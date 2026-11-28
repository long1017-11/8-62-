import json
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from agentdojo.task_suite.load_suites import get_suites

RESULTS_FILE = "lr5_artifacts/results.jsonl"
records = [json.loads(line) for line in open(RESULTS_FILE, "r", encoding="utf-8") if line.strip()]
df = pd.DataFrame(records)

ws_suite = get_suites("v1.2.2")["workspace"]
single_call_tasks = set()
for tid in df[df["suite"] == "workspace"]["task_id"].unique():
    env_tmp = ws_suite.load_and_inject_default_environment({})
    gt = ws_suite.get_user_task_by_id(tid).ground_truth(env_tmp)
    if len(gt) <= 1:
        single_call_tasks.add(tid)

df["is_single_call"] = df["task_id"].isin(single_call_tasks)

print("=================================================================")
print("1. ЭКСПЕРИМЕНТ 1: СТАБИЛЬНОСТЬ (T=0.0 и T=0.7)")
print("=================================================================")

# 1.1 两次 T=0.0 运行的一致性
t0_df = df[df["config"] == "exp1_t0"]
piv_t0 = t0_df.pivot(index="task_id", columns="run_idx", values="utility")
diff_t0_tasks = piv_t0[piv_t0[0] != piv_t0[1]].index.tolist()
print(f"T=0.0 (2 прогона): run0={int(piv_t0[0].sum())}/30, run1={int(piv_t0[1].sum())}/30, "
      f"разный исход на {len(diff_t0_tasks)} задачах: {diff_t0_tasks}")

# 1.2 无偏估计 pass^k 与 pass@k (k=1..5)
def pass_pow_k(n, c, k):
    return math.comb(c, k) / math.comb(n, k) if c >= k else 0.0

def pass_at_k(n, c, k):
    return 1.0 - (math.comb(n - c, k) / math.comb(n, k)) if (n - c) >= k else 1.0

samp_df = df[df["config"] == "exp1_sample"]
task_c = samp_df.groupby(["task_id", "is_single_call"])["utility"].sum().reset_index()
task_c.rename(columns={"utility": "c"}, inplace=True)

unstable_tasks = task_c[(task_c["c"] > 0) & (task_c["c"] < 5)]
print(f"T=0.7 (5 прогонов): доля нестабильных задач (0 < c < 5) = {len(unstable_tasks)}/30 ({len(unstable_tasks)*100/30:.1f}%)")
print(f"Всегда успешны (c=5): {int((task_c['c']==5).sum())}/30 | Всегда неуспешны (c=0): {int((task_c['c']==0).sum())}/30")

# 不稳定任务中成功 run 与失败 run 的开销对比
unst_ids = set(unstable_tasks["task_id"])
unst_runs = samp_df[samp_df["task_id"].isin(unst_ids)]
print(f"Затраты на нестабильных задачах: успешные прогоны медиана токенов = {unst_runs[unst_runs['utility']]['total_tokens'].median():.0f}, "
      f"неуспешные = {unst_runs[~unst_runs['utility']]['total_tokens'].median():.0f}")

curve_rows = []
for grp_name, sub_c in [
    ("Все задачи (30)", task_c),
    ("Один вызов (10)", task_c[task_c["is_single_call"]]),
    ("Несколько вызовов (20)", task_c[~task_c["is_single_call"]]),
]:
    for k in range(1, 6):
        p_pow = np.mean([pass_pow_k(5, int(c), k) for c in sub_c["c"]])
        p_at = np.mean([pass_at_k(5, int(c), k) for c in sub_c["c"]])
        curve_rows.append({"Группа": grp_name, "k": k, "pass^k": round(p_pow, 4), "pass@k": round(p_at, 4)})
df_curves = pd.DataFrame(curve_rows)
df_curves.to_csv("lr5_artifacts/table1_pass_curves.csv", index=False)
print("\nТаблица pass^k и pass@k (k=1..5):")
print(df_curves.to_string(index=False))

# 绘制 pass^k 与 pass@k 曲线图
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
for grp_name in ["Все задачи (30)", "Один вызов (10)", "Несколько вызовов (20)"]:
    sub = df_curves[df_curves["Группа"] == grp_name]
    axes[0].plot(sub["k"], sub["pass^k"], marker="o", label=grp_name)
    axes[1].plot(sub["k"], sub["pass@k"], marker="s", label=grp_name)
axes[0].set_title("Надёжность pass^k (все k успешны)")
axes[1].set_title("Потенциал pass@k (хотя бы 1 из k)")
for ax in axes:
    ax.set_xlabel("Число попыток k")
    ax.set_ylabel("Вероятность")
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_ylim(0, 1.05)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
plt.tight_layout()
plt.savefig("lr5_artifacts/pass_curves_lr5.png", dpi=200)

print("\n=================================================================")
print("2. ЭКСПЕРИМЕНТ 2: СБОИ ИНСТРУМЕНТОВ И ПОЛИТИКИ P0, P1, P2")
print("=================================================================")
base_t0 = t0_df[t0_df["run_idx"] == 0].set_index("task_id")
base_model_calls = base_t0["model_calls"].sum()
base_tool_calls = base_t0["tool_calls"].sum()

fault_rows = []
for pol in ("P0", "P1", "P2"):
    sub = df[df["config"] == f"exp2_fault_{pol}"].set_index("task_id")
    succ = int(sub["utility"].sum())
    lost_vs_base = int((base_t0["utility"] & (~sub["utility"])).sum())
    gained_vs_base = int(((~base_t0["utility"]) & sub["utility"]).sum())
    fault_rows.append({
        "Политика": pol,
        "Успешность": f"{succ}/30 ({succ*100/30:.1f}%)",
        "Сбоев (A / B)": f"{int(sub['injected_faults_total'].sum())} ({int(sub['injected_faults_A'].sum())} / {int(sub['injected_faults_B'].sum())})",
        "Лишние вызовы модели (vs база)": int(sub["model_calls"].sum() - base_model_calls),
        "Лишние вызовы инстр. (vs база)": int(sub["tool_calls"].sum() - base_tool_calls),
        "Повторные действия (дубли)": int(sub["duplicate_write_actions"].sum()),
        "Потеряно задач vs база": lost_vs_base,
    })
df_faults = pd.DataFrame(fault_rows)
df_faults.to_csv("lr5_artifacts/table2_faults.csv", index=False)
print(df_faults.to_string(index=False))

print("\n=================================================================")
print("3. ЭКСПЕРИМЕНТ 3: ПЕРЕНОС НА BANKING И ИНТЕРВАЛЫ УИЛСОНА (95%)")
print("=================================================================")
def wilson_ci(k, n, z=1.96):
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = (z * math.sqrt((p * (1 - p) / n) + (z**2 / (4 * n**2)))) / denom
    return round(max(0.0, center - half), 4), round(min(1.0, center + half), 4)

ws_k = int(base_t0["utility"].sum())
bk_df = df[df["config"] == "exp3_banking"]
bk_k = int(bk_df["utility"].sum())
print(f"workspace (n=30): {ws_k}/30 ({ws_k*100/30:.1f}%), 95% ДИ Уилсона = {wilson_ci(ws_k, 30)}")
print(f"banking   (n=16): {bk_k}/16 ({bk_k*100/16:.1f}%), 95% ДИ Уилсона = {wilson_ci(bk_k, 16)}")
print("Неуспешные задачи banking:", bk_df[~bk_df["utility"]]["task_id"].tolist())

print("\n=================================================================")
print("4. ВЫБОРКА 15 ЭПИЗОДОВ ДЛЯ ПРОВЕРКИ ПРОВЕРКИ (п. 4.4)")
print("=================================================================")
pool_eval = pd.concat([base_t0.reset_index(), bk_df])
passed_10 = pool_eval[pool_eval["utility"]].sample(n=min(10, int(pool_eval["utility"].sum())), random_state=42)
failed_5 = pool_eval[~pool_eval["utility"]].sample(n=min(5, int((~pool_eval["utility"]).sum())), random_state=42)
audit_15 = pd.concat([passed_10, failed_5])
audit_records = []
for _, r in audit_15.iterrows():
    tr_data = json.load(open(r["trace_path"], "r", encoding="utf-8"))["trace"]
    audit_records.append({
        "suite": r["suite"],
        "task_id": r["task_id"],
        "stand_utility": r["utility"],
        "query": tr_data["query"],
        "final_answer": tr_data["final_answer"],
    })
with open("lr5_artifacts/audit_15_episodes.json", "w", encoding="utf-8") as f:
    json.dump(audit_records, f, indent=2, ensure_ascii=False)
print("Сохранено 15 эпизодов в lr5_artifacts/audit_15_episodes.json для ручной сверки.")