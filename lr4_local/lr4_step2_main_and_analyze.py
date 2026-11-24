import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from agentdojo.task_suite.load_suites import get_suites
from agentdojo.functions_runtime import FunctionsRuntime
from agent_harness import CustomHarnessAgent

# 1. 记录运行前预测（第 4.1 节要求） / Прогноз до основной серии (п. 4.1)
forecast_lr4 = """ПРОГНОЗ ДО ОСНОВНОЙ СЕРИИ (ЛР4) / 实验四主实验前预测:
1. По общей успешности на 30 итоговых задачах ReAct превзойдёт plan-then-execute, особенно на многошаговых задачах и задачах, где аргументы следующего вызова (например, email клиента, ID события или файла) зависят от результата предыдущего чтения.
   （在 30 个正式任务上，ReAct 的总体成功率将高于 Plan-then-execute，尤其是在多步调用以及下一步参数（如客户邮箱、日程/文件 ID）依赖于前一步读取结果的任务上。）
2. Режим plan-then-execute справится сопоставимо с ReAct на простых задачах с 1–2 предсказуемыми шагами чтения, где нужные имена файлов или поисковые запросы известны сразу из текста задания.
   （在仅需 1–2 步且检索关键词或文件名直接包含在任务描述中的简单只读任务上，Plan-then-execute 的表现将与 ReAct 相当。）
3. По затратам: на простых задачах plan-then-execute сделает больше вызовов модели (минимум 3 вызова: план + шаг + ответ против 2 у ReAct), однако медианный суммарный расход входных токенов у plan-then-execute будет ниже, так как на шагах выполнения передаётся описание только одного запланированного инструмента вместо всех 24.
   （在开销方面：对于简单任务，Plan-then-execute 的模型调用次数更多（至少 3 次：规划+执行+汇总，而 ReAct 仅需 2 次），但单次 Episode 的中位总输入 token 消耗更低，因为执行阶段每步仅传入 1 个计划工具的描述而非全部 24 个工具。）
"""
with open("lr4_artifacts/pre_run_forecast_lr4.txt", "w", encoding="utf-8") as f:
    f.write(forecast_lr4)
print(forecast_lr4)

suite = get_suites("v1.2.2")["workspace"]
with open("lr4_artifacts/manifest_lr4.json", "r", encoding="utf-8") as f:
    manifest = json.load(f)
eval_ids = manifest["eval_ids"]  # 30 个正式任务

# 2. 按任务分组标注属性（单步 vs 多步，只读 vs 修改环境状态） / Разметка групп задач (п. 2.5)
task_meta = {}
for tid in eval_ids:
    t_obj = suite.get_user_task_by_id(tid)
    env_before = suite.load_and_inject_default_environment({})
    gt_calls = t_obj.ground_truth(env_before)
    env_after = suite.load_and_inject_default_environment({})
    rt_tmp = FunctionsRuntime(suite.tools)
    for fc in gt_calls:
        rt_tmp.run_function(env_after, fc.function, fc.args)
    modifies_state = env_before != env_after
    task_meta[tid] = {
        "gt_calls": len(gt_calls),
        "is_single_call": len(gt_calls) <= 1,
        "modifies_state": modifies_state,
    }

# 3. 交替顺序运行 30 个任务 × 2 模式 = 60 个 Episode（第 4.2 节要求）
# Чередование порядка режимов от задачи к задаче: 30 задач * 2 режима = 60 эпизодов
episodes = []
print("Запуск основной серии (30 задач, 60 эпизодов с чередованием режимов)...")
for idx, tid in enumerate(tqdm(eval_ids)):
    t_obj = suite.get_user_task_by_id(tid)
    modes_order = ("react", "plan_then_execute") if idx % 2 == 0 else ("plan_then_execute", "react")

    for mode in modes_order:
        agent = CustomHarnessAgent({"mode": mode})
        util, _ = suite.run_task_with_pipeline(agent, t_obj, None, {})
        tr = agent.last_trace
        rec = {
            "task_id": tid,
            "mode": mode,
            "utility": bool(util),
            "gt_calls": task_meta[tid]["gt_calls"],
            "is_single_call": task_meta[tid]["is_single_call"],
            "modifies_state": task_meta[tid]["modifies_state"],
            "stop_reason": tr["stop_reason"],
            "model_calls": tr["model_calls"],
            "tool_calls": tr["tool_calls"],
            "tool_errors": tr["tool_errors"],
            "prompt_tokens": tr["total_prompt_tokens"],
            "completion_tokens": tr["total_completion_tokens"],
            "total_tokens": tr["total_prompt_tokens"] + tr["total_completion_tokens"],
            "time_s": tr["episode_time_s"],
        }
        episodes.append(rec)
        with open(f"lr4_artifacts/traces_main/{tid}_{mode}.json", "w", encoding="utf-8") as f:
            json.dump({"meta": rec, "trace": tr}, f, indent=2, ensure_ascii=False)

df = pd.DataFrame(episodes)
df.to_csv("lr4_artifacts/final_episodes.csv", index=False)

# 4. 表 1：按任务分组统计成功率（第 4.2.1 节） / Таблица успешности по группам (п. 4.2.1)
groups = [
    ("Все задачи", df["task_id"].notna()),
    ("Один эталонный вызов", df["is_single_call"]),
    ("Несколько эталонных вызовов", ~df["is_single_call"]),
    ("Только чтение", ~df["modifies_state"]),
    ("Изменение среды", df["modifies_state"]),
]
succ_rows = []
for g_name, mask in groups:
    sub_r = df[(df["mode"] == "react") & mask]
    sub_p = df[(df["mode"] == "plan_then_execute") & mask]
    n_g = len(sub_r)
    kr, kp = int(sub_r["utility"].sum()), int(sub_p["utility"].sum())
    succ_rows.append(
        {
            "Группа задач": g_name,
            "N": n_g,
            "ReAct": f"{kr}/{n_g} ({kr*100/n_g:.1f}%)",
            "Plan-then-execute": f"{kp}/{n_g} ({kp*100/n_g:.1f}%)",
        }
    )
df_succ = pd.DataFrame(succ_rows)
df_succ.to_csv("lr4_artifacts/table1_success.csv", index=False)
print("\n=== ТАБЛИЦА 1: Успешность по группам задач (30 итоговых задач) ===")
print(df_succ.to_string(index=False))

# 5. 表 2：成对对比（第 4.2.2 节） / Попарное сравнение ReAct -> plan-then-execute
piv = df.pivot(index="task_id", columns="mode", values="utility")
both_ok = int((piv["react"] & piv["plan_then_execute"]).sum())
only_r = int((piv["react"] & (~piv["plan_then_execute"])).sum())
only_p = int(((~piv["react"]) & piv["plan_then_execute"]).sum())
neither = int(((~piv["react"]) & (~piv["plan_then_execute"])).sum())
print(f"\nПопарно (30 задач): Оба режима={both_ok}, Только ReAct={only_r}, Только Plan={only_p}, Ни один={neither}")

# 6. 表 3：资源消耗统计（全部 / 成功 / 失败 的中位数与范围，第 4.2.3 节） / Затраты (медианы и диапазоны)
def fmt_med_range(s):
    return f"{np.median(s):.2f} ({np.min(s):.2f}–{np.max(s):.2f})" if s.dtype == float else f"{int(np.median(s))} ({int(np.min(s))}–{int(np.max(s))})"

cost_rows = []
for subset_name, cond in [
    ("Все эпизоды", df["task_id"].notna()),
    ("Успешные", df["utility"]),
    ("Неуспешные", ~df["utility"]),
]:
    for metric_col, metric_label in [
        ("model_calls", "Вызовы модели"),
        ("tool_calls", "Попытки вызова инструментов"),
        ("prompt_tokens", "Входные токены"),
        ("completion_tokens", "Выходные токены"),
        ("total_tokens", "Всего токенов"),
        ("time_s", "Время эпизода (с)"),
    ]:
        sr = df[(df["mode"] == "react") & cond][metric_col]
        sp = df[(df["mode"] == "plan_then_execute") & cond][metric_col]
        cost_rows.append(
            {
                "Подмножество": subset_name,
                "Метрика": metric_label,
                f"ReAct (n={len(sr)})": fmt_med_range(sr) if len(sr) else "—",
                f"Plan (n={len(sp)})": fmt_med_range(sp) if len(sp) else "—",
            }
        )
df_cost = pd.DataFrame(cost_rows)
df_cost.to_csv("lr4_artifacts/table3_costs.csv", index=False)
print("\n=== ТАБЛИЦА 3: Затраты (медиана и диапазон) ===")
print(df_cost.to_string(index=False))

# 7. 失败原因分布（第 4.2.4 节） / Причины неудач по режимам
print("\n=== ТАБЛИЦА 4: Причины неудач (utility == False) ===")
for m in ("react", "plan_then_execute"):
    fail_df = df[(df["mode"] == m) & (~df["utility"])]
    print(f"Режим {m} (всего неудач: {len(fail_df)}):")
    for reason, cnt in fail_df["stop_reason"].value_counts().items():
        print(f"  - stop_reason='{reason}': {cnt}")
    with_err = int((fail_df["tool_errors"] > 0).sum())
    print(f"  - эпизодов с ошибками аргументов/инструментов: {with_err}")