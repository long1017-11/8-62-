import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mlflow
from experiment_core import prepare_dataset, build_prompt, timed_chat

mlflow.set_tracking_uri("sqlite:///lr1_artifacts/mlflow.db")
mlflow.set_experiment("LR1_GTX1650_Local")

tokens = prepare_dataset()

# === 1. 预检 ===
msg_chk2 = build_prompt(tokens, 256, "chk2")
res_chk2 = timed_chat(msg_chk2, max_tokens=64, req_id="chk2", condition="check_len", ignore_eos=True)
print(f"Проверка длины: запрошено 64, получено {res_chk2['output_tokens']}")

msg_chk3 = [{"role": "user", "content": "Explain quantum entanglement in 30 words."}]
r3_a = timed_chat(msg_chk3, max_tokens=50, req_id="chk3_a", condition="check_det", ignore_eos=False, temperature=0.0)
r3_b = timed_chat(msg_chk3, max_tokens=50, req_id="chk3_b", condition="check_det", ignore_eos=False, temperature=0.0)
print(f"Проверка повторяемости: {r3_a['text'] == r3_b['text']}")

# === 2. 实验 1.1-1.2: 输入长度 vs TTFT ===
_ = timed_chat(build_prompt(tokens, 512, "warmup_exp1"), max_tokens=16, req_id="warmup", condition="warmup")

prompt_lengths = [256, 1024, 2048, 4096, 8192]
exp1_records = []
for plen in prompt_lengths:
    for i in range(5):
        msgs = build_prompt(tokens, plen, f"exp1_{plen}_{i}", offset=i)
        rec = timed_chat(msgs, max_tokens=16, req_id=f"exp1_{plen}_{i}", condition=f"exp1_len_{plen}")
        rec["target_plen"] = plen
        exp1_records.append(rec)
    print(f"Длина {plen} завершена")

df_exp1 = pd.DataFrame(exp1_records)
df_exp1["prefill_tps"] = df_exp1["prompt_tokens"] / df_exp1["ttft_s"]
df_exp1.to_csv("lr1_artifacts/exp1_ttft.csv", index=False)

summary1 = df_exp1.groupby("target_plen").agg(
    actual_prompt_tokens=("prompt_tokens", "median"),
    ttft_med_s=("ttft_s", "median"),
    ttft_min_s=("ttft_s", "min"),
    ttft_max_s=("ttft_s", "max"),
    prefill_tps_med=("prefill_tps", "median")
).reset_index()
print("\nСводка Эксперимента 1:")
print(summary1.to_string(index=False))

plt.figure(figsize=(8, 5))
plt.errorbar(summary1["actual_prompt_tokens"], summary1["ttft_med_s"],
             yerr=[summary1["ttft_med_s"] - summary1["ttft_min_s"], summary1["ttft_max_s"] - summary1["ttft_med_s"]],
             fmt="-o", capsize=5, label="Медиана и диапазон (min-max)")
plt.title("Эксперимент 1: Зависимость TTFT от длины запроса (GTX 1650 4GB)")
plt.xlabel("Длина запроса (токены)")
plt.ylabel("TTFT (секунды)")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig("lr1_artifacts/fig1_ttft_vs_prompt.png", dpi=200)

# === 3. 实验 2: 输出长度 vs E2E ===
_ = timed_chat(build_prompt(tokens, 1024, "warmup_exp2"), max_tokens=16, req_id="warmup", condition="warmup")

output_lengths = [16, 64, 256, 1024]
exp2_records = []
for olen in output_lengths:
    for i in range(5):
        msgs = build_prompt(tokens, 1024, f"exp2_{olen}_{i}", offset=i+10)
        rec = timed_chat(msgs, max_tokens=olen, req_id=f"exp2_{olen}_{i}", condition=f"exp2_out_{olen}", ignore_eos=True)
        exp2_records.append(rec)
    print(f"Ответ {olen} токенов завершён")

df_exp2 = pd.DataFrame(exp2_records)
df_exp2.to_csv("lr1_artifacts/exp2_tpot.csv", index=False)

slope, intercept = np.polyfit(df_exp2["output_tokens"], df_exp2["e2e_s"], 1)
print(f"\nЛинейная регрессия: E2E ≈ {intercept:.4f} с (TTFT) + {slope*1000:.2f} мс (TPOT) * N")

plt.figure(figsize=(8, 5))
plt.scatter(df_exp2["output_tokens"], df_exp2["e2e_s"], alpha=0.6, label="Замеры запросов")
x_vals = np.linspace(16, 1024, 100)
plt.plot(x_vals, intercept + slope * x_vals, "r--", label=f"Аппроксимация: {intercept:.3f}с + {slope*1000:.1f}мс · N")
plt.title("Эксперимент 2: Полное время ответа от длины генерации (GTX 1650)")
plt.xlabel("Длина ответа N (токены)")
plt.ylabel("Полное время ответа E2E (секунды)")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig("lr1_artifacts/fig2_e2e_vs_output.png", dpi=200)

with mlflow.start_run(run_name="Exp1_and_Exp2_GTX1650"):
    mlflow.log_metric("exp2_fit_intercept_ttft_s", intercept)
    mlflow.log_metric("exp2_fit_slope_tpot_s", slope)
    mlflow.log_artifact("lr1_artifacts/fig1_ttft_vs_prompt.png")
    mlflow.log_artifact("lr1_artifacts/fig2_e2e_vs_output.png")