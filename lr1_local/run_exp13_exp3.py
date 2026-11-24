import sys
import asyncio
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mlflow
from openai import AsyncOpenAI
from experiment_core import client, BASE_URL, MODEL_NAME, prepare_dataset, build_prompt

# 修复 Windows + Python 3.14 asyncio ProactorEventLoop 的 WinError 10054 噪音
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# 为本地 4GB GPU 高并发排队设置充足的超时时间 (600秒)
long_timeout_async_client = AsyncOpenAI(base_url=BASE_URL, api_key="none", timeout=600.0)

mlflow.set_tracking_uri("sqlite:///lr1_artifacts/mlflow.db")
mlflow.set_experiment("LR1_GTX1650_Local")

tokens = prepare_dataset()

# === 实验 1.3: 前缀缓存测试 ===
print("\n--- Запуск Эксперимента 1.3 (Prefix Caching) ---")
same_msgs = build_prompt(tokens, 3500, "prefix_cache_test", offset=99)
cache_ttfts = []
for attempt in range(1, 4):
    t0 = time.perf_counter()
    stream = client.chat.completions.create(
        model=MODEL_NAME, messages=same_msgs, max_tokens=16, temperature=0.0,
        stream=True, stream_options={"include_usage": True},
        extra_body={"ignore_eos": True, "cache_prompt": True}
    )
    t1 = None
    p_tok = 0
    for chunk in stream:
        if chunk.usage: p_tok = chunk.usage.prompt_tokens
        if chunk.choices and chunk.choices[0].delta.content and t1 is None:
            t1 = time.perf_counter()
    ttft = t1 - t0
    cache_ttfts.append(ttft)
    print(f"Попытка {attempt}: prompt_tokens={p_tok}, TTFT={ttft*1000:.2f} мс")
print(f"Ускорение TTFT со второго раза: {cache_ttfts[0] / cache_ttfts[1]:.2f}x")

# === 实验 3: 并发负载测试 ===
print("\n--- Запуск Эксперимента 3 (Нагрузочное тестирование) ---")
SLA_TTFT_P90_LIMIT = 2.0
SLA_TPOT_MED_LIMIT = 0.100

async def single_async_request(messages, max_tokens, req_id):
    t0 = time.perf_counter()
    stream = await long_timeout_async_client.chat.completions.create(
        model=MODEL_NAME, messages=messages, max_tokens=max_tokens, temperature=0.0,
        stream=True, stream_options={"include_usage": True},
        extra_body={"ignore_eos": True, "cache_prompt": False}
    )
    t1 = t_last = None
    usage = None
    async for chunk in stream:
        if chunk.usage: usage = chunk.usage
        if chunk.choices and chunk.choices[0].delta.content:
            now = time.perf_counter()
            if t1 is None: t1 = now
            t_last = now
    n = usage.completion_tokens if usage else 0
    ttft = (t1 - t0) if t1 else 0.0
    tpot = ((t_last - t1) / (n - 1)) if (n > 1 and t_last and t1) else 0.0
    return {"req_id": req_id, "ttft_s": ttft, "tpot_s": tpot, "output_tokens": n}

async def run_load_level(concurrency):
    # 每个并发级别发送 max(8, 2 * concurrency) 条请求，既保证统计充分又避免单次排队超 10 分钟
    total_reqs = max(8, 2 * concurrency)
    queue = asyncio.Queue()
    for i in range(total_reqs):
        msgs = build_prompt(tokens, 1024, f"load_c{concurrency}_r{i}", offset=i + concurrency*100)
        queue.put_nowait((i, msgs))

    results = []
    async def worker():
        while not queue.empty():
            try:
                idx, msgs = queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            res = await single_async_request(msgs, 128, f"c{concurrency}_r{idx}")
            results.append(res)
            queue.task_done()

    t_start = time.perf_counter()
    await asyncio.gather(*[asyncio.create_task(worker()) for _ in range(concurrency)])
    total_duration = time.perf_counter() - t_start

    total_out = sum(r["output_tokens"] for r in results)
    ttfts = [r["ttft_s"] for r in results]
    tpots = [r["tpot_s"] for r in results]
    ttft_med = float(np.median(ttfts))
    ttft_p90 = float(np.percentile(ttfts, 90))
    tpot_med = float(np.median(tpots))
    good_reqs = sum(1 for r in results if r["ttft_s"] <= SLA_TTFT_P90_LIMIT and r["tpot_s"] <= SLA_TPOT_MED_LIMIT)

    return {
        "concurrency": concurrency,
        "requests": total_reqs,
        "throughput_tps": round(total_out / total_duration, 2),
        "rps": round(len(results) / total_duration, 2),
        "goodput_rps": round(good_reqs / total_duration, 2),
        "ttft_med_s": round(ttft_med, 4),
        "ttft_p90_s": round(ttft_p90, 4),
        "tpot_med_ms": round(tpot_med * 1000, 2),
        "sla_passed": (ttft_p90 <= SLA_TTFT_P90_LIMIT) and (tpot_med <= SLA_TPOT_MED_LIMIT)
    }

async def main():
    await run_load_level(1)  # 预热
    load_summary = []
    for c in [1, 2, 4, 8, 16]:
        print(f"Тестирование нагрузки: c={c}...")
        stats = await run_load_level(c)
        load_summary.append(stats)
        print(stats)

    df_load = pd.DataFrame(load_summary)
    df_load.to_csv("lr1_artifacts/exp3_load_table.csv", index=False)
    print("\nСводка Эксперимента 3:")
    print(df_load.to_string(index=False))

    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.set_xlabel("Число одновременных клиентов (c)")
    ax1.set_ylabel("Пропускная способность (токен/с)", color="tab:blue")
    ax1.plot(df_load["concurrency"], df_load["throughput_tps"], "o-", color="tab:blue", label="Throughput (токен/с)")
    ax1.grid(True, linestyle="--", alpha=0.5)

    ax2 = ax1.twinx()
    ax2.set_ylabel("Задержка TTFT P90 (с) / TPOT медиана (x100 мс)", color="tab:red")
    ax2.plot(df_load["concurrency"], df_load["ttft_p90_s"], "s--", color="tab:red", label="TTFT P90 (с)")
    ax2.plot(df_load["concurrency"], df_load["tpot_med_ms"] / 100.0, "^:", color="tab:orange", label="TPOT медиана (x100 мс)")
    ax2.axhline(SLA_TTFT_P90_LIMIT, color="gray", linestyle="-.", label="Предел TTFT P90 (2.0 с)")

    plt.title("Эксперимент 3: Пропускная способность и задержка (GTX 1650 4GB)")
    fig.tight_layout()
    plt.savefig("lr1_artifacts/fig3_throughput_latency.png", dpi=200)

    with mlflow.start_run(run_name="Exp3_Load_Testing_GTX1650"):
        mlflow.log_artifact("lr1_artifacts/exp3_load_table.csv")
        mlflow.log_artifact("lr1_artifacts/fig3_throughput_latency.png")

if __name__ == "__main__":
    asyncio.run(main())