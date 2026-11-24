import json
import numpy as np
import pandas as pd


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


claims_dev_map = {c["id"]: c for c in load_jsonl("lr2_artifacts/scifact_raw/claims_dev.jsonl")}
with open("lr2_artifacts/manifest.json", "r", encoding="utf-8") as f:
    manifest = json.load(f)
with open("lr3_artifacts/selection.json", "r", encoding="utf-8") as f:
    selection = json.load(f)
with open("lr3_artifacts/dev_all_retrieval.json", "r", encoding="utf-8") as f:
    dev_ret = json.load(f)
with open("lr3_artifacts/holdout_all_retrieval.json", "r", encoding="utf-8") as f:
    holdout_ret = json.load(f)

selected_method = selection["selected_method"]
holdout_ids = manifest["split"]["holdout_ids"]

# 1. Holdout 检索指标对比 (Dense vs Selected) / Метрики поиска на holdout (128 Q+)
q_plus_holdout = [claims_dev_map[cid] for cid in holdout_ids if claims_dev_map[cid].get("evidence")]
h_ret_rows = []
for m in ("dense", "bm25", "rrf", "rrf_ce"):
    h3, r3, mrr3, r20 = [], [], [], []
    for c in q_plus_holdout:
        rel = set(c["evidence"].keys())
        ranked = [x["doc_id"] for x in holdout_ret[str(c["id"])][m]]
        top3, top20 = ranked[:3], ranked[:20]
        h3.append(1.0 if (rel & set(top3)) else 0.0)
        r3.append(len(rel & set(top3)) / len(rel))
        r20.append(len(rel & set(top20)) / len(rel))
        rr = next((1.0 / i for i, d in enumerate(top3, 1) if d in rel), 0.0)
        mrr3.append(rr)
    h_ret_rows.append(
        {
            "Выборка / Метод": f"Holdout / {m}",
            "Hit@3": round(float(np.mean(h3)), 4),
            "Recall@3": round(float(np.mean(r3)), 4),
            "MRR@3": round(float(np.mean(mrr3)), 4),
            "Recall@20": round(float(np.mean(r20)), 4),
        }
    )
df_h_ret = pd.DataFrame(h_ret_rows)
df_h_ret.to_csv("lr3_artifacts/holdout_retrieval_metrics.csv", index=False)
print("=== Таблица поиска на holdout (Q+, 128 утверждений) / holdout 集检索指标 ===")
print(df_h_ret.to_string(index=False))
print()


# 2. 生成质量汇总表 (dev 对比 + holdout 盲测对比，含按类别召回率)
# Сводная таблица качества ответов на dev и holdout (включая полноту по классам)
def summarize_gen(records: list, title: str):
    df = pd.DataFrame(records)
    n = len(df)
    q_sub = df[df["gold_label"].isin(["SUPPORT", "CONTRADICT"])]
    return {
        "Выборка / Метод": title,
        "N": n,
        "Accuracy (%)": round(df["acc_ok"].mean() * 100, 2),
        "Recall SUPPORT (%)": round(df[df["gold_label"] == "SUPPORT"]["acc_ok"].mean() * 100, 2),
        "Recall CONTRADICT (%)": round(df[df["gold_label"] == "CONTRADICT"]["acc_ok"].mean() * 100, 2),
        "Recall NOINFO (%)": round(df[df["gold_label"] == "NOINFO"]["acc_ok"].mean() * 100, 2),
        "Верная ссылка Q+ (%)": round(q_sub["ref_ok"].mean() * 100, 2),
        "Пер. обосн. Q+ (%)": round(q_sub["contains_gold"].mean() * 100, 2),
        "Ошибки разбора": int((df["parse_status"] != "ok").sum()),
    }


dev_dense = load_jsonl("lr3_artifacts/gen_dev_dense.jsonl")
dev_rrf = load_jsonl("lr3_artifacts/gen_dev_rrf.jsonl")
dev_ce = load_jsonl("lr3_artifacts/gen_dev_rrf_ce.jsonl")
hold_dense = load_jsonl("lr3_artifacts/gen_holdout_dense.jsonl")
hold_sel = load_jsonl(f"lr3_artifacts/gen_holdout_{selected_method}.jsonl")

gen_rows = [
    summarize_gen(dev_dense, "dev / Dense"),
    summarize_gen(dev_rrf, "dev / RRF"),
    summarize_gen(dev_ce, "dev / RRF + CE"),
    summarize_gen(hold_dense, "holdout / Dense"),
    summarize_gen(hold_sel, f"holdout / {selected_method}"),
]
df_gen = pd.DataFrame(gen_rows)
df_gen.to_csv("lr3_artifacts/generation_summary_all.csv", index=False)
print("=== Сводная таблица генерации (dev и holdout) / 生成质量综合对比表 ===")
print(df_gen.to_string(index=False))
print()


# 3. 成对转移统计 (Dense -> RRF+CE on dev, Dense -> selected on holdout)
def paired_stats(base_recs, new_recs, label):
    b = pd.DataFrame(base_recs).set_index("claim_id")
    n = pd.DataFrame(new_recs).set_index("claim_id")
    fixed = int(((~b["acc_ok"]) & (n["acc_ok"])).sum())
    broken = int(((b["acc_ok"]) & (~n["acc_ok"])).sum())
    both_ok = int(((b["acc_ok"]) & (n["acc_ok"])).sum())
    both_bad = int(((~b["acc_ok"]) & (~n["acc_ok"])).sum())
    print(f"Попарно {label}: исправлено={fixed}, испорчено={broken}, оба верны={both_ok}, оба неверны={both_bad}")


paired_stats(dev_dense, dev_ce, "dev (Dense -> RRF+CE)")
paired_stats(hold_dense, hold_sel, f"holdout (Dense -> {selected_method})")
print()

# 4. 自动提取第 4.4 节 4 条完整链路追踪案例 (Разбор трейсов)
print("=== РАЗБОР ТРЕЙСОВ (Раздел 4.4) / 四类完整链路追踪剖析 ===")
d_map = {r["claim_id"]: r for r in dev_dense}
r_map = {r["claim_id"]: r for r in dev_rrf}
c_map = {r["claim_id"]: r for r in dev_ce}


def trace_claim(title, cid):
    c = claims_dev_map[cid]
    ret = dev_ret[str(cid)]
    rel = list(c["evidence"].keys())

    def find_ranks(lst):
        return {d: next((x["rank"] for x in lst if x["doc_id"] == d), None) for d in rel}

    print(f"\n--- {title} ---")
    print(f"Claim ID: {cid} | Gold: {c['evidence']} | Текст: {c['claim']}")
    print(f"Ранги эталона: Dense={find_ranks(ret['dense'])}, BM25={find_ranks(ret['bm25'])}, "
          f"RRF={find_ranks(ret['rrf'])}, RRF+CE={find_ranks(ret['rrf_ce'])}")
    for d in rel:
        rrf_item = next((x for x in ret["rrf"] if x["doc_id"] == d), None)
        if rrf_item:
            print(f"Вклады RRF для {d}: {rrf_item['contrib']}, сумма={rrf_item['score']}")
    print(f"Передано (Dense): {d_map[cid]['passed_docs']} -> Ответ: {d_map[cid]['raw_response']}")
    print(f"Передано (RRF):   {r_map[cid]['passed_docs']} -> Ответ: {r_map[cid]['raw_response']}")
    print(f"Передано (CE):    {c_map[cid]['passed_docs']} -> Ответ: {c_map[cid]['raw_response']}")


q_plus_ids = [cid for cid in manifest["split"]["dev_ids"] if claims_dev_map[cid].get("evidence")]

# 案例 1：BM25 top-3 命中而 Dense top-3 漏掉
case1 = [
    cid for cid in q_plus_ids
    if (set(claims_dev_map[cid]["evidence"].keys()) & {x["doc_id"] for x in dev_ret[str(cid)]["bm25"][:3]})
    and not (set(claims_dev_map[cid]["evidence"].keys()) & {x["doc_id"] for x in dev_ret[str(cid)]["dense"][:3]})
]
if case1:
    trace_claim("Случай 1: BM25 нашёл в топ-3, Dense пропустил", case1[0])

# 案例 2：Dense top-3 命中而 BM25 top-3 漏掉
case2 = [
    cid for cid in q_plus_ids
    if (set(claims_dev_map[cid]["evidence"].keys()) & {x["doc_id"] for x in dev_ret[str(cid)]["dense"][:3]})
    and not (set(claims_dev_map[cid]["evidence"].keys()) & {x["doc_id"] for x in dev_ret[str(cid)]["bm25"][:3]})
]
if case2:
    trace_claim("Случай 2: Dense нашёл в топ-3, BM25 пропустил", case2[0])

# 案例 3：RRF 改变了 top-3 文档组合
case3 = [
    cid for cid in q_plus_ids
    if [x["doc_id"] for x in dev_ret[str(cid)]["rrf"][:3]] != [x["doc_id"] for x in dev_ret[str(cid)]["dense"][:3]]
    and (set(claims_dev_map[cid]["evidence"].keys()) & {x["doc_id"] for x in dev_ret[str(cid)]["rrf"][:3]})
]
if case3:
    trace_claim("Случай 3: RRF изменил первую тройку", case3[0])

# 案例 4：Cross-Encoder 重排改变了最终模型回答
case4 = [
    cid for cid in manifest["split"]["dev_ids"]
    if r_map[cid]["acc_ok"] != c_map[cid]["acc_ok"]
]
if case4:
    trace_claim("Случай 4: Переранжирование CE изменило ответ модели", case4[0])