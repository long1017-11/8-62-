import os
import json
import pandas as pd
from tqdm import tqdm
from rag_core import (
    TOKEN_BUDGET,
    build_context_within_budget,
    check_contains_gold_rationale,
    call_llm_verdict,
)

# 记录实验前预测（第 4 节要求） / Фиксация прогноза до запуска генерации (Раздел 4)
forecast_lr3 = """ПРОГНОЗ ДО ЗАПУСКОВ (ЛР3) / 实验三运行前预测:
1. BM25 выиграет у плотного поиска на утверждениях с редкими биомедицинскими терминами, аббревиатурами генов и белков (например, ALDH1, PPM1D), где требуется точное лексическое совпадение.
   （BM25 将在包含罕见生物医学术语、基因/蛋白质缩写的论断上胜过稠密检索，因为这些词需要精确词面匹配。）
2. Объединение RRF объединит находки обоих методов и повысит Recall@3 и Recall@20 за счёт документов, найденных только одним из ретриверов.
   （RRF 融合将整合两路检索结果，利用各自独有的命中提升 Recall@3 和 Recall@20。）
3. Переранжирование cross-encoder (RRF + CE) улучшит точность ранжирования в топ-3 (рост MRR@3) при неизменном Recall@20, но прирост итоговой Accuracy модели будет умеренным, так как часть ошибок связана с рассуждением самой LLM.
   （Cross-Encoder 重排将在保持 Recall@20 不变的前提下提高前 3 名的排序精度（提升 MRR@3），但最终模型 Accuracy 的提升幅度将较为温和，因为部分错误源于大模型自身的推理限制。）
"""
with open("lr3_artifacts/pre_run_forecast_lr3.txt", "w", encoding="utf-8") as f:
    f.write(forecast_lr3)
print(forecast_lr3)


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


corpus_map = {str(d["doc_id"]): d for d in load_jsonl("lr2_artifacts/scifact_raw/corpus.jsonl")}
claims_dev_map = {c["id"]: c for c in load_jsonl("lr2_artifacts/scifact_raw/claims_dev.jsonl")}
lr2_results = load_jsonl("lr2_artifacts/results.jsonl")

with open("lr2_artifacts/manifest.json", "r", encoding="utf-8") as f:
    manifest = json.load(f)
with open("lr3_artifacts/dev_all_retrieval.json", "r", encoding="utf-8") as f:
    dev_ret = json.load(f)
with open("lr3_artifacts/holdout_all_retrieval.json", "r", encoding="utf-8") as f:
    holdout_ret = json.load(f)
with open("lr3_artifacts/selection.json", "r", encoding="utf-8") as f:
    selection = json.load(f)

selected_method = selection["selected_method"]
dev_ids = manifest["split"]["dev_ids"]
holdout_ids = manifest["split"]["holdout_ids"]


def get_gold_label(c):
    if not c.get("evidence"):
        return "NOINFO"
    return next(iter({r["label"] for rs in c["evidence"].values() for r in rs}))


def evaluate_prediction(gold_label: str, claim: dict, parsed: dict):
    if parsed is None:
        return False, False
    pred_label = parsed["label"]
    acc_ok = pred_label == gold_label
    if gold_label in ("SUPPORT", "CONTRADICT"):
        if not acc_ok:
            return False, False
        cited_did = parsed["doc_id"]
        cited_sents = set(parsed["sentences"])
        gold_evs = claim["evidence"].get(cited_did, [])
        ref_ok = any(set(ev["sentences"]).issubset(cited_sents) for ev in gold_evs)
        return True, ref_ok
    else:
        return acc_ok, acc_ok


def run_generation_batch(split_name: str, id_list: list, ret_dict: dict, method: str, out_path: str):
    if os.path.exists(out_path):
        os.remove(out_path)
    records = []
    print(f"Запуск генерации: {split_name} / {method} ({len(id_list)} утверждений)...")
    for cid in tqdm(id_list):
        c = claims_dev_map[cid]
        gold = get_gold_label(c)
        top3 = ret_dict[str(cid)][method][:3]
        cand_docs = [corpus_map[item["doc_id"]] for item in top3]
        msgs, passed_docs, p_toks = build_context_within_budget(c["claim"], cand_docs, TOKEN_BUDGET)
        contains_gold = check_contains_gold_rationale(c, passed_docs) if gold != "NOINFO" else False
        raw, parsed, status, _, g_time = call_llm_verdict(msgs, "rag", passed_docs)
        acc_ok, ref_ok = evaluate_prediction(gold, c, parsed)

        rec = {
            "split": split_name,
            "method": method,
            "claim_id": cid,
            "gold_label": gold,
            "found_docs": top3,
            "passed_docs": passed_docs,
            "contains_gold": contains_gold,
            "prompt_tokens": p_toks,
            "raw_response": raw,
            "parsed": parsed,
            "parse_status": status,
            "gen_time_s": g_time,
            "acc_ok": acc_ok,
            "ref_ok": ref_ok,
        }
        with open(out_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        records.append(rec)
    return records


# 1. 复用 ЛР2 的 dev/dense (k=3) 90 条结果（第 4.2 节允许直接复用） / Переиспользование Dense k=3 из ЛР2
lr2_dense_dev = [r for r in lr2_results if r["mode"] == "rag_k3"]
assert len(lr2_dense_dev) == 90
with open("lr3_artifacts/gen_dev_dense.jsonl", "w", encoding="utf-8") as f:
    for r in lr2_dense_dev:
        r_copy = dict(r)
        r_copy["split"] = "dev"
        r_copy["method"] = "dense"
        f.write(json.dumps(r_copy, ensure_ascii=False) + "\n")

# 2. 在 dev (90) 上运行 RRF 和 RRF + CE（共 180 次调用） / Эксперимент 2: RRF и RRF+CE на dev (180 вызовов)
run_generation_batch("dev", dev_ids, dev_ret, "rrf", "lr3_artifacts/gen_dev_rrf.jsonl")
run_generation_batch("dev", dev_ids, dev_ret, "rrf_ce", "lr3_artifacts/gen_dev_rrf_ce.jsonl")

# 3. 在 holdout (210) 上一次性运行 Dense 和选定方案（共 420 次调用） / Эксперимент 3: holdout (420 вызовов)
run_generation_batch("holdout", holdout_ids, holdout_ret, "dense", "lr3_artifacts/gen_holdout_dense.jsonl")
run_generation_batch(
    "holdout", holdout_ids, holdout_ret, selected_method, f"lr3_artifacts/gen_holdout_{selected_method}.jsonl"
)

print("\n✅ Все 600 новых вызовов завершены! Запустите lr3_step3_analyze.py для получения таблиц и разбора трейсов.")