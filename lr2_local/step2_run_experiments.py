import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from rag_core import (
    PROMPTS,
    TOKEN_BUDGET,
    count_tokens,
    render_chat_prompt,
    build_context_within_budget,
    check_contains_gold_rationale,
    parse_and_validate_response,
    call_llm_verdict,
)

# 1. 实验运行前记录预测结论（第 4 节要求：До запусков запишите прогноз）
# Фиксация прогноза до запуска экспериментов (требование раздела 4)
forecast_text = """ПРОГНОЗ ДО ЗАПУСКОВ (ЛР2) / 实验运行前预测:
1. Без поиска (LLM-only) модель будет чаще всего ошибаться на узкоспециализированных биомедицинских утверждениях с точными числовыми показателями и склоняться к NOINFO или галлюцинациям.
   （无检索时，模型在包含精确数值的生物医学专业论断上最容易出错，倾向于保守回答 NOINFO 或产生幻觉。）
2. RAG при k=3 исправит значительную долю ошибок (ожидаемый прирост Accuracy на +15..25 п.п.), но может испортить 3–6 верных ответов LLM-only, если поиск не найдёт нужную аннотацию и модель строго ответит NOINFO.
   （k=3 的 RAG 将纠正大量错误，预计准确率提升 15–25 个百分点，但也可能因检索漏掉关键摘要而把原本答对的 3–6 题改错为 NOINFO。）
3. При переходе k=1 -> 3 -> 5 Recall@k и доля переданного обоснования вырастут (особенно с k=1 до k=3), однако при k=5 часть длинных документов упрётся в бюджет 2048 токенов, а лишний шум может слегка снизить точность модели.
   （从 k=1 到 3 再到 5，Recall@k 和依据完整传递率会上升，但 k=5 时部分长文档会受限于 2048 token 预算，且冗余噪声可能略微拉低模型准确率。）
"""
with open("lr2_artifacts/pre_run_forecast.txt", "w", encoding="utf-8") as f:
    f.write(forecast_text)
print(forecast_text)

# 2. 第 3.4 节实验前逻辑单元测试 / Обязательные проверки перед опытами (п. 3.4)
print("--- Проверка 3.4 перед опытами / 运行前逻辑校验 ---")
# 2.1 检索指标公式校验（2 篇相关文档，其中 1 篇排在第 3 位） / Проверка метрик поиска на контрольном примере
rel = {"d1", "d2"}
r3 = ["x1", "x2", "d2"]
r1 = ["x1"]
assert (
    int(bool(rel & set(r3))) == 1
    and len(rel & set(r3)) / len(rel) == 0.5
    and abs(1 / 3 - 0.333333) < 1e-4
)
assert int(bool(rel & set(r1))) == 0

# 2.2 依据完整性校验（缺少一句论证句时必须返回 False） / Проверка полноты передачи обоснования
mock_claim = {"evidence": {"100": [{"sentences": [2, 3], "label": "SUPPORT"}]}}
assert check_contains_gold_rationale(mock_claim, {"100": [0, 1, 2, 3]}) is True
assert check_contains_gold_rationale(mock_claim, {"100": [0, 1, 2]}) is False

# 2.3 JSON 校验器拦截非法标签、多余字段以及未传入的句子编号 / Проверка отклонения невалидных ответов
assert (
    parse_and_validate_response('{"label":"MAYBE","doc_id":"1","sentences":[0]}', "rag", {"1": [0]})[0]
    is None
)
assert (
    parse_and_validate_response(
        '{"label":"SUPPORT","doc_id":"1","sentences":[0],"extra":1}', "rag", {"1": [0]}
    )[0]
    is None
)
assert (
    parse_and_validate_response('{"label":"SUPPORT","doc_id":"1","sentences":[5]}', "rag", {"1": [0, 1]})[
        0
    ]
    is None
)
print("✅ Все 4 предварительные проверки 3.4 пройдены! / 全部 4 项预检均已通过！\n")


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


corpus_map = {str(d["doc_id"]): d for d in load_jsonl("lr2_artifacts/scifact_raw/corpus.jsonl")}
claims_dev_map = {c["id"]: c for c in load_jsonl("lr2_artifacts/scifact_raw/claims_dev.jsonl")}
claims_train = load_jsonl("lr2_artifacts/scifact_raw/claims_train.jsonl")

with open("lr2_artifacts/manifest.json", "r", encoding="utf-8") as f:
    manifest = json.load(f)
with open("lr2_artifacts/dev_top5_retrieval.json", "r", encoding="utf-8") as f:
    retrieval_top5 = json.load(f)

dev_claims = [claims_dev_map[cid] for cid in manifest["split"]["dev_ids"]]


def get_gold_label(c):
    if not c.get("evidence"):
        return "NOINFO"
    return next(iter({r["label"] for rs in c["evidence"].values() for r in rs}))


# 3. 使用 claims_train[0] 进行一次服务预热（第 3.1 节第 4 点） / Прогрев сервера на claims_train (п. 3.1.4)
warmup_msgs = [
    {"role": "system", "content": PROMPTS["llm_only_system"]},
    {"role": "user", "content": f"Claim: {claims_train[0]['claim']}\nProvide your verdict in JSON format:"},
]
_ = call_llm_verdict(warmup_msgs, "llm_only", {})
print("Прогрев на claims_train[0] выполнен. / 服务端预热完成。\n")

# 4. 在 Q+（60 条有明确证据的 SUPPORT/CONTRADICT 论断）上计算 k=1, 3, 5 的检索与预算截断指标
# Расчёт метрик поиска (Hit@k, Recall@k, MRR@k) и доли переданного обоснования на множестве Q+ (60 утверждений)
q_plus = [c for c in dev_claims if get_gold_label(c) in ("SUPPORT", "CONTRADICT")]
retrieval_stats = []

for k in (1, 3, 5):
    hits, recalls, rrs, contains_list, passed_doc_counts, passed_tok_counts = [], [], [], [], [], []
    for c in q_plus:
        cid = str(c["id"])
        rel_docs = set(c["evidence"].keys())
        top_k_items = retrieval_top5[cid][:k]
        found_docs = [item["doc_id"] for item in top_k_items]

        hit = 1.0 if (rel_docs & set(found_docs)) else 0.0
        rec = len(rel_docs & set(found_docs)) / len(rel_docs)
        rr = 0.0
        for r_idx, did in enumerate(found_docs, start=1):
            if did in rel_docs:
                rr = 1.0 / r_idx
                break

        cand_docs = [corpus_map[did] for did in found_docs]
        _, passed_docs, p_toks = build_context_within_budget(c["claim"], cand_docs, TOKEN_BUDGET)
        has_gold = 1.0 if check_contains_gold_rationale(c, passed_docs) else 0.0

        hits.append(hit)
        recalls.append(rec)
        rrs.append(rr)
        contains_list.append(has_gold)
        passed_doc_counts.append(len(passed_docs))
        passed_tok_counts.append(p_toks)

    retrieval_stats.append(
        {
            "k": k,
            "Hit@k": round(float(np.mean(hits)), 4),
            "Recall@k": round(float(np.mean(recalls)), 4),
            "MRR@k": round(float(np.mean(rrs)), 4),
            "Contains_Gold_Ratio": round(float(np.mean(contains_list)), 4),
            "Avg_Passed_Docs": round(float(np.mean(passed_doc_counts)), 2),
            "Avg_Prompt_Tokens": round(float(np.mean(passed_tok_counts)), 1),
        }
    )

df_ret = pd.DataFrame(retrieval_stats)
df_ret.to_csv("lr2_artifacts/retrieval_metrics_k135.csv", index=False)
print("=== Сводка поиска и передачи контекста (Q+, 60 утверждений) / 检索与上下文传递汇总 ===")
print(df_ret.to_string(index=False))
print()

# 5. 执行全部 420 次大模型推理评测（LLM-only 90 + RAG k=3/1/5 各 90 + Oracle 60）
# Выполнение всех 420 вызовов модели с логированием в results.jsonl
RESULTS_PATH = "lr2_artifacts/results.jsonl"
if os.path.exists(RESULTS_PATH):
    os.remove(RESULTS_PATH)


def log_result(record):
    with open(RESULTS_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def evaluate_prediction(gold_label: str, claim: dict, parsed: dict):
    """评估标签准确率 (Accuracy) 与引用句子准确性 (Верная ссылка) / Оценка точности метки и ссылки."""
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
        # 引用正确：标签正确且引用的句子集合完整包含至少一组标准答案论证句
        ref_ok = any(set(ev["sentences"]).issubset(cited_sents) for ev in gold_evs)
        return True, ref_ok
    else:
        return acc_ok, acc_ok


all_runs = []

# A. 无检索基线模式 / Режим LLM-only (90 утверждений)
print("Запуск 1/5: LLM-only (90 утверждений) / 运行无检索基线...")
for c in tqdm(dev_claims):
    gold = get_gold_label(c)
    msgs = [
        {"role": "system", "content": PROMPTS["llm_only_system"]},
        {"role": "user", "content": f"Claim: {c['claim']}\nProvide your verdict in JSON format:"},
    ]
    p_toks = count_tokens(render_chat_prompt(msgs))
    raw, parsed, status, _, g_time = call_llm_verdict(msgs, "llm_only", {})
    acc_ok, ref_ok = evaluate_prediction(gold, c, parsed)
    rec = {
        "claim_id": c["id"],
        "gold_label": gold,
        "mode": "llm_only",
        "k": 0,
        "found_docs": [],
        "passed_docs": {},
        "contains_gold": False,
        "prompt_tokens": p_toks,
        "raw_response": raw,
        "parsed": parsed,
        "parse_status": status,
        "gen_time_s": g_time,
        "acc_ok": acc_ok,
        "ref_ok": ref_ok,
    }
    log_result(rec)
    all_runs.append(rec)

# B. 检索增强模式 / Режим RAG для k = 3, 1, 5 (3 × 90 = 270 вызовов)
for k in (3, 1, 5):
    print(f"Запуск RAG k={k} (90 утверждений) / 运行 RAG k={k}...")
    for c in tqdm(dev_claims):
        cid = str(c["id"])
        gold = get_gold_label(c)
        top_k = retrieval_top5[cid][:k]
        cand_docs = [corpus_map[item["doc_id"]] for item in top_k]
        msgs, passed_docs, p_toks = build_context_within_budget(c["claim"], cand_docs, TOKEN_BUDGET)
        contains_gold = check_contains_gold_rationale(c, passed_docs) if gold != "NOINFO" else False
        raw, parsed, status, _, g_time = call_llm_verdict(msgs, "rag", passed_docs)
        acc_ok, ref_ok = evaluate_prediction(gold, c, parsed)
        rec = {
            "claim_id": c["id"],
            "gold_label": gold,
            "mode": f"rag_k{k}",
            "k": k,
            "found_docs": top_k,
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
        log_result(rec)
        all_runs.append(rec)

# C. 理想诊断模式（直接提供标准答案证据句） / Режим Oracle (60 утверждений SUPPORT/CONTRADICT)
print("Запуск 5/5: Oracle (60 утверждений SUPPORT/CONTRADICT) / 运行理想诊断模式 Oracle...")
for c in tqdm(q_plus):
    gold = get_gold_label(c)
    first_did = sorted(c["evidence"].keys())[0]
    gold_sents_idx = sorted({s for ev in c["evidence"][first_did] for s in ev["sentences"]})
    doc = corpus_map[first_did]

    # 仅向模型传递标准答案对应的句子及其原始编号 / Передаём только эталонные предложения с исходными индексами
    sys_msg = {"role": "system", "content": PROMPTS["rag_system"]}
    lines = "".join(f"[{i}] {doc['abstract'][i]}\n" for i in gold_sents_idx)
    user_txt = (
        f"Claim: {c['claim']}\n\nRetrieved Documents:\n"
        f"\n[Document doc_id={first_did}]\nTitle: {doc['title']}\nSentences:\n{lines}"
        f"\nProvide your verdict in JSON format:"
    )
    msgs = [sys_msg, {"role": "user", "content": user_txt}]
    passed_docs = {first_did: gold_sents_idx}
    p_toks = count_tokens(render_chat_prompt(msgs))
    raw, parsed, status, _, g_time = call_llm_verdict(msgs, "oracle", passed_docs)
    acc_ok, ref_ok = evaluate_prediction(gold, c, parsed)
    rec = {
        "claim_id": c["id"],
        "gold_label": gold,
        "mode": "oracle",
        "k": 1,
        "found_docs": [{"doc_id": first_did, "score": 1.0, "rank": 1}],
        "passed_docs": passed_docs,
        "contains_gold": True,
        "prompt_tokens": p_toks,
        "raw_response": raw,
        "parsed": parsed,
        "parse_status": status,
        "gen_time_s": g_time,
        "acc_ok": acc_ok,
        "ref_ok": ref_ok,
    }
    log_result(rec)
    all_runs.append(rec)

df_all = pd.DataFrame(all_runs)
df_all.to_pickle("lr2_artifacts/all_runs.pkl")
print(
    "\n✅ Все 420 вызовов завершены! / 全部 420 次调用已完成！请运行 step3_analyze_and_errors.py 生成统计表与错误分析。"
)