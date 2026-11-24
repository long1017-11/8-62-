import json
import pandas as pd

# 加载全部运行记录 / Загрузка всех результатов экспериментов
df = pd.read_pickle("lr2_artifacts/all_runs.pkl")


def summarize_mode(sub_df, mode_name):
    """汇总单个实验模式的 Accuracy、各类别 Recall、引用准确率与解析错误数 / Сводка метрик по режиму."""
    total = len(sub_df)
    acc = sub_df["acc_ok"].mean() * 100
    rec_sup = sub_df[sub_df["gold_label"] == "SUPPORT"]["acc_ok"].mean() * 100
    rec_con = sub_df[sub_df["gold_label"] == "CONTRADICT"]["acc_ok"].mean() * 100
    rec_noi = (
        sub_df[sub_df["gold_label"] == "NOINFO"]["acc_ok"].mean() * 100
        if (sub_df["gold_label"] == "NOINFO").any()
        else float("nan")
    )
    q_plus_sub = sub_df[sub_df["gold_label"].isin(["SUPPORT", "CONTRADICT"])]
    valid_ref = q_plus_sub["ref_ok"].mean() * 100
    parse_err = (sub_df["parse_status"] != "ok").sum()
    return {
        "Режим": mode_name,
        "N": total,
        "Accuracy (%)": round(acc, 2),
        "Recall SUPPORT (%)": round(rec_sup, 2),
        "Recall CONTRADICT (%)": round(rec_con, 2),
        "Recall NOINFO (%)": round(rec_noi, 2) if not pd.isna(rec_noi) else "-",
        "Верная ссылка Q+ (%)": round(valid_ref, 2),
        "Ошибки разбора": int(parse_err),
    }


# 1. 实验 1 汇总表（90 条全集对比 + 60 条 Q+ 子集上与 Oracle 对比）
# Таблица Эксперимента 1 (на 90 утверждениях и сравнение с Oracle на 60 Q+)
rows_exp1 = [
    summarize_mode(df[df["mode"] == "llm_only"], "LLM-only (90)"),
    summarize_mode(df[df["mode"] == "rag_k3"], "RAG k=3 (90)"),
    summarize_mode(df[(df["mode"] == "llm_only") & (df["gold_label"] != "NOINFO")], "LLM-only (60 Q+)"),
    summarize_mode(df[(df["mode"] == "rag_k3") & (df["gold_label"] != "NOINFO")], "RAG k=3 (60 Q+)"),
    summarize_mode(df[df["mode"] == "oracle"], "Oracle (60 Q+)"),
]
df_exp1 = pd.DataFrame(rows_exp1)
print("=== ТАБЛИЦА 1: Сравнение LLM-only, RAG (k=3) и Oracle / 实验 1 对比汇总表 ===")
print(df_exp1.to_string(index=False))
df_exp1.to_csv("lr2_artifacts/table1_exp1_summary.csv", index=False)

# 2. 成对转移统计：LLM-only -> RAG (k=3) 纠正了多少错题、改错了多少对题
# Попарный переход LLM-only -> RAG (k=3): сколько ответов исправлено и сколько испорчено
m_llm = df[df["mode"] == "llm_only"].set_index("claim_id")
m_rag3 = df[df["mode"] == "rag_k3"].set_index("claim_id")
fixed = ((~m_llm["acc_ok"]) & (m_rag3["acc_ok"])).sum()
broken = ((m_llm["acc_ok"]) & (~m_rag3["acc_ok"])).sum()
both_ok = ((m_llm["acc_ok"]) & (m_rag3["acc_ok"])).sum()
both_bad = ((~m_llm["acc_ok"]) & (~m_rag3["acc_ok"])).sum()
print(
    f"\nПопарно LLM-only -> RAG k=3 (90) / 成对变化统计: исправлено (纠正) = {fixed} (+{fixed*100/90:.1f} п.п.), "
    f"испорчено (改错) = {broken} (-{broken*100/90:.1f} п.п.), оба верны (均对) = {both_ok}, оба неверны (均错) = {both_bad}"
)

# 3. 实验 2 汇总表：比较 k = 1, 3, 5 的表现 / Таблица Эксперимента 2 (k = 1, 3, 5)
rows_exp2 = [
    summarize_mode(df[df["mode"] == "rag_k1"], "RAG k=1"),
    summarize_mode(df[df["mode"] == "rag_k3"], "RAG k=3"),
    summarize_mode(df[df["mode"] == "rag_k5"], "RAG k=5"),
]
df_exp2 = pd.DataFrame(rows_exp2)
print("\n=== ТАБЛИЦА 2: Сравнение k = 1, 3, 5 / 实验 2 不同检索文档数对比 ===")
print(df_exp2.to_string(index=False))
df_exp2.to_csv("lr2_artifacts/table2_exp2_k135.csv", index=False)

# 4. 自动提取第 4.3 节要求的 4 类典型错误案例（定位信息在哪个环节丢失）
# Автоматический подбор 4 случаев для разбора ошибок (Раздел 4.3)
with open("lr2_artifacts/scifact_raw/claims_dev.jsonl", "r", encoding="utf-8") as f:
    claims_map = {json.loads(line)["id"]: json.loads(line) for line in f if line.strip()}

rag3 = df[df["mode"] == "rag_k3"]
q_plus_rag = rag3[rag3["gold_label"].isin(["SUPPORT", "CONTRADICT"])]


def print_case(title, row):
    c = claims_map[row["claim_id"]]
    gold_docs = list(c["evidence"].keys())
    print(f"\n--- {title} ---")
    print(f"Claim ID: {c['id']} | Gold Label: {row['gold_label']} | Gold Docs: {gold_docs}")
    print(f"Текст утверждения / 论断文本: {c['claim']}")
    print(f"Выдача поиска (top-3) / 检索结果: {row['found_docs']}")
    print(f"Переданные предложения / 实际传入句号: {row['passed_docs']}")
    print(f"Сырой ответ модели / 模型原始输出: {row['raw_response']}")


# 情况 1：检索未找到标准答案文档 / Случай 1: Свидетельство не найдено поиском (потеря на этапе Retrieval)
c1 = q_plus_rag[
    q_plus_rag.apply(
        lambda r: not bool(
            set(claims_map[r["claim_id"]]["evidence"].keys())
            & {x["doc_id"] for x in r["found_docs"]}
        ),
        axis=1,
    )
]
if len(c1) > 0:
    print_case("Случай 1: Свидетельство не найдено поиском / 情况 1：检索阶段未召回目标文档", c1.iloc[0])

# 情况 2：已检索到目标文档，但受限于 2048 token 预算关键句未能放入上下文
# Случай 2: Найдено в top-3, но обоснование не поместилось в бюджет 2048 токенов (потеря на этапе Context Assembly)
c2 = q_plus_rag[
    q_plus_rag.apply(
        lambda r: bool(
            set(claims_map[r["claim_id"]]["evidence"].keys())
            & {x["doc_id"] for x in r["found_docs"]}
        )
        and (not r["contains_gold"]),
        axis=1,
    )
]
if len(c2) > 0:
    print_case(
        "Случай 2: Найдено, но не поместилось в 2048 токенов / 情况 2：已检索到但受预算限制未完整传入",
        c2.iloc[0],
    )
else:
    print(
        "\n--- Случай 2: Найдено, но не поместилось / 情况 2 ---\n"
        "В dev при k=3 и бюджете 2048 токенов таких случаев не возникло (все найденные документы поместились)."
    )

# 情况 3：标准证据句已完整传入，但模型推理或引用判断出错
# Случай 3: Обоснование передано целиком, но ответ модели неверен (потеря на этапе Generation/Reasoning)
c3 = q_plus_rag[(q_plus_rag["contains_gold"]) & (~q_plus_rag["acc_ok"])]
if len(c3) > 0:
    print_case(
        "Случай 3: Обоснование передано целиком, но ответ неверен / 情况 3：证据已传入但模型判断出错",
        c3.iloc[0],
    )

# 情况 4：标准标签为 NOINFO（无证据），但模型产生幻觉错误引用了无关文档
# Случай 4: Ошибка на утверждении NOINFO (ложное срабатывание / галлюцинация ссылки)
c4 = rag3[(rag3["gold_label"] == "NOINFO") & (~rag3["acc_ok"])]
if len(c4) > 0:
    print_case(
        "Случай 4: Ошибка на утверждении NOINFO / 情况 4：无证据论断上的误判与幻觉引用",
        c4.iloc[0],
    )