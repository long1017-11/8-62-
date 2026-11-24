import os
import re
import json
import math
import time
import hashlib
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification

os.makedirs("lr3_artifacts", exist_ok=True)


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# 1. 加载 ЛР2 数据、严格划分清单与向量索引 / Загрузка данных, разбиения и индекса из ЛР2
corpus = load_jsonl("lr2_artifacts/scifact_raw/corpus.jsonl")
corpus_map = {str(d["doc_id"]): d for d in corpus}
claims_dev_map = {c["id"]: c for c in load_jsonl("lr2_artifacts/scifact_raw/claims_dev.jsonl")}

with open("lr2_artifacts/manifest.json", "r", encoding="utf-8") as f:
    manifest = json.load(f)
with open("lr2_artifacts/dev_top5_retrieval.json", "r", encoding="utf-8") as f:
    lr2_dev_top5 = json.load(f)
with open("lr2_artifacts/doc_ids.json", "r", encoding="utf-8") as f:
    doc_ids = json.load(f)

doc_embeddings = np.load("lr2_artifacts/corpus_embeddings.npy")
dev_ids = manifest["split"]["dev_ids"]
holdout_ids = manifest["split"]["holdout_ids"]


# 2. 纯 Python/NumPy 精确实现 Lucene BM25（严格符合附录分词与公式）
# Точная реализация BM25 (вариант Lucene, k1=1.2, b=0.75) по спецификации ЛР3
def bm25_tokenize(text: str) -> list:
    """附录规定分词：小写 + 连续拉丁字母与数字，无词干还原与停用词 / Токенизация из приложения."""
    return re.findall(r"[a-z0-9]+", text.lower())


class LuceneBM25:
    def __init__(self, docs: list, k1: float = 1.2, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_ids = [str(d["doc_id"]) for d in docs]
        self.N = len(docs)
        self.doc_lens = np.zeros(self.N, dtype=np.float64)
        self.inverted_index = {}  # term -> list of (doc_idx, tf)
        df = {}

        for idx, d in enumerate(docs):
            full_text = f"{d['title']}\n{' '.join(d['abstract'])}"
            tokens = bm25_tokenize(full_text)
            self.doc_lens[idx] = len(tokens)
            tf_map = {}
            for t in tokens:
                tf_map[t] = tf_map.get(t, 0) + 1
            for t, tf in tf_map.items():
                self.inverted_index.setdefault(t, []).append((idx, tf))
                df[t] = df.get(t, 0) + 1

        self.avgdl = float(np.mean(self.doc_lens)) if self.N > 0 else 0.0
        # Lucene IDF 公式: ln(1 + (N - n_t + 0.5) / (n_t + 0.5))
        self.idf = {
            t: math.log(1.0 + (self.N - n_t + 0.5) / (n_t + 0.5))
            for t, n_t in df.items()
        }

    def search(self, query: str, top_k: int = 20) -> list:
        """按查询去重词项计算 float64 得分，返回得分 > 0 的前 top_k 个文档 / Поиск BM25 по уникальным словам запроса."""
        q_terms = list(dict.fromkeys(bm25_tokenize(query)))  # 去重且保持顺序 / уникальные слова запроса
        scores = np.zeros(self.N, dtype=np.float64)
        for t in q_terms:
            if t not in self.inverted_index:
                continue
            idf_t = self.idf[t]
            for doc_idx, tf in self.inverted_index[t]:
                dl = self.doc_lens[doc_idx]
                denom = tf + self.k1 * (1.0 - self.b + self.b * (dl / self.avgdl))
                scores[doc_idx] += idf_t * (tf * (self.k1 + 1.0)) / denom

        pos_indices = np.where(scores > 0.0)[0]
        if len(pos_indices) == 0:
            return []
        # 同分时按 int(doc_id) 升序排列 / При равных оценках первым идёт меньший int(doc_id)
        ranked = sorted(
            pos_indices,
            key=lambda i: (-float(scores[i]), int(self.doc_ids[i])),
        )[:top_k]
        return [
            {"doc_id": self.doc_ids[i], "score": round(float(scores[i]), 6), "rank": r + 1}
            for r, i in enumerate(ranked)
        ]


# 3. RRF 倒数排名融合（完全遵循附录函数逻辑与同分按 int(doc_id) 排序）
# Объединение выдач RRF по рангу (c=60, depth=20)
def rrf_fusion(dense_list: list, bm25_list: list, depth: int = 20, c: int = 60, top: int = 20) -> list:
    scores = {}
    contrib = {}
    for m_name, ranked in (("dense", dense_list[:depth]), ("bm25", bm25_list[:depth])):
        for rank, item in enumerate(ranked, start=1):
            did = item["doc_id"]
            val = 1.0 / (c + rank)
            scores[did] = scores.get(did, 0.0) + val
            contrib.setdefault(did, {"dense_rank": None, "bm25_rank": None, "dense_rrf": 0.0, "bm25_rrf": 0.0})
            contrib[did][f"{m_name}_rank"] = rank
            contrib[did][f"{m_name}_rrf"] = round(val, 6)

    sorted_dids = sorted(scores.keys(), key=lambda d: (-scores[d], int(d)))[:top]
    return [
        {
            "doc_id": d,
            "score": round(scores[d], 6),
            "rank": r + 1,
            "contrib": contrib[d],
        }
        for r, d in enumerate(sorted_dids)
    ]


# 4. 加载 Dense 编码器与 Cross-Encoder 模型 / Загрузка Dense-кодировщика и Cross-Encoder
ENCODER_NAME = "BAAI/bge-small-en-v1.5"
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
dense_encoder = SentenceTransformer(ENCODER_NAME, device="cpu")

CE_NAME = "cross-encoder/ms-marco-MiniLM-L6-v2"
ce_device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Загрузка Cross-Encoder ({CE_NAME}) на {ce_device}... / 加载重排模型...")
ce_tokenizer = AutoTokenizer.from_pretrained(CE_NAME)
ce_model = AutoModelForSequenceClassification.from_pretrained(CE_NAME).to(ce_device).eval()


def dense_search(query: str, top_k: int = 20):
    q_vec = dense_encoder.encode([BGE_QUERY_PREFIX + query], normalize_embeddings=True).astype(np.float32)[0]
    scores = doc_embeddings @ q_vec
    # 同分按 int(doc_id) 升序 / При равных оценках первым идёт меньший doc_id
    top_idx = sorted(range(len(doc_ids)), key=lambda i: (-float(scores[i]), int(doc_ids[i])))[:top_k]
    return [
        {"doc_id": doc_ids[i], "score": round(float(scores[i]), 6), "rank": r + 1}
        for r, i in enumerate(top_idx)
    ]


def rerank_with_ce(query: str, rrf_candidates: list) -> list:
    if not rrf_candidates:
        return []
    docs_texts = [
        f"{corpus_map[item['doc_id']]['title']}\n{' '.join(corpus_map[item['doc_id']]['abstract'])}"
        for item in rrf_candidates
    ]
    batch = ce_tokenizer(
        [query] * len(docs_texts),
        docs_texts,
        truncation="only_second",
        max_length=512,
        padding=True,
        return_tensors="pt",
    ).to(ce_device)
    with torch.no_grad():
        logits = ce_model(**batch).logits.squeeze(-1).detach().cpu().numpy().tolist()
    if isinstance(logits, float):
        logits = [logits]

    paired = list(zip(rrf_candidates, logits))
    paired.sort(key=lambda x: (-float(x[1]), int(x[0]["doc_id"])))
    return [
        {
            "doc_id": item["doc_id"],
            "score": round(float(ce_score), 6),
            "rank": r + 1,
            "rrf_rank": item["rank"],
            "contrib": item["contrib"],
        }
        for r, (item, ce_score) in enumerate(paired)
    ]


# 5. 构建完整语料 BM25 索引并执行第 3.3 节 4 项运行前校验
# Построение индекса BM25 и выполнение 4 проверок перед опытами (п. 3.3)
print("Построение индекса BM25 по 5183 аннотациям... / 正在构建 BM25 倒排索引...")
bm25_index = LuceneBM25(corpus, k1=1.2, b=0.75)

print("--- Проверка 3.3 перед опытами / 运行前 4 项逻辑校验 ---")
# 3.3.1 玩具语料手工公式校验（含重复词与无匹配查询） / Ручная проверка формулы BM25 на 3 коротких документах
toy_docs = [
    {"doc_id": "1", "title": "alpha beta", "abstract": ["alpha"]},  # alpha=2, beta=1, dl=3
    {"doc_id": "2", "title": "beta gamma", "abstract": []},         # beta=1, gamma=1, dl=2
    {"doc_id": "3", "title": "delta", "abstract": []},              # delta=1, dl=1
]
toy_bm25 = LuceneBM25(toy_docs, k1=1.2, b=0.75)
# avgdl = 2.0; query = "alpha alpha" (去重后仅算一次 alpha): N=3, n_t=1 -> idf = ln(1 + 2.5/1.5) = ln(8/3)
expected_idf = math.log(1.0 + (3 - 1 + 0.5) / (1 + 0.5))
expected_s1 = expected_idf * (2 * 2.2) / (2 + 1.2 * (1 - 0.75 + 0.75 * (3 / 2.0)))
res_toy = toy_bm25.search("alpha alpha", top_k=3)
assert len(res_toy) == 1 and res_toy[0]["doc_id"] == "1" and abs(res_toy[0]["score"] - expected_s1) < 1e-5
assert toy_bm25.search("nonexistentword", top_k=3) == []

# 3.3.2 RRF 玩具示例：双榜同时出现的文档超越单榜第 1 名 / Проверка RRF на игрушечном примере
l_a = [{"doc_id": "10", "score": 0.9}, {"doc_id": "20", "score": 0.8}]
l_b = [{"doc_id": "30", "score": 5.0}, {"doc_id": "20", "score": 4.0}]
rrf_toy = rrf_fusion(l_a, l_b, depth=20, c=60, top=3)
assert rrf_toy[0]["doc_id"] == "20"  # 1/62 + 1/62 > 1/61

# 3.3.3 重排前后 Top-20 候选集合严格一致 / Множество 20 кандидатов после CE не меняется
sample_claim = claims_dev_map[dev_ids[0]]
d20_0 = dense_search(sample_claim["claim"], 20)
b20_0 = bm25_index.search(sample_claim["claim"], 20)
rrf20_0 = rrf_fusion(d20_0, b20_0, 20, 60, 20)
ce20_0 = rerank_with_ce(sample_claim["claim"], rrf20_0)
assert {x["doc_id"] for x in rrf20_0} == {x["doc_id"] for x in ce20_0}

# 3.3.4 Dense 前 5 篇与 ЛР2 严格一致 / Первые 5 результатов Dense совпадают с ЛР2
for cid in dev_ids:
    d_now = [x["doc_id"] for x in dense_search(claims_dev_map[cid]["claim"], 5)]
    d_lr2 = [x["doc_id"] for x in lr2_dev_top5[str(cid)]]
    assert d_now == d_lr2, f"Mismatch on claim {cid}: {d_now} vs {d_lr2}"
print("✅ Все 4 предварительные проверки 3.3 пройдены! / 全部 4 项预检均已通过！\n")


# 6. 执行全量检索与计时统计（dev 90 条 + holdout 210 条）
# Выполнение поиска всеми 4 методами и замер времени (медианы и диапазоны)
def run_retrieval_for_split(split_name: str, id_list: list):
    records = {}
    timings = {"dense_ms": [], "bm25_ms": [], "rrf_ms": [], "ce_ms": []}

    for cid in id_list:
        q = claims_dev_map[cid]["claim"]

        t0 = time.perf_counter()
        d20 = dense_search(q, 20)
        t_dense = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        b20 = bm25_index.search(q, 20)
        t_bm25 = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        rrf20 = rrf_fusion(d20, b20, 20, 60, 20)
        t_rrf = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        ce20 = rerank_with_ce(q, rrf20)
        t_ce = (time.perf_counter() - t0) * 1000.0

        timings["dense_ms"].append(t_dense)
        timings["bm25_ms"].append(t_bm25)
        timings["rrf_ms"].append(t_rrf)
        timings["ce_ms"].append(t_ce)

        records[str(cid)] = {
            "dense": d20,
            "bm25": b20,
            "rrf": rrf20,
            "rrf_ce": ce20,
            "timing_ms": {
                "dense": round(t_dense, 3),
                "bm25": round(t_bm25, 3),
                "rrf": round(t_rrf, 3),
                "ce": round(t_ce, 3),
            },
        }

    with open(f"lr3_artifacts/{split_name}_all_retrieval.json", "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

    if split_name == "dev":
        t_rows = []
        for stage, label in (
            ("dense_ms", "Dense (кодирование + поиск)"),
            ("bm25_ms", "BM25 (поиск по словам)"),
            ("rrf_ms", "Слияние RRF"),
            ("ce_ms", "Переранжирование Cross-Encoder"),
        ):
            arr = np.array(timings[stage])
            t_rows.append(
                {
                    "Этап": label,
                    "Медиана (мс)": round(float(np.median(arr)), 3),
                    "Мин (мс)": round(float(np.min(arr)), 3),
                    "Макс (мс)": round(float(np.max(arr)), 3),
                }
            )
        df_t = pd.DataFrame(t_rows)
        df_t.to_csv("lr3_artifacts/dev_timings.csv", index=False)
        print("=== Время поиска на запрос (dev, 90 запросов) / 检索各阶段耗时统计 ===")
        print(df_t.to_string(index=False))
        print()

    return records


print("Запуск поиска на dev (90)... / 正在计算 dev 集四路检索...")
dev_ret = run_retrieval_for_split("dev", dev_ids)
print("Запуск поиска на holdout (210)... / 正在计算 holdout 集四路检索...")
holdout_ret = run_retrieval_for_split("holdout", holdout_ids)


# 7. 计算 dev 检索指标、互补性矩阵，并按第 2.4 节规则锁定 selection.json
# Расчёт метрик поиска на dev, таблицы дополнительности и фиксация selection.json
def eval_retrieval_metrics(id_list: list, ret_dict: dict):
    q_plus = [claims_dev_map[cid] for cid in id_list if claims_dev_map[cid].get("evidence")]
    rows = []
    for method in ("dense", "bm25", "rrf", "rrf_ce"):
        h3, r3, mrr3, r20 = [], [], [], []
        for c in q_plus:
            rel = set(c["evidence"].keys())
            ranked_ids = [x["doc_id"] for x in ret_dict[str(c["id"])][method]]
            top3 = ranked_ids[:3]
            top20 = ranked_ids[:20]

            h3.append(1.0 if (rel & set(top3)) else 0.0)
            r3.append(len(rel & set(top3)) / len(rel))
            r20.append(len(rel & set(top20)) / len(rel))
            rr = 0.0
            for idx, did in enumerate(top3, start=1):
                if did in rel:
                    rr = 1.0 / idx
                    break
            mrr3.append(rr)
        rows.append(
            {
                "Метод": method,
                "Hit@3": round(float(np.mean(h3)), 4),
                "Recall@3": round(float(np.mean(r3)), 4),
                "MRR@3": round(float(np.mean(mrr3)), 4),
                "Recall@20": round(float(np.mean(r20)), 4),
            }
        )
    return pd.DataFrame(rows)


df_dev_ret = eval_retrieval_metrics(dev_ids, dev_ret)
df_dev_ret.to_csv("lr3_artifacts/dev_retrieval_metrics.csv", index=False)
print("=== Таблица поиска на dev (Q+, 60 утверждений) / dev 集四种检索方法对比 ===")
print(df_dev_ret.to_string(index=False))
print()

# 互补性统计表 (Эксперимент 1: Таблица дополнительности на глубине 3 и 20)
q_plus_dev = [claims_dev_map[cid] for cid in dev_ids if claims_dev_map[cid].get("evidence")]
comp_rows = []
for depth in (3, 20):
    groups = {"Оба": [0, 0], "Только Dense": [0, 0], "Только BM25": [0, 0], "Ни один": [0, 0]}
    for c in q_plus_dev:
        rel = set(c["evidence"].keys())
        item = dev_ret[str(c["id"])]
        in_d = bool(rel & {x["doc_id"] for x in item["dense"][:depth]})
        in_b = bool(rel & {x["doc_id"] for x in item["bm25"][:depth]})
        in_rrf_top3 = bool(rel & {x["doc_id"] for x in item["rrf"][:3]})

        if in_d and in_b:
            g = "Оба"
        elif in_d:
            g = "Только Dense"
        elif in_b:
            g = "Только BM25"
        else:
            g = "Ни один"
        groups[g][0] += 1
        if in_rrf_top3:
            groups[g][1] += 1
    comp_rows.append(
        {
            "Глубина": depth,
            "Оба (всего / в RRF@3)": f"{groups['Оба'][0]} / {groups['Оба'][1]}",
            "Только Dense (всего / в RRF@3)": f"{groups['Только Dense'][0]} / {groups['Только Dense'][1]}",
            "Только BM25 (всего / в RRF@3)": f"{groups['Только BM25'][0]} / {groups['Только BM25'][1]}",
            "Ни один (всего / в RRF@3)": f"{groups['Ни один'][0]} / {groups['Ни один'][1]}",
        }
    )
df_comp = pd.DataFrame(comp_rows)
df_comp.to_csv("lr3_artifacts/dev_complementarity.csv", index=False)
print("=== Таблица дополнительности Dense и BM25 на dev / 检索互补性统计表 ===")
print(df_comp.to_string(index=False))
print()

# 按第 2.4 节事前选型规则选择最优方案（在 BM25, RRF, RRF+CE 中选 MRR@3 最高者）
# Применение правила выбора (п. 2.4) и сохранение selection.json ДО запуска на holdout
simplicity_rank = {"bm25": 3, "rrf": 2, "rrf_ce": 1}
candidates_df = df_dev_ret[df_dev_ret["Метод"].isin(["bm25", "rrf", "rrf_ce"])].copy()
candidates_df["simplicity"] = candidates_df["Метод"].map(simplicity_rank)
candidates_df = candidates_df.sort_values(by=["MRR@3", "Recall@3", "simplicity"], ascending=[False, False, False])
best_row = candidates_df.iloc[0]
selected_method = str(best_row["Метод"])

config_str = json.dumps(
    {"bm25": {"k1": 1.2, "b": 0.75}, "rrf": {"depth": 20, "c": 60}, "ce": CE_NAME, "budget": 2048, "k": 3},
    sort_keys=True,
)
selection_obj = {
    "selected_method": selected_method,
    "selection_rule": "max MRR@3 on dev among {bm25, rrf, rrf_ce}, tie-break by Recall@3, then simplicity",
    "dev_metrics": {
        "Hit@3": float(best_row["Hit@3"]),
        "Recall@3": float(best_row["Recall@3"]),
        "MRR@3": float(best_row["MRR@3"]),
        "Recall@20": float(best_row["Recall@20"]),
    },
    "config_sha256": hashlib.sha256(config_str.encode()).hexdigest(),
}
with open("lr3_artifacts/selection.json", "w", encoding="utf-8") as f:
    json.dump(selection_obj, f, indent=2, ensure_ascii=False)

print(f"✅ Шаг 1 завершён! Выбранный по dev вариант: {selected_method} (MRR@3 = {best_row['MRR@3']}), сохранено в lr3_artifacts/selection.json")