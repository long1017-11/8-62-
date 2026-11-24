import os
import json
import tarfile
import hashlib
import urllib.request
import numpy as np
from sentence_transformers import SentenceTransformer

# 创建保存实验产物的目录 / Создаём директорию для сохранения артефактов эксперимента
os.makedirs("lr2_artifacts", exist_ok=True)
DATA_URL = "https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz"
TAR_PATH = "lr2_artifacts/data.tar.gz"
EXTRACT_DIR = "lr2_artifacts/scifact_raw"


def sha256_file(path):
    """计算文件的 SHA-256 哈希值 / Вычисляет контрольную сумму SHA-256 для файла."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# 1. 下载并解压官方 SciFact 数据集 / Скачивание и распаковка официального релиза SciFact
if not os.path.exists(TAR_PATH):
    print("Скачивание SciFact data.tar.gz... / 正在下载 SciFact 数据集...")
    urllib.request.urlretrieve(DATA_URL, TAR_PATH)

tar_sha256 = sha256_file(TAR_PATH)
print(f"SHA-256 архива / 压缩包哈希值: {tar_sha256}")

os.makedirs(EXTRACT_DIR, exist_ok=True)
with tarfile.open(TAR_PATH, "r:gz") as tar:
    for member in tar.getmembers():
        base = os.path.basename(member.name)
        # 跳过 macOS 隐藏的元数据文件 (._*) / Пропускаем служебные файлы macOS на '._'
        if base.startswith("._") or not base.endswith(".jsonl"):
            continue
        member.name = base
        tar.extract(member, EXTRACT_DIR, filter="data")


def load_jsonl(path):
    """读取 JSONL 文件并返回对象列表 / Загружает файл JSONL и возвращает список объектов."""
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


corpus = load_jsonl(f"{EXTRACT_DIR}/corpus.jsonl")
claims_train = load_jsonl(f"{EXTRACT_DIR}/claims_train.jsonl")
claims_dev = load_jsonl(f"{EXTRACT_DIR}/claims_dev.jsonl")


# 2. 严格按附录代码提取标签并划分 dev (90) 与 holdout (210)
# Определение метки утверждения и разбиение на dev (90) и holdout (210) строго по приложению
def claim_label(claim):
    if not claim.get("evidence"):
        return "NOINFO"
    labels = {r["label"] for rs in claim["evidence"].values() for r in rs}
    # 若存在冲突标签则返回 None 以便剔除 / Если встретятся обе метки, возвращаем None для исключения
    return labels.pop() if len(labels) == 1 else None


def split_dev(claims, per_class=30):
    by_class = {}
    excluded = []
    for c in claims:
        lbl = claim_label(c)
        if lbl is None:
            excluded.append(c["id"])
            continue
        by_class.setdefault(lbl, []).append(c["id"])
    dev_ids, holdout_ids = [], []
    # 按 SHA-256("scifact:<id>") 确定性排序，每类取前 30 条作为 dev / Детерминированная сортировка по SHA-256
    for label in ("SUPPORT", "CONTRADICT", "NOINFO"):
        ids = sorted(
            by_class[label],
            key=lambda i: hashlib.sha256(f"scifact:{i}".encode()).hexdigest(),
        )
        dev_ids += ids[:per_class]
        holdout_ids += ids[per_class:]
    return dev_ids, holdout_ids, excluded


dev_ids, holdout_ids, excluded_ids = split_dev(claims_dev, per_class=30)
print(
    f"Разбиение claims_dev / 数据集划分结果: dev={len(dev_ids)}, holdout={len(holdout_ids)}, excluded={len(excluded_ids)}"
)

# 3. 构建稠密向量索引 (在 CPU 上运行 BAAI/bge-small-en-v1.5)
# Построение плотного векторного индекса на CPU (кодировщик BAAI/bge-small-en-v1.5)
ENCODER_NAME = "BAAI/bge-small-en-v1.5"
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
print(f"Загрузка кодировщика {ENCODER_NAME} на CPU... / 正在 CPU 上加载编码器...")
encoder = SentenceTransformer(ENCODER_NAME, device="cpu")
max_seq_len = encoder.max_seq_length  # 512 tokens

doc_ids = []
doc_texts = []
truncated_count = 0

for doc in corpus:
    did = str(doc["doc_id"])
    # 文档文本格式：标题 + 换行 + 空格拼接的摘要句子 / Формат документа: заголовок + перевод строки + предложения аннотации
    full_text = f"{doc['title']}\n{' '.join(doc['abstract'])}"
    tok_ids = encoder.tokenizer.encode(full_text, add_special_tokens=True)
    # 统计超过编码器最大上下文长度 (512) 的摘要数量 / Подсчёт доли обрезанных аннотаций (> 512 токенов)
    if len(tok_ids) > max_seq_len:
        truncated_count += 1
    doc_ids.append(did)
    doc_texts.append(full_text)

trunc_ratio = truncated_count / len(corpus)
print(
    f"Доля обрезанных аннотаций / 被截断摘要比例 (> {max_seq_len} токенов): "
    f"{truncated_count}/{len(corpus)} ({trunc_ratio*100:.2f}%)"
)

# 预检：同一文本作为查询和文档时的余弦相似度应接近 1.0 / Предварительная проверка: близость текста с самим собой ≈ 1.0
v_q = encoder.encode([BGE_QUERY_PREFIX + doc_texts[0]], normalize_embeddings=True)[0]
v_d = encoder.encode([doc_texts[0]], normalize_embeddings=True)[0]
print(f"Проверка близости / 同文本相似度自检: {float(np.dot(v_q, v_d)):.4f}")

print("Кодирование корпуса (5183 аннотации)... / 正在编码语料库（5183 篇摘要）...")
doc_embeddings = encoder.encode(
    doc_texts, batch_size=64, normalize_embeddings=True, show_progress_bar=True
).astype(np.float32)
np.save("lr2_artifacts/corpus_embeddings.npy", doc_embeddings)
emb_sha256 = sha256_file("lr2_artifacts/corpus_embeddings.npy")

with open("lr2_artifacts/doc_ids.json", "w", encoding="utf-8") as f:
    json.dump(doc_ids, f)

# 4. 为 dev 集的 90 条论断预计算 Top-5 检索结果（供 k=1, 3, 5 实验复用）
# Предварительный расчёт Top-5 документов для 90 утверждений dev (переиспользуется для k=1, 3, 5)
dev_id_set = set(dev_ids)
dev_claims = [c for c in claims_dev if c["id"] in dev_id_set]
dev_claims.sort(key=lambda c: dev_ids.index(c["id"]))

query_texts = [BGE_QUERY_PREFIX + c["claim"] for c in dev_claims]
query_embeddings = encoder.encode(query_texts, normalize_embeddings=True).astype(np.float32)
# 向量已 L2 归一化，矩阵点积即余弦相似度 / Векторы нормализованы, скалярное произведение равно косинусной близости
scores_matrix = query_embeddings @ doc_embeddings.T  # (90, 5183)

retrieval_cache = {}
for idx, c in enumerate(dev_claims):
    row_scores = scores_matrix[idx]
    top5_idx = np.argsort(-row_scores)[:5]
    retrieval_cache[str(c["id"])] = [
        {"doc_id": doc_ids[i], "score": round(float(row_scores[i]), 6), "rank": r + 1}
        for r, i in enumerate(top5_idx)
    ]

with open("lr2_artifacts/dev_top5_retrieval.json", "w", encoding="utf-8") as f:
    json.dump(retrieval_cache, f, indent=2)

# 保存完整数据清单与索引护照 / Сохранение манифеста данных и паспорта индекса
manifest = {
    "dataset_source": DATA_URL,
    "archive_sha256": tar_sha256,
    "files_sha256": {
        "corpus.jsonl": sha256_file(f"{EXTRACT_DIR}/corpus.jsonl"),
        "claims_dev.jsonl": sha256_file(f"{EXTRACT_DIR}/claims_dev.jsonl"),
        "claims_train.jsonl": sha256_file(f"{EXTRACT_DIR}/claims_train.jsonl"),
    },
    "split": {
        "dev_count": len(dev_ids),
        "holdout_count": len(holdout_ids),
        "excluded_ids": excluded_ids,
        "dev_ids": dev_ids,
        "holdout_ids": holdout_ids,
    },
    "index_passport": {
        "encoder_model": ENCODER_NAME,
        "dimension": int(doc_embeddings.shape[1]),
        "max_seq_length": int(max_seq_len),
        "normalized": True,
        "query_prefix": BGE_QUERY_PREFIX,
        "passage_prefix": "",
        "corpus_size": len(corpus),
        "truncated_abstracts": truncated_count,
        "truncated_ratio": round(trunc_ratio, 4),
        "embeddings_sha256": emb_sha256,
    },
}
with open("lr2_artifacts/manifest.json", "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2, ensure_ascii=False)

print("✅ Шаг 1 завершён / 第 1 步完成：索引与数据清单已保存至 lr2_artifacts/")