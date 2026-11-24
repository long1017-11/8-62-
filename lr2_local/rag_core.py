import json
import time
import requests
from openai import OpenAI

# 本地 llama-server API 配置（使用 127.0.0.1 避免 Windows IPv6 回退延迟，使用 Session 复用 TCP 连接）
# Конфигурация локального сервера llama.cpp (используем 127.0.0.1 и requests.Session для переиспользования соединений)
SERVER_ROOT = "http://127.0.0.1:8000"
BASE_URL = f"{SERVER_ROOT}/v1"
MODEL_NAME = "Qwen_Qwen3-4B-Instruct-2507-Q4_K_M.gguf"
TOKEN_BUDGET = 2048  # 输入上下文最大预算（tokens） / Бюджет входного контекста в токенах

session = requests.Session()
client = OpenAI(base_url=BASE_URL, api_key="none")

# 官方附录严格 JSON Schema（用于 structured outputs 约束解码）
# Строгая схема JSON из приложения (для параметра response_format)
RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "verdict",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "required": ["label", "doc_id", "sentences"],
            "properties": {
                "label": {"enum": ["SUPPORT", "CONTRADICT", "NOINFO"]},
                "doc_id": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                "sentences": {"type": "array", "items": {"type": "integer"}},
            },
        },
    },
}

# 英文系统提示词（严格区分 RAG 仅依据上下文 vs LLM-only 依据自身知识）
# Системные промпты на английском языке для режимов RAG и LLM-only
PROMPTS = {
    "rag_system": (
        "You are a scientific fact-checking assistant. Verify the given claim STRICTLY using ONLY "
        "the provided document abstracts. Do NOT use outside knowledge.\n"
        "Labels:\n"
        "- SUPPORT: the provided context explicitly supports the claim.\n"
        "- CONTRADICT: the provided context explicitly refutes/contradicts the claim.\n"
        "- NOINFO: the provided context does not contain enough evidence to support or contradict the claim.\n"
        "Output a valid JSON object matching the schema. For SUPPORT or CONTRADICT, cite the exact doc_id "
        "and the 0-based sentence indices from that document containing the evidence. For NOINFO, set doc_id to null and sentences to []."
    ),
    "llm_only_system": (
        "You are a scientific fact-checking assistant. Verify the given claim using your scientific knowledge.\n"
        "Labels:\n"
        "- SUPPORT: the claim is scientifically supported.\n"
        "- CONTRADICT: the claim is scientifically false/contradicted.\n"
        "- NOINFO: there is insufficient established evidence to determine whether it is true or false.\n"
        "Output a valid JSON object matching the schema. Since no documents are provided, set doc_id to null and sentences to []."
    ),
}


def count_tokens(text: str) -> int:
    """调用本地 /tokenize 接口精确计算 token 数量（复用 Session 连接池） / Точный подсчёт токенов через /tokenize."""
    resp = session.post(f"{SERVER_ROOT}/tokenize", json={"content": text}, timeout=30)
    resp.raise_for_status()
    return len(resp.json()["tokens"])


def render_chat_prompt(messages: list) -> str:
    """还原 Qwen3 ChatML 对话模板以计算真实输入长度 / Применение шаблона диалога ChatML для точного учёта токенов."""
    parts = []
    for m in messages:
        parts.append(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n")
    parts.append("<|im_start|>assistant\n")
    return "".join(parts)


def build_context_within_budget(claim_text: str, candidate_docs: list, budget: int = TOKEN_BUDGET):
    """
    在 2048 token 预算内按排名逐篇、逐句组装上下文（第 3.3 节规则）：
    如果某篇文档无法完整放入，则尽量放入能容纳的前几句并立即停止。
    Сборка контекста в пределах бюджета токенов по рангу документов и целым предложениям (п. 3.3):
    если очередной документ не помещается целиком, добавляются поместившиеся предложения, и сборка останавливается.
    """
    sys_msg = {"role": "system", "content": PROMPTS["rag_system"]}
    header = f"Claim: {claim_text}\n\nRetrieved Documents:\n"
    footer = "\nProvide your verdict in JSON format:"

    base_msgs = [sys_msg, {"role": "user", "content": header + footer}]
    used_tokens = count_tokens(render_chat_prompt(base_msgs))

    context_blocks = []
    passed_docs = {}  # 记录实际传入提示词的文档及其句子编号 / Словарь реально переданных документов и индексов предложений

    for doc in candidate_docs:
        did = str(doc["doc_id"])
        doc_header = f"\n[Document doc_id={did}]\nTitle: {doc['title']}\nSentences:\n"
        h_toks = count_tokens(doc_header)
        if used_tokens + h_toks > budget:
            break

        added_sents_lines = []
        added_indices = []
        cur_toks = used_tokens + h_toks
        stop_all = False

        for idx, sent in enumerate(doc["abstract"]):
            line = f"[{idx}] {sent}\n"
            s_toks = count_tokens(line)
            if cur_toks + s_toks <= budget:
                added_sents_lines.append(line)
                added_indices.append(idx)
                cur_toks += s_toks
            else:
                # 预算已满，截断后续句子并停止添加新文档 / Бюджет исчерпан — останавливаем добавление
                stop_all = True
                break

        if added_indices:
            context_blocks.append(doc_header + "".join(added_sents_lines))
            passed_docs[did] = added_indices
            used_tokens = cur_toks

        if stop_all:
            break

    full_user = header + "".join(context_blocks) + footer
    final_msgs = [sys_msg, {"role": "user", "content": full_user}]
    final_tokens = count_tokens(render_chat_prompt(final_msgs))
    return final_msgs, passed_docs, final_tokens


def check_contains_gold_rationale(claim: dict, passed_docs: dict) -> bool:
    """
    检查实际传入的文本中是否完整包含至少一组标准答案的论证句子（содержит обоснование）：
    Проверка, содержится ли в переданном контексте целиком хотя бы одно эталонное обоснование.
    """
    evidence = claim.get("evidence", {})
    for did, ev_list in evidence.items():
        if did not in passed_docs:
            continue
        passed_set = set(passed_docs[did])
        for ev in ev_list:
            gold_sents = set(ev["sentences"])
            if gold_sents.issubset(passed_set):
                return True
    return False


def parse_and_validate_response(raw_text: str, mode: str, passed_docs: dict):
    """
    解析并校验模型输出的 JSON（检查字段名、标签合法性、引用文档及句子是否存在于已传入上下文中）：
    Разбор и семантическая валидация JSON-ответа модели (проверка ключей, метки, принадлежности документа и предложений).
    """
    try:
        obj = json.loads(raw_text)
    except Exception as e:
        return None, f"json_decode_error: {e}"

    if not isinstance(obj, dict) or set(obj.keys()) != {"label", "doc_id", "sentences"}:
        return None, "invalid_keys"

    label = obj["label"]
    doc_id = obj["doc_id"]
    sents = obj["sentences"]

    if label not in {"SUPPORT", "CONTRADICT", "NOINFO"}:
        return None, "invalid_label"
    if not isinstance(sents, list) or not all(isinstance(x, int) and x >= 0 for x in sents):
        return None, "invalid_sentences_type"

    # LLM-only 模式下不要求提供引用 / В режиме LLM-only ссылки на документы не требуются
    if mode == "llm_only":
        return {"label": label, "doc_id": None, "sentences": []}, "ok"

    # NOINFO 必须对应 doc_id=null 且 sentences=[] / Для NOINFO требуются null и пустой список
    if label == "NOINFO":
        if doc_id is not None or len(sents) != 0:
            return None, "noinfo_must_have_null_and_empty"
        return obj, "ok"

    # RAG / Oracle 下的 SUPPORT 或 CONTRADICT 必须引用已实际传入的文档和句子编号
    # Для SUPPORT и CONTRADICT проверяем, что doc_id и номера предложений реально были переданы в промпте
    if not isinstance(doc_id, str) or doc_id not in passed_docs:
        return None, "cited_doc_not_in_passed"
    if len(sents) == 0 or not set(sents).issubset(set(passed_docs[doc_id])):
        return None, "cited_sentence_not_in_passed"

    return obj, "ok"


def call_llm_verdict(messages: list, mode: str, passed_docs: dict):
    """调用本地大模型接口执行确定性生成 (temperature=0, max_tokens=128) / Вызов модели со structured outputs."""
    t0 = time.perf_counter()
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            max_tokens=128,
            temperature=0.0,
            response_format=RESPONSE_FORMAT,
        )
        gen_time = time.perf_counter() - t0
        raw_text = resp.choices[0].message.content or ""
        prompt_toks = resp.usage.prompt_tokens if resp.usage else 0
        parsed, status = parse_and_validate_response(raw_text, mode, passed_docs)
        return raw_text, parsed, status, prompt_toks, round(gen_time, 4)
    except Exception as e:
        return "", None, f"api_error: {e}", 0, round(time.perf_counter() - t0, 4)