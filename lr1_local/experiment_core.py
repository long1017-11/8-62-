import os
import time
import json
import hashlib
import urllib.request
import requests
import nltk
from openai import OpenAI, AsyncOpenAI

MODEL_NAME = "Qwen_Qwen3-4B-Instruct-2507-Q4_K_M.gguf"
SERVER_ROOT = "http://localhost:8000"
BASE_URL = f"{SERVER_ROOT}/v1"

client = OpenAI(base_url=BASE_URL, api_key="none")
async_client = AsyncOpenAI(base_url=BASE_URL, api_key="none")

def server_tokenize(text: str) -> list:
    """调用本地 llama-server 的 /tokenize 接口精确分词（无需联网下载 HuggingFace 分词器）"""
    resp = requests.post(f"{SERVER_ROOT}/tokenize", json={"content": text}, timeout=30)
    resp.raise_for_status()
    return resp.json()["tokens"]

def server_detokenize(token_ids: list) -> str:
    """调用本地 llama-server 的 /detokenize 接口还原文本"""
    resp = requests.post(f"{SERVER_ROOT}/detokenize", json={"tokens": token_ids}, timeout=30)
    resp.raise_for_status()
    return resp.json()["content"]

def prepare_dataset():
    """加载《Pride and Prejudice》并生成带 SHA-256 的数据清单"""
    os.makedirs("lr1_artifacts", exist_ok=True)
    txt_path = "lr1_artifacts/pride_and_prejudice.txt"
    source_url = "https://www.gutenberg.org/files/1342/1342-0.txt"

    if not os.path.exists(txt_path) or os.path.getsize(txt_path) < 100000:
        try:
            req = urllib.request.Request(
                "https://www.gutenberg.org/cache/epub/1342/pg1342.txt",
                headers={"User-Agent": "Mozilla/5.0"}
            )
            with urllib.request.urlopen(req, timeout=15) as resp, open(txt_path, "wb") as out:
                out.write(resp.read())
        except Exception:
            nltk.download("gutenberg", quiet=True)
            austen_files = [f for f in nltk.corpus.gutenberg.fileids() if "austen" in f]
            full_text = "\n\n".join(nltk.corpus.gutenberg.raw(f) for f in austen_files)
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(full_text)
            source_url += " (NLTK Gutenberg fallback)"

    with open(txt_path, "rb") as f:
        raw_bytes = f.read()
    sha256_hash = hashlib.sha256(raw_bytes).hexdigest()
    text = raw_bytes.decode("utf-8", errors="ignore")

    start_idx = text.find("Chapter 1")
    clean_text = text[start_idx:] if start_idx != -1 else text[2000:]

    tokens = server_tokenize(clean_text)
    manifest = {
        "source": source_url,
        "file": txt_path,
        "sha256": sha256_hash,
        "total_tokens": len(tokens)
    }
    with open("lr1_artifacts/data_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"Манифест данных сохранён: SHA-256 = {sha256_hash}, токенов = {len(tokens)}")
    return tokens

def build_prompt(tokens, target_prompt_tokens, unique_id, offset=0):
    """构造指定 token 长度的请求（开头带唯一标记防止前缀命中）"""
    prefix_text = f"[ReqID-{unique_id}] Summarize the following text in one sentence:\n\n"
    # Qwen3 ChatML 模板固定开销约 14 tokens + prefix_text token 数
    overhead_len = len(server_tokenize(prefix_text)) + 14

    body_len = max(16, target_prompt_tokens - overhead_len)
    start = (offset * 500) % max(1, len(tokens) - body_len - 10)
    body_text = server_detokenize(tokens[start : start + body_len])

    return [{"role": "user", "content": prefix_text + body_text}]

def timed_chat(messages, max_tokens, req_id="test", condition="default", ignore_eos=True, temperature=0.0):
    """流式请求并精确计算 TTFT、TPOT 和 E2E"""
    t0 = time.perf_counter()
    stream = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        stream=True,
        stream_options={"include_usage": True},
        extra_body={"ignore_eos": ignore_eos, "cache_prompt": False}
    )
    t1 = t_last = None
    usage = None
    chunks_text = []
    finish_reason = None

    for chunk in stream:
        if chunk.usage is not None:
            usage = chunk.usage
        if chunk.choices:
            if chunk.choices[0].finish_reason:
                finish_reason = chunk.choices[0].finish_reason
            delta = chunk.choices[0].delta.content
            if delta:
                now = time.perf_counter()
                if t1 is None:
                    t1 = now
                t_last = now
                chunks_text.append(delta)

    n = usage.completion_tokens if usage else 0
    ttft = (t1 - t0) if t1 else 0.0
    tpot = ((t_last - t1) / (n - 1)) if (n > 1 and t_last and t1) else 0.0
    e2e = (t_last - t0) if t_last else 0.0

    record = {
        "req_id": req_id,
        "condition": condition,
        "prompt_tokens": usage.prompt_tokens if usage else 0,
        "output_tokens": n,
        "ttft_s": round(ttft, 6),
        "tpot_s": round(tpot, 6),
        "e2e_s": round(e2e, 6),
        "finish_reason": finish_reason,
        "status": "ok" if n > 0 else "error",
        "text": "".join(chunks_text)
    }
    with open("lr1_artifacts/results.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return record