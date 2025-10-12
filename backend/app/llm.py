import os, httpx, time
import re

BASE = os.environ.get("GROQ_API_BASE", "https://api.groq.com")
MODEL = os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant")
API_KEY = os.environ.get("GROQ_API_KEY", "")

SYSTEM = """You are a precise tutor. Use ONLY the provided CONTEXT to answer.
CRITICAL: Cite immediately after the specific sentence/claim using EXACT tokens:
- For slides: [Lecture {n} Slide {m}]
- For lecture notes: [Lecture {n} Notes]
- For global KB: [Global]
- For user memory: [User]
Citations MUST sit right next to the claim they support. Prefer multiple short, inline citations rather than one big list at the end.
If the context is insufficient, say so explicitly."""

_LEC_SLIDE = re.compile(r"\[LEC\s+lec_(\d+)\s*/\s*SLIDE\s+(\d+)\]", re.I)
_LEC_NOTE  = re.compile(r"\[LEC\s+lec_(\d+)\s*/\s*LECTURE NOTE\]", re.I)
_GLOBAL    = re.compile(r"\[GLOBAL\]", re.I)
_USER      = re.compile(r"\[USER\]", re.I)

def normalize_citations(txt: str) -> str:
    txt = _LEC_SLIDE.sub(lambda m: f"[Lecture {m.group(1)} Slide {m.group(2)}]", txt)
    txt = _LEC_NOTE.sub(lambda m: f"[Lecture {m.group(1)} Notes]", txt)
    txt = _GLOBAL.sub("[Global]", txt)
    txt = _USER.sub("[Readings]", txt)
    return txt


def _post_json(url: str, payload: dict) -> httpx.Response:
    with httpx.Client(
        timeout=45.0,
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        },
    ) as c:
        return c.post(url, json=payload)

def answer_with_ctx(question: str, context: str, max_tokens: int = 600) -> dict:
    if not API_KEY:
        return {"answer": "(LLM disabled: missing GROQ_API_KEY)", "model": MODEL, "usage": {}}

    payload = {
        "model": MODEL,
        "temperature": 0.2,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"CONTEXT:\n{context}\n\nQUESTION: {question}"},
        ],
    }

    t0 = time.time()

    # Try OpenAI-compatible path first
    url1 = f"{BASE}/openai/v1/chat/completions"
    r = _post_json(url1, payload)
    if r.status_code == 404:
        # Fallback to plain /v1 if an intermediate proxy strips '/openai'
        url2 = f"{BASE}/v1/chat/completions"
        r = _post_json(url2, payload)

    if r.status_code >= 400:
        return {
            "answer": f"(LLM error: {r.status_code}) {r.text[:400]}",
            "model": MODEL,
            "usage": {},
            "latency_s": round(time.time() - t0, 3),
            "endpoint": r.request.url.__str__(),
        }

    data = r.json()
    content = data["choices"][0]["message"]["content"]
    content = normalize_citations(content)  # <— add this line
    usage = data.get("usage", {})
    return {
        "answer": content,
        "usage": usage,
        "latency_s": round(time.time() - t0, 3),
        "model": MODEL,
        "endpoint": r.request.url.__str__(),
    }


def groq_get_models() -> dict:
    if not API_KEY:
        return {"error": "missing GROQ_API_KEY"}
    url = f"{BASE}/openai/v1/models"  # OpenAI-compatible discovery
    with httpx.Client(timeout=20.0, headers={"Authorization": f"Bearer {API_KEY}"}) as c:
        r = c.get(url)
        return {"status": r.status_code, "text": r.text}
