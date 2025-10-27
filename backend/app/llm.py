import os, httpx, time
import re
from .settings import settings

BASE = os.environ.get("GROQ_API_BASE", "https://api.groq.com")
MODEL = os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant")
API_KEY = os.environ.get("GROQ_API_KEY", "")

GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"
GEMINI_KEY = "AIzaSyBfhGNJ5FOHgn0rY3zzjemEX20KKz18A5o"


SYSTEM = """You are a precise tutor. Use ONLY the provided CONTEXT to answer.
CRITICAL: Cite immediately after the specific sentence/claim using EXACT tokens:
- For slides: [Lecture {n} Slide {m}]
- For lecture notes: [Lecture {n} Notes]
- For readings: [From Lecture {n}]
- For global KB: [Global]
- For user memory: [User]  *THIS ONE MUST ONLY BE USED IF IT DOES NOT EXIST IN OTHER SOURCES, AND THIS IS NOT FOR LLM ANSWER BUT ONLY USER QUESTION/PROMPT/QUERY*
Citations MUST sit right next to the claim they support. Prefer multiple short, inline citations rather than one big list at the end.
If the context is insufficient, say so explicitly."""

_LEC_SLIDE = re.compile(r"\[LEC\s+lec_(\d+)\s*/\s*SLIDE\s+(\d+)\]", re.I)
_LEC_NOTE  = re.compile(r"\[LEC\s+lec_(\d+)\s*/\s*LECTURE NOTE\]", re.I)
_GLOBAL    = re.compile(r"\[GLOBAL\]", re.I)
_USER      = re.compile(r"\[USER\]", re.I)
_READINGS = re.compile(r"^lec_(\d+)_readings_(\d+)\.txt$", re.I)

def normalize_citations(txt: str) -> str:
    txt = _LEC_SLIDE.sub(lambda m: f"[Lecture {m.group(1)} Slide {m.group(2)}]", txt)
    txt = _LEC_NOTE.sub(lambda m: f"[Lecture {m.group(1)} Notes]", txt)
    txt = _GLOBAL.sub("[Global]", txt)
    txt = _USER.sub("[From Previous Conversations]", txt)
    txt = _READINGS.sub(lambda m: f"[From Lecture {m.group(1)}]", txt)
    return txt

def call_sig_gpt5(messages, temperature: float = 0.7, max_tokens: int = 2048):
    """
    Calls GPT-5-mini via SIG Azure API Mgmt gateway (OpenAI-compatible /chat/completions).
    """
    url = (
        f"{settings.SIG_GPT5_BASE}/deployments/"
        f"{settings.SIG_GPT5_DEPLOYMENT}/chat/completions"
        f"?api-version={settings.SIG_API_VERSION}"
    )
    payload = {
        "model": settings.SIG_GPT5_DEPLOYMENT,  # not strictly required, but fine
        "messages": messages,
        # "temperature": temperature,
        # "max_completion_tokens": max_tokens,
    }
    headers = {
        "Content-Type": "application/json",
        "api-key": settings.SIG_API_KEY,
    }
    r = httpx.post(url, headers=headers, json=payload, timeout=90)
    r.raise_for_status()
    data = r.json()
    return data["choices"][0]["message"]["content"]


def _post_json_gemini(payload: dict) -> httpx.Response:
    headers = {
        "Content-Type": "application/json",
        "X-goog-api-key": GEMINI_KEY,
    }
    with httpx.Client(timeout=45.0, headers=headers) as c:
        return c.post(GEMINI_URL, json=payload)


def _post_json_sig_gpt5(payload: dict) -> httpx.Response:
    url = (
        f"{settings.SIG_GPT5_BASE}/deployments/"
        f"{settings.SIG_GPT5_DEPLOYMENT}/chat/completions"
        f"?api-version={settings.SIG_API_VERSION}"
    )
    headers = {
        "Content-Type": "application/json",
        "api-key": settings.SIG_API_KEY,
    }
    return httpx.post(url, headers=headers, json=payload, timeout=90)


def answer_with_ctx(question: str, context: str, max_tokens: int = 600) -> dict:
    """
    Send the question + context to GPT-5-mini (SIG endpoint) and return a structured answer dict.
    """
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"QUESTION:\n{question}\n\nCONTEXT:\n{context}"},
    ]

    payload = {
        "model": settings.SIG_GPT5_DEPLOYMENT,
        "messages": messages,
        # "temperature": 0.3,
        # "max_completion_tokens": max_tokens,
    }

    t0 = time.time()
    r = _post_json_sig_gpt5(payload)

    if r.status_code >= 400:
        return {
            "answer": f"(GPT-5-mini error: {r.status_code}) {r.text[:400]}",
            "model": settings.SIG_GPT5_DEPLOYMENT,
            "latency_s": round(time.time() - t0, 3),
            "endpoint": settings.SIG_GPT5_BASE,
        }

    data = r.json()
    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError):
        return {
            "answer": "(GPT-5-mini returned unexpected format)",
            "raw": data,
            "model": settings.SIG_GPT5_DEPLOYMENT,
            "latency_s": round(time.time() - t0, 3),
            "endpoint": settings.SIG_GPT5_BASE,
        }

    content = normalize_citations(content)
    return {
        "answer": content,
        "model": settings.SIG_GPT5_DEPLOYMENT,
        "latency_s": round(time.time() - t0, 3),
        "endpoint": settings.SIG_GPT5_BASE,
    }


def groq_get_models() -> dict:
    if not API_KEY:
        return {"error": "missing GROQ_API_KEY"}
    url = f"{BASE}/openai/v1/models"  # OpenAI-compatible discovery
    with httpx.Client(timeout=20.0, headers={"Authorization": f"Bearer {API_KEY}"}) as c:
        r = c.get(url)
        return {"status": r.status_code, "text": r.text}
