import json
import os, httpx, time
import re
from typing import Any, Dict, List, Tuple

import spacy
from spacy.cli import download as spacy_download

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


def answer_with_ctx(
    question: str,
    context: str,
    max_tokens: int = 600,
    temperature: float | None = None,
) -> dict:
    """
    Send the question + context to GPT-5-mini (SIG endpoint) and return a structured answer dict.
    """
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"QUESTION:\n{question}\n\nCONTEXT:\n{context}"},
    ]

    payload: Dict[str, Any] = {
        "model": settings.SIG_GPT5_DEPLOYMENT,
        "messages": messages,
        # "max_completion_tokens": max_tokens,
    }
    if temperature is not None:
        payload["temperature"] = temperature

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


FACT_THRESHOLD = 0.75
MAX_FACT_ATTEMPTS = 3

FACT_CHECK_SYSTEM = """You are an expert fact-checking system.
You must review the QUESTION, CONTEXT, and ANSWER.
Determine whether the ANSWER is fully supported by the CONTEXT.
Respond in strict JSON format with keys: verdict (\"pass\" or \"fail\"),
confidence (0.0-1.0), and rationale (string explaining the decision).
Do not include any additional commentary outside of valid JSON."""

_NLP = None


def _get_nlp():
    global _NLP
    if _NLP is not None:
        return _NLP
    try:
        _NLP = spacy.load("en_core_web_sm")
    except OSError:
        try:
            spacy_download("en_core_web_sm")
            _NLP = spacy.load("en_core_web_sm")
        except Exception:
            _NLP = spacy.blank("en")
            if "ner" not in _NLP.pipe_names:
                _NLP.add_pipe("ner")
    return _NLP


def fact_check_with_llm(question: str, context: str, answer: str) -> Dict[str, Any]:
    messages = [
        {"role": "system", "content": FACT_CHECK_SYSTEM},
        {
            "role": "user",
            "content": (
                f"QUESTION:\n{question}\n\nCONTEXT:\n{context}\n\nANSWER:\n{answer}"
            ),
        },
    ]

    payload = {
        "model": settings.SIG_GPT5_DEPLOYMENT,
        "messages": messages,
        "temperature": 0.0,
    }

    result: Dict[str, Any] = {
        "passed": False,
        "confidence": 0.0,
        "verdict": "undetermined",
        "rationale": "",
    }

    try:
        response = _post_json_sig_gpt5(payload)
    except Exception as exc:
        result["rationale"] = f"fact check call failed: {exc}"
        return result

    if response.status_code >= 400:
        result["rationale"] = f"fact check error {response.status_code}: {response.text[:200]}"
        return result

    try:
        data = response.json()
        content = data["choices"][0]["message"]["content"]
    except Exception as exc:  # noqa: BLE001
        result["rationale"] = f"malformed response: {exc}"
        return result

    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        result["rationale"] = content.strip()
        return result

    verdict = str(parsed.get("verdict", "")).lower()
    confidence = parsed.get("confidence", 0.0)
    rationale = str(parsed.get("rationale", "")).strip()

    result["verdict"] = verdict or "undetermined"
    result["confidence"] = float(confidence) if isinstance(confidence, (int, float)) else 0.0
    result["rationale"] = rationale
    result["passed"] = verdict in {"pass", "supported", "true"} and result["confidence"] >= 0.5
    return result


def _normalize_entity(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def _entities_and_pairs(doc) -> Tuple[List[str], List[Tuple[str, str]]]:
    entities = [_normalize_entity(ent.text) for ent in doc.ents if ent.text.strip()]
    pairs: List[Tuple[str, str]] = []
    if not entities:
        return entities, pairs

    if doc.has_annotation("SENT_START"):
        sentences = list(doc.sents)
    else:
        sentences = [doc]

    for sent in sentences:
        sent_entities = [
            _normalize_entity(ent.text)
            for ent in sent.ents
            if ent.text.strip()
        ]
        for i, ent_a in enumerate(sent_entities):
            for ent_b in sent_entities[i + 1 :]:
                pair = tuple(sorted((ent_a, ent_b)))
                pairs.append(pair)
    return entities, pairs


def fact_check_entities(context: str, answer: str) -> Dict[str, Any]:
    nlp = _get_nlp()
    ctx_doc = nlp(context or "")
    ans_doc = nlp(answer or "")

    ctx_entities, ctx_pairs = _entities_and_pairs(ctx_doc)
    ans_entities, ans_pairs = _entities_and_pairs(ans_doc)

    ctx_pair_set = set(ctx_pairs)
    ans_pair_set = set(ans_pairs)

    if not ans_pair_set:
        return {
            "score": 1.0,
            "matched_pairs": [],
            "missing_pairs": [],
            "answer_entities": ans_entities,
            "context_entities": ctx_entities,
            "details": "No entity relationships detected in answer.",
        }

    matched = sorted(pair for pair in ans_pair_set if pair in ctx_pair_set)
    missing = sorted(pair for pair in ans_pair_set if pair not in ctx_pair_set)
    score = len(matched) / max(len(ans_pair_set), 1)

    return {
        "score": score,
        "matched_pairs": matched,
        "missing_pairs": missing,
        "answer_entities": sorted(set(ans_entities)),
        "context_entities": sorted(set(ctx_entities)),
    }


def run_fact_check_pipeline(
    question: str,
    context: str,
    max_attempts: int = MAX_FACT_ATTEMPTS,
    threshold: float = FACT_THRESHOLD,
) -> Dict[str, Any]:
    attempts: List[Dict[str, Any]] = []
    final_llm: Dict[str, Any] | None = None
    final_answer = ""

    temperatures = [0.2, 0.35, 0.5]

    for attempt_idx in range(max_attempts):
        temp = temperatures[attempt_idx] if attempt_idx < len(temperatures) else temperatures[-1]
        llm_result = answer_with_ctx(question, context, temperature=temp)
        final_llm = llm_result
        final_answer = llm_result.get("answer", "")

        llm_check = fact_check_with_llm(question, context, final_answer)
        entity_check = fact_check_entities(context, final_answer)

        needs_retry = (
            not llm_check.get("passed", False)
            or entity_check.get("score", 0.0) < threshold
        )

        attempt_record = {
            "attempt": attempt_idx + 1,
            "temperature": temp,
            "needs_retry": needs_retry,
            "ai_check": llm_check,
            "ner_check": entity_check,
            "answer_excerpt": final_answer[:200],
        }
        attempts.append(attempt_record)

        if not needs_retry:
            return {
                "answer": final_answer,
                "llm": llm_result,
                "fact_check": {
                    "status": "passed",
                    "retry_count": attempt_idx,
                    "threshold": threshold,
                    "max_attempts": max_attempts,
                    "attempts": attempts,
                },
            }

    return {
        "answer": final_answer,
        "llm": final_llm or {},
        "fact_check": {
            "status": "failed",
            "retry_count": max(0, len(attempts) - 1),
            "threshold": threshold,
            "max_attempts": max_attempts,
            "attempts": attempts,
            "message": "Unable to validate answer after retries.",
        },
    }


def groq_get_models() -> dict:
    if not API_KEY:
        return {"error": "missing GROQ_API_KEY"}
    url = f"{BASE}/openai/v1/models"  # OpenAI-compatible discovery
    with httpx.Client(timeout=20.0, headers={"Authorization": f"Bearer {API_KEY}"}) as c:
        r = c.get(url)
        return {"status": r.status_code, "text": r.text}
