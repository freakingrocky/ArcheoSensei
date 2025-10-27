import json
import os, httpx, time
import re
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .settings import settings
from .embed import embed_texts

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
    directives: Optional[str] = None,
) -> dict:
    """
    Send the question + context to GPT-5-mini (SIG endpoint) and return a structured answer dict.
    """
    user_content = f"QUESTION:\n{question}\n\nCONTEXT:\n{context}"
    if directives:
        user_content += "\n\nDIRECTIONS:\n" + directives.strip()

    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": user_content},
    ]

    payload: Dict[str, Any] = {
        "model": settings.SIG_GPT5_DEPLOYMENT,
        "messages": messages,
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


FACT_THRESHOLD = 0.75
MAX_FACT_ATTEMPTS = 3

FACT_CHECK_SYSTEM = """You are an expert fact-checking system.
You must review the QUESTION, CONTEXT, and ANSWER.
Determine whether the ANSWER is fully supported by the CONTEXT.
Respond in strict JSON format with keys: verdict (\"pass\" or \"fail\"),
confidence (0.0-1.0), and rationale (string explaining the decision).
Do not include any additional commentary outside of valid JSON."""

_MNLI_MODEL = None
_MNLI_TOKENIZER = None
_MNLI_MODEL_NAME = "microsoft/deberta-v3-base-mnli"

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n", re.MULTILINE)
_CITATION_RE = re.compile(r"\[[^\]]+\]")


def _load_mnli():
    global _MNLI_MODEL, _MNLI_TOKENIZER
    if _MNLI_MODEL is None or _MNLI_TOKENIZER is None:
        _MNLI_TOKENIZER = AutoTokenizer.from_pretrained(_MNLI_MODEL_NAME)
        _MNLI_MODEL = AutoModelForSequenceClassification.from_pretrained(
            _MNLI_MODEL_NAME
        )
        _MNLI_MODEL.eval()
    return _MNLI_MODEL, _MNLI_TOKENIZER


def _strip_citations(text: str) -> str:
    return _CITATION_RE.sub("", text).strip()


def _split_sentences(text: str) -> List[str]:
    sentences: List[str] = []
    for block in text.splitlines():
        block = block.strip()
        if not block:
            continue
        parts = _SENTENCE_SPLIT_RE.split(block)
        for part in parts:
            cleaned = part.strip()
            if cleaned:
                sentences.append(cleaned)
    return sentences


def _extract_sentences(text: str, min_chars: int = 20) -> List[str]:
    sentences = [_strip_citations(s) for s in _split_sentences(text)]
    return [s for s in sentences if len(s) >= min_chars]


def generate_claims(answer: str, min_chars: int = 32, min_words: int = 5) -> List[str]:
    candidates = _extract_sentences(answer, min_chars=min_chars)
    claims: List[str] = []
    for cand in candidates:
        cleaned = cand.lstrip("•*- ").strip()
        if len(cleaned.split()) >= min_words:
            claims.append(cleaned)
    return claims


def _claim_context_pairs(
    claims: List[str], context_sentences: List[str]
) -> List[tuple[str, str, int]]:
    if not claims or not context_sentences:
        return []

    ctx_embeddings = embed_texts(context_sentences)
    claim_embeddings = embed_texts(claims)
    sims = np.matmul(claim_embeddings, ctx_embeddings.T)
    indices = np.argmax(sims, axis=1)
    pairs: List[tuple[str, str, int]] = []
    for idx, claim in enumerate(claims):
        ctx_idx = int(indices[idx])
        pairs.append((claim, context_sentences[ctx_idx], ctx_idx))
    return pairs


def fact_check_claims(
    context: str, answer: str, threshold: float
) -> Dict[str, Any]:
    context_sentences = _extract_sentences(context)
    claims = generate_claims(answer)

    if not claims:
        return {
            "score": 1.0,
            "entailed": 0,
            "total_claims": 0,
            "claims": [],
            "passed": True,
            "details": "No substantial claims extracted from answer.",
        }

    if not context_sentences:
        return {
            "score": 0.0,
            "entailed": 0,
            "total_claims": len(claims),
            "claims": [],
            "passed": False,
            "details": "No context sentences available for verification.",
        }

    model, tokenizer = _load_mnli()
    pairs = _claim_context_pairs(claims, context_sentences)

    entailments = 0
    claim_details: List[Dict[str, Any]] = []

    for claim, ctx_sentence, ctx_idx in pairs:
        inputs = tokenizer(
            ctx_sentence,
            claim,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512,
        )
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)[0].cpu()

        label_idx = int(torch.argmax(probs))
        label = model.config.id2label.get(label_idx, "neutral").lower()
        entail_prob = float(probs[2].item()) if probs.numel() >= 3 else 0.0
        neutral_prob = float(probs[1].item()) if probs.numel() >= 2 else 0.0
        contra_prob = float(probs[0].item()) if probs.numel() >= 1 else 0.0

        if label == "entailment":
            entailments += 1

        claim_details.append(
            {
                "claim": claim,
                "context": ctx_sentence,
                "context_index": ctx_idx,
                "label": label,
                "entailment_probability": entail_prob,
                "neutral_probability": neutral_prob,
                "contradiction_probability": contra_prob,
            }
        )

    score = entailments / max(len(claims), 1)
    return {
        "score": score,
        "entailed": entailments,
        "total_claims": len(claims),
        "claims": claim_details,
        "threshold": threshold,
        "passed": score >= threshold,
    }


def _summarize_claim_failures(
    claim_check: Dict[str, Any], limit: int = 4
) -> List[str]:
    details = []
    for entry in claim_check.get("claims", [])[:limit]:
        if entry.get("label") == "entailment":
            continue
        claim = entry.get("claim", "")
        ctx = entry.get("context", "")
        label = entry.get("label", "neutral").capitalize()
        details.append(
            f"Claim '{claim[:160]}' was rated {label}. Align it with context: '{ctx[:160]}'"
        )
    if not details and claim_check.get("total_claims"):
        details.append(
            "Some claims were not fully entailed by the context. Restate them using direct evidence."
        )
    return details


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


def _build_retry_directives(
    answer: str,
    llm_check: Dict[str, Any],
    claim_check: Dict[str, Any],
    threshold: float,
) -> str:
    lines: List[str] = []

    verdict = llm_check.get("verdict", "undetermined")
    if not llm_check.get("passed", False):
        rationale = llm_check.get("rationale", "") or "Previous answer was not fully supported by context."
        lines.append(
            "The previous draft answer failed the strict fact check. "
            f"Verdict: {verdict}. Rationale: {rationale}"
        )

    claim_score = claim_check.get("score", 0.0)
    if claim_score < threshold:
        lines.append(
            f"Evidence entailment rate was {claim_score:.2f}, below the target {threshold:.2f}."
        )
        for detail in _summarize_claim_failures(claim_check):
            lines.append(detail)

    lines.append(
        "Regenerate a new answer using only the provided CONTEXT. "
        "Fix the issues noted above, keep claims grounded in the cited context, and do not mention this validation process."
    )

    if answer:
        lines.append("Previous answer (for reference, do not repeat verbatim):\n" + answer[:600])

    return "\n".join(lines)


def run_fact_check_pipeline(
    question: str,
    context: str,
    max_attempts: int = MAX_FACT_ATTEMPTS,
    threshold: float = FACT_THRESHOLD,
    progress_cb: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    attempts: List[Dict[str, Any]] = []
    final_llm: Dict[str, Any] | None = None
    final_answer = ""
    directives: Optional[str] = None

    def emit(stage: str, **payload: Any) -> None:
        if progress_cb:
            progress_cb({"stage": stage, **payload})

    emit(
        "pipeline_start",
        max_attempts=max_attempts,
        threshold=threshold,
    )

    for attempt_idx in range(max_attempts):
        applied_directives = directives or ""
        emit(
            "attempt_start",
            attempt=attempt_idx + 1,
            directives=applied_directives,
        )
        llm_result = answer_with_ctx(
            question,
            context,
            directives=directives,
        )
        emit("llm_result", attempt=attempt_idx + 1, llm=llm_result)
        final_llm = llm_result
        final_answer = llm_result.get("answer", "")

        llm_check = fact_check_with_llm(question, context, final_answer)
        emit("fact_ai", attempt=attempt_idx + 1, ai_check=llm_check)
        claim_check = fact_check_claims(context, final_answer, threshold)
        emit("fact_claims", attempt=attempt_idx + 1, claims_check=claim_check)

        needs_retry = (
            not llm_check.get("passed", False)
            or not claim_check.get("passed", False)
        )

        attempt_record = {
            "attempt": attempt_idx + 1,
            "needs_retry": needs_retry,
            "directives": applied_directives,
            "ai_check": llm_check,
            "claims_check": claim_check,
            "answer_excerpt": final_answer[:200],
        }
        attempts.append(attempt_record)
        emit("attempt_complete", attempt_record=attempt_record)

        if not needs_retry:
            fact_check_payload = {
                "status": "passed",
                "retry_count": attempt_idx,
                "threshold": threshold,
                "max_attempts": max_attempts,
                "attempts": attempts,
            }
            emit(
                "completed",
                status="passed",
                answer=final_answer,
                llm=llm_result,
                fact_check=fact_check_payload,
            )
            return {
                "answer": final_answer,
                "llm": llm_result,
                "fact_check": fact_check_payload,
            }

        if attempt_idx + 1 < max_attempts:
            directives = _build_retry_directives(
                final_answer,
                llm_check,
                claim_check,
                threshold,
            )

    fact_check_payload = {
        "status": "failed",
        "retry_count": max(0, len(attempts) - 1),
        "threshold": threshold,
        "max_attempts": max_attempts,
        "attempts": attempts,
        "message": "Unable to validate answer after retries.",
    }
    emit(
        "completed",
        status="failed",
        answer=final_answer,
        llm=final_llm or {},
        fact_check=fact_check_payload,
    )

    return {
        "answer": final_answer,
        "llm": final_llm or {},
        "fact_check": fact_check_payload,
    }


def groq_get_models() -> dict:
    if not API_KEY:
        return {"error": "missing GROQ_API_KEY"}
    url = f"{BASE}/openai/v1/models"  # OpenAI-compatible discovery
    with httpx.Client(timeout=20.0, headers={"Authorization": f"Bearer {API_KEY}"}) as c:
        r = c.get(url)
        return {"status": r.status_code, "text": r.text}
