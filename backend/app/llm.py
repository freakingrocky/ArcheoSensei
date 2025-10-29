import json
import os, httpx, time
import re
from typing import Any, Callable, Dict, List, Optional
import math
import numpy as np
import torch


from .settings import settings
from .embed import embed_texts

BASE = os.environ.get("GROQ_API_BASE", "https://api.groq.com")
MODEL = os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant")
API_KEY = os.environ.get("GROQ_API_KEY", "")


QUIZ_ALLOWED_TYPES = {
    "true_false",
    "mcq_single",
    "mcq_multi",
    "short_answer",
}

_JSON_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def _parse_structured_json(text: str) -> Dict[str, Any]:
    if not text:
        raise ValueError("Empty response from model")
    candidate = text.strip()
    match = _JSON_FENCE.search(candidate)
    if match:
        candidate = match.group(1).strip()
    try:
        return json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Failed to parse JSON payload: {exc}: {candidate[:200]}") from exc


def _ensure_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    return [str(value).strip()] if str(value).strip() else []


def generate_quiz_item(context: str, topic: str, question_type: str) -> Dict[str, Any]:
    if question_type not in QUIZ_ALLOWED_TYPES:
        raise ValueError(f"Unsupported quiz question type: {question_type}")

    safe_context = (context or "").strip()
    if len(safe_context) > 6500:
        safe_context = safe_context[:6500]

    topic_label = topic.strip() or "the course material"
    instructions = (
        "You are a helpful tutor creating a quiz question. "
        "Craft a question that tests conceptual understanding without relying on obscure trivia. "
        "Keep the language clear and student-friendly."
    )
    schema_description = (
        "Return JSON with keys: question_type, question_prompt, options, correct_answer, "
        "answer_rubric, hint, answer_explanation. "
        "question_type must be one of true_false, mcq_single, mcq_multi, short_answer. "
        "For true_false use the options ['True','False']. "
        "For mcq_single or mcq_multi provide between 3 and 5 options. "
        "correct_answer should be a string for true_false, mcq_single, short_answer; "
        "for mcq_multi return an array of the correct option strings. "
        "answer_rubric should outline the key points a good answer must include. "
        "hint should give a gentle nudge without revealing the answer. "
        "answer_explanation should clearly explain the correct answer in 2-4 sentences."
    )

    prompt = (
        f"Context to ground the question (may be empty):\n{safe_context}\n\n"
        f"Desired question type: {question_type}\n"
        f"Focus topic or lecture: {topic_label}\n"
        "Create one question following the schema instructions."
    )

    messages = [
        {"role": "system", "content": instructions + " " + schema_description},
        {"role": "user", "content": prompt},
    ]

    raw = call_sig_gpt5(messages, temperature=0.6, max_tokens=700)
    payload = _parse_structured_json(raw)

    q_type = str(payload.get("question_type", question_type)).strip().lower()
    if q_type not in QUIZ_ALLOWED_TYPES:
        q_type = question_type

    question_prompt = str(payload.get("question_prompt", "")).strip()
    if not question_prompt:
        raise ValueError("Model did not return a question prompt")

    options = payload.get("options") if isinstance(payload.get("options"), list) else None
    if q_type in {"mcq_single", "mcq_multi"}:
        options = [str(opt).strip() for opt in (options or []) if str(opt).strip()]
        if len(options) < 3:
            raise ValueError("Multiple-choice questions require at least three options")
    elif q_type == "true_false":
        options = ["True", "False"]
    else:
        options = None

    correct_answer = payload.get("correct_answer")
    if q_type == "mcq_multi":
        answers = _ensure_list(correct_answer)
        if not answers:
            raise ValueError("MCQ multi questions require at least one correct answer")
        correct_answer = answers
    else:
        if isinstance(correct_answer, list):
            correct_answer = ", ".join(str(v).strip() for v in correct_answer if str(v).strip())
        correct_answer = str(correct_answer or "").strip()
        if not correct_answer:
            raise ValueError("Question is missing a correct answer")

    hint = str(payload.get("hint", "")).strip()
    answer_rubric = str(payload.get("answer_rubric", "")).strip()
    explanation = str(payload.get("answer_explanation", "")).strip()

    if not answer_rubric:
        answer_rubric = "Highlight the central concept referenced in the question."
    if not hint:
        hint = "Think about the main idea presented in the lecture."
    if not explanation:
        explanation = "Review the core concept discussed to understand why this answer is correct."

    return {
        "question_type": q_type,
        "question_prompt": question_prompt,
        "options": options,
        "correct_answer": correct_answer,
        "answer_rubric": answer_rubric,
        "hint": hint,
        "answer_explanation": explanation,
    }


def grade_quiz_answer(
    question: Dict[str, Any],
    user_answer: Any,
    context: str,
) -> Dict[str, Any]:
    safe_context = (context or "").strip()
    if len(safe_context) > 6500:
        safe_context = safe_context[:6500]

    serialized_question = json.dumps(question, ensure_ascii=False)
    if isinstance(user_answer, list):
        answer_text = "\n".join(str(v).strip() for v in user_answer if str(v).strip())
    else:
        answer_text = str(user_answer or "").strip()

    if not answer_text:
        answer_text = "(no answer provided)"

    grading_instructions = (
        "You are grading a student's quiz response. Use the provided rubric, correct answer, and context. "
        "Return JSON with keys: correct (boolean), score (0.0-1.0), assessment (concise summary), "
        "good_points (array of strengths), bad_points (array of improvement areas). "
        "Score should reflect partial credit when appropriate."
    )

    prompt = (
        f"Context (may be empty):\n{safe_context}\n\n"
        f"Question JSON:\n{serialized_question}\n\n"
        f"Student answer:\n{answer_text}"
    )

    messages = [
        {"role": "system", "content": grading_instructions},
        {"role": "user", "content": prompt},
    ]

    raw = call_sig_gpt5(messages, temperature=0.2, max_tokens=600)
    payload = _parse_structured_json(raw)

    correct = bool(payload.get("correct"))
    score_val = payload.get("score")
    try:
        score = float(score_val)
    except (TypeError, ValueError):
        score = 1.0 if correct else 0.0
    score = max(0.0, min(1.0, score))

    assessment = str(payload.get("assessment", "")).strip()
    good_points = _ensure_list(payload.get("good_points"))
    bad_points = _ensure_list(payload.get("bad_points"))

    return {
        "correct": correct,
        "score": score,
        "assessment": assessment or ("Strong answer." if correct else "Needs improvement."),
        "good_points": good_points,
        "bad_points": bad_points,
    }

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
If the context is insufficient, say so explicitly.

IF SOMETHING CANNOT BE VERIFIED THEN MAKE SURE NOT TO INCLUDE IT IN THE FINAL RESPOSNE.
IF THERE IS ANY ERROR OR CONTRADICTION IN THE CONTEXT, POINT IT OUT CLEARLY.
IF YOU ARE UNSURE ABOUT THE ANSWER, STATE THAT CLEARLY RATHER THAN GUESSING.
Respond in a clear, concise manner suitable for a student audience.


Respond in Markdown format.
"""

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

def fact_check_claims(context: str, answer: str, threshold: float) -> Dict[str, Any]:
    """
    Topic-preservation via context-only k-means + soft coverage rules and centroid matching.
    Score = 1 - JSD over (context topic dist vs. answer-assigned dist).
    Pass if score >= threshold OR coverage_of_significant_topics >= coverage_min.

    Returns keys compatible with callers: score, entailed, total_claims, claims, threshold, passed, details.
    """
    # ---------- Tunables (adjust if you still see false MISSING_TOPIC) ----------
    k_max = 24            # upper bound on clusters
    k_min = 1             # lower bound
    centroid_match_tau = 0.83   # if any answer sentence cosine to centroid >= tau => topic covered
    mass_ratio_beta = 0.45      # topic covered if p_ans >= beta * p_ctx (so 45% of ctx mass)
    min_ctx_share = 0.15       # ignore tiny topics (<15% of context mass)
    min_ans_count = 1           # cover if >= this many answer sents assigned to the topic
    merge_tau = 0.90            # merge context centroids with cosine >= 0.90
    coverage_min = 0.70         # if >=70% significant topics covered, treat as pass even if score < threshold
    # ---------------------------------------------------------------------------

    def _extract(text: str, min_chars: int = 20) -> List[str]:
        sents = _extract_sentences(text)
        return [s for s in sents if len(s) >= min_chars]

    def _normalize_rows(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        n = np.linalg.norm(x, axis=1, keepdims=True)
        return x / np.maximum(n, eps)

    def _kmeans_pp_init(X: np.ndarray, k: int, rng: np.random.RandomState) -> np.ndarray:
        n = X.shape[0]
        centroids = np.empty((k, X.shape[1]), dtype=X.dtype)
        idx = rng.randint(0, n)
        centroids[0] = X[idx]
        d2 = np.sum((X - centroids[0])**2, axis=1)
        for i in range(1, k):
            probs = d2 / np.maximum(d2.sum(), 1e-12)
            idx = rng.choice(n, p=probs)
            centroids[i] = X[idx]
            d2 = np.minimum(d2, np.sum((X - centroids[i])**2, axis=1))
        return centroids

    def _kmeans(X: np.ndarray, k: int, max_iter: int = 60, tol: float = 1e-4, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.RandomState(seed)
        centroids = _kmeans_pp_init(X, k, rng)
        labels = np.zeros(X.shape[0], dtype=np.int32)
        for _ in range(max_iter):
            dists = np.sum((X[:, None, :] - centroids[None, :, :])**2, axis=2)  # [n, k]
            new_labels = np.argmin(dists, axis=1)
            if np.all(new_labels == labels):
                break
            labels = new_labels
            new_centroids = centroids.copy()
            for j in range(k):
                mask = (labels == j)
                if np.any(mask):
                    new_centroids[j] = X[mask].mean(axis=0)
            shift = np.linalg.norm(new_centroids - centroids)
            centroids = new_centroids
            if shift <= tol:
                break
        return labels, centroids

    def _dist_js(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
        """Jensen–Shannon distance in [0,1] using log base 2."""
        p = np.clip(p, eps, 1.0); p = p / p.sum()
        q = np.clip(q, eps, 1.0); q = q / q.sum()
        m = 0.5 * (p + q)
        def _kl(a, b):
            return float(np.sum(a * (np.log(a) - np.log(b)) / math.log(2)))
        jsd = 0.5 * (_kl(p, m) + _kl(q, m))
        return math.sqrt(jsd)

    def _merge_centroids(C: np.ndarray, labels: np.ndarray, X: np.ndarray, tau: float) -> tuple[np.ndarray, np.ndarray]:
        """
        Greedy merge of near-duplicate centroids by cosine >= tau.
        Returns new centroids and remapped labels.
        """
        if C.shape[0] <= 1:
            return C, labels
        Cn = _normalize_rows(C)
        k = C.shape[0]
        used = np.zeros(k, dtype=bool)
        new_centroids = []
        remap = -np.ones(k, dtype=int)
        for i in range(k):
            if used[i]: continue
            group = [i]
            for j in range(i+1, k):
                if used[j]: continue
                if float(np.dot(Cn[i], Cn[j])) >= tau:
                    group.append(j)
                    used[j] = True
            used[i] = True
            # average the members (by mass in X if desired; here equal)
            new_centroids.append(C[group].mean(axis=0))
            new_id = len(new_centroids) - 1
            for g in group:
                remap[g] = new_id
        new_centroids = np.vstack(new_centroids)
        new_labels = np.array([remap[l] for l in labels], dtype=int)
        return new_centroids, new_labels

    # ---------- Prepare ----------
    ctx_sents = _extract(context)
    ans_sents = _extract(answer)

    if not ctx_sents and not ans_sents:
        return {"score": 0.0, "entailed": 0, "total_claims": 0, "claims": [], "threshold": threshold, "passed": False, "details": "No sentences found."}
    if not ctx_sents:
        return {"score": 0.0, "entailed": 0, "total_claims": 0, "claims": [], "threshold": threshold, "passed": False, "details": "No context sentences."}
    if not ans_sents:
        return {"score": 0.0, "entailed": 0, "total_claims": 0, "claims": [], "threshold": threshold, "passed": False, "details": "No answer sentences."}

    ctx_emb = embed_texts(ctx_sents); ctx_emb = np.asarray(ctx_emb, dtype=np.float32)
    ans_emb = embed_texts(ans_sents); ans_emb = np.asarray(ans_emb, dtype=np.float32)
    ctx_emb = _normalize_rows(ctx_emb); ans_emb = _normalize_rows(ans_emb)

    n_ctx = len(ctx_sents)
    n_ans = len(ans_sents)

    # ---------- Choose k on context only ----------
    k = int(max(k_min, min(k_max, round(np.sqrt(n_ctx)))))
    if k > n_ctx:
        k = max(k_min, n_ctx)

    # Edge: if n_ctx small, just compare means
    if k <= 1:
        jsd = _dist_js(np.array([1.0]), np.array([1.0]))
        score = 1.0 - jsd
        return {
            "score": score, "entailed": 1 if n_ans > 0 else 0, "total_claims": 1,
            "claims": [], "threshold": threshold, "passed": score >= threshold,
            "details": {"k": 1, "jsd": jsd, "note": "degenerate single-topic case"}
        }

    # ---------- Cluster context; then possibly merge near-duplicate topics ----------
    ctx_labels, ctx_centroids = _kmeans(ctx_emb, k=k, seed=17)
    ctx_centroids, ctx_labels = _merge_centroids(ctx_centroids, ctx_labels, ctx_emb, tau=merge_tau)
    k_eff = ctx_centroids.shape[0]
    # recompute distributions
    def _hist(labels: np.ndarray, k: int) -> np.ndarray:
        h = np.bincount(labels, minlength=k).astype(np.float64)
        if h.sum() == 0: h[0] = 1.0
        return h / h.sum()

    p_ctx = _hist(ctx_labels, k_eff)

    # ---------- Assign answer to context centroids ----------
    # Use cosine since rows are normalized: argmax dot
    sims = ans_emb @ _normalize_rows(ctx_centroids).T  # [n_ans, k_eff]
    ans_labels = np.argmax(sims, axis=1) if n_ans else np.zeros((0,), dtype=int)
    p_ans = _hist(ans_labels, k_eff)

    # ---------- Score by JSD ----------
    jsd = _dist_js(p_ctx, p_ans)
    score = float(1.0 - jsd)

    # ---------- Soft topic coverage ----------
    # A topic j is significant if it holds enough mass in context.
    significant = (p_ctx >= min_ctx_share)
    covered_flags = np.zeros(k_eff, dtype=bool)

    # rule A: proportional mass covered
    covered_flags |= (p_ans >= mass_ratio_beta * p_ctx)

    # rule B: at least 'min_ans_count' answer sentences assigned
    if n_ans:
        for j in range(k_eff):
            if (ans_labels == j).sum() >= min_ans_count:
                covered_flags[j] = True

    # rule C: ANY answer sentence close to centroid (semantic hit)
    if n_ans:
        cen = _normalize_rows(ctx_centroids)
        # max cosine by topic
        max_cos_by_topic = (ans_emb @ cen.T).max(axis=0)  # [k_eff]
        covered_flags |= (max_cos_by_topic >= centroid_match_tau)

    # Compute coverage over significant topics only
    sig_count = int(significant.sum()) if significant.any() else k_eff
    covered_sig = int((covered_flags & significant).sum()) if significant.any() else int(covered_flags.sum())
    coverage = covered_sig / max(sig_count, 1)

    passed = bool((score >= threshold) or (coverage >= coverage_min))

    # ---------- Build "missing topics" only for truly uncovered significant topics ----------
    missing = []
    if significant.any():
        cen = _normalize_rows(ctx_centroids)
        for j in range(k_eff):
            if not significant[j] or covered_flags[j]:
                continue
            # pick representative context sentence nearest to centroid j
            members = np.where(ctx_labels == j)[0]
            if members.size == 0:
                continue
            # find representative
            ctx_emb_j = ctx_emb[members]
            d = 1.0 - (ctx_emb_j @ cen[j])  # cosine distance
            m_idx_local = int(np.argmin(d))
            m_idx = int(members[m_idx_local])
            rep = ctx_sents[m_idx]
            missing.append({
                "claim": f"Missing topic: {rep[:180]}",
                "context": rep,
                "context_index": m_idx,
                "label": "missing_topic",
                "topic_index": j,
                "topic_share_context": float(round(p_ctx[j], 4)),
                "topic_share_answer": float(round(p_ans[j], 4)),
                "max_centroid_cosine": float(round((ans_emb @ cen[j]).max() if n_ans else 0.0, 4)),
            })

    # For compatibility with upstream fields:
    total_topics = int((p_ctx > 0).sum())
    well_covered = int(covered_sig)

    return {
        "score": score,
        "entailed": well_covered,           # number of significant topics covered
        "total_claims": sig_count,          # number of significant context topics
        "claims": missing,                  # “missing_topic” entries only when clearly uncovered
        "threshold": threshold,
        "passed": passed,
        "details": {
            "k": int(k_eff),
            "jsd": float(jsd),
            "coverage": float(coverage),
            "coverage_min": float(coverage_min),
            "centroid_match_tau": float(centroid_match_tau),
            "mass_ratio_beta": float(mass_ratio_beta),
            "min_ctx_share": float(min_ctx_share),
            "topic_shares_context": p_ctx.tolist(),
            "topic_shares_answer": p_ans.tolist(),
        },
    }



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
        _MNLI_TOKENIZER = AutoTokenizer.from_pretrained(_MNLI_MODEL_NAME, token=settings.HUGGINGFACE_TOKEN or None)
        _MNLI_MODEL = AutoModelForSequenceClassification.from_pretrained(
            _MNLI_MODEL_NAME, token=settings.HUGGINGFACE_TOKEN or None
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
