import json
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
import numpy as np
from .db import conn_cursor
from .embed import embed_texts

def knn(limit: int, qvec: np.ndarray, where_sql: str, params: tuple):
    sql = f"""
    select id, text, metadata, 1 - (embedding <=> %s) as score
    from chunks
    where {where_sql}
    order by embedding <=> %s
    limit {limit}
    """
    with conn_cursor() as cur:
        cur.execute(sql, (qvec.tolist(), qvec.tolist(), *params) if False else (qvec.tolist(), qvec.tolist()))
        # note: psycopg may not accept extra params without placeholders; keep simple 2-param form
        rows = cur.fetchall()
    # rows: (id, text, metadata, score)
    return [{"id": r[0], "text": r[1], "metadata": r[2], "score": float(r[3])} for r in rows]

def knn_store(store_kind: int, qvec: np.ndarray, limit=60):
    with conn_cursor() as cur:
        cur.execute(
            f"""select id, text, metadata, 1 - (embedding <=> %s::vector) as score
                from chunks
                where store_kind = %s
                order by embedding <=> %s::vector
                limit {limit}""",
            (qvec.tolist(), store_kind, qvec.tolist()),
        )
        return [{"id": r[0], "text": r[1], "metadata": r[2], "score": float(r[3])} for r in cur.fetchall()]

def knn_lecture(lecture_key: str, qvec: np.ndarray, limit=20):
    with conn_cursor() as cur:
        cur.execute(
            f"""select id, text, metadata, 1 - (embedding <=> %s::vector) as score
                from chunks
                where store_kind = 2 and metadata->>'lecture_key' = %s
                order by embedding <=> %s::vector
                limit {limit}""",
            (qvec.tolist(), lecture_key, qvec.tolist()),
        )
        return [{"id": r[0], "text": r[1], "metadata": r[2], "score": float(r[3])} for r in cur.fetchall()]

def detect_lecture(candidates: List[Dict]) -> Tuple[Optional[str], Dict[str, float]]:
    scores = defaultdict(float)
    counts = defaultdict(int)
    for r in candidates:
        md = r["metadata"]
        key = md.get("lecture_key")
        if not key: 
            continue
        # small source weights: slide > slide_note > lecture_note
        src = (md.get("source") or "")
        w = 1.0 if src == "slide" else (0.9 if src == "slide_note" else 0.6)
        scores[key] += r["score"] * w
        counts[key] += 1
    if not scores:
        return None, {}
    # normalize by counts a bit
    for k in list(scores.keys()):
        scores[k] = scores[k] / max(1, counts[k])
    best = max(scores.items(), key=lambda kv: kv[1])
    return best[0], dict(scores)

def retrieve(query: str, lecture_force: Optional[str], use_global=True, user_id: Optional[str]=None):
    qvec = embed_texts([query])[0]
    results = {"diagnostics": {}, "hits": []}

    # specialized (lectures)
    if lecture_force:
        det = lecture_force
        results["diagnostics"]["lecture_forced"] = det
        hits = knn_lecture(det, qvec, limit=20)
    else:
        coarse = knn_store(2, qvec, limit=60)
        det, vote = detect_lecture(coarse)
        results["diagnostics"]["lecture_detected"] = det
        results["diagnostics"]["lecture_votes"] = vote
        hits = knn_lecture(det, qvec, limit=20) if det else []

    # global KB (optional)
    global_hits = knn_store(1, qvec, limit=10) if use_global else []

    # user memory (optional)
    user_hits = []
    if user_id:
        with conn_cursor() as cur:
            cur.execute(
                """select id, text, metadata, 1 - (embedding <=> %s) as score
                   from chunks
                   where store_kind=3 and tenant_id = %s
                   order by embedding <=> %s
                   limit 10""",
                (qvec.tolist(), user_id, qvec.tolist()),
            )
            user_hits = [{"id": r[0], "text": r[1], "metadata": r[2], "score": float(r[3])} for r in cur.fetchall()]

    # merge (prioritize lecture → global → user)
    merged = []
    tag = lambda md: (
        f"[LEC {md.get('lecture_key')} / SLIDE {md.get('slide_no')}]" if md.get("slide_no") is not None
        else (f"[LEC {md.get('lecture_key')} / LECTURE NOTE]" if md.get("source") == "lecture_note"
              else "[GLOBAL]" if md.get("store") == "global" else "[USER]")
    )
    for r in hits:       merged.append({**r, "tag": tag(r["metadata"])})
    for r in global_hits: 
        md = dict(r["metadata"]); md["store"]="global"; r["metadata"]=md
        merged.append({**r, "tag": tag(md)})
    for r in user_hits:  merged.append({**r, "tag": "[USER]"})

    # truncate to ~top 12 by score
    merged.sort(key=lambda x: x["score"], reverse=True)
    results["hits"] = merged[:12]
    results["label"] = _readable_label(md)
    return results

def _readable_label(md):
    if md.get("slide_no") is not None and md.get("lecture_key"):
        n = md["lecture_key"].split("_")[-1]
        return f"Lecture {n} Slide {md['slide_no']}"
    if md.get("source") == "lecture_note" and md.get("lecture_key"):
        n = md["lecture_key"].split("_")[-1]
        return f"Lecture {n} Notes"
    if md.get("store") == "global":
        return "Global"
    return "User"
