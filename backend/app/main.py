from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional, Dict, Any
import random
import orjson
from fastapi.responses import JSONResponse

from .settings import Settings, settings
from .ingest import ingest_files
from .retrieve import retrieve
from .db import conn_cursor
from .embed import embed_texts
from .schemas import (
    QueryRequest,
    UploadLectureRequest,
    MemorizeRequest,
    QueryOptions,
    QuizQuestionRequest,
    QuizQuestionResponse,
    QuizGradeRequest,
    QuizGradeResponse,
)
from .llm import groq_get_models, run_fact_check_pipeline, generate_quiz_item, grade_quiz_answer
from .jobs import jobs


app = FastAPI(title="RAG Backend", version="0.2.0")

origins = ["http://localhost:3000", "http://127.0.0.1:3000"]
app.add_middleware(
    CORSMiddleware, allow_origins=origins,
    allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

@app.get("/health")
def health():
    # quick DB ping
    db_ok = False
    try:
        with conn_cursor() as cur:
            cur.execute("select 1")
            db_ok = True
    except Exception:
        db_ok = False
    return {"status":"ok", "db": db_ok, "embedding_dim": settings.EMBEDDING_DIM}

@app.post("/upload/lectures")
async def upload_lectures(
    course: Optional[str] = Form(None),
    files: List[UploadFile] = File(...)
):
    blobs = []
    for f in files:
        content = await f.read()
        blobs.append((f.filename, content))
    res = ingest_files(blobs, course=course)
    return {"ok": True, **res}

@app.post("/memorize")
def memorize(req: MemorizeRequest):
    # store raw and vectorize into store_kind=3
    with conn_cursor() as cur:
        cur.execute(
            "insert into user_interactions(user_id, q_text) values (%s,%s) returning id",
            (req.user_id, req.text)
        )
    # embed & insert into chunks
    vec = embed_texts([req.text])[0]
    with conn_cursor() as cur:
        cur.execute(
            """insert into chunks (store_kind, tenant_id, doc_id, chunk_index, text, embedding, metadata)
               values (3, %s, null, 0, %s, %s, %s)""",
            (req.user_id, req.text, vec.tolist(), orjson.dumps({"source":"user_note"}).decode())
        )
    return {"ok": True}

def _build_context_from_hits(hits: List[Dict[str, Any]]) -> str:
    blocks, total = [], 0
    for h in hits:
        snippet = h.get("text", "").strip()
        tag = h.get("tag", "[CTX]")
        md = h.get("metadata") or {}
        filename = md.get("filename")
        file_hint = f" [FILE {filename}]" if filename else ""
        block = f"{tag}{file_hint}\n{snippet}" if snippet else f"{tag}{file_hint}"
        if total + len(block) > 30_000:
            break
        blocks.append(block)
        total += len(block)
    return "\n\n".join(blocks) if blocks else "(no context)"


def _process_query_job(job_id: str, req: QueryRequest) -> None:
    opts = req.options or QueryOptions()
    try:
        jobs.update_job(job_id, status="running", phase="retrieving")
        retrieval = retrieve(
            query=req.query,
            lecture_force=opts.force_lecture_key,
            use_global=opts.use_global,
            user_id=opts.user_id,
        )
    except Exception as exc:  # noqa: BLE001
        jobs.update_job(
            job_id,
            status="failed",
            phase="done",
            message=f"Retrieval failed: {exc}",
            answer=f"Error: Retrieval failed ({exc})",
            fact_check={},
            diagnostics={},
            hits=[],
        )
        return

    hits = retrieval.get("hits", [])
    context = _build_context_from_hits(hits)
    diagnostics = retrieval.get("diagnostics", {})

    jobs.update_job(
        job_id,
        phase="llm",
        diagnostics=diagnostics,
        hits=hits,
        top_k=len(hits),
        context_len=len(context),
    )

    def progress(event: Dict[str, Any]) -> None:
        stage = event.get("stage")
        if stage == "pipeline_start":
            jobs.update_job(
                job_id,
                max_attempts=event.get("max_attempts"),
                threshold=event.get("threshold"),
            )
        elif stage == "attempt_start":
            jobs.update_job(
                job_id,
                phase="llm",
                current_attempt=event.get("attempt"),
                directives=event.get("directives", ""),
            )
        elif stage == "llm_result":
            jobs.update_job(job_id, llm=event.get("llm"))
        elif stage == "fact_ai":
            ai_check = event.get("ai_check") or {}
            jobs.update_job(
                job_id,
                phase="fact_ai",
                last_ai_check=ai_check,
                fact_ai_status="passed" if ai_check.get("passed") else "failed",
            )
        elif stage == "fact_claims":
            claims_check = event.get("claims_check") or {}
            jobs.update_job(
                job_id,
                phase="fact_claims",
                last_claim_check=claims_check,
                fact_claims_status="passed" if claims_check.get("passed") else "failed",
            )
        elif stage == "attempt_complete":
            attempt = event.get("attempt_record") or {}
            jobs.append_attempt(job_id, attempt)
            retry_count = max(0, int(attempt.get("attempt", 1)) - 1)
            message = "AI response could not be validated, retrying..." if attempt.get("needs_retry") else ""
            jobs.update_job(job_id, retry_count=retry_count, message=message)
        elif stage == "completed":
            status = event.get("status")
            final_status = "succeeded" if status == "passed" else "failed"
            message = ""
            fact_payload = event.get("fact_check") or {}
            if final_status == "failed":
                message = fact_payload.get(
                    "message",
                    "AI response could not be validated after retries.",
                )
            jobs.update_job(
                job_id,
                status=final_status,
                phase="done",
                answer=event.get("answer"),
                fact_check=fact_payload,
                llm=event.get("llm"),
                message=message,
            )

    result = run_fact_check_pipeline(
        req.query,
        context,
        progress_cb=progress,
    )

    # Ensure final payload is stored even if the completed event didn't fire (e.g. progress callback absent)
    existing = jobs.get_job(job_id) or {}
    status = "succeeded" if (result.get("fact_check", {}).get("status") == "passed") else existing.get("status", "failed")
    jobs.update_job(
        job_id,
        status=status,
        phase="done",
        answer=result.get("answer"),
        fact_check=result.get("fact_check"),
        llm=result.get("llm"),
    )


# Quiz helpers
def _sanitize_question_type(qtype: Optional[str]) -> str:
    allowed = {"true_false", "mcq_single", "mcq_multi", "short_answer"}
    if qtype and qtype.lower() in allowed:
        return qtype.lower()
    return random.choice(sorted(allowed))


@app.post("/quiz/question", response_model=QuizQuestionResponse)
def quiz_question(req: QuizQuestionRequest):
    if not req.topic and not req.lecture_key:
        raise HTTPException(status_code=400, detail="Provide a topic or lecture key")

    question_type = _sanitize_question_type(req.question_type)
    query = (req.topic or "").strip()
    if not query and req.lecture_key:
        query = f"Key ideas from {req.lecture_key}"

    retrieval = retrieve(
        query=query or "Key ideas from the course",
        lecture_force=req.lecture_key,
        use_global=not bool(req.lecture_key),
        user_id=None,
    )
    context = _build_context_from_hits(retrieval.get("hits", []))

    try:
        question = generate_quiz_item(context, query or (req.lecture_key or "course material"), question_type)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Failed to generate quiz question: {exc}") from exc

    return QuizQuestionResponse(
        question=question,
        context=context,
        lecture_key=req.lecture_key,
        topic=req.topic,
    )


@app.post("/quiz/grade", response_model=QuizGradeResponse)
def quiz_grade(req: QuizGradeRequest):
    try:
        result = grade_quiz_answer(req.question.dict(), req.user_answer, req.context)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Failed to grade response: {exc}") from exc

    return QuizGradeResponse(**result)


@app.post("/query")
def query(req: QueryRequest):
    opts = req.options or QueryOptions()
    out = retrieve(
        query=req.query,
        lecture_force=opts.force_lecture_key,
        use_global=opts.use_global,
        user_id=opts.user_id
    )
    hits = out.get("hits", [])
    context = _build_context_from_hits(hits)
    fact_checked = run_fact_check_pipeline(req.query, context)
    llm = fact_checked.get("llm", {})
    answer = fact_checked.get("answer")
    fact_check = fact_checked.get("fact_check", {})

    return {
        "diagnostics": out["diagnostics"],
        "top_k": len(hits),
        "hits": hits,
        "context_len": len(context),
        "llm_model": llm.get("model"),
        "llm_latency_s": llm.get("latency_s"),
        "llm_usage": llm.get("usage", {}),
        "answer": answer,
        "fact_check": fact_check,
    }


@app.post("/query/async")
def query_async(req: QueryRequest, background: BackgroundTasks):
    job_id = jobs.create_job({"status": "queued", "phase": "queued"})
    background.add_task(_process_query_job, job_id, req)
    return {"job_id": job_id}


@app.get("/query/async/{job_id}")
def query_async_status(job_id: str):
    job = jobs.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    return job

@app.get("/llm/models")
def llm_models():
    return groq_get_models()


@app.get("/lectures")
def list_lectures() -> Dict[str, List[Dict]]:
    """
    Returns distinct lecture keys from the 'chunks' table (store_kind=2).
    """
    rows = []
    with conn_cursor() as cur:
        cur.execute(
            """
            select metadata->>'lecture_key' as lecture_key,
                   count(*) as n
            from chunks
            where store_kind=2 and metadata ? 'lecture_key'
            group by 1
            order by lecture_key
            """
        )
        rows = [{"lecture_key": r[0], "count": int(r[1])} for r in cur.fetchall()]
    return {"lectures": rows}

@app.get("/source/{lecture_key}/{slide_no}")
def get_source(lecture_key: str, slide_no: int):
    """
    Return the slide image URL and corresponding text for a given lecture slide.
    """
    with conn_cursor() as cur:
        cur.execute(
            "SELECT text, metadata FROM documents WHERE metadata->>'lecture_key'=%s AND (metadata->>'slide_no')::int=%s LIMIT 1",
            (lecture_key, slide_no),
        )
        row = cur.fetchone()
        if not row:
            return JSONResponse({"error": "Not found"}, status_code=404)

        text, metadata = row
        # Assuming images are stored with predictable names or in Supabase storage
        img_url = f"{settings.SUPABASE_URL}/storage/v1/object/public/slides/{lecture_key}_slide_{slide_no}.jpg"
        return {"lecture_key": lecture_key, "slide_no": slide_no, "text": text, "image_url": img_url}
