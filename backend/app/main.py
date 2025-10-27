from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
import orjson
from typing import List, Dict
from fastapi.responses import JSONResponse

from .settings import Settings, settings
from .ingest import ingest_files
from .retrieve import retrieve
from .db import conn_cursor
from .embed import embed_texts
from .schemas import QueryRequest, UploadLectureRequest, MemorizeRequest, QueryOptions
from .llm import answer_with_ctx, groq_get_models

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

@app.post("/query")
def query(req: QueryRequest):
    opts = req.options or QueryOptions()
    out = retrieve(
        query=req.query,
        lecture_force=opts.force_lecture_key,
        use_global=opts.use_global,
        user_id=opts.user_id
    )
    # Build prompt context capped to ~3–4k chars to keep costs low
    blocks, total = [], 0
    for h in out["hits"]:
        snippet = h["text"].strip()
        tag = h.get("tag","[CTX]")
        block = f"{tag} {snippet}"
        if total + len(block) > 30_000:  # crude cap
            break
        blocks.append(block); total += len(block)
    context = "\n\n".join(blocks) if blocks else "(no context)"
    llm = answer_with_ctx(req.query, context)

    return {
        "diagnostics": out["diagnostics"],
        "top_k": len(out["hits"]),
        "hits": out["hits"],
        "context_len": len(context),
        "llm_model": llm.get("model"),
        "llm_latency_s": llm.get("latency_s"),
        "llm_usage": llm.get("usage", {}),
        "answer": llm.get("answer"),
    }

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
