import os, re, io, json
from typing import Dict, List, Optional, Tuple
from PIL import Image
import pytesseract
from .embed import embed_texts
from .db import conn_cursor
from .settings import settings

SLIDE_RE = re.compile(r"^slide_(\d+)_lec_(\d+)\.(jpg|jpeg|png|txt)$", re.I)
ABOUT_RE = re.compile(r"^lec_(\d+)_about\.txt$", re.I)
READINGS_RE = re.compile(r"^lec_(\d+)_readings_(\d+)\.txt$", re.I)

def parse_filename(name: str) -> Optional[Dict]:
    m = SLIDE_RE.match(name)
    if m:
        slide_no = int(m.group(1))
        lec_no = int(m.group(2))
        ext = m.group(3).lower()
        return {"kind": "slide" if ext in ("jpg","jpeg","png") else "slide_note",
                "slide_no": slide_no, "lecture_key": f"lec_{lec_no}"}
    m = ABOUT_RE.match(name)
    if m:
        lec_no = int(m.group(1))
        return {"kind": "lecture_note", "lecture_key": f"lec_{lec_no}"}
    if m := READINGS_RE.match(name):
        lec_no = int(m.group(1))
        return {"kind": "readings", "lecture_key": f"lec_{lec_no}"}
    return None

def ocr_image(content: bytes) -> str:
    img = Image.open(io.BytesIO(content)).convert("RGB")
    text = pytesseract.image_to_string(img)
    return text

def chunk_text(text: str, max_chars=1200, overlap=240) -> List[str]:
    out, i, n = [], 0, len(text)
    while i < n:
        out.append(text[i:i+max_chars])
        i += max_chars - overlap
    return [c.strip() for c in out if c.strip()]

def insert_document(store_kind: int, title: str, source_uri: str = None, mime_type: str = None, extra: dict = None):
    extra = extra or {}
    with conn_cursor() as cur:
        cur.execute(
            "insert into documents(store_kind,title,source_uri,mime_type,extra) values (%s,%s,%s,%s,%s) returning id",
            (store_kind, title, source_uri, mime_type, json.dumps(extra)),
        )
        return cur.fetchone()[0]

def insert_chunks(rows: List[dict]):
    # rows: {store_kind, tenant_id, doc_id, chunk_index, text, embedding(np), metadata(dict)}
    with conn_cursor() as cur:
        data = []
        for r in rows:
            try:
                # Clean null bytes from text (Postgres TEXT cannot contain them)
                clean_text = r["text"].replace("\x00", "")
                # Also strip any other binary control chars that may break encoding
                clean_text = clean_text.encode("utf-8", errors="ignore").decode("utf-8", errors="ignore")

                data.append((
                    r["store_kind"],
                    r.get("tenant_id"),
                    r["doc_id"],
                    r["idx"],
                    clean_text,
                    r["embedding"].tolist(),
                    json.dumps(r["metadata"])
                ))
            except Exception as e:
                # Skip this row entirely and continue with others
                print(f"[WARN] Skipping problematic chunk idx={r.get('idx')} doc_id={r.get('doc_id')}: {e}")
                continue

        # If no valid data remains, just return
        if not data:
            return

        try:
            cur.executemany(
                """INSERT INTO chunks (store_kind, tenant_id, doc_id, chunk_index, text, embedding, metadata)
                   VALUES (%s,%s,%s,%s,%s,%s,%s)""",
                data
            )
        except Exception as e:
            # Log and skip if batch insert fails — don't crash the whole ingestion
            print(f"[ERROR] Batch insert failed: {e}")
    # rows: {store_kind, tenant_id, doc_id, chunk_index, text, embedding(np), metadata(dict)}
def parse_filename(name: str) -> Optional[Dict]:
    """
    Parse a filename to extract information about the file type and associated lecture.

    This function attempts to match the given filename against three regular expressions
    (SLIDE_RE, ABOUT_RE, and READINGS_RE) to determine the file type and extract relevant
    information.

    Parameters:
    name (str): The filename to be parsed.

    Returns:
    Optional[Dict]: A dictionary containing parsed information if the filename matches
                    any of the expected patterns, or None if no match is found.
                    The dictionary may contain the following keys:
                    - 'kind': The type of file ('slide', 'slide_note', 'lecture_note', or 'readings')
                    - 'slide_no': The slide number (for slide or slide_note files)
                    - 'lecture_key': A string identifier for the associated lecture

    """
    m = SLIDE_RE.match(name)
    if m:
        slide_no = int(m.group(1))
        lec_no = int(m.group(2))
        ext = m.group(3).lower()
        return {"kind": "slide" if ext in ("jpg","jpeg","png") else "slide_note",
                "slide_no": slide_no, "lecture_key": f"lec_{lec_no}"}
    m = ABOUT_RE.match(name)
    if m:
        lec_no = int(m.group(1))
        return {"kind": "lecture_note", "lecture_key": f"lec_{lec_no}"}
    if m := READINGS_RE.match(name):
        lec_no = int(m.group(1))
        return {"kind": "readings", "lecture_key": f"lec_{lec_no}"}
    return None
    with conn_cursor() as cur:
        data = [
            (
                r["store_kind"], r.get("tenant_id"), r["doc_id"], r["idx"], r["text"],
                r["embedding"].tolist(), json.dumps(r["metadata"])
            )
            for r in rows
        ]
        cur.executemany(
            """insert into chunks (store_kind, tenant_id, doc_id, chunk_index, text, embedding, metadata)
               values (%s,%s,%s,%s,%s,%s,%s)""",
            data
        )

def ingest_files(file_blobs: List[Tuple[str, bytes]], course: Optional[str] = None) -> Dict:
    """
    file_blobs: list of (filename, bytes). Accepts:
      - slide_1_lec_1.jpg/png
      - slide_1_lec_1.txt
      - lec_1_about.txt
    """
    prepared = []  # (doc_title, text, metadata)
    count  = 0
    for fname, blob in file_blobs:
        print(f"[ingest {count + 1}] processing {fname} ({len(blob)} bytes)")
        meta = parse_filename(os.path.basename(fname))
        if not meta:
            continue
        if meta["kind"] == "slide":
            text = ocr_image(blob)
        else:
            text = blob.decode("utf-8", errors="ignore")
        # attach course + filename
        if course:
            meta["course"] = course
        meta["filename"] = fname
        prepared.append((fname, text, meta))

    if not prepared:
        return {"inserted": 0}

    # create one doc per distinct logical file
    id_by_title = {}
    for title, _, _ in prepared:
        if title not in id_by_title:
            doc_id = insert_document(2, title=title, source_uri=title, mime_type=None, extra={})
            id_by_title[title] = doc_id

    # chunk + embed
    rows = []
    for title, text, meta in prepared:
        chunks = chunk_text(text)
        if not chunks:
            continue
        embs = embed_texts(chunks)
        doc_id = id_by_title[title]
        for idx, (ch, vec) in enumerate(zip(chunks, embs)):
            rows.append({
                "store_kind": 2,
                "tenant_id": None,
                "doc_id": doc_id,
                "idx": idx,
                "text": ch,
                "embedding": vec,
                "metadata": {
                    "store": "lectures",
                    **meta
                }
            })
    insert_chunks(rows)
    count += 1
    return {"inserted": len(rows)}
