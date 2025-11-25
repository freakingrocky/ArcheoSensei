use std::collections::HashMap;
use std::io::Write;
use std::process::{Command, Stdio};

use anyhow::{Context, Result};
use base64::Engine as _;
use base64::engine::general_purpose::STANDARD as BASE64;
use lazy_static::lazy_static;
use pgvector::Vector;
use regex::Regex;
use serde_json::{Map, Value, json};
use sqlx::Row;

use crate::{db::DbPool, embedder::Embedder};

lazy_static! {
    static ref SLIDE_RE: Regex =
        Regex::new(r"^slide_(\d+)_lec_(\d+)\.(jpg|jpeg|png|txt)$").unwrap();
    static ref ABOUT_RE: Regex = Regex::new(r"^lec_(\d+)_about\.txt$").unwrap();
    static ref READINGS_RE: Regex = Regex::new(r"^lec_(\d+)_readings_(\d+)\.txt$").unwrap();
}

#[derive(Debug, Clone)]
struct PreparedFile {
    title: String,
    text: String,
    metadata: Value,
}

fn parse_filename(name: &str) -> Option<Map<String, Value>> {
    if let Some(caps) = SLIDE_RE.captures(name) {
        let slide_no: i64 = caps.get(1)?.as_str().parse().ok()?;
        let lec_no: i64 = caps.get(2)?.as_str().parse().ok()?;
        let ext = caps.get(3)?.as_str().to_lowercase();
        let kind = if ["jpg", "jpeg", "png"].contains(&ext.as_str()) {
            "slide"
        } else {
            "slide_note"
        };
        let mut map = Map::new();
        map.insert("kind".to_string(), json!(kind));
        map.insert("slide_no".to_string(), json!(slide_no));
        map.insert("lecture_key".to_string(), json!(format!("lec_{}", lec_no)));
        return Some(map);
    }
    if let Some(caps) = ABOUT_RE.captures(name) {
        let lec_no: i64 = caps.get(1)?.as_str().parse().ok()?;
        let mut map = Map::new();
        map.insert("kind".to_string(), json!("lecture_note"));
        map.insert("lecture_key".to_string(), json!(format!("lec_{}", lec_no)));
        return Some(map);
    }
    if let Some(caps) = READINGS_RE.captures(name) {
        let lec_no: i64 = caps.get(1)?.as_str().parse().ok()?;
        let reading_no: i64 = caps.get(2)?.as_str().parse().ok()?;
        let mut map = Map::new();
        map.insert("kind".to_string(), json!("readings"));
        map.insert("lecture_key".to_string(), json!(format!("lec_{}", lec_no)));
        map.insert("reading_no".to_string(), json!(reading_no));
        return Some(map);
    }
    None
}

fn ocr_image(bytes: &[u8]) -> Result<String> {
    let encoded = BASE64.encode(bytes);
    let script = r#"
import base64, io, sys

try:
    from PIL import Image
    import pytesseract
except Exception:
    sys.exit(3)

data = base64.b64decode(sys.stdin.read())
img = Image.open(io.BytesIO(data))
text = pytesseract.image_to_string(img)
sys.stdout.write(text)
"#;

    let mut child = match Command::new("python3")
        .arg("-c")
        .arg(script)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
    {
        Ok(child) => child,
        Err(_) => return Ok(String::new()),
    };
    if let Some(stdin) = child.stdin.as_mut() {
        stdin.write_all(encoded.as_bytes()).ok();
    }
    let output = child.wait_with_output()?;
    if output.status.success() {
        Ok(String::from_utf8_lossy(&output.stdout).to_string())
    } else {
        Ok(String::new())
    }
}

fn chunk_text(text: &str, max_chars: usize, overlap: usize) -> Vec<String> {
    if text.trim().is_empty() {
        return Vec::new();
    }
    let mut out = Vec::new();
    let mut start = 0usize;
    let len = text.len();
    while start < len {
        let end = (start + max_chars).min(len);
        let chunk = text[start..end].trim();
        if !chunk.is_empty() {
            out.push(chunk.to_string());
        }
        if end == len {
            break;
        }
        start = start + max_chars.saturating_sub(overlap);
    }
    out
}

async fn insert_document(
    pool: &DbPool,
    store_kind: i32,
    title: &str,
    source_uri: &str,
) -> Result<i64> {
    let row = sqlx::query("INSERT INTO documents(store_kind, title, source_uri, mime_type, extra) VALUES ($1,$2,$3,$4,$5) RETURNING id")
        .bind(store_kind)
        .bind(title)
        .bind(source_uri)
        .bind(None::<String>)
        .bind(json!({}))
        .fetch_one(pool)
        .await?;
    Ok(row.get::<i64, _>("id"))
}

async fn insert_chunks(pool: &DbPool, rows: &[PreparedChunk]) -> Result<()> {
    if rows.is_empty() {
        return Ok(());
    }
    let mut builder = sqlx::QueryBuilder::new(
        "INSERT INTO chunks (store_kind, tenant_id, doc_id, chunk_index, text, embedding, metadata)",
    );
    builder.push(" VALUES ");
    let mut separated = builder.separated(", ");
    for row in rows {
        separated.push("(");
        separated.push_bind(row.store_kind);
        separated.push(", ");
        separated.push_bind(row.tenant_id.as_deref());
        separated.push(", ");
        separated.push_bind(row.doc_id);
        separated.push(", ");
        separated.push_bind(row.idx);
        separated.push(", ");
        separated.push_bind(&row.text);
        separated.push(", ");
        separated.push_bind(Vector::from(row.embedding.clone()));
        separated.push(", ");
        separated.push_bind(&row.metadata);
        separated.push(")");
    }
    builder.build().execute(pool).await?;
    Ok(())
}

#[derive(Debug)]
struct PreparedChunk {
    store_kind: i32,
    tenant_id: Option<String>,
    doc_id: i64,
    idx: i32,
    text: String,
    embedding: Vec<f32>,
    metadata: Value,
}

pub async fn ingest_files(
    pool: &DbPool,
    embedder: &Embedder,
    files: Vec<(String, Vec<u8>)>,
    course: Option<&str>,
) -> Result<Value> {
    if files.is_empty() {
        return Ok(json!({"inserted": 0}));
    }

    let mut prepared: Vec<PreparedFile> = Vec::new();
    for (filename, bytes) in files {
        let name = filename.split('/').last().unwrap_or(&filename).to_string();
        let Some(mut meta) = parse_filename(&name) else {
            continue;
        };
        let text = match meta.get("kind").and_then(|v| v.as_str()) {
            Some("slide") => ocr_image(&bytes).unwrap_or_else(|_| String::new()),
            _ => String::from_utf8(bytes).unwrap_or_default(),
        };
        if let Some(course) = course {
            meta.insert("course".to_string(), json!(course));
        }
        meta.insert("filename".to_string(), json!(filename));
        prepared.push(PreparedFile {
            title: filename.clone(),
            text,
            metadata: Value::Object(meta),
        });
    }

    if prepared.is_empty() {
        return Ok(json!({"inserted": 0}));
    }

    let mut id_by_title: HashMap<String, i64> = HashMap::new();
    for file in &prepared {
        if !id_by_title.contains_key(&file.title) {
            let doc_id = insert_document(pool, 2, &file.title, &file.title).await?;
            id_by_title.insert(file.title.clone(), doc_id);
        }
    }

    let mut all_chunks: Vec<PreparedChunk> = Vec::new();
    for file in prepared {
        let doc_id = *id_by_title.get(&file.title).context("missing doc id")?;
        let chunks = chunk_text(&file.text, 1200, 240);
        if chunks.is_empty() {
            continue;
        }
        let embeddings = embedder.embed(chunks.iter().map(String::as_str)).await?;
        for (idx, (chunk, embedding)) in chunks.into_iter().zip(embeddings.into_iter()).enumerate()
        {
            let mut metadata = file.metadata.as_object().cloned().unwrap_or_default();
            metadata.insert("store".to_string(), json!("lectures"));
            let text = chunk.replace('\u{0000}', "");
            let text = text.replace('\r', "\n");
            let clean_text = text
                .chars()
                .filter(|c| *c != '\u{0000}')
                .collect::<String>();
            all_chunks.push(PreparedChunk {
                store_kind: 2,
                tenant_id: None,
                doc_id,
                idx: idx as i32,
                text: clean_text,
                embedding,
                metadata: Value::Object(metadata.clone()),
            });
        }
    }

    insert_chunks(pool, &all_chunks).await?;

    Ok(json!({"inserted": all_chunks.len()}))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunk_text_splits_with_overlap() {
        let text = "abcdefghij";
        let chunks = chunk_text(text, 4, 1);
        assert_eq!(chunks, vec!["abcd", "defg", "ghij"]);
    }

    #[test]
    fn parse_slide_filename() {
        let meta = parse_filename("slide_3_lec_2.jpg").expect("meta");
        assert_eq!(meta.get("kind").unwrap(), "slide");
        assert_eq!(meta.get("slide_no").unwrap(), &json!(3));
        assert_eq!(meta.get("lecture_key").unwrap(), &json!("lec_2"));
    }
}
