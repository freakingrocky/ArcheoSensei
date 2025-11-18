use std::collections::HashMap;

use anyhow::Context;
use pgvector::Vector;
use serde_json::json;
use sqlx::{Row, postgres::PgRow};
use uuid::Uuid;

use crate::{
    db::DbPool,
    embedder::Embedder,
    models::{LectureImageAsset, RetrieveDiagnostics, RetrieveHit, RetrieveResult},
};
use tracing::warn;

async fn knn_store(
    pool: &DbPool,
    qvec: &Vector,
    store_kind: i32,
    limit: i64,
) -> anyhow::Result<Vec<RetrieveHit>> {
    let sql = "SELECT c.id, c.text, c.metadata, 1 - (c.embedding <=> $1::vector) AS score, \
                      d.\"Citation\", d.\"FILE_URL\", (d.extra->>'priority')::int AS priority \
               FROM chunks c \
               LEFT JOIN documents d ON d.id = c.doc_id \
               WHERE c.store_kind = $2 \
               ORDER BY c.embedding <=> $1::vector \
               LIMIT $3";
    let rows: Vec<PgRow> = sqlx::query(sql)
        .bind(qvec.clone())
        .bind(store_kind)
        .bind(limit)
        .fetch_all(pool)
        .await?;
    Ok(rows.into_iter().map(row_to_hit).collect())
}

async fn knn_lecture(
    pool: &DbPool,
    qvec: &Vector,
    lecture_key: &str,
    limit: i64,
) -> anyhow::Result<Vec<RetrieveHit>> {
    let sql = "SELECT c.id, c.text, c.metadata, 1 - (c.embedding <=> $1::vector) AS score, \
                      d.\"Citation\", d.\"FILE_URL\", (d.extra->>'priority')::int AS priority \
               FROM chunks c \
               LEFT JOIN documents d ON d.id = c.doc_id \
               WHERE c.store_kind = 2 AND c.metadata->>'lecture_key' = $2 \
               ORDER BY c.embedding <=> $1::vector \
               LIMIT $3";
    let rows: Vec<PgRow> = sqlx::query(sql)
        .bind(qvec.clone())
        .bind(lecture_key)
        .bind(limit)
        .fetch_all(pool)
        .await?;
    Ok(rows.into_iter().map(row_to_hit).collect())
}

async fn knn_store_priority(
    pool: &DbPool,
    qvec: &Vector,
    store_kind: i32,
    limit: i64,
) -> anyhow::Result<Vec<RetrieveHit>> {
    let sql = "SELECT c.id, c.text, c.metadata, 1 - (c.embedding <=> $1::vector) AS score, \
                      d.\"Citation\", d.\"FILE_URL\", (d.extra->>'priority')::int AS priority \
               FROM chunks c \
               LEFT JOIN documents d ON d.id = c.doc_id \
               WHERE c.store_kind = $2 AND COALESCE((d.extra->>'priority')::int, 0) = 1 \
               ORDER BY c.embedding <=> $1::vector \
               LIMIT $3";
    let rows: Vec<PgRow> = sqlx::query(sql)
        .bind(qvec.clone())
        .bind(store_kind)
        .bind(limit)
        .fetch_all(pool)
        .await?;
    Ok(rows.into_iter().map(row_to_hit).collect())
}

async fn knn_lecture_priority(
    pool: &DbPool,
    qvec: &Vector,
    lecture_key: &str,
    limit: i64,
) -> anyhow::Result<Vec<RetrieveHit>> {
    let sql = "SELECT c.id, c.text, c.metadata, 1 - (c.embedding <=> $1::vector) AS score, \
                      d.\"Citation\", d.\"FILE_URL\", (d.extra->>'priority')::int AS priority \
               FROM chunks c \
               LEFT JOIN documents d ON d.id = c.doc_id \
               WHERE c.store_kind = 2 AND c.metadata->>'lecture_key' = $2 \
                 AND COALESCE((d.extra->>'priority')::int, 0) = 1 \
               ORDER BY c.embedding <=> $1::vector \
               LIMIT $3";
    let rows: Vec<PgRow> = sqlx::query(sql)
        .bind(qvec.clone())
        .bind(lecture_key)
        .bind(limit)
        .fetch_all(pool)
        .await?;
    Ok(rows.into_iter().map(row_to_hit).collect())
}

fn row_to_hit(row: PgRow) -> RetrieveHit {
    RetrieveHit {
        id: row.get::<Uuid, _>("id"),
        text: row.get::<String, _>("text"),
        metadata: row.get::<serde_json::Value, _>("metadata"),
        score: row.get::<f64, _>("score") as f32,
        priority: row.try_get::<Option<i32>, _>("priority").ok().flatten(),
        citation: row.try_get::<Option<String>, _>("Citation").ok().flatten(),
        file_url: row.try_get::<Option<String>, _>("FILE_URL").ok().flatten(),
        tag: None,
    }
}

fn detect_lecture(candidates: &[RetrieveHit]) -> Option<(String, HashMap<String, f32>)> {
    let mut scores: HashMap<String, f32> = HashMap::new();
    let mut counts: HashMap<String, u32> = HashMap::new();
    for hit in candidates {
        let metadata = hit.metadata.as_object();
        let Some(metadata) = metadata else {
            continue;
        };
        let key = metadata.get("lecture_key").and_then(|v| v.as_str());
        if let Some(key) = key {
            let source = metadata
                .get("source")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            let weight = match source {
                "slide" => 1.0,
                "slide_note" => 0.9,
                "lecture_note" => 0.6,
                _ => 0.8,
            };
            *scores.entry(key.to_string()).or_insert(0.0) += hit.score * weight;
            *counts.entry(key.to_string()).or_insert(0) += 1;
        }
    }
    if scores.is_empty() {
        return None;
    }
    for (key, score) in scores.clone() {
        let count = counts.get(&key).copied().unwrap_or(1).max(1) as f32;
        scores.insert(key.clone(), score / count);
    }
    let best = scores
        .iter()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(k, v)| (k.clone(), *v))?;
    Some((best.0, scores))
}

fn readable_label(metadata: &serde_json::Value) -> Option<String> {
    let obj = metadata.as_object()?;
    if let (Some(lecture), Some(slide)) = (obj.get("lecture_key"), obj.get("slide_no")) {
        if let (Some(lecture), Some(slide)) = (lecture.as_str(), slide.as_i64()) {
            if let Some(num) = lecture.split('_').last() {
                return Some(format!("Lecture {} Slide {}", num, slide));
            }
        }
    }
    if let (Some(lecture), Some(source)) = (obj.get("lecture_key"), obj.get("source")) {
        if let Some(lecture) = lecture.as_str() {
            if let Some(num) = lecture.split('_').last() {
                if source == "lecture_note" {
                    return Some(format!("Lecture {} Notes", num));
                }
                if source == "readings" {
                    return Some(format!("From Lecture {}", num));
                }
            }
        }
    }
    if obj.get("store").and_then(|v| v.as_str()) == Some("global") {
        return Some("Global".to_string());
    }
    if obj.get("store").and_then(|v| v.as_str()) == Some("user")
        || obj.get("source").and_then(|v| v.as_str()) == Some("user_note")
    {
        return Some("From Previous Conversations".to_string());
    }
    None
}

fn citation_from_hit(hit: &RetrieveHit) -> Option<String> {
    hit.citation
        .as_ref()
        .map(|citation| citation.trim())
        .filter(|citation| !citation.is_empty())
        .map(|citation| citation.to_string())
}

fn tag_for_hit(_hit: &RetrieveHit, citation: Option<&str>) -> String {
    if let Some(citation) = citation {
        let clean = citation.trim();
        if clean.starts_with('[') && clean.ends_with(']') {
            clean.to_string()
        } else {
            format!("[{}]", clean)
        }
    } else {
        "[CTX]".to_string()
    }
}

fn build_context_from_hits(hits: &[RetrieveHit]) -> String {
    let mut blocks = Vec::new();
    let mut total = 0usize;
    for hit in hits {
        let snippet = hit.text.trim();
        let citation = hit
            .citation
            .as_deref()
            .filter(|v| !v.is_empty())
            .map(|v| v.to_string())
            .or_else(|| readable_label(&hit.metadata));
        let tag = hit
            .tag
            .clone()
            .unwrap_or_else(|| tag_for_hit(hit, citation.as_deref()));
        let block = if snippet.is_empty() {
            tag.clone()
        } else {
            format!("{}\n{}", tag, snippet)
        };
        if total + block.len() > 30_000 {
            break;
        }
        total += block.len();
        blocks.push(block);
    }
    blocks.join("\n\n")
}

async fn fetch_lecture_images(
    pool: &DbPool,
    lecture_key: &str,
    limit: i64,
) -> anyhow::Result<Vec<LectureImageAsset>> {
    let sql = "SELECT id, img_url, title, description, notes, lecture_key, area_description \
               FROM lecture_image_assets \
               WHERE lecture_key = $1 \
               ORDER BY updated_at DESC \
               LIMIT $2";
    let rows = sqlx::query_as::<_, LectureImageAsset>(sql)
        .bind(lecture_key)
        .bind(limit)
        .fetch_all(pool)
        .await?;
    Ok(rows)
}

pub async fn retrieve(
    pool: &DbPool,
    embedder: &Embedder,
    query: &str,
    lecture_force: Option<&str>,
    use_global: bool,
    user_id: Option<&str>,
) -> anyhow::Result<RetrieveResult> {
    let embeddings = embedder.embed([query]).await?;
    let qvec = Vector::from(embeddings.into_iter().next().context("missing embedding")?);

    let mut diagnostics = RetrieveDiagnostics::default();
    let mut merged: Vec<RetrieveHit> = Vec::new();

    let mut lecture_for_images: Option<String> = None;
    let lecture_hits = if let Some(force) = lecture_force {
        diagnostics.lecture_forced = Some(force.to_string());
        lecture_for_images = Some(force.to_string());
        knn_lecture(pool, &qvec, force, 20).await?
    } else {
        let coarse = knn_store(pool, &qvec, 2, 60).await?;
        let det = detect_lecture(&coarse);
        if let Some((key, votes)) = det {
            diagnostics.lecture_detected = Some(key.clone());
            diagnostics.lecture_votes = votes;
            lecture_for_images = Some(key.clone());
            knn_lecture(pool, &qvec, &key, 20).await?
        } else {
            Vec::new()
        }
    };

    let add_hit =
        |mut hit: RetrieveHit, default_citation: Option<String>, store_override: Option<&str>| {
            if let Some(store) = store_override {
                let mut metadata = hit.metadata.as_object().cloned().unwrap_or_default();
                metadata.insert("store".to_string(), json!(store));
                hit.metadata = serde_json::Value::Object(metadata);
            }
            let citation = hit
                .citation
                .clone()
                .and_then(|value| {
                    let trimmed = value.trim();
                    (!trimmed.is_empty()).then(|| trimmed.to_string())
                })
                .or(default_citation);
            let tag = tag_for_hit(&hit, citation.as_deref());
            RetrieveHit {
                tag: Some(tag),
                citation,
                ..hit
            }
        };

    for hit in lecture_hits {
        let citation = citation_from_hit(&hit);
        merged.push(add_hit(hit, citation, None));
    }

    if use_global {
        for hit in knn_store(pool, &qvec, 1, 10).await? {
            let citation = citation_from_hit(&hit);
            merged.push(add_hit(hit, citation, Some("global")));
        }
    }

    if let Some(user_id) = user_id {
        let sql = "SELECT id, text, metadata, 1 - (embedding <=> $1::vector) AS score \
                   FROM chunks \
                   WHERE store_kind = 3 AND tenant_id = $2 \
                   ORDER BY embedding <=> $1::vector LIMIT 10";
        let rows: Vec<PgRow> = sqlx::query(sql)
            .bind(qvec.clone())
            .bind(user_id)
            .fetch_all(pool)
            .await?;
        for row in rows {
            let hit = RetrieveHit {
                id: row.get::<Uuid, _>("id"),
                text: row.get("text"),
                metadata: row.get("metadata"),
                score: row.get::<f64, _>("score") as f32,
                priority: None,
                citation: None,
                file_url: None,
                tag: None,
            };
            let citation = citation_from_hit(&hit);
            merged.push(add_hit(hit, citation, Some("user")));
        }
    }

    merged.sort_by(|a, b| {
        let pa = a.priority.unwrap_or(i32::MAX);
        let pb = b.priority.unwrap_or(i32::MAX);
        pa.cmp(&pb).then_with(|| b.score.total_cmp(&a.score))
    });
    let mut hits = merged;
    if hits.len() > 12 {
        hits.truncate(12);
    }

    if lecture_for_images.is_none() {
        lecture_for_images = hits
            .iter()
            .find_map(|hit| hit.metadata.get("lecture_key").and_then(|v| v.as_str()))
            .map(|s| s.to_string());
    }

    let label = hits.first().and_then(|hit| {
        hit.citation
            .clone()
            .or_else(|| readable_label(&hit.metadata))
    });

    let images = if let Some(key) = lecture_for_images.clone() {
        match fetch_lecture_images(pool, &key, 4).await {
            Ok(rows) => rows,
            Err(err) => {
                warn!("lecture image fetch failed for {}: {:?}", key, err);
                Vec::new()
            }
        }
    } else {
        Vec::new()
    };

    Ok(RetrieveResult {
        diagnostics,
        hits,
        label,
        images,
    })
}

fn clean_text(value: &Option<String>) -> Option<String> {
    value
        .as_ref()
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string())
}

fn build_image_context(images: &[LectureImageAsset]) -> String {
    if images.is_empty() {
        return String::new();
    }
    let mut blocks = Vec::new();
    blocks.push("LECTURE_IMAGES:".to_string());
    for (idx, image) in images.iter().enumerate() {
        let mut lines = Vec::new();
        lines.push(format!("[IMG {}] {}", idx + 1, image.title.trim()));
        if let Some(lecture) = image
            .lecture_key
            .as_ref()
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
        {
            lines.push(format!("Lecture: {}", lecture));
        }
        lines.push(format!("URL: {}", image.img_url.trim()));
        if let Some(desc) = clean_text(&image.description) {
            lines.push(format!("Description: {}", desc));
        }
        if let Some(notes) = clean_text(&image.notes) {
            lines.push(format!("Notes: {}", notes));
        }
        if let Some(areas) = image.area_description.as_array() {
            let mut area_lines = Vec::new();
            for area in areas.iter().take(3) {
                let label = area
                    .get("label")
                    .and_then(|v| v.as_str())
                    .unwrap_or("Highlight");
                let help_text = area.get("help_text").and_then(|v| v.as_str()).unwrap_or("");
                let x = area.get("x").and_then(|v| v.as_f64());
                let y = area.get("y").and_then(|v| v.as_f64());
                let w = area.get("w").and_then(|v| v.as_f64());
                let h = area.get("h").and_then(|v| v.as_f64());
                let mut highlight = format!("- {}", label);
                if let (Some(x), Some(y), Some(w), Some(h)) = (x, y, w, h) {
                    highlight.push_str(&format!(" @ ({:.2}, {:.2}) size {:.2}x{:.2}", x, y, w, h));
                }
                if !help_text.trim().is_empty() {
                    highlight.push_str(&format!(": {}", help_text.trim()));
                }
                area_lines.push(highlight);
            }
            if !area_lines.is_empty() {
                lines.push("Highlights:".to_string());
                lines.extend(area_lines);
            }
        }
        blocks.push(lines.join("\n"));
    }
    blocks.join("\n\n")
}

pub fn build_context(hits: &[RetrieveHit], images: &[LectureImageAsset]) -> String {
    let mut context = build_context_from_hits(hits);
    let image_block = build_image_context(images);
    if !image_block.is_empty() {
        if !context.trim().is_empty() {
            context.push_str("\n\n");
        }
        context.push_str(&image_block);
    }
    context
}

pub async fn retrieve_priority_hits(
    pool: &DbPool,
    embedder: &Embedder,
    query: &str,
    lecture_key: Option<&str>,
    limit: i64,
) -> anyhow::Result<Vec<RetrieveHit>> {
    let embeddings = embedder.embed([query]).await?;
    let qvec = Vector::from(embeddings.into_iter().next().context("missing embedding")?);
    if let Some(key) = lecture_key {
        knn_lecture_priority(pool, &qvec, key, limit).await
    } else {
        knn_store_priority(pool, &qvec, 2, limit).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use uuid::Uuid;

    #[test]
    fn readable_label_for_slide() {
        let metadata = json!({"lecture_key": "lec_2", "slide_no": 4});
        let label = readable_label(&metadata).unwrap();
        assert_eq!(label, "Lecture 2 Slide 4");
    }

    #[test]
    fn context_truncates() {
        let hit = RetrieveHit {
            id: Uuid::nil(),
            text: "example text".to_string(),
            metadata: json!({"lecture_key": "lec_1", "slide_no": 1}),
            score: 0.9,
            priority: None,
            citation: Some("[Lecture 1 Slide 1]".to_string()),
            file_url: None,
            tag: None,
        };
        let context = build_context(&[hit], &[]);
        assert!(context.contains("example text"));
    }
}
