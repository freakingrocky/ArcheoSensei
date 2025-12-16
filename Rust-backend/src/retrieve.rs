use std::collections::HashMap;

use serde_json::{Value, json};

use anyhow::Context;
use pgvector::Vector;
use sqlx::{Row, postgres::PgRow};
use uuid::Uuid;

use crate::{
    db::DbPool,
    embedder::Embedder,
    models::{ImageAsset, RetrieveDiagnostics, RetrieveHit, RetrieveResult},
};
use tracing::{info, warn};

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

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    dot / (norm_a * norm_b)
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

fn serialize_area_description(area: &Value) -> String {
    serde_json::to_string(area).unwrap_or_else(|_| "[]".to_string())
}

fn build_image_context(asset: &ImageAsset) -> String {
    let mut lines = Vec::new();
    lines.push("IMAGE_ASSET".to_string());
    lines.push(format!("IMG_URL: {}", asset.img_url));
    if let Some(title) = asset.title.as_deref() {
        if !title.trim().is_empty() {
            lines.push(format!("TITLE: {}", title.trim()));
        }
    }
    if let Some(desc) = asset.description.as_deref() {
        if !desc.trim().is_empty() {
            lines.push(format!("DESCRIPTION: {}", desc.trim()));
        }
    }
    if let Some(notes) = asset.notes.as_deref() {
        if !notes.trim().is_empty() {
            lines.push(format!("NOTES: {}", notes.trim()));
        }
    }
    if let Some(lecture) = asset.lecture_key.as_deref() {
        if !lecture.trim().is_empty() {
            lines.push(format!("LECTURE: {}", lecture.trim()));
        }
    }
    lines.push(format!(
        "AREA_DESCRIPTION: {}",
        serialize_area_description(&asset.area_description)
    ));
    lines.join("\n")
}

async fn fetch_image_assets(
    pool: &DbPool,
    lecture_key: Option<&str>,
    limit: i64,
) -> anyhow::Result<Vec<ImageAsset>> {
    let rows: Vec<PgRow> = match lecture_key {
        Some(key) => sqlx::query(
            "SELECT img_url, title, description, notes, lecture_key, COALESCE(area_description, '[]'::jsonb) as area_description \n             FROM lecture_image_assets WHERE lecture_key = $1 ORDER BY updated_at DESC LIMIT $2",
        )
        .bind(key)
        .bind(limit)
        .fetch_all(pool)
        .await?,
        None => sqlx::query(
            "SELECT img_url, title, description, notes, lecture_key, COALESCE(area_description, '[]'::jsonb) as area_description \n             FROM lecture_image_assets ORDER BY updated_at DESC LIMIT $1",
        )
        .bind(limit)
        .fetch_all(pool)
        .await?,
    };

    Ok(rows
        .into_iter()
        .map(|row| ImageAsset {
            img_url: row.get::<String, _>("img_url"),
            title: row.try_get::<String, _>("title").ok(),
            description: row.try_get::<String, _>("description").ok(),
            notes: row.try_get::<String, _>("notes").ok(),
            lecture_key: row.try_get::<String, _>("lecture_key").ok(),
            area_description: row
                .try_get::<Value, _>("area_description")
                .unwrap_or_else(|_| json!([])),
            similarity: None,
        })
        .collect())
}

async fn select_best_image_asset(
    pool: &DbPool,
    embedder: &Embedder,
    query_embedding: &[f32],
    query: &str,
    lecture_hint: Option<&str>,
) -> anyhow::Result<Option<ImageAsset>> {
    let mut candidates = fetch_image_assets(pool, lecture_hint, 32).await?;
    if candidates.is_empty() {
        candidates = fetch_image_assets(pool, None, 24).await?;
    }
    if candidates.is_empty() {
        return Ok(None);
    }

    let embed_inputs: Vec<String> = candidates
        .iter()
        .map(|c| {
            format!(
                "{}\n{}\n{}\n{}\nQuery: {}",
                c.title.clone().unwrap_or_default(),
                c.description.clone().unwrap_or_default(),
                c.notes.clone().unwrap_or_default(),
                c.lecture_key.clone().unwrap_or_default(),
                query
            )
        })
        .collect();
    let embeddings = embedder
        .embed(embed_inputs.iter().map(|s| s.as_str()))
        .await?;

    let mut best: Option<ImageAsset> = None;
    let mut best_score = f32::MIN;
    for (mut asset, emb) in candidates.into_iter().zip(embeddings) {
        let sim = cosine_similarity(query_embedding, &emb);
        asset.similarity = Some(sim);
        if sim > best_score {
            best_score = sim;
            best = Some(asset);
        }
    }

    Ok(best)
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
    let query_embedding = embeddings.into_iter().next().context("missing embedding")?;
    let qvec = Vector::from(query_embedding.clone());

    let mut diagnostics = RetrieveDiagnostics::default();
    let mut merged: Vec<RetrieveHit> = Vec::new();

    info!(
        lecture_force = ?lecture_force,
        use_global,
        user_id_present = user_id.is_some(),
        "starting retrieval"
    );

    let lecture_hits = if let Some(force) = lecture_force {
        diagnostics.lecture_forced = Some(force.to_string());
        knn_lecture(pool, &qvec, force, 20).await?
    } else {
        let coarse = knn_store(pool, &qvec, 2, 60).await?;
        let det = detect_lecture(&coarse);
        if let Some((key, votes)) = det {
            diagnostics.lecture_detected = Some(key.clone());
            diagnostics.lecture_votes = votes;
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
        let user_uuid = Uuid::parse_str(user_id).map_err(|error| {
            warn!(%user_id, %error, "failed to parse user_id as UUID");
            anyhow::anyhow!("invalid user_id UUID")
        })?;
        let sql = "SELECT id, text, metadata, 1 - (embedding <=> $1::vector) AS score \
                   FROM chunks \
                   WHERE store_kind = 3 AND tenant_id = $2::uuid \
                   ORDER BY embedding <=> $1::vector LIMIT 10";
        let rows: Vec<PgRow> = sqlx::query(sql)
            .bind(qvec.clone())
            .bind(user_uuid)
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

    info!(
        hit_count = hits.len(),
        top_score = hits.first().map(|h| h.score),
        "retrieval completed"
    );

    let label = hits.first().and_then(|hit| {
        hit.citation
            .clone()
            .or_else(|| readable_label(&hit.metadata))
    });

    let lecture_hint = lecture_force.or(diagnostics.lecture_detected.as_deref());
    let image_asset =
        select_best_image_asset(pool, embedder, &query_embedding, query, lecture_hint).await?;

    Ok(RetrieveResult {
        diagnostics,
        hits,
        label,
        image_asset,
    })
}

pub fn build_context(hits: &[RetrieveHit]) -> String {
    build_context_from_hits(hits)
}

pub fn build_context_with_image(hits: &[RetrieveHit], image: Option<&ImageAsset>) -> String {
    let mut context = build_context_from_hits(hits);
    if let Some(asset) = image {
        let image_block = build_image_context(asset);
        if !context.is_empty() {
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
        let context = build_context(&[hit]);
        assert!(context.contains("example text"));
    }
}
