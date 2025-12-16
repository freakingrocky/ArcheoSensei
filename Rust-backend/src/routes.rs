use anyhow::Result;
use axum::{
    Json,
    extract::{Multipart, Path, State},
    http::StatusCode,
    response::{IntoResponse, Response},
};
use chrono::Utc;
use serde_json::{Value, json};
use sqlx::Row;
use std::sync::Arc;
use tokio::task;
use tracing::{error, warn};
use uuid::Uuid;

use crate::{
    config::Settings,
    db::DbPool,
    embedder::Embedder,
    ingest,
    jobs::{JobManager, JobMetadata, JobRecord},
    llm::{self, FactCheckOutput, build_learning_insights},
    models::{
        AsyncJobResponse, FactCheckResult, FactProgressCallback, FactProgressEvent, HealthResponse,
        ImageAsset, InsightStats, InsightsRequest, InsightsResponse, MemorizeRequest, QueryOptions,
        QueryRequest, QueryResponse, QuizGradeRequest, QuizGradeResponse, QuizQuestionRequest,
        QuizQuestionResponse, RetrieveDiagnostics, RetrieveHit, RetrieveResult,
    },
    retrieve,
};

#[derive(Clone)]
pub struct AppState {
    pub settings: Settings,
    pub pool: DbPool,
    pub embedder: Embedder,
    pub jobs: JobManager,
}

impl AppState {
    pub fn new(settings: Settings, pool: DbPool, embedder: Embedder, jobs: JobManager) -> Self {
        Self {
            settings,
            pool,
            embedder,
            jobs,
        }
    }
}

#[derive(Debug)]
pub struct ApiError(anyhow::Error);

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let status = StatusCode::INTERNAL_SERVER_ERROR;
        let body = Json(json!({ "detail": format!("{}", self.0) }));
        (status, body).into_response()
    }
}

impl<E> From<E> for ApiError
where
    E: Into<anyhow::Error>,
{
    fn from(err: E) -> Self {
        ApiError(err.into())
    }
}

pub async fn health(State(state): State<AppState>) -> Json<HealthResponse> {
    let mut db_ok = false;
    if let Ok(value) = sqlx::query_scalar::<_, i32>("SELECT 1")
        .fetch_one(&state.pool)
        .await
    {
        db_ok = value == 1;
    }
    Json(HealthResponse {
        status: "ok".to_string(),
        db: db_ok,
        embedding_dim: state.embedder.dimension(),
    })
}

fn build_annotated_image_block(asset: &ImageAsset) -> Option<String> {
    let payload = json!({
        "img_url": asset.img_url,
        "title": asset.title.clone().unwrap_or_default(),
        "description": asset.description.clone().unwrap_or_default(),
        "lecture": asset.lecture_key.clone().unwrap_or_default(),
        "notes": asset.notes.clone().unwrap_or_default(),
        "area_description": asset.area_description.clone(),
    });
    serde_json::to_string_pretty(&payload)
        .ok()
        .map(|body| format!("```annotated-image\n{}\n```", body))
}

fn answer_with_image_fallback(
    answer: &Option<String>,
    asset: Option<&ImageAsset>,
) -> Option<String> {
    let existing = answer.clone();
    let Some(asset_block) = asset.and_then(build_annotated_image_block) else {
        return existing;
    };
    match existing {
        Some(text) if text.contains("```annotated-image") => Some(text),
        Some(text) => Some(format!("{}\n\n{}", text.trim_end(), asset_block)),
        None => Some(asset_block),
    }
}

async fn load_user_context(pool: &DbPool, options: &QueryOptions) -> Option<String> {
    if let Some(raw) = options.user_context.as_ref().and_then(|ctx| {
        let trimmed = ctx.trim();
        (!trimmed.is_empty()).then(|| trimmed.to_string())
    }) {
        return Some(raw);
    }

    let Some(user_id) = options.user_id.as_deref() else {
        return None;
    };

    let user_uuid = match Uuid::parse_str(user_id) {
        Ok(id) => id,
        Err(error) => {
            warn!(%user_id, %error, "invalid user_id for profile context");
            return None;
        }
    };

    match sqlx::query("SELECT context FROM user_profiles WHERE id=$1")
        .bind(user_uuid)
        .fetch_optional(pool)
        .await
    {
        Ok(Some(row)) => {
            let value: Option<String> = row.get("context");
            value.and_then(|ctx| {
                let trimmed = ctx.trim();
                (!trimmed.is_empty()).then(|| trimmed.to_string())
            })
        }
        Ok(None) => None,
        Err(error) => {
            warn!(%user_id, %error, "failed to load profile context");
            None
        }
    }
}

pub async fn upload_lectures(
    State(state): State<AppState>,
    mut multipart: Multipart,
) -> Result<Json<Value>, ApiError> {
    let mut files: Vec<(String, Vec<u8>)> = Vec::new();
    let mut course: Option<String> = None;
    while let Some(field) = multipart.next_field().await? {
        let name = field.name().map(|s| s.to_string());
        if name.as_deref() == Some("course") {
            course = Some(field.text().await.unwrap_or_default());
        } else if let Some(filename) = field.file_name().map(|s| s.to_string()) {
            let data = field.bytes().await?.to_vec();
            files.push((filename, data));
        }
    }
    let result =
        ingest::ingest_files(&state.pool, &state.embedder, files, course.as_deref()).await?;
    Ok(Json(result))
}

pub async fn memorize(
    State(state): State<AppState>,
    Json(payload): Json<MemorizeRequest>,
) -> Result<Json<Value>, ApiError> {
    let user_uuid = uuid::Uuid::parse_str(&payload.user_id)
        .map_err(|e| anyhow::anyhow!("invalid user_id UUID: {}", e))?;
    let interaction_id: i64 =
        sqlx::query("INSERT INTO user_interactions(user_id, q_text) VALUES ($1,$2) RETURNING id")
            .bind(user_uuid)
            .bind(&payload.text)
            .fetch_one(&state.pool)
            .await?
            .get("id");
    let embeddings = state.embedder.embed([payload.text.as_str()]).await?;
    let vector = pgvector::Vector::from(embeddings[0].clone());
    sqlx::query("INSERT INTO chunks (store_kind, tenant_id, doc_id, chunk_index, text, embedding, metadata) VALUES (3,$1,NULL,0,$2,$3,$4)")
        .bind(user_uuid)
        .bind(&payload.text)
        .bind(vector)
        .bind(json!({"source": "user_note", "interaction_id": interaction_id}))
        .execute(&state.pool)
        .await?;
    Ok(Json(json!({"ok": true})))
}

pub async fn insights(
    State(state): State<AppState>,
    Json(payload): Json<InsightsRequest>,
) -> Result<Json<InsightsResponse>, ApiError> {
    let user_uuid = uuid::Uuid::parse_str(&payload.user_id)
        .map_err(|e| anyhow::anyhow!("invalid user_id UUID: {}", e))?;
    let max_chats = payload.max_chats.unwrap_or(6).clamp(1, 20);
    let max_messages = payload.max_messages_per_chat.unwrap_or(12).clamp(1, 50);

    let rows = sqlx::query(
        "SELECT name, messages FROM user_chats WHERE user_id=$1 ORDER BY updated_at DESC LIMIT $2",
    )
    .bind(user_uuid)
    .bind(max_chats as i64)
    .fetch_all(&state.pool)
    .await?;

    let mut transcript = String::new();
    let mut chat_count = 0usize;
    let mut message_count = 0usize;
    let mut quiz_signals = 0usize;

    for row in rows {
        let chat_name: String = row.get("name");
        let raw_messages: Value = row.get("messages");
        let Some(messages) = raw_messages.as_array() else {
            continue;
        };
        let mut window: Vec<Value> = messages.iter().rev().take(max_messages).cloned().collect();
        window.reverse();
        if window.is_empty() {
            continue;
        }
        chat_count += 1;
        transcript.push_str(&format!("Chat: {}\n", chat_name));
        for msg in window {
            let content = msg.get("content").and_then(Value::as_str).unwrap_or("");
            if content.trim().is_empty() {
                continue;
            }
            let role = msg.get("role").and_then(Value::as_str).unwrap_or("user");
            transcript.push_str(&format!("- {}: {}\n", role, content.trim()));
            message_count += 1;

            let kind = msg.get("kind").and_then(Value::as_str).unwrap_or("");
            let lowered = content.to_ascii_lowercase();
            if kind == "quiz"
                || lowered.contains("quiz question")
                || lowered.contains("quiz:")
                || lowered.contains("quiz me")
            {
                quiz_signals += 1;
            }
        }
        transcript.push('\n');
    }

    let (payload, llm) = build_learning_insights(&state.settings, &transcript).await?;

    Ok(Json(InsightsResponse {
        payload,
        llm,
        stats: InsightStats {
            chat_count,
            message_count,
            quiz_signals,
        },
    }))
}

pub async fn query(
    State(state): State<AppState>,
    Json(payload): Json<QueryRequest>,
) -> Result<Json<QueryResponse>, ApiError> {
    let options = payload.options.clone().unwrap_or(QueryOptions {
        force_lecture_key: None,
        use_global: true,
        user_id: None,
        chat_id: None,
        chat_name: None,
        user_context: None,
    });
    let user_context = load_user_context(&state.pool, &options).await;
    let RetrieveResult {
        diagnostics,
        hits,
        image_asset,
        ..
    } = retrieve::retrieve(
        &state.pool,
        &state.embedder,
        &payload.query,
        options.force_lecture_key.as_deref(),
        options.use_global,
        options.user_id.as_deref(),
    )
    .await?;
    let mut context = retrieve::build_context_with_image(&hits, image_asset.as_ref());
    if let Some(extra) = user_context.as_deref() {
        context = format!("User context:\n{}\n\n{}", extra, context);
    }
    let FactCheckOutput {
        answer,
        fact_check,
        llm,
    } = llm::run_fact_check_pipeline(
        &state.settings,
        &state.embedder,
        &payload.query,
        &context,
        None,
    )
    .await?;
    let final_answer = answer_with_image_fallback(&answer, image_asset.as_ref());
    let response = QueryResponse {
        diagnostics,
        top_k: hits.len(),
        hits,
        context_len: context.len(),
        llm_model: llm.model.clone(),
        llm_latency_s: llm.latency_s,
        llm_usage: llm.usage.clone(),
        answer: final_answer,
        fact_check,
        image_asset,
    };
    Ok(Json(response))
}

pub async fn query_async(
    State(state): State<AppState>,
    Json(payload): Json<QueryRequest>,
) -> Result<Json<AsyncJobResponse>, ApiError> {
    let metadata = JobMetadata {
        user_id: payload
            .options
            .as_ref()
            .and_then(|opts| opts.user_id.clone()),
        chat_id: payload
            .options
            .as_ref()
            .and_then(|opts| opts.chat_id.clone()),
        chat_name: payload
            .options
            .as_ref()
            .and_then(|opts| opts.chat_name.clone()),
        user_context: payload
            .options
            .as_ref()
            .and_then(|opts| opts.user_context.clone()),
        query: Some(payload.query.clone()),
        last_fetch: None,
    };
    let job_id = state.jobs.create_job(Some(metadata));
    let job_id_for_task = job_id.clone();
    let state_clone = state.clone();
    task::spawn(async move {
        if let Err(err) = process_query_job(state_clone, job_id_for_task, payload).await {
            error!("query job failed: {:?}", err);
        }
    });
    Ok(Json(AsyncJobResponse { job_id }))
}

async fn process_query_job(state: AppState, job_id: String, payload: QueryRequest) -> Result<()> {
    state
        .jobs
        .update_job(&job_id, json!({"status": "running", "phase": "retrieving"}));
    let options = payload.options.clone().unwrap_or(QueryOptions {
        force_lecture_key: None,
        use_global: true,
        user_id: None,
        chat_id: None,
        chat_name: None,
        user_context: None,
    });
    let user_context = load_user_context(&state.pool, &options).await;
    let result = retrieve::retrieve(
        &state.pool,
        &state.embedder,
        &payload.query,
        options.force_lecture_key.as_deref(),
        options.use_global,
        options.user_id.as_deref(),
    )
    .await;
    let retrieval = match result {
        Ok(r) => r,
        Err(err) => {
            state.jobs.update_job(
                &job_id,
                json!({
                    "status": "failed",
                    "phase": "done",
                    "message": format!("Retrieval failed: {}", err),
                }),
            );
            if let Some(job) = state.jobs.get_job(&job_id) {
                let empty_diag = RetrieveDiagnostics::default();
                let empty_fact = FactCheckResult::default();
                persist_chat_if_stale(
                    &state.pool,
                    &job,
                    &payload,
                    &Some(format!("Retrieval failed: {}", err)),
                    &empty_diag,
                    &[],
                    &empty_fact,
                )
                .await
                .ok();
            }
            return Ok(());
        }
    };
    let mut context =
        retrieve::build_context_with_image(&retrieval.hits, retrieval.image_asset.as_ref());
    if let Some(extra) = user_context.as_deref() {
        context = format!("User context:\n{}\n\n{}", extra, context);
    }
    state.jobs.update_job(
        &job_id,
        json!({
            "phase": "llm",
            "diagnostics": serde_json::to_value(&retrieval.diagnostics).unwrap_or(json!({})),
            "hits": serde_json::to_value(&retrieval.hits).unwrap_or(json!([])),
            "top_k": retrieval.hits.len(),
            "context_len": context.len(),
            "image_asset": serde_json::to_value(&retrieval.image_asset).unwrap_or(json!(null)),
        }),
    );
    let jobs_for_progress = state.jobs.clone();
    let job_id_for_progress = job_id.clone();
    let progress_cb: FactProgressCallback =
        Arc::new(move |event: &FactProgressEvent| match event.stage {
            "pipeline_start" => {
                let max_attempts = event
                    .data
                    .get("max_attempts")
                    .and_then(Value::as_u64)
                    .unwrap_or(0) as usize;
                let threshold = event
                    .data
                    .get("threshold")
                    .and_then(Value::as_f64)
                    .unwrap_or(0.0);
                jobs_for_progress.update_job(
                    &job_id_for_progress,
                    json!({
                        "max_attempts": max_attempts,
                        "retry_count": 0usize,
                        "threshold": threshold,
                    }),
                );
            }
            "attempt_start" => {
                jobs_for_progress.update_job(
                    &job_id_for_progress,
                    json!({
                        "phase": "llm",
                        "fact_ai_status": Value::Null,
                        "fact_claims_status": Value::Null,
                    }),
                );
            }
            "fact_ai" => {
                let passed = event
                    .data
                    .get("ai_check")
                    .and_then(|v| v.get("passed"))
                    .and_then(Value::as_bool);
                let payload = if let Some(passed) = passed {
                    let status = if passed { "passed" } else { "failed" };
                    json!({ "phase": "fact_ai", "fact_ai_status": status })
                } else {
                    json!({ "phase": "fact_ai" })
                };
                jobs_for_progress.update_job(&job_id_for_progress, payload);
            }
            "fact_claims" => {
                let passed = event
                    .data
                    .get("claims_check")
                    .and_then(|v| v.get("passed"))
                    .and_then(Value::as_bool);
                let payload = if let Some(passed) = passed {
                    let status = if passed { "passed" } else { "failed" };
                    json!({ "phase": "fact_claims", "fact_claims_status": status })
                } else {
                    json!({ "phase": "fact_claims" })
                };
                jobs_for_progress.update_job(&job_id_for_progress, payload);
            }
            "attempt_complete" => {
                if let Some(attempts) = event.data.get("attempts").cloned() {
                    let retry_count = event
                        .data
                        .get("retry_count")
                        .and_then(Value::as_u64)
                        .unwrap_or_default() as usize;
                    jobs_for_progress.update_job(
                        &job_id_for_progress,
                        json!({
                            "attempts": attempts,
                            "retry_count": retry_count,
                        }),
                    );
                }
            }
            _ => {}
        });

    match llm::run_fact_check_pipeline(
        &state.settings,
        &state.embedder,
        &payload.query,
        &context,
        Some(progress_cb),
    )
    .await
    {
        Ok(FactCheckOutput {
            answer,
            fact_check,
            llm,
        }) => {
            let final_answer = answer_with_image_fallback(&answer, retrieval.image_asset.as_ref());
            state.jobs.update_job(
                &job_id,
                json!({
                    "status": "succeeded",
                    "phase": "done",
                    "answer": final_answer,
                    "fact_check": serde_json::to_value(&fact_check).unwrap_or(json!({})),
                    "llm": serde_json::to_value(llm).unwrap_or(json!({})),
                    "image_asset": serde_json::to_value(&retrieval.image_asset).unwrap_or(json!(null)),
                }),
            );
            if let Some(job) = state.jobs.get_job(&job_id) {
                persist_chat_if_stale(
                    &state.pool,
                    &job,
                    &payload,
                    &final_answer,
                    &retrieval.diagnostics,
                    &retrieval.hits,
                    &fact_check,
                )
                .await
                .ok();
            }
        }
        Err(err) => {
            state.jobs.update_job(
                &job_id,
                json!({
                    "status": "failed",
                    "phase": "done",
                    "message": format!("LLM pipeline failed: {}", err),
                }),
            );
            if let Some(job) = state.jobs.get_job(&job_id) {
                let empty_fact = FactCheckResult::default();
                persist_chat_if_stale(
                    &state.pool,
                    &job,
                    &payload,
                    &Some(format!("LLM pipeline failed: {}", err)),
                    &retrieval.diagnostics,
                    &retrieval.hits,
                    &empty_fact,
                )
                .await
                .ok();
            }
        }
    }

    Ok(())
}

pub async fn query_async_status(
    State(state): State<AppState>,
    Path(job_id): Path<String>,
) -> Result<Json<JobRecord>, ApiError> {
    let Some(job) = state.jobs.get_job(&job_id) else {
        return Err(ApiError(anyhow::anyhow!("job not found")));
    };
    state.jobs.touch_fetch(&job_id);
    Ok(Json(job))
}

fn should_backend_persist(job: &JobRecord) -> bool {
    job.metadata
        .last_fetch
        .map(|ts| Utc::now().signed_duration_since(ts) > chrono::Duration::seconds(30))
        .unwrap_or(true)
}

async fn persist_chat_if_stale(
    pool: &DbPool,
    job: &JobRecord,
    payload: &QueryRequest,
    answer: &Option<String>,
    diagnostics: &RetrieveDiagnostics,
    hits: &[RetrieveHit],
    fact_check: &FactCheckResult,
) -> Result<()> {
    if !should_backend_persist(job) {
        return Ok(());
    }

    let user_id = match job.metadata.user_id.as_ref() {
        Some(id) => id,
        None => return Ok(()),
    };
    let chat_id = match job.metadata.chat_id.as_ref() {
        Some(id) => id,
        None => return Ok(()),
    };
    let chat_name = job
        .metadata
        .chat_name
        .clone()
        .unwrap_or_else(|| "Chat".to_string());

    let limited_hits: Vec<Value> = hits
        .iter()
        .take(3)
        .filter_map(|h| serde_json::to_value(h).ok())
        .collect();
    let fact_check_json = serde_json::to_value(fact_check).unwrap_or(json!({}));
    let diag_json = serde_json::to_value(diagnostics).unwrap_or(json!({}));
    let messages = json!([
        {"role": "user", "content": payload.query.clone()},
        {
            "role": "assistant",
            "content": answer.clone().unwrap_or_else(|| "(no answer)".to_string()),
            "hits": limited_hits,
            "diagnostics": diag_json,
            "fact_check": fact_check_json,
        }
    ]);

    sqlx::query(
        "INSERT INTO user_chats (id, user_id, name, messages)
         VALUES ($1, $2, $3, $4)
         ON CONFLICT (id) DO UPDATE SET
           name = EXCLUDED.name,
           messages = coalesce(user_chats.messages, '[]'::jsonb) || EXCLUDED.messages,
           updated_at = now()",
    )
    .bind(chat_id)
    .bind(user_id)
    .bind(chat_name)
    .bind(messages)
    .execute(pool)
    .await?;

    Ok(())
}

pub async fn llm_models(State(state): State<AppState>) -> Result<Json<Value>, ApiError> {
    let models = llm::groq_get_models(&state.settings).await?;
    Ok(Json(models))
}

pub async fn list_lectures(State(state): State<AppState>) -> Result<Json<Value>, ApiError> {
    let rows = sqlx::query("SELECT metadata->>'lecture_key' AS lecture_key, COUNT(*) as n FROM chunks WHERE store_kind=2 AND metadata ? 'lecture_key' GROUP BY 1 ORDER BY lecture_key")
        .fetch_all(&state.pool)
        .await?;
    let lectures: Vec<Value> = rows
        .into_iter()
        .map(|row| {
            json!({
                "lecture_key": row.get::<String, _>("lecture_key"),
                "count": row.get::<i64, _>("n"),
            })
        })
        .collect();
    Ok(Json(json!({"lectures": lectures})))
}

pub async fn get_source(
    State(state): State<AppState>,
    Path((lecture_key, slide_no)): Path<(String, i32)>,
) -> Result<Json<Value>, ApiError> {
    let row = sqlx::query("SELECT text, metadata FROM documents WHERE metadata->>'lecture_key'=$1 AND (metadata->>'slide_no')::int=$2 LIMIT 1")
        .bind(&lecture_key)
        .bind(slide_no)
        .fetch_optional(&state.pool)
        .await?;
    let Some(row) = row else {
        let not_found = Json(json!({"error": "Not found"}));
        return Ok(not_found);
    };
    let text: String = row.get("text");
    let metadata: Value = row.get("metadata");
    let supabase = state
        .settings
        .supabase_url
        .clone()
        .or_else(|| std::env::var("SUPABASE_URL").ok())
        .unwrap_or_default();
    let image_url = format!(
        "{}/storage/v1/object/public/slides/{}_slide_{}.jpg",
        supabase.trim_end_matches('/'),
        lecture_key,
        slide_no
    );
    Ok(Json(json!({
        "lecture_key": lecture_key,
        "slide_no": slide_no,
        "text": text,
        "metadata": metadata,
        "image_url": image_url,
    })))
}

pub async fn quiz_question(
    State(state): State<AppState>,
    Json(payload): Json<QuizQuestionRequest>,
) -> Result<Json<QuizQuestionResponse>, ApiError> {
    let has_lecture_selection = payload.lecture_key.is_some()
        || payload
            .lecture_keys
            .as_ref()
            .map(|keys| keys.iter().any(|k| !k.trim().is_empty()))
            .unwrap_or(false);
    if payload.topic.is_none() && !has_lecture_selection {
        return Err(ApiError(anyhow::anyhow!("Provide a topic or lecture key")));
    }
    let mut question_type = payload
        .question_type
        .clone()
        .unwrap_or_else(|| "short_answer".to_string());
    let allowed = ["true_false", "mcq_single", "mcq_multi", "short_answer"];
    if !allowed.contains(&question_type.as_str()) {
        question_type = "short_answer".to_string();
    }
    let mut lecture_cycle: Vec<String> = payload
        .lecture_keys
        .clone()
        .unwrap_or_default()
        .into_iter()
        .filter(|k| !k.trim().is_empty())
        .collect();
    if lecture_cycle.is_empty() {
        if let Some(single) = payload.lecture_key.clone() {
            if !single.trim().is_empty() {
                lecture_cycle.push(single);
            }
        }
    }
    let wants_all = lecture_cycle
        .iter()
        .any(|key| key.eq_ignore_ascii_case("all"));
    let lecture_for_question = if wants_all || lecture_cycle.is_empty() {
        None
    } else {
        let index = payload.question_index.unwrap_or(0);
        let idx = index % lecture_cycle.len();
        lecture_cycle.get(idx).cloned()
    };
    let query = payload
        .topic
        .clone()
        .or_else(|| {
            lecture_for_question
                .clone()
                .or(payload.lecture_key.clone())
                .map(|k| format!("Key ideas from {}", k))
        })
        .unwrap_or_else(|| "Key ideas from the course".to_string());
    let priority_hits = retrieve::retrieve_priority_hits(
        &state.pool,
        &state.embedder,
        &query,
        lecture_for_question.as_deref(),
        20,
    )
    .await?;
    if priority_hits.is_empty() {
        return Err(ApiError(anyhow::anyhow!(
            "No priority 1 documents available for quizzes"
        )));
    }
    let context = retrieve::build_context(&priority_hits);
    let question =
        llm::generate_quiz_item(&state.settings, &context, &query, &question_type).await?;
    Ok(Json(QuizQuestionResponse {
        question,
        context,
        lecture_key: lecture_for_question,
        topic: payload.topic.clone(),
    }))
}

pub async fn quiz_grade(
    State(state): State<AppState>,
    Json(payload): Json<QuizGradeRequest>,
) -> Result<Json<QuizGradeResponse>, ApiError> {
    let result = llm::grade_quiz_answer(
        &state.settings,
        &payload.question,
        &payload.user_answer,
        &payload.context,
    )
    .await?;
    Ok(Json(result))
}
