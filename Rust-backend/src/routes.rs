use anyhow::Result;
use axum::{
    Json,
    extract::{Multipart, Path, State},
    http::StatusCode,
    response::{IntoResponse, Response},
};
use serde_json::{Value, json};
use sqlx::Row;
use std::sync::Arc;
use tokio::task;
use tracing::error;

use crate::{
    config::Settings,
    db::DbPool,
    embedder::Embedder,
    ingest,
    jobs::{JobManager, JobRecord},
    llm::{self, FactCheckOutput},
    models::{
        AsyncJobResponse, FactProgressCallback, FactProgressEvent, HealthResponse, MemorizeRequest,
        QueryOptions, QueryRequest, QueryResponse, QuizGradeRequest, QuizGradeResponse,
        QuizQuestionRequest, QuizQuestionResponse, RetrieveResult,
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
    let interaction_id: i64 =
        sqlx::query("INSERT INTO user_interactions(user_id, q_text) VALUES ($1,$2) RETURNING id")
            .bind(&payload.user_id)
            .bind(&payload.text)
            .fetch_one(&state.pool)
            .await?
            .get("id");
    let embeddings = state.embedder.embed([payload.text.as_str()]).await?;
    let vector = pgvector::Vector::from(embeddings[0].clone());
    sqlx::query("INSERT INTO chunks (store_kind, tenant_id, doc_id, chunk_index, text, embedding, metadata) VALUES (3,$1,NULL,0,$2,$3,$4)")
        .bind(&payload.user_id)
        .bind(&payload.text)
        .bind(vector)
        .bind(json!({"source": "user_note", "interaction_id": interaction_id}))
        .execute(&state.pool)
        .await?;
    Ok(Json(json!({"ok": true})))
}

pub async fn query(
    State(state): State<AppState>,
    Json(payload): Json<QueryRequest>,
) -> Result<Json<QueryResponse>, ApiError> {
    let options = payload.options.unwrap_or(QueryOptions {
        force_lecture_key: None,
        use_global: true,
        user_id: None,
    });
    let RetrieveResult {
        diagnostics, hits, ..
    } = retrieve::retrieve(
        &state.pool,
        &state.embedder,
        &payload.query,
        options.force_lecture_key.as_deref(),
        options.use_global,
        options.user_id.as_deref(),
    )
    .await?;
    let context = retrieve::build_context(&hits);
    let FactCheckOutput {
        answer,
        fact_check,
        llm,
    } = llm::run_fact_check_pipeline(
        &state.settings,
        Some(&state.embedder),
        &payload.query,
        &context,
        None,
    )
    .await?;
    let response = QueryResponse {
        diagnostics,
        top_k: hits.len(),
        hits,
        context_len: context.len(),
        llm_model: llm.model.clone(),
        llm_latency_s: llm.latency_s,
        llm_usage: llm.usage.clone(),
        answer,
        fact_check,
    };
    Ok(Json(response))
}

pub async fn query_async(
    State(state): State<AppState>,
    Json(payload): Json<QueryRequest>,
) -> Result<Json<AsyncJobResponse>, ApiError> {
    let job_id = state.jobs.create_job(None);
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
    let options = payload.options.unwrap_or(QueryOptions {
        force_lecture_key: None,
        use_global: true,
        user_id: None,
    });
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
            return Ok(());
        }
    };
    let context = retrieve::build_context(&retrieval.hits);
    state.jobs.update_job(
        &job_id,
        json!({
            "phase": "llm",
            "diagnostics": serde_json::to_value(&retrieval.diagnostics).unwrap_or(json!({})),
            "hits": serde_json::to_value(&retrieval.hits).unwrap_or(json!([])),
            "top_k": retrieval.hits.len(),
            "context_len": context.len(),
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
                let retry_count = event
                    .data
                    .get("retry_count")
                    .and_then(Value::as_u64)
                    .unwrap_or_default() as usize;
                let needs_retry = event
                    .data
                    .get("needs_retry")
                    .and_then(Value::as_bool)
                    .unwrap_or(false);
                let mut payload = json!({
                    "retry_count": retry_count,
                    "message": if needs_retry {
                        "AI response could not be validated, retrying..."
                    } else {
                        ""
                    },
                });
                if let Some(attempts) = event.data.get("attempts").cloned() {
                    payload["attempts"] = attempts;
                }
                jobs_for_progress.update_job(&job_id_for_progress, payload);
            }
            _ => {}
        });

    match llm::run_fact_check_pipeline(
        &state.settings,
        Some(&state.embedder),
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
            let passed = fact_check.status == "passed";
            let final_status = if passed { "succeeded" } else { "failed" };
            let message = if passed {
                String::new()
            } else {
                fact_check.message.clone().unwrap_or_else(|| {
                    "AI response could not be validated after retries.".to_string()
                })
            };
            state.jobs.update_job(
                &job_id,
                json!({
                    "status": final_status,
                    "phase": "done",
                    "answer": answer,
                    "fact_check": serde_json::to_value(&fact_check).unwrap_or(json!({})),
                    "llm": serde_json::to_value(llm).unwrap_or(json!({})),
                    "message": message,
                }),
            );
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
    Ok(Json(job))
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
    if payload.topic.is_none() && payload.lecture_key.is_none() {
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
    let query = payload
        .topic
        .clone()
        .or_else(|| {
            payload
                .lecture_key
                .clone()
                .map(|k| format!("Key ideas from {}", k))
        })
        .unwrap_or_else(|| "Key ideas from the course".to_string());
    let retrieval = retrieve::retrieve(
        &state.pool,
        &state.embedder,
        &query,
        payload.lecture_key.as_deref(),
        payload.lecture_key.is_none(),
        None,
    )
    .await?;
    let context = retrieve::build_context(&retrieval.hits);
    let question =
        llm::generate_quiz_item(&state.settings, &context, &query, &question_type).await?;
    Ok(Json(QuizQuestionResponse {
        question,
        context,
        lecture_key: payload.lecture_key.clone(),
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
