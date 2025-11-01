use serde::{Deserialize, Serialize};
use serde_with::{DisplayFromStr, serde_as};
use std::collections::HashMap;
use uuid::Uuid;

#[derive(Debug, Clone, Deserialize, Default)]
#[serde(default)]
pub struct QueryOptions {
    pub force_lecture_key: Option<String>,
    pub use_global: bool,
    pub user_id: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct QueryRequest {
    pub query: String,
    pub options: Option<QueryOptions>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct MemorizeRequest {
    pub user_id: String,
    pub text: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct RetrieveDiagnostics {
    pub lecture_detected: Option<String>,
    pub lecture_votes: HashMap<String, f32>,
    pub lecture_forced: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct RetrieveHit {
    pub id: Uuid,
    pub text: String,
    pub metadata: serde_json::Value,
    pub score: f32,
    pub citation: Option<String>,
    pub file_url: Option<String>,
    pub tag: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct RetrieveResult {
    pub diagnostics: RetrieveDiagnostics,
    pub hits: Vec<RetrieveHit>,
    pub label: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct LlmUsage {
    pub prompt_tokens: Option<u64>,
    pub completion_tokens: Option<u64>,
    pub total_tokens: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct LlmInfo {
    pub model: Option<String>,
    pub latency_s: Option<f32>,
    pub usage: LlmUsage,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct FactCheckResult {
    pub status: Option<String>,
    pub score: Option<f32>,
    pub message: Option<String>,
    pub details: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct QueryResponse {
    pub diagnostics: RetrieveDiagnostics,
    pub top_k: usize,
    pub hits: Vec<RetrieveHit>,
    pub context_len: usize,
    pub llm_model: Option<String>,
    pub llm_latency_s: Option<f32>,
    pub llm_usage: LlmUsage,
    pub answer: Option<String>,
    pub fact_check: FactCheckResult,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuizQuestion {
    pub question_type: String,
    pub question_prompt: String,
    pub options: Option<Vec<String>>,
    pub correct_answer: serde_json::Value,
    pub answer_rubric: String,
    pub hint: String,
    pub answer_explanation: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct QuizQuestionRequest {
    pub lecture_key: Option<String>,
    pub topic: Option<String>,
    pub question_type: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct QuizQuestionResponse {
    pub question: QuizQuestion,
    pub context: String,
    pub lecture_key: Option<String>,
    pub topic: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct QuizGradeRequest {
    pub question: QuizQuestion,
    pub context: String,
    pub user_answer: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuizGradeResponse {
    pub correct: bool,
    pub score: f32,
    pub assessment: String,
    pub good_points: Vec<String>,
    pub bad_points: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UploadLectureForm {
    pub course: Option<String>,
}

#[serde_as]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LectureSummary {
    pub lecture_key: String,
    #[serde_as(as = "DisplayFromStr")]
    pub count: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LectureListResponse {
    pub lectures: Vec<LectureSummary>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceResponse {
    pub lecture_key: String,
    pub slide_no: i32,
    pub text: String,
    pub image_url: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthResponse {
    pub status: String,
    pub db: bool,
    pub embedding_dim: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AsyncJobResponse {
    pub job_id: String,
}
