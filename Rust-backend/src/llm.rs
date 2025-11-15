use std::time::Instant;

use anyhow::{Context, Result};
use regex::Regex;
use reqwest::Client;
use serde_json::{Value, json};
use tokio::time::Duration;

use crate::{
    config::Settings,
    models::{
        ClaimCheckClaim, ClaimCheckResult, FactAiCheck, FactCheckAttempt, FactCheckResult,
        FactProgressCallback, FactProgressEvent, LlmInfo, LlmUsage, QuizGradeResponse,
        QuizQuestion,
    },
};

use std::collections::HashSet;

use tracing::warn;

const DEFAULT_GROQ_BASE: &str = "https://api.groq.com";

fn json_from_response(body: &str) -> Result<Value> {
    lazy_static::lazy_static! {
        static ref JSON_FENCE: Regex = Regex::new(r"```(?:json)?\s*(?P<body>.*?)```" ).unwrap();
    }
    if let Some(caps) = JSON_FENCE.captures(body) {
        let trimmed = caps
            .name("body")
            .map(|m| m.as_str().trim())
            .unwrap_or(body.trim());
        Ok(serde_json::from_str(trimmed)?)
    } else {
        Ok(serde_json::from_str(body.trim())?)
    }
}

async fn call_sig_gpt5(settings: &Settings, messages: &[Value]) -> Result<(String, LlmInfo)> {
    let base = settings
        .sig_gpt5_base
        .clone()
        .or_else(|| std::env::var("SIG_GPT5_BASE").ok())
        .context("SIG_GPT5_BASE not configured")?;
    let api_key = settings
        .sig_api_key
        .clone()
        .or_else(|| std::env::var("SIG_API_KEY").ok())
        .context("SIG_API_KEY not configured")?;
    let deployment = settings.sig_gpt5_deployment.clone();
    let api_version = settings.sig_api_version.clone();

    let url = format!(
        "{}/deployments/{}/chat/completions?api-version={}",
        base.trim_end_matches('/'),
        deployment,
        api_version
    );

    let payload = json!({
        "model": deployment,
        "messages": messages,
    });

    let client = Client::new();
    let start = Instant::now();
    let resp = client
        .post(url)
        .header("api-key", api_key)
        .header("Content-Type", "application/json")
        .json(&payload)
        .timeout(Duration::from_secs(90))
        .send()
        .await?;
    if !resp.status().is_success() {
        let text = resp.text().await.unwrap_or_default();
        anyhow::bail!("SIG GPT5 request failed: {}", text);
    }
    let elapsed = start.elapsed().as_secs_f32();
    let data: Value = resp.json().await?;
    let content = data["choices"][0]["message"]["content"]
        .as_str()
        .context("missing message content")?
        .to_string();
    let usage = data.get("usage").cloned().unwrap_or_else(
        || json!({"prompt_tokens": null, "completion_tokens": null, "total_tokens": null}),
    );
    let llm = LlmInfo {
        model: data["model"]
            .as_str()
            .map(|s| s.to_string())
            .or_else(|| Some(settings.sig_gpt5_deployment.clone())),
        latency_s: Some(elapsed),
        usage: LlmUsage {
            prompt_tokens: usage["prompt_tokens"].as_u64(),
            completion_tokens: usage["completion_tokens"].as_u64(),
            total_tokens: usage["total_tokens"].as_u64(),
        },
    };
    Ok((content, llm))
}

pub async fn groq_get_models(settings: &Settings) -> Result<Value> {
    let base = std::env::var("GROQ_API_BASE").unwrap_or_else(|_| DEFAULT_GROQ_BASE.to_string());
    let key = settings
        .groq_api_key
        .clone()
        .or_else(|| std::env::var("GROQ_API_KEY").ok())
        .context("GROQ_API_KEY not configured")?;
    let url = format!("{}/openai/v1/models", base.trim_end_matches('/'));
    let client = Client::new();
    let resp = client.get(url).bearer_auth(key).send().await?;
    if !resp.status().is_success() {
        let body = resp.text().await.unwrap_or_default();
        anyhow::bail!("groq get models failed: {}", body);
    }
    Ok(resp.json().await?)
}

pub struct FactCheckOutput {
    pub answer: Option<String>,
    pub fact_check: FactCheckResult,
    pub llm: LlmInfo,
}

const MAX_CONTEXT_LEN: usize = 30_000;
const MAX_DIRECTIVES_LEN: usize = 800;

fn emit_progress(handler: &Option<FactProgressCallback>, stage: &'static str, data: Value) {
    if let Some(callback) = handler {
        callback(&FactProgressEvent { stage, data });
    }
}

pub async fn run_fact_check_pipeline(
    settings: &Settings,
    query: &str,
    context: &str,
    progress: Option<FactProgressCallback>,
) -> Result<FactCheckOutput> {
    let max_attempts = resolve_fact_attempts();
    let threshold = resolve_fact_threshold(settings);
    let mut attempts: Vec<FactCheckAttempt> = Vec::new();
    let mut directives: Option<String> = None;
    let mut llm_info = LlmInfo::default();
    let mut final_answer = String::new();

    emit_progress(
        &progress,
        "pipeline_start",
        json!({
            "max_attempts": max_attempts,
            "threshold": threshold,
        }),
    );

    for attempt_idx in 0..max_attempts {
        emit_progress(
            &progress,
            "attempt_start",
            json!({
                "attempt": attempt_idx + 1,
                "directives": directives,
            }),
        );
        let (answer, llm) =
            answer_with_context(settings, query, context, directives.as_deref()).await?;
        final_answer = answer;
        llm_info = llm;

        emit_progress(
            &progress,
            "llm_result",
            json!({
                "attempt": attempt_idx + 1,
                "llm": llm_info,
            }),
        );
        let ai_check = fact_check_with_llm(settings, query, context, &final_answer).await;
        emit_progress(
            &progress,
            "fact_ai",
            json!({
                "attempt": attempt_idx + 1,
                "ai_check": ai_check,
            }),
        );
        let claim_check = claim_check(context, &final_answer, threshold);
        emit_progress(
            &progress,
            "fact_claims",
            json!({
                "attempt": attempt_idx + 1,
                "claims_check": claim_check,
            }),
        );

        let needs_retry = !(ai_check.passed && claim_check.passed);
        let attempt_record = FactCheckAttempt {
            attempt: attempt_idx + 1,
            needs_retry,
            directives: directives.clone(),
            ai_check: ai_check.clone(),
            claims_check: claim_check.clone(),
            answer_excerpt: excerpt(&final_answer, 200),
        };
        attempts.push(attempt_record);

        emit_progress(
            &progress,
            "attempt_complete",
            json!({
                "attempt": attempt_idx + 1,
                "needs_retry": needs_retry,
                "attempts": attempts.clone(),
                "retry_count": attempts.len().saturating_sub(1),
            }),
        );

        if !needs_retry {
            let fact_check = FactCheckResult {
                status: "passed".to_string(),
                retry_count: attempts.len().saturating_sub(1),
                threshold,
                max_attempts,
                attempts,
                message: None,
            };
            let answer_value = value_if_not_blank(&final_answer);
            emit_progress(
                &progress,
                "completed",
                json!({
                    "status": "passed",
                    "answer": answer_value,
                    "fact_check": fact_check.clone(),
                    "llm": llm_info.clone(),
                }),
            );
            return Ok(FactCheckOutput {
                answer: answer_value,
                fact_check,
                llm: llm_info,
            });
        }

        if attempt_idx + 1 < max_attempts {
            directives = Some(build_retry_directives(&ai_check, &claim_check));
            if let Some(directive) = &mut directives {
                if directive.len() > MAX_DIRECTIVES_LEN {
                    directive.truncate(MAX_DIRECTIVES_LEN);
                }
            }
        }
    }

    let last_details = attempts
        .last()
        .and_then(|a| a.claims_check.details.clone())
        .or_else(|| {
            attempts.last().and_then(|a| {
                (!a.ai_check.rationale.is_empty()).then(|| a.ai_check.rationale.clone())
            })
        });

    let fact_check = FactCheckResult {
        status: "failed".to_string(),
        retry_count: attempts.len().saturating_sub(1),
        threshold,
        max_attempts,
        attempts,
        message: last_details
            .or_else(|| Some("Unable to validate answer after retries.".to_string())),
    };
    let answer_value = value_if_not_blank(&final_answer);
    emit_progress(
        &progress,
        "completed",
        json!({
            "status": "failed",
            "answer": answer_value,
            "fact_check": fact_check.clone(),
            "llm": llm_info.clone(),
        }),
    );

    Ok(FactCheckOutput {
        answer: answer_value,
        fact_check,
        llm: llm_info,
    })
}

fn resolve_fact_attempts() -> usize {
    std::env::var("MAX_FACT_RETRIES")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&v| v > 0)
        .unwrap_or(3)
}

fn resolve_fact_threshold(settings: &Settings) -> f32 {
    settings.fact_check_threshold.unwrap_or_else(|| {
        std::env::var("FACT_THRESHOLD")
            .ok()
            .and_then(|v| v.parse::<f32>().ok())
            .unwrap_or(0.75)
    })
}

async fn answer_with_context(
    settings: &Settings,
    query: &str,
    context: &str,
    directives: Option<&str>,
) -> Result<(String, LlmInfo)> {
    let safe_context = if context.len() > MAX_CONTEXT_LEN {
        &context[..MAX_CONTEXT_LEN]
    } else {
        context
    };
    let mut system = "You are a precise tutor. Use ONLY the provided CONTEXT to answer."
        .to_string()
        + " Cite evidence inline using the original citation markers such as [Lecture X Slide Y]."
        + " If the context is insufficient, say so explicitly.";
    if let Some(extra) = directives.and_then(value_if_not_blank) {
        system.push_str(" Follow these directives carefully: ");
        system.push_str(&extra);
    }
    let prompt = format!("CONTEXT:\n{}\n\nQUESTION:\n{}", safe_context, query);
    let messages = vec![
        json!({"role": "system", "content": system}),
        json!({"role": "user", "content": prompt}),
    ];
    call_sig_gpt5(settings, &messages).await
}

async fn fact_check_with_llm(
    settings: &Settings,
    question: &str,
    context: &str,
    answer: &str,
) -> FactAiCheck {
    if answer.trim().is_empty() {
        return FactAiCheck {
            passed: false,
            confidence: 0.0,
            verdict: "fail".to_string(),
            rationale: "No answer produced for fact checking.".to_string(),
        };
    }

    let system = "You are an expert fact-checking system.\nYou must review the QUESTION, CONTEXT, and ANSWER.\nDetermine whether the ANSWER is fully supported by the CONTEXT.\nRespond in strict JSON format with keys: verdict (\"pass\" or \"fail\"), confidence (0.0-1.0), and rationale (string explaining the decision).\nDo not include any additional commentary outside of valid JSON.";
    let user = format!(
        "QUESTION:\n{}\n\nCONTEXT:\n{}\n\nANSWER:\n{}",
        question,
        truncate_for_fact_check(context),
        answer
    );
    let messages = vec![
        json!({"role": "system", "content": system}),
        json!({"role": "user", "content": user}),
    ];

    match call_sig_gpt5(settings, &messages).await {
        Ok((content, _)) => parse_fact_check_response(&content),
        Err(err) => {
            warn!("fact check LLM call failed: {:?}", err);
            FactAiCheck {
                passed: false,
                confidence: 0.0,
                verdict: "fail".to_string(),
                rationale: format!("Fact check call failed: {err}"),
            }
        }
    }
}

fn truncate_for_fact_check(context: &str) -> &str {
    if context.len() > MAX_CONTEXT_LEN {
        &context[..MAX_CONTEXT_LEN]
    } else {
        context
    }
}

fn parse_fact_check_response(content: &str) -> FactAiCheck {
    match json_from_response(content) {
        Ok(value) => {
            let verdict = value
                .get("verdict")
                .and_then(Value::as_str)
                .unwrap_or("fail")
                .to_lowercase();
            let confidence = value
                .get("confidence")
                .and_then(Value::as_f64)
                .map(|v| v.clamp(0.0, 1.0) as f32)
                .unwrap_or(0.0);
            let rationale = value
                .get("rationale")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .trim()
                .to_string();
            FactAiCheck {
                passed: verdict == "pass",
                confidence,
                verdict,
                rationale,
            }
        }
        Err(err) => {
            warn!("fact check JSON parse failed: {:?}", err);
            FactAiCheck {
                passed: false,
                confidence: 0.0,
                verdict: "fail".to_string(),
                rationale: format!("Invalid fact check response: {err}"),
            }
        }
    }
}

fn claim_check(context: &str, answer: &str, threshold: f32) -> ClaimCheckResult {
    let context_sentences = extract_sentences(context);
    let claims = generate_claims(answer);

    if context_sentences.is_empty() {
        return ClaimCheckResult {
            score: 0.0,
            entailed: 0,
            total_claims: claims.len(),
            threshold,
            passed: false,
            claims: Vec::new(),
            details: Some("No context available for claim validation.".to_string()),
        };
    }

    if claims.is_empty() {
        return ClaimCheckResult {
            score: 1.0,
            entailed: 0,
            total_claims: 0,
            threshold,
            passed: true,
            claims: Vec::new(),
            details: Some("No actionable claims detected in the answer.".to_string()),
        };
    }

    let mut records = Vec::with_capacity(claims.len());
    let mut entailed = 0usize;
    let mut sum_scores = 0.0f32;
    let mut failing_details = None;

    for claim in claims {
        let (best_idx, best_context, best_score) = best_context_match(&claim, &context_sentences);
        let label = classify_similarity(best_score, threshold);
        if label == "entailment" {
            entailed += 1;
        } else if failing_details.is_none() {
            failing_details = Some(format!(
                "Claim '{}' not fully supported by context '{}'.",
                truncate(&claim, 160),
                truncate(best_context, 160)
            ));
        }

        records.push(ClaimCheckClaim {
            claim: claim.clone(),
            context: best_context.to_string(),
            context_index: best_idx,
            label: label.to_string(),
            entailment_probability: best_score,
            neutral_probability: (1.0 - best_score) * 0.6,
            contradiction_probability: (1.0 - best_score) * 0.4,
        });
        sum_scores += best_score;
    }

    let avg_score = if records.is_empty() {
        0.0
    } else {
        sum_scores / records.len() as f32
    };
    let passed = !records.is_empty() && avg_score >= threshold;

    ClaimCheckResult {
        score: avg_score,
        entailed,
        total_claims: records.len(),
        threshold,
        passed,
        claims: records,
        details: if passed { None } else { failing_details },
    }
}

fn build_retry_directives(ai_check: &FactAiCheck, claim_check: &ClaimCheckResult) -> String {
    let mut parts = vec![
        "Revise the answer using only evidence from the provided context.".to_string(),
        "Cite sources using the original citation markers.".to_string(),
    ];

    if !ai_check.passed && !ai_check.rationale.trim().is_empty() {
        parts.push(format!("Validator feedback: {}", ai_check.rationale.trim()));
    }

    if !claim_check.passed {
        if let Some(failing) = claim_check
            .claims
            .iter()
            .find(|c| c.label.to_lowercase() != "entailment")
        {
            parts.push(format!(
                "Avoid unsupported claim: '{}'. Ground the answer near: '{}'.",
                truncate(&failing.claim, 160),
                truncate(&failing.context, 160)
            ));
        }
    }

    parts.join(" ")
}

fn value_if_not_blank(value: &str) -> Option<String> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_string())
    }
}

fn excerpt(text: &str, limit: usize) -> String {
    let mut excerpt = text.trim().chars().take(limit).collect::<String>();
    if text.trim().chars().count() > limit {
        excerpt.push_str("…");
    }
    excerpt
}

fn truncate(text: &str, limit: usize) -> String {
    let mut truncated = text.trim().chars().take(limit).collect::<String>();
    if text.trim().chars().count() > limit {
        truncated.push_str("…");
    }
    truncated
}

fn classify_similarity(score: f32, threshold: f32) -> &'static str {
    if score >= threshold {
        "entailment"
    } else if score >= threshold * 0.6 {
        "neutral"
    } else {
        "contradiction"
    }
}

fn best_context_match<'a>(claim: &str, sentences: &'a [String]) -> (usize, &'a str, f32) {
    let mut best_idx = 0usize;
    let mut best_sentence = "";
    let mut best_score = 0.0f32;
    for (idx, sentence) in sentences.iter().enumerate() {
        let score = overlap_score(claim, sentence);
        if score > best_score {
            best_score = score;
            best_idx = idx;
            best_sentence = sentence;
        }
    }
    (best_idx, best_sentence, best_score)
}

fn strip_citations(text: &str) -> String {
    lazy_static::lazy_static! {
        static ref CITE_RE: Regex = Regex::new(r"\[[^\]]+\]").unwrap();
    }
    CITE_RE.replace_all(text, "").to_string()
}

fn tokenize(text: &str) -> Vec<String> {
    strip_citations(text)
        .split_whitespace()
        .map(|token| {
            token
                .trim_matches(|c: char| !c.is_alphanumeric())
                .to_lowercase()
        })
        .filter(|token| !token.is_empty())
        .collect()
}

fn overlap_score(claim: &str, sentence: &str) -> f32 {
    let claim_tokens = tokenize(claim);
    if claim_tokens.is_empty() {
        return 0.0;
    }
    let context_tokens: HashSet<_> = tokenize(sentence).into_iter().collect();
    let common = claim_tokens
        .iter()
        .filter(|token| context_tokens.contains(*token))
        .count();
    common as f32 / claim_tokens.len() as f32
}

fn extract_sentences(text: &str) -> Vec<String> {
    text.split(|c| matches!(c, '.' | '!' | '?' | '\n'))
        .filter_map(|part| {
            let trimmed = part.trim();
            (!trimmed.is_empty()).then(|| strip_citations(trimmed))
        })
        .collect()
}

fn generate_claims(answer: &str) -> Vec<String> {
    extract_sentences(answer)
        .into_iter()
        .map(|s| {
            s.trim_start_matches(|c| matches!(c, '•' | '*' | '-' | ' '))
                .trim()
                .to_string()
        })
        .filter(|s| s.split_whitespace().count() >= 5 && s.chars().count() >= 20)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{Json, Router, routing::post};
    use serde_json::Value;
    use tokio::net::TcpListener;

    async fn mock_completion(Json(payload): Json<Value>) -> Json<Value> {
        let messages = payload
            .get("messages")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        let system_prompt = messages
            .get(0)
            .and_then(|m| m.get("content"))
            .and_then(Value::as_str)
            .unwrap_or("");
        let content = if system_prompt.contains("fact-checking") {
            "{\"verdict\":\"pass\",\"confidence\":0.92,\"rationale\":\"supported\"}".to_string()
        } else {
            "The capital of Italy is Rome [Lecture 1 Slide 2].".to_string()
        };
        Json(json!({
            "id": "mock",
            "object": "chat.completion",
            "created": 0,
            "model": "mock-model",
            "choices": [{
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": content}
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 15,
                "total_tokens": 25
            }
        }))
    }

    fn test_settings(base: String) -> Settings {
        Settings {
            bind_host: "127.0.0.1".to_string(),
            port: 8000,
            supabase_url: None,
            supabase_service_role_key: None,
            database_url: "postgres://example".to_string(),
            groq_api_key: None,
            embedding_dim: 384,
            embedding_model: None,
            sig_name: None,
            sig_api_key: Some("test-key".to_string()),
            sig_gpt5_base: Some(base),
            sig_api_version: "2025-01-01-preview".to_string(),
            sig_gpt5_deployment: "mock".to_string(),
            huggingface_token: None,
            job_max_attempts: None,
            fact_check_threshold: Some(0.6),
        }
    }

    #[tokio::test]
    async fn fact_check_pipeline_passes() {
        let router =
            Router::new().route("/deployments/mock/chat/completions", post(mock_completion));
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            axum::serve(listener, router).await.unwrap();
        });

        unsafe {
            std::env::set_var("MAX_FACT_RETRIES", "1");
        }

        let settings = test_settings(format!("http://{}", addr));
        let context = "Lecture 1 Slide 2 discusses the capital city of Italy, which is Rome.";
        let result =
            run_fact_check_pipeline(&settings, "What is the capital of Italy?", context, None)
                .await
                .unwrap();

        assert_eq!(result.fact_check.status, "passed");
        assert_eq!(result.fact_check.attempts.len(), 1);
        let attempt = &result.fact_check.attempts[0];
        assert!(attempt.ai_check.passed);
        assert!(attempt.claims_check.passed);
        assert!(
            result
                .answer
                .as_ref()
                .unwrap()
                .to_lowercase()
                .contains("rome")
        );

        server.abort();
    }
}

pub async fn generate_quiz_item(
    settings: &Settings,
    context: &str,
    topic: &str,
    question_type: &str,
) -> Result<QuizQuestion> {
    let safe_context = if context.len() > 6500 {
        &context[..6500]
    } else {
        context
    };
    let system =
        "You are a helpful tutor creating a quiz question. Return JSON following the schema.";
    let prompt = format!(
        "Context:\n{}\n\nTopic: {}\nQuestion type: {}\nRespond with the JSON fields question_type, question_prompt, options, correct_answer, answer_rubric, hint, answer_explanation.",
        safe_context, topic, question_type
    );
    let messages = vec![
        json!({"role": "system", "content": system}),
        json!({"role": "user", "content": prompt}),
    ];
    let (raw, _) = call_sig_gpt5(settings, &messages).await?;
    let payload = json_from_response(&raw)?;
    let question = serde_json::from_value(payload)?;
    Ok(question)
}

pub async fn grade_quiz_answer(
    settings: &Settings,
    question: &QuizQuestion,
    user_answer: &Value,
    context: &str,
) -> Result<QuizGradeResponse> {
    let serialized_question = serde_json::to_string(question)?;
    let answer_text = match user_answer {
        Value::Array(items) => items
            .iter()
            .map(|v| v.as_str().unwrap_or(""))
            .filter(|s| !s.is_empty())
            .collect::<Vec<_>>()
            .join("\n"),
        Value::String(s) => s.clone(),
        Value::Null => String::new(),
        other => other.to_string(),
    };
    let safe_context = if context.len() > 6500 {
        &context[..6500]
    } else {
        context
    };
    let system = "You are grading a student's quiz response. Return JSON with keys correct, score, assessment, good_points, bad_points.";
    let prompt = format!(
        "Context:\n{}\n\nQuestion JSON:\n{}\n\nStudent answer:\n{}",
        safe_context, serialized_question, answer_text
    );
    let messages = vec![
        json!({"role": "system", "content": system}),
        json!({"role": "user", "content": prompt}),
    ];
    let (raw, _) = call_sig_gpt5(settings, &messages).await?;
    let payload = json_from_response(&raw)?;
    let grade = serde_json::from_value(payload)?;
    Ok(grade)
}
