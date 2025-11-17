use std::time::Instant;

use anyhow::{Context, Result};
use rand::{Rng, SeedableRng, rngs::StdRng};
use regex::Regex;
use reqwest::Client;
use serde_json::{Value, json};
use tokio::time::Duration;

use crate::{
    config::Settings,
    embedder::SentenceEmbedder,
    models::{
        ClaimCheckClaim, ClaimCheckResult, FactAiCheck, FactCheckAttempt, FactCheckResult,
        FactProgressCallback, FactProgressEvent, LlmInfo, LlmUsage, QuizGradeResponse,
        QuizQuestion,
    },
};

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

pub async fn run_fact_check_pipeline<E>(
    settings: &Settings,
    embedder: &E,
    query: &str,
    context: &str,
    progress: Option<FactProgressCallback>,
) -> Result<FactCheckOutput>
where
    E: SentenceEmbedder,
{
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
        let claim_check = claim_check(embedder, context, &final_answer, threshold).await?;
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
        + " If the context is insufficient, say so explicitly."
        + " When the context includes lecture image metadata (IMG_URL, TITLE, DESCRIPTION, NOTES, LECTURE, AREA_DESCRIPTION) you may embed a visual reference.";
    system.push_str(
        "\nTo embed an annotated image, emit a fenced code block labelled annotated-image that contains JSON like:\n"
    );
    system.push_str(
        "```annotated-image\\n{\n  \"img_url\": \"https://...\",\n  \"title\": \"Temple Pediment\",\n  \"description\": \"Key scene\",\n  \"lecture\": \"Lecture 5\",\n  \"notes\": \"Use to discuss figure posture\",\n  \"highlights\": [\n    {\n      \"label\": \"Central Column\",\n      \"x\": 0.15,\n      \"y\": 0.1,\n      \"w\": 0.2,\n      \"h\": 0.4,\n      \"color\": \"#FFB020\",\n      \"help_text\": \"Weathering on column capital\"\n    }\n  ]\n}\n```\n"
    );
    system.push_str(
        "Coordinates are normalized 0-1. Always pair each annotated-image block with surrounding prose that explains why the image matters and references the original slide citation."
    );
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

async fn claim_check<E: SentenceEmbedder>(
    embedder: &E,
    context: &str,
    answer: &str,
    threshold: f32,
) -> Result<ClaimCheckResult> {
    const K_MAX: usize = 24;
    const K_MIN: usize = 1;
    const CENTROID_MATCH_TAU: f32 = 0.83;
    const MASS_RATIO_BETA: f32 = 0.45;
    const MIN_CTX_SHARE: f32 = 0.15;
    const MIN_ANS_COUNT: usize = 1;
    const MERGE_TAU: f32 = 0.90;
    const COVERAGE_MIN: f32 = 0.70;
    const MIN_SENTENCE_CHARS: usize = 20;

    let ctx_sents = extract_sentences(context, MIN_SENTENCE_CHARS);
    let ans_sents = extract_sentences(answer, MIN_SENTENCE_CHARS);

    if ctx_sents.is_empty() && ans_sents.is_empty() {
        return Ok(ClaimCheckResult {
            score: 0.0,
            entailed: 0,
            total_claims: 0,
            threshold,
            passed: false,
            claims: Vec::new(),
            details: Some("No sentences found for validation.".to_string()),
        });
    }

    if ctx_sents.is_empty() {
        return Ok(ClaimCheckResult {
            score: 0.0,
            entailed: 0,
            total_claims: ans_sents.len(),
            threshold,
            passed: false,
            claims: Vec::new(),
            details: Some("No context sentences available.".to_string()),
        });
    }

    if ans_sents.is_empty() {
        return Ok(ClaimCheckResult {
            score: 0.0,
            entailed: 0,
            total_claims: ctx_sents.len(),
            threshold,
            passed: false,
            claims: Vec::new(),
            details: Some("No answer sentences detected.".to_string()),
        });
    }

    let mut ctx_emb = embedder.embed_strings(&ctx_sents).await?;
    let mut ans_emb = embedder.embed_strings(&ans_sents).await?;
    normalize_rows(&mut ctx_emb);
    normalize_rows(&mut ans_emb);

    let mut k = (f32::sqrt(ctx_sents.len() as f32).round() as usize).clamp(K_MIN, K_MAX);
    if k > ctx_sents.len() {
        k = ctx_sents.len().max(K_MIN);
    }

    if k <= 1 {
        let score = 1.0;
        return Ok(ClaimCheckResult {
            score,
            entailed: if !ans_sents.is_empty() { 1 } else { 0 },
            total_claims: 1,
            threshold,
            passed: score >= threshold,
            claims: Vec::new(),
            details: None,
        });
    }

    let (labels, centroids) = kmeans(&ctx_emb, k, 60, 1e-4);
    if centroids.is_empty() {
        return Ok(ClaimCheckResult {
            score: 0.0,
            entailed: 0,
            total_claims: 0,
            threshold,
            passed: false,
            claims: Vec::new(),
            details: Some("Unable to derive context topics for validation.".to_string()),
        });
    }
    let (centroids, labels) = merge_centroids(centroids, labels, MERGE_TAU);
    let k_eff = centroids.len();
    if k_eff == 0 {
        return Ok(ClaimCheckResult {
            score: 0.0,
            entailed: 0,
            total_claims: 0,
            threshold,
            passed: false,
            claims: Vec::new(),
            details: Some("Context clustering collapsed during validation.".to_string()),
        });
    }

    let p_ctx = histogram(&labels, k_eff);
    let ans_labels = assign_to_centroids(&ans_emb, &centroids);
    let p_ans = histogram(&ans_labels, k_eff);
    let jsd = js_distance(&p_ctx, &p_ans);
    let score = (1.0 - jsd).clamp(0.0, 1.0);

    let significant: Vec<bool> = p_ctx
        .iter()
        .map(|&share| share >= MIN_CTX_SHARE as f64)
        .collect();
    let significant_any = significant.iter().any(|&flag| flag);
    let mut covered_flags = vec![false; k_eff];

    for topic_idx in 0..k_eff {
        if p_ans[topic_idx] >= (MASS_RATIO_BETA as f64) * p_ctx[topic_idx] {
            covered_flags[topic_idx] = true;
        }
        let count = ans_labels
            .iter()
            .filter(|&&label| label == topic_idx)
            .count();
        if count >= MIN_ANS_COUNT {
            covered_flags[topic_idx] = true;
        }
    }

    let mut normalized_centroids = centroids.clone();
    normalize_rows(&mut normalized_centroids);
    let mut max_cos_by_topic = vec![0.0f32; k_eff];
    for (topic_idx, centroid) in normalized_centroids.iter().enumerate() {
        let mut best = 0.0f32;
        for ans in &ans_emb {
            let cos = dot(ans, centroid);
            if cos > best {
                best = cos;
            }
        }
        if best >= CENTROID_MATCH_TAU {
            covered_flags[topic_idx] = true;
        }
        max_cos_by_topic[topic_idx] = best;
    }

    let sig_count = if significant_any {
        significant.iter().filter(|&&flag| flag).count().max(1)
    } else {
        k_eff.max(1)
    };
    let covered_sig = if significant_any {
        significant
            .iter()
            .zip(covered_flags.iter())
            .filter(|pair| *pair.0 && *pair.1)
            .count()
    } else {
        covered_flags.iter().filter(|&&covered| covered).count()
    };
    let coverage = covered_sig as f32 / sig_count as f32;
    let passed = (score >= threshold) || (coverage >= COVERAGE_MIN);

    let mut claims = Vec::new();
    if significant_any {
        for (topic_idx, (&sig, &covered)) in
            significant.iter().zip(covered_flags.iter()).enumerate()
        {
            if !sig || covered {
                continue;
            }
            let members: Vec<usize> = labels
                .iter()
                .enumerate()
                .filter_map(|(idx, &label)| (label == topic_idx).then_some(idx))
                .collect();
            if members.is_empty() {
                continue;
            }
            let mut best_idx = members[0];
            let mut best_cos = -1.0f32;
            for &idx in &members {
                let cos = dot(&ctx_emb[idx], &normalized_centroids[topic_idx]);
                if cos > best_cos {
                    best_cos = cos;
                    best_idx = idx;
                }
            }
            let rep = ctx_sents[best_idx].clone();
            claims.push(ClaimCheckClaim {
                claim: format!("Missing topic: {}", truncate(&rep, 180)),
                context: rep,
                context_index: best_idx,
                label: "missing_topic".to_string(),
                entailment_probability: 0.0,
                neutral_probability: 0.0,
                contradiction_probability: 1.0,
                topic_index: Some(topic_idx),
                topic_share_context: Some(p_ctx[topic_idx] as f32),
                topic_share_answer: Some(p_ans[topic_idx] as f32),
                max_cos_to_answer: Some(max_cos_by_topic[topic_idx]),
            });
        }
    }

    let detail = if passed {
        None
    } else {
        Some(format!(
            "Evidence coverage {:.0}% (target {:.0}%) and topic alignment {:.0}% (target {:.0}%).",
            coverage * 100.0,
            COVERAGE_MIN * 100.0,
            score * 100.0,
            threshold * 100.0
        ))
    };

    Ok(ClaimCheckResult {
        score,
        entailed: covered_sig,
        total_claims: sig_count,
        threshold,
        passed,
        claims,
        details: detail,
    })
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
            let claim = truncate(&failing.claim, 160);
            let ctx = truncate(&failing.context, 160);
            if failing.label.to_lowercase() == "missing_topic" {
                parts.push(format!(
                    "Address the missing topic '{}'. Incorporate evidence near '{}'.",
                    claim, ctx
                ));
            } else {
                parts.push(format!(
                    "Avoid unsupported claim: '{}'. Ground the answer near: '{}'.",
                    claim, ctx
                ));
            }
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

fn strip_citations(text: &str) -> String {
    lazy_static::lazy_static! {
        static ref CITE_RE: Regex = Regex::new(r"\[[^\]]+\]").unwrap();
    }
    CITE_RE.replace_all(text, "").to_string()
}

fn normalize_rows(data: &mut [Vec<f32>]) {
    for row in data.iter_mut() {
        let norm = row.iter().map(|v| (*v as f64).powi(2)).sum::<f64>().sqrt() as f32;
        if norm > 1e-6 {
            for value in row.iter_mut() {
                *value /= norm;
            }
        }
    }
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn squared_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let diff = x - y;
            diff * diff
        })
        .sum()
}

fn kmeans_pp_init(data: &[Vec<f32>], k: usize, rng: &mut StdRng) -> Vec<Vec<f32>> {
    let mut centroids = Vec::with_capacity(k);
    let mut chosen = rng.gen_range(0..data.len());
    centroids.push(data[chosen].clone());
    let mut distances: Vec<f32> = data
        .iter()
        .map(|point| squared_distance(point, &centroids[0]))
        .collect();
    for _ in 1..k {
        let total: f32 = distances.iter().sum::<f32>().max(f32::EPSILON);
        let target = rng.gen_range(0.0..total);
        let mut cumulative = 0.0;
        chosen = 0;
        for (idx, dist) in distances.iter().enumerate() {
            cumulative += *dist;
            if cumulative >= target {
                chosen = idx;
                break;
            }
        }
        centroids.push(data[chosen].clone());
        for (idx, point) in data.iter().enumerate() {
            let dist = squared_distance(point, centroids.last().unwrap());
            if dist < distances[idx] {
                distances[idx] = dist;
            }
        }
    }
    centroids
}

fn kmeans(data: &[Vec<f32>], k: usize, max_iter: usize, tol: f32) -> (Vec<usize>, Vec<Vec<f32>>) {
    let dim = data[0].len();
    let mut rng = StdRng::seed_from_u64(42);
    let mut centroids = kmeans_pp_init(data, k, &mut rng);
    let mut labels = vec![0usize; data.len()];
    let mut counts = vec![0usize; k];

    for _ in 0..max_iter {
        let mut changed = false;
        for (idx, point) in data.iter().enumerate() {
            let mut best_idx = 0usize;
            let mut best_dist = f32::MAX;
            for (c_idx, centroid) in centroids.iter().enumerate() {
                let dist = squared_distance(point, centroid);
                if dist < best_dist {
                    best_dist = dist;
                    best_idx = c_idx;
                }
            }
            if labels[idx] != best_idx {
                labels[idx] = best_idx;
                changed = true;
            }
        }

        let mut new_centroids = vec![vec![0.0f32; dim]; k];
        counts.fill(0);
        for (point, &label) in data.iter().zip(labels.iter()) {
            for (value, coord) in new_centroids[label].iter_mut().zip(point.iter()) {
                *value += *coord;
            }
            counts[label] += 1;
        }

        for (centroid, &count) in new_centroids.iter_mut().zip(counts.iter()) {
            if count > 0 {
                for value in centroid.iter_mut() {
                    *value /= count as f32;
                }
            } else {
                let idx = rng.gen_range(0..data.len());
                *centroid = data[idx].clone();
            }
        }

        let shift = centroids
            .iter()
            .zip(new_centroids.iter())
            .map(|(old, new)| squared_distance(old, new))
            .sum::<f32>()
            .sqrt();
        centroids = new_centroids;
        if !changed || shift <= tol {
            break;
        }
    }

    (labels, centroids)
}

fn merge_centroids(
    centroids: Vec<Vec<f32>>,
    labels: Vec<usize>,
    tau: f32,
) -> (Vec<Vec<f32>>, Vec<usize>) {
    if centroids.len() <= 1 {
        return (centroids, labels);
    }
    let mut normalized = centroids.clone();
    normalize_rows(&mut normalized);
    let mut used = vec![false; centroids.len()];
    let mut remap = vec![usize::MAX; centroids.len()];
    let mut merged = Vec::new();

    for i in 0..centroids.len() {
        if used[i] {
            continue;
        }
        used[i] = true;
        let mut group = vec![i];
        for j in (i + 1)..centroids.len() {
            if used[j] {
                continue;
            }
            if dot(&normalized[i], &normalized[j]) >= tau {
                used[j] = true;
                group.push(j);
            }
        }
        let mut avg = vec![0.0f32; centroids[0].len()];
        for &idx in &group {
            for (value, coord) in avg.iter_mut().zip(centroids[idx].iter()) {
                *value += *coord;
            }
        }
        for value in avg.iter_mut() {
            *value /= group.len() as f32;
        }
        let new_idx = merged.len();
        for &idx in &group {
            remap[idx] = new_idx;
        }
        merged.push(avg);
    }

    let mut new_labels = labels;
    for label in new_labels.iter_mut() {
        if let Some(mapped) = remap.get(*label) {
            *label = *mapped;
        }
    }

    (merged, new_labels)
}

fn histogram(labels: &[usize], k: usize) -> Vec<f64> {
    let mut hist = vec![0.0f64; k];
    for &label in labels {
        if label < k {
            hist[label] += 1.0;
        }
    }
    let sum: f64 = hist.iter().sum();
    if sum > 0.0 {
        for value in hist.iter_mut() {
            *value /= sum;
        }
    } else if k > 0 {
        hist[0] = 1.0;
    }
    hist
}

fn assign_to_centroids(points: &[Vec<f32>], centroids: &[Vec<f32>]) -> Vec<usize> {
    if centroids.is_empty() {
        return Vec::new();
    }
    let mut normalized = centroids.to_vec();
    normalize_rows(&mut normalized);
    points
        .iter()
        .map(|point| {
            let mut best_idx = 0usize;
            let mut best_dot = f32::MIN;
            for (idx, centroid) in normalized.iter().enumerate() {
                let sim = dot(point, centroid);
                if sim > best_dot {
                    best_dot = sim;
                    best_idx = idx;
                }
            }
            best_idx
        })
        .collect()
}

fn js_distance(p: &[f64], q: &[f64]) -> f32 {
    fn normalize(dist: &[f64]) -> Vec<f64> {
        let mut norm = dist.iter().cloned().collect::<Vec<_>>();
        let sum: f64 = norm.iter().sum();
        if sum > 0.0 {
            for value in norm.iter_mut() {
                *value = (*value).clamp(1e-12, 1.0);
            }
            let new_sum: f64 = norm.iter().sum();
            for value in norm.iter_mut() {
                *value /= new_sum;
            }
        } else if !norm.is_empty() {
            norm[0] = 1.0;
        }
        norm
    }

    let p_norm = normalize(p);
    let q_norm = normalize(q);
    let m: Vec<f64> = p_norm
        .iter()
        .zip(q_norm.iter())
        .map(|(a, b)| 0.5 * (a + b))
        .collect();

    fn kl(a: &[f64], b: &[f64]) -> f64 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| x * ((x / y).ln() / std::f64::consts::LN_2))
            .sum::<f64>()
    }

    let jsd = 0.5 * (kl(&p_norm, &m) + kl(&q_norm, &m));
    jsd.sqrt() as f32
}

fn split_sentences(text: &str) -> Vec<String> {
    let mut sentences = Vec::new();
    let mut current = String::new();
    for ch in text.chars() {
        if ch == '\n' {
            if !current.trim().is_empty() {
                sentences.push(current.trim().to_string());
            }
            current.clear();
            continue;
        }
        current.push(ch);
        if matches!(ch, '.' | '!' | '?') {
            if !current.trim().is_empty() {
                sentences.push(current.trim().to_string());
            }
            current.clear();
        }
    }
    if !current.trim().is_empty() {
        sentences.push(current.trim().to_string());
    }
    sentences
}

fn extract_sentences(text: &str, min_chars: usize) -> Vec<String> {
    split_sentences(text)
        .into_iter()
        .map(|s| {
            strip_citations(&s)
                .trim_start_matches(|c: char| matches!(c, '•' | '*' | '-' | ' '))
                .trim()
                .to_string()
        })
        .filter(|s| s.chars().count() >= min_chars)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{Json, Router, routing::post};
    use serde_json::Value;
    use tokio::net::TcpListener;

    use crate::embedder::SentenceEmbedder;

    #[derive(Clone, Default)]
    struct StubEmbedder;

    #[async_trait::async_trait]
    impl SentenceEmbedder for StubEmbedder {
        async fn embed_strings(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
            Ok(texts.iter().map(|t| simple_embed(t)).collect())
        }
    }

    fn simple_embed(text: &str) -> Vec<f32> {
        let mut vec = vec![0.0f32; 8];
        for (idx, ch) in text.chars().enumerate() {
            let bucket = idx % vec.len();
            vec[bucket] += ((ch as u32) % 17) as f32 / 17.0;
        }
        vec
    }

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
        let embedder = StubEmbedder::default();
        let result = run_fact_check_pipeline(
            &settings,
            &embedder,
            "What is the capital of Italy?",
            context,
            None,
        )
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
