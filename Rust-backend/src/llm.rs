use std::time::Instant;

use anyhow::{Context, Result};
use regex::Regex;
use reqwest::Client;
use serde_json::{Value, json};
use tokio::time::Duration;

use crate::{
    config::Settings,
    embedder::Embedder,
    models::{
        ClaimCheckClaim, ClaimCheckResult, FactAiCheck, FactCheckAttempt, FactCheckResult,
        FactProgressCallback, FactProgressEvent, LlmInfo, LlmUsage, QuizGradeResponse,
        QuizQuestion,
    },
};

use rand::{Rng, SeedableRng, rngs::StdRng};
use std::collections::HashSet;
use std::f32::consts::LN_2;

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
const CLAIM_SENTENCE_MIN_CHARS: usize = 20;
const CLAIM_SENTENCE_LIMIT: usize = 240;
const KMEANS_MAX_ITER: usize = 60;
const KMEANS_TOL: f32 = 1e-4;
const K_MAX: usize = 24;
const K_MIN: usize = 1;
const CENTROID_MATCH_TAU: f32 = 0.83;
const MASS_RATIO_BETA: f32 = 0.45;
const MIN_CTX_SHARE: f32 = 0.15;
const MIN_ANS_COUNT: usize = 1;
const MERGE_TAU: f32 = 0.90;
const COVERAGE_MIN: f32 = 0.70;

fn emit_progress(handler: &Option<FactProgressCallback>, stage: &'static str, data: Value) {
    if let Some(callback) = handler {
        callback(&FactProgressEvent { stage, data });
    }
}

pub async fn run_fact_check_pipeline(
    settings: &Settings,
    query: &str,
    context: &str,
    embedder: Option<&Embedder>,
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
        let claim_check = claim_check(embedder, context, &final_answer, threshold).await;
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

async fn claim_check(
    embedder: Option<&Embedder>,
    context: &str,
    answer: &str,
    threshold: f32,
) -> ClaimCheckResult {
    if let Some(embedder) = embedder {
        match claim_check_topics(embedder, context, answer, threshold).await {
            Ok(result) => return result,
            Err(err) => {
                warn!("claim check embedding path failed: {:?}", err);
            }
        }
    }
    heuristic_claim_check(context, answer, threshold)
}

async fn claim_check_topics(
    embedder: &Embedder,
    context: &str,
    answer: &str,
    threshold: f32,
) -> Result<ClaimCheckResult> {
    let mut ctx_sentences = extract_claim_sentences(context);
    if ctx_sentences.is_empty() {
        return Ok(ClaimCheckResult {
            score: 0.0,
            entailed: 0,
            total_claims: 0,
            threshold,
            passed: false,
            claims: Vec::new(),
            details: Some("No context sentences available for validation.".to_string()),
        });
    }
    if ctx_sentences.len() > CLAIM_SENTENCE_LIMIT {
        ctx_sentences.truncate(CLAIM_SENTENCE_LIMIT);
    }

    let mut ans_sentences = extract_claim_sentences(answer);
    if ans_sentences.is_empty() {
        return Ok(ClaimCheckResult {
            score: 0.0,
            entailed: 0,
            total_claims: 0,
            threshold,
            passed: false,
            claims: Vec::new(),
            details: Some("No answer sentences available for validation.".to_string()),
        });
    }
    if ans_sentences.len() > CLAIM_SENTENCE_LIMIT {
        ans_sentences.truncate(CLAIM_SENTENCE_LIMIT);
    }

    let mut ctx_emb = embedder
        .embed(ctx_sentences.iter().map(|s| s.as_str()))
        .await?;
    let mut ans_emb = embedder
        .embed(ans_sentences.iter().map(|s| s.as_str()))
        .await?;

    normalize_rows(&mut ctx_emb);
    normalize_rows(&mut ans_emb);

    let n_ctx = ctx_emb.len();
    let n_ans = ans_emb.len();
    let mut k = (n_ctx as f32).sqrt().round() as usize;
    k = k.clamp(K_MIN, K_MAX);
    if k > n_ctx {
        k = n_ctx.max(K_MIN);
    }
    if k <= 1 {
        let score = 1.0;
        let passed = score >= threshold;
        return Ok(ClaimCheckResult {
            score,
            entailed: if n_ans > 0 { 1 } else { 0 },
            total_claims: 1,
            threshold,
            passed,
            claims: Vec::new(),
            details: (!passed)
                .then_some("Insufficient context diversity for topic clustering.".to_string()),
        });
    }

    let (ctx_labels, centroids) = kmeans(&ctx_emb, k, KMEANS_MAX_ITER, KMEANS_TOL);
    let (centroids, ctx_labels) = merge_centroids(centroids, &ctx_labels, MERGE_TAU);
    let k_eff = centroids.len();
    let p_ctx = histogram(&ctx_labels, k_eff);
    let normalized_centroids: Vec<Vec<f32>> =
        centroids.iter().map(|c| normalized_copy(c)).collect();
    let (_, p_ans, ans_counts, max_cos_by_topic) =
        assign_answer_topics(&ans_emb, &normalized_centroids);

    let jsd = jensen_shannon_distance(&p_ctx, &p_ans);
    let score = (1.0 - jsd).clamp(0.0, 1.0);

    let mut covered = vec![false; k_eff];
    for j in 0..k_eff {
        if p_ans[j] >= MASS_RATIO_BETA * p_ctx[j] {
            covered[j] = true;
        }
    }
    for (j, count) in ans_counts.iter().enumerate() {
        if *count >= MIN_ANS_COUNT {
            covered[j] = true;
        }
    }
    for (j, cos) in max_cos_by_topic.iter().enumerate() {
        if *cos >= CENTROID_MATCH_TAU {
            covered[j] = true;
        }
    }

    let significant: Vec<bool> = p_ctx.iter().map(|share| *share >= MIN_CTX_SHARE).collect();
    let any_significant = significant.iter().any(|flag| *flag);
    let sig_count = if any_significant {
        significant.iter().filter(|flag| **flag).count()
    } else {
        k_eff
    };
    let covered_sig = if any_significant {
        significant
            .iter()
            .enumerate()
            .filter(|(idx, flag)| **flag && covered[*idx])
            .count()
    } else {
        covered.iter().filter(|flag| **flag).count()
    };
    let coverage = if sig_count > 0 {
        covered_sig as f32 / sig_count as f32
    } else {
        1.0
    };
    let passed = (score >= threshold) || (coverage >= COVERAGE_MIN);

    let mut missing_claims = Vec::new();
    if any_significant {
        for topic_idx in 0..k_eff {
            if !significant[topic_idx] || covered[topic_idx] {
                continue;
            }
            if let Some(rep_idx) =
                representative_sentence(topic_idx, &ctx_labels, &ctx_emb, &normalized_centroids)
            {
                let sentence = ctx_sentences.get(rep_idx).cloned().unwrap_or_default();
                missing_claims.push(ClaimCheckClaim {
                    claim: format!("Missing topic: {}", truncate(&sentence, 180)),
                    context: sentence,
                    context_index: rep_idx,
                    label: "missing_topic".to_string(),
                    entailment_probability: p_ans.get(topic_idx).copied().unwrap_or(0.0),
                    neutral_probability: p_ctx.get(topic_idx).copied().unwrap_or(0.0),
                    contradiction_probability: 1.0 - coverage,
                });
            }
        }
    }

    let mut details = None;
    if !passed {
        let mut summary = format!(
            "Evidence coverage {:.0}% (target {:.0}%) · similarity {:.0}% (target {:.0}%).",
            coverage * 100.0,
            COVERAGE_MIN * 100.0,
            score * 100.0,
            threshold * 100.0
        );
        if let Some(first_missing) = missing_claims.first() {
            summary.push_str(" Key gap: ");
            summary.push_str(&first_missing.claim);
        }
        details = Some(summary);
    }

    Ok(ClaimCheckResult {
        score,
        entailed: covered_sig,
        total_claims: sig_count,
        threshold,
        passed,
        claims: missing_claims,
        details,
    })
}

fn heuristic_claim_check(context: &str, answer: &str, threshold: f32) -> ClaimCheckResult {
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

fn extract_claim_sentences(text: &str) -> Vec<String> {
    extract_sentences(text)
        .into_iter()
        .map(|s| s.trim().to_string())
        .filter(|s| s.chars().count() >= CLAIM_SENTENCE_MIN_CHARS)
        .collect()
}

fn normalize_rows(rows: &mut [Vec<f32>]) {
    for row in rows {
        normalize(row);
    }
}

fn normalize(row: &mut [f32]) {
    let norm = row.iter().map(|v| v * v).sum::<f32>().sqrt();
    if norm > 1e-6 {
        for v in row {
            *v /= norm;
        }
    }
}

fn normalized_copy(row: &[f32]) -> Vec<f32> {
    let mut copy = row.to_vec();
    normalize(&mut copy);
    copy
}

fn histogram(labels: &[usize], k: usize) -> Vec<f32> {
    let mut hist = vec![0.0f32; k];
    for &label in labels {
        if let Some(slot) = hist.get_mut(label) {
            *slot += 1.0;
        }
    }
    let sum: f32 = hist.iter().sum();
    if sum > 0.0 {
        for value in &mut hist {
            *value /= sum;
        }
    }
    hist
}

fn assign_answer_topics(
    answers: &[Vec<f32>],
    centroids: &[Vec<f32>],
) -> (Vec<usize>, Vec<f32>, Vec<usize>, Vec<f32>) {
    let k = centroids.len();
    if answers.is_empty() || k == 0 {
        return (Vec::new(), vec![0.0; k], vec![0; k], vec![0.0; k]);
    }
    let mut labels = vec![0usize; answers.len()];
    let mut counts = vec![0usize; k];
    for (idx, answer) in answers.iter().enumerate() {
        let mut best_label = 0usize;
        let mut best_score = -1.0f32;
        for (j, centroid) in centroids.iter().enumerate() {
            let score = dot(answer, centroid);
            if score > best_score {
                best_score = score;
                best_label = j;
            }
        }
        labels[idx] = best_label;
        counts[best_label] += 1;
    }
    let mut hist = vec![0.0f32; k];
    for &label in &labels {
        hist[label] += 1.0;
    }
    for value in &mut hist {
        *value /= labels.len() as f32;
    }
    let mut max_cos = vec![0.0f32; k];
    for answer in answers {
        for (j, centroid) in centroids.iter().enumerate() {
            let score = dot(answer, centroid);
            if score > max_cos[j] {
                max_cos[j] = score;
            }
        }
    }
    (labels, hist, counts, max_cos)
}

fn jensen_shannon_distance(p: &[f32], q: &[f32]) -> f32 {
    fn normalize_prob(vec: &[f32]) -> Vec<f32> {
        let mut out = Vec::with_capacity(vec.len());
        let sum: f32 = vec.iter().map(|v| v.max(1e-12)).sum();
        for &v in vec {
            out.push(v.max(1e-12) / sum.max(1e-12));
        }
        out
    }
    fn kl(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| x * ((*x / *y).ln()))
            .sum::<f32>()
            / LN_2
    }
    let p_norm = normalize_prob(p);
    let q_norm = normalize_prob(q);
    let mut m = vec![0.0f32; p_norm.len()];
    for i in 0..p_norm.len() {
        m[i] = 0.5 * (p_norm[i] + q_norm[i]);
    }
    let jsd = 0.5 * (kl(&p_norm, &m) + kl(&q_norm, &m));
    jsd.sqrt()
}

fn representative_sentence(
    topic_idx: usize,
    labels: &[usize],
    embeddings: &[Vec<f32>],
    centroids: &[Vec<f32>],
) -> Option<usize> {
    let mut best_idx = None;
    let mut best_distance = f32::MAX;
    let centroid = centroids.get(topic_idx)?;
    for (idx, label) in labels.iter().enumerate() {
        if *label != topic_idx {
            continue;
        }
        let distance = 1.0 - dot(&embeddings[idx], centroid);
        if distance < best_distance {
            best_distance = distance;
            best_idx = Some(idx);
        }
    }
    best_idx
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| x * y)
        .sum::<f32>()
        .clamp(-1.0, 1.0)
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

fn kmeans(data: &[Vec<f32>], k: usize, max_iter: usize, tol: f32) -> (Vec<usize>, Vec<Vec<f32>>) {
    let mut rng = StdRng::seed_from_u64(17);
    let mut centroids = kmeans_pp_init(data, k, &mut rng);
    let mut labels = vec![0usize; data.len()];
    for _ in 0..max_iter {
        let mut changed = false;
        for (idx, vector) in data.iter().enumerate() {
            let mut best = 0usize;
            let mut best_dist = f32::MAX;
            for (j, centroid) in centroids.iter().enumerate() {
                let dist = squared_distance(vector, centroid);
                if dist < best_dist {
                    best_dist = dist;
                    best = j;
                }
            }
            if labels[idx] != best {
                labels[idx] = best;
                changed = true;
            }
        }
        let mut new_centroids = vec![vec![0.0f32; data[0].len()]; k];
        let mut counts = vec![0usize; k];
        for (label, vector) in labels.iter().zip(data.iter()) {
            if let Some(row) = new_centroids.get_mut(*label) {
                for (dst, src) in row.iter_mut().zip(vector.iter()) {
                    *dst += *src;
                }
                counts[*label] += 1;
            }
        }
        for (idx, row) in new_centroids.iter_mut().enumerate() {
            if counts[idx] > 0 {
                let inv = 1.0 / counts[idx] as f32;
                for value in row.iter_mut() {
                    *value *= inv;
                }
            } else {
                *row = data[rng.gen_range(0..data.len())].clone();
            }
        }
        let mut shift = 0.0f32;
        for (old, new) in centroids.iter().zip(new_centroids.iter()) {
            shift += squared_distance(old, new);
        }
        centroids = new_centroids;
        if !changed || shift <= tol {
            break;
        }
    }
    (labels, centroids)
}

fn kmeans_pp_init(data: &[Vec<f32>], k: usize, rng: &mut StdRng) -> Vec<Vec<f32>> {
    let mut centroids = Vec::with_capacity(k);
    let first = rng.gen_range(0..data.len());
    centroids.push(data[first].clone());
    let mut distances: Vec<f32> = data
        .iter()
        .map(|point| squared_distance(point, &centroids[0]))
        .collect();
    while centroids.len() < k {
        let sum: f32 = distances.iter().sum();
        if sum == 0.0 {
            centroids.push(data[rng.gen_range(0..data.len())].clone());
        } else {
            let mut target = rng.r#gen::<f32>() * sum;
            let mut idx = 0usize;
            while target > 0.0 && idx < distances.len() {
                target -= distances[idx];
                idx += 1;
            }
            let chosen = idx.saturating_sub(1).min(data.len() - 1);
            centroids.push(data[chosen].clone());
        }
        for (i, point) in data.iter().enumerate() {
            let dist = squared_distance(point, centroids.last().unwrap());
            if dist < distances[i] {
                distances[i] = dist;
            }
        }
    }
    centroids
}

fn merge_centroids(
    centroids: Vec<Vec<f32>>,
    labels: &[usize],
    tau: f32,
) -> (Vec<Vec<f32>>, Vec<usize>) {
    if centroids.len() <= 1 {
        return (centroids, labels.to_vec());
    }
    let normalized: Vec<Vec<f32>> = centroids.iter().map(|c| normalized_copy(c)).collect();
    let mut used = vec![false; centroids.len()];
    let mut mapping = vec![0usize; centroids.len()];
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
        let mut aggregate = vec![0.0f32; centroids[0].len()];
        for &idx in &group {
            for (dst, src) in aggregate.iter_mut().zip(centroids[idx].iter()) {
                *dst += *src;
            }
        }
        let inv = 1.0 / group.len() as f32;
        for value in &mut aggregate {
            *value *= inv;
        }
        let new_idx = merged.len();
        for idx in group {
            mapping[idx] = new_idx;
        }
        merged.push(aggregate);
    }
    let mut new_labels = Vec::with_capacity(labels.len());
    for &label in labels {
        new_labels.push(mapping[label]);
    }
    (merged, new_labels)
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
        let result = run_fact_check_pipeline(
            &settings,
            "What is the capital of Italy?",
            context,
            None,
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
