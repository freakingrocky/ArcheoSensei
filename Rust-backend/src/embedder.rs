use anyhow::{anyhow, Result};
use blake3::Hasher;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use tracing::warn;

use crate::config::Settings;

#[derive(Clone)]
pub struct Embedder {
    dim: usize,
    backend: Option<HttpBackend>,
}

#[derive(Clone)]
struct HttpBackend {
    client: Client,
    url: String,
    api_key: Option<String>,
}

#[derive(Serialize)]
struct HttpEmbedRequest {
    inputs: Vec<String>,
}

#[derive(Deserialize)]
struct HttpEmbedResponse {
    embeddings: Vec<Vec<f32>>,
    #[serde(default)]
    dimension: Option<usize>,
}

impl Embedder {
    pub async fn new(settings: &Settings) -> Result<Self> {
        let dim = settings.embedding_dim;
        let service_url = std::env::var("EMBEDDING_SERVICE_URL")
            .ok()
            .or_else(|| settings.embedding_service_url.clone());
        let api_key = std::env::var("EMBEDDING_SERVICE_API_KEY")
            .ok()
            .or_else(|| settings.embedding_service_api_key.clone());

        let backend = if let Some(url) = service_url {
            let client = Client::builder()
                .user_agent("archeosensei-rust-backend/0.1")
                .build()?;
            Some(HttpBackend {
                client,
                url,
                api_key,
            })
        } else {
            None
        };

        Ok(Self { dim, backend })
    }

    pub fn dimension(&self) -> usize {
        self.dim
    }

    pub async fn embed<'a>(&self, texts: impl IntoIterator<Item = &'a str>) -> Result<Vec<Vec<f32>>> {
        let payload: Vec<String> = texts.into_iter().map(|t| t.to_string()).collect();
        if payload.is_empty() {
            return Ok(Vec::new());
        }

        if let Some(http) = &self.backend {
            match http.embed(&payload, self.dim).await {
                Ok(embeds) => return Ok(embeds),
                Err(err) => warn!(error = %err, "remote embedding failed, falling back to deterministic hashing"),
            }
        }

        Ok(payload
            .iter()
            .map(|text| deterministic_embedding(text, self.dim))
            .collect())
    }
}

impl HttpBackend {
    async fn embed(&self, texts: &[String], expected_dim: usize) -> Result<Vec<Vec<f32>>> {
        let request = HttpEmbedRequest {
            inputs: texts.to_vec(),
        };
        let mut req = self.client.post(&self.url).json(&request);
        if let Some(key) = &self.api_key {
            req = req.header("Authorization", key);
        }
        let response = req.send().await?.error_for_status()?;
        let parsed: HttpEmbedResponse = response.json().await?;
        if let Some(dim) = parsed.dimension {
            if dim != expected_dim {
                return Err(anyhow!(
                    "embedding dimension mismatch: expected {expected_dim}, got {dim}"
                ));
            }
        }
        if parsed.embeddings.len() != texts.len() {
            return Err(anyhow!(
                "embedding service returned {} embeddings for {} inputs",
                parsed.embeddings.len(),
                texts.len()
            ));
        }
        for emb in &parsed.embeddings {
            if emb.len() != expected_dim {
                return Err(anyhow!(
                    "embedding service produced vector of length {} (expected {})",
                    emb.len(),
                    expected_dim
                ));
            }
        }
        Ok(parsed.embeddings)
    }
}

fn deterministic_embedding(text: &str, dim: usize) -> Vec<f32> {
    let mut hasher = Hasher::new();
    hasher.update(text.as_bytes());
    let mut reader = hasher.finalize_xof();
    let mut buf = [0u8; 8];
    let mut values = Vec::with_capacity(dim);
    let mut sum_sq = 0.0f64;

    for _ in 0..dim {
        reader.fill(&mut buf);
        let value = u64::from_le_bytes(buf);
        let normalized = (value as f64 / u64::MAX as f64) * 2.0 - 1.0;
        sum_sq += normalized * normalized;
        values.push(normalized as f32);
    }

    if sum_sq > 0.0 {
        let scale = (1.0 / sum_sq.sqrt()) as f32;
        for v in &mut values {
            *v *= scale;
        }
    }

    values
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn hashing_backend_returns_expected_length() {
        let settings = Settings {
            bind_host: "0.0.0.0".to_string(),
            port: 8000,
            supabase_url: None,
            supabase_service_role_key: None,
            database_url: "postgres://localhost/test".to_string(),
            groq_api_key: None,
            embedding_dim: 16,
            embedding_model: None,
            embedding_service_url: None,
            embedding_service_api_key: None,
            sig_name: None,
            sig_api_key: None,
            sig_gpt5_base: None,
            sig_api_version: "2025-01-01-preview".to_string(),
            sig_gpt5_deployment: "gpt-5-mini".to_string(),
            huggingface_token: None,
            job_max_attempts: None,
            fact_check_threshold: None,
        };
        let embedder = Embedder::new(&settings).await.expect("init");
        let vectors = embedder.embed(["hello", "world"]).await.expect("embed");
        assert_eq!(vectors.len(), 2);
        assert!(vectors.iter().all(|v| v.len() == 16));
    }
}
