use std::str::FromStr;
use std::sync::Arc;

use anyhow::{Result, anyhow};
use async_trait::async_trait;
use fastembed::{EmbeddingModel, TextEmbedding, TextInitOptions};
use parking_lot::Mutex;
use tokio::task;

use crate::config::Settings;

#[derive(Clone)]
pub struct Embedder {
    inner: Arc<Mutex<TextEmbedding>>,
    dim: usize,
}

impl Embedder {
    pub async fn new(settings: &Settings) -> Result<Self> {
        let model_name = settings
            .embedding_model
            .clone()
            .unwrap_or_else(|| "bge-small-en-v1.5".to_string());
        let embedding_model = resolve_model(&model_name)?;
        let info = TextEmbedding::get_model_info(&embedding_model)?;
        let options =
            TextInitOptions::new(embedding_model.clone()).with_show_download_progress(false);
        let text_embedding = TextEmbedding::try_new(options)?;
        Ok(Self {
            inner: Arc::new(Mutex::new(text_embedding)),
            dim: info.dim as usize,
        })
    }

    pub fn dimension(&self) -> usize {
        self.dim
    }

    pub async fn embed<'a>(
        &self,
        texts: impl IntoIterator<Item = &'a str>,
    ) -> Result<Vec<Vec<f32>>> {
        let payload: Vec<String> = texts.into_iter().map(|t| t.to_string()).collect();
        let model = self.inner.clone();
        let embeddings = task::spawn_blocking(move || {
            let mut guard = model.lock();
            let result = guard.embed(payload, None)?;
            Ok::<_, anyhow::Error>(result)
        })
        .await??;
        Ok(embeddings)
    }
}

#[async_trait]
pub trait SentenceEmbedder: Send + Sync {
    async fn embed_strings(&self, texts: &[String]) -> Result<Vec<Vec<f32>>>;
}

#[async_trait]
impl SentenceEmbedder for Embedder {
    async fn embed_strings(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        self.embed(texts.iter().map(|s| s.as_str())).await
    }
}

fn resolve_model(name: &str) -> Result<EmbeddingModel> {
    let normalized = name.trim();
    if let Ok(model) = EmbeddingModel::from_str(normalized) {
        return Ok(model);
    }
    // map common aliases
    let alias = normalized.to_lowercase();
    let mapped = match alias.as_str() {
        "baai/bge-small-en-v1.5" | "bge-small-en-v1.5" => Some(EmbeddingModel::BGESmallENV15),
        "baai/bge-base-en-v1.5" | "bge-base-en-v1.5" => Some(EmbeddingModel::BGEBaseENV15),
        "baai/bge-large-en-v1.5" | "bge-large-en-v1.5" => Some(EmbeddingModel::BGELargeENV15),
        _ => None,
    };
    mapped.ok_or_else(|| anyhow!("Unsupported embedding model: {name}"))
}
