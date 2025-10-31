use serde::Deserialize;

#[allow(dead_code)]
#[derive(Debug, Clone, Deserialize)]
pub struct Settings {
    #[serde(default = "default_bind_host")]
    pub bind_host: String,
    #[serde(default = "default_port")]
    pub port: u16,
    pub supabase_url: Option<String>,
    pub supabase_service_role_key: Option<String>,
    pub database_url: String,
    pub groq_api_key: Option<String>,
    #[serde(default = "default_embedding_dim")]
    pub embedding_dim: usize,
    #[serde(default)]
    pub embedding_model: Option<String>,
    #[serde(default)]
    pub embedding_service_url: Option<String>,
    #[serde(default)]
    pub embedding_service_api_key: Option<String>,
    pub sig_name: Option<String>,
    pub sig_api_key: Option<String>,
    pub sig_gpt5_base: Option<String>,
    #[serde(default = "default_sig_api_version")]
    pub sig_api_version: String,
    #[serde(default = "default_sig_deployment")]
    pub sig_gpt5_deployment: String,
    #[serde(default)]
    pub huggingface_token: Option<String>,
    #[serde(default)]
    pub job_max_attempts: Option<u32>,
    #[serde(default)]
    pub fact_check_threshold: Option<f32>,
}

impl Settings {
    pub fn load() -> anyhow::Result<Self> {
        let mut cfg =
            config::Config::builder().add_source(config::Environment::default().separator("__"));

        if let Ok(path) = std::env::var("BACKEND_CONFIG_FILE") {
            cfg = cfg.add_source(config::File::with_name(&path).required(false));
        }

        let built = cfg.build()?;
        let mut settings: Settings = built.try_deserialize()?;
        if settings.embedding_dim == 0 {
            settings.embedding_dim = default_embedding_dim();
        }
        if settings.port == 0 {
            settings.port = default_port();
        }
        if settings.bind_host.is_empty() {
            settings.bind_host = default_bind_host();
        }
        Ok(settings)
    }
}

fn default_bind_host() -> String {
    "0.0.0.0".to_string()
}

fn default_port() -> u16 {
    std::env::var("BACKEND_PORT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(8000)
}

fn default_embedding_dim() -> usize {
    std::env::var("EMBEDDING_DIM")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(384)
}

fn default_sig_api_version() -> String {
    std::env::var("SIG_API_VERSION").unwrap_or_else(|_| "2025-01-01-preview".to_string())
}

fn default_sig_deployment() -> String {
    std::env::var("SIG_GPT5_DEPLOYMENT").unwrap_or_else(|_| "gpt-5-mini".to_string())
}
