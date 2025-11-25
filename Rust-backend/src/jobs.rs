use chrono::{DateTime, Utc};
use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Duration;
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct JobMetadata {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub user_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chat_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chat_name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub query: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_fetch: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct JobRecord {
    pub job_id: String,
    pub status: String,
    pub phase: String,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    #[serde(default)]
    pub attempts: Vec<serde_json::Value>,
    #[serde(default)]
    pub fact_ai_status: Option<String>,
    #[serde(default)]
    pub fact_claims_status: Option<String>,
    #[serde(default)]
    pub metadata: JobMetadata,
    #[serde(flatten)]
    pub extra: serde_json::Map<String, serde_json::Value>,
}

impl JobRecord {
    pub fn new(initial: Option<JobMetadata>) -> Self {
        let job_id = Uuid::new_v4().simple().to_string();
        let now = Utc::now();
        let record = JobRecord {
            job_id: job_id.clone(),
            status: "queued".to_string(),
            phase: "queued".to_string(),
            created_at: now,
            updated_at: now,
            attempts: Vec::new(),
            fact_ai_status: None,
            fact_claims_status: None,
            metadata: initial.unwrap_or_default(),
            extra: serde_json::Map::new(),
        };
        record
    }
}

#[derive(Clone)]
pub struct JobManager {
    inner: Arc<DashMap<String, JobRecord>>,
    ttl: Duration,
}

impl Default for JobManager {
    fn default() -> Self {
        Self {
            inner: Arc::new(DashMap::new()),
            ttl: Duration::from_secs(900),
        }
    }
}

impl JobManager {
    fn prune(&self) {
        let cutoff = Utc::now()
            - chrono::Duration::from_std(self.ttl)
                .unwrap_or_else(|_| chrono::Duration::minutes(15));
        let to_remove: Vec<String> = self
            .inner
            .iter()
            .filter(|entry| entry.value().updated_at < cutoff)
            .map(|entry| entry.key().clone())
            .collect();
        for key in to_remove {
            self.inner.remove(&key);
        }
    }

    pub fn create_job(&self, initial: Option<JobMetadata>) -> String {
        self.prune();
        let record = JobRecord::new(initial);
        let job_id = record.job_id.clone();
        self.inner.insert(job_id.clone(), record);
        job_id
    }

    pub fn get_job(&self, job_id: &str) -> Option<JobRecord> {
        self.prune();
        self.inner.get(job_id).map(|entry| entry.value().clone())
    }

    pub fn update_job(&self, job_id: &str, fields: serde_json::Value) {
        if let Some(mut entry) = self.inner.get_mut(job_id) {
            if let serde_json::Value::Object(map) = fields {
                for (k, v) in map {
                    match k.as_str() {
                        "status" => entry.status = v.as_str().unwrap_or(&entry.status).to_string(),
                        "phase" => entry.phase = v.as_str().unwrap_or(&entry.phase).to_string(),
                        "attempts" => {
                            if let Some(arr) = v.as_array() {
                                entry.attempts = arr.clone();
                            }
                        }
                        "fact_ai_status" => {
                            entry.fact_ai_status = v.as_str().map(|s| s.to_string())
                        }
                        "fact_claims_status" => {
                            entry.fact_claims_status = v.as_str().map(|s| s.to_string())
                        }
                        other => {
                            entry.extra.insert(other.to_string(), v);
                        }
                    }
                }
            }
            entry.updated_at = Utc::now();
        }
    }

    #[allow(dead_code)]
    pub fn append_attempt(&self, job_id: &str, attempt: serde_json::Value) {
        if let Some(mut entry) = self.inner.get_mut(job_id) {
            entry.attempts.push(attempt);
            entry.updated_at = Utc::now();
        }
    }

    pub fn touch_fetch(&self, job_id: &str) {
        if let Some(mut entry) = self.inner.get_mut(job_id) {
            entry.metadata.last_fetch = Some(Utc::now());
            entry.updated_at = Utc::now();
        }
    }
}
