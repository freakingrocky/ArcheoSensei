mod config;
mod db;
mod embedder;
mod ingest;
mod jobs;
mod llm;
mod models;
mod retrieve;
mod routes;

use crate::config::Settings;
use crate::embedder::Embedder;
use crate::jobs::JobManager;
use crate::routes::{
    get_source, health, list_lectures, llm_models, memorize, query, query_async,
    query_async_status, quiz_grade, quiz_question, upload_lectures,
};
use axum::{
    routing::{get, post},
    Router,
};
use tokio::signal;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

use tower_http::cors::CorsLayer;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    dotenvy::dotenv().ok();
    tracing_subscriber::registry()
        .with(tracing_subscriber::EnvFilter::new(
            std::env::var("RUST_LOG").unwrap_or_else(|_| "info".into()),
        ))
        .with(tracing_subscriber::fmt::layer())
        .init();

    let settings = Settings::load()?;
    let pool = db::init_pool(&settings.database_url).await?;
    let embedder = Embedder::new(&settings).await?;
    let jobs = JobManager::default();

    let app_state = routes::AppState::new(
        settings.clone(),
        pool.clone(),
        embedder.clone(),
        jobs.clone(),
    );

    let app = Router::new()
        .route("/health", get(health))
        .route("/upload/lectures", post(upload_lectures))
        .route("/memorize", post(memorize))
        .route("/query", post(query))
        .route("/query/async", post(query_async))
        .route("/query/async/:job_id", get(query_async_status))
        .route("/llm/models", get(llm_models))
        .route("/lectures", get(list_lectures))
        .route("/source/:lecture_key/:slide_no", get(get_source))
        .route("/quiz/question", post(quiz_question))
        .route("/quiz/grade", post(quiz_grade))
        .with_state(app_state)
        .layer(CorsLayer::permissive());

    let listener =
        tokio::net::TcpListener::bind((settings.bind_host.as_str(), settings.port)).await?;
    tracing::info!("listening on {}:{}", settings.bind_host, settings.port);
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;

    Ok(())
}

async fn shutdown_signal() {
    let ctrl_c = async {
        signal::ctrl_c()
            .await
            .expect("failed to install Ctrl+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        use tokio::signal::unix::{signal, SignalKind};

        signal(SignalKind::terminate())
            .expect("failed to install signal handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {},
        _ = terminate => {},
    }
}
