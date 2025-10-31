use sqlx::{postgres::PgPoolOptions, Pool, Postgres};

pub type DbPool = Pool<Postgres>;

pub async fn init_pool(database_url: &str) -> anyhow::Result<DbPool> {
    let pool = PgPoolOptions::new()
        .max_connections(16)
        .min_connections(2)
        .connect(database_url)
        .await?;
    Ok(pool)
}
