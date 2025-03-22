use crate::core::config::Config as AppConfig;
use anyhow::Result;
use sqlx::postgres::{PgPool, PgPoolOptions};
use std::time::Duration;

pub async fn create_pool(config: &AppConfig) -> Result<PgPool> {
    let connection_string = format!(
        "postgres://{}:{}@{}:{}/{}",
        config.database.username,
        config.database.password,
        config.database.host,
        config.database.port,
        config.database.database
    );

    let pool = PgPoolOptions::new()
        .max_connections(5)
        .acquire_timeout(Duration::from_secs(3))
        .connect(&connection_string)
        .await
        .map_err(|e| anyhow::anyhow!("Failed to create database pool: {}", e))?;

    Ok(pool)
}

pub async fn check_connection(pool: &PgPool) -> Result<()> {
    sqlx::query("SELECT 1").execute(pool).await?;
    Ok(())
}
