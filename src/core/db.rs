use crate::core::config::Config as AppConfig;
use anyhow::Result;
use deadpool_postgres::{Config, Pool, Runtime};
use tokio_postgres::NoTls;

pub async fn create_pool(config: &AppConfig) -> Result<Pool> {
    let mut pg_config = tokio_postgres::config::Config::new();
    pg_config.host(&config.database.host);
    pg_config.port(config.database.port);
    pg_config.user(&config.database.username);
    pg_config.password(&config.database.password);
    pg_config.dbname(&config.database.database);

    let pool_config = Config::new();
    Pool::builder(pool_config)
        .build(pg_config, NoTls, Runtime::Tokio1)
        .map_err(|e| anyhow::anyhow!("Failed to create database pool: {}", e))
}

pub async fn check_connection(pool: &Pool) -> Result<()> {
    let client = pool.get().await?;
    client.execute("SELECT 1", &[]).await?;
    Ok(())
}
