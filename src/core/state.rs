use crate::core::config::Config;
use crate::core::embedding::ClipEmbedder;
use anyhow::Result;
use candle_core::Device;
use sqlx::postgres::PgPool;
use std::path::Path;
use std::sync::Arc;

/// Application state containing shared resources
pub struct AppState {
    pub config: Config,
    pub db_pool: PgPool,
    pub embedder: Arc<ClipEmbedder>,
}

// TODO: load models from huggingface ids instead? or make paths optional?
impl AppState {
    /// Create a new application state with all resources initialized
    pub async fn new(config: Config) -> Result<Self> {
        // Create database pool
        let db_pool = crate::core::db::create_pool(&config).await?;
        crate::core::db::check_connection(&db_pool).await?;

        // Initialize CLIP model
        let embedder = ClipEmbedder::new(
            config.embedding.model_path.as_ref().map(Path::new).unwrap(),
            config
                .embedding
                .tokenizer_path
                .as_ref()
                .map(Path::new)
                .unwrap(),
            Device::cuda_if_available(0)?,
        )?;

        Ok(Self {
            config,
            db_pool,
            embedder: Arc::new(embedder),
        })
    }
}
