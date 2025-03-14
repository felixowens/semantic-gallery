use anyhow::Result;
use serde::Deserialize;
use std::error::Error;
use std::path::PathBuf;

#[derive(Debug, Deserialize, Clone)]
pub struct Config {
    pub database: DatabaseConfig,
    pub embedding: EmbeddingConfig,
    pub storage: StorageConfig,
}

#[derive(Debug, Deserialize, Clone)]
pub struct DatabaseConfig {
    pub host: String,
    pub port: u16,
    pub username: String,
    pub password: String,
    pub database: String,
}

#[derive(Debug, Deserialize, Clone)]
pub struct EmbeddingConfig {
    pub model_path: PathBuf,
    pub dimension: usize,
}

#[derive(Debug, Deserialize, Clone)]
pub struct StorageConfig {
    pub media_path: PathBuf,
}

pub fn load_config() -> Result<Config, Box<dyn Error>> {
    // TODO: Implement configuration loading
    todo!("Implement configuration loading")
}
