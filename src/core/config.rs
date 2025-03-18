use anyhow::Result;
use config::{Config as ConfigSource, File};
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

/// Load the configuration from the environment variables and the config file.
///
/// This function builds a configuration source by setting default values for
/// database connection parameters and adding a file source based on the
/// APP_CONFIG environment variable. It also incorporates environment variables
/// with a specified prefix. The resulting configuration is validated before
/// being returned.
pub fn load_config() -> Result<Config, Box<dyn Error>> {
    let config_builder = ConfigSource::builder()
        .set_default("database.host", "localhost")?
        .set_default("database.port", 5432)?
        .add_source(
            File::with_name(&std::env::var("APP_CONFIG").unwrap_or_else(|_| "config".to_string()))
                .required(false),
        )
        .add_source(config::Environment::with_prefix("APP").separator("__"))
        .build()?;

    let config: Config = config_builder.try_deserialize()?;

    validate_config(&config)?;

    Ok(config)
}

fn validate_config(config: &Config) -> Result<(), Box<dyn Error>> {
    if config.database.host.is_empty() {
        return Err("Database host cannot be empty".into());
    }

    if !config.embedding.model_path.exists() {
        return Err(format!(
            "Embedding model path does not exist: {}",
            config.embedding.model_path.display()
        )
        .into());
    }

    if config.embedding.dimension == 0 {
        return Err("Embedding dimension must be greater than 0".into());
    }

    if !config.storage.media_path.exists() {
        std::fs::create_dir_all(&config.storage.media_path)?;
    }

    Ok(())
}
