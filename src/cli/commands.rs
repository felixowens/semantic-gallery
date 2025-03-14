use crate::core::config::Config;
use std::error::Error;
use std::path::PathBuf;

pub async fn ingest(path: PathBuf, recursive: bool, config: &Config) -> Result<(), Box<dyn Error>> {
    // TODO: Implement media ingestion
    todo!("Implement media ingestion")
}

pub async fn search(query: String, limit: usize, config: &Config) -> Result<(), Box<dyn Error>> {
    // TODO: Implement media search
    todo!("Implement media search")
}

pub async fn tag(
    media_id: String,
    add: Vec<String>,
    remove: Vec<String>,
    config: &Config,
) -> Result<(), Box<dyn Error>> {
    // TODO: Implement tag management
    todo!("Implement tag management")
}

pub async fn list_tags(config: &Config) -> Result<(), Box<dyn Error>> {
    // TODO: Implement tag listing
    todo!("Implement tag listing")
}
