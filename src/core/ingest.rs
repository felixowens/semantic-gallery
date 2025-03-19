use std::path::Path;

use crate::core::config::Config;
// use crate::core::db::{check_connection, create_pool}; // Assuming you have a function to create a DB pool
use crate::core::embedding::ClipEmbedder; // Import your ClipModel
use anyhow::Result;
use candle_core::Device;
use image::DynamicImage;

pub async fn process_image(image: DynamicImage, config: &Config) -> Result<()> {
    // Create a database connection pool
    // let pool = create_pool(config).await?;
    // check_connection(&pool).await?;
    // Load the CLIP model

    // TODO: load models from huggingface ids instead? or make paths optional?
    let clip_model = ClipEmbedder::new(
        config.embedding.model_path.as_ref().map(Path::new).unwrap(),
        config
            .embedding
            .tokenizer_path
            .as_ref()
            .map(Path::new)
            .unwrap(),
        Device::Cpu,
    )?;

    // Generate the embedding
    let embedding = clip_model.encode_image(&image)?;

    println!("Embedding: {:?}", embedding);

    // Save the embedding to the database
    todo!("Save the embedding to the database");
    // let client = pool.get().await?;
    // client
    //     .execute("INSERT INTO embeddings (data) VALUES ($1)", &[&embedding])
    //     .await?;

    // Ok(())
}
