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
        Device::cuda_if_available(0)?,
    )?;

    // Generate the embedding
    let embedding = clip_model.encode_image(&image)?;

    println!("Embedding: {:?}", embedding);
    // TODO: testing, will remove
    let text_embedding_1 = clip_model.encode_text("man")?;
    println!("Text embedding - man: {:?}", text_embedding_1);
    let similarity_1 = clip_model.compute_similarity(&image, "man")?;
    println!("Similarity - man: {:?}", similarity_1);

    let text_embedding_2 = clip_model.encode_text("woman")?;
    println!("Text embedding - woman: {:?}", text_embedding_2);
    let similarity_2 = clip_model.compute_similarity(&image, "woman")?;
    println!("Similarity - woman: {:?}", similarity_2);

    let text_embedding_3 = clip_model.encode_text("dog food")?;
    println!("Text embedding: {:?}", text_embedding_3);
    let similarity_3 = clip_model.compute_similarity(&image, "dog food")?;
    println!("Similarity - dog food: {:?}", similarity_3);

    // Save the embedding to the database
    todo!("Save the embedding to the database");
    // let client = pool.get().await?;
    // client
    //     .execute("INSERT INTO embeddings (data) VALUES ($1)", &[&embedding])
    //     .await?;

    // Ok(())
}
