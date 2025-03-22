use std::path::Path;

use crate::core::config::Config;
use crate::core::db::{check_connection, create_pool};
use crate::core::embedding::ClipEmbedder;
use anyhow::Result;
use candle_core::Device;
use image::DynamicImage;
use sqlx;
use tracing::info;
use uuid::Uuid;

pub async fn process_image(image: DynamicImage, config: &Config) -> Result<()> {
    // TODO: create pool in main.rs?
    // Create a database connection pool
    let pool = create_pool(config).await?;
    check_connection(&pool).await?;

    // TODO: load models from huggingface ids instead? or make paths optional?
    // Load the CLIP model and tokenizer
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
    info!("Generated embedding with shape: {:?}", embedding.shape());

    // Convert embedding to a format suitable for database storage
    let embedding_vec = embedding.flatten_all()?.to_vec1::<f32>()?;

    // Generate UUIDs for the database records
    let media_id = Uuid::new_v4();
    let embedding_id = Uuid::new_v4();

    // Save the media information to the database
    let mut tx = pool.begin().await?;

    // TODO: get these from the image metadata or parameters
    sqlx::query!(
        r#"
        INSERT INTO media (id, filename, content_type, file_path, file_size, width, height, metadata)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
        "#,
        media_id,
        "placeholder.jpg",
        "image/jpeg",
        "/path/to/image",
        0,
        image.width() as i32,
        image.height() as i32,
        serde_json::Value::Null
    )
    .execute(&mut *tx)
    .await?;

    // Insert the embedding into the database
    sqlx::query!(
        r#"
        INSERT INTO embeddings (id, media_id, model_name, model_version, embedding)
        VALUES ($1, $2, $3, $4, $5::vector)
        "#,
        embedding_id,
        media_id,
        "clip-vit-base-patch32", // TODO: get from config
        "v1",                    // TODO: get from config
        &embedding_vec as &[f32]
    )
    .execute(&mut *tx)
    .await?;

    // Commit the transaction
    tx.commit().await?;

    info!(
        "Saved media with ID: {} and embedding with ID: {}",
        media_id, embedding_id
    );

    // The testing code can remain for now
    println!("Embedding: {:?}", embedding);
    // TODO: testing, will remove
    let text_embedding_1 = clip_model.encode_text("man")?;
    println!("Text embedding - man: {:?}", text_embedding_1);
    let similarity_1 = clip_model.compute_similarity(&image, "A photo of a man")?;
    println!("Similarity - man: {:?}", similarity_1);

    let text_embedding_2 = clip_model.encode_text("woman")?;
    println!("Text embedding - woman: {:?}", text_embedding_2);
    let similarity_2 = clip_model.compute_similarity(&image, "A photo of a woman")?;
    println!("Similarity - woman: {:?}", similarity_2);

    let text_embedding_3 = clip_model.encode_text("dog food")?;
    println!("Text embedding: {:?}", text_embedding_3);
    let similarity_3 = clip_model.compute_similarity(&image, "A photo of a dog food")?;
    println!("Similarity - dog food: {:?}", similarity_3);

    Ok(())
}
