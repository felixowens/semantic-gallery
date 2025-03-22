use std::path::Path;

use crate::core::config::Config;
use crate::core::db::{check_connection, create_pool};
use crate::core::embedding::ClipEmbedder;
use anyhow::Result;
use candle_core::Device;
use sqlx;
use tracing::info;
use uuid::Uuid;

use super::media::MediaDetails;

pub async fn process_image(media_details: MediaDetails, config: &Config) -> Result<()> {
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
    let embedding = clip_model.encode_image(&media_details.image)?;
    info!("Generated embedding with shape: {:?}", embedding.shape());

    // Convert embedding to a format suitable for database storage
    let embedding_vec = embedding.flatten_all()?.to_vec1::<f32>()?;

    // Generate UUIDs for the database records
    let media_id = Uuid::new_v4();
    let embedding_id = Uuid::new_v4();

    // Save the media information to the database
    let mut tx = pool.begin().await?;

    sqlx::query!(
        r#"
        INSERT INTO media (id, filename, content_type, file_path, file_size, width, height, metadata)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
        "#,
        media_id,
        media_details.filename,
        "image/jpeg",
        media_details.file_path,
        media_details.file_size as i64,
        media_details.image.width() as i32,
        media_details.image.height() as i32,
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

    tx.commit().await?;

    info!(
        "Saved media with ID: {} and embedding with ID: {}",
        media_id, embedding_id
    );

    Ok(())
}
