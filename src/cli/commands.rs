use crate::core::config::Config;
use crate::core::embedding::ClipEmbedder;
use crate::core::ingest::process_image;
use candle_core::Device;
use std::error::Error;
use std::path::Path;
use std::path::PathBuf;
use tracing::info;

// TODO: support either a single image or a directory
pub async fn ingest(path: PathBuf, recursive: bool, config: &Config) -> Result<(), Box<dyn Error>> {
    // Load the image
    let image = image::ImageReader::open(path)?.decode()?;

    // Process the image to create embedding and save to the database
    process_image(image, config).await?;

    Ok(())
}

pub async fn search(query: String, limit: usize, config: &Config) -> Result<(), Box<dyn Error>> {
    // Create a database connection pool
    let pool = crate::core::db::create_pool(config).await?;
    crate::core::db::check_connection(&pool).await?;

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

    // Generate text embedding for the query
    let text_embedding = clip_model.encode_text(&query)?;
    let embedding_vec = text_embedding.flatten_all()?.to_vec1::<f32>()?;

    // Search for similar images in the database using vector similarity
    let results = sqlx::query!(
        r#"
        SELECT m.id, m.filename, m.file_path, 
               1 - (e.embedding <=> $1::vector) as similarity
        FROM media m
        JOIN embeddings e ON m.id = e.media_id
        ORDER BY similarity DESC
        LIMIT $2
        "#,
        &embedding_vec as &[f32],
        limit as i64
    )
    .fetch_all(&pool)
    .await?;

    // Display the results
    if results.is_empty() {
        info!("No results found for query: \"{}\"", query);
    } else {
        info!("Search results for: \"{}\"", query);
        info!("{:-<50}", "");

        for (i, result) in results.iter().enumerate() {
            let similarity_percentage = result.similarity.map(|s| s * 100.0).unwrap_or(0.0);
            print!(
                "{}. {} (ID: {})\n   Path: \x1b]8;;{}\x1b\\{}\x1b]8;;\x1b\\\n   Similarity: {:.2}%\n",
                i + 1,
                result.filename,
                result.id,
                result.file_path,
                result.file_path,
                similarity_percentage
            );
        }
    }

    Ok(())
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
