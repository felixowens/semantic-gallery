use crate::core::ingest::process_image;
use crate::core::media::extract_media_details_from_path;
use crate::core::state::AppState;
use std::error::Error;
use std::path::PathBuf;

// TODO: support either a single image or a directory
pub async fn ingest(
    path: PathBuf,
    recursive: bool,
    app_state: &AppState,
) -> Result<(), Box<dyn Error>> {
    // Load the image and extract details
    let media_details = extract_media_details_from_path(&path)?;

    // Process the image to create embedding and save to the database
    process_image(media_details, &app_state).await?;

    Ok(())
}

pub async fn search(query: String, limit: usize, state: &AppState) -> Result<(), Box<dyn Error>> {
    // Generate text embedding for the query
    let text_embedding = state.embedder.encode_text(&query)?;
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
    .fetch_all(&state.db_pool)
    .await?;

    // Display the results
    if results.is_empty() {
        println!("No results found for query: \"{}\"", query);
    } else {
        println!("Search results for: \"{}\"", query);
        println!("{:-<50}", "");

        for (i, result) in results.iter().enumerate() {
            let similarity_percentage = result.similarity.map(|s| s * 100.0).unwrap_or(0.0);
            print!(
                "{}. {} (ID: {})\n   Path: file://{}\n   Similarity: {:.2}%\n",
                i + 1,
                result.filename,
                result.id,
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
    state: &AppState,
) -> Result<(), Box<dyn Error>> {
    todo!("Implement tag management")
}

pub async fn list_tags(state: &AppState) -> Result<(), Box<dyn Error>> {
    todo!("Implement tag listing")
}
