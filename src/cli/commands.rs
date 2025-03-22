use crate::core::ingest::process_image;
use crate::core::media::extract_media_details_from_path;
use crate::core::state::AppState;
use indicatif;
use std::error::Error;
use std::path::PathBuf;

// TODO: support either a single image or a directory
pub async fn ingest(
    path: PathBuf,
    recursive: bool,
    app_state: &AppState,
    max_depth: Option<usize>,
) -> Result<(), Box<dyn Error>> {
    let max_depth = max_depth.unwrap_or(5); // Default max depth of 5

    // Collect all image files to process
    let files = collect_image_files(&path, recursive, max_depth)?;

    if files.is_empty() {
        println!("No image files found at path: {:?}", path);
        return Ok(());
    }

    // Prompt user with count
    println!("Found {} image files to process.", files.len());
    println!("Do you want to continue? [y/N]");

    let mut input = String::new();
    std::io::stdin().read_line(&mut input)?;

    if !input.trim().eq_ignore_ascii_case("y") {
        println!("Ingestion cancelled.");
        return Ok(());
    }

    // Process each file with a progress bar
    let progress_bar = indicatif::ProgressBar::new(files.len() as u64);
    progress_bar.set_style(
        indicatif::ProgressStyle::default_bar()
            .template(
                "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})",
            )
            .unwrap()
            .progress_chars("#>-"),
    );

    for file_path in files {
        // Extract details and process the image
        match extract_media_details_from_path(&file_path) {
            Ok(media_details) => {
                if let Err(e) = process_image(media_details, &app_state).await {
                    eprintln!("Error processing {:?}: {}", file_path, e);
                }
            }
            Err(e) => {
                eprintln!("Error extracting details from {:?}: {}", file_path, e);
            }
        }

        progress_bar.inc(1);
    }

    progress_bar.finish_with_message("Ingestion complete!");
    Ok(())
}

fn collect_image_files(
    path: &PathBuf,
    recursive: bool,
    max_depth: usize,
) -> Result<Vec<PathBuf>, Box<dyn Error>> {
    let mut files = Vec::new();

    if path.is_file() {
        // If path is a file, just return it if it's an image
        if is_image_file(path) {
            files.push(path.clone());
        }
    } else if path.is_dir() {
        // For directories, collect all image files
        collect_images_from_dir(path, &mut files, recursive, max_depth, 0)?;
    } else {
        return Err(format!("Path does not exist: {:?}", path).into());
    }

    Ok(files)
}

fn collect_images_from_dir(
    dir: &PathBuf,
    files: &mut Vec<PathBuf>,
    recursive: bool,
    max_depth: usize,
    current_depth: usize,
) -> Result<(), Box<dyn Error>> {
    if current_depth > max_depth {
        return Ok(());
    }

    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();

        if path.is_file() && is_image_file(&path) {
            files.push(path);
        } else if recursive && path.is_dir() {
            collect_images_from_dir(&path, files, recursive, max_depth, current_depth + 1)?;
        }
    }

    Ok(())
}

fn is_image_file(path: &PathBuf) -> bool {
    if let Some(ext) = path.extension() {
        let ext_str = ext.to_string_lossy().to_lowercase();
        matches!(
            ext_str.as_str(),
            "jpg" | "jpeg" | "png" | "gif" | "webp" | "tiff" | "bmp"
        )
    } else {
        false
    }
}

pub async fn search(query: String, limit: usize, state: &AppState) -> Result<(), Box<dyn Error>> {
    // Generate text embedding for the query
    let text_embedding = state.embedder.encode_text(&query)?;
    let embedding_vec = text_embedding.flatten_all()?.to_vec1::<f32>()?;

    // TODO: move to a core function and make implementation agnostic

    // TODO: add debug/trace logging for time taken to search
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
