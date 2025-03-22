use std::{error::Error, path::PathBuf};

use image::DynamicImage;

/// Media details extracted from a file path
pub struct MediaDetails {
    pub image: DynamicImage,
    pub filename: String,
    pub file_path: String,
    pub file_size: u64,
}

/// Extract the file details from the path
/// Including the image, filename, file path, and file size
pub fn extract_media_details_from_path(path: &PathBuf) -> Result<MediaDetails, Box<dyn Error>> {
    let image = image::ImageReader::open(path)?.decode()?;
    let filename = path
        .file_name()
        .ok_or("Image path not found")?
        .to_string_lossy()
        .to_string();
    let file_path = path.to_string_lossy().to_string();
    let file_size = path.metadata()?.len();

    Ok(MediaDetails {
        image,
        filename,
        file_path,
        file_size,
    })
}
