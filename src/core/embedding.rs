use anyhow::{Error as E, Result as AnyhowResult};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::clip;
use image::DynamicImage;
use std::path::Path;
use tokenizers::Tokenizer;

pub struct ClipEmbedder {
    model: clip::ClipModel,
    tokenizer: Tokenizer,
    device: Device,
    config: clip::ClipConfig,
}

impl ClipEmbedder {
    pub fn new(model_path: &Path, tokenizer_path: &Path, device: Device) -> AnyhowResult<Self> {
        let config = clip::ClipConfig::vit_base_patch32();
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[model_path.to_path_buf()], DType::F32, &device)?
        };
        let model = clip::ClipModel::new(vb, &config)?;
        let tokenizer = Tokenizer::from_file(tokenizer_path).map_err(E::msg)?;

        Ok(Self {
            model,
            tokenizer,
            device,
            config,
        })
    }

    /// Load an image into a tensor.
    /// The image is resized to the model's image size and converted to RGB.
    fn load_image_tensor(&self, image: &DynamicImage) -> AnyhowResult<Tensor> {
        let (height, width) = (self.config.image_size, self.config.image_size);
        let img = image.resize_to_fill(
            width as u32,
            height as u32,
            image::imageops::FilterType::Triangle,
        );
        let img = img.to_rgb8();
        let img = img.into_raw();
        let img = Tensor::from_vec(img, (height, width, 3), &self.device)?
            .permute((2, 0, 1))?
            .to_dtype(DType::F32)?
            .affine(2. / 255., -1.)?;

        Ok(img)
    }

    fn tokenize_sequence(&self, sequence: &str) -> AnyhowResult<Tensor> {
        let encoding = self.tokenizer.encode(sequence, true).map_err(E::msg)?;
        let tokens = encoding.get_ids().to_vec();

        let input_ids = Tensor::new(tokens, &self.device)?.unsqueeze(0)?;
        Ok(input_ids)
    }

    pub fn encode_image(&self, image: &DynamicImage) -> AnyhowResult<Tensor> {
        let tensor = self.load_image_tensor(image)?;
        // Create a batch dimension
        let tensor = tensor.unsqueeze(0)?;
        let embedding = self.model.get_image_features(&tensor)?;
        let embedding_normalized = clip::div_l2_norm(&embedding)?;
        Ok(embedding_normalized)
    }

    pub fn encode_text(&self, text: &str) -> AnyhowResult<Tensor> {
        let input_ids = self.tokenize_sequence(text)?;
        let embedding = self.model.get_text_features(&input_ids)?;
        let embedding_normalized = clip::div_l2_norm(&embedding)?;
        Ok(embedding_normalized)
    }

    pub fn compute_similarity(&self, image: &DynamicImage, text: &str) -> AnyhowResult<f32> {
        let image_embedding = self.encode_image(image)?;
        let text_embedding = self.encode_text(text)?;

        cosine_similarity(&image_embedding, &text_embedding)
    }
}

/// Compute the cosine similarity between two tensors.
fn cosine_similarity(emb1: &Tensor, emb2: &Tensor) -> AnyhowResult<f32> {
    let sum = emb1
        .matmul(&emb2.transpose(0, 1)?)?
        .sum_all()?
        .to_scalar::<f32>()?;
    Ok(sum)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use candle_core::Tensor;

    #[test]
    fn test_cosine_similarity_identical_vectors() -> AnyhowResult<()> {
        // Two identical vectors should have similarity of 1.0
        let vec = Tensor::new(&[1.0f32; 512], &Device::Cpu)?.unsqueeze(0)?;
        let normalized_vec = clip::div_l2_norm(&vec)?;
        let result = cosine_similarity(&normalized_vec, &normalized_vec)?;
        assert_relative_eq!(result, 1.0, epsilon = 1e-5);
        Ok(())
    }

    #[test]
    fn test_cosine_similarity_orthogonal_vectors() -> AnyhowResult<()> {
        // Create truly orthogonal vectors instead of using a zero vector
        let mut vec1_data = vec![0.0f32; 512];
        let mut vec2_data = vec![0.0f32; 512];

        // Set first half of vec1 to 1.0
        for i in 0..256 {
            vec1_data[i] = 1.0;
        }

        // Set second half of vec2 to 1.0
        for i in 256..512 {
            vec2_data[i] = 1.0;
        }

        let vec1 = Tensor::new(&vec1_data[..], &Device::Cpu)?.unsqueeze(0)?;
        let vec2 = Tensor::new(&vec2_data[..], &Device::Cpu)?.unsqueeze(0)?;
        let normalized_vec1 = clip::div_l2_norm(&vec1)?;
        let normalized_vec2 = clip::div_l2_norm(&vec2)?;
        let result = cosine_similarity(&normalized_vec1, &normalized_vec2)?;
        assert_relative_eq!(result, 0.0, epsilon = 1e-5);
        Ok(())
    }

    #[test]
    fn test_cosine_similarity_opposite_vectors() -> AnyhowResult<()> {
        // Opposite vectors should have similarity of -1.0
        let vec1 = Tensor::new(&[1.0f32; 512], &Device::Cpu)?.unsqueeze(0)?;
        let vec2 = Tensor::new(&[-1.0f32; 512], &Device::Cpu)?.unsqueeze(0)?;
        let normalized_vec1 = clip::div_l2_norm(&vec1)?;
        let normalized_vec2 = clip::div_l2_norm(&vec2)?;
        let result = cosine_similarity(&normalized_vec1, &normalized_vec2)?;
        assert_relative_eq!(result, -1.0, epsilon = 1e-5);
        Ok(())
    }
}
