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

    fn load_image_tensor(&self, image: &DynamicImage) -> AnyhowResult<Tensor> {
        let (height, width) = (self.config.image_size, self.config.image_size);
        let img = image.resize_to_fill(
            width as u32,
            height as u32,
            image::imageops::FilterType::Triangle,
        );
        let img = img.to_rgb8();
        let img = img.into_raw();
        let img = Tensor::from_vec(img, (height, width, 3), &Device::Cpu)?
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

        self.cosine_similarity(&image_embedding, &text_embedding)
    }

    pub fn cosine_similarity(
        &self,
        image_embedding: &Tensor,
        text_embedding: &Tensor,
    ) -> AnyhowResult<f32> {
        let sum = image_embedding
            .matmul(&text_embedding.transpose(0, 1)?)?
            .sum_all()?
            .to_scalar::<f32>()?;
        Ok(sum)
    }
}
