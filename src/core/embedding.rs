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

    fn tokenize_sequences(&self, sequences: &[String]) -> AnyhowResult<(Tensor, Vec<String>)> {
        let pad_id = *self
            .tokenizer
            .get_vocab(true)
            .get("<|endoftext|>")
            .ok_or(E::msg("No pad token"))?;

        let mut tokens = vec![];
        for seq in sequences {
            let encoding = self.tokenizer.encode(seq.clone(), true).map_err(E::msg)?;
            tokens.push(encoding.get_ids().to_vec());
        }

        let max_len = tokens.iter().map(|v| v.len()).max().unwrap_or(0);
        // Pad the sequences to have the same length
        for token_vec in tokens.iter_mut() {
            let len_diff = max_len - token_vec.len();
            if len_diff > 0 {
                token_vec.extend(vec![pad_id; len_diff]);
            }
        }

        let input_ids = Tensor::new(tokens, &self.device)?;
        Ok((input_ids, sequences.to_vec()))
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

    fn load_image_tensors(&self, images: &[DynamicImage]) -> AnyhowResult<Tensor> {
        let mut tensors = vec![];
        for image in images.iter() {
            let tensor = self.load_image_tensor(image)?;
            tensors.push(tensor);
        }
        let images_tensor = Tensor::stack(&tensors, 0)?;
        Ok(images_tensor)
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
        Ok(embedding)
    }

    pub fn encode_images(&self, images: &[DynamicImage]) -> AnyhowResult<Tensor> {
        let tensors = self.load_image_tensors(images)?;
        let embedding = self.model.get_image_features(&tensors)?;
        Ok(embedding)
    }

    pub fn encode_text(&self, text: &str) -> AnyhowResult<Tensor> {
        let input_ids = self.tokenize_sequence(text)?;
        let embedding = self.model.get_text_features(&input_ids)?;
        Ok(embedding)
    }

    pub fn compute_similarity(&self, image: &DynamicImage, text: &str) -> AnyhowResult<f32> {
        let image_embedding = self.encode_image(image)?;
        println!(
            "Image embedding - compute_similarity: {:?}",
            image_embedding
        );
        let text_embedding = self.encode_text(text)?;
        println!("Text embedding - compute_similarity: {:?}", text_embedding);

        let similarity = image_embedding.matmul(&text_embedding.t()?)?;
        println!("Similarity - compute_similarity: {:?}", similarity);

        // Extract the scalar value correctly from the 1x1 tensor
        let similarity_tensor = similarity.get(0)?;
        println!("similarity_tensor - compute_similarity: {similarity_tensor}");

        let similarity_score = similarity_tensor.get(0)?.to_scalar::<f32>()?;
        println!("similarity_score - compute_similarity: {similarity_score}");

        Ok(similarity_score)
    }
}

pub fn download_clip_model() -> AnyhowResult<(std::path::PathBuf, std::path::PathBuf)> {
    let api = hf_hub::api::sync::Api::new()?;
    let api = api.repo(hf_hub::Repo::with_revision(
        "openai/clip-vit-base-patch32".to_string(),
        hf_hub::RepoType::Model,
        "refs/pr/15".to_string(),
    ));

    let model_path = api.get("model.safetensors")?;
    let tokenizer_path = api.get("tokenizer.json")?;

    Ok((model_path, tokenizer_path))
}
