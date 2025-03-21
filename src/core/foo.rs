use anyhow::Error as E;
use clap::Parser;

use candle_core::{DType, Device, Tensor};
use candle_nn::{ops::softmax, VarBuilder};
use candle_transformers::models::clip::{self, div_l2_norm};

use tokenizers::Tokenizer;

#[derive(Parser)]
struct Args {
    #[arg(long)]
    model: Option<String>,

    #[arg(long)]
    tokenizer: Option<String>,

    #[arg(long, use_value_delimiter = true)]
    images: Option<Vec<String>>,

    #[arg(long)]
    cpu: bool,

    #[arg(long, use_value_delimiter = true)]
    sequences: Option<Vec<String>>,
}

/// CLIP model wrapper for embedding extraction and similarity computation
struct ClipEmbedder {
    model: clip::ClipModel,
    tokenizer: Tokenizer,
    device: Device,
    config: clip::ClipConfig,
}

impl ClipEmbedder {
    /// Create a new ClipEmbedder with the specified model and tokenizer
    pub fn new(
        model_path: Option<String>,
        tokenizer_path: Option<String>,
        device: Device,
    ) -> anyhow::Result<Self> {
        let model_file = match model_path {
            None => {
                let api = hf_hub::api::sync::Api::new()?;
                let api = api.repo(hf_hub::Repo::with_revision(
                    "openai/clip-vit-base-patch32".to_string(),
                    hf_hub::RepoType::Model,
                    "refs/pr/15".to_string(),
                ));
                api.get("model.safetensors")?
            }
            Some(model) => model.into(),
        };

        let tokenizer = Self::get_tokenizer(tokenizer_path)?;
        let config = clip::ClipConfig::vit_base_patch32();

        let vb =
            unsafe { VarBuilder::from_mmaped_safetensors(&[model_file], DType::F32, &device)? };

        let model = clip::ClipModel::new(vb, &config)?;

        Ok(Self {
            model,
            tokenizer,
            device,
            config,
        })
    }

    /// Get tokenizer from file or download from HuggingFace
    fn get_tokenizer(tokenizer_path: Option<String>) -> anyhow::Result<Tokenizer> {
        let tokenizer = match tokenizer_path {
            None => {
                let api = hf_hub::api::sync::Api::new()?;
                let api = api.repo(hf_hub::Repo::with_revision(
                    "openai/clip-vit-base-patch32".to_string(),
                    hf_hub::RepoType::Model,
                    "refs/pr/15".to_string(),
                ));
                api.get("tokenizer.json")?
            }
            Some(file) => file.into(),
        };
        Tokenizer::from_file(tokenizer).map_err(E::msg)
    }

    /// Extract embeddings from images
    pub fn get_image_embeddings(&self, image_paths: &[String]) -> anyhow::Result<Tensor> {
        let images = load_images(image_paths, self.config.image_size)?.to_device(&self.device)?;

        // Extract just the image embeddings
        self.model.get_image_features(pixel_values).forward(&images)
    }

    /// Extract embeddings from text sequences
    pub fn get_text_embeddings(&self, sequences: &[String]) -> anyhow::Result<Tensor> {
        let (input_ids, _) = self.tokenize_sequences(sequences)?;

        // Extract just the text embeddings
        self.model.text_model().forward(&input_ids)

        div_l2_norm(&text_features)
    }

    /// Compute similarity between image and text embeddings
    pub fn compute_similarity(
        &self,
        image_embeddings: &Tensor,
        text_embeddings: &Tensor,
    ) -> anyhow::Result<Tensor> {
        // Scale by the temperature factor
        let logit_scale = self.model.logit_scale;
        let scale = logit_scale.exp()?;

        // Normalize embeddings
        let image_embeddings =
            image_embeddings.broadcast_div(&image_embeddings.sqr()?.sum_keepdim(1)?.sqrt()?)?;
        let text_embeddings =
            text_embeddings.broadcast_div(&text_embeddings.sqr()?.sum_keepdim(1)?.sqrt()?)?;

        // Compute similarity matrix
        let similarity = image_embeddings.matmul(&text_embeddings.transpose(0, 1)?)?;
        let similarity = similarity.mul(&scale)?;

        // Apply softmax to get probabilities
        let probs = softmax(&similarity, 1)?;

        Ok(probs)
    }

    /// Tokenize text sequences for processing
    pub fn tokenize_sequences(
        &self,
        sequences: &[String],
    ) -> anyhow::Result<(Tensor, Vec<String>)> {
        let pad_id = *self
            .tokenizer
            .get_vocab(true)
            .get("<|endoftext|>")
            .ok_or(E::msg("No pad token"))?;

        let mut tokens = vec![];
        for seq in sequences.iter() {
            let encoding = self.tokenizer.encode(seq, true).map_err(E::msg)?;
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

    /// Format and print similarity results
    pub fn print_similarity_results(
        &self,
        probs: &Tensor,
        image_paths: &[String],
        sequences: &[String],
    ) -> anyhow::Result<()> {
        let probs_vec = probs.flatten_all()?.to_vec1::<f32>()?;
        let probability_vec = probs_vec.iter().map(|v| v * 100.0).collect::<Vec<f32>>();

        let probability_per_image = probability_vec.len() / image_paths.len();

        for (i, img) in image_paths.iter().enumerate() {
            let start = i * probability_per_image;
            let end = start + probability_per_image;
            let prob = &probability_vec[start..end];

            println!("\n\nResults for image: {}\n", img);
            for (j, p) in prob.iter().enumerate() {
                println!("Probability: {:.4}% Text: {} ", p, sequences[j]);
            }
        }

        Ok(())
    }
}

fn load_image<T: AsRef<std::path::Path>>(path: T, image_size: usize) -> anyhow::Result<Tensor> {
    let img = image::ImageReader::open(path)?.decode()?;
    let (height, width) = (image_size, image_size);
    let img = img.resize_to_fill(
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

fn load_images<T: AsRef<std::path::Path>>(
    paths: &[T],
    image_size: usize,
) -> anyhow::Result<Tensor> {
    let mut images = vec![];
    for path in paths {
        let tensor = load_image(path, image_size)?;
        images.push(tensor);
    }
    let images = Tensor::stack(&images, 0)?;
    Ok(images)
}

pub fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let device = if args.cpu {
        Device::Cpu
    } else {
        Device::cuda_if_available(0)?
    };

    // Initialize the CLIP embedder
    let clip_embedder = ClipEmbedder::new(args.model, args.tokenizer, device)?;

    // Get image paths
    let image_paths = match args.images {
        Some(imgs) => imgs,
        None => vec![
            "candle-examples/examples/stable-diffusion/assets/stable-diffusion-xl.jpg".to_string(),
            "candle-examples/examples/yolo-v8/assets/bike.jpg".to_string(),
        ],
    };

    // Get text sequences
    let sequences = match args.sequences {
        Some(seq) => seq,
        None => vec![
            "a cycling race".to_string(),
            "a photo of two cats".to_string(),
            "a robot holding a candle".to_string(),
        ],
    };

    // Extract embeddings (ready for database storage)
    let image_embeddings = clip_embedder.get_image_embeddings(&image_paths)?;
    let text_embeddings = clip_embedder.get_text_embeddings(&sequences)?;

    // Compute similarity between embeddings
    let similarity_probs = clip_embedder.compute_similarity(&image_embeddings, &text_embeddings)?;

    // Print results
    clip_embedder.print_similarity_results(&similarity_probs, &image_paths, &sequences)?;

    // Here you could store the embeddings in a database
    // Example (pseudocode):
    // for (i, path) in image_paths.iter().enumerate() {
    //     let embedding = image_embeddings.get(i)?;
    //     database.store_image_embedding(path, embedding);
    // }

    Ok(())
}
