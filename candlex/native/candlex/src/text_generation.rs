// #[cfg(feature = "mkl")]
// extern crate intel_mkl_src;

// #[cfg(feature = "accelerate")]
// extern crate accelerate_src;

use anyhow::{Error as E, Result};
use candle_transformers::models::bigcode::{Config, GPTBigCode};
use clap::Parser;
use rustler::resource::ResourceTypeProvider;
use rustler::{Atom, NifStruct, NifTaggedEnum, ResourceArc};
use std::cell::RefCell;

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::generation::LogitsProcessor;
use hf_hub::{api::sync::Api, Repo, RepoType};
use tokenizers::Tokenizer;

pub fn get_device(cpu: bool) -> Result<Device> {
    if cpu {
        Ok(Device::Cpu)
    } else {
        let device = Device::cuda_if_available(0)?;
        if !device.is_cuda() {
            println!("Running on CPU, to run on GPU, build this example with `--features cuda`");
        }
        Ok(device)
    }
}

pub type TextGeneratorWrapper = RefCell<TextGeneration>;

pub fn generate_text<'a>(
    text_generator: &'a TextGeneratorWrapper,
    prompt: &str,
    sample_len: usize,
) -> Result<Vec<String>> {
    text_generator.borrow_mut().run(prompt, sample_len, None)
}

pub fn make_wrapper(
    model_path: Option<String>,
    cpu: bool,
    temperature: Option<f64>,
    top_p: f64,
    seed: u64,
) -> Result<TextGeneratorWrapper> {
    TextGeneration::create(model_path, cpu, temperature, top_p, seed).map(RefCell::new)
}

pub struct TextGeneration {
    model: GPTBigCode,
    device: Device,
    tokenizer: Tokenizer,
    logits_processor: LogitsProcessor,
}

impl TextGeneration {
    pub fn new(
        model: GPTBigCode,
        tokenizer: Tokenizer,
        seed: u64,
        temp: Option<f64>,
        top_p: Option<f64>,
        device: &Device,
    ) -> Self {
        let logits_processor = LogitsProcessor::new(seed, temp, top_p);
        Self {
            model,
            tokenizer,
            logits_processor,
            device: device.clone(),
        }
    }

    pub fn create(
        model_path: Option<String>,
        cpu: bool,
        temperature: Option<f64>,
        top_p: f64,
        seed: u64,
    ) -> Result<Self> {
        let api = Api::new()?;
        let repo = api.repo(Repo::with_revision(
            "bigcode/starcoderbase-1b".to_string(),
            RepoType::Model,
            "main".to_string(),
        ));
        let tokenizer_filename = repo.get("tokenizer.json")?;
        let filenames = match model_path {
            Some(weight_file) => vec![std::path::PathBuf::from(weight_file)],
            None => ["model.safetensors"]
                .iter()
                .map(|f| repo.get(f))
                .collect::<std::result::Result<Vec<_>, _>>()?,
        };
        let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;

        let weights = filenames
            .iter()
            .map(|f| Ok(unsafe { candle_core::safetensors::MmapedFile::new(f)? }))
            .collect::<Result<Vec<_>>>()?;
        let weights = weights
            .iter()
            .map(|f| Ok(f.deserialize()?))
            .collect::<Result<Vec<_>>>()?;

        let device = get_device(cpu)?;
        let vb = VarBuilder::from_safetensors(weights, DType::F32, &device);
        let config = Config::starcoder_1b();
        let model = GPTBigCode::load(vb, config)?;

        Ok(TextGeneration::new(
            model,
            tokenizer,
            seed,
            temperature,
            Some(top_p),
            &device,
        ))
    }

    pub fn run(
        &mut self,
        prompt: &str,
        sample_len: usize,
        use_cache: Option<bool>,
    ) -> Result<Vec<String>> {
        println!("starting the inference loop");
        print!("{prompt}");
        let mut tokens = self
            .tokenizer
            .encode(prompt, true)
            .map_err(E::msg)?
            .get_ids()
            .to_vec();
        let do_use_cache = use_cache.unwrap_or(self.model.config().use_cache);

        let mut new_tokens = vec![];
        let mut out_tokens = vec![];
        let start_gen = std::time::Instant::now();
        for index in 0..sample_len {
            let (context_size, past_len) = if do_use_cache && index > 0 {
                (1, tokens.len().saturating_sub(1))
            } else {
                (tokens.len(), 0)
            };
            let ctxt = &tokens[tokens.len().saturating_sub(context_size)..];
            let input = Tensor::new(ctxt, &self.device)?.unsqueeze(0)?;
            let logits = self.model.forward(&input, past_len)?;
            let logits = logits.squeeze(0)?.to_dtype(DType::F32)?;

            let next_token = self.logits_processor.sample(&logits)?;
            tokens.push(next_token);
            new_tokens.push(next_token);
            let token = self.tokenizer.decode(&[next_token], true).map_err(E::msg)?;
            println!("{}", token);
            out_tokens.push(token);
        }
        Ok(out_tokens)
    }
}

// fn main() -> Result<()> {
//     let args = Args::parse();

//     let start = std::time::Instant::now();
//     let api = Api::new()?;
//     let repo = api.repo(Repo::with_revision(
//         args.model_id,
//         RepoType::Model,
//         args.revision,
//     ));
//     let tokenizer_filename = repo.get("tokenizer.json")?;
//     let filenames = match args.weight_file {
//         Some(weight_file) => vec![std::path::PathBuf::from(weight_file)],
//         None => ["model.safetensors"]
//             .iter()
//             .map(|f| repo.get(f))
//             .collect::<std::result::Result<Vec<_>, _>>()?,
//     };
//     println!("retrieved the files in {:?}", start.elapsed());
//     let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;

//     let weights = filenames
//         .iter()
//         .map(|f| Ok(unsafe { candle::safetensors::MmapedFile::new(f)? }))
//         .collect::<Result<Vec<_>>>()?;
//     let weights = weights
//         .iter()
//         .map(|f| Ok(f.deserialize()?))
//         .collect::<Result<Vec<_>>>()?;

//     let start = std::time::Instant::now();
//     let device = candle_examples::device(args.cpu)?;
//     let vb = VarBuilder::from_safetensors(weights, DType::F32, &device);
//     let config = Config::starcoder_1b();
//     let model = GPTBigCode::load(vb, config)?;
//     println!("loaded the model in {:?}", start.elapsed());

//     let mut pipeline = TextGeneration::new(
//         model,
//         tokenizer,
//         args.seed,
//         args.temperature,
//         args.top_p,
//         &device,
//     );
//     pipeline.run(&args.prompt, args.sample_len)?;
//     Ok(())
// }
