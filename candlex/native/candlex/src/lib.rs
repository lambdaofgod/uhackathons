pub mod text_generation;
mod utils;
use lazy_static::lazy_static;
use rustler::resource::ResourceTypeProvider;
use rustler::wrapper::NIF_TERM;
use rustler::{nif, Atom, NifStruct, NifTaggedEnum, ResourceArc};
use rustler::{Env, Term};
use std::cell::RefCell;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::Mutex;
use text_generation::TextGeneratorWrapper;
use text_generation::*;
use utils::anyhow_to_std_result;
mod model_registry;
use model_registry::*;

type ModelPointer = TextGeneratorWrapper;
lazy_static! {
    static ref MODEL_REGISTRY: ModelRegistry = ModelRegistry::new();
}

#[rustler::nif]
pub fn initialize_model(
    model_name: Atom,
    model_path: Option<String>,
    cpu: bool,
    temperature: Option<f64>,
    top_p: f64,
    seed: u64,
) -> Result<Atom, String> {
    MODEL_REGISTRY.try_initialize_model(model_name, model_path, cpu, temperature, top_p, seed)
}

#[rustler::nif]
fn generate_text(model_name: Atom, prompt: &str, sample_len: usize) -> Result<Vec<String>, String> {
    anyhow_to_std_result(MODEL_REGISTRY.try_generate_text(model_name, prompt, sample_len))
}

fn on_load(env: Env, _info: Term) -> bool {
    true
}

rustler::init!(
    "Elixir.Candlex.TextGenerationModel.Native",
    [initialize_model, generate_text]
);
