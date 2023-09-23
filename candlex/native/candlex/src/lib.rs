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

type ModelPointer = TextGeneratorWrapper;
lazy_static! {
    static ref model_registry: Mutex<HashMap<NIF_TERM, ModelPointer>> = Mutex::new(HashMap::new());
}

fn set_model_registry_value(key: NIF_TERM, value: ModelPointer) {
    let _ = model_registry
        .lock()
        .map(|mut x| x.insert(key, value))
        .unwrap();
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
    let can_create_model = model_registry
        .lock()
        .map(|x| x.get(&model_name.as_c_arg()).is_none())
        .unwrap_or(false);
    match can_create_model {
        false => Err("Model already initialized".to_string()),
        true => {
            let text_generator_result = make_wrapper(model_path, cpu, temperature, top_p, seed);
            match text_generator_result {
                Ok(text_generator) => {
                    set_model_registry_value(model_name.as_c_arg(), text_generator);
                    Ok(model_name)
                }
                Err(e) => Err(e.to_string()),
            }
        }
    }
}

#[rustler::nif]
fn generate_text(model_name: Atom, prompt: &str, sample_len: usize) -> Result<Vec<String>, String> {
    anyhow_to_std_result(generate_text_impl(model_name, prompt, sample_len))
}

fn generate_text_impl(
    model_name: Atom,
    prompt: &str,
    sample_len: usize,
) -> anyhow::Result<Vec<String>> {
    match model_registry.lock() {
        Ok(model_registry_) => {
            let text_generator = model_registry_.get(&model_name.as_c_arg()).unwrap();
            text_generation::generate_text(&text_generator, prompt, sample_len)
        }
        Err(e) => Err(anyhow::anyhow!(e.to_string())),
    }
}

fn on_load(env: Env, _info: Term) -> bool {
    true
}

rustler::init!(
    "Elixir.Candlex.TextGenerationModel",
    [initialize_model, generate_text]
);
