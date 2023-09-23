mod text_generation;
use std::io::Error;
use text_generation::*;

fn main() -> Result<(), Error> {
    let model_path: Option<String> = None;
    let cpu: bool = false;
    let temperature: Option<f64> = None;
    let top_p: f64 = 1.0;
    let seed: u64 = 0;

    let model_wrapper_result = make_wrapper(model_path, cpu, temperature, top_p, seed);

    match model_wrapper_result {
        Ok(model_wrapper) => {
            let results = generate_text(&model_wrapper, "a cat walks into a bar", 20);
            println!("{:?}", results);
            Ok(())
        }
        Err(e) => Err(Error::new(std::io::ErrorKind::Other, e.to_string())),
    }
}
