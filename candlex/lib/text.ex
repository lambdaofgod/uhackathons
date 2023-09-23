defmodule Candlex.TextGenerationModel do
  use Rustler, otp_app: :candlex, crate: "candlex", features: [:cuda]

  def initialize_model(model_name, model_path, cpu, temperature, top_p, seed), do: :erlang.nif_error(:nif_not_loaded)
  # When your NIF is loaded, it will override this function.
  def generate_text(model_name, prompt, sample_len), do: :erlang.nif_error(:nif_not_loaded)
end
