defmodule Candlex.TextGenerationModel do
  @enforce_keys [:model_name, :cpu, :top_p, :seed]
  defstruct [:model_name, :model_path, :cpu, :temperature, :top_p, :seed]

  def new(model_name, opts) do
    model_path = opts[:model_path]
    cpu = opts[:cpu] || false
    top_p = opts[:top_p] || 1.0
    seed = opts[:seed] || 0
    temperature = opts[:temperature]
    model_name |> Candlex.TextGenerationModel.Native.initialize_model(
      model_path,
      cpu,
      temperature,
      top_p,
      seed
    )
    %__MODULE__{
      model_name: model_name,
      model_path: model_path,
      cpu: cpu,
      temperature: temperature,
      top_p: top_p,
      seed: seed
    }
  end

  def generate(text_generation_model, prompt, sample_len) do
    text_generation_model.model_name |> Candlex.TextGenerationModel.Native.generate_text(prompt, sample_len)
  end
end

defmodule Candlex.TextGenerationModel.Native do
  use Rustler, otp_app: :candlex, features: ["default"]

  def initialize_model(model_name, model_path, cpu, temperature, top_p, seed), do: :erlang.nif_error(:nif_not_loaded)
  # When your NIF is loaded, it will override this function.
  def generate_text(model_name, prompt, sample_len), do: :erlang.nif_error(:nif_not_loaded)
end
