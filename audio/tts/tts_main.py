from transformers import pipeline
from datasets import load_dataset
import soundfile as sf
import torch
from transformers import BarkModel, BarkProcessor
from time import time
import tqdm
from pydantic import BaseModel
from typing import List
import os
import yaml
import fire


class TTSGenerationConfig(BaseModel):
    model_name: str = "suno/bark"
    voice_presets: List[str]
    texts: List[str]
    out_file_template: str
    device: str = "cuda"


def time_func(func):
    # This function shows the execution time of
    # the function object passed
    def wrap_func(*args, **kwargs):
        t1 = time()
        result = func(*args, **kwargs)
        t2 = time()
        elapsed = t2 - t1
        print(f"Function {func.__name__!r} executed in {(elapsed):.4f}s")
        return result

    return wrap_func


@time_func
def setup_model(model_name="suno/bark", device="cuda"):
    model = BarkModel.from_pretrained(model_name)
    processor = BarkProcessor.from_pretrained(model_name)
    if device == "cuda":
        model = model.cuda()
    return model.cuda(), processor


def generate_speech(model, processor, text, voice_preset="v2/pl_speaker_3"):
    inputs = processor(text, voice_preset=voice_preset).to(model.device)
    import ipdb
    ipdb.set_trace()
    speech = model.generate(**inputs)
    return speech.cpu().numpy()


def format_file_template(template, text_idx, voice_preset_idx):
    return template.format(text_idx, voice_preset_idx)


def main(config_path: str):
    assert os.path.exists(
        config_path), f"Config file {config_path} does not exist"
    with open(config_path, "r") as f:
        dict_config = yaml.safe_load(f)
        config = TTSGenerationConfig(**dict_config)

    model, processor = setup_model(config.model_name, config.device)
    sampling_rate = model.generation_config.sample_rate

    for text_idx, text in enumerate(config.texts):
        for voice_preset_idx, voice_preset in enumerate(tqdm.tqdm(config.voice_presets)):
            speech = generate_speech(model, processor, text, voice_preset)
            sf.write(format_file_template(config.out_file_template, text_idx, voice_preset_idx),
                     speech[0], samplerate=sampling_rate)


if __name__ == "__main__":
    fire.Fire(main)
