from pydantic import BaseModel, Field
from fastapi import FastAPI
from transformers import pipeline
from typing import Dict, List, Optional, Union

import fire
import uvicorn
import yaml
import logging

logging.basicConfig(level=logging.INFO)
app = FastAPI()


class AppConfig(BaseModel):
    # "facebook/bart-large-mnli"
    model: Optional[str] = "cross-encoder/nli-distilroberta-base"
    host: str = "0.0.0.0"
    port: int = 8765


class Examples:
    candidate_labels = [
        "natural language processing", "computer vision"]
    texts = [
        "Sentence transformers is a library using transformers to embed text"]


def initialize_model(app, config: AppConfig):
    app.pipe = pipeline(model=config.model)


class ZeroShotClassificationRequest(BaseModel):
    text: str = Field(default=Examples.texts[0])
    candidate_labels: List[str] = Field(default=Examples.candidate_labels)


class ZeroShotListClassificationRequest(BaseModel):
    texts: List[str] = Field(default=Examples.texts)
    candidate_labels: List[str] = Field(default=Examples.candidate_labels)


class ZeroShotResponse(BaseModel):
    text: str
    label: str
    label_scores: Dict[str, float]

    @classmethod
    def from_hf_zsl_result(cls, result):
        text = result["sequence"]
        label_scores = dict(zip(result["labels"], result["scores"]))
        label = max(label_scores.keys(), key=label_scores.get)
        return ZeroShotResponse(text=text, label=label, label_scores=label_scores)


@app.post("/zero_shot_classify_single", response_model=ZeroShotResponse)
async def generate(zsl_request: ZeroShotClassificationRequest):
    result = app.pipe(
        zsl_request.text, candidate_labels=zsl_request.candidate_labels)
    return ZeroShotResponse.from_hf_zsl_result(result)


@app.post("/zero_shot_classify", response_model=List[ZeroShotResponse])
async def generate(zsl_request: ZeroShotListClassificationRequest):
    results = app.pipe(
        zsl_request.texts, candidate_labels=zsl_request.candidate_labels)
    return [ZeroShotResponse.from_hf_zsl_result(result) for result in results]


def main(config_path: str = None):
    if config_path is None:
        config = AppConfig()
    else:
        with open(config_path, "r") as f:
            config = AppConfig.parse_obj(yaml.safe_load(f))
    logging.info(f"Starting app with config: {config.model_dump()}")
    initialize_model(app, config)
    uvicorn.run(app, host=config.host, port=config.port)


if __name__ == "__main__":
    fire.Fire(main)
    # config = AppConfig()
    # setup_app(app, config)
    # candidate_labels = ["natural language processing", "computer vision"]
    # texts = ["Sentence transformers is a library using transformers to embed text"]
    # results = app.pipe(texts,
    #                    candidate_labels=candidate_labels)

    # print([ZeroShotResponse.from_hf_zsl_result(res) for res in results])
