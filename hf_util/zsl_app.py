from pydantic import BaseModel, Field
from fastapi import FastAPI
from transformers import pipeline
from typing import Dict, List, Optional, Union, Any
import fire
import uvicorn
import yaml
import logging
import scipy

logging.basicConfig(level=logging.INFO)
app = FastAPI()


class AppConfig(BaseModel):
    model: Optional[str] = "cross-encoder/nli-distilroberta-base"
    host: str = "0.0.0.0"
    port: int = 8765
    batch_size: int = 64
    gpu: bool = True


class Examples:
    candidate_labels = [
        "natural language processing", "computer vision"]
    texts = [
        "Sentence transformers is a library using transformers to embed text"]


def initialize_model(app, config: AppConfig):
    pipe = pipeline(
        model=config.model, device=0 if config.gpu else -1)
    app.state.zsl_classifier = ZeroShotClassifier(
        pipe=pipe, batch_size=config.batch_size)


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
    entropy: float

    @classmethod
    def from_hf_zsl_result(cls, result):
        text = result["sequence"]
        label_scores = dict(zip(result["labels"], result["scores"]))
        label = max(label_scores.keys(), key=label_scores.get)
        entropy = scipy.stats.entropy(list(label_scores.values()))
        return ZeroShotResponse(text=text, label=label, label_scores=label_scores, entropy=entropy)


class ZeroShotClassifier(BaseModel):
    pipe: Any
    batch_size: int = 32

    def predict(self, texts, candidate_labels):
        return [
            res
            for batch in self._batched(texts, self.batch_size)
            for res in self.pipe(batch, candidate_labels=candidate_labels)
        ]

    def predict_response(self, zsl_request: Union[ZeroShotListClassificationRequest, ZeroShotListClassificationRequest]) -> ZeroShotResponse:
        if isinstance(zsl_request, ZeroShotClassificationRequest):
            results = self.predict(
                [zsl_request.text], zsl_request.candidate_labels)
            return ZeroShotResponse.from_hf_zsl_result(results[0])
        else:
            results = self.predict(
                zsl_request.texts, zsl_request.candidate_labels)
            return [ZeroShotResponse.from_hf_zsl_result(result) for result in results]

    @classmethod
    def _batched(cls, items, batch_size):
        for i in range(0, len(items), batch_size):
            yield items[i:i + batch_size]

    class Config:
        arbitrary_types_allowed = True


@app.post("/zero_shot_classify_single", response_model=ZeroShotResponse)
async def generate_single(zsl_request: ZeroShotClassificationRequest):
    return app.state.zsl_classifier.predict_response(zsl_request)


@app.post("/zero_shot_classify", response_model=List[ZeroShotResponse])
async def generate(zsl_request: ZeroShotListClassificationRequest):
    return app.state.zsl_classifier.predict_response(zsl_request)


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
