from pathlib import Path
from typing import Literal, Optional

import dspy
import dspy.retrieve
from qdrant_client import QdrantClient

from llms_dspy.vllm_client import OpenAIVLLMClient, VLLMClient
# we use a fixed QdrantRM
# original one has a bug in the forward method (returns strings instead of dotdict)
from llms_dspy.qdrant_rm import QdrantRM


def load_api_key(key_path):
    with open(Path(key_path).expanduser()) as f:
        key = f.read().strip()
    return key


def get_qdrant_retriever(collection_name):
    qdrant_path = Path(f"~/.cache/qdrant/{collection_name}").expanduser()
    client = QdrantClient(path=qdrant_path)
    return QdrantRM(
        qdrant_collection_name=collection_name, qdrant_client=client, k=3)


def get_llm(llm_type: Literal["vllm", "openai", "openai_vllm"], openai_key_path: Optional[str] = None):

    vllm_model = 'alpindale/mistral-7b-safetensors'
    if llm_type == "openai":
        assert openai_key_path is not None
        return dspy.OpenAI(api_key=load_api_key(openai_key_path))
    elif llm_type == "openai_vllm":
        return OpenAIVLLMClient(model=vllm_model, port=8000, url="http://localhost")
    else:
        return VLLMClient(model=vllm_model, port=8000, url="http://localhost")
