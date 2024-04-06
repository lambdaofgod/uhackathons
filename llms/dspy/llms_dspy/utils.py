from pathlib import Path
from typing import Literal, Optional

import dspy
import dspy.retrieve
from qdrant_client import QdrantClient

from llms_dspy.vllm_client import OpenAIVLLMClient, VLLMClient
# we use a fixed QdrantRM
# original one has a bug in the forward method (returns strings instead of dotdict)
from llms_dspy.qdrant_rm import QdrantRM
from dspy.retrieve.chromadb_rm import ChromadbRM
from chromadb.utils import embedding_functions


def load_api_key(key_path):
    with open(Path(key_path).expanduser()) as f:
        key = f.read().strip()
    return key


def get_qdrant_retriever(collection_name, host=None, port=None):
    qdrant_path = Path(f"~/.cache/qdrant/{collection_name}").expanduser()
    if host is None or port is None:
        client = QdrantClient(path=qdrant_path)
    else:
        client = QdrantClient(host=host, port=port)
    return QdrantRM(
        qdrant_collection_name=collection_name, qdrant_client=client, k=3)


def get_llm(llm_type: Literal["vllm", "openai", "openai_vllm"], openai_key_path: Optional[str] = None, model_name: Optional[str] = None):
    if llm_type == "openai":
        assert openai_key_path is not None
        if model_name is None:
            model_name = "gpt-3.5-turbo"
        return dspy.OpenAI(model=model_name, api_key=load_api_key(openai_key_path))
    elif llm_type == "openai_vllm":
        return OpenAIVLLMClient(model=model_name, port=8000, url="http://localhost")
    else:
        return VLLMClient(model=model_name, port=8000, url="http://localhost")


def get_chroma_retriever(chroma_dir, collection_name):
    assert Path(chroma_dir).exists()
    embedding_function = embedding_functions.DefaultEmbeddingFunction()
    return ChromadbRM(collection_name=collection_name, persist_directory=chroma_dir, embedding_function=embedding_function)
