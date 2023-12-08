from qdrant_client import QdrantClient
import dspy
from pathlib import Path
from dspy.retrieve.qdrant_rm import QdrantRM
import dspy.retrieve
from llm_search_dspy.vllm_client import VLLMClient, OpenAIVLLMClient
import fire
from typing import Literal


def load_api_key(path="~/.keys/openai_key.txt"):
    with open(Path(path).expanduser()) as f:
        key = f.read().strip()
    return key


class RAG(dspy.Module):
    def __init__(self, num_passages=3):
        super().__init__()
        self.retrieve = dspy.Retrieve(k=num_passages)
        self.generate_answer = dspy.ChainOfThought(
            "context, question -> answer")

    def forward(self, question):
        context = self.retrieve(question).passages
        answer = self.generate_answer(context=context, question=question)
        return context, answer


def get_retriever(collection_name):
    qdrant_path = Path(f"~/.cache/qdrant/{collection_name}").expanduser()
    client = QdrantClient(path=qdrant_path)
    return QdrantRM(
        qdrant_collection_name=collection_name, qdrant_client=client, k=3)


def get_llm(llm_type: Literal["vllm", "openai", "openai_vllm"]):

    vllm_model = 'alpindale/mistral-7b-safetensors'
    if llm_type == "openai":
        return dspy.OpenAI(api_key=load_api_key())
    elif llm_type == "openai_vllm":
        return OpenAIVLLMClient(model=vllm_model, port=8000, url="http://localhost")
    else:
        return VLLMClient(model=vllm_model, port=8000, url="http://localhost")


def main(query="What dspy modules I need to use qdrant?", llm_type="openai_vllm", collection_name="dspy"):
    print(f"using {llm_type}")
    retriever = get_retriever(collection_name)
    llm = get_llm(llm_type)
    dspy.settings.configure(lm=llm, rm=retriever)
    rag = RAG()  # zero-shot, uncompiled version of RAG
    context, answer = rag(query)
    print("#" * 50)
    print("ANSWER:")
    print("#" * 50)
    print(answer.answer)
    print("#" * 50)
    print("RATIONALE:")
    print("#" * 50)
    print(answer.rationale)
    print("#" * 50)
    print("CONTEXT:")
    for passage in context:
        print(passage)


if __name__ == "__main__":
    fire.Fire(main)
