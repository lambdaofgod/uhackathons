import dspy
from llms_dspy.dspy_modules import SimpleRAG
from pathlib import Path
from llms_dspy.utils import get_chroma_retriever

def get_anthropic_lm(model_name="claude-3-haiku-20240229", key_path="~/.keys/anthropic_key.txt"):
    with open(Path(key_path).expanduser()) as f:
        key = f.read().strip()

    return dspy.Claude(model=model_name, api_key=key)


def get_openai_lm(model_name="gpt-3.5-turbo", key_path="~/.keys/openai_key.txt"):
    with open(Path(key_path).expanduser()) as f:
        key = f.read().strip()
    return dspy.OpenAI(model=model_name, api_key=key)


def get_chroma_retriever(chroma_dir, collection_name):
    embedding_function = embedding_functions.DefaultEmbeddingFunction()
    return ChromadbRM(collection_name=collection_name, persist_directory=chroma_dir, embedding_function=embedding_function)


if __name__ == "__main__":

    query = "what kind of neural network is BART?"
    haiku = get_openai_lm()

    chroma_retriever = get_chroma_retriever("chroma_db", "pdf_example")
    dspy.settings.configure(lm=haiku, rm=chroma_retriever)
    rag = SimpleRAG()  # zero-shot, uncompiled version of RAG
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
        print(passage)
