import fire

import dspy
import dspy.retrieve
from llms_dspy.dspy_modules import SimpleRAG
from llms_dspy.utils import get_llm, get_qdrant_retriever


def main(query="What dspy modules I need to use qdrant?", llm_type="openai_vllm", collection_name="dspy"):
    print(f"using {llm_type}")
    retriever = get_qdrant_retriever(
        collection_name, host="localhost", port=6333)
    llm = get_llm(llm_type)
    dspy.settings.configure(lm=llm, rm=retriever)
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


if __name__ == "__main__":
    fire.Fire(main)
