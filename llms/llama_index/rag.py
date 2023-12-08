from pydantic_models import RetrievalRequest, RetrievalResult, RAGRequest, RAGResult
from typing import List


def get_rag_result(search_system, request: RAGRequest) -> RAGResult:
    raw_rag_response = search_system.run(**dict(request))
    if request.return_retrieval_results:
        retrieval_results = [
            RetrievalResult.from_llamaindex_node(r)
            for r in raw_rag_response.source_nodes
        ]
    else:
        retrieval_results = []
    return RAGResult(
        response=raw_rag_response.response,
        retrieval_results=retrieval_results,
    )


def get_retrieval_result(index, request: RetrievalRequest) -> List[RetrievalResult]:
    retriever = index.as_retriever(similarity_top_k=request.top_k)
    raw_results = retriever.retrieve(request.text)
    return [RetrievalResult.from_llamaindex_node(r) for r in raw_results]
