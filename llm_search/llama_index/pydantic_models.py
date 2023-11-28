from pydantic import BaseModel, Field
from typing import List


class RetrievalRequest(BaseModel):
    text: str
    top_k: int = Field(default=3, ge=1)


class RetrievalResult(BaseModel):
    text: str
    metadata: dict
    score: float
    start_char_idx: int
    end_char_idx: int

    @classmethod
    def from_llamaindex_node(cls, r):
        return RetrievalResult(
            text=r.node.text,
            score=r.score,
            metadata=r.node.metadata,
            start_char_idx=r.node.start_char_idx,
            end_char_idx=r.node.end_char_idx,
        )


class RAGRequest(BaseModel):
    query_str: str
    similarity_top_k: int = Field(default=3, ge=1)
    return_retrieval_results: bool = Field(default=False)


class RAGResult(BaseModel):
    response: str
    metadata: dict
    retrieval_results: List[RetrievalResult]
