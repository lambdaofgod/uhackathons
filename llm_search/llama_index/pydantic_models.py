from pydantic import BaseModel, Field


class QARequest(BaseModel):
    question: str
    similarity_top_k: int = Field(default=2, ge=1)


class RetrievalRequest(BaseModel):
    text: str
    similarity_top_k: int = Field(default=2, ge=1)


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
            end_char_idx=r.node.end_char_idx
        )
