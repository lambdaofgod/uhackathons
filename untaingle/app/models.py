from pydantic import BaseModel
from typing import List
from datetime import datetime


class Source(BaseModel):
    title: str
    url: str
    content: str


class Answer(BaseModel):
    thread_id: str
    question: str
    answer_text: str
    sources: List[Source]
    related: List[str]
    created_at: datetime
