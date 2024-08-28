from pydantic import BaseModel
from typing import List
from datetime import datetime


class Source(BaseModel):
    title: str
    url: str
    content: str


class Question(BaseModel):
    thread_id: str
    question: str
    created_at: datetime
