from llama_index import SimpleDirectoryReader
from llama_index.llama_pack import download_llama_pack
from contextlib import contextmanager
import uvicorn
from pydantic import BaseModel, Field
from llama_index.response.schema import Response
from llama_index.schema import TextNode
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from typing import List


app = FastAPI()


def setup_search_system(documents_dir: str):
    # load in some sample data
    reader = SimpleDirectoryReader(
        input_dir=documents_dir)
    documents = reader.load_data()

    # download and install dependencies
    ZephyrQueryEnginePack = download_llama_pack(
        "ZephyrQueryEnginePack", "./zephyr_pack"
    )
    return ZephyrQueryEnginePack(documents)


class ExceptionResponse(Exception):
    def __init__(self, e: Exception):
        self.reason = str(e)


@app.exception_handler(ExceptionResponse)
async def app_exception_handler(request: Request, exc: ExceptionResponse):
    return JSONResponse(
        status_code=418,
        content={"message": f"App error: {exc.name}\n Reason: {exc.reason}"},
    )


@contextmanager
def exception_handler(name: str):
    try:
        yield
    except Exception as e:
        raise ExceptionResponse(name, e)


class QARequest(BaseModel):
    question: str
    similarity_top_k: int = Field(default=2, ge=1)


class SearchResult(BaseModel):
    id: str

    score: float


# @app.post("/answer", response_model=Response)
# def answer(request: QARequest):
#     return app.state.search_system.run(**dict(request))


class RetrievalResult(BaseModel):
    text: str
    metadata: dict
    score: float

    @classmethod
    def from_llamaindex_node(cls, r):
        return RetrievalResult(text=r.node.text, score=r.score, metadata=r.node.metadata)


@app.post("/retrieve", response_model=List[RetrievalResult])
def retrieve(text: str):
    with exception_handler("retrieve"):
        result = [RetrievalResult.from_llamaindex_node(
            r) for r in app.state.retriever.retrieve(text)]
    return result


def main(host, port, documents_dir):
    search_system = setup_search_system(documents_dir)
    app.state.search_system = search_system
    app.state.retriever = search_system.index.as_retriever()
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main(host="0.0.0.0", port=8910, documents_dir="/home/kuba/Projects/org/roam")
