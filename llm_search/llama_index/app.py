from llama_index import SimpleDirectoryReader
from llama_index.llama_pack import download_llama_pack
from contextlib import contextmanager
import uvicorn
from pydantic_models import RetrievalRequest, RetrievalResult, RAGRequest, RAGResult
from llama_index.response.schema import Response
from llama_index.schema import TextNode
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from typing import List
from rag import get_rag_result, get_retrieval_result

app = FastAPI()


def setup_search_system(documents_dir: str):
    # load in some sample data
    reader = SimpleDirectoryReader(input_dir=documents_dir)
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
        content={"message": f"App error: {exc.reason}"},
    )


@contextmanager
def exception_handler(name: str):
    try:
        yield
    except Exception as e:
        raise ExceptionResponse(e)


@app.post("/rag", response_model=RAGResult)
def rag(request: RAGRequest):
    with exception_handler("rag"):
        rag_result = get_rag_result(app.state.search_system, request)
    return rag_result


@app.post("/retrieve", response_model=List[RetrievalResult])
def retrieve(request: RetrievalRequest):
    with exception_handler("retrieve"):
        result = get_retrieval_result(app.state.index, request)
    return result


def main(host, port, documents_dir):
    search_system = setup_search_system(documents_dir)
    app.state.search_system = search_system
    app.state.index = search_system.index
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main(host="0.0.0.0", port=8910, documents_dir="/home/kuba/Projects/org/roam")
