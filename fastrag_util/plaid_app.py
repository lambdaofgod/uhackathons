from query_index import PlaidIndex, IndexConfig
import fastapi
import uvicorn
import fire
from haystack import Document
from typing import List
from pydantic import BaseModel, Field
from app_util import ExceptionResponse, exception_handler, ipdb_exception_handler
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import logging

logging.basicConfig(level="INFO")

app = fastapi.FastAPI()


class SearchArgs(BaseModel):
    query: str
    top_k: int = Field(default=10)


class GroupedSearchArgs(SearchArgs):
    group_by: str = Field(default="title")
    raw_docs_topk: int = Field(default=1000)


@app.exception_handler(ExceptionResponse)
async def app_exception_handler(request: Request, exc: ExceptionResponse):
    return JSONResponse(
        status_code=418,
        content={"message": f"App error: {exc.name}\n Reason: {exc.reason}"},
    )


@app.post("/search")
def search(search_args: SearchArgs) -> List[dict]:
    with ipdb_exception_handler("search"):
        results = app.state.plaid_index.query(**search_args.dict())
        results = [doc.to_dict() for doc in results]
    return results


@app.post("/search_grouped")
def search_grouped(search_args: GroupedSearchArgs) -> List[dict]:
    with ipdb_exception_handler("search"):
        results = app.state.plaid_index.query_grouped(**search_args.dict())
    return results


def main(port=4321, host="0.0.0.0", config_path=None, extra_logging=None, log_filename=None):
    if config_path is None:
        logging.warn("no config path specified, loading default config")
        config = IndexConfig()
    else:
        config = IndexConfig.load_from_indexing_config_path(config_path)
    if log_filename is not None:
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        logging.basicConfig(level="INFO", filename=log_filename)
        logging.info(f"storing logs to {log_filename}")
    app.state.plaid_index = PlaidIndex.create_index(config, extra_logging)
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    fire.Fire(main)
