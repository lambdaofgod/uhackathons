from query_index import PlaidIndex, IndexConfig
import fastapi
import uvicorn
import fire
from haystack import Document
from typing import List
from pydantic import BaseModel

app = fastapi.FastAPI()


class SearchArgs(BaseModel):
    query: str
    k: int = 10


@app.post("/search")
def search(search_args: SearchArgs) -> List[Document]:
    results = app.state.plaid_index.query(search_args.query)
    return results


def main(port=4321, host="0.0.0.0"):
    app.state.plaid_index = PlaidIndex.create_index(IndexConfig())
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    fire.Fire(main)
