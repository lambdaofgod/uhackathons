import qdrant_client
import fastapi
import uvicorn
import fire
import os
from pydantic import Field, BaseModel


app = fastapi.FastAPI()
DEFAULT_COLLECTION_NAME = os.environ.get("DEFAULT_COLLECTION_NAME", "org")


def get_qdrant_client():
    return qdrant_client.QdrantClient(host="qdrant", port=6333)


class SearchArgs(BaseModel):
    query: str
    collection_name: str = Field(default=DEFAULT_COLLECTION_NAME)
    k: int = Field(default=10)


@app.post("/search")
def search(search_args: SearchArgs):
    results = app.state.qdrant_client.query(
        collection_name=search_args.collection_name,
        query_text=search_args.query,
        limit=search_args.k
    )
    return results


def dry_run(client):
    client.query(
        collection_name=DEFAULT_COLLECTION_NAME,
        query_text="What is the best way to learn deep learning?",
        limit=1
    )


def main(host, port):
    app.state.qdrant_client = get_qdrant_client()
    dry_run(app.state.qdrant_client)
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    fire.Fire(main)
