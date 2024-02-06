import time
from pathlib import Path
from typing import Optional
import os

from fastrag.stores import PLAIDDocumentStore

from create_doc_collection import \
    create_doc_collection as create_doc_collection_main
import fire
import yaml
from pydantic import BaseModel, Field


class SetupDocCollectionConfig(BaseModel):
    corpus_df_path: str
    text_col: str
    collection_path: str
    title_col: Optional[str]
    additional_cols: list
    n_docs: Optional[int] = Field(default=None)
    doc_len: int = Field(default=1024)


class SetupIndexConfig(BaseModel):
    collection_path: Path
    index_save_path: Path
    name: str
    checkpoint: Path = Field(default="Intel/ColBERT-NQ")
    gpus: int = Field(default=1)
    ranks: int = Field(default=1)
    doc_max_length: int = Field(default=256)
    query_max_length: int = Field(default=60)
    kmeans_iterations: int = Field(default=4)
    nbits: int = Field(default=2)


class IndexingConfig(SetupDocCollectionConfig, SetupIndexConfig):

    @classmethod
    def load(cls, path):
        with open(path, "r") as f:
            config = yaml.safe_load(f)
            if config.get("name") is None:
                config["name"] = cls.get_name(config["corpus_df_path"])
            if config.get("index_save_path") is None:
                config["index_save_path"] = f"{config['name']}_index"
        return IndexingConfig(**config)

    @classmethod
    def get_name(cls, corpus_df_path):
        return os.path.basename(corpus_df_path).split(".")[0]


class Main:

    def create_doc_collection(
        self,
        input_path: str,
        text_column: str,
        output_path: str = "doc_coll.tsv",
        title_column: Optional[str] = None,
        n_docs: Optional[int] = None,
        doc_len: int = 1024,
        additional_cols=["tasks"]
    ):
        create_doc_collection_main(input_path, text_column, output_path,
                                   title_column, n_docs, doc_len, additional_cols)

    def make_plaid_index(self, checkpoint: Path, collection: Path, index_save_path: Path, gpus: int = 0, ranks: int = 1, doc_max_length: int = 120, query_max_length: int = 60, kmeans_iterations: int = 4, name: str = "plaid_index", nbits: int = 2):
        t_0 = time.perf_counter()
        store = PLAIDDocumentStore(
            index_path=f"{index_save_path}",
            checkpoint_path=f"{checkpoint}",
            collection_path=f"{collection}",
            create=True,
            nbits=nbits,
            gpus=gpus,
            ranks=ranks,
            doc_maxlen=doc_max_length,
            query_maxlen=query_max_length,
            kmeans_niters=kmeans_iterations,
        )
        t_end = time.perf_counter()
        print(f"Indexing took {t_end - t_0:.2f} seconds")

    def create_plaid_index_with_config(self, config_path: Path):
        config = IndexingConfig.load(config_path)
        self.create_doc_collection(
            config.corpus_df_path, config.text_col, config.collection_path,
            config.title_col, config.n_docs, config.doc_len, config.additional_cols
        )
        self.make_plaid_index(
            config.checkpoint, config.collection_path, config.index_save_path,
            config.gpus, config.ranks, config.doc_max_length, config.query_max_length,
            config.kmeans_iterations, config.name, config.nbits
        )


if __name__ == "__main__":
    fire.Fire(Main())
