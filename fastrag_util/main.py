import time
from pathlib import Path
from typing import Optional
import os
import pandas as pd
from fastrag.stores import PLAIDDocumentStore

from create_doc_collection import CollectionBuilder
import fire
import yaml
from pydantic import BaseModel, Field
from configs import SetupDocCollectionConfig, SetupIndexConfig, IndexingConfig


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
        CollectionBuilder.create_doc_collection(input_path, text_column, output_path,
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

    def foo(self, config_path: Path):
        config = IndexingConfig.load(config_path)
        df = pd.read_json(config.corpus_df_path)
        import ipdb
        ipdb.set_trace()


if __name__ == "__main__":
    fire.Fire(Main())
