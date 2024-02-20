
import json
from pathlib import Path
from typing import List

import fire
import pandas as pd
from fastrag.stores import PLAIDDocumentStore
from haystack import Document
from pydantic import BaseModel

from configs import IndexingConfig


class IndexConfig(BaseModel):
    index_path: Path = Path("plaid_index")
    collection_path: Path = Path("doc_coll.tsv")
    gpus: int = 1
    checkpoint_path: str = "Intel/ColBERT-NQ"
    additional_metadata_cols: list = ["tasks"]

    @classmethod
    def create_from_indexing_config(cls, indexing_config):
        return IndexConfig(
            index_path=indexing_config.index_save_path,
            collection_path=indexing_config.collection_path,
            gpus=indexing_config.gpus,
            checkpoint_path=indexing_config.checkpoint,
            additional_metadata_cols=indexing_config.additional_cols
        )

    @classmethod
    def load_from_indexing_config_path(cls, path):
        indexing_config = IndexingConfig.load(path)
        return cls.create_from_indexing_config(indexing_config)


class PlaidIndex(BaseModel):

    config: IndexConfig
    store: PLAIDDocumentStore
    collection: pd.DataFrame

    @classmethod
    def create_index(cls, config):
        store = PLAIDDocumentStore(
            index_path=str(config.index_path), collection_path=str(config.collection_path), create=False, gpus=config.gpus, checkpoint_path=config.checkpoint_path)
        collection = pd.read_csv(
            config.collection_path, sep="\t").set_index("title")
        return cls(config=config, store=store, collection=collection)

    def query(self, query, top_k=10):
        results = self.store.query(query, top_k=top_k)
        return self.DocUtils.fill_doc_metas(results, set(self.config.additional_metadata_cols), self.collection)

    def query_grouped(self, query, group_by, top_k, raw_docs_topk):
        doc_results = self.query(query, raw_docs_topk)
        return self.DocUtils.doc_results_to_grouped_results([doc.to_dict() for doc in doc_results], group_by, top_k)

    class DocUtils:

        @classmethod
        def fill_doc_metas(cls, docs: List[Document], additional_metadata_cols, metadata_df) -> List[Document]:
            # some docs might have invalid metadata, we ignore them
            maybe_filled_docs = [
                cls.fill_doc_meta(doc, additional_metadata_cols, metadata_df)
                for doc in docs
            ]
            return [
                doc
                for doc in maybe_filled_docs
                if doc is not None
            ]

        @classmethod
        def fill_doc_meta(cls, doc: Document, additional_metadata_cols, metadata_df):
            doc_title = doc.meta["title"]
            if doc_title not in metadata_df.index:
                return None
            else:
                row = metadata_df.loc[doc_title]
                for col in additional_metadata_cols:
                    doc.meta[col] = row[col]
            return doc

        @classmethod
        def doc_results_to_grouped_results(cls, doc_results, group_by, top_k):
            doc_result_dicts = [cls._doc_with_grouping_field(
                doc, group_by) for doc in doc_results]
            doc_results_df = pd.DataFrame.from_records(doc_result_dicts)
            grouped_results_df = doc_results_df.drop_duplicates(subset=[
                                                                group_by])
            results = grouped_results_df.iloc[:top_k].to_dict(orient="records")
            return results

        @classmethod
        def _doc_with_grouping_field(cls, doc_dict, group_by):
            doc_dict[group_by] = doc_dict["meta"][group_by]
            return doc_dict

    class Config:
        arbitrary_types_allowed = True


def query_main(query, results_path=None):
    plaid_index = PlaidIndex.create_index(IndexConfig())
    results = plaid_index.query(query)
    print(results)
    if results_path:
        result_dicts = [result.to_dict() for result in results]
        with open(results_path, "w") as f:
            json.dump(result_dicts, f)


if __name__ == '__main__':
    fire.Fire(query_main)
