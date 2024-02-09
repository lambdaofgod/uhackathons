from pydantic import BaseModel, Field
from typing import Optional
from pathlib import Path
import yaml
import os


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
