import fire
import pandas as pd
from typing import Optional
import re
from configs import SetupDocCollectionConfig
import yaml
import logging

logging.basicConfig(level="INFO")


def load_text_df(path):
    if path.endswith(".csv") or path.endswith(".csv.gz"):
        return pd.read_csv(path)
    if path.endswith(".json") or path.endswith(".json.gz"):
        return pd.read_json(path)
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported file format: {path}")


def reorder_cols(df, title_col, additional_cols, starting_cols=["id", "content"]):
    ordered_cols = starting_cols
    if title_col:
        ordered_cols.append("title")
    ordered_cols.extend(additional_cols)
    return df[ordered_cols]


def process_list_col(column):
    return column.apply(lambda l: f"[{', '.join(l)}]")


def process_col(column):
    if type(column.iloc[0]) is not str:
        print(f"column {column.name}")
        print("contains something other than string, converting to string")
        return process_list_col(column)
    else:
        return column


def prepare_doc_coll_df(
    df: pd.DataFrame,
    text_col: str,
    doc_len: int,
    title_col: Optional[str] = None,
    additional_cols: list = [],
):
    whitespace_re = re.compile(r"\s+")

    text_coll = df[text_col].apply(lambda x: whitespace_re.sub(" ", x[:doc_len]))
    doc_coll_df = pd.DataFrame({"content": text_coll})

    doc_coll_df = doc_coll_df[["content"]]
    if title_col:
        doc_coll_df["title"] = df[title_col]
        doc_coll_df = doc_coll_df.drop_duplicates(subset=["title"])
    for col in additional_cols:
        assigned_column = process_col(df[col])
        doc_coll_df[col] = assigned_column
    doc_coll_df["id"] = range(1, len(doc_coll_df) + 1)
    return reorder_cols(doc_coll_df, title_col, additional_cols)


def get_ascii_ratio(s: str):
    return len(s.encode("ascii", "ignore")) / len(s)


def filter_non_english_texts(
    df: pd.DataFrame, text_col: str, min_ascii_ratio: float = 0.9
):
    """
    filter out text that has too many non-ascii chars
    """
    ascii_ratios = df[text_col].apply(get_ascii_ratio)
    return df[ascii_ratios > min_ascii_ratio]


def filter_texts(
    df: pd.DataFrame,
    text_col: str,
    title_col: Optional[str],
    min_ascii_ratio: float = 0.9,
    n_docs=None,
):
    """
    filter out text that has too many non-ascii chars
    """
    if n_docs is not None:
        df = df.iloc[: 2 * n_docs]
    nonna_cols = [text_col]
    if title_col:
        nonna_cols.append(title_col)
    df = df.dropna(subset=nonna_cols)
    df = df[df[text_col].apply(lambda x: len(x) > 0)]
    df = filter_non_english_texts(df, text_col, min_ascii_ratio)
    if n_docs is not None:
        df = df.iloc[:n_docs]
    return df


class CollectionBuilder:
    @staticmethod
    def create_doc_collection(
        corpus_df_path: str,
        text_col: str,
        collection_path: str = "doc_coll.tsv",
        title_col: Optional[str] = None,
        n_docs: Optional[int] = None,
        doc_len: int = 1024,
        additional_cols=["tasks"],
    ):
        df = load_text_df(corpus_df_path)
        if n_docs is not None:
            df = df.iloc[: 2 * n_docs]
            df = filter_texts(df, text_col, title_col, n_docs=n_docs)
            df = df.iloc[:n_docs]
        else:
            df = filter_texts(df, text_col, title_col)
        doc_coll_df = prepare_doc_coll_df(
            df, text_col, doc_len, title_col, additional_cols
        )

        logging.info(f"extracted {doc_coll_df.shape[0]} documents")
        logging.info(f"saving extracted collection to {collection_path}")
        (doc_coll_df.to_csv(collection_path, sep="\t", escapechar="\\", index=False))

    @staticmethod
    def from_config(config_path: str):
        with open(config_path) as f:
            config_dict = yaml.safe_load(f)
        config = SetupDocCollectionConfig(**config_dict)
        CollectionBuilder.create_doc_collection(**config.dict())


if __name__ == "__main__":
    fire.Fire(CollectionBuilder)
