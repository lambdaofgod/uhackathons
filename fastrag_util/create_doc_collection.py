import fire
import pandas as pd
from typing import Optional
import re
from configs import SetupDocCollectionConfig
import yaml
import logging

logging.basicConfig(level="INFO")


def load_text_df(path):
    if path.endswith('.csv') or path.endswith('.csv.gz'):
        return pd.read_csv(path)
    if path.endswith('.json') or path.endswith('.json.gz'):
        return pd.read_json(path)
    if path.endswith('.parquet'):
        return pd.read_parquet(path)
    raise ValueError(f'Unsupported file format: {path}')


def prepare_doc_coll_df(df: pd.DataFrame, text_col: str, doc_len: int, title_col: Optional[str] = None, additional_cols: list = []):
    whitespace_re = re.compile(r'\s+')

    text_coll = df[text_col].apply(
        lambda x: whitespace_re.sub(" ", x[:doc_len]))
    doc_coll_df = pd.DataFrame({"content": text_coll})

    doc_coll_df = doc_coll_df[['content']]
    if title_col:
        doc_coll_df['title'] = df[title_col]
        doc_coll_df = doc_coll_df.drop_duplicates(subset=["title"])
    for col in additional_cols:
        doc_coll_df[col] = df[col]
    doc_coll_df['id'] = range(1, len(doc_coll_df)+1)
    return doc_coll_df


def get_ascii_ratio(s: str):
    return len(s.encode("ascii", "ignore")) / len(s)


def filter_non_english_texts(df: pd.DataFrame, text_col: str, min_ascii_ratio: float = 0.9):
    """
    filter out text that has too many non-ascii chars
    """
    ascii_ratios = df[text_col].apply(get_ascii_ratio)
    return df[ascii_ratios > min_ascii_ratio]


def filter_texts(df: pd.DataFrame, text_col: str, title_col: Optional[str], min_ascii_ratio: float = 0.9):
    """
    filter out text that has too many non-ascii chars
    """
    nonna_cols = [text_col]
    if title_col:
        nonna_cols.append(title_col)
    df = df.dropna(subset=nonna_cols)
    df = df[
        df[text_col].apply(lambda x: len(x) > 0)]
    df = filter_non_english_texts(df, text_col, min_ascii_ratio)
    return df


class Main:

    @staticmethod
    def create_doc_collection(
        corpus_df_path: str,
        text_col: str,
        collection_path: str = "doc_coll.tsv",
        title_col: Optional[str] = None,
        n_docs: Optional[int] = None,
        doc_len: int = 1024,
        additional_cols=["tasks"]
    ):
        df = load_text_df(corpus_df_path)
        if n_docs is not None:
            df = df.iloc[:2*n_docs]
            df = filter_texts(df, text_col, title_col)
            df = df.iloc[:n_docs]
        else:
            df = filter_texts(df, text_col, title_col)
        doc_coll_df = prepare_doc_coll_df(
            df, text_col, doc_len, title_col, additional_cols)

        ordered_cols = ["id", "content"]
        if title_col:
            ordered_cols.append("title")
        ordered_cols.extend(additional_cols)

        logging.info(f"extracted {doc_coll_df.shape[0]} documents")
        logging.info(f"saving extracted collection to {collection_path}")
        (
            doc_coll_df[ordered_cols]
            .to_csv(collection_path, sep="\t", escapechar="\\", index=False)
        )

    @staticmethod
    def from_config(config_path: str):
        with open(config_path) as f:
            config_dict = yaml.safe_load(f)
        config = SetupDocCollectionConfig(**config_dict)
        Main.create_doc_collection(**config.dict())


if __name__ == '__main__':
    fire.Fire(Main)
