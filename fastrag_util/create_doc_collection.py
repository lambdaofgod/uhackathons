import fire
import pandas as pd
from typing import Optional
import re


def load_text_df(path):
    if path.endswith('.csv') or path.endswith('.csv.gz'):
        return pd.read_csv(path)
    if path.endswith('.json') or path.endswith('.json.gz'):
        return pd.read_json(path)
    if path.endswith('.parquet'):
        return pd.read_parquet(path)
    raise ValueError(f'Unsupported file format: {path}')


def prepare_doc_coll_df(df: pd.DataFrame, text_column: str, doc_len: int, title_column: Optional[str] = None, additional_cols: list = []):
    whitespace_re = re.compile(r'\s+')

    text_coll = df[text_column].apply(
        lambda x: whitespace_re.sub(" ", x[:doc_len]))
    doc_coll_df = pd.DataFrame({"content": text_coll})

    doc_coll_df['id'] = range(1, len(df)+1)
    doc_coll_df = doc_coll_df[['id', 'content']]
    if title_column:
        doc_coll_df['title'] = df[title_column]
    for col in additional_cols:
        doc_coll_df[col] = df[col]
    return doc_coll_df


def get_ascii_ratio(s: str):
    return len(s.encode("ascii", "ignore")) / len(s)


def filter_non_english_texts(df: pd.DataFrame, text_column: str, min_ascii_ratio: float = 0.9):
    """
    filter out text that has too many non-ascii chars
    """
    ascii_ratios = df[text_column].apply(get_ascii_ratio)
    return df[ascii_ratios > min_ascii_ratio]


def filter_texts(df: pd.DataFrame, text_column: str, title_column: Optional[str], min_ascii_ratio: float = 0.9):
    """
    filter out text that has too many non-ascii chars
    """
    nonna_cols = [text_column]
    if title_column:
        nonna_cols.append(title_column)
    df = df.dropna(subset=nonna_cols)
    df = df[
        df[text_column].apply(lambda x: len(x) > 0)]
    df = filter_non_english_texts(df, text_column, min_ascii_ratio)
    return df


def create_doc_collection(
    input_path: str,
    text_column: str,
    output_path: str = "doc_coll.tsv",
    title_column: Optional[str] = None,
    n_docs: Optional[int] = None,
    doc_len: int = 1024,
    additional_cols=["tasks"]
):
    df = load_text_df(input_path)
    if n_docs is not None:
        df = df.iloc[:2*n_docs]
        df = filter_texts(df, text_column, title_column)
        df = df.iloc[:n_docs]
    else:
        df = filter_texts(df, text_column, title_column)
    doc_coll_df = prepare_doc_coll_df(
        df, text_column, doc_len, title_column, additional_cols)
    # offending_row = doc_coll_df.iloc[5312]
    # readme = offending_row['content']
    # import ipdb
    # ipdb.set_trace()

    doc_coll_df.to_csv(output_path, sep="\t", index=False,
                       errors="ignore", escapechar="\\")


if __name__ == '__main__':
    fire.Fire(create_doc_collection)
