import polars
import pytantivy
from time import time
import pandas as pd
import logging
import eland
import elasticsearch
import warnings
from elasticsearch.exceptions import ElasticsearchWarning
warnings.simplefilter('ignore', ElasticsearchWarning)
logging.basicConfig(level=logging.WARNING)


class Timing:

    run_times = {}

    @staticmethod
    def timer_func(func):
        # This function shows the execution time of
        # the function object passed
        def wrap_func(*args, **kwargs):
            t1 = time()
            result = func(*args, **kwargs)
            t2 = time()
            elapsed = t2 - t1
            logging.warning(
                f'Function {func.__name__!r} executed in {(elapsed):.4f}s')
            Timing.run_times[func.__name__] = elapsed
            return result
        return wrap_func


@Timing.timer_func
def index_polars_pytantivy(index_name, df):
    pytantivy.initialize_index(index_name, "repo", [
                               "tasks", "true_tasks", "dependencies"])
    logging.warning("Indexing...")
    pytantivy.index_polars_dataframe("benchmarking", df)
    logging.warning("Searching...")
    pytantivy.search("benchmarking", "object detection")


@Timing.timer_func
def query_pytantivy(index_name, query):
    results = pytantivy.search(index_name, query)
    logging.warning(f"Found {len(results)} results for query '{query}'")


@Timing.timer_func
def load_polars(path):
    return polars.read_csv(path).drop_nulls()


def pytantivy_benchmark(path, query):
    logging.warning("loading df using polars")
    df = load_polars(path)

    index_name = "benchmarking"
    index_polars_pytantivy("benchmarking", df)
    query_pytantivy("benchmarking", query)


@Timing.timer_func
def index_pandas_eland(index_name, df):
    logging.warning("Indexing...")
    es_client = elasticsearch.Elasticsearch(hosts="http://localhost:9200")

    mapping = {col: "text" for col in df.columns}
    eland_df = eland.pandas_to_eland(
        df, es_dest_index="benchmark", es_client=es_client, es_if_exists="replace", es_type_overrides=mapping)
    return eland_df


@Timing.timer_func
def query_eland(edf, query):
    results = edf.es_match(
        query, columns=list(edf.columns))
    logging.warning(f"Found {len(results)} results for query '{query}'")


@Timing.timer_func
def load_pandas(path):
    return pd.read_csv(path).dropna()


def eland_benchmark(path, query):
    df = load_pandas(path)
    index_name = "benchmarking"
    invalid_cols = ["", "Unnamed: 0"]
    for col in invalid_cols:
        if col in df.columns:
            df = df.drop(columns=[col])
    eland_df = index_pandas_eland(index_name, df)
    query_eland(eland_df, query)


if __name__ == "__main__":

    data_path = "data/search_example.csv"
    query = "object detection"
    logging.warning("eland")
    eland_benchmark(data_path, query)

    logging.warning("pytantivy")
    pytantivy_benchmark(data_path, query)

    time_ratio = (
        Timing.run_times["index_pandas_eland"] /
        Timing.run_times["index_polars_pytantivy"]
    )
    logging.warning(
        f"pytantivy indexing is {time_ratio:.2f}x faster than eland")
