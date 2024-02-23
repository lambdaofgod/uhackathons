import pandas as pd
from pydantic import BaseModel
from search_client import SearchClient
import tqdm
from typing import Union, Sequence, Dict


class MetricsCalculator(BaseModel):
    id_field: str
    true_queries_field: str

    def get_metrics(self, results, query_label, k=10):
        checked_results = self._check_results(results, query_label)[:k]
        n_hits = len([res for res in checked_results if res["do_match"]])
        return {f"hits@{k}": n_hits, f"accuracy@{k}": n_hits > 0}

    def _check_results(self, results, query_label):
        return [self._get_match_result(res, query_label) for res in results]

    def _get_match_result(self, result, query_label):
        true_labels = result["meta"][self.true_queries_field]
        return {
            self.id_field: result["meta"][self.id_field],
            self.true_queries_field: true_labels,
            "do_match": self._do_labels_match(query_label, true_labels),
        }

    def _do_labels_match(self, query_label, true_labels):
        return query_label in true_labels


class ColBERTResultEvaluator(BaseModel):
    api_url: str
    true_queries_field: str

    def _run_evaluation(self, labeled_queries: Dict[str, str], id_field="repo_name"):
        searcher = SearchClient(api_url=self.api_url)
        query_results = {
            (q_label, q): searcher.search_grouped(
                query=q, group_by=id_field, raw_docs_topk=5000
            )
            for (q_label, q) in tqdm.tqdm(labeled_queries.items())
        }

        metric_calculator = MetricsCalculator(
            id_field=id_field, true_queries_field=self.true_queries_field
        )
        evaluation_results_df = pd.DataFrame.from_records(
            [
                {
                    "query_label": q_label,
                    "query": q,
                    **metric_calculator.get_metrics(results, q_label),
                }
                for ((q_label, q), results) in query_results.items()
            ]
        )
        return evaluation_results_df

    def prepare_queries_df(
        self,
        raw_queries_df,
        query_field,
        query_label_field=None,
        query_doc_count_field="doc_count",
    ):
        if query_label_field is None:
            query_label_field = query_field
        return pd.DataFrame(
            {
                "query": raw_queries_df[query_field],
                "query_label": raw_queries_df[query_label_field],
                "query_doc_count": raw_queries_df[query_doc_count_field],
            }
        )

    def get_evaluation_df(self, queries_df, id_field="repo_name"):
        labeled_queries = (
            queries_df[["query_label", "query"]]
            .set_index("query_label")["query"]
            .to_dict()
        )
        evaluation_df = self._run_evaluation(labeled_queries, id_field=id_field)
        return evaluation_df.assign(query_doc_count=queries_df["query_doc_count"])


if __name__ == "__main__":
    # Example usage
    api_url = "http://localhost:5432"
    true_queries_field = "tasks"
    evaluator = ColBERTResultEvaluator(
        api_url=api_url, true_queries_field=true_queries_field
    )

    res = evaluator._run_evaluation({"autonomous vehicles": "autonomous vehicles"})
    print(res)
    # raw_queries_df = pd.read_csv("/tmp/pwc_tasks_metadata.csv")

    # queries_df = evaluator.prepare_queries_df(
    #     raw_queries_df, query_field="task_description", query_label_field="task", query_doc_count_field="n_repos")
    # queries_df = pd.DataFrame.from_records([
    #     {"query": "object detection", "query_doc_count": 100,
    #         "query_label": "object detection"},
    # ])
    # evaluation_df = evaluator.get_evaluation_df(
    #     queries_df, id_field="repo_name")
    # print(evaluation_df)
