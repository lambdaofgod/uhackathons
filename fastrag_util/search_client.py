from pydantic import BaseModel
import requests
import json


class SearchClient(BaseModel):
    api_url: str = 'http://localhost:4321'

    @property
    def search_grouped_url(self):
        return f"{self.api_url}/search_grouped"

    @property
    def search_url(self):
        return f"{self.api_url}/search_grouped"

    def search_grouped(self, query, group_by="repo_name", top_k=10, raw_docs_topk=1000):
        headers = {
            'accept': 'application/json',
            'Content-Type': 'application/json',
        }

        data = {"query": query,
                "top_k":  top_k,
                "group_by": group_by,
                "raw_docs_topk": raw_docs_topk
                }

        response = requests.post(
            self.search_grouped_url, headers=headers, data=json.dumps(data))

        return response.json()


class ResultEvaluator(BaseModel):
    result_key: str
    true_queries_key: str

    def get_metrics(self, results, query, k=10):
        checked_results = self.check_results(results, query)[:k]
        n_hits = len([res for res in checked_results if res["do_match"]])
        return {f"hits@{k}": n_hits, f"accuracy@{k}": n_hits > 0}

    def check_results(self, results, query):
        return [self.get_match_result(res, query) for res in results]

    def get_match_result(self, result, query):
        true_queries = result["meta"][self.true_queries_key]
        return {
            self.result_key: result["meta"][self.result_key],
            self.true_queries_key: true_queries,
            "do_match": self.do_queries_match(query, true_queries)
        }

    def do_queries_match(self, query, true_queries):
        return query in true_queries
