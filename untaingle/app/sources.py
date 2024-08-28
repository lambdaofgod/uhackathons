from duckduckgo_search import DDGS
from app.models import Source
from typing import List


class SourceProvider:
    def __init__(self):
        self.search_tool = DDGS()

    def search(self, query, max_results=5) -> List[Source]:
        return self._ddg_search(query, max_results)

    def _ddg_search(self, query, max_results=5):
        results = self.search_tool.text(
            query, max_results=max_results)
        return [Source(title=result["title"], url=result["href"], content=result["body"]) for result in results]
