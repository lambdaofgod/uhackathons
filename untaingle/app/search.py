from langchain_community.tools import DuckDuckGoSearchResults


class Searcher:
    def __init__(self):
        self.search_tool = None

    def search(self, query):
        return self.search_tool.search(query)


tool = DuckDuckGoSearchResults()
