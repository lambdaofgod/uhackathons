import bs4
import requests


class ArxivExtractor:

    @classmethod
    def get_arxiv_html(cls, url):
        content = requests.get(url).content
        return bs4.BeautifulSoup(content)

    @classmethod
    def extract_bibliography_elements(cls, parsed_html):
        ref_fragment = parsed_html.find("ul", {"class": "ltx_biblist"})
        return ref_fragment.find_all("li")

    @classmethod
    def get_bibliography_elements(cls, url):
        return cls.extract_bibliography_elements(cls.get_arxiv_html(url))
