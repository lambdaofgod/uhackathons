import requests
import xmltodict
from bs4 import BeautifulSoup
from typing import Literal, Dict, List
from tqdm.contrib.concurrent import process_map
import re
import arxiv2bib


class ReferenceAbstractExtractor:

    @classmethod
    def get_referenced_paper_abstracts(cls, arxiv_id) -> List[Dict[str, str]]:
        referenced_ids = cls.get_referenced_arxiv_ids(arxiv_id)
        return list(process_map(cls.get_abstract_record, referenced_ids))

    @classmethod
    def get_abstract_record(cls, arxiv_id):
        return {
            "id": arxiv_id, "title": cls.get_title(arxiv_id), "abstract": cls.get_abstract(arxiv_id), "bibtex": cls._try_extract_bib(arxiv_id)
        }

    @classmethod
    def get_referenced_paper_abstracts_(cls, arxiv_id) -> List[Dict[str, str]]:
        referenced_ids = cls.get_referenced_arxiv_ids(arxiv_id)
        return [
            {"id": arxiv_id, "title": cls.get_title(
                arxiv_id), "abstract": cls.get_abstract(arxiv_id), "bibtex": cls._try_extract_bib(arxiv_id)}
            for arxiv_id in referenced_ids
        ]

    @classmethod
    def get_html(cls, arxiv_id, type_: Literal["abs", "html"]):
        url = f"http://arxiv.org/{type_}/{arxiv_id}"
        return requests.get(url).content

    @classmethod
    def get_abstract(cls, arxiv_id):
        abs_html = cls.get_html(arxiv_id, "abs")
        return cls._extract_abstract(abs_html)

    @classmethod
    def get_referenced_arxiv_ids(cls, arxiv_id):
        paper_html = cls.get_html(arxiv_id, "html")
        reference_elements = cls._extract_references(paper_html)
        reference_extracted_arxiv_ids = [
            cls._try_extract_arxiv_id(elem) for elem in reference_elements]
        return [
            aid for aid in reference_extracted_arxiv_ids
            if aid is not None
        ]

    @classmethod
    def get_title(cls, arxiv_id):
        abs_html = cls.get_html(arxiv_id, "abs")
        return cls._extract_title(abs_html)

    @classmethod
    def _extract_title(cls, html_content):
        soup = BeautifulSoup(html_content, 'html.parser')
        title_element = soup.find("title")
        return title_element.text

    @classmethod
    def _extract_abstract(cls, html_content):
        # Parse the HTML content
        soup = BeautifulSoup(html_content, 'html.parser')

        # Find the blockquote element with class "abstract mathjax"
        abstract_element = soup.find('blockquote', class_='abstract mathjax')

        # Check if the abstract element is found
        if abstract_element:
            # Extract the text inside the abstract element
            abstract_text = abstract_element.text
            return abstract_text.strip().strip("Abstract:")

        return None

    @classmethod
    def _extract_references(cls, html_section):
        soup = BeautifulSoup(html_section, 'html.parser')
        bibsections = soup.find_all(class_="ltx_bibitem")

        return [b.text for b in bibsections]

    @classmethod
    def _try_extract_arxiv_id(cls, elem):
        match = re.search(r'arXiv:(\d+\.\d+)', elem)
        if match:
            return match.group(1)
        else:
            return None

    @classmethod
    def _try_extract_bib(cls, arxiv_id):
        bibtexs = arxiv2bib.arxiv2bib([arxiv_id])
        if len(bibtexs) > 0:
            return bibtexs[0].bibtex()
        else:
            return None
