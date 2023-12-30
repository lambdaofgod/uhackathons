from typing import List, Optional
import re
from returns.maybe import Maybe
import pandas as pd
from pydantic import BaseModel
from arxivist.extractors import ArxivExtractor


class ArXivCitation(BaseModel):
    title: str
    year: Optional[int]
    source_type: Optional[str]
    source: Optional[str]
    authors: str
    ref_name: str

    @classmethod
    def create(cls, title, ref_name, authors_str, source):
        year_result = re.findall(r"\d+", ref_name)
        if len(year_result) == 0:
            year = None
        else:
            year = year_result[0]

        source_type = cls.get_source_type(Maybe.from_optional(source))
        return ArXivCitation(
            title=title.strip(),
            ref_name=ref_name.strip(),
            authors=authors_str.strip(),
            year=year,
            source=source,
            source_type=source_type
        )

    @classmethod
    def get_source_type(cls, source: Maybe[str]):
        return source.map(lambda s: s.strip().split(", ")[0]).value_or(None)


class CitationExtractor:

    @classmethod
    def bib_element_to_record(cls, element):
        element_items = [e.get_text() for e in element.find_all("span")]
        ref_name, authors_str, title = element_items[:3]
        if len(element_items) > 3:
            source = element_items[3]
        else:
            source = None
        return {"title": title, "ref_name": ref_name, "authors_str": authors_str, "source": source}

    @classmethod
    def get_citations(cls, url) -> List[ArXivCitation]:
        bib_records = [cls.bib_element_to_record(
            e) for e in ArxivExtractor.get_bibliography_elements(url)]
        return [ArXivCitation.create(**s) for s in bib_records]

    @classmethod
    def get_citations_df(cls, url, sort_by_year=True):
        df = pd.DataFrame([s.dict() for s in cls.get_citations(url)])
        if sort_by_year:
            return df.sort_values("year", ascending=False)
        else:
            return df
