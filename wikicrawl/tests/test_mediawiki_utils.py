from wikicrawl.mediawiki_utils import extract_links, extract_toc
from tests.fixtures import AGES_PAGE_TRUNCATED


def test_extract_links_page_only():
    text = "See [[Europa Universalis V]] for more info."
    result = extract_links(text)
    assert result == [{'page': 'Europa Universalis V', 'section': None}]


def test_extract_links_section_only():
    text = "Check [[#Institutions|institutions]] section."
    result = extract_links(text)
    assert result == [{'page': None, 'section': 'Institutions'}]


def test_extract_links_page_with_section():
    text = "See [[Government#Cabinet|cabinet actions]]."
    result = extract_links(text)
    assert result == [{'page': 'Government', 'section': 'Cabinet'}]


def test_extract_links_filters_file_category():
    text = "[[File:Icon.png]] and [[Category:Test]] should be ignored. [[Real Link]] stays."
    result = extract_links(text)
    assert result == [{'page': 'Real Link', 'section': None}]


def test_extract_links_multiple():
    text = """[[Europa Universalis V]] has [[#Institutions|institutions]]
    and [[#advances|advances]]."""
    result = extract_links(text)
    assert len(result) == 3
    assert result[0] == {'page': 'Europa Universalis V', 'section': None}
    assert result[1] == {'page': None, 'section': 'Institutions'}
    assert result[2] == {'page': None, 'section': 'advances'}


def test_extract_links_empty():
    assert extract_links("") == []
    assert extract_links(None) == []


def test_extract_toc_ages_page():
    result = extract_toc(AGES_PAGE_TRUNCATED)
    assert len(result) == 3
    assert result[0] == {'level': 2, 'title': 'List of ages'}
    assert result[1] == {'level': 2, 'title': 'Institutions'}
    assert result[2] == {'level': 3, 'title': 'Age of Traditions'}


def test_extract_toc_empty():
    assert extract_toc("") == []
    assert extract_toc(None) == []
