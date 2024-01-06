import pytest
import pytantivy
import polars


@pytest.fixture
def tantivity_index():
    pytantivy.initialize_index("foo", "title", ["body"])


@pytest.fixture
def polars_df():
    df = polars.DataFrame(
        {
            "title": ["a", "b", "c"],
            "body": ["a text", "another text", "something"],
        }
    )
    return df


def test_initialization(tantivity_index):
    assert True


def test_indexing(tantivity_index):
    pytantivy.index_document("foo", {"title": "a title", "text": "a text"})


def test_search(tantivity_index):
    pytantivy.index_document("foo", {"title": "a title", "text": "a text"})
    query = "text"
    results = pytantivy.search("foo", query)
    assert results == ["a title"]


def test_indexing_with_polars(tantivity_index, polars_df):
    pytantivy.index_polars_dataframe("foo", polars_df)
    query = "text"
    results = pytantivy.search("foo", query)
    assert results == ["a", "b"]
