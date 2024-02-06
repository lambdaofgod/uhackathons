# fastrag_util

Utilities for working with [fastRAG](https://github.com/IntelLabs/fastRAG)

## Dependencies

fastRAG contains its own ColBERT in a subproject, because of which we use it as a git submodule.

## How to - ColBERT + PLAID

Create doc collection -> Build and save index -> deploy search 

``` python
poetry run python create_doc_collection.py $CORPUS_DF_PATH $TEXT_COL --n_docs=1000 --title_column=$TITLE_COL
```

``` python
poetry run python create_plaid_index.py --checkpoint Intel/ColBERT-NQ --collection doc_coll.tsv --index-save-path plaid_index --gpus 1 --doc-max-length 256
```

``` python
poetry run python plaid_app.py 
```

