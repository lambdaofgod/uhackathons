# fastrag_util

Utilities for working with [fastRAG](https://github.com/IntelLabs/fastRAG)

## Dependencies

fastRAG contains its own ColBERT in a subproject, because of which we use it as a git submodule.

## How to - ColBERT + PLAID

Create doc collection -> Build and save index -> deploy search 


### Run the pipeline

Create a config (based on conf/index_code_config.yaml) and put it in $CONF_PATH

```bash
poetry run python main.py create_plaid_index_with_config $CONF_PATH
```

After that you can run the server (it runs on 4321 by default)

```bash 
poetry run python plaid_app.py -c $CONF_PATH
```

### Separate steps

```bash 
poetry run python create_doc_collection.py $CORPUS_DF_PATH $TEXT_COL --n_docs=1000 --title_column=$TITLE_COL
```

```bash 
poetry run python create_plaid_index.py --checkpoint Intel/ColBERT-NQ --collection doc_coll.tsv --index-save-path plaid_index --gpus 1 --doc-max-length 256
```

## Querying

See querying.org for an example
