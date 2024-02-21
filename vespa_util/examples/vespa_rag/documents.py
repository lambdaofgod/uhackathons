import hashlib
import unicodedata

import tqdm
from typing import Any
import pandas as pd
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from vespa.package import Parameter, RankProfile, Schema
from pydantic import BaseModel
from langchain_core.documents import Document as LDocument


def get_text_splitter(chunk_size=1024, chunk_overlap=0):
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,  # chars, not llm tokens
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )


def remove_control_characters(s):
    return "".join(ch for ch in s if unicodedata.category(ch)[0] != "C")


class CSVLoader(BaseModel):
    text_splitter: Any = get_text_splitter()

    def get_documents_feed(self, path, name_col, text_col, sep, user_name="admin"):
        docs = self.get_documents_iter(path, name_col, text_col, sep)
        for doc in tqdm.tqdm(docs):
            yield {
                "fields": doc,
                "id": doc["id"],
                "groupname": user_name
            }

    def get_documents_iter(self, path, name_col, text_col, sep):
        df = pd.read_csv(path, sep=sep)
        for (i, doc_row) in df.iterrows():
            yield self._make_doc(doc_row.to_dict(), name_col, text_col)

    def _make_doc(self, raw_doc: dict, name_col: str, text_col: str):
        name = raw_doc.pop(name_col)
        text = raw_doc.pop(text_col)
        hash_value = hashlib.sha1(name.encode()).hexdigest()
        text_chunks = self._get_text_chunks(text)
        return {
            "id": hash_value,
            "name": name,
            "chunks": text_chunks,
            "metadata": raw_doc
        }

    def _get_text_chunks(self, text):
        chunks = self.text_splitter.transform_documents(
            [LDocument(page_content=text)])
        text_chunks = [chunk.page_content for chunk in chunks]
        return [
            remove_control_characters(chunk)
            for chunk in text_chunks
        ]
