import hashlib
import unicodedata

import tqdm
from typing import Any
import pandas as pd
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
    multichunk_documents: bool = False

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
            for doc in self._make_docs(doc_row.to_dict(), name_col, text_col):
                yield doc

    def _make_docs(self, raw_doc: dict, name_col: str, text_col: str):
        if self.multichunk_documents:
            yield self._make_multichunk_doc(raw_doc, name_col, text_col)
        else:
            for chunked_doc in self._make_chunk_docs(raw_doc, name_col, text_col):
                yield chunked_doc

    def _make_multichunk_doc(self, raw_doc: dict, name_col: str, text_col: str):
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

    def _make_chunk_docs(self, raw_doc: dict, name_col: str, text_col: str):
        name = raw_doc.pop(name_col)
        text = raw_doc.pop(text_col)
        text_chunks = self._get_text_chunks(text)
        for chunk in text_chunks:
            hash_value = hashlib.sha1((name + chunk).encode()).hexdigest()
            yield {
                "id": hash_value,
                "name": name,
                "chunk": chunk,
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
