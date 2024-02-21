import fire
from vespa.application import Vespa
from vespa.io import VespaResponse
from vespa.package import (HNSW, ApplicationPackage, Component, Document,
                           Field, FieldSet, FirstPhaseRanking, Function,
                           Parameter, RankProfile, Schema)
from documents import CSVLoader
from pydantic import BaseModel
from typing import List


def make_schema(name):
    indexing_opts = ["summary", "index"]

    return Schema(
        name=name,
        mode="streaming",
        document=Document(
            fields=[
                Field(name="id", type="string",
                      indexing=indexing_opts),
                Field(name="name", type="string",
                      indexing=indexing_opts),
                Field(name="metadata", type="map<string,string>",
                      indexing=indexing_opts),
                Field(name="chunks", type="array<string>",
                      indexing=indexing_opts),
                Field(name="embedding", type="tensor<bfloat16>(chunk{}, x[384])",
                      indexing=["input chunks", "embed e5",
                                "attribute", "index"],
                      ann=HNSW(distance_metric="angular"),
                      is_document_field=False
                      )
            ],
        ),
        fieldsets=[
            FieldSet(name="default", fields=["chunks", "name"])
        ]
    )


def make_vespa_application_package(name, schema):

    return ApplicationPackage(
        name=name,
        schema=[schema],
        components=[Component(id="e5", type="hugging-face-embedder",
                              parameters=[
                                  Parameter("transformer-model", {
                                      "url": "https://github.com/vespa-engine/sample-apps/raw/master/simple-semantic-search/model/e5-small-v2-int8.onnx"}),
                                  Parameter(
                                      "tokenizer-model", {"url": "https://raw.githubusercontent.com/vespa-engine/sample-apps/master/simple-semantic-search/model/tokenizer.json"})
                              ]
                              )]
    )


def make_rank_profile():
    return RankProfile(
        name="hybrid",
        inputs=[("query(q)", "tensor<float>(x[384])")],
        functions=[Function(
            name="similarities",
            expression="cosine_similarity(query(q), attribute(embedding),x)"
        )],
        first_phase=FirstPhaseRanking(
            expression="nativeRank(name) + nativeRank(chunks) + reduce(similarities, max, chunk)",
            rank_score_drop_limit=0.0
        ),
        match_features=["closest(embedding)", "similarities", "nativeRank(chunks)",
                        "nativeRank(name)", "elementSimilarity(chunks)"]
    )


def _indexing_callback(response: VespaResponse, id: str):
    if not response.is_successful():
        print(
            f"Document {id} failed to feed with status code {response.status_code}, url={response.url} response={response.json}")


class Main:
    @staticmethod
    def export_app(export_path):
        vespa_app_name = "csvrag"
        schema = make_schema(vespa_app_name)
        schema.add_rank_profile(make_rank_profile())
        application_package = make_vespa_application_package(
            vespa_app_name, schema)
        application_package.to_files(export_path)

    @staticmethod
    def add_documents_from_csv(path, name_col="title", text_col="content", sep="\t"):
        app = Vespa(url="http://localhost", port=8080)
        loader = CSVLoader()
        documents_feed = loader.get_documents_feed(
            path, name_col, text_col, sep)
        app.feed_iterable(schema="csvrag", iter=documents_feed,
                          namespace="personal", callback=_indexing_callback)


if __name__ == "__main__":
    fire.Fire(Main())
