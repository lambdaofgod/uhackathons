from vespa.application import Vespa
from vespa.io import VespaResponse
from vespa.package import (HNSW, ApplicationPackage, Component, Document,
                           Field, FieldSet, FirstPhaseRanking, SecondPhaseRanking, Function,
                           Parameter, RankProfile, Schema)
from documents import CSVLoader
from pydantic import BaseModel
from typing import List


class AppBuilder(BaseModel):
    application_package: ApplicationPackage
    multichunk_documents: bool

    @classmethod
    def build_app(cls, vespa_app_name: str):
        schema = cls.make_schema(vespa_app_name)
        schema.add_rank_profile(cls.make_rank_profile())
        application_package = cls.setup_vespa_application_package(
            vespa_app_name, schema)
        return cls(application_package=application_package)

    class Config:
        arbitrary_types_allowed = True


class EmbeddingsAppBuilder(AppBuilder):
    multichunk_documents: bool = True

    @classmethod
    def make_schema(cls, name):
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

    @classmethod
    def setup_vespa_application_package(cls, name, schema):

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

    @classmethod
    def make_rank_profile(cls):
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


class ColbertAppBuilder(AppBuilder):
    multichunk_documents: bool = False

    @classmethod
    def make_schema(cls, name):
        return Schema(
            name=name,
            mode="streaming",
            document=Document(
                fields=[
                    Field(name="id", type="string", indexing=["summary"]),
                    Field(name="name", type="string",
                          indexing=["summary", "index"]),
                    Field(name="metadata", type="map<string,string>",
                          indexing=["summary", "index"]),
                    Field(name="chunkno", type="int",
                          indexing=["summary", "attribute"]),
                    Field(name="chunk", type="string",
                          indexing=["summary", "index"]),
                    Field(name="embedding", type="tensor<bfloat16>(x[384])",
                          indexing=[
                              '"passage: " . (input name || "") . " " . (input chunk || "")', "embed e5", "attribute"],
                          attribute=["distance-metric: angular"],
                          is_document_field=False
                          ),
                    Field(name="colbert", type="tensor<int8>(dt{}, x[16])",
                          indexing=[
                              '(input name || "") . " " . (input chunk || "")', "embed colbert", "attribute"],
                          is_document_field=False
                          )
                ],
            ),
            fieldsets=[
                FieldSet(name="default", fields=["name", "chunk"])
            ]
        )

    @classmethod
    def setup_vespa_application_package(cls, name, schema):
        return ApplicationPackage(
            name=name,
            schema=[schema],
            components=[
                Component(id="e5", type="hugging-face-embedder",
                          parameters=[
                              Parameter("transformer-model", {
                                  "url": "https://huggingface.co/intfloat/e5-small-v2/resolve/main/model.onnx"}),
                              Parameter(
                                  "tokenizer-model", {"url": "https://huggingface.co/intfloat/e5-small-v2/raw/main/tokenizer.json"})
                          ]
                          ),
                Component(id="colbert", type="colbert-embedder",
                          parameters=[
                              Parameter("transformer-model", {
                                  "url": "https://huggingface.co/colbert-ir/colbertv2.0/resolve/main/model.onnx"}),
                              Parameter(
                                  "tokenizer-model", {"url": "https://huggingface.co/colbert-ir/colbertv2.0/raw/main/tokenizer.json"})
                          ]
                          )
            ]
        )

    @classmethod
    def make_rank_profile(cls):
        return RankProfile(
            name="colbert",
            inputs=[
                ("query(q)", "tensor<float>(x[384])"),
                ("query(qt)", "tensor<float>(qt{}, x[128])")
            ],
            functions=[
                Function(
                    name="unpack",
                    expression="unpack_bits(attribute(colbert))"
                ),
                Function(
                    name="cos_sim",
                    expression="closeness(field, embedding)"
                ),
                Function(
                    name="max_sim",
                    expression="""
                        sum(
                            reduce(
                                sum(
                                    query(qt) * unpack() , x
                                ),
                                max, dt
                            ),
                            qt
                        )
                    """
                )
            ],
            first_phase=FirstPhaseRanking(
                expression="cos_sim"
            ),
            second_phase=SecondPhaseRanking(
                expression="max_sim"
            ),
            match_features=["max_sim", "cos_sim"]
        )
