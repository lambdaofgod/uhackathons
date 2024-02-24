import fire
from vespa.application import Vespa
from vespa.io import VespaResponse
from vespa.package import (HNSW, ApplicationPackage, Component, Document,
                           Field, FieldSet, FirstPhaseRanking, Function,
                           Parameter, RankProfile, Schema)
from documents import CSVLoader
from pydantic import BaseModel
from typing import List, Literal
from app_builders import AppBuilder, EmbeddingsAppBuilder, ColbertAppBuilder
from vespa.deployment import VespaDocker


def make_app_builder(app_builder_type):
    if app_builder_type == "semantic":
        return EmbeddingsAppBuilder.build_app(app_builder_type)
    elif app_builder_type == "colbert":
        return ColbertAppBuilder.build_app(app_builder_type)


def _indexing_callback(response: VespaResponse, id: str):
    if not response.is_successful():
        print(
            f"Document {id} failed to feed with status code {response.status_code}, url={response.url} response={response.json}")


class Main:
    @staticmethod
    def export_app(export_path, app_builder_type: Literal["semantic", "colbert"] = "semantic"):
        app_builder = make_app_builder(
            app_builder_type)
        app_builder.application_package.to_files(export_path)

    @staticmethod
    def add_documents_from_csv(path, name_col="title", text_col="content", sep="\t", app_builder_type: Literal["semantic", "colbert"] = "semantic"):
        app = Vespa(url="http://localhost", port=8080)
        loader = CSVLoader(multichunk_documents=app_builder_type != "colbert")
        documents_feed = loader.get_documents_feed(
            path, name_col, text_col, sep)
        app.feed_iterable(schema=app_builder_type, iter=documents_feed,
                          namespace="personal", callback=_indexing_callback)

    @staticmethod
    def deploy_to_docker(app_name, app_files_path):
        vespa_docker = VespaDocker()
        app = vespa_docker.deploy_from_disk(app_name, app_files_path)


if __name__ == "__main__":
    fire.Fire(Main())
