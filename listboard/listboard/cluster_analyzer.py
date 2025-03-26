from pydantic import BaseModel, Field
import pandas as pd
from sklearn.base import ClusterMixin, TransformerMixin
from sklearn.pipeline import Pipeline, make_pipeline
import numpy as np
from typing import List, Dict
from scipy.spatial.distance import cdist


class TextClusterer(BaseModel):
    feature_pipeline: Pipeline
    clusterer: ClusterMixin

    @classmethod
    def create(cls, clusterer: ClusterMixin, feature_extractor: TransformerMixin):
        return TextClusterer(
            feature_pipeline=make_pipeline(feature_extractor), clusterer=clusterer
        )

    class Config:
        arbitrary_types_allowed = True


class TextClustererResult(BaseModel):
    features: np.array
    cluster_labels: np.array
    labels: np.array

    class Config:
        arbitrary_types_allowed = True


class TextColumnMerger(BaseModel):
    columns: List[str]
    text_joiner: str

    def get_texts(self, df: pd.DataFrame) -> pd.Series:
        def format_row(row):
            formatted_parts = []
            for col in self.columns:
                formatted_parts.append(f"{col}:\n{row[col]}")
            return self.text_joiner.join(formatted_parts)

        return df.apply(format_row, axis=1)


class TextClusterAnalyzer(BaseModel):
    df: pd.DataFrame
    text_clustering_pipeline: Pipeline
    text_clusterer_result: TextClustererResult
    text_merger: TextColumnMerger
    texts: pd.Series

    @classmethod
    def create(
        cls,
        df: pd.DataFrame,
        text_columns: List[str],
        clusterer: ClusterMixin,
        feature_extractor: TransformerMixin,
        text_joiner: str = "\n\n",
    ):
        # Create a TextClusterer instance
        text_clusterer = TextClusterer.create(clusterer, feature_extractor)

        merger = TextColumnMerger(columns=text_columns, text_joiner=text_joiner)
        # Join the text columns into a single text field
        texts = merger.get_texts(df)

        # Fit the feature pipeline to the texts and transform in one step
        features = text_clusterer.feature_pipeline.fit_transform(texts)

        # Fit the clusterer to the features
        cluster_labels = text_clusterer.clusterer.fit_transform(features)
        labels = text_clusterer.clusterer.labels_

        # Create the result object
        result = TextClustererResult(
            features=features, cluster_labels=cluster_labels, labels=labels
        )

        # Return a new TextClusterAnalyzer instance
        return cls(
            df=df.assign(extracted_text=texts),
            texts=texts,
            text_clustering_pipeline=make_pipeline(
                feature_extractor, clusterer
            ),  # For backward compatibility
            text_clusterer_result=result,
            text_merger=merger,
        )

    def _get_cluster_representatives_idxs(
        self, n_per_cluster: int = 1
    ) -> Dict[int, List[int]]:
        """
        Find the indices of examples closest to each cluster centroid.

        Args:
            n_per_cluster: Number of representative examples to return per cluster

        Returns:
            Dictionary mapping cluster IDs to lists of example indices
        """
        features = self.text_clusterer_result.features
        labels = self.text_clusterer_result.labels

        # Get unique cluster labels
        unique_clusters = np.unique(labels)
        representatives = {}

        for cluster_id in unique_clusters:
            # Get features for this cluster
            cluster_features = features[labels == cluster_id]

            if len(cluster_features) == 0:
                representatives[cluster_id] = []
                continue

            # Calculate centroid for this cluster
            centroid = np.mean(cluster_features, axis=0)

            # Calculate distances from each point to the centroid
            distances = cdist([centroid], cluster_features, metric="euclidean")[0]

            # Get indices of the closest n_per_cluster points
            closest_indices = np.argsort(distances)[:n_per_cluster]

            # Map these back to the original dataframe indices
            original_indices = np.where(labels == cluster_id)[0][closest_indices]

            representatives[cluster_id] = original_indices.tolist()

        return representatives

    def get_cluster_examples(self, n_per_cluster: int = 3) -> pd.DataFrame:
        """
        Get representative examples from each cluster.

        Args:
            n_per_cluster: Number of examples to return per cluster

        Returns:
            DataFrame with examples from each cluster, with a 'cluster_id' column
        """
        # Get representative indices for each cluster
        representatives = self._get_cluster_representatives_idxs(
            n_per_cluster=n_per_cluster
        )

        # Create a list to hold all example rows
        example_rows = []

        # For each cluster, add the representative examples to our list
        for cluster_id, indices in representatives.items():
            for idx in indices:
                # Get the row from the original dataframe
                row = self.df.iloc[idx].copy()

                # Add the cluster ID to the row
                row_dict = row.to_dict()
                row_dict["cluster_id"] = cluster_id

                example_rows.append(row_dict)

        # Create a new dataframe from all the example rows
        if example_rows:
            examples_df = pd.DataFrame(example_rows)
        else:
            # If no examples were found, return an empty dataframe with a cluster_id column
            examples_df = pd.DataFrame(columns=list(self.df.columns) + ["cluster_id"])

        return examples_df

    def refit_clusterer(self, clusterer: ClusterMixin) -> "TextClusterAnalyzer":
        """
        Refit the clustering model using the existing features.

        Args:
            clusterer: A new clustering model to fit to the data

        Returns:
            Self with updated clustering results
        """
        # Get the existing features
        features = self.text_clusterer_result.features

        # Fit the new clusterer to the features
        cluster_labels = clusterer.fit_transform(features)
        labels = clusterer.labels_

        # Create a new result object
        result = TextClustererResult(
            features=features, cluster_labels=cluster_labels, labels=labels
        )

        # Update the clusterer result
        self.text_clusterer_result = result

        # Update the pipeline with the new clusterer
        # This assumes the first step is the feature extractor
        feature_extractor = self.text_clustering_pipeline[0]
        self.text_clustering_pipeline = make_pipeline(feature_extractor, clusterer)

        return self

    class Config:
        arbitrary_types_allowed = True
