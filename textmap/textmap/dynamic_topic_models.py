import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Union
from bertopic import BERTopic


class DynamicTopicModel:
    """
    A wrapper for BERTopic that provides a scikit-learn-like API for dynamic topic modeling.
    """

    def __init__(
        self,
        num_topics: Optional[int] = None,
        text_col: str = "text",
        time_col: str = "creation_date",
        bertopic_model: Optional[BERTopic] = None,
        verbose: bool = True,
        representation_model=None,
        **bertopic_kwargs,
    ):
        """
        Initializes the DynamicTopicModel using BERTopic.

        Args:
            num_topics: Number of topics to extract. If None, BERTopic will determine automatically.
            text_col: The name of the column in the DataFrame containing the text data.
            time_col: The name of the column in the DataFrame indicating the timestamp.
            bertopic_model: An optional pre-trained instance of BERTopic.
            verbose: Whether to display progress information during model training.
            representation_model: An optional representation model for BERTopic.
            **bertopic_kwargs: Additional keyword arguments to pass to BERTopic.
        """
        self.bertopic_model = bertopic_model
        self.text_col = text_col
        self.time_col = time_col
        self.num_topics = num_topics
        self.verbose = verbose
        self.representation_model = representation_model
        self.bertopic_kwargs = bertopic_kwargs
        self.topics_over_time = None

    def fit(self, df: pd.DataFrame, nr_bins: int = 20):
        """
        Fits the dynamic topic model using BERTopic.

        Args:
            df: A pandas DataFrame with columns for text and timestamp.
            nr_bins: Number of time bins to use for the topics over time visualization.

        Returns:
            Self for method chaining.
        """
        # Debug information
        print(f"DataFrame shape: {df.shape}")
        print(f"DataFrame columns: {list(df.columns)}")
        print(
            f"Looking for text column: '{self.text_col}' and time column: '{self.time_col}'"
        )

        # Check if columns exist
        if self.text_col not in df.columns:
            raise ValueError(
                f"Text column '{self.text_col}' not found in DataFrame. Available columns: {list(df.columns)}"
            )
        if self.time_col not in df.columns and not self.time_col is None:
            raise ValueError(
                f"Time column '{self.time_col}' not found in DataFrame. Available columns: {list(df.columns)}"
            )

        # Initialize BERTopic model if not provided
        if self.bertopic_model is None:
            kwargs = self.bertopic_kwargs.copy()
            if self.representation_model is not None:
                kwargs["representation_model"] = self.representation_model
            self.bertopic_model = BERTopic(
                nr_topics=self.num_topics, verbose=self.verbose, **kwargs
            )
        try:
            # Extract text and timestamps
            texts = df[self.text_col].tolist()

            # Fit the BERTopic model
            print("Training BERTopic model...")
            topics, probs = self.bertopic_model.fit_transform(texts)
            print("BERTopic model training complete.")

            if self.time_col is not None:

                timestamps = df[self.time_col].tolist()
                # Generate topics over time
                print(f"Generating topics over time with {nr_bins} bins...")
                self.topics_over_time = self.bertopic_model.topics_over_time(
                    texts, timestamps, nr_bins=nr_bins
                )
                print("Topics over time generation complete.")

        except Exception as e:
            import traceback

            error_trace = traceback.format_exc()
            raise RuntimeError(f"Failed to train BERTopic model: {e}\n{error_trace}")

        return self

    def transform(self, df: pd.DataFrame) -> Tuple[List[int], List[List[float]]]:
        """
        Transforms the documents in the DataFrame into their topic distributions.

        Args:
            df: A pandas DataFrame with a column for text.

        Returns:
            A tuple of (topics, probabilities) where:
            - topics is a list of the most likely topic for each document
            - probabilities is a list of probability distributions over all topics for each document
        """
        if self.bertopic_model is None:
            raise RuntimeError(
                "The BERTopic model is not initialized. Call fit() first or provide a model."
            )
        if self.text_col not in df.columns:
            raise ValueError(f"Text column '{self.text_col}' not found in DataFrame.")

        # Transform documents using BERTopic
        try:
            texts = df[self.text_col].tolist()
            topics, probs = self.bertopic_model.transform(texts)
            return topics, probs
        except Exception as e:
            import traceback

            error_trace = traceback.format_exc()
            raise RuntimeError(f"Failed to transform documents: {e}\n{error_trace}")

    def get_topics(self, top_n_topics: int = 10) -> pd.DataFrame:
        """
        Get the topics from the BERTopic model.

        Args:
            top_n_topics: Number of top topics to include.

        Returns:
            DataFrame with topic information.
        """
        if self.bertopic_model is None:
            raise RuntimeError(
                "The BERTopic model is not initialized. Call fit() first or provide a model."
            )

        try:
            # Get topic information from BERTopic
            topic_info = self.bertopic_model.get_topic_info()

            # Filter to top N topics (excluding -1 which is the outlier topic)
            filtered_topics = topic_info[topic_info["Topic"] != -1].head(top_n_topics)

            return filtered_topics
        except Exception as e:
            import traceback

            error_trace = traceback.format_exc()
            raise RuntimeError(f"Failed to get topics: {e}\n{error_trace}")

    def get_topics_over_time(self) -> pd.DataFrame:
        """
        Get the topics over time data.

        Returns:
            DataFrame with columns: 'Topic', 'Words', 'Frequency', 'Timestamp'
        """
        if self.topics_over_time is None:
            raise RuntimeError("Topics over time not available. Call fit() first.")

        return self.topics_over_time

    def visualize_topics(self, **kwargs):
        """
        Visualize the topics using BERTopic's visualization.

        Args:
            **kwargs: Additional keyword arguments to pass to BERTopic's visualize_topics.

        Returns:
            A visualization of the topics.
        """
        if self.bertopic_model is None:
            raise RuntimeError(
                "The BERTopic model is not initialized. Call fit() first or provide a model."
            )

        return self.bertopic_model.visualize_topics(**kwargs)

    def visualize_topics_over_time(self, top_n_topics: int = 20, **kwargs):
        """
        Visualize the topics over time using BERTopic's visualization.

        Args:
            top_n_topics: Number of top topics to include in the visualization.
            **kwargs: Additional keyword arguments to pass to BERTopic's visualize_topics_over_time.

        Returns:
            A visualization of the topics over time.
        """
        if self.bertopic_model is None or self.topics_over_time is None:
            raise RuntimeError(
                "The BERTopic model or topics over time not initialized. Call fit() first."
            )

        return self.bertopic_model.visualize_topics_over_time(
            self.topics_over_time, top_n_topics=top_n_topics, **kwargs
        )
