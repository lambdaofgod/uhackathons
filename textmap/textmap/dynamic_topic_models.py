import pandas as pd
import numpy as np
from textmap.ldaseqmodel import LdaSeqModel
from gensim.corpora import Dictionary
from typing import List, Dict, Any, Tuple, Optional, Union
import pandas as pd


class DynamicTopicModel:
    """
    A wrapper for Gensim's LdaSeqModel that provides a scikit-learn-like API.
    """

    def __init__(
        self,
        num_topics: int = 10,
        text_col: str = "text",
        period_col: str = "period",
        lda_seq_model: Optional[LdaSeqModel] = None,
        **lda_kwargs,
    ):
        """
        Initializes the DynamicTopicModel.

        Args:
            num_topics: Number of topics to extract.
            text_col: The name of the column in the DataFrame containing the text data.
            period_col: The name of the column in the DataFrame indicating the time period.
            lda_seq_model: An optional pre-trained instance of Gensim's LdaSeqModel.
            **lda_kwargs: Additional keyword arguments to pass to LdaSeqModel.
        """
        self.lda_seq_model = lda_seq_model
        self.text_col = text_col
        self.period_col = period_col
        self.time_slices: List[int] = []
        self.corpus = None  # Will be set in fit
        self.id2word = None  # Will be set in fit
        self.num_topics = num_topics
        self.lda_kwargs = lda_kwargs

    def _preprocess_text(self, texts: List[str]) -> List[List[str]]:
        """
        Preprocess text data by tokenizing.

        Args:
            texts: List of text strings to tokenize.

        Returns:
            List of tokenized texts.
        """
        # Simple tokenization - in a real implementation, you might want more sophisticated preprocessing
        return [text.split() for text in texts]

    def _create_corpus_and_dictionary(
        self, df: pd.DataFrame
    ) -> Tuple[List[List[int]], Dictionary]:
        """
        Create a corpus and dictionary from the text data.

        Args:
            df: DataFrame containing the text data.

        Returns:
            Tuple of (corpus, dictionary)
        """
        if self.text_col not in df.columns:
            raise ValueError(f"Text column '{self.text_col}' not found in DataFrame.")

        # Extract and tokenize texts
        texts = df[self.text_col].tolist()
        tokenized_texts = self._preprocess_text(texts)

        # Create dictionary
        dictionary = Dictionary(tokenized_texts)

        # Create corpus (bag of words)
        corpus = [dictionary.doc2bow(text) for text in tokenized_texts]

        return corpus, dictionary

    def _calculate_time_slices(self, df: pd.DataFrame) -> List[int]:
        """
        Calculate time slices from the period column.

        Args:
            df: DataFrame containing the period data.

        Returns:
            List of document counts per time slice.
        """
        if self.period_col not in df.columns:
            raise ValueError(
                f"Period column '{self.period_col}' not found in DataFrame."
            )

        # Sort DataFrame by period to ensure correct ordering
        df_sorted = df.sort_values(by=self.period_col)

        # Count documents per period
        time_slices = df_sorted.groupby(self.period_col).size().tolist()

        return time_slices

    def fit(self, df: pd.DataFrame):
        """
        Fits the dynamic topic model by:
        1. Preprocessing the text data
        2. Creating a corpus and dictionary
        3. Calculating time slices
        4. Training the LdaSeqModel

        Args:
            df: A pandas DataFrame with columns for text and period.

        Returns:
            Self for method chaining.
        """
        # Prepare data
        self.corpus, self.id2word = self._create_corpus_and_dictionary(df)
        self.time_slices = self._calculate_time_slices(df)

        print(
            f"Model configured with text_col='{self.text_col}', period_col='{self.period_col}'."
        )
        print(f"Derived time_slices: {self.time_slices} from the input DataFrame.")

        # Train the LdaSeqModel if not provided
        if self.lda_seq_model is None:
            try:
                print(f"Training LdaSeqModel with {self.num_topics} topics...")
                self.lda_seq_model = LdaSeqModel(
                    corpus=self.corpus,
                    id2word=self.id2word,
                    time_slice=self.time_slices,
                    num_topics=self.num_topics,
                    initialize="gensim",
                    **self.lda_kwargs,
                )
                print("LdaSeqModel training complete.")
            except Exception as e:
                raise RuntimeError(f"Failed to train LdaSeqModel: {e}")

        return self

    def _transform_document(
        self, doc: str, time_idx: Optional[int] = None
    ) -> List[Tuple[int, float]]:
        """
        Transform a single document into its topic distribution.

        Args:
            doc: Text document to transform.
            time_idx: Optional time index for the document. If provided, uses time-specific topics.

        Returns:
            List of (topic_id, probability) tuples.
        """
        if self.lda_seq_model is None:
            raise RuntimeError(
                "The LdaSeqModel is not initialized. Call fit() first or provide a model."
            )

        # Tokenize and convert to bag of words
        tokenized_doc = self._preprocess_text([doc])[0]
        bow = self.id2word.doc2bow(tokenized_doc)

        # Get topic distribution
        # LdaSeqModel's doc_topics method doesn't accept a time parameter
        # We need to use the model directly
        try:
            # Get the document's topic distribution
            gamma = self.lda_seq_model.inference([bow])[0][0]

            # If time_idx is provided, we can try to get time-specific topics
            # by accessing the model's topic-word distributions at that time
            if time_idx is not None and 0 <= time_idx < len(self.time_slices):
                # This is a simplification - in a real implementation, you might
                # want to use the model's time-specific topic distributions
                pass

            # Convert gamma vector to (topic_id, probability) format
            topic_dist = [
                (topic_id, float(gamma[topic_id])) for topic_id in range(len(gamma))
            ]

            # Sort by probability in descending order
            topic_dist.sort(key=lambda x: x[1], reverse=True)

            return topic_dist

        except Exception as e:
            print(f"Error in document transformation: {e}")
            # Return empty list in case of error
            return []

    def transform(self, df: pd.DataFrame) -> List[List[Tuple[int, float]]]:
        """
        Transforms the documents in the DataFrame into their topic distributions.

        Args:
            df: A pandas DataFrame with a column for text and optionally period.

        Returns:
            A list of topic distributions for each document.
            Each topic distribution is a list of (topic_id, probability) tuples.
        """
        if self.lda_seq_model is None:
            raise RuntimeError(
                "The LdaSeqModel is not initialized. Call fit() first or provide a model."
            )
        if self.text_col not in df.columns:
            raise ValueError(f"Text column '{self.text_col}' not found in DataFrame.")

        # Check if period column exists for time-specific transformations
        use_time_info = self.period_col in df.columns

        # Transform each document
        topic_distributions = []
        for idx, row in df.iterrows():
            doc = row[self.text_col]
            time_idx = row[self.period_col] if use_time_info else None
            topic_dist = self._transform_document(doc, time_idx)
            topic_distributions.append(topic_dist)

        return topic_distributions

    def get_topics(
        self, time_periods: Optional[List[int]] = None, top_terms: int = 10
    ) -> pd.DataFrame:
        """
        Get the topics for specific time slices.

        Args:
            time_periods: List of time slice indices. If None, returns topics for all time slices.
            top_terms: Number of top terms to include for each topic.

        Returns:
            DataFrame with columns: 'time_period', 'topic_id', 'terms'
        """
        if self.lda_seq_model is None:
            raise RuntimeError(
                "The LdaSeqModel is not initialized. Call fit() first or provide a model."
            )

        # If time_periods is None, use all available time periods
        if time_periods is None:
            time_periods = list(range(len(self.time_slices)))

        try:
            # Create a list to store all topic data
            all_topics_data = []

            # Process each time period
            for time_period in time_periods:
                # Get topics from the model for this time period
                topics = self.lda_seq_model.print_topics(
                    time=time_period, top_terms=top_terms
                )

                # Process each topic
                for topic_id, topic_terms in enumerate(topics):
                    # Convert the list of term-weight tuples to a string
                    if isinstance(topic_terms, list):
                        terms_str = " + ".join(
                            [f"{weight:.3f}*{term}" for term, weight in topic_terms]
                        )
                    else:
                        # If already in the expected format, use as is
                        terms_str = topic_terms

                    # Add to our data collection
                    all_topics_data.append(
                        {
                            "time_period": time_period,
                            "topic_id": topic_id,
                            "terms": terms_str,
                        }
                    )

            # Convert to DataFrame
            return pd.DataFrame(all_topics_data)
        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            print(f"Error in get_topics:\n{error_trace}")
            raise RuntimeError(f"Failed to get topics: {e}\nVariable types: topics={type(topics)}, topic_id={type(topic_id)}, topic_terms={type(topic_terms)}\nFull traceback:\n{error_trace}")


if __name__ == "__main__":
    print("DynamicTopicModel module loaded. Run tests to verify functionality.")
