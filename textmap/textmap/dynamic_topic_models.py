import pandas as pd
import numpy as np
from gensim.models.ldaseqmodel import LdaSeqModel
from gensim.corpora import Dictionary
from typing import List, Dict, Any, Tuple, Optional

class DynamicTopicModel:
    """
    A wrapper for Gensim's LdaSeqModel that provides a scikit-learn-like API.
    """

    def __init__(self, num_topics: int = 10, text_col: str = "text", period_col: str = "period", 
                 lda_seq_model: Optional[LdaSeqModel] = None, **lda_kwargs):
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
    
    def _create_corpus_and_dictionary(self, df: pd.DataFrame) -> Tuple[List[List[int]], Dictionary]:
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
            raise ValueError(f"Period column '{self.period_col}' not found in DataFrame.")
        
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
        
        print(f"Model configured with text_col='{self.text_col}', period_col='{self.period_col}'.")
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
                    initialize='gensim',
                    **self.lda_kwargs
                )
                print("LdaSeqModel training complete.")
            except Exception as e:
                raise RuntimeError(f"Failed to train LdaSeqModel: {e}")
        
        return self

    def _transform_document(self, doc: str, time_idx: Optional[int] = None) -> List[Tuple[int, float]]:
        """
        Transform a single document into its topic distribution.
        
        Args:
            doc: Text document to transform.
            time_idx: Optional time index for the document. If provided, uses time-specific topics.
            
        Returns:
            List of (topic_id, probability) tuples.
        """
        if self.lda_seq_model is None:
            raise RuntimeError("The LdaSeqModel is not initialized. Call fit() first or provide a model.")
            
        # Tokenize and convert to bag of words
        tokenized_doc = self._preprocess_text([doc])[0]
        bow = self.id2word.doc2bow(tokenized_doc)
        
        # Get topic distribution
        # LdaSeqModel doesn't have get_document_topics method
        # Instead, we need to use doc_topics method which returns gamma matrix
        if time_idx is not None and 0 <= time_idx < len(self.time_slices):
            # For a specific time slice
            gamma = self.lda_seq_model.doc_topics(bow, time=time_idx)
        else:
            # Default to the first time slice if not specified
            gamma = self.lda_seq_model.doc_topics(bow, time=0)
        
        # Convert gamma vector to (topic_id, probability) format
        topic_dist = [(topic_id, float(gamma[topic_id])) for topic_id in range(len(gamma))]
        
        # Sort by probability in descending order
        topic_dist.sort(key=lambda x: x[1], reverse=True)
        
        return topic_dist
    
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
            raise RuntimeError("The LdaSeqModel is not initialized. Call fit() first or provide a model.")
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

    def get_topics(self, time: int = 0, top_terms: int = 10) -> List[Tuple[int, str]]:
        """
        Get the topics for a specific time slice.
        
        Args:
            time: Time slice index.
            top_terms: Number of top terms to include for each topic.
            
        Returns:
            List of (topic_id, terms_string) tuples.
        """
        if self.lda_seq_model is None:
            raise RuntimeError("The LdaSeqModel is not initialized. Call fit() first or provide a model.")
            
        try:
            return self.lda_seq_model.print_topics(time=time, top_terms=top_terms)
        except Exception as e:
            raise RuntimeError(f"Failed to get topics: {e}")

if __name__ == '__main__':
    # Example Usage with the updated implementation
    from gensim.test.utils import common_texts
    
    # 1. Prepare sample data for three time slices
    texts_t0 = [doc for doc in common_texts[:3]]
    texts_t1 = [doc for doc in common_texts[3:6]]
    texts_t2 = [doc for doc in common_texts[6:]]
    all_texts = texts_t0 + texts_t1 + texts_t2
    
    # Create a DataFrame
    periods = [0]*len(texts_t0) + [1]*len(texts_t1) + [2]*len(texts_t2)
    data = {'text': [" ".join(doc) for doc in all_texts], 'period': periods}
    sample_df = pd.DataFrame(data)
    
    # 2. Initialize and fit the DynamicTopicModel
    try:
        # Create model with 2 topics and parameters to improve convergence
        dtm = DynamicTopicModel(
            num_topics=2, 
            text_col='text', 
            period_col='period',
            chain_variance=0.005,  # Lower chain variance for more stable topics
            passes=20,             # More passes for better convergence
            em_min_iter=6,         # Minimum EM iterations
            em_max_iter=20         # Maximum EM iterations
        )
        
        # Fit the model (this will train the LdaSeqModel internally)
        dtm.fit(sample_df)
        
        # 3. Transform data to get topic distributions
        print("\nTransforming documents to get topic distributions...")
        topic_distributions = dtm.transform(sample_df)
        
        print(f"\nSample topic distribution for first document:")
        if topic_distributions and len(topic_distributions) > 0:
            # Print in a more readable format
            print(f"Document: '{sample_df.iloc[0][dtm.text_col]}'")
            print("Topic distribution:")
            for topic_id, prob in topic_distributions[0]:
                print(f"  Topic {topic_id}: {prob:.4f}")
        else:
            print("No topic distributions returned")
        
        # 4. Get topics for a specific time slice
        try:
            topics_at_time_0 = dtm.get_topics(time=0, top_terms=5)
            print("\nTopics at time=0:")
            for topic_idx, terms in topics_at_time_0:
                print(f"Topic {topic_idx}: {terms}")
        except Exception as e:
            print(f"Could not print topics: {e}")
            import traceback
            traceback.print_exc()
            
    except Exception as e:
        print(f"Error in example: {e}")
        print("This example requires a properly setup LdaSeqModel training environment.")
