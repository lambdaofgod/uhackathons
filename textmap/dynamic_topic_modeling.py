import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any, Union, Callable
from sklearn.base import BaseEstimator, TransformerMixin
import gensim
from gensim.models import LdaSeqModel
from gensim.corpora import Dictionary
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


class DynamicTopicModel(BaseEstimator, TransformerMixin):
    """
    A scikit-learn compatible wrapper for dynamic topic models.
    
    This class provides a standardized interface for working with dynamic topic models
    like those from gensim. It handles the preprocessing of dataframes and manages
    the time periods for the underlying model.
    
    Parameters:
    -----------
    model : object
        The underlying dynamic topic model (e.g., gensim.models.ldaseqmodel.LdaSeqModel)
    text_col : str
        The name of the column in the dataframe that contains the text data
    period_col : str
        The name of the column in the dataframe that contains the time period information
    id_col : Optional[str], default=None
        The name of the column in the dataframe that contains document IDs
    preprocess_fn : Optional[callable], default=None
        A function to preprocess the text data before passing it to the model
    """
    
    def __init__(
        self,
        model: Any,
        text_col: str,
        period_col: str,
        id_col: Optional[str] = None,
        preprocess_fn: Optional[callable] = None
    ):
        self.model = model
        self.text_col = text_col
        self.period_col = period_col
        self.id_col = id_col
        self.preprocess_fn = preprocess_fn
        self.is_fitted_ = False
        self.periods_ = None
        self.period_mapping_ = None
        self.document_ids_ = None
        
    def fit(self, X: pd.DataFrame, y=None):
        """
        Fit the dynamic topic model to the data.
        
        Parameters:
        -----------
        X : pd.DataFrame
            The input dataframe containing text and time period columns
        y : ignored
            Not used, present for API consistency by convention
            
        Returns:
        --------
        self : object
            Returns self
        """
        # Extract the time periods and create a mapping
        self.periods_ = sorted(X[self.period_col].unique())
        self.period_mapping_ = {period: i for i, period in enumerate(self.periods_)}
        
        # Preprocess the text if a preprocessing function is provided
        texts = X[self.text_col].values
        if self.preprocess_fn is not None:
            texts = [self.preprocess_fn(text) for text in texts]
        
        # Get the period indices for each document
        period_indices = X[self.period_col].map(self.period_mapping_).values
        
        # Store document IDs if provided
        if self.id_col is not None:
            self.document_ids_ = X[self.id_col].values
        else:
            self.document_ids_ = np.arange(len(X))
            
        # Fit the underlying model
        # Note: The exact implementation depends on the specific model being used
        # This is a generic implementation that assumes the model has a fit method
        # that takes texts and time slices as input
        self.model.fit(texts, period_indices)
        
        self.is_fitted_ = True
        return self
    
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Transform the input data to topic distributions.
        
        Parameters:
        -----------
        X : pd.DataFrame
            The input dataframe containing text and time period columns
            
        Returns:
        --------
        np.ndarray
            The topic distributions for each document
        """
        if not self.is_fitted_:
            raise ValueError("Model has not been fitted yet.")
        
        # Preprocess the text if a preprocessing function is provided
        texts = X[self.text_col].values
        if self.preprocess_fn is not None:
            texts = [self.preprocess_fn(text) for text in texts]
        
        # Get the period indices for each document
        period_indices = X[self.period_col].map(self.period_mapping_).values
        
        # Transform using the underlying model
        # Note: The exact implementation depends on the specific model being used
        # This is a generic implementation that assumes the model has a transform method
        return self.model.transform(texts, period_indices)
    
    def fit_transform(self, X: pd.DataFrame, y=None) -> np.ndarray:
        """
        Fit the model and transform the input data in one step.
        
        Parameters:
        -----------
        X : pd.DataFrame
            The input dataframe containing text and time period columns
        y : ignored
            Not used, present for API consistency by convention
            
        Returns:
        --------
        np.ndarray
            The topic distributions for each document
        """
        return self.fit(X).transform(X)
    
    def get_topic_terms(self, topic_id: int, period: Optional[Union[int, Any]] = None, top_n: int = 10) -> List[tuple]:
        """
        Get the top terms for a specific topic and time period.
        
        Parameters:
        -----------
        topic_id : int
            The ID of the topic
        period : Optional[Union[int, Any]], default=None
            The time period. If None, returns the global topic terms.
            Can be either the period index or the actual period value.
        top_n : int, default=10
            The number of top terms to return
            
        Returns:
        --------
        List[tuple]
            A list of (term, weight) tuples
        """
        if not self.is_fitted_:
            raise ValueError("Model has not been fitted yet.")
        
        # Convert period to index if it's not already an index
        if period is not None and not isinstance(period, int):
            period = self.period_mapping_.get(period)
            if period is None:
                raise ValueError(f"Period not found in the fitted data.")
        
        # Get the top terms from the underlying model
        # Note: The exact implementation depends on the specific model being used
        # This is a generic implementation that assumes the model has a get_topic_terms method
        return self.model.get_topic_terms(topic_id, period, top_n)
    
    def get_topic_evolution(self, topic_id: int, top_n: int = 10) -> Dict[Any, List[tuple]]:
        """
        Get the evolution of a topic over time.
        
        Parameters:
        -----------
        topic_id : int
            The ID of the topic
        top_n : int, default=10
            The number of top terms to return for each period
            
        Returns:
        --------
        Dict[Any, List[tuple]]
            A dictionary mapping periods to lists of (term, weight) tuples
        """
        if not self.is_fitted_:
            raise ValueError("Model has not been fitted yet.")
        
        result = {}
        for period, period_idx in self.period_mapping_.items():
            result[period] = self.get_topic_terms(topic_id, period_idx, top_n)
        
        return result
    
    def get_document_topics(self, doc_id: Union[int, Any]) -> Dict[int, float]:
        """
        Get the topic distribution for a specific document.
        
        Parameters:
        -----------
        doc_id : Union[int, Any]
            The ID of the document
            
        Returns:
        --------
        Dict[int, float]
            A dictionary mapping topic IDs to their probabilities
        """
        if not self.is_fitted_:
            raise ValueError("Model has not been fitted yet.")
        
        # Find the document index
        if isinstance(doc_id, int) and self.id_col is None:
            doc_idx = doc_id
        else:
            doc_idx = np.where(self.document_ids_ == doc_id)[0]
            if len(doc_idx) == 0:
                raise ValueError(f"Document ID {doc_id} not found.")
            doc_idx = doc_idx[0]
        
        # Get the topic distribution from the underlying model
        # Note: The exact implementation depends on the specific model being used
        # This is a generic implementation that assumes the model has a get_document_topics method
        return self.model.get_document_topics(doc_idx)
    
    @classmethod
    def create_gensim_dtm(
        cls,
        text_col: str,
        period_col: str,
        id_col: Optional[str] = None,
        preprocess_fn: Optional[Callable] = None,
        num_topics: int = 10,
        time_slice_length: Optional[int] = None,
        chain_variance: float = 0.005,
        passes: int = 10,
        random_state: int = 42,
        chunksize: int = 100,
        em_min_iter: int = 6,
        em_max_iter: int = 20
    ):
        """
        Create a DynamicTopicModel using gensim's LdaSeqModel implementation.
        
        Parameters:
        -----------
        text_col : str
            The name of the column in the dataframe that contains the text data
        period_col : str
            The name of the column in the dataframe that contains the time period information
        id_col : Optional[str], default=None
            The name of the column in the dataframe that contains document IDs
        preprocess_fn : Optional[callable], default=None
            A function to preprocess the text data before passing it to the model.
            Should convert text to a list of tokens.
        num_topics : int, default=10
            The number of topics to extract
        time_slice_length : Optional[int], default=None
            The number of documents in each time slice. If None, will be determined from data.
        chain_variance : float, default=0.005
            The variance of the normal distribution used for initialization of the topic chains
        passes : int, default=10
            Number of passes through the corpus during training
        random_state : int, default=42
            Random seed for reproducibility
        chunksize : int, default=100
            Number of documents to be used in each training chunk
        em_min_iter : int, default=6
            Minimum number of iterations for EM algorithm
        em_max_iter : int, default=20
            Maximum number of iterations for EM algorithm
            
        Returns:
        --------
        DynamicTopicModel
            A DynamicTopicModel instance with a gensim LdaSeqModel
        """
        class GensimDTMWrapper:
            """
            A wrapper for gensim's LdaSeqModel to provide a consistent interface.
            """
            def __init__(self, num_topics, chain_variance, passes, random_state, chunksize, em_min_iter, em_max_iter):
                self.num_topics = num_topics
                self.chain_variance = chain_variance
                self.passes = passes
                self.random_state = random_state
                self.chunksize = chunksize
                self.em_min_iter = em_min_iter
                self.em_max_iter = em_max_iter
                self.model = None
                self.dictionary = None
                self.corpus = None
                self.time_slices = None
                
            def fit(self, texts, time_slices):
                """
                Fit the LdaSeqModel to the data.
                
                Parameters:
                -----------
                texts : list
                    List of tokenized texts (each text is a list of tokens)
                time_slices : list
                    List of time slice indices for each document
                """
                # Create a dictionary and corpus
                self.dictionary = Dictionary(texts)
                self.dictionary.filter_extremes(no_below=5, no_above=0.5)
                self.corpus = [self.dictionary.doc2bow(text) for text in texts]
                
                # Count documents per time slice
                unique_slices = sorted(set(time_slices))
                self.time_slices = [time_slices.count(slice_idx) for slice_idx in unique_slices]
                
                # Train the model
                self.model = LdaSeqModel(
                    corpus=self.corpus,
                    id2word=self.dictionary,
                    time_slice=self.time_slices,
                    num_topics=self.num_topics,
                    chain_variance=self.chain_variance,
                    passes=self.passes,
                    random_state=self.random_state,
                    chunksize=self.chunksize,
                    em_min_iter=self.em_min_iter,
                    em_max_iter=self.em_max_iter
                )
                return self
                
            def transform(self, texts, time_slices):
                """
                Transform texts to topic distributions.
                
                Parameters:
                -----------
                texts : list
                    List of tokenized texts
                time_slices : list
                    List of time slice indices for each document
                    
                Returns:
                --------
                np.ndarray
                    Topic distributions for each document
                """
                if self.model is None:
                    raise ValueError("Model has not been fitted yet.")
                
                # Convert texts to bow format
                corpus = [self.dictionary.doc2bow(text) for text in texts]
                
                # Get topic distributions for each document
                result = np.zeros((len(corpus), self.num_topics))
                for i, (doc, time_idx) in enumerate(zip(corpus, time_slices)):
                    # Get topic distribution for this document at this time
                    topics = self.model.doc_topics(doc, time=time_idx)
                    for topic_id, prob in enumerate(topics):
                        result[i, topic_id] = prob
                
                return result
                
            def get_topic_terms(self, topic_id, time=None, top_n=10):
                """
                Get the top terms for a topic at a specific time.
                
                Parameters:
                -----------
                topic_id : int
                    The ID of the topic
                time : int, optional
                    The time slice index. If None, returns the global topic.
                top_n : int, default=10
                    Number of top terms to return
                    
                Returns:
                --------
                list
                    List of (term, weight) tuples
                """
                if self.model is None:
                    raise ValueError("Model has not been fitted yet.")
                
                # Get topic terms
                if time is None:
                    # For global topic, use the average over all time slices
                    topic_terms = []
                    for t in range(len(self.time_slices)):
                        topic_terms.extend(self.model.print_topic(topic_id, t, top_n))
                    
                    # Count term frequencies across all time slices
                    term_counts = {}
                    for term, _ in topic_terms:
                        if term in term_counts:
                            term_counts[term] += 1
                        else:
                            term_counts[term] = 1
                    
                    # Sort by frequency and return top_n
                    sorted_terms = sorted(term_counts.items(), key=lambda x: x[1], reverse=True)
                    return [(term, count / len(self.time_slices)) for term, count in sorted_terms[:top_n]]
                else:
                    # For specific time slice
                    return self.model.print_topic(topic_id, time, top_n)
                
            def get_document_topics(self, doc_idx):
                """
                Get topic distribution for a document.
                
                Parameters:
                -----------
                doc_idx : int
                    The index of the document
                    
                Returns:
                --------
                dict
                    Dictionary mapping topic IDs to probabilities
                """
                if self.model is None:
                    raise ValueError("Model has not been fitted yet.")
                
                # Get the document's time slice
                time_idx = 0
                doc_count = 0
                for i, count in enumerate(self.time_slices):
                    if doc_idx < doc_count + count:
                        time_idx = i
                        break
                    doc_count += count
                
                # Get topic distribution
                topics = self.model.doc_topics(self.corpus[doc_idx], time=time_idx)
                return {i: float(prob) for i, prob in enumerate(topics)}
        
        # Create the gensim wrapper
        gensim_model = GensimDTMWrapper(
            num_topics=num_topics,
            chain_variance=chain_variance,
            passes=passes,
            random_state=random_state,
            chunksize=chunksize,
            em_min_iter=em_min_iter,
            em_max_iter=em_max_iter
        )
        
        # Create and return the DynamicTopicModel
        return cls(
            model=gensim_model,
            text_col=text_col,
            period_col=period_col,
            id_col=id_col,
            preprocess_fn=preprocess_fn
        )
