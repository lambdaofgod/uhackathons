import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any, Union
from sklearn.base import BaseEstimator, TransformerMixin


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
