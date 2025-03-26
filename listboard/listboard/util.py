from sentence_transformers import SentenceTransformer
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
from typing import Optional, List, Union


class SentenceTransformerEncoder(BaseEstimator, TransformerMixin):
    """
    A scikit-learn transformer wrapper for sentence-transformers models.
    
    Parameters:
    -----------
    model_name : str, default='all-MiniLM-L6-v2'
        The name of the sentence-transformers model to use.
    device : str, default=None
        Device to use for encoding ('cpu', 'cuda', etc.). If None, uses the default device.
    batch_size : int, default=32
        Batch size for encoding.
    show_progress : bool, default=True
        Whether to show a progress bar during encoding.
    """
    
    def __init__(
        self, 
        model_name: str = 'all-MiniLM-L6-v2', 
        device: Optional[str] = None,
        batch_size: int = 32,
        show_progress: bool = True
    ):
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.show_progress = show_progress
        self.model = None
    
    def fit(self, X: Union[List[str], np.ndarray], y=None):
        """
        Initialize the sentence-transformer model.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples,)
            The input texts.
        y : None
            Ignored.
            
        Returns:
        --------
        self : object
            Returns self.
        """
        self.model = SentenceTransformer(self.model_name, device=self.device)
        return self
    
    def transform(self, X: Union[List[str], np.ndarray]) -> np.ndarray:
        """
        Transform the input texts into embeddings.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples,)
            The input texts.
            
        Returns:
        --------
        X_transformed : ndarray of shape (n_samples, n_features)
            The text embeddings.
        """
        if self.model is None:
            self.fit(X)
        
        # Convert embeddings to numpy array
        embeddings = self.model.encode(
            X, 
            batch_size=self.batch_size,
            show_progress_bar=self.show_progress
        )
        return embeddings
