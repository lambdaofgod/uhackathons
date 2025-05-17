"""Performance testing for textmap models."""

import cProfile
import pstats
import io
import numpy as np
from textmap.vectorizers import DocVectorizer
from textmap.transformers import LdaSeqTransformer
import tempfile
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def profile_function(func, *args, **kwargs):
    """Profile a function and return the stats.
    
    Parameters
    ----------
    func : callable
        Function to profile
    *args, **kwargs
        Arguments to pass to the function
        
    Returns
    -------
    pstats.Stats
        Profile statistics
    """
    pr = cProfile.Profile()
    pr.enable()
    result = func(*args, **kwargs)
    pr.disable()
    
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats(30)  # Print top 30 functions by cumulative time
    
    logger.info(s.getvalue())
    return result, ps


def generate_test_data(n_docs=100, vocab_size=1000, doc_length=50, n_time_slices=5):
    """Generate synthetic data for testing.
    
    Parameters
    ----------
    n_docs : int
        Number of documents
    vocab_size : int
        Size of vocabulary
    doc_length : int
        Average document length
    n_time_slices : int
        Number of time slices
        
    Returns
    -------
    list
        List of documents
    list
        List of time slice indices
    """
    # Generate random documents
    docs = []
    time_slices = []
    
    words = [f"word_{i}" for i in range(vocab_size)]
    
    docs_per_slice = n_docs // n_time_slices
    
    for t in range(n_time_slices):
        # Create documents for this time slice
        for _ in range(docs_per_slice):
            # Generate a random document
            doc_words = np.random.choice(words, size=doc_length)
            doc = " ".join(doc_words)
            docs.append(doc)
            time_slices.append(t)
    
    return docs, time_slices


def test_ldaseq_performance(n_docs=200, vocab_size=1000, doc_length=50, n_time_slices=5, 
                           n_topics=10, n_iter=5):
    """Test the performance of LdaSeqTransformer.
    
    Parameters
    ----------
    n_docs : int
        Number of documents
    vocab_size : int
        Size of vocabulary
    doc_length : int
        Average document length
    n_time_slices : int
        Number of time slices
    n_topics : int
        Number of topics
    n_iter : int
        Number of iterations
        
    Returns
    -------
    tuple
        (transformer, vectorizer, X) - the trained model, vectorizer, and transformed data
    """
    logger.info(f"Generating test data with {n_docs} documents, {vocab_size} vocabulary size")
    docs, time_slices = generate_test_data(n_docs, vocab_size, doc_length, n_time_slices)
    
    # Vectorize documents
    logger.info("Vectorizing documents")
    vectorizer = DocVectorizer(min_df=1, max_df=1.0)
    X = vectorizer.fit_transform(docs)
    
    # Create and fit LdaSeqTransformer
    logger.info(f"Creating LdaSeqTransformer with {n_topics} topics, {n_time_slices} time slices")
    transformer = LdaSeqTransformer(
        n_components=n_topics,
        time_slice=time_slices,
        n_iter=n_iter,
        random_state=42,
        initialize='random',
        em_min_iter=2,
        em_max_iter=4,
        verbose=1
    )
    
    # Profile fit_transform
    logger.info("Profiling fit_transform")
    X_transformed, _ = profile_function(transformer.fit_transform, X)
    
    # Profile get_topics
    logger.info("Profiling get_topics")
    topics, _ = profile_function(transformer.get_topics, 10)
    
    return transformer, vectorizer, X_transformed


def main():
    """Run performance tests."""
    # Test with small dataset
    logger.info("Running performance test with small dataset")
    test_ldaseq_performance(n_docs=100, vocab_size=500, n_topics=5, n_iter=3)
    
    # Test with larger dataset if needed
    logger.info("Running performance test with larger dataset")
    test_ldaseq_performance(n_docs=200, vocab_size=1000, n_topics=10, n_iter=3)


if __name__ == "__main__":
    main()
