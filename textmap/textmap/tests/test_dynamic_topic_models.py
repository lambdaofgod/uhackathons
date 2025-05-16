import pytest
import pandas as pd
import numpy as np
from gensim.test.utils import common_texts
from ..dynamic_topic_models import DynamicTopicModel

@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    # Prepare sample data for three time slices
    texts_t0 = [doc for doc in common_texts[:3]]
    texts_t1 = [doc for doc in common_texts[3:6]]
    texts_t2 = [doc for doc in common_texts[6:]]
    all_texts = texts_t0 + texts_t1 + texts_t2
    
    # Create a DataFrame
    periods = [0]*len(texts_t0) + [1]*len(texts_t1) + [2]*len(texts_t2)
    data = {'text': [" ".join(doc) for doc in all_texts], 'period': periods}
    return pd.DataFrame(data)

@pytest.fixture
def model_params():
    """Define model parameters for testing."""
    return {
        'num_topics': 2,
        'chain_variance': 0.005,
        'passes': 5,
        'em_min_iter': 2,
        'em_max_iter': 5,
        'lda_inference_max_iter': 5
    }

def test_initialization(model_params):
    """Test model initialization."""
    dtm = DynamicTopicModel(**model_params)
    assert dtm.num_topics == 2
    assert dtm.text_col == "text"
    assert dtm.period_col == "period"
    assert dtm.lda_seq_model is None

def test_preprocessing(model_params):
    """Test text preprocessing."""
    dtm = DynamicTopicModel(**model_params)
    texts = ["this is a test", "another test document"]
    tokenized = dtm._preprocess_text(texts)
    
    assert len(tokenized) == 2
    assert tokenized[0] == ["this", "is", "a", "test"]
    assert tokenized[1] == ["another", "test", "document"]

def test_corpus_creation(model_params, sample_data):
    """Test corpus and dictionary creation."""
    dtm = DynamicTopicModel(**model_params)
    corpus, dictionary = dtm._create_corpus_and_dictionary(sample_data)
    
    assert corpus is not None
    assert dictionary is not None
    assert len(corpus) == len(sample_data)
    assert len(dictionary) > 0

def test_time_slices(model_params, sample_data):
    """Test time slice calculation."""
    dtm = DynamicTopicModel(**model_params)
    time_slices = dtm._calculate_time_slices(sample_data)
    
    assert len(time_slices) == 3  # We have 3 time periods
    assert sum(time_slices) == len(sample_data)  # Total should match df length

def test_fit_transform(model_params, sample_data):
    """Test fitting and transforming."""
    # Initialize model
    dtm = DynamicTopicModel(**model_params)
    
    # Fit the model
    dtm.fit(sample_data)
    
    # Check that model components are initialized
    assert dtm.lda_seq_model is not None
    assert dtm.corpus is not None
    assert dtm.id2word is not None
    assert len(dtm.time_slices) == 3
    
    # Transform documents
    topic_distributions = dtm.transform(sample_data)
    
    # Check output format
    assert len(topic_distributions) == len(sample_data)
    
    # Check first document's topic distribution
    first_doc_topics = topic_distributions[0]
    assert isinstance(first_doc_topics, list)
    
    if first_doc_topics:  # If not empty
        # Check topic distribution format
        assert isinstance(first_doc_topics[0], tuple)
        assert len(first_doc_topics[0]) == 2
        
        # Topic ID should be an integer
        assert isinstance(first_doc_topics[0][0], int)
        
        # Probability should be a float between 0 and 1
        assert isinstance(first_doc_topics[0][1], float)
        assert first_doc_topics[0][1] >= 0.0
        assert first_doc_topics[0][1] <= 1.0
        
        # Probabilities should sum to approximately 1
        total_prob = sum(prob for _, prob in first_doc_topics)
        assert abs(total_prob - 1.0) < 0.1
        
        # Topics should be sorted by probability (descending)
        for i in range(len(first_doc_topics) - 1):
            assert first_doc_topics[i][1] >= first_doc_topics[i + 1][1]

def test_get_topics(model_params, sample_data):
    """Test getting topics for a time slice."""
    # Initialize and fit model
    dtm = DynamicTopicModel(**model_params)
    dtm.fit(sample_data)
    
    # Get topics for time=0
    topics = dtm.get_topics(time=0, top_terms=5)
    
    # Check output format
    assert isinstance(topics, list)
    if topics:
        assert isinstance(topics[0], tuple)
        assert len(topics[0]) == 2
        assert isinstance(topics[0][0], int)  # Topic ID
        assert isinstance(topics[0][1], str)  # Terms string
