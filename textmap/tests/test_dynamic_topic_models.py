import pytest
import pandas as pd
import numpy as np
from gensim.test.utils import common_texts
from textmap.dynamic_topic_models import DynamicTopicModel


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    # Prepare sample data for three time slices
    texts_t0 = [doc for doc in common_texts[:3]]
    texts_t1 = [doc for doc in common_texts[3:6]]
    texts_t2 = [doc for doc in common_texts[6:]]
    all_texts = texts_t0 + texts_t1 + texts_t2

    # Create a DataFrame
    periods = [0] * len(texts_t0) + [1] * len(texts_t1) + [2] * len(texts_t2)
    data = {"text": [" ".join(doc) for doc in all_texts], "period": periods}
    return pd.DataFrame(data)


@pytest.fixture
def model_params():
    """Define model parameters for testing."""
    return {
        "num_topics": 10,
        "chain_variance": 0.005,
        "passes": 5,
        "em_min_iter": 2,
        "em_max_iter": 5,
        "lda_inference_max_iter": 5,
    }


def test_initialization(model_params):
    """Test model initialization."""
    dtm = DynamicTopicModel(**model_params)
    assert dtm.num_topics == 10
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
    import logging
    import traceback

    # Set up logging
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger("test_fit_transform")

    try:
        # Initialize model
        logger.debug("Initializing DynamicTopicModel")
        dtm = DynamicTopicModel(**model_params)

        # Fit the model
        logger.debug("Fitting model to sample data")
        dtm.fit(sample_data)

        # Check that model components are initialized
        logger.debug(f"lda_seq_model type: {type(dtm.lda_seq_model)}")
        logger.debug(f"lda_seq_model dir: {dir(dtm.lda_seq_model)}")

        # Debug the LdaModel instance used in make_lda_seq_slice
        if hasattr(dtm.lda_seq_model, "make_lda_seq_slice"):
            logger.debug("Checking make_lda_seq_slice method")
            # Get a sample LDA model
            try:
                from gensim.models import ldamodel

                lda_sample = ldamodel.LdaModel(num_topics=model_params["num_topics"])
                logger.debug(f"LdaModel sample dir: {dir(lda_sample)}")
                logger.debug(
                    f"LdaModel has topics attr: {hasattr(lda_sample, 'topics')}"
                )
                logger.debug(
                    f"LdaModel has expElogbeta attr: {hasattr(lda_sample, 'expElogbeta')}"
                )
            except Exception as e:
                logger.error(f"Error checking LdaModel: {e}")
                logger.error(traceback.format_exc())

        assert dtm.lda_seq_model is not None
        assert dtm.corpus is not None
        assert dtm.id2word is not None
        assert len(dtm.time_slices) == 3

        # Transform documents
        logger.debug("Transforming sample data")
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
    except Exception as e:
        logger.error(f"Exception in test_fit_transform: {e}")
        logger.error(traceback.format_exc())
        raise


def test_get_topics(model_params, sample_data):
    """Test getting topics for time slices."""
    import logging
    import traceback

    # Set up logging
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger("test_get_topics")

    try:
        # Initialize and fit model
        logger.debug("Initializing and fitting DynamicTopicModel")
        dtm = DynamicTopicModel(**model_params)
        dtm.fit(sample_data)

        # Debug the model's internal state
        logger.debug(f"lda_seq_model type: {type(dtm.lda_seq_model)}")

        # Check if the model has topic_chains
        if hasattr(dtm.lda_seq_model, "topic_chains"):
            logger.debug(f"topic_chains length: {len(dtm.lda_seq_model.topic_chains)}")
            # Check the first topic chain
            if len(dtm.lda_seq_model.topic_chains) > 0:
                chain = dtm.lda_seq_model.topic_chains[0]
                logger.debug(f"First chain type: {type(chain)}")
                logger.debug(f"First chain attributes: {dir(chain)}")

        # Debug the get_topics method
        logger.debug("Calling get_topics method")

        # Test with a single time period
        topics_df = dtm.get_topics(time_periods=[0], top_terms=5)
        logger.debug(f"topics_df shape: {topics_df.shape}")

        # Check output format
        assert isinstance(topics_df, pd.DataFrame)
        assert set(topics_df.columns) == {"time_period", "topic_id", "terms"}
        assert len(topics_df) > 0
        assert all(topics_df["time_period"] == 0)

        # Test with multiple time periods
        logger.debug("Testing with multiple time periods")
        topics_df_multi = dtm.get_topics(time_periods=[0, 1], top_terms=5)
        assert isinstance(topics_df_multi, pd.DataFrame)
        assert set(topics_df_multi["time_period"].unique()) == {0, 1}

        # Test with all time periods (None)
        logger.debug("Testing with all time periods")
        topics_df_all = dtm.get_topics(time_periods=None, top_terms=5)
        assert isinstance(topics_df_all, pd.DataFrame)
        assert len(topics_df_all["time_period"].unique()) == len(dtm.time_slices)
    except Exception as e:
        logger.error(f"Exception in test_get_topics: {e}")
        logger.error(traceback.format_exc())
        raise
