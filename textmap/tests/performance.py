import cProfile
import pstats
import io
import pandas as pd
import pytest
from contextlib import contextmanager
import logging
from textmap.dynamic_topic_models import DynamicTopicModel
from gensim.test.utils import common_texts

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("performance_tests")


@contextmanager
def profile_operation(operation_name):
    """Context manager to profile a code block and print stats."""
    pr = cProfile.Profile()
    pr.enable()
    yield
    pr.disable()
    
    # Print stats sorted by cumulative time
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats(30)  # Print top 30 functions by cumulative time
    
    logger.info(f"Performance profile for {operation_name}:")
    logger.info(s.getvalue())
    
    # Also save to file for later analysis
    filename = f"profile_{operation_name.replace(' ', '_')}.prof"
    pr.dump_stats(filename)
    logger.info(f"Full profile saved to {filename}")


def create_sample_data(size_multiplier=1):
    """Create sample data for profiling, with adjustable size."""
    # Prepare sample data for three time slices
    texts_t0 = [doc for doc in common_texts[:3]] * size_multiplier
    texts_t1 = [doc for doc in common_texts[3:6]] * size_multiplier
    texts_t2 = [doc for doc in common_texts[6:]] * size_multiplier
    all_texts = texts_t0 + texts_t1 + texts_t2

    # Create a DataFrame
    periods = [0] * len(texts_t0) + [1] * len(texts_t1) + [2] * len(texts_t2)
    data = {"text": [" ".join(doc) for doc in all_texts], "period": periods}
    return pd.DataFrame(data)


def get_model_params():
    """Define model parameters for profiling."""
    return {
        "num_topics": 5,
        "chain_variance": 0.005,
        "passes": 10,
        "em_min_iter": 3,
        "em_max_iter": 10,
        "lda_inference_max_iter": 10,
    }


def profile_model_fitting():
    """Profile the model fitting process."""
    logger.info("Starting model fitting profiling")
    
    # Create sample data - use a multiplier to increase dataset size if needed
    sample_data = create_sample_data(size_multiplier=2)
    logger.info(f"Created sample data with {len(sample_data)} documents")
    
    # Initialize model
    model_params = get_model_params()
    dtm = DynamicTopicModel(**model_params)
    
    # Profile the fit operation
    with profile_operation("model_fitting"):
        dtm.fit(sample_data)
    
    return dtm, sample_data


def profile_topic_extraction(dtm, sample_data):
    """Profile the topic extraction process."""
    logger.info("Starting topic extraction profiling")
    
    # Profile getting topics for all time periods
    with profile_operation("get_topics_all_periods"):
        topics_df = dtm.get_topics(time_periods=None, top_terms=10)
    
    # Profile getting topics for a specific time period
    with profile_operation("get_topics_single_period"):
        topics_df_single = dtm.get_topics(time_periods=[0], top_terms=10)
    
    # Profile document transformation
    with profile_operation("transform_documents"):
        topic_distributions = dtm.transform(sample_data)


def profile_preprocessing(dtm, sample_data):
    """Profile the preprocessing steps separately."""
    logger.info("Starting preprocessing profiling")
    
    # Profile text preprocessing
    with profile_operation("preprocess_text"):
        tokenized = dtm._preprocess_text(sample_data["text"].tolist())
    
    # Profile corpus creation
    with profile_operation("create_corpus_and_dictionary"):
        corpus, dictionary = dtm._create_corpus_and_dictionary(sample_data)
    
    # Profile time slice calculation
    with profile_operation("calculate_time_slices"):
        time_slices = dtm._calculate_time_slices(sample_data)


def main():
    """Run all profiling tests."""
    logger.info("Starting performance profiling")
    
    # Profile model fitting
    dtm, sample_data = profile_model_fitting()
    
    # Profile topic extraction
    profile_topic_extraction(dtm, sample_data)
    
    # Profile preprocessing steps
    profile_preprocessing(dtm, sample_data)
    
    logger.info("Performance profiling completed")


if __name__ == "__main__":
    main()
