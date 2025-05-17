import cProfile
import pstats
import io
import os
import pandas as pd
import numpy as np
import logging
from contextlib import contextmanager
from textmap.dynamic_topic_models import DynamicTopicModel
from gensim.test.utils import common_texts

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
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
    ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
    ps.print_stats(30)  # Print top 30 functions by cumulative time

    logger.info(f"Performance profile for {operation_name}:")
    logger.info(s.getvalue())

    # Also save to file for later analysis
    filename = f"profile_{operation_name.replace(' ', '_')}.prof"
    pr.dump_stats(filename)
    logger.info(f"Full profile saved to {filename}")


def profile_with_jax(func, *args, **kwargs):
    """Profile a function using JAX's profiler."""
    try:
        import jax
        from jax import profiler
        
        # Create a directory for the trace if it doesn't exist
        trace_dir = "/tmp/jax-trace"
        os.makedirs(trace_dir, exist_ok=True)
        
        logger.info(f"Starting JAX profiler trace in {trace_dir}")
        with profiler.trace(trace_dir, create_perfetto_link=True):
            result = func(*args, **kwargs)
            # Make sure computation is complete
            if hasattr(result, "block_until_ready"):
                result.block_until_ready()
        
        logger.info(f"JAX profiler trace completed. View with perfetto at {trace_dir}")
        return result
    except ImportError:
        logger.warning("JAX not available, falling back to regular profiling")
        return func(*args, **kwargs)


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
        "use_jax": True,  # Enable JAX for profiling
    }


def profile_model_fitting():
    """Profile the model fitting process."""
    logger.info("Starting model fitting profiling")

    # Create sample data - use a multiplier to increase dataset size if needed
    sample_data = create_sample_data(size_multiplier=2)
    logger.info(f"Created sample data with {len(sample_data)} documents")

    # Initialize model
    model_params = get_model_params()
    logger.info(f"Using model parameters: {model_params}")
    
    dtm = DynamicTopicModel(**model_params)
    
    # Profile the fit operation with both cProfile and JAX profiler
    try:
        # First with cProfile
        with profile_operation("model_fitting"):
            dtm.fit(sample_data)
        
        # Then try with JAX profiler if available
        try:
            import jax
            logger.info("JAX is available, attempting to profile with JAX profiler")
            
            # Create a fresh model for JAX profiling
            jax_dtm = DynamicTopicModel(**model_params)
            
            # Define a function to profile
            def fit_model(model, data):
                return model.fit(data)
            
            # Profile with JAX
            profile_with_jax(fit_model, jax_dtm, sample_data)
            
        except ImportError:
            logger.info("JAX not available, skipping JAX profiling")
        
        return dtm, sample_data
    except Exception as e:
        logger.error(f"Error during model fitting: {e}")
        # Return None to indicate failure
        return None, sample_data


def profile_topic_extraction(dtm, sample_data):
    """Profile the topic extraction process."""
    logger.info("Starting topic extraction profiling")

    # Skip if model fitting failed
    if dtm is None:
        logger.warning(
            "Skipping topic extraction profiling because model fitting failed"
        )
        return

    try:
        # Profile getting topics for all time periods
        with profile_operation("get_topics_all_periods"):
            topics_df = dtm.get_topics(time_periods=None, top_terms=10)

        # Profile getting topics for a specific time period
        with profile_operation("get_topics_single_period"):
            topics_df_single = dtm.get_topics(time_periods=[0], top_terms=10)

        # Profile document transformation
        with profile_operation("transform_documents"):
            topic_distributions = dtm.transform(sample_data)
            
        # Try JAX profiling for get_topics
        try:
            import jax
            logger.info("Profiling get_topics with JAX profiler")
            
            # Define functions to profile
            def get_all_topics(model):
                return model.get_topics(time_periods=None, top_terms=10)
                
            def transform_docs(model, data):
                return model.transform(data)
            
            # Profile with JAX
            profile_with_jax(get_all_topics, dtm)
            profile_with_jax(transform_docs, dtm, sample_data)
            
        except ImportError:
            logger.info("JAX not available, skipping JAX profiling")
            
    except Exception as e:
        logger.error(f"Error during topic extraction: {e}")


def profile_preprocessing(dtm, sample_data):
    """Profile the preprocessing steps separately."""
    logger.info("Starting preprocessing profiling")

    # Create a new model instance for preprocessing tests
    model_params = get_model_params()
    fresh_dtm = DynamicTopicModel(**model_params)

    try:
        # Profile text preprocessing
        with profile_operation("preprocess_text"):
            tokenized = fresh_dtm._preprocess_text(sample_data["text"].tolist())

        # Profile corpus creation
        with profile_operation("create_corpus_and_dictionary"):
            corpus, dictionary = fresh_dtm._create_corpus_and_dictionary(sample_data)

        # Profile time slice calculation
        with profile_operation("calculate_time_slices"):
            time_slices = fresh_dtm._calculate_time_slices(sample_data)
    except Exception as e:
        logger.error(f"Error during preprocessing profiling: {e}")


def profile_jax_operations():
    """Profile JAX operations directly."""
    try:
        import jax
        import jax.numpy as jnp
        
        logger.info("Profiling JAX matrix operations")
        
        def matrix_operations():
            # Generate random data
            key = jax.random.key(0)
            x = jax.random.normal(key, (1000, 1000))
            
            # Perform matrix operations
            y = x @ x.T
            z = jnp.exp(y / 10.0)
            result = jnp.sum(z)
            
            return result
        
        # Profile with JAX
        profile_with_jax(matrix_operations)
        
    except ImportError:
        logger.info("JAX not available, skipping JAX operations profiling")


def main():
    """Run all profiling tests."""
    logger.info("Starting performance profiling")

    try:
        # Profile JAX operations directly
        profile_jax_operations()
        
        # Profile model fitting
        dtm, sample_data = profile_model_fitting()

        # Profile topic extraction
        profile_topic_extraction(dtm, sample_data)

        # Profile preprocessing steps
        profile_preprocessing(dtm, sample_data)

        logger.info("Performance profiling completed")
    except Exception as e:
        logger.error(f"Unhandled exception in performance profiling: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
