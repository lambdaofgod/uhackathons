import cProfile
import pstats
import io
import pandas as pd
import pytest
import traceback
import sys
import inspect
from contextlib import contextmanager
import logging
from textmap.dynamic_topic_models import DynamicTopicModel
from gensim.test.utils import common_texts

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
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
        "use_jax": False,  # Disable JAX by default to avoid the 'jac' keyword error
    }


def debug_jax_error():
    """Run a specific test to debug the JAX error."""
    logger.info("Starting JAX error debugging")
    
    # Create a small sample for debugging
    sample_data = create_sample_data(size_multiplier=1)
    
    # Try with JAX enabled
    model_params = get_model_params()
    model_params["use_jax"] = True
    
    logger.info("Creating model with JAX enabled")
    dtm = DynamicTopicModel(**model_params)
    
    try:
        logger.info("Attempting to fit model with JAX")
        dtm.fit(sample_data)
        logger.info("JAX fitting succeeded!")
        return True
    except Exception as e:
        logger.error(f"JAX error: {e}")
        logger.error(traceback.format_exc())
        
        # Try to find the minimize function that's causing the error
        logger.info("Inspecting JAX optimization code...")
        
        try:
            # Import the modules that might contain the minimize function
            import scipy.optimize
            import jax.scipy.optimize
            
            # Check scipy.optimize.minimize signature
            logger.info("scipy.optimize.minimize signature:")
            logger.info(inspect.signature(scipy.optimize.minimize))
            logger.info("scipy.optimize.minimize parameters:")
            for name, param in inspect.signature(scipy.optimize.minimize).parameters.items():
                logger.info(f"  {name}: {param.default}")
            
            # Try to check jax.scipy.optimize.minimize if available
            try:
                logger.info("jax.scipy.optimize.minimize signature:")
                logger.info(inspect.signature(jax.scipy.optimize.minimize))
                logger.info("jax.scipy.optimize.minimize parameters:")
                for name, param in inspect.signature(jax.scipy.optimize.minimize).parameters.items():
                    logger.info(f"  {name}: {param.default}")
            except (ImportError, AttributeError) as e:
                logger.error(f"Could not inspect jax.scipy.optimize.minimize: {e}")
            
            # Check if textmap.ldaseqmodel_jax is being imported
            try:
                import textmap.ldaseqmodel_jax
                logger.info("textmap.ldaseqmodel_jax is available")
                
                # Check the _optimize_obs_word function
                if hasattr(textmap.ldaseqmodel_jax, '_optimize_obs_word'):
                    logger.info("Found _optimize_obs_word function")
                    logger.info(inspect.getsource(textmap.ldaseqmodel_jax._optimize_obs_word))
            except (ImportError, AttributeError) as e:
                logger.error(f"Could not inspect textmap.ldaseqmodel_jax: {e}")
            
        except Exception as inspect_error:
            logger.error(f"Error during inspection: {inspect_error}")
            logger.error(traceback.format_exc())
        
        return False


def profile_model_fitting():
    """Profile the model fitting process."""
    logger.info("Starting model fitting profiling")
    
    # First try to debug the JAX error
    debug_jax_error()
    
    # Create sample data - use a multiplier to increase dataset size if needed
    sample_data = create_sample_data(size_multiplier=2)
    logger.info(f"Created sample data with {len(sample_data)} documents")
    
    # Initialize model with JAX disabled
    model_params = get_model_params()
    logger.info(f"Using model parameters: {model_params}")
    
    dtm = DynamicTopicModel(**model_params)
    
    # Profile the fit operation
    try:
        with profile_operation("model_fitting"):
            dtm.fit(sample_data)
        return dtm, sample_data
    except Exception as e:
        logger.error(f"Error during model fitting: {e}")
        logger.error(traceback.format_exc())
        # Return None to indicate failure
        return None, sample_data


def profile_topic_extraction(dtm, sample_data):
    """Profile the topic extraction process."""
    logger.info("Starting topic extraction profiling")
    
    # Skip if model fitting failed
    if dtm is None:
        logger.warning("Skipping topic extraction profiling because model fitting failed")
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
    except Exception as e:
        logger.error(f"Error during topic extraction: {e}")
        logger.error(traceback.format_exc())


def profile_preprocessing(dtm, sample_data):
    """Profile the preprocessing steps separately."""
    logger.info("Starting preprocessing profiling")
    
    # Create a new model instance for preprocessing tests
    # This ensures we can profile preprocessing even if the main model fitting failed
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
        logger.error(traceback.format_exc())


def main():
    """Run all profiling tests."""
    logger.info("Starting performance profiling")
    
    try:
        # Profile model fitting
        dtm, sample_data = profile_model_fitting()
        
        # Profile topic extraction
        profile_topic_extraction(dtm, sample_data)
        
        # Profile preprocessing steps
        profile_preprocessing(dtm, sample_data)
        
        logger.info("Performance profiling completed")
    except Exception as e:
        logger.error(f"Unhandled exception in performance profiling: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
