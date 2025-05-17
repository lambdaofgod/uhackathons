"""
Performance check for JAX operations in the Dynamic Topic Model.
"""

import os
import jax
import logging
from textmap.dynamic_topic_models import DynamicTopicModel
from gensim.test.utils import common_texts
import pandas as pd


def create_sample_data():
    """Create sample data for profiling."""
    # Prepare sample data for three time slices
    texts_t0 = [doc for doc in common_texts[:3]]
    texts_t1 = [doc for doc in common_texts[3:6]]
    texts_t2 = [doc for doc in common_texts[6:]]
    all_texts = texts_t0 + texts_t1 + texts_t2

    # Create a DataFrame
    periods = [0] * len(texts_t0) + [1] * len(texts_t1) + [2] * len(texts_t2)
    data = {"text": [" ".join(doc) for doc in all_texts], "period": periods}
    return pd.DataFrame(data)


def run_profiling():
    """Run JAX profiling on the fit-transform operations."""
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("jax_profiler")
    
    # Create trace directory
    trace_dir = "/tmp/jax-dtm-trace"
    os.makedirs(trace_dir, exist_ok=True)
    
    # Create sample data
    sample_data = create_sample_data()
    
    # Initialize model
    model_params = {
        "num_topics": 10,
        "chain_variance": 0.005,
        "passes": 5,
        "em_min_iter": 2,
        "em_max_iter": 5,
        "lda_inference_max_iter": 5,
    }
    
    logger.info("Initializing DynamicTopicModel")
    dtm = DynamicTopicModel(**model_params)
    
    # Profile the fit operation
    logger.info(f"Starting JAX profiling for model fitting (trace dir: {trace_dir})")
    with jax.profiler.trace(trace_dir, create_perfetto_link=True):
        # Fit the model
        logger.info("Fitting model to sample data")
        dtm.fit(sample_data)
        
        # Ensure all JAX operations are complete
        jax.effects_barrier()
    
    logger.info("JAX profiling for model fitting completed")
    
    # Profile the transform operation
    transform_trace_dir = "/tmp/jax-dtm-transform-trace"
    os.makedirs(transform_trace_dir, exist_ok=True)
    
    logger.info(f"Starting JAX profiling for transform (trace dir: {transform_trace_dir})")
    with jax.profiler.trace(transform_trace_dir, create_perfetto_link=True):
        # Transform documents
        logger.info("Transforming sample data")
        dtm.transform(sample_data)
        
        # Ensure all JAX operations are complete
        jax.effects_barrier()
    
    logger.info("JAX profiling for transform completed")
    
    # Log profiling information
    logger.info("JAX profiling information saved to:")
    logger.info(f"  - Fit trace: {trace_dir}")
    logger.info(f"  - Transform trace: {transform_trace_dir}")
    logger.info("View these traces with TensorBoard or Perfetto UI")


if __name__ == "__main__":
    run_profiling()
