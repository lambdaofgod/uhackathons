import unittest
import pandas as pd
import numpy as np
from gensim.test.utils import common_texts
from ..dynamic_topic_models import DynamicTopicModel

class TestDynamicTopicModel(unittest.TestCase):
    """Test cases for the DynamicTopicModel class."""
    
    def setUp(self):
        """Set up test data."""
        # Prepare sample data for three time slices
        texts_t0 = [doc for doc in common_texts[:3]]
        texts_t1 = [doc for doc in common_texts[3:6]]
        texts_t2 = [doc for doc in common_texts[6:]]
        all_texts = texts_t0 + texts_t1 + texts_t2
        
        # Create a DataFrame
        periods = [0]*len(texts_t0) + [1]*len(texts_t1) + [2]*len(texts_t2)
        data = {'text': [" ".join(doc) for doc in all_texts], 'period': periods}
        self.sample_df = pd.DataFrame(data)
        
        # Model parameters for testing
        self.model_params = {
            'num_topics': 2,
            'chain_variance': 0.005,
            'passes': 5,
            'em_min_iter': 2,
            'em_max_iter': 5,
            'lda_inference_max_iter': 5
        }
    
    def test_initialization(self):
        """Test model initialization."""
        dtm = DynamicTopicModel(**self.model_params)
        self.assertEqual(dtm.num_topics, 2)
        self.assertEqual(dtm.text_col, "text")
        self.assertEqual(dtm.period_col, "period")
        self.assertIsNone(dtm.lda_seq_model)
    
    def test_preprocessing(self):
        """Test text preprocessing."""
        dtm = DynamicTopicModel(**self.model_params)
        texts = ["this is a test", "another test document"]
        tokenized = dtm._preprocess_text(texts)
        
        self.assertEqual(len(tokenized), 2)
        self.assertEqual(tokenized[0], ["this", "is", "a", "test"])
        self.assertEqual(tokenized[1], ["another", "test", "document"])
    
    def test_corpus_creation(self):
        """Test corpus and dictionary creation."""
        dtm = DynamicTopicModel(**self.model_params)
        corpus, dictionary = dtm._create_corpus_and_dictionary(self.sample_df)
        
        self.assertIsNotNone(corpus)
        self.assertIsNotNone(dictionary)
        self.assertEqual(len(corpus), len(self.sample_df))
        self.assertGreater(len(dictionary), 0)
    
    def test_time_slices(self):
        """Test time slice calculation."""
        dtm = DynamicTopicModel(**self.model_params)
        time_slices = dtm._calculate_time_slices(self.sample_df)
        
        self.assertEqual(len(time_slices), 3)  # We have 3 time periods
        self.assertEqual(sum(time_slices), len(self.sample_df))  # Total should match df length
    
    def test_fit_transform(self):
        """Test fitting and transforming."""
        try:
            # Initialize model
            dtm = DynamicTopicModel(**self.model_params)
            
            # Fit the model
            dtm.fit(self.sample_df)
            
            # Check that model components are initialized
            self.assertIsNotNone(dtm.lda_seq_model)
            self.assertIsNotNone(dtm.corpus)
            self.assertIsNotNone(dtm.id2word)
            self.assertEqual(len(dtm.time_slices), 3)
            
            # Transform documents
            topic_distributions = dtm.transform(self.sample_df)
            
            # Check output format
            self.assertEqual(len(topic_distributions), len(self.sample_df))
            
            # Check first document's topic distribution
            first_doc_topics = topic_distributions[0]
            self.assertIsInstance(first_doc_topics, list)
            
            if first_doc_topics:  # If not empty
                # Check topic distribution format
                self.assertIsInstance(first_doc_topics[0], tuple)
                self.assertEqual(len(first_doc_topics[0]), 2)
                
                # Topic ID should be an integer
                self.assertIsInstance(first_doc_topics[0][0], int)
                
                # Probability should be a float between 0 and 1
                self.assertIsInstance(first_doc_topics[0][1], float)
                self.assertGreaterEqual(first_doc_topics[0][1], 0.0)
                self.assertLessEqual(first_doc_topics[0][1], 1.0)
                
                # Probabilities should sum to approximately 1
                total_prob = sum(prob for _, prob in first_doc_topics)
                self.assertAlmostEqual(total_prob, 1.0, places=1)
                
                # Topics should be sorted by probability (descending)
                for i in range(len(first_doc_topics) - 1):
                    self.assertGreaterEqual(
                        first_doc_topics[i][1],
                        first_doc_topics[i + 1][1]
                    )
        
        except Exception as e:
            self.fail(f"Test failed with error: {e}")
    
    def test_get_topics(self):
        """Test getting topics for a time slice."""
        try:
            # Initialize and fit model
            dtm = DynamicTopicModel(**self.model_params)
            dtm.fit(self.sample_df)
            
            # Get topics for time=0
            topics = dtm.get_topics(time=0, top_terms=5)
            
            # Check output format
            self.assertIsInstance(topics, list)
            if topics:
                self.assertIsInstance(topics[0], tuple)
                self.assertEqual(len(topics[0]), 2)
                self.assertIsInstance(topics[0][0], int)  # Topic ID
                self.assertIsInstance(topics[0][1], str)  # Terms string
        
        except Exception as e:
            self.fail(f"Test failed with error: {e}")

if __name__ == '__main__':
    unittest.main()
