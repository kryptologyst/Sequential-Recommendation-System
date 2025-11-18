"""
Unit tests for sequential recommendation system.

This module contains comprehensive tests for all components
of the sequential recommendation system.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.sequential_models import (
    MarkovChainModel, GRU4RecModel, GRU4RecNetwork, 
    BaseSequentialModel, Config, set_random_seeds
)
from src.data.data_pipeline import (
    SequentialDataGenerator, DataLoader, SequenceProcessor, 
    DataConfig, generate_sample_data
)
from src.utils.evaluation import SequentialEvaluator


class TestMarkovChainModel:
    """Test cases for Markov Chain model."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.model = MarkovChainModel(order=1, smoothing=0.1)
        self.sample_sequences = [
            [1, 2, 3, 4],
            [1, 3, 5, 2],
            [2, 4, 5, 1]
        ]
    
    def test_fit(self):
        """Test model fitting."""
        self.model.fit(self.sample_sequences)
        
        # Check that transition matrix is built
        assert len(self.model.transition_matrix) > 0
        assert self.model.num_items > 0
    
    def test_predict_next(self):
        """Test next item prediction."""
        self.model.fit(self.sample_sequences)
        
        predictions = self.model.predict_next([1, 2], k=3)
        
        assert isinstance(predictions, list)
        assert len(predictions) <= 3
        assert all(isinstance(item, int) for item in predictions)
    
    def test_evaluate(self):
        """Test model evaluation."""
        self.model.fit(self.sample_sequences)
        
        test_sequences = [[1, 2, 3], [2, 4, 5]]
        metrics = self.model.evaluate(test_sequences, k_values=[5, 10])
        
        assert isinstance(metrics, dict)
        assert 'precision@5' in metrics
        assert 'recall@5' in metrics
        assert 'hit_rate@5' in metrics
        assert all(0 <= value <= 1 for value in metrics.values())


class TestGRU4RecModel:
    """Test cases for GRU4Rec model."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.model = GRU4RecModel(
            num_items=10,
            hidden_size=32,
            num_layers=1,
            dropout=0.1,
            epochs=1,  # Minimal for testing
            device="cpu"
        )
        self.sample_sequences = [
            [1, 2, 3, 4],
            [1, 3, 5, 2],
            [2, 4, 5, 1]
        ]
    
    def test_model_initialization(self):
        """Test model initialization."""
        assert self.model.num_items == 10
        assert self.model.hidden_size == 32
        assert self.model.num_layers == 1
    
    def test_fit(self):
        """Test model fitting."""
        # This should not raise an exception
        self.model.fit(self.sample_sequences)
    
    def test_predict_next(self):
        """Test next item prediction."""
        self.model.fit(self.sample_sequences)
        
        predictions = self.model.predict_next([1, 2], k=3)
        
        assert isinstance(predictions, list)
        assert len(predictions) <= 3
        assert all(isinstance(item, int) for item in predictions)
    
    def test_evaluate(self):
        """Test model evaluation."""
        self.model.fit(self.sample_sequences)
        
        test_sequences = [[1, 2, 3], [2, 4, 5]]
        metrics = self.model.evaluate(test_sequences, k_values=[5])
        
        assert isinstance(metrics, dict)
        assert 'precision@5' in metrics
        assert all(0 <= value <= 1 for value in metrics.values())


class TestGRU4RecNetwork:
    """Test cases for GRU4Rec neural network."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.network = GRU4RecNetwork(
            num_items=10,
            hidden_size=32,
            num_layers=1,
            dropout=0.1
        )
    
    def test_forward_pass(self):
        """Test forward pass of the network."""
        import torch
        
        # Create dummy input
        input_tensor = torch.tensor([[1, 2, 3]], dtype=torch.long)
        
        # Forward pass
        output = self.network(input_tensor)
        
        assert output.shape == (1, 10)  # batch_size, num_items
        assert not torch.isnan(output).any()


class TestSequentialDataGenerator:
    """Test cases for data generator."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = DataConfig(
            num_users=10,
            num_items=20,
            min_sequence_length=3,
            max_sequence_length=10
        )
        self.generator = SequentialDataGenerator(self.config)
    
    def test_generate_item_popularity(self):
        """Test item popularity generation."""
        popularity = self.generator.generate_item_popularity()
        
        assert len(popularity) == self.config.num_items
        assert np.isclose(popularity.sum(), 1.0)
        assert all(popularity >= 0)
    
    def test_generate_user_preferences(self):
        """Test user preference generation."""
        preferences = self.generator.generate_user_preferences()
        
        assert len(preferences) == self.config.num_users
        assert all(isinstance(prefs, list) for prefs in preferences.values())
    
    def test_generate_sequence(self):
        """Test sequence generation."""
        popularity = self.generator.generate_item_popularity()
        user_preferences = self.generator.generate_user_preferences()
        
        sequence = self.generator.generate_sequence(0, user_preferences, popularity)
        
        assert isinstance(sequence, list)
        assert len(sequence) >= self.config.min_sequence_length
        assert len(sequence) <= self.config.max_sequence_length
        assert all(0 <= item < self.config.num_items for item in sequence)
    
    def test_generate_interactions_data(self):
        """Test interactions data generation."""
        df = self.generator.generate_interactions_data()
        
        assert isinstance(df, pd.DataFrame)
        assert 'user_id' in df.columns
        assert 'item_id' in df.columns
        assert 'timestamp' in df.columns
        assert 'weight' in df.columns
        assert len(df) > 0
    
    def test_generate_items_data(self):
        """Test items data generation."""
        df = self.generator.generate_items_data()
        
        assert isinstance(df, pd.DataFrame)
        assert 'item_id' in df.columns
        assert 'title' in df.columns
        assert 'category' in df.columns
        assert len(df) == self.config.num_items
    
    def test_generate_users_data(self):
        """Test users data generation."""
        df = self.generator.generate_users_data()
        
        assert isinstance(df, pd.DataFrame)
        assert 'user_id' in df.columns
        assert 'age' in df.columns
        assert 'gender' in df.columns
        assert len(df) == self.config.num_users


class TestDataLoader:
    """Test cases for data loader."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.loader = DataLoader("test_data")
    
    def test_save_and_load_data(self):
        """Test saving and loading data."""
        # Create sample data
        interactions = pd.DataFrame({
            'user_id': [0, 0, 1],
            'item_id': [1, 2, 1],
            'timestamp': [1, 2, 3],
            'weight': [1.0, 1.0, 1.0]
        })
        
        items = pd.DataFrame({
            'item_id': [1, 2],
            'title': ['Item1', 'Item2'],
            'category': ['A', 'B'],
            'price': [10.0, 20.0],
            'rating': [4.0, 5.0],
            'features': ['f1', 'f2']
        })
        
        users = pd.DataFrame({
            'user_id': [0, 1],
            'age': [25, 30],
            'gender': ['M', 'F'],
            'location': ['US', 'UK'],
            'signup_date': ['2023-01-01', '2023-01-02']
        })
        
        # Save data
        self.loader.save_data(interactions, items, users)
        
        # Load data
        loaded_interactions = self.loader.load_interactions()
        loaded_items = self.loader.load_items()
        loaded_users = self.loader.load_users()
        
        # Verify data
        pd.testing.assert_frame_equal(interactions, loaded_interactions)
        pd.testing.assert_frame_equal(items, loaded_items)
        pd.testing.assert_frame_equal(users, loaded_users)


class TestSequenceProcessor:
    """Test cases for sequence processor."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.processor = SequenceProcessor(min_sequence_length=2, max_sequence_length=10)
        
        self.sample_interactions = pd.DataFrame({
            'user_id': [0, 0, 0, 1, 1],
            'item_id': [1, 2, 3, 1, 2],
            'timestamp': [1, 2, 3, 1, 2],
            'weight': [1.0, 1.0, 1.0, 1.0, 1.0]
        })
    
    def test_interactions_to_sequences(self):
        """Test converting interactions to sequences."""
        sequences = self.processor.interactions_to_sequences(self.sample_interactions)
        
        assert isinstance(sequences, list)
        assert len(sequences) > 0
        assert all(isinstance(seq, list) for seq in sequences)
        assert all(len(seq) >= self.processor.min_sequence_length for seq in sequences)
    
    def test_split_sequences(self):
        """Test splitting sequences."""
        sequences = [[1, 2, 3], [1, 3, 5], [2, 4, 5], [1, 2, 4], [3, 5, 1]]
        
        train, val, test = self.processor.split_sequences(sequences, 0.6, 0.2)
        
        assert len(train) + len(val) + len(test) == len(sequences)
        assert len(train) > 0
        assert len(val) >= 0
        assert len(test) >= 0


class TestSequentialEvaluator:
    """Test cases for sequential evaluator."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.evaluator = SequentialEvaluator(k_values=[5, 10])
    
    def test_precision_at_k(self):
        """Test precision@k calculation."""
        predictions = [1, 2, 3, 4, 5]
        ground_truth = 3
        
        precision = self.evaluator.precision_at_k(predictions, ground_truth, k=5)
        assert precision == 1.0
        
        precision = self.evaluator.precision_at_k(predictions, ground_truth, k=2)
        assert precision == 0.0
    
    def test_recall_at_k(self):
        """Test recall@k calculation."""
        predictions = [1, 2, 3, 4, 5]
        ground_truth = 3
        
        recall = self.evaluator.recall_at_k(predictions, ground_truth, k=5)
        assert recall == 1.0
        
        recall = self.evaluator.recall_at_k(predictions, ground_truth, k=2)
        assert recall == 0.0
    
    def test_ndcg_at_k(self):
        """Test NDCG@k calculation."""
        predictions = [1, 2, 3, 4, 5]
        ground_truth = 3
        
        ndcg = self.evaluator.ndcg_at_k(predictions, ground_truth, k=5)
        assert 0 <= ndcg <= 1
        
        ndcg = self.evaluator.ndcg_at_k(predictions, ground_truth, k=2)
        assert ndcg == 0.0
    
    def test_intra_list_diversity(self):
        """Test intra-list diversity calculation."""
        predictions = [1, 2, 3, 4, 5]
        diversity = self.evaluator.intra_list_diversity(predictions)
        
        assert 0 <= diversity <= 1
        assert diversity == 1.0  # All items are unique
    
    def test_coverage(self):
        """Test coverage calculation."""
        all_predictions = [[1, 2, 3], [2, 3, 4], [3, 4, 5]]
        all_items = [1, 2, 3, 4, 5, 6]
        
        coverage = self.evaluator.coverage(all_predictions, all_items)
        
        assert 0 <= coverage <= 1
        assert coverage == 5/6  # 5 out of 6 items covered


class TestConfig:
    """Test cases for configuration."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = Config()
        
        assert config.train_ratio == 0.7
        assert config.val_ratio == 0.15
        assert config.test_ratio == 0.15
        assert config.min_sequence_length == 3
        assert config.max_sequence_length == 50
    
    def test_config_from_yaml(self):
        """Test loading configuration from YAML."""
        # Create a temporary YAML file
        import tempfile
        import yaml
        
        config_data = {
            'data': {
                'train_ratio': 0.8,
                'val_ratio': 0.1,
                'test_ratio': 0.1
            },
            'training': {
                'random_seed': 123
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = f.name
        
        try:
            config = Config.from_yaml(temp_path)
            assert config.train_ratio == 0.8
            assert config.val_ratio == 0.1
            assert config.test_ratio == 0.1
            assert config.random_seed == 123
        finally:
            import os
            os.unlink(temp_path)


class TestUtilityFunctions:
    """Test cases for utility functions."""
    
    def test_set_random_seeds(self):
        """Test setting random seeds."""
        # This should not raise an exception
        set_random_seeds(42)
    
    @patch('src.data.data_pipeline.generate_sample_data')
    def test_generate_sample_data(self, mock_generate):
        """Test sample data generation."""
        mock_generate.return_value = None
        
        generate_sample_data("test_dir")
        
        mock_generate.assert_called_once()


# Integration tests
class TestIntegration:
    """Integration tests for the complete pipeline."""
    
    def test_end_to_end_pipeline(self):
        """Test complete end-to-end pipeline."""
        # Generate sample data
        config = DataConfig(num_users=5, num_items=10)
        generator = SequentialDataGenerator(config)
        
        interactions_df = generator.generate_interactions_data()
        items_df = generator.generate_items_data()
        users_df = generator.generate_users_data()
        
        # Process sequences
        processor = SequenceProcessor()
        sequences = processor.interactions_to_sequences(interactions_df)
        train_sequences, val_sequences, test_sequences = processor.split_sequences(sequences)
        
        # Train models
        markov_model = MarkovChainModel()
        markov_model.fit(train_sequences)
        
        # Evaluate
        evaluator = SequentialEvaluator(k_values=[5])
        metrics = evaluator.evaluate_model(markov_model, test_sequences)
        
        assert isinstance(metrics, dict)
        assert 'precision@5' in metrics
        assert all(0 <= value <= 1 for value in metrics.values())


if __name__ == "__main__":
    pytest.main([__file__])
