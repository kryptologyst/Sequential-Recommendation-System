"""
Sequential Recommendation System

A modern implementation of sequential recommendation systems using various approaches
including Markov chains, GRU4Rec, and SASRec for next-item prediction.
"""

from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import yaml
from pathlib import Path
import logging
from dataclasses import dataclass
from abc import ABC, abstractmethod

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Config:
    """Configuration class for the sequential recommendation system."""
    data_file: str = "data/interactions.csv"
    items_file: str = "data/items.csv"
    users_file: str = "data/users.csv"
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    min_sequence_length: int = 3
    max_sequence_length: int = 50
    random_seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    @classmethod
    def from_yaml(cls, config_path: str) -> 'Config':
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Flatten nested config
        flat_config = {}
        for section, values in config_dict.items():
            if isinstance(values, dict):
                for key, value in values.items():
                    flat_config[key] = value
            else:
                flat_config[section] = values
        
        return cls(**flat_config)


class SequentialDataset(Dataset):
    """Dataset class for sequential recommendation data."""
    
    def __init__(self, sequences: List[List[int]], max_length: int = 50):
        """
        Initialize the dataset.
        
        Args:
            sequences: List of user interaction sequences
            max_length: Maximum sequence length
        """
        self.sequences = sequences
        self.max_length = max_length
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a sequence and its target.
        
        Args:
            idx: Index of the sequence
            
        Returns:
            Tuple of (input_sequence, target_item)
        """
        sequence = self.sequences[idx]
        
        # Truncate or pad sequence
        if len(sequence) > self.max_length:
            sequence = sequence[-self.max_length:]
        
        # Input is all items except the last, target is the last item
        input_seq = torch.tensor(sequence[:-1], dtype=torch.long)
        target = torch.tensor(sequence[-1], dtype=torch.long)
        
        return input_seq, target


class BaseSequentialModel(ABC):
    """Abstract base class for sequential recommendation models."""
    
    @abstractmethod
    def fit(self, sequences: List[List[int]], **kwargs) -> None:
        """Fit the model to the training sequences."""
        pass
    
    @abstractmethod
    def predict_next(self, sequence: List[int], k: int = 10) -> List[int]:
        """Predict the next k items for a given sequence."""
        pass
    
    @abstractmethod
    def evaluate(self, test_sequences: List[List[int]], k_values: List[int] = [5, 10]) -> Dict[str, float]:
        """Evaluate the model on test sequences."""
        pass


class MarkovChainModel(BaseSequentialModel):
    """Markov Chain model for sequential recommendation."""
    
    def __init__(self, order: int = 1, smoothing: float = 0.1):
        """
        Initialize the Markov Chain model.
        
        Args:
            order: Order of the Markov chain
            smoothing: Smoothing parameter for unseen transitions
        """
        self.order = order
        self.smoothing = smoothing
        self.transition_matrix: Dict[Tuple, Dict[int, float]] = {}
        self.item_counts: Dict[int, int] = {}
        self.num_items: int = 0
    
    def fit(self, sequences: List[List[int]], **kwargs) -> None:
        """
        Fit the Markov Chain model to training sequences.
        
        Args:
            sequences: List of user interaction sequences
        """
        logger.info("Training Markov Chain model...")
        
        # Count items and build transition matrix
        for sequence in sequences:
            for item in sequence:
                self.item_counts[item] = self.item_counts.get(item, 0) + 1
        
        self.num_items = len(self.item_counts)
        
        # Build transition matrix
        for sequence in sequences:
            for i in range(len(sequence) - self.order):
                context = tuple(sequence[i:i + self.order])
                next_item = sequence[i + self.order]
                
                if context not in self.transition_matrix:
                    self.transition_matrix[context] = {}
                
                self.transition_matrix[context][next_item] = \
                    self.transition_matrix[context].get(next_item, 0) + 1
        
        # Normalize transition probabilities
        for context in self.transition_matrix:
            total_transitions = sum(self.transition_matrix[context].values())
            for item in self.transition_matrix[context]:
                self.transition_matrix[context][item] /= total_transitions
        
        logger.info(f"Markov Chain model trained on {len(sequences)} sequences")
    
    def predict_next(self, sequence: List[int], k: int = 10) -> List[int]:
        """
        Predict the next k items for a given sequence.
        
        Args:
            sequence: Input sequence
            k: Number of items to recommend
            
        Returns:
            List of recommended item IDs
        """
        if len(sequence) < self.order:
            # Fallback to popularity-based recommendation
            return sorted(self.item_counts.items(), key=lambda x: x[1], reverse=True)[:k]
        
        context = tuple(sequence[-self.order:])
        
        if context not in self.transition_matrix:
            # Fallback to popularity-based recommendation
            return sorted(self.item_counts.items(), key=lambda x: x[1], reverse=True)[:k]
        
        # Get transition probabilities for this context
        probabilities = self.transition_matrix[context]
        
        # Sort by probability and return top k
        sorted_items = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
        return [item for item, _ in sorted_items[:k]]
    
    def evaluate(self, test_sequences: List[List[int]], k_values: List[int] = [5, 10]) -> Dict[str, float]:
        """
        Evaluate the Markov Chain model on test sequences.
        
        Args:
            test_sequences: List of test sequences
            k_values: List of k values for evaluation
            
        Returns:
            Dictionary of evaluation metrics
        """
        metrics = {}
        
        for k in k_values:
            precision_scores = []
            recall_scores = []
            hit_rates = []
            
            for sequence in test_sequences:
                if len(sequence) < self.order + 1:
                    continue
                
                # Get ground truth
                ground_truth = sequence[-1]
                
                # Get predictions
                predictions = self.predict_next(sequence[:-1], k)
                
                # Calculate metrics
                hit = 1 if ground_truth in predictions else 0
                precision = hit / k
                recall = hit  # For single target
                
                hit_rates.append(hit)
                precision_scores.append(precision)
                recall_scores.append(recall)
            
            metrics[f'precision@{k}'] = np.mean(precision_scores)
            metrics[f'recall@{k}'] = np.mean(recall_scores)
            metrics[f'hit_rate@{k}'] = np.mean(hit_rates)
        
        return metrics


class GRU4RecModel(BaseSequentialModel):
    """GRU4Rec model for sequential recommendation."""
    
    def __init__(self, num_items: int, hidden_size: int = 128, num_layers: int = 2, 
                 dropout: float = 0.25, learning_rate: float = 0.001, 
                 batch_size: int = 256, epochs: int = 50, device: str = "cpu"):
        """
        Initialize the GRU4Rec model.
        
        Args:
            num_items: Number of unique items
            hidden_size: Hidden size of GRU layers
            num_layers: Number of GRU layers
            dropout: Dropout rate
            learning_rate: Learning rate for optimization
            batch_size: Batch size for training
            epochs: Number of training epochs
            device: Device to use for training
        """
        self.num_items = num_items
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = device
        
        # Initialize model
        self.model = GRU4RecNetwork(num_items, hidden_size, num_layers, dropout)
        self.model.to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()
    
    def fit(self, sequences: List[List[int]], **kwargs) -> None:
        """
        Fit the GRU4Rec model to training sequences.
        
        Args:
            sequences: List of user interaction sequences
        """
        logger.info("Training GRU4Rec model...")
        
        # Create dataset and dataloader
        dataset = SequentialDataset(sequences)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # Training loop
        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            
            for batch_inputs, batch_targets in dataloader:
                batch_inputs = batch_inputs.to(self.device)
                batch_targets = batch_targets.to(self.device)
                
                # Forward pass
                self.optimizer.zero_grad()
                outputs = self.model(batch_inputs)
                loss = self.criterion(outputs, batch_targets)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
            
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}, Loss: {total_loss / len(dataloader):.4f}")
        
        logger.info("GRU4Rec model training completed")
    
    def predict_next(self, sequence: List[int], k: int = 10) -> List[int]:
        """
        Predict the next k items for a given sequence.
        
        Args:
            sequence: Input sequence
            k: Number of items to recommend
            
        Returns:
            List of recommended item IDs
        """
        self.model.eval()
        
        with torch.no_grad():
            input_tensor = torch.tensor([sequence], dtype=torch.long).to(self.device)
            outputs = self.model(input_tensor)
            probabilities = F.softmax(outputs, dim=1)
            
            # Get top k items
            top_k = torch.topk(probabilities, k, dim=1)
            return top_k.indices[0].cpu().numpy().tolist()
    
    def evaluate(self, test_sequences: List[List[int]], k_values: List[int] = [5, 10]) -> Dict[str, float]:
        """
        Evaluate the GRU4Rec model on test sequences.
        
        Args:
            test_sequences: List of test sequences
            k_values: List of k values for evaluation
            
        Returns:
            Dictionary of evaluation metrics
        """
        metrics = {}
        
        for k in k_values:
            precision_scores = []
            recall_scores = []
            hit_rates = []
            
            for sequence in test_sequences:
                if len(sequence) < 2:
                    continue
                
                # Get ground truth
                ground_truth = sequence[-1]
                
                # Get predictions
                predictions = self.predict_next(sequence[:-1], k)
                
                # Calculate metrics
                hit = 1 if ground_truth in predictions else 0
                precision = hit / k
                recall = hit  # For single target
                
                hit_rates.append(hit)
                precision_scores.append(precision)
                recall_scores.append(recall)
            
            metrics[f'precision@{k}'] = np.mean(precision_scores)
            metrics[f'recall@{k}'] = np.mean(recall_scores)
            metrics[f'hit_rate@{k}'] = np.mean(hit_rates)
        
        return metrics


class GRU4RecNetwork(nn.Module):
    """GRU4Rec neural network architecture."""
    
    def __init__(self, num_items: int, hidden_size: int, num_layers: int, dropout: float):
        """
        Initialize the GRU4Rec network.
        
        Args:
            num_items: Number of unique items
            hidden_size: Hidden size of GRU layers
            num_layers: Number of GRU layers
            dropout: Dropout rate
        """
        super(GRU4RecNetwork, self).__init__()
        
        self.num_items = num_items
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Embedding layer
        self.item_embedding = nn.Embedding(num_items, hidden_size)
        
        # GRU layers
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers, 
                          batch_first=True, dropout=dropout if num_layers > 1 else 0)
        
        # Output layer
        self.output_layer = nn.Linear(hidden_size, num_items)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the network.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length)
            
        Returns:
            Output tensor of shape (batch_size, num_items)
        """
        # Embed items
        embedded = self.item_embedding(x)
        
        # Apply GRU
        gru_output, _ = self.gru(embedded)
        
        # Get the last output
        last_output = gru_output[:, -1, :]
        
        # Apply dropout and output layer
        output = self.dropout(last_output)
        output = self.output_layer(output)
        
        return output


def set_random_seeds(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def main():
    """Main function to demonstrate the sequential recommendation system."""
    # Set random seeds
    set_random_seeds(42)
    
    # Load configuration
    config = Config.from_yaml("configs/config.yaml")
    
    # Generate sample data
    logger.info("Generating sample sequential data...")
    
    # Create sample sequences (in practice, this would be loaded from data files)
    sample_sequences = [
        [1, 2, 3, 4, 5],
        [1, 3, 5, 2, 4],
        [2, 4, 5, 1, 3],
        [1, 2, 4, 3, 5],
        [3, 5, 1, 2, 4],
        [2, 3, 4, 5, 1],
        [1, 4, 2, 5, 3],
        [3, 1, 5, 4, 2],
        [4, 2, 1, 3, 5],
        [5, 1, 3, 2, 4]
    ]
    
    # Split data
    train_size = int(len(sample_sequences) * config.train_ratio)
    val_size = int(len(sample_sequences) * config.val_ratio)
    
    train_sequences = sample_sequences[:train_size]
    val_sequences = sample_sequences[train_size:train_size + val_size]
    test_sequences = sample_sequences[train_size + val_size:]
    
    logger.info(f"Train sequences: {len(train_sequences)}")
    logger.info(f"Validation sequences: {len(val_sequences)}")
    logger.info(f"Test sequences: {len(test_sequences)}")
    
    # Train and evaluate models
    models = {
        'Markov Chain': MarkovChainModel(order=1, smoothing=0.1),
        'GRU4Rec': GRU4RecModel(num_items=5, device=config.device)
    }
    
    results = {}
    
    for model_name, model in models.items():
        logger.info(f"Training {model_name}...")
        model.fit(train_sequences)
        
        logger.info(f"Evaluating {model_name}...")
        metrics = model.evaluate(test_sequences, k_values=[5, 10])
        results[model_name] = metrics
        
        # Print results
        logger.info(f"{model_name} Results:")
        for metric, value in metrics.items():
            logger.info(f"  {metric}: {value:.4f}")
    
    # Demonstrate recommendations
    logger.info("\nDemonstrating recommendations:")
    test_sequence = [1, 2, 3]
    logger.info(f"Input sequence: {test_sequence}")
    
    for model_name, model in models.items():
        recommendations = model.predict_next(test_sequence, k=3)
        logger.info(f"{model_name} recommendations: {recommendations}")


if __name__ == "__main__":
    main()
