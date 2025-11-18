"""
Data pipeline utilities for sequential recommendation system.

This module provides functions for generating, loading, and preprocessing
sequential recommendation data.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import logging
from dataclasses import dataclass
import random

logger = logging.getLogger(__name__)


@dataclass
class DataConfig:
    """Configuration for data generation and processing."""
    num_users: int = 1000
    num_items: int = 100
    avg_sequence_length: int = 10
    min_sequence_length: int = 3
    max_sequence_length: int = 50
    popularity_bias: float = 0.7  # Higher values = more popular items
    seasonality_strength: float = 0.3
    noise_level: float = 0.1
    random_seed: int = 42


class SequentialDataGenerator:
    """Generator for realistic sequential recommendation data."""
    
    def __init__(self, config: DataConfig):
        """
        Initialize the data generator.
        
        Args:
            config: Data generation configuration
        """
        self.config = config
        random.seed(config.random_seed)
        np.random.seed(config.random_seed)
    
    def generate_item_popularity(self) -> np.ndarray:
        """
        Generate item popularity distribution using power law.
        
        Returns:
            Array of popularity scores for each item
        """
        # Power law distribution for item popularity
        alpha = 1.5  # Power law exponent
        popularity = np.power(np.arange(1, self.config.num_items + 1), -alpha)
        popularity = popularity / popularity.sum()
        
        return popularity
    
    def generate_user_preferences(self) -> Dict[int, List[int]]:
        """
        Generate user preference profiles.
        
        Returns:
            Dictionary mapping user_id to list of preferred item categories
        """
        user_preferences = {}
        
        # Define item categories (simplified)
        categories = list(range(0, self.config.num_items, 10))  # 10 categories
        
        for user_id in range(self.config.num_users):
            # Each user has preferences for 2-4 categories
            num_preferences = random.randint(2, 4)
            preferences = random.sample(categories, num_preferences)
            user_preferences[user_id] = preferences
        
        return user_preferences
    
    def generate_sequence(self, user_id: int, user_preferences: Dict[int, List[int]], 
                         item_popularity: np.ndarray) -> List[int]:
        """
        Generate a sequence for a specific user.
        
        Args:
            user_id: User identifier
            user_preferences: User preference mapping
            item_popularity: Item popularity scores
            
        Returns:
            List of item IDs representing the user's interaction sequence
        """
        sequence_length = random.randint(
            self.config.min_sequence_length, 
            self.config.max_sequence_length
        )
        
        sequence = []
        user_categories = user_preferences.get(user_id, [])
        
        for step in range(sequence_length):
            # Determine item selection probability
            if step == 0:
                # First item: influenced by user preferences and popularity
                if user_categories and random.random() < 0.7:
                    # Choose from preferred categories
                    category = random.choice(user_categories)
                    item_range = range(category, min(category + 10, self.config.num_items))
                    item_id = random.choice(list(item_range))
                else:
                    # Choose based on popularity
                    item_id = np.random.choice(
                        self.config.num_items, 
                        p=item_popularity
                    )
            else:
                # Subsequent items: influenced by previous items and preferences
                if random.random() < 0.6:
                    # Sequential dependency: choose similar items
                    last_item = sequence[-1]
                    # Items in same category or nearby
                    category_start = (last_item // 10) * 10
                    item_range = range(
                        max(0, category_start - 5), 
                        min(self.config.num_items, category_start + 15)
                    )
                    item_id = random.choice(list(item_range))
                else:
                    # Choose based on popularity and user preferences
                    if user_categories and random.random() < 0.5:
                        category = random.choice(user_categories)
                        item_range = range(category, min(category + 10, self.config.num_items))
                        item_id = random.choice(list(item_range))
                    else:
                        item_id = np.random.choice(
                            self.config.num_items, 
                            p=item_popularity
                        )
            
            # Add some noise
            if random.random() < self.config.noise_level:
                item_id = random.randint(0, self.config.num_items - 1)
            
            sequence.append(item_id)
        
        return sequence
    
    def generate_interactions_data(self) -> pd.DataFrame:
        """
        Generate interactions data in the canonical format.
        
        Returns:
            DataFrame with columns: user_id, item_id, timestamp, weight
        """
        logger.info("Generating interactions data...")
        
        # Generate item popularity and user preferences
        item_popularity = self.generate_item_popularity()
        user_preferences = self.generate_user_preferences()
        
        interactions = []
        timestamp = 0
        
        for user_id in range(self.config.num_users):
            sequence = self.generate_sequence(user_id, user_preferences, item_popularity)
            
            for item_id in sequence:
                interactions.append({
                    'user_id': user_id,
                    'item_id': item_id,
                    'timestamp': timestamp,
                    'weight': 1.0  # Binary interaction
                })
                timestamp += random.randint(1, 100)  # Random time gaps
        
        df = pd.DataFrame(interactions)
        logger.info(f"Generated {len(df)} interactions for {self.config.num_users} users")
        
        return df
    
    def generate_items_data(self) -> pd.DataFrame:
        """
        Generate items metadata.
        
        Returns:
            DataFrame with columns: item_id, title, category, features
        """
        logger.info("Generating items data...")
        
        items = []
        categories = ['Electronics', 'Books', 'Clothing', 'Home', 'Sports', 
                     'Beauty', 'Toys', 'Automotive', 'Health', 'Food']
        
        for item_id in range(self.config.num_items):
            category_idx = item_id // 10
            category = categories[category_idx % len(categories)]
            
            items.append({
                'item_id': item_id,
                'title': f'Item_{item_id}',
                'category': category,
                'price': round(random.uniform(10, 500), 2),
                'rating': round(random.uniform(3.0, 5.0), 1),
                'features': f'feature_{item_id % 5}'  # Simplified features
            })
        
        df = pd.DataFrame(items)
        logger.info(f"Generated {len(df)} items")
        
        return df
    
    def generate_users_data(self) -> pd.DataFrame:
        """
        Generate users metadata.
        
        Returns:
            DataFrame with columns: user_id, age, gender, location
        """
        logger.info("Generating users data...")
        
        users = []
        genders = ['M', 'F', 'Other']
        locations = ['US', 'UK', 'CA', 'AU', 'DE', 'FR', 'JP', 'CN']
        
        for user_id in range(self.config.num_users):
            users.append({
                'user_id': user_id,
                'age': random.randint(18, 65),
                'gender': random.choice(genders),
                'location': random.choice(locations),
                'signup_date': f'2023-{random.randint(1, 12):02d}-{random.randint(1, 28):02d}'
            })
        
        df = pd.DataFrame(users)
        logger.info(f"Generated {len(df)} users")
        
        return df


class DataLoader:
    """Data loader for sequential recommendation data."""
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialize the data loader.
        
        Args:
            data_dir: Directory containing data files
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
    
    def load_interactions(self, file_path: Optional[str] = None) -> pd.DataFrame:
        """
        Load interactions data.
        
        Args:
            file_path: Path to interactions file
            
        Returns:
            DataFrame with interactions data
        """
        if file_path is None:
            file_path = self.data_dir / "interactions.csv"
        
        if not Path(file_path).exists():
            raise FileNotFoundError(f"Interactions file not found: {file_path}")
        
        df = pd.read_csv(file_path)
        logger.info(f"Loaded {len(df)} interactions from {file_path}")
        
        return df
    
    def load_items(self, file_path: Optional[str] = None) -> pd.DataFrame:
        """
        Load items data.
        
        Args:
            file_path: Path to items file
            
        Returns:
            DataFrame with items data
        """
        if file_path is None:
            file_path = self.data_dir / "items.csv"
        
        if not Path(file_path).exists():
            raise FileNotFoundError(f"Items file not found: {file_path}")
        
        df = pd.read_csv(file_path)
        logger.info(f"Loaded {len(df)} items from {file_path}")
        
        return df
    
    def load_users(self, file_path: Optional[str] = None) -> pd.DataFrame:
        """
        Load users data.
        
        Args:
            file_path: Path to users file
            
        Returns:
            DataFrame with users data
        """
        if file_path is None:
            file_path = self.data_dir / "users.csv"
        
        if not Path(file_path).exists():
            raise FileNotFoundError(f"Users file not found: {file_path}")
        
        df = pd.read_csv(file_path)
        logger.info(f"Loaded {len(df)} users from {file_path}")
        
        return df
    
    def save_data(self, interactions: pd.DataFrame, items: pd.DataFrame, 
                  users: pd.DataFrame) -> None:
        """
        Save data to CSV files.
        
        Args:
            interactions: Interactions DataFrame
            items: Items DataFrame
            users: Users DataFrame
        """
        interactions_path = self.data_dir / "interactions.csv"
        items_path = self.data_dir / "items.csv"
        users_path = self.data_dir / "users.csv"
        
        interactions.to_csv(interactions_path, index=False)
        items.to_csv(items_path, index=False)
        users.to_csv(users_path, index=False)
        
        logger.info(f"Saved data to {self.data_dir}")


class SequenceProcessor:
    """Processor for converting interactions to sequences."""
    
    def __init__(self, min_sequence_length: int = 3, max_sequence_length: int = 50):
        """
        Initialize the sequence processor.
        
        Args:
            min_sequence_length: Minimum sequence length
            max_sequence_length: Maximum sequence length
        """
        self.min_sequence_length = min_sequence_length
        self.max_sequence_length = max_sequence_length
    
    def interactions_to_sequences(self, interactions: pd.DataFrame) -> List[List[int]]:
        """
        Convert interactions DataFrame to sequences.
        
        Args:
            interactions: DataFrame with user interactions
            
        Returns:
            List of sequences, where each sequence is a list of item IDs
        """
        sequences = []
        
        # Group by user and sort by timestamp
        user_groups = interactions.groupby('user_id').apply(
            lambda x: x.sort_values('timestamp')
        ).reset_index(drop=True)
        
        for user_id, group in user_groups.groupby('user_id'):
            sequence = group['item_id'].tolist()
            
            # Filter sequences by length
            if len(sequence) >= self.min_sequence_length:
                if len(sequence) > self.max_sequence_length:
                    sequence = sequence[-self.max_sequence_length:]
                sequences.append(sequence)
        
        logger.info(f"Generated {len(sequences)} sequences")
        return sequences
    
    def split_sequences(self, sequences: List[List[int]], 
                       train_ratio: float = 0.7, 
                       val_ratio: float = 0.15) -> Tuple[List[List[int]], List[List[int]], List[List[int]]]:
        """
        Split sequences into train/validation/test sets.
        
        Args:
            sequences: List of sequences
            train_ratio: Ratio for training set
            val_ratio: Ratio for validation set
            
        Returns:
            Tuple of (train_sequences, val_sequences, test_sequences)
        """
        random.shuffle(sequences)
        
        train_size = int(len(sequences) * train_ratio)
        val_size = int(len(sequences) * val_ratio)
        
        train_sequences = sequences[:train_size]
        val_sequences = sequences[train_size:train_size + val_size]
        test_sequences = sequences[train_size + val_size:]
        
        logger.info(f"Split sequences: {len(train_sequences)} train, "
                   f"{len(val_sequences)} val, {len(test_sequences)} test")
        
        return train_sequences, val_sequences, test_sequences


def generate_sample_data(data_dir: str = "data", config: Optional[DataConfig] = None) -> None:
    """
    Generate sample data for the sequential recommendation system.
    
    Args:
        data_dir: Directory to save data files
        config: Data generation configuration
    """
    if config is None:
        config = DataConfig()
    
    # Create data directory
    data_path = Path(data_dir)
    data_path.mkdir(exist_ok=True)
    
    # Generate data
    generator = SequentialDataGenerator(config)
    loader = DataLoader(data_dir)
    
    interactions = generator.generate_interactions_data()
    items = generator.generate_items_data()
    users = generator.generate_users_data()
    
    # Save data
    loader.save_data(interactions, items, users)
    
    logger.info("Sample data generation completed")


if __name__ == "__main__":
    # Generate sample data
    generate_sample_data()
