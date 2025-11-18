"""
Main training script for sequential recommendation system.

This script demonstrates how to train and evaluate different sequential
recommendation models on generated or real data.
"""

import sys
import os
from pathlib import Path
import logging
import yaml
import pandas as pd
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.sequential_models import MarkovChainModel, GRU4RecModel, Config, set_random_seeds
from src.data.data_pipeline import SequentialDataGenerator, DataLoader, SequenceProcessor, DataConfig
from src.utils.evaluation import SequentialEvaluator, print_evaluation_summary

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_config(config_path: str = "configs/config.yaml") -> Config:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration object
    """
    try:
        config = Config.from_yaml(config_path)
        logger.info(f"Loaded configuration from {config_path}")
        return config
    except Exception as e:
        logger.warning(f"Failed to load config from {config_path}: {e}")
        logger.info("Using default configuration")
        return Config()


def generate_data_if_needed(data_dir: str = "data", config: DataConfig = None) -> None:
    """
    Generate sample data if it doesn't exist.
    
    Args:
        data_dir: Directory to check for data
        config: Data generation configuration
    """
    data_path = Path(data_dir)
    
    if not data_path.exists() or not any(data_path.glob("*.csv")):
        logger.info("No data found, generating sample data...")
        from src.data.data_pipeline import generate_sample_data
        generate_sample_data(data_dir, config)
    else:
        logger.info("Data files found, skipping data generation")


def load_and_process_data(config: Config) -> tuple:
    """
    Load and process data for training.
    
    Args:
        config: Configuration object
        
    Returns:
        Tuple of (train_sequences, val_sequences, test_sequences, items_df, users_df)
    """
    # Load data
    loader = DataLoader("data")
    interactions_df = loader.load_interactions()
    items_df = loader.load_items()
    users_df = loader.load_users()
    
    # Process sequences
    processor = SequenceProcessor(
        min_sequence_length=config.min_sequence_length,
        max_sequence_length=config.max_sequence_length
    )
    
    sequences = processor.interactions_to_sequences(interactions_df)
    train_sequences, val_sequences, test_sequences = processor.split_sequences(
        sequences, config.train_ratio, config.val_ratio
    )
    
    logger.info(f"Data loaded: {len(train_sequences)} train, {len(val_sequences)} val, {len(test_sequences)} test sequences")
    
    return train_sequences, val_sequences, test_sequences, items_df, users_df


def train_models(train_sequences: list, config: Config) -> dict:
    """
    Train all models on the training sequences.
    
    Args:
        train_sequences: Training sequences
        config: Configuration object
        
    Returns:
        Dictionary of trained models
    """
    models = {}
    
    # Determine number of unique items
    all_items = set()
    for seq in train_sequences:
        all_items.update(seq)
    num_items = len(all_items)
    
    logger.info(f"Training models with {num_items} unique items")
    
    # Train Markov Chain model
    logger.info("Training Markov Chain model...")
    markov_model = MarkovChainModel(order=1, smoothing=0.1)
    markov_model.fit(train_sequences)
    models['Markov Chain'] = markov_model
    
    # Train GRU4Rec model
    logger.info("Training GRU4Rec model...")
    gru_model = GRU4RecModel(
        num_items=num_items,
        hidden_size=128,
        num_layers=2,
        dropout=0.25,
        learning_rate=0.001,
        batch_size=256,
        epochs=20,  # Reduced for demo
        device=config.device
    )
    gru_model.fit(train_sequences)
    models['GRU4Rec'] = gru_model
    
    return models


def evaluate_models(models: dict, test_sequences: list, items_df: pd.DataFrame) -> pd.DataFrame:
    """
    Evaluate all models on test sequences.
    
    Args:
        models: Dictionary of trained models
        test_sequences: Test sequences
        items_df: Items DataFrame for diversity calculation
        
    Returns:
        DataFrame with evaluation results
    """
    evaluator = SequentialEvaluator(k_values=[5, 10, 20])
    
    # Prepare item features for diversity calculation
    item_features = {}
    for _, row in items_df.iterrows():
        item_features[row['item_id']] = [row['category'], row['features']]
    
    # Calculate item popularity
    item_popularity = {}
    for seq in test_sequences:
        for item in seq:
            item_popularity[item] = item_popularity.get(item, 0) + 1
    
    # Evaluate models
    results_df = evaluator.compare_models(
        models, test_sequences, item_features, item_popularity
    )
    
    return results_df


def save_results(results_df: pd.DataFrame, output_dir: str = "results") -> None:
    """
    Save evaluation results to files.
    
    Args:
        results_df: Results DataFrame
        output_dir: Output directory
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Save as CSV
    csv_path = output_path / "evaluation_results.csv"
    results_df.to_csv(csv_path)
    logger.info(f"Results saved to {csv_path}")
    
    # Save as markdown table
    md_path = output_path / "evaluation_results.md"
    with open(md_path, 'w') as f:
        f.write("# Sequential Recommendation Model Evaluation Results\n\n")
        f.write(results_df.to_markdown())
    logger.info(f"Markdown results saved to {md_path}")


def main():
    """Main training and evaluation pipeline."""
    logger.info("Starting Sequential Recommendation System Training")
    
    # Set random seeds for reproducibility
    set_random_seeds(42)
    
    # Load configuration
    config = load_config()
    
    # Generate data if needed
    data_config = DataConfig(
        num_users=500,  # Smaller dataset for demo
        num_items=50,
        min_sequence_length=config.min_sequence_length,
        max_sequence_length=config.max_sequence_length
    )
    generate_data_if_needed("data", data_config)
    
    # Load and process data
    train_sequences, val_sequences, test_sequences, items_df, users_df = load_and_process_data(config)
    
    # Train models
    models = train_models(train_sequences, config)
    
    # Evaluate models
    logger.info("Evaluating models...")
    results_df = evaluate_models(models, test_sequences, items_df)
    
    # Print results
    print_evaluation_summary(results_df)
    
    # Save results
    save_results(results_df)
    
    # Demonstrate recommendations
    logger.info("\nDemonstrating recommendations:")
    test_sequence = train_sequences[0][:-1] if train_sequences else [1, 2, 3]
    logger.info(f"Input sequence: {test_sequence}")
    
    for model_name, model in models.items():
        recommendations = model.predict_next(test_sequence, k=5)
        logger.info(f"{model_name} recommendations: {recommendations}")
    
    logger.info("Training and evaluation completed successfully!")


if __name__ == "__main__":
    main()
