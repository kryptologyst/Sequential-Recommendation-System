"""
Evaluation metrics for sequential recommendation systems.

This module provides comprehensive evaluation metrics including
precision, recall, NDCG, hit rate, diversity, and coverage metrics.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


class SequentialEvaluator:
    """Evaluator for sequential recommendation models."""
    
    def __init__(self, k_values: List[int] = [5, 10, 20]):
        """
        Initialize the evaluator.
        
        Args:
            k_values: List of k values for evaluation
        """
        self.k_values = k_values
    
    def precision_at_k(self, predictions: List[int], ground_truth: int, k: int) -> float:
        """
        Calculate precision@k for a single prediction.
        
        Args:
            predictions: List of predicted item IDs
            ground_truth: Ground truth item ID
            k: Number of top items to consider
            
        Returns:
            Precision@k score
        """
        top_k = predictions[:k]
        return 1.0 if ground_truth in top_k else 0.0
    
    def recall_at_k(self, predictions: List[int], ground_truth: int, k: int) -> float:
        """
        Calculate recall@k for a single prediction.
        
        Args:
            predictions: List of predicted item IDs
            ground_truth: Ground truth item ID
            k: Number of top items to consider
            
        Returns:
            Recall@k score
        """
        top_k = predictions[:k]
        return 1.0 if ground_truth in top_k else 0.0
    
    def hit_rate_at_k(self, predictions: List[int], ground_truth: int, k: int) -> float:
        """
        Calculate hit rate@k for a single prediction.
        
        Args:
            predictions: List of predicted item IDs
            ground_truth: Ground truth item ID
            k: Number of top items to consider
            
        Returns:
            Hit rate@k score
        """
        return self.precision_at_k(predictions, ground_truth, k)
    
    def ndcg_at_k(self, predictions: List[int], ground_truth: int, k: int) -> float:
        """
        Calculate NDCG@k for a single prediction.
        
        Args:
            predictions: List of predicted item IDs
            ground_truth: Ground truth item ID
            k: Number of top items to consider
            
        Returns:
            NDCG@k score
        """
        top_k = predictions[:k]
        
        if ground_truth not in top_k:
            return 0.0
        
        # Find position of ground truth item
        position = top_k.index(ground_truth) + 1
        
        # Calculate DCG
        dcg = 1.0 / np.log2(position + 1)
        
        # Calculate IDCG (ideal DCG)
        idcg = 1.0 / np.log2(2)  # Since we only have one relevant item
        
        return dcg / idcg
    
    def map_at_k(self, predictions: List[int], ground_truth: int, k: int) -> float:
        """
        Calculate MAP@k for a single prediction.
        
        Args:
            predictions: List of predicted item IDs
            ground_truth: Ground truth item ID
            k: Number of top items to consider
            
        Returns:
            MAP@k score
        """
        top_k = predictions[:k]
        
        if ground_truth not in top_k:
            return 0.0
        
        # Find position of ground truth item
        position = top_k.index(ground_truth) + 1
        
        # Calculate average precision
        return 1.0 / position
    
    def intra_list_diversity(self, predictions: List[int], item_features: Optional[Dict[int, List]] = None) -> float:
        """
        Calculate intra-list diversity for a prediction list.
        
        Args:
            predictions: List of predicted item IDs
            item_features: Optional item features for diversity calculation
            
        Returns:
            Intra-list diversity score
        """
        if len(predictions) <= 1:
            return 0.0
        
        if item_features is None:
            # Use simple Jaccard diversity based on item IDs
            unique_items = len(set(predictions))
            return unique_items / len(predictions)
        
        # Calculate diversity based on item features
        diversity_scores = []
        for i in range(len(predictions)):
            for j in range(i + 1, len(predictions)):
                item1_features = set(item_features.get(predictions[i], []))
                item2_features = set(item_features.get(predictions[j], []))
                
                if len(item1_features) == 0 and len(item2_features) == 0:
                    diversity = 1.0 if predictions[i] != predictions[j] else 0.0
                else:
                    union = len(item1_features.union(item2_features))
                    intersection = len(item1_features.intersection(item2_features))
                    diversity = 1.0 - (intersection / union) if union > 0 else 0.0
                
                diversity_scores.append(diversity)
        
        return np.mean(diversity_scores) if diversity_scores else 0.0
    
    def coverage(self, all_predictions: List[List[int]], all_items: List[int]) -> float:
        """
        Calculate coverage metric.
        
        Args:
            all_predictions: List of prediction lists for all users
            all_items: List of all available items
            
        Returns:
            Coverage score
        """
        recommended_items = set()
        for predictions in all_predictions:
            recommended_items.update(predictions)
        
        return len(recommended_items) / len(all_items)
    
    def popularity_bias(self, all_predictions: List[List[int]], item_popularity: Dict[int, int]) -> float:
        """
        Calculate popularity bias metric.
        
        Args:
            all_predictions: List of prediction lists for all users
            item_popularity: Dictionary mapping item_id to popularity count
            
        Returns:
            Popularity bias score
        """
        if not item_popularity:
            return 0.0
        
        # Calculate average popularity of recommended items
        total_popularity = 0
        total_recommendations = 0
        
        for predictions in all_predictions:
            for item_id in predictions:
                total_popularity += item_popularity.get(item_id, 0)
                total_recommendations += 1
        
        if total_recommendations == 0:
            return 0.0
        
        avg_recommended_popularity = total_popularity / total_recommendations
        
        # Calculate average popularity of all items
        all_popularities = list(item_popularity.values())
        avg_all_popularity = np.mean(all_popularities) if all_popularities else 0
        
        # Calculate bias as ratio
        return avg_recommended_popularity / avg_all_popularity if avg_all_popularity > 0 else 0
    
    def evaluate_model(self, model, test_sequences: List[List[int]], 
                      item_features: Optional[Dict[int, List]] = None,
                      item_popularity: Optional[Dict[int, int]] = None) -> Dict[str, float]:
        """
        Evaluate a model on test sequences.
        
        Args:
            model: Sequential recommendation model
            test_sequences: List of test sequences
            item_features: Optional item features for diversity calculation
            item_popularity: Optional item popularity for bias calculation
            
        Returns:
            Dictionary of evaluation metrics
        """
        metrics = {}
        all_predictions = []
        
        # Calculate basic metrics
        for k in self.k_values:
            precision_scores = []
            recall_scores = []
            hit_rate_scores = []
            ndcg_scores = []
            map_scores = []
            
            for sequence in test_sequences:
                if len(sequence) < 2:
                    continue
                
                # Get ground truth
                ground_truth = sequence[-1]
                
                # Get predictions
                predictions = model.predict_next(sequence[:-1], k=max(self.k_values))
                all_predictions.append(predictions)
                
                # Calculate metrics
                precision_scores.append(self.precision_at_k(predictions, ground_truth, k))
                recall_scores.append(self.recall_at_k(predictions, ground_truth, k))
                hit_rate_scores.append(self.hit_rate_at_k(predictions, ground_truth, k))
                ndcg_scores.append(self.ndcg_at_k(predictions, ground_truth, k))
                map_scores.append(self.map_at_k(predictions, ground_truth, k))
            
            metrics[f'precision@{k}'] = np.mean(precision_scores)
            metrics[f'recall@{k}'] = np.mean(recall_scores)
            metrics[f'hit_rate@{k}'] = np.mean(hit_rate_scores)
            metrics[f'ndcg@{k}'] = np.mean(ndcg_scores)
            metrics[f'map@{k}'] = np.mean(map_scores)
        
        # Calculate diversity metrics
        diversity_scores = []
        for predictions in all_predictions:
            diversity_scores.append(self.intra_list_diversity(predictions, item_features))
        
        metrics['intra_list_diversity'] = np.mean(diversity_scores)
        
        # Calculate coverage
        all_items = list(range(max([max(seq) for seq in test_sequences]) + 1))
        metrics['coverage'] = self.coverage(all_predictions, all_items)
        
        # Calculate popularity bias
        if item_popularity:
            metrics['popularity_bias'] = self.popularity_bias(all_predictions, item_popularity)
        
        return metrics
    
    def compare_models(self, models: Dict[str, any], test_sequences: List[List[int]],
                      item_features: Optional[Dict[int, List]] = None,
                      item_popularity: Optional[Dict[int, int]] = None) -> pd.DataFrame:
        """
        Compare multiple models and return results as a DataFrame.
        
        Args:
            models: Dictionary mapping model names to model objects
            test_sequences: List of test sequences
            item_features: Optional item features for diversity calculation
            item_popularity: Optional item popularity for bias calculation
            
        Returns:
            DataFrame with comparison results
        """
        import pandas as pd
        
        results = {}
        
        for model_name, model in models.items():
            logger.info(f"Evaluating {model_name}...")
            metrics = self.evaluate_model(model, test_sequences, item_features, item_popularity)
            results[model_name] = metrics
        
        # Convert to DataFrame
        df = pd.DataFrame(results).T
        
        # Round to 4 decimal places
        df = df.round(4)
        
        return df


def create_leaderboard(results_df: pd.DataFrame, primary_metric: str = 'ndcg@10') -> pd.DataFrame:
    """
    Create a leaderboard sorted by primary metric.
    
    Args:
        results_df: Results DataFrame from model comparison
        primary_metric: Primary metric to sort by
        
    Returns:
        Sorted leaderboard DataFrame
    """
    if primary_metric not in results_df.columns:
        logger.warning(f"Primary metric {primary_metric} not found, using first available metric")
        primary_metric = results_df.columns[0]
    
    leaderboard = results_df.sort_values(primary_metric, ascending=False)
    return leaderboard


def print_evaluation_summary(results_df: pd.DataFrame) -> None:
    """
    Print a formatted evaluation summary.
    
    Args:
        results_df: Results DataFrame from model comparison
    """
    print("\n" + "="*80)
    print("SEQUENTIAL RECOMMENDATION MODEL EVALUATION RESULTS")
    print("="*80)
    
    # Print main metrics
    main_metrics = [col for col in results_df.columns if any(metric in col for metric in ['precision', 'recall', 'ndcg', 'hit_rate'])]
    
    if main_metrics:
        print("\nMain Recommendation Metrics:")
        print("-" * 40)
        for metric in main_metrics:
            print(f"{metric:20s}: {results_df[metric].max():.4f} (best: {results_df[metric].idxmax()})")
    
    # Print diversity and coverage metrics
    diversity_metrics = [col for col in results_df.columns if any(metric in col for metric in ['diversity', 'coverage', 'bias'])]
    
    if diversity_metrics:
        print("\nDiversity & Coverage Metrics:")
        print("-" * 40)
        for metric in diversity_metrics:
            print(f"{metric:20s}: {results_df[metric].max():.4f} (best: {results_df[metric].idxmax()})")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    # Example usage
    evaluator = SequentialEvaluator(k_values=[5, 10])
    
    # Mock test data
    test_sequences = [
        [1, 2, 3, 4],
        [1, 3, 5, 2],
        [2, 4, 5, 1]
    ]
    
    # Mock model
    class MockModel:
        def predict_next(self, sequence, k):
            return list(range(k))
    
    model = MockModel()
    
    # Evaluate
    metrics = evaluator.evaluate_model(model, test_sequences)
    print("Evaluation metrics:", metrics)
