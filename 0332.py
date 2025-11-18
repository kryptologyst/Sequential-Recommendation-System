#!/usr/bin/env python3
"""
Project 332: Sequential Recommendation System

This is the original implementation that has been modernized and refactored
into a comprehensive sequential recommendation system.

The modernized version includes:
- Multiple model implementations (Markov Chain, GRU4Rec)
- Comprehensive evaluation metrics
- Interactive demo interface
- Production-ready code structure
- Extensive documentation and testing

To use the modernized system, run:
    python scripts/train.py          # Train and evaluate models
    streamlit run scripts/demo.py   # Launch interactive demo

For more information, see the README.md file.
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

def main():
    """Main entry point for the original sequential recommendation system."""
    print("Sequential Recommendation System - Original Implementation")
    print("=" * 60)
    print()
    print("This is the original implementation. For the modernized version,")
    print("please use the following commands:")
    print()
    print("  python scripts/train.py          # Train and evaluate models")
    print("  streamlit run scripts/demo.py     # Launch interactive demo")
    print()
    print("For more information, see README.md")
    print()
    
    # Original simple implementation
    import numpy as np
    import pandas as pd
    
    print("Running original Markov Chain implementation...")
    
    # 1. Simulate user-item interactions in sequence
    users = ['User1', 'User2', 'User3', 'User4', 'User5']
    items = ['Item1', 'Item2', 'Item3', 'Item4', 'Item5']
    sequences = {
        'User1': ['Item1', 'Item2', 'Item3', 'Item4'],
        'User2': ['Item1', 'Item3', 'Item5'],
        'User3': ['Item2', 'Item4', 'Item5'],
        'User4': ['Item1', 'Item2', 'Item4'],
        'User5': ['Item3', 'Item5']
    }
    
    # 2. Build a transition matrix based on item sequences
    transition_matrix = {item: {next_item: 0 for next_item in items} for item in items}
    
    # Update transition counts based on user interactions
    for user, seq in sequences.items():
        for i in range(len(seq) - 1):
            current_item = seq[i]
            next_item = seq[i+1]
            transition_matrix[current_item][next_item] += 1
    
    # 3. Normalize the transition matrix to get probabilities
    for current_item, next_items in transition_matrix.items():
        total_transitions = sum(next_items.values())
        if total_transitions > 0:
            for next_item in next_items:
                transition_matrix[current_item][next_item] /= total_transitions
    
    # 4. Recommend next item based on sequence
    def recommend_next_item(user, sequences, transition_matrix):
        last_item = sequences[user][-1]
        next_item_probabilities = transition_matrix[last_item]
        
        # Sort items by the highest probability of transition
        sorted_items = sorted(next_item_probabilities.items(), key=lambda x: x[1], reverse=True)
        
        # Recommend the item with the highest transition probability
        recommended_item = sorted_items[0][0]
        return recommended_item
    
    # 5. Recommend the next item for User1
    user = 'User1'
    recommended_item = recommend_next_item(user, sequences, transition_matrix)
    print(f"Recommended next item for {user}: {recommended_item}")
    
    print()
    print("✅ Original implementation completed!")
    print("For the full modernized system, please use the scripts in the scripts/ directory.")


if __name__ == "__main__":
    main()