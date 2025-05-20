from metrics import RecommendationEvaluator
import numpy as np
from typing import Dict, List, Tuple

def generate_sample_data() -> Tuple[Dict[int, List[Tuple[int, float]]], Dict[int, List[int]]]:
    """
    Generate sample data for demonstration
    
    Returns:
        Tuple of (user_predictions, test_interactions)
    """
    # Sample user predictions (user_id -> [(item_id, score)])
    user_predictions = {
        1: [(1, 0.9), (2, 0.8), (3, 0.7), (4, 0.6), (5, 0.5)],
        2: [(3, 0.95), (1, 0.85), (4, 0.75), (2, 0.65), (6, 0.55)],
        3: [(2, 0.88), (4, 0.77), (1, 0.66), (5, 0.55), (3, 0.44)]
    }
    
    # Sample test interactions (user_id -> [item_id])
    test_interactions = {
        1: [1, 3, 5],  # User 1 actually interacted with items 1, 3, and 5
        2: [3, 4, 6],  # User 2 actually interacted with items 3, 4, and 6
        3: [2, 4, 5]   # User 3 actually interacted with items 2, 4, and 5
    }
    
    return user_predictions, test_interactions

def main():
    # Initialize evaluator
    evaluator = RecommendationEvaluator()
    
    # Generate sample data
    user_predictions, test_interactions = generate_sample_data()
    
    # Evaluate recommendations at different k values
    k_values = [1, 3, 5]
    
    print("Evaluation Results:")
    print("-" * 50)
    
    for k in k_values:
        print(f"\nMetrics at k={k}:")
        metrics = evaluator.evaluate_ranking(user_predictions, test_interactions, k)
        
        for metric_name, score in metrics.items():
            print(f"{metric_name}: {score:.4f}")
            
    # Example of direct metric calculation
    print("\nDirect metric calculation example:")
    actual = [1, 3, 5]
    predicted = [1, 2, 3, 4, 5]
    k = 3
    
    precision = evaluator.precision_at_k(actual, predicted, k)
    recall = evaluator.recall_at_k(actual, predicted, k)
    ndcg = evaluator.ndcg_at_k(actual, predicted, k)
    map_score = evaluator.map_at_k(actual, predicted, k)
    
    print(f"For a single user:")
    print(f"Precision@{k}: {precision:.4f}")
    print(f"Recall@{k}: {recall:.4f}")
    print(f"NDCG@{k}: {ndcg:.4f}")
    print(f"MAP@{k}: {map_score:.4f}")

if __name__ == "__main__":
    main() 