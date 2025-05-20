import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from typing import List, Dict, Union, Tuple

class RecommendationEvaluator:
    def __init__(self):
        pass

    @staticmethod
    def precision_at_k(actual: List[int], predicted: List[int], k: int) -> float:
        """
        Calculate precision@k
        
        Args:
            actual: List of actual relevant items
            predicted: List of predicted items
            k: Number of recommendations to consider
            
        Returns:
            Precision@k score
        """
        if len(predicted) > k:
            predicted = predicted[:k]
        
        if not actual or not predicted:
            return 0.0
            
        relevant_count = len(set(actual) & set(predicted))
        return relevant_count / min(k, len(predicted))

    @staticmethod
    def recall_at_k(actual: List[int], predicted: List[int], k: int) -> float:
        """
        Calculate recall@k
        
        Args:
            actual: List of actual relevant items
            predicted: List of predicted items
            k: Number of recommendations to consider
            
        Returns:
            Recall@k score
        """
        if len(predicted) > k:
            predicted = predicted[:k]
            
        if not actual or not predicted:
            return 0.0
            
        relevant_count = len(set(actual) & set(predicted))
        return relevant_count / len(actual) if len(actual) > 0 else 0.0

    @staticmethod
    def ndcg_at_k(actual: List[int], predicted: List[int], k: int) -> float:
        """
        Calculate NDCG@k (Normalized Discounted Cumulative Gain)
        
        Args:
            actual: List of actual relevant items
            predicted: List of predicted items
            k: Number of recommendations to consider
            
        Returns:
            NDCG@k score
        """
        if len(predicted) > k:
            predicted = predicted[:k]
            
        if not actual or not predicted:
            return 0.0
            
        dcg = 0.0
        idcg = 0.0
        
        # Calculate DCG
        for i, item in enumerate(predicted):
            if item in actual:
                dcg += 1 / np.log2(i + 2)  # i+2 because i starts from 0
                
        # Calculate IDCG
        for i in range(min(len(actual), k)):
            idcg += 1 / np.log2(i + 2)
            
        return dcg / idcg if idcg > 0 else 0.0

    @staticmethod
    def map_at_k(actual: List[int], predicted: List[int], k: int) -> float:
        """
        Calculate MAP@k (Mean Average Precision)
        
        Args:
            actual: List of actual relevant items
            predicted: List of predicted items
            k: Number of recommendations to consider
            
        Returns:
            MAP@k score
        """
        if len(predicted) > k:
            predicted = predicted[:k]
            
        if not actual or not predicted:
            return 0.0
            
        running_sum = 0.0
        num_hits = 0
        
        for i, p in enumerate(predicted):
            if p in actual and p not in predicted[:i]:
                num_hits += 1
                running_sum += num_hits / (i + 1)
                
        return running_sum / min(len(actual), k)

    def evaluate_recommendations(self, 
                              actual_lists: List[List[int]], 
                              predicted_lists: List[List[int]], 
                              k: int) -> Dict[str, float]:
        """
        Evaluate recommendations using multiple metrics
        
        Args:
            actual_lists: List of lists containing actual relevant items for each user
            predicted_lists: List of lists containing predicted items for each user
            k: Number of recommendations to consider
            
        Returns:
            Dictionary containing evaluation metrics
        """
        if len(actual_lists) != len(predicted_lists):
            raise ValueError("Number of actual and predicted lists must be equal")
            
        precision_scores = []
        recall_scores = []
        ndcg_scores = []
        map_scores = []
        
        for actual, predicted in zip(actual_lists, predicted_lists):
            precision_scores.append(self.precision_at_k(actual, predicted, k))
            recall_scores.append(self.recall_at_k(actual, predicted, k))
            ndcg_scores.append(self.ndcg_at_k(actual, predicted, k))
            map_scores.append(self.map_at_k(actual, predicted, k))
            
        return {
            f'precision@{k}': np.mean(precision_scores),
            f'recall@{k}': np.mean(recall_scores),
            f'ndcg@{k}': np.mean(ndcg_scores),
            f'map@{k}': np.mean(map_scores)
        }

    def evaluate_ranking(self, 
                        user_predictions: Dict[int, List[Tuple[int, float]]], 
                        test_interactions: Dict[int, List[int]], 
                        k: int) -> Dict[str, float]:
        """
        Evaluate ranking predictions
        
        Args:
            user_predictions: Dictionary of user_id to list of (item_id, score) tuples
            test_interactions: Dictionary of user_id to list of actual interacted items
            k: Number of recommendations to consider
            
        Returns:
            Dictionary containing evaluation metrics
        """
        actual_lists = []
        predicted_lists = []
        
        for user_id in user_predictions:
            if user_id in test_interactions:
                # Sort predictions by score and get top k items
                sorted_predictions = sorted(user_predictions[user_id], 
                                         key=lambda x: x[1], 
                                         reverse=True)
                predicted_items = [item[0] for item in sorted_predictions[:k]]
                actual_items = test_interactions[user_id]
                
                actual_lists.append(actual_items)
                predicted_lists.append(predicted_items)
                
        return self.evaluate_recommendations(actual_lists, predicted_lists, k) 