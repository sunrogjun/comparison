"""
Evaluation metrics for ranking method comparison
"""
import math
from typing import List, Dict, Any, Tuple
import numpy as np

class RankingMetrics:
    """Calculate various metrics for ranking evaluation following AceCoder and LLM-as-a-Judge methodologies"""
    
    @staticmethod
    def calculate_pass_at_k(execution_results: List[Dict[str, Any]], k: int) -> float:
        """
        Calculate Pass@K metric
        
        Args:
            execution_results: List of execution results (in ranked order)
            k: Number of top candidates to consider
            
        Returns:
            Pass@K score (0.0 to 1.0)
        """
        if not execution_results or k <= 0:
            return 0.0
        
        # Consider only top k candidates
        top_k_results = execution_results[:k]
        
        # Check if any of the top k passed
        passed = any(result.get('passed', False) for result in top_k_results)
        return 1.0 if passed else 0.0
    
    @staticmethod
    def calculate_pass_at_k_batch(all_results: List[List[Dict[str, Any]]], k: int) -> float:
        """
        Calculate Pass@K across multiple problems
        
        Args:
            all_results: List of execution results for each problem
            k: Number of top candidates to consider
            
        Returns:
            Average Pass@K across all problems
        """
        if not all_results:
            return 0.0
        
        pass_at_k_scores = []
        for problem_results in all_results:
            pass_at_k = RankingMetrics.calculate_pass_at_k(problem_results, k)
            pass_at_k_scores.append(pass_at_k)
        
        return np.mean(pass_at_k_scores)
    
    @staticmethod
    def calculate_first_correct_position(execution_results: List[Dict[str, Any]]) -> int:
        """
        Find the position (1-indexed) of the first correct solution
        
        Returns:
            Position of first correct solution, or len(results)+1 if none found
        """
        for i, result in enumerate(execution_results):
            if result.get('passed', False):
                return i + 1  # 1-indexed position
        
        return len(execution_results) + 1  # No correct solution found
    
    @staticmethod
    def calculate_mrr(all_results: List[List[Dict[str, Any]]]) -> float:
        """
        Calculate Mean Reciprocal Rank
        
        Args:
            all_results: List of execution results for each problem
            
        Returns:
            Mean Reciprocal Rank (0.0 to 1.0)
        """
        if not all_results:
            return 0.0
        
        reciprocal_ranks = []
        for problem_results in all_results:
            first_correct_pos = RankingMetrics.calculate_first_correct_position(problem_results)
            if first_correct_pos <= len(problem_results):
                rr = 1.0 / first_correct_pos
            else:
                rr = 0.0
            reciprocal_ranks.append(rr)
        
        return np.mean(reciprocal_ranks)
    
    @staticmethod
    def calculate_ndcg_at_k(execution_results: List[Dict[str, Any]], k: int) -> float:
        """
        Calculate NDCG@K (Normalized Discounted Cumulative Gain)
        
        Args:
            execution_results: List of execution results (in ranked order)
            k: Number of top candidates to consider
            
        Returns:
            NDCG@K score (0.0 to 1.0)
        """
        if not execution_results or k <= 0:
            return 0.0
        
        # Consider only top k candidates
        top_k_results = execution_results[:k]
        
        # Calculate DCG
        dcg = 0.0
        for i, result in enumerate(top_k_results):
            relevance = 1.0 if result.get('passed', False) else 0.0
            dcg += relevance / math.log2(i + 2)  # i+2 because log2(1) = 0
        
        # Calculate IDCG (ideal DCG) - assume all correct solutions come first
        correct_count = sum(1 for result in execution_results if result.get('passed', False))
        ideal_positions = min(k, correct_count)
        
        idcg = 0.0
        for i in range(ideal_positions):
            idcg += 1.0 / math.log2(i + 2)
        
        # Return NDCG
        if idcg == 0:
            return 0.0
        return dcg / idcg
    
    @staticmethod
    def calculate_bradley_terry_agreement(scores1: List[float], scores2: List[float], 
                                        candidates1: List[str], candidates2: List[str]) -> float:
        """
        Calculate agreement between two ranking methods using Bradley-Terry model
        Following AceCoder's preference pair methodology
        
        Args:
            scores1: Scores from first ranking method
            scores2: Scores from second ranking method  
            candidates1: Candidates ranked by first method
            candidates2: Candidates ranked by second method
            
        Returns:
            Agreement score (0.0 to 1.0)
        """
        if len(scores1) != len(scores2) or len(scores1) == 0:
            return 0.0
        
        # Create preference pairs and check agreement
        agreements = 0
        total_pairs = 0
        
        for i in range(len(scores1)):
            for j in range(i + 1, len(scores1)):
                # Check if both methods agree on preference direction
                method1_prefers_i = scores1[i] > scores1[j]
                method2_prefers_i = scores2[i] > scores2[j]
                
                if method1_prefers_i == method2_prefers_i:
                    agreements += 1
                total_pairs += 1
        
        return agreements / total_pairs if total_pairs > 0 else 0.0
    
    @staticmethod  
    def calculate_reward_model_correlation(rm_scores: List[float], execution_results: List[Dict[str, Any]]) -> float:
        """
        Calculate correlation between reward model scores and execution results
        Following AceCoder's reward model evaluation methodology
        
        Args:
            rm_scores: Reward model scores
            execution_results: Execution results with pass/fail information
            
        Returns:
            Spearman correlation coefficient
        """
        if len(rm_scores) != len(execution_results) or len(rm_scores) == 0:
            return 0.0
            
        # Convert execution results to binary scores
        binary_scores = [1.0 if result.get('passed', False) else 0.0 for result in execution_results]
        
        # Calculate Spearman correlation
        try:
            from scipy.stats import spearmanr
            correlation, _ = spearmanr(rm_scores, binary_scores)
            return correlation if not math.isnan(correlation) else 0.0
        except ImportError:
            # Fallback to simple ranking correlation if scipy not available
            rm_ranks = [sorted(rm_scores, reverse=True).index(score) for score in rm_scores]
            binary_ranks = [sorted(binary_scores, reverse=True).index(score) for score in binary_scores]
            
            # Calculate Pearson correlation of ranks
            if len(set(rm_ranks)) == 1 or len(set(binary_ranks)) == 1:
                return 0.0
            
            mean_rm = np.mean(rm_ranks)
            mean_binary = np.mean(binary_ranks)
            
            numerator = sum((rm_ranks[i] - mean_rm) * (binary_ranks[i] - mean_binary) for i in range(len(rm_ranks)))
            denominator = math.sqrt(sum((r - mean_rm)**2 for r in rm_ranks) * sum((r - mean_binary)**2 for r in binary_ranks))
            
            return numerator / denominator if denominator != 0 else 0.0
    
    @staticmethod
    def calculate_llm_judge_consistency(scores1: List[float], scores2: List[float]) -> float:
        """
        Calculate consistency between multiple LLM judge evaluations
        Following LLM-as-a-Judge best practices for reliability assessment
        
        Args:
            scores1: First round of LLM judge scores
            scores2: Second round of LLM judge scores
            
        Returns:
            Consistency score (0.0 to 1.0)
        """
        if len(scores1) != len(scores2) or len(scores1) == 0:
            return 0.0
        
        # Calculate mean absolute difference normalized by score range
        differences = [abs(s1 - s2) for s1, s2 in zip(scores1, scores2)]
        mean_diff = np.mean(differences)
        
        # Normalize by maximum possible difference (assuming 0-10 scale)
        max_diff = 10.0
        consistency = 1.0 - (mean_diff / max_diff)
        
        return max(0.0, consistency)

    @staticmethod
    def calculate_ranking_quality_metrics(all_results: List[List[Dict[str, Any]]]) -> Dict[str, float]:
        """
        Calculate comprehensive ranking quality metrics
        
        Args:
            all_results: List of execution results for each problem (in ranked order)
            
        Returns:
            Dictionary containing various metrics
        """
        metrics = {}
        
        # Pass@K for different values of K
        for k in [1, 5, 10, 20, 50, 100]:
            pass_at_k = RankingMetrics.calculate_pass_at_k_batch(all_results, k)
            metrics[f'pass_at_{k}'] = pass_at_k
        
        # Mean Reciprocal Rank
        metrics['mrr'] = RankingMetrics.calculate_mrr(all_results)
        
        # NDCG@K for different values of K
        ndcg_scores = {}
        for k in [5, 10, 20]:
            ndcg_scores_k = []
            for problem_results in all_results:
                ndcg_k = RankingMetrics.calculate_ndcg_at_k(problem_results, k)
                ndcg_scores_k.append(ndcg_k)
            ndcg_scores[k] = np.mean(ndcg_scores_k)
            metrics[f'ndcg_at_{k}'] = ndcg_scores[k]
        
        # Average position of first correct solution
        first_correct_positions = []
        for problem_results in all_results:
            pos = RankingMetrics.calculate_first_correct_position(problem_results)
            first_correct_positions.append(pos)
        
        metrics['avg_first_correct_position'] = np.mean(first_correct_positions)
        metrics['median_first_correct_position'] = np.median(first_correct_positions)
        
        # Success rate (problems with at least one correct solution in top 100)
        success_count = sum(1 for results in all_results 
                          if any(r.get('passed', False) for r in results))
        metrics['success_rate'] = success_count / len(all_results) if all_results else 0.0
        
        return metrics
    
    @staticmethod
    def compare_ranking_methods(method_results: Dict[str, List[List[Dict[str, Any]]]]) -> Dict[str, Dict[str, float]]:
        """
        Compare multiple ranking methods
        
        Args:
            method_results: Dict mapping method names to their results
            
        Returns:
            Dict mapping method names to their metrics
        """
        comparison = {}
        
        for method_name, results in method_results.items():
            metrics = RankingMetrics.calculate_ranking_quality_metrics(results)
            comparison[method_name] = metrics
        
        return comparison
    
    @staticmethod
    def print_comparison_table(comparison_results: Dict[str, Dict[str, float]]):
        """Print a formatted comparison table"""
        if not comparison_results:
            print("No results to compare")
            return
        
        methods = list(comparison_results.keys())
        metrics = list(comparison_results[methods[0]].keys())
        
        # Print header
        print(f"{'Method':<20}", end='')
        for metric in metrics:
            print(f"{metric:<12}", end='')
        print()
        
        print("-" * (20 + 12 * len(metrics)))
        
        # Print results for each method
        for method in methods:
            print(f"{method:<20}", end='')
            for metric in metrics:
                value = comparison_results[method][metric]
                if 'pass_at' in metric or 'ndcg' in metric or 'mrr' in metric or 'success_rate' in metric:
                    print(f"{value:.4f}    ", end='')
                else:
                    print(f"{value:.2f}     ", end='')
            print()