"""
Base class for code ranking methods
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple
import json
import os

class BaseRanker(ABC):
    """Base class for all code ranking methods"""
    
    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    def rank_candidates(self, problem: Dict[str, Any], candidates: List[str]) -> List[Tuple[str, float]]:
        """
        Rank candidates for a given problem
        
        Args:
            problem: Problem dictionary containing task_id, prompt, etc.
            candidates: List of candidate code strings
            
        Returns:
            List of tuples (candidate_code, score) sorted by score descending
        """
        pass
    
    def process_problem_file(self, problem_file_path: str) -> Dict[str, Any]:
        """Process a single problem file and rank its candidates"""
        with open(problem_file_path, 'r') as f:
            data = json.load(f)
        
        problem = data['problem']
        candidates = data['candidates']
        
        # Rank candidates
        ranked_candidates = self.rank_candidates(problem, candidates)
        
        # Create result structure
        result = {
            'problem': problem,
            'model_name': data.get('model_name'),
            'dataset': data.get('dataset'),
            'ranker_name': self.name,
            'num_candidates': len(candidates),
            'ranked_candidates': ranked_candidates,
            'original_generation_params': data.get('generation_params')
        }
        
        return result
    
    def process_dataset(self, input_dir: str, output_dir: str, dataset_name: str, model_name: str):
        """Process all problems for a given dataset and model"""
        
        model_dataset_dir = os.path.join(input_dir, model_name, dataset_name)
        if not os.path.exists(model_dataset_dir):
            print(f"Directory not found: {model_dataset_dir}")
            return
        
        # Create output directory
        output_model_dir = os.path.join(output_dir, self.name, model_name, dataset_name)
        os.makedirs(output_model_dir, exist_ok=True)
        
        problem_files = [f for f in os.listdir(model_dataset_dir) if f.endswith('.json')]
        problem_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
        
        print(f"Processing {len(problem_files)} problems for {model_name} on {dataset_name} with {self.name}")
        
        for problem_file in problem_files:
            problem_file_path = os.path.join(model_dataset_dir, problem_file)
            
            try:
                result = self.process_problem_file(problem_file_path)
                
                # Save ranked result
                output_file_path = os.path.join(output_model_dir, problem_file)
                with open(output_file_path, 'w') as f:
                    json.dump(result, f, indent=2)
                
                problem_id = result['problem'].get('problem_id', 'unknown')
                print(f"  ✓ Problem {problem_id}: Ranked {len(result['ranked_candidates'])} candidates")
                
            except Exception as e:
                print(f"  ✗ Error processing {problem_file}: {e}")
                continue