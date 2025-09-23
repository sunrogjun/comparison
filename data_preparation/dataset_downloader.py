"""
Setup unified evaluation datasets for RankEF, LLM-as-a-Judge, and AceCoder comparison
"""
import os
import json
import requests
try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    print("Warning: datasets library not available, using sample data")
    DATASETS_AVAILABLE = False
import pickle

class DatasetManager:
    def __init__(self, data_dir="/home/fdse/srj/comparison/data"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
        # Create subdirectories for each dataset
        self.humaneval_dir = os.path.join(data_dir, "humaneval")
        self.mbpp_dir = os.path.join(data_dir, "mbpp") 
        self.apps_dir = os.path.join(data_dir, "apps")
        
        for dir_path in [self.humaneval_dir, self.mbpp_dir, self.apps_dir]:
            os.makedirs(dir_path, exist_ok=True)
    
    def download_humaneval(self):
        """Download and setup HumanEval dataset"""
        print("Setting up HumanEval dataset...")
        
        try:
            # Load HumanEval from huggingface datasets
            dataset = load_dataset("openai_humaneval", split="test")
            humaneval_problems = []
            
            for i, problem in enumerate(dataset):
                problem_data = {
                    "task_id": problem["task_id"],
                    "prompt": problem["prompt"], 
                    "canonical_solution": problem["canonical_solution"],
                    "test": problem["test"],
                    "entry_point": problem["entry_point"],
                    "docstring": problem.get("docstring", ""),
                    "problem_id": i
                }
                humaneval_problems.append(problem_data)
                
                # Save individual problem file
                with open(os.path.join(self.humaneval_dir, f"{i}.json"), "w") as f:
                    json.dump(problem_data, f, indent=2)
            
            # Save complete dataset
            with open(os.path.join(self.humaneval_dir, "humaneval_complete.json"), "w") as f:
                json.dump(humaneval_problems, f, indent=2)
                
            print(f"HumanEval: {len(humaneval_problems)} problems downloaded")
            return humaneval_problems
            
        except Exception as e:
            print(f"Error downloading HumanEval: {e}")
            # Fallback to manual data if needed
            return self._create_sample_humaneval()
    
    def download_mbpp(self):
        """Download and setup MBPP dataset"""
        print("Setting up MBPP dataset...")
        
        try:
            # Load MBPP from huggingface datasets
            dataset = load_dataset("mbpp", split="test")
            mbpp_problems = []
            
            for i, problem in enumerate(dataset):
                # 检查必需字段是否存在
                if "text" not in problem:
                    print(f"Warning: Problem {i} missing 'text' field, skipping...")
                    continue
                    
                if "code" not in problem:
                    print(f"Warning: Problem {i} missing 'code' field, skipping...")
                    continue


                problem_data = {
                    "task_id": f"MBPP/{problem['task_id']}",
                    "prompt": problem["text"],
                    "canonical_solution": problem["code"],
                    "test_list": problem["test_list"],
                    "challenge_test_list": problem.get("challenge_test_list", []),
                    "problem_id": i
                }
                mbpp_problems.append(problem_data)
                
                # Save individual problem file
                with open(os.path.join(self.mbpp_dir, f"{i}.json"), "w") as f:
                    json.dump(problem_data, f, indent=2)
            
            # Save complete dataset
            with open(os.path.join(self.mbpp_dir, "mbpp_complete.json"), "w") as f:
                json.dump(mbpp_problems, f, indent=2)
                
            print(f"MBPP: {len(mbpp_problems)} problems downloaded")
            return mbpp_problems
            
        except Exception as e:
            print(f"Error downloading MBPP: {e}")
            return self._create_sample_mbpp()
    
    def setup_apps_subset(self):
        """Setup APPS test subset for comparison"""
        print("Setting up APPS test subset...")
        
        try:
            # Download APPS dataset from huggingface datasets (test split - validation subset)
            dataset = load_dataset("codeparrot/apps", split="test")
            apps_problems = []
            
            # Use 598 validation problems as mentioned in RankEF paper
            max_problems = min(598, len(dataset))
            print(f"Loading {max_problems} APPS validation problems...")
            
            for i in range(max_problems):
                problem = dataset[i]
                
                problem_data = {
                    "task_id": f"APPS/{i}",
                    "problem_id": i,
                    "prompt": problem["question"],
                    "solutions": problem.get("solutions", []),
                    "input_output": problem.get("input_output", ""),
                    "difficulty": problem.get("difficulty", ""),
                    "url": problem.get("url", ""),
                    "starter_code": problem.get("starter_code", "")
                }
                apps_problems.append(problem_data)
                
                # Save individual problem file
                with open(os.path.join(self.apps_dir, f"{i}.json"), "w") as f:
                    json.dump(problem_data, f, indent=2)
            
            # Save complete dataset
            with open(os.path.join(self.apps_dir, "apps_complete.json"), "w") as f:
                json.dump(apps_problems, f, indent=2)
                
            print(f"APPS: {len(apps_problems)} problems downloaded from validation set")
            return apps_problems
            
        except Exception as e:
            print(f"Error downloading APPS dataset: {e}")
            print("Falling back to sample dataset")
            return self._create_sample_apps()
    
    def _convert_rankef_apps_format(self, apps_path):
        """Convert RankEF APPS format to unified format"""
        apps_problems = []
        
        try:
            prob_list = os.listdir(apps_path)
            prob_list = sorted(prob_list)[:50]  # Limit for comparison
            
            for i, fname in enumerate(prob_list):
                prob_id = int(fname)
                prob_path = os.path.join(apps_path, fname)
                
                # Read problem question
                question_path = os.path.join(prob_path, 'question.txt')
                if os.path.exists(question_path):
                    with open(question_path, 'r') as f:
                        question = f.read()
                        
                    # Read solutions if available
                    solutions_path = os.path.join(prob_path, 'solutions.json')
                    solutions = []
                    if os.path.exists(solutions_path):
                        with open(solutions_path, 'r') as f:
                            solutions_data = json.load(f)
                            solutions = solutions_data if isinstance(solutions_data, list) else [solutions_data]
                    
                    problem_data = {
                        "task_id": f"APPS/{prob_id}",
                        "problem_id": prob_id,
                        "prompt": question,
                        "solutions": solutions,
                        "original_index": i
                    }
                    apps_problems.append(problem_data)
                    
                    # Save individual problem file
                    with open(os.path.join(self.apps_dir, f"{prob_id}.json"), "w") as f:
                        json.dump(problem_data, f, indent=2)
            
            # Save complete dataset
            with open(os.path.join(self.apps_dir, "apps_complete.json"), "w") as f:
                json.dump(apps_problems, f, indent=2)
                
        except Exception as e:
            print(f"Error converting APPS format: {e}")
            apps_problems = self._create_sample_apps()
        
        print(f"APPS: {len(apps_problems)} problems prepared")
        return apps_problems
    
    def _create_sample_humaneval(self):
        """Create sample HumanEval problems for testing"""
        sample_problems = [
            {
                "task_id": "HumanEval/0",
                "prompt": "def has_close_elements(numbers, threshold):\n    \"\"\"\n    Check if in given list of numbers, are any two numbers closer to each other than given threshold.\n    \"\"\"",
                "canonical_solution": "    for idx, elem in enumerate(numbers):\n        for idx2, elem2 in enumerate(numbers):\n            if idx != idx2:\n                distance = abs(elem - elem2)\n                if distance < threshold:\n                    return True\n    return False",
                "test": "def check(candidate):\n    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.3) == True\n    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.05) == False",
                "entry_point": "has_close_elements",
                "problem_id": 0
            },
            {
                "task_id": "HumanEval/1",
                "prompt": "def separate_paren_groups(paren_string):\n    \"\"\"\n    Input is a string consisting of multiple groups of nested parentheses.\n    \"\"\"",
                "canonical_solution": "    result = []\n    current_string = []\n    current_depth = 0\n    \n    for c in paren_string:\n        if c == '(':\n            current_depth += 1\n            current_string.append(c)\n        elif c == ')':\n            current_depth -= 1\n            current_string.append(c)\n            \n            if current_depth == 0:\n                result.append(''.join(current_string))\n                current_string = []\n    \n    return result",
                "test": "def check(candidate):\n    assert candidate('( ) (( )) (( )( ))') == ['()', '(())', '(()())']",
                "entry_point": "separate_paren_groups",
                "problem_id": 1
            }
        ]
        
        # Save sample problems
        for problem in sample_problems:
            with open(os.path.join(self.humaneval_dir, f"{problem['problem_id']}.json"), "w") as f:
                json.dump(problem, f, indent=2)
        
        with open(os.path.join(self.humaneval_dir, "humaneval_complete.json"), "w") as f:
            json.dump(sample_problems, f, indent=2)
        
        return sample_problems
    
    def _create_sample_mbpp(self):
        """Create sample MBPP problems for testing"""
        sample_problems = [
            {
                "task_id": "MBPP/1",
                "prompt": "Write a function to find the minimum cost path to reach (m, n) from (0, 0) for the given cost matrix cost[][] and a position (m, n) in cost[][].",
                "canonical_solution": "def min_cost(cost, m, n): \n\tR = 3\n\tC = 3\n\ttc = [[0 for x in range(C)] for x in range(R)] \n\ttc[0][0] = cost[0][0] \n\tfor i in range(1, m + 1): \n\t\ttc[i][0] = tc[i-1][0] + cost[i][0] \n\tfor j in range(1, n + 1): \n\t\ttc[0][j] = tc[0][j-1] + cost[0][j] \n\tfor i in range(1, m + 1): \n\t\tfor j in range(1, n + 1): \n\t\t\ttc[i][j] = min(tc[i-1][j-1], tc[i-1][j], tc[i][j-1]) + cost[i][j] \n\treturn tc[m][n]",
                "test_list": ["assert min_cost([[1, 2, 3], [4, 8, 2], [1, 5, 3]], 2, 2) == 8"],
                "problem_id": 0
            }
        ]
        
        # Save sample problems
        for problem in sample_problems:
            with open(os.path.join(self.mbpp_dir, f"{problem['problem_id']}.json"), "w") as f:
                json.dump(problem, f, indent=2)
        
        with open(os.path.join(self.mbpp_dir, "mbpp_complete.json"), "w") as f:
            json.dump(sample_problems, f, indent=2)
        
        return sample_problems
    
    def _create_sample_apps(self):
        """Create sample APPS problems for testing"""
        sample_problems = [
            {
                "task_id": "APPS/0",
                "problem_id": 0,
                "prompt": "Given an array of integers, return the maximum sum of any subarray.",
                "solutions": ["def max_subarray_sum(arr):\n    max_sum = float('-inf')\n    current_sum = 0\n    for num in arr:\n        current_sum = max(num, current_sum + num)\n        max_sum = max(max_sum, current_sum)\n    return max_sum"]
            }
        ]
        
        # Save sample problems
        for problem in sample_problems:
            with open(os.path.join(self.apps_dir, f"{problem['problem_id']}.json"), "w") as f:
                json.dump(problem, f, indent=2)
        
        with open(os.path.join(self.apps_dir, "apps_complete.json"), "w") as f:
            json.dump(sample_problems, f, indent=2)
        
        return sample_problems
    
    def create_unified_test_set(self):
        """Create unified test set across all datasets"""
        print("Creating unified test set...")
        
        # Setup all datasets
        humaneval_data = self.download_humaneval()
        mbpp_data = self.download_mbpp() 
        apps_data = self.setup_apps_subset()
        
        unified_dataset = {
            "humaneval": humaneval_data,
            "mbpp": mbpp_data,
            "apps": apps_data
        }
        
        # Save unified dataset
        with open(os.path.join(self.data_dir, "unified_test_set.json"), "w") as f:
            json.dump(unified_dataset, f, indent=2)
        
        print("Unified test set created successfully!")
        return unified_dataset

if __name__ == "__main__":
    print("Setting up evaluation datasets...")
    
    dataset_manager = DatasetManager()
    unified_dataset = dataset_manager.create_unified_test_set()
    
    # Print summary
    print("\n=== Dataset Summary ===")
    for dataset_name, problems in unified_dataset.items():
        print(f"{dataset_name}: {len(problems)} problems")
    
    print(f"\nData saved to: {dataset_manager.data_dir}")