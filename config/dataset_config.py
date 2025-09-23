"""
Dataset configuration for comparison experiments
"""

# Dataset information
DATASETS = {
    "mbpp": {
        "hf_name": "mbpp",
        "split": "test", 
        "num_problems": 500,
        "description": "Mostly Basic Python Problems",
        "prompt_field": "text",
        "solution_field": "code",
        "test_field": "test_list",
    },
    "humaneval": {
        "hf_name": "openai_humaneval",
        "split": "test",
        "num_problems": 164, 
        "description": "Human Eval Python Problems",
        "prompt_field": "prompt",
        "solution_field": "canonical_solution",
        "test_field": "test",
    },
    "apps": {
        "hf_name": "codeparrot/apps",
        "split": "test",
        "num_problems": 598,  # Validation subset used for comparison
        "description": "APPS Programming Problems",
        "prompt_field": "question",
        "solution_field": "solutions",
        "test_field": "input_output",
    }
}

# Output paths for different experiments
OUTPUT_PATHS = {
    "candidates": {
        "non_finetuned": "./results/candidates/non_finetuned",
        "finetuned": "./results/candidates/finetuned",
    },
    "rankings": "./results/rankings",
    "evaluation": "./results/evaluation",
    "logs": "./logs",
}

# Data format specifications
DATA_FORMAT = {
    "candidate_file": {
        "problem_id": "int",
        "task_id": "str", 
        "prompt": "str",
        "candidates": "list[str]",
        "metadata": "dict",
    },
    "ranking_file": {
        "problem_id": "int",
        "method": "str",
        "rankings": "list[int]",  # Indices of candidates in ranked order
        "scores": "list[float]",  # Optional scores
    }
}

def get_dataset_config(dataset_name):
    """Get configuration for specific dataset"""
    return DATASETS.get(dataset_name, {})

def get_output_path(path_type, subpath=None):
    """Get output path for given type"""
    base_path = OUTPUT_PATHS.get(path_type, "./results")
    if subpath:
        if isinstance(base_path, dict):
            return base_path.get(subpath, "./results")
        else:
            return f"{base_path}/{subpath}"
    return base_path

def get_all_datasets():
    """Get list of all available datasets"""
    return list(DATASETS.keys())