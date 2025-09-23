"""
Prompt builder for different code datasets
Creates appropriate prompts for code generation based on dataset type
"""
import os
import json

class PromptBuilder:
    def __init__(self):
        """Initialize prompt builder"""
        pass
    
    def build_prompt(self, problem: dict, dataset_name: str) -> str:
        """
        Build prompt based on dataset type and problem
        
        Args:
            problem: Problem dictionary containing task information
            dataset_name: Name of the dataset (humaneval, mbpp, apps)
            
        Returns:
            Formatted prompt string for code generation
        """
        if dataset_name == "humaneval":
            return self._build_humaneval_prompt(problem)
        elif dataset_name == "mbpp":
            return self._build_mbpp_prompt(problem)
        elif dataset_name == "apps":
            return self._build_apps_prompt(problem)
        else:
            # Default prompt construction
            return problem.get("prompt", "")
    
    def _build_humaneval_prompt(self, problem: dict) -> str:
        """Build prompt for HumanEval dataset"""
        # For HumanEval, we just need the function signature and docstring
        prompt = problem.get("prompt", "")
        return prompt
    
    def _build_mbpp_prompt(self, problem: dict) -> str:
        """Build prompt for MBPP dataset"""
        # For MBPP, we have a natural language description of the task
        description = problem.get("prompt", "")
        # Add standard header for code generation
        prompt = f"\"\"\"\n{description}\n\"\"\"\n"
        return prompt
    
    def _build_apps_prompt(self, problem: dict) -> str:
        """Build prompt for APPS dataset - based on RankEF implementation"""
        # Start with the question
        question = problem.get("prompt", "")
        _input = "\nQUESTION:\n" + question
        
        # Add starter code if available
        starter_code = problem.get("starter_code", None)
        if starter_code:
            _input += "\n" + starter_code
        
        # Determine format based on problem properties
        if problem.get("difficulty", ""):
            # Check if it should use call-based format
            _input += "\nUse Call-Based format"
        else:
            _input += "\nUse Standard Input format"
            
        _input += "\nANSWER:\n"
        return _input
