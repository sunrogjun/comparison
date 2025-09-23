"""
Candidate code generator with proper batch processing
Generates multiple candidate solutions for code problems using different models
"""
import os
import json
import torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, 
    T5ForConditionalGeneration
)
from typing import List, Dict, Any
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.model_config import get_model_path, get_model_config, NON_FINETUNED_MODELS
from config.generation_config import GENERATION_PARAMS
from data_preparation.prompt_builder import PromptBuilder

class CandidateGenerator:
    def __init__(self, model_name: str):
        """Initialize candidate generator with specified model"""
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.prompt_builder = PromptBuilder()
        
        # Get model configuration
        self.model_path = get_model_path(model_name, finetuned=False)
        self.model_config = get_model_config(model_name)
        
        if not self.model_path:
            raise ValueError(f"Model {model_name} not found in configuration")
        
        print(f"Initializing {model_name} from {self.model_path}")
        self._load_model()
    
    def _load_model(self):
        """Load model and tokenizer based on model type"""
        try:
            # Use RankEF's tokenizer approach for CodeT5
            if "codet5" in self.model_name.lower():
                # RankEF approach: use the model's own tokenizer but with RobertaTokenizer class
                from transformers import RobertaTokenizer
                print(f"Loading RobertaTokenizer from {self.model_path} (RankEF approach)")
                self.tokenizer = RobertaTokenizer.from_pretrained(self.model_path)
            else:
                # For other models, use AutoTokenizer
                tokenizer_path = self.model_config.get("tokenizer", self.model_path)
                self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
            
            # Set pad token if not exists
            if self.tokenizer.pad_token is None:
                if self.tokenizer.eos_token is not None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                else:
                    self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            
            # Load model based on type
            if "codet5" in self.model_name.lower():
                self.model = T5ForConditionalGeneration.from_pretrained(
                    self.model_path,
                    device_map=self.model_config.get("device_map", "auto"),
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    trust_remote_code=True
                )
            else:
                # For CodeGen and CodeLlama (causal LM models)
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    device_map=self.model_config.get("device_map", "auto"),
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    trust_remote_code=True
                )
            
            self.model.eval()
            print(f"Model {self.model_name} loaded successfully")
            
        except Exception as e:
            print(f"Error loading model {self.model_name}: {e}")
            raise
    
    def process_dataset(self, dataset_name: str, output_dir: str, 
                       num_candidates: int = 100, max_problems: int = None):
        """Process entire dataset with proper batch processing"""
        
        # Load dataset problems
        dataset_dir = f"/home/fdse/srj/comparison/data/{dataset_name}"
        problems = self._load_dataset_problems(dataset_dir, max_problems)
        
        if not problems:
            print(f"No problems found for dataset {dataset_name}")
            return
        
        # Create output directory
        model_output_dir = os.path.join(output_dir, self.model_name, dataset_name)
        os.makedirs(model_output_dir, exist_ok=True)
        
        print(f"Processing {len(problems)} problems from {dataset_name} dataset")
        
        # Get generation parameters
        gen_params = GENERATION_PARAMS.copy()
        inference_batch_size = gen_params.get("inference_batch_size", 64)
        num_seqs_per_iter = gen_params.get("num_seqs_per_iter", 10)
        
        print(f"Inference batch size: {inference_batch_size}, Sequences per iteration: {num_seqs_per_iter}")
        
        # Process problems in TRUE BATCHES using inference_batch_size
        for batch_start in range(0, len(problems), inference_batch_size):
            batch_end = min(batch_start + inference_batch_size, len(problems))
            problem_batch = problems[batch_start:batch_end]
            
            print(f"\n[Batch {batch_start//inference_batch_size + 1}/{(len(problems) + inference_batch_size - 1)//inference_batch_size}] Processing problems {batch_start+1}-{batch_end}")
            
            # Process each problem in the batch and save immediately
            for i, problem in enumerate(problem_batch):
                problem_id = problem.get("problem_id", batch_start + i)
                task_id = problem.get("task_id", f"{dataset_name}/{problem_id}")
                
                print(f"  [{batch_start + i + 1}/{len(problems)}] Processing {task_id}")
                
                try:
                    # Generate candidates for this problem
                    candidates = self._generate_for_problem(problem, dataset_name, num_candidates)
                    
                    # Save immediately after generating
                    result = {
                        "problem": problem,
                        "model_name": self.model_name,
                        "dataset": dataset_name,
                        "num_candidates": len(candidates),
                        "candidates": candidates,
                        "generation_params": gen_params
                    }
                    
                    output_file = os.path.join(model_output_dir, f"problem_{problem_id}.json")
                    
                    with open(output_file, "w") as f:
                        json.dump(result, f, indent=2)
                    
                    print(f"    ✓ {task_id}: {len(candidates)} candidates saved")
                    
                except Exception as e:
                    print(f"    ✗ Error processing {task_id}: {e}")
                    continue
            
            # Clear GPU cache after each batch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                print(f"    GPU cache cleared after batch")
        
        print(f"\nCompleted processing {dataset_name} with {self.model_name}")
    
    def _generate_for_problem(self, problem: Dict, dataset_name: str, num_candidates: int) -> List[str]:
        """Generate candidates for a single problem"""
        
        # Build prompt
        prompt = self.prompt_builder.build_prompt(problem, dataset_name)
        
        # Get RankEF-style generation parameters
        gen_params = GENERATION_PARAMS.copy()
        num_seqs_per_iter = gen_params.get("num_seqs_per_iter", 10)
        
        # Apply RankEF-specific settings for CodeT5
        if "codet5" in self.model_name.lower():
            gen_params["temperature"] = 0.6  # RankEF uses 0.6 for better quality
            gen_params["source_len"] = 600   # RankEF input length
        
        # Calculate iterations needed
        num_iterations = (num_candidates + num_seqs_per_iter - 1) // num_seqs_per_iter
        
        all_candidates = []
        
        # Generate in iterations
        for iter_idx in range(num_iterations):
            current_num_seqs = min(num_seqs_per_iter, num_candidates - len(all_candidates))
            
            if current_num_seqs <= 0:
                break
            
            try:
                candidates = self._generate_for_single_problem(prompt, current_num_seqs, gen_params)
                all_candidates.extend(candidates)
                print(f"    Iteration {iter_idx+1}/{num_iterations}: {len(candidates)} candidates")
                
            except Exception as e:
                print(f"    Error in iteration {iter_idx+1}: {e}")
                # Add empty candidates to maintain count
                for _ in range(current_num_seqs):
                    all_candidates.append("")
        
        # Ensure exactly the right number of candidates
        while len(all_candidates) < num_candidates:
            all_candidates.append("")
        
        return all_candidates[:num_candidates]
    
    def _generate_batch_parallel(self, prompts: List[str], num_seqs: int, gen_params: Dict) -> List[List[str]]:
        """Generate candidates for multiple prompts in parallel - TRUE BATCH PROCESSING"""
        
        # RankEF-style tokenization
        if "codet5" in self.model_name.lower():
            # Use RankEF's exact encoding approach
            input_ids_list = []
            for prompt in prompts:
                input_ids = torch.LongTensor(
                    self.tokenizer.encode(prompt, verbose=False, max_length=gen_params.get("source_len", 600))
                ).unsqueeze(0)
                input_ids_list.append(input_ids)
            
            # Pad sequences to same length
            max_len = max(ids.shape[1] for ids in input_ids_list)
            padded_input_ids = []
            for ids in input_ids_list:
                if ids.shape[1] < max_len:
                    pad_length = max_len - ids.shape[1]
                    padded = torch.cat([ids, torch.full((1, pad_length), self.tokenizer.pad_token_id)], dim=1)
                    padded_input_ids.append(padded)
                else:
                    padded_input_ids.append(ids)
            
            inputs = {"input_ids": torch.cat(padded_input_ids, dim=0)}
        else:
            # Standard tokenization for other models
            inputs = self.tokenizer(
                prompts,
                return_tensors="pt", 
                padding=True, 
                truncation=True,
                max_length=self.model_config.get("max_length", 512)
            )
        
        # RankEF-style explicit CUDA handling
        if torch.cuda.is_available():
            if "codet5" in self.model_name.lower():
                # RankEF explicitly uses .cuda()
                inputs = {k: v.cuda() for k, v in inputs.items()}
            else:
                # Standard device handling for other models
                device = next(self.model.parameters()).device
                inputs = {k: v.to(device) for k, v in inputs.items()}
        
        batch_results = []
        
        with torch.no_grad():
            if "codet5" in self.model_name.lower():
                # RankEF-style T5 generation
                outputs = self.model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask"),
                    do_sample=True,
                    temperature=gen_params["temperature"],  # RankEF: 0.6
                    max_length=gen_params.get("max_length", 512),  # RankEF uses max_length not max_new_tokens
                    num_return_sequences=num_seqs,
                    top_p=0.95,  # RankEF setting
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    early_stopping=True
                )
                
                # Decode outputs and group by original prompt
                candidates_per_prompt = []
                outputs_per_prompt = outputs.view(len(prompts), num_seqs, -1)
                
                for prompt_outputs in outputs_per_prompt:
                    prompt_candidates = []
                    for output in prompt_outputs:
                        candidate = self.tokenizer.decode(output, skip_special_tokens=True)
                        prompt_candidates.append(candidate.strip())
                    candidates_per_prompt.append(prompt_candidates)
                
                batch_results = candidates_per_prompt
                    
            else:
                # Causal LM batch generation
                outputs = self.model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask"),
                    max_new_tokens=gen_params.get("max_new_tokens", 512),
                    temperature=gen_params["temperature"],
                    top_p=gen_params["top_p"],
                    do_sample=gen_params["do_sample"],
                    num_return_sequences=num_seqs,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    early_stopping=True
                )
                
                # Decode outputs and extract only new tokens, group by original prompt
                input_lengths = inputs["input_ids"].shape[1]
                outputs_per_prompt = outputs.view(len(prompts), num_seqs, -1)
                
                candidates_per_prompt = []
                for prompt_outputs in outputs_per_prompt:
                    prompt_candidates = []
                    for output in prompt_outputs:
                        generated_tokens = output[input_lengths:]
                        candidate = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                        prompt_candidates.append(candidate.strip())
                    candidates_per_prompt.append(prompt_candidates)
                
                batch_results = candidates_per_prompt
        
        return batch_results
    
    def _generate_for_single_problem(self, prompt: str, num_seqs: int, gen_params: Dict) -> List[str]:
        """Generate candidates for a single problem - fallback for single prompt"""
        batch_results = self._generate_batch_parallel([prompt], num_seqs, gen_params)
        return batch_results[0] if batch_results else []
    
    def _load_dataset_problems(self, dataset_dir: str, max_problems: int = None) -> List[Dict]:
        """Load problems from dataset directory"""
        problems = []
        
        try:
            # Try to load complete dataset file first
            complete_files = [f for f in os.listdir(dataset_dir) if "complete" in f and f.endswith(".json")]
            
            if complete_files:
                complete_file = os.path.join(dataset_dir, complete_files[0])
                with open(complete_file, "r") as f:
                    problems = json.load(f)
            else:
                # Load individual problem files
                problem_files = [f for f in os.listdir(dataset_dir) 
                               if f.endswith(".json") and not "complete" in f]
                problem_files.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]) if "_" in x else int(x.split(".")[0]))
                
                for filename in problem_files:
                    filepath = os.path.join(dataset_dir, filename)
                    with open(filepath, "r") as f:
                        problem = json.load(f)
                        problems.append(problem)
            
            if max_problems:
                problems = problems[:max_problems]
                
        except Exception as e:
            print(f"Error loading dataset from {dataset_dir}: {e}")
        
        return problems

def check_model_dataset_completed(model_name: str, dataset_name: str, output_dir: str) -> bool:
    """Check if a model-dataset combination is already completed"""
    model_output_dir = os.path.join(output_dir, model_name, dataset_name)
    if not os.path.exists(model_output_dir):
        return False
    
    # Count existing result files
    existing_files = [f for f in os.listdir(model_output_dir) if f.endswith('.json')]
    
    # Load dataset to get expected number of problems
    dataset_dir = f"/home/fdse/srj/comparison/data/{dataset_name}"
    expected_problems = 0
    try:
        if os.path.exists(dataset_dir):
            complete_files = [f for f in os.listdir(dataset_dir) if "complete" in f and f.endswith(".json")]
            if complete_files:
                with open(os.path.join(dataset_dir, complete_files[0]), "r") as f:
                    problems = json.load(f)
                    expected_problems = len(problems)
            else:
                problem_files = [f for f in os.listdir(dataset_dir) if f.endswith(".json")]
                expected_problems = len(problem_files)
    except:
        pass
    
    return len(existing_files) >= expected_problems and expected_problems > 0

def generate_all_candidates(models: List[str] = None, datasets: List[str] = None,
                           output_dir: str = "/home/fdse/srj/comparison/results/candidates/non_finetuned",
                           num_candidates: int = 100):
    """Generate candidates for all specified models and datasets"""
    
    if models is None:
        models = ["codet5-770m", "codegen-2b", "codellama-7b"]
    
    if datasets is None:
        datasets = ["humaneval", "mbpp"]  # Only HumanEval and MBPP as requested
    
    print(f"Starting candidate generation with proper batch processing...")
    print(f"Models: {models}")
    print(f"Datasets: {datasets}")
    print(f"Candidates per problem: {num_candidates}")
    print(f"Output directory: {output_dir}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Check which combinations are already completed
    for model_name in models:
        for dataset_name in datasets:
            if check_model_dataset_completed(model_name, dataset_name, output_dir):
                print(f"✓ {model_name} on {dataset_name} already completed, skipping...")
    
    for model_name in models:
        print(f"\n{'='*60}")
        print(f"Processing model: {model_name}")
        print(f"{'='*60}")
        
        # Check if all datasets for this model are completed
        all_completed = all(check_model_dataset_completed(model_name, dataset_name, output_dir) 
                           for dataset_name in datasets)
        
        if all_completed:
            print(f"All datasets for {model_name} already completed, skipping entire model...")
            continue
        
        try:
            generator = CandidateGenerator(model_name)
            
            for dataset_name in datasets:
                # Check if this specific combination is completed
                if check_model_dataset_completed(model_name, dataset_name, output_dir):
                    print(f"\n{'-'*40}")
                    print(f"Dataset: {dataset_name} - ALREADY COMPLETED, SKIPPING")
                    print(f"{'-'*40}")
                    continue
                
                print(f"\n{'-'*40}")
                print(f"Dataset: {dataset_name}")
                print(f"{'-'*40}")
                generator.process_dataset(dataset_name, output_dir, num_candidates)
            
            # Clean up model to free memory
            del generator.model
            del generator.tokenizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"Error with model {model_name}: {e}")
            continue
    
    print(f"\n{'='*60}")
    print(f"Candidate generation completed!")
    print(f"Results saved to: {output_dir}")
    print(f"{'='*60}")

if __name__ == "__main__":
    # Generate candidates for HumanEval and MBPP only
    generate_all_candidates(
        models=["codet5-770m", "codegen-2b", "codellama-7b"],
        datasets=["humaneval", "mbpp"],
        num_candidates=100
    )