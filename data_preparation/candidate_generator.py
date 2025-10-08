"""
Candidate generator for code models
Generates code candidates using pre-trained models for different datasets
"""
import os
import sys
import json
import torch
from typing import Dict, List
from transformers import T5ForConditionalGeneration, AutoModelForCausalLM, AutoTokenizer
from .prompt_builder import PromptBuilder

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import model configuration functions
from config.model_config import get_model_path, get_model_config, NON_FINETUNED_MODELS


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
                    trust_remote_code=True,
                    low_cpu_mem_usage=self.model_config.get("low_cpu_mem_usage", False),
                    offload_folder=self.model_config.get("offload_folder", None)
                )
            else:
                # For CodeGen and CodeLlama (causal LM models)
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    device_map=self.model_config.get("device_map", "auto"),
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    trust_remote_code=True,
                    low_cpu_mem_usage=self.model_config.get("low_cpu_mem_usage", False),
                    offload_folder=self.model_config.get("offload_folder", None)
                )
            
            self.model.eval()
            print(f"Model {self.model_name} loaded successfully")
            
        except Exception as e:
            print(f"Error loading model {self.model_name}: {e}")
            raise
    
    def _generate_batch(self, prompt: str, batch_size: int, gen_params: Dict) -> List[str]:
        """Generate a batch of candidates"""
        
        # Tokenize input with increased max_length
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt", 
            padding=True, 
            truncation=True,
            max_length=self.model_config.get("max_length", 1024)  # Increased from 512 to 1024
        )
        
        # Move to GPU if available
        if torch.cuda.is_available() and hasattr(self.model, 'device'):
            device = next(self.model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            if "codet5" in self.model_name.lower():
                # T5 generation with increased max_new_tokens
                outputs = self.model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask"),
                    max_new_tokens=gen_params.get("max_new_tokens", 1024),  # Increased from 512 to 1024
                    temperature=gen_params["temperature"],
                    top_p=gen_params["top_p"],
                    do_sample=gen_params["do_sample"],
                    num_return_sequences=batch_size,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    early_stopping=True
                )
                
                # Decode outputs (T5 generates only the new tokens)
                candidates = []
                for output in outputs:
                    candidate = self.tokenizer.decode(output, skip_special_tokens=True)
                    candidates.append(candidate.strip())
                    
            else:
                # Causal LM generation (CodeGen, CodeLlama) with increased max_new_tokens
                outputs = self.model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask"),
                    max_new_tokens=gen_params.get("max_new_tokens", 1024),  # Increased from 512 to 1024
                    temperature=gen_params["temperature"],
                    top_p=gen_params["top_p"],
                    do_sample=gen_params["do_sample"],
                    num_return_sequences=batch_size,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    early_stopping=True
                )
                
                # Decode outputs and extract only new tokens
                input_length = inputs["input_ids"].shape[1]
                candidates = []
                for output in outputs:
                    # Extract only the generated part (after input prompt)
                    generated_tokens = output[input_length:]
                    candidate = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                    candidates.append(candidate.strip())
        
        return candidates
    
    def _generate_batch_parallel(self, prompts: List[str], num_seqs: int, gen_params: Dict) -> List[List[str]]:
        """Generate candidates for multiple prompts in parallel - TRUE BATCH PROCESSING"""
        
        # RankEF-style tokenization with increased max_length
        if "codet5" in self.model_name.lower():
            # Use RankEF's exact encoding approach
            input_ids_list = []
            for prompt in prompts:
                input_ids = torch.LongTensor(
                    self.tokenizer.encode(prompt, verbose=False, max_length=gen_params.get("source_len", 1024))  # Increased from 600 to 1024
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
            # Standard tokenization for other models with increased max_length
            inputs = self.tokenizer(
                prompts,
                return_tensors="pt", 
                padding=True, 
                truncation=True,
                max_length=self.model_config.get("max_length", 1024)  # Increased from 512 to 1024
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
                # RankEF-style T5 generation with increased max_length
                outputs = self.model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask"),
                    do_sample=True,
                    temperature=gen_params["temperature"],  # RankEF: 0.6
                    max_length=gen_params.get("max_length", 1024),  # Increased from 512 to 1024, RankEF uses max_length not max_new_tokens
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
                # Causal LM batch generation with increased max_new_tokens
                outputs = self.model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask"),
                    do_sample=True,
                    temperature=gen_params["temperature"],
                    max_new_tokens=gen_params.get("max_new_tokens", 1024),  # Increased from 512 to 1024
                    num_return_sequences=num_seqs,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    early_stopping=True,
                    top_p=gen_params.get("top_p", 0.95)
                )
                
                # Decode outputs and extract only new tokens
                input_length = inputs["input_ids"].shape[1]
                batch_results = []
                outputs_per_prompt = outputs.view(len(prompts), num_seqs, -1)
                
                for prompt_outputs in outputs_per_prompt:
                    prompt_candidates = []
                    for output in prompt_outputs:
                        # Extract only the generated part (after input prompt)
                        generated_tokens = output[input_length:]
                        candidate = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                        prompt_candidates.append(candidate.strip())
                    batch_results.append(prompt_candidates)
        
        return batch_results
    
    def generate_candidates(self, problem: Dict, dataset_name: str, num_candidates: int = 100) -> List[str]:
        """Generate multiple candidate solutions for a problem"""
        try:
            # Build prompt using PromptBuilder
            prompt = self.prompt_builder.build_prompt(problem, dataset_name)
            
            # Handle different model types
            if "codet5" in self.model_name.lower():
                # For CodeT5, use RankEF approach with encoder-decoder
                input_ids = self.tokenizer(
                    prompt, 
                    return_tensors="pt", 
                    truncation=True, 
                    max_length=self.model_config.get("max_length", 512)
                ).input_ids.to(self.model.device)
                
                # Generate multiple candidates using different seeds
                candidates = []
                for i in range(num_candidates):
                    # Set seed for reproducibility
                    torch.manual_seed(i)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(i)
                    
                    with torch.no_grad():
                        outputs = self.model.generate(
                            input_ids,
                            max_length=512,
                            num_return_sequences=1,
                            do_sample=True,
                            temperature=0.8,
                            top_p=0.95,
                            pad_token_id=self.tokenizer.pad_token_id,
                        )
                        candidate = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                        # Only add non-empty candidates
                        if candidate.strip():
                            candidates.append(candidate)
                
            else:
                # For causal LM models (CodeGen, CodeLlama)
                input_ids = self.tokenizer(
                    prompt, 
                    return_tensors="pt", 
                    truncation=True, 
                    max_length=self.model_config.get("max_length", 512)
                ).input_ids.to(self.model.device)
                
                # Generate multiple candidates
                candidates = []
                for i in range(num_candidates):
                    # Set seed for reproducibility
                    torch.manual_seed(i)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(i)
                    
                    with torch.no_grad():
                        outputs = self.model.generate(
                            input_ids,
                            max_length=input_ids.shape[1] + 256,  # Add 256 tokens for completion
                            num_return_sequences=1,
                            do_sample=True,
                            temperature=0.8,
                            top_p=0.95,
                            pad_token_id=self.tokenizer.pad_token_id,
                        )
                        # Extract only the generated part (remove prompt)
                        generated_tokens = outputs[0][input_ids.shape[1]:]
                        candidate = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                        # Only add non-empty candidates
                        if candidate.strip():
                            candidates.append(candidate)
            
            return candidates
            
        except Exception as e:
            print(f"Error generating candidates: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def process_dataset(self, dataset_name: str, output_dir: str, num_candidates: int = 100):
        """Process entire dataset with this generator"""
        # Load dataset problems
        problems = self._load_dataset_problems(dataset_name)
        print(f"Loaded {len(problems)} problems from {dataset_name}")
        
        # Create output directory
        dataset_output_dir = os.path.join(output_dir, self.model_name, dataset_name)
        os.makedirs(dataset_output_dir, exist_ok=True)
        
        # Process each problem
        for i, (problem_id, problem) in enumerate(problems.items()):
            try:
                output_file = os.path.join(dataset_output_dir, f"problem_{problem_id}.json")
                
                # Skip if already processed
                if os.path.exists(output_file):
                    print(f"  {i+1}/{len(problems)}: {problem_id} - ALREADY PROCESSED")
                    continue
                
                print(f"  {i+1}/{len(problems)}: {problem_id} - GENERATING CANDIDATES")
                
                # Generate candidates
                candidates = self.generate_candidates(problem, dataset_name, num_candidates)
                
                # Save results
                result = {
                    "problem_id": problem_id,
                    "prompt": problem["prompt"],
                    "canonical_solution": problem.get("canonical_solution", ""),
                    "test_cases": problem.get("test_cases", []),
                    "example_tests": problem.get("example_tests", ""),
                    "generated_candidates": candidates
                }
                
                with open(output_file, 'w') as f:
                    json.dump(result, f, indent=2)
                    
                print(f"  {i+1}/{len(problems)}: {problem_id} - COMPLETED ({len(candidates)} candidates)")
                
            except Exception as e:
                print(f"  {i+1}/{len(problems)}: {problem_id} - ERROR: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        print(f"Dataset {dataset_name} processing completed!")
    
    def _load_dataset_problems(self, dataset_name: str) -> Dict[str, Dict]:
        """Load problems from specified dataset"""
        problems = {}
        
        if dataset_name == "humaneval":
            # Load HumanEval dataset from local directory
            humaneval_dir = "/home/fdse/srj/comparison/data/humaneval"
            if not os.path.exists(humaneval_dir):
                raise FileNotFoundError(f"HumanEval dataset directory not found at: {humaneval_dir}")
            
            # Process all json files except the complete file
            for file_name in sorted(os.listdir(humaneval_dir)):
                if file_name.endswith('.json') and file_name != 'humaneval_complete.json':
                    file_path = os.path.join(humaneval_dir, file_name)
                    with open(file_path, 'r') as f:
                        item = json.load(f)
                        problem_id = f"humaneval_{item['problem_id']}"
                        problems[problem_id] = {
                            "prompt": item["prompt"],
                            "canonical_solution": item["canonical_solution"],
                            "test_cases": item["test"],
                            "entry_point": item["entry_point"]
                        }
        elif dataset_name == "mbpp":
            # Load MBPP dataset from local directory
            mbpp_dir = "/home/fdse/srj/comparison/data/mbpp"
            if not os.path.exists(mbpp_dir):
                raise FileNotFoundError(f"MBPP dataset directory not found at: {mbpp_dir}")
                
            # Process all json files except the complete file
            for file_name in sorted(os.listdir(mbpp_dir)):
                if file_name.endswith('.json') and file_name != 'mbpp_complete.json':
                    file_path = os.path.join(mbpp_dir, file_name)
                    with open(file_path, 'r') as f:
                        item = json.load(f)
                        problem_id = f"mbpp_{item['problem_id']}"
                        problems[problem_id] = {
                            "prompt": item["prompt"],
                            "canonical_solution": item["canonical_solution"],
                            "test_cases": item["test_list"],
                            "example_tests": item.get("example_tests", "")
                        }
        elif dataset_name == "apps":
            apps_dir = "/home/fdse/srj/comparison/data/apps"
            if not os.path.exists(apps_dir):
                raise FileNotFoundError(f"APPS dataset directory not found: {apps_dir}")
            
            # Process APPS problems in size order to handle memory better
            problem_files = [f for f in os.listdir(apps_dir) if f.endswith(".json")]
            file_sizes = []
            
            for f in problem_files:
                file_path = os.path.join(apps_dir, f)
                size = os.path.getsize(file_path)
                file_sizes.append((f, size))
            
            # Sort by size (smaller first)
            file_sizes.sort(key=lambda x: x[1])
            sorted_problem_files = [f for f, _ in file_sizes]
            
            for filename in sorted_problem_files:
                try:
                    file_path = os.path.join(apps_dir, filename)
                    problem_id = filename.replace(".json", "")
                    
                    with open(file_path, "r") as f:
                        problem_data = json.load(f)
                        # Ensure the problem has the expected structure
                        problems[problem_id] = {
                            "prompt": problem_data.get("prompt", ""),
                            "canonical_solution": problem_data.get("solutions", [None])[0] if problem_data.get("solutions") else "",
                            "test_cases": problem_data.get("input_output", "{}"),
                            "difficulty": problem_data.get("difficulty", ""),
                            "url": problem_data.get("url", ""),
                            "starter_code": problem_data.get("starter_code", "")
                        }
                except Exception as e:
                    print(f"Error loading APPS problem {filename}: {e}")
                    continue
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")
            
        return problems

def check_model_dataset_completed(model_name: str, dataset_name: str, output_dir: str) -> bool:
    """Check if a specific model/dataset combination has already been completed"""
    model_output_dir = os.path.join(output_dir, model_name, dataset_name)
    
    if not os.path.exists(model_output_dir):
        return False
    
    # Count existing result files
    existing_files = [f for f in os.listdir(model_output_dir) if f.endswith('.json')]
    
    # For APPS dataset, check if we have results for all problems
    if dataset_name == "apps":
        apps_dir = "/home/fdse/srj/comparison/data/apps"
        if os.path.exists(apps_dir):
            expected_problems = len([f for f in os.listdir(apps_dir) if f.endswith(".json")])
            return len(existing_files) >= expected_problems and expected_problems > 0
    
    # For other datasets, use original logic
    dataset_dir = f"/home/fdse/srj/comparison/data/{dataset_name}"
    expected_problems = 0
    try:
        if os.path.exists(dataset_dir):
            if dataset_name == "humaneval":
                # Load dataset to get expected number of problems
                from datasets import load_dataset
                dataset = load_dataset("openai_humaneval")
                expected_problems = len(dataset["test"])
            elif dataset_name == "mbpp":
                # Load dataset to get expected number of problems
                from datasets import load_dataset
                dataset = load_dataset("mbpp")
                expected_problems = len(dataset["test"])
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
        datasets = ["humaneval", "mbpp", "apps"]  # Add apps dataset
    
    print(f"Starting candidate generation for {len(models)} models and {len(datasets)} datasets")
    print(f"Models: {', '.join(models)}")
    print(f"Datasets: {', '.join(datasets)}")
    print(f"Output directory: {output_dir}")
    print(f"Number of candidates per problem: {num_candidates}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Check which combinations are already completed
    skipped_combinations = []
    for model_name in models:
        for dataset_name in datasets:
            if check_model_dataset_completed(model_name, dataset_name, output_dir):
                skipped_combinations.append(f"{model_name}/{dataset_name}")
    
    if skipped_combinations:
        print(f"\nSkipping {len(skipped_combinations)} already completed model/dataset combinations:")
        for combo in skipped_combinations:
            print(f"  ✓ {combo}")
    
    for model_idx, model_name in enumerate(models):
        print(f"\n{'='*60}")
        print(f"Model {model_idx+1}/{len(models)}: {model_name}")
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
                try:
                    generator.process_dataset(dataset_name, output_dir, num_candidates)
                except Exception as e:
                    print(f"Error processing dataset {dataset_name} with model {model_name}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
            
            # Clean up model to free memory
            del generator.model
            del generator.tokenizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"Error with model {model_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n{'='*60}")
    print("ALL CANDIDATE GENERATION COMPLETED")
    print(f"Results saved to: {output_dir}")
    print(f"{'='*60}")

if __name__ == "__main__":
    # Generate candidates for HumanEval, MBPP and APPS
    print("Starting candidate generation...")
    print("Models: codet5-770m, codegen-2b, codellama-7b")
    print("Datasets: humaneval, mbpp, apps")
    generate_all_candidates(
        models=["codet5-770m", "codegen-2b", "codellama-7b"],
        datasets=["humaneval", "mbpp", "apps"],  # Add apps dataset
        num_candidates=100
    )
    print("Candidate generation completed.")
