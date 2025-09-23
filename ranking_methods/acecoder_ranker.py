"""
AceCoderRM-based ranking method for candidate code evaluation
Based on the official AceCoder implementation and paper methodology
"""
import os
import torch
from typing import List, Dict, Any, Tuple
from transformers import AutoTokenizer
# Import AceCoder components based on the official implementation
import sys

sys.path.append('/home/fdse/srj/AceCoder/src')

try:
    # Import the correct class from acecoder package
    from acecoder import AceCodeRM
    ACECODER_AVAILABLE = "official"
except ImportError:
    try:
        # Try local AceCoder implementation
        from acecoder.rm_utils import AceCodeRM
        ACECODER_AVAILABLE = "local"
    except ImportError:
        try:
            # Manual implementation based on official code
            import torch
            import torch.nn as nn
            from transformers import Qwen2ForCausalLM
            ACECODER_AVAILABLE = "manual"
        except ImportError:
            ACECODER_AVAILABLE = False
from .base_ranker import BaseRanker

class AceCoderRMRanker(BaseRanker):
    """Ranking using AceCoderRM reward model following official implementation"""
    
    def __init__(self, model_path: str = "TIGER-Lab/AceCodeRM-7B", batch_size: int = 4):
        super().__init__("acecoder_rm")
        self.model_path = model_path
        self.batch_size = batch_size  # Reduced default for memory efficiency
        self.model = None
        self.tokenizer = None
        self._load_model()
    
    def _load_model(self):
        """Load AceCoderRM model and tokenizer using official implementation"""
        print(f"Loading AceCoderRM from {self.model_path}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path, 
                trust_remote_code=True,
                padding_side="left"  # Important for reward model
            )
            
            # Set pad token if not exists
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model based on availability - following official AceCoder implementation
            if ACECODER_AVAILABLE in ["official", "local"]:
                # Use official AceCodeRM implementation
                self.model = AceCodeRM.from_pretrained(
                    self.model_path,
                    device_map="auto",
                    dtype=torch.float16,  # Use 'dtype' instead of deprecated 'torch_dtype'
                    trust_remote_code=True
                )
                print(f"Loaded using {ACECODER_AVAILABLE} AceCodeRM implementation")
            elif ACECODER_AVAILABLE == "manual":
                # Manual implementation based on official code structure
                self.model = self._load_manual_acecoderm()
                print("Loaded using manual AceCodeRM implementation")
            else:
                raise ImportError("AceCoder not available - please install: pip install git+https://github.com/TIGER-AI-Lab/AceCoder.git")
                
            self.model.eval()
            print("AceCoderRM loaded successfully")
        except Exception as e:
            print(f"Error loading AceCoderRM: {e}")
            raise
    
    def _load_manual_acecoderm(self):
        """Manual implementation of AceCodeRM based on official code"""
        from transformers import AutoModelForCausalLM
        
        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
        
        # Try to add value head if the model supports it
        if hasattr(base_model.config, 'hidden_size'):
            import torch.nn as nn
            
            class ValueHead(nn.Module):
                def __init__(self, hidden_size):
                    super().__init__()
                    self.dropout = nn.Dropout(0.1)
                    self.summary = nn.Linear(hidden_size, 1)
                    
                def forward(self, hidden_states):
                    output = self.dropout(hidden_states)
                    if output.dtype != self.summary.weight.dtype:
                        output = output.to(self.summary.weight.dtype)
                    return self.summary(output)
            
            base_model.v_head = ValueHead(base_model.config.hidden_size)
            if torch.cuda.is_available():
                base_model.v_head = base_model.v_head.cuda()
        
        return base_model
    
    def _create_chat_format(self, problem: Dict[str, Any], candidate_code: str) -> List[Dict[str, str]]:
        """Create chat format for AceCoderRM input following the official format"""
        
        # Extract problem information - format according to AceCoder training data
        if 'prompt' in problem:
            # HumanEval format - clean up the prompt
            question = problem['prompt'].strip()
            if question.endswith('"""'):
                # Remove incomplete docstring if present
                lines = question.split('\n')
                # Find the function definition line
                for i, line in enumerate(lines):
                    if line.strip().startswith('def '):
                        # Keep everything up to and including the function definition
                        question = '\n'.join(lines[:i+1])
                        break
        else:
            # MBPP or other formats
            question = problem.get('text', problem.get('description', str(problem)))
        
        # Format according to AceCoder's training format
        chat = [
            {
                "role": "user",
                "content": f"Please complete the following Python code:\n\n{question}"
            },
            {
                "role": "assistant", 
                "content": candidate_code.strip()
            }
        ]
        return chat
    
    def _batch_score_candidates(self, problem: Dict[str, Any], candidates: List[str]) -> List[float]:
        """Score candidates in batches using official AceCoderRM methodology"""
        scores = []
        
        # Process candidates in batches
        for i in range(0, len(candidates), self.batch_size):
            batch_candidates = candidates[i:i + self.batch_size]
            
            # Create chat formats for the batch
            batch_chats = []
            for candidate in batch_candidates:
                if candidate.strip():  # Only process non-empty candidates
                    chat = self._create_chat_format(problem, candidate)
                    batch_chats.append(chat)
                else:
                    batch_chats.append([{"role": "user", "content": ""}, {"role": "assistant", "content": ""}])
            
            try:
                # Tokenize using the exact format from official AceCoder example
                input_tokens = self.tokenizer.apply_chat_template(
                    batch_chats,
                    tokenize=True,
                    return_dict=True,
                    padding=True,
                    return_tensors="pt",
                    add_generation_prompt=False
                )
                
                # Move to the correct device
                input_tokens = {k: v.to(next(self.model.parameters()).device) for k, v in input_tokens.items()}
                
                with torch.no_grad():
                    # Forward pass using EXACT official AceCoder implementation
                    if ACECODER_AVAILABLE in ["official", "local"]:
                        # Call model with exact parameters from official example
                        rm_scores = self.model(
                            input_ids=input_tokens["input_ids"],
                            attention_mask=input_tokens["attention_mask"],
                            output_hidden_states=True,
                            return_dict=True,
                            use_cache=False,
                        )
                        # rm_scores is already the final reward score tensor
                    elif ACECODER_AVAILABLE == "manual":
                        # Manual implementation following exact official forward logic
                        outputs = self.model(
                            input_ids=input_tokens["input_ids"],
                            attention_mask=input_tokens["attention_mask"],
                            output_hidden_states=True,
                            return_dict=True,
                            use_cache=False,
                        )
                        
                        # Exact implementation from official rm_utils.py
                        last_hidden_state = outputs.hidden_states[-1]
                        if hasattr(self.model, 'v_head'):
                            if last_hidden_state.device != self.model.v_head.summary.weight.device:
                                last_hidden_state = last_hidden_state.to(self.model.v_head.summary.weight.device)
                            value = self.model.v_head(last_hidden_state).squeeze(-1)
                            # Extract score from last token (EOS) position
                            masks = input_tokens["attention_mask"]
                            # Ensure masks are on the same device as value
                            if masks.device != value.device:
                                masks = masks.to(value.device)
                            indices = (masks.sum(dim=-1, keepdim=True) - 1)
                            # Ensure indices are on the same device as value
                            if indices.device != value.device:
                                indices = indices.to(value.device)
                            rm_scores = value.gather(dim=-1, index=indices).squeeze()
                        else:
                            # Fallback without value head
                            rm_scores = last_hidden_state.mean(dim=-1)
                    
                    # Handle tensor dimensions
                    if rm_scores.dim() == 0:
                        rm_scores = rm_scores.unsqueeze(0)
                    elif rm_scores.dim() == 2:
                        rm_scores = rm_scores.squeeze(-1)
                    
                    batch_scores = rm_scores.cpu().float().tolist()
                    
                    # Ensure correct number of scores
                    if isinstance(batch_scores, float):
                        batch_scores = [batch_scores]
                    
                    while len(batch_scores) < len(batch_candidates):
                        batch_scores.append(batch_scores[-1] if batch_scores else -10.0)
                    
                    batch_scores = batch_scores[:len(batch_candidates)]
                    scores.extend(batch_scores)
                
                print(f"  Scored batch {i//self.batch_size + 1}/{(len(candidates) + self.batch_size - 1)//self.batch_size}")
                
                # Clear cache to prevent memory issues
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"  Error scoring batch starting at {i}: {e}")
                # Add default low scores for failed batch
                scores.extend([-10.0] * len(batch_candidates))
                continue
        
        return scores
    
    def rank_candidates(self, problem: Dict[str, Any], candidates: List[str]) -> List[Tuple[str, float]]:
        """
        Rank candidates using AceCoderRM scores
        
        Args:
            problem: Problem dictionary
            candidates: List of candidate code strings
            
        Returns:
            List of (candidate_code, score) tuples sorted by score descending
        """
        print(f"  Ranking {len(candidates)} candidates with AceCoderRM...")
        
        # Filter out empty candidates
        valid_candidates = [(i, candidate) for i, candidate in enumerate(candidates) if candidate.strip()]
        
        if not valid_candidates:
            print("  No valid candidates found")
            return [(candidate, 0.0) for candidate in candidates]
        
        # Extract just the code strings for scoring
        valid_codes = [candidate for _, candidate in valid_candidates]
        
        # Score the valid candidates
        scores = self._batch_score_candidates(problem, valid_codes)
        
        # Create full results list with original order preserved
        full_results = []
        valid_idx = 0
        
        for i, original_candidate in enumerate(candidates):
            if original_candidate.strip():
                # This was a valid candidate, use its score
                score = scores[valid_idx]
                valid_idx += 1
            else:
                # This was an empty candidate, give it a very low score
                score = -100.0
            
            full_results.append((original_candidate, float(score)))
        
        # Sort by score in descending order
        ranked_results = sorted(full_results, key=lambda x: x[1], reverse=True)
        
        print(f"  AceCoderRM ranking completed. Score range: {ranked_results[0][1]:.3f} to {ranked_results[-1][1]:.3f}")
        
        return ranked_results
    
    def cleanup(self):
        """Clean up model to free memory"""
        if self.model is not None:
            del self.model
        if self.tokenizer is not None:
            del self.tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("AceCoderRM model cleaned up")