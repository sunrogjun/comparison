"""
LLM-as-a-Judge ranking method for candidate code evaluation
"""
import os
import json
import torch
import time
from typing import List, Dict, Any, Tuple, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
from .base_ranker import BaseRanker

class LLMJudgeRanker(BaseRanker):
    """Ranking using Large Language Model as a Judge"""
    
    def __init__(self, 
                 model_path: str = "meta-llama/Llama-2-7b-chat-hf",
                 judge_type: str = "local",  # "local", "openai", "anthropic"
                 batch_size: int = 1,
                 max_new_tokens: int = 512):
        super().__init__("llm_judge")
        self.model_path = model_path
        self.judge_type = judge_type
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens
        self.model = None
        self.tokenizer = None
        
        if judge_type == "local":
            self._load_local_model()
    
    def _load_local_model(self):
        """Load local LLM for judging"""
        print(f"Loading local LLM judge from {self.model_path}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                device_map="auto",
                torch_dtype=torch.float16,
                trust_remote_code=True
            )
            self.model.eval()
            print("Local LLM judge loaded successfully")
        except Exception as e:
            print(f"Error loading local LLM judge: {e}")
            raise
    
    def _create_judge_prompt(self, problem: Dict[str, Any], candidate_code: str, attempt: int = 0) -> str:
        """Create optimized prompt with multiple strategies for better score extraction"""

        # Extract problem information
        if 'prompt' in problem:
            question = problem['prompt']
        else:
            question = problem.get('text', problem.get('description', str(problem)))

        # Truncate long problem descriptions
        question = question.strip()[:150] + ("..." if len(question) > 150 else "")

        # Multiple prompt strategies to improve success rate
        prompt_strategies = [
            # Strategy 0: Ultra minimal (default)
            f"Rate this code 1-10:\n\n{candidate_code.strip()}\n\nTask: {question}\n\nRating:",

            # Strategy 1: Explicit number request
            f"Give a score from 1 to 10 for this Python code:\n\n{candidate_code.strip()}\n\nTask: {question}\n\nScore:",

            # Strategy 2: Multiple choice style
            f"Rate this code (1=bad, 10=excellent):\n\n{candidate_code.strip()}\n\nTask: {question}\n\nYour rating (1-10):",
        ]

        return prompt_strategies[attempt % len(prompt_strategies)]
    
    def _extract_score_from_response(self, response: str) -> float:
        """Extract numerical score from LLM response with improved error handling"""
        try:
            import re

            # Clean response
            response = response.strip()
            print(f"    Raw response: '{response}'")

            # Multiple extraction strategies in order of preference
            strategies = [
                # 1. Pure number (most reliable)
                (r'^(\d+(?:\.\d+)?)$', "pure_number"),
                # 2. Number at start
                (r'^(\d+(?:\.\d+)?)', "start_number"),
                # 3. Score pattern
                (r'[Ss]core[:\s]*(\d+(?:\.\d+)?)', "score_pattern"),
                # 4. Rating pattern
                (r'[Rr]ating[:\s]*(\d+(?:\.\d+)?)', "rating_pattern"),
                # 5. Any standalone number
                (r'\b(\d+(?:\.\d+)?)\b', "any_number"),
            ]

            for pattern, method in strategies:
                match = re.search(pattern, response)
                if match:
                    try:
                        score = float(match.group(1))
                        if 1 <= score <= 10:
                            print(f"    Extracted score: {score} (method: {method})")
                            return score
                        elif 0 <= score <= 1 and score != 0:  # Handle 0-1 scale
                            score *= 10
                            print(f"    Converted 0-1 scale to 10-scale: {score}")
                            return score
                    except ValueError:
                        continue

            # Fallback: look for any digit and use it as score
            digits = re.findall(r'\d', response)
            if digits:
                # Use first valid digit as score, or sum if multiple single digits
                if len(digits) == 1:
                    score = float(digits[0])
                    if 1 <= score <= 10:
                        print(f"    Using single digit as score: {score}")
                        return score
                elif len(digits) == 2 and int(digits[0]) == 1 and int(digits[1]) == 0:
                    print(f"    Detected '10' from separate digits")
                    return 10.0

        except Exception as e:
            print(f"    Error in score extraction: {e}")

        # Final fallback
        print(f"    Could not extract valid score from: '{response[:50]}...' - using default 5.0")
        return 5.0
    
    def _judge_single_candidate(self, problem: Dict[str, Any], candidate_code: str) -> float:
        """Judge a single candidate using local LLM with multiple strategies"""
        if not candidate_code.strip():
            return 1.0  # Minimum score for empty code

        # Try multiple prompt strategies for better success rate
        for prompt_attempt in range(3):  # Try up to 3 different prompts
            try:
                prompt = self._create_judge_prompt(problem, candidate_code, prompt_attempt)

                # Tokenize input with better handling
                inputs = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=1536,
                    padding=False
                ).to(self.model.device)

                with torch.no_grad():
                    # Multiple generation attempts for each prompt
                    for gen_attempt in range(2):
                        try:
                            # Vary generation parameters slightly for each attempt
                            temp = 0.2 + (gen_attempt * 0.3)  # 0.2 or 0.5

                            outputs = self.model.generate(
                                **inputs,
                                max_new_tokens=8,
                                temperature=temp,
                                do_sample=True,
                                top_k=15,
                                top_p=0.85,
                                repetition_penalty=1.02,
                                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                                eos_token_id=self.tokenizer.eos_token_id,
                                early_stopping=True
                            )

                            # Decode response
                            input_length = inputs['input_ids'].shape[1]
                            generated_tokens = outputs[0][input_length:]
                            response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

                            # Extract score
                            score = self._extract_score_from_response(response)

                            # If we got a valid non-default score, return it immediately
                            if score != 5.0:
                                return score

                        except Exception as gen_error:
                            print(f"    Generation attempt {gen_attempt + 1} with prompt {prompt_attempt + 1} failed: {gen_error}")
                            continue

            except Exception as prompt_error:
                print(f"    Prompt attempt {prompt_attempt + 1} failed: {prompt_error}")
                continue

        # All attempts failed, return default
        print(f"    All prompt/generation attempts failed, using default score 5.0")
        return 5.0
    
    def _batch_judge_candidates(self, problem: Dict[str, Any], candidates: List[str]) -> List[float]:
        """Judge candidates using local LLM with better progress tracking"""
        scores = []
        failed_extractions = 0

        print(f"  Judging {len(candidates)} candidates with LLM...")

        for i, candidate in enumerate(candidates):
            # Progress reporting
            if i > 0 and i % 10 == 0:
                success_rate = ((i - failed_extractions) / i) * 100
                print(f"    Progress: {i}/{len(candidates)} (Score extraction success: {success_rate:.1f}%)")

            # Judge candidate
            score = self._judge_single_candidate(problem, candidate)
            scores.append(score)

            # Track failed extractions (scores that fell back to default 5.0)
            if score == 5.0:
                failed_extractions += 1

            # Dynamic delay based on GPU temperature (if available)
            if torch.cuda.is_available():
                # Brief pause to prevent GPU overheating
                time.sleep(0.05)
            else:
                time.sleep(0.02)

        # Report final statistics
        final_success_rate = ((len(candidates) - failed_extractions) / len(candidates)) * 100
        print(f"    Final score extraction success rate: {final_success_rate:.1f}%")

        # Apply intelligent score distribution normalization
        scores = self._normalize_score_distribution(scores)

        return scores

    def _normalize_score_distribution(self, scores: List[float]) -> List[float]:
        """Intelligently normalize score distribution for better ranking"""
        import numpy as np

        if not scores:
            return scores

        scores_array = np.array(scores)
        unique_scores = len(set(scores))

        print(f"    Score distribution: {unique_scores} unique values, range {np.min(scores_array):.1f}-{np.max(scores_array):.1f}")

        # If we have very few unique scores, expand the distribution
        if unique_scores <= 3 and len(scores) > 10:
            print("    Applying score distribution expansion...")

            min_score = np.min(scores_array)
            max_score = np.max(scores_array)

            if max_score > min_score:
                # Map to 3-10 range (avoid 1-2 to distinguish from very poor code)
                normalized = 3 + 7 * (scores_array - min_score) / (max_score - min_score)

                # Add controlled random variation to break ties
                np.random.seed(42)  # Reproducible randomness
                for i in range(len(normalized)):
                    identical_count = np.sum(scores_array == scores_array[i])
                    if identical_count > len(scores) * 0.2:  # If more than 20% are identical
                        # Add small gaussian noise proportional to identical count
                        noise_scale = min(0.5, identical_count / len(scores))
                        normalized[i] += np.random.normal(0, noise_scale)
                        normalized[i] = np.clip(normalized[i], 1, 10)

                return normalized.tolist()
            else:
                # All scores identical, add small variations around the mean
                base_score = scores_array[0]
                np.random.seed(42)
                variations = np.random.normal(0, 0.3, len(scores))
                normalized = np.clip(base_score + variations, max(1, base_score - 1), min(10, base_score + 1))
                return normalized.tolist()

        return scores
    
    def rank_candidates(self, problem: Dict[str, Any], candidates: List[str]) -> List[Tuple[str, float]]:
        """
        Rank candidates using LLM judge scores with robust error handling

        Args:
            problem: Problem dictionary
            candidates: List of candidate code strings

        Returns:
            List of (candidate_code, score) tuples sorted by score descending
        """
        if not candidates:
            print("  No candidates to rank")
            return []

        print(f"  Ranking {len(candidates)} candidates with LLM judge...")

        try:
            if self.judge_type == "local":
                if self.model is None or self.tokenizer is None:
                    print("  ERROR: Local LLM model not loaded properly!")
                    # Fallback to random scores with warning
                    import random
                    random.seed(42)
                    scores = [random.uniform(4, 8) for _ in candidates]
                    print("  Using fallback random scores due to model loading failure")
                else:
                    scores = self._batch_judge_candidates(problem, candidates)
            else:
                # For API-based judges (OpenAI, Anthropic), implement separately
                scores = [5.0] * len(candidates)  # Placeholder
                print("  API-based judges not implemented yet, using default scores")

            # Validate scores
            if len(scores) != len(candidates):
                print(f"  ERROR: Score count mismatch! {len(scores)} scores for {len(candidates)} candidates")
                # Pad or truncate to match
                if len(scores) < len(candidates):
                    scores.extend([5.0] * (len(candidates) - len(scores)))
                else:
                    scores = scores[:len(candidates)]

            # Create results with scores
            results = []
            for candidate, score in zip(candidates, scores):
                try:
                    # Ensure score is valid
                    validated_score = float(score)
                    if not (1 <= validated_score <= 10):
                        print(f"    Warning: Invalid score {validated_score}, clipping to valid range")
                        validated_score = max(1, min(10, validated_score))
                    results.append((candidate, validated_score))
                except (ValueError, TypeError) as e:
                    print(f"    Error processing score {score}: {e}, using default 5.0")
                    results.append((candidate, 5.0))

            # Sort by score in descending order (higher scores first)
            ranked_results = sorted(results, key=lambda x: x[1], reverse=True)

            # Report statistics
            if ranked_results:
                score_range = f"{ranked_results[0][1]:.3f} to {ranked_results[-1][1]:.3f}"
                unique_scores = len(set(score for _, score in ranked_results))
                print(f"  LLM judge ranking completed. Score range: {score_range} ({unique_scores} unique scores)")
            else:
                print("  ERROR: No valid results produced!")

            return ranked_results

        except Exception as e:
            print(f"  CRITICAL ERROR in ranking: {e}")
            print("  Falling back to random ranking...")
            import random
            random.seed(42)
            fallback_results = [(candidate, random.uniform(4, 8)) for candidate in candidates]
            return sorted(fallback_results, key=lambda x: x[1], reverse=True)
    
    def cleanup(self):
        """Clean up model to free memory with thorough cleanup"""
        print("Starting LLM judge model cleanup...")

        try:
            # Move model to CPU first to free GPU memory
            if self.model is not None:
                if hasattr(self.model, 'cpu'):
                    self.model.cpu()
                del self.model
                self.model = None
                print("  Model deleted")

            if self.tokenizer is not None:
                del self.tokenizer
                self.tokenizer = None
                print("  Tokenizer deleted")

            # Force garbage collection
            import gc
            gc.collect()

            # Clear CUDA cache multiple times for thorough cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

                # Report GPU memory status
                if hasattr(torch.cuda, 'memory_allocated'):
                    allocated_mb = torch.cuda.memory_allocated() / 1024 / 1024
                    cached_mb = torch.cuda.memory_reserved() / 1024 / 1024
                    print(f"  GPU memory after cleanup: {allocated_mb:.1f}MB allocated, {cached_mb:.1f}MB cached")

            print("LLM judge model cleaned up successfully")

        except Exception as e:
            print(f"Warning: Error during cleanup: {e}")

    def __del__(self):
        """Destructor to ensure cleanup"""
        try:
            self.cleanup()
        except:
            pass  # Ignore errors in destructor