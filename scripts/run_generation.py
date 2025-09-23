#!/usr/bin/env python3
"""
Run candidate generation with proper batch processing
Generate candidates for 3 models on HumanEval and MBPP datasets
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '7'  # Use GPU 7

from data_preparation.candidate_generator import generate_all_candidates

if __name__ == "__main__":
    print("Starting candidate generation with proper batch processing...")
    print("Target: 3 models × 2 datasets × 100 candidates per problem")
    print("Models: CodeT5-770M, CodeGen-2B, CodeLlama-7B") 
    print("Datasets: HumanEval (164 problems), MBPP (500 problems)")
    print("Expected output: 664 problems × 3 models = 1,992 result files")
    
    # Generate candidates using proper batch processing
    generate_all_candidates(
        models=["codet5-770m", "codegen-2b", "codellama-7b"],
        datasets=["humaneval", "mbpp"],  # Only these two datasets as requested
        num_candidates=100
    )
    
    print("Generation completed! Results saved in results/candidates/non_finetuned/")