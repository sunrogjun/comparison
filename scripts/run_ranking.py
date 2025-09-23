#!/usr/bin/env python3
"""
Run code ranking using different methods
"""
import os
import sys
import argparse
from typing import List, Optional

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ranking_methods.acecoder_ranker import AceCoderRMRanker
from ranking_methods.llm_judge_ranker import LLMJudgeRanker

def main():
    parser = argparse.ArgumentParser(description="Run code ranking using different methods")
    parser.add_argument("--input_dir", type=str, 
                       default="/home/fdse/srj/comparison/results/candidates/non_finetuned",
                       help="Input directory containing candidate codes")
    parser.add_argument("--output_dir", type=str,
                       default="/home/fdse/srj/comparison/results/ranked",
                       help="Output directory for ranked results")
    parser.add_argument("--methods", nargs="+",
                       default=["llm_judge"],
                       choices=["acecoder_rm", "llm_judge"],
                       help="Ranking methods to use")
    parser.add_argument("--models", nargs="+",
                       default=["codet5-770m", "codegen-2b", "codellama-7b"],
                       help="Models to process")
    parser.add_argument("--datasets", nargs="+",
                       default=["humaneval", "mbpp"],
                       help="Datasets to process")
    parser.add_argument("--acecoder_batch_size", type=int, default=8,
                       help="Batch size for AceCoderRM")
    parser.add_argument("--llm_judge_model", type=str,
                       default="meta-llama/Llama-2-7b-chat-hf",
                       help="Model path for LLM judge")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Code Ranking Pipeline")
    print("=" * 60)
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Methods: {args.methods}")
    print(f"Models: {args.models}")
    print(f"Datasets: {args.datasets}")
    print()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process each method
    for method_name in args.methods:
        print(f"\n{'='*60}")
        print(f"Processing with method: {method_name}")
        print(f"{'='*60}")
        
        # Initialize ranker based on method
        if method_name == "acecoder_rm":
            ranker = AceCoderRMRanker(batch_size=args.acecoder_batch_size)
        elif method_name == "llm_judge":
            ranker = LLMJudgeRanker(model_path=args.llm_judge_model)
        elif method_name == "simple_judge":
            ranker = SimpleLLMJudge()
        else:
            print(f"Unknown method: {method_name}")
            continue
        
        try:
            # Process each model and dataset combination
            for model_name in args.models:
                print(f"\n{'-'*40}")
                print(f"Model: {model_name}")
                print(f"{'-'*40}")
                
                for dataset_name in args.datasets:
                    print(f"\nProcessing {dataset_name}...")
                    
                    try:
                        ranker.process_dataset(
                            input_dir=args.input_dir,
                            output_dir=args.output_dir,
                            dataset_name=dataset_name,
                            model_name=model_name
                        )
                    except Exception as e:
                        print(f"Error processing {model_name}/{dataset_name}: {e}")
                        continue
            
        finally:
            # Clean up model to free memory
            try:
                ranker.cleanup()
            except:
                pass
    
    print(f"\n{'='*60}")
    print("Ranking completed!")
    print(f"Results saved to: {args.output_dir}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()