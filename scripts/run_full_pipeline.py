#!/usr/bin/env python3
"""
Run the complete ranking and evaluation pipeline
"""
import os
import sys
import subprocess
import argparse

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n{'='*60}")
    print(f"STEP: {description}")
    print(f"{'='*60}")
    print(f"Command: {command}")
    print()
    
    try:
        result = subprocess.run(command, shell=True, check=True, 
                              capture_output=False, text=True)
        print(f"\n✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ {description} failed with exit code {e.returncode}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Run complete ranking and evaluation pipeline")
    parser.add_argument("--skip_ranking", action="store_true",
                       help="Skip ranking step (use existing ranked results)")
    parser.add_argument("--skip_evaluation", action="store_true", 
                       help="Skip evaluation step")
    parser.add_argument("--methods", nargs="+",
                       default=["acecoder_rm"],
                       choices=["acecoder_rm", "llm_judge"],
                       help="Ranking methods to use")
    parser.add_argument("--models", nargs="+",
                       default=["codet5-770m", "codegen-2b", "codellama-7b"],
                       help="Models to process")
    parser.add_argument("--datasets", nargs="+",
                       default=["humaneval", "mbpp"],
                       help="Datasets to process")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("COMPLETE RANKING AND EVALUATION PIPELINE")
    print("=" * 60)
    print(f"Methods: {args.methods}")
    print(f"Models: {args.models}")
    print(f"Datasets: {args.datasets}")
    print(f"Skip ranking: {args.skip_ranking}")
    print(f"Skip evaluation: {args.skip_evaluation}")
    print()
    
    # Step 1: Generate candidates (unless skipped)
    if not args.skip_generation:
        gen_cmd = "python scripts/run_generation.py"
        if not run_command(gen_cmd, "Candidate Generation"):
            print("Generation failed. Exiting.")
            return
    else:
        print("\nSkipping candidate generation step...")
    
    # Step 2: Rank candidates
    rank_cmd = f"python scripts/run_ranking.py --methods {' '.join(args.methods)} --models {' '.join(args.models)} --datasets {' '.join(args.datasets)}"
    if not run_command(rank_cmd, "Candidate Ranking"):
        print("Ranking failed. Exiting.")
        return
    
    # Step 3: Evaluate rankings
    eval_cmd = (f"python scripts/run_evaluation.py "
                f"--results_dir {args.results_dir} "
                f"--output_dir {args.evaluation_dir} "
                f"--methods {' '.join(args.methods)} "
                f"--models {' '.join(args.models)} "
                f"--datasets {' '.join(args.datasets)}")
    
    if not run_command(eval_cmd, "Ranking Evaluation"):
        print("Evaluation failed.")
        return
    
    print(f"\n{'='*60}")
    print("Complete pipeline finished successfully!")
    print(f"Results are in: {args.results_dir}")
    print(f"Evaluation outputs are in: {args.evaluation_dir}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()