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
    
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    success = True
    
    # Step 1: Run ranking (if not skipped)
    if not args.skip_ranking:
        ranking_cmd = f"cd {script_dir} && python run_ranking.py"
        ranking_cmd += f" --methods {' '.join(args.methods)}"
        ranking_cmd += f" --models {' '.join(args.models)}"  
        ranking_cmd += f" --datasets {' '.join(args.datasets)}"
        
        success = run_command(ranking_cmd, "Code Ranking")
        
        if not success:
            print("\n❌ Pipeline failed at ranking step")
            return 1
    else:
        print("\n⏭️  Skipping ranking step")
    
    # Step 2: Run evaluation (if not skipped)
    if not args.skip_evaluation:
        evaluation_cmd = f"cd {script_dir} && python run_evaluation.py"
        evaluation_cmd += f" --methods {' '.join(args.methods)}"
        evaluation_cmd += f" --models {' '.join(args.models)}"
        evaluation_cmd += f" --datasets {' '.join(args.datasets)}"
        
        success = run_command(evaluation_cmd, "Ranking Evaluation")
        
        if not success:
            print("\n❌ Pipeline failed at evaluation step")
            return 1
    else:
        print("\n⏭️  Skipping evaluation step")
    
    # Final summary
    print(f"\n{'='*60}")
    print("PIPELINE COMPLETED SUCCESSFULLY! 🎉")
    print(f"{'='*60}")
    print("\nResults can be found in:")
    print("- Ranked candidates: /home/fdse/srj/comparison/results/ranked/")
    print("- Evaluation results: /home/fdse/srj/comparison/results/evaluation/")
    print("\nNext steps:")
    print("1. Check the evaluation results in ranking_comparison.json")
    print("2. Compare Pass@K metrics with RankEF paper results")
    print("3. Analyze which ranking method performs better")
    print(f"{'='*60}")
    
    return 0

if __name__ == "__main__":
    exit(main())