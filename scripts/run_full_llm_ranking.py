#!/usr/bin/env python3
"""
Complete LLM ranking for all models and datasets with GPU selection
Processes: codegen-2b, codellama-7b, codet5-770m on humaneval, mbpp
"""

import os
import sys
import json
import time
import logging
import argparse
from datetime import datetime
from pathlib import Path

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ranking_methods.llm_judge_ranker import LLMJudgeRanker

def setup_logging(log_file: str):
    """Setup logging to both file and console"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

def set_gpu_device(gpu_id: int):
    """Set GPU device for CUDA operations"""
    if gpu_id is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        print(f"Set CUDA_VISIBLE_DEVICES to {gpu_id}")

def load_candidates(candidates_dir: str, model: str, dataset: str):
    """Load candidate codes from directory"""
    candidates_path = Path(candidates_dir) / model / dataset

    if not candidates_path.exists():
        raise FileNotFoundError(f"Candidates directory not found: {candidates_path}")

    problem_files = list(candidates_path.glob("problem_*.json"))
    logger.info(f"Found {len(problem_files)} problem files for {model}/{dataset}")

    # Sort by problem number (numeric) instead of filename (string)
    def get_problem_number(file_path):
        try:
            return int(file_path.stem.split('_')[1])
        except (IndexError, ValueError):
            return 0

    return sorted(problem_files, key=get_problem_number)

def save_ranked_results(results, output_path: Path, model: str, dataset: str, problem_num: int):
    """Save ranking results with metadata"""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Add metadata
    output_data = {
        "timestamp": datetime.now().isoformat(),
        "method": "improved_llm_judge",
        "judge_model": "codellama/CodeLlama-7b-Python-hf",
        "target_model": model,
        "dataset": dataset,
        "problem_number": problem_num,
        "total_candidates": len(results),
        "results": []
    }

    for i, (candidate, score) in enumerate(results):
        output_data["results"].append({
            "rank": i + 1,
            "score": float(score),
            "candidate": candidate
        })

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

def process_model_dataset(judge, model: str, dataset: str, args, total_stats):
    """Process all problems for a specific model-dataset combination"""

    logger.info("="*80)
    logger.info(f"Processing {model} on {dataset}")
    logger.info("="*80)

    try:
        # Load problem files
        problem_files = load_candidates(args.input_dir, model, dataset)

        if args.start_problem > 0:
            problem_files = problem_files[args.start_problem:]
            logger.info(f"Starting from problem {args.start_problem}")

        if args.max_problems:
            problem_files = problem_files[:args.max_problems]
            logger.info(f"Limiting to {args.max_problems} problems")

        logger.info(f"Processing {len(problem_files)} problems for {model}/{dataset}")

        # Track statistics
        processed_count = 0
        error_count = 0
        start_time = time.time()

        for i, problem_file in enumerate(problem_files):
            try:
                # Extract problem number from filename
                problem_num = int(problem_file.stem.split('_')[1])
                actual_problem_num = args.start_problem + i if args.start_problem > 0 else problem_num

                logger.info(f"[{model}/{dataset}] Processing problem {actual_problem_num}: {problem_file.name}")

                # Load problem data
                with open(problem_file, 'r') as f:
                    problem_data = json.load(f)

                problem = problem_data.get('problem', {})
                candidates = problem_data.get('candidates', [])

                if not candidates:
                    logger.warning(f"No candidates found for problem {actual_problem_num}")
                    continue

                # Rank candidates
                ranked_results = judge.rank_candidates(problem, candidates)

                # Save results
                output_dir = Path(args.output_dir) / "llm_judge" / model / dataset
                output_file = output_dir / f"problem_{actual_problem_num}.json"
                save_ranked_results(ranked_results, output_file, model, dataset, actual_problem_num)

                processed_count += 1

                # Progress reporting
                elapsed_time = time.time() - start_time
                avg_time_per_problem = elapsed_time / (i + 1)
                estimated_remaining = avg_time_per_problem * (len(problem_files) - i - 1)

                logger.info(f"✓ [{model}/{dataset}] Problem {actual_problem_num}: Ranked {len(candidates)} candidates")
                logger.info(f"  Progress: {processed_count}/{len(problem_files)} problems completed")
                logger.info(f"  Time: {elapsed_time/60:.1f}min elapsed, {estimated_remaining/60:.1f}min remaining")

                # Brief pause to prevent overheating
                time.sleep(0.5)

            except Exception as e:
                error_count += 1
                logger.error(f"✗ [{model}/{dataset}] Error processing problem {actual_problem_num}: {e}")
                continue

        # Update total statistics
        total_stats["processed"] += processed_count
        total_stats["errors"] += error_count
        total_stats["models_completed"] += 1

        # Model-dataset completion summary
        success_rate = (processed_count / (processed_count + error_count) * 100) if (processed_count + error_count) > 0 else 0
        logger.info(f"")
        logger.info(f"[{model}/{dataset}] COMPLETION SUMMARY")
        logger.info(f"  Processed: {processed_count} problems")
        logger.info(f"  Errors: {error_count}")
        logger.info(f"  Success rate: {success_rate:.1f}%")
        logger.info(f"  Results saved to: {args.output_dir}/llm_judge/{model}/{dataset}/")

    except Exception as e:
        logger.error(f"CRITICAL ERROR processing {model}/{dataset}: {e}")
        total_stats["critical_errors"] += 1

def main():
    global logger

    parser = argparse.ArgumentParser(description="Run comprehensive LLM ranking for all models and datasets")
    parser.add_argument("--input_dir", type=str,
                       default="/home/fdse/srj/comparison/results/candidates/non_finetuned",
                       help="Input directory containing candidates")
    parser.add_argument("--output_dir", type=str,
                       default="/home/fdse/srj/comparison/results/ranked",
                       help="Output directory for ranked results")
    parser.add_argument("--gpu", type=int, default=None,
                       help="GPU device to use (e.g., --gpu 7 for GPU 7)")
    parser.add_argument("--models", nargs="+",
                       default=["codet5-770m", "codegen-2b", "codellama-7b"],
                       help="Models to process")
    parser.add_argument("--datasets", nargs="+",
                       default=["humaneval", "mbpp"],
                       help="Datasets to process")
    parser.add_argument("--start_problem", type=int, default=0,
                       help="Start from problem number (for resuming)")
    parser.add_argument("--max_problems", type=int, default=None,
                       help="Maximum number of problems per model/dataset")

    args = parser.parse_args()

    # Set GPU device if specified
    if args.gpu is not None:
        set_gpu_device(args.gpu)

    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    gpu_suffix = f"_gpu{args.gpu}" if args.gpu is not None else ""
    log_file = f"full_llm_ranking{gpu_suffix}_{timestamp}.log"
    logger = setup_logging(log_file)

    logger.info("="*80)
    logger.info("COMPREHENSIVE LLM JUDGE RANKING - FULL PIPELINE")
    logger.info("="*80)
    logger.info(f"Input directory: {args.input_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"GPU device: {args.gpu if args.gpu is not None else 'auto'}")
    logger.info(f"Models: {', '.join(args.models)}")
    logger.info(f"Datasets: {', '.join(args.datasets)}")
    logger.info(f"Start problem: {args.start_problem}")
    logger.info(f"Max problems per model/dataset: {args.max_problems}")
    logger.info("")

    # Calculate total workload
    total_combinations = len(args.models) * len(args.datasets)
    logger.info(f"Total model-dataset combinations: {total_combinations}")

    # Estimate total problems
    total_problems_estimate = 0
    for model in args.models:
        for dataset in args.datasets:
            candidates_path = Path(args.input_dir) / model / dataset
            if candidates_path.exists():
                problem_count = len(list(candidates_path.glob("problem_*.json")))
                if args.max_problems:
                    problem_count = min(problem_count, args.max_problems)
                total_problems_estimate += problem_count

    logger.info(f"Estimated total problems to process: {total_problems_estimate}")
    logger.info(f"Estimated time: {total_problems_estimate * 0.6:.1f} minutes (~{total_problems_estimate * 0.6/60:.1f} hours)")
    logger.info("")

    try:
        # Initialize LLM Judge
        logger.info("Initializing improved LLM judge with CodeLlama-7b-Python...")
        judge = LLMJudgeRanker(
            model_path="codellama/CodeLlama-7b-Python-hf",
            judge_type="local",
            max_new_tokens=512
        )
        logger.info("LLM judge loaded successfully")
        logger.info("")

        # Track overall statistics
        total_stats = {
            "processed": 0,
            "errors": 0,
            "models_completed": 0,
            "critical_errors": 0,
            "start_time": time.time()
        }

        # Process each model-dataset combination
        combination_count = 0
        for model in args.models:
            for dataset in args.datasets:
                combination_count += 1
                logger.info(f"Starting combination {combination_count}/{total_combinations}: {model} + {dataset}")

                process_model_dataset(judge, model, dataset, args, total_stats)

                # Brief pause between model-dataset combinations
                time.sleep(2)

        # Final statistics
        total_time = time.time() - total_stats["start_time"]
        logger.info("")
        logger.info("="*80)
        logger.info("FINAL COMPLETION SUMMARY")
        logger.info("="*80)
        logger.info(f"Total runtime: {total_time/60:.1f} minutes ({total_time/3600:.1f} hours)")
        logger.info(f"Model-dataset combinations completed: {total_stats['models_completed']}/{total_combinations}")
        logger.info(f"Total problems processed: {total_stats['processed']}")
        logger.info(f"Total errors: {total_stats['errors']}")
        logger.info(f"Critical errors: {total_stats['critical_errors']}")

        if total_stats["processed"] > 0:
            overall_success_rate = (total_stats["processed"] / (total_stats["processed"] + total_stats["errors"]) * 100)
            avg_time_per_problem = total_time / total_stats["processed"]
            logger.info(f"Overall success rate: {overall_success_rate:.1f}%")
            logger.info(f"Average time per problem: {avg_time_per_problem:.1f} seconds")

        logger.info(f"Results saved to: {args.output_dir}/llm_judge/")

        # Cleanup
        judge.cleanup()
        logger.info("LLM judge cleanup completed")
        logger.info("COMPREHENSIVE RANKING PROCESS FINISHED SUCCESSFULLY!")

    except Exception as e:
        logger.error(f"CRITICAL SYSTEM ERROR: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1

    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)