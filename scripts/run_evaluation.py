#!/usr/bin/env python3
"""
Unified Code Ranking Evaluation Script
Combines the best features from all evaluation scripts and ensures correct execution
"""

import os
import sys
import json
import math
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import argparse
import time
import logging
from datetime import datetime

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from evaluation.metrics import RankingMetrics
from evaluation.code_executor import CodeExecutor



def _to_serializable(value: Any) -> Any:
    """Convert numpy types to plain Python types for JSON serialization."""
    if isinstance(value, (np.floating, np.integer)):
        value = float(value)
    elif isinstance(value, (float, int)):
        value = float(value)
    elif isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return 0.0
    return value


def _sanitize_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Return a metrics dict with JSON-serializable numeric values."""
    return {key: _to_serializable(value) for key, value in metrics.items()}



def _make_problem_summary(problem_id: str, execution_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Build a summary for a single problem that preserves ranking-sensitive data."""
    total = len(execution_results)
    passed_positions = [index + 1 for index, result in enumerate(execution_results)
                        if result.get('passed', False)]
    first_correct = passed_positions[0] if passed_positions else None

    return {
        'problem': problem_id,
        'passed': len(passed_positions),
        'total': total,
        'first_correct_position': first_correct,
        'passed_positions': passed_positions,
        'pass_at_1': 1.0 if first_correct and first_correct <= 1 else 0.0,
        'pass_at_2': 1.0 if first_correct and first_correct <= 2 else 0.0,
        'pass_at_5': 1.0 if first_correct and first_correct <= 5 else 0.0,
        'pass_at_10': 1.0 if first_correct and first_correct <= 10 else 0.0,
        'pass_at_20': 1.0 if first_correct and first_correct <= 20 else 0.0,
    }
def setup_logging(log_dir="logs"):
    """Setup logging for the evaluation"""
    log_dir = Path(log_dir)
    log_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"evaluation_{timestamp}.log"

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )

    return logging.getLogger(__name__)

def check_data_availability(results_dir):
    """Check if ranking results are available"""
    results_path = Path(results_dir) / "ranked"

    if not results_path.exists():
        logging.error(f"Ranked results directory not found: {results_path}")
        return False

    # Count available result files
    total_files = 0
    for json_file in results_path.rglob("*.json"):
        total_files += 1

    logging.info(f"Found {total_files} ranking result files in {results_path}")
    return total_files > 0

def load_ranking_results(results_dir, method, model, dataset):
    """Load ranked results for a specific method/model/dataset combination"""
    results_path = Path(results_dir) / "ranked" / method / model / dataset

    if not results_path.exists():
        logging.warning(f"Results path does not exist: {results_path}")
        return []

    all_results = []
    for result_file in sorted(results_path.glob("*.json")):
        try:
            with open(result_file, 'r') as f:
                data = json.load(f)

            # Handle different data formats
            if 'results' in data:  # LLM Judge format
                # IMPORTANT: Sort by rank to maintain correct order!
                sorted_results = sorted(data['results'], key=lambda x: x['rank'])
                candidates = [item['candidate'] for item in sorted_results]
                scores = [item['score'] for item in sorted_results]
                all_results.append({
                    'problem_id': result_file.stem,
                    'ranked_candidates': candidates,
                    'execution_results': [],  # Always start with empty execution results
                    'scores': scores,
                    'method': method  # Add method identifier to avoid confusion
                })
            else:  # AceCoder format
                # Extract candidates from [code, score] pairs
                ranked_candidates_raw = data.get('ranked_candidates', [])
                if ranked_candidates_raw and isinstance(ranked_candidates_raw[0], list):
                    # Format: [[code, score], [code, score], ...]
                    candidates = [item[0] for item in ranked_candidates_raw]
                    scores = [item[1] for item in ranked_candidates_raw]
                else:
                    # Format: [code, code, code, ...]
                    candidates = ranked_candidates_raw
                    scores = data.get('scores', [])

                all_results.append({
                    'problem_id': result_file.stem,
                    'ranked_candidates': candidates,
                    'execution_results': [],  # Always start with empty execution results to force re-execution
                    'scores': scores,
                    'method': method  # Add method identifier to avoid confusion
                })
        except Exception as e:
            logging.error(f"Error loading {result_file}: {e}")
            continue

    return all_results

def load_problem_info(problem_id, dataset, data_dir="data"):
    """Load real problem information for execution"""
    try:
        # Extract number from problem_id (e.g., "problem_161" -> "161")
        if problem_id.startswith("problem_"):
            problem_number = problem_id.replace("problem_", "")
        else:
            problem_number = problem_id

        # Check various possible locations for problem data
        possible_paths = [
            Path(f"{data_dir}/{dataset}/{problem_number}.json"),
            Path(f"{data_dir}/{dataset}/problems/{problem_number}.json"),
            Path(f"{data_dir}/{dataset}/{problem_id}.json"),
            Path(f"data/problems/{dataset}/{problem_number}.json"),
            Path(f"data/{dataset}/{problem_number}.json"),
        ]

        for problem_file in possible_paths:
            if problem_file.exists():
                with open(problem_file, 'r') as f:
                    problem_data = json.load(f)
                    logging.debug(f"Loaded problem info for {problem_id} from {problem_file}")

                    # Convert to standard format for code executor
                    if dataset == "mbpp" and 'test_list' in problem_data:
                        # MBPP format: convert test_list to executable test code
                        test_code = "def check(candidate_func):\n"
                        test_code += "    try:\n"

                        # Extract function name from the first test
                        first_test = problem_data['test_list'][0]
                        func_name = first_test.split('(')[0].replace('assert ', '').strip()

                        # Check if this function modifies in-place or returns a value
                        # For sorting functions, we need to handle in-place modifications
                        if 'sort' in func_name.lower():
                            # Handle in-place sorting functions
                            for test in problem_data['test_list']:
                                # Convert "assert comb_sort([5, 15, 37, 25, 79]) == [5, 15, 25, 37, 79]"
                                # to proper in-place test
                                test_parts = test.replace('assert ', '').split(' == ')
                                if len(test_parts) == 2:
                                    func_call = test_parts[0].strip()
                                    expected = test_parts[1].strip()

                                    # Extract array from function call
                                    array_start = func_call.find('[')
                                    array_end = func_call.find(']') + 1
                                    if array_start != -1 and array_end != -1:
                                        array_str = func_call[array_start:array_end]
                                        test_code += f"        test_arr = {array_str}\n"
                                        test_code += f"        {func_name}(test_arr)\n"
                                        test_code += f"        assert test_arr == {expected}\n"
                                    else:
                                        test_code += f"        {test}\n"
                                else:
                                    test_code += f"        {test}\n"
                        else:
                            # Regular function that returns a value
                            for test in problem_data['test_list']:
                                test_code += f"        {test}\n"

                        test_code += "        return True\n"
                        test_code += "    except Exception as e:\n"
                        test_code += "        return False\n"

                        return {
                            'problem_id': problem_id,
                            'dataset': dataset,
                            'prompt': problem_data.get('prompt', ''),
                            'test': test_code,
                            'entry_point': func_name
                        }
                    elif dataset == "humaneval" and 'test' in problem_data:
                        # HumanEval format: already has test code
                        return {
                            'problem_id': problem_id,
                            'dataset': dataset,
                            'prompt': problem_data.get('prompt', ''),
                            'test': problem_data['test'],
                            'entry_point': problem_data.get('entry_point', 'solution')
                        }
                    else:
                        return problem_data

        # If no problem file found, log warning and return None
        logging.warning(f"Problem file not found for {problem_id} in dataset {dataset}")
        return None

    except Exception as e:
        logging.error(f"Could not load problem info for {problem_id}: {e}")
        return None

def execute_candidates_with_real_executor(result_data, code_executor, problem_info):
    """Execute candidates using real code executor"""

    # Always execute candidates fresh to ensure different ranking methods are evaluated correctly
    # This prevents caching issues that could cause different ranking methods to have identical results

    if not result_data.get('ranked_candidates'):
        return []

    candidates = result_data['ranked_candidates']
    execution_results = []

    method = result_data.get('method', 'unknown')
    logging.info(f"Executing {len(candidates)} candidates for problem {result_data['problem_id']} using method {method}")

    # Log first few candidates to verify ranking differences
    logging.debug(f"First 3 candidates for {method}: {[c[:50] + '...' if len(c) > 50 else c for c in candidates[:3]]}")

    for i, candidate in enumerate(candidates):
        try:
            if problem_info:
                result = code_executor.execute_candidate_safe(problem_info, candidate)
            else:
                result = {
                    'passed': False,
                    'error': 'No problem info available',
                    'timeout': False,
                    'execution_time': 0.0
                }

            execution_results.append(result)

        except Exception as e:
            logging.error(f"Execution error for candidate {i}: {e}")
            execution_results.append({
                'passed': False,
                'error': f"Execution error: {str(e)}",
                'timeout': False,
                'execution_time': 0.0
            })

    passed_count = sum(1 for r in execution_results if r['passed'])
    logging.info(f"Problem {result_data['problem_id']}: {passed_count}/{len(candidates)} candidates passed")

    return execution_results


def evaluate_method(results_dir, method, model, dataset, code_executor) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    # Evaluate a single ranking method and return metrics plus per-problem summaries.
    logging.info(f"Evaluating {method} with {model} on {dataset}...")

    results = load_ranking_results(results_dir, method, model, dataset)

    if not results:
        logging.warning(f"No results found for {method}/{model}/{dataset}")
        return get_empty_metrics(), []

    all_execution_results: List[List[Dict[str, Any]]] = []
    problem_summaries: List[Dict[str, Any]] = []

    for result in results:
        problem_info = load_problem_info(result['problem_id'], dataset)
        execution_results = execute_candidates_with_real_executor(result, code_executor, problem_info)
        all_execution_results.append(execution_results)
        problem_summaries.append(_make_problem_summary(result['problem_id'], execution_results))

    if not all_execution_results or all(len(r) == 0 for r in all_execution_results):
        logging.warning(f"No execution results for {method}/{model}/{dataset}")
        return get_empty_metrics(), problem_summaries

    metrics = RankingMetrics.calculate_ranking_quality_metrics(all_execution_results)

    for k in [1, 2, 5, 10]:
        if f'pass_at_{k}' not in metrics:
            metrics[f'pass_at_{k}'] = RankingMetrics.calculate_pass_at_k_batch(all_execution_results, k)

    return metrics, problem_summaries

def get_empty_metrics():
    """Return empty metrics dictionary for failed evaluations"""
    return {
        'pass_at_1': 0.0,
        'pass_at_2': 0.0,
        'pass_at_5': 0.0,
        'pass_at_10': 0.0,
        'pass_at_20': 0.0,
        'mrr': 0.0,
        'ndcg_at_5': 0.0,
        'ndcg_at_10': 0.0,
        'success_rate': 0.0,
        'avg_first_correct_position': float('inf'),
        'median_first_correct_position': float('inf')
    }

def aggregate_results(method_results):
    """Aggregate metrics across multiple evaluations"""
    if not method_results:
        return get_empty_metrics()

    aggregated = {}
    all_keys = set()
    for result in method_results:
        all_keys.update(result.keys())

    for key in all_keys:
        values = [result.get(key, 0.0) for result in method_results]
        finite_values = [v for v in values if np.isfinite(v)]

        if finite_values:
            aggregated[key] = np.mean(finite_values)
        else:
            aggregated[key] = 0.0

    return aggregated

def create_comparison_table(results):
    """Create a comprehensive comparison table"""
    key_metrics = ['pass_at_1', 'pass_at_2', 'pass_at_5', 'pass_at_10', 'mrr', 'success_rate']

    df_data = []
    for method, metrics in results.items():
        row = {'Method': method}
        for metric in key_metrics:
            value = metrics.get(metric, 0.0)
            if np.isnan(value) or np.isinf(value):
                row[metric] = 0.0
            else:
                row[metric] = value
        df_data.append(row)

    df = pd.DataFrame(df_data)
    numeric_cols = [col for col in df.columns if col != 'Method']
    df[numeric_cols] = df[numeric_cols].round(4)

    return df

def save_results(results, output_dir):
    """Save all evaluation results and return structured outputs."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    df = create_comparison_table(results)
    table_path = output_path / "ranking_comparison_table.csv"
    df.to_csv(table_path, index=False)
    logging.info(f"✓ Comparison table saved to: {table_path}")

    results_path = output_path / "raw_evaluation_results.json"
    json_results = {method: _sanitize_metrics(metrics) for method, metrics in results.items()}

    with open(results_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    logging.info(f"✓ Raw results saved to: {results_path}")

    return df, json_results

def save_single_result(method, model, dataset, metrics, problem_summaries, output_dir):
    """Save results for a single evaluation run."""
    output_path = Path(output_dir)
    output_path.mkdir(exosts=True)
    
    # Create a unique filename for this specific evaluation
    # Add "apps_" prefix for apps dataset to distinguish from existing files
    if dataset == "apps":
        filename = f"{dataset}_{method}_{model}_results.json"
    else:
        filename = f"{method}_{model}_{dataset}_results.json"
    result_file = output_path / filename
    
    result_data = {
        'method': method,
        'model': model,
        'dataset': dataset,
        'metrics': _sanitize_metrics(metrics),
        'problems': problem_summaries
    }
    
    with open(result_file, 'w') as f:
        json.dump(result_data, f, indent=2)
    logging.info(f"✓ Results saved to: {result_file}")
    
    # Also save checkpoint
    checkpoint_file = output_path / "evaluation_checkpoint.json"
    checkpoint_data = {
        "last_evaluated": {
            "method": method,
            "model": model,
            "dataset": dataset
        },
        "timestamp": datetime.now().isoformat()
    }
    
    with open(checkpoint_file, 'w') as f:
        json.dump(checkpoint_data, f, indent=2)
    logging.info(f"✓ Checkpoint saved to: {checkpoint_file}")

def load_checkpoint(output_dir):
    """Load the last evaluation checkpoint to resume from where we left off."""
    checkpoint_file = Path(output_dir) / "evaluation_checkpoint.json"
    if checkpoint_file.exists():
        try:
            with open(checkpoint_file, 'r') as f:
                checkpoint = json.load(f)
            logging.info(f"Found checkpoint: {checkpoint}")
            return checkpoint.get("last_evaluated", {})
        except Exception as e:
            logging.warning(f"Could not load checkpoint: {e}")
    return {}

def should_skip_evaluation(method, model, dataset, checkpoint, args):
    """Determine if we should skip this evaluation based on checkpoint and resume flag."""
    if not args.resume:
        return False
    
    if not checkpoint:
        return False
        
    last = checkpoint
    methods = args.methods
    models = args.models
    datasets = args.datasets
    
    # Create a list of all combinations to determine the order
    all_combinations = []
    for m in methods:
        for mo in models:
            for d in datasets:
                all_combinations.append((m, mo, d))
    
    # Find the index of the last evaluated combination and current combination
    try:
        last_index = all_combinations.index((last["method"], last["model"], last["dataset"]))
        current_index = all_combinations.index((method, model, dataset))
        # Skip if current combination was evaluated before the checkpoint
        return current_index <= last_index
    except ValueError:
        # If combination not found, don't skip
        return False

def export_legacy_reports(combination_details: List[Dict[str, Any]], method_metrics: Dict[str, Dict[str, Any]],
                          output_dir: str = "results/evaluation") -> None:
    """Recreate legacy summary artifacts while preserving ranking information."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    detailed_payload: Dict[str, Any] = {}
    comprehensive_payload: Dict[str, Any] = {}
    detailed_rows: List[Dict[str, Any]] = []
    comprehensive_rows: List[Dict[str, Any]] = []

    for detail in combination_details:
        method = detail['method']
        model = detail['model']
        dataset = detail['dataset']
        metrics = _sanitize_metrics(detail.get('metrics', {}))
        problems = detail.get('problems', [])
        # Add "apps_" prefix for apps dataset to distinguish from existing files
        if dataset == "apps":
            config_key = f"{dataset}_{method}_{model}"
        else:
            config_key = f"{method}_{model}_{dataset}"
        total_problems = len(problems)
        candidate_counts = [problem.get('total', 0) for problem in problems]
        uniform_candidates = candidate_counts[0] if candidate_counts and len(set(candidate_counts)) == 1 else None

        sanitized_problems = []
        for problem in problems:
            sanitized_problems.append({
                'problem': problem.get('problem'),
                'passed': int(problem.get('passed', 0)),
                'total': int(problem.get('total', 0)),
                'first_correct_position': problem.get('first_correct_position'),
                'passed_positions': problem.get('passed_positions', []),
                'pass_at_1': float(problem.get('pass_at_1', 0.0)),
                'pass_at_2': float(problem.get('pass_at_2', 0.0)),
                'pass_at_5': float(problem.get('pass_at_5', 0.0)),
                'pass_at_10': float(problem.get('pass_at_10', 0.0)),
                'pass_at_20': float(problem.get('pass_at_20', 0.0)),
            })

        detail_entry = {
            'method': method,
            'model': model,
            'dataset': dataset,
            'total_problems': total_problems,
            'total_candidates_per_problem': uniform_candidates,
            'problems': sanitized_problems,
            'metrics': metrics,
            'valid': detail.get('valid', True),
        }
        if uniform_candidates is None and candidate_counts:
            detail_entry['candidate_counts'] = candidate_counts
        detailed_payload[config_key] = detail_entry

        problems_with_passing = sum(1 for problem in sanitized_problems if problem['passed'] > 0)
        total_passing_candidates = sum(problem['passed'] for problem in sanitized_problems)
        total_candidates = sum(problem['total'] for problem in sanitized_problems)

        comprehensive_payload[config_key] = {
            'method': method,
            'model': model,
            'dataset': dataset,
            'total_problems': total_problems,
            'total_ranking_files': total_problems,
            'metrics': metrics,
            'execution_summary': {
                'problems_with_passing_candidates': problems_with_passing,
                'total_passing_candidates': total_passing_candidates,
                'total_candidates': total_candidates,
            }
        }

        detailed_rows.append({
            'method': method,
            'model': model,
            'dataset': dataset,
            'configuration': config_key,
            'pass_at_1': metrics.get('pass_at_1', 0.0),
            'pass_at_2': metrics.get('pass_at_2', 0.0),
            'pass_at_5': metrics.get('pass_at_5', 0.0),
            'pass_at_10': metrics.get('pass_at_10', 0.0),
            'mrr': metrics.get('mrr', 0.0),
            'success_rate': metrics.get('success_rate', 0.0),
            'total_problems': total_problems,
        })

        comprehensive_rows.append({
            'method': method,
            'model': model,
            'dataset': dataset,
            'configuration': config_key,
            'pass_at_1': metrics.get('pass_at_1', 0.0),
            'pass_at_2': metrics.get('pass_at_2', 0.0),
            'pass_at_5': metrics.get('pass_at_5', 0.0),
            'pass_at_10': metrics.get('pass_at_10', 0.0),
            'mrr': metrics.get('mrr', 0.0),
            'success_rate': metrics.get('success_rate', 0.0),
            'total_problems': total_problems,
            'problems_with_solutions': problems_with_passing,
        })

    # Add timestamp to distinguish new files from existing ones
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    (output_path / f'detailed_evaluation_results_{timestamp}.json').write_text(json.dumps(detailed_payload, indent=2))
    (output_path / f'comprehensive_evaluation_results_{timestamp}.json').write_text(json.dumps(comprehensive_payload, indent=2))

    detailed_df = pd.DataFrame(detailed_rows)
    if not detailed_df.empty:
        detailed_df = detailed_df.round({'pass_at_1': 4, 'pass_at_2': 4, 'pass_at_5': 4, 'pass_at_10': 4, 'mrr': 4, 'success_rate': 4})
    detailed_df.to_csv(output_path / f'detailed_comparison_table_{timestamp}.csv', index=False)

    comprehensive_df = pd.DataFrame(comprehensive_rows)
    if not comprehensive_df.empty:
        comprehensive_df = comprehensive_df.round({'pass_at_1': 4, 'pass_at_2': 4, 'pass_at_5': 4, 'pass_at_10': 4, 'mrr': 4, 'success_rate': 4})
    comprehensive_df.to_csv(output_path / f'comprehensive_comparison_table_{timestamp}.csv', index=False)

    (output_path / f'raw_evaluation_results_{timestamp}.json').write_text(json.dumps(method_metrics, indent=2))

def evaluate_ranking_results(results_dir: str, datasets: List[str] = None, 
                           methods: List[str] = None, models: List[str] = None,
                           output_dir: str = "evaluation_outputs"):
    """Evaluate all ranking results and generate comprehensive metrics"""
    
    if datasets is None:
        datasets = ["humaneval", "mbpp", "apps"]  # Add apps dataset
    
    if methods is None:
        methods = ["llm_judge", "acecoder_rm"]
        
    if models is None:
        models = ["codet5-770m", "codegen-2b", "codellama-7b"]

    # Setup logging
    logger = setup_logging()

    logger.info("Starting comprehensive evaluation...")
    logger.info(f"Methods: {methods}")
    logger.info(f"Models: {models}")
    logger.info(f"Datasets: {datasets}")

    # Check data availability
    if not check_data_availability(results_dir):
        logger.error("No ranking results found. Please run ranking first.")
        return

    # Initialize results storage
    all_metrics = {}
    problem_summaries = {}
    
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    
    # Initialize code executor
    executor = CodeExecutor()
    
    # Evaluate each method/model/dataset combination
    for method in methods:
        all_metrics[method] = {}
        problem_summaries[method] = {}
        
        for model in models:
            all_metrics[method][model] = {}
            problem_summaries[method][model] = {}
            
            for dataset in datasets:
                logger.info(f"Evaluating {method} / {model} / {dataset}")
                
                # Load ranking results
                ranking_results = load_ranking_results(results_dir, method, model, dataset)
                
                if not ranking_results:
                    logger.warning(f"No results found for {method}/{model}/{dataset}")
                    continue
                
                all_metrics[method][model][dataset] = {}
                problem_summaries[method][model][dataset] = {}
                
                # Process each problem
                problem_executions = []
                for problem_result in ranking_results:
                    problem_id = problem_result['problem_id']
                    ranked_candidates = problem_result['ranked_candidates']
                    
                    # Load problem info
                    problem_info = load_problem_info(problem_id, dataset)
                    if not problem_info:
                        logger.warning(f"Could not load problem info for {problem_id}")
                        continue
                    
                    # Execute candidates
                    execution_results = []
                    for i, candidate in enumerate(ranked_candidates):
                        if not candidate or candidate.strip() == "":
                            execution_results.append({
                                'passed': False,
                                'output': '',
                                'error': 'Empty candidate'
                            })
                            continue
                        
                        try:
                            result = executor.execute_candidate(problem_info, candidate)
                            execution_results.append(result)
                        except Exception as e:
                            execution_results.append({
                                'passed': False,
                                'output': '',
                                'error': str(e)
                            })
                    
                    # Store execution results
                    problem_result['execution_results'] = execution_results
                    
                    # Create problem summary
                    summary = _make_problem_summary(problem_id, execution_results)
                    problem_summaries[method][model][dataset][problem_id] = summary
                    problem_executions.append(summary)
                
                # Calculate dataset-level metrics
                if problem_executions:
                    metrics = RankingMetrics.calculate_metrics(problem_executions)
                    all_metrics[method][model][dataset] = _sanitize_metrics(metrics)
                    logger.info(f"  Results: {metrics}")
    
    # Generate comprehensive report
    logger.info("Generating comprehensive evaluation report...")
    
    # Save detailed results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save problem summaries
    with open(f"{output_dir}/problem_summaries_{timestamp}.json", 'w') as f:
        json.dump(_sanitize_metrics(problem_summaries), f, indent=2)
    
    # Save metrics
    with open(f"{output_dir}/comprehensive_evaluation_results_{timestamp}.json", 'w') as f:
        json.dump(_sanitize_metrics(all_metrics), f, indent=2)
    
    # Generate comparison tables
    comparison_data = []
    for method in methods:
        for model in models:
            for dataset in datasets:
                if (method in all_metrics and 
                    model in all_metrics[method] and 
                    dataset in all_metrics[method][model]):
                    
                    metrics = all_metrics[method][model][dataset]
                    comparison_data.append({
                        'Method': method,
                        'Model': model,
                        'Dataset': dataset,
                        'pass@1': metrics.get('pass_at_1', 0),
                        'pass@2': metrics.get('pass_at_2', 0),
                        'pass@5': metrics.get('pass_at_5', 0),
                        'pass@10': metrics.get('pass_at_10', 0),
                        'pass@20': metrics.get('pass_at_20', 0),
                        'mrr': metrics.get('mrr', 0),
                        'success_rate': metrics.get('success_rate', 0)
                    })
    
    if comparison_data:
        df = pd.DataFrame(comparison_data)
        df.to_csv(f"{output_dir}/detailed_comparison_table_{timestamp}.csv", index=False)
        
        # Generate summary table
        summary_data = []
        for dataset in datasets:
            for model in models:
                row = {'Dataset': dataset, 'Model': model}
                for method in methods:
                    for metric in ['pass@1', 'pass@2', 'pass@5', 'pass@10', 'mrr']:
                        col_name = f"{method}_{metric}"
                        val = 0
                        for item in comparison_data:
                            if (item['Method'] == method and 
                                item['Model'] == model and 
                                item['Dataset'] == dataset):
                                val = item.get(metric, 0)
                                break
                        row[col_name] = round(val, 4)
                summary_data.append(row)
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(f"{output_dir}/comparison_table_{timestamp}.csv", index=False)
        
        logger.info("Evaluation completed successfully!")
        logger.info(f"Results saved to {output_dir}")
        print("\n" + "="*80)
        print("EVALUATION RESULTS")
        print("="*80)
        print(summary_df.to_string(index=False))
        
        return all_metrics
    
    logger.warning("No evaluation data generated")
    return {}

def main():
    parser = argparse.ArgumentParser(description='Run unified code ranking evaluation')
    parser.add_argument('--results-dir', default='results',
                       help='Directory containing ranking results')
    parser.add_argument('--methods', nargs='+',
                       default=['llm_judge', 'acecoder_rm'],
                       help='Ranking methods to evaluate')
    parser.add_argument('--models', nargs='+',
                       default=['codet5-770m', 'codellama-7b', 'codegen-2b'],
                       help='Models to evaluate')
    parser.add_argument('--datasets', nargs='+',
                       default=['humaneval', 'mbpp', 'apps'],  # Add apps dataset
                       help='Datasets to evaluate')
    parser.add_argument('--output-dir', default='evaluation_outputs',
                       help='Output directory for results')
    parser.add_argument('--timeout', type=int, default=60,
                       help='Timeout for code execution (seconds)')
    parser.add_argument('--single-eval', action='store_true',
                       help='Evaluate one method/model/dataset combination at a time and output results immediately')
    parser.add_argument('--resume', action='store_true',
                       help='Resume evaluation from the last checkpoint')

    args = parser.parse_args()

    # Setup logging
    logger = setup_logging()

    logger.info("=" * 80)
    logger.info("UNIFIED CODE RANKING EVALUATION")
    logger.info("=" * 80)
    logger.info(f"Methods: {args.methods}")
    logger.info(f"Models: {args.models}")
    logger.info(f"Datasets: {args.datasets}")
    logger.info(f"Results directory: {args.results_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info("=" * 80)

    # Check if ranking results are available
    if not check_data_availability(args.results_dir):
        logger.error("No ranking results found! Please run the ranking methods first.")
        return

    # Load checkpoint if resume is enabled
    checkpoint = {}
    if args.resume:
        checkpoint = load_checkpoint(args.output_dir)
        if checkpoint:
            logger.info(f"Resuming evaluation from checkpoint: {checkpoint}")

    # Initialize code executor
    code_executor = CodeExecutor(timeout=args.timeout)

    # Evaluate all methods
    all_results = {}
    combination_details: List[Dict[str, Any]] = []
    start_time = time.time()

    for method in args.methods:
        method_results = []

        for model in args.models:
            for dataset in args.datasets:
                # Check if we should skip this evaluation
                if should_skip_evaluation(method, model, dataset, checkpoint, args):
                    logger.info(f"Skipping already evaluated combination: {method} + {model} + {dataset}")
                    continue
                    
                logger.info(f"\n{'-'*40}")
                logger.info(f"Evaluating: {method} + {model} + {dataset}")
                logger.info(f"{'-'*40}")

                metrics, problem_summaries = evaluate_method(
                    args.results_dir, method, model, dataset, code_executor
                )

                metric_values = [
                    value for value in metrics.values()
                    if isinstance(value, (int, float)) and np.isfinite(value)
                ]
                has_valid_metrics = any(value > 0 for value in metric_values)

                combination_details.append({
                    'method': method,
                    'model': model,
                    'dataset': dataset,
                    'metrics': metrics,
                    'problems': problem_summaries,
                    'valid': has_valid_metrics,
                })

                if args.single_eval:
                    # Save and display results immediately for single evaluation mode
                    save_single_result(method, model, dataset, metrics, problem_summaries, args.output_dir)
                    logger.info(f"✓ Completed evaluation for {method}/{model}/{dataset}")
                    logger.info("Metrics:")
                    for key, value in metrics.items():
                        if isinstance(value, (int, float)) and np.isfinite(value):
                            logger.info(f"  {key}: {value:.4f}")
                else:
                    # Original behavior - collect results for aggregation
                    if has_valid_metrics:
                        method_results.append(metrics)
                        logger.info(f"✓ {method}/{model}/{dataset}: Valid results found")
                    else:
                        logger.warning(f"✗ {method}/{model}/{dataset}: No valid results")

        if not args.single_eval:
            # Aggregate results across models and datasets
            if method_results:
                aggregated = aggregate_results(method_results)
                all_results[method] = aggregated
                logger.info(f"✓ {method}: Aggregated {len(method_results)} configurations")
            else:
                all_results[method] = get_empty_metrics()
                logger.warning(f"✗ {method}: No valid results")

    total_time = time.time() - start_time

    if args.single_eval:
        logger.info(f"\n{'='*80}")
        logger.info(f"SINGLE EVALUATION MODE COMPLETE - Total time: {total_time/60:.1f} minutes")
        logger.info(f"{'='*80}")
        logger.info(f"📊 Individual results saved to: {args.output_dir}")
    else:
        # Check if we have any valid results
        if not any(any(v > 0 for v in metrics.values()
                      if isinstance(v, (int, float)) and np.isfinite(v))
                  for metrics in all_results.values()):
            logger.warning("WARNING: No valid evaluation results found!")
            logger.warning("This might be due to:")
            logger.warning("1. Missing problem data files in the data directory")
            logger.warning("2. Incorrect file paths or formats")
            logger.warning("3. Code execution issues")
            logger.warning("Please check the logs above for specific error messages.")

        # Save results and create summary
        df, raw_method_metrics = save_results(all_results, args.output_dir)
        export_legacy_reports(combination_details, raw_method_metrics, args.output_dir)  # Pass output_dir to export_legacy_reports

        # Print final summary
        logger.info(f"\n{'='*80}")
        logger.info(f"EVALUATION COMPLETE - Total time: {total_time/60:.1f} minutes")
        logger.info(f"{'='*80}")
        logger.info("\nFinal Comparison Table:")
        logger.info(df.to_string(index=False))

        logger.info(f"\n📊 All outputs saved to: {args.output_dir}")
        logger.info('Legacy-style summaries updated in results/evaluation/')

if __name__ == "__main__":
    main()