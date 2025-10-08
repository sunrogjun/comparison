#!/usr/bin/env python3
"""
Run candidate code generation for specific datasets only
"""
import os
import sys
import logging
from typing import List

# Set CUDA device before importing torch
os.environ["CUDA_VISIBLE_DEVICES"] = "6"

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_preparation.candidate_generator import generate_all_candidates

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    print("=" * 60)
    print("CODE GENERATION PHASE")
    print("=" * 60)
    
    # Configuration
    models = ["codet5-770m", "codegen-2b", "codellama-7b"]
    # 只生成APPS数据集的候选代码
    datasets = ["apps"]  # 修改为只包含apps数据集
    output_dir = "/home/fdse/srj/comparison/results/candidates/non_finetuned"
    num_candidates = 100
    
    print(f"Models: {', '.join(models)}")
    print(f"Datasets: {', '.join(datasets)}")  # 更新打印信息
    print(f"Output: {output_dir}")
    print()
    
    # Verify output directory
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Results directory verified: {output_dir}")
    
    # Generate candidates for all model/dataset combinations
    try:
        generate_all_candidates(
            models=models,
            datasets=datasets,  # 使用修改后的数据集列表
            output_dir=output_dir,
            num_candidates=num_candidates
        )
        logger.info("Candidate generation completed successfully.")
    except Exception as e:
        logger.error(f"Error in generation phase: {e}")
        raise

if __name__ == "__main__":
    main()