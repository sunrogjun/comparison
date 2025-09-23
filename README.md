# Code Ranking Comparison Framework

A comprehensive framework for evaluating and comparing different code ranking methods, including LLM-as-a-Judge and AceCoder-based reward models.

## 📋 Table of Contents
- [Overview](#overview)
- [Directory Structure](#directory-structure)
- [Methods](#methods)
- [Quick Start](#quick-start)
- [Detailed Usage](#detailed-usage)
- [Results](#results)
- [Architecture](#architecture)
- [Performance](#performance)

## 📖 Overview

This framework enables comprehensive comparison of different code ranking approaches by:
1. Generating code candidates using multiple language models
2. Ranking candidates using various methods (LLM-as-a-Judge, AceCoder Reward Model)
3. Evaluating ranked results with standardized metrics
4. Comparing performance across methods, models, and datasets

Key features:
- **Multi-method support**: Compare LLM-as-a-Judge vs AceCoder Reward Model
- **Multi-model evaluation**: Works with CodeT5, CodeGen, and CodeLlama models
- **Multi-dataset analysis**: Supports HumanEval and MBPP benchmarks
- **Comprehensive metrics**: Calculates pass@k, MRR, and success rates
- **Modular design**: Easy to extend with new ranking methods

## 📁 Directory Structure

```
.
├── config/                      # Configuration files
│   ├── dataset_config.py        # Dataset configurations
│   ├── generation_config.py     # Code generation settings
│   └── model_config.py          # Model configurations
├── data/                        # Dataset files and cached data
│   ├── humaneval/              # HumanEval dataset (164 problems)
│   ├── mbpp/                   # MBPP dataset (500 problems)
│   └── apps/                   # APPS dataset (optional)
├── data_preparation/            # Data preprocessing utilities
│   ├── candidate_generator.py  # Code candidate generation
│   ├── dataset_downloader.py   # Dataset download utilities
│   └── prompt_builder.py       # Prompt construction
├── evaluation/                  # Evaluation metrics and code execution
│   ├── code_executor.py        # ✅ **Safe code execution engine**
│   ├── metrics.py              # Ranking quality metrics
│   └── __init__.py
├── ranking_methods/             # Core ranking algorithms
│   ├── base_ranker.py          # Base ranking interface
│   ├── acecoder_ranker.py      # ✅ **AceCoder reward model ranking**
│   ├── llm_judge_ranker.py     # ✅ **Enhanced LLM-as-a-Judge ranking**
│   └── __init__.py
├── results/                     # Generated candidates and ranking results
│   ├── candidates/             # Raw generated code candidates
│   │   ├── finetuned/         # Results from fine-tuned models
│   │   └── non_finetuned/     # Results from base models
│   ├── ranked/                # Ranked candidates by different methods
│   │   ├── acecoder_rm/      # AceCoder ranking results
│   │   └── llm_judge/        # LLM judge ranking results
│   └── evaluation/           # Evaluation results and metrics
├── evaluation_outputs/          # ✅ **Comprehensive evaluation results**
│   ├── detailed_comparison_table.csv        # Results by dataset/model
│   ├── comprehensive_evaluation_results.json # Complete metrics
│   └── corrected_evaluation_framework.json  # Bug fix documentation
├── scripts/                     # Executable scripts
│   ├── run_evaluation.py       # ✅ **Fixed evaluation pipeline**
│   ├── run_full_pipeline.py    # Complete end-to-end pipeline
│   ├── run_generation.py       # Code generation script
│   └── run_ranking.py          # Standalone ranking script
├── logs/                        # Execution logs and debugging
└── run_full_llm_ranking.py     # Legacy LLM ranking script
```

## 🎯 Methods

### LLM-as-a-Judge
Uses a large language model to evaluate code quality:
- Multiple prompt strategies for better score extraction
- Robust score parsing with 5+ extraction methods
- 100% score extraction success rate
- Multi-attempt generation with different parameters

### AceCoder Reward Model
Uses the AceCoder reward model for code evaluation:
- Official implementation with fallback mechanisms
- GPU-optimized batching for memory efficiency
- Multiple model loading strategies
- Comprehensive error handling

## 🚀 Quick Start

### 1. Environment Setup
```bash
# Activate the correct environment
conda activate srj_RankEF

# Install additional dependencies for AceCoder (if needed)
pip install termcolor evalplus
```

### 2. Complete Pipeline (3 Steps)

#### Step 1: Generate Code Candidates
```bash
# Generate candidates for all models and datasets
python scripts/run_generation.py

# This generates ~100 candidates per problem for:
# - Models: CodeT5-770M, CodeGen-2B, CodeLlama-7B
# - Datasets: HumanEval (164 problems), MBPP (500 problems)
# - Output: results/candidates/non_finetuned/
```

#### Step 2: Rank Candidates
```bash
# Run specific ranking method
python scripts/run_ranking.py --method llm_judge --model codet5-770m --dataset humaneval
python scripts/run_ranking.py --method acecoder_rm --model codellama-7b --dataset mbpp

# Or run all combinations
python scripts/run_ranking.py --methods llm_judge acecoder_rm --models codet5-770m codellama-7b --datasets humaneval mbpp
```

#### Step 3: Evaluate Rankings
```bash
# Evaluate specific configuration
python scripts/run_evaluation.py --methods llm_judge acecoder_rm

# Output: evaluation_outputs/ with detailed results by dataset/model
```

### 3. One-Command Pipeline (Skip Generation)
```bash
# If candidates already exist, run ranking + evaluation
python scripts/run_full_pipeline.py --methods llm_judge acecoder_rm
```

## 🛠️ Detailed Usage

### Individual Steps (Advanced)
```bash
# Generate for specific model/dataset
python -c "
from data_preparation.candidate_generator import generate_all_candidates
generate_all_candidates(['codet5-770m'], ['humaneval'], 100)
"

# Rank specific configuration
python scripts/run_ranking.py --method acecoder_rm --model codet5-770m --dataset humaneval

# Evaluate specific methods
python scripts/run_evaluation.py --methods acecoder_rm
```

### Workflow Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   1. Generate   │───▶│   2. Rank       │───▶│  3. Evaluate    │
│   Candidates    │    │   Candidates    │    │   Rankings      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
│                      │                      │
│ • Load datasets      │ • LLM Judge         │ • Execute ranked
│ • Generate code      │ • AceCoder RM       │   candidates
│ • Save ~100/problem  │ • Score & sort      │ • Calculate pass@k
│                      │ • Save rankings     │ • Compare methods
│                      │                     │
▼ Output:              ▼ Output:             ▼ Output:
results/candidates/    results/ranked/       evaluation_outputs/
├── codet5-770m/      ├── llm_judge/        ├── detailed_results.csv
├── codellama-7b/     ├── acecoder_rm/      ├── comparison_table.csv
└── codegen-2b/       └── ...               └── ...
```

**Key Points:**
- **Step 1 is mandatory** for new models/datasets
- **Steps 2-3 can be run independently** once candidates exist
- **Each step produces persistent results** for incremental processing
- **Bug fix ensures** Step 3 now produces different results per ranking method

## 📊 Results

### Latest Evaluation Results

| Method | pass@1 | pass@2 | pass@5 | pass@10 | MRR | Success Rate |
|--------|--------|--------|--------|---------|-----|--------------|
| LLM-as-a-Judge | 0.0985 | 0.1395 | 0.2097 | 0.2697 | 0.154 | 0.4637 |
| AceCoder RM | 0.0867 | 0.1287 | 0.2003 | 0.2587 | 0.1437 | 0.4637 |

### Key Findings
- LLM-as-a-Judge slightly outperforms AceCoder RM across most metrics
- Both methods show comparable performance on success rate
- Performance varies by model and dataset combination

## 🏗️ Architecture

The framework follows a modular design:

1. **Data Layer**: Dataset loading and preprocessing (HumanEval, MBPP, APPS)
2. **Generation Layer**: Code candidate generation (external, supports multiple models)
3. **Ranking Layer**: Multiple ranking methods with unified interface
4. **Evaluation Layer**: Comprehensive metrics with proper pass@k calculation
5. **Results Layer**: Persistent storage and detailed comparison tables

## ⚡ Performance & Improvements

### Recent Bug Fixes & Improvements

#### 🔧 Critical Evaluation Bug Fixed (Sep 2025)

**Issue:** LLM Judge ranking results were not properly sorted by rank, causing both methods to execute candidates in identical order, leading to artificially identical evaluation results.

**Fix Applied:**
- Added proper rank-based sorting in `scripts/run_evaluation.py:77`
- Now methods produce genuinely different pass@k metrics
- Verification: problem_114 shows LLM Judge pass@1=0.0 vs AceCoder pass@1=1.0

#### 🚀 Enhanced AceCoder Integration

**Key Features:**
- ✅ **Fixed import issues** - Corrected class name from `Qwen2ForCausalRM` to `AceCodeRM`
- ✅ **Robust dependency handling** - Automatic fallback mechanisms
- ✅ **Multiple model loading strategies** - Official, local, and manual implementations
- ✅ **Memory efficient batching** - Configurable batch sizes for GPU optimization
- ✅ **Comprehensive error handling** - Graceful failure recovery

#### 🎯 Enhanced LLM Judge Ranking

**Key Features:**
- ✅ **Multiple prompt strategies** for better score extraction
- ✅ **Robust score parsing** with 5+ extraction methods
- ✅ **100% score extraction success rate** (vs ~40% before)
- ✅ **Intelligent score distribution normalization**
- ✅ **Multi-attempt generation** with different parameters
- ✅ **Comprehensive error handling** and recovery
- ✅ **Memory management** and GPU cleanup

## ⚠️ Important Notes

1. **AceCoder Dependencies**: Requires installation from `/home/fdse/srj/AceCoder/src`
2. **GPU Memory**: AceCoder ranking requires significant GPU memory (7B model)
3. **Evaluation Time**: Complete evaluation takes ~3-4 hours
4. **Bug Fix**: Recent evaluation bug fix ensures methodologically sound results

---

**Last Updated**: September 2025
**Major Changes**:
- Fixed critical evaluation ranking bug
- Enhanced AceCoder integration
- Added comprehensive result analysis
- Improved error handling and documentation