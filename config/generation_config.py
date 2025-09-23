"""
Generation configuration for candidate code generation
"""

# RankEF paper generation parameters
# RankEF-matched generation parameters for better quality
GENERATION_PARAMS = {
    "num_candidates": 100,        # Total number of candidates per problem (num_seqs in RankEF)
    "num_seqs_per_iter": 10,     # Number of sequences per generation call 
    "inference_batch_size": 64,     # RankEF uses smaller batches for stability
    "temperature": 0.6,          # RankEF uses 0.6 (more focused than 0.8)
    "top_p": 0.95,              # Top-p sampling  
    "max_length": 512,          # Maximum total sequence length (RankEF style)
    "source_len": 600,          # Input sequence max length (RankEF: 600)
    "do_sample": True,          # Enable sampling
    "pad_token_id": None,       # Will be set based on tokenizer
    "eos_token_id": None,       # Will be set based on tokenizer
}

# Dataset specific prompting strategies
PROMPT_CONFIGS = {
    "mbpp": {
        "prompt_style": "few_shot",
        "num_examples": 1,
        "format": "natural_language",
    },
    "humaneval": {
        "prompt_style": "zero_shot", 
        "num_examples": 0,
        "format": "function_completion",
    },
    "apps": {
        "prompt_style": "zero_shot",
        "num_examples": 0, 
        "format": "natural_language",
    }
}

# Generation timeout and resource limits
RESOURCE_LIMITS = {
    "generation_timeout": 300,   # 5 minutes per problem
    "max_memory_gb": 24,        # Maximum GPU memory usage
    "batch_timeout": 600,       # 10 minutes per batch
}

def get_generation_config(dataset_name=None):
    """Get generation configuration, optionally dataset-specific"""
    config = GENERATION_PARAMS.copy()
    
    # RankEF-specific adjustments for CodeT5
    if dataset_name:
        config["dataset_name"] = dataset_name
    
    if dataset_name and dataset_name in PROMPT_CONFIGS:
        config.update(PROMPT_CONFIGS[dataset_name])
    
    return config

# RankEF-style tokenizer configuration
TOKENIZER_CONFIG = {
    "codet5": {
        "tokenizer_name": "Salesforce/codet5-large",  # RankEF uses codet5-large tokenizer
        "tokenizer_class": "RobertaTokenizer",
        "max_length": 600,  # source_len from RankEF
        "verbose": False,
    }
}

def get_resource_limits():
    """Get resource limit configuration"""
    return RESOURCE_LIMITS.copy()