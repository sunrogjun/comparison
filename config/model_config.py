"""
Model configuration for code generation comparison experiments
"""

# Non-finetuned models (matching RankEF Table 4&5)
NON_FINETUNED_MODELS = {
    "codet5-770m": "Salesforce/codet5p-770m-py",  # RankEF uses codet5p-770m-py
    "codegen-2b": "Salesforce/codegen-2B-multi",
    "codellama-7b": "codellama/CodeLlama-7b-Python-hf",
}

# Finetuned models (for future experiments)
FINETUNED_MODELS = {
    # Will be populated when models are finetuned
    "codet5-770m-finetuned": "./models/codet5_finetuned",
    "codegen-2b-finetuned": "./models/codegen_2b_finetuned",
    "codellama-7b-finetuned": "./models/codellama_7b_finetuned",
}

# Model specific configurations
MODEL_CONFIGS = {
    "codet5-770m": {
        "tokenizer": "Salesforce/codet5p-770m-py",  # Match RankEF tokenizer
        "max_length": 512,
        "device_map": "auto",
    },
    "codegen-2b": {
        "tokenizer": "Salesforce/codegen-2B-multi", 
        "max_length": 512,
        "device_map": "auto",
    },
    "codellama-7b": {
        "tokenizer": "codellama/CodeLlama-7b-Python-hf",
        "max_length": 512,
        "device_map": "auto",
    }
}

def get_model_path(model_name, finetuned=False):
    """Get model path for given model name"""
    if finetuned:
        return FINETUNED_MODELS.get(f"{model_name}-finetuned", None)
    else:
        return NON_FINETUNED_MODELS.get(model_name, None)

def get_model_config(model_name):
    """Get model configuration"""
    return MODEL_CONFIGS.get(model_name, {})