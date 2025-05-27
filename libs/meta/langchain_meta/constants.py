LLAMA_KNOWN_MODELS = {
    "Llama-3.3-70B-Instruct": {
        "model_name": "Llama-3.3-70B-Instruct",
        "context_window": 8192,
    },
    "Llama-3.3-8B-Instruct": {
        "model_name": "Llama-3.3-8B-Instruct",
        "context_window": 8192,
    },
    "Llama-4-Scout-17B-16E-Instruct-FP8": {  # Example, adjust as needed
        "model_name": "Llama-4-Scout-17B-16E-Instruct-FP8",
        "context_window": 8192,  # Placeholder
    },
    "Llama-4-Maverick-17B-128E-Instruct-FP8": {  # Example, adjust as needed
        "model_name": "Llama-4-Maverick-17B-128E-Instruct-FP8",
        "context_window": 8192,  # Placeholder
    },
}

LLAMA_DEFAULT_MODEL_NAME = "Llama-4-Maverick-17B-128E-Instruct-FP8"

# Also include VALID_MODELS if it's used as a constant set
VALID_MODELS = set(LLAMA_KNOWN_MODELS.keys()) 