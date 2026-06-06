import yaml
import os
import sys

DEFAULT_CONFIG_PATH = "config.yaml"

TRAINING_DEFAULTS = {
    'beta1': 0.9,
    'beta2': 0.999,
    'epsilon': 1e-8,
    'warmup_steps': 1000,
}

def load_config(config_path=DEFAULT_CONFIG_PATH):
    if not os.path.exists(config_path):
        print("Configuration file not found. Please copy 'config.yaml.example' to 'config.yaml' and edit it as needed. Exiting.")
        sys.exit(1)
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    if 'training' in config:
        for key, default_value in TRAINING_DEFAULTS.items():
            config['training'].setdefault(key, default_value)

    return config