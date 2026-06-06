import yaml
import os
import sys

DEFAULT_CONFIG_PATH = "config.yaml"

def load_config(config_path=DEFAULT_CONFIG_PATH):
    if not os.path.exists(config_path):
        print("Configuration file not found. Please copy 'config.yaml.example' to 'config.yaml' and edit it as needed. Exiting.")
        sys.exit(1)
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config