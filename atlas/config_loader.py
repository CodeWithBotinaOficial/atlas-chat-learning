import yaml
import os

DEFAULT_CONFIG_PATH = "config.yaml"

DEFAULT_CONFIG_CONTENT = """
model:
  embed_dim: 32
  num_heads: 2
  ff_dim: 64
  num_layers: 2
  max_seq_len: 50
  dropout_rate: 0.1

training:
  learning_rate: 0.001
  lr_decay_rate: 0.95
  lr_decay_steps: 100
  replay_buffer_size: 10
  replay_sample_rate: 0.3

generation:
  temperature: 0.8
  repetition_penalty: 1.2
  top_k: 20
  top_p: 0.9
  beam_width: 0
  max_new_tokens: 20

memory:
  max_history_length: 5
"""

def load_config(config_path=DEFAULT_CONFIG_PATH):
    if not os.path.exists(config_path):
        print(f"Config file not found at {config_path}. Generating default config.yaml.")
        with open(config_path, "w") as f:
            f.write(DEFAULT_CONFIG_CONTENT.strip())
        print("Generated default config.yaml. Please edit and restart.")
        exit()
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

