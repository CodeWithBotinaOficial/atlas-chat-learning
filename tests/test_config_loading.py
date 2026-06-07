# tests/test_config_loading.py
import os
import tempfile
import yaml
import pytest
from atlas.config_loader import load_config
from atlas.brain import AtlasBrain


def test_default_config_loading():
    # Create a dummy config missing the new training fields to verify fallbacks work
    with tempfile.TemporaryDirectory() as tmpdir:
        dummy_config_path = os.path.join(tmpdir, "config.yaml")
        dummy_config = {
            'model': {
                'embed_dim': 16,
                'num_heads': 2,
                'ff_dim': 32,
                'num_layers': 1,
                'max_seq_len': 10,
                'dropout_rate': 0.0
            },
            'training': {
                'learning_rate': 0.001,
                'lr_decay_rate': 0.95,
                'lr_decay_steps': 100,
                'replay_buffer_size': 10,
                'replay_sample_rate': 0.3
            },
            'generation': {
                'temperature': 0.8,
                'repetition_penalty': 1.2,
                'top_k': 20,
                'top_p': 0.9,
                'beam_width': 0,
                'max_new_tokens': 20
            },
            'memory': {
                'max_history_length': 5
            }
        }
        
        with open(dummy_config_path, "w") as f:
            yaml.dump(dummy_config, f)
            
        config = load_config(dummy_config_path)
        
        # Verify fallbacks are populated by config_loader
        assert config['training']['epochs'] == 50
        assert config['training']['chunk_size'] == 1000
        assert config['training']['nan_check_interval'] == 10
        assert config['training']['auto_save_interval'] == 5
        assert config['training']['warmup_steps'] == 1000
        assert config['training']['beta1'] == 0.9
        assert config['training']['beta2'] == 0.999
        assert config['training']['epsilon'] == 1e-8


def test_custom_config_values_applied():
    with tempfile.TemporaryDirectory() as tmpdir:
        custom_config_path = os.path.join(tmpdir, "config.yaml")
        custom_config = {
            'model': {
                'embed_dim': 8,
                'num_heads': 1,
                'ff_dim': 16,
                'num_layers': 1,
                'max_seq_len': 10,
                'dropout_rate': 0.0
            },
            'training': {
                'learning_rate': 0.001,
                'lr_decay_rate': 0.95,
                'lr_decay_steps': 100,
                'replay_buffer_size': 10,
                'replay_sample_rate': 0.3,
                'epochs': 15,
                'chunk_size': 500,
                'nan_check_interval': 3,
                'auto_save_interval': 2,
                'warmup_steps': 100,
                'beta1': 0.8,
                'beta2': 0.98,
                'epsilon': 1e-7
            },
            'generation': {
                'temperature': 0.8,
                'repetition_penalty': 1.2,
                'top_k': 20,
                'top_p': 0.9,
                'beam_width': 0,
                'max_new_tokens': 20
            },
            'memory': {
                'max_history_length': 5
            }
        }
        
        with open(custom_config_path, "w") as f:
            yaml.dump(custom_config, f)
            
        config = load_config(custom_config_path)
        
        # Test loader loaded them
        assert config['training']['epochs'] == 15
        assert config['training']['chunk_size'] == 500
        assert config['training']['nan_check_interval'] == 3
        assert config['training']['auto_save_interval'] == 2
        assert config['training']['warmup_steps'] == 100
        assert config['training']['beta1'] == 0.8
        assert config['training']['beta2'] == 0.98
        assert config['training']['epsilon'] == 1e-7
        
        # Initialize AtlasBrain with this config
        model_path = os.path.join(tmpdir, "test_model.npz")
        vocab_path = os.path.join(tmpdir, "test_vocab.pkl")
        brain = AtlasBrain(model_path=model_path, vocab_path=vocab_path, config=config)
        
        # Verify values propagated to AtlasBrain
        assert brain.nan_check_interval == 3
        
        # Verify values propagated to Transformer
        assert brain.transformer.warmup_steps == 100
        assert brain.transformer.beta1 == 0.8
        assert brain.transformer.beta2 == 0.98
        assert brain.transformer.epsilon == 1e-7
