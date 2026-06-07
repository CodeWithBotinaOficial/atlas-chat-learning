# tests/test_analytics.py
import os
import tempfile
import json
from datetime import datetime, timedelta
import numpy as np
import pytest
from unittest.mock import MagicMock

from atlas.analytics import MetricsRecorder, generate_pdf_report


def test_metrics_recorder_initialization():
    config = {
        'reporting': {
            'enabled': True,
            'output_dir': 'test_reports',
            'generate_graphs': True,
            'max_history_points': 500
        }
    }
    history = []
    start_time = datetime.now()
    recorder = MetricsRecorder(config, history, start_time)
    
    assert recorder.enabled is True
    assert recorder.output_dir == 'test_reports'
    assert recorder.generate_graphs_enabled is True
    assert recorder.max_history_points == 500
    assert recorder.metrics_history is history
    assert recorder.start_time == start_time


def test_metrics_recorder_add_point_and_capping():
    config = {
        'reporting': {
            'enabled': True,
            'max_history_points': 3
        }
    }
    history = []
    start_time = datetime.now()
    recorder = MetricsRecorder(config, history, start_time)
    
    # Add 5 points (should cap at 3)
    for i in range(5):
        recorder.add_point(loss=0.5 - i * 0.1, learning_rate=0.001, vocab_size=10 + i)
        
    assert len(history) == 3
    # Steps should be tracked sequentially
    assert history[0]['step'] == 3
    assert history[1]['step'] == 4
    assert history[2]['step'] == 5
    
    assert history[2]['loss'] == pytest.approx(0.1)
    assert history[2]['vocab_size'] == 14
    assert history[2]['learning_rate'] == 0.001
    assert 'timestamp' in history[2]


def test_metrics_recorder_disabled():
    config = {
        'reporting': {
            'enabled': False
        }
    }
    history = []
    recorder = MetricsRecorder(config, history, datetime.now())
    
    recorder.add_point(loss=0.5, learning_rate=0.001, vocab_size=10)
    assert len(history) == 0


def test_generate_report_creates_files_and_directories():
    with tempfile.TemporaryDirectory() as tmpdir:
        config = {
            'reporting': {
                'enabled': True,
                'output_dir': tmpdir,
                'generate_graphs': True,
                'max_history_points': 10
            },
            'model': {
                'embed_dim': 16,
                'num_heads': 2,
                'ff_dim': 32,
                'num_layers': 1,
                'max_seq_len': 10,
                'dropout_rate': 0.0
            }
        }
        
        history = [
            {'step': 1, 'loss': 0.5, 'learning_rate': 0.001, 'vocab_size': 8, 'timestamp': (datetime.now() - timedelta(minutes=5)).isoformat()},
            {'step': 2, 'loss': 0.4, 'learning_rate': 0.001, 'vocab_size': 9, 'timestamp': datetime.now().isoformat()},
        ]
        
        # Mock AtlasBrain and Transformer parameters
        brain = MagicMock()
        brain.vocab_size = 10
        brain.idx_to_word = {i: f"word_{i}" for i in range(10)}
        brain.SPECIAL_TOKENS = ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]
        
        transformer = MagicMock()
        transformer.vocab_size = 10
        transformer.token_embedding = np.random.randn(10, 16)
        transformer.output_layer = np.random.randn(16, 10)
        transformer.output_bias = np.zeros(10)
        transformer.params = {
            'token_embedding': transformer.token_embedding,
            'output_layer': transformer.output_layer,
            'output_bias': transformer.output_bias
        }
        brain.transformer = transformer
        
        start_time = datetime.now() - timedelta(minutes=5)
        recorder = MetricsRecorder(config, history, start_time)
        
        recorder.generate_report(brain)
        
        # Check that files were created
        created_dirs = os.listdir(tmpdir)
        assert len(created_dirs) == 1
        
        report_dir = os.path.join(tmpdir, created_dirs[0])
        assert os.path.isdir(report_dir)
        
        # Check raw JSON file exists and contains history
        metrics_json_path = os.path.join(report_dir, "metrics.json")
        assert os.path.exists(metrics_json_path)
        with open(metrics_json_path, 'r') as f:
            saved_history = json.load(f)
        assert len(saved_history) == 2
        assert saved_history[0]['loss'] == 0.5
        
        # Check report file exists (PDF or fallback Text)
        pdf_path = os.path.join(report_dir, "analytics.pdf")
        txt_path = os.path.join(report_dir, "analytics_summary.txt")
        assert os.path.exists(pdf_path) or os.path.exists(txt_path)


def test_generate_pdf_report_direct():
    with tempfile.TemporaryDirectory() as tmpdir:
        config = {
            'model': {
                'embed_dim': 8,
                'num_heads': 1,
                'ff_dim': 16,
                'num_layers': 1,
                'max_seq_len': 5,
                'dropout_rate': 0.0
            }
        }
        
        history = [
            {'step': 1, 'loss': 0.8, 'learning_rate': 0.001, 'vocab_size': 5, 'timestamp': datetime.now().isoformat()},
            {'step': 2, 'loss': 0.6, 'learning_rate': 0.001, 'vocab_size': 5, 'timestamp': datetime.now().isoformat()},
        ]
        
        brain = MagicMock()
        brain.vocab_size = 5
        brain.idx_to_word = {0: "<PAD>", 1: "<UNK>", 2: "<BOS>", 3: "<EOS>", 4: "hello"}
        brain.SPECIAL_TOKENS = ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]
        
        transformer = MagicMock()
        transformer.vocab_size = 5
        transformer.token_embedding = np.random.randn(5, 8)
        transformer.output_layer = np.random.randn(8, 5)
        transformer.output_bias = np.zeros(5)
        transformer.params = {
            'token_embedding': transformer.token_embedding,
            'output_layer': transformer.output_layer,
            'output_bias': transformer.output_bias
        }
        brain.transformer = transformer
        
        output_pdf = os.path.join(tmpdir, "test_report.pdf")
        
        # Directly generate the PDF report (will compile matplotlib pages)
        generate_pdf_report(
            metrics_history=history,
            config=config,
            start_time=datetime.now() - timedelta(minutes=2),
            end_time=datetime.now(),
            output_path=output_pdf,
            brain=brain
        )
        
        assert os.path.exists(output_pdf)
        assert os.path.getsize(output_pdf) > 0
