"""
conftest.py

This file is used by pytest to provide fixtures and configuration for tests.
It ensures that the project root is added to sys.path so that modules
like 'atlas.brain' can be imported correctly during testing.
"""

import sys
import os
import pytest

# Add the project root to sys.path
# This allows pytest to find the 'atlas' package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

@pytest.fixture(scope='session', autouse=True)
def setup_pythonpath():
    """
    Fixture to ensure the project root is in PYTHONPATH for the entire test session.
    """
    print(f"\nAdding {os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))} to sys.path")
    # The sys.path.insert(0, ...) above already handles this for the current process.
    # This fixture primarily serves as documentation and to confirm the path is set.
    pass
