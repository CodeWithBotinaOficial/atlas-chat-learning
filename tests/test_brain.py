"""
test_brain.py

This module contains unit tests for the AtlasBrain class,
including tests for n-gram learning, response generation, and persistence.
"""

import pytest
import os
import tempfile
import shutil
from atlas.brain import AtlasBrain

@pytest.fixture
def brain_instance():
    """
    Fixture to create a new AtlasBrain instance with a temporary model file
    for each test.
    """
    test_dir = tempfile.mkdtemp()
    model_file = os.path.join(test_dir, "test_atlas_memory.pkl")
    brain = AtlasBrain(model_file=model_file)
    yield brain
    shutil.rmtree(test_dir)

def test_initialization_empty(brain_instance):
    """
    Tests that AtlasBrain initializes correctly with an empty model.
    """
    assert isinstance(brain_instance, AtlasBrain)
    assert len(brain_instance.unigrams) == 0
    assert len(brain_instance.bigrams) == 0
    assert len(brain_instance.trigrams) == 0
    assert len(brain_instance.vocabulary) == 0

def test_clean_token(brain_instance):
    """
    Tests the _clean_token helper method.
    """
    assert brain_instance._clean_token("Hello,") == "hello"
    assert brain_instance._clean_token(" World!") == "world"
    assert brain_instance._clean_token("PyThoN") == "python"
    assert brain_instance._clean_token("123-test") == "123test"
    assert brain_instance._clean_token("!@#$") == ""

def test_tokenize(brain_instance):
    """
    Tests the _tokenize helper method.
    """
    assert brain_instance._tokenize("Hello, world!") == ["hello", "world"]
    assert brain_instance._tokenize("  Python is fun.  ") == ["python", "is", "fun"]
    assert brain_instance._tokenize("") == []
    assert brain_instance._tokenize("123 test") == ["123", "test"]
    assert brain_instance._tokenize("  !@#$  ") == [] # Should handle only punctuation

def test_learn_unigrams(brain_instance):
    """
    Tests that learn() correctly updates unigram counts.
    """
    brain_instance.learn("Hello world")
    assert "hello" in brain_instance.vocabulary
    assert "world" in brain_instance.vocabulary
    assert brain_instance.unigrams["hello"] == 1
    assert brain_instance.unigrams["world"] == 1
    brain_instance.learn("Hello again")
    assert brain_instance.unigrams["hello"] == 2
    assert brain_instance.unigrams["again"] == 1

def test_learn_bigrams(brain_instance):
    """
    Tests that learn() correctly updates bigram counts.
    """
    brain_instance.learn("Hello world")
    assert brain_instance.bigrams[("hello",)]["world"] == 1
    brain_instance.learn("world peace")
    assert brain_instance.bigrams[("world",)]["peace"] == 1
    assert brain_instance.bigrams[("hello",)]["world"] == 1 # Should remain 1

def test_learn_trigrams(brain_instance):
    """
    Tests that learn() correctly updates trigram counts.
    """
    brain_instance.learn("The quick brown fox")
    assert brain_instance.trigrams[("the", "quick")]["brown"] == 1
    assert brain_instance.trigrams[("quick", "brown")]["fox"] == 1
    brain_instance.learn("quick brown dog")
    assert brain_instance.trigrams[("quick", "brown")]["dog"] == 1
    assert brain_instance.trigrams[("quick", "brown")]["fox"] == 1 # Should remain 1

def test_respond_empty_brain(brain_instance):
    """
    Tests respond() when the brain has no learned data.
    """
    response = brain_instance.respond()
    assert response in ["I don't have enough information to respond yet.", "I am still learning..."]

def test_respond_basic(brain_instance):
    """
    Tests that respond() returns a string after learning.
    """
    brain_instance.learn("hello world this is a test")
    brain_instance.learn("this is another test sentence")
    response = brain_instance.respond()
    assert isinstance(response, str)
    assert len(response.split()) > 0
    assert len(response.split()) <= 15

def test_respond_with_prompt(brain_instance):
    """
    Tests that respond() attempts to use the prompt as a seed.
    """
    brain_instance.learn("the quick brown fox jumps over the lazy dog")
    brain_instance.learn("the lazy cat sleeps all day long")
    
    # Test with a prompt that matches a trigram prefix
    response = brain_instance.respond(prompt="brown fox", max_words=5)
    assert response.startswith("brown fox") or response.startswith("fox") # Could start with bigram or trigram
    assert isinstance(response, str)

    # Test with a prompt that matches a bigram prefix
    response = brain_instance.respond(prompt="quick brown", max_words=5)
    assert response.startswith("quick brown") or response.startswith("brown")
    assert isinstance(response, str)

def test_save_load_persistence(brain_instance):
    """
    Tests that the brain's state can be saved and loaded correctly.
    """
    # 1. Create a brain, learn something, and save
    brain_instance.learn("apple banana cherry")
    brain_instance.learn("banana cherry date")
    brain_instance.save()

    # Check if file exists
    assert os.path.exists(brain_instance.model_file)

    # 2. Create a new brain instance and load from the same file
    brain2 = AtlasBrain(model_file=brain_instance.model_file)

    # 3. Verify that the loaded brain has the same data
    assert brain_instance.vocabulary == brain2.vocabulary
    assert brain_instance.unigrams == brain2.unigrams
    assert brain_instance.bigrams == brain2.bigrams
    assert brain_instance.trigrams == brain2.trigrams

def test_load_non_existent_file(brain_instance):
    """
    Tests loading from a file that does not exist.
    Should initialize an empty brain without errors.
    """
    # Ensure the model file does not exist
    if os.path.exists(brain_instance.model_file):
        os.remove(brain_instance.model_file)

    brain = AtlasBrain(model_file=brain_instance.model_file)
    assert len(brain.unigrams) == 0
    assert len(brain.bigrams) == 0
    assert len(brain.trigrams) == 0
    assert len(brain.vocabulary) == 0

def test_load_empty_file(brain_instance):
    """
    Tests loading from an empty file.
    Should handle the error and initialize an empty brain.
    """
    with open(brain_instance.model_file, 'w') as f:
        pass # Create an empty file
    
    brain = AtlasBrain(model_file=brain_instance.model_file)
    assert len(brain.unigrams) == 0
    assert len(brain.bigrams) == 0
    assert len(brain.trigrams) == 0
    assert len(brain.vocabulary) == 0

def test_load_corrupted_file(brain_instance):
    """
    Tests loading from a corrupted file.
    Should handle the error and initialize an empty brain.
    """
    with open(brain_instance.model_file, 'w') as f:
        f.write("this is not a pickle file")
    
    brain = AtlasBrain(model_file=brain_instance.model_file)
    assert len(brain.unigrams) == 0
    assert len(brain.bigrams) == 0
    assert len(brain.trigrams) == 0
    assert len(brain.vocabulary) == 0

def test_respond_diversity(brain_instance):
    """
    Tests that the response generation avoids immediate repetition.
    This is a probabilistic test, so it might fail rarely.
    """
    brain_instance.learn("hello hello hello world")
    brain_instance.learn("world world world peace")
    
    # If it always repeats, this test will likely fail
    response = brain_instance.respond(prompt="hello", max_words=3)
    tokens = response.split()
    assert len(tokens) > 1 # Ensure it generated more than one word
    if len(tokens) >= 2:
        assert tokens[0] != tokens[1] # Should avoid "hello hello" if possible
    
    response = brain_instance.respond(prompt="world", max_words=3)
    tokens = response.split()
    assert len(tokens) > 1
    if len(tokens) >= 2:
        assert tokens[0] != tokens[1]
