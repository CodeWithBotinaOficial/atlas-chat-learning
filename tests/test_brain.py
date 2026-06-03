"""
test_brain.py

This module contains unit tests for the AtlasBrain class,
including tests for n-gram learning, response generation, and persistence.
"""

import unittest
import os
import tempfile
import shutil
from atlas.brain import AtlasBrain

class TestAtlasBrain(unittest.TestCase):
    """
    Test suite for the AtlasBrain class.
    """

    def setUp(self):
        """
        Set up a temporary directory and model file for testing.
        """
        self.test_dir = tempfile.mkdtemp()
        self.model_file = os.path.join(self.test_dir, "test_atlas_memory.pkl")
        # Suppress print statements from AtlasBrain during tests
        # import io
        # self._stdout = io.StringIO()
        # self._stderr = io.StringIO()
        # self.__stdout__ = sys.stdout
        # self.__stderr__ = sys.stderr
        # sys.stdout = self._stdout
        # sys.stderr = self._stderr

    def tearDown(self):
        """
        Clean up the temporary directory after tests.
        """
        shutil.rmtree(self.test_dir)
        # sys.stdout = self.__stdout__
        # sys.stderr = self.__stderr__

    def test_initialization_empty(self):
        """
        Tests that AtlasBrain initializes correctly with an empty model.
        """
        brain = AtlasBrain(model_file=self.model_file)
        self.assertIsInstance(brain, AtlasBrain)
        self.assertEqual(len(brain.ngrams), 0)
        self.assertEqual(len(brain.vocabulary), 0)

    def test_tokenize(self):
        """
        Tests the _tokenize helper method.
        """
        brain = AtlasBrain(model_file=self.model_file)
        self.assertEqual(brain._tokenize("Hello, world!"), ["hello", "world"])
        self.assertEqual(brain._tokenize("  Python is fun.  "), ["python", "is", "fun"])
        self.assertEqual(brain._tokenize(""), [])
        self.assertEqual(brain._tokenize("123 test"), ["123", "test"])

    def test_learn_unigrams(self):
        """
        Tests that learn() correctly updates unigram counts.
        """
        brain = AtlasBrain(model_file=self.model_file)
        brain.learn("Hello world")
        self.assertIn("hello", brain.vocabulary)
        self.assertIn("world", brain.vocabulary)
        self.assertEqual(brain.ngrams[1][("",)]["hello"], 1)
        self.assertEqual(brain.ngrams[1][("",)]["world"], 1)
        brain.learn("Hello again")
        self.assertEqual(brain.ngrams[1][("",)]["hello"], 2)
        self.assertEqual(brain.ngrams[1][("",)]["again"], 1)

    def test_learn_bigrams(self):
        """
        Tests that learn() correctly updates bigram counts.
        """
        brain = AtlasBrain(model_file=self.model_file)
        brain.learn("Hello world")
        self.assertEqual(brain.ngrams[2][("hello",)]["world"], 1)
        brain.learn("world peace")
        self.assertEqual(brain.ngrams[2][("world",)]["peace"], 1)
        self.assertEqual(brain.ngrams[2][("hello",)]["world"], 1) # Should remain 1

    def test_learn_trigrams(self):
        """
        Tests that learn() correctly updates trigram counts.
        """
        brain = AtlasBrain(model_file=self.model_file)
        brain.learn("The quick brown fox")
        self.assertEqual(brain.ngrams[3][("the", "quick")]["brown"], 1)
        self.assertEqual(brain.ngrams[3][("quick", "brown")]["fox"], 1)
        brain.learn("quick brown dog")
        self.assertEqual(brain.ngrams[3][("quick", "brown")]["dog"], 1)
        self.assertEqual(brain.ngrams[3][("quick", "brown")]["fox"], 1) # Should remain 1

    def test_respond_empty_brain(self):
        """
        Tests respond() when the brain has no learned data.
        """
        brain = AtlasBrain(model_file=self.model_file)
        response = brain.respond()
        self.assertIn(response, ["I don't have enough information to respond yet.", "I am still learning..."])

    def test_respond_basic(self):
        """
        Tests that respond() returns a string after learning.
        """
        brain = AtlasBrain(model_file=self.model_file)
        brain.learn("hello world this is a test")
        brain.learn("this is another test sentence")
        response = brain.respond()
        self.assertIsInstance(response, str)
        self.assertGreater(len(response.split()), 0)
        self.assertLessEqual(len(response.split()), 15)

    def test_respond_with_prompt(self):
        """
        Tests that respond() attempts to use the prompt as a seed.
        """
        brain = AtlasBrain(model_file=self.model_file)
        brain.learn("the quick brown fox jumps over the lazy dog")
        brain.learn("the lazy cat sleeps all day long")
        
        # Test with a prompt that matches a trigram prefix
        response = brain.respond(prompt="brown fox", max_words=5)
        self.assertTrue(response.startswith("brown fox") or response.startswith("fox")) # Could start with bigram or trigram
        self.assertIsInstance(response, str)

        # Test with a prompt that matches a bigram prefix
        response = brain.respond(prompt="quick brown", max_words=5)
        self.assertTrue(response.startswith("quick brown") or response.startswith("brown"))
        self.assertIsInstance(response, str)

    def test_save_load_persistence(self):
        """
        Tests that the brain's state can be saved and loaded correctly.
        """
        # 1. Create a brain, learn something, and save
        brain1 = AtlasBrain(model_file=self.model_file)
        brain1.learn("apple banana cherry")
        brain1.learn("banana cherry date")
        brain1.save()

        # Check if file exists
        self.assertTrue(os.path.exists(self.model_file))

        # 2. Create a new brain instance and load from the same file
        brain2 = AtlasBrain(model_file=self.model_file)

        # 3. Verify that the loaded brain has the same data
        self.assertEqual(brain1.vocabulary, brain2.vocabulary)
        self.assertEqual(brain1.ngrams[1][("",)]["apple"], brain2.ngrams[1][("",)]["apple"])
        self.assertEqual(brain1.ngrams[2][("apple",)]["banana"], brain2.ngrams[2][("apple",)]["banana"])
        self.assertEqual(brain1.ngrams[3][("apple", "banana")]["cherry"], brain2.ngrams[3][("apple", "banana")]["cherry"])
        self.assertEqual(brain1.ngrams[3][("banana", "cherry")]["date"], brain2.ngrams[3][("banana", "cherry")]["date"])

    def test_load_non_existent_file(self):
        """
        Tests loading from a file that does not exist.
        Should initialize an empty brain without errors.
        """
        # Ensure the model file does not exist
        if os.path.exists(self.model_file):
            os.remove(self.model_file)

        brain = AtlasBrain(model_file=self.model_file)
        self.assertEqual(len(brain.ngrams), 0)
        self.assertEqual(len(brain.vocabulary), 0)

    def test_load_corrupted_file(self):
        """
        Tests loading from a corrupted file.
        Should handle the error and initialize an empty brain.
        """
        with open(self.model_file, 'w') as f:
            f.write("this is not a pickle file")
        
        brain = AtlasBrain(model_file=self.model_file)
        self.assertEqual(len(brain.ngrams), 0)
        self.assertEqual(len(brain.vocabulary), 0)


if __name__ == '__main__':
    unittest.main()
