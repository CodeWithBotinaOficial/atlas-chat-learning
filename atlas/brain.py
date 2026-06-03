"""
brain.py

This module contains the core logic for the Atlas AI's learning and response mechanisms,
implementing an n-gram based incremental learning model.
"""

import collections
import random
import pickle
import re
import os

class AtlasBrain:
    """
    The core brain of the Atlas AI.
    Handles learning from user input and generating responses using n-grams.
    """

    def __init__(self, model_file: str = "atlas_memory.pkl"):
        """
        Initializes the AtlasBrain.
        Loads previously saved model (ngrams and vocabulary) if it exists,
        otherwise initializes an empty model.

        Args:
            model_file (str): The filename to save/load the model from.
        """
        self.model_file = model_file
        self.ngrams = collections.defaultdict(
            lambda: collections.defaultdict(collections.Counter)
        )
        self.vocabulary = set()
        self.load()
        print(f"AtlasBrain initialized. Vocabulary size: {len(self.vocabulary)}")

    def _tokenize(self, text: str) -> list[str]:
        """
        Tokenizes the input text into words, converting to lowercase
        and removing punctuation.

        Args:
            text (str): The input string to tokenize.

        Returns:
            list[str]: A list of cleaned, lowercase words.
        """
        text = text.lower()
        # Remove punctuation, but keep spaces for splitting
        text = re.sub(r"[^\w\s]", "", text)
        tokens = text.split()
        return tokens

    def learn(self, text: str):
        """
        Processes user input to learn and update Atlas's knowledge base
        by building unigrams, bigrams, and trigrams.

        Args:
            text (str): The input text from the user.
        """
        tokens = self._tokenize(text)
        if not tokens:
            return

        # Add tokens to vocabulary
        self.vocabulary.update(tokens)

        # Unigrams (for starting sentences)
        for token in tokens:
            self.ngrams[1][("",)][token] += 1 # Use empty tuple as prefix for unigrams

        # Bigrams
        for i in range(len(tokens) - 1):
            prefix = (tokens[i],)
            next_word = tokens[i+1]
            self.ngrams[2][prefix][next_word] += 1

        # Trigrams
        for i in range(len(tokens) - 2):
            prefix = (tokens[i], tokens[i+1])
            next_word = tokens[i+2]
            self.ngrams[3][prefix][next_word] += 1
        
        print(f"Atlas learned from: '{text}'")

    def respond(self, prompt: str = None, max_words: int = 15) -> str:
        """
        Generates a response based on the current knowledge and an optional prompt.
        Uses weighted random choice based on n-gram frequencies.

        Args:
            prompt (str, optional): The user's last input. If provided, the last
                                    (n-1) words are used as a seed. Defaults to None.
            max_words (int): The maximum number of words in the generated response.

        Returns:
            str: A generated response from Atlas.
        """
        response_tokens = []
        current_prefix = ()

        if prompt:
            prompt_tokens = self._tokenize(prompt)
            if len(prompt_tokens) >= 2:
                current_prefix = (prompt_tokens[-2], prompt_tokens[-1])
            elif len(prompt_tokens) == 1:
                current_prefix = (prompt_tokens[-1],)

        # Try to start with a relevant prefix or a random one
        if current_prefix and current_prefix in self.ngrams[3]:
            pass # Use trigram prefix
        elif current_prefix and (current_prefix[-1],) in self.ngrams[2]:
            current_prefix = (current_prefix[-1],) # Fallback to bigram prefix
        else:
            # Pick a random starting word (unigram) if no relevant prefix or no prompt
            if self.ngrams[1][("",)]:
                words, counts = zip(*self.ngrams[1][("",)].items())
                current_word = random.choices(words, weights=counts, k=1)[0]
                response_tokens.append(current_word)
                current_prefix = (current_word,)
            else:
                return "I don't have enough information to respond yet."

        while len(response_tokens) < max_words:
            next_word = None
            
            # Try trigrams first
            if len(current_prefix) == 2 and current_prefix in self.ngrams[3]:
                possible_next_words = self.ngrams[3][current_prefix]
                if possible_next_words:
                    words, counts = zip(*possible_next_words.items())
                    next_word = random.choices(words, weights=counts, k=1)[0]
            
            # Fallback to bigrams if trigram not found or prefix too short
            if next_word is None and len(current_prefix) >= 1 and (current_prefix[-1],) in self.ngrams[2]:
                possible_next_words = self.ngrams[2][(current_prefix[-1],)]
                if possible_next_words:
                    words, counts = zip(*possible_next_words.items())
                    next_word = random.choices(words, weights=counts, k=1)[0]
            
            # Fallback to unigrams if no bigram found
            if next_word is None and self.ngrams[1][("",)]:
                words, counts = zip(*self.ngrams[1][("",)].items())
                next_word = random.choices(words, weights=counts, k=1)[0]

            if next_word:
                response_tokens.append(next_word)
                # Update prefix for next iteration
                if len(current_prefix) == 2:
                    current_prefix = (current_prefix[1], next_word)
                elif len(current_prefix) == 1:
                    current_prefix = (current_prefix[0], next_word)
                else: # Started with unigram
                    current_prefix = (next_word,)
            else:
                # No continuation found
                break

        if not response_tokens:
            return "I am still learning..."
            
        return " ".join(response_tokens)

    def save(self):
        """
        Saves the current state of the AtlasBrain (ngrams and vocabulary)
        to the specified model file using pickle.
        """
        try:
            with open(self.model_file, 'wb') as f:
                pickle.dump({'ngrams': self.ngrams, 'vocabulary': self.vocabulary}, f)
            print(f"Atlas memory saved to {self.model_file}")
        except Exception as e:
            print(f"Error saving Atlas memory: {e}")

    def load(self):
        """
        Loads the AtlasBrain state (ngrams and vocabulary) from the model file.
        If the file does not exist, the model remains empty.
        """
        if os.path.exists(self.model_file):
            try:
                with open(self.model_file, 'rb') as f:
                    data = pickle.load(f)
                    self.ngrams = data.get('ngrams', self.ngrams)
                    self.vocabulary = data.get('vocabulary', self.vocabulary)
                print(f"Atlas memory loaded from {self.model_file}")
            except Exception as e:
                print(f"Error loading Atlas memory from {self.model_file}: {e}")
                # Reset to empty if loading fails
                self.ngrams = collections.defaultdict(
                    lambda: collections.defaultdict(collections.Counter)
                )
                self.vocabulary = set()
        else:
            print(f"No existing model found at {self.model_file}. Starting fresh.")
