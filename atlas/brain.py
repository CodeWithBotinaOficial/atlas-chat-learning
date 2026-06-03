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
import sys

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
        self.unigrams = collections.Counter() # Stores counts of individual words
        self.bigrams = collections.defaultdict(collections.Counter) # Stores { (word1,): Counter(word2, word3, ...) }
        self.trigrams = collections.defaultdict(collections.Counter) # Stores { (word1, word2): Counter(word3, word4, ...) }
        self.vocabulary = set()
        self.load()
        print(f"AtlasBrain initialized. Vocabulary size: {len(self.vocabulary)}")

    def _clean_token(self, word: str) -> str:
        """
        Cleans a single word by lowercasing and removing non-alphanumeric characters.

        Args:
            word (str): The input word to clean.

        Returns:
            str: The cleaned, lowercase word.
        """
        return re.sub(r"[^\w]", "", word).lower()

    def _tokenize(self, text: str) -> list[str]:
        """
        Tokenizes the input text into words, cleaning each token.

        Args:
            text (str): The input string to tokenize.

        Returns:
            list[str]: A list of cleaned, lowercase words, with empty tokens removed.
        """
        tokens = [self._clean_token(word) for word in text.split()]
        return [token for token in tokens if token] # Remove empty strings

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

        self.vocabulary.update(tokens)

        # Unigrams
        for token in tokens:
            self.unigrams[token] += 1

        # Bigrams
        for i in range(len(tokens) - 1):
            prefix = (tokens[i],)
            next_word = tokens[i+1]
            self.bigrams[prefix][next_word] += 1

        # Trigrams
        for i in range(len(tokens) - 2):
            prefix = (tokens[i], tokens[i+1])
            next_word = tokens[i+2]
            self.trigrams[prefix][next_word] += 1
        
        # print(f"Atlas learned from: '{text}'") # Commented out to reduce console noise

    def respond(self, prompt: str = None, max_words: int = 15) -> str:
        """
        Generates a response based on the current knowledge and an optional prompt.
        Uses weighted random choice based on n-gram frequencies, with a simple
        diversity mechanism to avoid immediate repetitions.

        Args:
            prompt (str, optional): The user's last input. If provided, the last
                                    words are used as a seed. Defaults to None.
            max_words (int): The maximum number of words in the generated response.

        Returns:
            str: A generated response from Atlas.
        """
        response_tokens = []
        current_prefix = () # For bigram/trigram context

        # 1. Determine starting point
        start_candidates = []
        if prompt:
            prompt_tokens = self._tokenize(prompt)
            if len(prompt_tokens) >= 2:
                potential_prefix = (prompt_tokens[-2], prompt_tokens[-1])
                if potential_prefix in self.trigrams:
                    start_candidates.append(potential_prefix)
            if len(prompt_tokens) >= 1:
                potential_prefix = (prompt_tokens[-1],)
                if potential_prefix in self.bigrams:
                    start_candidates.append(potential_prefix)
            
            # If prompt tokens are not in vocabulary, or don't form known n-grams, fall back
            if not start_candidates and prompt_tokens and any(t in self.vocabulary for t in prompt_tokens):
                if prompt_tokens[-1] in self.vocabulary:
                    start_candidates.append((prompt_tokens[-1],))

        # If no specific start from prompt, or prompt didn't yield a good start, pick a random unigram
        if not start_candidates and self.unigrams:
            start_word = random.choices(list(self.unigrams.keys()), weights=list(self.unigrams.values()), k=1)[0]
            response_tokens.append(start_word)
            current_prefix = (start_word,)
        elif start_candidates:
            # Pick the longest valid prefix from prompt
            chosen_prefix = max(start_candidates, key=len)
            response_tokens.extend(list(chosen_prefix))
            current_prefix = chosen_prefix
        else:
            return "I don't have enough information to respond yet."

        # 2. Generate subsequent words
        while len(response_tokens) < max_words:
            next_word = None
            possible_next_words_counts = collections.Counter()

            # Try trigrams
            if len(current_prefix) == 2 and current_prefix in self.trigrams:
                possible_next_words_counts.update(self.trigrams[current_prefix])
            
            # Fallback to bigrams
            if not possible_next_words_counts and len(current_prefix) >= 1 and (current_prefix[-1],) in self.bigrams:
                possible_next_words_counts.update(self.bigrams[(current_prefix[-1],)])
            
            if possible_next_words_counts:
                candidates = list(possible_next_words_counts.items())
                
                # Simple diversity: try to avoid immediate repetition (e.g., "word word")
                if len(response_tokens) > 0:
                    non_repeating_candidates = [
                        (word, count) for word, count in candidates
                        if word != response_tokens[-1]
                    ]
                    if non_repeating_candidates:
                        candidates = non_repeating_candidates
                
                words, counts = zip(*candidates)
                next_word = random.choices(words, weights=counts, k=1)[0]
            
            if next_word:
                response_tokens.append(next_word)
                # Update prefix for next iteration
                if len(current_prefix) == 2:
                    current_prefix = (current_prefix[1], next_word)
                elif len(current_prefix) == 1:
                    current_prefix = (current_prefix[0], next_word)
                else: # This case should ideally not happen if response_tokens is populated
                    current_prefix = (next_word,)
            else:
                # No continuation found
                break

        if not response_tokens:
            return "I am still learning..."
            
        return " ".join(response_tokens)

    def save(self):
        """
        Saves the current state of the AtlasBrain (unigrams, bigrams, trigrams, and vocabulary)
        to the specified model file using pickle.
        """
        try:
            with open(self.model_file, 'wb') as f:
                pickle.dump({
                    'unigrams': self.unigrams,
                    'bigrams': self.bigrams,
                    'trigrams': self.trigrams,
                    'vocabulary': self.vocabulary
                }, f)
            print(f"Atlas memory saved to {self.model_file}")
        except Exception as e:
            print(f"Error saving Atlas memory: {e}")

    def load(self):
        """
        Loads the AtlasBrain state (unigrams, bigrams, trigrams, and vocabulary) from the model file.
        If the file does not exist, is empty, or corrupted, the model remains empty.
        """
        if os.path.exists(self.model_file):
            try:
                with open(self.model_file, 'rb') as f:
                    data = pickle.load(f)
                    self.unigrams = data.get('unigrams', collections.Counter())
                    self.bigrams = data.get('bigrams', collections.defaultdict(collections.Counter))
                    self.trigrams = data.get('trigrams', collections.defaultdict(collections.Counter))
                    self.vocabulary = data.get('vocabulary', set())
                print(f"Atlas memory loaded from {self.model_file}")
            except (EOFError, pickle.UnpicklingError) as e:
                print(f"Warning: Could not load Atlas memory from {self.model_file} (possibly empty or corrupted): {e}. Starting fresh.")
                # Reset to empty if loading fails
                self.unigrams = collections.Counter()
                self.bigrams = collections.defaultdict(collections.Counter)
                self.trigrams = collections.defaultdict(collections.Counter)
                self.vocabulary = set()
            except Exception as e:
                print(f"Error loading Atlas memory from {self.model_file}: {e}. Starting fresh.")
                self.unigrams = collections.Counter()
                self.bigrams = collections.defaultdict(collections.Counter)
                self.trigrams = collections.defaultdict(collections.Counter)
                self.vocabulary = set()
        else:
            print(f"No existing model found at {self.model_file}. Starting fresh.")
