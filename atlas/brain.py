# atlas/brain.py
import numpy as np
import pickle
import re
from collections import defaultdict
import os

from atlas.transformer import Transformer, cross_entropy_loss, softmax


class AtlasBrain:
    """
    Manages the conversational AI's brain, including vocabulary,
    Transformer model, learning, and response generation.
    """
    SPECIAL_TOKENS = ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]
    PAD_TOKEN_ID = 0
    UNK_TOKEN_ID = 1
    BOS_TOKEN_ID = 2
    EOS_TOKEN_ID = 3

    def __init__(self, model_path="atlas_model.npz", vocab_path="atlas_vocab.pkl"):
        self.model_path = model_path
        self.vocab_path = vocab_path

        self.word_to_idx = {}
        self.idx_to_word = {}
        self.vocab_size = 0

        # Initialize vocabulary with special tokens
        for token in self.SPECIAL_TOKENS:
            self._add_word_to_vocab(token)

        # Transformer hyperparameters
        self.embed_dim = 16
        self.num_heads = 2
        self.ff_dim = 32
        self.num_layers = 2
        self.max_seq_len = 50
        self.learning_rate = 0.01
        self.repetition_penalty = 1.2  # A small penalty to discourage immediate repetition

        # Initialize Transformer with current vocab_size
        self.transformer = Transformer(
            vocab_size=self.vocab_size,
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            ff_dim=self.ff_dim,
            num_layers=self.num_layers,
            max_seq_len=self.max_seq_len
        )

        self.load()  # Attempt to load existing model and vocab

        # Ensure transformer's vocab_size is up-to-date after loading or initial setup
        # This call is crucial if vocab was loaded and expanded beyond initial SPECIAL_TOKENS count
        self.transformer.update_vocab_size(self.vocab_size)

    def _add_word_to_vocab(self, word):
        """
        Adds a word to the vocabulary if it doesn't exist.
        Returns True if a new word was added, False otherwise.
        """
        if word not in self.word_to_idx:
            self.word_to_idx[word] = self.vocab_size
            self.idx_to_word[self.vocab_size] = word
            self.vocab_size += 1
            return True
        return False

    def _tokenize(self, text):
        """
        Tokenizes input text into a list of words.
        Converts to lowercase and removes punctuation.
        """
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
        tokens = text.split()
        return tokens

    def _detokenize(self, token_ids):
        """
        Converts a list of token IDs back to a human-readable string.
        """
        words = [self.idx_to_word.get(idx, "<UNK>") for idx in token_ids]
        return " ".join(words)

    def _words_to_ids(self, words):
        """
        Converts a list of words to a list of token IDs.
        Dynamically adds new words to the vocabulary and updates the transformer.
        """
        ids = []
        for word in words:
            if self._add_word_to_vocab(word):  # Add word and check if new
                self.transformer.update_vocab_size(self.vocab_size)  # Update transformer's vocab
            ids.append(self.word_to_idx.get(word, self.UNK_TOKEN_ID))
        return ids

    def _ids_to_words(self, ids):
        """
        Converts a list of token IDs to a list of words.
        """
        return [self.idx_to_word.get(idx, "<UNK>") for idx in ids]

    def learn(self, text):
        """
        Processes input text, updates vocabulary, and performs one training step
        on the Transformer model.
        """
        tokens = self._tokenize(text)
        if not tokens:
            return None

        # Add BOS and EOS tokens for training sequence
        # The model learns to predict the next token given the previous ones.
        # So, input sequence will be <BOS> token1 token2 ... tokenN
        # And target sequence will be token1 token2 ... tokenN <EOS>
        full_sequence_ids = [self.BOS_TOKEN_ID] + self._words_to_ids(tokens) + [self.EOS_TOKEN_ID]

        # Truncate sequence if too long, keeping the most recent tokens
        if len(full_sequence_ids) > self.max_seq_len:
            full_sequence_ids = full_sequence_ids[len(full_sequence_ids) - self.max_seq_len:]

        # Prepare input and target for transformer
        # Input: all tokens except the last one
        # Target: all tokens except the first one
        input_ids = full_sequence_ids[:-1]
        target_ids = full_sequence_ids[1:]

        # Pad sequences to max_seq_len if needed
        # This is important for consistent tensor shapes
        input_ids_padded = np.full(self.max_seq_len, self.PAD_TOKEN_ID, dtype=int)
        target_ids_padded = np.full(self.max_seq_len, self.PAD_TOKEN_ID, dtype=int)

        input_ids_padded[:len(input_ids)] = input_ids
        target_ids_padded[:len(target_ids)] = target_ids

        # Transformer expects batch_size, seq_len
        input_batch = np.array([input_ids_padded])
        target_batch = np.array([target_ids_padded])

        loss = self.transformer.train_step(input_batch, target_batch, self.learning_rate)
        # print(f"Learning loss: {loss:.4f}")
        return loss

    def respond(self, prompt=None):
        """
        Generates a response based on an optional prompt or starts from BOS.
        """
        if self.vocab_size <= len(self.SPECIAL_TOKENS):  # Only special tokens in vocab
            return "I'm learning to speak... please teach me more!"

        if prompt:
            prompt_tokens = self._tokenize(prompt)
            prompt_ids = self._words_to_ids(prompt_tokens)
            # Start generation with BOS and prompt
            initial_sequence = [self.BOS_TOKEN_ID] + prompt_ids
        else:
            # Start generation with BOS token
            initial_sequence = [self.BOS_TOKEN_ID]

        # Truncate initial sequence to fit within max_seq_len for generation
        initial_sequence = initial_sequence[-self.max_seq_len:]

        # Transformer expects batch_size, seq_len
        prompt_tensor = np.array([initial_sequence])

        generated_ids = self.transformer.generate(
            prompt_tensor,
            max_new_tokens=15,
            temperature=0.8,
            repetition_penalty=self.repetition_penalty,
            pad_token_id=self.PAD_TOKEN_ID,
            eos_token_id=self.EOS_TOKEN_ID
        )

        # Decode generated tokens, stopping at EOS
        response_tokens = []
        # Skip the initial prompt tokens if they were part of the generated_ids
        # And skip BOS token if it's the first generated token
        start_idx = len(initial_sequence) if prompt else 1  # If no prompt, skip BOS

        for token_id in generated_ids[start_idx:]:
            if token_id == self.EOS_TOKEN_ID:
                break
            # Only include actual words, not special tokens (except if they are part of the response)
            if token_id not in [self.PAD_TOKEN_ID, self.UNK_TOKEN_ID, self.BOS_TOKEN_ID]:
                response_tokens.append(self.idx_to_word.get(token_id, "<UNK>"))

        return " ".join(response_tokens)

    def save(self):
        """
        Saves the Transformer model parameters and the vocabulary.
        """
        # Save transformer parameters
        np.savez(self.model_path, **self.transformer.params)

        # Save vocabulary
        with open(self.vocab_path, 'wb') as f:
            pickle.dump(
                {'word_to_idx': self.word_to_idx, 'idx_to_word': self.idx_to_word, 'vocab_size': self.vocab_size}, f)
        # print(f"Model and vocabulary saved to {self.model_path} and {self.vocab_path}")

    def load(self):
        """
        Loads the Transformer model parameters and the vocabulary from saved files.
        """
        if os.path.exists(self.model_path) and os.path.exists(self.vocab_path):
            # Load vocabulary first to correctly initialize transformer's vocab_size
            with open(self.vocab_path, 'rb') as f:
                vocab_data = pickle.load(f)
                self.word_to_idx = vocab_data['word_to_idx']
                self.idx_to_word = vocab_data['idx_to_word']
                self.vocab_size = vocab_data['vocab_size']

            # Update transformer's vocab size before loading weights
            # This ensures the parameter matrices are correctly sized to receive loaded weights
            self.transformer.update_vocab_size(self.vocab_size)

            # Load transformer parameters
            loaded_params = np.load(self.model_path, allow_pickle=True)
            for k, v in loaded_params.items():
                # Direct assignment for top-level params
                if k in self.transformer.params:
                    self.transformer.params[k][:] = v  # Use [:] to modify in-place

                # Handle nested parameters for blocks, attention, and feed_forward
                for i, block in enumerate(self.transformer.transformer_blocks):
                    if k.startswith(f'block_{i}_'):
                        if k.startswith(f'block_{i}_attn_'):
                            attn_param_name = k[len(f'block_{i}_attn_'):]
                            if attn_param_name in block.attention.params:
                                block.attention.params[attn_param_name][:] = v
                        elif k.startswith(f'block_{i}_ff_'):
                            ff_param_name = k[len(f'block_{i}_ff_'):]
                            if ff_param_name in block.feed_forward.params:
                                block.feed_forward.params[ff_param_name][:] = v
                        else:  # Block's own parameters (e.g., layer norm gamma/beta)
                            block_param_name = k[len(f'block_{i}_'):]
                            if block_param_name in block.params:
                                block.params[block_param_name][:] = v
            # print(f"Model and vocabulary loaded from {self.model_path} and {self.vocab_path}")
        else:
            print("No saved model or vocabulary found. Starting with a fresh model.")
