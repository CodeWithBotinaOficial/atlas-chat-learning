# atlas/brain.py
import numpy as np
import pickle
import re
import os
import random

from atlas.transformer import Transformer
from atlas.grammar import GrammarHelper # Import GrammarHelper
from atlas.config_loader import load_config # Import load_config


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

    def __init__(self, model_path="atlas_model.npz", vocab_path="atlas_vocab.pkl", config=None):
        self.model_path = model_path
        self.vocab_path = vocab_path

        if config is None:
            self.config = load_config()
        else:
            self.config = config

        # Apply low_memory preset if enabled
        if self.config.get('performance', {}).get('low_memory', False):
            print("Low-memory mode enabled. Applying memory-optimized configuration.")
            self.config['model']['embed_dim'] = 16
            self.config['model']['num_heads'] = 2
            self.config['model']['ff_dim'] = 32
            self.config['model']['num_layers'] = 1
            self.config['model']['max_seq_len'] = 25
            self.config['model']['dropout_rate'] = 0.0
            self.config['generation']['beam_width'] = 0
            self.config['generation']['top_k'] = 10
            self.config['generation']['max_new_tokens'] = 30
            self.config['training']['replay_buffer_size'] = 5
            self.config['training']['replay_sample_rate'] = 0.0
            self.config['memory']['max_history_length'] = 2

        self.word_to_idx = {}
        self.idx_to_word = {}
        self.vocab_size = 0

        # Initialize vocabulary with special tokens
        for token in self.SPECIAL_TOKENS:
            self._add_word_to_vocab(token)

        # Transformer hyperparameters from config
        self.embed_dim = self.config['model']['embed_dim']
        self.num_heads = self.config['model']['num_heads']
        self.ff_dim = self.config['model']['ff_dim']
        self.num_layers = self.config['model']['num_layers']
        self.max_seq_len = self.config['model']['max_seq_len']
        self.dropout_rate = self.config['model']['dropout_rate']

        # Training hyperparameters from config
        self.learning_rate = self.config['training']['learning_rate']
        self.lr_decay_rate = self.config['training']['lr_decay_rate']
        self.lr_decay_steps = self.config['training']['lr_decay_steps']
        self.replay_buffer_size = self.config['training']['replay_buffer_size']
        self.replay_sample_rate = self.config['training']['replay_sample_rate']

        # Generation hyperparameters from config
        self.temperature = self.config['generation']['temperature']
        self.repetition_penalty = self.config['generation']['repetition_penalty']
        self.top_k = self.config['generation']['top_k']
        self.top_p = self.config['generation']['top_p']
        self.beam_width = self.config['generation']['beam_width']
        self.max_new_tokens = self.config['generation']['max_new_tokens']

        # Memory hyperparameters from config
        self.max_history_length = self.config['memory']['max_history_length']

        # NaN check frequency (not in config, keep as default or add to config if desired)
        self.nan_check_interval = 10 # Check every 10 interactions

        # Conversation history buffer
        self.conversation_history = []  # Stores (user_message_ids, atlas_response_ids)

        # Replay buffer for incremental learning
        self.replay_buffer = []  # Stores (input_ids, target_ids) for past user messages

        # Learning rate decay
        self.interaction_count = 0

        # Initialize Transformer with current vocab_size and config
        transformer_config = {
            'vocab_size': self.vocab_size,
            'model': self.config['model'],
            'generation': self.config['generation'],
            'performance': self.config.get('performance', {}) # Pass performance config safely
        }
        self.transformer = Transformer(transformer_config)

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

    def _prepare_sequence_for_training(self, text_ids):
        """
        Prepares a sequence of token IDs for transformer training.
        Adds BOS/EOS, truncates, and pads.
        """
        full_sequence_ids = [self.BOS_TOKEN_ID] + text_ids + [self.EOS_TOKEN_ID]

        # Skip if sequence is only special tokens (length <= 2 after BOS/EOS)
        if len(full_sequence_ids) <= len(self.SPECIAL_TOKENS) - 2: # BOS, EOS, and potentially PAD/UNK
             return None, None

        if len(full_sequence_ids) > self.max_seq_len:
            full_sequence_ids = full_sequence_ids[-self.max_seq_len:]

        input_ids = full_sequence_ids[:-1]
        target_ids = full_sequence_ids[1:]

        input_ids_padded = np.full(self.max_seq_len, self.PAD_TOKEN_ID, dtype=int)
        target_ids_padded = np.full(self.max_seq_len, self.PAD_TOKEN_ID, dtype=int)

        input_ids_padded[:len(input_ids)] = input_ids
        target_ids_padded[:len(target_ids)] = target_ids

        return np.array([input_ids_padded]), np.array([target_ids_padded])

    def _check_weights_for_nan(self):
        """
        Scans transformer parameters for NaN/inf and reinitializes if found.
        """
        nan_found = False
        for name, param in self.transformer.params.items():
            if np.any(np.isnan(param)) or np.any(np.isinf(param)):
                print(f"WARNING: NaN or Inf found in parameter: {name}. Reinitializing model.")
                nan_found = True
                break
        
        if nan_found:
            # Reinitialize the entire transformer
            transformer_config = {
                'vocab_size': self.vocab_size,
                'model': self.config['model'],
                'generation': self.config['generation'],
                'performance': self.config.get('performance', {})
            }
            self.transformer = Transformer(transformer_config)
            # Ensure vocab size is updated for the new transformer
            self.transformer.update_vocab_size(self.vocab_size)
            print("Model reinitialized due to NaN/Inf weights.")
            return True
        return False


    def learn(self, user_message):
        """
        Processes user input, updates vocabulary, and performs one training step
        on the Transformer model. Also manages conversation history and replay buffer.
        """
        user_tokens = self._tokenize(user_message)
        
        # Skip learning if user message is too short after tokenization
        if not user_tokens or len(user_tokens) < 2:
            print("Skipping learning: user message too short after tokenization.")
            return None
        
        self.interaction_count += 1 # Increment only for valid interactions
        current_learning_rate = self.learning_rate * (self.lr_decay_rate ** (self.interaction_count // self.lr_decay_steps))

        user_ids = self._words_to_ids(user_tokens)

        input_batch, target_batch = self._prepare_sequence_for_training(user_ids)
        
        # Skip training if the sequence is only special tokens
        if input_batch is None or target_batch is None:
            print("Skipping training: input sequence contains only special tokens.")
            return None

        # Add to replay buffer
        if len(self.replay_buffer) >= self.replay_buffer_size:
            self.replay_buffer.pop(0)  # Remove oldest
        self.replay_buffer.append((input_batch, target_batch))

        # Store initial embedding for "hello" if it exists
        hello_embedding_before = None
        if "hello" in self.word_to_idx:
            hello_id = self.word_to_idx["hello"]
            hello_embedding_before = np.copy(self.transformer.token_embedding[hello_id])

        # Train on current user message
        loss = self.transformer.train_step(input_batch, target_batch, current_learning_rate, training=True, pad_token_id=self.PAD_TOKEN_ID)

        # Check embedding for "hello" after training
        if "hello" in self.word_to_idx and hello_embedding_before is not None:
            hello_id = self.word_to_idx["hello"]
            hello_embedding_after = self.transformer.token_embedding[hello_id]
            if not np.array_equal(hello_embedding_before, hello_embedding_after):
                print(f"DEBUG: 'hello' embedding changed after training step {self.interaction_count}.")
            else:
                print(f"DEBUG: 'hello' embedding DID NOT change after training step {self.interaction_count}.")
        elif "hello" not in self.word_to_idx and self.vocab_size > self.UNK_TOKEN_ID:
            # If 'hello' is not in vocab, check UNK token embedding as a fallback
            unk_embedding_before = np.copy(self.transformer.token_embedding[self.UNK_TOKEN_ID])
            # Re-run train_step to ensure it's not just the first check
            _ = self.transformer.train_step(input_batch, target_batch, current_learning_rate, training=True, pad_token_id=self.PAD_TOKEN_ID)
            unk_embedding_after = self.transformer.token_embedding[self.UNK_TOKEN_ID]
            if not np.array_equal(unk_embedding_before, unk_embedding_after):
                # print(f"DEBUG: UNK token embedding changed after training step {self.interaction_count}.")
                pass
            else:
                # print(f"DEBUG: UNK token embedding DID NOT change after training step {self.interaction_count}.")
                pass


        # Occasionally sample from replay buffer for additional training
        if self.replay_buffer and random.random() < self.replay_sample_rate:
            replay_input, replay_target = random.choice(self.replay_buffer)
            replay_loss = self.transformer.train_step(replay_input, replay_target, current_learning_rate, training=True, pad_token_id=self.PAD_TOKEN_ID)
            # if replay_loss is not None:
            #     print(f"Replay loss: {replay_loss:.4f}")

        # Check for NaN/inf weights occasionally
        if self.interaction_count % self.nan_check_interval == 0:
            self._check_weights_for_nan()

        # if loss is not None:
        #     print(f"Learning loss: {loss:.4f}, Current LR: {current_learning_rate:.6f}")
        return loss

    def respond(self, prompt=None):
        """
        Generates a response based on an optional prompt, incorporating conversation history.
        """
        default_message_learning = "I'm learning to speak... please teach me more!"
        specific_empty_response_warning = "I need more training before I can respond properly."

        if self.vocab_size <= len(self.SPECIAL_TOKENS):  # Only special tokens in vocab
            return default_message_learning

        # Construct context from conversation history
        context_ids = []
        for user_hist_ids, atlas_hist_ids in self.conversation_history:
            context_ids.extend(user_hist_ids + [self.EOS_TOKEN_ID] + atlas_hist_ids + [self.EOS_TOKEN_ID])

        if prompt:
            prompt_tokens = self._tokenize(prompt)
            prompt_ids = self._words_to_ids(prompt_tokens)
            # Prepend BOS to the actual prompt for generation
            current_input_ids = context_ids + [self.BOS_TOKEN_ID] + prompt_ids
        else:
            current_input_ids = context_ids + [self.BOS_TOKEN_ID]

        # Truncate initial sequence to fit within max_seq_len for generation
        # Keep only the most recent tokens if too long
        if len(current_input_ids) > self.max_seq_len - 1: # -1 to leave space for at least one generated token
            current_input_ids = current_input_ids[len(current_input_ids) - (self.max_seq_len - 1):]

        prompt_tensor = np.array([current_input_ids])

        response_tokens_ids = self.transformer.generate(
            prompt_tensor,
            pad_token_id=self.PAD_TOKEN_ID,
            eos_token_id=self.EOS_TOKEN_ID,
        )

        # Decode generated tokens, stopping at EOS
        response_tokens = []
        # Find where the actual generation starts (after the prompt/context)
        # The generated_ids will include the prompt_tensor, so we need to slice it
        start_idx = len(current_input_ids)

        # Filter out special tokens and collect actual response tokens
        for token_id in response_tokens_ids[start_idx:]:
            if token_id == self.EOS_TOKEN_ID:
                break
            if token_id not in [self.PAD_TOKEN_ID, self.UNK_TOKEN_ID, self.BOS_TOKEN_ID]:
                response_tokens.append(self.idx_to_word.get(token_id, "<UNK>"))

        raw_atlas_response = " ".join(response_tokens).strip()

        # If raw_atlas_response is empty after token filtering, return a specific warning
        if not raw_atlas_response:
            return specific_empty_response_warning

        # Apply grammatical post-processing
        # Pass the original prompt as previous_user_message to GrammarHelper
        corrected_atlas_response = GrammarHelper.apply_all(raw_atlas_response, prompt)
        
        # Update conversation history with the grammatically corrected version
        if prompt:
            user_hist_ids = self._words_to_ids(self._tokenize(prompt))
        else:
            user_hist_ids = []

        # Tokenize the corrected_atlas_response for history storage
        atlas_hist_ids = self._words_to_ids(self._tokenize(corrected_atlas_response))

        self.conversation_history.append((user_hist_ids, atlas_hist_ids))
        if len(self.conversation_history) > self.max_history_length:
            self.conversation_history.pop(0)

        return corrected_atlas_response

    def save(self):
        """
        Saves the Transformer model parameters, vocabulary, conversation history,
        replay buffer, and interaction count.
        """
        # Save transformer parameters
        np.savez(self.model_path, **self.transformer.params)

        # Save vocabulary and other brain state
        brain_state = {
            'word_to_idx': self.word_to_idx,
            'idx_to_word': self.idx_to_word,
            'vocab_size': self.vocab_size,
            'conversation_history': self.conversation_history,
            'replay_buffer': self.replay_buffer,
            'interaction_count': self.interaction_count,
            'config': self.config, # Save the entire config
        }
        with open(self.vocab_path, 'wb') as f:
            pickle.dump(brain_state, f)
        # print(f"Model and brain state saved to {self.model_path} and {self.vocab_path}")

    def load(self):
        """
        Loads the Transformer model parameters, vocabulary, conversation history,
        replay buffer, and interaction count from saved files.
        """
        if os.path.exists(self.model_path) and os.path.exists(self.vocab_path):
            # Load brain state first to correctly initialize transformer's vocab_size
            with open(self.vocab_path, 'rb') as f:
                brain_state = pickle.load(f)
                self.word_to_idx = brain_state['word_to_idx']
                self.idx_to_word = brain_state['idx_to_word']
                self.vocab_size = brain_state['vocab_size']
                self.conversation_history = brain_state.get('conversation_history', [])
                self.replay_buffer = brain_state.get('replay_buffer', [])
                self.interaction_count = brain_state.get('interaction_count', 0)
                
                # Load config from saved state, if available, otherwise use current config
                loaded_config = brain_state.get('config')
                if loaded_config:
                    # Merge loaded config with current config to handle new parameters
                    # This prioritizes loaded values but allows new config defaults to be used
                    def merge_configs(base, new):
                        for k, v in new.items():
                            if k in base and isinstance(base[k], dict) and isinstance(v, dict):
                                base[k] = merge_configs(base[k], v)
                            else:
                                base[k] = v
                        return base
                    self.config = merge_configs(self.config, loaded_config)

                # Apply low_memory preset if enabled AFTER loading and merging config
                if self.config.get('performance', {}).get('low_memory', False):
                    print("Low-memory mode enabled during load. Applying memory-optimized configuration.")
                    self.config['model']['embed_dim'] = 16
                    self.config['model']['num_heads'] = 2
                    self.config['model']['ff_dim'] = 32
                    self.config['model']['num_layers'] = 1
                    self.config['model']['max_seq_len'] = 25
                    self.config['model']['dropout_rate'] = 0.0
                    self.config['generation']['beam_width'] = 0
                    self.config['generation']['top_k'] = 10
                    self.config['generation']['max_new_tokens'] = 30
                    self.config['training']['replay_buffer_size'] = 5
                    self.config['training']['replay_sample_rate'] = 0.0
                    self.config['memory']['max_history_length'] = 2

                # Apply loaded config values
                self.embed_dim = self.config['model']['embed_dim']
                self.num_heads = self.config['model']['num_heads']
                self.ff_dim = self.config['model']['ff_dim']
                self.num_layers = self.config['model']['num_layers']
                self.max_seq_len = self.config['model']['max_seq_len']
                self.dropout_rate = self.config['model']['dropout_rate']

                self.learning_rate = self.config['training']['learning_rate']
                self.lr_decay_rate = self.config['training']['lr_decay_rate']
                self.lr_decay_steps = self.config['training']['lr_decay_steps']
                self.replay_buffer_size = self.config['training']['replay_buffer_size']
                self.replay_sample_rate = self.config['training']['replay_sample_rate']

                self.temperature = self.config['generation']['temperature']
                self.repetition_penalty = self.config['generation']['repetition_penalty']
                self.top_k = self.config['generation']['top_k']
                self.top_p = self.config['generation']['top_p']
                self.beam_width = self.config['generation']['beam_width']
                self.max_new_tokens = self.config['generation']['max_new_tokens']

                self.max_history_length = self.config['memory']['max_history_length']


            # Re-initialize transformer with loaded hyperparameters
            transformer_config = {
                'vocab_size': self.vocab_size,
                'model': self.config['model'],
                'generation': self.config['generation'],
                'performance': self.config.get('performance', {}) # Pass performance config safely
            }
            self.transformer = Transformer(transformer_config)
            # Update transformer's vocab size after re-initialization
            self.transformer.update_vocab_size(self.vocab_size)

            # Load transformer parameters
            loaded_params = np.load(self.model_path, allow_pickle=True)
            for k, v in loaded_params.items():
                # Direct assignment for top-level params
                if k in self.transformer.params:
                    # Ensure shapes match before assignment, especially if hyperparameters changed
                    if self.transformer.params[k].shape == v.shape:
                        self.transformer.params[k][:] = v  # Use [:] to modify in-place
                    else:
                        print(f"WARNING: Parameter {k} shape mismatch. This may happen if you changed model hyperparameters. Consider deleting atlas_model.npz and retraining. Expected {self.transformer.params[k].shape}, got {v.shape}.")

                # Handle nested parameters for blocks, attention, and feed_forward
                for i, block in enumerate(self.transformer.transformer_blocks):
                    if k.startswith(f'block_{i}_'):
                        if k.startswith(f'block_{i}_attn_'):
                            attn_param_name = k[len(f'block_{i}_attn_'):]
                            if attn_param_name in block.attention.params:
                                if block.attention.params[attn_param_name].shape == v.shape:
                                    block.attention.params[attn_param_name][:] = v
                                else:
                                    print(f"WARNING: Parameter {k} shape mismatch. This may happen if you changed model hyperparameters. Consider deleting atlas_model.npz and retraining. Skipping load for this parameter.")
                        elif k.startswith(f'block_{i}_ff_'):
                            ff_param_name = k[len(f'block_{i}_ff_'):]
                            if ff_param_name in block.feed_forward.params:
                                if block.feed_forward.params[ff_param_name].shape == v.shape:
                                    block.feed_forward.params[ff_param_name][:] = v
                                else:
                                    print(f"WARNING: Parameter {k} shape mismatch. This may happen if you changed model hyperparameters. Consider deleting atlas_model.npz and retraining. Skipping load for this parameter.")
                        else:  # Block's own parameters (e.g., layer norm gamma/beta)
                            block_param_name = k[len(f'block_{i}_'):]
                            if block_param_name in block.params:
                                if block.params[block_param_name].shape == v.shape:
                                    block.params[block_param_name][:] = v
                                else:
                                    print(f"WARNING: Parameter {k} shape mismatch. This may happen if you changed model hyperparameters. Consider deleting atlas_model.npz and retraining. Skipping load for this parameter.")
            # print(f"Model and brain state loaded from {self.model_path} and {self.vocab_path}")
        else:
            print("No saved model or vocabulary found. Starting with a fresh model.")