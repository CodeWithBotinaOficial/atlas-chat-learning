import numpy as np
import math
from atlas.models.initialization import xavier_init, clip_gradients
from atlas.models.positional_encoding import PositionalEncoding
from atlas.models.transformer_block import TransformerBlock
from atlas.models.activations import softmax
from atlas.models.losses import label_smoothing_loss, label_smoothing_backward
from atlas.models.sampling import _top_k_top_p_sampling

class Transformer:
    """
    A small Transformer model for sequence-to-sequence tasks.
    """

    def __init__(self, config):
        self.vocab_size = config['vocab_size']
        self.embed_dim = config['model']['embed_dim']
        self.num_heads = config['model']['num_heads']
        self.ff_dim = config['model']['ff_dim']
        self.num_layers = config['model']['num_layers']
        self.max_seq_len = config['model']['max_seq_len']
        self.dropout_rate = config['model']['dropout_rate']
        self.half_precision = config['performance']['half_precision']
        self.dtype = np.float16 if self.half_precision else np.float32

        # Generation parameters
        self.temperature = config['generation']['temperature']
        self.repetition_penalty = config['generation']['repetition_penalty']
        self.top_k = config['generation']['top_k']
        self.top_p = config['generation']['top_p']
        self.beam_width = config['generation']['beam_width']
        self.max_new_tokens = config['generation']['max_new_tokens']

        # Special token IDs (assuming common defaults if not in config)
        # These are now defaults for the generate method, but still useful to store
        # for internal logic or if generate is called without explicit overrides.
        self._unk_token_id = config.get('unk_token_id', 1)
        self._bos_token_id = config.get('bos_token_id', 2)


        self.token_embedding = xavier_init(self.vocab_size, self.embed_dim, dtype=self.dtype)
        self.positional_encoding = PositionalEncoding(self.embed_dim, self.max_seq_len, dtype=self.dtype)
        self.transformer_blocks = [
            TransformerBlock(self.embed_dim, self.num_heads, self.ff_dim, self.dropout_rate, dtype=self.dtype) for _ in range(self.num_layers)
        ]
        self.output_layer = (np.random.randn(self.embed_dim, self.vocab_size) * 0.01).astype(self.dtype)
        self.output_bias = np.zeros(self.vocab_size, dtype=self.dtype)

        # Adam optimizer state and warmup
        self.warmup_steps = 1000
        self.train_step_count = 0
        self.m = {}
        self.v = {}

        # Collect all parameters into a single dictionary for easy access and saving
        self.params = {
            'token_embedding': self.token_embedding,
            'output_layer': self.output_layer,
            'output_bias': self.output_bias,
        }
        # Initialize gradients for top-level parameters
        self.grads = {k: np.zeros_like(v) for k, v in self.params.items()}

        # Add parameters and gradients from transformer blocks and their sub-layers
        for i, block in enumerate(self.transformer_blocks):
            for k, v in block.params.items():
                self.params[f'block_{i}_{k}'] = v
            for k, v in block.grads.items():
                self.grads[f'block_{i}_{k}'] = v
            for k, v in block.attention.params.items():
                self.params[f'block_{i}_attn_{k}'] = v
            for k, v in block.attention.grads.items():
                self.grads[f'block_{i}_attn_{k}'] = v
            for k, v in block.feed_forward.params.items():
                self.params[f'block_{i}_ff_{k}'] = v
            for k, v in block.feed_forward.grads.items():
                self.grads[f'block_{i}_ff_{k}'] = v

        self._init_optimizer_state()

    def _init_optimizer_state(self):
        """Initialize Adam first and second moment estimates for all parameters."""
        self.m = {k: np.zeros_like(v) for k, v in self.params.items()}
        self.v = {k: np.zeros_like(v) for k, v in self.params.items()}

    def _collect_all_grads(self):
        """Collect gradients from all model parameters into a single dictionary."""
        all_grads = {}
        for k in ['token_embedding', 'output_layer', 'output_bias']:
            if k in self.grads:
                all_grads[k] = self.grads[k]

        for i, block in enumerate(self.transformer_blocks):
            for k in ['norm1_gamma', 'norm1_beta', 'norm2_gamma', 'norm2_beta']:
                if k in block.grads:
                    all_grads[f'block_{i}_{k}'] = block.grads[k]
            for k in block.attention.grads.keys():
                if k in block.attention.grads:
                    all_grads[f'block_{i}_attn_{k}'] = block.attention.grads[k]
            for k in block.feed_forward.grads.keys():
                if k in block.feed_forward.grads:
                    all_grads[f'block_{i}_ff_{k}'] = block.feed_forward.grads[k]
        return all_grads

    def update_vocab_size(self, new_vocab_size):
        """
        Dynamically updates the vocabulary size of the transformer.
        Expands token embedding and output layer weights/biases with new random values.
        Preserves existing weights.
        """
        if new_vocab_size > self.vocab_size:
            # print(f"Updating vocab size from {self.vocab_size} to {new_vocab_size}")

            # Expand token embedding matrix
            new_embeddings = xavier_init(new_vocab_size - self.vocab_size, self.embed_dim, dtype=self.dtype)
            self.token_embedding = np.vstack((self.token_embedding, new_embeddings))
            self.params['token_embedding'] = self.token_embedding  # Update reference
            self.grads['token_embedding'] = np.zeros_like(self.token_embedding)  # Reset grads for new shape

            # Expand output layer weights
            new_output_weights = xavier_init(self.embed_dim, new_vocab_size - self.vocab_size, dtype=self.dtype)
            self.output_layer = np.hstack((self.output_layer, new_output_weights))
            self.params['output_layer'] = self.output_layer  # Update reference
            self.grads['output_layer'] = np.zeros_like(self.output_layer)  # Reset grads for new shape

            # Expand output bias
            new_output_bias = np.zeros(new_vocab_size - self.vocab_size, dtype=self.dtype)
            self.output_bias = np.hstack((self.output_bias, new_output_bias))
            self.params['output_bias'] = self.output_bias  # Update reference
            self.grads['output_bias'] = np.zeros_like(self.output_bias)  # Reset grads for new shape

            # Expand Adam optimizer state for new vocabulary entries
            self.m['token_embedding'] = np.zeros_like(self.token_embedding)
            self.v['token_embedding'] = np.zeros_like(self.token_embedding)
            self.m['output_layer'] = np.zeros_like(self.output_layer)
            self.v['output_layer'] = np.zeros_like(self.output_layer)
            self.m['output_bias'] = np.zeros_like(self.output_bias)
            self.v['output_bias'] = np.zeros_like(self.output_bias)

            self.vocab_size = new_vocab_size

    def _create_look_ahead_mask(self, seq_len):
        """
        Creates a look-ahead mask to prevent attention to future tokens.
        Returns: (1, 1, seq_len, seq_len)
        """
        mask = np.triu(np.ones((seq_len, seq_len)), k=1).astype(np.float32)
        mask = (mask * -1e9)  # Convert to additive mask (large negative value)
        return mask[np.newaxis, np.newaxis, :, :]  # (1, 1, seq_len, seq_len)

    def forward(self, x, training=True, cache=None):
        """
        Forward pass for the Transformer model.
        x: (batch_size, seq_len) - token IDs
        training: bool, if True, applies look-ahead mask and dropout.
        cache: optional KV cache dict for faster autoregressive inference.
               Maps layer index -> {'k': array, 'v': array}, plus '_past_seq_len'.
        Returns: (batch_size, seq_len, vocab_size) - logits for each token.
        """
        batch_size, seq_len = x.shape
        use_kv_cache = cache is not None and not training
        past_seq_len = cache.get('_past_seq_len', 0) if use_kv_cache else 0
        incremental = use_kv_cache and past_seq_len > 0 and seq_len == 1

        # Store token IDs for backward pass (for embedding gradients)
        self.cache_x_token_ids = x

        # Token and positional embeddings (scale embeddings per original Transformer paper)
        embeddings = self.token_embedding[x].astype(np.float32)  # Cast to float32 for computation
        embeddings = embeddings * math.sqrt(self.embed_dim)
        start_pos = past_seq_len if incremental else 0
        x = self.positional_encoding.forward(embeddings, start_pos=start_pos)

        if use_kv_cache and not incremental:
            for layer_idx in list(cache.keys()):
                if isinstance(layer_idx, int):
                    cache[layer_idx] = {'k': None, 'v': None}
            cache['_past_seq_len'] = 0

        # Create look-ahead mask for decoder if training
        mask = self._create_look_ahead_mask(seq_len) if training else None
        self.cache_mask = mask  # Cache the mask for backward pass in blocks

        # Transformer blocks
        for layer_idx, block in enumerate(self.transformer_blocks):
            x = block.forward(x, mask, training, cache=cache, layer_idx=layer_idx)

        if use_kv_cache:
            cache['_past_seq_len'] = past_seq_len + seq_len

        # Store output of last block for backward pass (input to final linear layer)
        self.cache_x_after_blocks = x

        # Final linear layer
        output_logits = x @ self.output_layer.astype(np.float32) + self.output_bias.astype(np.float32)  # (batch_size, seq_len, vocab_size)
        return output_logits

    def backward(self, d_output_logits):
        """
        Backward pass for the Transformer model.
        d_output_logits: (batch_size, seq_len, vocab_size) - gradient from the loss function.
        """
        batch_size, seq_len, _ = d_output_logits.shape

        # Gradients for output layer
        self.grads['output_layer'] = np.einsum('bsd,bsv->dv', self.cache_x_after_blocks, d_output_logits)
        self.grads['output_bias'] = np.sum(d_output_logits, axis=(0, 1))
        d_x_from_output = d_output_logits @ self.output_layer.astype(np.float32).T

        # Backward through transformer blocks
        d_x = d_x_from_output
        for block in reversed(self.transformer_blocks):
            d_x = block.backward(d_x)

        # Gradients for token embeddings (account for embedding scaling in forward pass)
        embed_scale = math.sqrt(self.embed_dim)
        d_token_embedding = np.zeros_like(self.token_embedding, dtype=np.float32)
        for i in range(batch_size):
            for j in range(seq_len):
                d_token_embedding[self.cache_x_token_ids[i, j]] += d_x[i, j] * embed_scale
        self.grads['token_embedding'] = d_token_embedding

    def train_step(self, input_tokens, target_tokens, learning_rate=0.01, training=True, pad_token_id=0):
        """
        Performs one training step (forward, loss, backward, update).
        input_tokens: (batch_size, seq_len) - input token IDs.
        target_tokens: (batch_size, seq_len) - target token IDs (shifted input).
        learning_rate: float.
        training: bool, if True, apply dropout and look-ahead mask.
        pad_token_id: int - ID of the padding token.
        Returns: float - computed loss, or None if NaN/Inf detected.
        """
        # Forward pass
        output_logits = self.forward(input_tokens, training=training)

        # Check for NaN/inf in output_logits
        if np.any(np.isnan(output_logits)) or np.any(np.isinf(output_logits)):
            print("WARNING: NaN or Inf detected in output_logits. Skipping gradient update.")
            return None

        # Reshape for loss calculation: (batch_size * seq_len, vocab_size)
        predictions = softmax(output_logits.reshape(-1, self.vocab_size))
        targets_flat = target_tokens.flatten()

        # Create padding mask
        padding_mask = (targets_flat != pad_token_id)

        loss = label_smoothing_loss(predictions, targets_flat, self.vocab_size, smoothing=0.1, padding_mask=padding_mask)

        # Check for NaN/inf in loss
        if np.isnan(loss) or np.isinf(loss):
            print("WARNING: NaN or Inf detected in loss. Skipping gradient update.")
            return None

        # Backward pass
        d_predictions = label_smoothing_backward(predictions, targets_flat, self.vocab_size, smoothing=0.1, padding_mask=padding_mask)
        d_output_logits = d_predictions.reshape(output_logits.shape)

        self.backward(d_output_logits)

        self.train_step_count += 1
        if self.train_step_count < self.warmup_steps:
            effective_lr = learning_rate * (self.train_step_count / self.warmup_steps)
        else:
            effective_lr = learning_rate

        self.update_weights_adam(effective_lr)
        return loss

    def update_weights_adam(self, learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8):
        """
        Updates all model parameters using Adam with the computed gradients.
        Applies gradient clipping before the Adam update.
        """
        all_grads = self._collect_all_grads()

        # Apply gradient clipping
        clip_gradients(all_grads, max_norm=5.0)

        t = self.train_step_count

        def _adam_update(param_name, grad, m_key):
            # Retrieve the parameter by name
            param = self.params[param_name]

            # Cast parameter and gradient to float32 for Adam computation
            param_f32 = param.astype(np.float32)
            grad_f32 = grad.astype(np.float32)

            self.m[m_key] = beta1 * self.m[m_key] + (1 - beta1) * grad_f32
            self.v[m_key] = beta2 * self.v[m_key] + (1 - beta2) * (grad_f32 ** 2)
            m_hat = self.m[m_key] / (1 - beta1 ** t)
            v_hat = self.v[m_key] / (1 - beta2 ** t)
            
            # Update in float32, then cast back to original dtype
            updated_param_f32 = param_f32 - learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
            self.params[param_name][:] = updated_param_f32.astype(self.dtype)


        for k in ['token_embedding', 'output_layer', 'output_bias']:
            if k in self.params and k in all_grads:
                _adam_update(k, all_grads[k], k)

        for i, block in enumerate(self.transformer_blocks):
            for k in ['norm1_gamma', 'norm1_beta', 'norm2_gamma', 'norm2_beta']:
                param_name = f'block_{i}_{k}'
                grad_key = f'block_{i}_{k}'
                if param_name in self.params and grad_key in all_grads:
                    _adam_update(param_name, all_grads[grad_key], grad_key)

            for k_attn in block.attention.params.keys():
                param_name = f'block_{i}_attn_{k_attn}'
                grad_key = f'block_{i}_attn_{k_attn}'
                if param_name in self.params and grad_key in all_grads:
                    _adam_update(param_name, all_grads[grad_key], grad_key)

            for k_ff in block.feed_forward.params.keys():
                param_name = f'block_{i}_ff_{k_ff}'
                grad_key = f'block_{i}_ff_{k_ff}'
                if param_name in self.params and grad_key in all_grads:
                    _adam_update(param_name, all_grads[grad_key], grad_key)

    def _beam_search_generate(self, prompt_tokens, pad_token_id, eos_token_id):
        """
        Generates tokens using beam search.
        """
        # Beams are (log_probability, sequence_of_token_ids)
        beams = [(0.0, list(prompt_tokens[0]))]
        
        # Define special_token_ids for this generation call
        special_token_ids = {pad_token_id, self._unk_token_id, self._bos_token_id, eos_token_id}

        for _ in range(self.max_new_tokens):
            all_candidates = []
            for log_prob, current_sequence in beams:
                if current_sequence[-1] == eos_token_id:
                    all_candidates.append((log_prob, current_sequence))
                    continue

                # Truncate if sequence exceeds max_seq_len
                input_seq = current_sequence[-self.max_seq_len:]
                input_tensor = np.array(input_seq).reshape(1, -1)

                output_logits = self.forward(input_tensor, training=False)
                last_token_logits = output_logits[0, -1, :] / self.temperature

                if self.repetition_penalty > 1.0:
                    unique_current_tokens = np.unique([t for t in current_sequence if t != pad_token_id])
                    for token_id in unique_current_tokens:
                        if 0 <= token_id < self.vocab_size:
                            last_token_logits[token_id] /= self.repetition_penalty

                probabilities = softmax(last_token_logits[np.newaxis, :])[0]
                
                # Check for NaN or Inf in probabilities
                if np.any(np.isnan(probabilities)) or np.any(np.isinf(probabilities)):
                    print("WARNING: NaN or Inf detected in probabilities during beam search sampling.")
                    # Fallback strategy: if logits are also problematic, choose randomly from top 10 valid indices
                    valid_indices = np.where(~np.isnan(last_token_logits) & ~np.isinf(last_token_logits))[0]
                    
                    # Filter out special tokens for a more meaningful fallback
                    filtered_valid_indices = [idx for idx in valid_indices if idx not in special_token_ids]

                    if len(filtered_valid_indices) > 0:
                        top_valid_indices = filtered_valid_indices[np.argsort(last_token_logits[filtered_valid_indices])][::-1][:min(10, len(filtered_valid_indices))]
                        print(f"Falling back to random choice from top {len(top_valid_indices)} valid non-special logits for beam search.")
                        next_token_id = np.random.choice(top_valid_indices)
                    elif len(valid_indices) > 0: # If only special tokens are valid
                        top_valid_indices = valid_indices[np.argsort(last_token_logits[valid_indices])][::-1][:min(10, len(valid_indices))]
                        print(f"Falling back to random choice from top {len(top_valid_indices)} valid (possibly special) logits for beam search.")
                        next_token_id = np.random.choice(top_valid_indices)
                    else:
                        print("WARNING: All logits are NaN/Inf. Falling back to random non-special token from vocabulary for beam search.")
                        non_special_indices = [i for i in range(self.vocab_size) if i not in special_token_ids]
                        if len(non_special_indices) > 0:
                            next_token_id = np.random.choice(non_special_indices)
                        else:
                            next_token_id = np.random.randint(0, self.vocab_size) # If only special tokens exist
                    
                    # Assign a very low probability to this fallback token to keep beam search stable
                    new_log_prob = log_prob + np.log(1e-6)
                    new_sequence = current_sequence + [next_token_id]
                    all_candidates.append((new_log_prob, new_sequence))
                    continue # Skip other candidates for this beam if NaN

                # Get top beam_width candidates for the next token
                # Ensure indices are within vocab_size, which they should be if last_token_logits is vocab_size long
                top_token_indices = np.argsort(probabilities)[::-1][:self.beam_width]

                for next_token_id in top_token_indices:
                    prob = max(probabilities[next_token_id], 1e-9)
                    new_log_prob = log_prob + np.log(prob)
                    new_sequence = current_sequence + [next_token_id]
                    all_candidates.append((new_log_prob, new_sequence))
            
            # Sort all candidates by log probability and select top beam_width
            beams = sorted(all_candidates, key=lambda x: x[0], reverse=True)[:self.beam_width]

            # If all beams have ended, stop early
            if all(b[1][-1] == eos_token_id for b in beams):
                break
        
        # Return the sequence from the best beam
        return np.array(beams[0][1])


    def generate(self, prompt_tokens, pad_token_id=0, eos_token_id=3):
        """
        Generates new tokens based on a prompt using various sampling strategies.
        prompt_tokens: (1, seq_len) - initial token IDs.
        pad_token_id: int - ID of the padding token.
        eos_token_id: int - ID of the end-of-sequence token.
        Returns: np.ndarray - array of generated token IDs.
        """

        # Define special_token_ids for this generation call
        special_token_ids = {pad_token_id, self._unk_token_id, self._bos_token_id, eos_token_id}

        if self.beam_width > 0:
            return self._beam_search_generate(prompt_tokens, pad_token_id, eos_token_id)

        generated_tokens = list(prompt_tokens[0])
        initial_prompt_len = len(generated_tokens)
        batch_size = 1  # Generation is always batch size 1
        kv_cache = {}
        last_token_logits = None # To store logits for fallback if no tokens are generated

        for step in range(self.max_new_tokens):
            if len(generated_tokens) == 0:
                print("WARNING: generated_tokens became empty during generation loop. Breaking.")
                break

            if len(generated_tokens) > self.max_seq_len:
                # Truncation requires a full recomputation of the KV cache.
                input_seq = generated_tokens[-self.max_seq_len:]
                kv_cache = {}
                input_tensor = np.array(input_seq).reshape(batch_size, -1)
                output_logits = self.forward(input_tensor, training=False, cache=kv_cache)
            elif step == 0:
                input_tensor = np.array(generated_tokens).reshape(batch_size, -1)
                kv_cache = {}
                output_logits = self.forward(input_tensor, training=False, cache=kv_cache)
            else:
                input_tensor = np.array([[generated_tokens[-1]]])
                output_logits = self.forward(input_tensor, training=False, cache=kv_cache)

            # Get logits for the last token in the sequence
            last_token_logits = output_logits[0, -1, :]

            if self.repetition_penalty > 1.0:
                # Apply repetition penalty
                unique_current_tokens = np.unique([t for t in generated_tokens if t != pad_token_id])
                for token_id in unique_current_tokens:
                    if 0 <= token_id < self.vocab_size:
                        last_token_logits[token_id] /= self.repetition_penalty

            # Sample next token using top-k and top-p
            next_token = _top_k_top_p_sampling(
                last_token_logits,
                self.vocab_size,
                self.top_k,
                self.top_p,
                self.temperature,
                special_token_ids=special_token_ids
            )
            
            generated_tokens.append(next_token)

            # Stop if EOS token is generated
            if next_token == eos_token_id:
                break

        # Ensure at least one new token is generated if the loop didn't add any
        if len(generated_tokens) == initial_prompt_len:
            print("WARNING: No new tokens were generated. Forcing generation of one token.")
            if last_token_logits is not None:
                # Try to pick the most probable non-special token
                # Exclude special tokens from consideration for forced generation
                non_special_token_logits = np.copy(last_token_logits)
                for st_id in special_token_ids:
                    if st_id < self.vocab_size:
                        non_special_token_logits[st_id] = -np.inf # Effectively remove special tokens

                if np.all(non_special_token_logits == -np.inf): # If all non-special tokens are masked
                    print("  All non-special tokens masked. Picking a random non-special token.")
                    non_special_indices = [i for i in range(self.vocab_size) if i not in special_token_ids]
                    if len(non_special_indices) > 0:
                        forced_token = np.random.choice(non_special_indices)
                    else:
                        # If only special tokens exist in vocab, pick UNK or a random token
                        forced_token = self._unk_token_id if self._unk_token_id < self.vocab_size else np.random.randint(0, self.vocab_size)
                else:
                    forced_token = np.argmax(non_special_token_logits)
                
                print(f"  Forced generation of token ID: {forced_token}")
                generated_tokens.append(forced_token)
            else:
                # If last_token_logits is None (e.g., prompt was empty or some error before first step)
                print("  last_token_logits was None. Appending UNK token.")
                generated_tokens.append(self._unk_token_id) # Fallback to UNK token

        return np.array(generated_tokens)
