import numpy as np
import math


# Helper functions
def softmax(x, epsilon=1e-12):
    """
    Compute softmax values for each row of x, with numerical stability.
    Clips input to prevent overflow and adds epsilon to denominator to prevent division by zero.
    """
    # Clip values to prevent overflow in exp()
    x = np.clip(x, -100, 100)
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / (np.sum(e_x, axis=-1, keepdims=True) + epsilon)

def dropout(x, dropout_rate, training):
    """
    Applies dropout to the input.
    x: input array
    dropout_rate: probability of an element being zeroed.
    training: boolean, if True, apply dropout, else return x as is.
    """
    if not training or dropout_rate == 0:
        return x
    mask = (np.random.rand(*x.shape) > dropout_rate) / (1 - dropout_rate)
    return x * mask

def layer_norm(x, gamma, beta, epsilon=1e-5):
    """Apply layer normalization."""
    mean = np.mean(x, axis=-1, keepdims=True)
    variance = np.var(x, axis=-1, keepdims=True)
    x_normalized = (x - mean) / np.sqrt(variance + epsilon)
    return gamma * x_normalized + beta, mean, variance, x_normalized


def relu(x):
    """ReLU activation function."""
    return np.maximum(0, x)


def relu_backward(dA, Z):
    """Backward pass for ReLU activation."""
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    return dZ

def xavier_init(input_dim, output_dim):
    """
    Xavier/Glorot uniform initialization for weights.
    """
    limit = np.sqrt(6 / (input_dim + output_dim))
    return np.random.uniform(-limit, limit, size=(input_dim, output_dim))

def clip_gradients(grads, max_norm=5.0):
    """
    Clips the global norm of gradients.
    grads: A dictionary of gradient arrays.
    max_norm: The maximum allowed global norm.
    """
    total_norm = 0.0
    # Calculate total norm
    for g in grads.values():
        if g is not None:
            total_norm += np.sum(g**2)
    total_norm = np.sqrt(total_norm)

    # Apply clipping if total_norm exceeds max_norm
    if total_norm > max_norm:
        clip_coef = max_norm / (total_norm + 1e-6) # Add epsilon to prevent division by zero
        for k in grads:
            if grads[k] is not None:
                grads[k] *= clip_coef


def label_smoothing_loss(predictions, targets, vocab_size, smoothing=0.1, epsilon=1e-9, padding_mask=None):
    """
    Compute cross-entropy loss with label smoothing.
    predictions: (batch_size * seq_len, vocab_size) - softmax probabilities
    targets: (batch_size * seq_len,) - integer class labels
    vocab_size: int - total number of possible classes
    smoothing: float - smoothing factor (e.g., 0.1)
    padding_mask: (batch_size * seq_len,) - boolean mask, True for valid tokens, False for padding.
    """
    # Clip predictions to avoid log(0)
    predictions = np.clip(predictions, epsilon, 1. - epsilon)

    # One-hot encode targets
    targets_one_hot = np.eye(vocab_size)[targets]

    # Apply label smoothing
    smoothed_targets = targets_one_hot * (1 - smoothing) + smoothing / vocab_size

    # Apply padding mask
    if padding_mask is not None:
        # Expand padding_mask to match predictions shape
        expanded_padding_mask = padding_mask[:, np.newaxis] # (batch_size * seq_len, 1)
        smoothed_targets = smoothed_targets * expanded_padding_mask
        # Mask predictions too, so log(0) doesn't contribute to loss for padded positions
        predictions = predictions * expanded_padding_mask + (1 - expanded_padding_mask) * epsilon # Avoid log(0) for masked positions

    # Calculate loss, only for non-padded positions
    if padding_mask is not None:
        num_valid_tokens = np.sum(padding_mask)
        if num_valid_tokens == 0:
            return 0.0 # No valid tokens, no loss
        loss = -np.sum(smoothed_targets * np.log(predictions)) / num_valid_tokens
    else:
        loss = -np.sum(smoothed_targets * np.log(predictions)) / predictions.shape[0]
    return loss


def label_smoothing_backward(predictions, targets, vocab_size, smoothing=0.1, padding_mask=None):
    """
    Compute gradient of cross-entropy loss with label smoothing with respect to predictions.
    predictions: (batch_size * seq_len, vocab_size) - softmax probabilities
    targets: (batch_size * seq_len,) - integer class labels
    vocab_size: int - total number of possible classes
    smoothing: float - smoothing factor (e.g., 0.1)
    padding_mask: (batch_size * seq_len,) - boolean mask, True for valid tokens, False for padding.
    Returns: (batch_size * seq_len, vocab_size)
    """
    targets_one_hot = np.eye(vocab_size)[targets]
    smoothed_targets = targets_one_hot * (1 - smoothing) + smoothing / vocab_size

    grad = (predictions - smoothed_targets)

    if padding_mask is not None:
        expanded_padding_mask = padding_mask[:, np.newaxis]
        grad = grad * expanded_padding_mask # Zero out gradients for padded positions
        num_valid_tokens = np.sum(padding_mask)
        if num_valid_tokens == 0:
            return np.zeros_like(grad) # No valid tokens, no gradient
        return grad / num_valid_tokens
    else:
        return grad / predictions.shape[0]  # Average over batch size


class PositionalEncoding:
    """
    Applies sinusoidal positional encoding to input embeddings.
    """

    def __init__(self, embed_dim, max_seq_len):
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        self.pe = self._create_positional_encoding()

    def _create_positional_encoding(self):
        pe = np.zeros((self.max_seq_len, self.embed_dim))
        position = np.arange(0, self.max_seq_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, self.embed_dim, 2) * -(math.log(10000.0) / self.embed_dim))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        return pe[np.newaxis, :, :]  # Add batch dimension (1, max_seq_len, embed_dim)

    def forward(self, x):
        """
        Adds positional encoding to the input embeddings.
        x: (batch_size, seq_len, embed_dim)
        Returns: (batch_size, seq_len, embed_dim)
        """
        seq_len = x.shape[1]
        # Positional encoding is broadcasted across the batch dimension
        return x + self.pe[:, :seq_len, :]


class MultiHeadSelfAttention:
    """
    Multi-Head Self-Attention mechanism.
    """

    def __init__(self, embed_dim, num_heads, dropout_rate=0.0):
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.head_dim = embed_dim // num_heads
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.dropout_rate = dropout_rate

        # Weights for Q, K, V for all heads (Xavier initialization)
        self.W_q = xavier_init(embed_dim, embed_dim)
        self.b_q = np.zeros(embed_dim)
        self.W_k = xavier_init(embed_dim, embed_dim)
        self.b_k = np.zeros(embed_dim)
        self.W_v = xavier_init(embed_dim, embed_dim)
        self.b_v = np.zeros(embed_dim)

        # Output projection (Xavier initialization)
        self.W_o = xavier_init(embed_dim, embed_dim)
        self.b_o = np.zeros(embed_dim)

        self.params = {
            'W_q': self.W_q, 'b_q': self.b_q,
            'W_k': self.W_k, 'b_k': self.b_k,
            'W_v': self.W_v, 'b_v': self.b_v,
            'W_o': self.W_o, 'b_o': self.b_o,
        }
        # Initialize gradients with zeros of the same shape as parameters
        self.grads = {k: np.zeros_like(v) for k, v in self.params.items()}

    def _split_heads(self, x, batch_size, seq_len):
        """
        Splits the input into multiple heads.
        x: (batch_size, seq_len, embed_dim)
        Returns: (batch_size, num_heads, seq_len, head_dim)
        """
        return x.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

    def _combine_heads(self, x, batch_size, seq_len):
        """
        Combines the output from multiple heads.
        x: (batch_size, num_heads, seq_len, head_dim)
        Returns: (batch_size, seq_len, embed_dim)
        """
        return x.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.embed_dim)

    def forward(self, x, mask=None, training=True):
        """
        Forward pass for Multi-Head Self-Attention.
        x: (batch_size, seq_len, embed_dim)
        mask: (1, 1, seq_len, seq_len) or None. Additive mask for attention scores.
        training: boolean, if True, apply dropout.
        Returns: (batch_size, seq_len, embed_dim)
        """
        batch_size, seq_len, _ = x.shape

        # Linear projections
        Q = x @ self.W_q + self.b_q
        K = x @ self.W_k + self.b_k
        V = x @ self.W_v + self.b_v

        # Split into multiple heads
        Q_heads = self._split_heads(Q, batch_size, seq_len)
        K_heads = self._split_heads(K, batch_size, seq_len)
        V_heads = self._split_heads(V, batch_size, seq_len)

        # Scaled Dot-Product Attention
        attention_scores = Q_heads @ K_heads.transpose(0, 1, 3, 2)
        attention_scores = attention_scores / math.sqrt(self.head_dim)

        if mask is not None:
            attention_scores = attention_scores + mask  # Additive mask

        attention_weights = softmax(attention_scores)
        attention_weights = dropout(attention_weights, self.dropout_rate, training) # Dropout after softmax

        attention_output_heads = attention_weights @ V_heads

        # Combine heads
        attention_output = self._combine_heads(attention_output_heads, batch_size, seq_len)

        # Output projection
        output = attention_output @ self.W_o + self.b_o
        output = dropout(output, self.dropout_rate, training) # Dropout after output projection

        # Save for backward pass
        self.cache = (x, Q, K, V, Q_heads, K_heads, V_heads, attention_scores, attention_weights,
                      attention_output_heads, attention_output, mask, training)
        return output

    def backward(self, d_output):
        """
        Backward pass for Multi-Head Self-Attention.
        d_output: (batch_size, seq_len, embed_dim) - gradient from the next layer.
        Returns: (batch_size, seq_len, embed_dim) - gradient with respect to the input x.
        """
        x, Q, K, V, Q_heads, K_heads, V_heads, attention_scores, attention_weights, attention_output_heads, attention_output, mask, training = self.cache
        batch_size, seq_len, _ = x.shape

        # Backward pass for dropout after output projection
        if training and self.dropout_rate > 0:
            d_output = d_output * ((np.random.rand(*d_output.shape) > self.dropout_rate) / (1 - self.dropout_rate))

        # Gradients for output projection
        self.grads['W_o'] = np.einsum('bsd,bse->de', attention_output, d_output)
        self.grads['b_o'] = np.sum(d_output, axis=(0, 1))
        d_attention_output = d_output @ self.W_o.T

        # Gradients for combining heads
        d_attention_output_heads = self._split_heads(d_attention_output, batch_size, seq_len)

        # Gradients for attention_weights and V_heads
        d_V_heads = np.einsum('bhls,bhld->bhsd', attention_weights.transpose(0, 1, 3, 2), d_attention_output_heads)
        d_attention_weights = np.einsum('bhld,bhds->bhls', d_attention_output_heads, V_heads.transpose(0, 1, 3, 2))

        # Backward pass for dropout after softmax
        if training and self.dropout_rate > 0:
            d_attention_weights = d_attention_weights * ((np.random.rand(*d_attention_weights.shape) > self.dropout_rate) / (1 - self.dropout_rate))

        # Correct softmax backward
        d_attention_scores = attention_weights * (d_attention_weights - np.sum(d_attention_weights * attention_weights, axis=-1, keepdims=True))

        if mask is not None:
            # Ensure mask broadcasting works correctly with d_attention_scores shape
            mask = np.broadcast_to(mask, d_attention_scores.shape)
            d_attention_scores[mask == -np.inf] = 0

        d_attention_scores = d_attention_scores / math.sqrt(self.head_dim)

        # Gradients for Q_heads and K_heads
        d_Q_heads = np.einsum('bhls,bhsd->bhld', d_attention_scores, K_heads)
        d_K_heads = np.einsum('bhls,bhld->bhsd', d_attention_scores.transpose(0, 1, 3, 2), Q_heads)

        # Combine heads for Q, K, V gradients
        d_Q = self._combine_heads(d_Q_heads, batch_size, seq_len)
        d_K = self._combine_heads(d_K_heads, batch_size, seq_len)
        d_V = self._combine_heads(d_V_heads, batch_size, seq_len)

        # Gradients for linear projections W_q, b_q, W_k, b_k, W_v, b_v
        self.grads['W_q'] = np.einsum('bsd,bse->de', x, d_Q)
        self.grads['b_q'] = np.sum(d_Q, axis=(0, 1))
        self.grads['W_k'] = np.einsum('bsd,bse->de', x, d_K)
        self.grads['b_k'] = np.sum(d_K, axis=(0, 1))
        self.grads['W_v'] = np.einsum('bsd,bse->de', x, d_V)
        self.grads['b_v'] = np.sum(d_V, axis=(0, 1))

        # Gradient with respect to the input x
        d_x = d_Q @ self.W_q.T + d_K @ self.W_k.T + d_V @ self.W_v.T
        return d_x


class FeedForward:
    """
    Position-wise Feed-Forward Network.
    """

    def __init__(self, embed_dim, ff_dim, dropout_rate=0.0):
        self.W1 = xavier_init(embed_dim, ff_dim)
        self.b1 = np.zeros(ff_dim)
        self.W2 = xavier_init(ff_dim, embed_dim)
        self.b2 = np.zeros(embed_dim)
        self.dropout_rate = dropout_rate

        self.params = {
            'W1': self.W1, 'b1': self.b1,
            'W2': self.W2, 'b2': self.b2,
        }
        self.grads = {k: np.zeros_like(v) for k, v in self.params.items()}

    def forward(self, x, training=True):
        """
        Forward pass for Feed-Forward Network.
        x: (batch_size, seq_len, embed_dim)
        training: boolean, if True, apply dropout.
        Returns: (batch_size, seq_len, embed_dim)
        """
        self.x = x  # Save for backward
        self.Z1 = x @ self.W1 + self.b1
        self.A1 = relu(self.Z1)
        self.A1_dropped = dropout(self.A1, self.dropout_rate, training) # Dropout after ReLU
        output = self.A1_dropped @ self.W2 + self.b2
        self.cache_training = training # Save training state for backward
        return output

    def backward(self, d_output):
        """
        Backward pass for Feed-Forward Network.
        d_output: (batch_size, seq_len, embed_dim) - gradient from the next layer.
        Returns: (batch_size, seq_len, embed_dim) - gradient with respect to the input x.
        """
        # d_output: (batch_size, seq_len, embed_dim)
        batch_size, seq_len, _ = d_output.shape

        # Gradients for W2, b2
        self.grads['W2'] = np.einsum('bsf,bse->fe', self.A1_dropped, d_output)
        self.grads['b2'] = np.sum(d_output, axis=(0, 1))
        d_A1_dropped = d_output @ self.W2.T

        # Backward pass for dropout after ReLU
        if self.cache_training and self.dropout_rate > 0:
            d_A1_dropped = d_A1_dropped * ((np.random.rand(*d_A1_dropped.shape) > self.dropout_rate) / (1 - self.dropout_rate))

        # Gradients for A1 (ReLU)
        d_Z1 = relu_backward(d_A1_dropped, self.Z1)

        # Gradients for W1, b1
        self.grads['W1'] = np.einsum('bsd,bsf->df', self.x, d_Z1)
        self.grads['b1'] = np.sum(d_Z1, axis=(0, 1))
        d_x = d_Z1 @ self.W1.T
        return d_x


class TransformerBlock:
    """
    A single Transformer block consisting of Multi-Head Self-Attention and a Feed-Forward Network.
    Includes Layer Normalization and Residual Connections.
    """

    def __init__(self, embed_dim, num_heads, ff_dim, dropout_rate=0.0):
        self.attention = MultiHeadSelfAttention(embed_dim, num_heads, dropout_rate)
        self.feed_forward = FeedForward(embed_dim, ff_dim, dropout_rate)

        # Layer normalization parameters
        self.norm1_gamma = np.ones(embed_dim)
        self.norm1_beta = np.zeros(embed_dim)
        self.norm2_gamma = np.ones(embed_dim)
        self.norm2_beta = np.zeros(embed_dim)

        self.params = {
            'norm1_gamma': self.norm1_gamma, 'norm1_beta': self.norm1_beta,
            'norm2_gamma': self.norm2_gamma, 'norm2_beta': self.norm2_beta,
        }
        self.grads = {k: np.zeros_like(v) for k, v in self.params.items()}

    def forward(self, x, mask=None, training=True):
        """
        Forward pass for a Transformer block.
        x: (batch_size, seq_len, embed_dim)
        mask: (1, 1, seq_len, seq_len) or None.
        training: boolean, if True, apply dropout.
        Returns: (batch_size, seq_len, embed_dim)
        """
        # Layer norm 1
        norm1_out, self.norm1_mean, self.norm1_variance, self.norm1_x_normalized = layer_norm(x, self.norm1_gamma,
                                                                                              self.norm1_beta)

        # Attention
        attn_output = self.attention.forward(norm1_out, mask, training)

        # Residual connection 1
        attn_output_residual = x + attn_output

        # Layer norm 2
        norm2_out, self.norm2_mean, self.norm2_variance, self.norm2_x_normalized = layer_norm(attn_output_residual,
                                                                                              self.norm2_gamma,
                                                                                              self.norm2_beta)

        # Feed forward
        ff_output = self.feed_forward.forward(norm2_out, training)

        # Residual connection 2
        output = attn_output_residual + ff_output

        self.cache = (x, attn_output_residual)  # For residual connections
        return output

    def backward(self, d_output):
        """
        Backward pass for a Transformer block.
        d_output: (batch_size, seq_len, embed_dim) - gradient from the next layer.
        Returns: (batch_size, seq_len, embed_dim) - gradient with respect to the input x.
        """
        x, attn_output_residual_input = self.cache

        # Gradients for residual connection 2
        d_ff_output = d_output
        d_attn_output_residual_from_ff = d_output

        # Backward through Feed Forward
        d_norm2_out = self.feed_forward.backward(d_ff_output)

        # Backward through Layer norm 2
        self.grads['norm2_gamma'] = np.sum(d_norm2_out * self.norm2_x_normalized, axis=(0, 1))
        self.grads['norm2_beta'] = np.sum(d_norm2_out, axis=(0, 1))

        d_norm2_x_normalized = d_norm2_out * self.norm2_gamma

        N = attn_output_residual_input.shape[-1]  # embed_dim
        d_var = np.sum(d_norm2_x_normalized * (attn_output_residual_input - self.norm2_mean) * -0.5 * np.power(
            self.norm2_variance + 1e-5, -1.5), axis=-1, keepdims=True)
        d_mean = np.sum(d_norm2_x_normalized * (-1 / np.sqrt(self.norm2_variance + 1e-5)), axis=-1,
                        keepdims=True) + d_var * np.sum(-2 * (attn_output_residual_input - self.norm2_mean) / N,
                                                        axis=-1, keepdims=True)

        d_attn_output_residual_input_from_norm2 = d_norm2_x_normalized / np.sqrt(
            self.norm2_variance + 1e-5) + d_var * 2 * (attn_output_residual_input - self.norm2_mean) / N + d_mean / N

        # Gradients for residual connection 1
        d_attn_output = d_attn_output_residual_from_ff + d_attn_output_residual_input_from_norm2
        d_x_from_attn_residual = d_attn_output

        # Backward through Attention
        d_norm1_out = self.attention.backward(d_attn_output)

        # Backward through Layer norm 1
        self.grads['norm1_gamma'] = np.sum(d_norm1_out * self.norm1_x_normalized, axis=(0, 1))
        self.grads['norm1_beta'] = np.sum(d_norm1_out, axis=(0, 1))

        d_norm1_x_normalized = d_norm1_out * self.norm1_gamma

        N = x.shape[-1]  # embed_dim
        d_var = np.sum(d_norm1_x_normalized * (x - self.norm1_mean) * -0.5 * np.power(self.norm1_variance + 1e-5, -1.5),
                       axis=-1, keepdims=True)
        d_mean = np.sum(d_norm1_x_normalized * (-1 / np.sqrt(self.norm1_variance + 1e-5)), axis=-1,
                        keepdims=True) + d_var * np.sum(-2 * (x - self.norm1_mean) / N, axis=-1, keepdims=True)

        d_x_from_norm1 = d_norm1_x_normalized / np.sqrt(self.norm1_variance + 1e-5) + d_var * 2 * (
                    x - self.norm1_mean) / N + d_mean / N

        d_x = d_x_from_attn_residual + d_x_from_norm1
        return d_x

def _top_k_top_p_sampling(logits, vocab_size, top_k=0, top_p=1.0, temperature=1.0):
    """
    Applies top-k and top-p sampling to logits.
    logits: (vocab_size,) - raw logits for the next token.
    vocab_size: int - total number of possible classes.
    top_k: int - if > 0, only consider the top_k most likely tokens.
    top_p: float - if < 1.0, only consider tokens whose cumulative probability
                   exceeds top_p.
    temperature: float - controls randomness.
    Returns: int - sampled token ID.
    """
    # Handle very low temperature for deterministic greedy behavior
    if temperature <= 0.01: # A very small epsilon
        return np.argmax(logits)

    # Apply temperature
    logits = logits / temperature

    # Cap top_k to the size of the logits array (vocab_size)
    _top_k = min(top_k, vocab_size) if top_k > 0 else 0

    # Top-k filtering
    if _top_k > 0:
        # Get the _top_k values
        top_k_values = np.partition(logits, -_top_k)[-_top_k:]
        min_val_for_top_k = top_k_values[0] if len(top_k_values) > 0 else -np.inf
        logits[logits < min_val_for_top_k] = -np.inf

    probabilities = softmax(logits[np.newaxis, :])[0]

    # Check for NaN in probabilities
    if np.any(np.isnan(probabilities)):
        print("WARNING: NaN detected in probabilities during sampling. Falling back to greedy (argmax).")
        # Fallback to greedy (argmax)
        return np.argmax(logits)

    # Top-p filtering
    if top_p < 1.0:
        sorted_indices = np.argsort(probabilities)[::-1]
        sorted_probabilities = probabilities[sorted_indices]
        cumulative_probabilities = np.cumsum(sorted_probabilities)

        # Find the first index where cumulative probability > top_p
        cutoff_idx = np.where(cumulative_probabilities > top_p)[0]
        
        if len(cutoff_idx) > 0:
            cutoff_idx = cutoff_idx[0]
            # Ensure at least one token is kept
            if cutoff_idx == 0 and len(sorted_probabilities) > 1:
                cutoff_idx = 1 # Keep at least the top 1 token
            elif cutoff_idx == 0 and len(sorted_probabilities) == 1:
                pass # Keep the only token
            else:
                # If the first token already exceeds top_p, keep only that one.
                # Otherwise, keep tokens up to the cutoff.
                if cumulative_probabilities[0] > top_p:
                    cutoff_idx = 1
            
            indices_to_remove = sorted_indices[cutoff_idx:]
            probabilities[indices_to_remove] = 0.0
            
            # Re-normalize probabilities
            if np.sum(probabilities) > 0:
                probabilities = probabilities / np.sum(probabilities)
            # else: Fallback handled below

    # Fallback if all probabilities are zero after filtering
    if np.sum(probabilities) == 0:
        # Revert to original (temperature-scaled) logits for fallback
        # Get top 10 indices from the (temperature-scaled) logits
        top_10_indices = np.argsort(logits)[::-1][:min(10, vocab_size)]
        
        if len(top_10_indices) > 0:
            return np.random.choice(top_10_indices)
        else:
            # This case should ideally not happen if vocab_size > 0
            return np.random.randint(0, vocab_size) # Random valid token

    return np.random.choice(vocab_size, p=probabilities)


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

        # Generation parameters
        self.temperature = config['generation']['temperature']
        self.repetition_penalty = config['generation']['repetition_penalty']
        self.top_k = config['generation']['top_k']
        self.top_p = config['generation']['top_p']
        self.beam_width = config['generation']['beam_width']
        self.max_new_tokens = config['generation']['max_new_tokens']


        self.token_embedding = xavier_init(self.vocab_size, self.embed_dim)
        self.positional_encoding = PositionalEncoding(self.embed_dim, self.max_seq_len)
        self.transformer_blocks = [
            TransformerBlock(self.embed_dim, self.num_heads, self.ff_dim, self.dropout_rate) for _ in range(self.num_layers)
        ]
        self.output_layer = xavier_init(self.embed_dim, self.vocab_size)
        self.output_bias = np.zeros(self.vocab_size)

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

    def update_vocab_size(self, new_vocab_size):
        """
        Dynamically updates the vocabulary size of the transformer.
        Expands token embedding and output layer weights/biases with new random values.
        Preserves existing weights.
        """
        if new_vocab_size > self.vocab_size:
            # print(f"Updating vocab size from {self.vocab_size} to {new_vocab_size}")

            # Expand token embedding matrix
            new_embeddings = xavier_init(new_vocab_size - self.vocab_size, self.embed_dim)
            self.token_embedding = np.vstack((self.token_embedding, new_embeddings))
            self.params['token_embedding'] = self.token_embedding  # Update reference
            self.grads['token_embedding'] = np.zeros_like(self.token_embedding)  # Reset grads for new shape

            # Expand output layer weights
            new_output_weights = xavier_init(self.embed_dim, new_vocab_size - self.vocab_size)
            self.output_layer = np.hstack((self.output_layer, new_output_weights))
            self.params['output_layer'] = self.output_layer  # Update reference
            self.grads['output_layer'] = np.zeros_like(self.output_layer)  # Reset grads for new shape

            # Expand output bias
            new_output_bias = np.zeros(new_vocab_size - self.vocab_size)
            self.output_bias = np.hstack((self.output_bias, new_output_bias))
            self.params['output_bias'] = self.output_bias  # Update reference
            self.grads['output_bias'] = np.zeros_like(self.output_bias)  # Reset grads for new shape

            self.vocab_size = new_vocab_size

    def _create_look_ahead_mask(self, seq_len):
        """
        Creates a look-ahead mask to prevent attention to future tokens.
        Returns: (1, 1, seq_len, seq_len)
        """
        mask = np.triu(np.ones((seq_len, seq_len)), k=1).astype('float32')
        mask = (mask * -1e9)  # Convert to additive mask (large negative value)
        return mask[np.newaxis, np.newaxis, :, :]  # (1, 1, seq_len, seq_len)

    def forward(self, x, training=True):
        """
        Forward pass for the Transformer model.
        x: (batch_size, seq_len) - token IDs
        training: bool, if True, applies look-ahead mask and dropout.
        Returns: (batch_size, seq_len, vocab_size) - logits for each token.
        """
        batch_size, seq_len = x.shape

        # Store token IDs for backward pass (for embedding gradients)
        self.cache_x_token_ids = x

        # Token and positional embeddings
        embeddings = self.token_embedding[x]  # (batch_size, seq_len, embed_dim)
        x = self.positional_encoding.forward(embeddings)

        # Create look-ahead mask for decoder if training
        mask = self._create_look_ahead_mask(seq_len) if training else None
        self.cache_mask = mask  # Cache the mask for backward pass in blocks

        # Transformer blocks
        for block in self.transformer_blocks:
            x = block.forward(x, mask, training)

        # Store output of last block for backward pass (input to final linear layer)
        self.cache_x_after_blocks = x

        # Final linear layer
        output_logits = x @ self.output_layer + self.output_bias  # (batch_size, seq_len, vocab_size)
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
        d_x_from_output = d_output_logits @ self.output_layer.T

        # Backward through transformer blocks
        d_x = d_x_from_output
        for block in reversed(self.transformer_blocks):
            d_x = block.backward(d_x)

        # Gradients for token embeddings
        d_token_embedding = np.zeros_like(self.token_embedding)
        for i in range(batch_size):
            for j in range(seq_len):
                d_token_embedding[self.cache_x_token_ids[i, j]] += d_x[i, j]
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
        self.update_weights(learning_rate)
        return loss

    def update_weights(self, learning_rate):
        """
        Updates all model parameters using SGD with the computed gradients.
        Applies gradient clipping.
        """
        all_grads = {}
        # Collect all gradients into a single dictionary
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
        
        # Apply gradient clipping
        clip_gradients(all_grads, max_norm=5.0)

        # Update weights using clipped gradients
        for k in ['token_embedding', 'output_layer', 'output_bias']:
            if k in self.params and k in all_grads:
                self.params[k] -= learning_rate * all_grads[k]

        for i, block in enumerate(self.transformer_blocks):
            for k in ['norm1_gamma', 'norm1_beta', 'norm2_gamma', 'norm2_beta']:
                if k in block.params and f'block_{i}_{k}' in all_grads:
                    block.params[k] -= learning_rate * all_grads[f'block_{i}_{k}']

            for k in block.attention.params.keys():
                if k in block.attention.params and f'block_{i}_attn_{k}' in all_grads:
                    block.attention.params[k] -= learning_rate * all_grads[f'block_{i}_attn_{k}']

            for k in block.feed_forward.params.keys():
                if k in block.feed_forward.params and f'block_{i}_ff_{k}' in all_grads:
                    block.feed_forward.params[k] -= learning_rate * all_grads[f'block_{i}_ff_{k}']

    def _beam_search_generate(self, prompt_tokens, pad_token_id, eos_token_id):
        """
        Generates tokens using beam search.
        """
        # Beams are (log_probability, sequence_of_token_ids)
        beams = [(0.0, list(prompt_tokens[0]))]
        
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
                
                # Check for NaN in probabilities
                if np.any(np.isnan(probabilities)):
                    print("WARNING: NaN detected in probabilities during beam search sampling. Falling back to greedy (argmax).")
                    next_token_id = np.argmax(last_token_logits) # Use raw logits for greedy
                    new_log_prob = log_prob + np.log(probabilities[next_token_id]) if probabilities[next_token_id] > 0 else -np.inf
                    new_sequence = current_sequence + [next_token_id]
                    all_candidates.append((new_log_prob, new_sequence))
                    continue # Skip other candidates for this beam if NaN

                # Get top beam_width candidates for the next token
                # Ensure indices are within vocab_size, which they should be if last_token_logits is vocab_size long
                top_token_indices = np.argsort(probabilities)[::-1][:self.beam_width]

                for next_token_id in top_token_indices:
                    new_log_prob = log_prob + np.log(probabilities[next_token_id])
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
        if self.beam_width > 0:
            return self._beam_search_generate(prompt_tokens, pad_token_id, eos_token_id)

        generated_tokens = list(prompt_tokens[0])
        batch_size = 1  # Generation is always batch size 1

        for _ in range(self.max_new_tokens):
            current_seq_len = len(generated_tokens)
            if current_seq_len == 0:
                break

            # Truncate if sequence exceeds max_seq_len
            input_seq = generated_tokens[-self.max_seq_len:]
            input_tensor = np.array(input_seq).reshape(batch_size, -1)

            # Forward pass (no training mask)
            output_logits = self.forward(input_tensor, training=False)

            # Get logits for the last token in the sequence
            last_token_logits = output_logits[0, -1, :]

            if self.repetition_penalty > 1.0:
                # Apply repetition penalty
                unique_current_tokens = np.unique([t for t in generated_tokens if t != pad_token_id])
                for token_id in unique_current_tokens:
                    if 0 <= token_id < self.vocab_size:
                        last_token_logits[token_id] /= self.repetition_penalty

            # Sample next token using top-k and top-p
            next_token = _top_k_top_p_sampling(last_token_logits, self.vocab_size, self.top_k, self.top_p, self.temperature)
            
            generated_tokens.append(next_token)

            # Stop if EOS token is generated
            if next_token == eos_token_id:
                break

        return np.array(generated_tokens)