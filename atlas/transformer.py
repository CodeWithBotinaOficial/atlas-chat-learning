# atlas/transformer.py
import numpy as np
import math


# Helper functions
def softmax(x):
    """Compute softmax values for each row of x."""
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)


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


def cross_entropy_loss(predictions, targets, epsilon=1e-9):
    """
    Compute cross-entropy loss.
    predictions: (batch_size * seq_len, vocab_size)
    targets: (batch_size * seq_len,)
    """
    # Clip predictions to avoid log(0)
    predictions = np.clip(predictions, epsilon, 1. - epsilon)

    # One-hot encode targets
    num_classes = predictions.shape[-1]
    targets_one_hot = np.eye(num_classes)[targets]

    loss = -np.sum(targets_one_hot * np.log(predictions)) / predictions.shape[0]
    return loss


def cross_entropy_backward(predictions, targets):
    """
    Compute gradient of cross-entropy loss with respect to predictions.
    predictions: (batch_size * seq_len, vocab_size) - softmax probabilities
    targets: (batch_size * seq_len,) - integer class labels
    Returns: (batch_size * seq_len, vocab_size)
    """
    num_classes = predictions.shape[-1]
    targets_one_hot = np.eye(num_classes)[targets]
    grad = predictions - targets_one_hot
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

    def __init__(self, embed_dim, num_heads):
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.head_dim = embed_dim // num_heads
        self.num_heads = num_heads
        self.embed_dim = embed_dim

        # Weights for Q, K, V for all heads
        self.W_q = np.random.randn(embed_dim, embed_dim) * 0.01
        self.b_q = np.zeros(embed_dim)
        self.W_k = np.random.randn(embed_dim, embed_dim) * 0.01
        self.b_k = np.zeros(embed_dim)
        self.W_v = np.random.randn(embed_dim, embed_dim) * 0.01
        self.b_v = np.zeros(embed_dim)

        # Output projection
        self.W_o = np.random.randn(embed_dim, embed_dim) * 0.01
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

    def forward(self, x, mask=None):
        """
        Forward pass for Multi-Head Self-Attention.
        x: (batch_size, seq_len, embed_dim)
        mask: (1, 1, seq_len, seq_len) or None. Additive mask for attention scores.
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
        # (batch_size, num_heads, seq_len, head_dim) @ (batch_size, num_heads, head_dim, seq_len)
        attention_scores = Q_heads @ K_heads.transpose(0, 1, 3, 2)
        attention_scores = attention_scores / math.sqrt(self.head_dim)

        if mask is not None:
            attention_scores = attention_scores + mask  # Additive mask

        attention_weights = softmax(attention_scores)

        # (batch_size, num_heads, seq_len, seq_len) @ (batch_size, num_heads, seq_len, head_dim)
        attention_output_heads = attention_weights @ V_heads

        # Combine heads
        attention_output = self._combine_heads(attention_output_heads, batch_size, seq_len)

        # Output projection
        output = attention_output @ self.W_o + self.b_o

        # Save for backward pass
        self.cache = (x, Q, K, V, Q_heads, K_heads, V_heads, attention_scores, attention_weights,
                      attention_output_heads, attention_output, mask)
        return output

    def backward(self, d_output):
        """
        Backward pass for Multi-Head Self-Attention.
        d_output: (batch_size, seq_len, embed_dim) - gradient from the next layer.
        Returns: (batch_size, seq_len, embed_dim) - gradient with respect to the input x.
        """
        x, Q, K, V, Q_heads, K_heads, V_heads, attention_scores, attention_weights, attention_output_heads, attention_output, mask = self.cache
        batch_size, seq_len, _ = x.shape

        # Gradients for output projection
        self.grads['W_o'] = np.einsum('bsd,bse->de', attention_output, d_output)  # (embed_dim, embed_dim)
        self.grads['b_o'] = np.sum(d_output, axis=(0, 1))  # (embed_dim,)
        d_attention_output = d_output @ self.W_o.T  # (batch_size, seq_len, embed_dim)

        # Gradients for combining heads
        d_attention_output_heads = self._split_heads(d_attention_output, batch_size,
                                                     seq_len)  # (batch_size, num_heads, seq_len, head_dim)

        # Gradients for attention_weights and V_heads
        # dL/dV_heads = attention_weights.T @ d_attention_output_heads
        # dL/d_attention_weights = d_attention_output_heads @ V_heads.T
        d_V_heads = np.einsum('bhls,bhld->bhsd', attention_weights.transpose(0, 1, 3, 2),
                              d_attention_output_heads)  # (batch_size, num_heads, seq_len, head_dim)
        d_attention_weights = np.einsum('bhld,bhsd->bhls', d_attention_output_heads,
                                        V_heads.transpose(0, 1, 3, 2))  # (batch_size, num_heads, seq_len, seq_len)

        # Correct softmax backward
        # dL/d_attention_scores = attention_weights * (d_attention_weights - sum(d_attention_weights * attention_weights, axis=-1, keepdims=True))
        d_attention_scores = attention_weights * (
                    d_attention_weights - np.sum(d_attention_weights * attention_weights, axis=-1, keepdims=True))

        if mask is not None:
            # Mask out gradients where attention scores were masked
            d_attention_scores[mask == -np.inf] = 0

        d_attention_scores = d_attention_scores / math.sqrt(self.head_dim)

        # Gradients for Q_heads and K_heads
        # dL/dQ_heads = d_attention_scores @ K_heads.T
        # dL/dK_heads = d_attention_scores.T @ Q_heads
        d_Q_heads = np.einsum('bhls,bhsd->bhld', d_attention_scores,
                              K_heads)  # (batch_size, num_heads, seq_len, head_dim)
        d_K_heads = np.einsum('bhls,bhld->bhsd', d_attention_scores.transpose(0, 1, 3, 2),
                              Q_heads)  # (batch_size, num_heads, seq_len, head_dim)

        # Combine heads for Q, K, V gradients
        d_Q = self._combine_heads(d_Q_heads, batch_size, seq_len)  # (batch_size, seq_len, embed_dim)
        d_K = self._combine_heads(d_K_heads, batch_size, seq_len)  # (batch_size, seq_len, embed_dim)
        d_V = self._combine_heads(d_V_heads, batch_size, seq_len)  # (batch_size, seq_len, embed_dim)

        # Gradients for linear projections W_q, b_q, W_k, b_k, W_v, b_v
        self.grads['W_q'] = np.einsum('bsd,bse->de', x, d_Q)  # (embed_dim, embed_dim)
        self.grads['b_q'] = np.sum(d_Q, axis=(0, 1))  # (embed_dim,)
        self.grads['W_k'] = np.einsum('bsd,bse->de', x, d_K)  # (embed_dim, embed_dim)
        self.grads['b_k'] = np.sum(d_K, axis=(0, 1))  # (embed_dim,)
        self.grads['W_v'] = np.einsum('bsd,bse->de', x, d_V)  # (embed_dim, embed_dim)
        self.grads['b_v'] = np.sum(d_V, axis=(0, 1))  # (embed_dim,)

        # Gradient with respect to the input x
        d_x = d_Q @ self.W_q.T + d_K @ self.W_k.T + d_V @ self.W_v.T
        return d_x


class FeedForward:
    """
    Position-wise Feed-Forward Network.
    """

    def __init__(self, embed_dim, ff_dim):
        self.W1 = np.random.randn(embed_dim, ff_dim) * 0.01
        self.b1 = np.zeros(ff_dim)
        self.W2 = np.random.randn(ff_dim, embed_dim) * 0.01
        self.b2 = np.zeros(embed_dim)

        self.params = {
            'W1': self.W1, 'b1': self.b1,
            'W2': self.W2, 'b2': self.b2,
        }
        self.grads = {k: np.zeros_like(v) for k, v in self.params.items()}

    def forward(self, x):
        """
        Forward pass for Feed-Forward Network.
        x: (batch_size, seq_len, embed_dim)
        Returns: (batch_size, seq_len, embed_dim)
        """
        self.x = x  # Save for backward
        self.Z1 = x @ self.W1 + self.b1
        self.A1 = relu(self.Z1)
        output = self.A1 @ self.W2 + self.b2
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
        self.grads['W2'] = np.einsum('bsf,bse->fe', self.A1, d_output)  # (ff_dim, embed_dim)
        self.grads['b2'] = np.sum(d_output, axis=(0, 1))  # (embed_dim,)
        d_A1 = d_output @ self.W2.T  # (batch_size, seq_len, ff_dim)

        # Gradients for A1 (ReLU)
        d_Z1 = relu_backward(d_A1, self.Z1)  # (batch_size, seq_len, ff_dim)

        # Gradients for W1, b1
        self.grads['W1'] = np.einsum('bsd,bsf->df', self.x, d_Z1)  # (embed_dim, ff_dim)
        self.grads['b1'] = np.sum(d_Z1, axis=(0, 1))  # (ff_dim,)
        d_x = d_Z1 @ self.W1.T  # (batch_size, seq_len, embed_dim)
        return d_x


class TransformerBlock:
    """
    A single Transformer block consisting of Multi-Head Self-Attention and a Feed-Forward Network.
    Includes Layer Normalization and Residual Connections.
    """

    def __init__(self, embed_dim, num_heads, ff_dim):
        self.attention = MultiHeadSelfAttention(embed_dim, num_heads)
        self.feed_forward = FeedForward(embed_dim, ff_dim)

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

    def forward(self, x, mask=None):
        """
        Forward pass for a Transformer block.
        x: (batch_size, seq_len, embed_dim)
        mask: (1, 1, seq_len, seq_len) or None.
        Returns: (batch_size, seq_len, embed_dim)
        """
        # Layer norm 1
        norm1_out, self.norm1_mean, self.norm1_variance, self.norm1_x_normalized = layer_norm(x, self.norm1_gamma,
                                                                                              self.norm1_beta)

        # Attention
        attn_output = self.attention.forward(norm1_out, mask)

        # Residual connection 1
        attn_output_residual = x + attn_output

        # Layer norm 2
        norm2_out, self.norm2_mean, self.norm2_variance, self.norm2_x_normalized = layer_norm(attn_output_residual,
                                                                                              self.norm2_gamma,
                                                                                              self.norm2_beta)

        # Feed forward
        ff_output = self.feed_forward.forward(norm2_out)

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
        # dL/dgamma = sum(dL/d_norm_out * x_normalized)
        # dL/dbeta = sum(dL/d_norm_out)
        self.grads['norm2_gamma'] = np.sum(d_norm2_out * self.norm2_x_normalized, axis=(0, 1))
        self.grads['norm2_beta'] = np.sum(d_norm2_out, axis=(0, 1))

        # dL/dx_normalized = dL/d_norm_out * gamma
        d_norm2_x_normalized = d_norm2_out * self.norm2_gamma

        # dL/d_variance and dL/d_mean calculations for LayerNorm
        # These are derived from the chain rule for LayerNorm
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


class Transformer:
    """
    A small Transformer model for sequence-to-sequence tasks.
    """

    def __init__(self, vocab_size, embed_dim, num_heads, ff_dim, num_layers, max_seq_len):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len

        self.token_embedding = np.random.randn(vocab_size, embed_dim) * 0.01
        self.positional_encoding = PositionalEncoding(embed_dim, max_seq_len)
        self.transformer_blocks = [
            TransformerBlock(embed_dim, num_heads, ff_dim) for _ in range(num_layers)
        ]
        self.output_layer = np.random.randn(embed_dim, vocab_size) * 0.01
        self.output_bias = np.zeros(vocab_size)

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
            new_embeddings = np.random.randn(new_vocab_size - self.vocab_size, self.embed_dim) * 0.01
            self.token_embedding = np.vstack((self.token_embedding, new_embeddings))
            self.params['token_embedding'] = self.token_embedding  # Update reference
            self.grads['token_embedding'] = np.zeros_like(self.token_embedding)  # Reset grads for new shape

            # Expand output layer weights
            new_output_weights = np.random.randn(self.embed_dim, new_vocab_size - self.vocab_size) * 0.01
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
        training: bool, if True, applies look-ahead mask.
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
            x = block.forward(x, mask)

        # Store output of last block for backward pass (input to final linear layer)
        self.cache_x_after_blocks = x

        # Final linear layer
        output_logits = x @ self.output_layer + self.output_bias  # (batch_size, seq_len, vocab_size)
        return output_logits

    def backward(self, d_output_logits, learning_rate):
        """
        Backward pass for the Transformer model.
        d_output_logits: (batch_size, seq_len, vocab_size) - gradient from the loss function.
        learning_rate: float - learning rate for SGD.
        """
        batch_size, seq_len, _ = d_output_logits.shape

        # Gradients for output layer
        # dL/dW_output = (input_to_output_layer).T @ dL/d(output_logits)
        # dL/db_output = sum(dL/d(output_logits))
        self.grads['output_layer'] = np.einsum('bsd,bsv->dv', self.cache_x_after_blocks, d_output_logits)
        self.grads['output_bias'] = np.sum(d_output_logits, axis=(0, 1))
        d_x_from_output = d_output_logits @ self.output_layer.T

        # Backward through transformer blocks
        d_x = d_x_from_output
        for block in reversed(self.transformer_blocks):
            d_x = block.backward(d_x)

        # Gradients for token embeddings
        # d_x at this point is dL/d(embeddings_after_pos_enc)
        # Since positional encoding is not trainable, this is also dL/d(token_embeddings)
        d_token_embedding = np.zeros_like(self.token_embedding)
        for i in range(batch_size):
            for j in range(seq_len):
                # Accumulate gradients for each token ID
                d_token_embedding[self.cache_x_token_ids[i, j]] += d_x[i, j]
        self.grads['token_embedding'] = d_token_embedding

        # Update weights
        self.update_weights(learning_rate)

    def update_weights(self, learning_rate):
        """
        Updates all model parameters using SGD with the computed gradients.
        """
        # Update top-level parameters
        for k in ['token_embedding', 'output_layer', 'output_bias']:
            if k in self.params and k in self.grads:
                self.params[k] -= learning_rate * self.grads[k]

        # Update block parameters
        for i, block in enumerate(self.transformer_blocks):
            # Update block's layer norm parameters
            for k in ['norm1_gamma', 'norm1_beta', 'norm2_gamma', 'norm2_beta']:
                if k in block.params and k in block.grads:
                    block.params[k] -= learning_rate * block.grads[k]

            # Update attention parameters
            for k in block.attention.params.keys():
                if k in block.attention.grads:
                    block.attention.params[k] -= learning_rate * block.attention.grads[k]

            # Update feed_forward parameters
            for k in block.feed_forward.params.keys():
                if k in block.feed_forward.grads:
                    block.feed_forward.params[k] -= learning_rate * block.feed_forward.grads[k]

    def train_step(self, input_tokens, target_tokens, learning_rate=0.01):
        """
        Performs one training step (forward, loss, backward, update).
        input_tokens: (batch_size, seq_len) - input token IDs.
        target_tokens: (batch_size, seq_len) - target token IDs (shifted input).
        learning_rate: float.
        Returns: float - computed loss.
        """
        # Forward pass
        output_logits = self.forward(input_tokens, training=True)

        # Reshape for loss calculation: (batch_size * seq_len, vocab_size)
        predictions = softmax(output_logits.reshape(-1, self.vocab_size))
        targets_flat = target_tokens.flatten()

        loss = cross_entropy_loss(predictions, targets_flat)

        # Backward pass
        # d_predictions is dL/d(softmax_output)
        d_predictions = cross_entropy_backward(predictions, targets_flat)
        # d_output_logits is dL/d(logits_before_softmax)
        # This is the gradient that needs to be propagated back through the model
        d_output_logits = d_predictions.reshape(output_logits.shape)

        self.backward(d_output_logits, learning_rate)
        return loss

    def generate(self, prompt_tokens, max_new_tokens=15, temperature=0.8, repetition_penalty=1.0, pad_token_id=0,
                 eos_token_id=3):
        """
        Generates new tokens based on a prompt.
        prompt_tokens: (1, seq_len) - initial token IDs.
        max_new_tokens: int - maximum number of tokens to generate.
        temperature: float - controls randomness in sampling. Higher means more random.
        repetition_penalty: float - penalizes tokens that have already appeared. > 1.0 to penalize.
        pad_token_id: int - ID of the padding token.
        eos_token_id: int - ID of the end-of-sequence token.
        Returns: np.ndarray - array of generated token IDs.
        """
        generated_tokens = list(prompt_tokens[0])
        batch_size = 1  # Generation is always batch size 1

        for _ in range(max_new_tokens):
            current_seq_len = len(generated_tokens)
            if current_seq_len == 0:
                break  # Should not happen if BOS is handled correctly

            # Truncate if sequence exceeds max_seq_len
            input_seq = generated_tokens[-self.max_seq_len:]
            input_tensor = np.array(input_seq).reshape(batch_size, -1)

            # Forward pass (no training mask)
            output_logits = self.forward(input_tensor, training=False)

            # Get logits for the last token in the sequence
            last_token_logits = output_logits[0, -1, :] / temperature

            if repetition_penalty > 1.0:
                # Apply repetition penalty
                # Get tokens in the current sequence (excluding PAD for penalty)
                current_tokens_in_seq = [t for t in input_tensor[0] if t != pad_token_id]
                unique_current_tokens = np.unique(current_tokens_in_seq)

                for token_id in unique_current_tokens:
                    if 0 <= token_id < self.vocab_size:  # Ensure token_id is valid
                        last_token_logits[token_id] /= repetition_penalty

            # Apply softmax to get probabilities
            probabilities = softmax(last_token_logits[np.newaxis, :])[0]

            # Sample next token
            next_token = np.random.choice(self.vocab_size, p=probabilities)
            generated_tokens.append(next_token)

            # Stop if EOS token is generated
            if next_token == eos_token_id:
                break

        return np.array(generated_tokens)
