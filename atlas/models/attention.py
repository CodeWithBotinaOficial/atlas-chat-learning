import numpy as np
import math
from atlas.models.initialization import xavier_init
from atlas.models.activations import softmax, dropout

class MultiHeadSelfAttention:
    """
    Multi-Head Self-Attention mechanism.
    """

    def __init__(self, embed_dim, num_heads, dropout_rate=0.0, dtype=np.float32):
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.head_dim = embed_dim // num_heads
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.dropout_rate = dropout_rate
        self.dtype = dtype

        # Weights for Q, K, V for all heads (Xavier initialization)
        self.W_q = xavier_init(embed_dim, embed_dim, dtype=dtype)
        self.b_q = np.zeros(embed_dim, dtype=dtype)
        self.W_k = xavier_init(embed_dim, embed_dim, dtype=dtype)
        self.b_k = np.zeros(embed_dim, dtype=dtype)
        self.W_v = xavier_init(embed_dim, embed_dim, dtype=dtype)
        self.b_v = np.zeros(embed_dim, dtype=dtype)

        # Output projection (Xavier initialization)
        self.W_o = xavier_init(embed_dim, embed_dim, dtype=dtype)
        self.b_o = np.zeros(embed_dim, dtype=dtype)

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

    def forward(self, x, mask=None, training=True, kv_cache_entry=None):
        """
        Forward pass for Multi-Head Self-Attention.
        x: (batch_size, seq_len, embed_dim)
        mask: (1, 1, seq_len, seq_len) or None.
        training: boolean, if True, apply dropout.
        kv_cache_entry: optional dict with cached 'k' and 'v' arrays for inference.
        Returns: (batch_size, seq_len, embed_dim)
        """
        batch_size, seq_len, _ = x.shape

        # Cast to float32 for computation if using half-precision
        if self.dtype == np.float16:
            x = x.astype(np.float32)
            W_q, b_q = self.W_q.astype(np.float32), self.b_q.astype(np.float32)
            W_k, b_k = self.W_k.astype(np.float32), self.b_k.astype(np.float32)
            W_v, b_v = self.W_v.astype(np.float32), self.b_v.astype(np.float32)
            W_o, b_o = self.W_o.astype(np.float32), self.b_o.astype(np.float32)
        else:
            W_q, b_q = self.W_q, self.b_q
            W_k, b_k = self.W_k, self.b_k
            W_v, b_v = self.W_v, self.b_v
            W_o, b_o = self.W_o, self.b_o

        # Linear projections
        Q = x @ W_q + b_q
        K = x @ W_k + b_k
        V = x @ W_v + b_v

        # Split into multiple heads
        Q_heads = self._split_heads(Q, batch_size, seq_len)
        K_heads_new = self._split_heads(K, batch_size, seq_len)
        V_heads_new = self._split_heads(V, batch_size, seq_len)

        use_kv_cache = kv_cache_entry is not None and not training
        if use_kv_cache and kv_cache_entry.get('k') is not None:
            K_heads = np.concatenate([kv_cache_entry['k'], K_heads_new], axis=2)
            V_heads = np.concatenate([kv_cache_entry['v'], V_heads_new], axis=2)
        else:
            K_heads = K_heads_new
            V_heads = V_heads_new

        if use_kv_cache:
            kv_cache_entry['k'] = K_heads
            kv_cache_entry['v'] = V_heads

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
        output = attention_output @ W_o + b_o
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
