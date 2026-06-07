import numpy as np
from atlas.models.activations import layer_norm
from atlas.models.attention import MultiHeadSelfAttention
from atlas.models.feed_forward import FeedForward

class TransformerBlock:
    """
    A single Transformer block consisting of Multi-Head Self-Attention and a Feed-Forward Network.
    Includes Layer Normalization and Residual Connections.
    """

    def __init__(self, embed_dim, num_heads, ff_dim, dropout_rate=0.0, dtype=np.float32):
        self.dtype = dtype
        self.attention = MultiHeadSelfAttention(embed_dim, num_heads, dropout_rate, dtype=dtype)
        self.feed_forward = FeedForward(embed_dim, ff_dim, dropout_rate, dtype=dtype)

        # Layer normalization parameters
        self.norm1_gamma = np.ones(embed_dim, dtype=dtype)
        self.norm1_beta = np.zeros(embed_dim, dtype=dtype)
        self.norm2_gamma = np.ones(embed_dim, dtype=dtype)
        self.norm2_beta = np.zeros(embed_dim, dtype=dtype)

        self.params = {
            'norm1_gamma': self.norm1_gamma, 'norm1_beta': self.norm1_beta,
            'norm2_gamma': self.norm2_gamma, 'norm2_beta': self.norm2_beta,
        }
        self.grads = {k: np.zeros_like(v) for k, v in self.params.items()}

    def forward(self, x, mask=None, training=True, cache=None, layer_idx=0):
        """
        Forward pass for a Transformer block.
        x: (batch_size, seq_len, embed_dim)
        mask: (1, 1, seq_len, seq_len) or None.
        training: boolean, if True, apply dropout.
        cache: optional KV cache dict keyed by layer index.
        layer_idx: int - index of this block in the transformer stack.
        Returns: (batch_size, seq_len, embed_dim)
        """
        # Cast to float32 for computation if using half-precision
        if self.dtype == np.float16:
            x = x.astype(np.float32)
            norm1_gamma, norm1_beta = self.norm1_gamma.astype(np.float32), self.norm1_beta.astype(np.float32)
            norm2_gamma, norm2_beta = self.norm2_gamma.astype(np.float32), self.norm2_beta.astype(np.float32)
        else:
            norm1_gamma, norm1_beta = self.norm1_gamma, self.norm1_beta
            norm2_gamma, norm2_beta = self.norm2_gamma, self.norm2_beta

        # Layer norm 1
        norm1_out, self.norm1_mean, self.norm1_variance, self.norm1_x_normalized = layer_norm(x, norm1_gamma,
                                                                                              norm1_beta)

        kv_cache_entry = None
        if cache is not None and not training:
            if layer_idx not in cache:
                cache[layer_idx] = {'k': None, 'v': None}
            kv_cache_entry = cache[layer_idx]

        # Attention
        attn_output = self.attention.forward(norm1_out, mask, training, kv_cache_entry=kv_cache_entry)

        # Residual connection 1
        attn_output_residual = x + attn_output

        # Layer norm 2
        norm2_out, self.norm2_mean, self.norm2_variance, self.norm2_x_normalized = layer_norm(attn_output_residual,
                                                                                              norm2_gamma,
                                                                                              norm2_beta)

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
