import numpy as np
import math

class PositionalEncoding:
    """
    Applies sinusoidal positional encoding to input embeddings.
    """

    def __init__(self, embed_dim, max_seq_len, dtype=np.float32):
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        self.pe = self._create_positional_encoding(dtype)

    def _create_positional_encoding(self, dtype):
        pe = np.zeros((self.max_seq_len, self.embed_dim), dtype=dtype)
        position = np.arange(0, self.max_seq_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, self.embed_dim, 2) * -(math.log(10000.0) / self.embed_dim)).astype(dtype)
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        return pe[np.newaxis, :, :]  # Add batch dimension (1, max_seq_len, embed_dim)

    def forward(self, x, start_pos=0):
        """
        Adds positional encoding to the input embeddings.
        x: (batch_size, seq_len, embed_dim)
        start_pos: int - position offset for incremental decoding with KV cache.
        Returns: (batch_size, seq_len, embed_dim)
        """
        seq_len = x.shape[1]
        # Positional encoding is broadcasted across the batch dimension
        return x + self.pe[:, start_pos:start_pos + seq_len, :]
