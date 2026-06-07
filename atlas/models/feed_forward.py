import numpy as np
from atlas.models.initialization import xavier_init
from atlas.models.activations import relu, relu_backward, dropout

class FeedForward:
    """
    Position-wise Feed-Forward Network.
    """

    def __init__(self, embed_dim, ff_dim, dropout_rate=0.0, dtype=np.float32):
        self.dtype = dtype
        self.W1 = xavier_init(embed_dim, ff_dim, dtype=dtype)
        self.b1 = np.zeros(ff_dim, dtype=dtype)
        self.W2 = xavier_init(ff_dim, embed_dim, dtype=dtype)
        self.b2 = np.zeros(embed_dim, dtype=dtype)
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

        # Cast to float32 for computation if using half-precision
        if self.dtype == np.float16:
            x = x.astype(np.float32)
            W1, b1 = self.W1.astype(np.float32), self.b1.astype(np.float32)
            W2, b2 = self.W2.astype(np.float32), self.b2.astype(np.float32)
        else:
            W1, b1 = self.W1, self.b1
            W2, b2 = self.W2, self.b2

        self.Z1 = x @ W1 + b1
        self.A1 = relu(self.Z1)
        self.A1_dropped = dropout(self.A1, self.dropout_rate, training) # Dropout after ReLU
        output = self.A1_dropped @ W2 + b2
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
