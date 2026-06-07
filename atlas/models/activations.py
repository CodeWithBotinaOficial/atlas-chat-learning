import numpy as np

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
