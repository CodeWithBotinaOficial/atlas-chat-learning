import numpy as np

def xavier_init(input_dim, output_dim, dtype=np.float32):
    """
    Xavier/Glorot uniform initialization for weights.
    """
    limit = np.sqrt(6 / (input_dim + output_dim))
    return np.random.uniform(-limit, limit, size=(input_dim, output_dim)).astype(dtype)

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
