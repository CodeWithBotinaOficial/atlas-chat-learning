import numpy as np

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
    targets: (batch_size * seq_len) - integer class labels
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
