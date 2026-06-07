import numpy as np
from atlas.models.activations import softmax

def _top_k_top_p_sampling(logits, vocab_size, top_k=0, top_p=1.0, temperature=1.0, special_token_ids=None):
    """
    Applies top-k and top-p sampling to logits.
    logits: (vocab_size,) - raw logits for the next token.
    vocab_size: int - total number of possible classes.
    top_k: int - if > 0, only consider the top_k most likely tokens.
    top_p: float - if < 1.0, only consider tokens whose cumulative probability
                   exceeds top_p.
    temperature: float - controls randomness.
    special_token_ids: list or set of int - IDs of special tokens to potentially exclude from random fallback.
    Returns: int - sampled token ID.
    """
    if special_token_ids is None:
        special_token_ids = {0, 1, 2, 3} # Default: PAD, UNK, BOS, EOS

    # Check for degenerate logits (all -inf, NaN, or effectively zero)
    if np.all(np.isinf(logits)) or np.all(np.isnan(logits)) or np.all(logits <= -1e9): # Check for very small values too
        print("WARNING: All logits are degenerate (NaN, Inf, or very small). Falling back to random non-special token.")
        
        # Try to find a non-special token
        non_special_indices = [i for i in range(vocab_size) if i not in special_token_ids]
        if len(non_special_indices) > 0:
            return np.random.choice(non_special_indices)
        else:
            # If only special tokens exist or vocab_size is very small, pick any token
            return np.random.randint(0, vocab_size)

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

    # Check for NaN or Inf in probabilities
    if np.any(np.isnan(probabilities)) or np.any(np.isinf(probabilities)):
        print("WARNING: NaN or Inf detected in probabilities during sampling after softmax.")
        # Fallback strategy: find valid indices in logits
        valid_indices = np.where(~np.isnan(logits) & ~np.isinf(logits))[0]
        if len(valid_indices) > 0:
            # Choose randomly from the top 10 valid indices based on logits, excluding special tokens if possible
            top_valid_indices = valid_indices[np.argsort(logits[valid_indices])][::-1]
            
            # Filter out special tokens from top_valid_indices for a more meaningful fallback
            filtered_top_valid_indices = [idx for idx in top_valid_indices if idx not in special_token_ids]
            
            if len(filtered_top_valid_indices) > 0:
                final_choice_indices = filtered_top_valid_indices[:min(10, len(filtered_top_valid_indices))]
                print(f"Falling back to random choice from top {len(final_choice_indices)} valid non-special logits.")
                return np.random.choice(final_choice_indices)
            elif len(top_valid_indices) > 0: # If only special tokens are valid, pick from them
                final_choice_indices = top_valid_indices[:min(10, len(top_valid_indices))]
                print(f"Falling back to random choice from top {len(final_choice_indices)} valid (possibly special) logits.")
                return np.random.choice(final_choice_indices)
            else:
                # If no valid indices at all, choose a random non-special token from the entire vocab
                print("Falling back to random non-special token from vocabulary.")
                non_special_indices = [i for i in range(vocab_size) if i not in special_token_ids]
                if len(non_special_indices) > 0:
                    return np.random.choice(non_special_indices)
                else:
                    return np.random.randint(0, vocab_size) # If only special tokens exist
        else:
            # If all logits are NaN/Inf, choose a random non-special token from the entire vocab
            print("WARNING: All logits are NaN/Inf. Falling back to random non-special token from vocabulary.")
            non_special_indices = [i for i in range(vocab_size) if i not in special_token_ids]
            if len(non_special_indices) > 0:
                return np.random.choice(non_special_indices)
            else:
                return np.random.randint(0, vocab_size) # If only special tokens exist


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
        print("WARNING: All probabilities became zero after filtering. Falling back to random non-special token.")
        # Revert to original (temperature-scaled) logits for fallback
        # Get top 10 indices from the (temperature-scaled) logits
        top_10_indices = np.argsort(logits)[::-1]
        
        # Filter out special tokens from top_10_indices for a more meaningful fallback
        filtered_top_10_indices = [idx for idx in top_10_indices if idx not in special_token_ids]

        if len(filtered_top_10_indices) > 0:
            return np.random.choice(filtered_top_10_indices[:min(10, len(filtered_top_10_indices))])
        elif len(top_10_indices) > 0: # If only special tokens are valid, pick from them
            return np.random.choice(top_10_indices[:min(10, len(top_10_indices))])
        else:
            # This case should ideally not happen if vocab_size > 0
            non_special_indices = [i for i in range(vocab_size) if i not in special_token_ids]
            if len(non_special_indices) > 0:
                return np.random.choice(non_special_indices)
            else:
                return np.random.randint(0, vocab_size) # If only special tokens exist

    return np.random.choice(vocab_size, p=probabilities)
