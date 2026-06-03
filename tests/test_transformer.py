# tests/test_transformer.py
import numpy as np
import pytest
from atlas.transformer import (Transformer, PositionalEncoding, MultiHeadSelfAttention, FeedForward, TransformerBlock,
                               softmax, label_smoothing_loss, label_smoothing_backward, _top_k_top_p_sampling)
from atlas.brain import AtlasBrain  # Import AtlasBrain to get special token IDs


# --- Helper functions for testing ---
def _create_dummy_transformer(vocab_size=10, embed_dim=16, num_heads=2, ff_dim=32, num_layers=2, max_seq_len=50, dropout_rate=0.0):
    return Transformer(vocab_size, embed_dim, num_heads, ff_dim, num_layers, max_seq_len, dropout_rate)


# --- PositionalEncoding Tests ---
def test_positional_encoding_forward_shape():
    embed_dim = 16
    max_seq_len = 50
    pe_layer = PositionalEncoding(embed_dim, max_seq_len)

    batch_size = 2
    seq_len = 10
    dummy_input = np.random.rand(batch_size, seq_len, embed_dim)
    output = pe_layer.forward(dummy_input)

    assert output.shape == (batch_size, seq_len, embed_dim)
    assert not np.allclose(output, dummy_input)  # Should add positional info


# --- MultiHeadSelfAttention Tests ---
def test_mha_forward_shape():
    embed_dim = 16
    num_heads = 2
    mha = MultiHeadSelfAttention(embed_dim, num_heads)

    batch_size = 2
    seq_len = 10
    dummy_input = np.random.rand(batch_size, seq_len, embed_dim)
    output = mha.forward(dummy_input)

    assert output.shape == (batch_size, seq_len, embed_dim)


def test_mha_forward_with_mask_shape():
    embed_dim = 16
    num_heads = 2
    mha = MultiHeadSelfAttention(embed_dim, num_heads)

    batch_size = 2
    seq_len = 10
    dummy_input = np.random.rand(batch_size, seq_len, embed_dim)

    # Create a proper look-ahead mask
    mask = np.triu(np.ones((seq_len, seq_len)), k=1).astype('float32')
    mask = (mask * -1e9)  # Additive mask
    mask = mask[np.newaxis, np.newaxis, :, :]  # (1, 1, seq_len, seq_len)

    output = mha.forward(dummy_input, mask)

    assert output.shape == (batch_size, seq_len, embed_dim)


def test_mha_backward_shape():
    embed_dim = 16
    num_heads = 2
    mha = MultiHeadSelfAttention(embed_dim, num_heads)

    batch_size = 2
    seq_len = 10
    dummy_input = np.random.rand(batch_size, seq_len, embed_dim)
    output = mha.forward(dummy_input)  # Call forward to populate cache

    d_output = np.random.rand(*output.shape)
    d_input = mha.backward(d_output)

    assert d_input.shape == dummy_input.shape
    # Check if gradients are populated with correct shapes
    for k, v in mha.params.items():
        assert k in mha.grads
        assert mha.grads[k].shape == v.shape


def test_mha_backward_with_mask_shape():
    embed_dim = 16
    num_heads = 2
    mha = MultiHeadSelfAttention(embed_dim, num_heads)

    batch_size = 2
    seq_len = 10
    dummy_input = np.random.rand(batch_size, seq_len, embed_dim)

    # Create a proper look-ahead mask for testing the mask logic
    look_ahead_mask = np.triu(np.ones((seq_len, seq_len)), k=1).astype('float32')
    look_ahead_mask = (look_ahead_mask * -1e9)
    look_ahead_mask = look_ahead_mask[np.newaxis, np.newaxis, :, :]  # (1, 1, seq_len, seq_len)

    output = mha.forward(dummy_input, look_ahead_mask)  # Call forward with mask

    d_output = np.random.rand(*output.shape)
    d_input = mha.backward(d_output)

    assert d_input.shape == dummy_input.shape
    for k, v in mha.params.items():
        assert k in mha.grads
        assert mha.grads[k].shape == v.shape

def test_dropout_mha():
    embed_dim = 16
    num_heads = 2
    dropout_rate = 0.5
    mha = MultiHeadSelfAttention(embed_dim, num_heads, dropout_rate=dropout_rate)

    batch_size = 1
    seq_len = 5
    dummy_input = np.ones((batch_size, seq_len, embed_dim)) # Use ones to easily detect zeros

    # Test training mode (dropout active)
    np.random.seed(0) # For reproducibility
    output_training = mha.forward(dummy_input, training=True)
    # Check if some values are zeroed out (or scaled)
    assert not np.allclose(output_training, np.ones_like(output_training) * (1 / (1 - dropout_rate)))
    assert np.any(output_training == 0) or np.any(output_training != (1 / (1 - dropout_rate)))

    # Test inference mode (dropout inactive)
    np.random.seed(0) # Reset seed for comparison
    output_inference = mha.forward(dummy_input, training=False)
    # In inference, dropout should not zero out anything, and values should not be scaled
    # The output will still be different from input due to attention mechanism,
    # but it should be deterministic given the same input and weights.
    # We can compare two inference runs to ensure no randomness.
    output_inference_2 = mha.forward(dummy_input, training=False)
    assert np.allclose(output_inference, output_inference_2)


# --- FeedForward Tests ---
def test_ffn_forward_shape():
    embed_dim = 16
    ff_dim = 32
    ffn = FeedForward(embed_dim, ff_dim)

    batch_size = 2
    seq_len = 10
    dummy_input = np.random.rand(batch_size, seq_len, embed_dim)
    output = ffn.forward(dummy_input)

    assert output.shape == (batch_size, seq_len, embed_dim)


def test_ffn_backward_shape():
    embed_dim = 16
    ff_dim = 32
    ffn = FeedForward(embed_dim, ff_dim)

    batch_size = 2
    seq_len = 10
    dummy_input = np.random.rand(batch_size, seq_len, embed_dim)
    output = ffn.forward(dummy_input)  # Call forward to populate cache

    d_output = np.random.rand(*output.shape)
    d_input = ffn.backward(d_output)

    assert d_input.shape == dummy_input.shape
    # Check if gradients are populated with correct shapes
    for k, v in ffn.params.items():
        assert k in ffn.grads
        assert ffn.grads[k].shape == v.shape

def test_dropout_ffn():
    embed_dim = 16
    ff_dim = 32
    dropout_rate = 0.5
    ffn = FeedForward(embed_dim, ff_dim, dropout_rate=dropout_rate)

    batch_size = 1
    seq_len = 5
    dummy_input = np.ones((batch_size, seq_len, embed_dim))

    # Test training mode (dropout active)
    np.random.seed(0)
    output_training = ffn.forward(dummy_input, training=True)
    assert np.any(output_training == 0) or np.any(output_training != (1 / (1 - dropout_rate)))

    # Test inference mode (dropout inactive)
    np.random.seed(0)
    output_inference = ffn.forward(dummy_input, training=False)
    output_inference_2 = ffn.forward(dummy_input, training=False)
    assert np.allclose(output_inference, output_inference_2)


# --- TransformerBlock Tests ---
def test_transformer_block_forward_shape():
    embed_dim = 16
    num_heads = 2
    ff_dim = 32
    block = TransformerBlock(embed_dim, num_heads, ff_dim)

    batch_size = 2
    seq_len = 10
    dummy_input = np.random.rand(batch_size, seq_len, embed_dim)
    output = block.forward(dummy_input)

    assert output.shape == (batch_size, seq_len, embed_dim)


def test_transformer_block_backward_shape():
    embed_dim = 16
    num_heads = 2
    ff_dim = 32
    block = TransformerBlock(embed_dim, num_heads, ff_dim)

    batch_size = 2
    seq_len = 10
    dummy_input = np.random.rand(batch_size, seq_len, embed_dim)
    output = block.forward(dummy_input)  # Call forward to populate cache

    d_output = np.random.rand(*output.shape)
    d_input = block.backward(d_output)

    assert d_input.shape == dummy_input.shape
    # Check if gradients are populated with correct shapes
    for k, v in block.params.items():
        assert k in block.grads
        assert block.grads[k].shape == v.shape
    # Also check sub-layers
    for k, v in block.attention.params.items():
        assert k in block.attention.grads
        assert block.attention.grads[k].shape == v.shape
    for k, v in block.feed_forward.params.items():
        assert k in block.feed_forward.grads
        assert block.feed_forward.grads[k].shape == v.shape


# --- Transformer Tests ---
def test_transformer_forward_shape():
    transformer = _create_dummy_transformer()

    batch_size = 2
    seq_len = 10
    dummy_input_ids = np.random.randint(0, transformer.vocab_size, (batch_size, seq_len))
    output_logits = transformer.forward(dummy_input_ids)

    assert output_logits.shape == (batch_size, seq_len, transformer.vocab_size)


def test_transformer_generate_returns_tokens():
    transformer = _create_dummy_transformer()

    # Use BOS_TOKEN_ID from AtlasBrain for consistency
    prompt_tokens = np.array([[AtlasBrain.BOS_TOKEN_ID]])
    generated_tokens = transformer.generate(prompt_tokens, max_new_tokens=5, eos_token_id=AtlasBrain.EOS_TOKEN_ID)

    assert isinstance(generated_tokens, np.ndarray)
    assert generated_tokens.ndim == 1
    assert len(generated_tokens) > 0
    assert all(isinstance(t, (int, np.integer)) for t in generated_tokens)

def test_generate_non_empty():
    transformer = _create_dummy_transformer()
    prompt_tokens = np.array([[AtlasBrain.BOS_TOKEN_ID]])
    generated_tokens = transformer.generate(prompt_tokens, max_new_tokens=1, eos_token_id=AtlasBrain.EOS_TOKEN_ID)
    # The generated sequence should always contain at least the prompt token(s)
    # and potentially one new token before EOS.
    assert len(generated_tokens) >= len(prompt_tokens[0])
    assert generated_tokens[0] == AtlasBrain.BOS_TOKEN_ID


def test_transformer_learning_reduces_loss():
    vocab_size = 10
    embed_dim = 16
    num_heads = 2
    ff_dim = 32
    num_layers = 1  # Use fewer layers for faster test
    max_seq_len = 5
    learning_rate = 0.1  # Higher LR for faster convergence in test

    transformer = Transformer(vocab_size, embed_dim, num_heads, ff_dim, num_layers, max_seq_len)

    # Simple repeating pattern for training
    # Input: [1, 2, 3, 1] -> Target: [2, 3, 1, 2]
    input_sequence = np.array([[1, 2, 3, 1]])
    target_sequence = np.array([[2, 3, 1, 2]])

    # Pad to max_seq_len
    input_padded = np.full((1, max_seq_len), AtlasBrain.PAD_TOKEN_ID, dtype=int)
    target_padded = np.full((1, max_seq_len), AtlasBrain.PAD_TOKEN_ID, dtype=int)
    input_padded[0, :input_sequence.shape[1]] = input_sequence[0]
    target_padded[0, :target_sequence.shape[1]] = target_sequence[0]

    initial_loss = transformer.train_step(input_padded, target_padded, learning_rate, training=True)

    # Train for a few steps
    losses = [initial_loss]
    for _ in range(10):  # 10 training steps
        loss = transformer.train_step(input_padded, target_padded, learning_rate, training=True)
        losses.append(loss)

    # Assert that loss generally decreases
    # It might not strictly decrease every step due to small model/data,
    # but the final loss should be significantly lower than initial.
    assert losses[-1] < initial_loss * 0.9  # Expect at least 10% reduction


def test_transformer_update_vocab_size():
    transformer = _create_dummy_transformer(vocab_size=10)
    initial_vocab_size = transformer.vocab_size

    # Make copies of initial weights to compare against
    initial_token_embedding = np.copy(transformer.token_embedding)
    initial_output_layer = np.copy(transformer.output_layer)
    initial_output_bias = np.copy(transformer.output_bias)

    new_vocab_size = 15
    transformer.update_vocab_size(new_vocab_size)

    assert transformer.vocab_size == new_vocab_size
    assert transformer.token_embedding.shape == (new_vocab_size, transformer.embed_dim)
    assert transformer.output_layer.shape == (transformer.embed_dim, new_vocab_size)
    assert transformer.output_bias.shape == (new_vocab_size,)

    # Ensure existing embeddings/weights are preserved (first `initial_vocab_size` entries)
    assert np.allclose(transformer.token_embedding[:initial_vocab_size, :], initial_token_embedding)
    assert np.allclose(transformer.output_layer[:, :initial_vocab_size], initial_output_layer)
    assert np.allclose(transformer.output_bias[:initial_vocab_size], initial_output_bias)


# --- Loss function tests ---
def test_label_smoothing_loss_and_backward():
    vocab_size = 5
    predictions = softmax(np.array([[1.0, 2.0, 3.0, 0.5, 0.1], [0.1, 0.2, 0.7, 1.5, 0.3]]))
    targets = np.array([2, 3]) # Target indices
    smoothing = 0.1

    # Manual calculation for expected loss
    # Smoothed targets for first example (target 2):
    # (1 - 0.1) * [0,0,1,0,0] + 0.1/5 * [1,1,1,1,1]
    # = 0.9 * [0,0,1,0,0] + 0.02 * [1,1,1,1,1]
    # = [0.02, 0.02, 0.92, 0.02, 0.02]
    # Smoothed targets for second example (target 3):
    # = [0.02, 0.02, 0.02, 0.92, 0.02]

    smoothed_targets_0 = np.array([0.02, 0.02, 0.92, 0.02, 0.02])
    smoothed_targets_1 = np.array([0.02, 0.02, 0.02, 0.92, 0.02])

    expected_loss_0 = -np.sum(smoothed_targets_0 * np.log(predictions[0]))
    expected_loss_1 = -np.sum(smoothed_targets_1 * np.log(predictions[1]))
    expected_loss = (expected_loss_0 + expected_loss_1) / 2

    loss = label_smoothing_loss(predictions, targets, vocab_size, smoothing)
    assert np.isclose(loss, expected_loss)

    # Manual calculation for expected gradient
    expected_grad_0 = predictions[0] - smoothed_targets_0
    expected_grad_1 = predictions[1] - smoothed_targets_1
    expected_grad = np.array([expected_grad_0, expected_grad_1]) / 2

    grad = label_smoothing_backward(predictions, targets, vocab_size, smoothing)
    assert np.allclose(grad, expected_grad)

def test_top_k_top_p_sampling():
    np.random.seed(42) # For reproducibility

    logits = np.array([1.0, 2.0, 3.0, 0.5, 0.1]) # vocab_size = 5
    vocab_size = 5
    temperature = 1.0

    # Test top-k only (k=3)
    # Top 3 logits: 3.0, 2.0, 1.0 (indices 2, 1, 0)
    # Others should be -inf
    sampled_token_k = _top_k_top_p_sampling(logits, vocab_size, top_k=3, top_p=1.0, temperature=temperature)
    assert sampled_token_k in [0, 1, 2]

    # Test top-p only (p=0.8)
    # Logits: [1.0, 2.0, 3.0, 0.5, 0.1]
    # Softmax: [0.106, 0.289, 0.788, 0.064, 0.039] (approx, after temp)
    # Sorted probs: [0.788 (idx 2), 0.289 (idx 1), 0.106 (idx 0), 0.064 (idx 3), 0.039 (idx 4)]
    # Cumulative: [0.788, 1.077, ...]
    # With p=0.8, only index 2 should be considered (0.788 < 0.8, but 0.788 + 0.289 > 0.8, so cutoff after first)
    # The logic in _top_k_top_p_sampling ensures at least one token is kept.
    # If cumulative_probabilities[0] > top_p, it keeps only the first.
    # Here, 0.788 is not > 0.8, so it should keep index 2 and index 1.
    # Let's re-evaluate the expected behavior based on the code:
    # sorted_probabilities = [0.788, 0.289, 0.106, 0.064, 0.039]
    # cumulative_probabilities = [0.788, 1.077, 1.183, 1.247, 1.286]
    # cutoff_idx = np.where(cumulative_probabilities > 0.8)[0][0] = 1
    # indices_to_remove = sorted_indices[1:] = [1, 0, 3, 4]
    # So only index 2 should remain.
    sampled_token_p = _top_k_top_p_sampling(logits, vocab_size, top_k=0, top_p=0.8, temperature=temperature)
    assert sampled_token_p == 2 # Index 2 has the highest probability and is within p=0.8

    # Test combined top-k (k=3) and top-p (p=0.8)
    # Top-k first: logits become [-inf, 2.0, 3.0, 0.5, -inf]
    # Softmax on these: [0, 0.289, 0.788, 0.064, 0] (renormalized)
    # Sorted probs: [0.788 (idx 2), 0.289 (idx 1), 0.064 (idx 3)]
    # Cumulative: [0.788, 1.077, ...]
    # Cutoff for p=0.8 is still at index 1 of sorted list, so only index 2 remains.
    sampled_token_kp = _top_k_top_p_sampling(logits, vocab_size, top_k=3, top_p=0.8, temperature=temperature)
    assert sampled_token_kp == 2

    # Test with very low temperature (should pick highest logit)
    sampled_token_low_temp = _top_k_top_p_sampling(logits, vocab_size, top_k=0, top_p=1.0, temperature=0.01)
    assert sampled_token_low_temp == 2 # Index 2 has logit 3.0, highest

    # Test with high temperature (more uniform sampling)
    # This is hard to assert deterministically, just ensure it runs
    sampled_token_high_temp = _top_k_top_p_sampling(logits, vocab_size, top_k=0, top_p=1.0, temperature=100.0)
    assert sampled_token_high_temp in [0, 1, 2, 3, 4]

def test_transformer_generate_beam_search():
    np.random.seed(42)
    vocab_size = 10
    embed_dim = 16
    num_heads = 2
    ff_dim = 32
    num_layers = 1
    max_seq_len = 10
    transformer = _create_dummy_transformer(vocab_size, embed_dim, num_heads, ff_dim, num_layers, max_seq_len)

    prompt_tokens = np.array([[AtlasBrain.BOS_TOKEN_ID]])
    max_new_tokens = 5
    beam_width = 2

    # For this test, we mostly want to ensure it runs and produces a sequence
    # The actual content is hard to predict without a trained model.
    generated_tokens = transformer.generate(
        prompt_tokens,
        max_new_tokens=max_new_tokens,
        eos_token_id=AtlasBrain.EOS_TOKEN_ID,
        beam_width=beam_width
    )

    assert isinstance(generated_tokens, np.ndarray)
    assert generated_tokens.ndim == 1
    assert len(generated_tokens) > 0
    # The generated sequence should start with the prompt token
    assert generated_tokens[0] == AtlasBrain.BOS_TOKEN_ID
    # The length should be at most max_new_tokens + len(prompt)
    assert len(generated_tokens) <= max_new_tokens + len(prompt_tokens[0])
    # If EOS is generated, it should be the last token
    if AtlasBrain.EOS_TOKEN_ID in generated_tokens:
        assert generated_tokens[-1] == AtlasBrain.EOS_TOKEN_ID