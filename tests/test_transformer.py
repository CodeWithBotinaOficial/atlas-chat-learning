# tests/test_transformer.py
import numpy as np
import pytest
from atlas.transformer import Transformer, PositionalEncoding, MultiHeadSelfAttention, FeedForward, TransformerBlock, \
    cross_entropy_loss, cross_entropy_backward, softmax
from atlas.brain import AtlasBrain  # Import AtlasBrain to get special token IDs


# --- Helper functions for testing ---
def _create_dummy_transformer(vocab_size=10, embed_dim=16, num_heads=2, ff_dim=32, num_layers=2, max_seq_len=50):
    return Transformer(vocab_size, embed_dim, num_heads, ff_dim, num_layers, max_seq_len)


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

    initial_loss = transformer.train_step(input_padded, target_padded, learning_rate)

    # Train for a few steps
    losses = [initial_loss]
    for _ in range(10):  # 10 training steps
        loss = transformer.train_step(input_padded, target_padded, learning_rate)
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
def test_cross_entropy_loss():
    predictions = np.array([[0.1, 0.9], [0.8, 0.2]])
    targets = np.array([1, 0])  # Corresponds to [0, 1] and [1, 0] one-hot

    # Expected loss calculation:
    # - (log(0.9) + log(0.8)) / 2
    expected_loss = - (np.log(0.9) + np.log(0.8)) / 2

    loss = cross_entropy_loss(predictions, targets)
    assert np.isclose(loss, expected_loss)


def test_cross_entropy_backward():
    predictions = np.array([[0.1, 0.9], [0.8, 0.2]])
    targets = np.array([1, 0])

    # Expected gradient: predictions - one_hot_targets
    # (predictions - targets_one_hot) / batch_size
    # targets_one_hot = [[0, 1], [1, 0]]
    # grad_raw = [[0.1, -0.1], [-0.2, 0.2]]
    # grad = grad_raw / 2
    expected_grad = np.array([[0.05, -0.05], [-0.1, 0.1]])

    grad = cross_entropy_backward(predictions, targets)
    assert np.allclose(grad, expected_grad)
