# Atlas: A Tiny Transformer Chatbot

Atlas is a conversational AI designed to learn and adapt through interaction. This version of Atlas features an **enhanced Transformer model built from scratch using only NumPy**, demonstrating online learning capabilities with advanced generation strategies and conversational memory.

## How it Works

Unlike traditional chatbots that rely on large pre-trained models or fixed rules, Atlas learns incrementally from each conversation. Every message you send updates the model's weights via backpropagation, allowing it to improve its understanding and response generation over time.

The core of Atlas is a Transformer neural network, implemented without any external machine learning frameworks (like TensorFlow or PyTorch). It uses:
-   **Positional Encoding**: To understand word order.
-   **Multi-Head Self-Attention**: To weigh the importance of different words in a sentence, now with **dropout** for regularization.
-   **Feed-Forward Networks**: For processing information, also with **dropout**.
-   **Layer Normalization and Residual Connections**: For stable training.
-   **Label Smoothing Cross-Entropy with Padding Mask**: A more robust loss function that ignores padding tokens.
-   **Xavier/Glorot Initialization**: For more stable weight initialization.

The model's capacity has been adjusted to smaller, more stable values (`embed_dim=32`, `num_heads=2`, `ff_dim=64`, `num_layers=2`) to prevent model collapse and improve training stability. The vocabulary grows dynamically as new words are encountered, and the model's parameters are saved and loaded to retain its "memory" across sessions.

## Features

-   **Online Learning**: Learns from every user input, with **learning rate decay** and a **replay buffer** to reinforce past patterns.
-   **Dynamic Vocabulary**: Adapts to new words.
-   **Persistent Memory**: Saves and loads its learned weights and vocabulary.
-   **Short-Term Conversation Memory**: Maintains a buffer of recent exchanges to provide context-aware responses.
-   **Advanced Generation Strategies**:
    -   **Top-K and Top-P (Nucleus) Sampling**: For more diverse and coherent text generation.
    -   **Beam Search**: An optional strategy for finding the most probable sequence of words.
    -   **Temperature and Repetition Penalty**: Controls randomness and discourages repetitive phrases.
-   **Dropout Regularization**: Improves generalization and prevents overfitting.
-   **Robustness**: Includes checks for NaN/Inf weights and empty responses.
-   **Minimal Dependencies**: Built primarily with Python's standard library and NumPy.

## Getting Started

### Prerequisites

-   Python 3.8+
-   `numpy`

### Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/your-username/atlas-chat-learning.git
    cd atlas-chat-learning
    ```
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

### Running Atlas

To start chatting with Atlas, run the `main.py` script:

```bash
python main.py
```

### Running Tests

To run the unit tests and ensure everything is working correctly:

```bash
python -m pytest
```

### Interacting with Atlas

-   Type your messages at the `You: ` prompt.
-   Atlas will respond based on what it has learned, leveraging its conversation history.
-   The more you chat, the smarter Atlas should become!
-   Type `quit` or `exit` to save the model and exit the conversation.
-   The model automatically saves every 5 interactions.

### Training

Every message you send to Atlas serves as a training example. The model performs one gradient step (backpropagation) after each of your inputs to update its internal parameters. This process now includes:
-   **Conversation Context**: Your current message is combined with recent conversation history to provide a richer training signal.
-   **Replay Buffer**: Past user messages are occasionally re-sampled and used for additional training steps, helping the model remember and reinforce earlier learnings.
-   **Learning Rate Decay**: The learning rate gradually decreases over time to ensure stable convergence.

This continuous learning process allows Atlas to adapt to your conversational style and the topics you discuss, producing more structured and context-aware answers.

## Project Structure

-   `main.py`: The main script to run the chatbot.
-   `atlas/`: Contains the core logic.
    -   `brain.py`: Manages the chatbot's overall logic, including tokenization, vocabulary management, conversation history, replay buffer, and orchestrating the Transformer.
    -   `transformer.py`: Implements the Transformer architecture from scratch using NumPy, including Multi-Head Self-Attention, Feed-Forward Networks, Positional Encoding, Layer Normalization, Residual Connections, Dropout, and advanced generation methods (Top-K, Top-P, Beam Search).
-   `requirements.txt`: Lists Python dependencies.
-   `README.md`: This file.
-   `tests/`: Unit tests for the components.

## Known Limitations

-   **Small Model Size**: Due to its small size, Atlas may produce repetitive or less coherent responses compared to larger, pre-trained models. For better quality, training with a much larger dataset and a more powerful model would be necessary.
-   **Limited Context**: `max_seq_len` is set to 50, meaning it can only consider a short window of past tokens.
-   **Computational Cost**: Training is done on the CPU using NumPy, which is slower than GPU-accelerated frameworks.

## Troubleshooting

-   **Atlas stops responding or gives strange output**: This can happen if the model's weights become corrupted (e.g., due to NaN/Inf values during training). To reset Atlas, delete the `atlas_model.npz` and `atlas_vocab.pkl` files in the project root. Atlas will then start learning from scratch.

## Future Enhancements

-   More sophisticated tokenization (e.g., BPE).
-   Integration with a GUI or web interface.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Educational Project

This project is intended for educational purposes to demonstrate the implementation of a Transformer model and its application in a conversational AI from first principles using NumPy. It serves as a learning resource for understanding the internal workings of such models.
