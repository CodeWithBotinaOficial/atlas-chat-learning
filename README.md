# Atlas: A Tiny Transformer Chatbot

Atlas is a conversational AI designed to learn and adapt through interaction. This version of Atlas features a **minimal Transformer model built from scratch using only NumPy**, demonstrating online learning capabilities.

## How it Works

Unlike traditional chatbots that rely on large pre-trained models or fixed rules, Atlas learns incrementally from each conversation. Every message you send updates the model's weights via backpropagation, allowing it to improve its understanding and response generation over time.

The core of Atlas is a small Transformer neural network, implemented without any external machine learning frameworks (like TensorFlow or PyTorch). It uses:
-   **Positional Encoding**: To understand word order.
-   **Multi-Head Self-Attention**: To weigh the importance of different words in a sentence.
-   **Feed-Forward Networks**: For processing information.
-   **Layer Normalization and Residual Connections**: For stable training.

The vocabulary grows dynamically as new words are encountered, and the model's parameters are saved and loaded to retain its "memory" across sessions.

## Features

-   **Online Learning**: Learns from every user input.
-   **Dynamic Vocabulary**: Adapts to new words.
-   **Persistent Memory**: Saves and loads its learned weights and vocabulary.
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
-   Atlas will respond based on what it has learned.
-   The more you chat, the smarter Atlas should become!
-   Type `quit` or `exit` to save the model and exit the conversation.
-   The model automatically saves every 5 interactions.

### Training

Every message you send to Atlas serves as a training example. The model performs one gradient step (backpropagation) after each of your inputs to update its internal parameters. This means:
-   **Input**: Your message (e.g., "hello there")
-   **Training**: Atlas processes this message, trying to predict the next word in the sequence. It adjusts its weights to minimize the prediction error.
-   **Output**: Atlas then generates a response based on its updated knowledge.

This continuous learning process allows Atlas to adapt to your conversational style and the topics you discuss.

## Project Structure

-   `main.py`: The main script to run the chatbot.
-   `atlas/`: Contains the core logic.
    -   `brain.py`: Manages the chatbot's overall logic, including tokenization, vocabulary management, and orchestrating the Transformer.
    -   `transformer.py`: Implements the Transformer architecture from scratch using NumPy.
-   `requirements.txt`: Lists Python dependencies.
-   `README.md`: This file.
-   `tests/`: Unit tests for the components.

## Future Enhancements

-   More sophisticated tokenization (e.g., BPE).
-   Larger model sizes and more complex architectures.
-   Improved generation strategies (e.g., beam search, repetition penalties).
-   Integration with a GUI or web interface.

## Contributing

Contributions are welcome! Please feel free to open issues or submit pull requests.