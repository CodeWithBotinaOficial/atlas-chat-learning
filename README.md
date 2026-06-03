# Atlas

Atlas is a conversational AI that learns from user input on the fly, without relying on pre-trained models. It's designed to build its knowledge base and conversational abilities directly from interactions.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Cloning the Repository

```bash
git clone https://github.com/CodeWithBotinaOficial/atlas-chat-learning.git
cd atlas-chat-learning
```

### Installation

Atlas uses a minimal set of dependencies. You can install them using pip:

```bash
pip install -r requirements.txt
```

### Running Atlas

To start an interactive session with Atlas, run the `main.py` script:

```bash
python main.py
```

### Basic Usage Example

Once running, Atlas will prompt you for input. Type your message and press Enter. Atlas will respond based on its current learning state.

```
You: Hello Atlas!
Atlas: hello world this is a test
You: What is your name?
Atlas: this is another test sentence
You: Tell me more about yourself.
Atlas: i am still learning
```
Type `quit` or `exit` to save Atlas's memory and end the session.

## Folder Structure

The project is organized as follows:

```
.
├── atlas/                  # Core AI logic and modules
│   ├── __init__.py         # Makes 'atlas' a Python package
│   └── brain.py            # Contains the core learning and response logic (AtlasBrain)
├── tests/                  # Unit tests for the project
│   └── test_brain.py       # Tests for the AtlasBrain module
├── main.py                 # Main entry point for the application
├── requirements.txt        # Lists project dependencies
├── atlas_memory.pkl        # (Generated) Persistent memory file for Atlas's learned knowledge
└── README.md               # This file
```

## Learning Mechanism (N-grams)

Atlas learns incrementally from your conversations using an n-gram model. Specifically, it builds:

*   **Unigrams**: Individual words and their frequencies. These are used to start new sentences.
*   **Bigrams**: Pairs of consecutive words (e.g., "hello world"). Atlas learns which word is likely to follow another.
*   **Trigrams**: Sequences of three consecutive words (e.g., "the quick brown"). This allows Atlas to generate more coherent and contextually relevant phrases.

When you provide input, Atlas tokenizes your text, updates the frequencies of these n-grams, and expands its vocabulary. When generating a response, it tries to predict the next word based on the preceding one or two words, choosing from learned patterns with higher frequency.

The more you interact with Atlas, the larger its vocabulary and n-gram knowledge base become, leading to more varied and "intelligent" responses over time. Its memory is saved to `atlas_memory.pkl` to ensure persistence across sessions.
