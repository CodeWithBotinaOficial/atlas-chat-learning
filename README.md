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
Atlas: I am learning...
You: What is your name?
Atlas: I am learning...
```

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
└── README.md               # This file
```
