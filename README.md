[![CI](https://github.com/CodeWithBotinaOficial/atlas-chat-learning/actions/workflows/test.yml/badge.svg)](https://github.com/CodeWithBotinaOficial/atlas-chat-learning/actions/workflows/test.yml)

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
-   **Xavier/Glorot Uniform Initialization**: For more stable weight initialization.
-   **Adam Optimizer with Learning Rate Warmup**: Adaptive per-parameter learning rates for faster convergence, with a linear warmup over the first 1000 training steps.
-   **Embedding Scaling**: Token embeddings are scaled by `sqrt(embed_dim)` before positional encoding, following the original Transformer paper.
-   **Gradient Clipping**: To prevent exploding gradients and improve training stability.

The model's capacity has been adjusted to smaller, more stable values (`embed_dim=32`, `num_heads=2`, `ff_dim=64`, `num_layers=2`) to prevent model collapse and improve training stability. The vocabulary grows dynamically as new words are encountered, and the model's parameters are saved and loaded to retain its "memory" across sessions.

## Features

-   **Online Learning**: Learns from every user input, with **Adam optimization**, **learning rate warmup**, **learning rate decay**, and a **replay buffer** to reinforce past patterns.
-   **Dynamic Vocabulary**: Adapts to new words.
-   **Persistent Memory**: Saves and loads its learned weights and vocabulary.
-   **Short-Term Conversation Memory**: Maintains a buffer of recent exchanges to provide context-aware responses.
-   **Grammatical Post-Processing**: Responses are now automatically corrected for basic grammar, including capitalization, punctuation, and removal of excessive word repetition, to enhance readability and structure.
-   **Advanced Generation Strategies**:
    -   **Top-K and Top-P (Nucleus) Sampling**: For more diverse and coherent text generation.
    -   **Beam Search**: An optional strategy for finding the most probable sequence of words.
    -   **Temperature and Repetition Penalty**: Controls randomness and discourages repetitive phrases.
-   **Dropout Regularization**: Improves generalization and prevents overfitting.
-   **Robustness**: Includes checks for NaN/Inf weights, numerically stable softmax, and empty responses.
-   **Execution Modes**: Supports training-only, production-only, and dual (learn & respond) modes.
-   **Document Training**: Learn from local or remote PDF, DOCX, TXT, and Markdown files via `--file` and `--url`.
-   **Minimal Dependencies**: Built primarily with Python's standard library and NumPy.

## Getting Started

### Prerequisites

-   Python 3.8+
-   `numpy`
-   `pyyaml`
-   `requests` (for web scraping and document downloads)
-   `beautifulsoup4` (for web scraping)
-   `lxml` (recommended for `beautifulsoup4` parsing)
-   `pypdf` (for PDF text extraction)
-   `python-docx` (for DOCX text extraction)

### Installation

1.  Clone the repository:
    ```bash
    git clone https://https://github.com/CodeWithBotinaOficial/atlas-chat-learning.git
    cd atlas-chat-learning
    ```
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

### Configuration

Atlas uses a `config.yaml` file to manage its hyperparameters. To get started, copy the example configuration:

```bash
cp config.yaml.example config.yaml
```

Then, open `config.yaml` in a text editor and adjust the parameters as needed.

Here's a breakdown of the configuration parameters:

-   **`model`**: Defines the Transformer model's architecture.
    -   `embed_dim`: Dimension of the token embeddings. This is the size of the vector used to represent each word. Larger values allow for more complex representations but increase memory usage and computation.
    -   `num_heads`: Number of attention heads in the multi-head attention mechanism. More heads allow the model to jointly attend to information from different representation subspaces.
    -   `ff_dim`: Dimension of the feed-forward network within each Transformer block.
    -   `num_layers`: Number of Transformer encoder/decoder layers. More layers increase the model's depth and capacity.
    -   `max_seq_len`: Maximum sequence length the model can handle. This limits how many tokens (words) can be in an input or generated response.
    -   `dropout_rate`: Dropout rate for regularization. A fraction of neurons are randomly ignored during training to prevent overfitting.

-   **`training`**: Parameters related to the model's learning process.
    -   `learning_rate`: The initial learning rate for the optimizer. Controls the step size at each iteration while moving towards a minimum of the loss function.
    -   `lr_decay_rate`: Rate at which the learning rate decays over time.
    -   `lr_decay_steps`: Number of interactions after which the learning rate decays.
    -   `replay_buffer_size`: Maximum size of the replay buffer for experience replay. This stores past interactions for re-training.
    -   `replay_sample_rate`: Probability of sampling from the replay buffer during training.
    -   `beta1`, `beta2`, `epsilon`: Optional Adam optimizer hyperparameters (currently hardcoded in `transformer.py`; included in config for future use).
    -   `warmup_steps`: Number of training steps for linear learning rate warmup (currently hardcoded to 1000 in `transformer.py`).

-   **`generation`**: Parameters controlling how Atlas generates responses.
    -   `temperature`: Controls the randomness of predictions. Lower values (e.g., 0.1-0.5) make the model more deterministic and focused, while higher values (e.g., 0.8-1.0) increase creativity and diversity.
    -   `repetition_penalty`: Penalizes repeated tokens to encourage diverse responses. A value greater than 1.0 discourages repetition.
    -   `top_k`: Limits the sampling pool to the top K most likely next tokens. For example, `top_k: 40` means only the 40 most probable next words are considered.
    -   `top_p`: (Nucleus Sampling) Limits the sampling pool to the smallest set of tokens whose cumulative probability exceeds P. For example, `top_p: 0.95` means tokens are selected from the smallest set whose probabilities sum up to at least 95%.
    -   `beam_width`: If greater than 0, enables beam search decoding with this width. Beam search explores multiple possible sequences simultaneously and is slower. Default is `0` (disabled) for faster generation; use sampling or `--fast` instead.
    -   `max_new_tokens`: Maximum number of new tokens to generate in a response.

-   **`memory`**: Parameters for managing conversational memory.
    -   `max_history_length`: Maximum number of previous turns (user input + Atlas response) to keep in the conversation history. This provides context for future responses.

-   **`scraping`**: Parameters for configuring the web scraping functionality.
    -   `connect_timeout`: Timeout for establishing the connection to the web server in seconds.
    -   `read_timeout`: Timeout for receiving a response from the web server in seconds.
    -   `max_retries`: The number of times the scraper will retry fetching a URL if a transient network error (like a timeout or connection error) occurs.
    -   `backoff_factor`: A multiplier for the exponential backoff delay between retry attempts. The wait time is calculated as `backoff_factor * (2 ** (attempt - 1))` seconds.

-   **`performance`**: Parameters for optimizing Atlas for different hardware.
    -   `low_memory`: If `true`, Atlas will automatically configure itself with a very small model and conservative generation settings to minimize RAM usage. This is ideal for devices with limited memory (e.g., 4GB or less).
    -   `half_precision`: If `true`, model weights will be stored in `float16` (half-precision) format, reducing memory footprint by approximately 50%. Computations are still performed in `float32` for numerical stability during training. This is an experimental feature and may introduce slight numerical differences.
    -   `max_ram_gb`: (Future) Optional: auto-adjust model if RAM is limited.

**Important**: After modifying `config.yaml`, you need to restart Atlas for the changes to take effect.

### Execution Modes

To start chatting with Atlas, run the `main.py` script. You can specify different execution modes using command-line arguments:

-   **`--dual` (Default Mode)**:
    -   **Use Case**: General interactive conversation where Atlas learns from your input and generates responses. This is the standard mode for continuous improvement.
    -   **Behavior**: Processes your input, updates its model, and then generates a reply.
    ```bash
    python main.py
    # or explicitly
    python main.py --dual
    ```

-   **`--training` (Training-Only Mode)**:
    -   **Use Case**: Focused training without interaction overhead. Useful for feeding large datasets or specific learning materials to Atlas without needing immediate responses.
    -   **Behavior**: Only learns from your input and updates its model. It will not generate responses.
    -   **Output**: Prints a `.` for each successful learning step or `x` if learning was skipped (e.g., due to short input).
    ```bash
    python main.py --training
    ```

-   **`--production` (Production-Only Mode)**:
    -   **Use Case**: Deployment scenarios where you want stable responses without further learning. Ideal for integrating Atlas into applications where its behavior should be consistent.
    -   **Behavior**: Only generates responses using its current learned model. It will not update its weights or learn from new input.
    ```bash
    python main.py --production
    ```

-   **`--fast` (Low-Latency Generation)**:
    -   **Use Case**: Interactive deployments where response speed matters more than sampling diversity. Best combined with `--production`.
    -   **Behavior**: Enables greedy decoding (`temperature=0.01`, `top_k=1`, `beam_width=0`, `top_p=1.0`). Generation also uses **KV caching** in the Transformer so each new token reuses cached attention keys/values instead of recomputing the full sequence.
    ```bash
    python main.py --production --fast
    ```
    For fastest responses, also set `beam_width: 0` in `config.yaml` (now the default in `config.yaml.example`).

### Training from Documents

Atlas can learn from document files in **PDF**, **DOCX**, **TXT**, and **Markdown (MD)** formats. The text is extracted, sanitized, split into chunks, and fed to the model. After training completes, the model is saved automatically and the program exits (no interactive chat).

**Dependencies**: Install document parsing libraries with:

```bash
pip install pypdf python-docx
```

These are also included in `requirements.txt`.

**Local files:**

```bash
python main.py --train --file /home/user/documents/report.pdf
python main.py --file ./notes.docx
python main.py --file data.txt
python main.py --file readme.md
```

**Remote URLs:**

```bash
python main.py --url https://example.com/document.pdf
python main.py --url https://example.com/file.docx
```

**Notes:**
- `--file`, `--url`, and `--scraping` are mutually exclusive.
- Only one document is processed per run.
- Supported formats are detected from the file extension (local) or URL extension / HTTP `Content-Type` (remote).
- Chunks are limited to about 1000 characters, respecting word boundaries.
- If a required library is missing or a file is corrupted, Atlas prints a clear error and exits.

**Example output:**

```
[✓] Loading local document: /home/user/documents/report.pdf
[✓] Detected format: .pdf
[✓] Extracting text...
[✓] Extracted 45231 characters.
[✓] Chunking text for training...
[✓] Training Atlas with 48 chunk(s)... (This may take a moment)
...
[✓] Training complete! Learned from 48 chunk(s), skipped 0. Model saved. Exiting.
```

### Web Scraping Training

Atlas can also learn directly from web content. Use the `--scraping` argument followed by a URL to fetch, sanitize, and train the model with the text from a webpage.

**Dependencies**: For web scraping, ensure you have `requests`, `beautifulsoup4`, and `lxml` installed (`pip install requests beautifulsoup4 lxml`). `lxml` is highly recommended for performance; if not available, `html.parser` will be used with a warning.

**Important Considerations:**
-   **`robots.txt` Compliance**: The script will first check the target website's `robots.txt` file. If scraping the specified URL is disallowed for `User-agent: AtlasBot` or `User-agent: *`, the process will be halted. If `robots.txt` is not found, scraping is assumed to be allowed.
-   **User Confirmation**: After a successful `robots.txt` check, you will be prompted to confirm whether you wish to proceed with scraping and training.
-   **Text Extraction**: The script intelligently extracts the main article or post content, removing boilerplate like navigation, ads, and comments.
-   **Training Process**: The extracted and sanitized text is then fed to the Atlas model in chunks for training. This process may take some time for very large pages.
-   **Model Saving**: After training from the scraped content, the model will be automatically saved.

**Example Usage:**
```bash
python main.py --scraping https://blog.codewithbotina.com/es/posts/recreacion-moderna-de-pac-man-aprende-a-desarrollar-juegos-cross-platform-con-net-y-avalonia-ui
```
Expected output:
```
[✓] Attempting to scrape and train from: https://blog.codewithbotina.com/es/posts/recreacion-moderna-de-pac-man-aprende-a-desarrollar-juegos-cross-platform-con-net-y-avalonia-ui
[✓] Successfully fetched and parsed robots.txt from https://blog.codewithbotina.com/robots.txt.
[✓] robots.txt check passed. Scraping of https://blog.codewithbotina.com/es/posts/recreacion-moderna-de-pac-man-aprende-a-desarrollar-juegos-cross-platform-con-net-y-avalonia-ui is allowed.
Do you want to scrape this page and train the AI? (y/n): y
[✓] Scraping and sanitizing text... (This may take a moment)
[✓] Training Atlas with the new content... (This may take a moment)
[✓] Training complete! Model saved. Exiting.
```

### Grammar Post-processing

Atlas's responses undergo a post-processing step to improve their readability and grammatical correctness. This includes:
-   **Capitalization**: Ensures sentences start with a capital letter.
-   **Punctuation**: Adds appropriate punctuation (e.g., periods, question marks) at the end of sentences.
-   **Repetition Removal**: Detects and removes excessive repetition of words or phrases within a short span.
-   **Gibberish Detection**: Attempts to identify and filter out responses that appear to be random or nonsensical.

This step aims to make the AI's output feel more natural and polished, even if the underlying model is still learning.

### Model Analytics and Reports

Atlas features an automatic training session analytics reporting system. Every time a training session completes—whether through interactive chat (`--dual`, `--training`), document training (`--file`, `--url`), or web scraping (`--scraping`)—Atlas generates a timestamped report folder under `.reports/statistics/dd-mm-yyyy-hh-mm/`.

This folder contains:
1. **`metrics.json`**: Raw recorded metrics per interaction step (loss, learning rate, vocabulary size, timestamp).
2. **`analytics.pdf`**: A multi-page PDF report with mathematical details and visual plots (requires `matplotlib`):
   - **Executive Summary (Page 1)**: Meta details of the training session (duration, total steps, final loss, final learning rate) and model architecture hyperparameters.
   - **Training Curves (Page 2)**: Subplots tracking Loss progress (including a moving average line), Learning Rate schedule, and Vocabulary Size growth over time.
   - **Embedding Analysis (Page 3)**: A histogram of token embedding L2 norms, embedding statistics (mean, std, percentiles), and a lookup table of the most active (highest norm) and least active (lowest norm) vocabulary terms.
   - **Parameter Complexity & Confidence (Page 4)**: A color-coded heatmap slice of the output projection matrix, shapes and norms of all parameter groups, and a prediction confidence entropy index.

*Note: If `matplotlib` is not installed or the system is missing a graphical display server, Atlas will output a warning and generate a plain text report (`analytics_summary.txt`) alongside the raw `metrics.json` instead of the PDF.*

#### Configuration
You can configure or disable reporting in `config.yaml`:
```yaml
reporting:
  enabled: true             # Set false to disable recording and report generation
  output_dir: .reports/statistics
  generate_graphs: true      # Set false to only generate metrics.json and text summary
  max_history_points: 1000   # Caps history size to prevent high RAM usage in very long runs
```

#### Example Output Directory Structure
```text
.reports/
└── statistics/
    └── 07-06-2026-12-42/
        ├── analytics.pdf
        └── metrics.json
```

## Pre-trained Model (Optional)

You can download pre-trained model files (`atlas_model.npz` and `atlas_vocab.pkl`) from the [Releases](https://github.com/CodeWithBotinaOficial/atlas-chat-learning/releases) page.

To use the pre-trained model:
1.  Download both `atlas_model.npz` and `atlas_vocab.pkl` from the latest release.
2.  Place both files in the root directory of this project.
3.  Run `python main.py` to start chatting with the pre-trained model.

**Note:** The pre-trained model has been lightly trained on sample conversations. For better and more robust results, it is highly recommended to train your own model with `python main.py --training` using a larger and more diverse text corpus relevant to your application.

## Testing

To run the unit tests and ensure everything is working correctly:

```bash
python -m pytest
```

### Interacting with Atlas

-   Type your messages at the `You: ` prompt.
-   Atlas will respond based on what it has learned, leveraging its conversation history.
-   The more you chat, the smarter Atlas should become!
-   Type `quit` or `exit` to save the model and exit the conversation.
-   The model automatically saves every 5 interactions (except in production mode).

### Training

Every message you send to Atlas (in `dual` or `training` mode) serves as a training example. The model performs one gradient step (backpropagation) after each of your inputs to update its internal parameters. This process now includes:
-   **Adam Optimizer**: Adaptive per-parameter learning rates replace plain SGD for faster and more stable convergence.
-   **Learning Rate Warmup**: The effective learning rate ramps up linearly over the first 1000 training steps before reaching the configured base rate.
-   **Conversation Context**: Your current message is combined with recent conversation history to provide a richer training signal.
-   **Replay Buffer**: Past user messages are occasionally re-sampled and used for additional training steps, helping the model remember and reinforce earlier learnings.
-   **Learning Rate Decay**: The learning rate gradually decreases over time to ensure stable convergence.
-   **Gradient Clipping**: Prevents numerical instability during training.

This continuous learning process allows Atlas to adapt to your conversational style and the topics you discuss, producing more structured and context-aware answers.

## Project Structure

-   `main.py`: The main script to run the chatbot, with argument parsing for execution modes, document training, and config path.
-   `config.yaml`: Configuration file for hyperparameters.
-   `atlas/`: Contains the core logic.
    -   `brain.py`: Manages the chatbot's overall logic, including tokenization, vocabulary management, conversation history, replay buffer, and orchestrating the Transformer.
    -   `transformer.py`: Implements the Transformer architecture from scratch using NumPy, including Multi-Head Self-Attention, Feed-Forward Networks, Positional Encoding, Layer Normalization, Residual Connections, Dropout, KV caching for fast inference, and advanced generation methods (Top-K, Top-P, Beam Search). Now includes numerical stability improvements and gradient clipping.
    -   `grammar.py`: Contains the `GrammarHelper` class for post-processing generated responses to improve readability and structure.
    -   `config_loader.py`: Module to load and manage the `config.yaml` file.
    -   `document_loader.py`: Loads and extracts text from PDF, DOCX, TXT, and MD files (local or remote).
-   `web_scraper.py`: Contains functions for checking `robots.txt` and scraping web content.
-   `text_processor.py`: Handles HTML parsing and text sanitization.
-   `requirements.txt`: Lists Python dependencies.
-   `README.md`: This file.
-   `tests/`: Unit tests for the components.

## Known Limitations

-   **Small Model Size**: Due to its small size, Atlas may produce repetitive or less coherent responses compared to larger, pre-trained models. For better quality, training with a much larger dataset and a more powerful model would be necessary.
-   **Limited Context**: `max_seq_len` is set to 50, meaning it can only consider a short window of past tokens.
-   **Grammar Post-Processing**: While improving readability, the rule-based grammar corrections do not make the underlying model smarter or improve its semantic understanding. They are purely for presentational enhancement.
-   **Computational Cost**: Training is done on the CPU using NumPy, which is slower than GPU-accelerated frameworks.

## Hardware Requirements

The computational resources required depend heavily on the `model` configuration parameters, especially `embed_dim`, `num_heads`, `ff_dim`, `num_layers`, and `max_seq_len`.

-   **Low-Memory Mode (`performance.low_memory: true`)**:
    -   **RAM**: Minimum 1GB, 2GB recommended.
    -   **CPU**: Any modern CPU, including single-board computers like Raspberry Pi.
    -   **Disk Space**: Negligible (model files are typically < 1MB).
    -   **Changes**: This mode drastically reduces model size and disables heavy generation features. Specifically, it sets:
        -   `model.embed_dim = 16`
        -   `model.num_heads = 2`
        -   `model.ff_dim = 32`
        -   `model.num_layers = 1`
        -   `model.max_seq_len = 25`
        -   `model.dropout_rate = 0.0`
        -   `generation.beam_width = 0`
        -   `generation.top_k = 10`
        -   `generation.max_new_tokens = 30`
        -   `training.replay_buffer_size = 5`
        -   `training.replay_sample_rate = 0.0`
        -   `memory.max_history_length = 2`

-   **Standard Configuration (e.g., default `config.yaml`)**:
    -   **RAM**: Minimum 4GB, 8GB recommended.
    -   **CPU**: Any modern multi-core CPU (e.g., Intel i3/i5 or AMD Ryzen 3/5 equivalent) from the last 5-7 years should suffice for interactive use. Training will be slower but manageable.
    -   **Disk Space**: Negligible (model files are small, typically < 10MB).

-   **Larger Configuration (e.g., `embed_dim: 512`, `num_layers: 8`, `max_seq_len: 256`)**:
    -   **RAM**: Minimum 16GB, 32GB+ recommended. The memory footprint grows significantly with `embed_dim` and `max_seq_len`.
    -   **CPU**: A powerful multi-core CPU (e.g., Intel i7/i9 or AMD Ryzen 7/9 equivalent) is highly recommended for reasonable training and response generation times.
    -   **Disk Space**: Model files can grow to tens or hundreds of MBs.

**Half-Precision (`performance.half_precision: true`)**:
This option stores model weights in `float16` format, effectively halving the memory usage of the model parameters. While computations are performed in `float32` for numerical stability during training, the reduced storage can be beneficial for memory-constrained environments. This feature is experimental and may subtly affect model stability or convergence.

**General Recommendation**: For a smooth interactive experience and faster training, a system with at least 8GB RAM and a quad-core CPU is advisable. If you plan to experiment with significantly larger models or extensive web scraping, consider a machine with 16GB+ RAM.

## Troubleshooting

-   **Atlas stops responding or gives strange output**: This can happen if the model's weights become corrupted (e.g., due to NaN/Inf values during training). To reset Atlas, delete the `atlas_model.npz` and `atlas_vocab.pkl` files in the project root. Atlas will then start learning from scratch.
-   **After upgrading to Adam optimizer**: If you have an existing `atlas_model.npz` from a previous version, delete it (and optionally `atlas_vocab.pkl` if you want a fresh vocabulary) before training. The new Adam optimizer maintains internal momentum state that is not stored in saved model files and is incompatible with weights trained under plain SGD.

## Future Enhancements

-   More sophisticated tokenization (e.g., BPE).
-   Integration with a GUI or web interface.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Educational Project

This project is intended for educational purposes to demonstrate the implementation of a Transformer model and its application in a conversational AI from first principles using NumPy. It serves as a learning resource for understanding the internal workings of such models.