# main.py
"""
main.py

This is the main entry point for the Atlas conversational AI.
It handles the interactive loop with the user, incorporating
n-gram based learning and persistent memory.
"""

from atlas.brain import AtlasBrain
import sys
import os
import argparse
from atlas.config_loader import load_config # Import load_config

# Import web scraping and text processing modules
try:
    import requests
    from bs4 import BeautifulSoup
    from atlas import web_scraper
    from atlas import text_processor
except ImportError:
    # These will be handled specifically if --scraping is used
    pass

# --- Readline setup for command history and arrow key navigation ---
try:
    import readline
    import atexit

    histfile = os.path.expanduser("~/.atlas_history")
    try:
        readline.read_history_file(histfile)
    except FileNotFoundError:
        pass
    # Set history length to avoid unlimited growth
    readline.set_history_length(1000)
    atexit.register(readline.write_history_file, histfile)
    print("Readline enabled for command history and editing.")
except ImportError:
    print("Readline module not available (likely on Windows). Command history and arrow keys will not work.")
except Exception as e:
    print(f"An error occurred while setting up readline: {e}. Command history and arrow keys may not work.")
# --- End Readline setup ---

MAX_CHUNK_SIZE = 1000 # Max characters per chunk for training


def chunk_text_for_training(text, max_chunk_size=MAX_CHUNK_SIZE):
    """
    Split text into chunks for training, respecting word boundaries.
    """
    words = text.split()
    current_chunk = []
    current_chunk_len = 0
    chunks = []

    for word in words:
        separator_len = 1 if current_chunk else 0
        if current_chunk and current_chunk_len + separator_len + len(word) > max_chunk_size:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]
            current_chunk_len = len(word)
        else:
            current_chunk.append(word)
            current_chunk_len += separator_len + len(word)

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def train_brain_from_text(brain, text, source_label):
    """
    Parse text into alternating prompt/response pairs and train using brain.learn_pair() over multiple epochs.
    """
    print("[✓] Parsing text into prompt-response pairs...")
    
    def clean_prefix(line):
        line_lower = line.lower()
        if line_lower.startswith("usuario "):
            return line[len("usuario "):].strip()
        if line_lower.startswith("usuario:"):
            return line[len("usuario:"):].strip()
        if line_lower.startswith("atlas "):
            return line[len("atlas "):].strip()
        if line_lower.startswith("atlas:"):
            return line[len("atlas:"):].strip()
        return line

    cleaned_lines = []
    for line in text.split('\n'):
        line = line.strip()
        if not line:
            continue
        cleaned = clean_prefix(line)
        if cleaned:
            cleaned_lines.append(cleaned)

    pairs = []
    for i in range(0, len(cleaned_lines) - 1, 2):
        pairs.append((cleaned_lines[i], cleaned_lines[i+1]))

    if not pairs:
        print("[✗] No valid prompt-response pairs found after parsing.", file=sys.stderr)
        sys.exit(1)

    epochs = 50
    print(f"[✓] Training Atlas with {len(pairs)} pair(s) over {epochs} epochs... (This may take a moment)")
    
    total_pairs = len(pairs)
    learned_pairs = 0

    for epoch in range(1, epochs + 1):
        epoch_learned = 0
        for prompt, response in pairs:
            result = brain.learn_pair(prompt, response)
            if result is not None:
                epoch_learned += 1
        
        if epoch == 1:
            learned_pairs = epoch_learned
            
        print(f"[✓] Epoch {epoch}/{epochs} complete - Learned: {epoch_learned}/{total_pairs}")

    if learned_pairs == 0:
        print(f"[✗] No pairs from '{source_label}' could be used for training.", file=sys.stderr)
        sys.exit(1)

    brain.save()
    print(f"[✓] Training complete! Learned from {learned_pairs}/{total_pairs} pair(s). Model saved.")
    brain.generate_report()


def run_document_training(brain, file_path=None, url=None):
    """
    Train the model from a local document file or remote URL.
    """
    from atlas.document_loader import DocumentLoadError, load_from_local, load_from_url

    try:
        if file_path:
            text = load_from_local(file_path)
            source_label = file_path
        else:
            text = load_from_url(url)
            source_label = url

        train_brain_from_text(brain, text, source_label)
        sys.exit(0)
    except DocumentLoadError as exc:
        print(f"[✗] Document training error: {exc}", file=sys.stderr)
        sys.exit(1)
    except requests.exceptions.RequestException as exc:
        print(f"[✗] Network error during document download: {exc}", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        print(f"[✗] An unexpected error occurred during document training: {exc}", file=sys.stderr)
        sys.exit(1)


def main():
    """
    Main function to run the Atlas conversational AI.
    Initializes the AtlasBrain, handles user interaction,
    learning, response generation, and persistent saving.
    Also handles web scraping and document training features.
    """
    parser = argparse.ArgumentParser(description="Run the Atlas conversational AI in different modes.")
    parser.add_argument('--training', action='store_true', help='Only learn from input, do not generate responses.')
    parser.add_argument('--production', action='store_true', help='Only generate responses, do not learn.')
    parser.add_argument('--dual', action='store_true', help='Both learn and generate responses (default).')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to the configuration YAML file.')
    parser.add_argument('--train', action='store_true', help='Train from a local document (use with --file).')
    parser.add_argument('--file', type=str, help='Local document path (PDF, DOCX, TXT, MD) for training.')
    parser.add_argument('--url', type=str, help='Remote document URL (PDF, DOCX, TXT, MD) for training.')
    parser.add_argument('--scraping', type=str, help='Scrape content from the given URL, sanitize it, and use it to train the AI.')
    parser.add_argument('--fast', action='store_true', help='Enable fast greedy generation (recommended with --production).')
    args = parser.parse_args()

    if args.train and not args.file:
        print("Error: --train requires --file <path>.", file=sys.stderr)
        sys.exit(1)

    training_sources = [bool(args.file), bool(args.url), bool(args.scraping)]
    if sum(training_sources) > 1:
        print("Error: --file, --url, and --scraping are mutually exclusive.", file=sys.stderr)
        sys.exit(1)

    # Load configuration
    config = load_config(args.config)
    brain = AtlasBrain(model_path="atlas_model.npz", vocab_path="atlas_vocab.pkl", config=config)

    # --- Document Training Feature ---
    if args.file or args.url:
        run_document_training(brain, file_path=args.file, url=args.url)
    # --- End Document Training Feature ---

    # --- Web Scraping and Training Feature ---
    if args.scraping:
        print(f"[✓] Attempting to scrape and train from: {args.scraping}")
        try:
            # Check for dependencies
            import requests
            from bs4 import BeautifulSoup
        except ImportError:
            print("Error: 'requests' or 'beautifulsoup4' library not found.", file=sys.stderr)
            print("Please install them using: pip install requests beautifulsoup4 lxml", file=sys.stderr)
            sys.exit(1)

        try:
            if not web_scraper.check_robots_txt(args.scraping):
                print("[✗] Scraping disallowed by robots.txt. Exiting.")
                sys.exit(1)

            user_confirm = input("Do you want to scrape this page and train the AI? (y/n): ").lower()
            if user_confirm != 'y':
                print("[✗] User cancelled. Exiting.")
                sys.exit(0)

            print("[✓] Scraping and sanitizing text... (This may take a moment)")
            
            # Extract scraping configuration
            scraping_config = config.get('scraping', {})
            scraped_text = web_scraper.scrape_article_text(args.scraping, scraping_config)
            
            if not scraped_text.strip():
                print("[✗] No significant text content found on the page. Exiting.")
                sys.exit(1)

            train_brain_from_text(brain, scraped_text, args.scraping)
            sys.exit(0)

        except requests.exceptions.RequestException as e:
            print(f"[✗] Network error during scraping: {e}", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            print(f"[✗] An unexpected error occurred during scraping or training: {e}", file=sys.stderr)
            sys.exit(1)
    # --- End Web Scraping Feature ---

    if args.fast:
        if not args.production:
            print("Warning: --fast is intended for use with --production.", file=sys.stderr)
        brain.temperature = 0.01
        brain.top_k = 1
        brain.beam_width = 0
        brain.top_p = 1.0
        brain.transformer.temperature = 0.01
        brain.transformer.top_k = 1
        brain.transformer.beam_width = 0
        brain.transformer.top_p = 1.0
        print("Fast mode enabled: greedy sampling, no beam search.")

    # Determine the operating mode for interactive chat
    mode = 'dual'
    if args.training:
        mode = 'training'
    elif args.production:
        mode = 'production'
    elif args.dual:
        mode = 'dual' # Explicitly set if --dual is passed, though it's the default

    print(f"Atlas AI starting in {mode.upper()} mode.")
    print("Welcome to Atlas! Type 'quit' or 'exit' to save and exit.")

    interaction_count = 0
    auto_save_interval = 5

    try:
        while True:
            user_input = input("You: ")

            if user_input.lower() in ['quit', 'exit']:
                print("Atlas: Goodbye!")
                brain.save()
                break
            
            if mode in ['training', 'dual']:
                learn_result = brain.learn(user_input)
                if mode == 'training':
                    if learn_result is not None:
                        sys.stdout.write(".")
                        sys.stdout.flush()
                    else:
                        sys.stdout.write("x") # Indicate skipped learning
                        sys.stdout.flush()
                    continue # Skip response generation in training mode

            if mode in ['production', 'dual']:
                try:
                    response = brain.respond(prompt=user_input)
                except Exception as e:
                    print(f"Atlas: An error occurred while generating a response: {e}")
                    response = "I'm having trouble responding right now, please try again."
                
                print(f"Atlas: {response}")
            elif mode == 'training':
                # This branch should ideally not be reached due to 'continue' above,
                # but as a safeguard, if somehow reached, don't print "Atlas: "
                pass


            interaction_count += 1
            if interaction_count % auto_save_interval == 0:
                if mode != 'production': # Only save if learning happened
                    brain.save()
                    if mode == 'training':
                        print("\n(Auto-saved model during training)")


    except KeyboardInterrupt:
        print("\nAtlas: Interrupted. Saving memory before exiting...")
        brain.save()
        brain.generate_report()
        sys.exit(0)
    except EOFError: # Handles Ctrl+D
        print("\nAtlas: EOF detected. Saving memory before exiting...")
        brain.save()
        brain.generate_report()
        sys.exit(0)

    # Generate report on normal interactive exit
    brain.generate_report()

if __name__ == "__main__":
    main()
