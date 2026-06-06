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

def main():
    """
    Main function to run the Atlas conversational AI.
    Initializes the AtlasBrain, handles user interaction,
    learning, response generation, and persistent saving.
    Also handles the new web scraping and training feature.
    """
    parser = argparse.ArgumentParser(description="Run the Atlas conversational AI in different modes.")
    parser.add_argument('--training', action='store_true', help='Only learn from input, do not generate responses.')
    parser.add_argument('--production', action='store_true', help='Only generate responses, do not learn.')
    parser.add_argument('--dual', action='store_true', help='Both learn and generate responses (default).')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to the configuration YAML file.')
    parser.add_argument('--scraping', type=str, help='Scrape content from the given URL, sanitize it, and use it to train the AI.')
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    brain = AtlasBrain(model_path="atlas_model.npz", vocab_path="atlas_vocab.pkl", config=config)

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

            print("[✓] Training Atlas with the new content... (This may take a moment)")
            # Split text into chunks for learning
            # A simple chunking by character count, trying to respect word boundaries
            words = scraped_text.split()
            current_chunk = []
            current_chunk_len = 0
            
            for word in words:
                # Add 1 for space
                if current_chunk_len + len(word) + 1 > MAX_CHUNK_SIZE and current_chunk:
                    brain.learn(" ".join(current_chunk))
                    current_chunk = [word]
                    current_chunk_len = len(word)
                else:
                    current_chunk.append(word)
                    current_chunk_len += len(word) + 1
            
            if current_chunk: # Learn the last chunk
                brain.learn(" ".join(current_chunk))

            brain.save()
            print("[✓] Training complete! Model saved. Exiting.")
            sys.exit(0)

        except requests.exceptions.RequestException as e:
            print(f"[✗] Network error during scraping: {e}", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            print(f"[✗] An unexpected error occurred during scraping or training: {e}", file=sys.stderr)
            sys.exit(1)
    # --- End Web Scraping Feature ---

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
        sys.exit(0)
    except EOFError: # Handles Ctrl+D
        print("\nAtlas: EOF detected. Saving memory before exiting...")
        brain.save()
        sys.exit(0)

if __name__ == "__main__":
    main()