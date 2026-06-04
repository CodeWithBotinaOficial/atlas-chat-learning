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


def main():
    """
    Main function to run the Atlas conversational AI.
    Initializes the AtlasBrain, handles user interaction,
    learning, response generation, and persistent saving.
    """
    parser = argparse.ArgumentParser(description="Run the Atlas conversational AI in different modes.")
    parser.add_argument('--training', action='store_true', help='Only learn from input, do not generate responses.')
    parser.add_argument('--production', action='store_true', help='Only generate responses, do not learn.')
    parser.add_argument('--dual', action='store_true', help='Both learn and generate responses (default).')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to the configuration YAML file.')
    args = parser.parse_args()

    # Determine the operating mode
    mode = 'dual'
    if args.training:
        mode = 'training'
    elif args.production:
        mode = 'production'
    elif args.dual:
        mode = 'dual' # Explicitly set if --dual is passed, though it's the default

    print(f"Atlas AI starting in {mode.upper()} mode.")

    # Load configuration
    config = load_config(args.config)

    brain = AtlasBrain(model_path="atlas_model.npz", vocab_path="atlas_vocab.pkl", config=config)
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