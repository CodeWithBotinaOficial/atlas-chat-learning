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
    # Updated to use new model and vocab paths
    brain = AtlasBrain(model_path="atlas_model.npz", vocab_path="atlas_vocab.pkl")
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
            
            brain.learn(user_input)
            response = brain.respond(prompt=user_input)
            print(f"Atlas: {response}")

            interaction_count += 1
            if interaction_count % auto_save_interval == 0:
                brain.save()

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
