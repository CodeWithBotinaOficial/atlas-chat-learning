"""
main.py

This is the main entry point for the Atlas conversational AI.
It handles the interactive loop with the user, incorporating
n-gram based learning and persistent memory.
"""

from atlas.brain import AtlasBrain
import sys

def main():
    """
    Main function to run the Atlas conversational AI.
    Initializes the AtlasBrain, handles user interaction,
    learning, response generation, and persistent saving.
    """
    model_file = "atlas_memory.pkl"
    brain = AtlasBrain(model_file=model_file)
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
