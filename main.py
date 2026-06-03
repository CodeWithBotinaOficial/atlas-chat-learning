"""
main.py

This is the main entry point for the Atlas conversational AI.
It handles the interactive loop with the user.
"""

from atlas.brain import AtlasBrain

def main():
    """
    Main function to run the Atlas conversational AI.
    Initializes the AtlasBrain and enters an interactive loop
    to receive user input and provide responses.
    """
    brain = AtlasBrain()
    print("Welcome to Atlas! Type 'quit' to exit.")

    while True:
        user_input = input("You: ")
        if user_input.lower() == 'quit':
            print("Atlas: Goodbye!")
            break
        
        # For now, Atlas just echoes input and indicates learning
        # In future steps, this will involve more complex processing
        brain.learn(user_input)
        response = brain.respond(user_input)
        print(f"Atlas: {response}")

if __name__ == "__main__":
    main()
