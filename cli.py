from rag.smart_fashion_assistant import SmartFashionAssistant
import questionary

def main():
    sfa = SmartFashionAssistant()
    #question = "I am a women and need to dress for an indian wedding?"

    print("Welcome to the Fashion Assistant app!.You can ask questions related to fashion recommendation and products")
    print("You can exit the program at any time when prompted.")
    while True:
        question = questionary.text("What is you Fashion search query?").ask()
        answer = sfa.rag(question)
        print("Answer:")
        print(answer)
        print()
        feedback = questionary.select(
            "How would you rate this response?",
            choices=["+1 (Positive)", "-1 (Negative)", "Pass (Skip feedback)"],
        ).ask()
        if feedback == "+1 (Positive)":
            print("Thank you for your positive feedback!")
        elif feedback == "-1 (Negative)":
            print("Thank you for your feedback! We'll strive to improve.")
        else:
            print("Feedback skipped.")

        continue_prompt = questionary.confirm("Do you want to continue?").ask()
        if not continue_prompt:
            print("Thank you for using the app. Goodbye!")
            break
    


if __name__ == "__main__":
    main()  