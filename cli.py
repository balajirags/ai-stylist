from rag.smart_fashion_assistant import SmartFashionAssistant
import questionary
import requests
import json
import uuid


def ask_question(url, question):
    data = {"question": question}
    response = requests.post(f"{url}/question", json=data)
    return response.json()


def send_feedback(url, conversation_id, feedback):
    feedback_data = {"conversation_id": conversation_id, "feedback": feedback}
    response = requests.post(f"{url}/feedback", json=feedback_data)
    return response.status_code


def main1():
    sfa = SmartFashionAssistant()
    base_url = "http://localhost:5001"
    print("Welcome to the Fashion Assistant app!.You can ask questions related to fashion recommendation and products")
    print("You can exit the program at any time when prompted.")
    while True:
        question = questionary.text("What is you Fashion search query?").ask()
        #answer, metadata, raw_response = sfa.rag(question)
        print("Answer:")
        print(answer)
        print()
        print("Usage Metadata:")
        print(f"Input token - {metadata['input_tokens']}, Output tokens - {metadata['output_tokens']}, Total tokens - {metadata['total_tokens']}")
        print(f"model:{raw_response.response_metadata['model']}")
        print()

        print(" ----- Evaluation ----")
        eval_response, metadata, raw_response = sfa.evaluate_relevance(question, answer)
        print("Evaluation:")
        print(eval_response)
        print()
        print("metadata:")
        print(f"Input token - {metadata['input_tokens']}, Output tokens - {metadata['output_tokens']}, Total tokens - {metadata['total_tokens']}")
        print(" -----------------")

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
    

def main():
    sfa = SmartFashionAssistant()
    base_url = "http://localhost:5001"
    print("Welcome to the Fashion Assistant app!.You can ask questions related to fashion recommendation and products")
    print("You can exit the program at any time when prompted.")
    while True:
        question = questionary.text("What is you Fashion search query?").ask()
        response = ask_question(base_url, question)
        print("\nAnswer:", response.get("answer", "No answer provided"))
        conversation_id = response.get("conversation_id", str(uuid.uuid4()))
        
        feedback = questionary.select(
            "How would you rate this response?",
            choices=["+1 (Positive)", "-1 (Negative)", "Pass (Skip feedback)"],
        ).ask()

        if feedback != "Pass (Skip feedback)":
            feedback_value = 1 if feedback == "+1 (Positive)" else -1
            status = send_feedback(base_url, conversation_id, feedback_value)
            print(f"Feedback sent. Status code: {status}")
        else:
            print("Feedback skipped.")
            
        continue_prompt = questionary.confirm("Do you want to continue?").ask()
        if not continue_prompt:
            print("Thank you for using the app. Goodbye!")
            break


if __name__ == "__main__":
    main()  