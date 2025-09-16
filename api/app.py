import uuid

from flask import Flask, request, jsonify

from rag.smart_fashion_assistant import SmartFashionAssistant   

import api.db as db


app = Flask(__name__)


@app.route("/question", methods=["POST"])
def handle_question():
    print("Received request:", request.json)
    data = request.json
    question = data["question"]

    if not question:
        return jsonify({"error": "No question provided"}), 400

    conversation_id = str(uuid.uuid4())

    sfa = SmartFashionAssistant()
    answer, answer_metadata, answer_raw_response = sfa.rag(question)  # Call once to get the raw response
    eval_response, eval_metadata, eval_raw_response = sfa.evaluate_relevance(question,answer)

    db.save_conversation(
        conversation_id=conversation_id,
        question=question,
        answer=answer,
        answer_metadata=answer_metadata,
        model_used=answer_raw_response.response_metadata["model"],
        evaluation=eval_response,
        evaluation_metadata=eval_metadata
    )

    result = {
        "conversation_id": conversation_id,
        "question": question,
        "answer": answer,
    }
    return jsonify(result)

@app.route("/feedback", methods=["POST"])
def handle_feedback():
    data = request.json
    conversation_id = data["conversation_id"]
    feedback = data["feedback"]

    if not conversation_id or feedback not in [1, -1]:
        return jsonify({"error": "Invalid input"}), 400
    
    db.save_feedback(
        conversation_id=conversation_id,
        feedback=feedback,
    )

    result = {
        "message": f"Feedback received for conversation {conversation_id}: {feedback}"
    }
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, port=5001)