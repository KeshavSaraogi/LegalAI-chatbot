from flask import Flask, request, jsonify
from chatbot import get_legal_answer  # Ensure function is correctly imported

app = Flask(__name__)

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    question = data.get("question")

    if not question:
        return jsonify({"error": "No question provided"}), 400

    # Get answer from chatbot model
    answer = get_legal_answer(question)

    return jsonify({"answer": answer})

if __name__ == "__main__":
    app.run(debug=True)
