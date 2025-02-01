from flask import Flask, request, jsonify
from chatbot import getLegalAnswers

app = Flask(__name__)

legalContext = "The Fair Housing Act prohibits discrimination in renting or buying homes based on race, color, religion, sex, national origin, disability, or familial status. Landlords cannot refuse to rent, charge different rents, or impose different terms based on these factors."


@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    question = data.get("question")
    
    answer = getLegalAnswers(question, legalContext)
    return jsonify({"answer" : answer})

if __name__ == "__main__":
    app.run(debug=True)