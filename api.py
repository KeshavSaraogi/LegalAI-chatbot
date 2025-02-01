from flask import Flask, request, jsonify
from chatbot import get_legal_answer  # Ensure correct function name

app = Flask(__name__)

# Define the legal context before calling it
legal_context = """
The Fair Housing Act prohibits discrimination in renting or buying homes based on race, color, religion, sex, national origin, disability, or familial status. 
Landlords cannot refuse to rent, charge different rents, or impose different terms based on these factors. 
This law also covers mortgage lending practices, housing advertisements, and the construction of accessible housing.
If a landlord discriminates, the affected person has the right to file a complaint with the U.S. Department of Housing and Urban Development (HUD).
"""

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    question = data.get("question")

    # Make sure the function name matches the one in chatbot.py
    answer = get_legal_answer(question, legal_context)  
    return jsonify({"answer": answer})

if __name__ == "__main__":
    app.run(debug=True)
