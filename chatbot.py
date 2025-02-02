from transformers import BertTokenizer, BertForQuestionAnswering
import torch

# Load fine-tuned LegalBERT model
model_path = "./legalbert_finetuned"
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForQuestionAnswering.from_pretrained(model_path)

def get_legal_answer(question):
    """
    Processes the question using the fine-tuned LegalBERT model.
    Returns the extracted legal answer.
    """

    # Placeholder context (Replace this with actual legal text retrieval)
    context = "Legal information related to housing, rights, and compliance."

    # Tokenize input
    inputs = tokenizer(question, context, return_tensors="pt", truncation=True, max_length=1024)

    with torch.no_grad():
        outputs = model(**inputs)

    answer_start = torch.argmax(outputs.start_logits)
    answer_end = torch.argmax(outputs.end_logits) + 1

    if answer_start >= answer_end:
        return "I'm sorry, I couldn't find a relevant legal answer."

    answer_tokens = inputs["input_ids"][0][answer_start:answer_end]
    answer = tokenizer.decode(answer_tokens, skip_special_tokens=True)

    return answer if answer.strip() else "I'm sorry, I couldn't find a relevant legal answer."
