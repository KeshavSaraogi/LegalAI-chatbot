from transformers import BertTokenizer, BertForQuestionAnswering
import torch

# Load LegalBERT
tokenizer = BertTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased")
model = BertForQuestionAnswering.from_pretrained("nlpaueb/legal-bert-base-uncased")

def get_legal_answer(question, context):
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
