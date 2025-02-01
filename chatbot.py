from transformers import BertTokenizer, BertForQuestionAnswering
import torch

tokenizer = BertTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased")
model = BertForQuestionAnswering.from_pretrained("nlpaueb/legal-bert-base-uncased")

def getLegalAnswers(question, context):
    inputs = tokenizer(question, context, return_tensors="pt", truncation=True, max_length=512)
    
    with torch.no_grad():
        outputs = model(**inputs)
        
    answer_start = torch.argmax(outputs.start_logits)
    answer_end = torch.argmax(outputs.end_logits) + 1
    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][answer_start:answer_end]))
    
    return answer

