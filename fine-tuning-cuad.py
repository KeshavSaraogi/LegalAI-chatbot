import torch
from datasets import load_dataset
from transformers import BertTokenizer, BertForQuestionAnswering, Trainer, TrainingArguments, default_data_collator
from transformers import DataCollatorWithPadding

import os

# Preprocessing script (preprocess.py)
def preprocess_data(example):
    if "context" not in example or "question" not in example or "answers" not in example:
        print(f"Skipping example due to missing keys: {example.keys()}")
        return {"input_ids": [0], "attention_mask": [0], "labels": [0]}  # Avoid empty tensors
    
    # Ensure valid input values
    context = example["context"] if example["context"] else "No context available."
    question = example["question"] if example["question"] else "What is this document about?"

    # Ensure answers exist
    if not example["answers"]["text"]:
        answer_text = "No answer available"
    else:
        answer_text = example["answers"]["text"][0]

    # Print example before tokenization for debugging
    print(f"Processing Example:\nContext: {context}\nQuestion: {question}\nAnswer: {answer_text}")

    inputs = tokenizer(
        context, question,
        truncation=True, padding="max_length", max_length=512, return_tensors="pt"
    )

    labels = tokenizer(
        answer_text,
        truncation=True, padding="max_length", max_length=128, return_tensors="pt"
    )

    # Ensure `input_ids` are not empty
    if inputs["input_ids"].numel() == 0:
        print(f"Skipping example with empty input_ids: {example}")
        return {"input_ids": [0], "attention_mask": [0], "labels": [0]}  # Placeholder to prevent failure

    return {
        "input_ids": inputs["input_ids"].squeeze().tolist(),
        "attention_mask": inputs["attention_mask"].squeeze().tolist(),
        "labels": labels["input_ids"].squeeze().tolist()
    }



# Clear Hugging Face cache to force dataset reload
os.system("rm -rf ~/.cache/huggingface/datasets")

# Load CUAD dataset from local JSON files
cuad = load_dataset("json", data_files={"train": "/Users/keshavsaraogi/data/cuad/CUAD_v1.json"})

# Ensure dataset is properly loaded
print(f"Train Dataset Size Before Processing: {len(cuad['train'])}")
if len(cuad["train"]) > 1:
    cuad_split = cuad["train"].train_test_split(test_size=0.1)
    train_dataset = cuad_split["train"]
    eval_dataset = cuad_split["test"]
else:
    print("Dataset too small for splitting. Using full dataset for training.")
    train_dataset = cuad["train"]
    eval_dataset = None  # No evaluation set

print("---------------------------------------------")
print("First few raw examples before preprocessing:")
for i in range(min(5, len(train_dataset))):  # Print first 5 examples
    print(train_dataset[i])

# Apply preprocessing
# Remove non-essential columns
columns_to_keep = ["context", "question", "answers"]
train_dataset = train_dataset.remove_columns([col for col in train_dataset.column_names if col not in columns_to_keep])
if eval_dataset is not None:
    eval_dataset = eval_dataset.remove_columns([col for col in eval_dataset.column_names if col not in columns_to_keep])


# Verify dataset structure
print(f"Train Dataset Size After Processing: {len(train_dataset)}")
if eval_dataset is not None:
    print(f"Eval Dataset Size After Processing: {len(eval_dataset)}")


# Verify dataset structure
print(f"Train Dataset Size After Processing: {len(train_dataset)}")
if eval_dataset is not None:
    print(f"Eval Dataset Size After Processing: {len(eval_dataset)}")

if len(train_dataset) == 0:
    raise ValueError("Train dataset is empty! Check data loading and preprocessing.")
if eval_dataset is not None and len(eval_dataset) == 0:
    raise ValueError("Eval dataset is empty! Check data loading and preprocessing.")

# Save processed dataset
torch.save(train_dataset, "processed_cuad_train.pt")
if eval_dataset is not None:
    torch.save(eval_dataset, "processed_cuad_eval.pt")
print("Preprocessing complete! Train and eval datasets saved.")

# Initialize LegalBERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased")
model = BertForQuestionAnswering.from_pretrained("nlpaueb/legal-bert-base-uncased")

# Training script (train.py)
training_args = TrainingArguments(
    output_dir="./legalbert_finetuned",
    evaluation_strategy="epoch" if eval_dataset is not None else "no",
    save_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=500,
    save_total_limit=2,
    remove_unused_columns=False
)

data_collator = DataCollatorWithPadding(tokenizer)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset if eval_dataset is not None else None,
    data_collator=data_collator
)


# Train model
trainer.train()

# Save the fine-tuned model
model.save_pretrained("./legalbert_finetuned")
tokenizer.save_pretrained("./legalbert_finetuned")

print("Fine-tuning complete! Model saved to ./legalbert_finetuned")
