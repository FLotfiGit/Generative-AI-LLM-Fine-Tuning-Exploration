pip install transformers datasets torch


import torch
from transformers import GPT2Tokenizer, GPT2ForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
'''
Fine-Tuning GPT-2 on a Sentiment Dataset
@ Flotfi 
Feb 2025
'''
# Load GPT-2 tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token  # GPT-2 does not have a pad token

# Load sentiment dataset (IMDb as an example)
dataset = load_dataset("imdb")

# Tokenization function
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

# Tokenize dataset
tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Load GPT-2 model for sequence classification
model = GPT2ForSequenceClassification.from_pretrained("gpt2", num_labels=2)  # 2 labels (positive/negative)

# Training arguments
training_args = TrainingArguments(
    output_dir="./gpt2_sentiment_model",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=100,
    save_total_limit=2,
    load_best_model_at_end=True,
)

# Trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    tokenizer=tokenizer,
)

# Fine-tune the model
trainer.train()

# Save trained model
trainer.save_model("./gpt2_sentiment_finetuned")

print("Fine-tuning completed and model saved!")
