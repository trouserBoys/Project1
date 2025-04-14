import pandas as pd
from datasets import Dataset
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)
import torch

# Load and prepare dataset
df = pd.read_csv("sentiment.txt")  # Replace with your file path
label_map = {"negative": 0, "positive": 1}
df["label"] = df["label"].map(label_map)

dataset = Dataset.from_pandas(df)

# Split into train and test
dataset = dataset.train_test_split(test_size=0.1)

# Load tokenizer and model
model_name = "distilbert-base-uncased"
tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)
model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Preprocessing
def preprocess(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=256)

tokenized = dataset.map(preprocess, batched=True)

# Training arguments
training_args = TrainingArguments(
    output_dir="./finetuned-sentiment-model",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="epoch",
    report_to="none"  # disables wandb
)

# Data collator
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["test"],
    tokenizer=tokenizer,
    data_collator=data_collator
)

# Train
trainer.train()

# Save model and tokenizer
model.save_pretrained("finetuned-sentiment-model")
tokenizer.save_pretrained("finetuned-sentiment-model")


from transformers import TextClassificationPipeline

pipe = TextClassificationPipeline(model=model, tokenizer=tokenizer, top_k=None)

print(pipe("I love this!"))
print(pipe("This is bad."))