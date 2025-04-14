import pandas as pd
from datasets import Dataset
from transformers import BartForConditionalGeneration, BartTokenizer, TrainingArguments, Trainer


data = pd.read_csv("summary.txt",sep = '\t')
sample_size = min(100, len(data))
data = data.sample(n=sample_size, random_state=42)
dataset = Dataset.from_pandas(data)

model_name = "sshleifer/distilbart-cnn-12-6"
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)

# Preprocess
def preprocess_function(examples):
    inputs = tokenizer(examples["text"], max_length=512, truncation=True, padding="max_length")
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["summary"], max_length=128, truncation=True, padding="max_length")
    inputs["labels"] = labels["input_ids"]
    return inputs

tokenized_dataset = dataset.map(preprocess_function, batched=True)


training_args = TrainingArguments(
    output_dir="./finetuned_bart_summary_cpu",
    evaluation_strategy="no",  # skip eval
    learning_rate=5e-5,
    per_device_train_batch_size=2,  # keep low for CPU
    num_train_epochs=3,  # 1 epoch for testing
    logging_strategy="no",
    save_strategy="no",
    report_to="none"  # no wandb or hub
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer
)


trainer.train()

model.save_pretrained("finetuned-bart-summary-cpu")
tokenizer.save_pretrained("finetuned-bart-summary-cpu")

from transformers import pipeline

def summarize(text, model_path="finetuned-bart-summary-cpu"):
    summarization_model = pipeline(
        "summarization",
        model=model_path,
        tokenizer=model_path
    )

    # Generate the summary
    summary = summarization_model(text)[0]['summary_text']
    return summary

text= ""
print(summarize(text))