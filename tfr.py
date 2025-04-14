#----------------
#Machine Translation
#----------------
import pandas as pd
from datasets import Dataset
from transformers import (
    MarianTokenizer,
    MarianMTModel,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq
)

# Config
model_name = "Helsinki-NLP/opus-mt-en-fr"
batch_size = 4
epochs = 3

# Load dataset (CSV with 'src' and 'tgt' columns)
df = pd.read_csv("translation.txt",sep="\t")  # Your dataset here
print(df)
dataset = Dataset.from_pandas(df)

# Load model and tokenizer
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

# Preprocessing
def preprocess(example):
    model_inputs = tokenizer(example["source"], max_length=128, truncation=True, padding="max_length")
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(example["target"], max_length=128, truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_dataset = dataset.map(preprocess, batched=True)

# Training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="./finetuned-en-fr",
    evaluation_strategy="no",
    learning_rate=5e-5,
    per_device_train_batch_size=batch_size,
    num_train_epochs=epochs,
    weight_decay=0.01,
    save_total_limit=1,
    logging_dir='./logs',
    report_to="none"  # Disable WandB, Comet, etc.
)

# Data collator
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# Fine-tune the model
trainer.train()

# Save model
model.save_pretrained("finetuned-marian-en-fr")
tokenizer.save_pretrained("finetuned-marian-en-fr")

from transformers import (MarianMTModel,MarianTokenizer,Seq2SeqTrainingArguments,Seq2SeqTrainer,DataCollatorForSeq2Seq)

from datasets import Dataset

data=pd.read_csv("translation.txt",sep="\t")

model_name='Helsinki-NLP/opus-mt-en-fr'

batch_size=4
epoch=3

model=MarianMTModel.from_pretrained(model_name)

tokenizer=MarianTokenizer.from_pretrained(model_name)

def preprocess(example):
  print(example)
  model_inputs=tokenizer(example['source'],max_length=128,truncation=True,padding='max_length')
  with tokenizer.as_target_tokenizer():
    labels=tokenizer(example['target'],max_length=128,truncation=True,padding='max_length')
  model_inputs['labels']=labels['input_ids']
  return model_inputs

dataset=Dataset.from_pandas(data)
tokenized_data=dataset.map(preprocess)

training_args=Seq2SeqTrainingArguments(output_dir='./123',report_to="none",evaluation_strategy='no',learning_rate=5e-5,num_train_epochs=3,per_device_train_batch_size=4,weight_decay=0.01)

datacollator=DataCollatorForSeq2Seq(tokenizer, model=model)

trainer=Seq2SeqTrainer(model=model,tokenizer=tokenizer,data_collator=datacollator,train_dataset=tokenized_data,args=training_args)

trainer.train()

model.save_pretrained("finetuned-marian-en-fr")
tokenizer.save_pretrained("finetuned-marian-en-fr")

from transformers import MarianMTModel, MarianTokenizer

def translate(text, model_dir="finetuned-marian-en-fr"):
    tokenizer = MarianTokenizer.from_pretrained(model_dir)
    model = MarianMTModel.from_pretrained(model_dir)
    encoded = tokenizer(text, return_tensors='pt',padding=True,max_length=True)
    generated = model.generate(**encoded)
    return tokenizer.decode(generated[0],skip_special_tokens=True)

# Example
print(translate("Hi, I am Partha"))

import nltk
nltk.download('punkt')
import evaluate

# Load BLEU metric
bleu = evaluate.load("bleu")

# Example list of predictions and references
predictions = ["Je t'aime beaucoup."]
references = [["Je vous aime beaucoup."]]  # note: reference must be a list of lists

# Compute BLEU
results = bleu.compute(predictions=predictions, references=references)
print(f"BLEU score: {results['bleu']:.4f}")
