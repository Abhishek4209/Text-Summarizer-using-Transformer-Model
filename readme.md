# Text Summarizer using Transformer Model

This project implements a Text Summarizer using Hugging Face Transformers, specifically using the `DataCollatorForSeq2Seq`, `AutoModelForSeq2SeqLM`, `Seq2SeqTrainingArguments`, and `Seq2SeqTrainer`. The model is trained to generate concise summaries of long text inputs.

## Table of Contents
- [Project Overview](#project-overview)
- [Installation](#installation)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Usage](#usage)
- [References](#references)

## Project Overview

This project demonstrates how to fine-tune a pre-trained sequence-to-sequence (Seq2Seq) model for text summarization tasks. The training process leverages Hugging Face’s `transformers` library, which makes it easy to implement modern transformer-based models like BART, T5, or Pegasus for summarization.

The workflow includes:
1. Loading a pre-trained Seq2Seq model.
2. Fine-tuning the model on a custom dataset.
3. Summarizing input texts and evaluating the performance of the model.

## Installation

To run this project, you'll need Python and the following dependencies installed:

```bash
pip install torch transformers datasets evaluate
```

### Dataset
- You can use any summarization dataset, such as the CNN/DailyMail dataset or your custom dataset. Ensure that your dataset has two main columns:

- source: The text to be summarized.
- target: The reference summary for the source text.
- Here’s an example of loading the dataset:

```bash

from datasets import load_dataset
news = load_dataset('multi_news', split = 'test')
```


### Model Architecture
This project uses a pre-trained model from the Hugging Face transformers library. For example, you can use BART (facebook/bart-large-cnn) or T5 (t5-base) as the base model.

### Key Components:
- `AutoModelForSeq2SeqLM`: Automatically loads a pre-trained Seq2Seq model.
- `Seq2SeqTrainer`: Handles the fine-tuning of Seq2Seq models with built-in support for distributed training and mixed precision.
- `DataCollatorForSeq2Seq`: Pads and collates the input sequences for training.


### Training
The training process is handled by Hugging Face's Seq2SeqTrainer. Here’s how the model is trained:

```bash

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('t5-small')


prefix = "summarize: "

def preprocess_function(examples):
    inputs = [prefix + doc for doc in examples["document"]]
    model_inputs = tokenizer(inputs, max_length=1024, truncation=True)
    labels = tokenizer(text=examples["summary"], max_length=128, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized = train_news.map(preprocess_function , batched=True)



from transformers import DataCollatorForSeq2Seq , AutoModelForSeq2SeqLM , Seq2SeqTrainingArguments, Seq2SeqTrainer

data_collator = DataCollatorForSeq2Seq(tokenizer = tokenizer , model = 't5-small')

model = AutoModelForSeq2SeqLM.from_pretrained('t5-small')

training_args = Seq2SeqTrainingArguments(
output_dir="./results",
evaluation_strategy="epoch",
learning_rate=2e-5,
per_device_train_batch_size=10,
per_device_eval_batch_size=10,
weight_decay=0.01,
save_total_limit=3,
num_train_epochs=10,
# fp16=True,
)

trainer = Seq2SeqTrainer(
    model = model,
    args = training_args,
    train_dataset = tokenized['train'],
    eval_dataset = tokenized['test'],
    tokenizer = tokenizer,
    data_collator = data_collator
)

trainer.train()
```


### Evaluation
Once training is complete, you can evaluate the model using various metrics, including ROUGE (Recall-Oriented Understudy for Gisting Evaluation).



### Results
After training, the model can be used to summarize new input text.

```bash
def pred(document):
  device = model.device
  tokenized= tokenizer([document], return_tensors = 'pt')
  print(tokenized)
  tokenized ={k:v.to(device) for k , v in tokenized.items()}
  results = model.generate(**tokenized, max_length = 128)
  results = results.to('cpu')
  pred = tokenizer.decode(results[0])
  return pred

```


### Usage
- Train the Model: Use the provided code to fine-tune the pre-trained Seq2Seq model.
- Evaluate the Model: Measure its performance using your preferred metrics.
- Summarize Text: Once trained, use the model to summarize new text inputs.


### References

- Hugging Face Transformers Documentation
- 
- Seq2SeqTrainer Documentation


- [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers/index)

- [Seq2SeqTrainer Documentation](https://huggingface.co/docs/transformers/en/main_classes/trainer)

