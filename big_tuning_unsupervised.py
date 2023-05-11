import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (BigBirdForMaskedLM, BigBirdTokenizer, AutoTokenizer,
                          BigBirdConfig, DataCollatorForLanguageModeling,
                          Trainer, TrainingArguments)
import os


class LineByLineTextDatasetLazy(Dataset):
    def __init__(self, tokenizer, file_path, block_size):
        self.tokenizer = tokenizer
        self.file_path = file_path
        self.block_size = block_size

        with open(file_path, encoding="utf-8") as f:
            self.num_lines = sum(1 for _ in f)

    def __len__(self):
        return self.num_lines

    def __getitem__(self, idx):
        with open(self.file_path, encoding="utf-8") as f:
            line = next(x for i, x in enumerate(f) if i == idx).strip()
        tokenized_input = self.tokenizer(line, max_length=self.block_size, truncation=True, return_tensors="pt")
        input_ids = tokenized_input['input_ids'].squeeze()
        return input_ids


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("device: ", device)

MODEL_NAME = 'google/bigbird-roberta-base'

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
config = BigBirdConfig.from_pretrained(MODEL_NAME)
print("Loading Model...")
model = BigBirdForMaskedLM.from_pretrained(MODEL_NAME).to(device)

print("Model loaded...")

train_file = "train_data/train_data.txt"
train_dataset = LineByLineTextDatasetLazy(tokenizer=tokenizer, file_path=train_file, block_size=config.block_size)
print("Loaded Training set...")

eval_file = "train_data/test_data.txt"
eval_dataset = LineByLineTextDatasetLazy(tokenizer=tokenizer, file_path=eval_file, block_size=config.block_size)
print("Loading Test set...")

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)
print("Collator created...")

# output_dir = "./artifacts/model_output_full_opt"
# training_args = TrainingArguments(
#     # your existing training arguments
# )

# Set up training arguments
output_dir = "./artifacts/model_output_full_opt" 
training_args = TrainingArguments(
    output_dir=output_dir,
    overwrite_output_dir=True,
    num_train_epochs=5,
    per_device_train_batch_size=16,
    save_steps=1_000,
    save_total_limit=5,
    logging_dir="./artifacts/logs_output_full_opt",
    logging_steps=1_000,
    learning_rate=5e-6,
    weight_decay=0.01,
    evaluation_strategy="steps",
    load_best_model_at_end=True,
)
print("TrainingArguments created...")

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset
)

trainer.train()
