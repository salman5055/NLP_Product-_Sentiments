import os
import pandas as pd
import torch

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)

# ==========================================
# MODEL
# ==========================================

MODEL_NAME = "nlptown/bert-base-multilingual-uncased-sentiment"

# ==========================================
# DATA FILES
# ==========================================

TRAIN_FILE = "dataset/processed/train.csv"
VAL_FILE = "dataset/processed/val.csv"

# ==========================================
# OUTPUT DIRECTORY
# ==========================================

OUTPUT_DIR = "models/nlptown_bert_product"


# ==========================================
# 1. LOAD DATASETS
# ==========================================

train_df = pd.read_csv(TRAIN_FILE)
val_df = pd.read_csv(VAL_FILE)

print("Original train samples:", len(train_df))
print("Validation samples:", len(val_df))


# ==========================================
# 2. SELECT TRAINING SUBSET
# ==========================================

# CPU-friendly training
train_df = train_df.sample(
    n=min(800, len(train_df)),
    random_state=42
)

print("Training samples used:", len(train_df))


# ==========================================
# 3. BERT LABELS
# ==========================================

# BERT labels must be 0-4
train_df["label"] = train_df["rating"] - 1
val_df["label"] = val_df["rating"] - 1


# Keep only required columns
train_df = train_df[["reviewText", "label"]]
val_df = val_df[["reviewText", "label"]]


# ==========================================
# 4. CONVERT TO HUGGING FACE DATASET
# ==========================================

train_dataset = Dataset.from_pandas(
    train_df,
    preserve_index=False
)

val_dataset = Dataset.from_pandas(
    val_df,
    preserve_index=False
)


# ==========================================
# 5. LOAD TOKENIZER
# ==========================================

print("\nLoading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)


# ==========================================
# 6. TOKENIZATION
# ==========================================

def tokenize_function(examples):

    return tokenizer(
        examples["reviewText"],
        padding="max_length",
        truncation=True,
        max_length=64
    )


print("Tokenizing datasets...")

train_dataset = train_dataset.map(
    tokenize_function,
    batched=True
)

val_dataset = val_dataset.map(
    tokenize_function,
    batched=True
)


# ==========================================
# 7. LOAD BERT MODEL
# ==========================================

print("\nLoading BERT model...")

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=5
)


# ==========================================
# 8. TRAINING SETTINGS
# ==========================================

training_args = TrainingArguments(

    output_dir=OUTPUT_DIR,

    eval_strategy="epoch",

    save_strategy="epoch",

    learning_rate=2e-5,

    # CPU-friendly batch size
    per_device_train_batch_size=4,

    per_device_eval_batch_size=4,

    # One epoch
    num_train_epochs=1,

    weight_decay=0.01,

    logging_steps=20,

    load_best_model_at_end=True,

    metric_for_best_model="eval_loss",

    greater_is_better=False,

    report_to="none"
)


# ==========================================
# 9. TRAINER
# ==========================================

trainer = Trainer(

    model=model,

    args=training_args,

    train_dataset=train_dataset,

    eval_dataset=val_dataset,

    processing_class=tokenizer
)


# ==========================================
# 10. START TRAINING
# ==========================================

print("\n===================================")
print("Starting BERT training...")
print("===================================")

trainer.train()


# ==========================================
# 11. SAVE MODEL
# ==========================================

print("\nSaving model...")

trainer.save_model(OUTPUT_DIR)

tokenizer.save_pretrained(OUTPUT_DIR)


# ==========================================
# 12. COMPLETION
# ==========================================

print("\n===================================")
print("Training completed successfully!")
print("===================================")

print("Model saved at:", OUTPUT_DIR)