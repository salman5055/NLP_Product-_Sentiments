import pandas as pd
import numpy as np

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer
)

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix
)


# ==========================================
# FILE PATHS
# ==========================================

MODEL_DIR = "models/nlptown_bert_product"
TEST_FILE = "dataset/processed/test.csv"


# ==========================================
# 1. LOAD TEST DATA
# ==========================================

test_df = pd.read_csv(TEST_FILE)

print("Test samples:", len(test_df))


# BERT labels: 1-5 stars -> 0-4
test_df["label"] = test_df["rating"].astype(int) - 1

test_df = test_df[["reviewText", "label"]]


# ==========================================
# 2. CONVERT TO DATASET
# ==========================================

test_dataset = Dataset.from_pandas(
    test_df,
    preserve_index=False
)


# ==========================================
# 3. LOAD TOKENIZER
# ==========================================

print("\nLoading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_DIR
)


# ==========================================
# 4. TOKENIZE TEST DATA
# ==========================================

def tokenize_function(examples):

    return tokenizer(
        examples["reviewText"],
        padding="max_length",
        truncation=True,
        max_length=64
    )


print("Tokenizing test data...")

test_dataset = test_dataset.map(
    tokenize_function,
    batched=True
)


# ==========================================
# 5. LOAD TRAINED MODEL
# ==========================================

print("\nLoading trained model...")

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_DIR
)


# ==========================================
# 6. CREATE TRAINER
# ==========================================

trainer = Trainer(
    model=model
)


# ==========================================
# 7. PREDICTION
# ==========================================

print("\nRunning predictions...")

predictions = trainer.predict(test_dataset)


# ==========================================
# 8. GET PREDICTED LABELS
# ==========================================

predicted_labels = np.argmax(
    predictions.predictions,
    axis=1
)

true_labels = test_df["label"].values


# ==========================================
# 9. METRICS
# ==========================================

accuracy = accuracy_score(
    true_labels,
    predicted_labels
)

precision, recall, f1, _ = precision_recall_fscore_support(
    true_labels,
    predicted_labels,
    average="weighted",
    zero_division=0
)


# ==========================================
# 10. RESULTS
# ==========================================

print("\n===================================")
print("BERT MODEL EVALUATION")
print("===================================")

print(f"Accuracy : {accuracy * 100:.2f}%")
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall   : {recall * 100:.2f}%")
print(f"F1 Score : {f1 * 100:.2f}%")


# ==========================================
# 11. CONFUSION MATRIX
# ==========================================

cm = confusion_matrix(
    true_labels,
    predicted_labels
)

print("\nConfusion Matrix:")
print(cm)

print("\nEvaluation completed!")