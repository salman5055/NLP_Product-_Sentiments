import pandas as pd
from sklearn.model_selection import train_test_split

INPUT_FILE = "dataset/cleaned/cleaned_reviews.csv"

TRAIN_FILE = "dataset/processed/train.csv"
VAL_FILE = "dataset/processed/val.csv"
TEST_FILE = "dataset/processed/test.csv"

# Load cleaned dataset
df = pd.read_csv(INPUT_FILE)

# Convert rating to integer
df["rating"] = df["rating"].astype(int)

print("Total dataset:", len(df))

# First split: 80% train, 20% temporary
train_df, temp_df = train_test_split(
    df,
    test_size=0.20,
    random_state=42,
    stratify=df["rating"]
)

# Second split: temporary → 10% validation + 10% test
val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    random_state=42,
    stratify=temp_df["rating"]
)

# Save files
train_df.to_csv(TRAIN_FILE, index=False)
val_df.to_csv(VAL_FILE, index=False)
test_df.to_csv(TEST_FILE, index=False)

print("\nSplit completed!")
print("Train:", len(train_df))
print("Validation:", len(val_df))
print("Test:", len(test_df))

print("\nTrain rating distribution:")
print(train_df["rating"].value_counts().sort_index())

print("\nValidation rating distribution:")
print(val_df["rating"].value_counts().sort_index())

print("\nTest rating distribution:")
print(test_df["rating"].value_counts().sort_index())