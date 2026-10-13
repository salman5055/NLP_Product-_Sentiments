import pandas as pd
import re

INPUT_FILE = "dataset/cleaned/product_reviews.csv"
OUTPUT_FILE = "dataset/cleaned/cleaned_reviews.csv"


def clean_text(text):
    if pd.isna(text):
        return ""

    text = str(text)

    # Remove HTML tags
    text = re.sub(r"<.*?>", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text)

    # Remove leading/trailing spaces
    text = text.strip()

    return text


# Load dataset
df = pd.read_csv(INPUT_FILE)

print("Original reviews:", len(df))

# Clean review text
df["reviewText"] = df["reviewText"].apply(clean_text)

# Remove empty reviews
df = df[df["reviewText"] != ""]

# Remove duplicate reviews
df = df.drop_duplicates(subset=["reviewText"])

# Save cleaned dataset
df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")

print("Cleaned reviews:", len(df))
print("File saved:", OUTPUT_FILE)

print("\nRating distribution:")
print(df["rating"].value_counts().sort_index())