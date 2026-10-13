import json
import pandas as pd

INPUT_FILE = "dataset/raw/Electronics_5.json"
OUTPUT_FILE = "dataset/cleaned/product_reviews.csv"

PRODUCT_ASIN = "B007WTAJTO"

reviews = []

with open(INPUT_FILE, "r", encoding="utf-8") as file:
    for line in file:
        line = line.strip()

        if not line:
            continue

        data = json.loads(line)

        if data.get("asin") == PRODUCT_ASIN:
            reviews.append({
                "reviewText": data.get("reviewText", ""),
                "summary": data.get("summary", ""),
                "rating": data.get("overall", None),
                "asin": data.get("asin", "")
            })

df = pd.DataFrame(reviews)

df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")

print("Product:", PRODUCT_ASIN)
print("Total reviews:", len(df))
print("File saved:", OUTPUT_FILE)
print("\nRating distribution:")
print(df["rating"].value_counts().sort_index())