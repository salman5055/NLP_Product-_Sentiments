import json
from collections import Counter

INPUT_FILE = "dataset/raw/Electronics_5.json"
PRODUCT_ASIN = "B007WTAJTO"

reviews = []

with open(INPUT_FILE, "r", encoding="utf-8") as file:
    for line in file:
        line = line.strip()

        if not line:
            continue

        data = json.loads(line)

        if data.get("asin") == PRODUCT_ASIN:
            reviews.append(data)

print("===================================")
print("PRODUCT INFORMATION")
print("===================================")

print("ASIN:", PRODUCT_ASIN)
print("Total Reviews:", len(reviews))

ratings = Counter()

for review in reviews:
    ratings[review.get("overall")] += 1

print("\nRating Distribution:")

for rating in sorted(ratings):
    print(f"{rating} Star: {ratings[rating]} reviews")

print("\nSample Reviews:")

for i, review in enumerate(reviews[:5], start=1):
    print(f"\nReview {i}:")
    print("Rating:", review.get("overall"))
    print("Summary:", review.get("summary"))
    print("Review:", review.get("reviewText"))