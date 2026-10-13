import json
from collections import Counter

INPUT_FILE = "dataset/raw/Electronics_5.json"

product_counts = Counter()

with open(INPUT_FILE, "r", encoding="utf-8") as file:
    for line in file:
        line = line.strip()

        if not line:
            continue

        data = json.loads(line)

        asin = data.get("asin")

        if asin:
            product_counts[asin] += 1

print("Top 20 products by number of reviews:\n")

for asin, count in product_counts.most_common(20):
    print(f"ASIN: {asin}  |  Reviews: {count}")