import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# ==========================================
# MODEL
# ==========================================

MODEL_DIR = "models/nlptown_bert_product"


# ==========================================
# LOAD MODEL
# ==========================================

print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_DIR
)

print("Loading model...")

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_DIR
)

model.eval()


# ==========================================
# SENTIMENT MAPPING
# ==========================================

sentiment_mapping = {
    1: "Very Negative",
    2: "Negative",
    3: "Neutral",
    4: "Positive",
    5: "Very Positive"
}


# ==========================================
# PREDICTION FUNCTION
# ==========================================

def predict_sentiment(review):

    inputs = tokenizer(
        review,
        return_tensors="pt",
        truncation=True,
        max_length=64
    )

    with torch.no_grad():

        outputs = model(**inputs)

        probabilities = torch.softmax(
            outputs.logits,
            dim=1
        )

        predicted_class = torch.argmax(
            probabilities,
            dim=1
        ).item()

    stars = predicted_class + 1

    confidence = probabilities[0][predicted_class].item()

    sentiment = sentiment_mapping[stars]

    return stars, sentiment, confidence


# ==========================================
# TEST REVIEWS
# ==========================================

test_reviews = [

    "This product is terrible. It stopped working and I regret buying it.",

    "The product is poor and has many problems.",

    "The product is okay. Nothing special.",

    "The product works well and I am satisfied.",

    "This product is excellent. It works perfectly and I am very happy with it."

]


# ==========================================
# RUN TEST
# ==========================================

print("\n===================================")
print("SENTIMENT MAPPING TEST")
print("===================================")


for i, review in enumerate(test_reviews, start=1):

    stars, sentiment, confidence = predict_sentiment(
        review
    )

    print(f"\nTest {i}")
    print("-----------------------------------")
    print("Review:", review)
    print("Predicted Rating:", stars, "Stars")
    print("Sentiment:", sentiment)
    print(f"Confidence: {confidence * 100:.2f}%")


print("\n===================================")
print("TEST COMPLETED")
print("===================================")