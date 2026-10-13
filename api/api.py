from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)


# ==========================================
# FASTAPI APP
# ==========================================

app = FastAPI(
    title="Product Sentiment Analysis API",
    description="NLP Product Sentiment Analysis using NLPTown BERT",
    version="1.0.0"
)


# ==========================================
# CORS CONFIGURATION
# ==========================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==========================================
# MODEL PATH
# ==========================================

MODEL_DIR = "models/nlptown_bert_product"


# ==========================================
# LOAD TOKENIZER
# ==========================================

print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_DIR
)


# ==========================================
# LOAD MODEL
# ==========================================

print("Loading BERT model...")

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_DIR
)

model.eval()

print("Model loaded successfully!")


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
# REQUEST MODEL
# ==========================================

class ReviewRequest(BaseModel):

    review: str


# ==========================================
# HEALTH CHECK
# ==========================================

@app.get("/health")
def health():

    return {
        "status": "ok",
        "model": "NLPTown BERT",
        "model_loaded": True
    }


# ==========================================
# SENTIMENT PREDICTION
# ==========================================

@app.post("/predict")
def predict(request: ReviewRequest):

    review = request.review.strip()

    # Check empty review
    if not review:

        return {
            "error": "Review cannot be empty"
        }


    # ======================================
    # TOKENIZATION
    # ======================================

    inputs = tokenizer(
        review,
        return_tensors="pt",
        truncation=True,
        max_length=64
    )


    # ======================================
    # MODEL PREDICTION
    # ======================================

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


    # ======================================
    # CONVERT CLASS TO STARS
    # ======================================

    stars = predicted_class + 1


    # ======================================
    # CONFIDENCE
    # ======================================

    confidence = probabilities[
        0
    ][predicted_class].item()


    # ======================================
    # SENTIMENT
    # ======================================

    sentiment = sentiment_mapping[stars]


    # ======================================
    # RESPONSE
    # ======================================

    return {

        "review": review,

        "rating": stars,

        "sentiment": sentiment,

        "confidence": round(
            confidence,
            4
        )

    }


# ==========================================
# MODEL PERFORMANCE METRICS
# ==========================================

@app.get("/metrics")
def metrics():

    return {

        "accuracy": 0.8191,

        "precision": 0.7408,

        "recall": 0.8191,

        "f1_score": 0.7696

    }