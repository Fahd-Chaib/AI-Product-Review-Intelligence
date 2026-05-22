
from pathlib import Path
from typing import Dict

import torch
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification
)

# ---------------------------------------------------
# MODEL PATH
# ---------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "model" / "sentiment_distilbert"

# ---------------------------------------------------
# DEVICE
# ---------------------------------------------------

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# ---------------------------------------------------
# LOAD TOKENIZER + MODEL
# ---------------------------------------------------

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model folder not found: {MODEL_PATH}"
    )

tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_PATH)
model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)

model.to(DEVICE)
model.eval()

# ---------------------------------------------------
# MAIN API
# ---------------------------------------------------

def predict_sentiment(text: str) -> Dict:
    """
    Predict sentiment from product review text.

    Returns:
    {
        "label": "positive",
        "score": 0.95
    }
    """

    if not isinstance(text, str):
        raise ValueError("Input must be a string.")

    if len(text.strip()) == 0:
        raise ValueError("Input text cannot be empty.")

    try:
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=128
        )

        inputs = {
            key: value.to(DEVICE)
            for key, value in inputs.items()
        }

        with torch.no_grad():
            outputs = model(**inputs)

            probabilities = torch.softmax(
                outputs.logits,
                dim=1
            )

            score, predicted_class_id = torch.max(
                probabilities,
                dim=1
            )

        label = model.config.id2label[
            predicted_class_id.item()
        ]

        return {
            "label": label,
            "score": round(score.item(), 4)
        }

    except Exception as error:
        raise RuntimeError(
            f"Prediction failed: {error}"
        )
