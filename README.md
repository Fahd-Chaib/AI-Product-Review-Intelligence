# AI Product Review Intelligence System

## Description

This project uses a fine-tuned DistilBERT model to analyze product review sentiment.

The model predicts:
- positive
- negative

The model was trained using:
- PyTorch
- HuggingFace Transformers
- Google Colab GPU

Final validation accuracy:
~93%

---

# Install

pip install -r requirements.txt

---

# Example Usage

from src.sentiment_model import predict_sentiment

result = predict_sentiment(
    "This product is amazing."
)

print(result)

Expected output:

{
    "label": "positive",
    "score": 0.95
}

---

# Folder Structure

AI-Product-Review-Intelligence/

├── model/
├── src/
├── scripts/
├── results/
├── logs/
├── notebooks/
├── requirements.txt
├── README.md
└── .gitignore

---

# Saved Evaluation Results

results/classification_report.txt

results/confusion_matrix.png

results/evaluation_metrics.json

logs/training.log

---

# Stable API

Main function:

predict_sentiment(text)

Returns:

{
    "label": "positive",
    "score": 0.95
}

This stable interface is designed for:
- Streamlit
- CrewAI
- multi-agent orchestration

---

# GitHub Workflow

git init

git checkout -b model-part

git add .

git commit -m "Add fine-tuned DistilBERT sentiment model"

git remote add origin YOUR_GITHUB_REPOSITORY

git push -u origin model-part

---

# Merge Later

Teammate can merge using:

git checkout main

git pull origin main

git fetch origin

git merge origin/model-part

git push origin main

---

# Oral Defense Summary

We fine-tuned DistilBERT on Amazon product reviews using:
- PyTorch
- HuggingFace Transformers
- Google Colab GPU

We:
- cleaned the dataset
- tokenized reviews
- trained DistilBERT
- evaluated the model
- generated confusion matrix and reports
- saved the model
- created a stable prediction API

Final validation accuracy:
~93%