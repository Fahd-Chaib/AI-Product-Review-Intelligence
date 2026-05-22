"""
Sentiment tool.

This module represents the Sentiment Agent tool.

Priority:
1. Use teammate trained DistilBERT model:
   from src.sentiment_model import predict_sentiment

2. If the model folder is missing, use the temporary transformers pipeline.

3. If transformers fails, use a small keyword fallback.
This keeps the demo stable.
"""

from typing import Dict, List

from src.utils.errors import SentimentAnalysisError
from src.utils.logger import log_event


_hf_pipeline = None
_teammate_model_available = None


def _normalize_label(label: str, score: float) -> str:
    """
    Normalize possible labels into:
    positive / negative / neutral

    The teammate model may return:
    - positive / negative
    - LABEL_0 / LABEL_1
    - POSITIVE / NEGATIVE
    """
    label = str(label).lower()

    if score < 0.60:
        return "neutral"

    if label in ["label_0", "negative"] or "neg" in label:
        return "negative"

    if label in ["label_1", "positive"] or "pos" in label:
        return "positive"

    if "neu" in label:
        return "neutral"

    return "neutral"


def _keyword_fallback(text: str) -> Dict:
    """
    Last-resort fallback for demo stability.
    This is not the final deep learning model.
    """
    positive_words = [
        "excellent", "great", "good", "love", "premium",
        "easy", "fast", "comfortable", "recommend", "quality",
        "useful", "bright", "nice"
    ]

    negative_words = [
        "bad", "slow", "late", "cheap", "fragile", "crashes",
        "short", "disappointing", "noisy", "unstable", "too small",
        "not helpful"
    ]

    text_lower = text.lower()

    positive_count = sum(word in text_lower for word in positive_words)
    negative_count = sum(word in text_lower for word in negative_words)

    if positive_count > negative_count:
        return {
            "label": "positive",
            "score": 0.70,
            "model_used": "keyword_fallback"
        }

    if negative_count > positive_count:
        return {
            "label": "negative",
            "score": 0.70,
            "model_used": "keyword_fallback"
        }

    return {
        "label": "neutral",
        "score": 0.55,
        "model_used": "keyword_fallback"
    }


def predict_review_sentiment(text: str) -> Dict:
    """
    Predict sentiment for one review.

    Expected output:
    {
        "label": "positive",
        "score": 0.95,
        "model_used": "teammate_distilbert"
    }
    """
    global _hf_pipeline
    global _teammate_model_available

    agent_name = "Sentiment Agent"

    try:
        if not text or not text.strip():
            return {
                "label": "neutral",
                "score": 0.0,
                "model_used": "empty_input"
            }

        # 1. Try teammate model first.
        if _teammate_model_available is not False:
            try:
                from src.sentiment_model import predict_sentiment

                _teammate_model_available = True

                result = predict_sentiment(text)

                score = float(result.get("score", 0.0))
                label = _normalize_label(
                    result.get("label", "neutral"),
                    score
                )

                return {
                    "label": label,
                    "score": score,
                    "model_used": "teammate_distilbert"
                }

            except Exception as error:
                _teammate_model_available = False

                log_event(
                    agent=agent_name,
                    action="teammate_model_unavailable",
                    status="warning",
                    error=str(error)
                )

        # 2. Temporary placeholder pipeline.
        try:
            if _hf_pipeline is None:
                from transformers import pipeline

                _hf_pipeline = pipeline("sentiment-analysis")

            hf_result = _hf_pipeline(text)[0]

            score = float(hf_result.get("score", 0.0))
            label = _normalize_label(
                hf_result.get("label", "neutral"),
                score
            )

            return {
                "label": label,
                "score": score,
                "model_used": "transformers_pipeline_placeholder"
            }

        except Exception as error:
            log_event(
                agent=agent_name,
                action="transformers_pipeline_failed",
                status="warning",
                error=str(error)
            )

            return _keyword_fallback(text)

    except Exception as error:
        raise SentimentAnalysisError(str(error))


def analyze_reviews_sentiment(reviews: List[Dict]) -> List[Dict]:
    """
    Run sentiment analysis on a list of reviews.
    """
    agent_name = "Sentiment Agent"

    try:
        log_event(
            agent=agent_name,
            action="sentiment_analysis_started",
            status="started",
            input_data={
                "review_count": len(reviews)
            }
        )

        analyzed_reviews = []

        for review in reviews:
            text = review.get("text", "")

            sentiment = predict_review_sentiment(text)

            analyzed_reviews.append(
                {
                    **review,
                    "sentiment_label": sentiment["label"],
                    "sentiment_score": sentiment["score"],
                    "model_used": sentiment.get("model_used", "unknown")
                }
            )

        log_event(
            agent=agent_name,
            action="sentiment_analysis_completed",
            status="success",
            output_data={
                "analyzed_reviews": len(analyzed_reviews)
            }
        )

        return analyzed_reviews

    except Exception as error:
        log_event(
            agent=agent_name,
            action="sentiment_analysis_failed",
            status="error",
            error=str(error)
        )

        raise SentimentAnalysisError(str(error))
