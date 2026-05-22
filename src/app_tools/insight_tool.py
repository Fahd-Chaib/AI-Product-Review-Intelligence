"""
Insight tool.

This module represents the Insight Agent tool.

It transforms sentiment predictions into business insights:
- positive / negative / neutral percentages
- common complaints
- product strengths

The logic is simple on purpose so it is easy to defend during the oral exam.
"""

from collections import Counter
from typing import Dict, List

from src.utils.errors import InsightGenerationError
from src.utils.logger import log_event


COMPLAINT_CATEGORIES = {
    "battery": ["battery", "charging", "charge", "autonomy"],
    "delivery": ["delivery", "late", "shipping", "delayed"],
    "quality": ["cheap", "fragile", "broken", "quality", "defect"],
    "price": ["expensive", "price", "cost"],
    "app": ["app", "crash", "crashes", "bug", "slow"],
    "comfort": ["uncomfortable", "heavy", "comfort"],
    "noise": ["noisy", "noise"],
    "connection": ["bluetooth", "connection", "unstable"],
}

STRENGTH_CATEGORIES = {
    "sound quality": ["sound", "audio", "clarity"],
    "design": ["design", "premium", "beautiful", "nice"],
    "battery": ["battery", "lasts", "autonomy"],
    "value for money": ["price", "value", "money"],
    "ease of use": ["easy", "simple", "clean", "useful"],
    "performance": ["fast", "performance", "works"],
    "comfort": ["comfortable", "comfort"],
    "quality": ["quality", "build", "excellent"],
}


def _extract_categories(texts: List[str], categories: Dict[str, List[str]]) -> List[str]:
    """
    Count repeated themes in review texts.

    Example output:
    ["battery mentioned 2 time(s)", "delivery mentioned 1 time(s)"]
    """
    counter = Counter()

    for text in texts:
        lower_text = text.lower()

        for category, keywords in categories.items():
            if any(keyword in lower_text for keyword in keywords):
                counter[category] += 1

    return [
        f"{category} mentioned {count} time(s)"
        for category, count in counter.most_common(5)
    ]


def generate_insights(analyzed_reviews: List[Dict]) -> Dict:
    """
    Generate business insights from analyzed reviews.
    """
    agent_name = "Insight Agent"

    try:
        log_event(
            agent=agent_name,
            action="insight_generation_started",
            status="started",
            input_data={
                "review_count": len(analyzed_reviews)
            }
        )

        if not analyzed_reviews:
            raise InsightGenerationError("No analyzed reviews provided.")

        total = len(analyzed_reviews)

        labels = [
            review.get("sentiment_label", "neutral")
            for review in analyzed_reviews
        ]

        positive_count = labels.count("positive")
        negative_count = labels.count("negative")
        neutral_count = labels.count("neutral")

        percentages = {
            "positive": round((positive_count / total) * 100, 2),
            "negative": round((negative_count / total) * 100, 2),
            "neutral": round((neutral_count / total) * 100, 2),
        }

        negative_texts = [
            review["text"]
            for review in analyzed_reviews
            if review.get("sentiment_label") == "negative"
        ]

        positive_texts = [
            review["text"]
            for review in analyzed_reviews
            if review.get("sentiment_label") == "positive"
        ]

        complaints = _extract_categories(
            negative_texts,
            COMPLAINT_CATEGORIES
        )

        strengths = _extract_categories(
            positive_texts,
            STRENGTH_CATEGORIES
        )

        if not complaints:
            complaints = ["No major repeated complaint detected."]

        if not strengths:
            strengths = ["No major repeated strength detected."]

        insights = {
            "total_reviews": total,
            "sentiment_counts": {
                "positive": positive_count,
                "negative": negative_count,
                "neutral": neutral_count,
            },
            "sentiment_percentages": percentages,
            "common_complaints": complaints,
            "strengths": strengths,
        }

        log_event(
            agent=agent_name,
            action="insight_generation_completed",
            status="success",
            output_data=insights
        )

        return insights

    except Exception as error:
        log_event(
            agent=agent_name,
            action="insight_generation_failed",
            status="error",
            error=str(error)
        )

        raise InsightGenerationError(str(error))
