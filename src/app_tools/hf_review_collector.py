"""
Hugging Face review collector.

This tool collects real Amazon reviews from a Hugging Face dataset.

Dataset used:
mteb/amazon_polarity

Why this dataset?
- It is available through the Hugging Face Dataset Viewer.
- It is auto-converted to Parquet.
- It contains real Amazon reviews.
- It works well with streaming=True.

The Collector Agent uses this source when the user wants automatic real reviews.
"""

from typing import Dict, List

from datasets import load_dataset

from src.utils.errors import ReviewCollectionError
from src.utils.logger import log_event


def collect_reviews_from_huggingface(
    product_name: str,
    max_reviews: int = 20,
    max_scan: int = 3000
) -> List[Dict]:
    """
    Collect real Amazon reviews automatically from Hugging Face.

    Important:
    mteb/amazon_polarity does not contain a clean product_name column.
    So we search inside the review text.

    If no exact match is found, we return general real Amazon reviews.
    """
    agent_name = "Collector Agent"

    try:
        log_event(
            agent=agent_name,
            action="huggingface_collection_started",
            status="started",
            input_data={
                "dataset": "mteb/amazon_polarity",
                "product_name": product_name,
                "max_reviews": max_reviews,
                "max_scan": max_scan
            }
        )

        if not product_name or not product_name.strip():
            raise ReviewCollectionError("Product name is empty.")

        product_query = product_name.strip().lower()

        dataset = load_dataset(
            "mteb/amazon_polarity",
            split="train",
            streaming=True
        )

        matched_reviews = []
        fallback_reviews = []

        for index, row in enumerate(dataset):
            if index >= max_scan:
                break

            review_text = str(row.get("text", "")).strip()

            if not review_text:
                continue

            review_item = {
                "review_id": index,
                "product": product_name.strip(),
                "text": review_text,
                "source": "huggingface_amazon_polarity"
            }

            if len(fallback_reviews) < max_reviews:
                fallback_reviews.append(review_item)

            if product_query in review_text.lower():
                matched_reviews.append(review_item)

            if len(matched_reviews) >= max_reviews:
                break

        reviews = matched_reviews if matched_reviews else fallback_reviews

        if not reviews:
            raise ReviewCollectionError("No reviews found from Hugging Face dataset.")

        log_event(
            agent=agent_name,
            action="huggingface_collection_completed",
            status="success",
            output_data={
                "reviews_collected": len(reviews),
                "matched_product": bool(matched_reviews),
                "source": "huggingface_amazon_polarity"
            }
        )

        return reviews[:max_reviews]

    except Exception as error:
        log_event(
            agent=agent_name,
            action="huggingface_collection_failed",
            status="error",
            error=str(error)
        )

        raise ReviewCollectionError(str(error))
