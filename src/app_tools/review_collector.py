"""
Collector tool.

This module represents the Collector Agent tool.

For the demo, reviews are loaded from data/sample_reviews.csv.
Later, we can replace this source with:
- Hugging Face datasets
- Amazon Reviews dataset
- API
- Web scraping
- uploaded CSV

The rest of the app will not change because this function always returns
the same clean structure.
"""

from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from src.utils.errors import ReviewCollectionError
from src.utils.logger import log_event


DATA_PATH = Path("data/sample_reviews.csv")


def collect_reviews(product_name: str, max_reviews: int = 20) -> List[Dict]:
    """
    Collect reviews for a product.

    Args:
        product_name: Product name entered by the user.
        max_reviews: Maximum number of reviews to return.

    Returns:
        List of dictionaries:
        [
            {
                "review_id": 1,
                "product": "Wireless Headphones",
                "text": "The product is excellent.",
                "source": "sample_csv"
            }
        ]
    """
    agent_name = "Collector Agent"

    try:
        log_event(
            agent=agent_name,
            action="collect_reviews_started",
            status="started",
            input_data={
                "product_name": product_name,
                "max_reviews": max_reviews
            }
        )

        if not product_name or not product_name.strip():
            raise ReviewCollectionError("Product name is empty.")

        if not DATA_PATH.exists():
            raise ReviewCollectionError(f"Dataset not found: {DATA_PATH}")

        df = pd.read_csv(DATA_PATH)

        if "product" not in df.columns or "review" not in df.columns:
            raise ReviewCollectionError(
                "CSV file must contain 'product' and 'review' columns."
            )

        product_name_clean = product_name.strip().lower()

        matched_df = df[
            df["product"].str.lower().str.contains(product_name_clean, na=False)
        ]

        # Fallback for demo stability.
        if matched_df.empty:
            matched_df = df[df["product"].str.lower() == "general product"]

        # Last fallback.
        if matched_df.empty:
            matched_df = df

        matched_df = matched_df.head(max_reviews)

        reviews = []

        for index, row in matched_df.iterrows():
            reviews.append(
                {
                    "review_id": int(index),
                    "product": product_name.strip(),
                    "text": str(row["review"]),
                    "source": "sample_csv"
                }
            )

        log_event(
            agent=agent_name,
            action="collect_reviews_completed",
            status="success",
            output_data={
                "reviews_collected": len(reviews)
            }
        )

        return reviews

    except Exception as error:
        log_event(
            agent=agent_name,
            action="collect_reviews_failed",
            status="error",
            error=str(error)
        )

        raise ReviewCollectionError(str(error))


def build_reviews_from_user_input(
    product_name: str,
    raw_reviews: str
) -> Optional[List[Dict]]:
    """
    Convert reviews pasted by the user into the same structure.

    One line = one review.
    """
    if not raw_reviews or not raw_reviews.strip():
        return None

    lines = [
        line.strip()
        for line in raw_reviews.splitlines()
        if line.strip()
    ]

    if not lines:
        return None

    return [
        {
            "review_id": index,
            "product": product_name,
            "text": line,
            "source": "user_input"
        }
        for index, line in enumerate(lines)
    ]
