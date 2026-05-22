"""
Custom exceptions used across the application.

The goal is to keep error handling clean and easy to explain.
"""


class ProductReviewError(Exception):
    """Base exception for the Product Review Intelligence system."""
    pass


class ReviewCollectionError(ProductReviewError):
    """Raised when review collection fails."""
    pass


class SentimentAnalysisError(ProductReviewError):
    """Raised when sentiment analysis fails."""
    pass


class InsightGenerationError(ProductReviewError):
    """Raised when insight generation fails."""
    pass


class HumanApprovalRequiredError(ProductReviewError):
    """Raised when human approval is missing."""
    pass


class ReportGenerationError(ProductReviewError):
    """Raised when report generation fails."""
    pass
