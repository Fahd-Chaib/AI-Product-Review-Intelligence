"""
Report tool.

This module represents the Report Agent.

It generates the final product intelligence report using Gemini API.
If Gemini is not configured, it returns a fallback report so the demo never crashes.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from dotenv import load_dotenv

from src.utils.errors import ReportGenerationError
from src.utils.logger import log_event


load_dotenv()

REPORT_DIR = Path("reports")


def _build_prompt(product_name: str, analyzed_reviews: List[Dict], insights: Dict) -> str:
    """
    Build a structured prompt for Gemini.
    """
    compact_reviews = [
        {
            "text": review["text"],
            "sentiment": review["sentiment_label"],
            "score": review["sentiment_score"],
            "model_used": review.get("model_used", "unknown")
        }
        for review in analyzed_reviews[:15]
    ]

    return f"""
You are the Report Agent of a university multi-agent AI system.

Product: {product_name}

Sentiment insights:
{json.dumps(insights, indent=2)}

Sample analyzed reviews:
{json.dumps(compact_reviews, indent=2)}

Generate a professional product review intelligence report with:

1. Executive summary
2. Sentiment distribution
3. Main customer complaints
4. Main product strengths
5. Business recommendations
6. Risks and limitations
7. Conclusion

Keep the report clear, structured, and suitable for a university project demo.
"""


def _fallback_report(product_name: str, insights: Dict) -> str:
    """
    Fallback report when Gemini API is unavailable.
    """
    return f"""
# Product Review Intelligence Report

## Product
{product_name}

## Executive Summary
The system analyzed customer reviews using a multi-agent architecture. The sentiment model classified each review as positive, negative, or neutral. The Insight Agent then extracted the main business trends.

## Sentiment Distribution
- Positive: {insights["sentiment_percentages"]["positive"]}%
- Negative: {insights["sentiment_percentages"]["negative"]}%
- Neutral: {insights["sentiment_percentages"]["neutral"]}%

## Common Complaints
{chr(10).join("- " + item for item in insights["common_complaints"])}

## Product Strengths
{chr(10).join("- " + item for item in insights["strengths"])}

## Business Recommendations
The company should reduce the most frequent complaints and use the strongest positive points in marketing communication.

## Risks and Limitations
This report is based on the available review sample. More reviews would improve the reliability of the analysis.

## Conclusion
The Product Review Intelligence system successfully combines review collection, sentiment analysis, insight extraction, human validation, and report generation.
"""


def generate_final_report(product_name: str, analyzed_reviews: List[Dict], insights: Dict) -> Dict:
    """
    Generate the final Markdown report.
    """
    agent_name = "Report Agent"

    try:
        log_event(
            agent=agent_name,
            action="report_generation_started",
            status="started",
            input_data={
                "product_name": product_name,
                "review_count": len(analyzed_reviews)
            }
        )

        REPORT_DIR.mkdir(parents=True, exist_ok=True)

        api_key = os.getenv("GEMINI_API_KEY")
        model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

        prompt = _build_prompt(
            product_name=product_name,
            analyzed_reviews=analyzed_reviews,
            insights=insights
        )

        if api_key:
            try:
                from google import genai

                client = genai.Client(api_key=api_key)

                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt
                )

                report_text = response.text

            except Exception as error:
                log_event(
                    agent=agent_name,
                    action="gemini_generation_failed",
                    status="warning",
                    error=str(error)
                )

                report_text = _fallback_report(product_name, insights)

        else:
            log_event(
                agent=agent_name,
                action="gemini_api_key_missing",
                status="warning",
                error="GEMINI_API_KEY not found. Fallback report used."
            )

            report_text = _fallback_report(product_name, insights)

        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        report_path = REPORT_DIR / f"report_{timestamp}.md"

        report_path.write_text(report_text, encoding="utf-8")

        output = {
            "report_text": report_text,
            "report_path": str(report_path)
        }

        log_event(
            agent=agent_name,
            action="report_generation_completed",
            status="success",
            output_data={
                "report_path": str(report_path)
            }
        )

        return output

    except Exception as error:
        log_event(
            agent=agent_name,
            action="report_generation_failed",
            status="error",
            error=str(error)
        )

        raise ReportGenerationError(str(error))
