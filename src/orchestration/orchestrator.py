"""
Main orchestrator.

The orchestrator coordinates the full workflow:

1. Collector Agent
2. Sentiment Agent
3. Insight Agent
4. Human-in-the-loop checkpoint
5. Report Agent

This design is simple, stable, and easy to defend orally.
"""

from typing import Dict, Optional

from src.app_agents.crew_agents import create_agents, describe_architecture
from src.app_tools.insight_tool import generate_insights
from src.app_tools.report_tool import generate_final_report
from src.app_tools.review_collector import build_reviews_from_user_input, collect_reviews
from src.app_tools.sentiment_tool import analyze_reviews_sentiment
from src.utils.errors import HumanApprovalRequiredError, ProductReviewError
from src.utils.logger import log_event


class ProductReviewOrchestrator:
    """
    Central orchestrator for the Product Review Intelligence system.
    """

    def __init__(self) -> None:
        """
        Initialize CrewAI agents and architecture description.
        """
        self.agents = create_agents()
        self.architecture = describe_architecture()

        log_event(
            agent="Orchestrator",
            action="orchestrator_initialized",
            status="success",
            output_data={
                "agents_loaded": list(self.agents.keys()),
                "architecture": self.architecture
            }
        )

    def run_analysis(
        self,
        product_name: str,
        max_reviews: int = 20,
        raw_reviews: Optional[str] = None
    ) -> Dict:
        """
        Run the analysis workflow until the human approval checkpoint.

        The final report is not generated in this function.
        """
        try:
            log_event(
                agent="Orchestrator",
                action="analysis_workflow_started",
                status="started",
                input_data={
                    "product_name": product_name,
                    "max_reviews": max_reviews,
                    "has_user_reviews": bool(raw_reviews)
                }
            )

            user_reviews = build_reviews_from_user_input(
                product_name=product_name,
                raw_reviews=raw_reviews or ""
            )

            if user_reviews:
                reviews = user_reviews[:max_reviews]

                log_event(
                    agent="Collector Agent",
                    action="user_reviews_loaded",
                    status="success",
                    output_data={
                        "reviews_collected": len(reviews)
                    }
                )
            else:
                reviews = collect_reviews(
                    product_name=product_name,
                    max_reviews=max_reviews
                )

            analyzed_reviews = analyze_reviews_sentiment(reviews)

            insights = generate_insights(analyzed_reviews)

            result = {
                "product_name": product_name,
                "reviews": reviews,
                "analyzed_reviews": analyzed_reviews,
                "insights": insights,
                "architecture": self.architecture,
                "checkpoint": "Human approval required before final report generation."
            }

            log_event(
                agent="Orchestrator",
                action="analysis_workflow_completed",
                status="success",
                output_data={
                    "product_name": product_name,
                    "reviews_analyzed": len(analyzed_reviews)
                }
            )

            return result

        except ProductReviewError:
            raise

        except Exception as error:
            log_event(
                agent="Orchestrator",
                action="analysis_workflow_failed",
                status="error",
                error=str(error)
            )

            raise ProductReviewError(str(error))

    def generate_report_after_approval(
        self,
        analysis_result: Dict,
        human_approved: bool
    ) -> Dict:
        """
        Generate final report only if the human approves.
        """
        try:
            log_event(
                agent="Orchestrator",
                action="human_checkpoint_reached",
                status="started",
                input_data={
                    "human_approved": human_approved
                }
            )

            if not human_approved:
                log_event(
                    agent="Orchestrator",
                    action="human_checkpoint_rejected",
                    status="blocked",
                    error="Human approval missing."
                )

                raise HumanApprovalRequiredError(
                    "Human approval is required before generating the final report."
                )

            log_event(
                agent="Orchestrator",
                action="human_checkpoint_approved",
                status="success"
            )

            product_name = analysis_result["product_name"]
            analyzed_reviews = analysis_result["analyzed_reviews"]
            insights = analysis_result["insights"]

            report = generate_final_report(
                product_name=product_name,
                analyzed_reviews=analyzed_reviews,
                insights=insights
            )

            log_event(
                agent="Orchestrator",
                action="report_workflow_completed",
                status="success",
                output_data={
                    "report_path": report.get("report_path")
                }
            )

            return report

        except ProductReviewError:
            raise

        except Exception as error:
            log_event(
                agent="Orchestrator",
                action="report_workflow_failed",
                status="error",
                error=str(error)
            )

            raise ProductReviewError(str(error))
