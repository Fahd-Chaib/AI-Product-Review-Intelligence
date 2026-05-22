"""
Streamlit application for AI Product Review Intelligence System.

Features:
- Product name input
- Optional manual reviews input
- Multi-agent orchestration
- Sentiment percentages
- Common complaints
- Product strengths
- Human-in-the-loop approval checkpoint
- Final report generation
- JSON logs display
"""

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from src.orchestration.orchestrator import ProductReviewOrchestrator
from src.utils.errors import HumanApprovalRequiredError, ProductReviewError
from src.utils.logger import read_logs


load_dotenv()


st.set_page_config(
    page_title="AI Product Review Intelligence",
    layout="wide"
)


@st.cache_resource
def get_orchestrator() -> ProductReviewOrchestrator:
    """
    Cache the orchestrator to avoid recreating agents at every refresh.
    """
    return ProductReviewOrchestrator()


def display_sentiment_metrics(insights):
    """
    Display sentiment percentages as Streamlit metrics.
    """
    percentages = insights["sentiment_percentages"]

    col1, col2, col3 = st.columns(3)

    col1.metric("Positive reviews", f"{percentages['positive']}%")
    col2.metric("Negative reviews", f"{percentages['negative']}%")
    col3.metric("Neutral reviews", f"{percentages['neutral']}%")


def display_sentiment_chart(insights):
    """
    Display sentiment distribution as a bar chart.
    """
    percentages = insights["sentiment_percentages"]

    chart_df = pd.DataFrame(
        {
            "Sentiment": ["Positive", "Negative", "Neutral"],
            "Percentage": [
                percentages["positive"],
                percentages["negative"],
                percentages["neutral"],
            ],
        }
    )

    st.bar_chart(chart_df.set_index("Sentiment"))


def display_review_table(analyzed_reviews):
    """
    Display analyzed reviews in a table.
    """
    table_df = pd.DataFrame(
        [
            {
                "Review": review["text"],
                "Sentiment": review["sentiment_label"],
                "Score": round(float(review["sentiment_score"]), 3),
                "Model used": review.get("model_used", "unknown"),
                "Source": review.get("source", "unknown"),
            }
            for review in analyzed_reviews
        ]
    )

    st.dataframe(table_df, width='stretch')


def main():
    """
    Main Streamlit UI.
    """
    st.title("AI Product Review Intelligence System")

    st.caption(
        "Multi-agent product review analysis using CrewAI, DistilBERT sentiment model, "
        "Gemini report generation, JSON logging and human-in-the-loop validation."
    )

    orchestrator = get_orchestrator()

    with st.sidebar:
        st.header("Architecture")

        st.markdown(
            """
            **Orchestrator**
            
            Coordinates all agents and controls the workflow.

            **Specialist Agents**
            - Collector Agent
            - Sentiment Agent
            - Insight Agent
            - Report Agent
            """
        )

        st.divider()

        st.header("Settings")

        max_reviews = st.slider(
            "Maximum reviews to analyze",
            min_value=3,
            max_value=30,
            value=10
        )

        st.info(
            "If no manual reviews are pasted, the app uses the demo review dataset."
        )

    product_name = st.text_input(
        "Product name",
        placeholder="Example: Wireless Headphones"
    )

    raw_reviews = st.text_area(
        "Optional: paste your own reviews, one review per line",
        placeholder=(
            "The product is excellent and easy to use.\n"
            "The battery life is too short.\n"
            "Delivery was late but the quality is good."
        ),
        height=150
    )

    analyze_button = st.button("Analyze Reviews", type="primary")

    if analyze_button:
        try:
            if not product_name.strip():
                st.error("Please enter a product name.")
                return

            with st.spinner("Agents are analyzing product reviews..."):
                analysis_result = orchestrator.run_analysis(
                    product_name=product_name,
                    max_reviews=max_reviews,
                    raw_reviews=raw_reviews
                )

            st.session_state["analysis_result"] = analysis_result

            if "report_result" in st.session_state:
                del st.session_state["report_result"]

            st.success(
                "Analysis completed. Human approval is required before final report generation."
            )

        except ProductReviewError as error:
            st.error(f"Application error: {error}")

        except Exception as error:
            st.error(f"Unexpected error: {error}")

    if "analysis_result" in st.session_state:
        result = st.session_state["analysis_result"]
        insights = result["insights"]

        st.divider()

        st.header("Sentiment Overview")

        display_sentiment_metrics(insights)
        display_sentiment_chart(insights)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Common Complaints")
            for complaint in insights["common_complaints"]:
                st.write(f"- {complaint}")

        with col2:
            st.subheader("Product Strengths")
            for strength in insights["strengths"]:
                st.write(f"- {strength}")

        st.subheader("Analyzed Reviews")
        display_review_table(result["analyzed_reviews"])

        st.divider()

        st.header("Human-in-the-loop Checkpoint")

        st.warning(
            "The final report can only be generated after human approval."
        )

        human_approved = st.checkbox(
            "I reviewed the insights and approve final report generation."
        )

        generate_report_button = st.button("Generate Final Report")

        if generate_report_button:
            try:
                with st.spinner("Report Agent is generating the final report..."):
                    report_result = orchestrator.generate_report_after_approval(
                        analysis_result=result,
                        human_approved=human_approved
                    )

                st.session_state["report_result"] = report_result

                st.success("Final report generated successfully.")

            except HumanApprovalRequiredError as error:
                st.error(str(error))

            except ProductReviewError as error:
                st.error(f"Report generation error: {error}")

            except Exception as error:
                st.error(f"Unexpected error: {error}")

    if "report_result" in st.session_state:
        st.divider()

        st.header("Final Report")

        st.markdown(st.session_state["report_result"]["report_text"])

        st.info(
            f"Report saved at: {st.session_state['report_result']['report_path']}"
        )

    st.divider()

    with st.expander("View JSON Logs"):
        logs = read_logs(limit=100)

        if logs:
            st.json(logs)
        else:
            st.write("No logs yet.")


if __name__ == "__main__":
    main()
