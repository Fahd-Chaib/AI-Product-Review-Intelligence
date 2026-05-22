"""
CrewAI agent definitions.

This file defines the multi-agent architecture:
- Collector Agent
- Sentiment Agent
- Insight Agent
- Report Agent

The actual execution is controlled by our Python orchestrator for stability.
CrewAI is used to represent the agent roles clearly.
"""

import os
from typing import Dict

from dotenv import load_dotenv

from src.utils.logger import log_event


load_dotenv()


def _build_llm():
    """
    Build Gemini LLM for CrewAI if GEMINI_API_KEY exists.

    If there is no API key, the app still works because the workflow
    is handled by deterministic Python tools.
    """
    try:
        from crewai import LLM

        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            return None

        return LLM(
            model=os.getenv("CREWAI_LLM_MODEL", "gemini/gemini-2.5-flash"),
            api_key=api_key
        )

    except Exception as error:
        log_event(
            agent="CrewAI Setup",
            action="llm_initialization_failed",
            status="warning",
            error=str(error)
        )
        return None


def create_agents() -> Dict:
    """
    Create CrewAI specialist agents.
    """
    try:
        from crewai import Agent

        llm = _build_llm()
        llm_config = {"llm": llm} if llm else {}

        collector_agent = Agent(
            role="Collector Agent",
            goal="Collect relevant customer reviews for the selected product.",
            backstory=(
                "You collect customer reviews and prepare clean input data "
                "for the sentiment analysis agent."
            ),
            verbose=True,
            allow_delegation=False,
            **llm_config
        )

        sentiment_agent = Agent(
            role="Sentiment Agent",
            goal="Classify each review using the trained DistilBERT sentiment model.",
            backstory=(
                "You are responsible for using the deep learning sentiment model "
                "as a real functional tool in the workflow."
            ),
            verbose=True,
            allow_delegation=False,
            **llm_config
        )

        insight_agent = Agent(
            role="Insight Agent",
            goal="Transform sentiment predictions into business insights.",
            backstory=(
                "You analyze customer feedback to detect strengths, complaints, "
                "and product-level trends."
            ),
            verbose=True,
            allow_delegation=False,
            **llm_config
        )

        report_agent = Agent(
            role="Report Agent",
            goal="Generate a final structured product intelligence report.",
            backstory=(
                "You create the final business report only after human approval."
            ),
            verbose=True,
            allow_delegation=False,
            **llm_config
        )

        agents = {
            "collector": collector_agent,
            "sentiment": sentiment_agent,
            "insight": insight_agent,
            "report": report_agent
        }

        log_event(
            agent="CrewAI Setup",
            action="agents_created",
            status="success",
            output_data=list(agents.keys())
        )

        return agents

    except Exception as error:
        log_event(
            agent="CrewAI Setup",
            action="agents_creation_failed",
            status="error",
            error=str(error)
        )

        return {}


def describe_architecture() -> Dict:
    """
    Return a simple architecture description for Streamlit display.
    """
    return {
        "orchestrator": "Controls the full workflow and coordinates all specialist agents.",
        "collector_agent": "Collects product reviews.",
        "sentiment_agent": "Uses the trained DistilBERT sentiment model.",
        "insight_agent": "Computes percentages, complaints, and strengths.",
        "report_agent": "Generates the final Gemini report after human approval."
    }
