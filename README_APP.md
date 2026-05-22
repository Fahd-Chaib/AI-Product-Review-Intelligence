# App Part - AI Product Review Intelligence System

## Objective

This part implements the application layer of the AI Product Review Intelligence System.

It includes:
- Streamlit user interface
- Multi-agent architecture
- CrewAI agent definitions
- Orchestrator
- Sentiment model integration
- Insight generation
- Gemini report generation
- Human-in-the-loop checkpoint
- JSON logging
- Error handling

## Main Files

app.py

src/app_agents/crew_agents.py
src/app_tools/review_collector.py
src/app_tools/sentiment_tool.py
src/app_tools/insight_tool.py
src/app_tools/report_tool.py
src/orchestration/orchestrator.py
src/utils/logger.py
src/utils/errors.py

## Multi-Agent Architecture

The system uses one orchestrator and four specialist agents.

### Orchestrator

The orchestrator controls the full workflow:
1. Collect reviews
2. Analyze sentiment
3. Generate insights
4. Wait for human approval
5. Generate final report

### Collector Agent

The Collector Agent collects product reviews.

For the demo, reviews are loaded from:

data/sample_reviews.csv

### Sentiment Agent

The Sentiment Agent analyzes each review using the sentiment model.

The app first tries to use the teammate model:

from src.sentiment_model import predict_sentiment

If the trained model folder is missing, the app uses a temporary Transformers sentiment pipeline.

### Insight Agent

The Insight Agent generates:
- Positive percentage
- Negative percentage
- Neutral percentage
- Common complaints
- Product strengths

### Report Agent

The Report Agent generates the final report.

It uses Gemini API if GEMINI_API_KEY is configured.

If Gemini is not configured, the app generates a fallback report to keep the demo stable.

## Human-in-the-loop Checkpoint

Before generating the final report, the user must approve the analysis in the Streamlit interface.

If approval is missing, the report is blocked.

## Run the App

streamlit run app.py

## Model Integration

The trained model folder must be placed here:

model/sentiment_distilbert/

The app automatically uses:

from src.sentiment_model import predict_sentiment

Expected model output:

{
    "label": "positive",
    "score": 0.95
}

## Logs

Logs are saved in:

logs/agent_actions.jsonl

Each log contains:
- timestamp
- agent
- action
- status
- input
- output
- error

## Error Handling

Custom errors are defined in:

src/utils/errors.py

Main errors:
- ProductReviewError
- ReviewCollectionError
- SentimentAnalysisError
- InsightGenerationError
- HumanApprovalRequiredError
- ReportGenerationError

## Git Branch

This app part is developed on:

app-part

## Important Note

This part does not train the model.

The deep learning model is handled by the ML teammate.

This part focuses on:
- Multi-agent architecture
- Streamlit app
- CrewAI orchestration
- Logging
- Error handling
- Human approval
- Report generation
- GitHub integration
