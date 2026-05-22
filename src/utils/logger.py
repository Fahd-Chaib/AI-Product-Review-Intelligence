"""
JSON logging system.

Each important action is saved as one JSON line with:
- timestamp
- agent name
- action
- status
- input
- output
- error

This satisfies the project requirement: every agent action must be logged with timestamps.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


LOG_DIR = Path("logs")
LOG_FILE = LOG_DIR / "agent_actions.jsonl"


def log_event(
    agent: str,
    action: str,
    status: str,
    input_data: Optional[Any] = None,
    output_data: Optional[Any] = None,
    error: Optional[str] = None,
) -> None:
    """
    Save one event in JSON Lines format.

    JSONL format:
    one action = one JSON object = one line.
    """
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)

        event: Dict[str, Any] = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "agent": agent,
            "action": action,
            "status": status,
            "input": input_data,
            "output": output_data,
            "error": error,
        }

        with LOG_FILE.open("a", encoding="utf-8") as file:
            file.write(json.dumps(event, ensure_ascii=False) + "\n")

    except Exception:
        # Logging should never crash the application.
        pass


def read_logs(limit: int = 100) -> List[Dict[str, Any]]:
    """
    Read latest logs for Streamlit display.
    """
    if not LOG_FILE.exists():
        return []

    logs: List[Dict[str, Any]] = []

    try:
        lines = LOG_FILE.read_text(encoding="utf-8").splitlines()

        for line in lines[-limit:]:
            try:
                logs.append(json.loads(line))
            except json.JSONDecodeError:
                logs.append({"raw": line})

        return logs

    except Exception as error:
        return [
            {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "agent": "Logger",
                "action": "read_logs",
                "status": "error",
                "input": None,
                "output": None,
                "error": str(error),
            }
        ]
