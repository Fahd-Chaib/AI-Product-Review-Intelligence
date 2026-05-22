
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    classification_report,
    confusion_matrix
)

# ------------------------------------------------
# SAVE RESULTS FUNCTION
# ------------------------------------------------

def save_all_results(
    y_true,
    y_pred,
    eval_results,
    output_dir="results",
    log_dir="logs"
):

    output_path = Path(output_dir)

    log_path = Path(log_dir)

    output_path.mkdir(
        parents=True,
        exist_ok=True
    )

    log_path.mkdir(
        parents=True,
        exist_ok=True
    )

    logging.basicConfig(
        filename=log_path / "training.log",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    logging.info("Saving results started.")

    # --------------------------------------------
    # CLASSIFICATION REPORT
    # --------------------------------------------

    report = classification_report(
        y_true,
        y_pred,
        target_names=["negative", "positive"]
    )

    with open(
        output_path / "classification_report.txt",
        "w",
        encoding="utf-8"
    ) as file:

        file.write(report)

    logging.info("Classification report saved.")

    # --------------------------------------------
    # EVALUATION METRICS JSON
    # --------------------------------------------

    clean_eval_results = {
        key: float(value)
        if isinstance(value, (int, float))
        else value
        for key, value in eval_results.items()
    }

    with open(
        output_path / "evaluation_metrics.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            clean_eval_results,
            file,
            indent=4
        )

    logging.info("Evaluation metrics saved.")

    # --------------------------------------------
    # CONFUSION MATRIX PNG
    # --------------------------------------------

    cm = confusion_matrix(
        y_true,
        y_pred
    )

    plt.figure(figsize=(6, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["negative", "positive"],
        yticklabels=["negative", "positive"]
    )

    plt.xlabel("Predicted")

    plt.ylabel("True")

    plt.title("Confusion Matrix")

    plt.savefig(
        output_path / "confusion_matrix.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    logging.info("Confusion matrix saved.")

    print("Results saved successfully.")
