"""Evaluate the final ConvNeXt classifier on the held-out test set."""

import json

import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor, AutoModelForImageClassification
from tqdm.auto import tqdm
from sklearn.metrics import classification_report, confusion_matrix

from Dataprep import BASE_DIR, CastingImageDataset, ID2LABEL, LABEL2ID, OUTPUT_DIR
from preprocessing import DEFECTIVE_THRESHOLD, classify_probabilities
from visualization import display_error_samples, plot_test_metrics
from train_model import BATCH_SIZE, FINAL_MODEL_DIR, metric_values


ARTIFACT_DIR = BASE_DIR / "submission_artifacts"
PREDICTIONS_FILE = ARTIFACT_DIR / "test_predictions.csv"
EXAMPLES_FILE = ARTIFACT_DIR / "example_predictions.csv"
PLOT_FILE = ARTIFACT_DIR / "test_metrics_roc_confusion_matrix.png"
ERROR_EXAMPLES_FILE = ARTIFACT_DIR / "false_positive_false_negative_examples.png"
METRICS_FILE = ARTIFACT_DIR / "test_metrics.json"


def evaluate_test_set() -> None:
    """Evaluate the final model once on the held-out test split."""
    manifest = OUTPUT_DIR / "image_labels.csv"
    if not manifest.is_file():
        raise FileNotFoundError(f"Missing dataset manifest: {manifest}")

    dataframe = pd.read_csv(manifest)
    test_data = dataframe[dataframe["split"] == "test"].reset_index(drop=True)
    if test_data.empty:
        raise ValueError("The manifest contains no test images.")
    if not FINAL_MODEL_DIR.is_dir():
        raise FileNotFoundError(
            f"Missing final model checkpoint: {FINAL_MODEL_DIR}; run main.py first."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = AutoImageProcessor.from_pretrained(FINAL_MODEL_DIR)
    dataset = CastingImageDataset(test_data, OUTPUT_DIR, processor)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    model = AutoModelForImageClassification.from_pretrained(FINAL_MODEL_DIR).to(device)
    model.eval()
    decision_threshold = float(
        getattr(model.config, "defective_threshold", DEFECTIVE_THRESHOLD)
    )
    all_logits = []
    actual = []
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Evaluating test set"):
            all_logits.append(model(pixel_values=images.to(device)).logits.cpu())
            actual.extend(labels.tolist())

    logits = torch.cat(all_logits)
    probabilities = logits.softmax(dim=1)[:, 1].tolist()
    predicted = classify_probabilities(probabilities, decision_threshold)
    scores = metric_values(actual, predicted, probabilities)
    print("Test evaluation using the final model:")
    print(f"Defective decision threshold: {decision_threshold:.4f}")
    for name, key in (
        ("Accuracy", "accuracy"),
        ("Precision", "precision"),
        ("Recall", "recall"),
        ("F1-score", "f1"),
        ("ROC-AUC", "roc_auc"),
    ):
        print(f"{name}: {scores[key]:.4f}")

    matrix = confusion_matrix(actual, predicted, labels=[0, 1])
    report = classification_report(
        actual,
        predicted,
        labels=[0, 1],
        target_names=["Normal", "Defective"],
        output_dict=True,
        zero_division=0,
    )
    metrics = {
        "checkpoint": str(FINAL_MODEL_DIR.relative_to(BASE_DIR)),
        "device": device.type,
        "split": "test",
        "sample_count": len(actual),
        "defective_threshold": decision_threshold,
        "positive_class": "Defective",
        "metrics": scores,
        "per_class": {
            label: {
                metric: report[label][metric]
                for metric in ("precision", "recall", "f1-score", "support")
            }
            for label in ("Normal", "Defective")
        },
        "confusion_matrix": {
            "labels": ["Normal", "Defective"],
            "actual_rows_predicted_columns": matrix.tolist(),
        },
    }
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    with METRICS_FILE.open("w", encoding="utf-8") as metrics_file:
        json.dump(metrics, metrics_file, indent=2)

    plot_test_metrics(actual, predicted, probabilities, PLOT_FILE)

    results = test_data[["image_name", "label"]].rename(
        columns={"label": "true_label"}
    )
    results["predicted_label"] = [ID2LABEL[index] for index in predicted]
    results["defective_probability"] = probabilities
    results["confidence_score"] = [
        probability if prediction == LABEL2ID["Defective"] else 1.0 - probability
        for probability, prediction in zip(probabilities, predicted)
    ]
    results["decision_threshold"] = decision_threshold
    results["error_type"] = [
        "false_positive"
        if true_label == "Normal" and predicted_label == "Defective"
        else "false_negative"
        if true_label == "Defective" and predicted_label == "Normal"
        else ""
        for true_label, predicted_label in zip(
            results["true_label"], results["predicted_label"]
        )
    ]
    results.to_csv(PREDICTIONS_FILE, index=False)
    results["example_type"] = ""
    example_types = (
        ("correct_normal", "Normal", "Normal"),
        ("correct_defective", "Defective", "Defective"),
        ("false_positive", "Normal", "Defective"),
        ("false_negative", "Defective", "Normal"),
    )
    for example_type, true_label, predicted_label in example_types:
        matches = results.index[
            (results["true_label"] == true_label)
            & (results["predicted_label"] == predicted_label)
        ]
        if len(matches):
            results.loc[matches[0], "example_type"] = example_type
    results.loc[results["example_type"] != ""].to_csv(EXAMPLES_FILE, index=False)
    print(f"Per-image predictions saved to: {PREDICTIONS_FILE}")
    print(f"Example predictions saved to: {EXAMPLES_FILE}")
    print(f"Test metrics saved to: {METRICS_FILE}")
    print(f"Metric plots saved to: {PLOT_FILE}")
    display_error_samples(
        test_data,
        OUTPUT_DIR,
        actual,
        predicted,
        plot_file=ERROR_EXAMPLES_FILE,
    )
