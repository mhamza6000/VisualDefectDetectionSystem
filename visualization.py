"""Display training/test metrics and labeled prediction samples."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve


METRIC_NAMES = ("accuracy", "precision", "recall", "f1", "roc_auc")
LABEL_NAMES = ("Normal", "Defective")


def display_dataset_samples(
    dataframe: pd.DataFrame,
    dataset_dir: Path,
    source_classes: dict[str, str],
    samples_per_class: int,
) -> None:
    """Display selected source images with their dataset labels."""
    labels = sorted(dataframe["label"].unique())
    figure, axes = plt.subplots(
        len(labels),
        samples_per_class,
        squeeze=False,
        figsize=(3 * samples_per_class, 3 * len(labels)),
    )

    for row, label in enumerate(labels):
        images = dataframe[dataframe["label"] == label].head(samples_per_class)
        for column, image_name in enumerate(images["image_name"]):
            image_path = dataset_dir / source_classes[label] / image_name
            with Image.open(image_path) as image:
                axes[row, column].imshow(image.convert("RGB"))
            axes[row, column].set_title(label)

    for axis in axes.flat:
        axis.axis("off")
    figure.tight_layout()
    plt.show()
    plt.close(figure)


def display_mismatches(
    mismatches: pd.DataFrame,
    dataset_dir: Path,
    source_classes: dict[str, str],
) -> None:
    """Display up to 12 candidate mismatches with both labels."""
    samples = mismatches.head(12)
    if samples.empty:
        print("No candidate mismatches to display.")
        return

    columns = 4
    rows = (len(samples) + columns - 1) // columns
    figure, axes = plt.subplots(rows, columns, figsize=(12, 3 * rows), squeeze=False)
    for axis, sample in zip(axes.flat, samples.itertuples(index=False)):
        image_path = dataset_dir / source_classes[sample.dataset_label] / sample.image_name
        with Image.open(image_path) as image:
            axis.imshow(image.convert("RGB"))
        axis.set_title(
            f"{sample.image_name}\n"
            f"Dataset: {sample.dataset_label} | Cluster: {sample.cluster_label}",
            fontsize=8,
        )
        axis.axis("off")

    for axis in axes.flat[len(samples) :]:
        axis.axis("off")
    figure.tight_layout()
    plt.show(block=True)
    plt.close(figure)


def plot_training_history(
    train_history: dict[str, list[float]],
    validation_history: dict[str, list[float]],
    validation_actual: list[int],
    validation_probabilities: list[float],
    plot_file: Path,
) -> None:
    """Plot mean cross-validation metrics and pooled validation diagnostics."""
    figure, axes = plt.subplots(2, 4, figsize=(18, 9))
    epochs = range(1, len(train_history["accuracy"]) + 1)

    for axis, metric in zip(axes.flat, METRIC_NAMES):
        axis.plot(epochs, train_history[metric], marker="o", label="Train")
        axis.plot(
            epochs,
            validation_history[metric],
            marker="o",
            label="Validation",
        )
        axis.set(
            title=metric.replace("_", " ").title(),
            xlabel="Epoch",
            ylabel="Score",
            ylim=(0, 1),
        )
        axis.grid(True, alpha=0.3)
        axis.legend()

    validation_predicted = [
        int(probability > 0.5) for probability in validation_probabilities
    ]
    if len(set(validation_actual)) > 1:
        false_positive, true_positive, _ = roc_curve(
            validation_actual, validation_probabilities
        )
        axes.flat[5].plot(false_positive, true_positive, label="Pooled validation")
        axes.flat[5].legend()
    axes.flat[5].plot([0, 1], [0, 1], "--", color="gray")
    axes.flat[5].set(
        title="Pooled validation ROC",
        xlabel="False positive rate",
        ylabel="True positive rate",
    )

    matrix = confusion_matrix(
        validation_actual, validation_predicted, labels=[0, 1]
    )
    axes.flat[6].imshow(matrix, cmap="Blues")
    axes.flat[6].set(
        title="Pooled validation confusion matrix",
        xlabel="Predicted label",
        ylabel="True label",
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=LABEL_NAMES,
        yticklabels=LABEL_NAMES,
    )
    for (row, column), value in np.ndenumerate(matrix):
        axes.flat[6].text(column, row, str(value), ha="center", va="center")
    axes.flat[7].axis("off")

    figure.suptitle("Mean 5-fold training and validation metrics")
    figure.tight_layout()
    plot_file.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(plot_file, dpi=150)
    plt.show()
    plt.close(figure)
    print(f"Training and validation plots saved to: {plot_file}")


def plot_test_metrics(
    actual: list[int],
    predicted: list[int],
    probabilities: list[float],
    plot_file: Path,
) -> None:
    """Plot only the test ROC-AUC curve and confusion matrix."""
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))

    if len(set(actual)) > 1:
        false_positive, true_positive, _ = roc_curve(actual, probabilities)
        auc = roc_auc_score(actual, probabilities)
        axes[0].plot(false_positive, true_positive, label=f"AUC = {auc:.3f}")
        axes[0].legend()
    axes[0].plot([0, 1], [0, 1], "--", color="gray")
    axes[0].set(
        title="Test ROC-AUC curve",
        xlabel="False positive rate",
        ylabel="True positive rate",
    )

    matrix = confusion_matrix(actual, predicted, labels=[0, 1])
    axes[1].imshow(matrix, cmap="Blues")
    axes[1].set(
        title="Test confusion matrix",
        xlabel="Predicted label",
        ylabel="True label",
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=LABEL_NAMES,
        yticklabels=LABEL_NAMES,
    )
    for (row, column), value in np.ndenumerate(matrix):
        axes[1].text(column, row, str(value), ha="center", va="center")

    figure.suptitle("Held-out test results")
    figure.tight_layout()
    plot_file.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(plot_file, dpi=150)
    plt.show()
    plt.close(figure)


def display_error_samples(
    dataframe: pd.DataFrame,
    image_root: Path,
    true_labels: list[int],
    predicted_labels: list[int],
    samples_per_error: int = 4,
    plot_file: Path | None = None,
) -> None:
    """Display and optionally save false-positive and false-negative examples."""
    false_positives = [
        index
        for index, (actual, predicted) in enumerate(zip(true_labels, predicted_labels))
        if actual == 0 and predicted == 1
    ][:samples_per_error]
    false_negatives = [
        index
        for index, (actual, predicted) in enumerate(zip(true_labels, predicted_labels))
        if actual == 1 and predicted == 0
    ][:samples_per_error]
    samples = [
        ("False positives (Normal predicted as Defective)", false_positives),
        ("False negatives (Defective predicted as Normal)", false_negatives),
    ]
    max_samples = max(map(len, (false_positives, false_negatives)), default=0)
    if max_samples == 0:
        print("No false positives or false negatives in the test predictions.")
        return

    figure, axes = plt.subplots(
        2,
        max_samples,
        figsize=(4 * max_samples, 8),
        squeeze=False,
    )

    for row, (title, indices) in enumerate(samples):
        for column, axis in enumerate(axes[row]):
            if column >= len(indices):
                axis.axis("off")
                continue
            index = indices[column]
            sample = dataframe.iloc[index]
            image_path = image_root / sample["split"] / sample["image_name"]
            with Image.open(image_path) as image:
                axis.imshow(image.convert("RGB"))
            axis.set_title(
                f"{sample['image_name']}\n"
                f"Predicted: {LABEL_NAMES[predicted_labels[index]]}\n"
                f"Actual: {LABEL_NAMES[true_labels[index]]}",
                color="red",
                fontsize=9,
            )
            axis.axis("off")
        axes[row, 0].set_ylabel(title, fontsize=10)

    figure.tight_layout()
    if plot_file is not None:
        plot_file.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(plot_file, dpi=150)
        print(f"Error examples saved to: {plot_file}")
    plt.show()
    plt.close(figure)
