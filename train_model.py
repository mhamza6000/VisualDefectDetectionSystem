"""Fine-tune ConvNeXt with stratified five-fold cross-validation."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    precision_recall_curve,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from Dataprep import CLASS_NAMES, CastingImageDataset, OUTPUT_DIR, RANDOM_SEED
from model import load_model
from preprocessing import BATCH_SIZE, TRAINING_CONFIG, classify_probabilities
from visualization import plot_training_history


FREEZE_BACKBONE = False
BACKBONE_LR_MULTIPLIER = 0.01
FOLDS = 5
EPOCHS = 5
IMBALANCE_STRATEGY = TRAINING_CONFIG["imbalance_strategy"]
LEARNING_RATE = 1e-3
CHECKPOINT_DIR = OUTPUT_DIR / "cv_models"
FINAL_MODEL_DIR = OUTPUT_DIR / "final_model"
PLOT_DIR = OUTPUT_DIR / "training_plots"


@dataclass(frozen=True)
class TrainingConfig:
    """Hyperparameters used by cross-validation training."""

    learning_rate: float = LEARNING_RATE
    batch_size: int = BATCH_SIZE
    weight_decay: float = 0.01
    epochs: int = EPOCHS


def build_optimizer(model: nn.Module, config: TrainingConfig) -> AdamW:
    """Use a smaller learning rate for pretrained features than for the head."""
    parameter_groups = [
        {
            "params": [
                parameter
                for parameter in model.base_model.parameters()
                if parameter.requires_grad
            ],
            "lr": config.learning_rate * BACKBONE_LR_MULTIPLIER,
        },
        {
            "params": [
                parameter
                for parameter in model.classifier.parameters()
                if parameter.requires_grad
            ],
            "lr": config.learning_rate,
        },
    ]
    parameter_groups = [group for group in parameter_groups if group["params"]]
    return AdamW(parameter_groups, weight_decay=config.weight_decay)


def get_predictions(
    model: nn.Module,
    loader: DataLoader,
    loss_function: nn.Module,
    device: torch.device,
    description: str = "Evaluating batches",
) -> tuple[list[int], list[int], list[float], float]:
    """Collect labels, probabilities, and mean loss without model updates."""
    model.eval()
    actual, predicted, probabilities = [], [], []
    total_loss = total = 0
    with torch.no_grad():
        for images, labels in tqdm(loader, desc=description, leave=False):
            labels = labels.to(device)
            logits = model(pixel_values=images.to(device)).logits
            total_loss += loss_function(logits, labels).item() * labels.size(0)
            total += labels.size(0)
            actual.extend(labels.tolist())
            batch_probabilities = logits.softmax(dim=1)[:, 1].cpu().tolist()
            predicted.extend(classify_probabilities(batch_probabilities))
            probabilities.extend(batch_probabilities)
    return actual, predicted, probabilities, total_loss / total


def metric_values(
    actual: list[int], predicted: list[int], probabilities: list[float]
) -> dict[str, float]:
    """Calculate the requested binary classification metrics."""
    return {
        "accuracy": accuracy_score(actual, predicted),
        "precision": precision_score(actual, predicted, zero_division=0),
        "recall": recall_score(actual, predicted, zero_division=0),
        "f1": f1_score(actual, predicted, zero_division=0),
        "roc_auc": (
            roc_auc_score(actual, probabilities)
            if len(set(actual)) > 1
            else float("nan")
        ),
    }


def select_f1_threshold(actual: list[int], probabilities: list[float]) -> float:
    """Choose the validation threshold that maximizes Defective-class F1."""
    precision, recall, thresholds = precision_recall_curve(actual, probabilities)
    if not len(thresholds):
        raise ValueError("At least two distinct validation probabilities are required.")
    f1_scores = (
        2 * precision[:-1] * recall[:-1]
        / (precision[:-1] + recall[:-1] + 1e-12)
    )
    best_index = max(
        range(len(thresholds)),
        key=lambda index: (f1_scores[index], thresholds[index]),
    )
    return float(thresholds[best_index])


def class_weighted_loss(
    training_data: pd.DataFrame, device: torch.device
) -> nn.CrossEntropyLoss:
    """Weight each class inversely to its frequency in the training partition."""
    if IMBALANCE_STRATEGY == "none":
        return nn.CrossEntropyLoss()
    counts = training_data["label"].value_counts().reindex(CLASS_NAMES, fill_value=0)
    if (counts == 0).any():
        raise ValueError("The training partition must contain every class.")
    weights = torch.tensor(
        len(training_data) / (len(CLASS_NAMES) * counts.to_numpy()),
        dtype=torch.float32,
        device=device,
    )
    return nn.CrossEntropyLoss(weight=weights)


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_function: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Train for one epoch and return mean training loss."""
    model.train()
    total_loss = 0.0
    total = 0
    progress = tqdm(loader, desc="Training batches", leave=False)
    for images, labels in progress:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = loss_function(model(pixel_values=images).logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * labels.size(0)
        total += labels.size(0)
        progress.set_postfix(loss=f"{loss.item():.4f}")
    return total_loss / total


def train_final_model(
    training_data: pd.DataFrame,
    validation_data: pd.DataFrame,
    config: TrainingConfig,
    device: torch.device,
) -> None:
    """Train on training data and select the saved model by validation loss."""
    training_data = training_data.copy()
    training_data["split"] = "train"
    validation_data = validation_data.copy()
    validation_data["split"] = "validation"
    torch.manual_seed(RANDOM_SEED)
    processor, model = load_model(freeze_backbone=FREEZE_BACKBONE)
    model.to(device)
    train_loader = DataLoader(
        CastingImageDataset(training_data, OUTPUT_DIR, processor, augment=True),
        batch_size=config.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(RANDOM_SEED),
    )
    validation_loader = DataLoader(
        CastingImageDataset(validation_data, OUTPUT_DIR, processor),
        batch_size=config.batch_size,
        shuffle=False,
    )
    optimizer = build_optimizer(model, config)
    loss_function = class_weighted_loss(training_data, device)

    best_validation_loss = float("inf")
    FINAL_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    print("\nTraining final model with validation-based checkpoint selection")
    for epoch in tqdm(range(1, config.epochs + 1), desc="Final model epochs"):
        train_loss = train_epoch(model, train_loader, loss_function, optimizer, device)
        validation_actual, _, validation_probabilities, validation_loss = get_predictions(
            model, validation_loader, loss_function, device,
            description="Final model validation",
        )
        validation_threshold = select_f1_threshold(
            validation_actual, validation_probabilities
        )
        print(
            f"Final model epoch {epoch}/{config.epochs} | "
            f"train loss {train_loss:.4f} | validation loss {validation_loss:.4f} | "
            f"validation F1 threshold {validation_threshold:.4f}"
        )
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            model.config.defective_threshold = validation_threshold
            model.save_pretrained(FINAL_MODEL_DIR)
            processor.save_pretrained(FINAL_MODEL_DIR)

    print(f"Final model saved to: {FINAL_MODEL_DIR}")
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()


def train_cross_validation(config: TrainingConfig | None = None) -> None:
    """Run stratified cross-validation using only the training split."""
    config = config or TrainingConfig()
    print("Training hyperparameters:")
    print(f"  learning rate: {config.learning_rate:g}")
    print(f"  batch size: {config.batch_size}")
    print(f"  weight decay: {config.weight_decay:g}")
    print(f"  epochs: {config.epochs}")

    manifest = OUTPUT_DIR / "image_labels.csv"
    if not manifest.is_file():
        raise FileNotFoundError(f"Run Dataprep.py first; missing {manifest}")

    dataframe = pd.read_csv(manifest)
    training_data = dataframe[dataframe["split"] == "train"].reset_index(drop=True)
    validation_data = dataframe[
        dataframe["split"] == "validation"
    ].reset_index(drop=True)
    if training_data.empty:
        raise ValueError("No training images found in the manifest.")
    if validation_data.empty:
        raise ValueError("No validation images found in the manifest.")
    if training_data["label"].value_counts().min() < FOLDS:
        raise ValueError("Each class needs at least five images for 5-fold CV.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    folds = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=RANDOM_SEED)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    fold_train_histories = []
    fold_validation_histories = []
    pooled_validation_actual = []
    pooled_validation_probabilities = []

    fold_splits = list(folds.split(training_data, training_data["label"]))
    for fold, (train_idx, validation_idx) in enumerate(
        tqdm(fold_splits, desc="Cross-validation folds"), start=1
    ):
        print(f"\nFold {fold}/{FOLDS}", flush=True)
        train_data = training_data.iloc[train_idx].copy()
        fold_validation_data = training_data.iloc[validation_idx].copy()
        train_data["split"] = fold_validation_data["split"] = "train"

        torch.manual_seed(RANDOM_SEED + fold)
        processor, model = load_model(freeze_backbone=FREEZE_BACKBONE)
        model.to(device)
        train_loader = DataLoader(
            CastingImageDataset(train_data, OUTPUT_DIR, processor, augment=True),
            batch_size=config.batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(RANDOM_SEED + fold),
        )
        train_metrics_loader = DataLoader(
            CastingImageDataset(train_data, OUTPUT_DIR, processor),
            batch_size=config.batch_size,
            shuffle=False,
        )
        validation_loader = DataLoader(
            CastingImageDataset(fold_validation_data, OUTPUT_DIR, processor),
            batch_size=config.batch_size,
            shuffle=False,
        )
        optimizer = build_optimizer(model, config)
        loss_function = class_weighted_loss(train_data, device)
        best_validation_loss = float("inf")
        best_validation_predictions = None
        train_history = {metric: [] for metric in ("accuracy", "precision", "recall", "f1", "roc_auc")}
        validation_history = {
            metric: [] for metric in ("accuracy", "precision", "recall", "f1", "roc_auc")
        }
        fold_dir = CHECKPOINT_DIR / f"fold_{fold}"

        epoch_progress = tqdm(
            range(1, config.epochs + 1), desc=f"Fold {fold} epochs"
        )
        for epoch in epoch_progress:
            train_loss = train_epoch(
                model, train_loader, loss_function, optimizer, device
            )
            train_actual, train_pred, train_prob, _ = get_predictions(
                model,
                train_metrics_loader,
                loss_function,
                device,
                description="Train metrics",
            )
            val_actual, val_pred, val_prob, val_loss = get_predictions(
                model,
                validation_loader,
                loss_function,
                device,
                description="Validation batches",
            )
            train_scores = metric_values(train_actual, train_pred, train_prob)
            val_scores = metric_values(val_actual, val_pred, val_prob)
            for metric in train_history:
                train_history[metric].append(train_scores[metric])
                validation_history[metric].append(val_scores[metric])
            epoch_progress.set_postfix(
                train_loss=f"{train_loss:.4f}",
                val_loss=f"{val_loss:.4f}",
                val_accuracy=f"{val_scores['accuracy']:.3f}",
            )

            print(
                f"\nEpoch {epoch}/{config.epochs} | train loss {train_loss:.4f} | "
                f"train acc {train_scores['accuracy']:.3f}, "
                f"precision {train_scores['precision']:.3f}, "
                f"recall {train_scores['recall']:.3f}, "
                f"F1 {train_scores['f1']:.3f}, AUC {train_scores['roc_auc']:.3f} | "
                f"validation loss {val_loss:.4f}, "
                f"validation acc {val_scores['accuracy']:.3f}, "
                f"precision {val_scores['precision']:.3f}, "
                f"recall {val_scores['recall']:.3f}, "
                f"F1 {val_scores['f1']:.3f}, AUC {val_scores['roc_auc']:.3f}"
            )
            if val_loss < best_validation_loss:
                best_validation_loss = val_loss
                model.save_pretrained(fold_dir)
                processor.save_pretrained(fold_dir)
                best_validation_predictions = (val_actual, val_prob)

        fold_train_histories.append(train_history)
        fold_validation_histories.append(validation_history)
        if best_validation_predictions is not None:
            pooled_validation_actual.extend(best_validation_predictions[0])
            pooled_validation_probabilities.extend(best_validation_predictions[1])
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    mean_train_history = {
        metric: np.mean([history[metric] for history in fold_train_histories], axis=0)
        for metric in fold_train_histories[0]
    }
    mean_validation_history = {
        metric: np.mean(
            [history[metric] for history in fold_validation_histories], axis=0
        )
        for metric in fold_validation_histories[0]
    }
    plot_training_history(
        mean_train_history,
        mean_validation_history,
        pooled_validation_actual,
        pooled_validation_probabilities,
        PLOT_DIR / "cross_validation_metrics.png",
    )
    train_final_model(training_data, validation_data, config, device)
    print(f"\nFold models saved in: {CHECKPOINT_DIR}")
