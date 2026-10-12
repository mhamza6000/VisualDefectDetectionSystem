"""Quickly compare training settings using one training epoch and validation split."""

import json
import random

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from Dataprep import CastingImageDataset, OUTPUT_DIR, RANDOM_SEED
from model import load_model
from train_model import (
    EPOCHS,
    FREEZE_BACKBONE,
    TrainingConfig,
    build_optimizer,
    class_weighted_loss,
    get_predictions,
    metric_values,
    train_epoch,
)

SEARCH_TRIALS = 5
SEARCH_EPOCHS = 1
BATCH_SIZE_OPTIONS = (8, 16, 32)
WEIGHT_DECAY_OPTIONS = (0.0, 0.001, 0.01)
RESULTS_FILE = OUTPUT_DIR / "random_search_results.csv"
BEST_CONFIG_FILE = OUTPUT_DIR / "best_hyperparameters.json"


def random_configurations(seed: int, trials: int) -> list[TrainingConfig]:
    """Sample unique configurations from the supported training options."""
    generator = random.Random(seed)
    configurations = []
    seen = set()
    while len(configurations) < trials:
        values = (
            round(10 ** generator.uniform(-5, -3), 8),
            generator.choice(BATCH_SIZE_OPTIONS),
            generator.choice(WEIGHT_DECAY_OPTIONS),
        )
        if values not in seen:
            seen.add(values)
            configurations.append(
                TrainingConfig(
                    learning_rate=values[0],
                    batch_size=values[1],
                    weight_decay=values[2],
                    epochs=SEARCH_EPOCHS,
                )
            )
    return configurations


def tune_hyperparameters() -> TrainingConfig:
    """Find a quick configuration using one pass and the existing validation split."""
    manifest = OUTPUT_DIR / "image_labels.csv"
    if not manifest.is_file():
        raise FileNotFoundError(f"Run dataset preparation first; missing {manifest}")

    dataframe = pd.read_csv(manifest)
    training_data = dataframe[dataframe["split"] == "train"].reset_index(drop=True)
    validation_data = dataframe[
        dataframe["split"] == "validation"
    ].reset_index(drop=True)
    if training_data.empty:
        raise ValueError("No training images found in the manifest.")
    if validation_data.empty or validation_data["label"].nunique() < 2:
        raise ValueError("The validation split must contain examples of both classes.")

    configurations = random_configurations(RANDOM_SEED, SEARCH_TRIALS)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trial_results = []

    for trial, config in enumerate(
        tqdm(configurations, desc="Random-search trials"), start=1
    ):
        print(
            f"\nTrial {trial}/{len(configurations)}: "
            f"learning_rate={config.learning_rate:g}, "
            f"batch_size={config.batch_size}, "
            f"weight_decay={config.weight_decay:g}, "
            f"search_epochs={SEARCH_EPOCHS}",
            flush=True,
        )

        seed = RANDOM_SEED + trial
        torch.manual_seed(seed)
        processor, model = load_model(freeze_backbone=FREEZE_BACKBONE)
        model.to(device)
        train_loader = DataLoader(
            CastingImageDataset(
                training_data, OUTPUT_DIR, processor, augment=True
            ),
            batch_size=config.batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(seed),
        )
        validation_loader = DataLoader(
            CastingImageDataset(validation_data, OUTPUT_DIR, processor),
            batch_size=config.batch_size,
            shuffle=False,
        )
        optimizer = build_optimizer(model, config)
        loss_function = class_weighted_loss(training_data, device)
        train_epoch(model, train_loader, loss_function, optimizer, device)
        actual, predicted, probabilities, _ = get_predictions(
            model,
            validation_loader,
            loss_function,
            device,
            description=f"Trial {trial} validation",
        )
        validation_auc = metric_values(actual, predicted, probabilities)["roc_auc"]

        result = {
            "trial": trial,
            "learning_rate": config.learning_rate,
            "batch_size": config.batch_size,
            "weight_decay": config.weight_decay,
            "search_epochs": SEARCH_EPOCHS,
            "validation_roc_auc": validation_auc,
        }
        trial_results.append(result)
        print(f"Trial {trial} validation ROC-AUC: {validation_auc:.4f}")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    results = pd.DataFrame(trial_results).sort_values(
        "validation_roc_auc", ascending=False
    )
    results.to_csv(RESULTS_FILE, index=False)
    best_trial = results.iloc[0]
    best_config = TrainingConfig(
        learning_rate=float(best_trial["learning_rate"]),
        batch_size=int(best_trial["batch_size"]),
        weight_decay=float(best_trial["weight_decay"]),
        epochs=EPOCHS,
    )
    with BEST_CONFIG_FILE.open("w", encoding="utf-8") as config_file:
        json.dump(best_config.__dict__, config_file, indent=2)

    print("\nBest random-search hyperparameters:")
    print(f"  learning rate: {best_config.learning_rate:g}")
    print(f"  batch size: {best_config.batch_size}")
    print(f"  weight decay: {best_config.weight_decay:g}")
    print(f"  final training epochs: {best_config.epochs}")
    print(f"  search epochs per trial: {SEARCH_EPOCHS}")
    print(f"  validation ROC-AUC: {best_trial['validation_roc_auc']:.4f}")
    print(f"Search results saved to: {RESULTS_FILE}")
    print(f"Best hyperparameters saved to: {BEST_CONFIG_FILE}")
    return best_config
