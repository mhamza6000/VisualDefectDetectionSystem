"""Run the complete casting-image preparation, training, and evaluation pipeline."""

import os

os.environ.setdefault("MPLBACKEND", "Agg")

from Dataprep import DATASET_DIR, prepare_dataset
from evaluate_model import evaluate_test_set
from hyperparameter_search import tune_hyperparameters
from kmeans_check import run_kmeans_check
from model import MODEL_CHECKPOINT
from train_model import TrainingConfig, train_cross_validation


RUN_KMEANS_LABEL_CHECK = False
RUN_HYPERPARAMETER_SEARCH = False
SHOW_DATASET_PREVIEW = False
SHOW_KMEANS_PREVIEW = False


def main() -> None:
    """Prepare data, tune hyperparameters, train with CV, then test."""
    print("1/5: Preparing dataset")
    dataframe = prepare_dataset(show_samples=SHOW_DATASET_PREVIEW)

    if RUN_KMEANS_LABEL_CHECK:
        print("\n2/5: Checking labels with K-means")
        run_kmeans_check(
            dataset_dir=DATASET_DIR,
            dataframe=dataframe,
            show_images=SHOW_KMEANS_PREVIEW,
        )
    else:
        print("\n2/5: Skipping K-means label check")

    if RUN_HYPERPARAMETER_SEARCH:
        print(f"\n3/5: Searching hyperparameters for {MODEL_CHECKPOINT}")
        training_config = tune_hyperparameters()
    else:
        print("\n3/5: Skipping hyperparameter search; using default settings")
        training_config = TrainingConfig()

    print("\n4/5: Training with the selected hyperparameters")
    train_cross_validation(training_config)

    print("\n5/5: Evaluating on the held-out test split")
    evaluate_test_set()


if __name__ == "__main__":
    main()
