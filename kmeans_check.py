"""Compare K-means image clusters with the dataset's folder labels."""

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

from Dataprep import (
    BASE_DIR,
    SOURCE_CLASSES,
    create_image_dataframe,
    extract_dataset,
)
from visualization import display_mismatches as show_mismatches


FEATURE_SIZE = (32, 32)
RANDOM_SEED = 42
RESULTS_FILE = BASE_DIR / "kmeans_label_check.csv"


def image_features(dataframe: pd.DataFrame, dataset_dir: Path) -> np.ndarray:
    """Represent each image as a small normalized grayscale pixel vector."""
    features = []
    for row in dataframe.itertuples(index=False):
        image_path = dataset_dir / SOURCE_CLASSES[row.label] / row.image_name
        with Image.open(image_path) as image:
            resized = image.convert("L").resize(FEATURE_SIZE)
            features.append(np.asarray(resized, dtype=np.float32).ravel() / 255)
    return np.stack(features)


def run_kmeans_check(
    dataset_dir: Path | None = None,
    dataframe: pd.DataFrame | None = None,
    show_images: bool = True,
) -> pd.DataFrame:
    """Compare unsupervised clusters with labels and return mismatch candidates."""
    dataset_dir = dataset_dir or extract_dataset()
    dataframe = dataframe.copy() if dataframe is not None else create_image_dataframe(dataset_dir)
    features = image_features(dataframe, dataset_dir)
    labels = sorted(dataframe["label"].unique())

    clusters = KMeans(n_clusters=len(labels), random_state=RANDOM_SEED, n_init=10)
    dataframe["cluster"] = clusters.fit_predict(features)

    counts = pd.crosstab(dataframe["label"], dataframe["cluster"])
    label_indices, cluster_indices = linear_sum_assignment(-counts.to_numpy())
    cluster_labels = {
        counts.columns[cluster_index]: counts.index[label_index]
        for label_index, cluster_index in zip(label_indices, cluster_indices)
    }
    dataframe["cluster_label"] = dataframe["cluster"].map(cluster_labels)
    dataframe["label_matches"] = dataframe["label"] == dataframe["cluster_label"]

    agreement = dataframe["label_matches"].mean()
    ari = adjusted_rand_score(dataframe["label"], dataframe["cluster"])
    print(f"Cluster/label agreement (best cluster mapping): {agreement:.1%}")
    print(f"Adjusted Rand index: {ari:.3f}")
    print("\nCluster to label mapping:")
    print(cluster_labels)

    class_stats = dataframe.groupby("label")["label_matches"].agg(
        matched="sum", total="count"
    )
    class_stats["mismatched"] = class_stats["total"] - class_stats["matched"]
    class_stats["agreement"] = class_stats["matched"] / class_stats["total"]
    print("\nAgreement by class:")
    print(class_stats.to_string())

    mismatches = dataframe.loc[
        ~dataframe["label_matches"], ["image_name", "label", "cluster_label"]
    ].rename(columns={"label": "dataset_label"})
    mismatches.to_csv(RESULTS_FILE, index=False)
    print(f"\nPotentially mislabeled images: {len(mismatches)}")
    if not mismatches.empty:
        print("\nFirst 12 candidates with their dataset and cluster labels:")
        print(mismatches.head(12).to_string(index=False))
        if show_images:
            show_mismatches(mismatches, dataset_dir, SOURCE_CLASSES)
    print(f"\nMismatched image names and labels saved to: {RESULTS_FILE}")
    print("Note: cluster disagreement flags candidates; it does not prove a label is wrong.")
    return mismatches
