"""Tests for class-balanced loss and stratified dataset partitions."""

from tempfile import TemporaryDirectory
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch
import warnings

import pandas as pd
import torch
from PIL import Image

from Dataprep import (
    CLASS_NAMES,
    IMAGE_SIZE,
    SOURCE_CLASSES,
    CastingImageDataset,
    build_image_transform,
    split_images,
)
from train_model import (
    TrainingConfig,
    build_optimizer,
    class_weighted_loss,
    select_f1_threshold,
)


class TrainingDataTests(TestCase):
    """Check split ratios/stratification and imbalance-aware loss weights."""

    def test_split_creates_stratified_train_validation_and_test_sets(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source"
            output = root / "output"
            rows = []
            for class_index, label in enumerate(CLASS_NAMES):
                source_class = source / SOURCE_CLASSES[label]
                source_class.mkdir(parents=True)
                for index in range(50):
                    filename = f"{class_index}_{index}.jpeg"
                    (source_class / filename).write_bytes(
                        f"{label}-{index}".encode()
                    )
                    rows.append(
                        {
                            "image_name": filename,
                            "label": label,
                            "sha256": f"{label}-{index}",
                        }
                    )

            manifest = split_images(pd.DataFrame(rows), source, output, seed=13)

            self.assertEqual(manifest["split"].value_counts().to_dict(), {
                "train": 70,
                "validation": 15,
                "test": 15,
            })
            class_split_counts = pd.crosstab(manifest["split"], manifest["label"])
            self.assertTrue((class_split_counts.min(axis=1) >= 7).all())
            for split in ("train", "validation", "test"):
                self.assertEqual(len(list((output / split).iterdir())), 15 if split != "train" else 70)

    def test_loss_weights_upweight_the_less_frequent_class(self) -> None:
        training_data = pd.DataFrame(
            {"label": ["Normal"] * 8 + ["Defective"] * 2}
        )
        loss_function = class_weighted_loss(training_data, torch.device("cpu"))

        self.assertIsNotNone(loss_function.weight)
        self.assertGreater(loss_function.weight[1], loss_function.weight[0])

    def test_f1_threshold_is_selected_from_validation_predictions(self) -> None:
        threshold = select_f1_threshold(
            actual=[0, 1, 1, 1],
            probabilities=[0.1, 0.45, 0.6, 0.9],
        )

        self.assertAlmostEqual(threshold, 0.45)

    def test_optimizer_uses_smaller_learning_rate_for_pretrained_backbone(self) -> None:
        class TinyClassifier(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.base_model = torch.nn.Linear(2, 2)
                self.classifier = torch.nn.Linear(2, 2)

        config = TrainingConfig(learning_rate=1e-3)
        optimizer = build_optimizer(TinyClassifier(), config)

        self.assertEqual(len(optimizer.param_groups), 2)
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 1e-5)
        self.assertAlmostEqual(optimizer.param_groups[1]["lr"], 1e-3)

    def test_preprocessing_outputs_normalized_grayscale_convnext_tensor(self) -> None:
        processor = type(
            "Processor",
            (),
            {"image_mean": [0.5, 0.5, 0.5], "image_std": [0.5, 0.5, 0.5]},
        )()
        image = Image.new("RGB", (40, 20), (128, 128, 128))

        pixels = build_image_transform(processor)(image)

        self.assertEqual(tuple(pixels.shape), (3, IMAGE_SIZE, IMAGE_SIZE))
        self.assertTrue(torch.allclose(pixels[0], pixels[1]))
        self.assertTrue(torch.allclose(pixels[1], pixels[2]))
        self.assertTrue(torch.all((pixels >= -1) & (pixels <= 1)))

    def test_configured_gaussian_blur_is_applied(self) -> None:
        processor = type(
            "Processor",
            (),
            {"image_mean": [0.5, 0.5, 0.5], "image_std": [0.5, 0.5, 0.5]},
        )()
        image = Image.new("RGB", (40, 40), "black")
        image.putpixel((20, 20), (255, 255, 255))

        pixels = build_image_transform(processor)(image)
        center = pixels[0, 112, 112]
        adjacent = pixels[0, 112, 113]

        self.assertLess(center, 1.0)
        self.assertGreater(adjacent, -1.0)

    def test_dataset_loads_original_when_split_copy_is_missing(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            image_root = root / "output"
            source_root = root / "source"
            label = "Defective"
            image_name = "defect.png"
            source_image = source_root / SOURCE_CLASSES[label] / image_name
            source_image.parent.mkdir(parents=True)
            Image.new("RGB", (16, 16), "white").save(source_image)
            (image_root / "validation").mkdir(parents=True)
            processor = type(
                "Processor",
                (),
                {"image_mean": [0.5, 0.5, 0.5], "image_std": [0.5, 0.5, 0.5]},
            )()
            dataset = CastingImageDataset(
                pd.DataFrame(
                    [{"image_name": image_name, "label": label, "split": "validation"}]
                ),
                image_root,
                processor,
            )

            with patch("Dataprep.DATASET_DIR", source_root), warnings.catch_warnings(
                record=True
            ) as caught_warnings:
                warnings.simplefilter("always")
                pixels, class_id = dataset[0]

        self.assertEqual(tuple(pixels.shape), (3, IMAGE_SIZE, IMAGE_SIZE))
        self.assertEqual(class_id, 1)
        self.assertTrue(
            any(
                "Split copy is missing" in str(item.message)
                for item in caught_warnings
            )
        )
