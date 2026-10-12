"""Load image preprocessing settings and build the shared image transform."""

from pathlib import Path
from typing import Any

import torch
import yaml
from PIL import Image
from torchvision import transforms
from transformers import AutoImageProcessor


CONFIG_FILE = Path(__file__).resolve().parent / "config.yaml"
with CONFIG_FILE.open(encoding="utf-8") as config_file:
    CONFIG: dict[str, Any] = yaml.safe_load(config_file)

if not isinstance(CONFIG, dict):
    raise ValueError(f"Configuration file must contain a YAML mapping: {CONFIG_FILE}")

PREPROCESSING_CONFIG = CONFIG["preprocessing"]
TRAINING_CONFIG = CONFIG["training"]
INFERENCE_CONFIG = CONFIG["inference"]
IMAGE_SIZE = int(PREPROCESSING_CONFIG["image_size"])
BATCH_SIZE = int(TRAINING_CONFIG["batch_size"])
DEFECTIVE_THRESHOLD = float(INFERENCE_CONFIG["defective_threshold"])
if IMAGE_SIZE < 1:
    raise ValueError("preprocessing.image_size must be greater than zero.")
if BATCH_SIZE < 1:
    raise ValueError("training.batch_size must be greater than zero.")
if not 0.0 < DEFECTIVE_THRESHOLD < 1.0:
    raise ValueError("inference.defective_threshold must be between 0 and 1.")
if TRAINING_CONFIG["imbalance_strategy"] not in {"inverse_frequency", "none"}:
    raise ValueError(
        "training.imbalance_strategy must be 'inverse_frequency' or 'none'."
    )


def build_image_transform(
    processor: AutoImageProcessor, augment: bool = False
) -> transforms.Compose:
    """Build ConvNeXt preprocessing with optional training-only augmentation."""
    pipeline = [
        transforms.Grayscale(num_output_channels=3)
        if PREPROCESSING_CONFIG["grayscale"]
        else transforms.Lambda(lambda image: image.convert("RGB"))
    ]
    blur_config = PREPROCESSING_CONFIG["gaussian_blur"]
    if blur_config["enabled"]:
        pipeline.append(
            transforms.GaussianBlur(
                kernel_size=blur_config["kernel_size"],
                sigma=blur_config["sigma"],
            )
        )
    pipeline.append(transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)))

    augmentation = PREPROCESSING_CONFIG["augmentation"]
    if augment:
        if augmentation["horizontal_flip"]:
            pipeline.append(transforms.RandomHorizontalFlip())
        if augmentation["vertical_flip"]:
            pipeline.append(transforms.RandomVerticalFlip())
        if augmentation["rotation_degrees"]:
            pipeline.append(
                transforms.RandomRotation(augmentation["rotation_degrees"])
            )
        if augmentation["translation_fraction"]:
            fraction = augmentation["translation_fraction"]
            pipeline.append(
                transforms.RandomAffine(degrees=0, translate=(fraction, fraction))
            )

    pipeline.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=processor.image_mean,
                std=processor.image_std,
            ),
        ]
    )
    return transforms.Compose(pipeline)


def preprocess_image(image: Image.Image, processor: AutoImageProcessor) -> torch.Tensor:
    """Preprocess one image for inference using the training pipeline."""
    return build_image_transform(processor)(image)


def classify_probabilities(
    probabilities: list[float], threshold: float = DEFECTIVE_THRESHOLD
) -> list[int]:
    """Apply the configured or validation-selected defect threshold."""
    return [int(probability >= threshold) for probability in probabilities]
