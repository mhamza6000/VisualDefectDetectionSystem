"""Load a pretrained ConvNeXt image classifier for casting images."""

import torch.nn as nn
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
)

from Dataprep import ID2LABEL, LABEL2ID, CLASS_NAMES


MODEL_CHECKPOINT = "facebook/convnext-tiny-224"


def load_model(freeze_backbone: bool = True):
    """Return the image processor and binary ConvNeXt classifier."""
    processor = AutoImageProcessor.from_pretrained(MODEL_CHECKPOINT)
    model = AutoModelForImageClassification.from_pretrained(MODEL_CHECKPOINT)
    model.classifier = nn.Linear(model.classifier.in_features, len(CLASS_NAMES))
    nn.init.normal_(
        model.classifier.weight,
        mean=0.0,
        std=model.config.initializer_range,
    )
    nn.init.zeros_(model.classifier.bias)
    model.config.num_labels = len(CLASS_NAMES)
    model.config.id2label = ID2LABEL
    model.config.label2id = LABEL2ID
    if freeze_backbone:
        for parameter in model.base_model.parameters():
            parameter.requires_grad = False
        for parameter in model.classifier.parameters():
            parameter.requires_grad = True
    return processor, model
