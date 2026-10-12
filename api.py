"""FastAPI inference service for the trained product classifier."""

import logging
import os
from contextlib import asynccontextmanager
from io import BytesIO
from pathlib import Path
from typing import Annotated

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel
from transformers import AutoImageProcessor, AutoModelForImageClassification

from preprocessing import (
    DEFECTIVE_THRESHOLD,
    classify_probabilities,
    preprocess_image,
)


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = Path(os.getenv("MODEL_DIR", BASE_DIR / "casting_imagefolder" / "final_model"))
MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_IMAGE_PIXELS = 40_000_000
logger = logging.getLogger("classifier_api")


class PredictionResponse(BaseModel):
    """Prediction output returned by the image-classification endpoint."""

    predicted_class: str
    confidence_score: float
    class_probabilities: dict[str, float]


def load_inference_model():
    """Load the saved processor and classifier onto the available device."""
    if not MODEL_DIR.is_dir():
        raise FileNotFoundError(
            f"Model checkpoint not found at {MODEL_DIR}; train the model first."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = AutoImageProcessor.from_pretrained(
        MODEL_DIR, local_files_only=True
    )
    model = AutoModelForImageClassification.from_pretrained(
        MODEL_DIR, local_files_only=True
    ).to(device)
    model.eval()
    return processor, model, device


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model once when the service starts."""
    try:
        processor, model, device = load_inference_model()
    except Exception:
        logger.exception("Could not load classifier from %s", MODEL_DIR)
        raise

    app.state.processor = processor
    app.state.model = model
    app.state.device = device
    logger.info("Classifier loaded from %s on %s", MODEL_DIR, device)
    yield


app = FastAPI(
    title="Product Defect Classifier",
    description="Classify submitted product images as Normal or Defective.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health_check() -> dict[str, str]:
    """Report whether the application has loaded its inference model."""
    if not hasattr(app.state, "model"):
        raise HTTPException(status_code=503, detail="Model is not loaded.")
    return {"status": "ready"}


@app.post("/predict", response_model=PredictionResponse)
async def predict_image(
    image_file: Annotated[UploadFile, File(description="Product image to classify")],
) -> PredictionResponse:
    """Validate and classify one uploaded image."""
    try:
        if not image_file.content_type or not image_file.content_type.startswith("image/"):
            raise HTTPException(
                status_code=415,
                detail="Upload a file with an image content type.",
            )

        image_bytes = await image_file.read(MAX_IMAGE_BYTES + 1)
        if not image_bytes:
            raise HTTPException(status_code=400, detail="The uploaded image is empty.")
        if len(image_bytes) > MAX_IMAGE_BYTES:
            raise HTTPException(status_code=413, detail="Image exceeds the 10 MB size limit.")

        try:
            with Image.open(BytesIO(image_bytes)) as image:
                if image.width * image.height > MAX_IMAGE_PIXELS:
                    raise HTTPException(
                        status_code=413,
                        detail="Image dimensions exceed the supported pixel limit.",
                    )
                image.verify()
            with Image.open(BytesIO(image_bytes)) as image:
                rgb_image = image.convert("RGB")
        except (
            UnidentifiedImageError,
            OSError,
            ValueError,
            Image.DecompressionBombError,
        ) as exc:
            logger.info("Rejected an invalid uploaded image.")
            raise HTTPException(
                status_code=400,
                detail="Uploaded file is not a valid image.",
            ) from exc
    finally:
        await image_file.close()

    processor = app.state.processor
    model = app.state.model
    device = app.state.device
    inputs = preprocess_image(rgb_image, processor).unsqueeze(0)
    try:
        with torch.inference_mode():
            logits = model(pixel_values=inputs.to(device)).logits[0]
            probabilities = logits.softmax(dim=0)
    except Exception as exc:
        logger.exception("Classifier inference failed.")
        raise HTTPException(status_code=500, detail="Image classification failed.") from exc

    label_map = {
        int(index): label for index, label in model.config.id2label.items()
    }
    positive_probability = float(probabilities[1].item())
    decision_threshold = float(
        getattr(model.config, "defective_threshold", DEFECTIVE_THRESHOLD)
    )
    predicted_index = classify_probabilities(
        [positive_probability], decision_threshold
    )[0]
    class_probabilities = {
        label_map[index]: float(probabilities[index].item())
        for index in range(len(probabilities))
    }
    prediction = PredictionResponse(
        predicted_class=label_map[predicted_index],
        confidence_score=float(probabilities[predicted_index].item()),
        class_probabilities=class_probabilities,
    )
    logger.info(
        "Image classified as %s with confidence %.4f",
        prediction.predicted_class,
        prediction.confidence_score,
    )
    return prediction
