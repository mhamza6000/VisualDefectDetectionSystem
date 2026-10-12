"""Tests for inference API input validation and prediction response."""

from io import BytesIO
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

import torch
from fastapi.testclient import TestClient
from PIL import Image

import api


class StubProcessor:
    """Create the model input tensor without image preprocessing dependencies."""

    image_mean = [0.5, 0.5, 0.5]
    image_std = [0.5, 0.5, 0.5]

    def __call__(self, *, images: Image.Image, return_tensors: str) -> dict:
        _ = images, return_tensors
        return {"pixel_values": torch.zeros((1, 3, 224, 224))}


class StubModel:
    """Return a fixed prediction for API tests."""

    config = SimpleNamespace(id2label={0: "Normal", 1: "Defective"})

    def __call__(self, *, pixel_values: torch.Tensor) -> SimpleNamespace:
        logits = torch.tensor([[0.5, 1.5]]).expand(pixel_values.shape[0], -1)
        return SimpleNamespace(logits=logits)


class LowConfidenceDefectModel(StubModel):
    """Return a defective score below 0.5 but above the configured threshold."""

    def __call__(self, *, pixel_values: torch.Tensor) -> SimpleNamespace:
        logits = torch.tensor([[0.2, 0.0]]).expand(pixel_values.shape[0], -1)
        return SimpleNamespace(logits=logits)


class HigherThresholdModel(LowConfidenceDefectModel):
    """Use the threshold selected for the saved model during validation."""

    config = SimpleNamespace(
        id2label={0: "Normal", 1: "Defective"},
        defective_threshold=0.6,
    )


class InferenceApiTests(TestCase):
    """Exercise health, successful inference, and rejected upload cases."""

    def client(self) -> TestClient:
        return TestClient(api.app)

    def runtime_patch(self):
        return patch(
            "api.load_inference_model",
            return_value=(StubProcessor(), StubModel(), torch.device("cpu")),
        )

    @staticmethod
    def image_bytes() -> bytes:
        buffer = BytesIO()
        Image.new("RGB", (16, 16), "white").save(buffer, format="PNG")
        return buffer.getvalue()

    def test_health_and_prediction_response(self) -> None:
        with self.runtime_patch(), self.client() as client:
            self.assertEqual(client.get("/health").json(), {"status": "ready"})
            response = client.post(
                "/predict",
                files={"image_file": ("product.png", self.image_bytes(), "image/png")},
            )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["predicted_class"], "Defective")
        self.assertGreater(body["confidence_score"], 0.5)
        self.assertEqual(set(body["class_probabilities"]), {"Normal", "Defective"})

    def test_rejects_non_image_content_type(self) -> None:
        with self.runtime_patch(), self.client() as client:
            response = client.post(
                "/predict",
                files={"image_file": ("product.txt", b"not an image", "text/plain")},
            )
        self.assertEqual(response.status_code, 415)

    def test_rejects_corrupt_image(self) -> None:
        with self.runtime_patch(), self.client() as client:
            response = client.post(
                "/predict",
                files={"image_file": ("broken.png", b"not an image", "image/png")},
            )
        self.assertEqual(response.status_code, 400)

    def test_rejects_oversized_upload(self) -> None:
        with self.runtime_patch(), self.client() as client:
            response = client.post(
                "/predict",
                files={
                    "image_file": (
                        "large.png",
                        b"x" * (api.MAX_IMAGE_BYTES + 1),
                        "image/png",
                    )
                },
            )
        self.assertEqual(response.status_code, 413)

    def test_uses_recall_focused_defective_threshold(self) -> None:
        with patch(
            "api.load_inference_model",
            return_value=(
                StubProcessor(),
                LowConfidenceDefectModel(),
                torch.device("cpu"),
            ),
        ), self.client() as client:
            response = client.post(
                "/predict",
                files={"image_file": ("product.png", self.image_bytes(), "image/png")},
            )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["predicted_class"], "Defective")
        self.assertAlmostEqual(
            body["confidence_score"],
            body["class_probabilities"]["Defective"],
        )

    def test_uses_validation_selected_model_threshold(self) -> None:
        with patch(
            "api.load_inference_model",
            return_value=(
                StubProcessor(),
                HigherThresholdModel(),
                torch.device("cpu"),
            ),
        ), self.client() as client:
            response = client.post(
                "/predict",
                files={"image_file": ("product.png", self.image_bytes(), "image/png")},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["predicted_class"], "Normal")
