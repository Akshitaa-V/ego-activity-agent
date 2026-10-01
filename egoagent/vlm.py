"""Zero-shot labelling of real video frames with a vision-language model (CLIP).

The synthetic frames in this repo are 32x32 grey blobs, so CLIP has nothing meaningful to
see there; this module is for running the agent on real first-person footage. The model
backend is injectable, which keeps the scoring logic testable without downloading weights.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

DEFAULT_PROMPTS = {
    "idle": "a first-person photo of someone sitting still",
    "walking": "a first-person photo taken while walking down a hallway",
    "stirring": "a first-person photo of a hand stirring a pot on a stove",
    "typing": "a first-person photo of hands typing on a keyboard",
    "reaching": "a first-person photo of a hand reaching for an object on a shelf",
}


class ImageTextBackend(Protocol):
    def encode_images(self, frames: np.ndarray) -> np.ndarray: ...  # (T, H, W, 3) uint8 -> (T, D)
    def encode_texts(self, texts: list[str]) -> np.ndarray: ...  # -> (K, D)


class HFClipBackend:
    """CLIP from Hugging Face ``transformers`` (``pip install transformers``)."""

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32") -> None:
        import torch
        from transformers import CLIPModel, CLIPProcessor

        self._torch = torch
        self.model = CLIPModel.from_pretrained(model_name).eval()
        self.processor = CLIPProcessor.from_pretrained(model_name)

    def encode_images(self, frames: np.ndarray) -> np.ndarray:
        with self._torch.no_grad():
            inputs = self.processor(images=list(frames), return_tensors="pt")
            return self.model.get_image_features(**inputs).numpy()

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        with self._torch.no_grad():
            inputs = self.processor(text=texts, return_tensors="pt", padding=True)
            return self.model.get_text_features(**inputs).numpy()


class ZeroShotLabeller:
    """Scores a clip against text prompts: cosine similarity per frame, averaged over the
    clip, then a temperature-scaled softmax over activities."""

    def __init__(
        self,
        backend: ImageTextBackend | None = None,
        prompts: dict[str, str] | None = None,
        temperature: float = 100.0,
    ) -> None:
        self.backend = backend or HFClipBackend()
        self.prompts = prompts or DEFAULT_PROMPTS
        self.temperature = temperature
        text = self.backend.encode_texts(list(self.prompts.values()))
        self._text = text / np.linalg.norm(text, axis=1, keepdims=True)

    def score_clip(self, frames: np.ndarray) -> dict[str, float]:
        img = self.backend.encode_images(frames)
        img = img / np.linalg.norm(img, axis=1, keepdims=True)
        logits = self.temperature * (img @ self._text.T).mean(axis=0)
        probs = np.exp(logits - logits.max())
        probs /= probs.sum()
        return {name: float(p) for name, p in zip(self.prompts, probs, strict=True)}
