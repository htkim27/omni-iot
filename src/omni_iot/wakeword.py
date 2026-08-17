from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

SAMPLE_RATE = 16_000
FRAME_SAMPLES = 1_280
FRAME_BYTES = FRAME_SAMPLES * 2
DEFAULT_MODEL_NAME = "hey_jarvis"
DEFAULT_MODEL_FILE = "hey_jarvis_v0.1.onnx"


@dataclass(frozen=True)
class WakeDetection:
    model: str
    score: float


class WakeWordModel(Protocol):
    def predict(self, audio: np.ndarray) -> dict[str, float]: ...

    def reset(self) -> None: ...


class WakeWordDetector:
    """Frame arbitrary PCM chunks for openWakeWord's 80 ms streaming API."""

    def __init__(
        self,
        model_path: str | Path,
        threshold: float = 0.5,
        inference_framework: str = "onnx",
        model: WakeWordModel | None = None,
    ) -> None:
        self.threshold = threshold
        self._pending = bytearray()

        if model is None:
            try:
                from openwakeword.model import Model
            except ImportError as exc:
                raise RuntimeError(
                    "openwakeword is not installed; run `uv sync` first."
                ) from exc

            path = Path(model_path).expanduser()
            if not path.is_file():
                raise RuntimeError(
                    f"Wake-word model not found: {path}. "
                    "Run `uv run omni-iot-wakeword-models`."
                )
            support_models = {
                "melspec_model_path": path.parent / "melspectrogram.onnx",
                "embedding_model_path": path.parent / "embedding_model.onnx",
            }
            missing = [
                item.name for item in support_models.values() if not item.is_file()
            ]
            if missing:
                raise RuntimeError(
                    f"Wake-word support model(s) missing beside {path.name}: "
                    f"{', '.join(missing)}. Run `uv run omni-iot-wakeword-models`."
                )
            model = Model(
                wakeword_models=[str(path)],
                inference_framework=inference_framework,
                **{key: str(value) for key, value in support_models.items()},
            )

        self._model = model

    def process(self, pcm: bytes) -> WakeDetection | None:
        if len(pcm) % 2:
            raise ValueError("PCM payload must contain complete 16-bit samples.")

        self._pending.extend(pcm)
        best: WakeDetection | None = None
        while len(self._pending) >= FRAME_BYTES:
            frame = bytes(self._pending[:FRAME_BYTES])
            del self._pending[:FRAME_BYTES]
            samples = np.frombuffer(frame, dtype="<i2")
            scores = self._model.predict(samples)
            for name, raw_score in scores.items():
                score = float(raw_score)
                if score >= self.threshold and (best is None or score > best.score):
                    best = WakeDetection(model=name, score=score)
        return best

    def reset(self) -> None:
        self._pending.clear()
        reset = getattr(self._model, "reset", None)
        if reset is not None:
            reset()


def download_model_main() -> None:
    """Download official openWakeWord model files once for offline inference."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--target", type=Path, default=Path("models/openwakeword"))
    args = parser.parse_args()

    from openwakeword.utils import download_models

    target = args.target.expanduser().resolve()
    download_models(model_names=[args.model], target_directory=str(target))
    print(f"openWakeWord model ready: {target}")
