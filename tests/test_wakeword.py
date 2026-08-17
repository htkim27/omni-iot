from __future__ import annotations

import io
import unittest
import wave

import numpy as np

from omni_iot.server import _pcm_to_wav
from omni_iot.wakeword import FRAME_BYTES, SAMPLE_RATE, WakeWordDetector


class FakeWakeWordModel:
    def __init__(self, scores: list[dict[str, float]]) -> None:
        self.scores = iter(scores)
        self.frames: list[np.ndarray] = []
        self.reset_count = 0

    def predict(self, audio: np.ndarray) -> dict[str, float]:
        self.frames.append(audio.copy())
        return next(self.scores)

    def reset(self) -> None:
        self.reset_count += 1


class WakeWordDetectorTest(unittest.TestCase):
    def test_reframes_chunks_and_detects_hey_jarvis(self) -> None:
        model = FakeWakeWordModel([{"hey_jarvis_v0.1": 0.2}, {"hey_jarvis_v0.1": 0.8}])
        detector = WakeWordDetector("unused", threshold=0.5, model=model)

        self.assertIsNone(detector.process(bytes(FRAME_BYTES // 2)))
        detection = detector.process(bytes(FRAME_BYTES * 3 // 2))

        self.assertEqual(len(model.frames), 2)
        self.assertEqual(model.frames[0].dtype, np.dtype("int16"))
        self.assertEqual(detection.model, "hey_jarvis_v0.1")
        self.assertEqual(detection.score, 0.8)

    def test_reset_clears_partial_frame_and_model_state(self) -> None:
        model = FakeWakeWordModel([{"hey_jarvis_v0.1": 0.9}])
        detector = WakeWordDetector("unused", model=model)
        detector.process(bytes(FRAME_BYTES // 2))

        detector.reset()
        self.assertIsNone(detector.process(bytes(FRAME_BYTES // 2)))

        self.assertEqual(model.frames, [])
        self.assertEqual(model.reset_count, 1)

    def test_rejects_incomplete_int16_sample(self) -> None:
        detector = WakeWordDetector("unused", model=FakeWakeWordModel([]))

        with self.assertRaisesRegex(ValueError, "complete 16-bit samples"):
            detector.process(b"\x00")


class PcmWaveTest(unittest.TestCase):
    def test_wraps_pcm_as_16khz_mono_wave(self) -> None:
        pcm = np.array([-32768, 0, 32767], dtype="<i2").tobytes()

        with wave.open(io.BytesIO(_pcm_to_wav(pcm)), "rb") as wav_file:
            self.assertEqual(wav_file.getnchannels(), 1)
            self.assertEqual(wav_file.getsampwidth(), 2)
            self.assertEqual(wav_file.getframerate(), SAMPLE_RATE)
            self.assertEqual(wav_file.readframes(3), pcm)


if __name__ == "__main__":
    unittest.main()
