from __future__ import annotations

import base64
import io
import unittest

import numpy as np
import soundfile as sf

from omni_iot.vllm_omni import _parse_stream_event, decode_audio_chunk


class VllmOmniStreamTest(unittest.TestCase):
    def test_parses_text_and_audio_sse_choices(self) -> None:
        audio = b"RIFF-test"

        text_chunks = list(
            _parse_stream_event(
                {
                    "modality": "text",
                    "choices": [{"delta": {"content": "안녕"}}],
                }
            )
        )
        audio_chunks = list(
            _parse_stream_event(
                {
                    "modality": "audio",
                    "choices": [
                        {
                            "delta": {
                                "content": base64.b64encode(audio).decode("ascii")
                            }
                        }
                    ],
                }
            )
        )

        self.assertEqual(text_chunks[0].modality, "text")
        self.assertEqual(text_chunks[0].content, "안녕")
        self.assertEqual(audio_chunks[0].modality, "audio")
        self.assertEqual(audio_chunks[0].content, audio)

    def test_decodes_wav_chunk_to_mono_pcm16(self) -> None:
        samples = np.array([[0.5, -0.5], [-1.0, 1.0]], dtype=np.float32)
        wav = io.BytesIO()
        sf.write(wav, samples, 24_000, format="WAV", subtype="FLOAT")

        decoded = decode_audio_chunk(wav.getvalue())

        self.assertEqual(decoded.sample_rate, 24_000)
        self.assertEqual(decoded.channels, 1)
        self.assertEqual(len(decoded.data), 4)


if __name__ == "__main__":
    unittest.main()
