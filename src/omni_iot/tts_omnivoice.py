from __future__ import annotations

import argparse
import re
from functools import lru_cache
from pathlib import Path
from threading import Lock

import soundfile as sf
import torch
from omnivoice import OmniVoice


SAMPLE_RATE = 24_000
_generation_lock = Lock()


def split_text_chunks(text: str, max_chars: int = 80) -> list[str]:
    """Split text into independently playable, sentence-like TTS chunks."""
    normalized = " ".join(text.split())
    if not normalized:
        return []
    if max_chars < 1:
        raise ValueError("max_chars must be positive.")

    sentences = [
        part.strip()
        for part in re.findall(r".+?(?:[.!?。！？~]+(?=\s|$)|$)", normalized)
        if part.strip()
    ]
    chunks: list[str] = []
    for sentence in sentences:
        remainder = sentence
        while len(remainder) > max_chars:
            split_at = max(
                remainder.rfind(delimiter, 0, max_chars + 1)
                for delimiter in (" ", ",", "、")
            )
            if split_at <= 0:
                split_at = max_chars
            chunk = remainder[:split_at].strip()
            if chunk:
                chunks.append(chunk)
            remainder = remainder[split_at:].strip(" ,、")
        if remainder:
            chunks.append(remainder)
    return chunks


def _generation_kwargs(
    text: str,
    ref_audio: Path | None = None,
    ref_text: str | None = None,
    language: str | None = "ko",
    instruct: str | None = None,
    speed: float | None = None,
    num_steps: int | None = None,
) -> dict[str, str | float | int]:
    kwargs: dict[str, str | float | int] = {"text": text}
    if language:
        kwargs["language"] = language
    if ref_audio:
        kwargs["ref_audio"] = str(ref_audio)
    if ref_text:
        kwargs["ref_text"] = ref_text
    if instruct:
        kwargs["instruct"] = instruct
    if speed is not None:
        kwargs["speed"] = speed
    if num_steps is not None:
        kwargs["num_step"] = num_steps
    return kwargs


def synthesize(
    text: str,
    output_path: Path,
    ref_audio: Path | None = None,
    ref_text: str | None = None,
    language: str | None = "ko",
    instruct: str | None = None,
    speed: float | None = None,
    model_id: str = "k2-fsa/OmniVoice",
    num_steps: int | None = None,
) -> None:
    kwargs = _generation_kwargs(
        text=text,
        ref_audio=ref_audio,
        ref_text=ref_text,
        language=language,
        instruct=instruct,
        speed=speed,
        num_steps=num_steps,
    )

    # OmniVoice is shared by every turn in this process. Serialize generation so
    # concurrent HTTP requests cannot mutate the model state at the same time.
    with _generation_lock:
        model = load_model(model_id)
        audio = model.generate(**kwargs)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path, audio[0], SAMPLE_RATE)


@lru_cache(maxsize=1)
def load_model(model_id: str) -> OmniVoice:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    return OmniVoice.from_pretrained(
        model_id,
        device_map=device,
        dtype=dtype,
    )


def warmup_model(
    model_id: str,
    language: str | None = "ko",
    instruct: str | None = None,
    speed: float | None = None,
    num_steps: int | None = None,
) -> None:
    with _generation_lock:
        model = load_model(model_id)
        model.generate(
            **_generation_kwargs(
                text="준비됐어요.",
                language=language,
                instruct=instruct,
                speed=speed,
                num_steps=num_steps,
            )
        )


def clear_model_cache() -> None:
    load_model.cache_clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate TTS with k2-fsa/OmniVoice.")
    parser.add_argument("--text", help="Text to synthesize.")
    parser.add_argument("--text-file", type=Path, help="UTF-8 file containing text to synthesize.")
    parser.add_argument("--output", type=Path, required=True, help="Output WAV path.")
    parser.add_argument("--ref-audio", type=Path, help="Reference voice WAV for voice cloning.")
    parser.add_argument("--ref-text", help="Transcript for --ref-audio.")
    parser.add_argument("--language", default="ko", help="Language code or name. Default: ko.")
    parser.add_argument("--instruct", help="Voice design instruction.")
    parser.add_argument("--speed", type=float, help="Speaking speed factor.")
    parser.add_argument("--model-id", default="k2-fsa/OmniVoice")
    args = parser.parse_args()

    if args.text_file:
        text = args.text_file.read_text(encoding="utf-8").strip()
    elif args.text:
        text = args.text.strip()
    else:
        raise SystemExit("Either --text or --text-file is required.")

    synthesize(
        text=text,
        output_path=args.output,
        ref_audio=args.ref_audio,
        ref_text=args.ref_text,
        language=args.language,
        instruct=args.instruct,
        speed=args.speed,
        model_id=args.model_id,
    )


if __name__ == "__main__":
    main()
