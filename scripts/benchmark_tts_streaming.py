from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import tempfile
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from threading import Lock

import soundfile as sf

from omni_iot.config import get_settings
from omni_iot.mcp_client import McpCallResult
from omni_iot.omni_agent import OmniAgent
from omni_iot.omni_llama import LlamaServer
from omni_iot.tts_omnivoice import clear_model_cache, synthesize, warmup_model


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUTS = {
    "no_tool": PROJECT_ROOT / ".runtime/211444b31dd64558a4a1fb9cfdfd7118/input.wav",
    "tool_command_1": PROJECT_ROOT / ".runtime/be61361333ab4c7fa8e4b337c54f0de3/input.wav",
}
DEFAULT_BASELINE = (
    PROJECT_ROOT
    / ".runtime/benchmarks/natural-latency-20260901/experiment-2-flash-on/samples.jsonl"
)
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "switchbot__list_devices",
            "description": "List the user's SwitchBot devices.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "switchbot__send_command",
            "description": "Send a command to a SwitchBot device.",
            "parameters": {
                "type": "object",
                "properties": {
                    "device_id": {"type": "string"},
                    "command": {"type": "string"},
                    "parameter": {},
                },
                "required": ["device_id", "command"],
            },
        },
    },
]


@dataclass
class ReplayMcp:
    expose_tools: bool = True

    async def prepare_turn(self) -> list[dict[str, object]]:
        return TOOLS if self.expose_tools else []

    async def call_tool(
        self, name: str, arguments: dict[str, object]
    ) -> McpCallResult:
        started_at = time.perf_counter()
        if name.endswith("list_devices"):
            payload = {
                "isError": False,
                "devices": [
                    {"deviceName": "에어컨", "deviceId": "benchmark-aircon"},
                    {"deviceName": "선풍기", "deviceId": "benchmark-fan"},
                    {"deviceName": "조명", "deviceId": "benchmark-light"},
                ],
            }
            tool_name = "list_devices"
        else:
            payload = {"isError": False, "ok": True, "arguments": arguments}
            tool_name = "send_command"
        return McpCallResult(
            content=json.dumps(payload, ensure_ascii=False),
            is_error=False,
            server_name="benchmark-replay",
            tool_name=tool_name,
            seconds=time.perf_counter() - started_at,
        )


class TextChunkBuffer:
    """Turn arbitrary LLM deltas into sentence chunks without waiting for EOS."""

    def __init__(self, max_chars: int) -> None:
        self.max_chars = max_chars
        self.buffer = ""

    def feed(self, text: str) -> list[str]:
        self.buffer += text
        chunks: list[str] = []
        while True:
            boundary = next(
                (
                    index + 1
                    for index, character in enumerate(self.buffer)
                    if character in ".!?。！？~"
                ),
                None,
            )
            if boundary is not None:
                chunk = self.buffer[:boundary].strip()
                self.buffer = self.buffer[boundary:].lstrip()
                if chunk:
                    chunks.append(chunk)
                continue
            if len(self.buffer) < self.max_chars:
                break
            boundary = self.buffer.rfind(" ", 0, self.max_chars + 1)
            if boundary <= 0:
                boundary = self.max_chars
            chunk = self.buffer[:boundary].strip()
            self.buffer = self.buffer[boundary:].lstrip()
            if chunk:
                chunks.append(chunk)
        return chunks

    def flush(self) -> list[str]:
        chunk = self.buffer.strip()
        self.buffer = ""
        return [chunk] if chunk else []


def _mean_sd(values: list[float]) -> dict[str, float]:
    return {
        "mean": round(statistics.mean(values), 3),
        "standard_deviation": round(
            statistics.stdev(values) if len(values) > 1 else 0.0,
            3,
        ),
    }


def _baseline(path: Path) -> dict[str, list[float]]:
    grouped: dict[str, list[float]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        grouped.setdefault(str(row["input"]), []).append(
            float(row["speech_end_to_audio_ready_seconds"])
        )
    return grouped


async def run_trial(
    *,
    label: str,
    repeat: int,
    audio_path: Path,
    agent: OmniAgent,
    server: LlamaServer,
    settings: object,
    temp_dir: Path,
    max_chars: int,
    num_steps: int,
    vad_seconds: float,
) -> dict[str, object]:
    await asyncio.to_thread(server.erase_slot)
    started_at = time.perf_counter()
    chunk_buffer = TextChunkBuffer(max_chars)
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tts-stream")
    futures: list[Future[dict[str, object]]] = []
    submitted_chunks: list[str] = []
    first_text_delta_at: float | None = None
    first_chunk_queued_at: float | None = None
    timing_lock = Lock()

    def synthesize_chunk(chunk: str, index: int) -> dict[str, object]:
        synthesis_started_at = time.perf_counter()
        output_path = temp_dir / f"{label}-{repeat:02d}-{index:02d}.wav"
        synthesize(
            chunk,
            output_path,
            language=settings.omnivoice_language,
            instruct=settings.omnivoice_instruct,
            speed=settings.omnivoice_speed,
            model_id=settings.omnivoice_model_id,
            num_steps=num_steps,
        )
        completed_at = time.perf_counter()
        return {
            "text": chunk,
            "queued_seconds": round(synthesis_started_at - started_at, 3),
            "synthesis_seconds": round(completed_at - synthesis_started_at, 3),
            "ready_seconds": round(completed_at - started_at, 3),
            "audio_seconds": round(float(sf.info(output_path).duration), 3),
        }

    def submit(chunks: list[str]) -> None:
        nonlocal first_chunk_queued_at
        for chunk in chunks:
            if first_chunk_queued_at is None:
                first_chunk_queued_at = time.perf_counter()
            index = len(submitted_chunks)
            submitted_chunks.append(chunk)
            futures.append(executor.submit(synthesize_chunk, chunk, index))

    def on_response_delta(delta: str) -> None:
        nonlocal first_text_delta_at
        with timing_lock:
            if first_text_delta_at is None:
                first_text_delta_at = time.perf_counter()
        submit(chunk_buffer.feed(delta))

    try:
        generation = await agent.generate_stream(
            audio_path,
            [],
            on_response_delta,
            max_tokens=64,
        )
        llm_complete_at = time.perf_counter()
        submit(chunk_buffer.flush())
        chunk_results = [await asyncio.wrap_future(future) for future in futures]
    finally:
        executor.shutdown(wait=True, cancel_futures=False)

    if not chunk_results:
        raise RuntimeError("Streaming turn produced no playable TTS chunk.")
    first_ready = float(chunk_results[0]["ready_seconds"])
    playback_cursor = first_ready
    predicted_gap = 0.0
    for chunk in chunk_results:
        ready = float(chunk["ready_seconds"])
        if ready > playback_cursor:
            predicted_gap += ready - playback_cursor
            playback_cursor = ready
        playback_cursor += float(chunk["audio_seconds"])

    payload = json.loads(generation.output)
    return {
        "timestamp": datetime.now().astimezone().isoformat(),
        "input": label,
        "repeat": repeat,
        "seed": settings.llama_seed,
        "temperature": settings.llama_temperature,
        "gpu_layers": int(settings.llama_gpu_layers),
        "flash_attention": settings.llama_flash_attn,
        "history_messages": 0,
        "slot_reset_before_trial": True,
        "vad_silence_seconds": vad_seconds,
        "response_text": payload["text"],
        "chunks": submitted_chunks,
        "chunk_count": len(chunk_results),
        "first_text_delta_seconds": round(
            (first_text_delta_at or llm_complete_at) - started_at,
            3,
        ),
        "first_chunk_queued_seconds": round(
            (first_chunk_queued_at or llm_complete_at) - started_at,
            3,
        ),
        "first_audio_ready_after_pipeline_start_seconds": round(first_ready, 3),
        "speech_end_to_first_audio_ready_seconds": round(first_ready + vad_seconds, 3),
        "llm_complete_seconds": round(llm_complete_at - started_at, 3),
        "all_chunks_ready_seconds": round(
            max(float(chunk["ready_seconds"]) for chunk in chunk_results),
            3,
        ),
        "predicted_playback_gap_seconds": round(predicted_gap, 3),
        "chunk_timings": chunk_results,
        "tool_calls": len(generation.tool_trace),
        **(generation.timings or {}),
    }


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark LLM SSE -> sentence TTS -> first playable audio."
    )
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--max-chars", type=int, default=40)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--inputs", default="no_tool,tool_command_1")
    parser.add_argument("--disable-tools", action="store_true")
    args = parser.parse_args()
    if args.trials < 1 or args.max_chars < 1:
        raise SystemExit("--trials and --max-chars must be positive")
    selected_inputs = {
        label: INPUTS[label]
        for label in (item.strip() for item in args.inputs.split(","))
        if label
    }

    base = get_settings()
    output_dir = args.output_dir or (
        base.runtime_dir
        / "benchmarks"
        / f"end-to-end-streaming-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    settings = replace(
        base,
        runtime_dir=output_dir / "turns",
        omni_backend="server",
        tts_backend="omnivoice",
        llama_gpu_layers="24",
        llama_flash_attn="on",
        llama_temperature=0.2,
        llama_seed=-1,
        llama_slot_save_path=output_dir / "slot-cache",
        llama_warmup=False,
        omnivoice_warmup=False,
    )
    settings.runtime_dir.mkdir(parents=True, exist_ok=True)
    vad_seconds = settings.vad_silence_end_ms / 1_000
    manifest = {
        "created_at": datetime.now().astimezone().isoformat(),
        "pipeline": "Qwen3-Omni llama.cpp SSE -> sentence buffer -> OmniVoice WAV chunks",
        "true_streaming_tts": False,
        "tts_limitation": "OmniVoice generate() has no incremental text/waveform API.",
        "trials_per_input": args.trials,
        "max_chunk_chars": args.max_chars,
        "baseline": str(args.baseline),
        "gpu_layers": 24,
        "flash_attention": "on",
        "temperature": 0.2,
        "seed": -1,
        "history_messages": 0,
        "slot_reset_before_trial": True,
        "vad_silence_seconds": vad_seconds,
        "tools_exposed": not args.disable_tools,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    server = LlamaServer(settings)
    measured: list[dict[str, object]] = []
    try:
        await asyncio.to_thread(server.start)
        warmup_agent = OmniAgent(
            server,
            ReplayMcp(not args.disable_tools),
            settings,
        )  # type: ignore[arg-type]
        await warmup_agent.generate(INPUTS["no_tool"], [], max_tokens=64)
        await asyncio.to_thread(
            warmup_model,
            settings.omnivoice_model_id,
            settings.omnivoice_language,
            settings.omnivoice_instruct,
            settings.omnivoice_speed,
            settings.omnivoice_num_steps,
        )
        with tempfile.TemporaryDirectory(prefix="omni-e2e-stream-") as temp_name:
            temp_dir = Path(temp_name)
            for label, audio_path in selected_inputs.items():
                for repeat in range(1, args.trials + 1):
                    row = await run_trial(
                        label=label,
                        repeat=repeat,
                        audio_path=audio_path,
                        agent=OmniAgent(
                            server,
                            ReplayMcp(not args.disable_tools),
                            settings,
                        ),  # type: ignore[arg-type]
                        server=server,
                        settings=settings,
                        temp_dir=temp_dir,
                        max_chars=args.max_chars,
                        num_steps=settings.omnivoice_num_steps,
                        vad_seconds=vad_seconds,
                    )
                    measured.append(row)
                    with (output_dir / "samples.jsonl").open("a", encoding="utf-8") as file:
                        file.write(json.dumps(row, ensure_ascii=False) + "\n")
                    print(
                        f"{label} {repeat:02d}: first_audio="
                        f"{row['speech_end_to_first_audio_ready_seconds']:.3f}s "
                        f"llm_done={row['llm_complete_seconds']:.3f}s "
                        f"chunks={row['chunk_count']}",
                        flush=True,
                    )
    finally:
        clear_model_cache()
        await asyncio.to_thread(server.stop)

    baselines = _baseline(args.baseline)
    summary = []
    for label in selected_inputs:
        group = [row for row in measured if row["input"] == label]
        streamed = [
            float(row["speech_end_to_first_audio_ready_seconds"]) for row in group
        ]
        baseline = baselines[label]
        baseline_mean = statistics.mean(baseline)
        streamed_mean = statistics.mean(streamed)
        delta = baseline_mean - streamed_mean
        summary.append(
            {
                "input": label,
                "streaming_samples": len(streamed),
                "baseline_samples": len(baseline),
                "baseline_speech_end_to_audio_ready_seconds": _mean_sd(baseline),
                "streaming_speech_end_to_first_audio_ready_seconds": _mean_sd(streamed),
                "saved_seconds": round(delta, 3),
                "reduction_percent": round(delta / baseline_mean * 100, 1),
                "first_text_delta_seconds": _mean_sd(
                    [float(row["first_text_delta_seconds"]) for row in group]
                ),
                "first_chunk_queued_seconds": _mean_sd(
                    [float(row["first_chunk_queued_seconds"]) for row in group]
                ),
                "llm_complete_seconds": _mean_sd(
                    [float(row["llm_complete_seconds"]) for row in group]
                ),
                "predicted_playback_gap_seconds": _mean_sd(
                    [float(row["predicted_playback_gap_seconds"]) for row in group]
                ),
                "mean_chunk_count": round(
                    statistics.mean(float(row["chunk_count"]) for row in group), 2
                ),
            }
        )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"results: {output_dir}")


if __name__ == "__main__":
    asyncio.run(main())
