from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import subprocess
import time
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path

from omni_iot.config import get_settings
from omni_iot.mcp_client import McpCallResult
from omni_iot.omni_agent import OmniAgent
from omni_iot.omni_llama import LlamaServer
from omni_iot.pipeline import run_turn_pipeline
from omni_iot.tts_omnivoice import clear_model_cache, warmup_model


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUTS = {
    "no_tool": PROJECT_ROOT / ".runtime/211444b31dd64558a4a1fb9cfdfd7118/input.wav",
    "tool_command_1": PROJECT_ROOT / ".runtime/be61361333ab4c7fa8e4b337c54f0de3/input.wav",
}

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
    """Expose tool schemas but never touch a real device during a benchmark."""

    async def prepare_turn(self) -> list[dict[str, object]]:
        return TOOLS

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


def gpu_snapshot() -> dict[str, float | str]:
    command = [
        "nvidia-smi",
        "--query-gpu=name,memory.used,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=True)
    name, used, total, utilization = [
        item.strip() for item in completed.stdout.strip().split(",")
    ]
    return {
        "name": name,
        "memory_used_mib": float(used),
        "memory_total_mib": float(total),
        "utilization_percent": float(utilization),
    }


async def benchmark_layer(
    layer: int,
    repeats: int,
    output_dir: Path,
    inputs: dict[str, Path],
    seed: int,
    flash_attn: str,
    temperature: float,
) -> list[dict[str, object]]:
    base = get_settings()
    layer_dir = output_dir / f"gpu-layers-{layer}"
    layer_dir.mkdir(parents=True, exist_ok=True)
    settings = replace(
        base,
        runtime_dir=layer_dir / "turns",
        llama_gpu_layers=str(layer),
        llama_temperature=temperature,
        llama_seed=seed,
        llama_flash_attn=flash_attn,
        llama_slot_save_path=layer_dir / "slot-cache",
        llama_warmup=False,
        omnivoice_warmup=False,
    )
    settings.runtime_dir.mkdir(parents=True, exist_ok=True)
    server = LlamaServer(settings)
    server_log = settings.runtime_dir / "llama-server.log"
    rows: list[dict[str, object]] = []
    try:
        startup_started_at = time.perf_counter()
        await asyncio.to_thread(server.start)
        startup_seconds = time.perf_counter() - startup_started_at
        gpu_ready = gpu_snapshot()

        # Exclude one LLM and one TTS warmup from reported samples.
        warmup_agent = OmniAgent(server, ReplayMcp(), settings)  # type: ignore[arg-type]
        await warmup_agent.generate(inputs["no_tool"], [], max_tokens=64)
        await asyncio.to_thread(
            warmup_model,
            settings.omnivoice_model_id,
            settings.omnivoice_language,
            settings.omnivoice_instruct,
            settings.omnivoice_speed,
            settings.omnivoice_num_steps,
        )
        gpu_warmed = gpu_snapshot()

        for label, audio_path in inputs.items():
            for repeat in range(1, repeats + 1):
                await asyncio.to_thread(server.erase_slot)
                wall_started_at = time.perf_counter()
                result = await run_turn_pipeline(
                    audio_path.read_bytes(),
                    settings,
                    omni_client=OmniAgent(  # type: ignore[arg-type]
                        server, ReplayMcp(), settings
                    ),
                    max_tokens=64,
                    tts_num_steps=settings.omnivoice_num_steps,
                )
                measured_wall = time.perf_counter() - wall_started_at
                row: dict[str, object] = {
                    "timestamp": datetime.now().astimezone().isoformat(),
                    "gpu_layers": layer,
                    "input": label,
                    "source_wav": str(audio_path.relative_to(PROJECT_ROOT)),
                    "repeat": repeat,
                    "seed": seed,
                    "temperature": temperature,
                    "flash_attention": flash_attn,
                    "history_messages": 0,
                    "slot_reset_before_trial": True,
                    "vad_silence_seconds": settings.vad_silence_end_ms / 1000,
                    "speech_end_to_audio_ready_seconds": round(
                        measured_wall + settings.vad_silence_end_ms / 1000, 3
                    ),
                    "pipeline_wall_seconds": round(measured_wall, 3),
                    "startup_seconds": round(startup_seconds, 3),
                    "gpu_at_server_ready": gpu_ready,
                    "gpu_after_warmup": gpu_warmed,
                    "turn_id": result.turn_id,
                    "response_text": result.text,
                    **result.timings,
                }
                rows.append(row)
                with (output_dir / "samples.jsonl").open("a", encoding="utf-8") as f:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
    except Exception as exc:
        failure = {
            "timestamp": datetime.now().astimezone().isoformat(),
            "gpu_layers": layer,
            "error": str(exc),
            "server_log": str(server_log),
        }
        with (output_dir / "failures.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(failure, ensure_ascii=False) + "\n")
    finally:
        clear_model_cache()
        await asyncio.to_thread(server.stop)
    return rows


def write_summary(output_dir: Path, rows: list[dict[str, object]]) -> None:
    grouped: dict[tuple[int, str, str], list[dict[str, object]]] = {}
    for row in rows:
        key = (
            int(row["gpu_layers"]),
            str(row["input"]),
            str(row["flash_attention"]),
        )
        grouped.setdefault(key, []).append(row)
    summary = []
    for (layers, label, flash_attn), group_rows in sorted(grouped.items()):
        values = [
            float(row["speech_end_to_audio_ready_seconds"])
            for row in group_rows
        ]
        completion_tokens = [
            float(row.get("completion_tokens", 0.0)) for row in group_rows
        ]
        summary.append(
            {
                "gpu_layers": layers,
                "input": label,
                "flash_attention": flash_attn,
                "samples": len(values),
                "mean_seconds": round(statistics.mean(values), 3),
                "median_seconds": round(statistics.median(values), 3),
                "min_seconds": round(min(values), 3),
                "max_seconds": round(max(values), 3),
                "standard_deviation_seconds": round(
                    statistics.stdev(values) if len(values) > 1 else 0.0, 3
                ),
                "mean_completion_tokens": round(
                    statistics.mean(completion_tokens), 3
                ),
                "standard_deviation_completion_tokens": round(
                    statistics.stdev(completion_tokens)
                    if len(completion_tokens) > 1
                    else 0.0,
                    3,
                ),
            }
        )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", default="0,8,16,20,24")
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--flash-attn", choices=("on", "off"), default="off")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    layers = [int(item) for item in args.layers.split(",")]
    output_dir = args.output or (
        PROJECT_ROOT
        / ".runtime/benchmarks"
        / datetime.now().strftime("gpu-offload-%Y%m%d-%H%M%S")
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    missing = [str(path) for path in DEFAULT_INPUTS.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing benchmark WAV: " + ", ".join(missing))
    manifest = {
        "created_at": datetime.now().astimezone().isoformat(),
        "definition": "last voiced sample + configured VAD silence to reply WAV ready",
        "browser_playback_scheduler_included": False,
        "layers": layers,
        "repeats": args.repeats,
        "seed": args.seed,
        "temperature": args.temperature,
        "flash_attention": args.flash_attn,
        "history_messages": 0,
        "slot_reset_before_every_trial": True,
        "inputs": {key: str(value) for key, value in DEFAULT_INPUTS.items()},
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    rows: list[dict[str, object]] = []
    for layer in layers:
        rows.extend(
            await benchmark_layer(
                layer,
                args.repeats,
                output_dir,
                DEFAULT_INPUTS,
                args.seed,
                args.flash_attn,
                args.temperature,
            )
        )
        write_summary(output_dir, rows)
    print(output_dir)


if __name__ == "__main__":
    asyncio.run(main())
