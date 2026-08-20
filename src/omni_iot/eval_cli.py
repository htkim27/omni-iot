from __future__ import annotations

import argparse

from .config import get_settings
from .observability import LangfuseObservability

SCORE_CONFIGS = (
    {
        "name": "transcript_accuracy",
        "data_type": "CATEGORICAL",
        "description": "Human review of the transcript against the input audio.",
        "categories": (
            (1.0, "exact"),
            (0.66, "minor_error"),
            (0.0, "wrong"),
            (0.33, "unintelligible"),
        ),
    },
    {
        "name": "transcript_correction",
        "data_type": "TEXT",
        "description": "Corrected transcript written by a human reviewer.",
        "categories": (),
    },
    {
        "name": "response_quality",
        "data_type": "CATEGORICAL",
        "description": "Human review of the final assistant response.",
        "categories": (
            (1.0, "good"),
            (0.5, "acceptable"),
            (0.0, "poor"),
        ),
    },
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate or initialize local Langfuse evaluation resources."
    )
    parser.add_argument("command", choices=("doctor", "setup"))
    args = parser.parse_args()

    settings = get_settings()
    if not settings.ai_eval_enabled:
        raise SystemExit("Set AI_EVAL_ENABLED=true before using omni-iot-eval.")

    observability = LangfuseObservability(settings)
    try:
        if not observability.authenticate():
            raise SystemExit("Langfuse authentication failed.")
        if args.command == "doctor":
            print("Langfuse authentication succeeded.")
            return
        _setup_resources(observability)
    finally:
        observability.shutdown()


def _setup_resources(observability: LangfuseObservability) -> None:
    from langfuse.api.commons.types import ConfigCategory, ScoreConfigDataType

    api = observability.api
    existing_configs = {
        config.name: config for config in api.score_configs.get(limit=100).data
    }
    config_ids: list[str] = []
    for specification in SCORE_CONFIGS:
        name = str(specification["name"])
        config = existing_configs.get(name)
        if config is None:
            categories = [
                ConfigCategory(value=value, label=label)
                for value, label in specification["categories"]
            ]
            config = api.score_configs.create(
                name=name,
                data_type=ScoreConfigDataType(str(specification["data_type"])),
                categories=categories or None,
                description=str(specification["description"]),
            )
            print(f"Created score config {name}.")
        else:
            print(f"Score config {name} already exists.")
        config_ids.append(config.id)

    queues = api.annotation_queues.list_queues(limit=100).data
    if any(queue.name == "voice-transcript-review" for queue in queues):
        print("Annotation queue voice-transcript-review already exists.")
        return
    api.annotation_queues.create_queue(
        name="voice-transcript-review",
        score_config_ids=config_ids,
        description="Review input audio, generated transcript, and final response.",
    )
    print("Created annotation queue voice-transcript-review.")


if __name__ == "__main__":
    main()
