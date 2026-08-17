from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Collection


TURN_DIRECTORY_PATTERN = re.compile(r"[0-9a-f]{32}")


def prune_runtime_turns(
    runtime_dir: Path,
    keep: int = 20,
    protected: Collection[Path] = (),
) -> int:
    """Delete old generated turn directories and return the deletion count."""
    if keep < 0:
        raise ValueError("keep must not be negative.")
    if not runtime_dir.exists():
        return 0

    protected_paths = {path.resolve() for path in protected}
    turn_dirs = [
        path
        for path in runtime_dir.iterdir()
        if TURN_DIRECTORY_PATTERN.fullmatch(path.name)
        and path.is_dir()
        and not path.is_symlink()
    ]
    turn_dirs.sort(key=lambda path: path.stat().st_mtime_ns, reverse=True)

    protected_turns = [path for path in turn_dirs if path.resolve() in protected_paths]
    unprotected_turns = [path for path in turn_dirs if path.resolve() not in protected_paths]
    remaining_slots = max(0, keep - len(protected_turns))
    expired_turns = unprotected_turns[remaining_slots:]

    for path in expired_turns:
        shutil.rmtree(path)

    return len(expired_turns)
