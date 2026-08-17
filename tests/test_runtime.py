from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from omni_iot.runtime import prune_runtime_turns


class PruneRuntimeTurnsTest(unittest.TestCase):
    def test_keeps_latest_turns_without_touching_other_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            runtime_dir = Path(temp_dir)
            turn_dirs: list[Path] = []
            for index in range(22):
                turn_dir = runtime_dir / f"{index:032x}"
                turn_dir.mkdir()
                (turn_dir / "input.wav").write_bytes(b"wav")
                os.utime(turn_dir, (index, index))
                turn_dirs.append(turn_dir)

            custom_dir = runtime_dir / "manual-check"
            custom_dir.mkdir()
            smoke_wav = runtime_dir / "smoke.wav"
            smoke_wav.write_bytes(b"wav")

            deleted = prune_runtime_turns(runtime_dir, keep=20)

            self.assertEqual(deleted, 2)
            self.assertFalse(turn_dirs[0].exists())
            self.assertFalse(turn_dirs[1].exists())
            self.assertTrue(all(path.exists() for path in turn_dirs[2:]))
            self.assertTrue(custom_dir.exists())
            self.assertTrue(smoke_wav.exists())

    def test_protected_turn_is_not_deleted(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            runtime_dir = Path(temp_dir)
            old_turn = runtime_dir / ("0" * 32)
            new_turn = runtime_dir / ("1" * 32)
            old_turn.mkdir()
            new_turn.mkdir()
            os.utime(old_turn, (1, 1))
            os.utime(new_turn, (2, 2))

            prune_runtime_turns(runtime_dir, keep=1, protected={old_turn})

            self.assertTrue(old_turn.exists())
            self.assertFalse(new_turn.exists())


if __name__ == "__main__":
    unittest.main()
