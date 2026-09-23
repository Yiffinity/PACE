"""Exercise the public launcher without loading models or starting training."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class LauncherTests(unittest.TestCase):
    def run_launcher(self, *, group="mmsd2_docmsu", fail_at=""):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            calls = root / "calls.jsonl"
            interpreter = root / "record-python"
            interpreter.write_text(
                f"#!{sys.executable}\n"
                "import json, os, sys\n"
                "with open(os.environ['PACE_TEST_CALLS'], 'a') as handle:\n"
                "    handle.write(json.dumps(sys.argv[1:]) + '\\n')\n"
                "if os.environ.get('PACE_TEST_FAIL_AT') in sys.argv[1:]:\n"
                "    sys.exit(17)\n",
                encoding="utf-8",
            )
            interpreter.chmod(0o755)
            env = {key: value for key, value in os.environ.items() if not key.startswith("PACE_PLUS_")}
            env.update(
                PYTHON_BIN=str(interpreter),
                PACE_PLUS_GROUP=group,
                PACE_TEST_CALLS=str(calls),
                PACE_TEST_FAIL_AT=fail_at,
            )
            result = subprocess.run(
                ["bash", str(ROOT / "scripts" / "run_pace_plus.sh")],
                cwd=root,
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )
            records = [json.loads(line) for line in calls.read_text().splitlines()] if calls.exists() else []
        return result, records

    def test_full_pipeline_uses_the_selected_source_pair(self):
        result, calls = self.run_launcher(group="docmsu_sarcnet")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(calls), 8)
        self.assertEqual(
            [call[3] for call in calls if Path(call[0]).name == "pace_plus_cli.py"],
            ["preflight", "generate-reasonings", "extract", "consolidate", "consolidate", "consolidate", "train"],
        )
        train = calls[-1]
        self.assertEqual(Path(train[train.index("--data") + 1]).name, "docmsu_sarcnet.jsonl")
        self.assertEqual(train[train.index("--group") + 1], "docmsu_sarcnet")

    def test_review_failure_stops_before_consolidation_and_training(self):
        result, calls = self.run_launcher(fail_at="extract")
        self.assertEqual(result.returncode, 17)
        self.assertEqual(calls[-1][3], "extract")
        self.assertEqual(len(calls), 4)

    def test_unknown_group_fails_before_running_any_stage(self):
        result, calls = self.run_launcher(group="unknown")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()
