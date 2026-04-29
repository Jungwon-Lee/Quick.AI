#!/usr/bin/env python3
import subprocess
import tempfile
import unittest
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TOOL = ROOT / "tools/model_onboarding_agent/run_onboarding_pipeline.py"


class PipelineCliTest(unittest.TestCase):
    def test_pipeline_mock(self):
        with tempfile.TemporaryDirectory() as td:
            reports = Path(td) / "reports"
            cmd = [
                "python",
                str(TOOL),
                "--model-id",
                "m1",
                "--hf-url",
                "https://huggingface.co/org/model",
                "--hf-revision",
                "rev1",
                "--reports-root",
                str(reports),
                "--mock",
            ]
            res = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
            self.assertEqual(res.returncode, 0, msg=res.stderr)
            self.assertTrue((reports / "m1" / "onboarding_summary.md").exists())
            self.assertTrue((reports / "m1" / "benchmark_results.json").exists())
            payload = json.loads((reports / "m1" / "benchmark_results.json").read_text())
            self.assertEqual(payload["runtime"]["runner_output_unit"], "tps")


if __name__ == "__main__":
    unittest.main()
