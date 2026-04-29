#!/usr/bin/env python3
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TOOL = ROOT / "tools/model_onboarding_agent/onboarding_cli.py"


class OnboardingCliTest(unittest.TestCase):
    def test_init_reports(self):
        with tempfile.TemporaryDirectory() as td:
            out_root = Path(td) / "reports"
            cmd = [
                "python",
                str(TOOL),
                "--model-id",
                "qwen-demo",
                "--hf-url",
                "https://huggingface.co/org/model",
                "--hf-revision",
                "deadbeef",
                "--root",
                str(out_root),
            ]
            res = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(res.returncode, 0, msg=res.stderr)

            model_dir = out_root / "qwen-demo"
            self.assertTrue((model_dir / "onboarding_summary.md").exists())
            self.assertTrue((model_dir / "benchmark_results.json").exists())
            self.assertTrue((model_dir / "optimization_log.md").exists())
            self.assertTrue((model_dir / "todo_smoke_test.md").exists())


if __name__ == "__main__":
    unittest.main()
