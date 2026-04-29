#!/usr/bin/env python3
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TOOL = ROOT / "tools/model_onboarding_agent/bench_tool.py"


class BenchToolCLITest(unittest.TestCase):
    def run_tool(self, extra_args):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "result.json"
            cmd = [
                "python",
                str(TOOL),
                "--model-id",
                "demo",
                "--hf-revision",
                "abc123",
                "--threads",
                "4",
                "--batch-size",
                "1",
                "--warmup",
                "3",
                "--repeat",
                "10",
                "--prompt-lengths",
                "128",
                "256",
                "512",
                "1024",
                "--output",
                str(out),
            ] + extra_args
            res = subprocess.run(cmd, capture_output=True, text=True)
            payload = json.loads(out.read_text()) if out.exists() else None
            return res, payload

    def test_mock_mode_creates_valid_schema(self):
        res, payload = self.run_tool(["--mock"])
        self.assertEqual(res.returncode, 0, msg=res.stderr)
        self.assertIsNotNone(payload)

        self.assertEqual(payload["runtime"]["runner_output_unit"], "tps")
        lengths = [row["prompt_length"] for row in payload["results"]]
        self.assertEqual(lengths, [128, 256, 512, 1024])

        for row in payload["results"]:
            self.assertGreater(row["prefill_tps"]["p50"], 0)
            self.assertGreater(row["decode_tps"]["p50"], 0)
            self.assertGreater(row["e2e"]["tps"]["p50"], 0)
            self.assertGreater(row["e2e"]["latency_ms"]["p50"], 0)

    def test_invalid_policy_is_rejected(self):
        res, _ = self.run_tool(["--mock", "--threads", "8"])
        self.assertNotEqual(res.returncode, 0)
        self.assertIn("threads must be 4", res.stderr)

    def test_requires_runner_or_mock(self):
        res, _ = self.run_tool([])
        self.assertNotEqual(res.returncode, 0)
        self.assertIn("Provide either --mock or --runner-cmd", res.stderr)


if __name__ == "__main__":
    unittest.main()
