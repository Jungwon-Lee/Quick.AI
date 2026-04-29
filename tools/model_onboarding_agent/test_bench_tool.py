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
            payload = None
            if out.exists():
                payload = json.loads(out.read_text())
            return res, payload

    def test_mock_mode_creates_valid_schema(self):
        res, payload = self.run_tool(["--mock"])
        self.assertEqual(res.returncode, 0, msg=res.stderr)
        self.assertIsNotNone(payload)

        self.assertEqual(payload["model_id"], "demo")
        self.assertEqual(payload["hf_revision"], "abc123")
        self.assertEqual(payload["runtime"]["threads"], 4)
        self.assertEqual(payload["runtime"]["batch_size"], 1)
        self.assertEqual(payload["runtime"]["warmup"], 3)
        self.assertEqual(payload["runtime"]["repeat"], 10)

        lengths = [row["prompt_length"] for row in payload["results"]]
        self.assertEqual(lengths, [128, 256, 512, 1024])

        for row in payload["results"]:
            self.assertIn("prefill_tps", row)
            self.assertIn("decode_tps", row)
            self.assertIn("e2e", row)
            self.assertGreater(row["prefill_tps"]["p50"], 0)
            self.assertGreater(row["decode_tps"]["p50"], 0)
            self.assertGreater(row["e2e"]["latency_ms_p50"], 0)

    def test_invalid_policy_is_rejected(self):
        res, _payload = self.run_tool(["--mock", "--threads", "8"])
        self.assertNotEqual(res.returncode, 0)
        self.assertIn("threads must be 4", res.stderr)


if __name__ == "__main__":
    unittest.main()
