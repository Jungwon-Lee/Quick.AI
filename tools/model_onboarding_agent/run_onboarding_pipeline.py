#!/usr/bin/env python3
"""Run end-to-end onboarding scaffolding + benchmark (mock/external)."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run onboarding pipeline helper")
    p.add_argument("--model-id", required=True)
    p.add_argument("--hf-url", required=True)
    p.add_argument("--hf-revision", required=True)
    p.add_argument("--reports-root", type=Path, default=Path("reports"))
    p.add_argument("--mock", action="store_true")
    p.add_argument("--runner-cmd")
    p.add_argument("--runner-output-unit", choices=["tps", "latency_ms"], default="tps")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--repeat", type=int, default=10)
    p.add_argument("--prompt-lengths", nargs="+", type=int, default=[128, 256, 512, 1024])
    return p.parse_args()


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()
    report_dir = args.reports_root / args.model_id

    run(
        [
            sys.executable,
            "tools/model_onboarding_agent/onboarding_cli.py",
            "--model-id",
            args.model_id,
            "--hf-url",
            args.hf_url,
            "--hf-revision",
            args.hf_revision,
            "--root",
            str(args.reports_root),
        ]
    )

    bench_cmd = [
            sys.executable,
            "tools/model_onboarding_agent/bench_tool.py",
        "--model-id",
        args.model_id,
        "--hf-revision",
        args.hf_revision,
        "--threads",
        str(args.threads),
        "--batch-size",
        str(args.batch_size),
        "--warmup",
        str(args.warmup),
        "--repeat",
        str(args.repeat),
        "--prompt-lengths",
        *[str(x) for x in args.prompt_lengths],
        "--output",
        str(report_dir / "benchmark_results.json"),
        "--runner-output-unit",
        args.runner_output_unit,
    ]

    if args.mock:
        bench_cmd.append("--mock")
    elif args.runner_cmd:
        bench_cmd.extend(["--runner-cmd", args.runner_cmd])
    else:
        raise SystemExit("Either --mock or --runner-cmd must be provided")

    run(bench_cmd)
    print(f"Onboarding pipeline completed for {args.model_id}: {report_dir}")


if __name__ == "__main__":
    main()
