#!/usr/bin/env python3
"""CPU-first benchmark runner for Quick.AI model onboarding.

This tool standardizes benchmark execution metadata and output schema.
Actual model inference hooks can be wired through `--runner-cmd`.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List


@dataclass
class RunConfig:
    model_id: str
    hf_revision: str
    threads: int
    batch_size: int
    warmup: int
    repeat: int
    prompt_lengths: List[int]
    output: Path
    runner_cmd: str | None
    mock: bool


def parse_args() -> RunConfig:
    parser = argparse.ArgumentParser(description="Quick.AI onboarding benchmark tool")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--hf-revision", required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--prompt-lengths", nargs="+", type=int, default=[128, 256, 512, 1024])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--runner-cmd",
        help=(
            "Optional external benchmark command template. Supports {phase} and {prompt_length}. "
            "Example: 'python run_bench.py --phase {phase} --len {prompt_length}'"
        ),
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Generate deterministic synthetic results (for CI/spec verification).",
    )
    args = parser.parse_args()

    if args.threads != 4:
        raise ValueError("threads must be 4 (fixed policy)")
    if args.batch_size != 1:
        raise ValueError("batch-size must be 1 (fixed policy)")
    if args.warmup != 3:
        raise ValueError("warmup must be 3 (fixed policy)")
    if args.repeat != 10:
        raise ValueError("repeat must be 10 (fixed policy)")
    allowed = {128, 256, 512, 1024}
    if set(args.prompt_lengths) != allowed:
        raise ValueError("prompt-lengths must include exactly: 128 256 512 1024")

    return RunConfig(
        model_id=args.model_id,
        hf_revision=args.hf_revision,
        threads=args.threads,
        batch_size=args.batch_size,
        warmup=args.warmup,
        repeat=args.repeat,
        prompt_lengths=sorted(args.prompt_lengths),
        output=args.output,
        runner_cmd=args.runner_cmd,
        mock=args.mock,
    )


def percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = (len(s) - 1) * p
    lo = int(idx)
    hi = min(lo + 1, len(s) - 1)
    w = idx - lo
    return s[lo] * (1 - w) + s[hi] * w


def run_external(cmd_template: str, phase: str, prompt_length: int) -> float:
    cmd = cmd_template.format(phase=phase, prompt_length=prompt_length)
    started = time.perf_counter()
    proc = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
    elapsed = time.perf_counter() - started
    stdout = proc.stdout.strip()
    if stdout:
        try:
            return float(stdout.splitlines()[-1])
        except ValueError:
            pass
    return elapsed


def run_mock(phase: str, prompt_length: int, i: int) -> float:
    base = {"prefill": 220.0, "decode": 80.0, "e2e": 65.0}[phase]
    scale = 128 / prompt_length
    return base * scale * (1.0 - 0.01 * (i % 3))


def measure_phase(cfg: RunConfig, phase: str, prompt_length: int) -> List[float]:
    for _ in range(cfg.warmup):
        if cfg.mock:
            _ = run_mock(phase, prompt_length, 0)
        elif cfg.runner_cmd:
            _ = run_external(cfg.runner_cmd, phase, prompt_length)
        else:
            raise ValueError("Provide --runner-cmd or --mock")

    samples: List[float] = []
    for i in range(cfg.repeat):
        if cfg.mock:
            value = run_mock(phase, prompt_length, i)
        else:
            value = run_external(cfg.runner_cmd or "", phase, prompt_length)
        samples.append(value)
    return samples


def summarize_tps(samples: List[float]) -> Dict[str, float]:
    return {"p50": round(percentile(samples, 0.5), 4), "p90": round(percentile(samples, 0.9), 4)}


def build_result(cfg: RunConfig) -> Dict[str, Any]:
    results = []
    for prompt_length in cfg.prompt_lengths:
        prefill = measure_phase(cfg, "prefill", prompt_length)
        decode = measure_phase(cfg, "decode", prompt_length)
        e2e = measure_phase(cfg, "e2e", prompt_length)
        results.append(
            {
                "prompt_length": prompt_length,
                "prefill_tps": summarize_tps(prefill),
                "decode_tps": summarize_tps(decode),
                "e2e": {
                    "latency_ms_p50": round(1000.0 / max(statistics.mean(e2e), 1e-9), 4),
                    "latency_ms_p90": round(1000.0 / max(percentile(e2e, 0.1), 1e-9), 4),
                },
            }
        )

    return {
        "model_id": cfg.model_id,
        "hf_revision": cfg.hf_revision,
        "runtime": {
            "device": "cpu",
            "threads": cfg.threads,
            "batch_size": cfg.batch_size,
            "warmup": cfg.warmup,
            "repeat": cfg.repeat,
        },
        "results": results,
    }


def main() -> None:
    cfg = parse_args()
    payload = build_result(cfg)
    cfg.output.parent.mkdir(parents=True, exist_ok=True)
    cfg.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Saved benchmark report: {cfg.output}")


if __name__ == "__main__":
    main()
