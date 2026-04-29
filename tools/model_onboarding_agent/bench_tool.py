#!/usr/bin/env python3
"""CPU-first benchmark runner for Quick.AI model onboarding."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
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
    runner_output_unit: str


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
        "--runner-output-unit",
        choices=["tps", "latency_ms"],
        default="tps",
        help="Unit emitted by --runner-cmd stdout's last line. Mock mode ignores this.",
    )
    parser.add_argument("--mock", action="store_true", help="Generate deterministic synthetic results.")
    args = parser.parse_args()

    if args.threads != 4:
        parser.error("threads must be 4 (fixed policy)")
    if args.batch_size != 1:
        parser.error("batch-size must be 1 (fixed policy)")
    if args.warmup != 3:
        parser.error("warmup must be 3 (fixed policy)")
    if args.repeat != 10:
        parser.error("repeat must be 10 (fixed policy)")

    allowed = {128, 256, 512, 1024}
    if set(args.prompt_lengths) != allowed:
        parser.error("prompt-lengths must include exactly: 128 256 512 1024")

    if not args.mock and not args.runner_cmd:
        parser.error("Provide either --mock or --runner-cmd")

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
        runner_output_unit=args.runner_output_unit,
    )


def percentile(values: List[float], p: float) -> float:
    s = sorted(values)
    idx = (len(s) - 1) * p
    lo = int(idx)
    hi = min(lo + 1, len(s) - 1)
    w = idx - lo
    return s[lo] * (1 - w) + s[hi] * w


def run_external(cmd_template: str, phase: str, prompt_length: int) -> float:
    cmd = cmd_template.format(phase=phase, prompt_length=prompt_length)
    proc = subprocess.run(shlex.split(cmd), check=True, capture_output=True, text=True)
    stdout = proc.stdout.strip()
    if not stdout:
        raise ValueError("runner command must print numeric value on stdout")
    try:
        return float(stdout.splitlines()[-1])
    except ValueError as e:
        raise ValueError(f"unable to parse numeric value from runner output: {stdout!r}") from e


def run_mock(phase: str, prompt_length: int, i: int) -> float:
    base = {"prefill": 220.0, "decode": 80.0, "e2e": 65.0}[phase]
    scale = 128 / prompt_length
    return base * scale * (1.0 - 0.01 * (i % 3))


def measure_phase(cfg: RunConfig, phase: str, prompt_length: int) -> List[float]:
    for _ in range(cfg.warmup):
        if cfg.mock:
            run_mock(phase, prompt_length, 0)
        else:
            run_external(cfg.runner_cmd or "", phase, prompt_length)

    samples: List[float] = []
    for i in range(cfg.repeat):
        value = run_mock(phase, prompt_length, i) if cfg.mock else run_external(cfg.runner_cmd or "", phase, prompt_length)
        samples.append(value)
    return samples


def summarize(values: List[float]) -> Dict[str, float]:
    return {"p50": round(percentile(values, 0.5), 4), "p90": round(percentile(values, 0.9), 4)}


def build_result(cfg: RunConfig) -> Dict[str, Any]:
    results = []
    for prompt_length in cfg.prompt_lengths:
        prefill = measure_phase(cfg, "prefill", prompt_length)
        decode = measure_phase(cfg, "decode", prompt_length)
        e2e = measure_phase(cfg, "e2e", prompt_length)

        if cfg.mock or cfg.runner_output_unit == "tps":
            # For tps inputs, lower tps implies higher latency. We map latency p90
            # from tps p10 intentionally (tail-latency proxy).
            e2e_summary = {
                "tps": summarize(e2e),
                "latency_ms": {
                    "p50": round(1000.0 / max(percentile(e2e, 0.5), 1e-9), 4),
                    "p90": round(1000.0 / max(percentile(e2e, 0.1), 1e-9), 4),
                },
            }
        else:
            e2e_summary = {
                "latency_ms": summarize(e2e),
                "tps": {
                    "p50": round(1000.0 / max(percentile(e2e, 0.5), 1e-9), 4),
                    "p90": round(1000.0 / max(percentile(e2e, 0.9), 1e-9), 4),
                },
            }

        results.append(
            {
                "prompt_length": prompt_length,
                "prefill_tps": summarize(prefill),
                "decode_tps": summarize(decode),
                "e2e": e2e_summary,
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
            "runner_output_unit": cfg.runner_output_unit,
        },
        "results": results,
    }


def main() -> None:
    cfg = parse_args()
    run_benchmark(cfg)


def run_benchmark(cfg: RunConfig) -> Dict[str, Any]:
    payload = build_result(cfg)
    cfg.output.parent.mkdir(parents=True, exist_ok=True)
    cfg.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return payload


if __name__ == "__main__":
    main()
