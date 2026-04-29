#!/usr/bin/env python3
"""Run end-to-end onboarding scaffolding + benchmark (mock/external)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.model_onboarding_agent.bench_tool import RunConfig, run_benchmark  # noqa: E402
from tools.model_onboarding_agent.onboarding_cli import initialize_workspace  # noqa: E402


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


def update_summary(report_dir: Path, benchmark_path: Path) -> None:
    summary_path = report_dir / "onboarding_summary.md"
    if not summary_path.exists():
        return

    payload = json.loads(benchmark_path.read_text(encoding="utf-8"))
    e2e_rows = payload.get("results", [])
    unit = payload.get("runtime", {}).get("runner_output_unit", "tps")
    first_row = e2e_rows[0] if e2e_rows else {}
    prompt = first_row.get("prompt_length", "-")
    metric = first_row.get("e2e", {}).get(unit, {})
    p50 = metric.get("p50", "-")
    p90 = metric.get("p90", "-")

    summary = summary_path.read_text(encoding="utf-8")
    summary = summary.replace("- [ ] Benchmark", "- [x] Benchmark")
    marker = "## Notes\n"
    note = (
        f"- Auto benchmark complete: unit={unit}, first_prompt_length={prompt}, "
        f"p50={p50}, p90={p90}, file={benchmark_path.name}\n"
    )
    if marker in summary:
        summary = summary.replace(marker, marker + note, 1)
    else:
        summary += "\n## Notes\n" + note
    summary_path.write_text(summary, encoding="utf-8")


def main() -> None:
    args = parse_args()
    report_paths = initialize_workspace(args.model_id, args.hf_url, args.hf_revision, args.reports_root)
    report_dir = report_paths["model_dir"]

    if not args.mock and not args.runner_cmd:
        raise SystemExit("Either --mock or --runner-cmd must be provided")
    cfg = RunConfig(
        model_id=args.model_id,
        hf_revision=args.hf_revision,
        threads=args.threads,
        batch_size=args.batch_size,
        warmup=args.warmup,
        repeat=args.repeat,
        prompt_lengths=sorted(args.prompt_lengths),
        output=report_dir / "benchmark_results.json",
        runner_cmd=args.runner_cmd,
        mock=args.mock,
        runner_output_unit=args.runner_output_unit,
    )
    run_benchmark(cfg)
    update_summary(report_dir, cfg.output)
    manifest = {
        "model_id": args.model_id,
        "report_dir": str(report_dir),
        "files": {k: str(v) for k, v in report_paths.items() if k != "model_dir"},
    }
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
