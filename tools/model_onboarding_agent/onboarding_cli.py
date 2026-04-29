#!/usr/bin/env python3
"""Initialize onboarding workspace/report files for a model."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Initialize model onboarding report scaffold")
    p.add_argument("--model-id", required=True)
    p.add_argument("--hf-url", required=True)
    p.add_argument("--hf-revision", required=True)
    p.add_argument("--root", type=Path, default=Path("reports"))
    return p.parse_args()


def initialize_workspace(model_id: str, hf_url: str, hf_revision: str, root: Path) -> Dict[str, Path]:
    model_dir = root / model_id
    model_dir.mkdir(parents=True, exist_ok=True)

    summary = model_dir / "onboarding_summary.md"
    bench = model_dir / "benchmark_results.json"
    opt_log = model_dir / "optimization_log.md"
    smoke = model_dir / "todo_smoke_test.md"

    if not summary.exists():
        summary.write_text(
            "\n".join(
                [
                    f"# Onboarding Summary: {model_id}",
                    "",
                    f"- Created At (UTC): {datetime.now(timezone.utc).isoformat()}",
                    f"- HF URL: {hf_url}",
                    f"- HF Revision: {hf_revision}",
                    "",
                    "## Status",
                    "- [ ] Analysis",
                    "- [ ] FP32 Validation",
                    "- [ ] Q4_0 Validation",
                    "- [ ] Benchmark",
                    "- [ ] Optimization",
                    "- [ ] Merge Gate",
                    "",
                    "## Notes",
                    "-",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

    if not bench.exists():
        bench.write_text("{}\n", encoding="utf-8")

    if not opt_log.exists():
        opt_log.write_text("# Optimization Log\n\n", encoding="utf-8")

    if not smoke.exists():
        smoke.write_text(
            "# Smoke Test TODO\n\n- Use `tools/model_onboarding_agent/SMOKE_TEST_CHECKLIST_TEMPLATE.md`.\n",
            encoding="utf-8",
        )

    return {
        "model_dir": model_dir,
        "summary": summary,
        "benchmark": bench,
        "optimization_log": opt_log,
        "smoke_todo": smoke,
    }


def main() -> None:
    args = parse_args()
    paths = initialize_workspace(args.model_id, args.hf_url, args.hf_revision, args.root)
    print(f"Initialized onboarding workspace: {paths['model_dir']}")


if __name__ == "__main__":
    main()
