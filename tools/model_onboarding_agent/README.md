# Model Onboarding Agent (Codex-first)

This directory contains tooling designed primarily for **Codex-driven automation** of new model onboarding, rather than manual script execution by end users.

## Quick User Guide

Users should submit a **natural-language request** to Codex like this:

```text
Please onboard a new model.
- HF URL: <huggingface_url>
- revision: <hf_revision_or_commit>
- model_id: <model_id>

Requirements:
1) Follow the Codex-First Execution Order in tools/model_onboarding_agent
2) Record FP32 / Q4_0 validation results in reports
3) Always generate/update benchmark_results.json and onboarding_summary.md
4) Summarize Merge Gate pass/fail status with evidence at the end
```

## Recommended Codex Execution Order

1. Initialize workspace  
   `onboarding_cli.initialize_workspace(model_id, hf_url, hf_revision, root)`
2. Run benchmark  
   `bench_tool.run_benchmark(RunConfig(...))`
3. Update summary  
   `run_onboarding_pipeline.update_summary(report_dir, benchmark_path)`
4. Optional one-shot entrypoint  
   `python tools/model_onboarding_agent/run_onboarding_pipeline.py ...`

## Required Inputs

- Hugging Face URL
- revision/commit hash (recommended; if omitted, default `main`)
- model_id

## Expected Output Files

- `reports/<model_id>/onboarding_summary.md`
- `reports/<model_id>/benchmark_results.json`
- `reports/<model_id>/optimization_log.md`
- `reports/<model_id>/todo_smoke_test.md`

## Retry Template (After Failure)

```text
Please retry onboarding based on the previous failure.
- Reuse the same model_id
- Update onboarding_summary.md and optimization_log.md with: failure cause, reproduction steps, and next action
- Summarize what changed in this retry (fixes/optimizations)
```
