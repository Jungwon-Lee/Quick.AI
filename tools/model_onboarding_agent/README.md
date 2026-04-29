# Model Onboarding Agent

This directory contains tooling designed primarily for **Agent-driven automation** of new model onboarding, rather than manual script execution by end users.

## Quick User Guide

Users should submit a **natural-language request** to the Agent like this:

```text
Please onboard a new model.
- HF URL: <huggingface_url>
- revision: <hf_revision_or_commit>
- model_id: <model_id>

Requirements:
1) Follow the Mandatory Agent Workflow in tools/model_onboarding_agent
2) Record FP32 / Q4_0 validation results in reports
3) Always generate/update benchmark_results.json and onboarding_summary.md
4) Summarize Merge Gate pass/fail status with evidence at the end
```

## Mandatory Agent Workflow

This is the single required workflow for the Agent (not optional/recommended):

1. Initialize workspace via `onboarding_cli.initialize_workspace(model_id, hf_url, hf_revision, root)` (creates `reports/<model_id>/` and the four required report artifacts with initial metadata/placeholders).
2. Download model from Hugging Face URL (with revision/hash).
3. Implement model code while downloading, referencing `transformers` or `modeling_<model_name>.py`.
4. Implement `weight_converter.py` to produce Quick.AI-loadable FP32 `.bin`.
5. Validate FP32 `.bin` load/execution correctness.
6. Quantize FP32 to Q4_0 using `nntrainer_quantize`.
7. Validate Q4_0 model functionality/stability.
8. Run baseline benchmark with fixed policy (`threads=4`, `batch=1`, `warmup=3`, `repeat=10`, lengths `128/256/512/1024`).
9. Run a mandatory optimization loop: analyze bottleneck -> apply one optimization -> re-validate FP32/Q4_0 -> re-benchmark -> keep/revert based on correctness + metric gain.
10. Update summary/report artifacts (including per-iteration benchmark + validation evidence).
11. Apply merge gate decision (`Quick.AI/models/*.py` update only on full pass).

## Agent Workflow Visualization

```mermaid
flowchart TD
    A[1. Initialize workspace] --> B[2. Download model from HF]
    B --> C[3. Implement model code]
    C --> D[4. Implement weight_converter.py]
    D --> E[5. Validate FP32 .bin]
    E --> F[6. Quantize FP32 to Q4_0]
    F --> G[7. Validate Q4_0]

    G --> H[8. Run baseline benchmark]
    H --> I[Analyze bottleneck evidence]
    I --> J{Optimization candidate exists?}

    J -->|Yes| K[Apply exactly one optimization]
    K --> L[Re-validate FP32/Q4_0]
    L --> M{Validation pass?}
    M -->|No| N[Revert change + log reason]
    N --> I
    M -->|Yes| O[Re-benchmark under identical settings]
    O --> P{Metric improved vs best?}
    P -->|No| Q[Revert change + log result]
    Q --> R{Exit criteria met?}
    P -->|Yes| S[Keep change + update best metrics]
    S --> R
    R -->|No| I
    R -->|Yes| T[10. Update summary/report]

    J -->|No| T
    T --> U{11. Merge Gate passed?}
    U -->|Yes| V[Update Quick.AI/models/*.py]
    U -->|No| W[Record failure/repro/next action]
```

## Optimization Loop Checklist

For each iteration, record in `optimization_log.md`:
1. Bottleneck hypothesis (where/why it is slow)
2. Single change applied
3. FP32/Q4_0 re-validation result
4. Benchmark deltas (prefill/decode/end-to-end)
5. Decision: keep or revert

Stop the loop when 2 consecutive iterations each improve < 3% on both prefill/decode TPS, or if correctness/stability regresses.

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


## FAQ

### What does "Initialize onboarding workspace and report artifacts" mean?

It means the Agent must first bootstrap `reports/<model_id>/` and create the required files:
- `onboarding_summary.md`
- `benchmark_results.json`
- `optimization_log.md`
- `todo_smoke_test.md`

This step guarantees that every later workflow stage appends evidence to a consistent report structure.
