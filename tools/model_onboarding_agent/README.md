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
1) Follow the Agent-First Execution Order in tools/model_onboarding_agent
2) Record FP32 / Q4_0 validation results in reports
3) Always generate/update benchmark_results.json and onboarding_summary.md
4) Summarize Merge Gate pass/fail status with evidence at the end
```

## Recommended Execution Order

1. Initialize workspace  
   `onboarding_cli.initialize_workspace(model_id, hf_url, hf_revision, root)`
2. Run benchmark  
   `bench_tool.run_benchmark(RunConfig(...))`
3. Update summary  
   `run_onboarding_pipeline.update_summary(report_dir, benchmark_path)`
4. Optional one-shot entrypoint  
   `python tools/model_onboarding_agent/run_onboarding_pipeline.py ...`

### Additional Onboarding Steps

5. Download model from Hugging Face URL.
5-1. While downloading, implement model code by referencing `transformers` or `modeling_<model_name>.py`.
6. Implement `weight_converter.py` to convert downloaded weights into a Quick.AI-loadable `.bin` file.
7. Validate that the FP32 `.bin` model loads correctly.
8. After FP32 validation, quantize FP32 to Q4_0 using `nntrainer_quantize`.
9. Validate the quantized Q4_0 model.

## Agent Workflow Visualization

```mermaid
flowchart TD
    A[1. Initialize workspace] --> B[2. Run benchmark]
    B --> C[3. Update summary report]
    C --> D{4. One-shot entrypoint?}
    D -->|Yes| E[Continue with generated artifacts]
    D -->|No| E
    E --> F[5. Download model from Hugging Face]
    F --> G[6. Implement model code
(transformers / modeling_<model_name>.py)]
    G --> H[7. Implement weight_converter.py
(.bin conversion)]
    H --> I[8. Validate FP32 .bin model]
    I --> J[9. Quantize FP32 to Q4_0
(nntrainer_quantize)]
    J --> K[10. Validate Q4_0 model]
    K --> L{11. Merge Gate passed?}
    L -->|Yes| M[Update Quick.AI/models/*.py]
    L -->|No| N[Record failure/repro/next action
in onboarding_summary.md or optimization_log.md]
```

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
