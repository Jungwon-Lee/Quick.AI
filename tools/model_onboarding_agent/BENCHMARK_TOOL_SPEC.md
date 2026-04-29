# Benchmark Tool Spec (CPU-first)

## 1) Goal
신규 모델 구조 온보딩 시 성능 측정을 표준화하기 위한 벤치마크 Tool 요구사항 문서.

## 2) Fixed Runtime Parameters
- Device: CPU only
- Threads: 4
- Batch size: 1
- Warm up: 3 iterations
- Measurement repeat: 10 iterations
- Prompt lengths: 128, 256, 512, 1024

## 3) Required Measurements
각 prompt length에 대해 아래 3개를 모두 측정한다.
1. Prefill TPS
2. Decode TPS
3. Prefill&Decode end-to-end latency/TPS

## 4) CLI Interface (Draft)
```bash
python tools/model_onboarding_agent/bench_tool.py \
  --model-id <model_id> \
  --hf-revision <revision_or_hash> \
  --threads 4 \
  --batch-size 1 \
  --warmup 3 \
  --repeat 10 \
  --prompt-lengths 128 256 512 1024 \
  --runner-output-unit tps \
  --output reports/<model_id>/benchmark_results.json
```

## 5) Output Schema (JSON)
```json
{
  "model_id": "string",
  "hf_revision": "string",
  "runtime": {
    "device": "cpu",
    "threads": 4,
    "batch_size": 1,
    "warmup": 3,
    "repeat": 10,
    "runner_output_unit": "tps"
  },
  "results": [
    {
      "prompt_length": 128,
      "prefill_tps": {"p50": 0.0, "p90": 0.0},
      "decode_tps": {"p50": 0.0, "p90": 0.0},
      "e2e": {
        "tps": {"p50": 0.0, "p90": 0.0},
        "latency_ms": {"p50": 0.0, "p90": 0.0}
      }
    }
  ]
}
```

## 6) Comparison Policy
- 최적화 전/후는 동일 파라미터로만 비교한다.
- 비교 결과는 p50, p90 기준으로 모두 기록한다.
- 개선 여부 판정은 onboarding_summary.md에 명시한다.

## 7) Integration Plan (To Do)
- [ ] 스모크 테스트 체크리스트와 통합 실행
- [ ] CI 야간 성능 회귀 체크와 연동
- [ ] 모델별 baseline 스냅샷 관리
