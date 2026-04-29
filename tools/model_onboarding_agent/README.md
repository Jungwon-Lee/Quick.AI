# Model Onboarding Agent (Codex-first)

이 디렉터리는 **사용자가 직접 스크립트를 실행하는 용도보다**, Codex가 신규 모델 온보딩을 자동 수행할 수 있도록 만든 도구 모음입니다.

## 사용자용 빠른 사용법

사용자는 아래처럼 **자연어 요청만** Codex에 전달하면 됩니다.

```text
새 모델 온보딩을 진행해줘.
- HF URL: <huggingface_url>
- revision: <hf_revision_or_commit>
- model_id: <model_id>

요구사항:
1) tools/model_onboarding_agent의 Codex-First Execution Order를 따를 것
2) FP32 / Q4_0 검증 결과를 리포트에 기록할 것
3) benchmark_results.json과 onboarding_summary.md를 반드시 생성/갱신할 것
4) Merge Gate 통과 여부와 근거를 마지막에 요약할 것
```

## Codex 권장 실행 순서

1. 워크스페이스 초기화  
   `onboarding_cli.initialize_workspace(model_id, hf_url, hf_revision, root)`
2. 벤치마크 실행  
   `bench_tool.run_benchmark(RunConfig(...))`
3. 요약 갱신  
   `run_onboarding_pipeline.update_summary(report_dir, benchmark_path)`
4. 필요 시 단일 엔트리포인트  
   `python tools/model_onboarding_agent/run_onboarding_pipeline.py ...`

## 필수 입력값

- Hugging Face URL
- revision/commit hash (권장: 명시, 미지정 시 `main`)
- model_id

## 결과물 확인 경로

- `reports/<model_id>/onboarding_summary.md`
- `reports/<model_id>/benchmark_results.json`
- `reports/<model_id>/optimization_log.md`
- `reports/<model_id>/todo_smoke_test.md`

## 실패 시 재요청 템플릿

```text
이전 온보딩에서 실패한 원인을 기준으로 재시도해줘.
- 동일 model_id 재사용
- 실패 원인/재현 방법/다음 액션을 onboarding_summary.md와 optimization_log.md에 업데이트
- 변경된 점(이번 시도에서 추가된 최적화/수정)을 마지막에 요약
```

