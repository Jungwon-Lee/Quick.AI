# Quick.AI Model Onboarding Agent Role

## 1) Mission
이 Agent의 목적은 Hugging Face의 신규 모델 구조를 Quick.AI에서 실행 가능하도록 구현하고,
정합성 검증 및 성능 최적화를 완료한 뒤, 기준을 통과한 경우에만 `Quick.AI/models`에 `.py` 코드를 추가하는 것이다.

---

## 2) Scope & Constraints
- 구현 위치: `tools/` (Agent 로직), `Quick.AI/models` (모델 구현 코드)
- `Quick.AI/models`에는 `.py` 파일만 추가한다.
- 우선 지원 대상: CPU only
- 기본 구현 대상 정밀도:
  - 필수: FP32, Q4_0
  - 선택(optional): FP16 (Android 지원 필요 시 검증)
- 모델 정보 입력은 Hugging Face URL(revision 포함 가능)로 받는다.
- HF `revision` 또는 `commit hash`를 반드시 기록한다. 미지정 시 기본값 `main`을 사용하되 재현성 경고를 남긴다.

---

## 3) Inputs / Outputs

### Inputs
- Hugging Face model URL
- (권장) revision or commit hash
- 테스트/벤치마크 설정값

### Outputs
- 모델 구조 구현 코드 (`Quick.AI/models/*.py`)
- 검증 코드 및 실행 로그
- 성능 벤치마크 결과
- 온보딩 요약 리포트

---

## 4) Mandatory Workflow (반드시 순서 준수)

1. 모델 분석
   - HF에서 config/modeling/weights 확인
   - Quick.AI 미지원 레이어/연산 탐지
2. 테스트 설계 및 작성 (구현 전에 먼저)
   - FP32 정합성 테스트
   - 대형 모델이면 layer-wise 테스트 포함
   - Q4_0 정상동작 테스트
3. 모델 구현
   - 신규 레이어/연산 구현
4. 정합성 검증
   - FP32 기준으로 참조 구현(PyTorch/HF)과 비교
   - Q4_0은 정상 출력/안정성 확인
5. 성능 측정 및 최적화
   - prefill / decode / prefill&decode 측정
   - 개선 시 코드 반영
6. 결과 기록
   - 검증/최적화/성능 측정 내역을 파일에 기록
7. 반영 결정
   - 모든 필수 게이트 통과 시에만 `Quick.AI/models`에 `.py` 추가/업데이트

---

## 5) Validation Policy

### FP32 (필수)
- 참조: HF FP32 모델 출력
- 목표: Quick.AI FP32 출력과 수치적으로 매우 근접
- 임계값은 충분히 작은 값 사용 (프로젝트 표준값 적용)
- 대형 모델은 layer-wise 비교를 병행

### Q4_0 (필수)
- FP32와 exact match는 요구하지 않음
- 정상동작 기준:
  - 추론이 중단 없이 수행됨
  - NaN/Inf 없음
  - 출력 형태/길이가 비정상적이지 않음

### FP16 (선택)
- Android 지원 필요 시 검증 수행
- 필요 시 별도 환경에서 검증 로그 첨부

---

## 6) Benchmark Policy (고정값)

- 대상: CPU
- batch size: 1
- threads: 4
- warm up: 3회
- measurement repeat: 10회
- prompt lengths: 128, 256, 512, 1024
- 측정 항목:
  1) prefill TPS
  2) decode TPS
  3) prefill&decode end-to-end TPS/latency

동일 조건에서 최적화 전/후 비교하며, 결과는 표 형태로 기록한다.

---

## 7) Smoke Test Policy
- 스모크 테스트는 현재 즉시 구현하지 않고 **To Do**로 관리한다.
- 추후 별도 측정 Tool에서 자동 실행되도록 통합한다.
- 본 문서의 체크리스트 템플릿과 `BENCHMARK_TOOL_SPEC.md`를 기준으로 Tool 요구사항을 정의한다.

---

## 8) Reporting Requirements

필수 기록 파일(예시):
- `reports/<model_id>/onboarding_summary.md`
- `reports/<model_id>/benchmark_results.json`
- `reports/<model_id>/optimization_log.md`
- `reports/<model_id>/todo_smoke_test.md` (To Do 관리용)

리포트에는 최소 아래 정보를 포함:
- HF URL + revision/hash
- Quick.AI commit hash
- 실행 환경(CPU 모델, OS, Python, thread 설정)
- FP32/Q4_0 검증 결과
- 벤치마크 설정 및 결과(길이별)
- 최적화 전/후 비교
- 잔여 TODO 및 리스크

---

## 9) Merge / Update Gate

아래를 모두 통과해야 `Quick.AI/models` 변경 반영 가능:
1. FP32 검증 통과
2. Q4_0 정상동작 검증 통과
3. 벤치마크 결과 기록 완료 (128/256/512/1024, warmup/repeat/threads 규칙 준수)
4. 최적화 내역 및 근거 기록 완료

미통과 시:
- `Quick.AI/models` 반영 금지
- 실패 원인, 재현 방법, 다음 액션 기록 필수

---

## 10) Non-Goals
- GPU 최적화는 현재 범위 밖
- 스모크 테스트 자동화 Tool 구현은 현재 범위 밖(To Do)

## 11) Appendix
- 스모크 테스트 체크리스트: `SMOKE_TEST_CHECKLIST_TEMPLATE.md`
- 벤치마크 Tool 요구사항: `BENCHMARK_TOOL_SPEC.md`
- 온보딩 리포트 초기화 CLI: `onboarding_cli.py`
