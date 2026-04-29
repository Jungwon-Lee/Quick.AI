# Smoke Test Checklist (To Do)

- Model ID:
- HF URL:
- Revision/Hash:
- Quick.AI Commit:
- Date:

## A. 준비
- [ ] 테스트 환경 정보 기록 (CPU/OS/Python/Threads)
- [ ] 테스트 입력 프롬프트 세트 준비 (short/medium)

## B. 로딩 스모크
- [ ] 모델 로딩 성공
- [ ] tokenizer/config 로딩 성공
- [ ] 첫 추론 호출 성공 (no crash)

## C. 기능 스모크 (FP32)
- [ ] 1-step 생성 성공
- [ ] NaN/Inf 없음
- [ ] 빈 출력/비정상 종료 없음

## D. 기능 스모크 (Q4_0)
- [ ] 양자화 모델 로딩 성공
- [ ] 짧은 생성 성공
- [ ] NaN/Inf 없음
- [ ] 출력 형식 비정상 없음

## E. 최소 성능 스모크 (빠른 게이트)
- [ ] length=128, batch=1, threads=4에서 prefill 측정 성공
- [ ] length=128, batch=1, threads=4에서 decode 측정 성공
- [ ] 결과 로그 저장 성공

## F. 회귀 스모크 (기존 대표 모델)
- [ ] 기존 모델 A 로딩/1-step 생성 성공
- [ ] 기존 모델 B 로딩/1-step 생성 성공

## G. 결과 판정
- [ ] PASS
- [ ] FAIL (원인 기록 필수)

## H. 실패 시 기록
- 증상:
- 재현 명령:
- 추정 원인:
- 다음 액션:
- 담당자:
- 목표 일정:
