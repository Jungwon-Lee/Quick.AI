# SmallThinker Slim 및 Sparse Expert 최적화 구현 계획

## 요약

SmallThinker 최적화는 한 번에 cached-slim으로 가지 않고, 먼저 FSU 기반
`slim` variant를 추가한 뒤 그 위에 `cache_slim` variant를 쌓는 순서로
진행한다. 이후 SmallThinker의 구조적 특징인 `relu(gate) * up` 형태를 활용해
gate가 0이 되는 intermediate channel의 up/down 연산을 건너뛰는 sparse expert
compute를 추가한다.

기존 `SmallThinkerForCausalLM`과 `smallthinker_moe`는 compatibility와
correctness baseline 용도로 그대로 유지한다.

## 1단계: SmallThinker Slim Variant

- 새 architecture key를 추가한다: `SmallThinkerSlimForCausalLM`.
  - 기존 SmallThinker model 구성과 config parsing을 재사용한다.
  - MoE layer만 새 `smallthinker_moe_slim` layer로 교체한다.
  - `config.json`의 `architectures[0]` 값으로 opt-in 선택한다.

- `smallthinker_moe_slim` layer를 추가한다.
  - SmallThinker의 2-input contract를 유지한다.
    - expert input: FFN norm 출력
    - router input: pre-router 입력
  - expert gate/up/down weight는 virtual tensor로 요청한다.
  - router/gate weight는 resident 상태로 유지한다.
  - routing 결과로 선택된 현재 top-k expert만 `activate()`하고 compute 후 바로
    `deactivate()`한다.

- 이 단계의 목표는 cached policy 없이 FSU expert loading 자체를 SmallThinker에
  맞게 안정화하는 것이다.
  - correctness 비교가 단순하다.
  - cache eviction, LRU, resident 상태 bug와 FSU 기본 동작 bug를 분리할 수 있다.

## 2단계: SmallThinker Cache-Slim Variant

- `SmallThinkerCachedSlimForCausalLM` architecture key를 추가한다.
  - `SmallThinkerSlimForCausalLM`와 같은 model 구성 흐름을 쓰되,
    MoE layer만 `smallthinker_moe_cached_slim`로 교체한다.

- `smallthinker_moe_cached_slim` layer를 추가한다.
  - 1단계 slim layer의 routing, expert assignment, virtual weight loading
    흐름을 재사용한다.
  - 선택된 top-k expert가 resident가 아니면 `activate()`한다.
  - 사용된 expert는 LRU 위치를 갱신한다.
  - compute 후 resident expert 수가 32개를 넘으면 least-recent expert부터
    `deactivate()`한다.

- v1 cache policy:
  - cache budget은 32 experts로 hard-code한다.
  - extra top-k margin preload와 future-token speculative preload는 넣지 않는다.
  - 현재 token/chunk의 정확한 top-k routing 결과만 pre-load 기준으로 사용한다.

## 3단계: ReLU Gate 기반 Sparse Expert Compute

- SmallThinker expert compute의 수식은 다음 형태다.
  - `gate_out = input * expert_gate`
  - `up_out = input * expert_up`
  - `hidden = relu(gate_out) * up_out`
  - `output += hidden * expert_down * router_weight`

- `relu(gate_out)`가 0인 intermediate channel은 최종 output에 기여하지 않는다.
  따라서 해당 channel에 대해서는 다음 연산을 생략할 수 있다.
  - `up_out[j]` 계산
  - `hidden[j]` 계산
  - `expert_down[j, :]`와의 누적

- Generation fast path부터 sparse compute를 적용한다.
  - `seq_len == 1` 또는 `total_tokens == 1`에서 먼저 구현한다.
  - 선택된 expert마다 gate projection을 먼저 계산한다.
  - `gate_out[j] > 0`인 index만 `active_intermediate_indices`로 수집한다.
  - active index 수가 threshold보다 작으면 sparse path를 사용한다.
  - active density가 높으면 기존 dense path로 fallback한다.

- Sparse generation path의 compute 방식:
  - `gate_out`은 전체 intermediate size에 대해 계산한다. 이는 active channel을
    알기 위한 필수 비용이다.
  - active channel `j`에 대해서만 `up_out[j] = dot(input, up_col_j)`를 계산한다.
  - `hidden_j = gate_out[j] * up_out[j]`를 계산한다.
  - output vector에 `hidden_j * down_row_j`를 누적한다.
  - 마지막에 router weight를 곱하거나, 누적 시 `hidden_j`에 router weight를
    미리 곱한다.

- Prefill sparse path는 2차 최적화로 둔다.
  - Prefill에서는 token 수가 많아 dense GEMM 효율이 좋을 수 있으므로 sparse가
    항상 이기지 않는다.
  - 먼저 token별 active density를 측정한다.
  - 평균 density가 낮고 active pattern이 충분히 sparse할 때만 sparse batching을
    추가한다.
  - 그렇지 않으면 현재 expert별 token batching dense path를 유지한다.

- Sparse fallback 기준은 benchmark로 조정한다.
  - 초기값은 active density 40% 이하에서 sparse path 사용으로 둔다.
  - `active_count == 0`이면 해당 expert compute를 즉시 skip한다.
  - `active_count`가 높거나 tensor dtype/quantized weight 접근이 불리하면 dense
    path로 fallback한다.

## 구현 순서

1. `smallthinker_moe_slim`을 추가하고 기존 SmallThinker dense MoE와 output
   equivalence를 확인한다.
2. `SmallThinkerSlimForCausalLM` 등록과 Meson build 구성을 추가한다.
3. Slim variant에서 peak memory와 latency를 측정한다.
4. `smallthinker_moe_cached_slim`과 `SmallThinkerCachedSlimForCausalLM`을
   추가한다.
5. 32-expert LRU cache를 적용하고 slim 대비 generation latency 개선을 확인한다.
6. `seq_len == 1` sparse expert compute fast path를 추가한다.
7. Sparse threshold별 benchmark 후 default threshold를 확정한다.
8. Prefill sparse batching은 density 측정 결과가 유리할 때만 추가한다.

## Public Interface

- 새 architecture key:
  - `SmallThinkerSlimForCausalLM`
  - `SmallThinkerCachedSlimForCausalLM`

- 새 internal layer type:
  - `smallthinker_moe_slim`
  - `smallthinker_moe_cached_slim`

- 기존 interface는 변경하지 않는다.
  - `SmallThinkerForCausalLM`
  - `smallthinker_moe`
  - 기존 SmallThinker config key와 weight name

## 테스트 계획

- Build:
  - `meson setup build`
  - `ninja -C build quick_dot_ai_run`

- Correctness:
  - 기존 `SmallThinkerForCausalLM`, slim, cache-slim을 같은 model과 같은 prompt로
    실행한다.
  - deterministic setting에서 logits 또는 generated token이 FP 오차 범위 내에서
    같은지 확인한다.
  - sparse path와 dense fallback path의 output을 같은 input/expert로 비교한다.

- Behavior:
  - Slim variant에서 선택된 top-k expert만 activate/deactivate되는지 확인한다.
  - Cache-slim variant에서 resident expert가 32개를 넘지 않는지 확인한다.
  - ReLU gate sparse path에서 `gate_out <= 0` channel이 output에 기여하지 않는지
    확인한다.
  - `moe_primary_router_apply_softmax=true`와 `false`를 모두 테스트한다.

- Performance:
  - 기존 dense, slim, cache-slim, cache-slim+sparse를 단계별로 benchmark한다.
  - Prefill TPS, generation TPS, total latency, peak memory를 기록한다.
  - Sparse path는 active density, sparse/dense 선택 횟수, fallback 횟수를 함께
    기록한다.

## 가정 및 주의점

- Slim을 먼저 만드는 이유는 FSU loading correctness와 cache policy를 분리해서
  검증하기 위해서다.
- Cache-slim은 slim layer가 안정화된 뒤 같은 routing/assignment 코드를 공유해
  구현한다.
- ReLU sparse 최적화는 CPU compute 감소가 주목표이며, expert weight 전체를
  activate하는 현재 FSU 구조에서는 storage I/O 자체를 row 단위로 줄이지는 않는다.
- Quantized tensor의 raw layout 접근이 복잡하거나 backend별 차이가 있으면 sparse
  path는 FP32-access 가능한 경우에만 켜고 나머지는 dense fallback한다.
- Prefill은 dense GEMM 효율이 좋아 sparse가 불리할 수 있으므로 generation fast
  path를 먼저 구현한다.
