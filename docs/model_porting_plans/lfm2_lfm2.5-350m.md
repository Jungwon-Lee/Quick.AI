# LFM2.5-350M Porting Plan

## Compatibility Report

- Hugging Face model: `LiquidAI/LFM2.5-350M`.
- Target mode: decoder-only causal LM.
- Architecture: `Lfm2ForCausalLM`, `model_type=lfm2`.
- Modeling source reviewed: Hugging Face Transformers
  `src/transformers/models/lfm2/modeling_lfm2.py` from the current main
  branch, plus the downloaded local `config.json` and `model.safetensors`.
- Layers: 16 total, with 10 gated short convolution blocks and 6 full GQA
  attention blocks. Attention layers use 16 query heads, 8 KV heads, 64 head
  size, RoPE theta 1000000, and Q/K RMSNorm.
- MLP: LFM2 `w1`, `w3`, `w2` SwiGLU with adjusted hidden size 4608.
- Tokenizer: ChatML-like template with `<|startoftext|>`,
  `<|im_start|>`, and `<|im_end|>`.
- New layer feature: model-specific `lfm2_short_conv` core under
  `models/lfm2/`. It implements HF's `C * causal_depthwise_conv(B * x)` after
  the standard in-projection and before the standard out-projection.
- Quantization scope: normal FC projections are eligible for `Q4_0`; the
  short-convolution depthwise kernel remains FP32 in both FP32 and Q4_0
  runtime configs.
- Unsupported or weakly verified behavior: full logits-level numeric parity has
  not been run yet. Deterministic FP32 output parity and FP32/Q4_0 smoke tests
  completed.

## Implementation Plan

1. Add `models/lfm2/` with `Lfm2CausalLM`, LFM2 graph construction, attention
   block wiring, convolution block wiring, MLP name mapping, custom-layer
   registration, and Meson integration.
2. Add a model-specific short convolution custom layer that stores the causal
   depthwise convolution cache and FP32 convolution kernel.
3. Register `Lfm2ForCausalLM` in `main.cpp`, `quantize.cpp`, and
   `api/causal_lm_api.cpp`.
4. Extend shared config parsing for HF-compatible `rope_parameters.rope_theta`,
   `tie_embedding`, and `norm_eps`.
5. Extend quantization dtype mapping to include LFM2 `ffn_w1`, `ffn_w3`,
   `conv_in`, and `conv_out` FC layers.
6. Add reviewable runtime assets under `res/lfm2/lfm2.5-350m/`:
   `config.json`, `generation_config.json`, FP32 and Q4_0 `nntr_config`
   files, and `weight_converter.py`.
7. Keep downloaded `model.safetensors`, tokenizer files, and model-card files
   ignored by git.
8. Validate with Linux build, converter tensor-key coverage, FP32 conversion,
   Q4_0 quantization, deterministic FP32 output parity, and FP32/Q4_0 smoke
   runs. Full logits-level numeric parity remains a follow-up gate.

## Validation Notes

- Python inspection dependencies available: `transformers 5.7.0`.
- Snapshot downloaded locally to `res/lfm2/lfm2.5-350m/` for inspection.
- Converter key-order audit against `model.safetensors`:
  148 expected tensor entries, 148 checkpoint keys, no missing keys.
- Standalone short-conv core check against PyTorch `conv1d` reference:
  max diff `1.1920928955078125e-07`.
- FP32 conversion command run:
  `python3 res/lfm2/lfm2.5-350m/weight_converter.py --model_path res/lfm2/lfm2.5-350m --output_name res/lfm2/lfm2.5-350m/nntr_lfm2.5_350m_fp32.bin`.
- Q4_0 quantization command run:
  `./build/quick_dot_ai_quantize res/lfm2/lfm2.5-350m --config res/lfm2/lfm2.5-350m/nntr_config_q4_0.json`.
- Q4_0 quantization result: source 1608 MB, output 410 MB, 25.5%
  compression.
- Build command run: `ninja -C build`.
- Build result: success, producing `libquick_dot_ai.so`, `quick_dot_ai_run`,
  `quick_dot_ai_quantize`, and `quick_dot_ai_test_api`.
- FP32 smoke command run:
  `./build/quick_dot_ai_run res/lfm2/lfm2.5-350m "<|startoftext|><|im_start|>user\nSay hi.<|im_end|>\n<|im_start|>assistant\n"`.
- FP32 smoke result: generated a coherent greeting response; prefill 13
  tokens, generation 57 tokens, peak memory 2074008 KB.
- Deterministic HF reference prompt:
  `<|im_start|>user\nSay hi.<|im_end|>\n<|im_start|>assistant\n`.
- Deterministic HF reference new tokens:
  `[36309, 510, 2213, 1011, 859, 1801, 1010, 4008, 540, 7]`.
- Deterministic HF reference text:
  `Hello! How can I help you today?<|im_end|>`.
- Deterministic Quick.AI FP32 prompt:
  `<|startoftext|><|im_start|>user\nSay hi.<|im_end|>\n<|im_start|>assistant\n`.
  The explicit `<|startoftext|>` is used because Quick.AI tokenizers-cpp does
  not add the tokenizer BOS automatically; this gives the same 13-token prefill
  sequence as the Hugging Face reference prompt above.
- Deterministic Quick.AI FP32 result:
  `Hello! How can I help you today?<|im_end|>`; prefill 13 tokens,
  generation 9 tokens, peak memory 1715684 KB.
- Q4_0 smoke command run from temporary directory `/tmp/lfm2-q4-smoke` using
  `nntr_config_q4_0.json` as `nntr_config.json`.
- Q4_0 smoke result: generated `Hi!<|im_end|>`; prefill 13 tokens,
  generation 2 tokens, peak memory 772308 KB.

## Follow-Up Validation Commands

```bash
python3 res/lfm2/lfm2.5-350m/weight_converter.py \
  --model_path res/lfm2/lfm2.5-350m \
  --output_name res/lfm2/lfm2.5-350m/nntr_lfm2.5_350m_fp32.bin

./build/quick_dot_ai_run res/lfm2/lfm2.5-350m
```

For deeper parity, compare full logits from a fixed prompt between Hugging Face
and Quick.AI.
