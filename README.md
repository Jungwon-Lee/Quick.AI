<div align="center">

<h1>☄️ Quick.AI</h1>

<h3><em>The fastest way to run an LLM on the device in your hand.</em></h3>

<p>
Production-grade causal-LM inference on top of <a href="https://github.com/nntrainer/nntrainer">NNTrainer</a> —<br/>
Qwen 3, GPT-OSS, Gemma 3, Llama and more, with <strong>MoE on phones</strong> via on-the-fly expert streaming.
</p>

<p>
  <a href="https://github.com/nntrainer/Quick.AI/actions/workflows/ci-linux.yml"><img src="https://github.com/nntrainer/Quick.AI/actions/workflows/ci-linux.yml/badge.svg" alt="Linux"/></a>
  <a href="https://github.com/nntrainer/Quick.AI/actions/workflows/ci-android.yml"><img src="https://github.com/nntrainer/Quick.AI/actions/workflows/ci-android.yml/badge.svg" alt="Android"/></a>
  <a href="https://github.com/nntrainer/Quick.AI/actions/workflows/cpp-linter.yml"><img src="https://github.com/nntrainer/Quick.AI/actions/workflows/cpp-linter.yml/badge.svg" alt="Format"/></a>
  <a href="https://github.com/nntrainer/Quick.AI/actions/workflows/codeql.yml"><img src="https://github.com/nntrainer/Quick.AI/actions/workflows/codeql.yml/badge.svg" alt="CodeQL"/></a>
  <br/>
  <img src="https://img.shields.io/badge/license-Apache%202.0-blue.svg?style=flat-square" alt="License"/>
  <img src="https://img.shields.io/badge/C%2B%2B-17-00599C?logo=c%2B%2B&logoColor=white&style=flat-square" alt="C++17"/>
  <img src="https://img.shields.io/badge/Android-NDK%20r26d-3DDC84?logo=android&logoColor=white&style=flat-square" alt="Android"/>
  <img src="https://img.shields.io/badge/platform-Linux%20·%20Android-lightgrey?style=flat-square" alt="Platform"/>
  <img src="https://img.shields.io/badge/offline-100%25-success?style=flat-square" alt="Offline"/>
</p>

<p>
  <a href="#quick-start">Quick start</a> ·
  <a href="#see-it-in-action">Demos</a> ·
  <a href="#supported-models">Models</a> ·
  <a href="#android-build">Android</a> ·
  <a href="#quantization">Quantization</a> ·
  <a href="docs/architecture.md">Architecture</a>
</p>

</div>

---

<div align="center">

### Quick.AI in three numbers

| Peak RAM | Library size | Network use |
|:---:|:---:|:---:|
| **16.5 GB → 1.3 GB** | **~13 MB** | **0 bytes** |
| Peak RAM for Qwen3-MoE 30B with FSU | Single core inference library | Sent over the network at runtime |

</div>

---

## Why Quick.AI?

<table>
<tr>
<td width="50%" valign="top">

### MoE that fits in your pocket
Run **30B-parameter Mixture-of-Experts** models in **~1.3 GB of RAM** with Flash Storage Utilization (FSU) — experts stream in from disk only when their tokens fire.

### Tuned for the metal
Hand-written kernels for **ARMv8.2-a** (FP16, dotprod, i8mm) and **AVX2** on x86_64. Multi-threaded with OpenMP, NEON-vectorized hot paths.

### Offline by design
Weights, prompts, and activations stay on the device. No telemetry, no Python runtime at inference time.

</td>
<td width="50%" valign="top">

### Pluggable layers
Each transformer building block (RMSNorm, SwiGLU, QKV, MHA core, tied embeddings…) ships as an **independently loadable `.so`** — drop in your own without recompiling the world.

### Embed anywhere
Native **C and C++ APIs** plus a clean Android JNI build. Same source tree builds for desktop, server, and mobile.

### Zero‑install quantizer
`quick_dot_ai_quantize` shrinks an FP32 checkpoint to **Q4_0 / Q4_K / Q6_K / FP16** in one command.

</td>
</tr>
</table>

---

## See it in action

<div align="center">

#### MoE inference on a phone

<table>
<tr>
<th align="center">GPT-OSS 20B</th>
<th align="center">Qwen3-MoE 30B-A3B</th>
</tr>
<tr>
<td align="center"><img src="docs/videos/GPT_OSS_20B_Demo.gif" width="300"/></td>
<td align="center"><img src="docs/videos/Qwen_30B_Demo.gif" width="300"/></td>
</tr>
</table>

#### FSU: the same model, the same machine, a 12× memory cut

<table>
<tr>
<th align="center">Load whole model<br/><sub>Qwen3-30B-A3B</sub></th>
<th align="center">Load experts on the fly<br/><sub>Quick.AI / FSU</sub></th>
</tr>
<tr>
<td align="center"><img src="docs/videos/moe-full.gif" width="300"/></td>
<td align="center"><img src="docs/videos/moe-on-the-fly.gif" width="300"/></td>
</tr>
<tr>
<td align="center"><b>Memory: 16.5 GB</b></td>
<td align="center"><b>Memory: 1.3 GB</b></td>
</tr>
</table>

</div>

---

## Supported models

| Family | Variants | Notes |
|---|---|---|
| **Llama** | 1B / 3B / 7B-class | reference architecture |
| **Qwen 2** | 0.5B – 7B | causal LM |
| **Qwen 3** | 0.6B · 1.7B · 4B · 8B · 14B · 32B | [HF: Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B) |
| **Qwen 3-MoE** | 30B-A3B | [HF: Qwen3-30B-A3B](https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507) · **FSU** |
| **GPT-OSS** | MoE 20B · 120B | [HF: gpt-oss-20b](https://huggingface.co/openai/gpt-oss-20b) · **FSU** |
| **SmallThinker** | 4B-A0.6B · 21B-A3B | [HF: SmallThinker-21BA3B-Instruct](https://huggingface.co/Tiiny/SmallThinker-21BA3B-Instruct) · CLI |
| **Gemma 3** | all causal variants | + sentence-embedding head |

> **Bring your own**: subclass the causal-LM template under `models/<your_family>/` and the [factory](factory.h) wires it in. See the [model author guide](models/README.md).

---

## Quick start

```bash
# 1 · Clone (with submodules — NNTrainer rides along)
git clone --recursive https://github.com/nntrainer/Quick.AI.git
cd Quick.AI

# 2 · System deps (Ubuntu 22.04 / 24.04)
sudo apt-get install -y libopenblas-dev libflatbuffers-dev flatbuffers-compiler \
                        build-essential pkg-config
pip install meson ninja

# 3 · Build (~1 min on a modern laptop)
meson setup build -Denable-fp16=true -Dthread-backend=omp -Domp-num-threads=4
ninja -C build

# 4 · Generate
export OMP_NUM_THREADS=4 OMP_WAIT_POLICY=active OMP_PROC_BIND=true OMP_PLACES=cores
./build/quick_dot_ai_run ./res/qwen3/qwen3-4b/
```

> **Model layout** — drop a model into `res/<name>/` containing
> `config.json`, `generation_config.json`, `tokenizer.json`, `tokenizer_config.json`,
> `vocab.json`, `nntr_config.json`, and the NNTrainer `.bin` weight file referenced from `nntr_config.json`.

---

## Android build

<details open>
<summary><b>Click to expand the modular Android pipeline</b></summary>

<br/>

**Prerequisites:** Android NDK r21d+, CMake, [Rust](https://rustup.rs) (for `tokenizers-cpp`), `adb`.

```bash
export ANDROID_NDK=/path/to/android-ndk
./build_android.sh        # libquick_dot_ai_core.so · quick_dot_ai · quick_dot_ai_quantize
./build_api_lib.sh        # (optional) libquick_dot_ai_api.so
./build_test_app.sh       # (optional) quick_dot_ai_test_api
./install_android.sh      # adb push to /data/local/tmp/quick_dot_ai/
```

| Script | Output(s) | Depends on |
|---|---|---|
| `build_android.sh` | `libquick_dot_ai_core.so`, `quick_dot_ai`, `quick_dot_ai_quantize` | NDK + Rust |
| `build_api_lib.sh` | `libquick_dot_ai_api.so` | core lib |
| `build_test_app.sh` | `quick_dot_ai_test_api` | core + api lib |
| `install_android.sh` | `/data/local/tmp/quick_dot_ai/*` | adb device |

Run on the phone:

```bash
adb shell /data/local/tmp/quick_dot_ai/run_causallm.sh <model_path>
adb shell /data/local/tmp/quick_dot_ai/run_quantize.sh <model_path>
adb shell /data/local/tmp/quick_dot_ai/run_test_api.sh <model_name> "<prompt>"
```

All artifacts land under `jni/libs/arm64-v8a/`.

</details>

---

## Quantization

```bash
# Default: FC → Q4_0, embedding → FP32
./build/quick_dot_ai_quantize /path/to/qwen3-4b

# Mix dtypes per layer family
./build/quick_dot_ai_quantize /path/to/qwen3-4b \
    --fc_dtype Q4_0 --embd_dtype Q6_K --lmhead_dtype FP16

# Write into a separate output directory
./build/quick_dot_ai_quantize /path/to/qwen3-4b -o /out/qwen3-4b-q40
```

| dtype | bits | typical use |
|---|---|---|
| `FP32` | 32 | embedding, LM head (default) |
| `FP16` | 16 | LM head when memory matters |
| `Q4_0` | 4 | FC layers (default), fastest path |
| `Q4_K` | 4 | FC layers, K-quant accuracy |
| `Q6_K` | 6 | embedding when 4-bit hurts quality |

> **Q4_0 is ISA-specific** — an x86-quantized Q4_0 binary is not byte-compatible with ARM. Quantize on the same architecture you serve from.

After quantization, point `quick_dot_ai_run` at the quantized directory (or `mv nntr_config_quantized.json nntr_config.json` in place and rerun).

---

## Continuous integration

Every PR is gated by:

| Check | What it does |
|---|---|
| **Linux build** | Meson + Ninja on Ubuntu 22.04 & 24.04 |
| **Android build** | `arm64-v8a`, NDK r26d, Rust `aarch64-linux-android` |
| **C++ format** | clang-format 14 against `.clang-format` |
| **CodeQL** | security & quality static analysis |

Workflows live under [`.github/workflows/`](.github/workflows/).

---

## Further reading

- [Architecture deep-dive](docs/architecture.md) — layered diagram, module-by-module breakdown, design choices
- [Model implementation guide](models/README.md)
- [C API reference](api/README.md)
- [Benchmark tooling](benchmarks/README.md)
- Talks & papers:
  - [Memory-Efficient LLM Inference on Edge Devices with NNTrainer](https://youtu.be/J2tUmi4bwMY?si=rJyiXkwr5iFrMhIK) — Open Source Summit 2025 Seoul
  - [A New Frontier of AI: On-Device AI Training and Personalization](https://dl.acm.org/doi/abs/10.1145/3639477.3639716) — ICSE-SEIP 2024
  - [NNTrainer: Light-Weight On-Device Training Framework](https://arxiv.org/pdf/2206.04688.pdf) — arXiv 2022

---

## Contributing

We love PRs. Before opening one:

1. `meson setup build && ninja -C build` — the same command CI runs.
2. `clang-format -i` on any changed C/C++ files (config in `.clang-format`).
3. Adding a new model family? Drop it under `models/<your_family>/`, wire it into `models/meson.build`, and register it in [`factory.h`](factory.h).

## License

Quick.AI is released under the [Apache License 2.0](LICENSE). NNTrainer, bundled as a submodule, is also Apache-2.0.

## Citation

If Quick.AI is useful for your research, please cite the NNTrainer paper it builds on:

```bibtex
@inproceedings{10.1145/3639477.3639716,
  author    = {Moon, Jijoong and Lee, Hyeonseok and Chu, Jiho and Park, Donghak and Hong, Seungbaek and Seo, Hyungjun and Jeong, Donghyeon and Kong, Sungsik and Ham, Myungjoo},
  title     = {A New Frontier of AI: On-Device AI Training and Personalization},
  booktitle = {Proceedings of the 46th International Conference on Software Engineering: Software Engineering in Practice},
  series    = {ICSE-SEIP '24},
  year      = {2024},
  pages     = {323--333},
  doi       = {10.1145/3639477.3639716}
}
```

<div align="center">

---

<sub>Built on top of <a href="https://github.com/nntrainer/nntrainer">NNTrainer</a>.</sub>

</div>
