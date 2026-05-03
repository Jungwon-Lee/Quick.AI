# Model Porting Harness

This document defines a reusable Codex subagent workflow for adding support for
a new Hugging Face model to Quick.AI.

The harness input is a Hugging Face repository URL or a local model directory.
The harness output is a reviewable patch set that can build and run in
Quick.AI: model C++ code, build wiring, factory/API registration,
`nntr_config.json`, `weight_converter.py`, and validation notes.

## Goals

- Support both decoder-only causal language models and embedding models.
- Reuse existing Quick.AI model and layer patterns before adding new layers.
- Download a local snapshot of the target model so inspection, tensor-key
  mapping, and converter smoke tests use the real files, not only remote
  metadata.
- Download and inspect the Hugging Face modeling source, especially
  `modeling_<model_name>.py` or the closest custom modeling file, so the C++
  graph follows the actual Python layer order, tensor transforms, masks, cache
  behavior, and logits path.
- When the target model requires a layer feature that Quick.AI does not yet
  support, prefer implementing the feature as a model-specific custom layer
  under `models/<family>/` when it is only needed for that family. Modify shared
  `layers/` code only when the behavior is clearly reusable across existing or
  expected model families and the compatibility risk is understood.
- Make tensor ordering explicit so the C++ graph and weight converter agree.
- Require an explicit implementation plan before coding, so new model features
  are classified as already supported, model-specific custom layer work,
  reusable shared layer work, model-specific graph work, converter work,
  verification work, or blocking runtime gaps.
- Write model-specific implementation plans to a separate plan file, not back
  into this harness document.
- Verify generated output against a Hugging Face reference run. A runtime that
  merely loads and emits text is not enough for acceptance when the model is
  expected to have parity.
- Verify incrementally: first prove at least one representative layer or newly
  implemented feature matches Python output, then compare the full model
  against Hugging Face output.
- Stop with a compatibility report only when the model is outside the current
  harness scope, cannot be inspected safely, or requires runtime capabilities
  that cannot be implemented in a reviewable patch.
- Keep model weights, tokens, generated binaries, and build outputs out of git.

## Inputs

Required input:

- `hf_model`: Hugging Face repository URL, repository id, or local model path.

Optional inputs:

- `target_mode`: `auto`, `causal_lm`, or `embedding`. Default: `auto`.
  `causal_lm` may be used for an explicitly text-only port of a multimodal
  conditional-generation repository, but the compatibility report must say that
  multimodal behavior is intentionally out of scope and list all remaining
  parity gaps.
- `family_name`: Quick.AI model family directory name. Default: derive from
  the normalized Hugging Face model family.
- `sample_input`: Prompt or embedding input used for smoke testing.
- `allow_remote_code`: Whether analysis may load HF custom code. Default:
  `true` for inspection only.
- `snapshot_dir`: Local directory for downloaded Hugging Face metadata and
  weights. Default: `res/<family>/<model>`. Keep downloaded weights, generated
  `.bin` files, and cache files ignored unless the task explicitly requires a
  local runtime artifact.
- `download_weights`: Whether to download weight files as well as config,
  tokenizer, and model code. Default: `true`, because tensor-key inspection and
  converter smoke tests require the real checkpoint.
- `modeling_source`: Hugging Face modeling source file to use as the Python
  reference. Default: discover from the snapshot, preferring
  `modeling_<model_name>.py`, then `modeling_*.py`, then the installed
  Transformers implementation for `config.model_type`.

## Output Package

A completed harness run must produce these review artifacts:

- Compatibility report describing the HF architecture, tensor names, tokenizer,
  custom code, modeling source file reviewed, newly implemented layer features,
  remaining unsupported operations, and chosen Quick.AI base pattern.
- Implementation plan that converts the compatibility findings into ordered
  coding tasks, ownership boundaries, validation tasks, and blocking decisions.
  This plan must be written to a new model-specific file, such as
  `docs/model_porting_plans/<family>_<model>.md`, instead of being appended to
  this harness document.
- C++ model implementation under `models/<family>/`.
- Meson integration through `models/<family>/meson.build` and
  `models/meson.build`.
- Factory registration in every runtime entry point that dispatches by HF
  architecture: `main.cpp`, `quantize.cpp`, and `api/causal_lm_api.cpp`.
- Runtime sample assets under `res/<family>/<model>/`, including
  `nntr_config.json` and `weight_converter.py`.
- Reference-output validation notes, including the exact prompt, decoding
  settings, Hugging Face output tokens/text or embeddings, Quick.AI output
  tokens/text or embeddings, and the comparison result.
- Build, conversion, quantization, and smoke-test commands that were run or are
  required before merge.

## Lead Agent Workflow

The lead agent owns the run from input to final patch. It may delegate the
subtasks below to subagents in parallel, then merge their outputs into one
implementation.

1. Normalize the input model URL/path and choose a local `snapshot_dir`.
2. Bootstrap the Python inspection environment before assigning HF-specific
   work. At minimum, verify `numpy`, `torch`, `transformers`, `safetensors`,
   `huggingface_hub`, and `accelerate` import successfully. For embedding
   ports, also verify `sentence_transformers`. For multimodal ports, verify the
   modality packages required by the model card, such as `torchvision`,
   `pillow`, and `librosa`.
3. Download the target model snapshot into `snapshot_dir`. Include
   `config.json`, `generation_config.json`, tokenizer files, processor files,
   model index files, safetensors files, and modeling source files. Do not
   commit the downloaded snapshot, weights, tokens, or cache files.
4. Collect from the local snapshot: `config.json`, `generation_config.json`,
   tokenizer files, processor files, model index files, safetensors files, and
   modeling source files. If the repo exposes a large single safetensors file,
   still download it when disk allows; converter tensor order must be based on
   real checkpoint keys and shapes.
5. Locate the Python modeling source used by the target model. Prefer a local
   `modeling_<model_name>.py` file from the snapshot, then another local
   `modeling_*.py` file, then the matching installed Transformers modeling
   module. Record the exact file path or module name and inspect the relevant
   layer classes before mapping C++ graph order or converter tensor order.
6. Decide `target_mode`. If `target_mode=auto`, choose `embedding` for encoder
   or sentence-transformer style models and `causal_lm` for decoder-only
   generation models. For multimodal conditional-generation models, either keep
   full multimodal support in scope or explicitly choose a text-only
   `causal_lm` subset. A text-only subset must not be presented as HF parity.
7. Assign the Snapshot Downloader, HF Inspector, and Architecture Mapper roles
   before implementation work. After those reports are available, assign the
   Model Feature Planner to analyze the target model features and produce the
   implementation plan before coding roles start.
8. Merge the inspector, mapper, and planner outputs into a single
   model-porting design and classify every required feature as already
   supported, model-specific custom layer work, reusable shared layer work,
   model-specific graph work, converter work, verification work, intentionally
   excluded, or blocked.
9. Assign implementation roles only after the planner output is available.
10. Implement missing Quick.AI layer features required by the target model
   before wiring the full model graph. Prefer new model-specific custom layers
   under `models/<family>/` for target-only behavior. Examples include RoPE
   variants, normalization shapes, activation variants, attention masks, logit
   transforms, routing features, cache behavior, or processor-adjacent tensor
   preparation that can fit inside the current runtime architecture. Promote
   the feature into shared `layers/` only when the planner shows that multiple
   model families should share the behavior and existing models remain
   compatible.
11. For every new or changed layer feature, run partial layer verification
   against a Python reference implementation before relying on full-model
   output. Compare fixed-input Quick.AI/NNTrainer layer output with the matching
   Hugging Face layer from `modeling_<model_name>.py` or a standalone PyTorch
   equivalent, and record tolerances and first-difference diagnostics.
12. Implement the model patch unless the compatibility report has blocking
   issues that are outside the patch scope.
13. Verify that C++ layer creation order and converter tensor save order are the
   same.
14. Generate a deterministic Hugging Face reference output from the same local
    snapshot used by the converter, then compare it with the Quick.AI runtime
    output from the converted weights. Treat mismatches as implementation bugs
    unless the compatibility report explicitly accepts a partial/text-only
    subset and lists the missing semantics that explain the mismatch.
15. Run the relevant validation commands, or record why they could not be run.
    For very large FP32 models, a runtime smoke test that loads the model and
    generates partial text may be acceptable as "partial" validation when
    generation does not finish promptly on CPU. Partial validation must not be
    reported as output parity; stop long-running processes and report exactly
    what completed.

## Python Dependency Bootstrap

Run this before any subagent tries to load a Hugging Face config, tokenizer, or
model:

```bash
python3 - <<'PY'
import importlib.util

required = [
    "numpy",
    "torch",
    "transformers",
    "safetensors",
    "huggingface_hub",
    "accelerate",
]

missing = [name for name in required if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit("Missing Python packages: " + ", ".join(missing))

import transformers
print("transformers", transformers.__version__)
PY
```

If packages are missing, install the baseline converter/inspection stack:

```bash
python3 -m pip install --user --upgrade \
  numpy torch transformers safetensors huggingface_hub accelerate
```

For embedding model ports:

```bash
python3 -m pip install --user --upgrade sentence-transformers
```

For multimodal model ports, install the modality dependencies needed for
inspection and converter authoring:

```bash
python3 -m pip install --user --upgrade pillow torchvision librosa
```

Some newly released models require a Transformers version newer than the latest
stable PyPI release. If `config.json` has a future or development
`transformers_version`, or the model card says support landed after the
installed release, install Transformers from source in the active user
environment:

```bash
python3 -m pip install --user --upgrade \
  git+https://github.com/huggingface/transformers.git
```

Use a virtual environment instead of `--user` if the machine has one dedicated
to Quick.AI model-porting work. Do not commit virtual environments, package
caches, downloaded model weights, or Hugging Face tokens.

## Model Snapshot Download

Every Hugging Face model port must use a local snapshot for inspection and
converter validation. Prefer `huggingface_hub.snapshot_download` so the snapshot
path is explicit and can live in the same `res/<family>/<model>` runtime
directory used by Quick.AI:

```bash
python3 - <<'PY'
from huggingface_hub import snapshot_download

repo_id = "<hf_repo_id>"
snapshot_dir = "res/<family>/<model>"

path = snapshot_download(
    repo_id=repo_id,
    local_dir=snapshot_dir,
    local_dir_use_symlinks=False,
)
print(path)
PY
```

For gated models, authenticate outside the repository before running the
harness. Do not paste or commit Hugging Face tokens. If the user supplied a
local model directory, verify it contains the same files a snapshot would
provide and use that directory as `snapshot_dir`.

When disk, network, authorization, or gating prevents downloading weights,
download at least metadata and tokenizer files, record the limitation in the
compatibility report, and do not claim tensor-order validation has passed.

After the snapshot is available, find the Python modeling source that defines
the target classes. Many custom repositories provide a file named
`modeling_<model_name>.py`; use that file as the first reference for module
composition and per-layer behavior. If the snapshot does not include custom
modeling code, record the installed Transformers source module used for the
architecture. The inspector and graph designer should cite the source file or
module they used, not only `config.json`.

After downloading weights, inspect actual safetensors keys and shapes before
authoring the converter. For example:

```bash
python3 - <<'PY'
from safetensors import safe_open

path = "res/<family>/<model>/model.safetensors"
with safe_open(path, framework="pt", device="cpu") as f:
    keys = list(f.keys())
    print("num_keys", len(keys))
    for key in keys[:100]:
        print(key, tuple(f.get_tensor(key).shape))
PY
```

For sharded checkpoints, iterate over every `.safetensors` file and build a
`key -> file` map instead of loading the full model into memory. Avoid
instantiating a large model with `AutoModel.from_config()` just to inspect key
names; it can allocate gigabytes unnecessarily.

## Output Verification

Every port must verify model output, not only graph construction and weight
loading. Full-model verification comes after isolated layer verification for
new or changed layer behavior. The verification target depends on
`target_mode`:

- For causal LM ports, run the Hugging Face model and Quick.AI with the same
  tokenizer, prompt, generation length, and deterministic decoding settings.
  Use greedy decoding unless the model requires a different deterministic
  setting. Record prompt text, input token ids, generated token ids, decoded
  text, EOS behavior, and any stop condition.
- For embedding ports, compare the Quick.AI embedding vector with the Hugging
  Face or Sentence Transformers reference vector for the same input. Record the
  shape, dtype, first few values, cosine similarity, max absolute difference,
  and mean absolute difference.
- For multimodal repositories implemented as text-only subsets, verify text
  output against the Hugging Face text path when possible. If the HF reference
  requires multimodal processor behavior that Quick.AI does not implement, the
  compatibility report must mark the port partial and explain why output parity
  cannot be claimed.

Use a small fixed sample that is cheap to run, such as a one-sentence prompt
and 8 to 32 generated tokens for causal LM. Disable sampling for the parity
check even if `generation_config.json` defaults to sampling. If the Quick.AI
CLI cannot expose token ids, record decoded text and the runtime log lines that
show generated text; prefer adding a narrow debug hook over relying on manual
visual inspection.

A causal LM parity check should compare at least one of these, in descending
preference:

- next-token logits or top-k token ids for the first generated step
- exact generated token ids for a short greedy decode
- exact decoded text for a short greedy decode

Exact text can still hide tokenizer or whitespace differences, so token ids are
preferred whenever the runtime can expose them. If exact token parity fails,
capture the first divergent step and classify the likely source: tensor order,
transpose, norm offset, RoPE, attention mask/cache, activation, tied output
head, logit transform, or intentionally unsupported HF behavior.

Do not accept "it prints plausible text" as validation. Plausible-but-different
text is a failed parity check unless the report explicitly documents the port
as partial and lists the unsupported semantics that make parity impossible.

## Subagent Roles

### HF Inspector

Prompt:

```text
You are the HF inspector for a Quick.AI model port.

Input: <hf_model>, <snapshot_dir>, <target_mode>, allow_remote_code=<true|false>.

Before inspecting model internals, verify the Python dependency bootstrap from
this document has passed. If imports fail, report the exact packages to install
and stop.

Inspect the downloaded snapshot or local model path. If the model has not been
downloaded yet, request that the Snapshot Downloader role run first and stop.
Report:
- architecture names from config["architectures"]
- required Transformers version and whether the installed version is sufficient
- model_type, hidden size, layer count, attention heads, KV heads, head_dim,
  rope settings, norm epsilon, vocab size, tied embedding flag, max positions
- nested sub-configs such as `text_config`, `vision_config`, or `audio_config`,
  and whether the requested target mode uses the top-level architecture or only
  a sub-config
- generation config tokens and defaults
- tokenizer files and required tokenizer path
- model shard/index layout and safetensors key prefixes from the local files
- representative safetensors tensor shapes from the local files
- whether custom modeling code is present; identify the exact
  `modeling_<model_name>.py`, other `modeling_*.py`, or installed Transformers
  modeling module used for inspection, and list the classes/functions that
  define layer behavior
- new or unusual modules, activations, attention variants, cache behavior, MoE
  routing, quantization wrappers, or multimodal components
- layer features that Quick.AI must add or extend for correctness

Do not implement code. End with either "compatible", "compatible with custom
layer feature", or "blocked". Use "blocked" only when the model cannot be
ported within the harness scope or cannot be inspected safely.
```

Expected output:

- A concise architecture summary.
- A list of tensor key prefixes grouped by embedding, attention, MLP, norm,
  head, and optional MoE experts.
- A compatibility status.

### Snapshot Downloader

Prompt:

```text
You are the Snapshot Downloader for a Quick.AI model port.

Input: <hf_model>, <snapshot_dir>, download_weights=<true|false>.

Verify the Python dependency bootstrap from this document has passed. Download
the target Hugging Face model into snapshot_dir using huggingface_hub, or verify
the supplied local model directory is complete. Include config, generation
config, tokenizer, processor, model index, safetensors, and custom modeling
source files when available.

Do not commit or move downloaded weights into the repository. Report the local
snapshot path, files present, total size if available, missing expected files,
and any authorization, gating, disk, or network issues.
```

Expected output:

- Local snapshot path.
- File inventory relevant to model porting.
- Whether tensor-key inspection and converter smoke tests can use this
  snapshot.

### Architecture Mapper

Prompt:

```text
You are the architecture mapper for a Quick.AI model port.

Input: HF inspector report and the current Quick.AI repo.

Map the model to the closest existing Quick.AI implementation:
- base CausalLM / Transformer
- Qwen2 or Qwen3 style attention
- Gemma3 style attention and norm ordering
- GPT-OSS or Qwen MoE
- slim/cached MoE variant
- Qwen/Gemma embedding path

Report the selected base pattern, required overrides, new or extended layer
features, and exact architecture strings that must be registered in the
factory.

If the target HF architecture is multimodal but the requested implementation is
text-only CausalLM, report both architecture strings separately:
- the HF top-level architecture that may need to resolve for user convenience
- the Quick.AI text-only architecture/class that will actually be instantiated

List all top-level modules intentionally excluded from the graph, such as
vision towers, audio towers, processors, or modality projection paths.

Do not implement code. Flag any mismatch that affects output correctness, and
classify it as either implementable layer work or a blocking runtime mismatch.
```

Expected output:

- Selected Quick.AI base class or nearest model family.
- Required C++ methods to override, such as `createAttention`,
  `createTransformerDecoderBlock`, `registerCustomLayers`, or embedding output
  construction.
- Factory architecture strings.

### Model Feature Planner

Prompt:

```text
You are the model feature planner for a Quick.AI model port.

Input: HF inspector report, architecture mapper report, local snapshot
inventory, requested target_mode, and the current Quick.AI repo.

Analyze the target model features before any coding starts. Convert the model
architecture findings into an ordered implementation plan that the lead agent
can assign to implementation roles.

Classify every relevant feature into one of these categories:
- already supported by an existing Quick.AI model or layer
- model-specific custom layer to implement under models/<family>/
- reusable shared layer feature to implement or extend under layers/
- model-specific graph behavior to implement under models/<family>/
- converter-only tensor transform or tensor ordering work
- runtime asset or integration work
- verification work required before full-model output parity
- intentionally excluded behavior for a partial or text-only port
- blocking runtime gap outside the current harness scope

Prefer model-specific custom layers under `models/<family>/` for newly added
target-model behavior. Choose shared `layers/` changes only when the behavior
is demonstrably reusable, does not encode one model family's assumptions, and
can preserve existing model behavior.

For every new model-specific or reusable layer feature, identify:
- the closest existing layer or custom layer to extend
- whether the implementation belongs under `models/<family>/` or shared
  `layers/`, with the reason
- the HF config fields and tensor keys that drive the behavior
- the Python source class/function from `modeling_<model_name>.py` or the
  installed Transformers module that defines the behavior
- the C++ property or constructor wiring that should expose it
- the smallest partial-layer parity test required before full-model validation
- backward-compatibility risks for existing model families

For model-specific graph behavior, identify:
- the base class to derive from
- methods that need overrides
- layer order changes and shape-sensitive constants
- tensor order implications for the converter

For converter work, identify:
- exact tensor groups that need transpose, reshape, duplication, omission, or
  tied-weight handling
- whether direct safetensors reads are required to avoid loading the full model
- failure conditions that must be explicit instead of silently falling back

Do not implement code. If a feature is unclear, name the exact snapshot file,
config field, modeling source class, or tensor key that must be inspected next.
End with an implementation sequence that minimizes risk: layer features first,
partial layer verification against Python second, model graph third, converter
fourth, integration fifth, full-model output parity sixth.
```

Expected output:

- Feature classification table.
- Ordered implementation plan with dependencies between tasks.
- Required layer-feature parity checks and tolerances to define.
- Converter tensor-order risks.
- Blocking or intentionally excluded behavior, if any.

### Layer Feature Implementer

Prompt:

```text
You are the layer feature implementer for a Quick.AI model port.

Input: HF inspector report, architecture mapper report, model feature planner
report, NNTrainer graph design, and the current Quick.AI repo.

Implement Quick.AI layer features needed by the target model when no existing
layer exactly matches the Hugging Face behavior. For behavior that is specific
to this model family, implement a new custom layer under `models/<family>/` and
register it from that model. Modify shared `layers/` only when the model
feature planner classified the behavior as reusable shared layer work. Examples
include RoPE variants, partial rotary dimensions, attention softcapping, final
logit softcapping, normalization shape variants, activation variants, MoE
routing details, cache semantics, and attention-mask/window behavior.

Use the downloaded `modeling_<model_name>.py` or the recorded Transformers
modeling source as the reference for equations, tensor reshapes, mask semantics,
and edge cases.

Update layer properties, serialization/export behavior, model graph property
wiring, and focused tests or smoke-test hooks where the repo has a suitable
place. Preserve existing behavior for already supported models.

Report changed files, new properties, compatibility risks, and validation
commands.
```

Expected output:

- Implemented layer feature summary.
- Files changed.
- Backward-compatibility notes.
- Validation commands and results.

### Partial Layer Verification Agent

Prompt:

```text
You are the partial layer verification agent for a Quick.AI model port.

Input: HF inspector report, model feature planner report, layer feature
implementation summary, NNTrainer graph design, and the current Quick.AI repo.

For each new or changed layer feature, build a minimal parity check that
compares Quick.AI/NNTrainer output with a Python reference output. Use the
Hugging Face model implementation from the local snapshot, especially
`modeling_<model_name>.py`, when the feature is implemented there; otherwise
write a small standalone PyTorch reference that matches the HF equations and
config fields.

Verify features independently before full-model generation. Start with one
representative layer or feature path and prove it matches Python before running
the whole model; then add additional isolated checks for other new features.
Examples include RoPE variants, partial rotary dimensions, sliding or causal
attention masks, attention softcapping, final logit softcapping, RMSNorm offset
handling, activation functions, Q/K normalization, MoE routing, shared-KV cache
behavior, and model-specific residual side paths.

Use deterministic fixed inputs and weights. Prefer tiny tensor shapes that make
failures easy to inspect, but include at least one shape from the target model
when shape-dependent behavior is the risk. Save or print:
- config values used by the test
- input tensor shape, dtype, seed, and representative values
- Python reference output summary
- Quick.AI/NNTrainer output summary
- max absolute difference, mean absolute difference, and tolerance
- first mismatching index and values when the tolerance fails

Do not treat full-model plausible text as layer verification. If Quick.AI does
not expose an executable path for the layer in isolation, define the smallest
temporary harness or debug hook needed to run the layer and document whether it
should be kept, converted into a regression test, or removed before merge.
```

Expected output:

- Per-feature parity status against Python output.
- Commands or harness entry points used to produce both outputs.
- Tolerances and numeric error summary.
- First-difference diagnostics for failures.
- Recommendation: keep as regression coverage, remove temporary harness, or
  block the port until the layer feature matches.

### NNTrainer Graph Designer

Prompt:

```text
You are the NNTrainer graph designer for a Quick.AI model port.

Input: HF inspector report, architecture mapper report, model feature planner
report, and existing Quick.AI model/layer patterns.

Design the C++ graph:
- class names and inheritance
- layer order for embedding, decoder blocks, attention, MLP, norms, output norm,
  LM head or embedding head
- exact NNTrainer layer types and key properties
- custom layer registration requirements
- required new or extended layer features and the properties that will expose
  them
- shape-sensitive constants from config
- any sub-config sanitization required before calling shared base constructors,
  such as replacing a top-level multimodal config with `text_config` or removing
  null config fields that Quick.AI base setup expects to be booleans
- layer-specific dimensional changes, such as full-attention layers using a
  different head dimension from sliding-attention layers
- unsupported-but-bypassed semantics that the graph intentionally does not
  model, such as per-layer embedding side paths or cross-layer shared K/V

Return the C++ implementation outline and the required tensor load order. Do
not write files.
```

Expected output:

- C++ graph outline.
- Ordered tensor list that the converter must write.
- Custom-layer requirements.
- Layer-feature implementation requirements.

### Weight Converter

Prompt:

```text
You are the weight converter author for a Quick.AI model port.

Input: HF inspector report, model feature planner report, and NNTrainer graph
tensor order.

Design `weight_converter.py`:
- use AutoConfig and AutoModel or AutoModelForCausalLM as appropriate
- use trust_remote_code only when the inspector approved custom code
- save tensors in exactly the C++ graph load order
- inspect tensor keys from the downloaded snapshot, not only from remote
  metadata
- prefer direct safetensors reads with a `key -> file` map for large models; do
  not instantiate the full HF model when key and tensor inspection is enough
- document all transpose rules
- handle tied versus untied output weights
- handle model-specific norm offsets, q/k norms, MoE experts, routers, and LoRA
  wrappers when present
- handle checkpoint-vs-graph mismatches explicitly. If the Quick.AI graph
  intentionally duplicates or ignores tensors to make a text-only subset
  loadable, document that as a parity gap in the report.
- expose --model_path, --output_name, and --data_type arguments

Return converter code structure and any parity checks that should be run.
```

Expected output:

- Ordered save plan matching the NNTrainer graph.
- Converter arguments and default output name.
- Any model-specific tensor transforms.
- Reference-output prerequisites, such as whether token ids, logits, or
  embeddings can be compared without loading the full model into memory.

### Integration Agent

Prompt:

```text
You are the Quick.AI integration agent for a model port.

Input: model feature planner report, selected family name, architecture
strings, C++ class names, and runtime asset plan.

Plan integration edits:
- models/<family>/meson.build
- models/meson.build
- includes and factory registration in main.cpp
- includes and factory registration in quantize.cpp
- includes and factory registration in api/causal_lm_api.cpp
- res/<family>/<model>/nntr_config.json
- model README notes if the model has unusual runtime requirements
- narrow `.gitignore` exceptions when the repository ignores `res/*`, so the
  intended `nntr_config.json` and `weight_converter.py` are reviewable while
  model weights remain ignored

Keep the changes consistent with existing repo style.
```

Expected output:

- Exact integration checklist.
- Public architecture strings and class names.
- Default `nntr_config.json` values.

### Verification Agent

Prompt:

```text
You are the verification agent for a Quick.AI model port.

Input: model feature planner report, generated patch summary, and model asset
paths.

Define and run, when possible:
- Python dependency bootstrap and package version report
- formatting checks for changed C/C++ files
- Linux Meson build
- converter syntax check
- converter smoke test
- quantization smoke test when requested
- partial layer verification before full-model validation: start with one
  representative newly implemented or changed layer, compare it against the
  Python output from `modeling_<model_name>.py` or an equivalent PyTorch
  reference, then cover every remaining new or changed layer feature
- Hugging Face reference-output generation from the same local snapshot and
  converted-weight Quick.AI output comparison
- causal LM inference smoke test or embedding API smoke test
- process cleanup checks for long-running download, conversion, or runtime smoke
  commands

Report commands, pass/fail status, missing prerequisites, and residual risk.
For output verification, include the prompt or input, deterministic decoding
settings, reference tokens/text or embedding summary, Quick.AI tokens/text or
embedding summary, the comparison result, and the first divergent token or
largest embedding error when available.
```

Expected output:

- Commands run.
- Artifacts produced.
- Partial layer parity results for changed layer features, clearly marked as
  completed before or blocking full-model parity.
- Output parity result, or a clearly labeled partial-validation result with the
  missing prerequisite or unsupported HF behavior.
- Any validation that remains manual.

## Implementation Checklist

Use this checklist when applying a model port generated by the harness.

### C++ Model

- Add `models/<family>/<family>_causallm.h` and `.cpp` for causal LM, or the
  embedding equivalent for embedding models.
- Derive from the closest existing Quick.AI base class.
- Override only the graph pieces that differ from the base implementation.
- Register every custom layer in `registerCustomLayers`.
- Implement and register any new model-specific custom layer required by the
  target model before relying on it in the model graph. Use shared reusable
  layer features only when the planner explicitly chose `layers/`.
- Keep layer names stable and deterministic, because the binary weight file is
  loaded in graph order.
- If the model uses a nested HF config, sanitize it before shared Quick.AI base
  setup reads mandatory fields. Remove or normalize null fields when existing
  base setup expects concrete booleans or numbers.
- If a text-only subset is implemented for a multimodal architecture, register a
  Quick.AI text-only architecture string and, only when intentional, map the HF
  top-level architecture to that text-only class.

### Layer Features

- Add or extend layer properties for target-model behavior that differs from
  existing Quick.AI semantics.
- Keep default property values backward compatible for existing model families.
- Prefer adding a model-specific custom layer under `models/<family>/` for
  newly added behavior that only the target family needs.
- Extend a general layer such as `mha_core`, `rms_norm`, or `lm_head` only when
  the behavior is reusable across model families and backward compatibility is
  covered.
- Add a dedicated custom layer when the feature cannot be represented by
  existing layer abstractions without changing shared semantics.
- Document the HF config field that drives each new property.
- Document the Python source class or function from `modeling_<model_name>.py`
  or the installed Transformers module that the C++ layer follows.
- Verify at least one representative new layer against Python output before
  running full-model parity.

### Build Wiring

- Add `models/<family>/meson.build` with source files and include directory.
- Add `subdir('<family>')` in `models/meson.build`.
- Add includes and factory entries in:
  - `main.cpp`
  - `quantize.cpp`
  - `api/causal_lm_api.cpp`
- For embedding models, update architecture resolution if an HF architecture
  must map to a Quick.AI embedding class name.

### Runtime Assets

- Add `res/<family>/<model>/nntr_config.json`.
- Add `res/<family>/<model>/weight_converter.py`.
- If `res/*` is ignored, add narrow `.gitignore` exceptions for only the new
  config and converter files. Do not unignore or commit downloaded weights.
- Set `model_type` to `CausalLM` or `Embedding`.
- Set `model_file_name` to the converter output file name.
- Set `tokenizer_file` to the expected tokenizer JSON path for the model
  directory used in smoke tests.
- Include a `sample_input` that exercises the model without requiring private
  data.

### Converter

- Read HF weights with `AutoConfig` and the matching `AutoModel*` class.
- Use `trust_remote_code=True` only when the compatibility report says custom
  code is expected.
- For large safetensors checkpoints, read tensors directly with `safe_open`
  rather than instantiating the full HF model.
- Save projection matrices transposed when the matching NNTrainer
  `fully_connected` layer expects input-major storage.
- Save norm weights with any required model-specific offset, such as Gemma-style
  RMSNorm `weight + 1.0`.
- Save LM head weights only when untied; otherwise rely on the shared embedding
  path used by the C++ graph.
- Write generated `.bin` artifacts into the local `res/<family>/<model>`
  validation directory when the runtime config points there. Never commit
  generated `.bin`, safetensors, downloaded snapshots, or Hugging Face cache
  files unless the task explicitly asks for a local runtime artifact.
- Keep the converter deterministic and fail fast on missing tensors. Silent
  tensor duplication, fallback to unrelated keys, or ignored HF tensors must be
  documented as parity gaps and reflected in output verification.

## Acceptance Criteria

A model port is acceptable when all applicable criteria are met:

- The HF architecture string resolves to a registered Quick.AI factory entry.
- The target model snapshot has been downloaded or a complete local model
  directory has been verified.
- The relevant Hugging Face modeling source, such as
  `modeling_<model_name>.py`, has been downloaded or identified from the
  installed Transformers package and used for the graph/layer mapping.
- Any newly required target-model layer feature is implemented, wired through
  graph properties, and documented in the compatibility report, with
  target-specific behavior kept under `models/<family>/` unless shared
  `layers/` changes are justified.
- Every newly implemented or changed layer feature has partial layer
  verification against Python reference output, or the compatibility report
  explains why the feature cannot be isolated and what full-model comparison
  covers instead.
- `ninja -C build` succeeds.
- The converter creates the configured `.bin` file from a local HF model
  directory.
- The C++ graph tensor load order matches the converter save order.
- Causal LM output verification passes against the Hugging Face reference for
  the accepted behavior using first-step logits/top-k, generated token ids, or
  decoded greedy text for a short deterministic prompt.
- Embedding output verification passes against the Hugging Face reference for
  the accepted behavior, with matching shape and recorded cosine similarity /
  absolute-error thresholds that are acceptable for the chosen dtype.
- `quick_dot_ai_run` can initialize the model directory for causal LM models,
  but initialization alone is only a smoke test and is not parity validation.
- `quick_dot_ai_test_api` covers API-facing embedding or causal LM changes.
- The compatibility report has no unresolved blocking items.
- Any accepted text-only subset of a larger HF architecture documents all
  excluded modules and parity gaps. The report status should say "text-only" or
  "partial" instead of simply "compatible" unless HF parity was validated for
  the accepted subset.
- No weights, credentials, or generated build artifacts are committed.

## Default Validation Commands

From the repository root:

```bash
python3 - <<'PY'
import numpy, torch, transformers, safetensors, huggingface_hub, accelerate
print("torch", torch.__version__)
print("transformers", transformers.__version__)
PY
python3 - <<'PY'
from huggingface_hub import snapshot_download
print(snapshot_download("<hf_repo_id>",
                        local_dir="res/<family>/<model>",
                        local_dir_use_symlinks=False))
PY
meson setup build
ninja -C build
python3 res/<family>/<model>/weight_converter.py \
  --model_path res/<family>/<model> \
  --output_name res/<family>/<model>/<model_file_name>
./build/quick_dot_ai_run res/<family>/<model> "<sample prompt>"
./build/quick_dot_ai_test_api
```

Before accepting the runtime result, generate a deterministic Hugging Face
reference for the same prompt and save it as an artifact:

```bash
python3 - <<'PY'
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_dir = "res/<family>/<model>"
prompt = "<sample prompt>"
output_path = "res/<family>/<model>/hf_reference_output.json"

tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    torch_dtype=torch.float32,
    trust_remote_code=True,
).eval()

inputs = tokenizer(prompt, return_tensors="pt")
with torch.no_grad():
    output = model.generate(
        **inputs,
        do_sample=False,
        max_new_tokens=16,
        return_dict_in_generate=True,
        output_scores=True,
    )

record = {
    "prompt": prompt,
    "input_ids": inputs["input_ids"][0].tolist(),
    "generated_ids": output.sequences[0].tolist(),
    "decoded": tokenizer.decode(output.sequences[0], skip_special_tokens=False),
}
if output.scores:
    top = torch.topk(output.scores[0][0], k=5)
    record["first_step_top5"] = list(zip(top.indices.tolist(), top.values.tolist()))

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(record, f, indent=2)

print(output_path)
PY
```

Then capture the Quick.AI runtime output and compare it with the saved
reference. Prefer token ids or first-step top-k comparison when available;
otherwise compare the deterministic decoded text and record the limitation:

```bash
./build/quick_dot_ai_run res/<family>/<model> "<sample prompt>" \
  | tee res/<family>/<model>/quick_ai_runtime_output.txt
```

For models too large to load as FP32 on the validation machine, use the smallest
available dtype/device that preserves the intended comparison and record that
choice. If no Hugging Face reference run is possible, the report must say
output parity is unverified and explain the blocker.

Run the converter syntax check before a full conversion:

```bash
python3 -m py_compile res/<family>/<model>/weight_converter.py
```

For large models, use the local `res/<family>/<model>` runtime directory for
validation, but keep generated weights and downloaded checkpoints ignored. This
verifies initialization without committing generated weights.

For quantization:

```bash
./build/quick_dot_ai_quantize res/<family>/<model>
```

For Android-facing ports, also run the existing Android build and device smoke
test when `ANDROID_NDK` and a device are available:

```bash
./build_test_app.sh
./run_test.sh
```

## Compatibility Report Template

```markdown
# Compatibility Report: <hf_model>

## Status
compatible | compatible with custom layer feature | text-only partial | blocked

## HF Architecture
- architectures:
- model_type:
- target_mode:
- modeling source reviewed:
- top-level architecture mapped to text-only class: yes/no
- hidden_size:
- num_hidden_layers:
- num_attention_heads:
- num_key_value_heads:
- head_dim:
- max_position_embeddings:
- rope_theta:
- rms_norm_eps:
- vocab_size:
- tie_word_embeddings:

## Tokenizer And Generation
- tokenizer files:
- bos/eos/pad ids:
- sample input:

## Tensor Map
- local snapshot path:
- safetensors files and total size:
- tensor key prefix:
- embedding:
- attention:
- mlp:
- norms:
- output:
- moe or custom:

## Quick.AI Mapping
- base pattern:
- new classes:
- architecture strings:
- custom layers:
- model-specific layers under models/<family>/:
- new or extended layer features:
- shared layers/ changes:
- intentionally excluded modules:
- known parity gaps:

## Implementation Plan
- already-supported features:
- model-specific custom layer work:
- reusable shared layer-feature work:
- model-specific graph work:
- converter work:
- integration work:
- partial-layer verification tasks:
- output-parity tasks:
- task order and dependencies:

## Blocking Issues
- none, or list only issues that cannot be implemented within this port

## Validation
- snapshot download:
- safetensors key/shape inspection:
- build:
- converter syntax:
- converter:
- partial layer verification:
- Hugging Face reference output:
- Quick.AI runtime output:
- output comparison:
- generated artifacts and locations:
- remaining manual checks:
```

## Patch Summary Template

```markdown
# Model Port Patch: <family>/<model>

## Summary
- Added Quick.AI support for <HF architecture>.
- Target mode: causal_lm | embedding.
- Base pattern: <existing Quick.AI family or base class>.
- Implementation plan source: model feature planner report.

## Public Registration
- Factory architecture string:
- C++ class:
- Runtime asset directory:

## Files Changed
- models/<family>/...
- res/<family>/<model>/...
- main.cpp / quantize.cpp / api/causal_lm_api.cpp

## Validation
- [ ] clang-format on changed C/C++ files
- [ ] meson setup build
- [ ] ninja -C build
- [ ] converter smoke test
- [ ] partial layer verification against Python reference output
- [ ] Hugging Face reference-output capture
- [ ] Quick.AI output comparison
- [ ] quick_dot_ai_run or quick_dot_ai_test_api

## Risks
- Tensor-order risks:
- Newly implemented layer-feature risks:
- Unsupported HF behavior:
- Planner assumptions that still need verification:
- Output-parity risks:
- Performance or memory notes:
```
