# Apple M4 SLM Local-Answer Baseline

The completed `apple-m4-slm-answer` campaign makes the practical Mac baseline explicit:

```text
prompt in
-> validated small dense instruct GGUF
-> Rust CLI
-> apple-m4-cpu-neon
-> strict loader and tokenizer routing
-> warm-session answer receipts
-> intelligible text out
```

This is the dense SLM user-facing Mac path. BitNet has a separate explicit
one-shot ask route after the strict answer-corpus proof, and remains separate
from dense SLM chat/server and future Metal acceleration work.

For the broader validation graph that separates dense SLM UX evidence from
BitNet / 1-bit model evidence, see
[Reference Topology](../architecture/reference-topology.md).

## Supported Baseline

| Field | Current baseline |
|---|---|
| Model family | Qwen2.5 0.5B Instruct GGUF |
| Proof artifact | Q8_0 companion used by the Rust-native warm-session proof |
| Backend label | `apple-m4-cpu-neon` |
| Runtime API | `cpu` |
| Fallback policy | `fallback_used=false` must be recorded |
| Prompt template | `qwen2.5` |
| Execution mode | Warm session: model and tokenizer load once, then multiple prompts run |

The proof artifact remains a local file under `target/` today:

```text
target/apple-m4-slm-answer/SLM-M4-003/candidates/qwen2_5_0_5b_q8_0/qwen2.5-0.5b-instruct-q8_0.gguf
```

Do not commit model binaries. Productized model cache commands are tracked as `M4-PROD-002`.

## Model Cache

This Mac path uses Qwen2.5 0.5B Instruct as a dense regular SLM. It is the Apple Silicon user-facing local-answer baseline for CLI, cache, receipt, warm-session, and quality harness behavior. It is not a BitNet substitute and does not prove 1-bit / 1.58-bit model quality, I2_S/TL1/TL2 kernel paths, QK256, or Apple BitNet inference.

`M4-PROD-002` adds a user cache for supported SLM artifacts. By default it uses:

```text
~/.cache/bitnet-rs/models/
```

The cache root can be overridden with `BITNET_MODEL_CACHE_DIR` or `--cache-dir`.

Supported model IDs:

```text
qwen2.5-0.5b-instruct-q8_0    Rust-native Apple M4 CPU/NEON baseline artifact
qwen2.5-0.5b-instruct-q4_k_m  Rust-native Apple M4 CPU/NEON storage-conscious artifact
qwen2.5-1.5b-instruct-q4_k_m  Rust-native Apple M4 CPU/NEON larger Qwen artifact; explicit non-default
```

Useful commands:

```bash
bitnet mac models
bitnet model list
bitnet model fetch qwen2.5-0.5b-instruct-q8_0
bitnet model verify qwen2.5-0.5b-instruct-q8_0
bitnet model prune qwen2.5-0.5b-instruct-q8_0
```

The default remains `qwen2.5-0.5b-instruct-q8_0`. Select either non-default
model explicitly with `--model-id` on Mac commands after fetching and verifying
that artifact:

```bash
bitnet model fetch qwen2.5-1.5b-instruct-q4_k_m
bitnet mac check --model-id qwen2.5-1.5b-instruct-q4_k_m
bitnet mac ask "What is 2+2? Answer briefly." \
  --model-id qwen2.5-1.5b-instruct-q4_k_m
```

Cache metadata records source repository, revision, filename, SHA256, size,
quantization, tokenizer metadata, chat-template presence, and Apple M4 CPU/NEON
support status. Fetch warns on low disk headroom and honors `--offline` /
`BITNET_OFFLINE`.
`bitnet mac models` is the Mac operator view for default, supported,
blocked, candidate, and rejected model states plus disk-headroom guidance for
first fetches. Its text output prints exact `Next fetch` and `Next verify`
commands for the recommended first supported model when disk headroom is
adequate. The BitNet row also prints receipt-only `bitnet mac bitnet-proof`
and fixed-prompt `bitnet mac bitnet-warm` bridge commands so operators can
validate the strict BitNet answer-corpus proof and the warm-session receipt.
BitNet is limited to explicit one-shot `bitnet mac ask` plus fixed-prompt
`bitnet mac bitnet-warm` with verified GGUF/tokenizer authority;
`M4-BITNET-ASK-001` adds one committed user-route receipt at
`ci/hardware/apple-m4-mac-mini/2026-05-13/bitnet-mac-ask/bitnet-mac-ask-runtime-receipt.json`.
It is not an enabled Mac chat/server route. `bitnet model list` is the
lower-level cache inventory; use `bitnet model verify <id>` or `bitnet mac
check` when SHA integrity matters.
Dense Mac answer/service wrappers reject diagnostic-only, candidate, rejected,
and unknown model IDs before cache repair guidance, so operators are not pointed
at a fetch command for a non-selectable M4 answer model.

First-run and repair guidance is intentionally explicit:

```bash
# Missing cache
bitnet model fetch qwen2.5-0.5b-instruct-q8_0

# Present artifact without cache metadata
bitnet model verify qwen2.5-0.5b-instruct-q8_0

# Partial, wrong-size, or wrong-hash artifact
bitnet model prune qwen2.5-0.5b-instruct-q8_0
bitnet model fetch qwen2.5-0.5b-instruct-q8_0
```

When a Mac wrapper finds a missing model cache, the failure includes both the
exact `bitnet model fetch <id>` repair command and a
`bitnet mac models --cache-dir ...` command with disk guidance. Operators should
use the model view first on low-space machines before choosing between the
default Q8_0 model and storage-conscious Q4_K_M support.

Offline mode cannot repair a missing or corrupt cache. Pre-seed the GGUF file and run `bitnet model verify qwen2.5-0.5b-instruct-q8_0`, or disable offline mode and fetch the supported artifact. On low-disk systems, prune unused cached models or set `BITNET_MODEL_CACHE_DIR` / `--cache-dir` to a larger volume before fetching.

## Working Commands

Fetch and verify the supported runtime artifact once:

```bash
bitnet model fetch qwen2.5-0.5b-instruct-q8_0
bitnet mac check
```

Ask one question through the supported Mac wrapper:

```bash
bitnet mac ask "What is 2+2? Answer briefly." \
  --json-out target/apple-m4-productization/mac-ask.json
```

Before generation, `bitnet mac ask` prints a compact operator summary on stderr
with the selected model ID, quantization, verified cache root, backend,
fallback status, receipt path, and short model SHA. The summary is metadata
only; the strict answer receipt remains the authority for proof.

The older flag form is kept for scripts:

```bash
bitnet mac ask \
  --question "What is 2+2? Answer briefly." \
  --json-out target/apple-m4-productization/mac-ask.json
```

Run multiple prompts through one resident Mac session:

```bash
bitnet mac chat \
  --prompt "What is 2+2? Answer briefly." \
  --prompt "Name the capital of France." \
  --json-out target/apple-m4-continuity/mac-chat.json
```

`bitnet mac chat` is a non-interactive resident session wrapper over the same
supported Apple M4 CPU/NEON dense-SLM warm-session path used by validation. It
loads the verified Qwen2.5 model and tokenizer once, streams token text by
default, writes an aggregate receipt plus per-prompt receipts, and keeps the
same dense-SLM claim boundary as `bitnet mac ask`. For one question, continue to
use `bitnet mac ask`.

For terminal use, collect a resident batch interactively and finish with
`/exit`, `/quit`, or EOF:

```bash
bitnet mac chat --interactive \
  --json-out target/apple-m4-continuity/mac-chat.json
```

Interactive mode still writes the aggregate receipt when the collected prompts
finish. Per-turn receipt files are enabled by default and can be disabled with
`--no-turn-receipts` when the operator only wants the aggregate receipt. Use
`--progress` to print stderr status lines such as model/tokenizer loaded-once
state; quiet default output keeps stdout focused on streamed token text.

Run the compact M4 health smoke:

```bash
bitnet mac smoke \
  --json-out target/apple-m4-continuity/mac-smoke.json
```

`bitnet mac smoke` verifies the supported dense-SLM cache, runs a fixed tiny
`2+2` prompt through `apple-m4-cpu-neon`, validates the generated answer
receipt, writes an aggregate golden-smoke receipt with backend/fallback identity
and disk/cache health, and keeps the same dense-SLM-only claim boundary. It is a
local appliance check, not a BitNet, Metal, Neural Engine, QK256, or broad
performance proof. The cache health receipt records `verification_passes=1` so
the smoke avoids hashing the same cached model twice before the tiny generation.

Run the one-command M4 health verdict:

```bash
bitnet mac doctor \
  --json-out target/apple-m4-slm-excellence/mac-doctor.json
```

`bitnet mac doctor` wraps the supported dense-SLM health checks into one local
operator verdict. By default it verifies the model cache and hash, checks disk
headroom, confirms `apple-m4-cpu-neon` with `fallback_used=false`, and verifies
that full `apple-m4-metal` inference remains rejected for the dense Mac wrapper
without running live generation. Pass `--run-smoke` when the operator explicitly
wants the compact smoke path and generated receipt validation. It does not
download models by default and does not make a BitNet, full Metal, Neural
Engine, QK256, or broad performance claim.

Run the deterministic warm-session validation corpus:

```bash
bitnet mac validate \
  --json-out target/apple-m4-productization/mac-validate.json
```

Run the operator timing profile set:

```bash
bitnet mac validate \
  --profile-set operator \
  --json-out target/apple-m4-productization/mac-operator-profiles.json
```

This writes a summary receipt plus per-profile warm-session receipts for
`warm_16`, `warm_32`, and `warm_64`. These profiles record cold model/tokenizer
load separately from warm prompt timing, show model/tokenizer reuse within each
profile, and keep latency numbers scoped to this model, backend, prompt set, and
machine context. The operator profile set intentionally runs one warm session per
token budget, so reuse is `within_profile`, not a single shared process across
all three budgets. The summary records `profile_set_model_loads=3` and
`profiles_loaded_independently=true` to keep that scope visible. These are not
broad performance or speedup claims.

Check answer or warm-session receipts:

```bash
bitnet mac receipts-check target/apple-m4-productization/mac-validate.json
bitnet mac receipts-check target/apple-m4-productization/mac-operator-profiles.json
```

The lower-level warm-session command remains available for debugging:

```bash
RUST_LOG=warn cargo run --locked -p bitnet-cli \
  --no-default-features --features cpu,full-cli -- \
  --device apple-m4-cpu-neon \
  slm-warm-session \
  --model target/apple-m4-slm-answer/SLM-M4-003/candidates/qwen2_5_0_5b_q8_0/qwen2.5-0.5b-instruct-q8_0.gguf \
  --corpus ci/quality/apple-m4-slm-quality-corpus.yaml \
  --strict-loader \
  --strict-tokenizer \
  --fail-on-quality \
  --require-determinism \
  --json-out target/apple-m4-productization/M4-PROD-001/slm-local-answer-baseline.json
```

Expected receipt properties:

```text
requested_backend = apple-m4-cpu-neon
selected_backend = apple-m4-cpu-neon
runtime_api = cpu
fallback_used = false
model_loaded_once = true
tokenizer_loaded_once = true
generated text and token IDs present
quality_summary.passed = true
determinism.passed = true
timing separates load, tokenize, prefill, decode, sampling, and total time
operator profile summaries include warm_16, warm_32, and warm_64 when requested
operator profile summaries disclose one warm session per token budget
operator profile summaries record profile_set_model_loads = 3
broad_performance_claim = false
speedup_claim = false
```

`bitnet mac ask` and `bitnet mac validate` intentionally route to
`apple-m4-cpu-neon`. Passing `--device apple-m4-metal`, `apple-m4-mpsgraph`, or
another accelerator label is rejected because full Metal/MPSGraph model
inference is not a proven user-facing path yet.

## Failure Boundaries

The Mac baseline must fail clearly when:

- the model file is missing;
- the model hash does not match the supported artifact manifest;
- strict loader or strict tokenizer mode would fall back;
- `selected_backend` differs from `requested_backend` without `fallback_reason`;
- `apple-m4-metal` is requested for full inference before a strict receipt proves it;
- MPSGraph output is counted as Neural Engine execution;
- QK256 support is inferred from SLM evidence.

## Claim Boundary

This path may claim:

```text
Rust-native Apple M4 CPU/NEON SLM local answers work for the validated model, corpus, backend, and receipt settings.
```

It must not claim:

```text
BitNet local-answer quality
full apple-m4-metal model inference
Neural Engine execution
QK256 on Apple Silicon
general M4 performance
```

Warm timing is measured for the recorded machine, model, corpus, backend, and run settings only. Broad performance claims require a later campaign item.
