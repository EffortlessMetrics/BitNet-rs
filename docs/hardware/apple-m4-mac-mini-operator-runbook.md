# Apple M4 Mac mini Operator Runbook

## Purpose

This runbook is for a local macOS operator who wants to validate the Apple M4
BitNet-rs path and run the receipt-backed local-answer path without hidden
fallbacks or hardware overclaims.

The current reliable user-facing path is `apple-m4-cpu-neon`. Native Metal and
MPSGraph evidence exists only where receipts say so. Do not treat Metal probe,
tiny Metal smoke, Metal I2_S parity, or MPSGraph smoke as full model inference.

## Supported Backend Labels

| Label | Meaning | Current operator claim |
|---|---|---|
| `apple-m4-cpu-neon` | Apple ARM64 CPU path used for strict BitNet proof, local answers, profile receipts, and CPU reference behavior. | Reliable local-answer path when strict real-model receipt passes. |
| `apple-m4-metal` | Native Metal proof or phase/subgraph path where receipt-backed. | Metal visibility, tiny compute, I2_S parity, and phase proof only unless a strict full-model Metal receipt later proves more. |
| `apple-m4-mpsgraph` | MPSGraph graph/reference lane. | Graph/reference evidence only; not native Metal proof and not Neural Engine proof. |

CPU fallback cannot count as Metal execution. MPSGraph cannot count as native
Metal or Neural Engine execution unless a receipt records the resolved target and
a campaign item explicitly allows that claim.

## Model Placement

Use the canonical BitNet GGUF used by the Apple receipts:

```text
repo: microsoft/bitnet-b1.58-2B-4T-gguf
file: ggml-model-i2_s.gguf
tokenizer: llama3
```

Place the model under the repo-local model directory:

```text
models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf
```

If you use Hugging Face tooling, one practical shape is:

```bash
huggingface-cli download microsoft/bitnet-b1.58-2B-4T-gguf \
  ggml-model-i2_s.gguf \
  --local-dir models/BitNet-b1.58-2B-4T
```

If tokenizer artifacts are distributed separately, keep them beside the model.
Strict tokenizer mode fails rather than silently accepting a mock or missing
tokenizer.

## One-Command Validation

Run the complete Apple M4 validation bundle from the repository root:

```bash
cargo run --locked -p xtask --no-default-features -- apple-m4 validate \
  --date 2026-05-07 \
  --model models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf \
  --out ci/hardware/apple-m4-mac-mini/2026-05-07
```

`--out-dir` is accepted as an alias for `--out`.

Use a current date for new local runs. For scratch validation that should not be
committed, write under `target/` instead:

```bash
cargo run --locked -p xtask --no-default-features -- apple-m4 validate \
  --date 2026-05-07 \
  --model models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf \
  --out target/apple-m4-validation-2026-05-07
```

The validation command runs the known-good proof sequence and then validates the
bundle. A successful run prints:

```text
Apple M4 validation receipts passed: <out>/summary.json
```

It does not prove full `apple-m4-metal` model inference.

## Expected Receipt Bundle

The validation output directory should contain:

```text
machine-profile.json
metal-probe.json
metal-smoke.json
metal-i2s-parity.json
mpsgraph-smoke.json
strict-bitnet-cpu-neon-proof.json
phase-profile.json
allocation-audit.json
summary.json
```

`summary.json` must state both what is proven and what is not proven. The current
bundle may prove:

- Apple M4 machine facts were recorded.
- Metal runtime visibility was recorded.
- Tiny native Metal compute smoke passed with `fallback_used = false`.
- I2_S-adjacent Metal parity matched the CPU/NEON reference with
  `fallback_used = false`.
- Tiny MPSGraph reference smoke ran as graph/reference evidence.
- Strict BitNet CPU/NEON proof passed for the selected Apple CPU backend.
- CPU/NEON profile and allocation receipts were emitted for the recorded
  profile.

The current bundle must not claim:

- full `apple-m4-metal` model inference;
- QK256 on Apple Silicon;
- Neural Engine execution;
- general M4 performance;
- MPSGraph as native Metal kernel proof.

## Receipt-Bundle Check

Validate an existing bundle with:

```bash
cargo run --locked -p xtask --no-default-features -- apple-m4 receipts-check \
  ci/hardware/apple-m4-mac-mini/2026-05-07
```

For machine-readable output:

```bash
cargo run --locked -p xtask --no-default-features -- apple-m4 receipts-check \
  ci/hardware/apple-m4-mac-mini/2026-05-07 \
  --json
```

The checker fails if:

- `fallback_used` is missing;
- `fallback_used = true` has no non-empty `fallback_reason`;
- `selected_backend` differs from `requested_backend` without a fallback reason;
- a Metal proof selected CPU fallback;
- an MPSGraph receipt claims Neural Engine without separate proof;
- an MPSGraph receipt claims native Metal proof;
- BitNet receipts are missing `model.tokenizer` or `tokenizer`;
- BitNet receipts are missing `bitnet.kernel_family` or
  `bitnet.execution_phase`;
- Apple receipts claim QK256 before an Apple QK256 item explicitly supports it;
- summary or receipt text claims full Metal model inference, Neural Engine
  execution, QK256, or broad performance outside the `not_proven` section.

Successful text output starts with:

```text
Apple M4 receipt bundle passed: <dir>
```

and then lists each expected receipt as `pass`.

## Strict Local CPU/NEON Answer Path

For local answers today, use the Apple CPU/NEON path. The CLI device selector is
a global option, so place it before the `run` subcommand:

```bash
BITNET_DISABLE_MINIMAL_LOADER=1 \
BITNET_STRICT_MODE=1 \
cargo run --locked -p bitnet-cli \
  --no-default-features \
  --features cpu,full-cli \
  -- \
  --device apple-m4-cpu-neon \
  run \
  --model models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf \
  --prompt "What is 2+2? Answer briefly." \
  --max-tokens 32 \
  --temperature 0.0 \
  --greedy \
  --deterministic \
  --strict-loader \
  --strict-tokenizer \
  --prompt-template raw \
  --json-out ci/hardware/apple-m4-mac-mini/2026-05-07/local-answer-cpu-neon.json
```

This is the closest supported user-facing path: prompt in, generated text out,
and a strict receipt showing the requested and selected Apple CPU backend. It may
be slow. Its receipt must not be described as Metal acceleration.

## Conservative Profile Names

Apple M4 operational profiles are named so a receipt cannot turn one tiny proof
into a broad performance claim.

| Profile | Meaning | Current status |
|---|---|---|
| `strict_cpu_neon_smoke_1` | One-token strict CPU/NEON proof/profile with timing fields. | Emitted by `apple-m4 validate` as `phase-profile.json` and `allocation-audit.json`. |
| `metal_i2s_parity` | I2_S-adjacent Metal parity fixture. | Receipt-backed parity only, not a throughput benchmark. |
| `prefill_512` | Future 512-token prefill profile. | Not proven by the current bundle. |
| `decode_128` | Future 128-token decode profile. | Not proven by the current bundle. |
| `context_4096` | Future 4096-token context profile. | Not proven by the current bundle. |

Profile receipts should include:

```text
timing.model_load_ms
timing.tokenize_ms
timing.prefill_ms
timing.first_token_ms
timing.decode_steady_state_tok_s
timing.sampling_ms_per_token
latency.total_ms
```

The receipt checker validates those fields for profile receipts. Missing values
may be `null` only where the profile cannot produce a meaningful number, such as
steady-state decode throughput for a one-token smoke run.

## Metal Phase Proof

Metal proof is currently phase/subgraph proof, not full model inference. The
operator validation command runs the live Metal proofs for:

- `metal-smoke.json`: tiny native Metal add smoke;
- `metal-i2s-parity.json`: I2_S-adjacent Metal parity against CPU/NEON.

If you run the lower-level live tests directly, write receipts under the same
artifact convention and do not commit local machine artifacts unless the work
item explicitly allows it:

```bash
BITNET_RUN_M4_METAL_I2S_PARITY=1 \
BITNET_M4_METAL_I2S_PARITY_RECEIPT=ci/hardware/apple-m4-mac-mini/2026-05-07/metal-i2s-parity.json \
cargo test --locked -p bitnet-kernels \
  --no-default-features --features metal \
  --test metal_tiny_smoke \
  tiny_m4_metal_i2s_matches_cpu_neon_reference_when_enabled \
  -- --nocapture
```

This proves only the recorded kernel or phase. It does not prove full model
inference, QK256 on Metal, MPSGraph execution, Neural Engine execution, or
general performance.

## Failure Modes

| Symptom | Meaning | Operator action |
|---|---|---|
| `Apple M4 validation model does not exist` | The `--model` path is wrong or the model was not downloaded. | Download the canonical GGUF and rerun with the correct path. |
| Strict tokenizer failure | No real tokenizer source was found or accepted. | Place the tokenizer artifact beside the model or use a model artifact with embedded tokenizer metadata. |
| Metal probe unavailable | macOS did not report an Apple M4-family Metal-visible runtime. | Treat Metal proof as unavailable; do not count CPU fallback as Metal. |
| Metal test command fails | Native Metal smoke/parity did not complete. | Inspect the failing test output; keep the claim at probe/CPU-only until a Metal receipt passes. |
| MPSGraph resolved target is `ane` or Neural Engine is claimed | The current checker rejects Neural Engine claims without separate proof. | Keep MPSGraph as graph/reference evidence only. |
| Receipt checker rejects QK256 | Apple Silicon QK256 is not currently supported by this operational lane. | Use I2_S/TL1 receipt-backed paths; do not relabel QK256 as supported. |
| `selected_backend` differs from `requested_backend` | A fallback happened or the receipt is inconsistent. | Require a clear `fallback_reason`; do not count fallback as the requested backend. |

## Artifact Hygiene

Use `ci/hardware/apple-m4-mac-mini/<date>/` only when the artifact is intended
for review or archival. Use `target/apple-m4-validation-<date>/` for local
scratch runs.

Do not commit bulky or machine-specific outputs unless the active campaign item
explicitly asks for them. When publishing artifacts, include the receipt bundle
and run `apple-m4 receipts-check` first.

## Current User-Facing Bar

The final Mac Silicon end-state is:

```text
local prompt
-> real supported model
-> explicit Apple backend routing
-> coherent generated text
-> receipt proves requested backend, selected backend, runtime API, fallback
   status, model, tokenizer, kernel family, execution phase, artifact path, and
   unsupported claims
```

Today, use `apple-m4-cpu-neon` for the reliable local-answer path. Use
`apple-m4-metal` only for receipt-backed Metal phases or future full-model Metal
inference once a strict real-model receipt proves it. Use `apple-m4-mpsgraph`
only as graph/reference evidence.
