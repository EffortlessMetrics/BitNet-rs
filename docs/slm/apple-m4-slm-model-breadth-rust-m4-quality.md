# Apple M4 Dense SLM Model Breadth Rust M4 Quality

`M4-MODEL-003` runs the reference-good SmolLM2 candidate from
`M4-MODEL-002` through the Rust M4 quality gate. The result is a rejection for
this round: SmolLM2 remains reference-good, but it is not supported by the Rust
M4 dense SLM path.

`M4-MODEL-011` runs the reference-good Qwen2.5 1.5B Q4_K_M candidate from
`M4-MODEL-010` through the same Rust M4 quality gate. The result is an
acceptance for cache-registration consideration only: the candidate passes
bounded Rust M4 output quality and duplicate-prompt greedy determinism, but it
is not selectable as a supported Mac model until the cache/model-registration
item lands.

Machine-readable evidence is recorded in
`ci/quality/apple-m4-slm-model-breadth-rust-m4-quality.toml`.

The larger Qwen2.5 candidate cycle records machine-readable evidence in
`ci/quality/apple-m4-slm-model-breadth-qwen15-rust-m4-quality.toml`.

## Candidate

```text
id = smollm2-360m-instruct-q8_0
repo = HuggingFaceTB/SmolLM2-360M-Instruct-GGUF
revision = 593b5a2e04c8f3e4ee880263f93e0bd2901ad47f
file = smollm2-360m-instruct-q8_0.gguf
size = 386404992 bytes
sha256 = 48ab3034d0dd401fbc721eb1df3217902fee7dab9078992d66431f09b7750201
gguf_architecture = llama
gguf_basename = smollm2
quantization = Q8_0
tokenizer.ggml.model = gpt2
tokenizer.ggml.pre = smollm
tokenizer.chat_template = present
```

`M4-MODEL-002` accepted this artifact under the reference runner because the
bounded prompt suite produced coherent short answers. That reference result does
not by itself prove Rust M4 support.

## Strict Rust M4 Gate

Command shape:

```text
cargo run --locked -p bitnet-cli --no-default-features --features cpu,full-cli -- \
  --device apple-m4-cpu-neon \
  run \
  --model <smollm2.gguf> \
  --prompt "What is 2+2? Answer briefly." \
  --max-new-tokens 1 \
  --temperature 0 \
  --top-k 1 \
  --top-p 1 \
  --prompt-template smollm-chat \
  --system-prompt "<smollm2 reference system>" \
  --greedy \
  --deterministic \
  --strict-loader \
  --strict-tokenizer \
  --repetition-penalty 1.0 \
  --json-out <target receipt> \
  --no-warnings
```

Observed result:

```text
exit_code = 1
phase = model_load
failure_class = strict_loader_layernorm_gamma_guard
error = LayerNorm gamma 'blk.0.ffn_norm.weight' suspicious: rms=0.09831
```

The current strict Rust loader rejects the artifact before generation, so no
model-cache registration or supported-model claim is allowed.

## Diagnostic Compatibility Probes

Local diagnostic probes temporarily bypassed the dense LayerNorm guard to check
whether the remaining path could produce coherent text. Those probes are not
committed runtime support and are not support evidence.

The best prompt-token parity probe matched the 39-token reference prompt and
routed through `apple-m4-cpu-neon` with `fallback_used=false`, but the output was
not plausible:

```text
generated_ids = [198, 1780, 314, 260, 1462, 282, 260, 1462, ...]
generated_text = "\nWhat is the name of the name of the name of the name of the"
```

The embedded GGUF chat-template variant with an assistant newline also failed:

```text
generated_ids = [1780, 314, 260, 1462, 282, 260, 1462, ...]
generated_text = "What is the name of the name of the name of the name of the name"
```

Speculative Q8_0 layout probes for square projections, non-square projections,
and tied embeddings all degraded output further. No Q8_0 layout change is
accepted from this item.

## Decision

```text
result = rejected_rust_m4_quality
supported_model = false
register_in_cache = false
default_model_change = false
next_item_unblocked = false
```

SmolLM2 remains a useful reference-good artifact, but it does not pass the Rust
M4 quality gate. `M4-MODEL-004` must remain blocked because there is no accepted
new model to register.

## Qwen2.5 1.5B Q4_K_M Rust M4 Acceptance

Candidate:

```text
id = qwen2.5-1.5b-instruct-q4_k_m
repo = Qwen/Qwen2.5-1.5B-Instruct-GGUF
revision = 91cad51170dc346986eccefdc2dd33a9da36ead9
file = qwen2.5-1.5b-instruct-q4_k_m.gguf
size = 1117320736 bytes
sha256 = 6a1a2eb6d15622bf3c96857206351ba97e1af16c30d7a74ee38970e434e9407e
gguf_architecture = qwen2
quantization = Q4_K_M
tokenizer.ggml.model = gpt2
tokenizer.ggml.pre = qwen2
tokenizer.chat_template = present
```

Strict Rust M4 warm-session command shape:

```text
cargo run --locked -p bitnet-cli --no-default-features --features cpu,full-cli -- \
  --device apple-m4-cpu-neon \
  slm-warm-session \
  --model <qwen2.5-1.5b-q4_k_m.gguf> \
  --prompt "What is 2+2? Answer briefly." \
  --prompt "Name the capital of France." \
  --prompt "Write one short sentence about Rust." \
  --max-new-tokens 16 \
  --temperature 0 \
  --top-k 1 \
  --top-p 1 \
  --greedy \
  --deterministic \
  --strict-loader \
  --strict-tokenizer \
  --repetition-penalty 1.0 \
  --prompt-template qwen2.5 \
  --fail-on-quality \
  --json-out <target receipt> \
  --quiet
```

Observed bounded outputs:

| Prompt | Normalized Rust output | Generated token IDs |
|---|---|---|
| `What is 2+2? Answer briefly.` | `2+2 equals 4.` | `[220, 17, 10, 17, 16819, 220, 19, 13, 151645]` |
| `Name the capital of France.` | `The capital of France is Paris.` | `[576, 6722, 315, 9625, 374, 12095, 13, 151645]` |
| `Write one short sentence about Rust.` | `Rust is a systems programming language known for its safety, speed, and` | `[198, 49, 590, 374, 264, 5942, 15473, 4128, 3881, 369, 1181, 7149, 11, 4628, 11, 323]` |

The third prompt is truncated by the 16-token Rust quality budget, but it is
valid UTF-8, non-empty, non-degenerate, and semantically plausible for the
bounded smoke gate.

Receipt highlights:

```text
requested_backend = apple-m4-cpu-neon
selected_backend = apple-m4-cpu-neon
runtime_api = cpu
fallback_used = false
loader_mode = real_gguf
tokenizer.source = gguf_metadata
tokenizer.pretokenizer_authority = present
model_load_ms = 30374.613
model_sha256_ms = 1962.101
tokenizer_load_ms = 215.5
total_session_ms = 99341.727
decode_steady_state_tok_s ~= 2.03 to 2.06
```

Duplicate-prompt determinism passed with stable generated token IDs and text:

```text
generated_ids = [220, 17, 10, 17, 16819, 220, 19, 13, 151645]
generated_text = " 2+2 equals 4.<|im_end|>"
```

Decision:

```text
result = accepted_rust_m4_quality
supported_model = false
register_in_cache = false
eligible_for_cache_registration = true
default_model_change = false
next_item_unblocked = true
```

The candidate may proceed to cache/model-registration review. It remains
unregistered and unsupported until that item records fetch/verify/list behavior,
cache metadata, receipt validation, and model selection.

## Claim Boundary

This page may claim only the recorded Rust M4 quality-gate outcomes for the
bounded model-breadth candidates. It does not change the M4 default, register a
model, prove BitNet behavior, prove full Apple Metal inference, prove Neural
Engine execution, prove MPSGraph model inference, prove QK256 support, prove
MacBook behavior, or make broad M4 performance claims.
