# Apple M4 Dense SLM Model Breadth Rust M4 Quality

`M4-MODEL-003` runs the reference-good SmolLM2 candidate from
`M4-MODEL-002` through the Rust M4 quality gate. The result is a rejection for
this round: SmolLM2 remains reference-good, but it is not supported by the Rust
M4 dense SLM path.

Machine-readable evidence is recorded in
`ci/quality/apple-m4-slm-model-breadth-rust-m4-quality.toml`.

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

## Claim Boundary

This item may claim only that SmolLM2 failed Rust M4 quality for this round. It
does not change the M4 default, register a model, prove BitNet behavior, prove
full Apple Metal inference, prove Neural Engine execution, prove MPSGraph model
inference, prove QK256 support, or make broad M4 performance claims.
