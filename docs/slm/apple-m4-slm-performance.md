# Apple M4 SLM Performance

The Apple M4 SLM path is now a working local-answer baseline. This document defines the next performance campaign: measure release-mode warm-session behavior first, then remove overhead and expand acceleration only where receipts prove it.

## Current Baseline

Works today:

```text
Rust-native dense SLM local answers on apple-m4-cpu-neon, with strict model/tokenizer routing, fallback status, warm-session receipts, deterministic quality checks, model cache management, Mac CLI wrappers, and a first named Metal phase proof.
```

Not claimed:

```text
BitNet local-answer quality
full apple-m4-metal inference
Neural Engine execution
QK256 on Apple Silicon
general M4 performance
```

## First Performance Item

`M4-SLM-PERF-001` adds a release-mode baseline profile set for these profiles:

```text
warm_16
warm_32
warm_64
warm_128
```

Each profile should separate cold load from warm prompt timing and record:

```text
model_load_ms
tokenize_ms
prefill_ms
first_token_ms
decode_ms
sampling_ms
total_ms
tokens_per_second
peak_memory_mb
model_loaded_once
tokenizer_loaded_once
requested_backend
selected_backend
runtime_api
fallback_used
```

The release-mode command shape is:

```bash
cargo run --release --locked -p bitnet-cli \
  --no-default-features --features cpu,full-cli -- \
  mac validate \
  --profile-set performance \
  --json-out target/apple-m4-slm-performance/M4-SLM-PERF-001/release-baseline.json
```

The `performance` profile set must be run from a release build. Debug builds should fail before producing a performance receipt so profile artifacts cannot look like release-mode evidence by accident.

Local M4 proof receipt:

```text
ci/hardware/apple-m4-mac-mini/2026-05-08/slm-performance/release-baseline.json
```

Observed release-mode summary on the recorded M4 Mac mini:

| Profile | Generated tokens | Warm prompt tok/s | Decode tok/s | First token mean ms | Total session ms | Peak memory MB |
|---|---:|---:|---:|---:|---:|---:|
| `warm_16` | 34 | 4.435 | 14.962 | 1885.000 | 12576.726 | 3772.469 |
| `warm_32` | 50 | 5.947 | 15.317 | 1779.333 | 13117.047 | 4009.078 |
| `warm_64` | 82 | 7.840 | 15.269 | 1763.000 | 15228.741 | 4026.438 |
| `warm_128` | 123 | 9.347 | 15.313 | 1775.333 | 17896.347 | 4033.422 |

The receipt records `requested_backend=apple-m4-cpu-neon`, `selected_backend=apple-m4-cpu-neon`, `runtime_api=cpu`, and `fallback_used=false`. It also records `release_mode_observed=true`, `warm_128_included=true`, `speedup_claim=false`, and `broad_performance_claim=false`.

## Allocation Audit

`M4-SLM-PERF-002` adds allocation-counter auditing to the SLM warm-session path and records the audit in the same release-mode profile set. The audit uses process-global allocator counter deltas scoped to prompt tokenize/setup, prompt prefill, decode substeps, token vector updates, token decode, stop-tail updates, and prompt receipt construction.

Audit command:

```bash
cargo run --release --locked -p bitnet-cli \
  --no-default-features --features cpu,full-cli -- \
  mac validate \
  --profile-set performance \
  --allocation-audit \
  --json-out ci/hardware/apple-m4-mac-mini/2026-05-08/slm-performance/allocation-audit.json
```

Local M4 audit receipt:

```text
ci/hardware/apple-m4-mac-mini/2026-05-08/slm-performance/allocation-audit.json
```

Aggregate ranked hotspots from the recorded audit:

| Component | Alloc count | Alloc bytes |
|---|---:|---:|
| `prompt_setup` | 2,977 | 9,663,994,696 |
| `decode_total` | 4,698,106 | 5,970,721,165 |
| `model.forward` | 4,681,800 | 5,434,345,232 |
| `prompt_prefill` | 6,740,800 | 3,875,563,648 |
| `prompt_tokenize` | 18,241,403 | 1,519,238,922 |
| `model.logits_and_extract` | 8,381 | 527,517,191 |
| `sampler.sample` | 56 | 7,306,608 |
| `model.embed` | 4,913 | 1,418,412 |
| `receipt_construction` | 3,276 | 331,534 |
| `tokenizer.decode` | 2,089 | 57,426 |

These are allocation counter deltas, not resident-memory measurements. They can include transient allocate/free churn and allocator reuse behavior. The first optimization targets should therefore be chosen from the ranked evidence, not from raw intuition:

1. `prompt_setup` is dominated by per-profile/per-prompt runtime setup such as KV cache/session objects. This is a resident-session hardening target, not a math kernel target.
2. `decode_total` is dominated by `model.forward`, so CPU/NEON hot-path work should start from the measured forward path after setup/session reuse is bounded.
3. `prompt_prefill` and `prompt_tokenize` are large enough to keep visible in before/after receipts.
4. `model.logits_and_extract` is smaller than forward but still material for decode.
5. `sampler.sample`, `tokenizer.decode`, token vector updates, and receipt construction are currently named rather than optimized; receipt construction is outside the decode hot loop.

Unavoidable candidates before optimization:

```text
model.embed/model.forward/model.logits tensor outputs from the current dense Qwen CPU execution path
tokenizer.decode allocation for per-token text and stop-tail checks
receipt construction outside the decode hot loop
prompt token vector growth until a reusable session buffer is introduced
```

This audit does not claim any performance improvement. It names and ranks allocation hotspots so `M4-SLM-PERF-003` and `M4-SLM-PERF-004` can remove overhead with before/after receipts.

## Resident Session Hardening

`M4-SLM-PERF-003` makes the existing SLM warm-session command more explicit about what stays resident across prompts. The model and tokenizer were already loaded once per aggregate warm-session run; this item adds CLI-owned reusable prompt buffers and receipt fields so operators can distinguish durable session state from prompt-local runtime state.

Resident-session receipts now record:

```text
session.reuse_scope = resident_session
session.session_owned_buffers = true
session.prompt_token_buffer_reused = true
session.generated_token_buffer_reused = true
session.timing_buffers_reused = true
session.allocation_audit_buffers_reused = true
session.stop_tail_buffer_reused = true
session.kv_cache_reuse_policy = recreated_per_prompt_for_prompt_isolation
session.sampler_reuse_policy = recreated_per_prompt_for_deterministic_prompt_independence
session.logits_buffer_reuse_policy = not_claimed_until_logits_extraction_uses_reusable_storage
```

The important boundary is that this is session hygiene, not a speed claim. KV cache and sampler state remain reset per prompt to preserve independent prompt correctness. Logits extraction still materializes a fresh vector, so logits buffer reuse is deliberately not claimed.

## CPU/NEON Hot-Path Optimization

`M4-SLM-PERF-004` starts with a narrow transformer-side optimization in the measured `model.forward` decode path. The single-token decode path no longer materializes and broadcasts an all-zero causal mask, and static debug/trace environment flags are cached so disabled diagnostics do not repeatedly query the process environment inside every layer and generated token.

This item preserves the existing generation path and claim boundary:

```text
requested_backend = apple-m4-cpu-neon
selected_backend = apple-m4-cpu-neon
runtime_api = cpu
fallback_used = false
```

The optimization is valid only if the same greedy prompt/profile receipts keep the same generated token IDs and quality status. It does not claim broad Apple M4 performance, BitNet quality, full Metal inference, or QK256 support.

Local M4 proof receipt:

```text
target/apple-m4-slm-performance/M4-SLM-PERF-004/cpu-neon-hotpath.json
```

Compared with the `M4-SLM-PERF-001` release baseline, the recorded run preserved generated token IDs for every warm profile and prompt:

| Profile | Baseline decode tok/s | Optimized decode tok/s | Baseline warm tok/s | Optimized warm tok/s |
|---|---:|---:|---:|---:|
| `warm_16` | 14.517 | 15.131 | 4.266 | 4.425 |
| `warm_32` | 15.187 | 15.246 | 5.879 | 5.923 |
| `warm_64` | 14.909 | 15.250 | 7.663 | 7.819 |
| `warm_128` | 14.925 | 15.132 | 8.847 | 9.061 |

## Optimization Order

1. Measure release-mode warm-session bottlenecks.
2. Audit decode-loop allocation churn and temporary-buffer creation.
3. Harden resident-session reuse as the normal multi-prompt path.
4. Optimize the measured CPU/NEON bottleneck while preserving greedy token IDs.
5. Expand only parity-gated Metal phases with phase-specific receipts.
6. Add streaming and time-to-first-token UX.
7. Publish a measured performance envelope for supported profiles only.

## Metal Phase Expansion

`M4-SLM-PERF-005` keeps the existing first Metal phase narrow: the deterministic
Qwen2.5 dense f32 prefill linear projection fixture. The receipt records:

```text
slm_pipeline.selected_backend = apple-m4-cpu-neon
metal_phase.selected_backend = apple-m4-metal
metal_phase.execution_phase = prefill_linear_projection
metal_phase.fallback_used = false
parity.greedy_token_ids_match_cpu_reference = true
timing.cpu_reference_ms
timing.metal_phase_ms
timing.timing_delta_ms
timing.speedup_claim = false
```

This timing is phase-local. It does not claim full SLM inference, full
`apple-m4-metal` routing, or broad M4 performance.

## Streaming And Operator UX

`M4-SLM-PERF-006` adds user-facing output controls around the resident
warm-session path without changing backend routing or speed claims:

```text
--stream    write generated token text to stdout as each token is decoded
--progress  write operator progress lines to stderr
--quiet     suppress status/progress lines while still writing receipts
```

Warm-session receipts now keep `timing.first_token_ms` and add the explicit
alias `timing.time_to_first_token_ms` per prompt. Aggregate receipts also record
`speed.timing.time_to_first_token_ms`, plus an `operator_ux` block that states
whether token streaming and progress were requested. The default remains quiet
for receipt-oriented runs: no progress output is emitted unless `--progress` is
passed, and token text is streamed only when `--stream` is passed.

`bitnet mac validate` exposes `--progress` and `--quiet` for operator profile
runs. The Mac validation receipts preserve the same claim boundary: they record
time-to-first-token and progress behavior, but do not claim broad performance,
BitNet quality, full Metal inference, Neural Engine execution, MPSGraph model
inference, or QK256 support.

## Published Performance Envelope

`M4-SLM-PERF-007` publishes the measured Apple M4 SLM envelope from current
validator-passing receipts:

```text
ci/hardware/apple-m4-mac-mini/2026-05-08/slm-performance/release-baseline.json
ci/hardware/apple-m4-mac-mini/2026-05-08/slm-performance/allocation-audit.json
ci/hardware/apple-m4-mac-mini/2026-05-08/slm-performance/metal-phase/metal-dense-prefill-linear.json
ci/hardware/apple-m4-mac-mini/2026-05-08/slm-performance/performance-envelope.md
```

Measured context:

```text
machine_id = apple-m4-mac-mini
chip = Apple M4
cpu_cores = 10
gpu_cores = 10
unified_memory_bytes = 17179869184
model = Qwen2.5 0.5B Instruct Q8_0
model_sha256 = ca59ca7f13d0e15a8cfa77bd17e65d24f6844b554a7b6c12e07a5f89ff76844e
requested_backend = apple-m4-cpu-neon
selected_backend = apple-m4-cpu-neon
runtime_api = cpu
fallback_used = false
```

The published envelope for the recorded release-mode CPU/NEON run is:

| Profile | Requested max tokens | Generated tokens | Warm prompt tok/s | Decode tok/s | First token mean ms | Total session ms | Peak memory MB |
|---|---:|---:|---:|---:|---:|---:|---:|
| `warm_16` | 16 | 34 | 4.435 | 14.962 | 1885.000 | 12576.726 | 3772.469 |
| `warm_32` | 32 | 50 | 5.947 | 15.317 | 1779.333 | 13117.047 | 4009.078 |
| `warm_64` | 64 | 82 | 7.840 | 15.269 | 1763.000 | 15228.741 | 4026.438 |
| `warm_128` | 128 | 123 | 9.347 | 15.313 | 1775.333 | 17896.347 | 4033.422 |

The allocation-audit envelope records the same four profiles with auditing
enabled. Its top allocation-counter hotspots are `prompt_setup`,
`decode_total`, `model.forward`, `prompt_prefill`, and `prompt_tokenize`. These
are process-global allocation counter deltas; they are not resident-memory
claims.

The Metal phase contribution remains a separate phase-local receipt:

```text
execution_phase = prefill_linear_projection
kernel_id = tiny_metal_dense_prefill_linear_projection
metal_phase_ms = 135.298875
cpu_reference_ms = 0.013333
timing_delta_ms = 135.285542
fallback_used = false
speedup_claim = false
full_metal_inference_claimed = false
```

This envelope may claim only that the named model, profiles, backend, machine,
and phase contribution were measured. It does not claim broad M4 performance,
full `apple-m4-metal` inference, BitNet quality or performance, QK256 on Apple
Silicon, MPSGraph model inference, or Neural Engine execution.

## Claim Boundary

Performance claims must be tied to a receipt with:

```text
model
tokenizer
profile
backend
machine context
fallback status
timing fields
quality and determinism status
```

Do not report a speedup unless the before and after receipts use the same model, prompt/profile, backend, and generation settings. A named Metal phase can claim only that phase, not full `apple-m4-metal` inference.
