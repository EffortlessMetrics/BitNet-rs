# Apple M4 SLM Performance Envelope

This envelope publishes measured Apple M4 SLM performance only for the named
model, backend, profiles, machine, and receipts below.

## Source Receipts

```text
ci/hardware/apple-m4-mac-mini/2026-05-08/slm-performance/release-baseline.json
ci/hardware/apple-m4-mac-mini/2026-05-08/slm-performance/allocation-audit.json
ci/hardware/apple-m4-mac-mini/2026-05-08/slm-performance/metal-phase/metal-dense-prefill-linear.json
```

All three source receipts pass `bitnet mac receipts-check`.

## Measurement Context

```text
machine_id = apple-m4-mac-mini
chip = Apple M4
cpu_cores = 10
gpu_cores = 10
unified_memory_bytes = 17179869184
model = Qwen2.5 0.5B Instruct Q8_0
model_sha256 = ca59ca7f13d0e15a8cfa77bd17e65d24f6844b554a7b6c12e07a5f89ff76844e
model_bytes = 675710816
tokenizer_model = gpt2
tokenizer_pre = qwen2
requested_backend = apple-m4-cpu-neon
selected_backend = apple-m4-cpu-neon
runtime_api = cpu
fallback_used = false
build_profile = release
```

## CPU/NEON Warm Profiles

| Profile | Requested max tokens | Generated tokens | Warm prompt tok/s | Decode tok/s | First token mean ms | Total session ms | Peak memory MB |
|---|---:|---:|---:|---:|---:|---:|---:|
| `warm_16` | 16 | 34 | 4.435 | 14.962 | 1885.000 | 12576.726 | 3772.469 |
| `warm_32` | 32 | 50 | 5.947 | 15.317 | 1779.333 | 13117.047 | 4009.078 |
| `warm_64` | 64 | 82 | 7.840 | 15.269 | 1763.000 | 15228.741 | 4026.438 |
| `warm_128` | 128 | 123 | 9.347 | 15.313 | 1775.333 | 17896.347 | 4033.422 |

The CPU/NEON receipt records `release_mode_observed=true`,
`cold_load_separated=true`, `warm_128_included=true`, `speedup_claim=false`, and
`broad_performance_claim=false`.

## Allocation Audit

The allocation audit repeats the same four profiles with process-global
allocator counter deltas enabled.

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

These are allocation counter deltas, not resident-memory measurements.

## Metal Phase Contribution

The Metal phase receipt is phase-local and does not claim full model inference.

```text
artifact_kind = phase_contribution
requested_backend = apple-m4-metal
selected_backend = apple-m4-metal
runtime_api = metal
fallback_used = false
execution_phase = prefill_linear_projection
kernel_id = tiny_metal_dense_prefill_linear_projection
kernel_family = dense_f32
cpu_reference_ms = 0.013333
metal_phase_ms = 135.298875
timing_delta_ms = 135.285542
greedy_token_ids_match_cpu_reference = true
speedup_claim = false
full_metal_inference_claimed = false
```

## Claim Boundary

This envelope may claim only that the named Apple M4 Mac mini, Qwen2.5 0.5B
Q8_0 model, CPU/NEON warm profiles, allocation audit, and named Metal phase
receipt were measured.

It must not claim broad Apple M4 performance, full `apple-m4-metal` inference,
BitNet quality or performance, QK256 on Apple Silicon, MPSGraph model inference,
or Neural Engine execution.
