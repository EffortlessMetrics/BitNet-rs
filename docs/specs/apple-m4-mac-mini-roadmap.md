# Apple M4 Mac mini Roadmap

## Purpose

This document defines the Apple M4 Mac mini validation lane for BitNet-rs. The M4 lane is Metal-first, with MPSGraph as a graph/reference lane and CPU/NEON as the fallback/parity lane.

Primary labels:

```text
apple-m4-metal
apple-m4-mpsgraph
apple-m4-cpu-neon
```

The first work after the lane scaffold is M4-002 machine profiling. It records stable machine facts and planned probe artifact paths before any Metal kernels, MPSGraph graph execution, CPU/Metal parity, receipts, or benchmarks.

## Hardware Baseline

Base M4 Mac mini:

| Property | Expected value |
|---|---|
| CPU | 10-core CPU |
| GPU | 10-core GPU |
| Neural Engine | 16-core Neural Engine |
| Neural Engine peak | Up to 38 TOPS |
| Memory | 16GB unified memory, configurable to 24GB or 32GB |
| Memory bandwidth | 120 GB/s |
| GPU features | Dynamic Caching, hardware ray tracing, mesh shading |

M4 Pro variants:

| Property | Expected value |
|---|---|
| CPU | 12-core or configurable 14-core CPU |
| GPU | 16-core or configurable 20-core GPU |
| Neural Engine | 16-core Neural Engine |
| Memory | Up to 64GB unified memory |
| Memory bandwidth | 273 GB/s |

Receipts and probe artifacts must record the actual chip, CPU/GPU core counts, unified memory size, and memory bandwidth class when known from confirmed docs/specs. Do not assume base M4 when validating M4 Pro.

## Claim Boundary

- Metal device visibility is not Metal execution.
- Metal compute smoke is not CPU/Metal parity.
- CPU/Metal parity is not full inference.
- MPSGraph smoke is not handwritten Metal kernel proof.
- MPSGraph smoke is not Neural Engine proof unless the resolved execution target is receipt-backed.
- CPU fallback cannot count as Metal execution.
- Apple CPU/NEON fallback is not AVX2 or AVX-512.

## Runtime Paths

### Machine Profile Path

M4-002 owns docs/artifact prep only. It must collect and document:

- macOS version and kernel/build.
- Native macOS vs virtualized execution.
- Apple chip name: M4 or M4 Pro.
- CPU core count.
- GPU core count when visible from system tools or confirmed machine spec.
- Unified memory size.
- Memory bandwidth class when known.
- Metal visibility.
- MPSGraph lane notes.
- CPU/NEON lane notes.
- Rust toolchain versions.

Expected probe artifact paths:

```text
ci/hardware/apple-m4-mac-mini/<date>/metal-probe.json
ci/hardware/apple-m4-mac-mini/<date>/cpu-neon-probe.json
ci/hardware/apple-m4-mac-mini/<date>/mpsgraph-probe.json
```

These paths are probe placeholders until a machine run produces receipts. Do not commit large machine-specific outputs as part of M4-002.

### Native Metal Path

Milestones:

1. Metal device visibility.
2. Tiny Metal compute pipeline compiles.
3. Tiny compute dispatch runs.
4. CPU/Metal output parity.
5. Receipt-backed Metal kernel/subgraph proof.
6. Benchmark baseline with macOS, chip, unified memory, and fallback context.

### MPSGraph Reference Path

MPSGraph is the Apple graph/reference lane. It may route through GPU, CPU, or Neural Engine depending on platform/runtime behavior.

Milestones:

1. MPSGraph availability and version/runtime notes.
2. Tiny graph smoke.
3. CPU reference comparison.
4. Receipt records resolved target when available.

Do not use MPSGraph smoke as native Metal kernel proof.

### CPU/NEON Path

The CPU lane supports fallback and parity:

```text
selected_backend = "apple-m4-cpu-neon"
runtime_api = "cpu"
```

This is useful for CPU/Metal parity and no-accelerator fallback receipts.

## Receipt Fields

Minimum Metal receipt:

```json
{
  "requested_backend": "apple-m4-metal",
  "selected_backend": "apple-m4-metal",
  "runtime_api": "metal",
  "resolved_device": {
    "chip": "Apple M4",
    "gpu_cores": 10,
    "unified_memory": true
  },
  "fallback_used": false,
  "proof_stage": "runtime_detected",
  "artifact_path": "ci/hardware/apple-m4-mac-mini/2026-05-05/metal-probe.json"
}
```

Minimum MPSGraph receipt:

```json
{
  "requested_backend": "apple-m4-mpsgraph",
  "selected_backend": "apple-m4-mpsgraph",
  "runtime_api": "mpsgraph",
  "resolved_device": {
    "chip": "Apple M4",
    "resolved_target": "gpu|cpu|neural-engine|unknown"
  },
  "fallback_used": false,
  "proof_stage": "runtime_detected",
  "artifact_path": "ci/hardware/apple-m4-mac-mini/2026-05-05/mpsgraph-probe.json",
  "graph": {
    "name": "tiny_matmul",
    "shape_mode": "static"
  }
}
```

Minimum CPU/NEON receipt:

```json
{
  "requested_backend": "apple-m4-cpu-neon",
  "selected_backend": "apple-m4-cpu-neon",
  "runtime_api": "cpu",
  "resolved_device": {
    "chip": "Apple M4",
    "cpu_cores": 10,
    "unified_memory": true
  },
  "fallback_used": false,
  "proof_stage": "runtime_detected",
  "artifact_path": "ci/hardware/apple-m4-mac-mini/2026-05-05/cpu-neon-probe.json"
}
```

BitNet-specific artifacts must add:

```json
{
  "model": {
    "repo": "microsoft/bitnet-b1.58-2B-4T-gguf",
    "file": "ggml-model-i2_s.gguf",
    "tokenizer": "llama3"
  },
  "bitnet": {
    "kernel_family": "i2_s|tl1|qk256|openvino_graph",
    "execution_phase": "probe|smoke|parity|prefill|decode"
  }
}
```

## Validation Bundle

The machine bundle lives in:

```text
docs/hardware/apple-m4-mac-mini-validation.md
```

It must collect:

- macOS version.
- Apple chip name.
- CPU/GPU core counts.
- Neural Engine presence.
- Unified memory size and bandwidth target.
- Metal device visibility.
- MPSGraph availability notes.
- CPU/NEON availability notes.
- Rust toolchain.

## Work Plan

### M4-001 - Add Backend Lane

Docs/tracking only. Add M4 Metal, MPSGraph, and CPU/NEON lanes.

### M4-002 - Machine Profile

Collect macOS, native-vs-virtualized status, Apple chip, CPU/GPU cores, unified memory, memory bandwidth class when known, Metal visibility, MPSGraph lane notes, CPU/NEON lane notes, Rust toolchain data, and expected probe artifact paths.

### M4-003 - Backend Identity

Preserve Apple M4 Metal, MPSGraph, and CPU/NEON requested and selected backend identities.

### M4-004 - Metal Probe

Report Metal device visibility, Apple chip, GPU family where available, unified memory, and macOS version.

This probe records runtime visibility only. It preserves `requested_backend`,
`selected_backend`, `runtime_api`, `fallback_used`, and `proof_stage` fields,
but it does not compile or dispatch a Metal compute pipeline.

### M4-005 - Metal Compute Smoke

Compile and dispatch a tiny Metal compute pipeline.

### M4-006 - CPU/Metal Parity

Compare one Metal kernel/subgraph output against CPU/NEON.

### M4-007 - MPSGraph Smoke

Run a tiny graph through MPSGraph and record the resolved target when available.

### M4-008 - Receipts

Record chip, GPU core count, unified memory, selected backend, fallback status, and kernel/graph IDs.

### M4-009 - Benchmark Baseline

Compare Apple CPU/NEON against M4 Metal for the validated kernel/subgraph.
The first baseline is limited to the validated `tiny_metal_add_smoke` kernel and
records `compile_ms`, `first_dispatch_ms`, `steady_state_ms`, and
`cpu_reference_ms` in `ci/hardware/apple-m4-mac-mini/<date>/metal-benchmark.json`.
It is not a general Metal performance claim and is not a BitNet inference claim.

### M4-010 - Apple CPU/NEON BitNet Reference

Run a BitNet reference path on Apple CPU/NEON or scalar fallback with model,
tokenizer, kernel family, and fallback status recorded.

The proof artifact is emitted through the CLI JSON path and must include:

```json
{
  "artifact_kind": "strict_bitnet_cpu_reference",
  "requested_backend": "apple-m4-cpu-neon",
  "selected_backend": "apple-m4-cpu-neon",
  "runtime_api": "cpu",
  "fallback_used": false,
  "model": {
    "repo": "microsoft/bitnet-b1.58-2B-4T-gguf",
    "file": "ggml-model-i2_s.gguf",
    "sha256": "...",
    "tokenizer": "llama3",
    "loader_mode": "real_gguf"
  },
  "bitnet": {
    "kernel_family": "i2_s",
    "execution_phase": "decode",
    "layout_source": "gguf_packed_i2_s_reference",
    "fallback_layout": null
  },
  "kernel": {
    "implementation": "scalar",
    "layout": "gguf_packed_i2_s",
    "dequantizes_before_compute": false,
    "kernel_id": "i2_s-scalar-reference"
  }
}
```

If scalar fallback is selected, the receipt must say so with
`fallback_used=true` and cannot count as Apple CPU/NEON proof. M4-010 does not
claim Metal BitNet execution, QK256 on Apple Silicon, or Apple CPU performance.

### M4-011 - Native Metal I2_S Smoke/Parity

Start BitNet-specific native Metal work with an I2_S-adjacent kernel or
subgraph, not QK256, and compare against Apple CPU/NEON. The first proof target
is `tiny_metal_i2s_parity`: a 1x4 output fixture with `k=32`, canonical I2_S
packed bytes, per-column scales, and a Metal storage-buffer transport layout of
`u32_le_words_from_i2s_bytes`.

The receipt must record:

```json
{
  "requested_backend": "apple-m4-metal",
  "selected_backend": "apple-m4-metal",
  "runtime_api": "metal",
  "kernel_id": "tiny_metal_i2s_parity",
  "fallback_used": false,
  "bitnet": {
    "kernel_family": "i2_s",
    "execution_phase": "parity",
    "layout_source": "fixture_packed_i2_s",
    "fallback_layout": null
  },
  "layout": {
    "source": "fixture_packed_i2_s",
    "transport_layout": "u32_le_words_from_i2s_bytes",
    "consumes_packed_i2_s_directly": true,
    "dequantizes_before_compute": false
  },
  "parity": {
    "reference_backend": "apple-m4-cpu-neon",
    "target_backend": "apple-m4-metal",
    "kernel_id": "tiny_metal_i2s_parity",
    "max_abs_error": 0.0,
    "mean_abs_error": 0.0
  }
}
```

M4-011 does not claim full BitNet Metal inference, QK256 on Metal, MPSGraph
execution, Neural Engine execution, or performance.

### M4-012 - TL1 / ARM-Oriented Investigation

Investigate whether TL1 is the right Apple CPU/NEON-oriented path and document
any layout conversion boundary before claiming Metal consumption.

Current TL1 evidence is CPU-oriented. The default TL1 layout uses unsigned
2-bit LUT codes packed four per byte, per-block scales, and optional zero
points for asymmetric quantization. Apple receipts should identify this as
`kernel_family = "tl1"`, `layout_source = "tl1_reference"`, and
`transport_layout = "tl1_packed_u2_codes_with_scales"`.

M4-012 may record:

```json
{
  "requested_backend": "apple-m4-cpu-neon",
  "selected_backend": "apple-m4-cpu-neon",
  "runtime_api": "cpu",
  "bitnet": {
    "kernel_family": "tl1",
    "execution_phase": "investigation",
    "layout_source": "tl1_reference"
  },
  "layout": {
    "transport_layout": "tl1_packed_u2_codes_with_scales",
    "conversion_boundary": "tl1_to_metal_transport_not_proven",
    "consumes_packed_tl1_directly_on_metal": false,
    "dequantizes_before_compute": true
  },
  "fallback_used": false
}
```

M4-012 must not claim native Metal TL1 execution. A later Metal item must either
prove direct packed TL1 consumption or record the exact conversion/dequantization
path before compute.

### M4-013 - Metal Prefill/Decode Contribution

Move from isolated kernels to a named BitNet phase such as prefill or decode
contribution, with CPU reference and explicit fallback status.

The first contribution target is `tiny_metal_i2s_prefill_contribution`, not a
full decode loop. It reuses the packed I2_S Metal fixture, expands it to two
prompt-token rows, and records the proof as a prefill projection fixture:

```json
{
  "artifact_kind": "phase_contribution",
  "requested_backend": "apple-m4-metal",
  "selected_backend": "apple-m4-metal",
  "runtime_api": "metal",
  "kernel_id": "tiny_metal_i2s_prefill_contribution",
  "fallback_used": false,
  "bitnet": {
    "kernel_family": "i2_s",
    "execution_phase": "prefill",
    "phase_scope": "prefill_projection_fixture",
    "layout_source": "fixture_packed_i2_s",
    "fallback_layout": null
  },
  "phase": {
    "name": "prefill",
    "prefill_tokens": 2,
    "kv_cache_behavior": "not_exercised",
    "full_autoregressive_decode": false
  },
  "parity": {
    "reference_backend": "apple-m4-cpu-neon",
    "target_backend": "apple-m4-metal",
    "max_abs_error": 0.0,
    "mean_abs_error": 0.0
  }
}
```

M4-013 may claim only that the named prefill contribution is receipt-backed
against CPU/NEON. It must not claim full decode, KV-cache correctness, full
BitNet inference, M4 acceleration, QK256 on Metal, MPSGraph execution, or Neural
Engine execution.

### M4-014 - Strict BitNet M4 Proof Run

Run strict real GGUF, real tokenizer, selected Apple backend,
`fallback_used=false`, deterministic prompt, and receipt emission before
claiming BitNet inference on M4.

The first proof target is `apple-m4-cpu-neon` with the canonical
`microsoft/bitnet-b1.58-2B-4T-gguf` I2_S GGUF. The receipt must keep Apple
machine identity, model identity, tokenizer authority, kernel family, execution
phase, and fallback state together:

```json
{
  "artifact_kind": "strict_bitnet_cpu_reference",
  "machine_id": "apple-m4-mac-mini",
  "requested_backend": "apple-m4-cpu-neon",
  "selected_backend": "apple-m4-cpu-neon",
  "runtime_api": "cpu",
  "fallback_used": false,
  "resolved_device": {
    "chip": "Apple M4",
    "gpu_cores": 10,
    "unified_memory": true
  },
  "model": {
    "repo": "microsoft/bitnet-b1.58-2B-4T-gguf",
    "file": "ggml-model-i2_s.gguf",
    "sha256": "...",
    "tokenizer": "llama3",
    "loader_mode": "real_gguf"
  },
  "bitnet": {
    "kernel_family": "i2_s",
    "execution_phase": "decode",
    "layout_source": "gguf_packed_i2_s_reference",
    "fallback_layout": null
  },
  "kernel": {
    "kernel_id": "i2_s-scalar-reference",
    "implementation": "scalar",
    "layout": "gguf_packed_i2_s"
  }
}
```

This proves BitNet inference only for the selected Apple backend and recorded
configuration. It is not Metal BitNet proof, QK256-on-Metal proof, Neural Engine
proof, or a performance claim.

### M4-015 - Steady Decode and Prefill Profile

Add timing evidence to the strict BitNet M4 proof without broadening the claim.
The profile receipt records a named profile, selected Apple backend, fallback
status, model/tokenizer identity, Apple machine context, prompt tokenization
time, prompt-prefix prefill timing, first-token decode timing, steady decode
timing, and sampling time.

The first profile target is a short CPU/NEON strict run:

```json
{
  "artifact_kind": "strict_bitnet_cpu_profile",
  "requested_backend": "apple-m4-cpu-neon",
  "selected_backend": "apple-m4-cpu-neon",
  "runtime_api": "cpu",
  "fallback_used": false,
  "profile": {
    "id": "smoke_4",
    "kind": "steady_decode_prefill",
    "phase": "decode",
    "machine_context_recorded": true,
    "prompt_prefill": {
      "exercised": true,
      "kv_cache_behavior": "prompt_prefix_prefilled_before_decode"
    },
    "decode": {
      "generated_tokens": 4,
      "warmup_tokens": 1,
      "steady_state_tokens": 3,
      "steady_state_tok_s": 0.0
    }
  },
  "timing": {
    "model_load_ms": 0.0,
    "tokenizer_load_ms": 0.0,
    "tokenize_ms": 0.0,
    "prefill_ms": 0.0,
    "first_token_decode_ms": 0.0,
    "decode_total_ms": 0.0,
    "decode_steady_state_tok_s": 0.0,
    "sampling_ms_per_token": 0.0
  }
}
```

M4-015 may claim only that timing is recorded for the named BitNet phase and
profile under captured machine context. It must not claim general M4
performance, Neural Engine execution, QK256 acceleration on Apple Silicon, or
Metal BitNet inference.

### M4-016 - Hot-Loop Allocation Audit

Add an opt-in allocation audit to the strict Apple profile path. The audit is
not a benchmark by itself; it records allocator counter deltas around prompt
prefill and decode steps so later performance claims can distinguish compute
timing from allocation overhead.

The receipt records:

```json
{
  "profile": {
    "allocation_audit": {
      "enabled": true,
      "method": "process_global_allocator_counter_delta",
      "claim_scope": "allocation counter deltas for the selected Apple BitNet profile only",
      "warmup_tokens": 1,
      "measured_tokens": 3,
      "per_token_alloc_count_delta": {
        "count": 4,
        "total": 0,
        "mean_per_token": 0.0
      },
      "per_token_alloc_bytes_delta": {
        "count": 4,
        "total": 0,
        "mean_per_token": 0.0
      },
      "decode": {
        "total": {
          "alloc_count_total": 0,
          "alloc_bytes_total": 0,
          "net_bytes_total": 0
        },
        "embed": {},
        "forward": {},
        "logits": {},
        "sample": {},
        "token_decode": {}
      }
    }
  }
}
```

The audit may claim only that per-token allocation behavior is measured or
bounded for the selected Apple BitNet path. It must not claim the decode path is
compute-bound unless allocation overhead is separated in the receipt, and it
must not claim QK256 acceleration, Neural Engine execution, Metal execution, or
general M4 performance.

### M4-017 - Metal I2_S Projection Residual Subgraph

Expand Metal kernel-family coverage with a tiny I2_S projection plus residual
subgraph. This keeps the proof below full inference while exercising more than a
single projection kernel:

```text
fixture_packed_i2_s
-> tiny_metal_i2s_projection_residual
-> residual_add
-> CPU/NEON parity
```

The live test is opt-in and writes a subgraph receipt:

```bash
BITNET_RUN_M4_METAL_I2S_PROJECTION_RESIDUAL=1 \
BITNET_M4_METAL_I2S_PROJECTION_RESIDUAL_RECEIPT=ci/hardware/apple-m4-mac-mini/<date>/metal-i2s-projection-residual.json \
cargo test --locked -p bitnet-kernels \
  --no-default-features \
  --features metal \
  --test metal_tiny_smoke tiny_m4_metal_i2s_projection_residual_subgraph_matches_cpu_reference_when_enabled -- --nocapture
```

The receipt records:

```json
{
  "artifact_kind": "subgraph",
  "graph_id": "tiny_i2s_projection_residual_subgraph",
  "requested_backend": "apple-m4-metal",
  "selected_backend": "apple-m4-metal",
  "runtime_api": "metal",
  "fallback_used": false,
  "bitnet": {
    "kernel_family": "i2_s",
    "execution_phase": "parity",
    "phase_scope": "projection_residual_subgraph"
  },
  "subgraph": {
    "kernel_id": "tiny_metal_i2s_projection_residual",
    "operations": ["packed_i2_s_matmul", "residual_add"],
    "full_bitnet_inference": false,
    "full_autoregressive_decode": false
  },
  "parity": {
    "reference_backend": "apple-m4-cpu-neon",
    "target_backend": "apple-m4-metal",
    "max_abs_error": 0.0,
    "mean_abs_error": 0.0
  }
}
```

M4-017 may claim only that this specific Apple Metal I2_S subgraph passes CPU
reference parity with `fallback_used=false`. It must not claim full Metal
inference, QK256 acceleration, Neural Engine execution, MPSGraph execution, or
general M4 performance.

### M4-018 - CLI and Package Surface Polish

Polish the package-facing Apple backend labels and failure-mode text after the
strict proof and kernel/subgraph expansion items. The CLI should present these
as separate proof lanes:

```text
apple-m4-metal     native Metal proof lane
apple-m4-mpsgraph  MPSGraph graph/reference proof lane
apple-m4-cpu-neon  Apple ARM64 CPU/NEON fallback and parity lane
```

`apple-m4-metal` must not alias to MPSGraph, Neural Engine, CPU fallback, or
generic `metal`. `apple-m4-mpsgraph` must not count as native Metal kernel
proof or Neural Engine proof without resolved-target evidence.
`apple-m4-cpu-neon` must not count as Metal acceleration.

Strict-mode failures should explain that unavailable Apple labels fail rather
than silently falling back. Non-strict fallback paths must remain receipt-backed
with `requested_backend`, `selected_backend`, `runtime_api`, `fallback_used`,
and `fallback_reason`.

Legacy CLI subcommands that do not emit Apple proof receipts should point users
to `bitnet run` for receipt-backed Apple M4 labels. The artifact path examples
remain:

```text
ci/hardware/apple-m4-mac-mini/<date>/strict-bitnet-cpu-neon-proof.json
ci/hardware/apple-m4-mac-mini/<date>/metal-i2s-parity.json
ci/hardware/apple-m4-mac-mini/<date>/metal-i2s-prefill-contribution.json
ci/hardware/apple-m4-mac-mini/<date>/metal-i2s-projection-residual.json
```

M4-018 may claim only that Apple backend CLI and package surfaces describe
supported backend labels and failure modes accurately. It must not claim new
Metal kernel execution, Neural Engine execution, or general M4 performance.

## Do Not

- Do not start with Apple Neural Engine inference claims.
- Do not count CPU fallback as Metal execution.
- Do not count MPSGraph smoke as native Metal kernel proof.
- Do not assume M4 Pro configuration from base M4 facts.
- Do not compare unified-memory results to discrete VRAM GPUs without memory context.

## Related Contract Docs

- `docs/hardware/HARDWARE_MATRIX.md`
- `docs/hardware/PROOF_STAGES.md`
- `docs/hardware/LANE_OWNERSHIP.md`
- `docs/hardware/BENCHMARK_PROTOCOL.md`
