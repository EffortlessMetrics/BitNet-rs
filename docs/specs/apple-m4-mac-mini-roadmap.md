# Apple M4 Mac mini Roadmap

## Purpose

This document defines the Apple M4 Mac mini validation lane for BitNet-rs. The M4 lane is Metal-first, with MPSGraph as a graph/reference lane and CPU/NEON as the fallback/parity lane.

Primary labels:

```text
apple-m4-metal
apple-m4-mpsgraph
apple-m4-cpu-neon
```

The first useful milestone is tiny Metal compute smoke with a receipt proving `selected_backend=apple-m4-metal` and `fallback_used=false`.

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

Receipts must record the actual chip, CPU/GPU core counts, unified memory size, and memory bandwidth. Do not assume base M4 when validating M4 Pro.

## Claim Boundary

- Metal device visibility is not Metal execution.
- Metal compute smoke is not CPU/Metal parity.
- CPU/Metal parity is not full inference.
- MPSGraph smoke is not handwritten Metal kernel proof.
- MPSGraph smoke is not Neural Engine proof unless the resolved execution target is receipt-backed.
- CPU fallback cannot count as Metal execution.
- Apple CPU/NEON fallback is not AVX2 or AVX-512.

## Runtime Paths

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
  "requested_backend": "apple-m4",
  "selected_backend": "apple-m4-metal",
  "runtime_api": "metal",
  "chip": "Apple M4",
  "gpu_cores": 10,
  "unified_memory": true,
  "memory_bandwidth_gbps": 120,
  "fallback_backend": null,
  "fallback_used": false
}
```

Minimum MPSGraph receipt:

```json
{
  "requested_backend": "apple-m4-mpsgraph",
  "selected_backend": "apple-m4-mpsgraph",
  "runtime_api": "mpsgraph",
  "resolved_target": "gpu|cpu|neural-engine|unknown",
  "fallback_used": false,
  "graph": {
    "name": "tiny_matmul",
    "shape_mode": "static"
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
- Rust toolchain.

## Work Plan

### M4-001 - Add Backend Lane

Docs/tracking only. Add M4 Metal, MPSGraph, and CPU/NEON lanes.

### M4-002 - Machine Profile

Collect macOS, Apple chip, GPU cores, unified memory, Metal visibility, and Rust toolchain data.

### M4-003 - Backend Identity

Preserve Apple M4 Metal, MPSGraph, and CPU/NEON requested and selected backend identities.

### M4-004 - Metal Probe

Report Metal device visibility, Apple chip, GPU family where available, unified memory, and macOS version.

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
