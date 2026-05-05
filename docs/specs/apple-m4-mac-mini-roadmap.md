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
