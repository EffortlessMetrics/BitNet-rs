# Apple Silicon GPU/NPU Backend Roadmap for BitNet-rs

## Executive Summary

BitNet-rs currently ships CPU SIMD kernels and a CUDA backend. There is no Apple-specific execution backend for Metal GPU or Apple Neural Engine (ANE). This document defines a phased plan to add Apple acceleration support with two complementary paths:

1. **Metal/MPS path (priority first):** native Rust-side kernel backend for Apple GPU.
2. **Core ML / MLCompute path (optional advanced):** model-export + runtime path that can leverage ANE where supported.

This staged approach gives broad macOS coverage quickly while preserving a path to maximal Apple Silicon performance.

## Current-State Audit (Codebase)

### Existing backend surface
- `bitnet-kernels` currently exposes `cpu` and (feature-gated) `cuda` modules.
- `bitnet-inference` dispatches CPU/CUDA execution paths.
- Build and setup docs focus on CUDA (`docs/GPU_SETUP.md`, `docs/cuda-configuration-guide.md`).

### Key missing capabilities
- No `metal`/`apple` kernel module in `bitnet-kernels`.
- No Apple-specific device variants in shared device abstractions.
- No runtime device selection for Apple GPU/NPU.
- No model conversion/export pipeline from GGUF to Core ML format.

## Design Goals

1. **Preserve existing CPU/CUDA behavior** by default.
2. **Add Apple acceleration behind feature flags** to avoid platform regressions.
3. **Avoid private Apple APIs**; rely only on public frameworks (Metal/MPS/Core ML/MLCompute/Accelerate).
4. **Guarantee deterministic fallback** to CPU when Apple runtime/backend requirements are unmet.

## Proposed Architecture

## 1) Device model and probing

Extend common device representation with Apple-specific targets:
- `AppleGpu` (Metal/MPS)
- `AppleNeuralEngine` (via Core ML/MLCompute execution path)

Update device probing crate(s) to:
- detect Apple Silicon runtime availability,
- detect Metal capability at runtime,
- mark ANE path as available only when OS/runtime prerequisites are met.

## 2) Kernel/backend wiring

Add feature-gated module(s):
- `bitnet-kernels/src/metal.rs` (or `apple.rs` with internal Metal implementation)

Wire kernel selection to route:
- CPU → existing SIMD path,
- CUDA → existing GPU path,
- AppleGpu → new Metal/MPS kernels.

## 3) Inference runtime integration

Add inference dispatch branches:
- `device=apple-gpu` → Metal/MPS kernel execution
- `device=apple-ne` → Core ML / MLCompute execution path (if model/runtime available)

Expose CLI/config toggles for backend selection with documented fallback behavior.

## 4) Model conversion and packaging

Introduce tooling (`xtask` or CLI subcommand) for model export:
- GGUF → intermediate format (if needed) → `.mlmodel` / `.mlpackage`

Implementation note:
- Conversion can be orchestrated via `coremltools` invocation in build/export tooling.
- Runtime should auto-detect exported Core ML artifacts when `apple-ne` is selected.

## 5) Memory/layout considerations

Plan data layout adaptation for Apple execution paths:
- evaluate 4D-friendly shapes where required,
- use FP16-friendly execution where practical,
- minimize host↔device copies (buffer reuse / zero-copy where API allows).

## Feature Flag Plan

- `cuda` (existing)
- `metal` (new): compiles Apple GPU backend
- `coreml` (new, optional): compiles Core ML / MLCompute integration

Recommended defaults:
- Keep Apple backends **off by default** for cross-platform stability.
- Enable `metal` and/or `coreml` explicitly by platform or user choice.

## Delivery Roadmap

### Sprint 1 — Device abstraction and UX
- Add Apple device enum variants.
- Add runtime probe signals.
- Add CLI/config backend selectors and fallback messaging.

### Sprint 2 — Metal MVP
- Implement core ops (matmul/add/softmax) on Metal/MPS.
- Integrate with kernel selector + inference loop.
- Validate output parity against CPU.

### Sprint 3 — Core ML/ANE path
- Add model export pipeline and artifact discovery.
- Implement runtime execution path for exported model.
- Validate parity and fallback when unsupported ops appear.

### Sprint 4 — Hardening and performance
- Profile on Apple Silicon hardware.
- Optimize layouts and transfer overhead.
- Add CI build coverage for macOS feature compilation.

## Testing Strategy

### Correctness
- Unit tests per new kernel op against CPU reference.
- End-to-end generation parity checks (tolerance-based) across backends.

### Performance
- Benchmark tokens/sec and latency on representative M1/M2 hardware.
- Track memory footprint and first-token latency.

### Reliability
- Backend availability tests for feature/platform combinations.
- Fallback-path tests (missing runtime/model artifacts).

## Risks and Mitigations

- **Core ML operator coverage gaps** → keep Metal backend as independent fallback path.
- **Conversion pipeline fragility** → version-pin export tooling and add validation checks.
- **Platform drift (Apple SDK changes)** → isolate backend code behind thin adapters and feature gates.

## Success Criteria

1. Apple GPU backend runs end-to-end generation on supported macOS devices.
2. Optional Core ML path can execute compatible exported models.
3. CPU fallback remains correct and stable when Apple paths are unavailable.
4. CI verifies compilation and baseline tests for Apple feature flags on macOS.
