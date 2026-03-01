# Intel GPU / A770 — Development Roadmap

> Phased plan for bringing the BitNet-rs OpenCL backend from CPU-reference
> scaffolding to production-grade inference on Intel Arc GPUs.

---

## Phase 1 — Foundation (Current)

**Goal:** Establish module structure, CPU-reference kernels, and CI
infrastructure so that all subsequent GPU work can be developed, tested, and
reviewed safely.

| Milestone                        | Status |
|----------------------------------|--------|
| `opencl_*` module scaffold       | ✅      |
| CPU-reference implementations    | ✅      |
| `.cl` kernel sources             | ✅      |
| PipelineConfig + validation      | ✅      |
| A770-tuned buffer alignment      | ✅      |
| WorkSizeConfig defaults          | ✅      |
| BackendRegistry + Dispatcher     | ✅      |
| SPIR-V compilation pipeline      | ✅      |
| `ci-core.yml` CPU-reference CI   | ✅      |
| `gpu-smoke.yml` weekly schedule  | ✅      |
| Architecture documentation       | ✅      |

**Exit criteria:** All CPU-reference tests pass in CI. Module interfaces are
stable enough that kernel implementations can be swapped in without API changes.

---

## Phase 2 — Hardware Kernel Execution

**Goal:** Execute real OpenCL kernels on Intel Arc hardware, validate
correctness against CPU references, and wire end-to-end inference through the
GPU pipeline.

| Milestone                                | Status |
|------------------------------------------|--------|
| OpenCL context creation on A770          | 🔲      |
| `opencl_embedding` — GPU dispatch        | 🔲      |
| `opencl_attention` — GPU SDPA            | 🔲      |
| `opencl_ffn` — GPU gated FFN             | 🔲      |
| `opencl_quantized` — GPU I2_S matmul     | 🔲      |
| `opencl_pipeline` — full GPU forward pass| 🔲      |
| Numerical validation pass (FP32 ± 1e-5)  | 🔲      |
| Hardware-gated CI (A770 self-hosted runner)| 🔲     |

**Key risks:**

* Driver-level precision differences (especially FP16 accumulation).
* Memory-transfer overhead dominating small-batch latency.
* Kernel launch overhead for per-layer dispatch.

**Mitigation:** Batch operations where possible; use pinned/zero-copy buffers
for streaming token I/O; fuse kernels (e.g., RMSNorm + matmul) when profiling
shows launch overhead > 5 %.

---

## Phase 3 — Performance Optimisation

**Goal:** Achieve competitive tokens/s on the A770 by exploiting tiled matmul,
FP16 mixed precision, and SLM-backed attention.

| Milestone                                     | Status |
|------------------------------------------------|--------|
| Tiled matmul (16×16 FP16 tiles, SLM)          | 🔲      |
| FP16 mixed-precision forward pass              | 🔲      |
| Subgroup shuffle attention                     | 🔲      |
| Fused RMSNorm + projection kernel              | 🔲      |
| INT8 DP4A quantised matmul (Xe-HPG dp4a)      | 🔲      |
| KV-cache paged attention on GPU                | 🔲      |
| Kernel binary caching (opencl_cache)           | 🔲      |
| `intel_gpu_top` profiling integration          | 🔲      |
| Benchmark suite (Criterion + receipt export)   | 🔲      |

**Performance targets (2B-parameter BitNet model on A770):**

| Metric          | Phase 2 (unoptimised) | Phase 3 target |
|-----------------|-----------------------|----------------|
| Tokens/s        | ~1–5                  | ≥ 20           |
| Time-to-first   | ~500 ms               | < 200 ms       |
| VRAM usage      | ~6 GB                 | ≤ 4 GB         |

---

## Phase 4 — Production Readiness

**Goal:** Harden the backend for real-world deployment: monitoring, graceful
degradation, multi-model serving, and documentation.

| Milestone                                       | Status |
|--------------------------------------------------|--------|
| `opencl_telemetry` — structured kernel metrics   | 🔲      |
| `opencl_profiling` — per-kernel wallclock/EU stats| 🔲     |
| `opencl_recovery` — error handling + GPU reset   | 🔲      |
| Graceful fallback on driver crash                | 🔲      |
| Context-pool multi-model serving                 | 🔲      |
| Receipt artefact export (schema v1.0.0)          | 🔲      |
| Production benchmarks published                  | 🔲      |
| `bitnet-server` GPU backend wiring               | 🔲      |
| Monitoring dashboard (Prometheus metrics)        | 🔲      |

**Exit criteria:** A770 backend can serve ≥ 20 tok/s sustained for a 2B model,
with automated nightly performance regression tests, structured telemetry, and
graceful fallback to CPU on any GPU failure.

---

## Timeline (indicative)

```
Phase 1 ████████████████  ← current
Phase 2          ░░░░░░░░░░░░░░░░
Phase 3                    ░░░░░░░░░░░░░░░░
Phase 4                              ░░░░░░░░░░░░░░░░
        ──────────────────────────────────────────────►
        now       +4 wk     +8 wk     +12 wk    +16 wk
```

---

## See Also

* [ARCHITECTURE.md](ARCHITECTURE.md) — Backend architecture reference
* [SETUP.md](SETUP.md) — Driver installation and verification
