# bitnet-rs Roadmap

## Roadmap Principles

This roadmap is a planning and sequencing document for bitnet-rs contributors.
It is not a release promise. Dates are intentionally omitted unless a tracked
release branch or signed tag adds them later.

Work should be planned in small PRs that keep the project honest about what is
validated:

- Prefer measured receipts over aspirational performance claims.
- Keep CPU correctness, tokenizer parity, and GGUF loading compatibility ahead
  of backend breadth.
- Treat GPU backends as experimental until they emit the same validation
  receipts as CPU.
- Keep default features empty; document every required feature flag in commands
  and CI.
- Avoid checking model binaries into git; use reproducible model provisioning
  instead.

## Where We Are

bitnet-rs is a pre-alpha (v0.2.1-dev) Rust inference engine for 1-bit BitNet
LLMs. CPU inference with SIMD optimization (AVX2/AVX-512/NEON) works end-to-end.
GPU backends are scaffolded but not yet validated.

**What works today:**

- CPU inference with I2_S BitNet32-F16 and QK256 quantization formats
- AVX2/AVX-512/NEON SIMD kernels with runtime dispatch
- GGUF and SafeTensors model loading with automatic format detection
- Interactive chat and one-shot Q&A with 59+ prompt template variants
- Cross-validation framework against C++ reference (per-token logits comparison)
- Honest-compute receipts (schema v1.0.0, 8 validation gates)
- SafeTensors-to-GGUF export with F16 LayerNorm preservation
- CI test-target gating: all optional-dep `[[test]]` entries guarded by
  `required-features`
- Receipt verification workflow with malformed-receipt fallback handling

**By the numbers:**

- ~200 workspace crates, 2,500+ .rs files
- ~58,700 `#[test]` annotations (~2,800 intentionally ignored with
  justification)
- 790 proptest macros, 1,442 snapshot files, 109 fuzz targets
- 44 bench files, 45 CI workflows

## Current Limitations

- **QK256 performance**: Scalar kernels only (~0.1 tok/s for 2B models). Not
  suitable for production.
- **No validated GPU inference**: CUDA is furthest along but receipt validation
  is still pending. Metal, Vulkan, OpenCL, ROCm are scaffolded stubs.
- **Model quality**: microsoft-bitnet-b1.58-2B-4T produces non-sensical output
  in some configurations (model limitation, not inference bug).
- **Server incomplete**: Health endpoints work; inference endpoints have TODOs.
- **Large scaffold surface**: ~2,800 ignored tests are TDD scaffolds,
  resource-gated, or slow.

## Completed: CI Foundation (v0.2.1-dev)

Merged in this dev cycle — no further action needed:

- [x] Gate all optional-dep `[[test]]` entries behind `required-features` in
      `crates/bitnet-kernels/Cargo.toml` (7 targets across Metal, OpenCL, CPU)
- [x] Harden `verify-receipts.yml` fallback to detect and replace malformed
      generated receipts missing `schema_version` — preserves raw output as
      `ci/inference.raw.json`
- [x] Replace blanket `#[allow(dead_code)]` on `PagedCacheBlock` with
      `#[cfg_attr(not(target_arch = "aarch64"), allow(dead_code))]`
- [x] SHA-pin all GitHub Actions; enforce `--locked` on all cargo invocations in
      CI
- [x] All `#[ignore]` attributes carry a justification string (enforced by
      pre-commit hook)

## Near-term Release Train: v0.2.1 Stabilization

The v0.2.1 track should make the current CPU path easier to validate and harder
to regress. It should not expand public API surface unless the API is required
for validation or benchmark receipts.

### 1. QK256 CPU Throughput

**Goal:** Make QK256 inference useful enough for repeatable validation runs on
developer hardware.

**Planned work:**

- [ ] Add AVX2 nibble-LUT unpacking with FMA-friendly tile layout.
- [ ] Add scalar/AVX2 differential tests over odd sizes, tails, and alignment
      offsets.
- [ ] Add deterministic microbench receipts for QK256 matmul tiles.
- [ ] Add runtime dispatch telemetry so receipts record the selected kernel
      path.
- [ ] Document hardware expectations for AVX2, AVX-512, and NEON paths.

**Acceptance gates:**

- QK256 AVX2 path is at least 3× faster than scalar on the project benchmark
  fixture.
- Differential tests pass with `--no-default-features --features cpu`.
- Receipts include CPU brand, detected features, kernel name, tensor shape, and
  measured latency.

### 2. Zero-Allocation Dense Forward Pass

**Goal:** Remove avoidable per-token allocations from the hot decode loop.

**Planned work:**

- [ ] Introduce a `BlockWorkspace`/scratch-buffer ownership model for dense
      layers.
- [ ] Use ping-pong activation buffers for layer-to-layer handoff.
- [ ] Add allocation counters around prefill and decode benchmark paths.
- [ ] Document lifetime and aliasing invariants for workspace reuse.

**Acceptance gates:**

- Decode step for the 2B fixture performs zero heap allocations after warmup.
- Allocation regression test fails if new hot-path allocations are introduced.
- Throughput receipt reports allocation count, peak scratch bytes, and token
  count.

### 3. Apple Silicon NEON Parity

**Goal:** Keep ARM64 developer and CI paths numerically aligned with x86 CPU
validation.

**Planned work:**

- [ ] Fill NEON coverage gaps for softmax, attention, and matmul helpers.
- [ ] Add NEON-vs-scalar differential fixtures for representative tensor shapes.
- [ ] Record NEON feature detection in honest-compute receipts.
- [ ] Ensure non-aarch64 builds keep a clean dead-code baseline without blanket
      allows.

**Acceptance gates:**

- NEON differential suite passes on Apple Silicon hardware.
- Receipts identify NEON execution instead of reporting generic CPU.
- Unsupported hosts skip hardware-only tests with explicit skip reasons.

### 4. Sampling and Softmax Correctness

**Goal:** Make decode output reproducible and explainable under every supported
sampling mode.

**Planned work:**

- [ ] Add AVX2+FMA log-softmax and vectorized softmax-exp kernels.
- [ ] Expand top-k, top-p, temperature, greedy, and repetition-penalty snapshot
      coverage.
- [ ] Add deterministic seed handling to CLI and server request paths.
- [ ] Document exact reproducibility limits across scalar/SIMD backends.

**Acceptance gates:**

- Sampling snapshots cover greedy, nucleus, temperature, and repetition penalty
  combinations.
- CLI and server can emit receipts that include seed, sampler config, and stop
  reason.
- Vectorized softmax stays within documented tolerance of scalar reference.

### 5. Test Scaffold Burn-down

**Goal:** Convert ignored TDD scaffolds into executable tests or explicitly
tracked backlog items.

**Planned work:**

- [ ] Categorize ignored tests by resource gate, TDD scaffold, slow-path, and
      hardware-only reason.
- [ ] Convert high-value tokenizer, GGUF loader, and receipt tests first.
- [ ] Add a dashboard or generated summary for ignored-test counts by crate.
- [ ] Require new `#[ignore]` entries to name an owner, reason, and unblock
      condition.

**Acceptance gates:**

- Ignored-test count trends downward in release notes.
- New ignores without justification fail pre-commit and CI guard checks.
- Resource-gated tests expose a documented local command for manual execution.

### 6. Documentation and Quickstart Accuracy

**Goal:** Ensure the README and quickstart do not overstate backend readiness or
omit required flags.

**Planned work:**

- [ ] Test the CPU quickstart from a clean checkout.
- [ ] Move stale implementation claims into archive docs or update them with
      current status.
- [ ] Add a model-provisioning guide that avoids committing model binaries.
- [ ] Add troubleshooting for toolchain, feature flags, and missing model paths.

**Acceptance gates:**

- Quickstart commands are copy-pasteable and use `--locked` plus explicit
  feature flags.
- README status matches the validation receipts available in-tree.
- GPU language is clearly marked experimental until GPU receipts are green.

## Validation Infrastructure Track

The validation track cuts across releases because it determines when a backend
or feature can be called supported.

### Phase A: Reproducible Model Provisioning

**Goal:** Download, verify, and cache models without adding binaries to the
repository.

**Deliverables:**

- `xtask fetch-models` or equivalent model-fetch command.
- Lockfile containing URL, file names, size, SHA-256, license notes, and
  intended test tier.
- Atomic cache writes under a user cache directory.
- CI-light and integration model tiers.

**Acceptance gates:**

- Re-running the command is idempotent.
- Corrupt or partial downloads are detected before use.
- CI can opt into small fixtures without pulling multi-GB models.

### Phase B: Real C++ Reference Bridge

**Goal:** Replace mock or placeholder parity layers with a gated
bitnet.cpp/llama.cpp-compatible reference bridge.

**Deliverables:**

- Feature-gated C++ bridge that links only when the reference checkout is
  present.
- Safe Rust ownership wrappers for model, context, tokenization, eval, and
  teardown.
- Graceful skips when `BITNET_CPP_DIR` or required artifacts are missing.

**Acceptance gates:**

- Tokenization parity checks compare exact token IDs for fixed prompts.
- Single-step logits parity reports cosine similarity and max absolute error.
- Tests never require local C++ artifacts unless the reference feature is
  enabled.

### Phase C: Parity Receipts

**Goal:** Emit durable, reviewable evidence for Rust-vs-reference correctness.

**Deliverables:**

- Per-token parity receipts containing prompt, token IDs, model hash, commit,
  backend, and metrics.
- Multi-step greedy decode comparison for short prompts.
- Receipt verifier checks for schema version, required fields, and numeric
  thresholds.

**Acceptance gates:**

- Receipts fail validation if required metadata or thresholds are missing.
- CPU parity is reproducible on at least one documented model fixture.
- CI publishes raw and normalized receipts for debugging.

### Phase D: Real Performance Baselines

**Goal:** Replace fabricated or static benchmark values with measured,
reproducible baselines.

**Deliverables:**

- Benchmark receipts for prefill latency, decode latency, tokens/sec, and
  memory.
- `xtask gen-baselines` or equivalent command to regenerate baseline JSON from
  receipts.
- Regression checks with documented tolerance and hardware labels.

**Acceptance gates:**

- Performance gates compare only compatible hardware/backend labels.
- Baseline updates require receipt evidence in the PR.
- Regression output names the exact metric and threshold that failed.

## Medium-term Release Train: v0.3.0

The v0.3.0 track should turn validated CPU inference into a usable developer
preview and graduate at least one GPU path from scaffold to receipt-backed
experimental support.

### 1. QK256 Validation-Usable Inference

- [ ] Reach 1+ tok/s on a documented 2B-model CPU validation host.
- [ ] Add prefill/decode split metrics to CLI receipts.
- [ ] Keep scalar fallback correct and available for debugging.
- [ ] Publish benchmark methodology and fixture details.

### 2. First Validated GPU Backend

- [ ] Complete CUDA receipt validation end-to-end before broadening backend
      claims.
- [ ] Add CPU-vs-CUDA numerical parity checks for fixed prompts and short decode
      windows.
- [ ] Record CUDA device name, driver/runtime versions, and kernel variant in
      receipts.
- [ ] Keep Metal, Vulkan, OpenCL, and ROCm labeled scaffolded until equivalent
      receipts exist.

### 3. Server MVP

- [ ] Wire inference endpoints to the validated generation loop.
- [ ] Add request validation, cancellation, timeout, and max-token enforcement.
- [ ] Add streaming and non-streaming response tests.
- [ ] Include honest-compute receipt metadata in server responses or sidecar
      logs.
- [ ] Document API compatibility and non-goals for OpenAI-compatible endpoints.

### 4. KV Cache Integration

- [ ] Wire KV cache optimization into the generation loop.
- [ ] Add correctness tests for prompt prefill, continuation, and cache reset.
- [ ] Add memory accounting to receipts.
- [ ] Benchmark long-prompt behavior before and after cache changes.

### 5. Workspace Consolidation

- [ ] Identify SRP microcrates that do not need independent publishing
      boundaries.
- [ ] Consolidate only where it reduces compile time or cognitive overhead.
- [ ] Keep public crate moves behind compatibility shims or explicit
      breaking-change notes.
- [ ] Track compile-time and dependency-graph changes before and after
      consolidation.

### 6. WebAssembly Proof of Concept

- [ ] Validate `bitnet-wasm` compilation with the minimal CPU feature set.
- [ ] Add a tiny browser or wasm-bindgen smoke test.
- [ ] Document model-size and performance constraints clearly.
- [ ] Keep browser inference marked proof-of-concept until real model loading is
      practical.

## Longer-term Themes

These tracks are intentionally less prescriptive. They should be promoted into
release-train work only after the validation infrastructure can prove
correctness and performance.

### Backend Expansion

- Metal backend validation after CUDA proves the receipt flow.
- Vulkan backend validation for portable desktop GPU experiments.
- OpenCL/ROCm reassessment after CUDA/Metal/Vulkan priorities are clear.
- Multi-GPU and tensor parallelism only after single-device receipts are
  reliable.

### Runtime and Memory

- Paged KV cache with eviction policies.
- Better long-context memory planning and receipt reporting.
- NUMA-aware CPU execution if benchmark receipts show a need.
- Disk-backed or mmap-friendly model loading improvements for large fixtures.

### Language Bindings and Packaging

- Python bindings via PyO3 for local experimentation.
- C ABI stabilization only after CLI/server APIs settle.
- Binary packaging once the CPU quickstart is stable and reproducible.
- Nix and container recipes that mirror the documented release commands.

## Future Directions

These are aspirational and may change:

- Metal and Vulkan backend validation (after CUDA is green)
- Paged KV cache with eviction policies
- Multi-GPU inference and tensor parallelism
- Python bindings via PyO3
- WebAssembly target for browser inference (beyond proof-of-concept)

## Non-goals

These are explicitly not planned:

- General-purpose LLM inference (this is BitNet-specific)
- Mobile deployment targets
- SaaS or hosted inference service
- Distributed inference across machines
