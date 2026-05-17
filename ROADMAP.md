# bitnet-rs Roadmap

This roadmap is the coordination document for bitnet-rs work after the v0.2.1-dev
CI foundation. It is intentionally evidence-first: work is considered done when
it has a reproducible command, receipt, benchmark, or cross-validation artifact,
not when an API is merely scaffolded.

## Product North Star

bitnet-rs is a Rust inference engine for 1-bit BitNet LLMs. The project should
make BitNet inference reproducible, inspectable, and fast enough to compare
honestly with the Microsoft BitNet.cpp reference path.

The north-star outcome is:

1. load the official BitNet GGUF artifacts without hidden conversion steps,
2. run deterministic CPU inference with strict tokenizer and loader validation,
3. emit receipts that describe model, tokenizer, backend, kernel, and sampling
   decisions,
4. cross-check logits and generated tokens against the C++ reference path, and
5. graduate accelerator backends only after receipt-backed parity is available.

## Where We Are

bitnet-rs is a pre-alpha (`v0.2.1-dev`) Rust inference engine. CPU inference with
SIMD optimization works end-to-end for diagnostic use. GPU backends are present
as scaffolded surfaces, but they are not yet validated as production answer
paths.

**What works today:**

- CPU inference with I2_S BitNet32-F16 and QK256 quantization formats.
- AVX2, AVX-512, and NEON SIMD kernel surfaces with runtime dispatch.
- GGUF and SafeTensors model loading with automatic format detection.
- Interactive chat and one-shot Q&A paths with 59+ prompt template variants.
- Cross-validation framework against the C++ reference path, including per-token
  logits comparison.
- Honest-compute receipts with schema versioning and validation gates.
- SafeTensors-to-GGUF export with F16 LayerNorm preservation.
- CI test-target gating for optional-dependency `[[test]]` targets through
  `required-features`.
- Receipt verification workflow with malformed-receipt fallback handling.

**By the numbers:**

- Roughly 200 workspace crates and 2,500+ Rust source files.
- Tens of thousands of test annotations, including a large intentionally ignored
  scaffold/resource-gated backlog.
- Hundreds of property tests, snapshot artifacts, fuzz targets, benchmark files,
  and CI workflow entries.

## Current Limitations

The main risks are correctness proof, kernel speed, and backend truthfulness.
These limitations should stay visible in release notes and user-facing docs until
closed with evidence.

- **QK256 performance:** scalar paths remain too slow for production-style 2B
  model use. SIMD tiling and allocation removal are the primary CPU performance
  blockers.
- **No validated GPU answer path:** CUDA is the furthest along, but GPU receipt
  validation and strict CPU/CUDA parity are not complete. Metal, Vulkan, OpenCL,
  and ROCm remain scaffolded or exploratory.
- **Model quality ambiguity:** poor generations can be caused by model artifact,
  tokenizer/template mismatch, loader interpretation, sampling configuration, or
  kernel defects. The roadmap prioritizes first-divergence evidence before
  throughput claims.
- **Server incompleteness:** health endpoints exist, but inference endpoints and
  production-serving behavior need dedicated completion and validation.
- **Large scaffold surface:** ignored tests and microcrates represent useful
  design intent, but they also create maintenance cost until promoted, archived,
  or removed.

## Operating Principles

All roadmap items should follow these rules:

1. **Correctness before speed.** Throughput improvements are not accepted unless
   parity gates still pass.
2. **Receipts before claims.** Backend, kernel, tokenizer, model, and sampling
   claims need machine-readable evidence.
3. **Strict paths stay strict.** Loader and tokenizer strict-mode failures should
   stay actionable rather than silently falling back.
4. **One workstream per PR where possible.** Do not mix MSRV/toolchain policy,
   kernel math, tokenizer semantics, server behavior, and docs cleanup in one
   change.
5. **Scaffolds must converge.** TDD placeholders should be promoted to real
   checks, explicitly resource-gated, or archived with rationale.
6. **CI economics matter.** Cheap static and targeted checks should run on PRs;
   expensive mutation, fixture, and hardware lanes should run where they add
   signal.

## Roadmap Overview

| Horizon | Theme | Exit signal |
|---|---|---|
| v0.2.1 stabilization | CPU correctness, CI hygiene, docs truthfulness | strict CPU run and receipt validation are reproducible on a fresh checkout |
| v0.2.x performance wave | QK256 CPU performance and allocation control | benchmark JSON shows repeatable kernel/generation improvement without parity regression |
| v0.3.0 validation release | first validated accelerator and server MVP | CUDA or another accelerator has receipt-backed parity; server inference has tests |
| v0.4.0 packaging release | consumable CLI/server/library workflows | release artifacts, examples, and compatibility docs are tested end-to-end |
| v1.0.0 stability | stable public API and support policy | semver, MSRV, security, and platform support policies are enforced |

## v0.2.1 Stabilization

Goal: make the current CPU answer path and CI foundation easy to verify, easy to
explain, and hard to misrepresent.

### Workstreams

- [ ] Keep `cargo build --locked --no-default-features --features cpu` green on
      clean Linux runners.
- [ ] Keep `cargo clippy --locked --workspace --all-targets --no-default-features
      --features cpu -- -D warnings` green or document scoped exceptions.
- [ ] Ensure `xtask benchmark --json` emits `schema_version` natively so CI no
      longer needs malformed-receipt fallback repair.
- [ ] Refresh README and quickstart commands against the current CLI flags,
      feature flags, and model artifact paths.
- [ ] Promote or retire the highest-value `#[ignore = "TDD scaffold: ..."]`
      tests in loader, tokenizer, quantization, and generation crates.
- [ ] Add a small deterministic CPU smoke corpus that records prompt, tokenizer,
      template, seed/sampling configuration, output tokens, and receipt path.

### Acceptance Gates

- `cargo fmt --all -- --check` passes.
- CPU build and targeted CPU tests pass with locked dependencies.
- A strict-loader, strict-tokenizer CLI run emits a schema-versioned receipt.
- Documentation contains no production-readiness claim for unvalidated GPU paths.

## v0.2.x CPU Performance Wave

Goal: make CPU BitNet inference fast enough to support practical validation loops
while preserving strict parity evidence.

### Workstreams

- [ ] Implement QK256 AVX2 nibble-LUT and FMA tiling behind runtime dispatch.
- [ ] Add AVX2+FMA log-softmax and vectorized softmax exponentiation paths.
- [ ] Introduce or complete zero-allocation dense forward-pass workspaces,
      including ping-pong activation buffers.
- [ ] Wire KV-cache optimization into the generation loop with receipts that
      expose cache configuration.
- [ ] Add benchmark baselines for scalar, AVX2, AVX-512, and NEON paths where
      hardware is available.
- [ ] Report tokens/sec, first-token latency, peak RSS, model hash, tokenizer
      hash, backend, kernel family, and CPU feature set in benchmark JSON.

### Acceptance Gates

- Kernel benchmarks show repeatable improvement over scalar baselines on the same
  host.
- Cross-validation does not regress for touched quantization, logits, sampling,
  or generation surfaces.
- Allocation-sensitive tests or benchmarks demonstrate reduced hot-path
  allocation.
- Benchmark reports are explicit about hardware and do not generalize from one
  machine to all CPUs.

## v0.3.0 Validation Release

Goal: validate the first accelerator path and make the server useful for local,
receipt-backed inference experiments.

### Workstreams

- [ ] Complete CUDA receipt validation end-to-end, including backend identity,
      device identity, kernel identity, fallback status, and parity status.
- [ ] Establish strict CPU/CUDA parity for a deterministic prompt corpus before
      advertising CUDA as an answer path.
- [ ] Finish server inference endpoints with request validation, error mapping,
      cancellation behavior, and receipt return/download support.
- [ ] Add server integration tests that cover health, generation, invalid model
      paths, invalid tokenizer paths, and malformed request bodies.
- [ ] Keep Metal, Vulkan, OpenCL, ROCm, and NPU surfaces clearly marked as
      scaffolded or experimental until equivalent receipts exist.
- [ ] Validate `bitnet-wasm` as a proof-of-concept target if it can share the
      same tokenizer, loader, and receipt expectations.

### Acceptance Gates

- At least one accelerator backend produces receipt-backed outputs that match the
  CPU reference policy for the selected corpus.
- Server MVP can run a local generation request and return a receipt without
  relying on undocumented environment state.
- GPU docs identify exact hardware, driver/runtime versions, feature flags, and
  fallback behavior.

## v0.4.0 Packaging and Ecosystem Release

Goal: make the project easier to consume without weakening the pre-production
truthfulness model.

### Workstreams

- [ ] Publish tested CLI examples for model download, strict inference,
      benchmarking, receipt verification, and cross-validation.
- [ ] Add Docker or Nix workflows for repeatable CPU validation environments.
- [ ] Stabilize C FFI headers and examples for the subset of APIs that have
      coverage and compatibility evidence.
- [ ] Reassess Python bindings after the CPU answer path and public API surface
      are less volatile.
- [ ] Consolidate artificial SRP microcrates where crate boundaries add build
      cost without ownership clarity.
- [ ] Create a compatibility matrix for Linux, macOS, Windows, CPU feature sets,
      and accelerator runtimes.

### Acceptance Gates

- A new contributor can reproduce the documented CPU smoke path from a fresh
  checkout.
- Packaging examples pin feature flags and do not rely on default features.
- Public examples include receipt output and verification steps.

## v1.0.0 Stability Track

Goal: define the stable surface area and operational policies needed before the
project is described as production-ready.

### Workstreams

- [ ] Define semver policy, deprecation policy, MSRV policy, and supported target
      matrix.
- [ ] Audit public APIs and split stable, experimental, and internal surfaces.
- [ ] Complete unsafe-code documentation and memory-safety review for core
      loader, quantization, kernel, and FFI surfaces.
- [ ] Add release-readiness gates for security audit results, vulnerability
      handling, mutation/readiness evidence, and benchmark regression reports.
- [ ] Publish migration guides for any breaking API or CLI changes.

### Acceptance Gates

- Stable APIs have examples, tests, and migration policy.
- Release candidates include security, correctness, and benchmark evidence.
- Unsupported or experimental backends cannot be mistaken for supported paths in
  docs, receipts, or CLI output.

## Hardware Validation Tracks

Hardware work should be tracked as evidence lanes, not marketing claims.

| Platform | Primary question | Required evidence |
|---|---|---|
| Intel 258V CPU | Reference CPU behavior and AVX2 diagnostics | strict CPU receipts, kernel dispatch logs, benchmarks |
| i5-8250U CPU | Low-power dense SLM comparison | CPU receipts, latency/RSS reports |
| Ryzen 9950X3D | AVX-512 and high-performance CPU diagnostics | AVX-512 dispatch evidence, benchmark JSON |
| RTX 5070 Ti | CUDA packed BitNet validation | CPU/CUDA parity receipts, CUDA runtime details |
| Apple M4 | NEON, Metal, and MPSGraph exploration | NEON parity, Metal fallback status, receipts |
| Arc A770 | Discrete Intel GPU validation | OpenCL/OpenVINO backend receipts when available |
| Arc 140V | Lunar Lake iGPU validation | OpenCL/OpenVINO backend receipts when available |
| Intel NPU | Static-shape NPU feasibility | model-shape constraints, fallback status, receipts |

## Documentation Tracks

Documentation should converge toward four reader journeys:

1. **First diagnostic run:** install, download model, run strict CPU inference,
   verify receipt.
2. **Contributor validation:** run focused tests, cross-validation, benchmarks,
   and receipt checks for a touched subsystem.
3. **Backend validation:** prove a CPU, CUDA, Metal, OpenCL, Vulkan, ROCm, NPU,
   or WASM path without hiding fallback.
4. **API consumption:** use CLI, server, library, C FFI, Python, or WASM surfaces
   only where stability and evidence are adequate.

## Non-goals

The following are not planned unless the project scope changes explicitly:

- General-purpose LLM inference unrelated to BitNet.
- Hosted SaaS inference service.
- Distributed inference across machines before single-host parity and receipts
  are mature.
- Production-readiness claims for unvalidated accelerator backends.
- Silent fallback from strict validation failures.

## Definition of Done for Roadmap Items

A roadmap checkbox can be marked complete only when the PR or tracking issue
links to the relevant evidence:

- command transcript or CI run,
- receipt JSON or schema update,
- benchmark JSON with hardware metadata,
- cross-validation report,
- test coverage or ignored-test disposition,
- documentation update that explains user-visible behavior, and
- rollback or fallback notes for risky backend changes.
