# bitnet-rs Roadmap

## Where We Are

bitnet-rs is a pre-alpha (v0.2.1-dev) Rust inference engine for 1-bit BitNet LLMs. CPU inference
with SIMD optimization (AVX2/AVX-512/NEON) works end-to-end. GPU backends are scaffolded but not
yet validated.

**What works today:**

- CPU inference with I2_S BitNet32-F16 and QK256 quantization formats
- AVX2/AVX-512/NEON SIMD kernels with runtime dispatch
- GGUF and SafeTensors model loading with automatic format detection
- Interactive chat and one-shot Q&A with 59+ prompt template variants
- Cross-validation framework against C++ reference (per-token logits comparison)
- Honest-compute receipts (schema v1.0.0, 8 validation gates)
- SafeTensors-to-GGUF export with F16 LayerNorm preservation
- CI test-target gating: all optional-dep `[[test]]` entries guarded by `required-features`
- Receipt verification workflow with malformed-receipt fallback handling

**By the numbers:**

- ~200 workspace crates, 2,500+ .rs files
- ~58,700 `#[test]` annotations (~2,800 intentionally ignored with justification)
- 790 proptest macros, 1,442 snapshot files, 109 fuzz targets
- 44 bench files, 45 CI workflows

## Current Limitations

- **QK256 performance**: Scalar kernels only (~0.1 tok/s for 2B models). Not suitable for
  production.
- **No validated GPU inference**: CUDA is furthest along but receipt validation is still pending.
  Metal, Vulkan, OpenCL, ROCm are scaffolded stubs.
- **Model quality**: microsoft-bitnet-b1.58-2B-4T produces non-sensical output in some
  configurations (model limitation, not inference bug).
- **Server incomplete**: Health endpoints work; inference endpoints have TODOs.
- **Large scaffold surface**: ~2,800 ignored tests are TDD scaffolds, resource-gated, or slow.

## Completed: CI Foundation (v0.2.1-dev)

Merged in this dev cycle — no further action needed:

- [x] Gate all optional-dep `[[test]]` entries behind `required-features` in
      `crates/bitnet-kernels/Cargo.toml` (7 targets across Metal, OpenCL, CPU)
- [x] Harden `verify-receipts.yml` fallback to detect and replace malformed generated receipts
      missing `schema_version` — preserves raw output as `ci/inference.raw.json`
- [x] Replace blanket `#[allow(dead_code)]` on `PagedCacheBlock` with
      `#[cfg_attr(not(target_arch = "aarch64"), allow(dead_code))]`
- [x] SHA-pin all GitHub Actions; enforce `--locked` on all cargo invocations in CI
- [x] All `#[ignore]` attributes carry a justification string (enforced by pre-commit hook)

## Near-term: v0.2.1

Active workstreams, no date commitments:

- [ ] QK256 AVX2 nibble-LUT + FMA tiling (target 3× over scalar)
- [ ] Zero-alloc dense forward pass (BlockWorkspace + ping-pong buffers)
- [ ] Apple Silicon NEON kernel parity (softmax, attention, matmul)
- [ ] AVX2+FMA log-softmax and vectorized softmax exp
- [ ] Unblock TDD placeholder tests — work through `#[ignore = "TDD scaffold: ..."]` backlog
- [ ] `xtask benchmark --json` emits `schema_version` natively (removes workflow fallback need)
- [ ] Documentation cleanup: README accuracy, quickstart tested end-to-end

## Medium-term: v0.3.0

- [ ] QK256 inference usable for validation (target 1+ tok/s on 2B)
- [ ] CUDA receipt validation end-to-end (first validated GPU backend)
- [ ] Server MVP: wire inference endpoints, fix remaining TODOs
- [ ] KV cache optimization wired into generation loop
- [ ] Reduce workspace crate count (consolidate SRP microcrates where boundaries are artificial)
- [ ] `bitnet-wasm` compilation target validated (browser inference proof-of-concept)

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
