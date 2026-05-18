# CPU AVX-512 Implementation Plan

Status: Draft
Owner: BitNet-rs maintainers
Created: 2026-05-18
Linked proposal: n/a
Linked specs: `docs/specs/BITNET-SPEC-CPU-AVX512-KERNEL-CONTRACT.md`, `docs/specs/BITNET-SPEC-CPU-ISA-SELECTION.md`, `docs/specs/amd-9950x3d-cpu-roadmap.md`
Linked ADRs: n/a
Linked plan: n/a
Linked issues: n/a
Linked PRs: n/a
Support-tier impact: AVX-512 remains explicit, strict, and proof-gated until parity, counter, phase, and sustained receipts support narrower claims.
Policy impact: n/a

## Goal

Bring the AVX-512 CPU lane from detection and receipt-label evidence to real
first-class BitNet-rs AVX-512 execution on the 9950X3D. The lane must provide
strict fallback rejection, scalar parity, AVX2 comparison, exact receipt
counters, phase benchmarks, sustained-power/cache-domain evidence, and no
speedup overclaims.

## Current State

- `bitnet-cpu-detect` has AVX-512-tier detection, but detection is not kernel
  execution proof.
- `bitnet-quantization` exposes AVX2 feature plumbing and scalar/AVX2 QK256
  kernel IDs, but no AVX-512 quantization feature or stable AVX-512 QK256 IDs.
- Existing AVX-512-labeled receipts are useful answer evidence, but labels such
  as `i2_s-avx512-reference` do not prove optimized AVX-512 QK256 execution.
- Existing CPU/CUDA parity diagnostics may record generated-token agreement for
  cases while still carrying logits/top-k divergence and `speedup_claim=false`.

## Claim Rails

The AVX-512 lane must preserve these rails across every PR:

- AVX-512 detection is not AVX-512 kernel proof.
- AVX2 proof is not AVX-512 proof.
- AVX-512 execution is not speedup.
- AVX-512 microbench speed is not decode speed.
- Short boost is not sustained performance.
- CPU proof is not GPU, NPU, CUDA, OpenCL, OpenVINO, Metal, server, or general
  chat-quality proof.
- Strict requested AVX-512 must fail when unavailable.
- Auto-selection must not promote AVX-512 until parity, phase, and sustained
  receipts justify profile-scoped promotion.

## Work Items

### CPU-AVX512-000 - Docs/spec rails

Files:

- `docs/specs/BITNET-SPEC-CPU-AVX512-KERNEL-CONTRACT.md`
- `docs/specs/BITNET-SPEC-CPU-ISA-SELECTION.md`
- `plans/cpu-avx512/implementation-plan.md`
- `docs/bitnet/BITNET_KERNEL_MATRIX.md`
- `docs/specs/amd-9950x3d-cpu-roadmap.md`
- `docs/tracking/campaigns/cpu-proof/active.toml`

Acceptance:

- No runtime changes.
- AVX-512 claim boundary is explicit.
- Strict CPU ISA selection behavior is specified.
- Stable AVX-512 QK256 kernel IDs are reserved.
- The PR queue is encoded in this plan and the cpu-proof tracker.

Proof commands:

```bash
git diff --check
cargo run --locked -p xtask --no-default-features -- campaign check cpu-proof
cargo run --locked -p xtask --no-default-features -- campaign generate --check
```

### CPU-AVX512-001 - AVX-512 subfeature detection

Expose subfeature helpers without changing dispatch:

```rust
pub fn avx512_f_available() -> bool;
pub fn avx512_bw_available() -> bool;
pub fn avx512_vl_available() -> bool;
pub fn avx512_vnni_available() -> bool;
pub fn avx512_f_bw_available() -> bool;
pub fn avx512_f_bw_vl_available() -> bool;
pub fn avx512_bitnet_i8s_available() -> bool;
```

Acceptance:

- Non-x86 returns `false` without panic.
- Tests assert helper ordering and no-panic behavior.
- Receipts can serialize detected, required, and used subfeatures.
- Model behavior and QK256 dispatch are unchanged.

### CPU-AVX512-002 - Quantization feature plumbing

Add `bitnet-quantization` AVX-512 feature gates and a gated AVX-512 module
surface without implementing the production kernel yet.

Acceptance:

```bash
cargo check --locked -p bitnet-quantization --no-default-features --features cpu
cargo check --locked -p bitnet-quantization --no-default-features --features cpu,avx2
cargo check --locked -p bitnet-quantization --no-default-features --features cpu,avx512
```

### CPU-AVX512-003 - AVX-512 F32/no-scale QK256 GEMV

Add `qk256-avx512-f32-gemv` as the first AVX-512 smoke kernel. It decodes
packed two-bit codes to F32 weights, multiplies by an F32 activation vector, and
compares against the scalar no-scale QK256 oracle.

Acceptance:

- Scalar parity across rows 1, 2, 7, and 32.
- Column coverage for 256, 257, 300, 512, 513, and 1024.
- Pattern coverage for all-zero, all-two, repeating, and pseudorandom codes.
- Activation coverage for constant, ramp, centered, and pseudorandom vectors.
- Repeated-run equality.
- Strict unavailable AVX-512 request fails.
- No answer-corpus or speed claim changes.

### CPU-AVX512-004 - AVX-512 kernel selection metadata

Extend QK256 selection to support explicit AVX-512 requests while keeping `auto`
from blindly promoting AVX-512.

Acceptance:

- Explicit strict AVX-512 unavailable returns an error.
- Explicit non-strict AVX-512 unavailable records fallback.
- Explicit AVX-512 available selects the stable AVX-512 kernel ID.
- AVX2 behavior remains unchanged.
- Receipts expose requested and selected kernel IDs.

### CPU-AVX512-005 - Hot-path counters

Add receipt counters for F32 and scaled I8S QK256 hot paths:

```text
qk256_f32_scalar_invocations
qk256_f32_avx2_invocations
qk256_f32_avx512_invocations
qk256_i8s_scaled_scalar_invocations
qk256_i8s_scaled_avx2_invocations
qk256_i8s_scaled_avx512_invocations
```

Acceptance:

- Answer-corpus receipts include counters.
- Strict AVX-512 F32 proof can show AVX-512 invocation count greater than zero.
- No speed claim is made.

### CPU-AVX512-006 - Scaled I2_S-by-I8_S fixtures

Before implementing the scaled AVX-512 kernel, lock scalar inline-scale behavior
with reusable fixtures.

Acceptance:

- Fixture scales include 0.125, 0.5, and 1.0.
- Column coverage includes 1, 2, 127, 128, 129, 255, 256, 257, 512, and 1024.
- Code patterns include 0, 1, 2, 3, cyclic, and pseudorandom.
- Activation ranges include small, large, signed, and finite max-ish values.
- The scalar oracle is documented and stable.

### CPU-AVX512-007 - Scaled I2_S-by-I8_S AVX-512 GEMV

Add the baseline `qk256-avx512-i8s-scaled-gemv` kernel. It must mirror scalar
BitNet.cpp semantics first and optimize second.

Acceptance:

- Scalar-vs-AVX512 parity passes.
- Tail coverage passes.
- Repeated-run equality passes.
- Runtime detection is strict.
- No model-level auto promotion occurs.

### CPU-AVX512-008 - Inline-scale dispatch wiring

Route inline-scale QK256 decode through scaled AVX-512 only when explicitly
selected and available.

Acceptance:

- A real BitNet run can record `selected_kernel=qk256-avx512-i8s-scaled-gemv`.
- `qk256_i8s_scaled_avx512_invocations > 0`.
- `fallback_used=false`.
- Generated IDs remain unchanged versus scalar or AVX2 where expected.
- No speed claim is made.

### CPU-AVX512-009 - Strict AVX-512 answer-corpus refresh

Run the official Microsoft BitNet I2_S GGUF with CPU AVX-512 strict selection.

Acceptance:

- Real GGUF and strict tokenizer metadata are recorded.
- Fallback is false.
- Selected kernel is the stable AVX-512 scaled kernel ID.
- AVX-512 invocation counters are non-zero.
- Tiny corpus pass/fail and exact divergence reports are emitted.
- Scalar-vs-AVX512 and AVX2-vs-AVX512 answer parity artifacts are generated.
- No speedup claim is made.

### CPU-AVX512-010 - AVX-512 microbench receipts

Compare scalar, AVX2, AVX-512 F32, and AVX-512 I8S-scaled kernels across QK256
micro shapes.

Acceptance:

- Micro receipts record median, p95, bandwidth estimate, selected kernel, CPU
  features, thread count, and affinity when known.
- No model-level speedup claim is made.

### CPU-AVX512-011 - 9950X3D phase receipts

Record prefill, first-token, decode, and warm-session profiles for scalar,
AVX2, AVX-512, and CUDA diagnostic comparison when available.

Acceptance:

- Phase receipts are emitted.
- Cache-domain and core-affinity context are recorded or explicitly unavailable.
- No sustained claim is made.

### CPU-AVX512-012 - Sustained-power/cache-domain proof

Run a sustained decode or warm-session loop comparing AVX2 and AVX-512.

Acceptance:

- Sustained receipt records duration, power mode, cooling, temperature and clock
  data when available, thread affinity, CCD context, and selected kernels.
- Short boost no longer drives the claim.
- Promotion is allowed only where the sustained profile is better or non-
  regressing.

### CPU-AVX512-013 - Profile-specific auto promotion

Promote AVX-512 auto-selection only by profile after parity, answer, phase, and
sustained receipts pass.

Acceptance:

- `auto` does not blindly choose AVX-512.
- Profile-specific promotion ledger or validator proof exists.
- Users can still force scalar, AVX2, AVX-512, or AVX-512 VNNI.

## Default Validation Set

Use the scoped proof commands from each work item. Runtime AVX-512 PRs should
start from this validation set and narrow only with an explicit reason:

```bash
cargo fmt --all -- --check
cargo test --locked -p bitnet-cpu-detect --no-default-features --features avx512
cargo test --locked -p bitnet-quantization --no-default-features --features cpu,avx512 i2s_qk256
cargo test --locked -p bitnet-quantization --no-default-features --features cpu,avx512 --test qk256_avx512_parity_tests
cargo test --locked -p bitnet-quantization --no-default-features --features cpu,avx2,avx512 --test qk256_avx2_parity_tests
cargo check --locked -p bitnet-cli --no-default-features --features cpu,full-cli
cargo run --locked -p xtask --no-default-features -- campaign check cpu-proof
cargo run --locked -p xtask --no-default-features -- campaign generate --check
git diff --check
```
