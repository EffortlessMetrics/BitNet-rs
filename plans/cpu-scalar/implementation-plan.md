# CPU Scalar BitNet Implementation Plan

Status: proposed

Linked specs:

- [CPU scalar kernel contract](../../docs/specs/BITNET-SPEC-CPU-SCALAR-KERNEL-CONTRACT.md)
- [CPU scalar hot-path contract](../../docs/specs/BITNET-SPEC-CPU-SCALAR-HOTPATH.md)
- [CPU scalar parity contract](../../docs/specs/BITNET-SPEC-CPU-SCALAR-PARITY.md)
- [CPU scalar performance contract](../../docs/specs/BITNET-SPEC-CPU-SCALAR-PERFORMANCE.md)

Linked references:

- [CPU path plan](../../docs/bitnet/BITNET_CPU_PATH_PLAN.md)
- [Kernel matrix](../../docs/bitnet/BITNET_KERNEL_MATRIX.md)
- [Receipt fields](../../docs/bitnet/BITNET_RECEIPT_FIELDS.md)
- [CPU proof active tracker](../../docs/tracking/campaigns/cpu-proof/active.toml)

## Goal

Make scalar CPU BitNet inference a first-class, accurate, efficient,
receipt-backed oracle and fallback path. The scaled I2_S x I8_S scalar path is
the production scalar BitNet path; F32/no-scale scalar remains a diagnostic and
reference path.

## Hard Rules

- Do not use scalar as a hidden fallback in strict accelerated runs.
- Do not use no-scale F32 scalar as a substitute for scaled BitNet I8_S scalar.
- Do not change tokenizer or prompt policy.
- Do not change scalar math without fixtures and answer receipts.
- Do not invent new tolerances.
- Do not touch GPU/NPU/server lanes.
- Do not claim speedup.
- Preserve generated IDs or record exact divergence.

## Work Items

### CPU-SCALAR-000 — Add scalar specs and tracker rails

Title: `docs(cpu): add scalar kernel contract and hot-path plan`

Add the scalar kernel, hot-path, parity, and performance specs plus this plan.
Update CPU path and kernel-matrix docs to point to the scalar contracts, and add
tracker rails for the scalar follow-on sequence.

Acceptance:

```text
docs only
no runtime changes
CPU-SCALAR-000 tracker item added
claim boundaries explicit
```

Proof commands:

```bash
cargo run --locked -p xtask --no-default-features -- campaign check cpu-proof
cargo run --locked -p xtask --no-default-features -- campaign generate --check
git diff --check
```

### CPU-SCALAR-001 — Split scalar kernel identity

Title: `feat(cpu): split scalar QK256 kernel IDs`

Add precise IDs:

```text
qk256-scalar-f32-gemv
qk256-scalar-f32-gemm
qk256-scalar-i8s-scaled-gemv
qk256-scalar-i8s-scaled-gemm
```

Keep compatibility aliases:

```text
qk256-scalar-gemv -> qk256-scalar-f32-gemv
qk256-scalar-gemm -> qk256-scalar-f32-gemm
```

Acceptance:

```bash
cargo test --locked -p bitnet-quantization --no-default-features --features cpu i2s_qk256 --lib
```

### CPU-SCALAR-002 — Add scaled scalar selection metadata

Title: `feat(cpu): add kernel selection for scaled scalar I8S GEMV`

Add selected-kernel metadata for scaled scalar GEMV. Strict scalar is not a
fallback, strict unavailable AVX2 cannot silently select scalar, and scaled
kernel identity reaches receipts.

### CPU-SCALAR-003 — Wire scaled scalar selection into QK256 dispatch

Title: `feat(cpu): route inline-scale QK256 through selected scalar kernel`

Route `inline_scale` QK256 dispatch through the selected scaled scalar/SIMD path
and record requested/selected/fallback fields.

### CPU-SCALAR-004 — Add scalar hot-path counters

Title: `diag(cpu): record scalar QK256 hot-path counters`

Expose the counters required by the scalar hot-path spec in receipts.

### CPU-SCALAR-005 — Harden scalar fixture suite

Title: `test(cpu): harden scalar I2S/I8S fixtures`

Add broad tail, row, scale, pattern, activation, wrapping-accumulation, and
repeatability fixtures for scaled scalar math.

### CPU-SCALAR-006 — Add strict scalar answer-corpus receipt

Title: `test(cpu): record strict scalar BitNet answer corpus`

Record official Microsoft I2_S answer-corpus evidence with strict loader,
strict tokenizer, precise scaled scalar kernel ID, and `fallback_used=false`.

### CPU-SCALAR-007 — Add scalar-vs-existing receipts comparison

Title: `test(cpu): compare scalar baseline against AVX2 receipts`

Compare scalar strict receipts to AVX2 strict receipts with same prompt IDs,
generated IDs, decoded text, and fallback status.

### CPU-SCALAR-008 — Remove per-call flat weight extraction

Title: `perf(cpu): cache QK256 flat packed bytes for scalar dispatch`

Expose borrowed packed views from loader/model state so scalar dispatch does not
extract flat packed bytes per token.

### CPU-SCALAR-009 — Replace row `Vec<Vec<f32>>` materialization

Title: `perf(cpu): use flat buffers for scalar QK256 rows`

Use flat input/output buffers and strides instead of per-layer `Vec<Vec<f32>>`
input and output materialization.

### CPU-SCALAR-010 — Add reusable scalar workspace

Title: `perf(cpu): add reusable scalar CPU workspace`

Introduce reusable scalar scratch buffers for activation quantization, scaled
GEMV/GEMM, and output construction.

### CPU-SCALAR-011 — Optimize scalar activation quantization

Title: `perf(cpu): optimize scalar I8S activation quantization`

Add exact-behavior `quantize_row_i8_s_activation_into` style helpers that write
into workspace without per-call allocation.

### CPU-SCALAR-012 — Scalar prefill GEMM real path

Title: `feat(cpu): add scaled scalar QK256 prefill GEMM`

Add scaled I2_S x I8_S scalar GEMM and prove batched GEMM equals repeated
scaled GEMV.

### CPU-SCALAR-013 — Scalar phase benchmark receipt

Title: `bench(cpu): add scalar-only BitNet phase receipts`

Record scalar-only micro, layer, prefill, first-token, decode, and warm-session
receipts with precise scalar IDs and `speedup_claim=false`.

### CPU-SCALAR-014 — Scalar CPU support-op audit

Title: `diag(cpu): audit scalar transformer support-op timing`

Measure decode-critical CPU support ops and rank the next optimization target
without changing runtime behavior.

### CPU-SCALAR-015 — Scalar support-op cleanup pass

Title: `perf(cpu): optimize highest-cost scalar support op`

Optimize the highest-cost support op selected by CPU-SCALAR-014 with before and
after receipts and unchanged generated IDs.

### CPU-SCALAR-016 — Threading and scheduling policy for scalar

Title: `bench(cpu): establish scalar thread-count envelope`

Measure scalar thread counts for prefill, first-token, and decode profiles and
record the default scalar thread-count decision from evidence.

### CPU-SCALAR-017 — Scalar product-status surface

Title: `docs(cpu): publish scalar BitNet status`

Publish user-facing scalar BitNet CPU status that distinguishes oracle,
fallback, answer-corpus, long-decode, and performance status without speedup,
GPU/NPU, or server claims.

## Minimum Done

```text
strict scalar kernel selectable
actual selected kernel recorded
scaled I2_S x I8_S scalar path first-class
answer corpus passes
fallback=false
scalar-vs-AVX2 parity available
hot-path counters exist
```

## Good Done

```text
no per-token flat weight extraction
no Vec<Vec> row materialization
workspace reuse
prefill and decode scalar phase receipts
long decode parity
support-op timing report
```

## Excellent Done

```text
scalar is a stable oracle for AVX2/AVX512/CUDA/A770/M4
scalar performance is measured and not embarrassingly wasteful
profile-specific scalar baselines exist for every hardware lane
all optimized lanes compare against scalar receipts
users can force scalar for diagnosis and get intelligible bounded answers
```
