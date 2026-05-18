# CPU AVX2 BitNet hot-path implementation plan

Status: Draft
Owner: BitNet-rs maintainers
Created: 2026-05-18
Linked proposal: n/a
Linked specs: ../../docs/specs/BITNET-SPEC-CPU-AVX2-HOTPATH.md
Linked ADRs: ../../docs/adr/BITNET-ADR-0005-proof-families-are-not-interchangeable.md
Linked plan: n/a
Linked issues: n/a
Linked PRs: n/a
Support-tier impact: Exact-profile CPU AVX2 promotions only after receipts pass.
Policy impact: none

## Guiding rails

- Scalar packed QK256 remains the correctness oracle.
- Strict requested AVX2 must fail closed if AVX2 cannot run or if a non-AVX2
  substitute is selected.
- Receipts must expose requested backend/kernel, selected backend/kernel,
  kernel family, runtime API, model and tokenizer authority, fallback truth, and
  QK256 hot-path counters.
- Performance claims require phase receipts and exact-profile review.
- This campaign is CPU AVX2 BitNet I2_S/QK256 only.

## Work items

### CPU-AVX2-HOTPATH-001 — docs/spec/plan/tracker rails

Title: `docs(cpu): add AVX2 BitNet hot-path implementation plan`

Scope: docs only.

Acceptance:

- create the CPU AVX2 hot-path spec;
- create the repo-local plan and status page;
- add the tracker item to `docs/tracking/campaigns/cpu-proof/active.toml`;
- do not change runtime code;
- run `cargo run --locked -p xtask --no-default-features -- campaign check cpu-proof`;
- run `git diff --check`.

### CPU-AVX2-HOTPATH-002 — hot-path counters

Title: `diag(cpu): record BitNet QK256 hot-path execution counters`

Add counters for no-scale F32 scalar/AVX2 GEMV, scaled I2_S x I8_S scalar/AVX2
GEMV, flat byte extraction, input-row materialization, output-row allocation,
and tensor-to-Vec conversion. Strict scalar and strict AVX2 answer-corpus
receipts must include these counters and preserve answer parity. No math changes
and no speed claim.

### CPU-AVX2-HOTPATH-003 — hot-path receipt validation

Title: `receipts(cpu): validate AVX2 hot-path counters`

Fail validation when requested AVX2 selects scalar in strict mode, an AVX2
selected kernel has zero AVX2 invocations, inline-scale proof records only the
no-scale F32 path, fallback truth conflicts with counters, or materialization
exceeds the audited boundary.

### CPU-AVX2-HOTPATH-004 — scaled I2S-I8S parity fixtures

Title: `test(cpu): add scaled I2S-I8S AVX2 parity fixtures`

Add fixture coverage for rows 1/2/7/32, cols 256/257/300/512/513/1024,
zero/one-ish/repeating/row-varying/pseudorandom packed patterns,
constant/ramp/centered/pseudorandom activations, and weight scales including
0.125, 0.5, 1.0, and finite edge values. Compare against
`gemv_qk256_bitnet_i8s_scaled` exactly; include code3, tail, and repeated-run
determinism coverage.

### CPU-AVX2-HOTPATH-005 — scaled I2S-I8S AVX2 GEMV

Title: `feat(cpu): add AVX2 scaled I2S-I8S QK256 GEMV`

Implement `gemv_qk256_bitnet_i8s_scaled_avx2` behind x86_64 AVX2 feature gates
and runtime feature checks. Do not fallback inside the function. Dimension,
tail, and weight-scale validation must match scalar semantics.

### CPU-AVX2-HOTPATH-006 — explicit scaled kernel selection

Title: `feat(cpu): select scaled AVX2 QK256 kernel explicitly`

Add stable kernel IDs for scalar and AVX2 scaled I8_S GEMV and selection tests
for auto, strict, non-strict fallback, and scalar requests when inline scale is
present.

### CPU-AVX2-HOTPATH-007 — transformer dispatch wiring

Title: `feat(cpu): route inline-scale BitNet QK256 through scaled AVX2`

Wire the inline-scale branch through the scaled AVX2 selector. Strict AVX2
answer-corpus receipts must show selected scaled AVX2 kernel, scaled AVX2
invocations greater than zero, scaled scalar invocations zero, fallback false,
and unchanged generated token IDs.

### CPU-AVX2-HOTPATH-008 — packed-view and materialization cleanup

Title: `perf(cpu): cache QK256 packed views for CPU dispatch`

Cache parsed QK256 layouts and flattened packed bytes, expose immutable packed
tensor views, use flat input/output buffers, and remove avoidable per-token
`Vec<Vec<_>>` materialization. Generate before/after phase receipts without
claiming speed until reviewed.

### CPU-AVX2-HOTPATH-009 — reusable CPU workspace

Title: `perf(cpu): add reusable BitNet CPU decode workspace`

Add reusable scratch for activation I8, output F32, optional QK256 code scratch,
attention, and logits. Receipts should show workspace reuse and memory
high-water while generated IDs remain unchanged.

### CPU-AVX2-HOTPATH-010 — strict AVX2 phase timing profiles

Title: `bench(cpu): add strict AVX2 phase timing profiles`

Add strict profiles for micro scaled GEMV, layer 0 decode, prefill 128, prefill
512, first token, decode 32, decode 128, and warm session 3 turns. Receipts must
split phase timing and keep speedup claims false until review.

### CPU-AVX2-HOTPATH-011 — exact-profile performance review

Title: `docs(cpu): review AVX2 performance qualification`

Turn scalar, AVX2, and previous-CPU timings into accepted/rejected decisions per
profile. Do not make global speedup claims.

### CPU-AVX2-HOTPATH-012 — answer corpus v2

Title: `test(cpu): add BitNet CPU answer corpus v2`

Expand answer categories while preserving scalar/AVX2 classification and first
divergence reporting. Do not make broad chat claims.

### CPU-AVX2-HOTPATH-013 — long-decode deterministic parity

Title: `test(cpu): add scalar-vs-AVX2 long decode parity`

Add greedy deterministic profiles for 16, 32, and 128 generated tokens with
prompt token equality, generated token equality or first divergence, top-k
logits where available, and fallback false.

### CPU-AVX2-HOTPATH-014 — prefill optimization

Title: `perf(cpu): optimize BitNet QK256 prefill path`

Optimize prefill without hot-path dequantization. Improve prefill_128 and
prefill_512 receipts or classify the blocker.

### CPU-AVX2-HOTPATH-015 — non-QK256 op bottleneck audit

Title: `diag(cpu): profile non-QK256 transformer CPU ops`

Rank RMSNorm, sub-layernorm, RoPE, QK score, softmax/masking, AV,
KV append/read, output head/logits, and sampling bottlenecks without runtime
behavior changes.

### CPU-AVX2-HOTPATH-016 — user-facing support status

Title: `docs(cpu): publish AVX2 BitNet support status`

Publish exact support rows for official Microsoft BitNet I2_S scalar, AVX2
scaled I8_S, answer corpus, long decode, speed profiles, and server false unless
separately proven.

## Default validation bundle

Runtime PRs normally run:

```bash
cargo fmt --all -- --check
cargo test --locked -p bitnet-quantization --no-default-features --features cpu,avx2 --test qk256_avx2_parity_tests
cargo test --locked -p bitnet-quantization --no-default-features --features cpu,avx2 i2s_qk256 --lib
cargo test --locked -p bitnet-qk256-dispatch --no-default-features --features cpu
cargo test --locked -p bitnet-cli --no-default-features --features cpu,full-cli answer_corpus
cargo run --locked -p xtask --no-default-features -- campaign check cpu-proof
git diff --check
```

Performance PRs also add the relevant benchmark build and JSON receipt checks.
