# BitNet CPU ISA Selection Spec

Status: draft
Owner: BitNet-rs maintainers
Created: 2026-05-18
Linked proposal: n/a
Linked specs:
- `docs/specs/BITNET-SPEC-CPU-AVX512-KERNEL-CONTRACT.md`
- `docs/specs/amd-9950x3d-cpu-roadmap.md`
Linked ADRs: n/a
Linked plan:
- `plans/cpu-avx512/implementation-plan.md`
Linked issues: n/a
Linked PRs: n/a
Support-tier impact: CPU auto-selection must not promote AVX-512 solely from CPUID.
Policy impact: No policy exception.

## Purpose

This spec defines strict CPU ISA selection for BitNet-rs QK256/I2_S execution.
It makes requested ISA, selected ISA, runtime CPU features, and fallback behavior
observable and enforceable.

## Request modes

The CPU kernel selector must support these request modes as the AVX-512 lane is
implemented:

```text
auto
scalar
avx2
avx512
avx512-vnni
```

`auto` is conservative. It may select scalar or AVX2 according to existing
feature and proof rules, but it must not select AVX-512 merely because CPUID
reports AVX-512 support.

## Selection table

| Request | Runtime features | Strict? | Result |
|---|---|---:|---|
| `auto` | AVX-512 available and profile promotion exists | n/a | AVX-512 for that promoted profile only |
| `auto` | AVX2/FMA available and no AVX-512 promotion applies | n/a | AVX2 |
| `auto` | neither AVX2/FMA nor promoted AVX-512 applies | n/a | scalar |
| `avx512` | required AVX-512 features available | true or false | AVX-512 |
| `avx512` | required AVX-512 features missing | true | error |
| `avx512` | required AVX-512 features missing | false | scalar/AVX2 fallback with `fallback_used=true` and a non-null reason |
| `avx512-vnni` | AVX-512 VNNI requirements available | true or false | AVX-512 VNNI kernel ID |
| `avx512-vnni` | VNNI requirements missing | true | error |
| `avx2` | AVX2/FMA available | true or false | AVX2 |
| `avx2` | AVX2/FMA missing | true | error |
| `avx2` | AVX2/FMA missing | false | scalar fallback with `fallback_used=true` and a non-null reason |
| `scalar` | any | true or false | scalar |

## AVX-512 feature rules

AVX-512 detection must be subfeature-aware. Receipts and selectors should
distinguish at least:

```rust
pub fn avx512_f_available() -> bool;
pub fn avx512_bw_available() -> bool;
pub fn avx512_vl_available() -> bool;
pub fn avx512_vnni_available() -> bool;
pub fn avx512_f_bw_available() -> bool;
pub fn avx512_f_bw_vl_available() -> bool;
pub fn avx512_bitnet_i8s_available() -> bool;
```

The baseline AVX-512 QK256 path must require only the subfeatures it actually
uses. VNNI must not be assumed unless probed and selected through a distinct
kernel ID.

## Strict fallback rule

Strict requested ISA is fatal on fallback. If a user requests an AVX-512 kernel
with strict selection and the host cannot execute that kernel, the run must
error. It must not write a receipt that reports scalar or AVX2 as the selected
kernel with `fallback_used=false`.

Non-strict fallback is permitted only when receipts record:

- requested kernel;
- selected kernel;
- `fallback_used=true`;
- non-null fallback reason;
- detected features;
- required features for the requested kernel.

## Auto-promotion rule

Do not make `auto` choose AVX-512 just because AVX-512 is detected. Auto may
choose AVX-512 only after all of these are true for the specific profile:

1. Scalar parity is proven.
2. AVX2 comparison is proven.
3. Answer-corpus or model-level parity evidence required by the profile passes.
4. Phase benchmark receipts show AVX-512 is better for the profile.
5. Sustained receipts show no sustained-power/cache-domain regression.
6. Fallback is false.
7. A promotion ledger or receipt validator accepts the promotion.

Until then, AVX-512 is explicit-request or campaign-only.
