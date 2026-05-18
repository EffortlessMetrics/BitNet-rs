# BITNET-SPEC-CPU-ISA-SELECTION: CPU ISA Selection Contract

Status: Draft
Owner: BitNet-rs maintainers
Created: 2026-05-18
Linked proposal: n/a
Linked specs: `docs/specs/BITNET-SPEC-CPU-AVX512-KERNEL-CONTRACT.md`, `docs/bitnet/BITNET_CPU_PATH_PLAN.md`, `docs/bitnet/BITNET_KERNEL_MATRIX.md`
Linked ADRs: n/a
Linked plan: `plans/cpu-avx512/implementation-plan.md`
Linked issues: n/a
Linked PRs: n/a
Support-tier impact: CPU ISA selection remains explicit and receipt-backed; automatic AVX-512 promotion is forbidden until profile-scoped receipts justify it.
Policy impact: n/a

## Purpose

This spec defines strict CPU ISA request and fallback behavior for scalar, AVX2,
AVX-512, and AVX-512 VNNI QK256 execution. It exists to prevent hidden fallback
and to keep auto-selection from promoting AVX-512 based only on CPUID detection.

## Request Modes

CPU kernel selection must support these request modes before the AVX-512 lane is
considered first-class:

```text
auto
scalar
avx2
avx512
avx512-vnni
```

A full kernel ID such as `qk256-avx512-i8s-scaled-gemv` may map onto one of
these ISA modes, but receipts must preserve both the user request and the stable
selected kernel ID.

## Selection Table

| Request | Runtime features | Strict? | Result |
|---|---|---:|---|
| `auto` | AVX-512 available and profile promotion exists | n/a | AVX-512 for that promoted profile |
| `auto` | AVX2 and FMA available | n/a | AVX2 |
| `auto` | neither AVX2/FMA nor promoted AVX-512 available | n/a | scalar |
| `avx512` | required AVX-512 features available | true or false | AVX-512 |
| `avx512` | required AVX-512 features missing | true | error |
| `avx512` | required AVX-512 features missing | false | scalar or AVX2 fallback with `fallback_used=true` |
| `avx512-vnni` | AVX-512 baseline and VNNI features available | true or false | AVX-512 VNNI kernel ID |
| `avx512-vnni` | VNNI or baseline features missing | true | error |
| `avx512-vnni` | VNNI or baseline features missing | false | non-VNNI AVX-512, AVX2, or scalar fallback with `fallback_used=true` |
| `avx2` | AVX2 and FMA available | true or false | AVX2 |
| `avx2` | AVX2 or FMA missing | true | error |
| `avx2` | AVX2 or FMA missing | false | scalar fallback with `fallback_used=true` |
| `scalar` | any | true or false | scalar |

## Auto-Selection Rail

`auto` must not choose AVX-512 just because runtime feature detection says
AVX-512 exists. Auto-selection may promote AVX-512 only after all of these are
true for the named profile:

- scalar-vs-AVX512 parity passes;
- AVX2-vs-AVX512 comparison receipts exist;
- answer-corpus or model-level proof for the relevant path records no hidden
  fallback;
- phase profile receipts show AVX-512 beats AVX2 for that profile;
- sustained-profile receipts show AVX-512 does not regress under sustained
  thermal and power conditions;
- the promotion is recorded in a profile-scoped promotion ledger or equivalent
  receipt validator accepted by the campaign.

Until then, AVX-512 is explicit-request or campaign-only.

## Strict Mode

Strict mode means the selected ISA and kernel must match the request. If the
requested ISA cannot run, the process must return an error before it emits a
success receipt. A strict AVX-512 request must never succeed with scalar or AVX2
execution.

## Non-Strict Fallback

Non-strict fallback may choose the best available lower ISA, but only if the
receipt records:

- `fallback_used=true`;
- the requested ISA or kernel;
- the selected ISA and stable kernel ID;
- the fallback reason;
- detected, required, and used CPU features.

## Required CPU Feature Helpers

The CPU detection layer should expose subfeature helpers before AVX-512 kernel
selection depends on them:

```rust
pub fn avx512_f_available() -> bool;
pub fn avx512_bw_available() -> bool;
pub fn avx512_vl_available() -> bool;
pub fn avx512_vnni_available() -> bool;
pub fn avx512_f_bw_available() -> bool;
pub fn avx512_f_bw_vl_available() -> bool;
pub fn avx512_bitnet_i8s_available() -> bool;
```

On non-x86 targets these helpers must return `false` without panicking. VNNI
helpers must not imply that baseline AVX-512 kernels are VNNI kernels.

## Receipt Requirements

ISA selection receipts must distinguish these states:

- AVX-512 detected;
- AVX-512 requested;
- AVX-512 selected;
- AVX-512 executed;
- AVX-512 faster for a named profile;
- AVX-512 sustained for a named sustained profile.

Each state requires different proof. User-facing status, support tiers, and
model readiness documentation must not collapse them into one claim.
