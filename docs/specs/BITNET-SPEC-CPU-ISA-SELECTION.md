# BitNet CPU ISA Selection

Status: Draft
Owner: BitNet CPU proof campaign
Created: 2026-05-18
Linked proposal: n/a
Linked specs: `docs/specs/BITNET-SPEC-CPU-AVX512-KERNEL-CONTRACT.md`, `docs/specs/amd-9950x3d-cpu-roadmap.md`
Linked ADRs: n/a
Linked plan: `plans/cpu-avx512/implementation-plan.md`
Linked issues: n/a
Linked PRs: n/a
Support-tier impact: CPU ISA auto-selection remains conservative until proof receipts promote a profile.
Policy impact: No policy exception.

## Purpose

This spec defines CPU ISA request modes, strict fallback behavior, and auto-selection rails for scalar, AVX2, AVX-512, and future AVX-512 VNNI BitNet CPU kernels.

## Request Modes

CPU kernel selection must support these request modes once the lane is implemented:

```text
auto
scalar
avx2
avx512
avx512-vnni
```

Concrete kernel IDs, such as `qk256-avx512-i8s-scaled-gemv`, may also be accepted by user-facing tools when the dispatch layer can map them to the same ISA contract.

## Required Feature Semantics

| ISA mode | Required runtime features | Notes |
| --- | --- | --- |
| `scalar` | none | Always available and the correctness oracle. |
| `avx2` | `avx2`, `fma` where the kernel requires FMA | Existing x86 SIMD comparison lane. |
| `avx512` | `avx512f`, `avx512bw` | Baseline AVX-512BW BitNet path. |
| `avx512-vnni` | `avx512f`, `avx512bw`, `avx512vl`, `avx512vnni` or the exact subfeature set required by the implementation | Separate kernel ID required. |

Detection helpers should be subfeature-aware and must return `false` on unsupported architectures without panicking.

## Selection Rules

| Request | Runtime features | Strict? | Result |
| --- | --- | ---: | --- |
| `auto` | AVX-512 available and profile is explicitly promoted by receipts | n/a | AVX-512 |
| `auto` | AVX2/FMA available and AVX-512 is not promoted | n/a | AVX2 |
| `auto` | Neither promoted AVX-512 nor AVX2/FMA available | n/a | scalar |
| `avx512` | Required AVX-512 features available | true or false | AVX-512 |
| `avx512` | Required AVX-512 features missing | true | error |
| `avx512` | Required AVX-512 features missing | false | scalar or AVX2 fallback with `fallback_used=true` |
| `avx2` | AVX2/FMA available | true or false | AVX2 |
| `avx2` | AVX2/FMA missing | true | error |
| `avx2` | AVX2/FMA missing | false | scalar fallback with `fallback_used=true` |
| `scalar` | any | true or false | scalar |

## Auto-Selection Rail

Auto-selection must not choose AVX-512 merely because the CPU reports AVX-512 support. AVX-512 auto-selection is allowed only for a profile with evidence that:

1. kernel parity passes against scalar;
2. answer-corpus or real-model parity is accepted for the claim scope;
3. phase benchmarks beat or justify replacing AVX2 for that profile;
4. sustained receipts do not regress the profile;
5. no fallback was used; and
6. a profile-specific promotion record or validator accepts the promotion.

Until that evidence exists, AVX-512 must remain explicit-request or campaign-only.

## Receipt Requirements

CPU ISA receipts must record:

- requested mode or requested kernel;
- selected kernel;
- strictness;
- fallback status and reason;
- detected CPU features;
- required CPU features;
- used CPU features;
- invocation counters for the selected hot path.

The selector must never emit scalar or AVX2 execution as AVX-512 with `fallback_used=false`.

## Error Requirements

A strict request must produce a fatal, user-visible error when the selected kernel cannot satisfy the requested ISA. The error should include the requested mode/kernel and the missing feature set when known.

Non-strict fallback may continue only when the fallback is explicit in receipts and user-facing output.
