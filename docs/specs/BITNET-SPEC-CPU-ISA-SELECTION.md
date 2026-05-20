# BITNET-SPEC-CPU-ISA-SELECTION: Strict CPU ISA Selection

Status: proposed
Linked spec: [CPU AVX-512 kernel contract](BITNET-SPEC-CPU-AVX512-KERNEL-CONTRACT.md)
Linked plan: [CPU AVX-512 implementation plan](../../plans/cpu-avx512/implementation-plan.md)
Applies to: CPU kernel request modes, QK256/I2_S dispatch, strict fallback
policy, CPU proof receipts, answer-corpus receipts, benchmark receipts

## Purpose

CPU ISA selection must be explicit enough to prove what ran. The selector must
separate auto-selection, user-forced scalar/AVX2/AVX-512 requests, strict
failure behavior, and non-strict fallback receipts.

This spec prevents hidden fallback where a user requests AVX-512 but scalar or
AVX2 execution is recorded as successful AVX-512 proof.

## Required Request Modes

CPU kernel selection must support these conceptual request modes. CLI spelling
may use stable kernel IDs or aliases, but receipts must preserve the normalized
mode or kernel ID that was requested.

```text
auto
scalar
avx2
avx512
avx512-vnni
```

The `avx512-vnni` mode is reserved until a separately identified VNNI kernel
lands and passes parity. It must not alias to the baseline AVX-512BW kernel in
strict receipts.

## Selection Rules

| Request | Runtime features | Strict? | Result |
| --- | --- | ---: | --- |
| `auto` | AVX-512 available and profile promotion accepted | n/a | Select promoted AVX-512 kernel for that profile. |
| `auto` | AVX2 and FMA available | n/a | Select AVX2 when no AVX-512 promotion applies. |
| `auto` | Neither AVX2/FMA nor promoted AVX-512 available | n/a | Select scalar. |
| `avx512` | Required AVX-512 features available | true or false | Select the requested AVX-512 kernel. |
| `avx512` | Required AVX-512 features missing | true | Error before fallback execution. |
| `avx512` | Required AVX-512 features missing | false | Select scalar or AVX2 fallback and record `fallback_used=true`. |
| `avx512-vnni` | Required AVX-512 and VNNI features available | true or false | Select the requested VNNI kernel. |
| `avx512-vnni` | Required VNNI features missing | true | Error before fallback execution. |
| `avx512-vnni` | Required VNNI features missing | false | Select baseline AVX-512, AVX2, or scalar fallback and record `fallback_used=true`. |
| `avx2` | AVX2 and FMA available | true or false | Select AVX2. |
| `avx2` | AVX2 or FMA missing | true | Error before fallback execution. |
| `avx2` | AVX2 or FMA missing | false | Select scalar fallback and record `fallback_used=true`. |
| `scalar` | Any | true or false | Select scalar. |

## Auto-Selection Rail

Auto mode must not choose AVX-512 merely because CPUID reports AVX-512. Auto may
select AVX-512 for a specific profile only after all of the following exist and
are accepted for that profile:

- scalar parity;
- AVX2 comparison;
- answer-corpus proof when model-level answers are in scope;
- phase benchmark showing the AVX-512 profile beats the comparator it replaces;
- sustained run showing no profile regression under recorded 9950X3D platform
  conditions;
- receipt validator or promotion ledger accepts the profile-specific rule;
- `fallback_used=false` in the promotion evidence.

Until those gates land, AVX-512 is explicit-request or campaign-only.

## Receipt Requirements

Each CPU dispatch receipt must record:

```text
requested_backend
selected_backend
requested_kernel
selected_kernel
kernel_family
fallback_used
fallback_reason
strict
features_detected
features_required
features_used
missing_features
profile_name
```

For auto-selection, receipts must also identify why AVX-512 was or was not
eligible for the profile. Absence of a promotion record must be interpreted as
"not promoted," not as permission to use AVX-512 globally.

## Error Requirements

Strict selection errors must be actionable. A strict AVX-512 failure must name:

- requested kernel or request mode;
- selected kernel, if any, before execution;
- required CPU features;
- detected CPU features;
- missing CPU features;
- whether the binary was compiled with the needed feature gate.

The process must not emit a success receipt that implies AVX-512 execution when
execution did not occur.

## Claim Boundary

- `auto` does not mean "widest detected ISA".
- A successful scalar or AVX2 fallback is not AVX-512 proof.
- A strict selection failure is valid proof that hidden fallback was rejected.
- `avx512-vnni` is not proven by baseline `avx512f,avx512bw` execution.
- Profile-specific AVX-512 promotion does not imply global AVX-512 promotion.

## Proof Commands

Documentation-only selection changes must run:

```bash
git diff --check
cargo run --locked -p xtask --no-default-features -- campaign check cpu-proof
cargo run --locked -p xtask --no-default-features -- campaign generate --check
```

Runtime selection PRs must add unit tests for strict and non-strict behavior for
each request mode they introduce.

## Non-Goals

- This spec does not implement AVX-512 kernels.
- This spec does not promote AVX-512 auto-selection.
- This spec does not change GPU, NPU, OpenVINO, CUDA, OpenCL, Metal, WGPU, or
  server selection rules.

## Related Sources

- [CPU AVX-512 kernel contract](BITNET-SPEC-CPU-AVX512-KERNEL-CONTRACT.md)
- [AMD Ryzen 9 9950X3D CPU Roadmap](amd-9950x3d-cpu-roadmap.md)
- [BitNet Kernel Matrix](../bitnet/BITNET_KERNEL_MATRIX.md)
- [BitNet CPU Path Plan](../bitnet/BITNET_CPU_PATH_PLAN.md)
