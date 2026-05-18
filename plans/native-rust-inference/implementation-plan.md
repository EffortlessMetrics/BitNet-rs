# Native Rust Inference Implementation Plan

This file gives the first executable sequence for the native Rust inference
product lane. Each item should become a narrow PR or a campaign work item before
runtime implementation starts.

## Sequence

| Order | Item | File |
| --- | --- | --- |
| 0 | Normalize autonomous PR operations | landed before this plan |
| 1 | Define BitNet source-of-truth model | landed before this plan |
| 2 | Add native Rust inference product proposal | landed before this plan |
| 3 | Add model onboarding proof ladder | landed before this plan |
| 4 | Add runtime performance contract | landed before this plan |
| 5 | Record non-interchangeable proof families ADR | landed before this plan |
| 6 | Add this implementation plan | this PR |
| 7 | Stable model-status and receipts-explain JSON | `server-readiness.md` |
| 8 | CUDA receipt triage guide | `server-readiness.md` |
| 9 | Exact-profile readiness display | `server-readiness.md` |
| 10 | Per-request receipt export | `server-readiness.md` |
| 11-14 | Qwen3 product promotion | `dense-qwen3.md` |
| 15-18 | Official BitNet performance and residency | `bitnet-official-i2s.md` |
| 19-22 | Dense Qwen2.5 optimization and requalification | `dense-qwen25.md` |
| 23-26 | SmolLM2 comparator, CPU, and CUDA path | `smollm2.md` |
| CI-1+ | CI economics rollout | `ci-economics.md` |

## Work item: NRI-PLAN-000

Status: ready
Linked proposal: `docs/proposals/BITNET-PROP-0003-native-rust-inference-product.md`
Linked specs: `docs/specs/BITNET-SPEC-0013-model-onboarding-proof-ladder.md`,
`docs/specs/BITNET-SPEC-0014-runtime-performance-contract.md`
Linked ADRs: `docs/adr/BITNET-ADR-0005-proof-families-are-not-interchangeable.md`
Campaign: `docs/tracking/campaigns/nvidia-5070ti/active.toml`
Blocks: product UX, server readiness, Qwen3 promotion, performance work, CI economics
Blocked by: none

### Goal

Add the native inference plan files so future runtime and CI PRs have a shared
sequence and proof contract.

### Production delta

No runtime delta. This is planning infrastructure for follow-on PRs.

### Non-goals

Do not edit model coverage rows, receipts, runtime code, CI workflows, policy
TOMLs, or generated dashboards.

### Acceptance

- Plan files exist under `plans/native-rust-inference/`.
- Each lane file names source links, non-goals, acceptance, proof commands, and
  rollback.
- Proof-family and exact-profile boundaries are repeated where the work might
  otherwise blur them.

### Proof commands

```bash
git diff --check
cargo run --locked -p xtask --no-default-features -- campaign check nvidia-5070ti
```

### Rollback

Revert `plans/native-rust-inference/` and leave active campaign state unchanged.
