# BITNET-SPEC-CUDA-SUPPORT-ISSUE

Status: proposed
Owner: BitNet-rs maintainers
Created: 2026-05-19
Linked proposal:
[BITNET-PROP-0003](../proposals/BITNET-PROP-0003-native-rust-inference-product.md)
Linked specs:
[BITNET-SPEC-SUPPORT-BUNDLE](BITNET-SPEC-SUPPORT-BUNDLE.md),
[BITNET-SPEC-MODEL-READINESS-STATUS-SURFACE](BITNET-SPEC-MODEL-READINESS-STATUS-SURFACE.md),
[BITNET-SPEC-RECEIPT-EXPLAIN-SCHEMA](BITNET-SPEC-RECEIPT-EXPLAIN-SCHEMA.md)
Linked ADRs:
[BITNET-ADR-0005](../adr/BITNET-ADR-0005-proof-families-are-not-interchangeable.md)
Linked plan: n/a
Linked issues: n/a
Linked PRs: n/a
Support-tier impact: CUDA support issues must preserve existing claim boundaries
Policy impact: none

## Purpose

CUDA support issues must be receipt-backed. A user should paste one support
bundle before writing free-form environment prose so triage can route by
structured proof instead of by inference history or guesswork.

The governed template is:

```text
.github/ISSUE_TEMPLATE/cuda-support.yml
```

## Required First Artifact

The template must ask for JSON from:

```powershell
bitnet support bundle --latest --device nvidia-rtx-5070-ti-cuda --format json
```

The support-bundle field must:

- appear before the free-form issue description;
- be required;
- render as JSON;
- show `kind = bitnet_support_bundle`;
- show `selected_backend = nvidia-rtx-5070-ti-cuda`;
- show `selected_route`;
- show `fallback_used`;
- show `speedup_claim`;
- show `claim_boundary`.

## Claim Boundary Checklist

The template must preserve these review boundaries:

```text
selected backend is nvidia-rtx-5070-ti-cuda, not generic cuda
fallback_used=false is required for strict selected-backend proof
speedup_claim=false unless exact-profile benchmark proof accepts speedup
Qwen2.5 exact-profile server readiness is not broad dense GGUF server readiness
dense CUDA proof is not BitNet I2_S/QK256 proof
Qwen2.5 evidence is not Qwen3 evidence
```

Checkboxes are allowed as user-facing reminders, but triage must still use the
support-bundle JSON and receipt explanation as the authority.

## Fallback Path

If the support-bundle command fails, issue triage should request:

```text
failed command
stderr
receipt path, if available
operating system
GPU model
driver/runtime versions
```

This fallback is for support collection only. It must not become proof of
selected CUDA execution, speedup, full residency, server readiness, or answer
quality.

## Non-Goals

- Do not ask users to paste unrelated logs before the support bundle.
- Do not infer broad CUDA support from one device label.
- Do not infer Qwen3, BitNet packed I2_S/QK256, speedup, or full residency
  from dense Qwen2.5 CUDA support evidence.
- Do not require private paths, secrets, tokens, credentials, or model files.

## Proof Commands

```bash
cargo test --locked -p bitnet-cli --no-default-features --features cpu,full-cli cuda_support_issue_template
git diff --check
```
