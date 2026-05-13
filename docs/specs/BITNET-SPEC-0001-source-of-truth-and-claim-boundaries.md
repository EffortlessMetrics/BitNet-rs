# BITNET-SPEC-0001: Source Of Truth And Claim Boundaries

Status: proposed
Linked proposal:
[BITNET-PROP-0001](../proposals/BITNET-PROP-0001-proof-convergence-and-ci-economics.md)
Applies to: docs, campaign tracking, model-artifact proof, hardware proof,
policy ledgers, release notes, README capability summaries

## Purpose

BitNet-rs needs one rule for deciding which artifact owns which truth. The repo
already has model gates, hardware matrices, policy TOMLs, campaign manifests,
receipts, and generated dashboards. This spec defines how those sources relate
so user-facing claims do not drift from proof artifacts or active work state.

This spec does not replace the operational gates. It defines where they sit in
the source-of-truth stack.

## Requirements

### 1. README Summarizes

`README.md` is the short user entry point. It may summarize current status,
common commands, limitations, and links to deeper docs.

It is not the final proof map. A README claim about model support, answer
quality, hardware execution, backend selection, speed, CI enforcement, or
release readiness must link to a maintained proof surface or remain explicitly
diagnostic, advisory, experimental, planned, or unsupported.

### 2. Model Answer Readiness Lives In Model Artifact Authorities

Model answer-readiness truth lives in:

- [Answer Artifact Gate](../model-artifacts/ANSWER_ARTIFACT_GATE.md)
- `ci/model-artifacts/artifact-manifest.toml`
- `ci/model-artifacts/candidate-artifacts.toml`
- `ci/model-artifacts/rejected-artifacts.toml`
- `ci/model-artifacts/tokenizer-authority.toml`
- `ci/model-artifacts/model-kernel-compatibility.toml`
- `ci/model-artifacts/model-coverage-matrix.toml`

A model artifact can be structurally valid and still fail answer readiness.
Coherent local-answer claims require the answer gate to pass for the relevant
model family and route. Loader success, tokenizer metadata, prompt-template
metadata, or a backend receipt alone is not enough.

### 3. Hardware Proof Lives In Hardware Authorities And Receipts

Hardware proof truth lives in:

- [Hardware Matrix](../hardware/HARDWARE_MATRIX.md)
- [Proof Stages](../hardware/PROOF_STAGES.md)
- `ci/hardware/**`
- durable hardware receipts produced by the lane

Hardware evidence must preserve:

```text
hardware identity
runtime identity
requested backend
selected backend
fallback_used
model artifact
proof stage
claim allowed
claim not allowed
```

Detected hardware is not backend proof. Backend proof is not answer quality.
Answer quality is not speed qualification. Each promotion must point to the
receipt or artifact that proves the next claim.

### 4. CI Lane Truth Lives In Policy Ledgers

CI lane inventory, cost posture, risk routing, and exception state live in:

- `policy/ci-lanes.toml`
- `policy/ci-budget.toml`
- `policy/ci-risk-packs.toml`
- `policy/ci-lane-whitelist.toml`
- `policy/ci-whitelist-exceptions.toml`
- workflow gates that enforce those ledgers

Narrative docs may explain the policy. They must not become a competing lane
inventory. If a doc and policy TOML disagree, the policy TOML is the
enforcement authority and the doc should be repaired.

Skipped expensive lanes must report a skip reason such as `skipped-by-policy`.
They must not be hidden as proof that the skipped lane passed.

### 5. Active Implementation State Lives In Campaign Active TOML

Current executable work state lives in campaign-local manifests:

```text
docs/tracking/campaigns/<campaign>/CAMPAIGN.md
docs/tracking/campaigns/<campaign>/active.toml
docs/tracking/campaigns/<campaign>/events/
docs/tracking/campaigns/<campaign>/generated/
```

Do not create `.adze/goals`, `.bitnet/goals`, or another hidden global active
goal file. Agents should use [Tracker Model](../tracking/TRACKER_MODEL.md) and
campaign `active.toml` files to decide what PR-sized work is ready, blocked,
open, merged, or superseded.

### 6. Generated Dashboards Are Generated

Generated dashboards are derived artifacts. Agents and maintainers must not
hand-edit generated dashboards to resolve conflicts, hide skipped lanes, change
hardware visibility, or mark work complete.

If generated content is stale, regenerate it with the campaign tooling. If the
generator is wrong, fix the generator or source manifest rather than editing
the generated output by hand.

### 7. Product Claims Must Link To Proof

Product claims in README, release notes, CLI docs, status docs, tutorials, and
server docs must point to proof or use a narrower tier.

Allowed tiers before full proof include:

- `planned`
- `experimental`
- `diagnostic`
- `advisory`
- `unsupported`
- `unsupported-on-this-hardware`
- `docs-only placeholder`

A supported or answer-ready claim needs proof from the relevant model,
tokenizer, prompt, backend, hardware, receipt, and CI surfaces.

## Claim Boundary

The following proof boundaries are mandatory:

| Evidence | May claim | Must not claim |
| --- | --- | --- |
| Structural GGUF loading | parse/load validity | coherent answer quality |
| Tokenizer authority | tokenizer/pre-tokenizer source status | answer readiness alone |
| Prompt-template authority | prompt formatting authority | backend correctness alone |
| Dense SLM receipt | dense SLM lane evidence | BitNet, I2_S, QK256, or 1-bit proof |
| BitNet CPU receipt | CPU lane diagnostic or answer proof as gated | CUDA, Metal, NPU, or speed proof |
| CUDA/Metal/OpenCL/OpenVINO/NPU receipt | selected-device execution as recorded | answer quality or generic GPU proof |
| Coverage | execution surface | oracle adequacy or model quality |
| ripr | static mutation-exposure signal | runtime mutation outcome |
| Mutation testing | runtime mutant evidence | hardware or model-answer proof alone |

## Proof Commands

This spec is documentation-only. Its current validation is:

```bash
git diff --check
cargo run --locked -p xtask --no-default-features -- campaign doctor
```

Future enforcement should live behind a dedicated docs source-of-truth check
once the proposal/spec/ADR/plan/campaign surfaces stabilize.

## Non-Goals

- Do not duplicate the answer artifact gate in this spec.
- Do not duplicate the hardware matrix in this spec.
- Do not duplicate CI lane TOMLs in this spec.
- Do not change README product claims in this PR.
- Do not change runtime behavior, workflows, model manifests, hardware
  receipts, policy TOMLs, or generated dashboards in this PR.
- Do not promote any diagnostic model, hardware, or CI state to supported.

## Related Policy Or Manifest Sources

- [Answer Artifact Gate](../model-artifacts/ANSWER_ARTIFACT_GATE.md)
- [Model Coverage Matrix](../model-artifacts/MODEL_COVERAGE_MATRIX.md)
- [Hardware Matrix](../hardware/HARDWARE_MATRIX.md)
- [Proof Stages](../hardware/PROOF_STAGES.md)
- [CI Cost and Verification Policy](../ci/cost-and-verification-policy.md)
- [Tracker Model](../tracking/TRACKER_MODEL.md)
- `policy/ci-lanes.toml`
- `policy/ci-budget.toml`
- `policy/ci-risk-packs.toml`
- `policy/ci-lane-whitelist.toml`
- `ci/model-artifacts/model-coverage-matrix.toml`
