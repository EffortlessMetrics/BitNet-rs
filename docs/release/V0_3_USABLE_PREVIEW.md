# v0.3 Usable Preview Release Contract

Status: proposed
Owner: release maintainers
Created: 2026-06-05
Linked proposal: [BITNET-PROP-0003](../proposals/BITNET-PROP-0003-native-rust-inference-product.md)
Linked specs:
[BITNET-SPEC-MODEL-READINESS-STATUS-SURFACE](../specs/BITNET-SPEC-MODEL-READINESS-STATUS-SURFACE.md),
[BITNET-SPEC-RECEIPT-EXPLAIN-SCHEMA](../specs/BITNET-SPEC-RECEIPT-EXPLAIN-SCHEMA.md),
[BITNET-SPEC-SUPPORT-BUNDLE](../specs/BITNET-SPEC-SUPPORT-BUNDLE.md),
[BITNET-SPEC-CLAIM-BOUNDARY-REVIEW](../specs/BITNET-SPEC-CLAIM-BOUNDARY-REVIEW.md),
[BITNET-SPEC-0010](../specs/BITNET-SPEC-0010-server-readiness-proof-boundary.md)
Linked ADRs: [BITNET-ADR-0005](../adr/BITNET-ADR-0005-proof-families-are-not-interchangeable.md)
Linked plan: n/a
Linked issues: n/a
Linked PRs: n/a
Support-tier impact: defines the v0.3 preview release gate; does not promote a tier by itself
Policy impact: release-readiness and claim-boundary input only

## Purpose

The v0.3 usable preview release turns BitNet-rs from a proof-heavy pre-alpha
workspace into a small, honest, installable local-inference preview. The release
is ready only when a user can:

```text
install or build BitNet-rs
inspect what model/device paths are supported
fetch or verify a supported model artifact
run one local answer path
explain the resulting receipt
prepare a useful support bundle
understand every unsupported, candidate, diagnostic, and experimental surface
```

This contract defines the release bar. It does not change runtime behavior,
model coverage rows, support tiers, receipts, CI routing, publish state, or
README positioning by itself.

## Source Of Truth

The README may summarize the release, but these surfaces own release truth:

| Surface | Release role |
| --- | --- |
| [Model coverage matrix](../model-artifacts/MODEL_COVERAGE_MATRIX.md) and `ci/model-artifacts/model-coverage-matrix.toml` | Model tiers, proof-family booleans, speed, residency, server, and next-proof fields. |
| [Status documents](../status/README.md) | User-facing claim maps and status-page ownership rules. |
| [CUDA capability matrix](../status/CUDA_CAPABILITY_MATRIX.md) | RTX 5070 Ti CUDA model and route status. |
| [OpenVINO capability matrix](../status/OPENVINO_CAPABILITY_MATRIX.md) | Lunar Lake OpenVINO candidate and diagnostic boundaries. |
| [Hardware validation matrix](../hardware/HARDWARE_MATRIX.md) | Device labels, hardware identity, and proof-stage boundaries. |
| [CI cost and verification policy](../ci/cost-and-verification-policy.md) | Default PR versus release-lane validation routing. |
| [Receipt explain schema](../specs/BITNET-SPEC-RECEIPT-EXPLAIN-SCHEMA.md) | Stable receipt support summary shape. |
| [Support bundle spec](../specs/BITNET-SPEC-SUPPORT-BUNDLE.md) | Pasteable issue artifact shape. |
| [5-minute quickstart](../quickstart.md) and [CUDA quickstart](../tutorials/9950x3d-5070ti-cuda-quickstart.md) | User command paths once rewritten or verified against this contract. |
| [Promotion contract](PROMOTE_TO_BITNET_RS.md) | Source/swarm release-promotion boundary. |

If any of these surfaces disagree, the release must use the narrower claim until
the mismatch is repaired.

## Release Posture

v0.3 is a usable preview, not a production release. The public claim is:

```text
BitNet-rs supports exact, receipt-backed local answer paths for selected
models and devices. Unsupported, candidate, diagnostic, server, speed, and
residency surfaces remain bounded by model coverage and receipt evidence.
```

The release must not claim:

- production readiness;
- broad GPU support;
- broad server readiness;
- global speedup;
- full device residency;
- dense SLM proof as BitNet packed I2_S/QK256 proof;
- CUDA execution as speedup;
- CLI readiness as server readiness;
- one exact-profile proof as another model, backend, route, or device profile.

## User Front Door

The preview should make this sequence the obvious supported path:

```bash
bitnet model status
bitnet model fetch <supported-model>
bitnet model verify <supported-model>
bitnet ask --model <supported-model> --device <supported-device> "What is 2+2?"
bitnet receipts explain --latest
bitnet support bundle --latest --device <supported-device> --format json
```

For a successful supported path, the user-facing summary must preserve:

```text
model coverage row
current tier
model artifact identity
tokenizer and prompt authority
requested backend
selected backend
selected route
fallback status
quality gate result
speedup claim
server readiness scope
full residency claim
claim boundary
receipt path
next proof
```

`bitnet model status` is the front door for what the repo allows the user to
claim. `bitnet receipts explain` describes what the last run actually proved.
The support bundle composes both for issue triage without running new inference.

## Releasable Model Rows

The model coverage matrix owns the final tier. A v0.3 release may present a row
as supported preview only when its current matrix row and linked receipts allow
the claim.

| Model row | Release posture | Required proof for v0.3 claim | Stop lines |
| --- | --- | --- | --- |
| `bitnet_official_2b_i2s_qk256` | Supported preview for exact official I2_S/QK256 CLI answer paths where the matrix keeps `product_cli_ready=true`. | Model artifact, tokenizer, prompt authority, CPU or exact selected accelerator receipt, fallback-free strict run where claimed, quality gate, receipt explanation, support bundle parity. | No TL1, TL2, BF16/GPU-int2, Apple, A770, dense SLM, speedup, full residency, or broad server inheritance. |
| `dense_qwen25_05b_q8_cuda` | Supported preview for exact dense Qwen2.5 CUDA CLI paths where the matrix keeps `product_cli_ready=true`. | Artifact verification, `dense_regular_llm_cuda` route, strict RTX 5070 Ti receipts, answer quality, receipt explanation, benchmark review preserving speed claims. | Not BitNet proof; no broad dense GGUF, speedup, full residency, concurrency, or deployment readiness. |
| `dense_qwen3_06b_q8_candidate` | Supported preview only for the bounded product-CLI paths already accepted by its row. | Own Qwen3 artifact, CPU sanity, all-layer plan, strict CUDA ask/chat receipts, receipt explanation, and separate review for each promoted scope. | Does not inherit Qwen2.5 or BitNet proof; no speedup, full residency, or broad dense support. |
| `small_llm_qwen25_15b_q4km_candidate` | Optional supported preview only for the exact Apple M4 CPU/NEON answer path if README and quickstart keep it non-default and matrix-backed. | CPU/NEON receipts, memory-envelope evidence, artifact and tokenizer authority, receipt explanation, support bundle fields. | No CUDA claim until strict CUDA support exists; no default model claim unless the release notes say so. |
| `dense_smollm2_360m_candidate` | Candidate. | Structurally valid evidence plus next same-prompt reference comparator or quality-gate proof before answer readiness. | No CPU answer, CUDA answer, product CLI, speed, or server claim. |
| Qwen3 larger, SmolLM2 1.7B, Llama 3.2, Gemma, Phi, modern placeholders | Candidate or registered. | Each needs its own artifact, tokenizer, CPU sanity, route, quality, backend, and receipt ladder before support. | No inherited Qwen2.5, Qwen3, BitNet, CUDA, speed, or server proof. |
| Other BitNet, 1bitLLM, Falcon, Llama3 1.58-bit, MCU fixture rows | Diagnostic or registered unless the matrix says otherwise. | Family-specific artifact, tokenizer, route, reference, quality, backend, and receipt gates. | Fixture, diagnostic, or family registration is not user answer support. |

## Releasable Device Rows

Device labels must match the hardware matrix and receipt identity. A friendly
alias such as `cuda` may route to a strict selected backend only when the status
surface shows the exact selected backend.

| Device row | Release posture | Required proof | Stop lines |
| --- | --- | --- | --- |
| `cpu` and exact CPU labels | Supported preview only for rows with accepted CPU answer readiness. | CPU model status row, artifact and tokenizer authority, fallback semantics, quality gate, receipt explanation. | CPU proof does not prove CUDA, Apple, OpenVINO, A770, NPU, speed, or server readiness. |
| `nvidia-rtx-5070-ti-cuda` | Supported preview for exact CUDA rows listed in the CUDA capability matrix. | Strict selected backend, route, runtime API, fallback-free receipt, quality gate, receipt explanation, benchmark review where speed is discussed. | Generic `cuda`, WGPU, Vulkan, visibility, or fallback receipts are not strict RTX 5070 Ti proof. |
| `apple-m4-cpu-neon` | Optional supported preview for exact CPU/NEON model rows that have accepted receipts. | Apple M4 CPU/NEON identity, artifact/tokenizer proof, quality gate, receipt explanation. | Apple CPU/NEON does not prove Metal, MPSGraph, ANE/NPU, CUDA, or non-Apple CPU behavior. |
| OpenVINO CPU/GPU/NPU rows | Candidate unless exact route promotion says otherwise. | Route promotion ledger, fallback-free OpenVINO receipts, quality and timing review, status and receipt explanation parity. | NPU or GPU candidate evidence is not BitNet QK256, native OpenCL, speed, low-power, or server proof. |
| A770, Metal, ROCm, WGPU, Vulkan, generic GPU, other accelerators | Diagnostic or registered unless a status matrix promotes an exact profile. | Exact device identity, selected backend, route, fallback-free receipt, quality gate, model coverage update, receipt explanation. | Detection or smoke is not answer readiness, speed, residency, or release support. |

## Required User Commands

The release is not usable until the supported path has copy-pasteable command
coverage in quickstart or status docs:

```bash
bitnet model status --format json
bitnet model status --device <supported-device> --format json
bitnet model fetch <supported-model>
bitnet model verify <supported-model>
bitnet ask --model <supported-model> --device <supported-device> "What is 2+2?"
bitnet receipts explain --latest --format json
bitnet support bundle --latest --device <supported-device> --format json
bitnet --version
bitnet doctor
```

For every supported model/device combination, docs must state whether:

```text
fallback is allowed or rejected
quality gate passed or is unavailable
speedup_claim is true or false
server_ready is false, preview, or exact-profile only
full_residency_claim is true or false
```

If a command cannot run on a user's machine without special hardware, the docs
must say so before the command and point to the cheapest status command that
does not require that hardware.

## Maintainer Release Gates

Before tagging a v0.3 preview candidate, maintainers must run or record why
they cannot run:

```bash
git diff --check
cargo run --locked -p xtask --no-default-features -- check-model-coverage
cargo run -p xtask -- release-ready --profile usable-preview
```

The release-ready profile must fail closed when:

- README links are stale;
- status matrices are missing required rows;
- model coverage rows disagree with status docs;
- a supported row lacks receipt or command evidence;
- `speedup_claim=true` lacks benchmark-qualified exact-profile receipts;
- `server_ready=true` lacks exact-profile server promotion evidence;
- crates.io or docs.rs badges imply publication before publication exists;
- release notes or known limitations are missing;
- stale campaign rows imply release support without current proof;
- source/swarm promotion boundaries are missing for promoted swarm evidence.

Until the `release-ready` command exists, this document is the manual checklist.
Each missing automated check must be listed in release notes as not run,
unknown, or blocked.

## Experimental And Unsupported Surfaces

The preview keeps these out of the supported claim unless their exact rows are
promoted before release:

- TL1, TL2, and BF16/GPU-int2 BitNet routes;
- A770/OpenCL, ROCm, WGPU, Vulkan, Metal, MPSGraph, and NPU full inference;
- broad OpenVINO route promotion;
- broad server or OpenAI-compatible API readiness;
- streaming, concurrency, uptime, deployment, or production serving claims;
- global speedup, benchmark-qualified speed, or full residency;
- all unlisted model families and quantizations;
- model downloads or artifacts without coverage rows;
- diagnostics, fixtures, traces, microbenchmarks, or hardware detection as
  product support.

## Release Documentation Requirements

The v0.3 preview documentation set must contain:

| Document | Required role |
| --- | --- |
| README | Short product entry point and precise warning; summarizes but does not invent claims. |
| Support matrix | One user-facing table of supported, candidate, diagnostic, and unsupported rows. |
| CUDA and Apple capability matrices | Exact-profile device status and proof pointers. |
| Quickstart | Copy-pasteable zero-to-answer path for the most reliable supported path first. |
| Known limitations | Explicit unsupported and experimental surfaces. |
| Release notes | What works, what does not, supported models/devices, performance posture, server posture, and issue path. |
| Support triage guide | How to interpret fallback, backend mismatch, tokenizer errors, quality failures, speed false, server false, dense-vs-BitNet proof, and issue payloads. |

## Definition Of Done

BitNet-rs is nice and releasable for v0.3 when:

```text
one supported model path works end-to-end
one command shows what is supported
one command fetches or verifies the supported model
one command runs a local answer
one command explains the receipt
one support bundle helps debug failures
one support matrix bounds all claims
one quickstart gets users to success
one release checklist blocks overclaiming
```

If any item is missing, the release may still publish only if release notes name
the missing item, mark the affected claim unsupported or unknown, and keep the
README warning precise.
