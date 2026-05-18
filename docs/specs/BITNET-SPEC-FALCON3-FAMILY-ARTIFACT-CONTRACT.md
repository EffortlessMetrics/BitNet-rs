# BITNET-SPEC-FALCON3-FAMILY-ARTIFACT-CONTRACT: Falcon3 Artifact Contract

Status: draft
Owner: model-artifacts
Created: 2026-05-18
Linked proposal: [BITNET-PROP-0012](../proposals/BITNET-PROP-0012-falcon3-family-supported-models.md)
Linked specs: n/a
Linked ADRs: [BITNET-ADR-0005](../adr/BITNET-ADR-0005-proof-families-are-not-interchangeable.md)
Linked plan: [Falcon3 family implementation plan](../../plans/falcon3-family/implementation-plan.md)
Linked issues: n/a
Linked PRs: n/a
Support-tier impact: defines future gates only; no promotion
Policy impact: no policy exception

## Purpose

Define exact artifact identity and receipt requirements for Falcon3 Family onboarding. This spec registers candidate artifacts only; it does not authorize answer, backend, speed, or server claims.

## Required Artifact Table

```toml
[[falcon3_artifact]]
id = "falcon3_1b_instruct_158_i2s_gguf"
source_repo = "tiiuae/Falcon3-1B-Instruct-1.58bit-GGUF"
source_format = "gguf"
file = "ggml-model-i2_s.gguf"
nominal_size = "1B"
hf_display_size = "recorded_from_probe"
route = "i2_s"
priority = 1

[[falcon3_artifact]]
id = "falcon3_7b_instruct_158_i2s_gguf"
source_repo = "tiiuae/Falcon3-7B-Instruct-1.58bit-GGUF"
source_format = "gguf"
file = "ggml-model-i2_s.gguf"
nominal_size = "7B"
hf_display_size = "recorded_from_probe"
route = "i2_s"
priority = 2

[[falcon3_artifact]]
id = "falcon3_3b_instruct_158_safetensors"
source_repo = "tiiuae/Falcon3-3B-Instruct-1.58bit"
source_format = "safetensors"
route = "conversion_required_i2_s"
priority = 3

[[falcon3_artifact]]
id = "falcon3_10b_instruct_158_safetensors"
source_repo = "tiiuae/Falcon3-10B-Instruct-1.58bit"
source_format = "safetensors"
route = "conversion_required_i2_s"
priority = 4
```

## Required Inventory Receipt Fields

```json
{
  "artifact_kind": "falcon3_artifact_inventory",
  "model_family": "falcon3_158bit",
  "source_repo": "tiiuae/Falcon3-1B-Instruct-1.58bit-GGUF",
  "source_revision": "...",
  "file": "ggml-model-i2_s.gguf",
  "size_bytes": 0,
  "sha256": "...",
  "nominal_model_size": "1B",
  "hf_display_model_size": "...",
  "format": "gguf",
  "quantization_route": "i2_s",
  "tokenizer_files": {},
  "license": "Falcon License",
  "claim_boundary": {
    "answer_ready": false,
    "cpu_ready": false,
    "cuda_ready": false,
    "apple_ready": false,
    "a770_ready": false,
    "speedup_claim": false,
    "server_ready": false
  }
}
```

Receipts must also record file list, cleanup status, local storage context, GGUF metadata when applicable, tokenizer metadata if embedded, and whether any HF displayed model-size metadata differs from the nominal size.

## Hard Rules

```text
No exact file hash, no artifact claim.
No tokenizer/prompt authority, no answer claim.
No GGUF/conversion authority, no backend proof.
No model binaries committed.
No third-party artifact substitution without artifact-authority decision.
```
