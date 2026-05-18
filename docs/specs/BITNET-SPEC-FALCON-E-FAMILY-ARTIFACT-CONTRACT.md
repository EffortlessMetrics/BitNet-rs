# BITNET-SPEC-FALCON-E-FAMILY-ARTIFACT-CONTRACT

Status: proposed
Owner: BitNet-rs maintainers
Created: 2026-05-18
Linked proposal: docs/proposals/BITNET-PROP-0013-falcon-e-family-supported-models.md
Linked specs: n/a
Linked ADRs: n/a
Linked plan: plans/falcon-e-family/implementation-plan.md
Linked issues: n/a
Linked PRs: n/a
Support-tier impact: artifact registration only; no answer/backend/speed promotion
Policy impact: no policy exception

## Candidate artifacts

```toml
[[falcon_e_artifact]]
id = "falcon_e_1b_instruct_i2s_gguf"
source_repo = "tiiuae/Falcon-E-1B-Instruct-GGUF"
source_format = "gguf"
file = "ggml-model-i2_s.gguf"
nominal_size = "1B"
hf_reported_model_size = "2B"
hf_repo_size = "666 MB"
quantization_route = "i2_s"
priority = 1

[[falcon_e_artifact]]
id = "falcon_e_3b_instruct_i2s_gguf"
source_repo = "tiiuae/Falcon-E-3B-Instruct-GGUF"
source_format = "gguf"
file = "ggml-model-i2_s.gguf"
nominal_size = "3B"
hf_reported_model_size = "3B"
hf_repo_size = "1,000 MB"
quantization_route = "i2_s"
priority = 2
```

## Required receipt fields

```json
{
  "artifact_kind": "falcon_e_artifact_inventory",
  "model_family": "falcon_e_158bit",
  "source_repo": "tiiuae/Falcon-E-1B-Instruct-GGUF",
  "source_revision": "...",
  "file": "ggml-model-i2_s.gguf",
  "size_bytes": 0,
  "sha256": "...",
  "format": "gguf",
  "quantization_route": "i2_s",
  "license": "falcon-llm-license",
  "tokenizer_authority": "pending",
  "prompt_authority": "pending",
  "claim_boundary": {
    "answer_ready": false,
    "cpu_ready": false,
    "cuda_ready": false,
    "speedup_claim": false
  }
}
```

## Acceptance

- Record source repo, source revision, exact file, file list, `size_bytes`,
  SHA256, GGUF metadata, license, tokenizer metadata if embedded, nominal size,
  displayed model size, storage context, and cleanup status.
- Keep 1B and 3B receipts separate.
- Treat prequantized, bfloat16, and fine-tuning variants as separate artifacts.
- Store receipts only; never commit model binaries.

## Hard rules

```text
No exact SHA256, no artifact claim.
No tokenizer/prompt authority, no answer claim.
No reference-good receipt, no CPU/backend answer claim.
No model binaries committed.
```
