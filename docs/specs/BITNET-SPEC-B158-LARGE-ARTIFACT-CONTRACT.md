# BITNET-SPEC-B158-LARGE-ARTIFACT-CONTRACT

Status: proposed
Owner: BitNet-rs maintainers
Created: 2026-05-18
Linked proposal: [BITNET-PROP-0009 bitnet_b1_58-large control model](../proposals/BITNET-PROP-0009-bitnet-b158-large-control-model.md)
Linked specs: [conversion](BITNET-SPEC-B158-LARGE-CONVERSION.md), [tokenizer/prompt](BITNET-SPEC-B158-LARGE-TOKENIZER-PROMPT.md), [reference quality](BITNET-SPEC-B158-LARGE-REFERENCE-QUALITY.md), [answer artifact gate](../model-artifacts/ANSWER_ARTIFACT_GATE.md)
Linked ADRs: n/a
Linked plan: [bitnet_b1_58-large implementation plan](../../plans/bitnet-b158-large/implementation-plan.md)
Linked issues: n/a
Linked PRs: n/a
Support-tier impact: no support promotion; artifact contract only
Policy impact: no policy exception

## Purpose

Define the exact artifact authority required before BitNet-rs can treat
`1bitLLM/bitnet_b1_58-large` as more than a registered control-model candidate.
This contract covers source inventory and claim boundaries only; it does not
approve answer, backend, server, or speed claims.

## Contract identity

```toml
id = "onebitllm_bitnet_b158_large_07b"
source_repo = "1bitLLM/bitnet_b1_58-large"
source_revision = "pending_exact_inventory"
source_files = [
  "model.safetensors",
  "config.json",
  "tokenizer.json",
  "tokenizer.model",
  "tokenizer_config.json",
  "special_tokens_map.json",
]
model_family = "bitnet_b1_58"
role = "smaller_control_candidate"
format_source = "safetensors"
target_formats = ["gguf_f16_reference", "gguf_i2_s", "gguf_tl1", "gguf_tl2"]
official_gguf_present = false
third_party_artifact_allowed = false
```

The recorded Apple candidate matrix currently blocks the model at revision
`85d047191dcb224f0e04f20d26110caaf8dc1a47` because the inspected official repo
contains safetensors and tokenizer/config files but no official GGUF. A future
inventory may update `source_revision`, but it must keep historical receipts
immutable.

## Required receipt shape

```json
{
  "artifact_kind": "bitnet_b158_large_artifact_inventory",
  "source_repo": "1bitLLM/bitnet_b1_58-large",
  "source_revision": "...",
  "files": [
    {
      "name": "model.safetensors",
      "size_bytes": "...",
      "sha256": "..."
    }
  ],
  "tokenizer_authority": {
    "tokenizer_json": "...",
    "tokenizer_model": "...",
    "pretokenizer": "recorded_or_unknown"
  },
  "claim_boundary": {
    "answer_ready": false,
    "backend_ready": false,
    "speedup_claim": false
  }
}
```

Receipts may include additional fields, but they must not omit revision, file
sizes, SHA256 hashes, tokenizer authority, or explicit claim booleans.

## Required fields

Every artifact inventory receipt must record:

- `source_repo` and source URL;
- exact `source_revision` or immutable commit ID;
- file names, byte sizes, and SHA256 hashes;
- tokenizer/config file hashes;
- model family and role;
- source format and intended target formats;
- whether an official GGUF is present;
- whether any third-party artifact is allowed;
- storage context for hardware-lane runs;
- cleanup status;
- claim boundary for answer, backend, server, and speed.

## Hard rules

- No exact file hashes, no artifact claim.
- No tokenizer authority, no answer claim.
- No reference runner output, no backend answer work.
- No third-party GGUF without an explicit artifact-authority decision.
- No model binaries committed.
- No inherited official Microsoft 2B proof.
- No inherited dense SLM proof.
