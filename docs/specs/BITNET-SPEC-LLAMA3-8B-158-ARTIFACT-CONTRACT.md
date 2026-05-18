# BITNET-SPEC-LLAMA3-8B-158-ARTIFACT-CONTRACT

Status: proposed
Owner: model-artifacts
Created: 2026-05-18
Linked proposal: [BITNET-PROP-0011](../proposals/BITNET-PROP-0011-llama3-8b-158-supported-model.md)
Linked plan: [Llama3 8B 1.58 implementation plan](../../plans/llama3-8b-158/implementation-plan.md)
Support-tier impact: registered candidate only
Policy impact: no policy exception

## Purpose

Define the artifact identity requirements for
`HF1BitLLM/Llama3-8B-1.58-100B-tokens`. This spec records inventory authority;
it does not approve inference, conversion, CPU, CUDA, Apple, server, or speed
claims.

## Required artifact receipt

```json
{
  "artifact_kind": "llama3_8b_158_artifact_inventory",
  "model_id": "hf1bitllm_llama3_8b_158_100b_tokens",
  "source_repo": "HF1BitLLM/Llama3-8B-1.58-100B-tokens",
  "source_revision": "...",
  "source_format": "safetensors",
  "official_gguf_present": false,
  "upstream_parameter_count": "8.0B",
  "hf_display_model_size": "3B",
  "identity_discrepancy_recorded": true,
  "files": [
    { "name": "model.safetensors", "size_bytes": 0, "sha256": "..." }
  ],
  "tokenizer_files": {
    "tokenizer_json_sha256": "...",
    "tokenizer_config_sha256": "..."
  },
  "claim_boundary": {
    "answer_ready": false,
    "cpu_ready": false,
    "cuda_ready": false,
    "speedup_claim": false
  }
}
```

## Required inventory fields

The artifact inventory must record source repo, source revision, file list, file
sizes, SHA256 values, `config.json` hash, `generation_config.json` hash,
`tokenizer.json` hash, `tokenizer_config.json` hash, `model.safetensors` hash,
HF metadata, tensor/file types, model-size display, upstream support-table
parameter count, identity-discrepancy note, storage context, and cleanup status.

## Hard rules

- No exact hashes, no artifact claim.
- No tokenizer authority, no answer claim.
- No approved GGUF/conversion/runner route, no Rust backend proof.
- Inventory may record that a GGUF is absent, but absence must not be converted
  into a backend failure claim.
- Model binaries must remain local-only and must not be committed.
