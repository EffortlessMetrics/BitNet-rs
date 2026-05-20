# BITNET-SPEC-B158-3B-ARTIFACT-CONTRACT

Status: proposed
Owner: BitNet-rs maintainers
Created: 2026-05-18
Linked proposal: [BITNET-PROP-0010](../proposals/BITNET-PROP-0010-bitnet-b158-3b-tl-model.md)
Linked specs: n/a
Linked ADRs: n/a
Linked plan: [3B TL implementation plan](../../plans/bitnet-b158-3b/implementation-plan.md)
Linked issues: n/a
Linked PRs: n/a
Support-tier impact: no support promotion; artifact inventory contract only
Policy impact: no policy exception

## Purpose

Define the minimum artifact inventory required before BitNet-rs may make any
artifact-known claim for `1bitLLM/bitnet_b1_58-3B`. The contract records source
identity, exact files, exact hashes, tokenizer/config authority, storage
context, cleanup status, and claim boundaries. It does not prove inference.

## Required artifact receipt

```json
{
  "artifact_kind": "bitnet_b158_3b_artifact_inventory",
  "model_id": "onebitllm_bitnet_b158_3b",
  "source_repo": "1bitLLM/bitnet_b1_58-3B",
  "source_revision": "af89e318d78a70802061246bf037199d2fb97020",
  "source_format": "safetensors_sharded",
  "official_gguf_present": false,
  "files": [
    {
      "name": "model-00001-of-00003.safetensors",
      "size_bytes": 4990000000,
      "sha256": "..."
    }
  ],
  "tokenizer_files": {
    "tokenizer_json_sha256": "...",
    "tokenizer_model_sha256": "...",
    "tokenizer_config_sha256": "...",
    "special_tokens_map_sha256": "..."
  },
  "claim_boundary": {
    "answer_ready": false,
    "cpu_ready": false,
    "cuda_ready": false,
    "apple_ready": false,
    "server_ready": false,
    "speedup_claim": false
  }
}
```

## Required inventory fields

The inventory receipt must record:

- source repository and exact source revision;
- all safetensors shard names, byte sizes, and SHA256 values;
- `model.safetensors.index.json` SHA256;
- `config.json` SHA256;
- `tokenizer.json` SHA256;
- `tokenizer.model` SHA256;
- `added_tokens.json` SHA256 when present;
- `special_tokens_map.json` SHA256;
- `tokenizer_config.json` SHA256;
- whether any official `.gguf` is present in the source listing;
- storage context including local cache path class, available space before and
  after probe, and cleanup status;
- explicit booleans for answer, CPU, CUDA, Apple, server, and speed claims.

## Hard rules

- No exact hashes, no artifact claim.
- No tokenizer authority, no answer claim.
- No approved GGUF or conversion route, no Rust backend proof.
- No model binaries, safetensors shards, GGUFs, tokenizer binaries from model
  repositories, or generated model outputs may be committed.
- Third-party GGUF substitution is diagnostic until an artifact-authority
  decision approves provenance, hash identity, tokenizer policy, and route
  metadata.
