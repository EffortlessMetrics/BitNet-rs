# SLM CPU Model Manifest

This document defines the model-selection contract for the 8250U dense SLM CPU lane. It is intentionally separate from the BitNet CPU proof path: dense GGUF candidates may reuse loader, tokenizer, CLI, and receipt infrastructure, but they must not reuse BitNet QK256/I2_S layout assumptions.

## First Target Policy

The first target should be the smallest official dense GGUF whose metadata can be verified in-repo.

Initial priority:

1. `Qwen/Qwen3-0.6B-GGUF`, `Qwen3-0.6B-Q8_0.gguf`, because it is an official GGUF artifact with a pinned SHA256.
2. `Qwen2.5-0.5B-Instruct` as the conservative architecture fallback if a trusted GGUF or local conversion is pinned.
3. A Qwen small-family 1B-ish GGUF only after the 0.6B path works.
4. A small Gemma or Phi-family GGUF as a cross-family adapter test.

Do not anchor the lane on an ambiguous model name. A candidate is accepted only after the exact artifact path, SHA256, GGUF `general.architecture`, tokenizer metadata, tensor naming, quant format, and chat-template policy are recorded.

`qwen35`, `qwen3_5`, and related Qwen3.5 hybrid architectures are out of scope for this dense CPU lane. They require linear-attention/state-space and vision-path support that must be tracked as a separate architecture effort.

Artifact source: <https://huggingface.co/Qwen/Qwen3-0.6B-GGUF/blob/main/Qwen3-0.6B-Q8_0.gguf>.

## Required Manifest Fields

Each candidate entry must record:

```toml
[[candidate]]
id = "qwen3_0_6b_gguf_q8_0"
priority = 1
family = "qwen"
expected_architecture = "qwen3"
target_role = "first 8250U dense SLM proof"
recommended_quant = ["Q8_0"]
claim_boundary = "bring-up and tiny answer corpus only"
artifact_status = "official_artifact_identified_metadata_unverified"
repo = "Qwen/Qwen3-0.6B-GGUF"
file = "Qwen3-0.6B-Q8_0.gguf"
sha256 = "9465e63a22add5354d9bb4b99e90117043c7124007664907259bd16d043bb031"
source_url = "https://huggingface.co/Qwen/Qwen3-0.6B-GGUF/blob/main/Qwen3-0.6B-Q8_0.gguf"

[candidate.requirements]
sha256_required = true
gguf_architecture_required = true
tokenizer_metadata_required = true
tensor_name_audit_required = true
chat_template_required = true
```

## Claim Boundary

The 8250U SLM lane may claim only correctness and operability evidence:

- real dense GGUF metadata was read
- tokenizer source and strictness were recorded
- architecture adapter selection was explicit
- CPU backend was selected
- fallback was false
- prompt/generated token IDs and decoded text were captured

It must not claim sustained throughput, broad chat quality, server inference, GPU/OpenVINO/UHD 620 execution, NPU execution, or BitNet QK256 coverage.
