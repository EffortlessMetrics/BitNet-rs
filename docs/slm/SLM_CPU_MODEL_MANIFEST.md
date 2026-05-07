# SLM CPU Model Manifest

This document defines the model-selection contract for the 8250U dense SLM CPU lane. It is intentionally separate from the BitNet CPU proof path: dense GGUF candidates may reuse loader, tokenizer, CLI, and receipt infrastructure, but they must not reuse BitNet QK256/I2_S layout assumptions.

## First Target Policy

The first target should be the smallest real instruct-capable dense GGUF whose metadata can be verified in-repo.

Initial priority:

1. `Qwen2.5-0.5B-Instruct` GGUF, `Q4_K_M` first and `Q8_0` if memory and time permit.
2. A Qwen small-family 1B-ish GGUF only after the 0.5B path works.
3. A small Gemma or Phi-family GGUF as a cross-family adapter test.

Do not anchor the lane on an ambiguous model name. A candidate is accepted only after the exact artifact path, SHA256, GGUF `general.architecture`, tokenizer metadata, tensor naming, quant format, and chat-template policy are recorded.

## Required Manifest Fields

Each candidate entry must record:

```toml
[[candidate]]
id = "qwen2_5_0_5b_instruct"
priority = 1
family = "qwen"
expected_architecture = "qwen2"
target_role = "first 8250U dense SLM proof"
recommended_quant = ["Q4_K_M", "Q8_0"]
claim_boundary = "bring-up and tiny answer corpus only"
artifact_status = "candidate_unverified"

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
