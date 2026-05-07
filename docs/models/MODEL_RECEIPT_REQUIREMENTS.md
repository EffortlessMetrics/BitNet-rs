# Model receipt requirements

Receipts are the only way to advance from planning to proof. Every receipt must bind the claim to one model family, one variant, one artifact, one tokenizer policy, one prompt template, one backend decision, and one task.

## Required fields

- `model_id` and `variant_id`.
- `model_hash_or_tbd`.
- `tokenizer_source` and tokenizer fallback status.
- `prompt_template` and mode flags.
- `requested_backend`, `selected_backend`, and backend fallback status.
- `task` and generated-token or labeled-token coverage.
- `full_inference_claim` and `speedup_claim` booleans.

## Boundary fields

Receipts must explicitly record false claims for untested areas, including multimodal input, MoE routing, long context, speculative decoding, and speedups.

