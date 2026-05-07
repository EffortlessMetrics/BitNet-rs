# Model receipt requirements

Receipts are the only way to advance a model-family claim past documentation/scaffold states.

## Common fields

Receipts must record `model_id`, `variant_id`, `model_hash_or_tbd`, `tokenizer_source`, `prompt_template`, `requested_backend`, `selected_backend`, `fallback_used`, `task`, `generated_tokens_or_labeled_tokens`, `full_inference_claim`, and `speedup_claim`.

## Coverage booleans

Every receipt must explicitly set false for unsupported coverage: multimodal, MoE, long-context, full inference, and speedup claims. Absence is not proof.

## Task-specific receipts

- Generative one-token receipts prove only deterministic one-token generation.
- Multimodal text-only receipts prove the text path only.
- MoE router receipts must include expert routing evidence.
- Token-classification receipts must include logits shape, label taxonomy, span decoder, and generation_claim=false.
- External-reference receipts must identify the external tool/backend and must not imply native bitnet-rs runtime support.
