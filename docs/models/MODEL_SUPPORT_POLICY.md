# Model support policy

Model support is per family, variant, backend, loader format, quantization, modality, task, and receipt. A receipt for one text-only GGUF variant does not generalize to safetensors, multimodal inputs, MoE routing, long context, or speedup.

## Required support dimensions

- Family and variant ID.
- Loader format and artifact hash or explicit `TBD`.
- Tokenizer source and fallback status.
- Prompt template ID and mode flags.
- Requested and selected backend.
- Task type: generation, token classification, multimodal text-only proof, router smoke, or external reference.
- Explicit false fields for unsupported claims.

## Non-BitNet boundary

Non-BitNet dense and MoE families must not use BitNet W1.58 or QK256 kernel proof as evidence unless a future model explicitly uses that representation and has its own receipts.

