# Model support policy

bitnet-rs must not claim support for a model family until proof exists for the exact family, variant, task, backend, modality, quantization, tokenizer, prompt template, and fallback state.

## Required support dimensions

- Family and variant ID.
- Architecture kind and unsupported feature gates.
- Tokenizer source and prompt-template ID.
- Loader format and tensor mapping status.
- Requested backend, selected backend, and fallback status.
- Task shape: generation, multimodal text-only, MoE router smoke, graph reference, or token classification.
- Receipt path and hash/TBD status.

## Non-BitNet boundary

Non-BitNet dense and MoE models must not use BitNet W1.58, QK256, or hardware-kernel receipts as model support evidence unless that exact model actually uses those kernels and the receipt says so.
