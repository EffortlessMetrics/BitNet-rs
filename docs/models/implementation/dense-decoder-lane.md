# Dense decoder lane

This lane covers non-BitNet dense autoregressive decoders.

It must not use QK256 or BitNet W1.58 kernel proof as evidence.

## First supported formats

- GGUF dense quantized CPU/GPU path.
- safetensors path later.
- OpenVINO graph/reference path later.

## Receipt requirements

Receipts must record:

- `dense_regular_llm=true`
- `bitnet_kernel_used=false`
- `qk256_used=false` unless the model actually uses QK256

This boundary exists because the current BitNet transformer path has QK256-specific dispatch in linear calls; non-BitNet models must not be silently routed through that and called supported.
