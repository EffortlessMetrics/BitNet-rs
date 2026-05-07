# Dense Decoder Lane

This lane covers non-BitNet dense autoregressive decoders.

It must not use QK256 or BitNet W1.58 kernel proof as evidence.

First supported formats:
- GGUF dense quantized CPU/GPU path.
- safetensors path later.
- OpenVINO graph/reference path later.

Receipts must record:
- `dense_regular_llm=true`
- `bitnet_kernel_used=false`
- `qk256_used=false` unless the model actually uses QK256

Current scope is planning only. Do not route non-BitNet models through QK256-specific dispatch and call them supported without receipts.
