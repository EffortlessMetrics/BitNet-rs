# GGUF dense quant lane

This lane covers dense quantized GGUF artifacts for regular non-BitNet LLMs.

## Required gates

- GGUF metadata and tensor map inspection.
- Tokenizer source and hash.
- Prompt template id.
- Quantization type and model hash.
- Requested/selected backend and fallback status.
- One-token deterministic proof before broader claims.

GGUF compatibility is per artifact. A GGUF text-only receipt does not prove safetensors, ONNX, OpenVINO IR, multimodal projectors, MoE routing, full context, or speed.
