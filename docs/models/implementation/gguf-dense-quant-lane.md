# GGUF dense quant lane

This lane covers GGUF dense quantized artifacts for non-BitNet models. It must track quantization type, tokenizer source, prompt template source, backend selection, fallback status, and whether CPU/GPU offload was used.

A GGUF text-only receipt does not prove safetensors loading, multimodal projector loading, MoE routing, full context, or speed.

