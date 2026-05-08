# Dense Decoder Architecture Lane

Status: design contract only. This lane describes how non-BitNet dense decoder families such as Gemma 4 should enter the existing truth system.

## Boundary

Dense decoder support is not BitNet support. Dense models must not route through QK256/I2S kernels or reuse BitNet receipts as proof of dense GGUF execution.

## Required model metadata

- Architecture family and variant.
- Context length as model capability, separate from loaded runtime context.
- Vocabulary size.
- Layer count.
- Attention schedule, including local/sliding versus global layers.
- RoPE mode and any per-layer p-RoPE/pruned-RoPE metadata.
- KV-cache ownership, including shared-KV mappings where applicable.
- Quantization family and tensor shapes from real GGUF inspection.

## Receipt fields for future proofs

A dense decoder receipt must include fields equivalent to:

```json
{
  "model_family": "gemma4",
  "variant": "e2b-it",
  "task": "text_generation",
  "dense_regular_llm": true,
  "bitnet_kernel_used": false,
  "qk256_used": false,
  "multimodal_claim": false,
  "moe_claim": false,
  "long_context_claim": false,
  "fallback_used": false
}
```

## Gemma 4 additions

Gemma 4 E2B/E4B require Per-Layer Embeddings and shared KV in strict mode. The dense lane therefore must fail honestly when those tensors or metadata are missing instead of silently falling back to older Gemma behavior.
