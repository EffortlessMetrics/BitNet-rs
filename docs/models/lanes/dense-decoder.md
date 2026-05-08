# Dense decoder lane

Status: design lane / no runtime claim.

Dense decoder models are non-BitNet causal transformer families that should use dense GGUF
quantization kernels and the generalized transformer stack where appropriate. They must not
be routed through BitNet packed I2S/QK256 kernels as proof of support.

## Scope

The lane covers text-only causal decoding for dense families such as Gemma 4 E2B/E4B,
Gemma 4 31B, Qwen dense variants, Phi, Mistral, and related GGUF models.

## Required boundaries

- Dense quant kernels must identify themselves separately from BitNet kernels.
- `qk256_used=false` is required for regular dense LLM proofs.
- KV-cache semantics must be model-specific; per-layer KV ownership cannot be assumed for
  families with shared-KV behavior.
- Prefill and decode claims must be receipt-backed separately when performance is claimed.
- Model long-context capability is not the same as runtime proof at that context length.

## Gemma 4 notes

Gemma 4 E2B/E4B are the first dense-decoder foundation targets in this lane, but they need
family-specific handling before inference can be claimed:

- Per-Layer Embeddings are required for E2B/E4B strict mode.
- Shared KV is required for E2B/E4B strict mode.
- Sliding/global attention and p-RoPE behavior must be validated from metadata.
- Multimodal and MoE support remain future-gated.
