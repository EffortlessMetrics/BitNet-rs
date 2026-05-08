# Dense Decoder Architecture Lane

The dense decoder lane covers non-BitNet decoder-only or decoder-centric models
that use dense matrix kernels. It is separate from the BitNet QK256/I2S lane even
when the same CLI, catalog, tokenizer, loader, and receipt machinery are reused.

## Scope

- Uses the existing model architecture and catalog control plane.
- Uses dense quantized or floating-point kernels, not packed BitNet QK256 kernels.
- Preserves transformer concepts such as token embeddings, attention blocks,
  RoPE variants, MLPs, KV cache, prefill, and decode.
- Records exact kernel family and fallback state in receipts.

## Non-Claims

A dense decoder catalog entry or architecture enum variant is not an inference
claim. It does not prove tokenizer compatibility, prompt-template correctness,
GGUF tensor-map coverage, kernel parity, long-context support, or performance.

## Gemma 4 E2B/E4B Notes

Gemma 4 E2B and E4B are dense decoder targets, but strict support requires
Gemma-4-specific behavior before any proof can be accepted:

- Per-Layer Embeddings are required.
- Shared KV behavior is required.
- Sliding/global attention schedule must be validated from model metadata or an
  explicit variant expectation.
- Standard RoPE versus p-RoPE must be selected per attention kind.
- Text-only support must not imply image, audio, video, MoE, MTP, or full-context
  support.

## Receipt Expectations

A future dense decoder proof should record at least:

- `model_family`
- `variant`
- `task`
- `kernel_family`
- `dense_regular_llm=true`
- `bitnet_kernel_used=false`
- `qk256_used=false`
- `fallback_used=false`
- model-capability fields separately from runtime-coverage fields
