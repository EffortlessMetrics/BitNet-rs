# Gemma 4 Foundation

Status: scaffold only. This document makes Gemma 4 visible in the existing architecture/catalog/receipt truth system; it does not claim runtime inference support.

## Claim boundary

- Generic `Gemma` support is not Gemma 4 support.
- Gemma 4 must use `ModelArchitecture::Gemma4`, not `ModelArchitecture::Gemma`.
- BitNet QK256/I2S kernels do not count as Gemma 4 dense inference proof.
- Text-only Gemma 4 support must not imply image, audio, video, MoE, MTP, full-context, or speedup support.
- Any future receipt must explicitly set `bitnet_kernel_used=false`, `qk256_used=false`, and a dense regular-LLM kernel family when claiming Gemma 4 dense inference.

## Variants

| Variant | Catalog id | First-class role | Context metadata | Vocab metadata | Key requirements | Current repo claim |
|---|---|---:|---:|---:|---|---|
| E2B IT | `gemma4-e2b-it` | First target | 128K | 262K | Dense decoder, PLE, shared KV, sliding/global attention | Catalog/detection scaffold only |
| E4B IT | `gemma4-e4b-it` | Second target | 128K | 262K | Dense decoder, PLE, shared KV, sliding/global attention | Catalog/detection scaffold only |
| 31B IT | `gemma4-31b-it` | Later dense target | 256K | 262K | Dense decoder, larger sliding window, image-capable model metadata | Partial/offload scaffold only |
| 26B-A4B IT | `gemma4-26b-a4b-it` | Future MoE target | 256K | 262K | Router, active-vs-total parameter accounting, expert dispatch | Design-only scaffold |

## First target

The first proof target is **Gemma 4 E2B IT, text-only, Q4 GGUF, CPU first, limited runtime context, strict receipt, no fallback**.

That proof does not exist yet. The first proof must show all of the following before this repository can claim Gemma 4 inference:

- Real GGUF metadata was loaded from an explicit model path.
- Tokenizer and prompt template authority came from the model or an explicit user-provided source.
- PLE tensors were loaded for E2B/E4B.
- Shared KV behavior was enabled for E2B/E4B.
- Sliding/global attention and per-layer RoPE behavior matched Gemma 4 metadata.
- A dense regular-LLM Q4/Q8/F16 kernel path was used.
- BitNet QK256 was not used.
- Fallback was not used.
- One token was generated and the receipt validated.

## Future-gated work

- Multimodal image/audio/video processors and projectors.
- 26B-A4B MoE router/expert kernels.
- Full 128K/256K long-context receipts.
- MTP draft model loading and speculative decode accounting.
- CUDA/Metal/OpenCL dense acceleration claims.
