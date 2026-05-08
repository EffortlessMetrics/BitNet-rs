# Gemma 4 Foundation

Status: scaffold only. This document records the Gemma 4 model-family boundary for `bitnet-rs`; it does not claim Gemma 4 inference support.

## Architecture identity

Gemma 4 is represented as `ModelArchitecture::Gemma4`, distinct from the older `ModelArchitecture::Gemma` family. Generic Gemma or Gemma 2 detection must not be treated as proof that Gemma 4 model semantics are implemented.

Detected names include:

- `gemma-4`
- `gemma4`
- `google/gemma-4-E2B-it`
- `google/gemma-4-E4B-it`
- `google/gemma-4-31B-it`
- `google/gemma-4-26B-A4B-it`

## Variants

| Variant | Lane | Catalog id | Context metadata | Runtime claim |
|---|---|---|---:|---|
| E2B IT | dense decoder | `gemma4-e2b-it` | 128K | first target, text-only scaffold |
| E4B IT | dense decoder | `gemma4-e4b-it` | 128K | second target, text-only scaffold |
| 31B IT | dense decoder | `gemma4-31b-it` | 256K | future-gated |
| 26B-A4B IT | MoE decoder | `gemma4-26b-a4b-it` | 256K | future-gated |

The first proof target is Gemma 4 E2B IT, text-only, Q4 GGUF, CPU first, strict receipt, and limited runtime context. The catalog context length records model metadata only; it is not a runtime long-context claim.

## Required implementation facts for E2B/E4B

E2B and E4B require Gemma 4-specific implementation work before strict inference can be claimed:

- Per-Layer Embeddings (PLE) must be loaded and enabled in strict mode.
- Shared KV behavior must be represented in the cache plan.
- Alternating sliding/global attention must be driven by model metadata or an explicit validated schedule.
- Global attention p-RoPE/unified-KV behavior must not be collapsed into the older generic Gemma path.
- Prompt-template and tokenizer authority must be explicit before receipts claim real model output.

## Claim boundaries

Gemma 4 is not a BitNet or QK256 proof path.

- BitNet QK256 kernels do not count as Gemma 4 dense inference proof.
- Text-only support must not imply image, audio, video, MoE, MTP, or full-context support.
- Dense Gemma 4 variants require dense regular-LLM kernels and receipts with `qk256_used=false`.
- The 26B-A4B MoE variant must report MoE status, active parameters, and total parameters before any future runtime claim.
- Multimodal capability must be receipt-gated separately from text-only loading.
- Long-context capability must be receipt-gated separately from model metadata.

## Acceptance boundary for the first real proof

Gemma 4 becomes a real runtime claim only when an E2B text-only strict receipt records:

- real GGUF weights;
- real tokenizer/template authority;
- `model_family=gemma4` and `variant=e2b-it`;
- PLE loaded and enabled;
- shared KV enabled;
- dense Q4 execution with `bitnet_kernel_used=false` and `qk256_used=false`;
- no fallback;
- no multimodal, MoE, MTP, speedup, or long-context claim; and
- at least one generated token from the strict CPU user path.
