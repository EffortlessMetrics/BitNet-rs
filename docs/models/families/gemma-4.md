# Gemma 4 Foundation

Gemma 4 is tracked as a first-class non-BitNet model family. Generic Gemma or
Gemma 2 support is not Gemma 4 support, and Gemma 4 proof must stay inside the
existing architecture, catalog, backend-status, and receipt truth system rather
than a parallel model tracker.

## First Target

The first implementation target is **Gemma 4 E2B IT, text-only, Q4 GGUF, CPU
first, limited context**. The initial proof must be a strict receipt-backed text
generation proof with no fallback and no BitNet QK256 kernel use. E4B follows
only after the E2B path has a validated text-only proof.

## Architecture Identity

Gemma 4 must use `ModelArchitecture::Gemma4`, not `ModelArchitecture::Gemma`.
Detection should recognize at least these strings as Gemma 4:

- `gemma-4`
- `gemma4`
- `google/gemma-4-E2B-it`
- `google/gemma-4-E4B-it`
- `google/gemma-4-31B-it`
- `google/gemma-4-26B-A4B-it`

Gemma 2 strings such as `google/gemma-2-2b-it` must continue to resolve to the
older `Gemma` family.

## Variant Expectations

| Variant | Runtime priority | Layers | Context | Sliding window | Vocab | Required notes |
|---|---:|---:|---:|---:|---:|---|
| E2B IT | 1 | 35 | 128K | 512 | 262K | Dense text-first; PLE and shared KV required for strict proof |
| E4B IT | 2 | 42 | 128K | 512 | 262K | Dense text-first after E2B; PLE and shared KV required |
| 31B IT | Later | 60 | 256K | 1024 | 262K | Dense but partial/offload tier; no local proof claim yet |
| 26B-A4B IT | Future | 30 | 256K | 1024 | 262K | MoE design-only locally; receipts must record active vs total params |

The 26B-A4B variant is recognized for metadata and catalog routing only. It must
not imply local MoE inference support.

## Required Metadata Fields

A Gemma 4 metadata record should carry these fields before runtime work attempts
strict loading:

- `variant`
- `layers`
- `vocab_size`
- `context_length`
- `sliding_window`
- `has_ple`
- `has_shared_kv`
- `is_moe`
- `active_experts` and `total_experts` for MoE variants
- model modalities as capability metadata, not runtime coverage

## Strict Claim Boundaries

Gemma 4 scaffold may claim only architecture/catalog recognition and documented
future gates. It must not claim inference, multimodal loading, MoE dispatch,
full-context support, MTP/speculative drafting, performance, or hardware
acceleration.

BitNet QK256 kernels do not count as Gemma 4 dense inference proof. A future
Gemma 4 receipt must identify a dense regular-LLM kernel family and explicitly
record `bitnet_kernel_used=false` and `qk256_used=false`.

## Future-Gated Work

- Multimodal image/audio support requires separate processor, projector/encoder,
  placeholder-token, and `modalities_loaded` receipt coverage.
- MoE support requires router logits, top-k expert selection, shared expert
  handling, expert loading, and active-vs-total parameter receipt fields.
- Long-context support requires an explicit runtime claim and receipt; the model
  context length alone is not a bitnet-rs runtime claim.
- MTP/speculative drafting requires target/draft loading, accept/reject
  accounting, and dedicated receipt fields.
