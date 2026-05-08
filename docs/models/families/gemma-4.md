# Gemma 4 foundation

Status: scaffold / metadata only. This document does not claim Gemma 4 inference works in
bitnet-rs.

## Architecture identity

Gemma 4 is represented as `ModelArchitecture::Gemma4`, distinct from the older
`ModelArchitecture::Gemma` family. Generic Gemma detection, Gemma 2 catalog entries, or
Gemma-style defaults are not sufficient proof of Gemma 4 support.

The first local target is:

```text
Gemma 4 E2B IT
text-only
Q4 GGUF
CPU-first reference execution
limited runtime context for proof runs
strict receipt with fallback=false
```

## Variants

| Variant | Foundation tier | Context metadata | Vocab metadata | Notes |
|---|---:|---:|---:|---|
| E2B IT | Local testable scaffold | 128K | 262K | Dense first target; requires PLE and shared KV. |
| E4B IT | Local testable scaffold | 128K | 262K | Dense second target; requires PLE and shared KV. |
| 31B IT | Partial/offload scaffold | 256K | 262K | Dense later target; no local inference claim. |
| 26B-A4B IT | Design-only scaffold | 256K | 262K | MoE future target; no local inference claim. |

## Required future metadata checks

A strict Gemma 4 loader must prefer model/GGUF metadata over hardcoded defaults and must
validate:

- exact variant (`e2b-it`, `e4b-it`, `31b-it`, or `26b-a4b-it`);
- layer count and hidden dimensions;
- vocabulary size;
- context length as a model capability, not an automatic runtime claim;
- sliding/global attention schedule;
- PLE tensors for E2B/E4B;
- shared-KV metadata for E2B/E4B;
- MoE router/expert metadata for 26B-A4B.

## Claim boundaries

- BitNet QK256 kernels do not count as Gemma 4 dense inference proof.
- Text-only support does not imply image, audio, video, MoE, MTP, tool, or full-context
  support.
- A catalog entry is not an inference claim.
- E2B/E4B strict mode must eventually fail if PLE tensors or shared-KV behavior are
  unavailable.
- The 26B-A4B entry is recognition/design metadata only until router, expert loading,
  shared expert handling, and active-vs-total parameter receipts exist.

## Receipt expectations for the first real proof

The first real proof must be a strict E2B text-only CPU receipt that records at least:

```json
{
  "model_family": "gemma4",
  "variant": "e2b-it",
  "task": "text_generation",
  "fallback_used": false,
  "dense_regular_llm": true,
  "bitnet_kernel_used": false,
  "qk256_used": false,
  "ple_enabled": true,
  "shared_kv_enabled": true,
  "multimodal_claim": false,
  "moe_claim": false,
  "mtp_claim": false,
  "long_context_claim": false
}
```

Until such a receipt exists, Gemma 4 remains scaffolded metadata only.
