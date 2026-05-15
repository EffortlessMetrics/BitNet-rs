# Apple M3 MacBook Air 0.7B 1bitLLM Preflight

Date: 2026-05-15
Work item: `M3MBA-006`

## Result

`M3MBA-006` is blocked before download or inference. The official
`1bitLLM/bitnet_b1_58-large` repository at revision
`85d047191dcb224f0e04f20d26110caaf8dc1a47` contains `model.safetensors` plus
tokenizer/config files, but no `.gguf` artifact for the BitNet.cpp GGUF command
shape named by the work item.

Evidence receipt:

- `ci/hardware/apple-silicon-macbook/2026-05-15/m3-air/1bitllm-07b-preflight.json`

## Source Probe

| Field | Value |
|---|---|
| Repository | `1bitLLM/bitnet_b1_58-large` |
| Revision | `85d047191dcb224f0e04f20d26110caaf8dc1a47` |
| API | `https://huggingface.co/api/models/1bitLLM/bitnet_b1_58-large` |
| Tree | `https://huggingface.co/1bitLLM/bitnet_b1_58-large/tree/main` |
| Largest model file | `model.safetensors`, 2,915,408,840 bytes |
| Tokenizer files | `tokenizer.json`, `tokenizer.model`, `tokenizer_config.json`, `special_tokens_map.json`, `added_tokens.json` |
| GGUF files | none |

## Local Preflight

| Field | Value |
|---|---:|
| Volume | `/System/Volumes/Data` |
| Available | 43,473,536 KiB |
| Total | 482,797,652 KiB |
| Capacity | 90% |
| Cache root | `/Users/sarahisaacs/Library/Caches/bitnet-rs/models` |
| BitNet.cpp runner | `/Users/sarahisaacs/.cache/bitnet_cpp/build/bin/llama-cli` |

The runner exists and the volume remains above the preferred free-space floor,
but there is no official GGUF file to pass to the runner for this candidate.

## Decision

Do not substitute an arbitrary third-party GGUF for `M3MBA-006`. The shared
artifact ledgers already reject prior 1bitLLM large-family third-party GGUFs for
answer readiness, including Q8_0 and TQ2_0 variants whose reference outputs
failed the deterministic prompt suite. A future unblocking PR needs one of:

- an official GGUF in `1bitLLM/bitnet_b1_58-large`,
- a reproducible conversion path from the official safetensors repository to a
  GGUF supported by the intended runner, or
- an explicitly approved third-party artifact with source revision, SHA-256,
  tokenizer/pre-tokenizer authority, runner command, prompt-suite output, and
  cleanup status.

## Claim Boundary

This report records source availability and local preflight state only. It does
not claim 0.7B answer readiness, BitNet quality, Rust Apple backend support, M4
Mac mini proof, Apple Metal BitNet inference, QK256 on Apple Silicon, or broad
Apple Silicon performance.
