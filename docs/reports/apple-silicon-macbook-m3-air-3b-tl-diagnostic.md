# Apple M3 MacBook Air 3B TL Diagnostic Preflight

Date: 2026-05-16
Work item: `M3MBA-007`

## Result

`M3MBA-007` is blocked before download or inference. The official
`1bitLLM/bitnet_b1_58-3B` repository at revision
`af89e318d78a70802061246bf037199d2fb97020` contains three safetensors shards
plus tokenizer/config files, but no `.gguf` artifact for the TL1/TL2
BitNet.cpp diagnostic command shape named by the work item.

Evidence receipt:

- `ci/hardware/apple-silicon-macbook/2026-05-16/m3-air/1bitllm-3b-tl-diagnostic-preflight.json`

## Source Probe

| Field | Value |
|---|---|
| Repository | `1bitLLM/bitnet_b1_58-3B` |
| Revision | `af89e318d78a70802061246bf037199d2fb97020` |
| API | `https://huggingface.co/api/models/1bitLLM/bitnet_b1_58-3B` |
| Tree | `https://huggingface.co/1bitLLM/bitnet_b1_58-3B/tree/main` |
| Model shards | `model-00001-of-00003.safetensors`, `model-00002-of-00003.safetensors`, `model-00003-of-00003.safetensors` |
| Model shard bytes | 13,297,592,664 |
| Tokenizer files | `tokenizer.json`, `tokenizer.model`, `tokenizer_config.json`, `special_tokens_map.json`, `added_tokens.json`, `tokenization_bitnet.py` |
| GGUF files | none |

## Local Preflight

| Field | Value |
|---|---:|
| Volume | `/System/Volumes/Data` |
| Available | 17,427,716 KiB |
| Total | 482,797,652 KiB |
| Capacity | 96% |
| Cache root | `/Users/sarahisaacs/Library/Caches/bitnet-rs/models` |
| BitNet.cpp runner | `/Users/sarahisaacs/.cache/bitnet_cpp/build/bin/llama-cli` |
| `huggingface-cli` | unavailable |

The runner exists, but there is no official GGUF file to pass to it. Downloading
the official safetensors shards would also leave about 4,435,547 KiB available,
below the 8 GiB hard free-space floor recorded for the MacBook artifact sweep.

## Decision

Do not download the safetensors shards and do not substitute a third-party GGUF
for `M3MBA-007`. A future unblocking PR needs one of:

- an official TL1/TL2 GGUF in `1bitLLM/bitnet_b1_58-3B`,
- a reproducible conversion path from the official safetensors repository to a
  GGUF supported by the intended runner,
- an explicitly approved third-party TL diagnostic artifact with source
  revision, SHA-256, tokenizer/pre-tokenizer authority, runner command,
  prompt-suite output, and cleanup status, and
- enough local free space that any approved large-candidate download leaves at
  least the hard free-space floor after download.

## Claim Boundary

This report records source availability and local preflight state only. It does
not claim 3B TL1/TL2 diagnostic output, 3B I2_S support, Rust Apple backend
support, Apple local-answer quality, Apple Metal BitNet inference, QK256 on
Apple Silicon, or broad Apple Silicon performance.
