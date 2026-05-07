# M4-QA-MODEL-002 Artifact Search

**Date:** 2026-05-07
**Campaign:** `apple-m4-local-answer`
**Status:** blocked; no tested artifact can unblock `M4-QA-001`

## Summary

`M4-QA-MODEL-002` attempted to find or regenerate a supported local-answer GGUF/tokenizer artifact that passes the Apple M4 local-answer prompt suite under the reference runner. No tested artifact met the gate.

The blocker remains model/output quality. `M4-QA-001` must stay blocked, and the Apple M4 CPU/NEON path must not claim prompt-in, coherent-answer-out behavior yet.

## Gate

The artifact must pass the campaign prompt suite under a reference runner before it can be used for Rust-native Apple M4 CPU/NEON local-answer proof:

```text
ci/quality/apple-m4-local-answer-corpus.yaml
max_new_tokens=16
temperature=0.0
greedy=true
quality gates: math contains 4/four, France contains Paris, Rust contains Rust/programming terms
```

## Tested Candidates

| Candidate | Size | Reference status | Decision |
|---|---:|---|---|
| `microsoft/bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf` | 1.1 GiB | Loads, but lacks `tokenizer.ggml.pre` and fails all prompt gates | rejected |
| metadata-repaired Microsoft I2_S | 1.1 GiB | Added `tokenizer.ggml.pre=llama-bpe`; output stayed non-coherent | rejected |
| `jpacifico/Aramis-2B-BitNet-b1.58-i2s-GGUF/aramis-ggml-model-i2_s.gguf` | 1.1 GiB | Lacks `tokenizer.ggml.pre`; raw and metadata-repaired outputs stayed non-coherent | rejected |
| `RichardErkhov/1bitLLM_-_bitnet_b1_58-large-gguf/bitnet_b1_58-large.Q8_0.gguf` | 740 MiB | Has `tokenizer.ggml.pre=default`; France and Rust pass raw, math repeats prompt | rejected |
| `imi2/oxenai_1bitLLM_bitnet_b1_58-large-instruct-v2-gguf/TQ2_0` | 262 MiB | Has `tokenizer.ggml.pre=default`; France and Rust pass, math answers incoherently | rejected |
| `BoscoTheDog/Bessie_bitnet_instruct_100k_gguf/bitnet_100k.gguf` | 740 MiB | Published GGUF aborts in reference loader with `GGML_ASSERT(n_dims >= 1...)` | rejected |
| `Green-Sky/bitnet_b1_58-3B-GGUF/q2_2`, `q1_3`, and `old/q2_2` | 730-874 MiB | Published checksums match, but reference loader reports tensor data outside file bounds | rejected |
| `larenspear/bitnet_b1_58-large-GGUF/q4_k_m` | 430 MiB | Loads with `tokenizer.ggml.pre=default`, but fails the math gate under raw, instruct, and LLaMA-3 chat prompt shapes | rejected |
| `Bifrost-AI/Bitnet-b1.58-Bifrost-SOL-2B-4T-gguf/I2_S` | 1.1 GiB | Loads with missing pre-tokenizer warning; raw and LLaMA-3 chat prompt shapes fail all gates | rejected |
| regenerated Bessie Q8_0 from `BoscoTheDog/bitnet_instruct_q16_gguf` source | 878 MiB | Regenerated GGUF loads after local conversion fixes, but raw, instruct, and `Human:/BITNETAssistant:` prompts all fail output quality | rejected |

## Regeneration Attempt

The source Bessie model was downloaded to `target/` only and regenerated with a target-local copy of the BitNet GGUF conversion script. Three target-local conversion fixes were needed before the reference runner could load the regenerated file:

```text
avoid duplicate tokenizer.ggml.add_bos_token
return quantized tensor data with the i2 scale metadata expected by the writer
skip per-layer rotary_emb.inv_freq tensors that the local reference loader does not expect
```

The regenerated artifact:

```text
source repo: BoscoTheDog/bitnet_instruct_q16_gguf
source revision: 8d872253fbfe1c77cb1e17bc68a7a94d1ac027b2
source model.safetensors sha256: 8561617743135eecb8e36aa06a9315f77dc15903609b9d277cc9398b92ea5579
generated file: bessie-bitnet-instruct-100k-q8_0-regenerated.gguf
generated sha256: 4c9652a10dd9b9944480f5bed8ffa133598d87821a88fd41c7f055bde68b69da
tokenizer: SentencePiece, tokenizer.ggml.model=llama, tokenizer.ggml.pre=default
```

It still failed semantic quality:

| Prompt shape | Math | France | Rust |
|---|---|---|---|
| raw | fail | fail | fail |
| instruct | fail | fail | fail |
| `Human:/BITNETAssistant:` | fail | fail | fail |

## Storage Hygiene

Large candidate GGUFs, the downloaded source `model.safetensors`, regenerated GGUFs, target-local helper artifacts, and Cargo build output were removed after hashes and evidence were recorded in this report and the artifact manifest.

## Decision

`M4-QA-MODEL-002` is blocked. No tested artifact is allowed to unblock `M4-QA-001`.

The next unblocker needs one of:

1. an upstream-supported BitNet GGUF/tokenizer artifact that passes the prompt suite under the reference runner;
2. a source-to-GGUF regeneration recipe that produces coherent reference output, not just a loadable GGUF;
3. an explicit campaign decision to use a different supported local-answer model family, with the corpus prompt template and Rust loader support updated honestly.

Until then, do not weaken the quality gate and do not claim Apple M4 CPU/NEON coherent local answers.
