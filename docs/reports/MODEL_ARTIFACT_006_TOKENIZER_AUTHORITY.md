# MODEL-ARTIFACT-006 Tokenizer Authority Audit

**Status:** diagnostic-only; no artifact is promoted to `answer_ready`

## Summary

`MODEL-ARTIFACT-006` audits tokenizer and pre-tokenizer authority for the
official Microsoft I2_S target and the `tdh111` alternate-quant control
artifact.

The result is narrower than the previous blocker:

- The official Microsoft I2_S GGUF truly lacks `tokenizer.ggml.pre` metadata.
- The source Microsoft `bitnet-b1.58-2B-4T` repository does publish an external
  `tokenizer.json` with explicit pre-tokenizer behavior.
- The external tokenizer's BPE merges hash matches the GGUF metadata.
- The GGUF chat template and the external `tokenizer_config.json` chat template
  are different authority sources.
- External tokenizer authority alone does not make the official I2_S GGUF
  answer-ready; the recorded intended-runner prompt suite still fails.
- The `tdh111` IQ2_BN_R4 repository does not publish tokenizer assets, so it
  remains alternate-quant control evidence with missing pre-tokenizer authority.

## Evidence Sources

| Source | Evidence |
|---|---|
| Local official GGUF | `models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf` |
| Local external tokenizer | `models/microsoft-bitnet-b1.58-2B-4T/tokenizer.json` |
| Hugging Face API | Repository file listings for Microsoft GGUF, Microsoft source model, and `tdh111` GGUF repos |
| Prior runner evidence | `docs/reports/MODEL_ARTIFACT_004_IKLLAMA_INTENDED_RUNNER.md` |

Machine-readable details are in
`ci/model-artifacts/tokenizer-authority.toml`.

## Official Microsoft I2_S

| Field | Value |
|---|---|
| Artifact | `microsoft_bitnet_b158_2b_4t_gguf_i2s_current` |
| Scope | `official_target` |
| Target alignment | `official_i2s_cuda_target` |
| GGUF SHA256 | `4221b252fdd5fd25e15847adfeb5ee88886506ba50b8a34548374492884c2162` |
| GGUF tokenizer model | `gpt2` |
| `tokenizer.ggml.pre` | missing |
| GGUF tokens | 128256 |
| GGUF merges | 280147 |
| GGUF merges SHA256 | `cfe2e1ca857e85780db48349c18f4dc6e5262573b9a25e79e15f93f2a2d43339` |
| GGUF chat template | `Human: ... BITNETAssistant:` |

The official GGUF repository listing exposes the GGUF file but no companion
tokenizer assets.

The source Microsoft model repository exposes tokenizer assets:

| File | SHA256 | Notes |
|---|---|---|
| `tokenizer.json` | `e134af98b985517b4f068e3755ae90d4e9cd2d45d328325dc503f1c6b2d06cc7` | BPE tokenizer with explicit Sequence pre-tokenizer. |
| `tokenizer_config.json` | `d27b698683435b0b0dd544a591a30196b2b63f5fd4c8c64b9060a84dc185386f` | `PreTrainedTokenizerFast`; `User:` / `Assistant:` chat template. |
| `special_tokens_map.json` | `462d91939dbc37178aa5a3eae7068d1990ccc92e09f288cc71f42cdf139d69cc` | Records BOS and EOS special tokens. |
| `config.json` | `2b43e80788972e6d53967b01ac3609d7df23cf0aabb0c866e51be694a59c1149` | `model_type = bitnet`; `tie_word_embeddings = true`. |
| `generation_config.json` | `af34a37cd006fd230106fcefc6bfdc1f775503aee05a80c174ccc5c6594f7054` | Generation defaults. |

The external tokenizer can supply pre-tokenizer authority for Rust diagnostics,
but it does not close the answer gate. Recorded intended-runner evidence still
shows the official I2_S artifact failing the deterministic prompt suite with
repeated colon output.

## tdh111 IQ2_BN_R4

| Field | Value |
|---|---|
| Artifact | `tdh111_bitnet_b158_2b_4t_iq2_bn_r4` |
| Scope | `alternate_quant_control` |
| Target alignment | `official_derived_alt_quant` |
| GGUF SHA256 | `a99001aaa5c1dc24acffe8035315c7d2970e82d8ccd3189383275c5d5a5287b5` |
| Runner | `ik_llama.cpp` |
| Prompt suite | passed |
| Tokenizer assets in repo | none recorded |
| Pre-tokenizer authority | missing |

This artifact remains useful control evidence: it proves an official-derived
alternate quant can produce readable tiny-suite outputs under its intended
runner. It does not unblock the official I2_S CUDA path and it still lacks
pre-tokenizer authority required by the shared gate.

## Answers to the Audit Questions

| Question | Answer |
|---|---|
| Is `tokenizer.ggml.pre` truly absent from the official GGUF? | Yes. Local GGUF metadata parsing found 24 metadata keys and no `tokenizer.ggml.pre`. |
| Does the Microsoft source repo include tokenizer assets? | Yes. It includes `tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json`, `config.json`, and `generation_config.json`. |
| Do those assets define the missing pre-tokenizer behavior? | Yes for external authority: `tokenizer.json` records a Sequence pre-tokenizer made from Regex Split plus ByteLevel. |
| Does the official GGUF repo itself include those assets? | No. The GGUF repo listing only exposes the GGUF file. |
| Does `tdh111` include tokenizer assets? | No tokenizer assets were listed for the `tdh111` GGUF repository. |
| Does external tokenizer authority promote official I2_S to `answer_ready`? | No. It supplies tokenizer evidence for diagnostics, but the official I2_S artifact still fails the recorded prompt-suite evidence. |

## Next Unblocker

The next artifact decision should be one of:

1. run official I2_S under a reference runner with the external tokenizer and
   prompt-template authority explicitly supplied, if the runner supports that;
2. regenerate official-target I2_S GGUF with embedded tokenizer/pre-tokenizer
   metadata and a prompt suite that passes under an intended reference runner;
3. keep official I2_S blocked and open a separate alternate-quant control lane
   for `tdh111`.

The direct RTX 5070 Ti answer path remains blocked until the official I2_S
target has both authority and coherent prompt-suite output.
