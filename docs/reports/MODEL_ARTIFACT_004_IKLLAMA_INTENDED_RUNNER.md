# MODEL-ARTIFACT-004 ik_llama.cpp Intended-Runner Evidence

**Date:** 2026-05-07
**Campaign:** `model-artifacts`
**Status:** diagnostic-only; no artifact is promoted to `answer_ready`

## Summary

`MODEL-ARTIFACT-004` records intended-runner evidence for the
official-derived `tdh111` IQ2_BN_R4 BitNet candidate after `MODEL-ARTIFACT-003`
showed that stock `llama.cpp` cannot load it. The `tdh111` model card says the
IQ2_BN files are intended for `ik_llama.cpp`, so this PR checks that runner
directly.

The result is useful but does not unblock the Rust CUDA answer path:

- `tdh111_bitnet_b158_2b_4t_iq2_bn_r4` loads under `ik_llama.cpp` and passes the
  deterministic prompt suite with readable outputs.
- The runner still reports missing pre-tokenizer metadata, so the artifact does
  not satisfy the shared answer-artifact gate.
- The artifact is IQ2_BN_R4, not the official Microsoft I2_S GGUF targeted by
  the RTX 5070 Ti QK256/I2_S CUDA answer path.
- The official Microsoft I2_S GGUF loads under the same `ik_llama.cpp` runner
  but fails the prompt suite with repeated colon output.

This report does not change runtime behavior, tokenizer behavior, model loader
behavior, CUDA code, or answer quality gates.

## Runner

The target-local intended runner was built from `ik_llama.cpp`:

```text
repo = ik_llama.cpp
commit = 9a26522af234f8db079ae3735f35ab6c20fe2c66
llama-cli --version = version: 1 (9a26522)
build = CPU-only CMake/Ninja release build
cmake flags = GGML_NATIVE=ON, GGML_CUDA=OFF, LLAMA_CURL=OFF, LLAMA_BUILD_TESTS=OFF, LLAMA_BUILD_SERVER=OFF
```

The runner reported AVX-512 support on the 9950X3D host:

```text
HAVE_FANCY_SIMD is defined
AVX = 1
AVX2 = 1
AVX512 = 1
AVX512_VBMI = 1
AVX512_VNNI = 1
AVX512_BF16 = 1
```

The runner, cloned source, build output, and GGUF files stayed under `target/`
and are not committed.

## Command Shape

Each prompt used the same deterministic CPU-only command shape:

```powershell
target\model-artifacts\tools\ik_llama.cpp\build-bitnet-cpu\bin\llama-cli.exe `
  -m <candidate.gguf> `
  -ngl 0 `
  -t 8 `
  -c 4096 `
  -n <max-new-tokens> `
  --temp 0 `
  --top-k 1 `
  --top-p 1 `
  --min-p 0 `
  --seed 42 `
  --no-display-prompt `
  --no-warmup `
  -no-fa `
  -p "User: <prompt><|eot_id|>Assistant:"
```

The explicit `User: ... <|eot_id|>Assistant:` prompt shape avoided the double-BOS
early-stop behavior seen in an initial diagnostic run.

## Candidate Results

| Candidate | File | SHA256 | Runner result | Decision |
|---|---|---|---|---|
| `tdh111_bitnet_b158_2b_4t_iq2_bn_r4` | `bitnet1582b4t-iq2_bn_r4.gguf` | `a99001aaa5c1dc24acffe8035315c7d2970e82d8ccd3189383275c5d5a5287b5` | Loads under `ik_llama.cpp`; prompt suite passes. | rejected for answer readiness because pre-tokenizer authority is missing and this alternate IQ2_BN_R4 artifact does not unblock the official I2_S CUDA target |
| `microsoft_bitnet_b158_2b_4t_gguf_i2s_current` | `ggml-model-i2_s.gguf` | `4221b252fdd5fd25e15847adfeb5ee88886506ba50b8a34548374492884c2162` | Loads under `ik_llama.cpp`; prompt suite fails with repeated colon output. | remains rejected |

Both files produced:

```text
load: missing pre-tokenizer type, using: 'default'
```

That warning is part of the answer-readiness decision. A readable prompt-suite
result is not enough to satisfy the shared gate when the artifact lacks the
required tokenizer/pre-tokenizer authority.

## Prompt Suite Outputs

### tdh111 IQ2_BN_R4 under ik_llama.cpp

| Prompt id | Gate | Output | Result |
|---|---|---|---|
| `math_2_plus_2` | answer is `4` or starts with `4` | `4` | pass |
| `capital_france` | mentions Paris | `The capital of France is Paris.` | pass |
| `yes_no_water` | starts yes/no | `Yes.` | pass |
| `colors_four` | readable color list | `1. Red 2. Blue 3. Green 4. Yellow` | pass |
| `bitnet_one_sentence` | readable one-sentence answer | `BitNet is a computer network protocol designed for the exchange of text and binary data between computers, developed by the University of California, Berkeley.` | pass |

The last answer is readable and satisfies the current tiny-suite readability
gate, but it is not a factual explanation of the BitNet model architecture.
This report does not treat the tiny suite as a semantic evaluator.

### Microsoft I2_S under ik_llama.cpp

| Prompt id | Output | Result |
|---|---|---|
| `math_2_plus_2` | `::::::::::::::::` | fail |
| `capital_france` | `::::::::::::::::::::::::::::::::` | fail |
| `yes_no_water` | `::::::::::::::::` | fail |
| `colors_four` | `::::::::::::::::::::::::::::::::::::::::::::::::` | fail |
| `bitnet_one_sentence` | `::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::` | fail |

## Decision

No artifact is promoted to `answer_ready`.

The `tdh111` evidence is a useful intended-runner signal and should not be
discarded: it proves that at least one official-derived alternate quant can
produce readable tiny-suite answers under the runner it names. It still cannot
unblock CPU, CUDA, Apple M4, NPU, SLM, server, or speed claims because it lacks
the required pre-tokenizer authority and is not the official Microsoft I2_S
artifact targeted by the CUDA product path.

The official Microsoft I2_S artifact remains the relevant blocker for the RTX
5070 Ti lane. It is structurally loadable under `ik_llama.cpp`, but its
deterministic prompt-suite output remains non-coherent.

## Next Unblocker

The next useful artifact work is one of:

1. regenerate or acquire an official-target I2_S GGUF with tokenizer and
   pre-tokenizer authority that passes the prompt suite under a reference
   runner;
2. obtain an upstream Microsoft BitNet reference-runner recipe for the official
   I2_S artifact that produces coherent prompt-suite output; or
3. explicitly create a separate alternate-quant answer lane for the `tdh111`
   IQ2_BN_R4 artifact, with claim wording that does not imply official I2_S
   Rust CUDA readiness.

Until then, backend answer-readiness lanes remain blocked or diagnostic-only.
