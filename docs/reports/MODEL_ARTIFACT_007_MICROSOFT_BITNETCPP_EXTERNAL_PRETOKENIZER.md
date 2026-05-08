# MODEL-ARTIFACT-007 Microsoft BitNet.cpp External Pre-Tokenizer Evidence

## Summary

`MODEL-ARTIFACT-007` records a new intended-runner result for the official
Microsoft BitNet I2_S GGUF. The GGUF still lacks embedded `tokenizer.ggml.pre`,
but the source Microsoft model repository publishes a `tokenizer.json` with
explicit Llama/GPT-style BPE pre-tokenizer authority. Supplying that authority
to Microsoft BitNet.cpp with:

```text
--override-kv tokenizer.ggml.pre=str:llama-bpe
```

makes the official I2_S artifact pass the committed deterministic answer
corpus under the Microsoft BitNet.cpp reference runner.

This report does not claim Rust CPU, Rust CUDA, Apple, NPU, SLM, server, or
speed readiness. It only records that the official artifact is no longer blocked
at the shared model-artifact gate when paired with the documented external
tokenizer/pre-tokenizer authority and prompt-template command shape.

## Artifact

| Field | Value |
|---|---|
| Artifact id | `microsoft_bitnet_b158_2b_4t_gguf_i2s_current` |
| Repo | `microsoft/bitnet-b1.58-2B-4T-gguf` |
| Repo revision | `a1f2f1c765812aa8af3f6eda4a313707064bba15` |
| File | `ggml-model-i2_s.gguf` |
| SHA256 | `4221b252fdd5fd25e15847adfeb5ee88886506ba50b8a34548374492884c2162` |
| Size | `1187801280` bytes |
| Format | GGUF |
| Architecture | `bitnet-b1.58` |
| Quantization | `i2_s` |
| Target alignment | `official_i2s_cuda_target` |

## Tokenizer Authority

The GGUF metadata still does not contain `tokenizer.ggml.pre`; that is a real
metadata omission, not a Rust parser omission. `MODEL-ARTIFACT-006` recorded the
external Microsoft tokenizer assets:

| Field | Value |
|---|---|
| Source repo | `microsoft/bitnet-b1.58-2B-4T` |
| Source revision | `04c3b9ad9361b824064a1f25ea60a8be9599b127` |
| Tokenizer file | `tokenizer.json` |
| Tokenizer SHA256 | `e134af98b985517b4f068e3755ae90d4e9cd2d45d328325dc503f1c6b2d06cc7` |
| Pre-tokenizer authority | `externally_supplied` |
| Compatibility decision | Use `tokenizer.ggml.pre=llama-bpe` for reference-runner and backend-gate diagnostics. |

The external tokenizer's BPE merges hash matches the GGUF metadata, so the
compatibility decision supplies the missing pre-tokenizer behavior without
changing the GGUF bytes.

## Reference Runner

The runner was built locally with:

```powershell
cargo run --release --locked -p xtask --no-default-features -- fetch-cpp --backend cpu
```

The Windows bootstrap used Git Bash after PR #3983:

```text
Shell: C:\Program Files\Git\bin\bash.exe
Repository: C:\Users\steven\.cache\bitnet_cpp
BitNet.cpp commit: 01eb415772c342d9f20dc42772f1583ae1e5b102
llama.cpp submodule commit: 1f86f058de0c3f4098dedae2ae8653c335c868a1
llama-cli version: 3962 (1f86f058)
compiler: gcc.exe 13.2.0 MinGW-W64 x86_64-ucrt-posix-seh
```

On this Windows host the default memory-mapped load path failed before prompt
execution with:

```text
llama_model_load: error loading model: PrefetchVirtualMemory unavailable
```

The reference-runner command therefore used `--no-mmap`. This is a runner/host
loader workaround, not a model quality condition.

## Command Shape

Each prompt used this deterministic CPU command shape:

```powershell
C:\Users\steven\.cache\bitnet_cpp\build\bin\llama-cli.exe `
  -m D:\Code\Rust\BitNet\models\microsoft-bitnet-b1.58-2B-4T-gguf\ggml-model-i2_s.gguf `
  --override-kv tokenizer.ggml.pre=str:llama-bpe `
  --no-mmap `
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
  -p "User: <question><|eot_id|>Assistant:"
```

The runner logs `sampler seed: 4294967295`, but generation is deterministic for
this suite because the sampler chain is greedy with `--temp 0` and `--top-k 1`.

## Prompt-Suite Result

The committed corpus is `ci/quality/bitnet-answer-corpus.yaml`.

| Prompt id | Gate | Output | Result |
|---|---|---|---|
| `math_2_plus_2` | exact trimmed `4` | `4` | pass |
| `capital_france` | contains `Paris` | `Paris` | pass |
| `repeat_colors` | contains `red blue green` | `red blue green` | pass |
| `say_ok` | exact trimmed `OK` | `OK` | pass |
| `yes_no_water` | starts with `yes` or `no` | `No. Water is` | pass |

Additional readability prompts also produced coherent, printable continuations:

| Prompt id | Output |
|---|---|
| `colors_four` | `1. Red 2. Blue 3. Green 4. Yellow` |
| `bitnet_one_sentence` | `BitNet is a computer network protocol designed for the exchange of text and binary data between computers, developed by the University of California, Berkeley, in the 1970s.` |

The `bitnet_one_sentence` answer is readable but factually about a historical
network protocol rather than the BitNet model architecture. The shared tiny
suite is a constrained answer-readiness gate, not a broad factual evaluator.

## Decision

The official Microsoft I2_S artifact is promoted to `answer_ready` for backend
answer-readiness gates when paired with:

- exact GGUF SHA256
  `4221b252fdd5fd25e15847adfeb5ee88886506ba50b8a34548374492884c2162`;
- external Microsoft tokenizer SHA256
  `e134af98b985517b4f068e3755ae90d4e9cd2d45d328325dc503f1c6b2d06cc7`;
- explicit `tokenizer.ggml.pre=llama-bpe` compatibility decision;
- `User: <question><|eot_id|>Assistant:` prompt envelope;
- Microsoft BitNet.cpp runner evidence above.

This can unblock strict Rust CPU and Rust CUDA answer-readiness runs. Those
backend lanes still have to prove their own prompt-token parity, prefill/decode,
selected backend, selected kernel, fallback=false, receipt, and quality gates.

## Next Unblocker

The next PR should run strict Rust CPU against this answer-ready artifact and
external tokenizer authority, then compare prompt IDs and generated output
against the Microsoft BitNet.cpp reference-runner evidence. Only after Rust CPU
passes should the RTX 5070 Ti strict CUDA answer path be promoted from
diagnostic-only to answer-readiness proof.
