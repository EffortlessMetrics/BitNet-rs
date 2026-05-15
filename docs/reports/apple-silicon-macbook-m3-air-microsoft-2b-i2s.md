# Apple M3 MacBook Air Microsoft 2B I2_S

Date: 2026-05-14
Work items: `M3MBA-005A`, `M3MBA-005B`, `M3MBA-005C`

## Result

The official Microsoft BitNet 2B I2_S GGUF identity was recorded on the Apple M3 MacBook Air lane.

Evidence receipts:

- `ci/hardware/apple-silicon-macbook/2026-05-12/m3-air/microsoft-2b-i2s-identity.json`
- `ci/hardware/apple-silicon-macbook/2026-05-12/m3-air/microsoft-2b-i2s-tokenizer-authority.json`
- `ci/hardware/apple-silicon-macbook/2026-05-12/m3-air/microsoft-2b-i2s-reference-output.json`

This is identity, hash, cache, storage, tokenizer-authority, diagnostic runner,
and M3 Air BitNet.cpp reference-output evidence only. `M3MBA-005C` accepts the
artifact only for the recorded M3 Air BitNet.cpp reference-runner context. It
does not accept the artifact for the repository's Rust Apple backend.

## Artifact

- Repository: `microsoft/bitnet-b1.58-2B-4T-gguf`
- Source: `https://huggingface.co/microsoft/bitnet-b1.58-2B-4T-gguf`
- Revision: `a1f2f1c765812aa8af3f6eda4a313707064bba15`
- Filename: `ggml-model-i2_s.gguf`
- Format: GGUF
- Architecture: `bitnet-b1.58`
- Quantization: `i2_s`
- Size: 1,187,801,280 bytes
- SHA-256: `4221b252fdd5fd25e15847adfeb5ee88886506ba50b8a34548374492884c2162`
- Header linked ETag: `4221b252fdd5fd25e15847adfeb5ee88886506ba50b8a34548374492884c2162`

The downloaded file size and local SHA-256 match the shared Apple candidate matrix and shared model artifact manifest.

## Cache And Storage

- Cache root: `/Users/sarahisaacs/Library/Caches/bitnet-rs/models`
- Local model path: `/Users/sarahisaacs/Library/Caches/bitnet-rs/models/microsoft-bitnet-2b-i2s/ggml-model-i2_s.gguf`
- Storage before download: 60,456,036 KiB available on `/System/Volumes/Data`
- Storage after hash verification: 59,294,360 KiB available on `/System/Volumes/Data`
- Hard free-space floor: 8,388,608 KiB
- Preferred free-space floor: 26,214,400 KiB
- Retention: retained for `M3MBA-005B` tokenizer authority and `M3MBA-005C` reference output decision
- Model binaries committed: false

Power was AC with the internal battery at 100% charged. `pmset -g therm` recorded no thermal, performance, or CPU power warning level.

## Authority References

- Shared answer gate: `docs/model-artifacts/ANSWER_ARTIFACT_GATE.md`
- Shared artifact manifest: `ci/model-artifacts/artifact-manifest.toml`
- Tokenizer authority ledger: `ci/model-artifacts/tokenizer-authority.toml`
- Model/kernel compatibility ledger: `ci/model-artifacts/model-kernel-compatibility.toml`
- Apple MacBook candidate matrix: `ci/hardware/apple-silicon-macbook/bitnet-candidate-matrix.toml`

The compatibility ledger records the Microsoft model as the official model family and records ARM `i2_s` as supported. The shared answer gate still requires external tokenizer/pre-tokenizer authority and reference prompt output before a backend lane can make local-answer claims.

## Tokenizer Authority

The M3 Air run confirms the shared tokenizer-authority ledger for the cached official Microsoft I2_S GGUF:

- GGUF tokenizer model: `gpt2`
- GGUF `tokenizer.ggml.pre`: missing
- GGUF token count: 128,256
- GGUF merges count: 280,147
- External tokenizer source: `microsoft/bitnet-b1.58-2B-4T`
- External tokenizer revision: `04c3b9ad9361b824064a1f25ea60a8be9599b127`
- External tokenizer file: `tokenizer.json`
- External tokenizer SHA-256: `e134af98b985517b4f068e3755ae90d4e9cd2d45d328325dc503f1c6b2d06cc7`
- Required runner override: `--override-kv tokenizer.ggml.pre=str:llama-bpe`

The required authority path is externally supplied pre-tokenizer behavior from the Microsoft source tokenizer, applied to the GGUF runner with `tokenizer.ggml.pre=llama-bpe`. This matches `ci/model-artifacts/tokenizer-authority.toml` and `docs/reports/MODEL_ARTIFACT_007_MICROSOFT_BITNETCPP_EXTERNAL_PRETOKENIZER.md`.

## M3 Runner Diagnostics

`cargo run --release --locked -p xtask --no-default-features -- fetch-cpp --backend cpu` built Microsoft BitNet.cpp at commit `01eb415772c342d9f20dc42772f1583ae1e5b102` with llama.cpp submodule commit `1f86f058de0c3f4098dedae2ae8653c335c868a1`. The local `llama-cli` reports version `3962 (1f86f058)`, built with Apple clang `17.0.0` for `arm64-apple-darwin25.3.0`.

The diagnostic prompt was:

```text
User: What is 2+2? Answer with one digit.<|eot_id|>Assistant:
```

With the required override:

```text
llama-cli -m <cache-root>/microsoft-bitnet-2b-i2s/ggml-model-i2_s.gguf \
  -p <diagnostic-prompt> \
  -n 8 --no-display-prompt --temp 0 --top-k 1 \
  --override-kv tokenizer.ggml.pre=str:llama-bpe \
  --no-mmap
```

the runner logged `validate_override: Using metadata override (  str) 'tokenizer.ggml.pre' = llama-bpe` and produced `4`.

Without the override:

```text
llama-cli -m <cache-root>/microsoft-bitnet-2b-i2s/ggml-model-i2_s.gguf \
  -p <diagnostic-prompt> \
  -n 8 --no-display-prompt --temp 0 --top-k 1 \
  --no-mmap
```

the runner logged `llm_load_vocab: missing pre-tokenizer type, using: 'default'` and the quality-degraded warning. That no-override path is therefore bad/no-authority evidence even though this single diagnostic prompt also produced `4`.

On this M3 build, BitNet.cpp selected Metal by default and logged `Apple M3`, `MTLGPUFamilyApple9`, unified memory, and `31/31` offloaded layers for both diagnostics. This report records that device behavior only as context for tokenizer-authority diagnostics; it does not promote Apple Metal BitNet inference or Apple local-answer readiness.

## M3 Reference Output Decision

`M3MBA-005C` ran the committed `ci/quality/bitnet-answer-corpus.yaml` prompt
suite through the local Microsoft BitNet.cpp `llama-cli` using the required
external pre-tokenizer override:

```text
llama-cli -m <cache-root>/microsoft-bitnet-2b-i2s/ggml-model-i2_s.gguf \
  --override-kv tokenizer.ggml.pre=str:llama-bpe \
  --no-mmap \
  -n <case.max_new_tokens> --temp 0 --top-k 1 --top-p 1 --min-p 0 \
  --seed 42 --no-display-prompt --no-warmup \
  -p "User: <case.question><|eot_id|>Assistant:"
```

The answer-gate result is accepted for the recorded M3 Air reference context:
5 passed, 0 failed, 0 not run, and no failing prompt IDs.

| Prompt id | Gate | Output | Result |
|---|---|---|---|
| `math_2_plus_2` | exact trimmed `4` | `4` | pass |
| `capital_france` | contains `Paris` | `Paris` | pass |
| `repeat_colors` | contains `red blue green` | `red blue green` | pass |
| `say_ok` | exact trimmed `OK` | `OK` | pass |
| `yes_no_water` | starts with `yes` or `no` | `Yes.` | pass |

The BitNet.cpp runner again selected the Apple M3 default device context and
logged `31/31` offloaded layers. That is the recorded reference-runner context
only. A separate forced `-ngl 0` diagnostic for the math prompt produced
`/controlphicewoman` and failed the gate; it is not used for this acceptance
decision because the M3MBA-005C command does not require `-ngl 0`. The failed
diagnostic is retained in the receipt so this item cannot be misread as a
CPU-only result.

The local GGUF remains retained in `/Users/sarahisaacs/Library/Caches/bitnet-rs/models`
for follow-on M3 Air qualification work. No model binary is committed.

## Claim Boundary

This report records official Microsoft 2B I2_S artifact identity,
tokenizer/pre-tokenizer authority, and accepted prompt-suite output for the M3
Air BitNet.cpp reference-runner context only. It does not claim Rust M4 BitNet
local answers, repository Rust Apple backend acceptance, M4 Mac mini proof,
Apple Metal BitNet inference, MPSGraph model inference, Neural Engine execution,
QK256 on Apple Silicon, speedup, or broad Apple Silicon performance.

Dense Qwen M3 evidence does not prove BitNet behavior.

## Next

Follow-on M3 Air work can now use this as reference-output evidence while still
proving each backend, device path, and performance claim separately.
