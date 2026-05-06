# BitNet Reference Runs

## Purpose

Reference runs define what a BitNet proof is compared against. A hardware artifact without a reference path is not enough for BitNet correctness or performance claims.

## Reference Lanes

| Reference | Purpose |
|---|---|
| `bitnet.cpp` | External correctness/performance reference |
| `bitnet-rs scalar` | Internal correctness reference |
| `bitnet-rs AVX2/TL2/QK256` | Optimized CPU target |
| `BF16 model` | Model-shape/reference-only, not deployment proof |
| `OpenVINO llama.cpp GGUF` | External graph/reference lane, not native bitnet-rs proof |

## Reference Hierarchy

| Reference | Claim allowed |
|---|---|
| `bitnet-rs scalar` | Scalar reference |
| `bitnet-rs avx2/tl2/qk256` | Optimized CPU proof after scalar parity |
| `bitnet.cpp` | Compatibility/parity reference |
| BF16 model | Model-shape/reference path, not deployment performance |
| OpenVINO llama.cpp GGUF | External graph/runtime reference, not native bitnet-rs kernel proof |

## Canonical Prompt Fixtures

Use deterministic prompts first:

```text
Answer with a single digit: 2+2=
```

Settings:

```text
max_tokens = 1
temperature = 0.0
batch_size = 1
context = 512 for smoke
```

Longer profiles are defined in `docs/bitnet/BITNET_BENCHMARK_PROTOCOL.md`.

The canonical fixture manifest is:

```text
docs/bitnet/fixtures.yaml
```

Do not add model binaries to the repository. Fixture entries identify expected paths, hashes, prompt settings, and command shapes only.

## bitnet.cpp Reference Shape

Command shape:

```bash
python run_inference.py \
  -m models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf \
  -p "Answer with a single digit: 2+2=" \
  -n 1 \
  -t 4 \
  -c 512 \
  -temp 0
```

Record:

- bitnet.cpp commit or version.
- Model path and hash.
- Thread count.
- Context size.
- Prompt.
- Generated token count.
- Temperature.
- Output text/tokens.

## bitnet-rs Strict Reference Shape

Command shape:

```bash
BITNET_DISABLE_MINIMAL_LOADER=1 \
BITNET_STRICT_MODE=1 \
cargo run --locked -p bitnet-cli \
  --no-default-features \
  --features cpu,full-cli \
  -- run \
  --model models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf \
  --prompt "Answer with a single digit: 2+2=" \
  --max-tokens 1 \
  --temperature 0.0 \
  --json-out ci/hardware/<machine>/<date>/strict-bitnet-proof.json
```

Strict proof must record:

- Real GGUF loader mode.
- Tokenizer source.
- Model hash.
- Selected backend.
- Selected kernel family.
- Fallback status.
- Prompt and generation settings.
- Output token(s).

## Reference Rules

- `bitnet.cpp` is external reference, not automatic bitnet-rs proof.
- `bitnet-rs scalar` is internal correctness reference.
- BF16 weights are not packed-kernel performance proof.
- OpenVINO llama.cpp GGUF is graph/reference proof, not native bitnet-rs packed-kernel proof.
- Sampling tests must use `temperature=0.0` unless a seed and sampling policy are recorded.

## Artifact Naming

Reference artifacts should use the hardware artifact policy:

```text
ci/hardware/<machine-id>/<date>/strict-bitnet-proof.json
ci/hardware/<machine-id>/<date>/bitnet-cpp-reference.json
ci/hardware/<machine-id>/<date>/bitnet-rs-scalar-reference.json
```
