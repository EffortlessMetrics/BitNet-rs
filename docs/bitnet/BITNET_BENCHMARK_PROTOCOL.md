# BitNet Benchmark Protocol

## Purpose

Generic hardware benchmark fields are not enough for BitNet. BitNet benchmarks must separate model loading, tokenization, prefill, first token, decode, sampling, and total generation.

## Required Timing Phases

Runtime phase definitions live in `docs/bitnet/BITNET_RUNTIME_PHASES.md`. Record these timings separately when available:

```json
{
  "timing": {
    "model_load_ms": 0,
    "tokenize_ms": 0,
    "prefill_ms": 0,
    "first_token_ms": 0,
    "decode_steady_state_tok_s": 0,
    "sampling_ms_per_token": 0,
    "total_ms": 0
  }
}
```

## Execution Fields

Every BitNet benchmark must record:

- Model repo/file/hash.
- Tokenizer.
- Quantization format.
- Kernel family.
- Selected backend.
- Runtime API.
- Execution phase.
- Prompt token count.
- Generated token count.
- Batch size.
- Thread count.
- Fallback status.
- Cache state.
- Hardware artifact path.

## Fixed Profiles

```yaml
profiles:
  smoke_1:
    prompt_tokens: small
    generated_tokens: 1
    purpose: proof path

  decode_128:
    prompt_tokens: 256
    generated_tokens: 128
    purpose: steady decode

  prefill_512:
    prompt_tokens: 512
    generated_tokens: 1
    purpose: prefill behavior

  long_context_4096:
    prompt_tokens: 4096
    generated_tokens: 16
    purpose: context limit behavior
```

## Phase Claims

| Phase | What it proves |
|---|---|
| `load_model` | Loader and model artifact path |
| `tokenize_prompt` | Tokenizer authority and prompt shape |
| `prefill` | Prompt processing performance |
| `first_token` | Time-to-first-token behavior |
| `decode_steady_state` | Autoregressive decode throughput |
| `sampling` | Sampling overhead |
| `total_generation` | End-to-end user-visible timing |

Do not use one phase to claim another.

## Hardware Linkage

BitNet benchmark artifacts must also follow:

- `docs/hardware/BENCHMARK_PROTOCOL.md`
- `ci/hardware/README.md`

Hardware context remains required:

- Machine ID.
- Driver/runtime versions.
- Power mode.
- Thermal state when available.
- Selected backend.
- Fallback status.

## Claim Rules

- A decode benchmark is not a prefill benchmark.
- A prefill benchmark is not full generation.
- First-token latency must not be hidden inside total generation.
- Sampling overhead must be separated when possible.
- Hardware fallback invalidates accelerator performance claims.
- Minimal loader fallback invalidates strict BitNet proof.
