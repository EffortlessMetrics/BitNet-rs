# Apple M4 SLM Performance

The Apple M4 SLM path is now a working local-answer baseline. This document defines the next performance campaign: measure release-mode warm-session behavior first, then remove overhead and expand acceleration only where receipts prove it.

## Current Baseline

Works today:

```text
Rust-native dense SLM local answers on apple-m4-cpu-neon, with strict model/tokenizer routing, fallback status, warm-session receipts, deterministic quality checks, model cache management, Mac CLI wrappers, and a first named Metal phase proof.
```

Not claimed:

```text
BitNet local-answer quality
full apple-m4-metal inference
Neural Engine execution
QK256 on Apple Silicon
general M4 performance
```

## First Performance Item

`M4-SLM-PERF-001` adds a release-mode baseline profile set for these profiles:

```text
warm_16
warm_32
warm_64
warm_128
```

Each profile should separate cold load from warm prompt timing and record:

```text
model_load_ms
tokenize_ms
prefill_ms
first_token_ms
decode_ms
sampling_ms
total_ms
tokens_per_second
peak_memory_mb
model_loaded_once
tokenizer_loaded_once
requested_backend
selected_backend
runtime_api
fallback_used
```

The release-mode command shape is:

```bash
cargo run --release --locked -p bitnet-cli \
  --no-default-features --features cpu,full-cli -- \
  mac validate \
  --profile-set performance \
  --json-out target/apple-m4-slm-performance/M4-SLM-PERF-001/release-baseline.json
```

The `performance` profile set must be run from a release build. Debug builds should fail before producing a performance receipt so profile artifacts cannot look like release-mode evidence by accident.

Local M4 proof receipt:

```text
ci/hardware/apple-m4-mac-mini/2026-05-08/slm-performance/release-baseline.json
```

Observed release-mode summary on the recorded M4 Mac mini:

| Profile | Generated tokens | Warm prompt tok/s | Decode tok/s | First token mean ms | Total session ms | Peak memory MB |
|---|---:|---:|---:|---:|---:|---:|
| `warm_16` | 34 | 4.266 | 14.517 | 1966.667 | 13085.842 | 3646.453 |
| `warm_32` | 50 | 5.879 | 15.187 | 1804.000 | 13513.646 | 4010.359 |
| `warm_64` | 82 | 7.663 | 14.909 | 1802.667 | 15483.482 | 4028.125 |
| `warm_128` | 123 | 8.847 | 14.925 | 1975.333 | 18977.237 | 4028.125 |

The receipt records `requested_backend=apple-m4-cpu-neon`, `selected_backend=apple-m4-cpu-neon`, `runtime_api=cpu`, and `fallback_used=false`. It also records `release_mode_observed=true`, `warm_128_included=true`, `speedup_claim=false`, and `broad_performance_claim=false`.

## Optimization Order

1. Measure release-mode warm-session bottlenecks.
2. Audit decode-loop allocation churn and temporary-buffer creation.
3. Harden resident-session reuse as the normal multi-prompt path.
4. Optimize the measured CPU/NEON bottleneck while preserving greedy token IDs.
5. Expand only parity-gated Metal phases with phase-specific receipts.
6. Add streaming and time-to-first-token UX.
7. Publish a measured performance envelope for supported profiles only.

## Claim Boundary

Performance claims must be tied to a receipt with:

```text
model
tokenizer
profile
backend
machine context
fallback status
timing fields
quality and determinism status
```

Do not report a speedup unless the before and after receipts use the same model, prompt/profile, backend, and generation settings. A named Metal phase can claim only that phase, not full `apple-m4-metal` inference.
