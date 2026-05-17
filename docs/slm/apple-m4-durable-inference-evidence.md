# Apple M4 Durable Inference Evidence

This page tracks the follow-on proof layer after the dense SLM eval v2, BitNet
eval/benchmark, BitNet productization, and inference-ops campaigns. Those
campaigns made the M4 path measured and operable; this lane keeps the evidence
fresh enough to support regression trends instead of one-time snapshots.

## Current Gap

The committed M4 report families are valid, but several dashboard groups still
have only one matching report identity. That is intentionally reported as
`insufficient_history`. To become a better appliance, the Mac mini needs repeat
refreshes under the same model/tokenizer/backend/fallback identity so the
dashboard can compare real drift.

Dense SLM stability also needs a longer resident benchmark profile. The v2
benchmark contract covered `resident_25` and `resident_50`; the durable refresh
adds `resident_100` as a bounded local/advisory profile.

`M4-DURABLE-002` records the first live dense SLM refresh with `resident_100`.
The new summaries live under:

```text
ci/hardware/apple-m4-mac-mini/2026-05-15T1845Z/slm-benchmark-v2/<model-id>/summary.json
```

The refresh validates as `apple_m4_slm_benchmark_v2`, uses
`apple-m4-cpu-neon`, records `fallback_used=false`, and keeps dense SLM evidence
separate from BitNet evidence. Strict `bitnet mac regression` comparisons
against the 2026-05-15 baseline intentionally stop with `profiles_required
mismatch` because the earlier baseline did not include `resident_100`; the
timestamped refresh therefore starts a new profile-set baseline.

## Dense SLM Refresh Contract

Run the full dense benchmark profile set in release mode:

```bash
target/release/bitnet --device apple-m4-cpu-neon mac benchmark \
  --model-id <model-id> \
  --profile short_prompt_16_out \
  --profile short_prompt_64_out \
  --profile long_prompt_16_out \
  --profile long_prompt_128_out \
  --profile context_1k \
  --profile context_4k \
  --profile resident_25 \
  --profile resident_50 \
  --profile resident_100 \
  --json-out ci/hardware/apple-m4-mac-mini/<date>/slm-benchmark-v2/<model-id>/summary.json
```

Each summary must validate as `apple_m4_slm_benchmark_v2`, record generated text
and token IDs in profile receipts, keep `fallback_used=false`, and retain the
claim boundary that this is bounded M4 Mac mini evidence, not a broad Apple
Silicon benchmark.

## Refresh Sequence

Use this order for the next durable evidence run:

```bash
bitnet mac status
bitnet mac report-refresh
bitnet mac regression-dashboard
target/release/bitnet mac benchmark ... --profile resident_100 ...
target/release/bitnet answer-corpus ...
target/release/bitnet mac bitnet-benchmark ...
target/release/bitnet mac bitnet-warm ...
bitnet mac report-refresh
bitnet mac regression-dashboard
bitnet mac receipts-check <new-summary.json> --json
bitnet mac regression <new-summary.json> --baseline <previous-summary.json>
```

Do not put live model downloads, long resident sessions, or hardware timing runs
in generic required PR CI. Keep them local, advisory, scheduled, or release-only.

## Dense Refresh Results

The 2026-05-15T1845Z dense refresh ran all supported dense M4 model IDs across
the full nine-profile set. Each summary records 201 prompts.

| Model | Generated | Overall TTFT p50 | Overall TTFT p99 | Input tok/s p50 | Output tok/s p50 | Decode tok/s p50 |
|---|---:|---:|---:|---:|---:|---:|
| `qwen2.5-0.5b-instruct-q8_0` | 2382 | 2150.0 ms | 262573.0 ms | 21.701 | 1.708 | 15.652 |
| `qwen2.5-0.5b-instruct-q4_k_m` | 2543 | 2150.0 ms | 262456.0 ms | 21.698 | 3.079 | 15.653 |
| `qwen2.5-1.5b-instruct-q4_k_m` | 2262 | 8184.0 ms | 822688.0 ms | 5.773 | 0.357 | 4.808 |

The long-context tail remains the main operator concern:

| Model | `context_1k` TTFT p50 | `context_4k` TTFT p50 | `context_4k` input tok/s p50 | `context_4k` decode tok/s p50 |
|---|---:|---:|---:|---:|
| `qwen2.5-0.5b-instruct-q8_0` | 52798.0 ms | 262608.0 ms | 15.526 | 9.958 |
| `qwen2.5-0.5b-instruct-q4_k_m` | 52772.0 ms | 262519.0 ms | 15.529 | 9.951 |
| `qwen2.5-1.5b-instruct-q4_k_m` | 182985.0 ms | 822691.0 ms | 4.954 | 3.572 |

The new `resident_100` profile gives a longer warm-session stability sample:

| Model | Generated | TTFT p50 | TTFT p99 | Decode tok/s p50 | Output tok/s p50 | Peak MB | Memory drift MB |
|---|---:|---:|---:|---:|---:|---:|---:|
| `qwen2.5-0.5b-instruct-q8_0` | 860 | 2150.0 ms | 2246.0 ms | 15.650 | 1.707 | 4156.750 | 1.875 |
| `qwen2.5-0.5b-instruct-q4_k_m` | 928 | 2151.0 ms | 2246.0 ms | 15.650 | 3.078 | 4159.609 | 0.968 |
| `qwen2.5-1.5b-instruct-q4_k_m` | 804 | 8078.0 ms | 8966.0 ms | 4.780 | 0.352 | 8395.047 | 0.000 |

## BitNet Refresh Results

The 2026-05-15T2214Z BitNet refresh reran the accepted Microsoft I2_S GGUF and
external tokenizer through the bounded eval, benchmark, and variable
warm-session surfaces. The new receipts live under:

```text
ci/hardware/apple-m4-mac-mini/2026-05-15T2214Z/bitnet-eval/answer-corpus.json
ci/hardware/apple-m4-mac-mini/2026-05-15T2214Z/bitnet-benchmark/summary.json
ci/hardware/apple-m4-mac-mini/2026-05-15T2214Z/bitnet-warm/variable-warm-session.json
```

All three receipts validate with `fallback_used=false`,
`selected_backend=apple-m4-cpu-neon`, `runtime_api=cpu`, and the accepted model
and tokenizer SHA256 identity.

The eval refresh matches the earlier 2026-05-15 BitNet eval context and has no
regression warnings:

| Cases | Passed | Failed | Timeout | Not run | Generated tokens | Regression warnings |
|---:|---:|---:|---:|---:|---:|---:|
| 100 | 75 | 25 | 0 | 0 | 765 | 0 |

The benchmark refresh also matches the earlier 2026-05-15 BitNet benchmark
context. Advisory regression comparison reports five warnings, all on sub-ms
prompt-tokenize timing fields (`0.205 ms` baseline to `0.241 ms` observed);
generation, fallback, identity, and prompt counts still validate.

| Prompts | Generated tokens | TTFT p50 | Input tok/s p50 | Output tok/s p50 | Decode tok/s p50 | Total wall p50 | Peak memory p50 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 8 | 7910.0 ms | 2.422 | 0.238 | 2.065 | 8388.460 ms | 4246.359 MiB |

The variable warm-session refresh ran five operator prompts with one exact
repeat. It loaded the model and tokenizer once, passed the repeated-prompt
determinism check, and produced the expected short outputs:

| Prompts | Generated tokens | Repeated prompt stable | TTFT p50 | Decode tok/s p50 | Total session |
|---:|---:|---|---:|---:|---:|
| 5 | 10 | yes | 7909.0 ms | 2.095 | 43955.621 ms |

`bitnet mac regression` supports BitNet eval, benchmark, and variable
warm-session receipts. `bitnet_apple_m4_warm_session` comparisons are
identity-strict and require the accepted artifact/tokenizer metadata, backend,
fallback state, prompt set, warm profile, timeout policy, and receipt schema to
match before timing or memory drift is reported.

## Dashboard History Results

`M4-DURABLE-004` regenerated the model-free report-refresh manifest and
regression dashboard from committed receipts only:

```text
target/apple-m4-durable-inference-evidence/report-refresh-manifest.json
target/apple-m4-durable-inference-evidence/regression-dashboard.json
target/apple-m4-durable-inference-evidence/regression-dashboard.md
```

The report-refresh manifest found five M4 evidence families and 14 committed
reports. It kept dense SLM and BitNet evidence separated, selected
`apple-m4-cpu-neon`, recorded `fallback_used=false`, and made no live model run
or model download.

The dashboard has nine matching-identity groups across those five families. Five
groups are comparable today:

| Family | Evidence | Comparable groups | Status |
|---|---|---:|---|
| `dense_slm_benchmark_v2` | dense SLM | 3 | `ready` |
| `bitnet_eval` | BitNet | 1 | `ready` |
| `bitnet_benchmark` | BitNet | 1 | `ready` |

Four groups still have insufficient matching history and must not be described
as trend-ready:

| Family | Evidence | Groups | Status |
|---|---|---:|---|
| `dense_slm_eval_v2` | dense SLM | 3 | `insufficient_history` |
| `bitnet_variable_warm` | BitNet | 1 | `insufficient_history` |

`bitnet mac receipts-check` passes for the regenerated dashboard receipt. The
dashboard may claim only that comparable matching-history groups now exist for
dense benchmark, BitNet eval, and BitNet benchmark reports; it must not claim a
trend for dense eval v2 or BitNet variable warm until another matching report is
committed for those identities.

## Operator Envelope V3

`M4-DURABLE-005` publishes the durable refresh layer in:

```text
docs/slm/apple-m4-operator-envelope-v3.md
```

The v3 envelope keeps the command-to-receipt map from
`docs/slm/apple-m4-operator-envelope-v2.md` and adds the operating contract from
the matching-history reports: refresh cadence, regression thresholds,
`resident_100` dense status, BitNet durable status, disk/cache guidance, and
claim boundaries.

The envelope is documentation and tracking only. It does not run a new model,
download an artifact, enable BitNet chat or serve, enable Metal, or widen the
M4 claim surface.

## Claim Boundary

This lane may claim only that the tooling and committed receipts support durable
M4 report refreshes and matching-identity comparisons. It must not claim:

- broad dense SLM quality;
- broad Apple Silicon performance;
- dense SLM evidence as BitNet evidence;
- BitNet chat or serve readiness beyond the existing gates;
- full `apple-m4-metal` inference;
- QK256, Neural Engine, MPSGraph, or MacBook evidence.
