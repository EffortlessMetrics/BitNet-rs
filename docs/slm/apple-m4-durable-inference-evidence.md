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
adds `resident_100` as a bounded local/advisory profile. Adding the profile does
not by itself claim a new 100-prompt result.

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
target/release/bitnet mac bitnet-benchmark ...
bitnet mac report-refresh
bitnet mac regression-dashboard
bitnet mac receipts-check <new-summary.json> --json
bitnet mac regression <new-summary.json> --baseline <previous-summary.json>
```

Do not put live model downloads, long resident sessions, or hardware timing runs
in generic required PR CI. Keep them local, advisory, scheduled, or release-only.

## Claim Boundary

This lane may claim only that the tooling and committed receipts support durable
M4 report refreshes and matching-identity comparisons. It must not claim:

- broad dense SLM quality;
- broad Apple Silicon performance;
- dense SLM evidence as BitNet evidence;
- BitNet chat or serve readiness beyond the existing gates;
- full `apple-m4-metal` inference;
- QK256, Neural Engine, MPSGraph, or MacBook evidence.
