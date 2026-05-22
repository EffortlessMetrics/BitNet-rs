# Apple M4 Inference Excellence

This page is the operator-facing map for the
`apple-m4-inference-excellence` campaign. It starts after the durable evidence
closeout: dense benchmark, BitNet eval, and BitNet benchmark groups have
matching-history comparisons, while dense SLM eval v2 and BitNet variable warm
were intentionally kept as `insufficient_history` until another matching
refresh landed. `M4-EXCELLENCE-001` records the second dense SLM eval-v2
refresh. `M4-EXCELLENCE-002` records the second BitNet variable warm refresh
under the dashboard-indexed `bitnet-productization` path.

The goal is not to prove that the M4 can run local inference. That is already
done for the supported dense SLM path and narrowly done for the accepted BitNet
one-shot and warm proof surfaces. The goal is to make the M4 a measured,
operator-ready appliance: repeatable evidence, larger mechanical evals,
complete benchmark envelopes, reproducible run identity, artifact provenance,
service conformance, BitNet-specific gates, better operator UX, and strict
claim boundaries.

## First Proof Gap

The first two items remove the remaining important matching-history gaps:

```text
M4-EXCELLENCE-001  second dense SLM eval-v2 refresh
M4-EXCELLENCE-002  second BitNet variable-warm refresh
```

The dense refresh writes
`ci/hardware/apple-m4-mac-mini/2026-05-16T0240Z/slm-eval-v2/<model-id>/summary.json`
for every supported dense M4 model identity. The BitNet variable warm refresh
writes
`ci/hardware/apple-m4-mac-mini/2026-05-16T0626Z/bitnet-productization/variable-warm-session.json`
for the accepted Microsoft I2_S GGUF and explicit external tokenizer identity.

`M4-EXCELLENCE-003` refreshes the model-free report manifest and regression
dashboard after both matching-history receipts landed. The refreshed dashboard
reports `status=ok`, five evidence families, 18 committed reports, nine
comparison groups, and nine comparable groups. Dense SLM and BitNet evidence
remain separate:

| Family | Evidence | Reports | Groups | Comparable groups |
|---|---|---:|---:|---:|
| `dense_slm_eval_v2` | dense SLM | 6 | 3 | 3 |
| `dense_slm_benchmark_v2` | dense SLM | 6 | 3 | 3 |
| `bitnet_eval` | BitNet | 2 | 1 | 1 |
| `bitnet_benchmark` | BitNet | 2 | 1 | 1 |
| `bitnet_variable_warm` | BitNet | 2 | 1 | 1 |

The dashboard refresh is not a live model run and does not change any chat,
serve, Metal, QK256, Neural Engine, MPSGraph, MacBook, broad quality, broad
performance, or speedup claim.

## Accuracy Depth

Before the large corpus work, the campaign freezes corpus and scorer identity:

```text
corpus IDs
seed generation rules
expected-output provenance
normalization rules
scoring schema
scorer self-tests
receipt version fields
```

`M4-ACCURACY-000` makes that contract machine-readable for the primary M4 eval
corpora. The dense SLM v2 corpus and BitNet eval corpus now carry
`metadata.corpus_contract` with:

| Field | Purpose |
|---|---|
| `contract_version` | Version of the shared corpus/scorer contract shape. |
| `corpus_id` and `corpus_version` | Stable corpus identity; prompt, expected-output, scoring, or family-count changes require a version bump. |
| `seed_generation_rules` | How deterministic cases are derived and how case IDs / `seed_material` preserve fixture inputs. |
| `expected_output_provenance` | Authority for expected answers; closed-form fixture answers are separate from model outputs or optional reference-runner evidence. |
| `normalization_rules` | The scoring normalization version and where strict exact matching remains strict. |
| `scoring_schema` | The mechanical scorer schema used by `answer-corpus`. |
| `scorer_self_tests` | Local tests that guard scorer behavior before pass rates are interpreted. |
| `receipt_contract` | Aggregate receipt contract expected from dry-run and live eval reports. |

`answer-corpus` receipts propagate these fields under `corpus.contract` and
`scoring_contract`, along with corpus metadata such as seed, generator policy,
case-count target, prompt template, and claim boundary. This is contract
readiness only; it does not create new live quality or performance evidence.

Dense SLM accuracy work expands the deterministic corpus in two stages:

```text
100 mechanical cases
500 mechanical cases
```

`M4-ACCURACY-001` is an evidence closeout for the 100-case stage, not a second
corpus expansion. The dense SLM eval-v2 corpus already contains 120
deterministic mechanically scored cases from seed `777331`, with coverage across
arithmetic, fixed-table QA, JSON/schema output, closed-label classification,
synthetic extraction, ordering/sorting, copy/edit/rewrite, constrained summary,
and instruction-following required/forbidden-token families. The earlier
`M4-SLM-EVAL2-001` item added that corpus; `M4-ACCURACY-000` adds the
machine-readable corpus/scorer contract needed before interpreting it inside the
excellence campaign.

`M4-ACCURACY-002` expands that same static dense SLM eval-v2 corpus to 500
deterministic cases while keeping scoring mechanical and reproducible. The
500-case distribution is `84` arithmetic, `42` numeric-tolerance, `50`
fixed-table QA, `42` JSON/schema, `50` closed-label classification, `50`
synthetic extraction, `50` ordering/sorting, `50` copy/edit/rewrite, `41`
constrained-summary, and `41` instruction-following required/forbidden-token
cases. This is corpus coverage only; it does not refresh runtime pass rates or
make a broad quality claim.

`M4-ACCURACY-003` then repairs the deterministic scorer and normalization path
exposed by the larger corpus. It keeps the 500 cases and expected answers
unchanged, bumps the dense corpus contract to `2.2.0`, strips known Qwen ChatML
stop tails and leading assistant separators before gates/scoring, allows
parseable fenced or embedded JSON payloads for JSON/schema scoring, and makes
required/forbidden keyword checks use token boundaries instead of raw substring
matches. It does not run models, refresh pass rates, prove BitNet behavior, or
make a broad quality claim.

`M4-ACCURACY-004` publishes
`ci/hardware/apple-m4-mac-mini/2026-05-16T0240Z/slm-eval-v2/task-family-pass-rates.json`
as a derived rollup from the committed 120-case `M4-EXCELLENCE-001` dense SLM
eval-v2 summaries. The rollup keeps exact model, tokenizer, backend,
`fallback_used`, prompt-template, source receipt, scoring, quality, task-family,
and claim-boundary fields per supported dense M4 model identity. It is a
current evidence publication for the matching-history refresh; it is not a
fresh runtime run and does not convert the 500-case static corpus into 500-case
runtime pass rates.

`M4-ACCURACY-005` adds deterministic failure-category fields for regression
triage: formatting, factual/table, extraction, refusal, timeout, schema, and
normalization. These fields are emitted alongside strict `failed_rules` and
legacy taxonomy labels; they are not an LLM judge and do not broaden dense SLM
quality claims.

`M4-ACCURACY-006` records the first full 500-case runtime pass-rate refresh on
the M4 for every supported dense model identity:

```text
ci/hardware/apple-m4-mac-mini/2026-05-16T1711Z/slm-eval-v2/<model-id>/answer-corpus.json
ci/hardware/apple-m4-mac-mini/2026-05-16T1711Z/slm-eval-v2/<model-id>/summary.json
ci/hardware/apple-m4-mac-mini/2026-05-16T1711Z/slm-eval-v2/task-family-pass-rates.json
```

The refresh keeps `apple-m4-cpu-neon`, `fallback_used=false`, catalog-pinned
GGUF SHA256 values, strict GGUF tokenizer authority, Qwen2.5 prompt template,
generated text, generated token IDs, task-family counts, and deterministic
failure categories. It is accuracy-depth evidence, not a broad quality or
performance claim. Memory and resident-stability fields in the summary receipts
remain sourced from the prior matching dense warm-session proof because
`answer-corpus` child receipts do not record fresh process RSS.

| Model | Strict score | Quality gate | TTFT p50 | TTFT p90 | Input tok/s p50 | Output tok/s p50 | Decode tok/s p50 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `qwen2.5-0.5b-instruct-q8_0` | 299 / 500 | 299 / 500 | 4214.0 ms | 11488.1 ms | 12.298 | 1.095 | 8.957 |
| `qwen2.5-0.5b-instruct-q4_k_m` | 297 / 500 | 297 / 500 | 2202.0 ms | 2793.8 ms | 21.988 | 2.237 | 15.626 |
| `qwen2.5-1.5b-instruct-q4_k_m` | 246 / 500 | 245 / 500 | 8724.0 ms | 11114.7 ms | 5.525 | 0.571 | 4.971 |

`M4-ACCURACY-007` repeats the full 500-case runtime refresh under the same
supported dense model identities:

```text
ci/hardware/apple-m4-mac-mini/2026-05-17T0045Z/slm-eval-v2/<model-id>/answer-corpus.json
ci/hardware/apple-m4-mac-mini/2026-05-17T0045Z/slm-eval-v2/<model-id>/summary.json
ci/hardware/apple-m4-mac-mini/2026-05-17T0045Z/slm-eval-v2/task-family-pass-rates.json
ci/hardware/apple-m4-mac-mini/2026-05-17T0045Z/report-refresh/report-refresh-manifest.json
ci/hardware/apple-m4-mac-mini/2026-05-17T0045Z/regression-dashboard/regression-dashboard.json
```

Each model summary validates with `bitnet mac receipts-check`, and each
matching regression check against `2026-05-16T1711Z` reports
`matched_context=true` with zero warnings. The refreshed dashboard reports
`status=ok`, `report_count=24`, `group_count=9`, and
`comparable_group_count=9`; the dense SLM eval-v2 family has three ready
groups. This is comparable matching-history evidence for the recorded 500-case
dense SLM identities only. It is not BitNet evidence, not a broad model-quality
claim, and not a broad performance benchmark.

| Model | Strict score | Quality gate | TTFT p50 | TTFT p90 | Input tok/s p50 | Output tok/s p50 | Decode tok/s p50 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `qwen2.5-0.5b-instruct-q8_0` | 299 / 500 | 299 / 500 | 2203 ms | 2771 ms | 22.010 | 2.206 | 15.628 |
| `qwen2.5-0.5b-instruct-q4_k_m` | 297 / 500 | 297 / 500 | 2201 ms | 2784 ms | 21.989 | 2.243 | 15.630 |
| `qwen2.5-1.5b-instruct-q4_k_m` | 246 / 500 | 245 / 500 | 8809 ms | 11336 ms | 5.505 | 0.574 | 4.949 |

Scoring stays mechanical:

```text
exact match
normalized match
numeric tolerance
JSON/schema validation
required keywords
forbidden tokens
closed-label classification
```

LLM-as-judge can be advisory only; it is not a required gate.

Small golden-token canaries stay separate from the full corpus. They record
prompt text, template identity, input token IDs, generated token IDs and text,
stop reason, sampler config, backend, fallback state, and artifact/tokenizer
identity so drift can be localized before running hundreds of cases.
`M4-CANARY-001` adds the `apple_m4_golden_token_canaries` receipt kind for a
compact dense SLM plus BitNet fixture. It is drift-localization evidence only,
not a fresh live runtime proof, broad quality claim, broad performance claim,
BitNet chat/serve enablement, or full Metal inference claim.

## Robustness And Negative Cases

`M4-ROBUSTNESS-001` adds a model-free dry-run lane for negative and robustness
fixtures before any live robustness claim is made:

```bash
bitnet mac eval --suite m4-robustness --dry-run --json
```

The suite is defined in `ci/quality/apple-m4-robustness-corpus.yaml` and covers
false-premise, ambiguous, instruction-conflict, prompt-injection-style,
format-trap, and unsupported-request cases. The receipt expands the same
fixtures into separate dense SLM and BitNet families with their own prompt
template, model identity scope, category summary, and mechanical scoring plan.

This lane is deliberately narrow. It proves that the M4 evidence system can
track robustness cases separately by model family, validate the corpus
contract, and reject overclaims through `bitnet mac receipts-check`. It does
not prove broad safety, alignment, factuality, dense-to-BitNet transfer,
BitNet chat, BitNet serve, Metal inference, QK256, Neural Engine, MPSGraph, or
speedup.

## Reproducible Run Identity

`M4-REPRO-001` introduces the shared `m4-run-identity-v1` contract in
`bitnet-receipts-core`. Receipts that opt into the contract carry a
top-level `run_identity` object and a matching `run_identity_sha256` digest.
The validator checks the reproducibility fields needed before dashboard,
regression, and release comparisons can trust that two reports are the same
kind of run:

| Field group | Requirement |
|---|---|
| Machine | M4 machine ID, SoC, OS name, OS version, and OS-version source. |
| Source and binary | Git commit plus source, crate version, and either build profile or binary SHA256. |
| Command | Command class and whether the receipt came from a live model run. |
| Model and tokenizer | Model ID/SHA or explicit model-free scope, tokenizer authority/SHA or explicit model-free scope. |
| Prompt template | Template ID and SHA256, even when the command is model-free. |
| Backend | Requested backend, selected backend, runtime API, and `fallback_used=false`; selected must match requested. |
| Evidence identity | Scope, seed, corpus ID, and profile ID. |
| Timing | Timing source used by the receipt family. |

The first schema-`1.2.0` model-free operator receipts using the contract are
`apple_m4_inference_status`, `apple_m4_operator_evidence_summary`,
`apple_m4_report_refresh_manifest`, and `apple_m4_regression_dashboard`.
`bitnet mac receipts-check` validates `run_identity` for schema `1.2.0`
receipts and for any receipt that includes `run_identity_sha256`.

Older committed M4 receipts remain valid with their existing schemas while
the excellence campaign refreshes eval, benchmark, warm-session, chat-gate,
serve-gate, and dashboard artifacts through later items. This is identity
infrastructure and operator receipt hardening only: it does not create a new
live model run, BitNet chat or serve proof, Metal route, QK256 route, broad
quality claim, broad performance claim, or speedup claim.

## Benchmark Depth

The benchmark envelope should cover:

```text
cold load
tokenizer load
prompt tokenization
prefill/input tokens per second
TTFT
output/decode tokens per second
sampling overhead
total wall time
peak memory
memory drift
```

Reports should include p50, p90, p99, and min/max where the receipt schema
supports them. Regression comparisons must match model, tokenizer, backend,
runtime API, fallback state, corpus or profile set, and machine identity before
describing drift.

`M4-BENCH-001` tightens the `apple_m4_slm_benchmark_v2` contract for future
receipts. New dense SLM benchmark summaries use `schema_version=1.1.0`, record
the supported profile set, require explicit timing / throughput / memory metric
lists, and include aggregate `sampling_ms_per_token_{p50,p90,p99}` alongside
the existing load, tokenize, prefill, TTFT, throughput, wall-time, and memory
fields. This is contract and validation readiness only; it does not add a new
live M4 benchmark result or publish a speed, memory, variance, or drift claim.

Benchmarkability also needs environment and variance evidence:

```text
macOS build
thermal and memory pressure when available
power state
disk/cache state
model cache root
background-load notes
run count and sample count
variance band
outlier handling
threshold derivation
```

`M4-BENCH-004` adds `bitnet mac benchmark-preflight`, a preflight receipt that
does not run model inference and does not record timing results. The receipt
captures the environment fields needed before timing drift can be interpreted:
git commit, crate/build profile, macOS product/build version, expected M4 SoC
and observed hardware probe, memory pressure snapshot, thermal pressure snapshot
when available, power state, supported model cache state, disk headroom, model
cache root, operator background-load notes, and explicit
`invalid_comparison_reasons`.

The preflight is a comparison gate, not a speed claim. A release benchmark can
only interpret timing drift when the benchmark receipt and preflight identity
match and `comparison_readiness.can_compare_timing=true`. Missing cache,
low-disk headroom, missing macOS build evidence, or missing git identity turn
the preflight into `invalid_for_comparison`; unavailable thermal/power/memory
pressure probes and missing background-load notes remain advisory warnings.

`M4-BENCH-007` adds `bitnet mac benchmark --calibrate`, a synthetic harness
calibration receipt that does not load a model and does not claim model speed.
The receipt records the `std::time::Instant` clock source and observed
resolution, runner overhead, warm-up and sample-discard policy, synthetic timing
fixtures, profile timeout rules, and invalid-comparison reasons. It is a
precondition for interpreting later benchmark envelopes, not a benchmark
envelope itself.

`M4-BENCH-002` publishes the committed dense SLM `slm-benchmark-v2` summaries
from `ci/hardware/apple-m4-mac-mini/2026-05-15T1845Z/slm-benchmark-v2/` as the
current full-profile benchmark envelope for the supported Qwen model set. Each
summary is a release-mode `apple_m4_slm_benchmark_v2` receipt with the nine
required profiles, 201 prompt runs, exact model SHA identity,
`selected_backend=apple-m4-cpu-neon`, `fallback_used=false`, p50/p90/p99 plus
min/max profile stats, and dense-only claim boundaries. The release
`mac receipts-check ... --json` validator passes for all three summaries.

| Model | TTFT p50 | Input tok/s p50 | Output tok/s p50 | Decode tok/s p50 | Peak memory p50 |
|---|---:|---:|---:|---:|---:|
| `qwen2.5-0.5b-instruct-q8_0` | 2150 ms | 21.701 | 1.708 | 15.652 | 4051.297 MB |
| `qwen2.5-0.5b-instruct-q4_k_m` | 2150 ms | 21.698 | 3.079 | 15.653 | 4053.719 MB |
| `qwen2.5-1.5b-instruct-q4_k_m` | 8184 ms | 5.773 | 0.357 | 4.808 | 8395.047 MB |

These are bounded M4 Mac mini dense SLM benchmark receipts only. They do not
prove BitNet performance, broad Apple Silicon performance, MacBook behavior,
Metal inference, QK256, Neural Engine, MPSGraph, or a speedup.

`M4-BENCH-006` keeps BitNet timing variance explicit instead of relying on the
shared benchmark envelope alone. It records one-shot and warm-session run
counts, sample counts, timeout-stage accounting, variance bands, outlier
handling, and advisory-vs-failure thresholds for the accepted BitNet artifact.

`M4-BENCH-008` is the harness prerequisite for `M4-BENCH-005`. The current
`bitnet mac benchmark --repeat <n>` surface writes an
`apple_m4_benchmark_variance_v1` aggregate over repeated dense SLM benchmark v2
child summaries. The aggregate records repeat count, completed count, profile
count, sample count, model cache identity, backend/fallback state, variance
band, outlier handling, invalid-comparison reasons, child summary receipts, and
strict claim boundaries. The validator requires release-mode
`apple-m4-cpu-neon`, `fallback_used=false`, a non-empty v2 profile set, one
metric sample per completed repeat, matching repeat/profile counts, generated
text/token-ID evidence flags, and no broad quality/performance, BitNet, Metal,
QK256, Neural Engine, MPSGraph, MacBook, or speedup claim.

The committed harness smoke at
`ci/hardware/apple-m4-mac-mini/2026-05-19T0746Z/benchmark-variance/harness-smoke.json`
uses `qwen2.5-0.5b-instruct-q8_0`, `short_prompt_16_out`, and `--repeat 2`.
It is harness evidence only. `M4-BENCH-005` remains responsible for publishing
live dense Qwen variance envelopes for the supported M4 dense SLM identities.
BitNet one-shot and warm-session timing variance stays in `M4-BENCH-006`,
using the accepted BitNet artifact and external tokenizer authority.

`M4-BENCH-009` closes the timeout-policy gap found while starting the live
dense variance run. The calibration receipt already declared per-profile
timeout rules, including `context_4k=720s`, but `mac benchmark` did not enforce
them. A local diagnostic run of the default dense model produced a first
`context_4k` prompt at roughly 1,032.7s, so that attempt is non-comparable and
must not be published as a variance envelope. The benchmark now runs each dense
profile through a child `slm-warm-session` process, kills the child when the
calibrated profile timeout is reached, writes an
`apple_m4_slm_benchmark_profile_timeout` receipt, and keeps that profile
invalid for envelope comparison. The committed enforcement smoke at
`ci/hardware/apple-m4-mac-mini/2026-05-19T0939Z/benchmark-timeout-enforcement/harness-smoke.json`
uses `qwen2.5-0.5b-instruct-q8_0`, `short_prompt_16_out`, and `--repeat 2` to
prove the child-process benchmark path still validates for a bounded profile.

`M4-BENCH-010` is the timeout-aware aggregation prerequisite found by the first
live `M4-BENCH-005` attempt. The default Q8 dense model reached the enforced
`context_4k` boundary at 720 seconds, wrote a valid
`apple_m4_slm_benchmark_profile_timeout` receipt with
`status=invalid_for_comparison`, `fallback_used=false`, and
`profile_timeout_exceeded:context_4k:720s`, then the variance command aborted
instead of writing a parent aggregate. The aggregate path now treats timed-out
profiles as invalid profile entries, skips them from timing and throughput
samples, propagates invalid-comparison reasons into child
`apple_m4_slm_benchmark_v2` and parent `apple_m4_benchmark_variance_v1`
receipts, and validates those receipts without converting timeout evidence into
a successful timing sample. This is still a prerequisite only:
`M4-BENCH-005` remains responsible for the live dense Qwen variance envelopes,
and BitNet variance remains `M4-BENCH-006`.

`M4-BENCH-005` publishes the first live dense Qwen repeatability receipts from
`ci/hardware/apple-m4-mac-mini/2026-05-19T1125Z/benchmark-variance/`. Each
supported dense Qwen identity was checked with `bitnet mac benchmark-preflight`
and then run through release-mode
`bitnet --device apple-m4-cpu-neon mac benchmark --repeat 2` over the nine
required profiles. The aggregate receipts validate as
`apple_m4_benchmark_variance_v1`, record `fallback_used=false`, preserve raw
repeat samples with no outlier filtering, and carry dense-only claim
boundaries.

| Model | Repeats | Samples | Prompt runs | Generated tokens | Comparison status | Invalid-comparison reasons |
|---|---:|---:|---:|---:|---|---|
| `qwen2.5-0.5b-instruct-q8_0` | 2 | 18 | 402 | 4300 | `invalid_for_comparison` | `profile_timeout_exceeded:context_4k:720s` |
| `qwen2.5-0.5b-instruct-q4_k_m` | 2 | 18 | 402 | 4780 | `invalid_for_comparison` | `profile_timeout_exceeded:context_4k:720s` |
| `qwen2.5-1.5b-instruct-q4_k_m` | 2 | 18 | 402 | 3586 | `invalid_for_comparison` | `profile_timeout_exceeded:long_prompt_128_out:420s`; `profile_timeout_exceeded:context_1k:360s`; `profile_timeout_exceeded:context_4k:720s` |

These receipts are useful repeatability and timeout-behavior evidence, not
final comparable timing envelopes. Because at least one calibrated profile
timed out for every model, `comparison_readiness.can_compare_timing=false` and
timing drift must not be interpreted from these aggregates. BitNet one-shot and
warm-session variance remains scoped to `M4-BENCH-006`; this evidence does not
claim BitNet timing variance, BitNet chat or serve behavior, Metal, QK256,
Neural Engine, MPSGraph, MacBook evidence, speedup, broad model quality, or
broad Apple Silicon performance.

`M4-BENCH-006` publishes the first explicit BitNet repeatability and variance
receipt for the accepted Microsoft I2_S artifact/tokenizer identity:

```text
ci/hardware/apple-m4-mac-mini/2026-05-19T2245Z/bitnet-benchmark-variance/summary.json
```

The release-mode command repeats the full `mac bitnet-benchmark` contract twice.
Each repeat runs one `mac ask` prompt and one fixed three-prompt warm session,
then the parent aggregate records repeat count, completed count, path count,
sample count, child summary receipts, timeout-stage accounting, raw p50/p90/p99
bands, min/max samples, outlier policy, and advisory-vs-failure threshold
language. The aggregate and all child receipts validate through
`target/release/bitnet mac receipts-check ... --json`.

| Field | Value |
|---|---:|
| Repeats requested/completed | 2 / 2 |
| Path samples | 4 |
| Prompt runs | 8 |
| Generated tokens | 16 |
| Timeouts | 0 |
| TTFT p50/p90/p99 | 7491 / 8486 / 8486 ms |
| Input tok/s p50/p90/p99 | 2.422 / 2.428 / 2.428 |
| Output tok/s p50/p90/p99 | 0.251 / 0.251 / 0.251 |
| Decode tok/s p50/p90/p99 | 2.082 / 2.083 / 2.083 |
| Peak memory p50/p90/p99 | 4322.875 / 4327.438 / 4327.438 MB |

The receipt preserves model SHA
`4221b252fdd5fd25e15847adfeb5ee88886506ba50b8a34548374492884c2162`, tokenizer
SHA `e134af98b985517b4f068e3755ae90d4e9cd2d45d328325dc503f1c6b2d06cc7`,
`selected_backend=apple-m4-cpu-neon`, and `fallback_used=false`. It is bounded
BitNet one-shot and fixed warm-session variance evidence only; it does not
enable BitNet chat or serve, prove dense SLM behavior, claim broad BitNet
quality, claim broad Apple Silicon performance, or widen Metal, QK256, Neural
Engine, MPSGraph, or MacBook support.

`M4-BENCH-003` wires benchmark receipts into the direct
`bitnet mac regression <current.json> --baseline <matching-baseline.json>`
path after the dense and BitNet variance lanes. Direct regression now accepts
`apple_m4_benchmark_variance_v1` baselines when the dense variance receipt is
valid for timing comparison, continues to compare `apple_m4_slm_benchmark_v2`
and `bitnet_apple_m4_benchmark_v1`, and labels drift warnings by category so
timing, memory, and quality signals stay separate from identity failures. A
dense variance receipt with `comparison_readiness.can_compare_timing=false`
or non-empty `invalid_comparison_reasons` remains a comparison blocker, not a
zero-warning timing result.

## Drift Thresholds

`M4-EXCELLENCE-004` publishes the current family-specific drift policy in the
operator envelope. Thresholds are only meaningful for dashboard groups whose
identity context is `ready`; identity mismatches start a new baseline instead
of creating a trend.

The published policy separates:

```text
identity mismatches and missing required fields: comparison blockers
quality and timeout regressions: zero-tolerance release blockers
timing drift: advisory warnings unless the lane uses --fail-on-drift
memory drift: advisory warnings unless the lane uses --fail-on-drift
unsupported claim flags: claim blockers
```

Dense SLM and BitNet share the same timing envelope shape where the receipt
families expose comparable fields: 20% higher load or sampling overhead, 15%
higher latency or lower input/output throughput, 12.5% lower decode throughput,
10% higher peak memory, and 15% higher memory drift. Quality thresholds stay
family-specific and strict: dense SLM eval v2 and BitNet eval allow no lower
mechanical pass counts and no higher timeout, failed, or not-run counts before
the release claim must stop for investigation.

These thresholds document the existing regression/dashboard behavior. They do
not add a live model run, prove new runtime quality, enable BitNet chat or
serve, or broaden Apple backend claims.

BitNet variable warm has matching-history dashboard evidence, receipt
validation, and direct `bitnet mac regression --baseline` support for
`bitnet_apple_m4_warm_session` receipts. The direct comparison remains
identity-strict: accepted artifact/tokenizer metadata, backend, fallback state,
prompt set, warm profile, timeout policy, and receipt schema must match before
timing or memory drift is reported.

## Reproducibility

Excellent M4 evidence needs enough identity to rerun or reject a comparison.
Receipts should record:

```text
machine ID and SoC
OS version
git commit
binary hash or build profile
command class
model ID and SHA256
tokenizer authority and SHA256
prompt template and stop criteria
generation parameters
backend and fallback state
corpus/profile seed
timing source
```

Artifact provenance is separate from runtime quality. A supported dense model
or accepted BitNet artifact should have a manifest for source, license or
redistribution boundary, file size, SHA256, tokenizer authority, prompt
template identity, local cache path, symlink target when used, and repair
command.

`M4-REPRO-002` publishes that manifest through `bitnet model verify <model-id>
--json` and the local `bitnet-model-cache.json` written after successful cache
verification. The manifest artifact kind is `m4_supported_model_provenance`;
it records upstream repo/revision/URL, license or redistribution boundary,
artifact size and SHA256, tokenizer authority and SHA status, prompt-template
identity, cache path, explicit verify path, symlink target when present, and
structured fetch/verify/prune repair commands. Dense Qwen tokenizer SHA is
recorded as embedded GGUF metadata bound to the model SHA; the accepted BitNet
artifact records the external tokenizer JSON SHA from the artifact authority
lane.

`M4-MODEL-LIFECYCLE-001` defines the supported-model lifecycle exposed by
`bitnet mac models` and `bitnet mac models --json`. The lifecycle is a policy
surface only: it does not add a supported model, change the default, enable
BitNet chat or serve, prove Metal/QK256/Neural Engine/MPSGraph/MacBook
behavior, or create broad quality, performance, speedup, or Apple Silicon
claims.

| State | Selectable | Required evidence before promotion or use | Cache migration behavior | Operator warning | Downgrade or rollback | Claim-boundary update |
|---|---|---|---|---|---|---|
| `default` | yes | Supported artifact provenance, dense eval-v2 and benchmark-v2 receipts for the exact identity, route-state and operator workload receipts, and a release-gate tracker event. | Fetch and verify the new default before docs or generated status change; keep the previous default cached until rollback guidance is published. | Treat default changes as release-gate changes, not broad model, backend, or platform support. | Revert the catalog row to the previous verified default, keep both cache entries until the rollback PR lands, and rerun `bitnet mac models` plus `bitnet model verify`. | Update the operator envelope, expectation envelope, route matrix, and tracker notes before any default claim changes. |
| `supported-non-default` | yes, explicit only | Supported artifact provenance, matching dense eval-v2 and benchmark-v2 receipts, route-state evidence that keeps the model explicit-only, and regression-dashboard history for the exact identity. | Add or refresh the model under its own cache id; do not replace the default cache or default model id. | Operators must pass `--model-id`; statements apply only to that exact non-default identity. | Mark the row deprecated or rejected when receipts regress; leave the default unchanged. | Update docs and dashboards only for the exact identity; do not widen dense, BitNet, or platform claims. |
| `supported-ask` | yes, explicit BitNet ask/warm only | Accepted BitNet artifact and external tokenizer authority, one-shot ask receipt, warm-session receipt, and route-state evidence that chat and serve stay separate. | Fetch and verify the BitNet artifact/tokenizer separately from dense cache entries; never make it the dense default. | BitNet chat and serve stay disabled unless later receipts explicitly enable those surfaces. | Mark the row deprecated or rejected if artifact, tokenizer, warm-session, timeout, or route-state receipts regress. | Update only BitNet one-shot or warm-session claims until separate chat and serve receipts pass. |
| `diagnostic-only` | no | Diagnosis receipt naming the blocker, exact artifact and tokenizer identity before promotion review, and a separate candidate item before user-facing routes. | Do not recommend fetch by default; keep any local diagnostic cache outside first-run guidance. | Diagnostic-only models are not user-ready and are not selectable by dense or BitNet M4 commands. | Keep diagnostic-only or move to rejected when the blocker is confirmed; do not promote without a new evidence item. | Docs may mention diagnosis scope only, not answer, chat, serve, quality, or performance readiness. |
| `candidate` | no | Exact source, revision, size, SHA256, tokenizer authority, prompt authority, cache verification, provenance manifest, and deterministic eval, benchmark, canary, and route-state receipts before promotion. | Use an explicit experiment cache path; do not add first-run fetch guidance or default cache migration. | Candidate rows are not supported runtime models and cannot satisfy release or operator readiness claims. | Move to rejected when artifact, tokenizer, quality, timing, or route evidence fails; otherwise keep as candidate until promotion gates land. | Candidate docs must say no supported-model, default, broad quality, performance, or platform claim is created. |
| `deprecated` | no | Replacement or regression receipt, operator migration warning, and rollback decision before restored support. | Stop recommending fetch; keep verify/prune guidance so existing operators can migrate deliberately. | Deprecated models are not user-ready for new work unless a rollback event says so. | Restore only through a fresh supported-model PR with current receipts, or continue to retired after migration evidence lands. | Remove or narrow active expectation-envelope claims and generated dashboard rows for the identity. |
| `rejected` | no | Rejection reason covering artifact, architecture, tokenizer, quality, timing, or scope failure. | Do not fetch; prune stale cache entries only when safe and operator-approved. | Rejected models are not M4 answer, chat, serve, quality, or benchmark evidence. | Do not roll back into rejected rows; open a new candidate item if new evidence appears. | Docs may only state why the identity is rejected and which separate work would be required. |
| `retired` | no | Retirement event naming replacement or end-of-support reason, cache cleanup guidance, and documentation removal or archival note. | Remove from first-run and recommendation surfaces; leave explicit prune instructions for old cache entries. | Retired models are unsupported and should not be used for new M4 inference receipts. | Return only as a new candidate with fresh artifact, cache, eval, benchmark, and route receipts. | Remove active support claims and keep only archival receipt references. |

The text and JSON catalog repeat the evidence, cache, warning, rollback, and
claim-boundary requirements on every row so operators can evaluate a model
state without reading tracker internals.

`M4-COMPAT-001` adds the compatibility-refresh contract for changes to macOS,
the Rust toolchain, binary build profile, or the supported-model manifest. The
model-free contract is written with:

```bash
bitnet mac compat-refresh \
  --json-out target/apple-m4-inference-excellence/compat/compat-refresh.json
```

The receipt requires follow-up compatibility evidence under
`ci/hardware/apple-m4-mac-mini/<date>/compat/`: `bitnet mac doctor`,
`bitnet mac smoke`, and `bitnet mac regression-dashboard`. It also records cache
repair cases for missing artifacts, SHA mismatches, tokenizer gaps, stale
symlinks, and low disk headroom, plus rollback guidance for OS, toolchain,
binary-profile, and model-manifest changes.

The compatibility receipt is not a live model run and does not prove unchanged
performance. Timing baselines stay advisory after a compatibility trigger until
matching benchmark identities pass again; quality claims still require matching
eval identity, and dense SLM evidence remains separate from BitNet evidence.

`M4-REPRO-003` adds runtime prompt-generation identity to the receipt surface.
Dense SLM and BitNet eval, ask, warm/chat, and serve receipts should carry a
`prompt_generation_identity` object with template family/source/hash, tokenizer
authority, stop criteria and stop hashes, generation parameters and parameter
hash, plus an overall identity hash. Receipt validation treats those hashes as
comparison hygiene only: they make mismatched templates, stop rules, or sampling
parameters visible before regression comparison, but they do not prove improved
quality, speed, chat enablement, serve readiness, or acceleration.
`M4-REPRO-004` tightens that validation by comparing canonical prompt-template
families instead of raw labels, so aliases for the same Qwen2.5 ChatML template
such as `qwen2.5` and `qwen25-chat` do not block otherwise valid dense M4
receipts while genuinely different template families still fail.

Dense SLMs also get a bounded reference-vs-Rust control so reference runner,
template, tokenizer, and Rust behavior can be distinguished without using that
control as broad model-quality evidence.
`M4-DENSE-REF-000` must land before live dense comparison receipts: it defines
the `apple_m4_slm_reference_vs_rust_comparison_v1` receipt contract and
validator for supported Qwen identities, reference-runner command identity,
prompt/template/tokenizer authority, Rust token IDs, mechanical-score deltas,
summary totals, token-ID availability, and claim-boundary flags. `M4-DENSE-REF-001`
then records the bounded live comparison evidence against that validator.
The validator is exercised by
`crates/bitnet-cli/tests/fixtures/apple-m4-dense-reference-vs-rust-comparison.json`;
that fixture is schema evidence only and does not stand in for reference-runner
or live Rust M4 output.
`M4-DENSE-REF-001` records the bounded live comparison under
`ci/hardware/apple-m4-mac-mini/2026-05-18T1115Z/slm-reference-vs-rust/` for
the three supported Qwen identities. The receipts compare 7 deterministic
quality prompts against `llama-cli` CPU-only `-ngl 0` reference output and Rust
`apple-m4-cpu-neon` `mac validate` output. Reference generated token IDs are
recorded as unavailable because the reference runner does not expose them in
this path; Rust generated token IDs remain recorded.

## BitNet Ladder

BitNet remains separate from dense Qwen evidence. The campaign keeps this proof
ladder:

```text
BitNet-specific corpus
reference-vs-Rust comparison
one-shot benchmark envelope
variable warm 25/50/100
progress and timeout UX
task-family pass rates
failure taxonomy
matching-history eval refresh
chat gate
serve gate
```

`M4-BITNET-EX-001` reuses the existing
`ci/quality/apple-m4-bitnet-eval-seeded-corpus.yaml` corpus from the earlier
BitNet eval/benchmark lane instead of creating a duplicate fixture. That corpus
already records the accepted Microsoft I2_S GGUF identity, external tokenizer
authority, `bitnetcpp-answer` prompt template, deterministic seed `912587`, and
100 mechanically scored BitNet-specific cases split evenly across ten task
families. The reuse is corpus/evidence hygiene only; it does not create a fresh
runtime run or widen BitNet chat, serve, Metal, QK256, Neural Engine, MPSGraph,
MacBook, speedup, broad quality, or broad performance claims.

`M4-BITNET-EX-002` adds a reference-vs-Rust comparison slice for that same
100-case corpus. The reference side is generated by the local Microsoft
BitNet.cpp `llama-cli` runner with the accepted I2_S GGUF, the
`tokenizer.ggml.pre=str:llama-bpe` override, greedy sampling, and the
`bitnetcpp-answer` prompt shape. It is compared against the existing validated
Rust M4 eval receipt from
`ci/hardware/apple-m4-mac-mini/2026-05-15T2214Z/bitnet-eval/answer-corpus.json`.
The comparison records 100 reference texts, Rust texts and generated token IDs,
reference token-ID unavailability, text matches, and mechanical scoring deltas.
It is not a fresh Rust runtime run and does not widen BitNet chat, serve, Metal,
QK256, Neural Engine, MPSGraph, MacBook, speedup, broad quality, or broad
performance claims.

Recorded comparison artifacts:

```text
ci/hardware/apple-m4-mac-mini/2026-05-17T0810Z/bitnet-eval/reference-runner-output.json
ci/hardware/apple-m4-mac-mini/2026-05-17T0810Z/bitnet-eval/reference-vs-rust-comparison.json
ci/hardware/apple-m4-mac-mini/2026-05-17T0810Z/bitnet-eval/answer-corpus-reference-comparison.json
```

Summary:

```text
reference prompts: 100 completed
text matches: 55 / 100
mechanical scoring matches: 77 / 100
reference pass / Rust fail: 9
Rust pass / reference fail: 14
reference generated token IDs: unavailable from the reference runner
Rust generated token IDs: recorded in the source M4 eval receipt
```

`M4-BITNET-EX-008` publishes a standalone BitNet task-family and
failure-taxonomy rollup derived from those committed receipts:

```text
ci/hardware/apple-m4-mac-mini/2026-05-17T0810Z/bitnet-eval/task-family-pass-rates.json
```

The report is BitNet-only, uses the accepted Microsoft I2_S GGUF and external
tokenizer identity, derives from the committed M4 Rust answer-corpus receipt
and the reference-vs-Rust comparison receipt, and explicitly records that it is
not a fresh runtime run. Dense Qwen evidence is not used. The source Rust
receipt remains the 100-case M4 CPU/NEON run from
`2026-05-15T2214Z`; reference generated token IDs remain unavailable because
the reference runner did not expose them.

Overall mechanical score:

```text
Rust M4 passes: 75 / 100
timeouts: 0
not run: 0
reference-vs-Rust text matches: 55 / 100
reference-vs-Rust mechanical scoring matches: 77 / 100
reference pass / Rust fail: 9
Rust pass / reference fail: 14
```

| Task family | Rust pass rate | Failure taxonomy | Reference text match | Reference scoring match |
|---|---:|---|---:|---:|
| arithmetic_exact | 10 / 10 | none | 10 / 10 | 10 / 10 |
| closed_label_classification | 9 / 10 | `answer_content=1` | 6 / 10 | 7 / 10 |
| constrained_summary | 9 / 10 | `answer_content=1` | 3 / 10 | 9 / 10 |
| fixed_table_qa | 6 / 10 | `answer_content=4` | 2 / 10 | 6 / 10 |
| format_constrained_json | 5 / 10 | `fenced_json=5`, `format_only=5` | 3 / 10 | 5 / 10 |
| numeric_tolerance | 5 / 10 | `answer_content=5` | 3 / 10 | 5 / 10 |
| ordering_sorting | 8 / 10 | `answer_content=2` | 8 / 10 | 9 / 10 |
| required_forbidden_tokens | 7 / 10 | `answer_content=3` | 4 / 10 | 9 / 10 |
| rewrite_normalized | 9 / 10 | `answer_content=1` | 7 / 10 | 8 / 10 |
| synthetic_extraction | 7 / 10 | `answer_content=3` | 9 / 10 | 9 / 10 |

This is a report-surface improvement only. It does not claim broad BitNet
quality, does not add matching eval history by itself, and does not enable or
broaden chat, serve, Metal, QK256, Neural Engine, MPSGraph, MacBook, speedup,
or broad Apple Silicon performance claims. `M4-BITNET-EX-009` remains the
matching-history eval refresh and larger-corpus decision point.

`M4-BITNET-EX-009` runs the second matching BitNet deterministic eval refresh
for the same accepted Microsoft I2_S GGUF, external tokenizer, prompt template,
and `apple-m4-cpu-neon` backend:

```text
ci/hardware/apple-m4-mac-mini/2026-05-17T1417Z/bitnet-eval/answer-corpus.json
ci/hardware/apple-m4-mac-mini/2026-05-17T1417Z/bitnet-eval/task-family-pass-rates.json
ci/hardware/apple-m4-mac-mini/2026-05-17T1417Z/bitnet-eval/reference-vs-rust-comparison.json
ci/hardware/apple-m4-mac-mini/2026-05-17T1417Z/bitnet-eval/regression-vs-2026-05-15T2214Z.json
ci/hardware/apple-m4-mac-mini/2026-05-17T1417Z/bitnet-eval/larger-corpus-decision.json
```

The fresh Rust M4 run records 100 mechanically scored cases, 79 passes, zero
timeouts, valid child receipts for every prompt, and `fallback_used=false`.
The matching regression against `2026-05-15T2214Z` reports
`matched_context=true` and four advisory warnings, all isolated to
`constrained_summary` moving from 9/10 to 8/10. The derived
reference-vs-current comparison records 54/100 text matches, 83/100 mechanical
scoring matches, four reference-pass/current-fail cases, 13
current-pass/reference-fail cases, and no comparable reference generated token
IDs because the reference runner still does not expose them.

| Task family | Current Rust pass rate | Regression note |
|---|---:|---|
| arithmetic_exact | 10 / 10 | stable |
| closed_label_classification | 9 / 10 | stable |
| constrained_summary | 8 / 10 | advisory warning |
| fixed_table_qa | 6 / 10 | weak |
| format_constrained_json | 10 / 10 | improved; needs another stage before broad claims |
| numeric_tolerance | 5 / 10 | weak |
| ordering_sorting | 8 / 10 | stable |
| required_forbidden_tokens | 7 / 10 | weak |
| rewrite_normalized | 9 / 10 | stable |
| synthetic_extraction | 7 / 10 | weak |

The recorded larger-corpus decision is conservative: stage a 250-case BitNet
corpus next, focused on weak or variable task families, before any 500-case
expansion or broad BitNet quality envelope. This is matching-history and
decision evidence only. It does not enable BitNet chat or serve, does not use
dense Qwen evidence, and does not widen Metal, QK256, Neural Engine, MPSGraph,
MacBook, broad Apple Silicon performance, or speedup claims.

The tracker now makes that decision executable as three separate gates:
`M4-BITNET-EX-010` defines the deterministic 250-case corpus and scorer
contract, `M4-BITNET-EX-011` runs the accepted artifact/tokenizer/backend
identity and publishes bounded receipts, and `M4-BITNET-EX-012` decides whether
the next honest step is a 500-case BitNet expansion or corpus/scoring repair.
Trend and release gates depend on that decision instead of treating the
100-case larger-corpus recommendation as sufficient.

`M4-BITNET-EX-010` adds
`ci/quality/apple-m4-bitnet-eval-seeded-corpus-250.yaml`, preserving the
accepted Microsoft I2_S GGUF identity, external tokenizer authority,
`bitnetcpp-answer` prompt template, mechanical scorer contract, and false
runtime/quality/performance/chat/serve claim boundaries. The 250-case
distribution is weighted toward weak or variable families from the 100-case
history: arithmetic_exact=15, numeric_tolerance=35, fixed_table_qa=35,
format_constrained_json=20, closed_label_classification=20,
synthetic_extraction=25, ordering_sorting=20, rewrite_normalized=20,
constrained_summary=30, and required_forbidden_tokens=30. This is corpus
definition and dry-run validation only; runtime pass rates wait for
`M4-BITNET-EX-011`.

`M4-BITNET-EX-011` runs that 250-case BitNet corpus on the M4 Mac mini through
the accepted Microsoft I2_S GGUF, accepted external tokenizer, and
`apple-m4-cpu-neon` backend:

```text
ci/hardware/apple-m4-mac-mini/2026-05-17T1903Z/bitnet-eval-250/answer-corpus.json
ci/hardware/apple-m4-mac-mini/2026-05-17T1903Z/bitnet-eval-250/answer-corpus-runs/*.json
ci/hardware/apple-m4-mac-mini/2026-05-17T1903Z/bitnet-eval-250/summary.json
ci/hardware/apple-m4-mac-mini/2026-05-17T1903Z/bitnet-eval-250/receipts-check.json
ci/hardware/apple-m4-mac-mini/2026-05-17T1903Z/bitnet-eval-250/regression-vs-2026-05-17T1417Z.json
ci/hardware/apple-m4-mac-mini/2026-05-17T1903Z/bitnet-eval-250/larger-corpus-decision.json
```

The receipt records 250 mechanically scored cases, 196 passes, 54
quality-failed cases, zero timeouts, `fallback_used=false`, generated token IDs
for every case, and 2,086 generated tokens. The compact summary records timing
and throughput p50/p90/p99 from per-case receipts; first-token p50 is 17,087 ms,
input tok/s p50 is 1.601, output tok/s p50 is 0.231, and decode steady-state
tok/s p50 is 1.423. These numbers describe this bounded run only, not a
performance envelope.

| Task family | Current Rust pass rate | Main failure categories |
|---|---:|---|
| arithmetic_exact | 14 / 15 | `answer_content=1` |
| numeric_tolerance | 24 / 35 | `answer_content=11`, `format_only=2` |
| fixed_table_qa | 23 / 35 | `factual_table=12` |
| format_constrained_json | 20 / 20 | none |
| closed_label_classification | 18 / 20 | `answer_content=2` |
| synthetic_extraction | 19 / 25 | `extraction=6` |
| ordering_sorting | 17 / 20 | `answer_content=3` |
| rewrite_normalized | 15 / 20 | `answer_content=5` |
| constrained_summary | 24 / 30 | `answer_content=6` |
| required_forbidden_tokens | 22 / 30 | `answer_content=8` |

Strict regression against the 100-case `2026-05-17T1417Z` baseline is blocked
by design because the corpus name, case count, and selected case IDs differ.
The context-mismatch artifact records `matched_context=false` instead of
pretending the two runs are comparable. This is bounded 250-case BitNet runtime
evidence only; it does not use dense Qwen evidence, does not make a broad
BitNet quality or performance claim, and does not enable chat, serve, Metal,
QK256, Neural Engine, MPSGraph, MacBook, broad Apple Silicon performance, or
speedup claims.

`M4-BITNET-EX-012` records the larger-corpus decision from the 100-case and
250-case evidence. The decision is repair-first, not 500-case expansion:
`larger-corpus-decision.json` keeps `expand_to_500_cases_now=false` and
`repair_corpus_scorer_template_first=true`. The reason is not runtime failure;
the 250-case run completed with zero timeouts, zero `not_run` cases, 2,086
generated tokens, and `fallback_used=false`. The blocker is evidence quality:
numeric tolerance still has format-only failures, fixed-table QA records twelve
factual-table misses, rewrite-normalized drops to 15/20, and no comparable
250-case reference-vs-Rust output exists. The next BitNet eval work should
repair scorer/template/normalization issues and rerun the 250-case receipt
before approving a 500-case runtime campaign.

The tracker keeps that repair-first decision explicit: `M4-BITNET-EX-013`
repairs the 250-case scorer, template, normalization, and reference-comparison
path without a runtime claim. `M4-BITNET-EX-014` now records the repaired
250-case runtime refresh:

```text
ci/hardware/apple-m4-mac-mini/2026-05-18T1806Z/bitnet-eval-250-repaired/answer-corpus.json
ci/hardware/apple-m4-mac-mini/2026-05-18T1806Z/bitnet-eval-250-repaired/summary.json
ci/hardware/apple-m4-mac-mini/2026-05-18T1806Z/bitnet-eval-250-repaired/receipts-check.json
ci/hardware/apple-m4-mac-mini/2026-05-18T1806Z/bitnet-eval-250-repaired/regression-vs-2026-05-17T1903Z.json
```

The repaired run completed all 250 cases with `fallback_used=false`, zero
timeouts, zero `not_run` cases, 2,086 generated tokens, and receipt validation
over 251 receipts: 250 child receipts plus the aggregate. It passed 205/250
quality gates and 210/250 mechanical scoring gates. The remaining quality
failures are still bounded evidence, not a broad BitNet quality claim:

| Family | Passed / Total | Notes |
| --- | ---: | --- |
| arithmetic_exact | 14 / 15 | `answer_content=1` |
| numeric_tolerance | 24 / 35 | `answer_content=10`, `format_only=4` |
| fixed_table_qa | 30 / 35 | `factual_table=5` |
| format_constrained_json | 20 / 20 | all schema checks passed |
| closed_label_classification | 18 / 20 | `answer_content=2` |
| synthetic_extraction | 19 / 25 | `extraction=6` |
| ordering_sorting | 17 / 20 | `answer_content=3` |
| rewrite_normalized | 15 / 20 | quality gate 15/20; scoring gate 19/20 |
| constrained_summary | 26 / 30 | `answer_content=4` |
| required_forbidden_tokens | 22 / 30 | `answer_content=8` |

The strict regression command against
`2026-05-17T1903Z/bitnet-eval-250/answer-corpus.json` correctly reports a
context mismatch because corpus version `2.1.0` adds `contains_expected` and
revised normalization while the prior receipt used corpus version `2.0.0`. The
context-only operator deltas are +9 quality passes, -9 quality failures, zero
timeout delta, +7 fixed-table passes, +2 constrained-summary passes, unchanged
numeric-tolerance quality passes, and +4 rewrite scoring passes. Treat this as
the first repaired 250-case baseline until a second repaired receipt exists.
`M4-BITNET-EX-015` repeats that repaired run for matching history and records
whether the next honest step is 500-case expansion, continued repair, or
freezing the current BitNet quality envelope.

`M4-BITNET-EX-015` now adds the second repaired 250-case BitNet CPU/NEON
receipt:

```text
ci/hardware/apple-m4-mac-mini/2026-05-20T0133Z/bitnet-eval-250-repaired/answer-corpus.json
```

The run used the same accepted Microsoft I2_S GGUF SHA
`4221b252fdd5fd25e15847adfeb5ee88886506ba50b8a34548374492884c2162`,
external tokenizer SHA
`e134af98b985517b4f068e3755ae90d4e9cd2d45d328325dc503f1c6b2d06cc7`,
`bitnetcpp-answer` prompt template, strict tokenizer, greedy deterministic
generation, and `apple-m4-cpu-neon` backend with `fallback_used=false`.
`mac receipts-check` passed for both the aggregate receipt and receipt
directory. The aggregate records 250/250 child receipts, 2,043 generated
tokens, and zero timeouts.

Matched-context regression against
`2026-05-18T1806Z/bitnet-eval-250-repaired/answer-corpus.json` is advisory and
`matched_context=true`, but it reports quality regressions: quality summary
199/250 versus 205/250 baseline, scoring summary 202/250 versus 210/250
baseline, and 26 quality/task-family warnings. The next decision is therefore
**keep repairing**, not 500-case expansion and not freezing a stronger BitNet
quality envelope.

| Family | Current Passed / Total | Baseline Passed / Total | Delta |
| --- | ---: | ---: | ---: |
| arithmetic_exact | 14 / 15 | 14 / 15 | 0 |
| numeric_tolerance | 21 / 35 | 24 / 35 | -3 |
| fixed_table_qa | 30 / 35 | 30 / 35 | 0 |
| format_constrained_json | 19 / 20 | 20 / 20 | -1 |
| closed_label_classification | 17 / 20 | 18 / 20 | -1 |
| synthetic_extraction | 19 / 25 | 19 / 25 | 0 |
| ordering_sorting | 16 / 20 | 17 / 20 | -1 |
| rewrite_normalized | 15 / 20 | 15 / 20 | 0 |
| constrained_summary | 27 / 30 | 26 / 30 | +1 |
| required_forbidden_tokens | 21 / 30 | 22 / 30 | -1 |

The observed wall run lasted about 5h08m from harness start to aggregate
timestamp. Case latency distribution was p50 20.2s, p90 58.2s, p99 105.2s, and
max 117.7s; the slowest cases were constrained-summary and
required/forbidden-token prompts. This is timing evidence for this repaired
BitNet eval run only, not a broad BitNet performance claim.

`M4-BITNET-EX-013` now stages the repair as a dry-run-only corpus/scorer
contract update. The repaired 250-case corpus is version `2.1.0` and keeps the
same accepted Microsoft I2_S GGUF identity, external tokenizer authority, and
`bitnetcpp-answer` prompt template. It records closed-form YAML expected
answers as authority, uses `contains_expected` for fixed-table prose answers,
uses final-answer numeric extraction for numeric tolerance, keeps rewrite
normalization limited to casing, punctuation, and whitespace, and records the
reference-vs-Rust 250-case sidecar as not yet supplied. This does not refresh
approve the 500-case expansion.

`M4-BITNET-EX-003` publishes the first BitNet one-shot benchmark envelope for
the accepted artifact/tokenizer identity through the `mac bitnet-benchmark`
route:

```text
ci/hardware/apple-m4-mac-mini/2026-05-17T0825Z/bitnet-benchmark/summary.json
ci/hardware/apple-m4-mac-mini/2026-05-17T0825Z/bitnet-benchmark/receipts/bitnet-mac-ask-benchmark.json
ci/hardware/apple-m4-mac-mini/2026-05-17T0825Z/bitnet-benchmark/receipts/bitnet-mac-bitnet-warm-benchmark.json
```

The successful run uses explicit authority for both required BitNet artifacts:

```text
model path: models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf
model sha256: 4221b252fdd5fd25e15847adfeb5ee88886506ba50b8a34548374492884c2162
tokenizer path: models/microsoft-bitnet-b1.58-2B-4T/tokenizer.json
tokenizer sha256: e134af98b985517b4f068e3755ae90d4e9cd2d45d328325dc503f1c6b2d06cc7
backend: apple-m4-cpu-neon
fallback_used: false
chat_enabled: false
serve_enabled: false
```

The benchmark summary records one `mac ask` prompt and a three-prompt fixed
warm session, all under the same accepted artifact/tokenizer identity. The
one-shot answer prompt produced text `4` with generated token IDs `[19,
128009]`. Aggregate summary metrics are:

| Metric | p50 | p90 | p99 |
|---|---:|---:|---:|
| Model load | 4133.650 ms | 4154.861 ms | 4154.861 ms |
| Tokenizer load | 158.859 ms | 174.108 ms | 174.108 ms |
| Prompt tokenize | 0.056 ms | 0.426 ms | 0.426 ms |
| Prefill | 7292.154 ms | 8044.345 ms | 8044.345 ms |
| TTFT | 7794.000 ms | 8531.000 ms | 8531.000 ms |
| Decode total | 943.894 ms | 949.428 ms | 949.428 ms |
| Input throughput | 2.468 tok/s | 2.486 tok/s | 2.486 tok/s |
| Output throughput | 0.242 tok/s | 0.243 tok/s | 0.243 tok/s |
| Decode throughput | 2.107 tok/s | 2.128 tok/s | 2.128 tok/s |
| Peak memory | 4245.688 MB | 4320.953 MB | 4320.953 MB |

This is a bounded BitNet benchmark envelope only. It does not claim BitNet
quality, enable BitNet chat or serve, prove broad Apple Silicon performance,
claim a speedup, or widen Metal, QK256, Neural Engine, MPSGraph, or MacBook
support.

`M4-BITNET-EX-004` adds named warm-session profiles to
`bitnet mac bitnet-warm` and records bounded 25/50/100 prompt evidence for the
accepted BitNet artifact/tokenizer identity. The profile run executes the
largest requested profile once as a single resident 100-prompt session, then
records `resident_25`, `resident_50`, and `resident_100` as prefix
checkpoints. Named profiles are mutually exclusive with explicit `--prompt`
values so operator-supplied prompts and campaign profiles stay separate.

Recorded warm-profile artifacts:

```text
ci/hardware/apple-m4-mac-mini/2026-05-17T0847Z/bitnet-warm/variable-warm-session.json
ci/hardware/apple-m4-mac-mini/2026-05-17T0847Z/bitnet-warm/variable-warm-session-prompts/
```

The run uses explicit authority for both required BitNet artifacts:

```text
model path: models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf
model sha256: 4221b252fdd5fd25e15847adfeb5ee88886506ba50b8a34548374492884c2162
tokenizer path: models/microsoft-bitnet-b1.58-2B-4T/tokenizer.json
tokenizer sha256: e134af98b985517b4f068e3755ae90d4e9cd2d45d328325dc503f1c6b2d06cc7
backend: apple-m4-cpu-neon
fallback_used: false
chat_enabled: false
serve_enabled: false
```

The aggregate receipt validates as `bitnet_apple_m4_warm_session`, records 100
per-prompt receipts plus the aggregate receipt, loads the model and tokenizer
once, enforces a 1200 second timeout without reaching it, and records 406
generated tokens. Determinism is checked across 12 repeated-prompt groups and
passes with stable generated token IDs and decoded text for each repeated
prompt. The quality gate here is the warm-session gate: non-empty valid output
with generated token IDs and no failed prompt indices. It is not a BitNet
accuracy or broad quality claim.

Profile checkpoint metrics:

| Profile | Prompts | Generated tokens | Quality gate | Determinism | TTFT p50 | Total wall p50 | Decode total p50 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `resident_25` | 25 | 100 | pass | pass | 7892.0 ms | 8375.524 ms | 957.319 ms |
| `resident_50` | 50 | 200 | pass | pass | 7892.0 ms | 8373.339 ms | 957.319 ms |
| `resident_100` | 100 | 406 | pass | pass | 7892.0 ms | 8375.301 ms | 957.472 ms |

Aggregate timing and memory:

```text
model_load_ms: 4336.870
tokenizer_load_ms: 166.163
prefill_ms: 647474.830
warm_prompt_wall_ms: 842793.459
total_session_ms: 847724.769
decode_generated_tok_s: 2.091
warm_prompt_generated_tok_s: 0.482
resident_memory_bytes: 2682978304
```

This is bounded BitNet warm-session evidence for the accepted artifact,
tokenizer, backend, and machine context. It does not claim BitNet chat, BitNet
serve, BitNet broad quality, full Metal inference, QK256, Neural Engine,
MPSGraph, MacBook evidence, speedup, or broad Apple Silicon performance.

`M4-BITNET-EX-005` hardens BitNet operator failure UX before any chat or serve
gate can move. The one-shot `bitnet mac ask` BitNet route now accepts
`--timeout-seconds` and records timeout state in failure receipts. Dense SLM
`mac ask` rejects that timeout flag for now so the new behavior stays scoped to
the explicit BitNet one-shot route.

The BitNet one-shot failure receipt now records the same operator diagnostics
shape expected from warm-session failure receipts:

```text
progress.enabled
progress.status_stream
progress.last_stage
progress.stage_taxonomy
timeout_boundary.configured_seconds
timeout_boundary.enforced
timeout_boundary.reached
timeout_boundary.stage
generation.partial_text
generation.partial_token_ids
generation.partial_generation_available
repair_guidance
```

The receipt validator rejects BitNet one-shot failure receipts that omit the
progress taxonomy, timeout boundary, repair guidance, or explicit partial
generation fields. The hardening remains conservative: setup, verification,
generation, and timeout failures preserve backend, fallback, model, tokenizer,
prompt, timeout, and claim-boundary fields, but decode-time partial text is
still recorded as unavailable unless the generation path can safely provide it.
This is operator UX and receipt-contract hardening only; it does not claim
BitNet quality and does not enable BitNet chat or serve.

`M4-BITNET-EX-006` adds the explicit BitNet chat route behind the chat gate.
`bitnet mac chat --model-family bitnet` still refuses before prompt collection
unless the operator passes `--bitnet-chat-gate-receipt <gate.json>` and that
receipt validates as `status=ready_to_enable` with all warm-session,
timeout/failure, streaming-semantics, backend, fallback, and claim-boundary
requirements passed.

When the gate is ready, the route remains narrow:

```bash
bitnet mac chat \
  --model-family bitnet \
  --model-id microsoft-bitnet-b1.58-2B-4T-i2s \
  --tokenizer models/microsoft-bitnet-b1.58-2B-4T/tokenizer.json \
  --bitnet-chat-gate-receipt <bitnet_apple_m4_chat_gate.json> \
  --prompt "Answer with a single digit: 2+2=" \
  --prompt "Name the capital of France. Answer with one word."
```

The successful chat receipt kind is `bitnet_apple_m4_chat_session`. It records
the accepted model/tokenizer identity, the consumed ready gate receipt SHA,
`apple-m4-cpu-neon`, `fallback_used=false`, generated text, token IDs, per-turn
receipt state, timing, memory, and `chat_enabled=true` for this gated route.
It keeps `serve_enabled=false` and does not claim BitNet serve, broad BitNet
quality, full Metal inference, QK256, Neural Engine, MPSGraph, MacBook
evidence, speedup, or broad Apple Silicon performance.

`M4-BITNET-EX-007` adds the BitNet serve route behind a stricter service gate.
`bitnet mac serve --model-family bitnet` refuses before cache lookup or bind
unless the operator passes `--bitnet-serve-gate-receipt <gate.json>` and that
receipt validates as `status=ready_to_enable`. The gate consumes a ready
`bitnet_apple_m4_chat_session`, BitNet serve streaming-semantics evidence,
BitNet serve timeout/failure evidence, and a `mac serve-check --completion`
receipt proving `/health`, `/ready`, completion, and `/receipts/{id}` export
on the gated BitNet route.

The route remains local and gate-scoped:

```bash
bitnet mac serve \
  --model-family bitnet \
  --model-id microsoft-bitnet-b1.58-2B-4T-i2s \
  --model-path models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf \
  --tokenizer models/microsoft-bitnet-b1.58-2B-4T/tokenizer.json \
  --bitnet-serve-gate-receipt <bitnet_apple_m4_serve_gate.json> \
  --host 127.0.0.1 \
  --port 8080
```

Successful BitNet server completions write
`bitnet_apple_m4_serve_completion` receipts with accepted model/tokenizer
identity, `apple-m4-cpu-neon`, `fallback_used=false`, generated text, token
IDs, timing, resident-server reuse, and the consumed serve-gate SHA. The route
does not claim production hosting, broad OpenAI compatibility, broad BitNet
quality, full Metal inference, QK256, Neural Engine, MPSGraph, MacBook
evidence, speedup, or broad Apple Silicon performance.

## Stability And Service

The appliance should prove that it stays useful after the first successful
command:

```text
mixed dense-model switching
cache reuse and unload/reload behavior
memory drift
cache repair and low-disk guidance
interrupted generation
client cancellation
interrupted receipt write
process restart
long-context guardrails
scheduled trend retention
stale-identity aging
```

`M4-STABILITY-001` records the first live mixed dense-model switch soak:

```text
ci/hardware/apple-m4-mac-mini/2026-05-20T1210Z/slm-soak/mixed-model-switch.json
```

The release-mode run exercises the three supported dense Qwen M4 identities in
sequence with `resident_25` child summaries per model:

| Model | Prompts | Generated tokens | Peak memory |
|---|---:|---:|---:|
| `qwen2.5-0.5b-instruct-q8_0` | 25 | 195 | 3756.375 MB |
| `qwen2.5-0.5b-instruct-q4_k_m` | 25 | 217 | 3615.313 MB |
| `qwen2.5-1.5b-instruct-q4_k_m` | 25 | 195 | 6416.563 MB |

The aggregate receipt validates with `bitnet mac receipts-check`, records
`prompt_count=75`, `generated_tokens=607`,
`requested_backend=selected_backend=apple-m4-cpu-neon`, `runtime_api=cpu`,
`fallback_used=false`, child receipt separation for each model identity, and
parent process peak drift of `1.922 MB`. This is bounded dense SLM stability
evidence for the recorded identities only. It is not BitNet evidence, not a
broad benchmark, and not a Metal, QK256, Neural Engine, MPSGraph, MacBook,
speedup, broad quality, or broad performance claim.

`M4-STABILITY-003` defines the scheduled trend-retention policy for committed
M4 dense SLM and BitNet evidence. The policy keeps trend claims tied to
receipts that can still be compared by identity, while letting dashboard and
operator summaries be regenerated from those retained receipts.

`M4-RELIABILITY-001` adds a bounded model-free recovery-drill receipt:

```text
ci/hardware/apple-m4-mac-mini/2026-05-20T194116Z/reliability-drills/summary.json
```

The receipt records eight required drill classes across dense SLM, BitNet
one-shot ask, and BitNet warm routes:

```text
interrupted_generation
client_cancellation
timeout
interrupted_receipt_write
missing_cache
corrupt_cache
low_disk
process_restart
```

Each drill records the expected failure receipt obligations, `fallback_used=false`,
operator-visible stage and elapsed-time requirements, and retry guidance. This
is recovery-diagnostics and receipt-contract evidence only. It does not execute
fresh live model interruption, enable BitNet chat or serve, prove service
production readiness, claim full Metal, QK256, Neural Engine, MPSGraph,
MacBook evidence, speedup, broad quality, or broad performance.

Retain these committed receipt families for each current supported M4 identity:

```text
dense SLM eval-v2 aggregate reports and per-model summaries
dense SLM benchmark-v2 summaries and profile timeout receipts
dense SLM chat, smoke, context, and stability receipts used by route classes
BitNet eval aggregate reports, including repaired-corpus summaries
BitNet benchmark, variance, one-shot, and warm-session receipts
BitNet gate receipts for any enabled chat or serve route
setup, doctor, status, prune dry-run, and cache-repair receipts used by the envelope
```

For trend comparison, at least the current report and the previous matching
baseline must remain committed for each dashboard group. Large child receipts
may be retained only when the accepted evidence bundle needs them for receipt
validation, generated text/token-ID audit, or failure taxonomy; otherwise the
committed aggregate must preserve the child count, identity, scoring totals,
failure categories, timing, memory, and receipt-validation status needed to
reproduce the dashboard decision. Model binaries, local cache copies, and
intermediate `target/` artifacts are not retention targets.

Regenerate these summaries from retained receipts instead of treating them as
source evidence:

```text
report-refresh manifest
regression-dashboard JSON and Markdown
operator evidence summary
status tables and open-target explanations
generated campaign status
operator envelope class tables
```

Generated summaries should land in `target/apple-m4-inference-excellence/`
during local refreshes unless the work item explicitly requires committing a
summary or generated tracking file. A regenerated summary can change operator
classification only when the retained source receipts support the same claim
boundary.

`M4-TREND-001` is the first committed rolling trend summary. It adds
`--since 7d` to `bitnet mac report-refresh` and
`bitnet mac regression-dashboard`, then records the model-free outputs under:

```text
ci/hardware/apple-m4-mac-mini/2026-05-21T1805Z/trend/report-refresh.json
ci/hardware/apple-m4-mac-mini/2026-05-21T1805Z/trend/regression-dashboard.json
ci/hardware/apple-m4-mac-mini/2026-05-21T1805Z/trend/regression-dashboard.md
```

The window covers `2026-05-15` through `2026-05-21` from committed receipts
only. The dashboard records 29 reports across 7 families, 13
matching-identity groups, and 10 comparable groups. Each family or group carries
skipped-day reasons, a threshold outcome, and operator-envelope impact text so a
maintainer can see whether a current route has matching history, needs another
matching report, or should block envelope updates until receipt issues are
repaired. These trend summaries have `prompt_count=0` and
`generated_tokens=0`; they do not run live inference, download models, replace
source receipts, enable BitNet chat or serve, or claim broad quality,
performance, speedup, full Metal, QK256, Neural Engine, MPSGraph, MacBook, or
broad Apple Silicon behavior.

An identity becomes stale for current operator claims when any comparison
identity field changes or is missing:

```text
model ID or model SHA256
tokenizer authority or tokenizer SHA256
prompt template, stop criteria, or generation identity
corpus or benchmark profile version
selected backend, runtime API, fallback state, or machine ID
route gate receipt consumed by chat or serve
receipt schema required for the dashboard group
```

Stale identities are not deleted, but they become historical evidence. They
must not be described as current trends after either a newer accepted identity
has two matching refreshes, the supported-model matrix removes or deprecates the
identity, or the identity misses two scheduled M4 refresh cycles while the
operator envelope still depends on it. A stale identity can return to current
status only through a fresh matching-history pair and a dashboard status that is
ready or explicitly accepted with documented warnings.

Refresh the operator envelope when any of these changes:

```text
default dense model or supported-model state
accepted BitNet artifact, tokenizer, prompt, stop, or generation identity
route state for ask, chat, warm session, serve, or streaming
quality, timeout, timing, or memory threshold result
context guardrail, long-context profile, or benchmark timeout boundary
dashboard status for a current evidence group
disk/cache readiness floor or repair guidance
stale identity aging that changes the current operator class
claim-boundary wording or newly supported backend surface
```

This policy is evidence-retention and operator-refresh guidance only. It does
not run a model, prove new dense SLM or BitNet quality, enable BitNet chat or
serve, or widen Metal, QK256, Neural Engine, MPSGraph, MacBook, speedup, broad
quality, or broad performance claims.

`M4-TREND-001` publishes the first seven-day matching-history summary derived
from retained receipts:

```text
ci/hardware/apple-m4-mac-mini/2026-05-22T0530Z/trend/seven-day-history.json
ci/hardware/apple-m4-mac-mini/2026-05-22T0530Z/trend/seven-day-history.md
```

The summary covers the five current dashboard families from
`2026-05-15T00:00:00Z` through `2026-05-22T00:00:00Z`: dense SLM eval-v2,
dense SLM benchmark-v2, BitNet eval, BitNet benchmark, and BitNet variable warm.
It records nine matching dashboard groups, all with `ready` history. Dense SLM
eval and benchmark groups remain within the published threshold envelope for
their retained latest-vs-baseline pairs. BitNet eval and benchmark also remain
within threshold for their retained pairs.

The only advisory trend issue is BitNet variable-warm resident memory: the
retained latest run increased resident memory from `2140225536` bytes to
`2688778240` bytes, a `25.63%` increase against the existing `10%`
higher-is-worse advisory threshold. Timing improved and quality remained
passing, so this is an operator follow-up, not a BitNet chat or serve enablement
signal.

Skipped-day reasons are explicit in the trend artifact. Later receipts in the
window, including BitNet repaired 250-case eval, setup, variance, context,
reliability, and serve failure-semantics evidence, remain separate route or
context baselines unless a second matching receipt makes them trend-comparable.
The trend summary does not replace per-run receipts, run live inference, prove
future performance, or widen Metal, QK256, Neural Engine, MPSGraph, MacBook,
speedup, broad quality, or broad performance claims.

`M4-CONTEXT-001` implements long-context guardrails for the M4 operator routes.
Dense SLM `mac ask`, dense `mac chat`, dense `mac chat-smoke`, and dense
`mac serve` classify requests against the recorded short, `context_1k`, and
`context_4k` evidence envelopes. Requests beyond the recorded dense
`context_4k` envelope fail closed with an `apple_m4_context_guardrail` receipt
instead of falling through to cache lookup or generation. BitNet `mac ask`,
`mac bitnet-warm`, and gated BitNet chat/serve routes classify requests against
the bounded accepted-artifact ask/warm prompt evidence and fail closed beyond
that boundary.

The guardrail receipt records `context_envelope` fields for route, model family,
model id, operator class, status, prompt-token count, exact-vs-estimated token
authority, max-new-token budget, recorded evidence profile, and claim boundary.
This is a routing and overclaim-prevention contract only; it does not prove new
long-context quality, enable unsupported contexts, enable BitNet chat or serve,
or widen Metal, QK256, Neural Engine, MPSGraph, MacBook, speedup, broad quality,
or broad performance claims.

`M4-CONTEXT-002` publishes the first live long-context proof receipt through the
release `bitnet mac eval --suite m4-long-context` route:

```text
ci/hardware/apple-m4-mac-mini/2026-05-20T1611Z/context/answer-corpus.json
```

The receipt validates as `apple_m4_long_context_answer_corpus` and covers the
default dense SLM identity `qwen2.5-0.5b-instruct-q8_0` on
`apple-m4-cpu-neon` with `fallback_used=false`. The four mechanical cases pass:
retrieval/copy, table extraction, late-context instruction following, and the
explicit unsupported-context boundary. The same evidence records Qwen prompt
template identity, stop/generation identity, generated token counts, model SHA,
and the claim boundary that dense SLM long-context evidence does not prove
BitNet long-context behavior.

The matching release `bitnet mac benchmark --profile context` receipt is:

```text
ci/hardware/apple-m4-mac-mini/2026-05-20T1611Z/context/benchmark.json
```

It validates as `apple_m4_slm_benchmark_v2` with `fallback_used=false`.
`context_1k` completes as a warm-session profile with 3 prompts and 48
generated tokens. `context_4k` reaches the calibrated 720 second timeout and is
recorded as `apple_m4_slm_benchmark_profile_timeout` with
`status=invalid_for_comparison`, so the aggregate is timing evidence with an
explicit timeout boundary, not a comparable performance envelope. This proof is
for the tested dense identity only. It does not claim non-default dense SLM
long-context quality, BitNet long-context behavior, BitNet chat or serve, full
Metal inference, QK256, Neural Engine, MPSGraph, MacBook behavior, speedup,
broad quality, or broad performance.

CLI proof is route-specific. Dense SLM ask/chat conformance needs bounded
multi-turn history, timeout/cancel behavior, per-turn receipts, generated text,
token IDs, backend, fallback state, and model/tokenizer identity before the CLI
surface is treated as excellent.

`M4-DENSE-CHAT-001` adds `bitnet mac chat-smoke` as a dense-only conformance
route for that proof. The command runs a bounded two-prompt resident session
through the same dense chat path, writes per-turn receipts plus an aggregate
`apple_m4_slm_chat_smoke` receipt, and validates prompt-template identity, stop
behavior, timeout/cancel metadata, generated text, token IDs, backend,
fallback state, model SHA, and tokenizer authority.

Current M4 evidence is recorded under:

```text
ci/hardware/apple-m4-mac-mini/2026-05-18T1238Z/slm-chat/
```

It covers the supported dense identities:

```text
qwen2.5-0.5b-instruct-q8_0
qwen2.5-0.5b-instruct-q4_k_m
qwen2.5-1.5b-instruct-q4_k_m
```

This is bounded dense SLM chat-route conformance. It is not BitNet chat
evidence, broad model-quality evidence, a server claim, a full Metal inference
claim, a QK256 claim, Neural Engine or MPSGraph execution, MacBook evidence, a
speedup claim, or a broad Apple Silicon performance envelope.

Service proof is separate from CLI proof. Dense SLM serve and later BitNet
serve need receipts for:

```text
health and ready
one-shot request
streaming completion
client cancellation
timeout stage
invalid request
missing cache
per-request receipt export
local-only safety defaults
queue limits and backpressure
resident model reuse
```

Local service claims stay bounded: local appliance operation, not production
hosting and not broad OpenAI compatibility.

## Operator UX

The M4 should explain itself without requiring a user to read the whole
receipt tree. Operator-facing commands should surface:

```text
default model
supported models
cache state
disk pressure
last successful dense report
last successful BitNet report
current regressions
unsupported claims
recommended next command
route envelope class
route-state matrix
```

`bitnet mac evidence` is the operator-facing summary for that view. It reads
the model catalog, disk/cache state, committed report inventory, and regression
dashboard groups, then writes an `apple_m4_operator_evidence_summary` receipt
without running live inference or downloading models. `bitnet mac status`,
`doctor`, `report-refresh`, and `regression-dashboard` remain model-free by
default. Live model runs belong in local, advisory, scheduled, or release lanes.
`bitnet mac report-refresh --explain --open-targets` and
`bitnet mac regression-dashboard --explain --open-targets` expose the same
model-free operator contract in a more navigable form: status meanings,
per-family or per-group reasons, receipt/Markdown/report targets, and the
command to run next. `comparable` means matching-history comparison can proceed;
`warning` requires operator review; `failed` blocks claims until the receipt is
repaired; `insufficient_history` means a second matching report is needed.
`bitnet mac status` and `bitnet mac doctor` also keep dense SLM and BitNet
readiness separate: dense routes report default-model cache repair, BitNet
routes report one-shot/warm readiness, BitNet chat and serve remain disabled
until their gates pass, and both surfaces point at the latest matching receipt
families used for operator context.

Envelope classes should translate evidence into local user expectations:

```text
interactive
advisory
batch
disabled
unsupported
```

`M4-OPS-SLO-001` publishes those classes in
`docs/slm/apple-m4-operator-envelope-v3.md`. The class map ties each dense Qwen
ask/chat route, dense local-server route, BitNet one-shot/warm route, gated
BitNet chat/serve route, and unsupported Apple backend route to the exact
committed evidence identity, max context or profile guidance, timing
expectation, and memory/disk posture. The map is an operator expectation layer
only; it does not enable a disabled route or turn bounded receipts into broad
quality or performance claims.

`M4-ROUTE-MATRIX-001` adds the model-free route-state matrix to `bitnet mac
status --json` and `bitnet mac evidence --json`. The matrix separates dense SLM
ask, chat, warm-session, serve, and streaming states from BitNet ask,
warm-session, chat, serve, and streaming states. Enabled or batch-only rows name
the exact evidence item and receipt family required for that state; disabled
BitNet rows name the required chat or serve gate; unsupported backend rows stay
unsupported until a separate full-route receipt exists.

`M4-EVIDENCE-REPLAY-001` adds replayable evidence bundle manifests for auditing
committed dense SLM and BitNet refreshes. The first bundle is
`ci/hardware/apple-m4-mac-mini/2026-05-21T145609Z/evidence-replay/manifest.json`.
It records exact model-free replay/audit commands, git and binary identity,
dense SLM and BitNet model/tokenizer identity, receipt inputs, dashboard
outputs, expected advisory regression result, and claim boundaries. Operators
audit it with `bitnet mac evidence replay --bundle <manifest.json> --dry-run
--json`; the dry-run checks the committed manifest and referenced receipts only.
It does not execute live inference, download models, validate uncommitted
artifacts, enable disabled BitNet routes, or create Metal, QK256, Neural Engine,
MPSGraph, MacBook, broad quality, broad performance, speedup, or broad Apple
Silicon claims.

`M4-WORKLOAD-001` adds a model-free operator workload suite manifest:
`ci/hardware/apple-m4-mac-mini/2026-05-21T171832Z/workload/summary.json`.
The `bitnet mac workload --suite m4-operator` receipt covers six operator
workflow families: `summarize`, `extract`, `classify`, `json`, `rewrite`, and
`table_qa`. For each workflow it records mechanical checks and route-plan entries
over dense SLM ask/chat/warm-session/serve surfaces plus BitNet ask/warm-session
surfaces. BitNet chat and serve remain disabled gate boundaries that name the
missing gate families instead of enabling those routes.

The committed workload receipt validates with `bitnet mac receipts-check` and
records `prompt_count=0`, `generated_tokens=0`, `no_live_model_run=true`, and
`workload_manifest_only=true`. It supplies exact follow-up commands for later
live route receipts, but it does not run those commands itself. This is operator
coverage and receipt-contract evidence only: it does not prove broad assistant
quality, enable BitNet chat or serve, claim production server readiness, claim
full Metal inference, QK256, Neural Engine, MPSGraph, MacBook behavior, speedup,
broad performance, or broad Apple Silicon support.

`M4-WORKLOAD-001` begins from that matrix with a model-free operator workload
suite contract:

```bash
bitnet mac workload --suite m4-operator --json-out target/apple-m4-inference-excellence/workload/summary.json
```

The receipt is `apple_m4_operator_workload_suite`; it enumerates summarize,
extract, classify, JSON/schema, rewrite, and table-QA cases across enabled dense
SLM and BitNet routes. It is a generic-PR-safe contract only, not live workload
proof. See `docs/slm/apple-m4-workload-suite.md`.

`M4-EVIDENCE-REPLAY-001` adds dry-run replay bundles for committed evidence
refreshes:

```bash
bitnet mac evidence replay --bundle ci/hardware/apple-m4-mac-mini/2026-05-22T0400Z/evidence-replay/dense-slm-q8-eval/manifest.json --dry-run --json
```

The replay receipt is `apple_m4_evidence_replay_dry_run`; it validates the
bundle manifest, SHA256-pinned receipt inputs, dashboard outputs, exact command
list, expected regression metadata, and claim boundary. It does not run a model,
download artifacts, execute the regression command, or validate uncommitted
local artifacts. See `docs/slm/apple-m4-evidence-replay.md`.

## Release Gates

Before the public M4 expectation envelope changes, the go/no-go matrix in
`docs/slm/apple-m4-release-go-no-go.md` says which dense SLM, BitNet,
benchmark, stability, service, operator, and claim-boundary gates must pass. A
missing BitNet chat or serve gate remains a missing feature, not a documentation
issue.

`M4-CLAIM-LINT-001` adds static publication hygiene for M4 docs, generated
status, operator envelope text, and operator-facing command strings. It should
reject unsupported broad Apple Silicon, MacBook, full Metal, Neural Engine,
MPSGraph, QK256, dense-as-BitNet, broad quality/performance, or speedup wording
unless the wording is tied to a matching accepted receipt gate.

`M4-GATE-HYGIENE-001` keeps the code-health gate surface honest for this lane:
`check-no-panic-family` must be clean or explicitly justified before the M4 CI
and release-gate items are treated as ready. This is gate hygiene only; it does
not prove runtime quality, speed, BitNet behavior, or any Apple backend support.

`M4-CI-001` codifies the M4 evidence CI lane contract in
`docs/slm/apple-m4-evidence-ci-lanes.md`. Generic PR Tier 0 remains model-free:
parser, scorer, receipt-schema, committed-summary, self-baseline regression,
generated-dashboard, and diff-hygiene checks only. Advisory local, scheduled M4,
and release-gate lanes are the only lanes that may produce fresh live M4
evidence, and hardware-only timing jobs are non-blocking for ordinary PRs unless
a release gate explicitly opts into drift failure.

`M4-STABILITY-002` keeps operator repair flows explicit and non-destructive:
`bitnet mac doctor --json --include-bitnet` records dense SLM cache state, BitNet
ask/warm readiness, stale-symlink state, disk pressure, and repair guidance, and
`bitnet model prune --dry-run --json` records what supported model-cache entries
would be removed before any deletion is attempted. These are cache and disk
repair receipts only; they do not run live inference by default or prove model
quality, speed, BitNet chat/serve readiness, or any wider backend support.

`M4-SETUP-001` records the first-run operator path as setup evidence under
`ci/hardware/apple-m4-mac-mini/2026-05-19T040154Z/setup/`. The setup bundle
captures `bitnet mac models` before and after fetching the accepted BitNet
artifact, `bitnet model verify` JSON for the dense default and Microsoft I2_S
BitNet artifact, `bitnet mac doctor --include-bitnet` with live inference
skipped by default, and a dense `bitnet mac smoke --model-id
qwen2.5-0.5b-instruct-q8_0` aggregate plus answer receipt. Its
`apple_m4_first_run_setup_summary` receipt is a setup summary only: BitNet chat
and serve remain disabled, BitNet live smoke is not claimed by that summary, and
broad quality, performance, speedup, full Metal, QK256, Neural Engine,
MPSGraph, or MacBook claims remain out of scope.

## Metal Boundary

Metal work is phase-scoped only:

```text
one named phase
CPU reference parity
same generated token IDs/text where required
fallback_used=false
phase-local timing
explicit CPU/NEON remainder
```

No full `apple-m4-metal`, QK256, Neural Engine, MPSGraph, MacBook, broad Apple
Silicon, broad quality, or speedup claim is allowed until a separate full-route
receipt proves it.

`M4-METAL-EX-001` selects the next future dense SLM phase target after the
completed Q/K/V projection work: prefill attention-score logits with CPU
reference parity, fallback-free phase receipts, phase-local timing, and
CPU/NEON retained for the rest of the answer path. See
`docs/slm/apple-m4-metal-ex-phase-choice.md`.

`M4-METAL-EX-002` implements that one named phase as a dense SLM fixture and
records the runtime receipt at:

```text
ci/hardware/apple-m4-mac-mini/2026-05-22/slm-metal-phases/metal-dense-prefill-attention-scores.json
```

The receipt is phase-scoped: prefill attention-score logits only, CPU reference
parity, `fallback_used=false`, phase-local timing, and CPU/NEON retained for
the rest of the answer path. It is not full `apple-m4-metal` inference, not a
BitNet route, not QK256, Neural Engine, MPSGraph, MacBook evidence, speedup,
broad quality, broad performance, or broad Apple Silicon support.

## Completion Audit

The campaign closeout audit is committed at:

```text
ci/hardware/apple-m4-mac-mini/2026-05-22/m4-inference-excellence-completion-audit.json
```

It maps the active thread objective to committed artifacts and records the
local validation commands used at closeout. The decision is complete because
all 78 tracker items are merged, `campaign next` reports no next item, the
campaign and generated dashboards validate, claim lint passes for the Apple M4
scope, and the final state keeps dense SLM evidence, BitNet evidence, service
proof, operator UX, release gates, and Metal phase claims separated.
