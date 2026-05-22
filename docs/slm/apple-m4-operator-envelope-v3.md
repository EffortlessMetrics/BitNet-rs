# Apple M4 Operator Envelope V3

This envelope refreshes the M4 Mac mini operator contract after the durable
evidence campaign added matching-history reports. It extends the command map in
`docs/slm/apple-m4-operator-envelope-v2.md` with refresh cadence, regression
thresholds, trend-retention and stale-identity rules, `resident_100` status,
disk/cache guidance, and claim boundaries from the committed dense SLM and
BitNet report history.

It is a local M4 Mac mini evidence envelope. It is not a broad Apple Silicon
benchmark and it is not a broad model-quality claim.

## Evidence Inputs

The durable envelope is based on these committed evidence surfaces:

| Surface | Receipt path | Evidence | Status |
|---|---|---|---|
| Dense SLM benchmark v2 refresh | `ci/hardware/apple-m4-mac-mini/2026-05-15T1845Z/slm-benchmark-v2/<model-id>/summary.json` | Supported Qwen dense models, nine benchmark profiles, including `resident_100` | comparable matching history exists |
| BitNet eval refresh | `ci/hardware/apple-m4-mac-mini/2026-05-15T2214Z/bitnet-eval/answer-corpus.json` | Accepted Microsoft I2_S GGUF plus external tokenizer, 100 deterministic cases | comparable matching history exists |
| BitNet benchmark refresh | `ci/hardware/apple-m4-mac-mini/2026-05-15T2214Z/bitnet-benchmark/summary.json` | Accepted BitNet one-shot benchmark profile | comparable matching history exists |
| BitNet variable warm refresh | `ci/hardware/apple-m4-mac-mini/2026-05-16T0626Z/bitnet-productization/variable-warm-session.json` | Five prompt warm session with one exact repeated prompt | comparable matching history exists |
| Report dashboard | `target/apple-m4-inference-excellence/regression-dashboard.json` | Model-free grouping of committed reports by matching identity | refreshed by `M4-EXCELLENCE-003`; five families, 18 reports, and nine comparable groups |
| Evidence replay bundle | `ci/hardware/apple-m4-mac-mini/2026-05-21T145609Z/evidence-replay/manifest.json` | Exact audit commands, git/binary identity, model/tokenizer identity, receipt inputs, dashboard outputs, expected advisory regression result, and claim boundary | dry-run audit only; no live model run |
| Operator workload suite | `ci/hardware/apple-m4-mac-mini/2026-05-21T171832Z/workload/summary.json` | Six workflow families across dense SLM ask/chat/warm-session/serve and enabled BitNet ask/warm-session route states, with disabled BitNet chat/serve boundaries | model-free manifest only; no live model run |

`M4-BENCH-002` validates the dense SLM benchmark surface above with
`target/release/bitnet mac receipts-check ... --json` for all three supported
Qwen summaries. The validation covers the receipt contract, full profile set,
p50/p90/p99 and min/max timing fields, model SHA identity, backend/fallback
state, generated text/token IDs, and dense-only claim boundaries.

All durable refresh receipts used by this envelope keep the supported local M4
route bounded to:

```text
machine_id or machine.id=apple-m4-mac-mini
selected_backend=apple-m4-cpu-neon
runtime_api=cpu
fallback_used=false
```

Dense SLM evidence and BitNet evidence stay separate. A dense Qwen report does
not prove BitNet behavior, and a BitNet report does not broaden dense model
support.

## Operator Envelope Classes

Operator envelope classes translate committed receipts into local expectations
for the M4 Mac mini. They are route guidance, not broad quality or performance
claims. A class does not override runtime gates: if a command requires a ready
gate receipt, missing cache, or disabled route check, that command still stops.

| Class | Local meaning | Required operator posture |
|---|---|---|
| `interactive` | Short or resident local use is expected to feel prompt on the recorded M4 identity. | Use the supported route normally, keep receipts, and rerun `mac status` or `doctor` after cache changes. |
| `advisory` | The route is usable, but timing, route maturity, or pending conformance work means operators should inspect receipts. | Use for local workflows where receipt review is acceptable; do not promote to a release expectation without the named follow-up evidence. |
| `batch` | The route is valid but slow enough that users should expect queued or unattended operation. | Use progress output, timeouts, and receipt validation; avoid interactive UX promises. |
| `disabled` | The route is intentionally blocked until a ready gate receipt or later item enables it. | Treat failure as the correct behavior and collect the missing gate evidence. |
| `unsupported` | The route or backend has no accepted M4 evidence in this envelope. | Do not route users there and do not make a support claim. |

The current class map is:

| Family and route | Class | Max context or profile guidance | Timing expectation | Memory and disk expectation | Evidence identity |
|---|---|---|---|---|---|
| Dense Qwen 0.5B `bitnet mac ask` and dense `bitnet mac chat` using `qwen2.5-0.5b-instruct-q8_0` | `interactive` for short prompts and resident sessions; `batch` for 1k/4k context profiles | Supported benchmark profiles through `resident_100`; `context_1k` and `context_4k` are recorded but not interactive | 2026-05-17 eval TTFT p50/p90 2203/2771 ms, decode p50 15.628 tok/s; 2026-05-15 `resident_100` TTFT p50/p99 2150/2246 ms; `context_1k` TTFT p50 52798 ms and `context_4k` TTFT p50 262608 ms | Dense cache 675710816 bytes; resident peak about 4.16 GiB; keep at least 10 GiB free for ordinary operation and 20 GiB before full refreshes | `qwen2.5-0.5b-instruct-q8_0`, SHA `ca59ca7f13d0e15a8cfa77bd17e65d24f6844b554a7b6c12e07a5f89ff76844e`, GGUF tokenizer authority, `qwen2.5` template, `apple-m4-cpu-neon`, `fallback_used=false`; eval `ci/hardware/apple-m4-mac-mini/2026-05-17T0045Z/slm-eval-v2/qwen2.5-0.5b-instruct-q8_0/summary.json`; benchmark `ci/hardware/apple-m4-mac-mini/2026-05-15T1845Z/slm-benchmark-v2/qwen2.5-0.5b-instruct-q8_0/summary.json`; chat conformance `ci/hardware/apple-m4-mac-mini/2026-05-18T1238Z/slm-chat/` |
| Dense Qwen 0.5B `bitnet mac ask` and dense `bitnet mac chat` using `qwen2.5-0.5b-instruct-q4_k_m` | `interactive` for short prompts and resident sessions; `batch` for 1k/4k context profiles | Supported benchmark profiles through `resident_100`; `context_1k` and `context_4k` are recorded but not interactive | 2026-05-17 eval TTFT p50/p90 2201/2784 ms, decode p50 15.630 tok/s; 2026-05-15 `resident_100` TTFT p50/p99 2151/2246 ms; `context_1k` TTFT p50 52772 ms and `context_4k` TTFT p50 262519 ms | Dense cache 491400032 bytes; resident peak about 4.16 GiB; keep the same disk floors as the dense default | `qwen2.5-0.5b-instruct-q4_k_m`, SHA `74a4da8c9fdbcd15bd1f6d01d621410d31c6fc00986f5eb687824e7b93d7a9db`, GGUF tokenizer authority, `qwen2.5` template, `apple-m4-cpu-neon`, `fallback_used=false`; eval `ci/hardware/apple-m4-mac-mini/2026-05-17T0045Z/slm-eval-v2/qwen2.5-0.5b-instruct-q4_k_m/summary.json`; benchmark `ci/hardware/apple-m4-mac-mini/2026-05-15T1845Z/slm-benchmark-v2/qwen2.5-0.5b-instruct-q4_k_m/summary.json`; chat conformance `ci/hardware/apple-m4-mac-mini/2026-05-18T1238Z/slm-chat/` |
| Dense Qwen 1.5B `bitnet mac ask` and dense `bitnet mac chat` using `qwen2.5-1.5b-instruct-q4_k_m` | `advisory` for short prompts and resident sessions; `batch` for long/context profiles | Supported benchmark profiles through `resident_100`; long and context profiles are recorded as slow paths | 2026-05-17 eval TTFT p50/p90 8809/11336 ms, decode p50 4.949 tok/s; 2026-05-15 `resident_100` TTFT p50/p99 8078/8966 ms; `context_1k` TTFT p50 182985 ms and `context_4k` TTFT p50 822691 ms | Dense cache 1117320736 bytes; resident peak about 8.40 GiB; keep at least 10 GiB free for ordinary use and more before full refreshes | `qwen2.5-1.5b-instruct-q4_k_m`, SHA `6a1a2eb6d15622bf3c96857206351ba97e1af16c30d7a74ee38970e434e9407e`, GGUF tokenizer authority, `qwen2.5` template, `apple-m4-cpu-neon`, `fallback_used=false`; eval `ci/hardware/apple-m4-mac-mini/2026-05-17T0045Z/slm-eval-v2/qwen2.5-1.5b-instruct-q4_k_m/summary.json`; benchmark `ci/hardware/apple-m4-mac-mini/2026-05-15T1845Z/slm-benchmark-v2/qwen2.5-1.5b-instruct-q4_k_m/summary.json`; chat conformance `ci/hardware/apple-m4-mac-mini/2026-05-18T1238Z/slm-chat/` |
| Dense Qwen `bitnet mac serve` | `advisory` | Use the dense local-server contract only for local loopback operation; do not treat it as production hosting or broad OpenAI compatibility | Route timing should be read from per-request receipts and the dense serve smoke/failure semantics receipts; use the dense ask/chat classes above as model-level expectations | Same model cache and resident memory floors as the selected dense model; server refreshes should use the 20 GiB full-refresh floor | Current implementation contract is `docs/slm/apple-m4-local-server-command-config.md`; dense serve conformance is recorded by `M4-SERVE-EX-001` and failure/streaming semantics by `M4-SERVE-EX-002` |
| BitNet `bitnet mac ask` and fixed `bitnet mac bitnet-warm` using the accepted Microsoft I2_S artifact | `batch` | One-shot and fixed or variable warm receipts are accepted; 250-case repaired eval is valid and comparable, but the latest matching run regressed and keeps repair first | 2026-05-19 BitNet variance TTFT p50/p90/p99 7491/8486/8486 ms, output p50 0.251 tok/s, decode p50 2.082 tok/s; 2026-05-17 warm `resident_100` TTFT p50 7892 ms and total session 847724.769 ms; 2026-05-20 250-case eval wall latency p50/p90/p99 about 20.2/58.2/105.2 s | Accepted GGUF is about 1.19 GB plus external tokenizer; benchmark peak memory p50/p90 about 4322.875/4327.438 MB; warm resident memory about 2682978304 bytes; keep 20 GiB free before full BitNet refreshes | Microsoft BitNet I2_S GGUF SHA `4221b252fdd5fd25e15847adfeb5ee88886506ba50b8a34548374492884c2162`; external tokenizer SHA `e134af98b985517b4f068e3755ae90d4e9cd2d45d328325dc503f1c6b2d06cc7`; `bitnetcpp-answer` identity SHA `8ccb3ad8cf3e3af19b5da2cf69b50a21b78cf01cd3b287de76a1497d2fbfeb3c`; `apple-m4-cpu-neon`, `fallback_used=false`; variance `ci/hardware/apple-m4-mac-mini/2026-05-19T2245Z/bitnet-benchmark-variance/summary.json`; warm `ci/hardware/apple-m4-mac-mini/2026-05-17T0847Z/bitnet-warm/variable-warm-session.json`; repaired eval `ci/hardware/apple-m4-mac-mini/2026-05-20T0133Z/bitnet-eval-250-repaired/answer-corpus.json` with quality 199/250 and scoring 202/250 versus 205/250 and 210/250 baseline |
| BitNet `bitnet mac chat --model-family bitnet` | `disabled` unless a ready `bitnet_apple_m4_chat_gate` receipt is supplied | The route is gate-required; one-shot, benchmark, or warm receipts alone are not chat enablement | No chat timing expectation is published by this envelope | Uses the same accepted BitNet artifact/tokenizer only after the gate passes | `M4-BITNET-EX-006` defines the gate; missing or blocked gate receipts must keep the route disabled |
| BitNet `bitnet mac serve --model-family bitnet` | `disabled` unless a ready `bitnet_apple_m4_serve_gate` receipt is supplied | The route is gate-required after chat and service evidence; dense serve receipts do not prove BitNet serve | No BitNet serve timing expectation is published by this envelope | Uses the same accepted BitNet artifact/tokenizer only after the gate passes | `M4-BITNET-EX-007` defines the gate; missing or blocked gate receipts must keep the route disabled |
| Full `apple-m4-metal`, QK256-on-Apple, Neural Engine, MPSGraph, MacBook, and broad Apple Silicon routes | `unsupported` | No supported inference profile in this envelope | No timing expectation | No memory or disk expectation | No accepted full-route receipt in this envelope |

`M4-WORKLOAD-001` translates this route-state map into an operator workload
manifest. `bitnet mac workload --suite m4-operator` records summarize, extract,
classify, JSON, rewrite, and table-QA workflow cases, mechanical checks, route
evidence, and exact follow-up live commands. The receipt is validated by
`bitnet mac receipts-check` and records zero prompts and zero generated tokens
because it is a manifest, not a live inference run.

## Route-State Matrix

`M4-ROUTE-MATRIX-001` makes the route state explicit in both operator docs and
the model-free `bitnet mac status --json` / `bitnet mac evidence --json`
receipts. The matrix is descriptive: it does not enable a route without the
runtime receipts and gates named below.

| Family | Surface | State | Command surface | Required evidence item and receipt family | Boundary |
|---|---|---|---|---|---|
| Dense SLM | ask | `enabled` | `bitnet mac ask` | `M4-ACCURACY-007` / `dense_slm_eval_v2` (`apple_m4_slm_eval_summary`); `M4-BENCH-002` / `dense_slm_benchmark_v2` (`apple_m4_slm_benchmark_v2`) | Supported dense Qwen identities only; not BitNet or broad quality proof. |
| Dense SLM | chat | `enabled` | `bitnet mac chat`; `bitnet mac chat-smoke` | `M4-DENSE-CHAT-001` / `dense_slm_chat_smoke` (`apple_m4_slm_chat_smoke`) | Dense chat conformance only; not BitNet chat or server proof. |
| Dense SLM | warm session | `enabled` | resident multi-prompt `bitnet mac chat` | `M4-DENSE-CHAT-001` / `dense_slm_chat_smoke`; `M4-BENCH-002` / `dense_slm_benchmark_v2` resident profiles | Resident dense session reuse only; not a long soak or broad performance claim. |
| Dense SLM | long/context profile | `batch_only` | ask/chat/serve inside recorded `context_1k` or `context_4k` envelopes | `M4-CONTEXT-002` / `dense_slm_context_eval` (`apple_m4_long_context_eval_summary`) and `dense_slm_context_benchmark` (`apple_m4_slm_benchmark_v2`) | Requests beyond the recorded envelope are `unsupported` guardrail blocks. |
| Dense SLM | serve | `enabled` | loopback `bitnet mac serve` | `M4-SERVE-EX-001` / `dense_slm_serve_smoke` (`bitnet_apple_m4_dense_local_server_smoke`); `M4-SERVE-EX-002` / `serve_failure_semantics` (`bitnet_apple_m4_serve_failure_semantics`) | Local appliance route only; not production hosting or broad OpenAI compatibility. |
| Dense SLM | streaming | `enabled` | dense local-server streaming completion | `M4-SERVE-EX-001` / `dense_slm_serve_smoke`; `M4-SERVE-EX-002` / `serve_failure_semantics` | Dense local-server streaming only; not BitNet serve evidence. |
| BitNet | ask | `batch_only` | explicit `bitnet mac ask --model-id microsoft-bitnet-b1.58-2B-4T-i2s ...` | `M4-BITNET-EX-003` / `bitnet_benchmark` (`bitnet_apple_m4_benchmark_v1`) | Accepted artifact/tokenizer one-shot only; not chat, serve, broad quality, or speed proof. |
| BitNet | warm session | `batch_only` | `bitnet mac bitnet-warm` | `M4-BITNET-EX-004` / `bitnet_variable_warm` (`bitnet_apple_m4_warm_session`); `M4-BITNET-REG-001` / warm-session regression | Accepted artifact/tokenizer warm evidence only; chat and serve remain separate gated surfaces. |
| BitNet | chat | `disabled` | `bitnet mac chat --model-family bitnet --bitnet-chat-gate-receipt <gate.json>` | Gate-required by `M4-BITNET-EX-006` / `bitnet_chat_gate` (`bitnet_apple_m4_chat_gate`) | The route must refuse without a ready chat gate; one-shot, warm, or dense receipts do not enable it. |
| BitNet | serve | `disabled` | `bitnet mac serve --model-family bitnet --bitnet-serve-gate-receipt <gate.json>` | Gate-required by `M4-BITNET-EX-007` / `bitnet_serve_gate` (`bitnet_apple_m4_serve_gate`) | The route must refuse without a ready serve gate; dense serve receipts do not enable it. |
| BitNet | streaming | `disabled` | gated BitNet chat or serve streaming | Gate-required by `M4-BITNET-EX-006` and `M4-BITNET-EX-007` / `bitnet_apple_m4_chat_streaming_semantics` or `bitnet_apple_m4_serve_streaming_semantics` | BitNet streaming is part of gate evidence and is not enabled by dense streaming receipts. |
| All | unsupported backend or machine | `unsupported` | full `apple-m4-metal`, QK256-on-Apple, Neural Engine, MPSGraph, MacBook runtime | No accepted full-route receipt | Do not route users there and do not make a support claim. |

Operator shortcuts:

```text
default dense short ask/chat: interactive
dense 1.5B short ask/chat: advisory
dense long/context profiles: batch
dense local serve: advisory
BitNet one-shot and warm: batch
BitNet chat and serve without ready gates: disabled
Metal/QK256/Neural Engine/MPSGraph/MacBook: unsupported
```

The machine-readable route-state matrix for these classes is embedded under
`route_state_matrix` in `bitnet mac status --json` and
`bitnet mac evidence --json`; see `docs/slm/apple-m4-route-state-matrix.md`.

## Supported Model Lifecycle

`bitnet mac models` now exposes the M4 supported-model lifecycle used by this
envelope. The lifecycle separates model selection from evidence claims:
`default`, `supported-non-default`, and `supported-ask` are selectable only for
their recorded scopes, while `diagnostic-only`, `candidate`, `deprecated`,
`rejected`, and `retired` are not selectable operator routes.

| State | Operator meaning | Required action before changing state |
|---|---|---|
| `default` | The implicit dense SLM M4 CPU/NEON route. | Treat as a release-gate change with exact artifact, eval, benchmark, route, cache, rollback, and envelope updates. |
| `supported-non-default` | Explicit dense SLM `--model-id` route on the recorded M4 identity. | Keep it explicit-only, verify its cache under its own id, and update claims only for that exact model. |
| `supported-ask` | Explicit BitNet one-shot ask and warm-session route only. | Keep BitNet chat and serve disabled until separate ready receipts enable those surfaces. |
| `diagnostic-only` | Debugging or blocker diagnosis, not user-ready. | Record the blocker and open a separate candidate item before any promotion review. |
| `candidate` | Pinned review identity, not a supported route. | Require artifact, tokenizer, prompt, provenance, eval, benchmark, canary, and route-state receipts before promotion. |
| `deprecated` | Transitional removal state after regression or replacement. | Stop recommending fetch and publish migration or rollback guidance before restoring support. |
| `rejected` | Failed or out-of-scope identity. | Do not fetch or route; reconsider only through a fresh candidate item. |
| `retired` | Archived identity removed from active support. | Remove active support claims and return only through a fresh candidate with current receipts. |

The lifecycle policy is not a live model run and does not add supported models,
change the default, prove BitNet chat or serve, or broaden Apple Silicon,
Metal, QK256, Neural Engine, MPSGraph, MacBook, quality, performance, or
speedup claims. JSON output includes per-row evidence, cache-migration,
operator-warning, rollback, and claim-boundary fields so downstream dashboards
can preserve those limits.

## Compatibility Refresh

`bitnet mac compat-refresh` writes the `M4-COMPAT-001` model-free contract for
compatibility refreshes after macOS, Rust toolchain, binary build-profile, or
supported-model manifest changes. The contract names the required follow-up
receipts:

```bash
bitnet mac compat-refresh \
  --json-out target/apple-m4-inference-excellence/compat/compat-refresh.json
bitnet mac doctor --json-out ci/hardware/apple-m4-mac-mini/<date>/compat/doctor.json
bitnet mac smoke --json-out ci/hardware/apple-m4-mac-mini/<date>/compat/smoke.json
bitnet mac regression-dashboard \
  --json-out ci/hardware/apple-m4-mac-mini/<date>/compat/regression-dashboard.json \
  --markdown-out ci/hardware/apple-m4-mac-mini/<date>/compat/regression-dashboard.md
```

The contract records cache repair and rollback obligations, but it is not a
runtime or performance proof. After a compatibility trigger, timing baselines
remain advisory until matching benchmark identities pass again, quality claims
still require matching eval identities, and the M4 Mac mini scope does not
extend to MacBook or broad Apple Silicon targets.

## Context Guardrails

`M4-CONTEXT-001` turns the class map into route-level guardrails for M4
operator paths. The guardrail is evidence-scoped: it classifies a request
against the recorded prompt/context envelope and records the result, but it does
not create new long-context quality or performance proof.

Current limits:

| Family and route | Recorded prompt envelope | Guardrail result beyond envelope |
|---|---:|---|
| Dense SLM `mac ask`, dense `mac chat`, dense `mac chat-smoke`, dense `mac serve` | 4096 prompt tokens, matching the recorded dense `context_4k` profile | `unsupported` with a blocked `apple_m4_context_guardrail` receipt |
| BitNet `mac ask`, BitNet `mac bitnet-warm`, gated BitNet `mac chat`, gated BitNet `mac serve` | 512 prompt tokens, matching bounded one-shot and warm-session evidence | `unsupported` with a blocked `apple_m4_context_guardrail` receipt |

Requests inside the dense short prompt envelope remain `interactive` for the
dense 0.5B models, `advisory` for dense 1.5B, and `advisory` for dense local
server requests. Dense requests inside `context_1k` or `context_4k` are
classified `batch`. BitNet requests inside the bounded envelope remain `batch`.
BitNet chat and serve still require their existing gate receipts; the context
guardrail does not enable those routes by itself.

The first `M4-CONTEXT-002` live proof receipt is recorded at
`ci/hardware/apple-m4-mac-mini/2026-05-20T1611Z/context/answer-corpus.json`.
It validates the release `bitnet mac eval --suite m4-long-context` route for
the tested default dense identity only: `qwen2.5-0.5b-instruct-q8_0`,
`apple-m4-cpu-neon`, `fallback_used=false`, 4/4 mechanical long-context cases
passed. The matching `context` benchmark receipt records `context_1k` as a
completed warm-session profile and `context_4k` as an enforced 720 second
timeout, making the aggregate invalid for timing comparison. Treat this as
bounded dense long-context evidence with an explicit timeout boundary, not as
BitNet, broad quality, or broad performance proof.

Every generated guardrail or successful route receipt records
`context_envelope` with:

```text
contract_version
work_item
route
model_family
model_id
operator_class
status
allowed
prompt_tokens
prompt_token_count_exact
max_new_tokens
requested_total_tokens
recorded_envelope
claim_boundary
```

Blocked guardrail receipts use `artifact_kind=apple_m4_context_guardrail`,
`fallback_used=false`, `runtime_api=cpu`, prompt text omitted with SHA256 hashes
only, and `claim_boundary.live_generation_executed=false`.

## Refresh Cadence

Use this cadence to keep the M4 appliance measured without moving live hardware
runs into generic PR CI:

| Lane | When | Required surface | Live model run |
|---|---|---|---|
| Generic PR | Every PR | Campaign checks, receipt schema checks, generated tracking status, docs diff hygiene | no |
| Advisory local | After `main` sync, model cache changes, receipt schema changes, or M4 CLI changes | `bitnet mac status`, `bitnet mac report-refresh`, `bitnet mac regression-dashboard`, `bitnet mac receipts-check` | no by default |
| Scheduled M4 | Nightly or weekly on the M4 Mac mini | Dense SLM eval/benchmark, BitNet eval/benchmark, BitNet variable warm, report-refresh, regression-dashboard | yes |
| Release gate | Before publishing a new M4 expectation envelope | Full supported dense matrix, accepted BitNet artifact/tokenizer eval and benchmark, warm-session receipts, dashboard refresh, operator docs | yes |

The model-free refresh sequence is:

```bash
bitnet mac models
bitnet mac status
bitnet mac evidence \
  --json-out target/apple-m4-inference-excellence/evidence-summary.json \
  --json
bitnet mac evidence replay \
  --bundle ci/hardware/apple-m4-mac-mini/2026-05-21T145609Z/evidence-replay/manifest.json \
  --dry-run \
  --json
bitnet mac report-refresh \
  --json-out target/apple-m4-inference-excellence/report-refresh-manifest.json \
  --explain \
  --open-targets \
  --json
bitnet mac regression-dashboard \
  --json-out target/apple-m4-inference-excellence/regression-dashboard.json \
  --markdown-out target/apple-m4-inference-excellence/regression-dashboard.md \
  --explain \
  --open-targets \
  --json
bitnet mac receipts-check target/apple-m4-inference-excellence/regression-dashboard.json --json
bitnet mac receipts-check target/apple-m4-inference-excellence/evidence-summary.json --json
bitnet mac receipts-check \
  ci/hardware/apple-m4-mac-mini/2026-05-21T145609Z/evidence-replay/manifest.json \
  --json
```

The `--explain` and `--open-targets` flags are operator affordances only. They
print status meanings, per-family or per-group reasons, and openable receipt,
Markdown, latest-report, and baseline-report targets without launching live
inference or downloading models.

The evidence replay dry-run is also model-free. It checks only the committed
bundle manifest and referenced receipt/dashboard paths, records a dry-run audit
receipt, and leaves live dense SLM or BitNet refresh execution in the scheduled
M4 lane.

The detailed CI and artifact-retention contract is
`docs/slm/apple-m4-evidence-ci-lanes.md`. Generic PR Tier 0 stays model-free;
fresh live M4 dense SLM or BitNet evidence belongs only in advisory local,
scheduled M4, or release-gate lanes.

`bitnet mac status` and `bitnet mac doctor` expose dense SLM and BitNet
readiness as separate operator states. Dense readiness is tied to supported
Qwen cache repair and the latest dense receipts. BitNet readiness is tied to
the accepted artifact/tokenizer, one-shot ask, warm-session evidence, and
explicit chat/serve-disabled boundaries.

The live refresh sequence belongs only in advisory, scheduled, or release lanes:

```bash
bitnet mac benchmark --calibrate \
  --json-out ci/hardware/apple-m4-mac-mini/<date>/benchmark/calibration.json

target/release/bitnet --device apple-m4-cpu-neon mac benchmark \
  --model-id <dense-model-id> \
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

target/release/bitnet --device apple-m4-cpu-neon mac bitnet-benchmark \
  --model-id microsoft-bitnet-b1.58-2B-4T-i2s \
  --json-out ci/hardware/apple-m4-mac-mini/<date>/bitnet-benchmark/summary.json

target/release/bitnet --device apple-m4-cpu-neon mac bitnet-warm \
  --model-id microsoft-bitnet-b1.58-2B-4T-i2s \
  --model-path models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf \
  --tokenizer models/microsoft-bitnet-b1.58-2B-4T/tokenizer.json \
  --prompt 'Answer with a single digit: 2+2=' \
  --prompt 'Name the capital of France. Answer with one word.' \
  --prompt 'Return exactly: ready' \
  --prompt 'Answer with a single digit: 3+1=' \
  --prompt 'Answer with a single digit: 2+2=' \
  --json-out ci/hardware/apple-m4-mac-mini/<date>/bitnet-productization/variable-warm-session.json

target/release/bitnet --device apple-m4-cpu-neon mac reliability-drill \
  --json-out ci/hardware/apple-m4-mac-mini/<date>/reliability-drills/summary.json

target/release/bitnet mac receipts-check <new-receipt.json> --json
target/release/bitnet mac regression <new-eval-or-benchmark.json> --baseline <matching-baseline.json>
target/release/bitnet mac report-refresh --json
target/release/bitnet mac regression-dashboard --json
```

## Trend Retention

Scheduled trend reports are derived from committed receipts, not from
previously generated dashboard output. Keep the current and previous matching
baseline for each dense SLM and BitNet dashboard group so the dashboard can
rebuild `ready`, `warning`, `failure`, or `insufficient_history` status from
source evidence.

Retained source receipts include committed dense SLM eval, benchmark, chat,
context, smoke, and stability summaries; committed BitNet eval, benchmark,
variance, one-shot, and warm-session summaries; any BitNet chat or serve gate
receipt after those routes are enabled; and setup, status, doctor, prune, or
cache-repair receipts that the envelope uses for operator guidance. Large child
receipts are retained only when the accepted evidence bundle needs them for
receipt validation, generated text/token-ID audit, or failure taxonomy. Local
model files, cache copies, and intermediate `target/` outputs are not retained
as evidence.

Regenerate the report-refresh manifest, regression-dashboard JSON and Markdown,
operator evidence summary, status tables, and open-target explanations from the
retained receipts. Generated outputs may be committed only when the campaign
item explicitly calls for that artifact; otherwise they remain local refresh
products under `target/apple-m4-inference-excellence/`.

An identity ages out of current operator claims when the model SHA, tokenizer
authority or SHA, prompt/template/stop/generation identity, corpus version,
benchmark profile set, backend, runtime API, fallback state, machine ID, route
gate receipt, or required receipt schema no longer matches the retained
baseline. It also ages out when a newer accepted identity has two matching
refreshes, the supported-model matrix deprecates or removes it, or it misses two
scheduled M4 refresh cycles while the envelope still depends on it.

Stale identities remain historical evidence. They must not be used for current
trend claims until a fresh matching-history pair exists and the dashboard group
is `ready` or accepted with documented warnings. Refresh this operator envelope
whenever stale aging changes a route class, default model, supported-model
state, BitNet artifact/tokenizer identity, route enablement, threshold result,
context boundary, disk/cache floor, dashboard status, or claim-boundary wording.

`M4-TREND-001` records the first committed seven-day trend summary under:

```text
ci/hardware/apple-m4-mac-mini/2026-05-22T0530Z/trend/seven-day-history.json
ci/hardware/apple-m4-mac-mini/2026-05-22T0530Z/trend/seven-day-history.md
```

The summary keeps nine dashboard groups in `ready` matching-history state for
dense SLM eval-v2, dense SLM benchmark-v2, BitNet eval, BitNet benchmark, and
BitNet variable warm. It also records skipped-day reasons for days that produced
setup, variance, context, reliability, serve, or repaired-eval evidence without
a second matching receipt for the dashboard family. The only current advisory
impact is BitNet variable-warm resident memory growth of `25.63%` against the
`10%` higher-is-worse advisory threshold. No route class changes and no BitNet
chat or serve enablement follow from this trend summary.

## Regression Thresholds

The dashboard may compare reports only when the identity context matches:

```text
artifact kind
evidence family
model id
model SHA256
tokenizer authority and tokenizer SHA256 when present
prompt template or benchmark profile set
selected backend
runtime API
fallback_used=false
machine id
```

If any identity field does not match, the report is a new baseline. It must not
be described as a trend against the previous context.

Dashboard states mean:

| State | Meaning | Operator action |
|---|---|---|
| `ready` | At least two committed reports share the same comparison identity. | Use dashboard warnings/failures as the regression signal. |
| `insufficient_history` | Only one matching report exists for that identity. | Commit another matching refresh before claiming a trend. |
| `identity_mismatch` or profile-set mismatch | The current report differs in model, tokenizer, backend, fallback, corpus, or benchmark profiles. | Treat the report as a new baseline. |
| warning | Context matched, but an advisory metric drifted. | Inspect the metric and receipt, then decide whether to refresh or file a follow-up. |
| failure | Context matched and a required quality, fallback, identity, or receipt invariant failed. | Block the claim and fix before publishing the envelope. |

The `M4-EXCELLENCE-003` model-free dashboard refresh reports all important
committed M4 evidence groups as comparable:

| Family | Evidence | Reports | Comparable groups |
|---|---|---:|---:|
| `dense_slm_eval_v2` | dense SLM | 6 | 3 |
| `dense_slm_benchmark_v2` | dense SLM | 6 | 3 |
| `bitnet_eval` | BitNet | 2 | 1 |
| `bitnet_benchmark` | BitNet | 2 | 1 |
| `bitnet_variable_warm` | BitNet | 2 | 1 |

This dashboard state removes the prior important `insufficient_history` gap for
dense SLM eval v2 and BitNet variable warm. It remains dashboard-only evidence:
no live model run, model download, BitNet chat/serve enablement, Metal, QK256,
Neural Engine, MPSGraph, MacBook, broad quality, broad performance, or speedup
claim is made by the refresh.

Quality gates must fail the release claim when any of these change in a matching
context:

```text
fallback_used becomes true
selected_backend changes away from apple-m4-cpu-neon
model SHA256 or tokenizer SHA256 changes without starting a new baseline
cases_total changes without an explicit corpus version update
timeouts or not_run cases become non-zero
pass rate drops beyond the dashboard threshold
valid UTF-8, generated text, or generated token IDs disappear
```

Performance drift is advisory unless the release gate explicitly promotes a
metric to required. Operators should still inspect p50, p90, and p99 for:

```text
cold_load_ms
tokenizer_load_ms
prompt_tokenize_ms
prefill_ms
time_to_first_token_ms
sampling_ms_per_token
input_tok_s
output_tok_s
decode_tok_s
total_wall_ms
peak_memory_mb
memory_drift_mb
```

### Published Drift Thresholds

`bitnet mac regression` is advisory by default. Use `--fail-on-drift` when a
scheduled or release lane needs any warning to fail the gate. These thresholds
only apply after the dashboard reports `ready` for a matching identity; an
identity mismatch starts a new baseline and must not be called a trend.

Hard comparison blockers:

| Class | Blocker | Operator action |
|---|---|---|
| Identity | artifact kind, evidence family, model ID/SHA, tokenizer authority/SHA, prompt template or profile set, backend, runtime API, fallback state, or machine mismatch | Start a new baseline or rerun with the intended identity. |
| Claim boundary | a receipt claims BitNet chat/serve, full Metal, QK256, Neural Engine, MPSGraph, MacBook, broad quality, broad performance, or speedup outside its proven lane | Block the public claim and file a follow-up. |
| Required fields | missing generated text, token IDs, timing, memory, fallback, tokenizer, or quality fields required by the receipt validator | Fix the receipt producer before comparing drift. |

Quality and timeout thresholds:

| Family | Fields | Threshold | Severity |
|---|---|---:|---|
| Dense SLM eval v2 | `cases_passed`, exact/normalized/schema/numeric/keyword/token pass counts, scoring summary, task-family pass counts | 0% lower allowed | warning by default; blocks release quality claims until explained |
| Dense SLM eval v2 | timeouts, `not_run`, scoring failures, `quality_passed=false` | 0% higher allowed or boolean mismatch | block release quality claims |
| BitNet eval | quality/scoring passed counts, task-family passed counts, reference matched/text/token-ID match counts | 0% lower allowed | warning by default; blocks BitNet quality claims until explained |
| BitNet eval | failed, timeout, `not_run`, reference mismatch/not-run/partial counts | 0% higher allowed | block BitNet quality claims |
| BitNet benchmark paths | prompt count, generated token count, model/tokenizer-loaded-once flags, timeout-boundary flags, quality flags | exact match required | comparison blocker if mismatched |
| BitNet variable warm | accepted artifact/tokenizer, backend, fallback state, repeated-prompt quality, per-turn receipts, and aggregate receipt validity | exact match before drift is reported | direct `bitnet mac regression --baseline` support for matching `bitnet_apple_m4_warm_session` receipts |

Timing drift thresholds:

| Family | Metric class | Direction | Advisory threshold |
|---|---|---|---:|
| Dense SLM eval v2 | `cold_load_ms_p50`, `tokenizer_load_ms_p50` | higher is worse | 20% |
| Dense SLM eval v2 | prompt tokenization, prefill, TTFT p50/p90, total wall p50 | higher is worse | 15% |
| Dense SLM eval v2 | sampling ms/token p50 | higher is worse | 20% |
| Dense SLM eval v2 | input/output throughput p50 | lower is worse | 15% |
| Dense SLM eval v2 | decode throughput p50 | lower is worse | 12.5% |
| Dense SLM benchmark v2 | load metrics p50/p90/p99 | higher is worse | 20% |
| Dense SLM benchmark v2 | prompt tokenization, prefill, TTFT, decode total, total wall p50/p90/p99 | higher is worse | 15% |
| Dense SLM benchmark v2 | sampling ms/token p50/p90/p99 | higher is worse | 20% |
| Dense SLM benchmark v2 | input/output throughput p50/p90/p99 | lower is worse | 15% |
| Dense SLM benchmark v2 | decode throughput p50/p90/p99 | lower is worse | 12.5% |
| Dense SLM warm/performance sessions | TTFT and total session | higher is worse | 15% |
| Dense SLM warm/performance sessions | warm-prompt throughput | lower is worse | 15% |
| Dense SLM warm/performance sessions | decode throughput | lower is worse | 12.5% |
| BitNet benchmark | cold/model/tokenizer load p50/p90/p99 | higher is worse | 20% |
| BitNet benchmark | prompt tokenization, prefill, TTFT, decode total, total wall p50/p90/p99 | higher is worse | 15% |
| BitNet benchmark | sampling ms/token p50/p90/p99 | higher is worse | 20% |
| BitNet benchmark | input/output throughput p50/p90/p99 | lower is worse | 15% |
| BitNet benchmark | decode throughput p50/p90/p99 | lower is worse | 12.5% |

Memory drift thresholds:

| Family | Metric | Direction | Advisory threshold |
|---|---|---|---:|
| Dense SLM eval v2 | peak memory | higher is worse | 10% |
| Dense SLM benchmark v2 | peak memory p50/p90/p99 | higher is worse | 10% |
| Dense SLM benchmark v2 | memory drift p50/p90/p99 | higher is worse | 15% |
| Dense SLM warm/performance sessions | peak memory | higher is worse | 10% |
| BitNet benchmark | peak memory p50/p90/p99 | higher is worse | 10% |
| BitNet benchmark | memory drift p50/p90/p99 and process peak drift | higher is worse | 15% |

BitNet variable warm has matching-history dashboard status, receipt validation,
and direct `bitnet mac regression --baseline` support for
`bitnet_apple_m4_warm_session`. The direct command rejects mismatched
artifact/tokenizer identity, backend, fallback state, prompt set, warm profile,
timeout policy, or receipt schema before it reports timing or memory drift.

The current BitNet benchmark comparison reports five advisory warnings, all on
sub-ms prompt-tokenize timing fields. Identity, fallback, prompt count, and
generation fields still validate.

## Resident-100 Status

The 2026-05-15T1845Z dense benchmark refresh ran all supported dense M4 model
IDs across the full nine-profile set. The `resident_100` profile is now the
longer warm-session stability sample for dense SLMs:

| Model | Generated | TTFT p50 | TTFT p99 | Decode tok/s p50 | Output tok/s p50 | Peak MB | Memory drift MB |
|---|---:|---:|---:|---:|---:|---:|---:|
| `qwen2.5-0.5b-instruct-q8_0` | 860 | 2150.0 ms | 2246.0 ms | 15.650 | 1.707 | 4156.750 | 1.875 |
| `qwen2.5-0.5b-instruct-q4_k_m` | 928 | 2151.0 ms | 2246.0 ms | 15.650 | 3.078 | 4159.609 | 0.968 |
| `qwen2.5-1.5b-instruct-q4_k_m` | 804 | 8078.0 ms | 8966.0 ms | 4.780 | 0.352 | 8395.047 | 0.000 |

This supports a bounded dense M4 resident-stability claim for the recorded
model identities and profile set. It does not prove BitNet behavior and it does
not claim broad Apple Silicon performance.

The same dense benchmark refresh shows that long-context prompts are the main
latency tail. Operators should consult
`docs/slm/apple-m4-durable-inference-evidence.md` before treating 1k or 4k
context profiles as interactive.

## BitNet Durable Status

The 2026-05-15T2214Z BitNet refresh uses:

```text
model_repo=microsoft/bitnet-b1.58-2B-4T-gguf
model_file=ggml-model-i2_s.gguf
model_sha256=4221b252fdd5fd25e15847adfeb5ee88886506ba50b8a34548374492884c2162
tokenizer_authority=external_tokenizer_json
tokenizer_sha256=e134af98b985517b4f068e3755ae90d4e9cd2d45d328325dc503f1c6b2d06cc7
selected_backend=apple-m4-cpu-neon
runtime_api=cpu
fallback_used=false
```

Current matching-history status:

| Family | Result | Trend status |
|---|---|---|
| BitNet eval | 100 cases, 75 passed, 25 failed, 0 timeout, 0 not_run | comparable |
| BitNet benchmark | 4 prompts, 8 generated tokens, TTFT p50 7910.0 ms, decode p50 2.065 tok/s, peak memory p50 4246.359 MiB | comparable with advisory prompt-tokenize warnings |
| BitNet variable warm | 5 prompts, 10 generated tokens, repeated prompt stable, total session 43809.341 ms | comparable |

BitNet chat and BitNet serve remain disabled. Variable warm receipts are
required evidence for future chat work, but they are not chat enablement by
themselves.

## Disk And Cache Guidance

Before a scheduled or release refresh, run:

```bash
df -h .
bitnet mac models
bitnet model verify <model-id>
bitnet model prune
```

Use these local thresholds:

| Situation | Minimum free disk | Guidance |
|---|---:|---|
| Full dense plus BitNet refresh | 20 GiB | Preferred floor before running benchmark, eval, warm, dashboard, and receipt validation in one session. |
| Single dense ask, smoke, or doctor | 10 GiB | Enough for ordinary operation if the required model is already cached. |
| New model fetch or cache repair | model size plus 10 GiB | Verify cache state first and avoid duplicate GGUF copies. |
| Below 10 GiB | stop | Prune caches or archive artifacts before running long M4 jobs. |

Supported dense cache sizes are small enough to keep together on the appliance:

| Model | Approx cache size |
|---|---:|
| `qwen2.5-0.5b-instruct-q4_k_m` | 468.64 MiB |
| `qwen2.5-0.5b-instruct-q8_0` | 644.41 MiB |
| `qwen2.5-1.5b-instruct-q4_k_m` | 1065.56 MiB |

The accepted BitNet GGUF is about 1.19 GB plus the external tokenizer. Do not
copy model files into the repository and never commit `models/**`. Prefer the
configured model cache and symlinks when a local runner needs an explicit path.

Memory guidance from the durable receipts:

| Surface | Peak memory signal |
|---|---:|
| Dense 0.5B `resident_100` | about 4.16 GiB |
| Dense 1.5B `resident_100` | about 8.40 GiB |
| BitNet one-shot benchmark | about 4.25 GiB p50 |

## Claim Boundary

Allowed claim:

```text
The M4 operator envelope describes the durable evidence refresh process,
matching-history regression boundaries, trend-retention and stale-identity
policy, resident_100 dense SLM status, disk and cache guidance, and claim
boundaries for the recorded M4 Mac mini receipts.
```

Still not allowed:

```text
BitNet chat works.
BitNet serve works.
Full apple-m4-metal inference works.
QK256 on Apple Silicon is supported.
Neural Engine execution is used.
MPSGraph model inference is used.
MacBook evidence exists.
The reports are broad Apple Silicon benchmarks.
The reports prove broad model quality.
Dense SLM evidence proves BitNet behavior.
The envelope proves a speedup.
```

When a future campaign changes any of those boundaries, it must publish a new
receipt family or matching-history report set before updating the operator
claim.
