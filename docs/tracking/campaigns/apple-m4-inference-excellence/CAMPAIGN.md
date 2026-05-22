# Apple M4 Inference Excellence Campaign

Campaign ID: `apple-m4-inference-excellence`

Status: active

## Objective

Move the M4 Mac mini from complete evidence lanes to excellent, repeatable
local inference across supported dense SLMs and BitNet: deeper deterministic
accuracy, fuller benchmark envelopes, enough matching-history receipts to
remove important `insufficient_history` gaps, BitNet-specific product proof,
reproducible run and artifact identity, service-surface conformance, clearer
operator UX, and phase-scoped acceleration discipline.

## Why This Exists

The M4 dense SLM, BitNet local-answer, BitNet eval/benchmark, BitNet
productization, inference-ops, local-server, Metal phase, and durable evidence
campaigns are complete. The durable closeout intentionally preserved two
important boundaries: dense SLM eval v2 and BitNet variable warm still needed
more matching-history receipts, and BitNet chat/serve remained disabled.

This campaign turns that honest state into the next proof ladder. It should not
repeat baseline work or use dense Qwen results as BitNet evidence. It should
make the M4 Mac mini easier to operate and harder to overclaim.

## End State

- Every supported dense M4 SLM model has repeated matching-history eval and
  benchmark receipts under the same model/tokenizer/backend/fallback identity.
- The accepted BitNet artifact has repeated matching-history eval, benchmark,
  and variable warm-session receipts before any chat or serve claim changes.
- Dense SLM and BitNet eval corpora are large enough to report useful
  mechanical pass rates by task family without required LLM judging.
- Benchmark reports include load, tokenization, prefill/input throughput,
  TTFT, output/decode throughput, total wall time, peak memory, memory drift,
  and percentile summaries.
- Every committed M4 receipt records enough run identity to reproduce and
  compare it: machine, OS, git commit, binary/build profile, model artifact,
  tokenizer authority, prompt template, backend, fallback state, corpus/profile
  seed, and command class.
- Supported model artifacts have provenance manifests and cache verification
  receipts that separate file identity, tokenizer authority, prompt template
  identity, license/source metadata, and local cache state.
- Corpus, scorer, normalization, and golden-token canary versions are recorded
  before pass rates or drift are interpreted.
- Benchmark summaries record environment preflight and repeatability variance
  before p50/p90/p99 drift thresholds are treated as meaningful.
- Operator docs translate receipts into local operating classes such as
  interactive, advisory, batch, and unsupported.
- Dense SLM and BitNet service surfaces have explicit health, ready,
  streaming, timeout, cancellation, and per-request receipt conformance before
  any user-facing service claim expands.
- Release gates state exactly which eval, benchmark, stability, service, and
  operator checks must pass before updating the public M4 expectation envelope.
- Operator UX exposes supported models, cache/disk health, last dense and
  BitNet reports, current regressions, unsupported claim boundaries, and
  recommended next commands.
- Metal work remains phase-scoped with CPU parity, fallback-free phase receipts,
  and explicit CPU/NEON remainder until a full route is separately proven.

## Hard Constraints

- This is an M4 Mac mini inference-excellence campaign.
- Do not reopen completed Apple M4 dense SLM, BitNet local-answer, BitNet
  eval/benchmark, BitNet productization, inference-ops, local server, durable
  evidence, or Metal campaigns unless a regression proves they are wrong.
- Do not use dense Qwen evidence as BitNet evidence.
- Do not use MacBook artifact-only evidence as M4 runtime proof.
- Do not enable BitNet chat or BitNet serve until the campaign item for that
  surface has matching receipts and gates.
- Do not claim full `apple-m4-metal` inference, QK256 support, Neural Engine
  execution, MPSGraph model inference, MacBook evidence, broad Apple Silicon
  performance, broad model quality, or speedup.
- Do not add live model downloads, hardware timing runs, BitNet runtime runs,
  or long resident soaks to generic required PR CI.
- Never commit model binaries.
- Do not let eval, benchmark, serve, or status commands contact external
  services except for explicit model fetch commands.

## Work Items

| Work item | Status | Notes |
|---|---|---|
| M4-EXCELLENCE-001 | merged | Add a second dense SLM eval-v2 refresh so dense eval groups can become comparable. |
| M4-EXCELLENCE-002 | merged | Add a second BitNet variable warm refresh so the warm group can become comparable. |
| M4-EXCELLENCE-003 | merged | Regenerate dashboard trend validation after the second matching-history receipts land. |
| M4-EXCELLENCE-004 | merged | Publish drift thresholds by dense SLM and BitNet family. |
| M4-ACCURACY-000 | merged | Freeze corpus/scorer version, seed, normalization, and expected-output contracts. |
| M4-ACCURACY-001 | merged | Expand the seeded dense SLM corpus to 100 deterministic mechanically scored cases. |
| M4-ACCURACY-002 | merged | Expand the dense SLM corpus to 500 deterministic cases once scoring is stable. |
| M4-ACCURACY-003 | merged | Fix stop-token, template, normalization, and scoring misses found by the larger corpus. |
| M4-ACCURACY-004 | merged | Publish dense pass rates by deterministic task family and supported model identity. |
| M4-ACCURACY-005 | merged | Add mechanical failure taxonomy for regression triage. |
| M4-ACCURACY-006 | merged | Run full 500-case dense SLM eval-v2 receipts for every supported M4 dense model identity. |
| M4-ACCURACY-007 | merged | Add a second matching 500-case dense eval-v2 refresh for trend history. |
| M4-CANARY-001 | proposed | Add dense SLM and BitNet golden-token trace canaries. |
| M4-DENSE-REF-001 | proposed | Add a dense reference-vs-Rust control for supported Qwen identities. |
| M4-DENSE-CHAT-001 | proposed | Prove dense CLI ask/chat conformance with per-turn receipts. |
| M4-ROBUSTNESS-001 | proposed | Add dense SLM and BitNet negative/robustness eval cases. |
| M4-BENCH-001 | merged | Ensure benchmark receipts include the full timing and memory metric contract. |
| M4-BENCH-004 | merged | Add benchmark environment preflight fields and invalid-comparison reasons. |
| M4-BENCH-002 | merged | Publish p50/p90/p99 and min/max summaries for the supported dense benchmark matrix. |
| M4-BENCH-008 | merged | Add the repeat-run benchmark variance harness before live variance envelopes. |
| M4-BENCH-009 | merged | Enforce calibrated dense benchmark profile timeouts before live variance envelopes. |
| M4-BENCH-010 | merged | Make benchmark variance aggregation timeout-aware before live dense envelopes. |
| M4-BENCH-005 | merged | Publish repeatability and variance envelopes for supported dense Qwen timing families. |
| M4-BENCH-006 | merged | Publish explicit BitNet one-shot and warm variance receipts. |
| M4-BENCH-003 | merged | Wire benchmark summaries into matching-identity regression comparisons. |
| M4-BITNET-REG-001 | merged | Add direct BitNet warm-session regression support. |
| M4-BITNET-EX-001 | merged | Add a BitNet-specific 100-case deterministic corpus. |
| M4-BITNET-EX-002 | merged | Compare reference-runner and Rust M4 BitNet answers under the accepted identity. |
| M4-BITNET-EX-003 | merged | Publish the BitNet one-shot benchmark envelope. |
| M4-BITNET-EX-004 | merged | Add BitNet variable warm 25/50/100 prompt soaks. |
| M4-BITNET-EX-005 | merged | Harden BitNet progress, timeout, partial-failure, and repair UX. |
| M4-BITNET-EX-006 | merged | Enable BitNet chat only after warm, timeout, determinism, and streaming gates pass. |
| M4-BITNET-EX-007 | merged | Enable BitNet serve only after chat and service receipts pass. |
| M4-BITNET-EX-008 | merged | Publish BitNet task-family and failure-taxonomy reports. |
| M4-BITNET-EX-009 | merged | Add matching BitNet eval history and a larger-corpus decision. |
| M4-BITNET-EX-010 | merged | Define the staged 250-case BitNet corpus and scorer contract. |
| M4-BITNET-EX-011 | merged | Run and publish bounded 250-case BitNet M4 receipts. |
| M4-BITNET-EX-012 | merged | Decide whether BitNet expands to 500 cases or repairs corpus/scoring first. |
| M4-OPS-UX-001 | merged | Add an operator-facing M4 evidence summary. |
| M4-OPS-UX-002 | merged | Add explain/open affordances for report-refresh and regression-dashboard outputs. |
| M4-OPS-UX-003 | merged | Extend doctor/status UX to report dense SLM and BitNet readiness separately. |
| M4-OPS-SLO-001 | merged | Translate receipts into local operator envelope classes. |
| M4-CONTEXT-001 | merged | Add long-context guardrails against recorded dense SLM and BitNet evidence envelopes. |
| M4-CONTEXT-HARNESS-001 | merged | #6137 registered the `m4-long-context` dry-run command contract and `context` benchmark alias without claiming live long-context proof. |
| M4-CONTEXT-002 | merged | Publish long-context quality and timing receipts where supported. |
| M4-REPRO-001 | merged | Define the reusable M4 run-identity contract for all evidence families. |
| M4-REPRO-002 | merged | Publish supported-model artifact provenance and cache verification manifests. |
| M4-REPRO-003 | proposed | Record prompt-template, stop-sequence, and generation-parameter identity. |
| M4-RECEIPT-001 | merged | Add receipt-schema compatibility and negative fixtures for M4 evidence families. |
| M4-STABILITY-HARNESS-001 | merged | Implement the mixed dense-model switch benchmark harness before recording soak evidence. |
| M4-STABILITY-001 | merged | Run a mixed dense-model switch soak with cache reuse and memory-drift evidence. |
| M4-STABILITY-002 | proposed | Add cache and disk-pressure repair receipts for operator flows. |
| M4-STABILITY-003 | merged | Define scheduled M4 trend-retention and stale-identity policy. |
| M4-RELIABILITY-001 | merged | Add recovery drills for cancellation, interruption, low disk, cache corruption, and restart. |
| M4-OBS-001 | merged | Correlate progress events, logs, receipts, and failure diagnostics. |
| M4-SERVE-EX-001 | pr_open | Refresh dense SLM local-server conformance receipts. |
| M4-SERVE-EX-002 | proposed | Prove dense and BitNet streaming/failure semantics after BitNet serve is gated. |
| M4-SERVE-EX-003 | merged | Document and test local-server safety defaults for appliance operation. |
| M4-SERVE-EX-004 | merged | Add bounded server queue, backpressure, and resident-state evidence. |
| M4-CI-001 | merged | Codify PR, advisory, scheduled, release, and retention evidence lanes. |
| M4-SETUP-001 | proposed | Prove first-run setup, fetch or repair, cache verification, and smoke receipts. |
| M4-BENCH-007 | proposed | Calibrate the benchmark harness before timing envelopes are interpreted. |
| M4-ROUTE-MATRIX-001 | merged | Publish the route-state matrix for dense SLM and BitNet command surfaces. |
| M4-WORKLOAD-001 | merged | Add end-to-end operator workload receipts across enabled M4 routes. |
| M4-EVIDENCE-REPLAY-001 | merged | Add replayable evidence bundles for dense SLM and BitNet refreshes. |
| M4-TREND-001 | merged | Publish seven-day matching-identity trend history and skipped-day reasons. |
| M4-MODEL-LIFECYCLE-001 | merged | Define supported-model lifecycle states and claim-boundary requirements. |
| M4-COMPAT-001 | merged | Define compatibility refresh receipts after OS, toolchain, binary, or manifest changes. |
| M4-CLAIM-LINT-001 | merged | #6190 added static M4 claim-boundary wording checks. |
| M4-RELEASE-001 | proposed | Publish the M4 inference release go/no-go matrix. |
| M4-METAL-EX-001 | proposed | Choose one named future Metal phase and document parity/receipt requirements. |
| M4-METAL-EX-002 | proposed | Implement that named phase only with CPU parity and fallback-free phase receipts. |

## External Dependencies

`apple-bitnet-artifact-sweep` remains a separate Apple Silicon artifact lane.
Its MacBook evidence can qualify candidate artifacts and tokenizers, but it
does not prove M4 runtime behavior. Any accepted artifact still needs M4
CPU/NEON receipts before it can change an M4 local-answer, chat, serve, or
benchmark claim.

## Review Policy

Each PR should own one item. Live model execution and timing refreshes belong
in local, advisory, scheduled, or release lanes and must record exact
model/tokenizer/backend/fallback identity before any drift claim is made.
Generic required PR CI remains model-free unless the item is a small parser,
schema, fixture, docs, or receipt-validation change.
