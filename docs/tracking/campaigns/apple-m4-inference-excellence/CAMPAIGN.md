# Apple M4 Inference Excellence Campaign

Campaign ID: `apple-m4-inference-excellence`

Status: complete; tracker has 78/78 work items merged and no next item.

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

`active.toml` is the authoritative item ledger, and
`generated/status.md` renders the current PR, branch, review, merge, and human
gate state. Do not update item state in this hand-maintained campaign overview;
run `cargo run --locked -p xtask --no-default-features -- campaign generate`
after editing the tracker.

Current generated state:

- 78 tracked work items.
- 78 merged work items.
- `docs/tracking/generated/global-dashboard.md` reports
  `M4-METAL-EX-002` as merged with next item `none`.
- Completion audit:
  `ci/hardware/apple-m4-mac-mini/2026-05-22/m4-inference-excellence-completion-audit.json`.

| Area | Merged tracker items |
|---|---|
| Excellence baseline | M4-EXCELLENCE-001 through M4-EXCELLENCE-004 |
| Dense and shared accuracy | M4-ACCURACY-000 through M4-ACCURACY-007, M4-CANARY-001, M4-DENSE-REF-000, M4-DENSE-REF-001, M4-DENSE-CHAT-001, M4-ROBUSTNESS-001 |
| Benchmark and regression | M4-BENCH-001 through M4-BENCH-010, M4-BITNET-REG-001 |
| BitNet proof ladder | M4-BITNET-EX-001 through M4-BITNET-EX-015 |
| Operator UX and envelopes | M4-OPS-UX-001 through M4-OPS-UX-003, M4-OPS-SLO-001 |
| Context and reproducibility | M4-CONTEXT-001, M4-CONTEXT-HARNESS-001, M4-CONTEXT-002, M4-REPRO-001 through M4-REPRO-004, M4-RECEIPT-001 |
| Stability, reliability, and observability | M4-GATE-HYGIENE-001, M4-STABILITY-HARNESS-001, M4-STABILITY-001 through M4-STABILITY-003, M4-RELIABILITY-001, M4-OBS-001 |
| Service, CI, route, and lifecycle | M4-SERVE-EX-001 through M4-SERVE-EX-004, M4-CI-001, M4-SETUP-001, M4-ROUTE-MATRIX-001, M4-WORKLOAD-001, M4-EVIDENCE-REPLAY-001, M4-TREND-001, M4-MODEL-LIFECYCLE-001, M4-COMPAT-001 |
| Claim and release gates | M4-CLAIM-LINT-001, M4-RELEASE-001 |
| Metal phase discipline | M4-METAL-EX-001, M4-METAL-EX-002 |

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
