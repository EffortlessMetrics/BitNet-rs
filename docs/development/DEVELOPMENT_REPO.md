# BitNet-rs Development Repository Cutover

Active development has moved to:

```text
https://github.com/EffortlessMetrics/bitnet-rs-swarm
```

This repository is now the release and publish repository for BitNet-rs. Treat
it as the stable source for public release history, tags, crates.io publish
flow, release notes, package metadata, signed artifacts when present, stable
release branches, and emergency release-blocking fixes.

## Accepted Work Here

Open or merge PRs in this repository only for:

- release promotion PRs from `bitnet-rs-swarm`;
- versioning, changelog, packaging, signing, and publish changes;
- emergency security or release-blocking hotfixes;
- documentation corrections needed for released artifacts.

Do not open normal feature, hardware-lane, performance, diagnostic, refactor,
or proof-tooling PRs here. Put that work in `bitnet-rs-swarm`.

## PR Disposition Rules

Closing is not backlog reduction. Do not close a PR because it is old, behind
main, noisy, from an old branch chain, diagnostic-only, or needs restacking.

Valid close reasons are limited to:

- the exact useful content already landed;
- the exact useful content was clean-ported and the successor landed;
- the PR is a true duplicate of a named kept PR;
- the PR is historical-only diagnostic evidence captured in a committed ledger
  or report;
- the idea was explicitly rejected after content review.

If future work remains, keep the PR open or create and link a tracking issue
before closing. Preserve PR identity where feasible; valid PR review history and
comments are useful state.

## Proof-Gated Work

Draft and proof-gated PRs remain draft or proof-gated until their stated proof
lands. In particular:

- Windows BitNet.cpp fetch/build/install hardening remains proof-gated until a
  full Windows fetch/build/install completes, or the PR is explicitly narrowed
  to script-hardening-only.
- AVX2 QK256 performance work remains proof-gated until official-shape parity,
  hot-path counters, behavior receipts, and repeatable benchmark evidence
  exist.
- A770 runtime or diagnostic work remains content-bearing until exact
  successor, duplicate, port, or historical ledger capture is proven.

Current migrated proof-gated work:

- `EffortlessMetrics/BitNet-rs#5092` was closed only after successor issue
  `EffortlessMetrics/bitnet-rs-swarm#96` was created to carry the AVX2 QK256
  GEMV LUT decode candidate and its proof checklist.

## Swarm Continuation Lanes

Move these lanes to `bitnet-rs-swarm`:

- A770 diagnostic continuation;
- A770 QK256 OpenCL implementation;
- CPU and 5700X correctness and AVX2 proof;
- behavior-suite runner;
- benchmark runner;
- device-history ledger;
- model support matrix;
- Rust-native proof tooling;
- broad model-family expansion.

## Guardrails

Do not use this repository to make the queue look smaller. Use it to publish
release-ready work and preserve traceable state.

Avoid:

- bulk closing, reopening, rebasing, or labeling;
- recreating valid PRs by default;
- running CI for queue archaeology;
- promoting partial hardware or quality evidence;
- turning drafts into merge candidates to tidy the queue.

The release repository should be boring by design. Active execution belongs in
`bitnet-rs-swarm`.
