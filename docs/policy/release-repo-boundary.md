# Release Repository Boundary

Status: active
Owner: release maintainers
Created: 2026-05-20
Linked proposal: n/a
Linked specs: n/a
Linked ADRs: n/a
Linked plan: `docs/release/SWARM_PROMOTION.md`
Linked issues: n/a
Linked PRs: n/a
Support-tier impact: release and publish intake only
Policy impact: defines accepted PR classes for this repository

## Purpose

`EffortlessMetrics/BitNet-rs` is the release and publish repository for
BitNet-rs. Active feature, hardware, performance, diagnostic, campaign,
refactor, and proof-tooling work belongs in
`EffortlessMetrics/bitnet-rs-swarm`.

This boundary keeps `BitNet-rs` from becoming a second active development queue.

## Allowed PR Classes

PRs may target this repository only when they are one of:

- `release-promotion`: a promotion from `bitnet-rs-swarm` into the release repo;
- `release-hotfix`: a release-blocking fix needed before publish;
- `publish-metadata`: versioning, changelog, packaging, signing, or publish
  metadata;
- `security-hotfix`: emergency security work;
- `docs-for-current-release`: documentation corrections needed for released
  artifacts.

## Redirected PR Classes

Open these in `bitnet-rs-swarm`, not here:

- hardware lane work;
- feature development;
- performance experiments;
- diagnostic receipt expansion;
- refactor-only development;
- campaign work;
- generated proof dashboard work;
- broad model-family expansion;
- A770, CUDA, Apple, Lunar Lake, CPU AVX2, or server proof lanes.

## Closure Rules

Closing a PR is not backlog reduction. Do not close a PR because it is old,
behind main, from an old branch chain, noisy, diagnostic-only, or needs a
restack.

Close only after content review proves one of:

- the exact useful content already landed;
- the exact useful content was clean-ported and the successor landed;
- the PR is a true duplicate of a named kept PR;
- the PR is historical-only evidence captured in a committed ledger or report;
- the idea was explicitly rejected after content review.

If future work remains, keep the PR open or create and link a tracking issue
before closing. Preserve PR identity where feasible.

## Initial Enforcement

The initial policy is documentation-first. A future advisory check may reject
obvious release-repo boundary violations, such as:

- changes under `crates/**` or `ci/hardware/**` without a release-promotion or
  release-hotfix label;
- campaign dashboard changes made directly in this repository;
- release-promotion PRs without a source `bitnet-rs-swarm` commit or PR list.

Do not make the first check too broad. It should protect the release boundary
without hiding valid migrated work or proof-gated source PRs.
