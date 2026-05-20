# Swarm Promotion Contract

Status: active
Owner: release maintainers
Created: 2026-05-20
Linked proposal: n/a
Linked specs: n/a
Linked ADRs: n/a
Linked plan: n/a
Linked issues: n/a
Linked PRs: n/a
Support-tier impact: release promotion only
Policy impact: defines the legal path from `bitnet-rs-swarm` to `BitNet-rs`

## Purpose

Active development lands in:

```text
EffortlessMetrics/bitnet-rs-swarm
```

Releases are promoted into:

```text
EffortlessMetrics/BitNet-rs
```

This document defines the one normal path from swarm development to release
publication.

## Required Promotion Metadata

Every release-promotion PR must include:

```text
source_repo = EffortlessMetrics/bitnet-rs-swarm
source_commit = <swarm main sha>
source_prs = [<included swarm PR numbers>]
release_target = patch | minor | prerelease
version = <x.y.z or prerelease version>
changelog = <summary or changelog path>
proof_pack = <receipt or manifest path>
excluded_work = <known swarm work intentionally not promoted>
```

If the promotion includes hardware, model, quality, performance, or residency
claims, the proof pack must name the exact receipts and claim ledgers that allow
those claims. Missing receipts mean the claim remains out of scope.

## Release PR Acceptance

A release-promotion PR should run release-grade package checks, not every swarm
hardware lane by default.

Expected checks include:

```text
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace --all-targets
cargo package --workspace --allow-dirty
cargo publish --dry-run, when applicable
```

When a command is unavailable or intentionally scoped down, the PR must record:

- the unavailable command;
- why it could not run;
- substitute evidence, if any;
- whether the missing proof blocks release.

## Hardware and Performance Claims

The release repo does not invent or broaden hardware claims. It consumes swarm
proof manifests.

For any promoted hardware/backend/profile claim, the release PR must preserve:

- source swarm receipt paths;
- selected backend;
- selected device;
- fallback status;
- model and tokenizer identity;
- quality gate;
- benchmark profile;
- explicit not-claims.

Do not promote A770, CUDA, Apple, Lunar Lake, CPU AVX2, server, residency,
speed, or broad model-family claims unless the swarm proof manifest explicitly
allows them.

## Excluded Work

Promotion PRs must say what is not included. Examples:

- draft or proof-gated PRs;
- diagnostic-only receipts;
- old PRs without content disposition;
- hardware lanes that have not passed quality gates;
- performance candidates without behavior-backed benchmarks.

Excluded work should stay in `bitnet-rs-swarm` or remain linked to its original
PR/issue. Do not close it in `BitNet-rs` merely because a release was promoted.

## Publish Authority

Tags, crates.io publication, release notes, signed artifacts when present, and
stable release branches belong to `BitNet-rs`.

Normal development belongs to `bitnet-rs-swarm`.
