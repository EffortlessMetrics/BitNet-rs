# Promote To BitNet-rs

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
Policy impact: defines swarm-to-release handoff

## Purpose

This document defines how selected work moves from
`EffortlessMetrics/bitnet-rs-swarm` to the public source-of-truth and release
repository, `EffortlessMetrics/BitNet-rs`.

During migration, swarm owns high-throughput same-repo execution and proof
work. `BitNet-rs` remains the public source-of-truth until an explicit
sync/cutover says otherwise, and it owns release history, tags, crates.io
publication, release notes, stable release branches, package metadata, and
release-blocking hotfixes.

Swarm-only commits do not become public-release authority until a promotion or
sync PR names the included work, proof inputs, claim boundaries, and excluded
work.

## Merge Method

Promotion is a repository-boundary operation. It must preserve swarm ancestry in
the public source repo.

Do not squash swarm-to-source promotion PRs. A promotion must land by regular
merge commit or by an explicitly approved fast-forward/direct update that keeps
the promoted swarm commits reachable from `BitNet-rs/main`.

Recommended branch shape:

```bash
git clone git@github.com:EffortlessMetrics/BitNet-rs.git bitnet-rs-promote
cd bitnet-rs-promote
git remote add swarm git@github.com:EffortlessMetrics/bitnet-rs-swarm.git
git fetch origin --prune
git fetch swarm --prune
git switch -c promote/swarm-YYYY-MM-DD origin/main
git merge --no-ff swarm/main -m "promote: merge bitnet-rs-swarm through <swarm_sha>"
```

The source repo may either temporarily/permanently allow merge commits for
`promote:swarm-to-source` PRs, or an admin may perform a non-force
fast-forward/direct update after proof. Force-push is not an accepted promotion
method.

## Promotion Inputs

Before opening a release-promotion PR against `BitNet-rs`, prepare:

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

The source commit should be on swarm `main` unless the release manager has
approved a narrower hotfix source.

## Promotion PR Body

The `BitNet-rs` promotion PR should include:

- merge method: regular merge commit or approved fast-forward/direct update;
- source swarm commit;
- included swarm PRs;
- release target and version;
- changelog summary;
- proof manifest or receipt pack;
- package/publish checks;
- excluded swarm work;
- explicit claim boundary.

For hardware, model, quality, performance, or residency claims, include the
exact receipt paths and claim ledgers that allow the claim. If a receipt is
missing, leave the claim out.

## Release Checks

Release PRs run package and publish gates instead of re-running every swarm
hardware lane by default.

Expected checks:

```text
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace --all-targets
cargo package --workspace --allow-dirty
cargo publish --dry-run, when applicable
```

If a command cannot run, record why, substitute evidence if available, and
whether the gap blocks release.

## Claim Boundary

Do not broaden claims during promotion.

The release PR must preserve:

- selected backend and device identity for hardware claims;
- model and tokenizer identity for model claims;
- fallback status;
- behavior or quality gate;
- benchmark profile for performance claims;
- explicit not-claims;
- excluded work.

Diagnostic-only swarm receipts remain diagnostic unless the promotion proof pack
makes them claim-grade.

## Source-Owned Surfaces

Do not casually overwrite source-owned release surfaces during promotion:

| File type | Promotion handling |
| --- | --- |
| Runtime/code/docs/specs/tests | Promote when included in the promotion scope. |
| Receipts intended for source | Promote when policy-compliant and claim-bounded. |
| Swarm-only CI | Usually exclude. |
| Swarm authority docs | Exclude or keep only if useful to source readers. |
| Release/signing/publish workflows | Source-owned; do not overwrite casually. |
| Secrets-heavy workflow changes | Source-owned; hold for explicit release/security review. |

## Excluded Work Examples

Promotion PRs should list work that remains in swarm:

- draft or proof-gated PRs;
- diagnostic-only receipts;
- old PRs without content disposition;
- hardware lanes without quality gates;
- performance candidates without behavior-backed benchmarks;
- broad model-family expansion not covered by the release.

Excluded work remains open or tracked in swarm. Do not close it in `BitNet-rs`
because a release was promoted.
