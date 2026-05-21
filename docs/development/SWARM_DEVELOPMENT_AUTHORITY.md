# Swarm Development Authority

Status: active
Owner: swarm maintainers
Created: 2026-05-20
Linked proposal: n/a
Linked specs: n/a
Linked ADRs: n/a
Linked plan: `docs/release/PROMOTE_TO_BITNET_RS.md`
Linked issues: n/a
Linked PRs: n/a
Support-tier impact: development authority only
Policy impact: routes new active development to `bitnet-rs-swarm` while
  `BitNet-rs` remains source-of-truth until sync/cutover

## Purpose

`EffortlessMetrics/BitNet-rs` remains the public source-of-truth, release, and
publish repository until an explicit sync/cutover says otherwise.

`EffortlessMetrics/bitnet-rs-swarm` is the high-throughput same-repo
development and proof execution repository for BitNet-rs.

Use this repository for:

- feature PRs;
- hardware lanes;
- A770, CUDA, Apple, Lunar Lake, CPU AVX2, server, and model-family proof work;
- campaign manifests and generated dashboards;
- diagnostic and receipt evolution;
- CI economics and self-hosted-runner routing;
- agent-swarm branches and high-throughput PR processing.

`BitNet-rs` owns tags, crates.io publication, release notes, stable release
branches, package metadata, signed artifacts when present, emergency
release-blocking fixes, and the public source-of-truth stack until cutover.

## Migration Boundary

Swarm is not a replacement source repo yet. Work that lands here is execution
and proof input until a release-promotion or sync PR carries the selected
content back to `BitNet-rs`.

The machine-readable repository boundary ledger is:

```text
policy/repo-boundary.toml
```

It names the source-owned and swarm-owned surfaces, forbidden history
operations, release-workflow guard, self-hosted runner boundary, shared
surfaces, and promotion requirements. Agents should treat the ledger as the
compact policy entrypoint before changing repository-boundary behavior.

If swarm and source-repo state disagree before cutover, do not treat the
swarm-only state as public-release truth. Resolve the difference through an
explicit sync or promotion PR that names included commits, included PRs, proof
inputs, claim boundaries, and excluded work.

## Development Rule

New active lane work lands here first. Do not open normal feature, hardware,
performance, diagnostic, refactor, or campaign PRs in `BitNet-rs` unless
explicitly directed, or unless the PR is a source-repo promotion, sync, release,
publish, or emergency hotfix.

When work is ready for public release, promote it through a release-promotion PR
against `BitNet-rs` using the contract in:

```text
docs/release/PROMOTE_TO_BITNET_RS.md
```

## Merge Method Boundary

Normal swarm PRs into `bitnet-rs-swarm/main` use squash merge. This keeps
high-volume agent work linear and reviewable.

Repository-boundary PRs must preserve ancestry and must not be squash-merged:

| PR kind | Required landing method |
| --- | --- |
| Ordinary swarm development PR | Squash merge |
| Source-history repair | Regular merge commit or fast-forward/direct update preserving ancestry |
| Source-to-swarm sync while source remains active | Regular merge commit |
| Swarm-to-source promotion | Regular merge commit or fast-forward/direct update preserving ancestry |

Never squash source-history repair, source-to-swarm sync, or swarm-to-source
promotion PRs. A squash can copy the file tree while dropping the commit graph
needed for contributor attribution and promotion auditability.

Never force-push `main`. If a repository-boundary branch must be refreshed,
merge or fast-forward from the current default branch and preserve both sides of
the ancestry.

Use these labels to make the required merge method explicit:

```text
merge:squash
merge:regular
sync:source-to-swarm
promote:swarm-to-source
repo-boundary
swarm-only
source-only
pre-sync
```

If a PR has `sync:source-to-swarm` or `promote:swarm-to-source`, it must use
`merge:regular` or an explicitly approved fast-forward/direct update. All other
ordinary swarm PRs default to `merge:squash`.

## CI And Runner Boundary

Trusted same-repo PRs may use routed self-hosted CI. Public fork PRs must not
run self-hosted runner jobs.

Release, signing, publish, secrets-heavy workflows, full platform matrices,
GPU/model-cache lanes, and public-fork self-hosted paths stay out of this swarm
cutover unless a separate approved PR deliberately moves them.

## Proof And Claim Discipline

Swarm can carry diagnostic and proof-building work that is not release-ready.
That does not make the evidence public-release claim-grade.

Every promoted claim must still name:

- source PRs;
- source commit;
- receipt or proof manifest;
- model, tokenizer, backend, and route identity where relevant;
- fallback status;
- explicit not-claims;
- excluded swarm work.

Hardware and model-support claims remain bounded by their campaign manifests and
receipt ledgers. A diagnostic receipt in swarm does not become a release claim
until the release-promotion PR includes and validates it.

## PR Queue Rules

Do not reduce queue count by discarding useful work.

Closing a PR is valid only after content review proves one of:

- exact useful content already landed;
- exact useful content was clean-ported and successor landed;
- true duplicate of a named kept PR;
- historical-only diagnostic evidence captured in a committed ledger or report;
- explicit content rejection after review.

Old, behind, branch-chain, diagnostic-only, and needs-restack are not close
reasons.

## Lane Coordination

Every active PR must name its lane, campaign, work item, orchestrator, branch,
base main SHA, allowed paths, shared surfaces touched, and whether closeout is
required. See:

```text
docs/tracking/LANE_OWNERSHIP.md
```

The campaign manifest and append-only campaign events are authoritative.
GitHub labels mirror the manifest for queue visibility; they do not replace it.

Use lane-qualified branch names such as:

```text
codex/<lane>/<work-item>-<slug>
claude/<lane>/<work-item>-<slug>
droid/<lane>/<work-item>-<slug>
dependabot/<ecosystem>/<dependency>
```

Generated dashboards, `AGENTS.md`, `README.md`, `.github/**`, `xtask/**`, and
hardware routing files are shared surfaces. If generated dashboards conflict,
preserve both campaign-source changes and regenerate instead of overwriting
another lane's state.

Hardware and runtime lanes are non-stackable by default unless the campaign
manifest explicitly allows the overlap.

## Release Handoff

A release handoff starts only after:

- swarm main is green enough for the selected release target;
- generated campaign dashboards are current;
- the release candidate manifest names included and excluded work;
- proof packs and claim boundaries are explicit;
- a promotion branch is opened against `BitNet-rs`.

The release repo then runs package and publish gates. It should not re-run every
swarm hardware lane by default.
