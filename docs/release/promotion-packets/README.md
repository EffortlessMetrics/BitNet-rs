# Promotion Packets

Status: active
Owner: release maintainers
Created: 2026-05-22
Linked proposal: n/a
Linked specs: n/a
Linked ADRs: n/a
Linked plan: `docs/release/PROMOTE_TO_BITNET_RS.md`
Linked issues: n/a
Linked PRs: n/a
Support-tier impact: release promotion only
Policy impact: durable home for reviewed swarm-to-source promotion packets

## Purpose

This directory holds committed promotion packets when a swarm-to-source
promotion needs a durable review artifact.

Draft packets can be generated under `target/promotion/` while operators are
choosing the batch boundary. Commit a packet here only when the promotion itself
needs repository-visible evidence that names the selected swarm range, included
PRs, proof commands, claim boundary, source impact, release impact, generated
artifacts, excluded work, and rollback.

The canonical promotion contract is:

```text
docs/release/PROMOTE_TO_BITNET_RS.md
```

This directory does not replace that contract. It stores packet instances that
follow it.

## Packet Naming

Use stable, reviewable filenames:

```text
YYYY-MM-DD-<short-range-or-lane>.md
```

Examples:

```text
2026-05-22-post370-control-plane.md
2026-05-22-slm-cpu-receipts.md
```

## Required Boundary

A committed packet must say what it promotes and what it does not promote. In
particular, it must not imply hardware, model, quality, performance, server,
residency, release, publish, or signing claims without exact receipts and
source-review acceptance.

If the packet is only a dry run, keep it under `target/promotion/` instead of
this directory.
