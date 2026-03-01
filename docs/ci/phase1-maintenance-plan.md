# Phase 1 Maintenance Instrumentation Plan (BitNet-rs)

This document translates the proposed “instrument rack” into a concrete, low-noise rollout for BitNet-rs.

## Selected publication pattern

**Pattern 3: artifacts for PRs, commits only on scheduled/release runs.**

Rationale:
- Keeps PRs reviewable (evidence attached as artifacts, not generated-file churn).
- Preserves deterministic snapshots on a predictable cadence (nightly/weekly or release).
- Avoids CI loops from docs-only generated commits while still retaining historical outputs.

## Scope of this phase

Phase 1 priorities from the rack:
1. Repo fingerprint diagram.
2. Workspace crate dependency graph.
3. Public API snapshot (selected crates).
4. SBOM + vulnerability + license policy checks.
5. Curated microbench summary.
6. Binary size/bloat report.
7. Churn/hotspot report.
8. CODEOWNERS hygiene.
9. Automated changelog for release.

## Workflow layout

### 1) PR Evidence workflow
- **File:** `.github/workflows/phase1-pr-evidence.yml`
- **Triggers:**
  - `pull_request`
  - path filters for `Cargo.toml`, `Cargo.lock`, `crates/**`, `src/**`, `benches/**`, `.github/CODEOWNERS`, `docs/**`
- **Outputs (artifacts only):**
  - `reports/deps-crates.svg`
  - `reports/public_api/*.txt`
  - `reports/sbom.json`
  - `reports/licenses.txt`
  - `reports/vuln-audit.txt`
  - `reports/bench-summary.md` (label-gated with `perf`)
  - `reports/size.md`
  - `reports/churn.md`
- **Checks (required):**
  - license policy
  - vulnerability scan (best effort on PR)
  - dependency graph generation
  - API diff generation

### 2) Nightly/Scheduled snapshots workflow
- **File:** `.github/workflows/phase1-snapshots.yml`
- **Triggers:**
  - `schedule` (nightly for security + quick reports)
  - weekly schedule for heavier reports (bench + churn)
  - `workflow_dispatch`
- **Outputs (committed):**
  - `docs/diagram.svg`
  - `docs/deps-crates.svg`
  - `docs/public_api/*.txt`
  - `docs/sbom.json`
  - `docs/bench.md`
  - `docs/size.md`
  - `docs/churn.md`
- **Commit rules:**
  - commit message suffix `[skip ci]`
  - write only if a file hash changes
  - push with a dedicated bot identity

### 3) Release discipline workflow extensions
- **File:** integrate into existing `.github/workflows/release.yml`
- **Triggers:** release/tag flow already present.
- **Additions:**
  - generate changelog section from conventional commits/labels
  - attach `sbom.json` to release assets
  - persist release-time API snapshots for selected crates

## Anti-spam/anti-loop rules

Apply these across phase-1 workflows:
- Use `concurrency` per workflow+ref and `cancel-in-progress: true`.
- Use strict `paths`/`paths-ignore` filters.
- Avoid writing to repo on PR jobs.
- For commit-producing scheduled jobs:
  - skip commit when no diff (`git diff --quiet`),
  - use `[skip ci]`,
  - keep generated outputs in a fixed file set,
  - optionally gate on default branch only.

## Tooling recommendations (pinned where possible)

- Repo diagram: `repo-visualizer`.
- Crate graph: `cargo tree` + Graphviz pipeline.
- Public API: `cargo-public-api`.
- SBOM: `cargo cyclonedx`.
- Vulnerability: `cargo audit`.
- License policy: `cargo deny`.
- Bench summary: `cargo bench` (curated subset) + markdown summarizer.
- Size report: `cargo bloat`/`cargo llvm-lines` (choose one stable baseline).
- Churn report: `git log --since` scripts.

## Rollout sequence (safe order)

1. Start with PR artifacts only (`phase1-pr-evidence.yml`).
2. After two weeks of stable signal/noise, enable scheduled commit snapshots.
3. Extend release workflow with changelog + attached SBOM/API snapshots.

## Minimal acceptance criteria

- PRs show dependency graph + API diff artifacts when Rust or manifest files change.
- License and vulnerability checks are visible on every PR.
- A scheduled run updates snapshot docs only when content changes.
- Release assets include SBOM and changelog entries.

## Mapping to current repository

BitNet-rs already contains strong CI/security/perf coverage in `.github/workflows/`.
This plan intentionally layers deterministic generated artifacts on top of the existing checks rather than replacing current pipelines.
