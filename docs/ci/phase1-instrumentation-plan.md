# Phase 1 CI Instrumentation Plan (Deterministic Artifacts)

This plan translates the Phase 1 "instrument rack" into concrete GitHub Actions layout for BitNet-rs with low-noise defaults.

## Publication Pattern

**Selected pattern: PR artifacts + scheduled/release commits.**

- **PRs**: generate evidence as workflow artifacts only (no direct docs commits).
- **`main`/scheduled/release**: commit stable generated outputs into `docs/generated/`.
- **Loop control**:
  - generated commits use `[skip ci]` in commit message.
  - CI workflows include `paths-ignore: ['docs/generated/**']` unless they validate generated content.
  - auto-commit job is protected with `if: github.actor != 'github-actions[bot]'`.

## Scope: Phase 1 items

Target list: **1, 2, 6, 21–23, 26–27, 30, 33, 36**.

## Workflow Layout

### 1) `ci-instrumentation-pr.yml` (PR evidence lane)

**Triggers**
- `pull_request` on key inputs:
  - `Cargo.toml`, `Cargo.lock`, `crates/**`, `src/**`, `.github/workflows/**`, `docs/**`
- `paths-ignore`: `docs/generated/**`

**Jobs**
- `repo-fingerprint` → upload `diagram.svg` (item 1)
- `crate-deps-graph` (guarded by Cargo file changes) → upload `deps-crates.svg` (item 2)
- `public-api-diff` (selected public crates) → upload text snapshots + diff summary (item 6)
- `sbom-preview` → upload `sbom.json` (item 21)
- `security-scan-preview` → upload advisories/license/vuln reports (items 22, 23)
- `perf-preview` (label `perf` or Cargo change) → upload `bench.json`, `size.txt` (items 26, 27)
- `churn-preview` → upload `churn.md` (item 30)

**Outputs**
- One artifact bundle per job + a single job summary section with links.

### 2) `ci-instrumentation-sync.yml` (main/scheduled materialization)

**Triggers**
- `push` to `main` (filtered to relevant source paths)
- `schedule` (nightly/weekly depending on artifact type)
- `workflow_dispatch`

**Jobs**
- Regenerate same outputs as PR lane, but write to:
  - `docs/generated/diagram.svg`
  - `docs/generated/deps-crates.svg`
  - `docs/generated/public-api/*.txt`
  - `docs/generated/sbom.json`
  - `docs/generated/security/{licenses.md,vulns.md}`
  - `docs/generated/perf/{bench.md,size.md}`
  - `docs/generated/churn.md`
- `changelog-dry` job for release-note draft data (item 36)
- commit step (`git diff --quiet || git commit -m "ci(generated): refresh deterministic artifacts [skip ci]"`)

### 3) `release-governance.yml` (release-only)

**Triggers**
- `release` (`published`)

**Jobs**
- Rebuild + attach SBOM (item 21)
- Re-run vulnerability/license reports and attach to release (items 22, 23)
- Persist public API snapshots for release tag (item 6)
- Generate changelog/release notes artifact and append to GH release body (item 36)

## Tooling map (recommended)

- Repo/crate graph: `repo-visualizer`, `cargo tree` + graph helper.
- Public API: `cargo public-api` (selected crates only).
- SBOM: `cargo cyclonedx`.
- Security/licenses: `cargo-deny`, `cargo-audit`.
- Perf/size: `criterion` summary + `cargo bloat`/`cargo llvm-lines`.
- Churn: small script over `git log --since`.
- CODEOWNERS routing (item 33): keep in existing policy lane; no generated output needed.

## Guardrails

- Pin all actions by full SHA.
- Use `--locked` on cargo commands.
- Use concurrency groups to cancel superseded PR runs.
- Restrict commit job permissions: `contents: write`, others `read`.
- Keep generated outputs deterministic:
  - stable sort order
  - fixed timestamps (or omit)
  - explicit locale (`LC_ALL=C`)

## Suggested rollout

1. Add PR evidence lane with artifacts only.
2. Add sync lane writing to `docs/generated/` on schedule + `main`.
3. Add release governance attachments.
4. Wire branch protection to require only stable, fast checks.

## Minimal acceptance criteria

- PR touching `Cargo.lock` produces dependency graph + security artifacts.
- Weekly schedule updates `docs/generated/sbom.json` and `docs/generated/churn.md`.
- Release contains attached SBOM + vulnerability report + generated notes.
- Generated commit does not recursively trigger full CI.
