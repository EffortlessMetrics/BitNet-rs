# Phase 1 Instrumentation Plan (BitNet-rs)

This plan translates the "instrument rack" into a concrete, low-noise setup for this repository.

## Chosen publication pattern

Use **Pattern 3**:
- **PRs:** upload generated outputs as CI artifacts only.
- **Main/scheduled/release:** commit deterministic snapshots to the repository.

Why this pattern here:
- Keeps PRs readable (no generated-file churn).
- Still gives maintainers reproducible snapshots on default branch and at release boundaries.
- Matches existing heavy workflow usage in `.github/workflows/` and avoids adding review overhead.

## Directory layout for generated outputs

Keep all deterministic outputs in one namespace:

- `docs/generated/repo/diagram.svg`
- `docs/generated/deps/workspace.svg`
- `docs/generated/api/rust/<crate>.public-api.txt`
- `docs/generated/security/sbom.cdx.json`
- `docs/generated/security/vuln-audit.json`
- `docs/generated/perf/bench-summary.md`
- `docs/generated/perf/binary-size.md`
- `docs/generated/ops/churn-30d.md`

This avoids scattering generated content under multiple docs trees.

## Workflow map (Phase 1 only)

## 1) Structure fingerprint diagram

- **Workflow file:** `.github/workflows/structure-fingerprint.yml`
- **Trigger:**
  - `push` on `main`
  - `schedule` nightly
  - `workflow_dispatch`
- **Output:** `docs/generated/repo/diagram.svg`
- **PR behavior:** artifact only
- **main/schedule behavior:** commit only when file hash changed

## 2) Workspace dependency graph

- **Workflow file:** `.github/workflows/deps-graph.yml`
- **Trigger:**
  - `pull_request` with paths `**/Cargo.toml`, `Cargo.lock`
  - `push` on `main` with same path filters
- **Output:** `docs/generated/deps/workspace.svg`
- **PR behavior:** artifact only
- **main behavior:** commit on change

## 6) Public API snapshot + diff

- **Workflow file:** `.github/workflows/public-api.yml`
- **Trigger:**
  - `pull_request` with paths `crates/**`, `Cargo.toml`, `Cargo.lock`
  - `release` (`published`)
- **Tracked crates (initial):** `bitnet-common`, `bitnet-inference`, `bitnet-cli`, `bitnet-ffi`, `bitnet-kernels`
- **Outputs:** `docs/generated/api/rust/*.public-api.txt`
- **PR behavior:**
  - generate snapshots in temp dir
  - diff against `docs/api/rust/*.public-api.txt` baseline
  - fail on unacknowledged breaking changes
  - upload diff artifact
- **release behavior:** commit fresh snapshots

## 21–23) SBOM + license + vulnerability scans

- **Workflow file:** `.github/workflows/supply-chain-phase1.yml`
- **Trigger:**
  - `pull_request` (best-effort, non-blocking for transient network failures)
  - `schedule` daily
  - `release`
- **Outputs:**
  - `docs/generated/security/sbom.cdx.json`
  - `docs/generated/security/vuln-audit.json`
  - license report artifact
- **PR behavior:** artifacts + failing checks on policy violations
- **schedule/release behavior:** commit SBOM + vulnerability snapshot

## 26–27) Perf summary + binary size report

- **Workflow file:** `.github/workflows/perf-phase1.yml`
- **Trigger:**
  - `pull_request` with label `perf`
  - `schedule` nightly
  - `push` on `main` with paths `crates/**`, `Cargo.lock`
- **Outputs:**
  - `docs/generated/perf/bench-summary.md`
  - `docs/generated/perf/binary-size.md`
- **PR behavior:** artifact only (plus check summary)
- **main/schedule behavior:** commit on change

## 30) Churn/hotspot report

- **Workflow file:** `.github/workflows/churn-report.yml`
- **Trigger:** weekly schedule
- **Output:** `docs/generated/ops/churn-30d.md`
- **Behavior:** commit on schedule (artifact also optional)

## 33) CODEOWNERS review routing

- Keep `.github/CODEOWNERS` required and up to date.
- Enforce branch protection requiring relevant CODEOWNER review.

## 36) Automated changelog

- **Workflow file:** `.github/workflows/release-notes.yml`
- **Trigger:** `release` events
- **Output:** `CHANGELOG.md` update + release notes body
- **Behavior:** commit only on release cut (not on every merge)

## Anti-spam rules (mandatory)

All Phase 1 workflows should include the following controls:

1. **Concurrency cancellation**
   ```yaml
   concurrency:
     group: ${{ github.workflow }}-${{ github.ref }}
     cancel-in-progress: true
   ```
2. **Path filters** so expensive jobs only run on relevant file changes.
3. **Commit only if content changed** using hash comparison (`git diff --quiet` check).
4. **Single bot identity** for generated commits (`bitnet-ci[bot]`).
5. **Generated commit marker** in message, e.g. `chore(ci): refresh generated artifacts [skip ci]`.
6. **Workflow-level guard** to ignore bot-generated commit loops (`if: github.actor != 'bitnet-ci[bot]'`).
7. **Retention policy** for PR artifacts (7–14 days) to control storage costs.

## Suggested rollout sequence

1. Add `public-api.yml` and `supply-chain-phase1.yml` first (highest governance value).
2. Add `deps-graph.yml` and `structure-fingerprint.yml` next.
3. Add `perf-phase1.yml` and `churn-report.yml` last.

This yields Phase 1 value without overwhelming maintainers with CI churn.
