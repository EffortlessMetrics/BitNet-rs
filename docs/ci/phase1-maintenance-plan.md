# Phase 1 CI Instrumentation Plan (Deterministic Artifacts)

This plan translates the Phase 1 instrument rack into a concrete GitHub Actions layout for BitNet-rs.

## Publication Pattern

This repository should use **Pattern 3**:

- **PRs**: generate artifacts and upload them (no repository commits)
- **`main` + scheduled + release**: generate the same artifacts and commit updates under `docs/generated/`

Rationale:

- Keeps PR noise low while still preserving evidence.
- Keeps deterministic snapshots close to code after merge.
- Avoids endless workflow loops by combining `[skip ci]` commits with path ignores.

## Output Layout

All generated outputs live under `docs/generated/phase1/`:

- `repo-fingerprint.txt` (Item 1)
- `deps-workspace.json` (Item 2)
- `public-api-manifest.md` (Item 6)
- `sbom-cargo-metadata.json` (Item 21 baseline)
- `security-license-report.txt` (Items 22–23)
- `bench-inventory.md` (Item 26 baseline)
- `size-inventory.md` (Item 27 baseline)
- `churn-30d.md` (Item 30)
- `release-notes-preview.md` (Item 36 baseline)
- `status.json` (hashes + UTC generation timestamp)

## Workflow Files

### 1) `.github/workflows/phase1-maintenance.yml`

Single workflow with three trigger classes:

- `pull_request` (artifact-only)
- `push` to `main` (commit generated outputs)
- `schedule` weekly + `release` published (commit generated outputs)

Suggested path filters:

- `Cargo.toml`, `Cargo.lock`
- `crates/**`
- `src/**`
- `benches/**`, `benchmarks/**`
- `.github/workflows/**`
- `docs/api/rust/**`

### 2) Optional helper script: `scripts/ci/generate-phase1-artifacts.sh`

Encapsulates deterministic generation logic so local runs and CI runs match.

## Job Breakdown

### Job A: `phase1-generate`

Runs on all triggers.

- Checkout with `fetch-depth: 0` (required for churn report).
- Set up stable Rust.
- Generate deterministic artifacts (sorted output, stable formatting).
- Upload artifact bundle on PRs.

### Job B: `phase1-commit-generated`

Runs only on `main`, `schedule`, and `release`.

- Configures bot git identity.
- Commits `docs/generated/phase1/*` if changed.
- Commit message: `ci(phase1): refresh generated maintenance artifacts [skip ci]`.
- Pushes to same branch.

## Anti-Spam / No-Loop Rules

1. Add `paths-ignore` for `docs/generated/**` in this workflow.
2. Use `[skip ci]` in auto-commit messages.
3. Use `concurrency` with `cancel-in-progress: true`.
4. Keep PR mode artifact-only.
5. Keep expensive checks label-gated (`perf`, `coverage`) and do not run them unconditionally in this workflow.

## Mapping to Phase 1 List

- **1 Repo fingerprint** → `repo-fingerprint.txt`
- **2 Crate dependency graph** → `deps-workspace.json`
- **6 Public API snapshot/diff** → `public-api-manifest.md` + existing `contracts.yml`
- **21 SBOM** → `sbom-cargo-metadata.json` (upgrade path: CycloneDX)
- **22–23 License/vuln scan** → `security-license-report.txt` + existing security workflows
- **26 Microbench suite** → `bench-inventory.md` (baseline inventory, not benchmark execution)
- **27 Size/dependency bloat** → `size-inventory.md` baseline
- **30 Churn report** → `churn-30d.md`
- **33 CODEOWNERS routing** → existing `CODEOWNERS` (no generation required)
- **36 Changelog/release notes** → `release-notes-preview.md`

## Incremental Upgrade Path

- Replace SBOM baseline with CycloneDX generation once tool pinning is finalized.
- Replace benchmark inventory with selected `cargo bench` targets under `perf` label.
- Replace size baseline with built binary size deltas for `bitnet-cli` and key crates.
