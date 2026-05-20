# CI Workflow Inventory

A flat listing of GitHub Actions workflows in `.github/workflows/` and
how they map (or don't yet map) to the CI lane whitelist
(`policy/ci-lane-whitelist.toml`).

This document is human-curated; the authoritative machine-readable
view is the whitelist itself, validated by `xtask ci-lane-whitelist check`.

## Default-PR workflows

| Workflow                          | Whitelist lane(s)                                  | Notes                              |
| --------------------------------- | -------------------------------------------------- | ---------------------------------- |
| `pr-plan.yml`                     | `pr-plan`                                          | Visibility-only                    |
| `ci-core.yml`                     | `ci-core-build-test`, `ci-core-clippy`, `ci-core-docs`, `bdd-grid-check` | BDD grid runs only when relevant   |
| `feature-matrix.yml`              | `feature-matrix-pr`, `feature-matrix-full-cli`, `feature-matrix-full` | Full CLI and full matrix are risk-routed |
| `compatibility.yml`               | `compatibility-msrv`, `compatibility-ffi-abi`      | MSRV path-gated; FFI label-gated   |

## Risk-gated and expensive workflows

| Workflow                          | Whitelist lane(s)                                  | Trigger model                       |
| --------------------------------- | -------------------------------------------------- | ----------------------------------- |
| `gpu-ci-matrix.yml`               | `gpu-native`, `gpu-docker`                          | Path-gated on `main`; PR lanes emit by labels |
| `apple-silicon.yml`, `macos-arm64.yml` | `macos-arm64-clippy`                            | Labels + scheduled                  |
| `coverage.yml`                    | (deep, not yet whitelisted)                         | Main / nightly                      |
| `crossval.yml`                    | (deep, not yet whitelisted)                         | Labels                              |
| `property-tests.yml`              | (deep, not yet whitelisted)                         | Labels + scheduled                  |
| `fuzz-ci.yml`                     | (deep, not yet whitelisted)                         | PR build + scheduled matrix         |
| `model-gates.yml`, `validation.yml`, `gguf_build_and_validate.yml` | (deep) | Labels                              |
| `intel-gpu-*.yml`, `rocm-smoke.yml`, `gpu-smoke.yml`, `gpu.yml`, `a770-nightly.yml` | (gpu deep) | Labels / scheduled |
| `tl-lut-nightly.yml`, `tl-lut-stress.yml`, `quant-matrix.yml` | (deep)         | Scheduled                           |
| `guards.yml`, `guards-nightly.yml` | (policy)                                            | Repo guards                         |
| `security.yml`, `verify-receipts.yml` | (policy/release)                                 | Receipts / advisories               |
| `perf-gate.yml`, `performance-tracking.yml`, `phase1-maintenance.yml` | (deep) | Scheduled / opt-in       |
| `release.yml`                     | (release)                                           | Tag-driven                          |
| `repin-actions.yml`               | (meta)                                              | Scheduled                           |
| `link-check.yml`                  | (docs/policy)                                       | Scheduled                           |
| `markdownlint.yml`, `shellcheck.yml`, `patch-policy-check.yml` | (policy)              | Default PR / hooks                  |
| `docs-automation.yml`             | (docs)                                              | Default PR (docs paths)             |
| `campaign-tracker.yml`            | (docs)                                              | Default PR (campaigns)              |
| `pr-size-guard.yml`               | (control)                                           | Default PR                          |
| `contracts.yml`, `test-framework.yml` | (policy)                                        | Default PR                          |
| `cache-bitnet-cpp.yml`, `rust-ci-image.yml` | (infra)                                       | Scheduled                           |

## New workflows added by this rollout

| Workflow                          | Whitelist lane                                     | Status                              |
| --------------------------------- | -------------------------------------------------- | ----------------------------------- |
| `policy.yml` (PR 02 / 03)         | `strict-policy`                                     | Frontdoor blocking, very cheap      |
| `ripr.yml` (PR 13)                | `ripr-advisory`                                     | Frontdoor advisory                  |

## Backlog

The whitelist intentionally covers the highest-signal lanes first.
Workflows in the "deep / not yet whitelisted" rows above will be
added in follow-up PRs once their proof obligation, owner, and cost
are pinned down. Any lane not in the whitelist is treated as
opt-in (labels / scheduled), never required for ordinary PRs.
