<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# CI coverage Campaign Status

- Campaign: `ci-coverage`
- State: `active`
- Objective: Make coverage upload and reporting reliable without turning forked PRs or missing secrets into failing unrelated work.

## Work Items

| Item | State | PR | Branch | Review | Merge | Human gate | Acceptance |
|---|---|---:|---|---|---|---|---|
| CI-COVERAGE-001 | merged | #3620 | `codex/implement-minimal-codecov-integration-vm5aks` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | Guard Codecov upload so forked PRs and missing tokens skip coverage upload without failing unrelated CI. |
| CI-COVERAGE-002 | merged | #5775 | `codex/coverage-container-cleanup-3394` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | Run coverage as a rust-ci job container and remove hosted-runner disk cleanup and nested Docker execution from the coverage workflow. |

## Hard Constraints

- Do not block unrelated runtime or tracker work on optional coverage uploads.
- Do not leak or assume Codecov secrets in forked PRs.
- Do not conflate coverage wiring with test quality claims.
