<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# CI coverage Campaign Status

- Campaign: `ci-coverage`
- State: `active`
- Objective: Make coverage upload and reporting reliable without turning forked PRs or missing secrets into failing unrelated work.

## Work Items

| Item | State | PR | Branch | Acceptance |
|---|---|---:|---|---|
| CI-COVERAGE-001 | pr_open | #3620 | `codex/implement-minimal-codecov-integration-vm5aks` | Guard Codecov upload so forked PRs and missing tokens skip coverage upload without failing unrelated CI. |

## Hard Constraints

- Do not block unrelated runtime or tracker work on optional coverage uploads.
- Do not leak or assume Codecov secrets in forked PRs.
- Do not conflate coverage wiring with test quality claims.
