# CI Coverage Campaign

Campaign ID: `ci-coverage`

Status: active

## Objective

Make coverage upload and reporting reliable without turning forked PRs or missing secrets into failing unrelated work.

## End State

- Coverage upload handles trusted and forked PR contexts explicitly.
- Duplicate Codecov PRs are normalized behind one canonical path.
- CI reports make skipped coverage reasons visible.

## Hard Constraints

- Do not block unrelated runtime or tracker work on optional coverage uploads.
- Do not leak or assume Codecov secrets in forked PRs.
- Do not conflate coverage wiring with test quality claims.

## Work Items

| Work item | Status | Notes |
|---|---|---|
| CI-COVERAGE-001 | pr_open | Canonical Codecov upload guard is open in #3620. |

## Review Policy

Coverage PRs should remain CI-only and avoid touching runtime, kernels, loaders, or tracker campaign semantics.
