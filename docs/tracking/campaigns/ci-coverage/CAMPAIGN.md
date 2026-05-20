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
| CI-COVERAGE-001 | merged | Canonical Codecov upload guard merged in #3620. |
| CI-COVERAGE-002 | pr_open | Move coverage onto the rust-ci job container and remove hosted-runner disk cleanup. |

## Review Policy

Coverage PRs should remain CI-only and avoid touching runtime, kernels, loaders, or tracker campaign semantics.
