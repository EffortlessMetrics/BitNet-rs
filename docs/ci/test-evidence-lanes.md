# Test Evidence Lanes

This document maps the Rust 1.95 / next minor rollout evidence stack to CI
lanes. It exists to keep ordinary PR validation cheap without weakening release
proof.

## Doctrine

```text
ripr is static mutation-exposure analysis.

It catches much of the same signal mutation testing catches -- weak test/oracle
exposure -- but earlier and cheaper, because it runs statically and can run
per PR.

Mutation testing remains the runtime empirical backstop, especially for
nightly and release readiness. The CI design should use ripr to shift
mutation signal left, not to pretend mutation is unnecessary.
```

## Lane Map

| Lane | Evidence | Cost tier | Trigger |
|---|---|---:|---|
| Default PR | fmt, check, clippy, tests, policy checks, `ripr` static mutation-exposure analysis | low | every relevant PR |
| Risk PR | default PR evidence plus targeted mutation for touched high-risk owner surfaces | medium-high | path, owner, or label |
| Nightly | broader mutation matrix, deeper coverage, dogfood/report drift | high | schedule |
| Release | publish dry-run, package list, policy status, no-panic status, Clippy status, file-policy status, `ripr` status, mutation readiness | high | release branch/tag |

## Default PR

Default PR lanes should answer:

```text
Did this change plausibly break the changed crate, its direct dependents, or
the canonical CPU/default-member path?
```

Default PR evidence includes:

- CI Core build/test/doc lanes,
- Clippy on the canonical CPU surface,
- strict policy checks,
- file-policy and lint-inheritance checks,
- no-panic and Clippy exception checks,
- `ripr` static mutation-exposure analysis for relevant Rust diffs.

Default PR evidence does not include broad mutation, broad hardware validation,
or full coverage by default.

## Risk PR

Risk PRs add targeted runtime confirmation when the diff touches high-risk
owner surfaces:

- kernel math,
- quantization format correctness,
- tokenizer/model compatibility,
- FFI and ABI boundaries,
- GPU backends and shader surfaces,
- authentication/request routing,
- policy checker logic,
- release/package metadata.

Targeted mutation should be scoped to the touched owner surface. It should not
be promoted into the default PR lane just because a narrow risk PR needed it.

## Nightly

Nightly lanes are the place for broad runtime confirmation:

- broader mutation matrix,
- deeper coverage,
- long-running property/fuzz surfaces,
- cross-validation and hardware jobs when scheduled resources are available,
- report and policy drift detection.

Nightly failures must produce actionable reports. They should not silently
pressure ordinary PR lanes to become broader.

## Release

Release readiness must be clean enough to ship:

- package and publish dry-run status,
- MSRV/toolchain status,
- policy-report status,
- no-panic baseline/no-new-debt status,
- Clippy ratchet status,
- file-policy status,
- `ripr` status,
- mutation status,
- GPU/FFI claim boundary,
- known non-blockers,
- tag procedure,
- rollback path.

## Skipped Lanes

Skipped lanes must report that they were skipped by policy. They must not be
hidden as passed proof.

Use explicit language:

```text
skipped-by-policy: docs-only diff
skipped-by-policy: no GPU-owned paths changed
skipped-by-policy: mutation reserved for nightly unless high-risk owner surface changed
```

Do not use language that implies validation ran when it did not.
