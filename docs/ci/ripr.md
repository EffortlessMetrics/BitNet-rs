# ripr Static Mutation-Exposure Analysis

`ripr` is static mutation-exposure analysis.

It catches much of the same signal mutation testing catches -- weak test/oracle
exposure -- but earlier and cheaper, because it runs statically and can run per
PR.

Mutation testing remains the runtime empirical backstop, especially for nightly
and release readiness. The CI design uses `ripr` to shift mutation signal left,
not to pretend mutation is unnecessary.

## Evidence Stack

```text
Default PR:
  ripr + normal gates + policy checks

Risk PR:
  ripr + targeted mutation for touched high-risk owner surfaces

Nightly:
  broader mutation matrix

Release:
  mutation/readiness clean enough to ship
```

This is one verification stack with multiple cost tiers. `ripr` and mutation
testing are not unrelated parallel lanes.

## Current State

The repo already has:

- `.github/workflows/ripr.yml`,
- `ripr.toml`,
- `policy/ripr-suppressions.toml`,
- `ripr-advisory` in `policy/ci-lane-whitelist.toml`.

The current workflow may record a no-op when the `ripr` binary is not present
on the runner. That is acceptable for the Rust 1.93 control plane, but it is
not the Rust 1.95 target state.

## Rust 1.95 Target State

PR 11 (`ci/ripr-real-advisory`) provisions and runs `ripr` consistently:

```yaml
- name: Install ripr
  run: cargo install ripr --locked

- name: Run ripr doctor
  run: ripr doctor || true

- name: Run ripr check
  run: |
    mkdir -p target/ripr
    ripr check \
      --base "origin/${{ github.base_ref }}" \
      --json target/ripr/ripr.json \
      --sarif target/ripr/ripr.sarif \
      --markdown target/ripr/ripr.md \
      --config ripr.toml || true
```

The job remains advisory in this wave:

- it may annotate or summarize findings,
- it must upload JSON/SARIF/Markdown evidence when available,
- it must not become a branch-protection blocker yet,
- it must not be used as an excuse to remove mutation testing.

Use synchronize-only cancellation:

```yaml
cancel-in-progress: ${{ github.event_name == 'pull_request' && github.event.action == 'synchronize' }}
```

## Outputs

Expected artifacts:

```text
target/ripr/ripr.json
target/ripr/ripr.sarif
target/ripr/ripr.md
```

The PR summary should make clear whether `ripr` ran, skipped by policy, or was
unavailable. Skipped analysis must not be presented as a passed proof.

## Suppressions

Known acceptable findings belong in `policy/ripr-suppressions.toml`. Each
suppression must include:

```toml
[[suppress]]
id = "ripr-0001"
path = "crates/example/src/lib.rs"
finding = "reachable_unrevealed"
owner = "core/runtime"
reason = "Reachable only through dyn dispatch; covered by integration test foo."
expires = "2026-08-01"
```

Suppressions are receipts with owner, reason, and expiry. They are not
permanent allows.

## Review Questions

For every `ripr` finding:

1. Is changed behavior reachable from tests?
2. Is there a meaningful discriminator, not just execution?
3. If the static answer is unknown, is a targeted runtime check warranted?
4. If the surface is high risk, should targeted mutation run for that owner
   surface?
