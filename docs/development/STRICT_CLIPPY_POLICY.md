# Strict Clippy Policy

This document describes the Effortless Metrics shared Clippy policy as it is
being rolled out across BitNet-rs. The model is:

> **Global deny by default. Local exception by structured receipt.**

It is meant to be the *same* baseline applied across the Rust estate so that
every repository has the same panic-free production-and-test posture, the
same suppression governance, the same disclosed exception policy, and the
same upgrade ladder for new lints.

## Design

| Concern | Mechanism |
| --- | --- |
| Code shape | Workspace-level `[lints]` block in root `Cargo.toml` |
| Tooling thresholds | `clippy.toml` (no test carve-outs) |
| Active + future lints | `policy/clippy-lints.toml` |
| Repo-local debt | `policy/clippy-debt.toml` (added per PR-5 onwards) |
| Panic-family exceptions | `policy/no-panic-allowlist.toml` (added per PR-3) |
| Non-Rust file exceptions | `policy/non-rust-allowlist.toml` (added per PR-2) |
| Enforcement | `cargo xtask check-lint-policy` + Clippy CI lanes |

Clippy says *"the code shape is acceptable"*. The complementary `ripr`
evidence layer (added per PR-10) says *"the changed behavior has test
grip"*.

## Suppression style

Bare `#[allow(...)]` is denied at policy level. Every Clippy suppression
must use `#[expect(..., reason = "...")]` and the reason must reference the
backing policy entry where applicable.

```rust
#[expect(
    clippy::unwrap_used,
    reason = "Tracked in policy/no-panic-allowlist.toml: \
              fixture setup helper; expires 2026-07-01."
)]
fn temp_fixture_dir() -> std::path::PathBuf {
    // ...
}
```

If `expect` triggers `unfulfilled_lint_expectations`, that is a real signal
the underlying violation is gone and the suppression should be removed.

## Test posture

Tests are workspace surface. The standard is *workspace panic-free*, not
*production panic-free*. That means `clippy.toml` does **not** add:

```toml
allow-unwrap-in-tests = true       # forbidden
allow-expect-in-tests = true       # forbidden
allow-panic-in-tests = true        # forbidden
allow-indexing-slicing-in-tests = true  # forbidden
allow-dbg-in-tests = true          # forbidden
```

PR-6 adds shared fallible test helpers (`ensure`, `ensure_eq`,
`require_some`, `require_ok`) in `bitnet-test-support` so tests can return
`anyhow::Result<()>` instead of panicking on setup or assertions where
practical.

## Unsafe code

BitNet-rs has legitimate unsafe islands in FFI (`bitnet-sys`,
`bitnet-ggml-ffi`, `bitnet-ffi`), GPU backends (`bitnet-cuda`,
`bitnet-vulkan`, `bitnet-metal`, `bitnet-opencl`, `bitnet-rocm`,
`bitnet-wgpu`), and SIMD kernels. Therefore the workspace uses:

```toml
unsafe_code = "deny"
unsafe_op_in_unsafe_fn = "deny"
```

instead of `forbid`. Crates that legitimately need unsafe use crate-level
`#![expect(unsafe_code, reason = "...")]` or function-level
`#[expect(unsafe_code, reason = "...")]` with a justification, and every
unsafe block is `// SAFETY:` documented (enforced via
`undocumented_unsafe_blocks` once the strict profile is flipped).

## Rollout stages

The rollout is multi-PR by design. Tracking lives in
[`BITNET_POLICY_ROLLOUT.md`](./BITNET_POLICY_ROLLOUT.md). The current state
is summarized in that document. Each rollout PR is discrete and reversible.

## Upgrade ledger (Rust 1.94 / 1.95)

`policy/clippy-lints.toml` maintains a `[[planned]]` entry for every lint
that will be flipped on a future toolchain bump. The `xtask` policy check
rejects activating planned lints before `activate_when_msrv` is met. When
the MSRV is bumped, the policy command surfaces planned lints due for
activation.

## CI command posture

The strict baseline uses explicit `deny` levels. CI does **not** run
`-D warnings` while warn-stage numeric/good-taste lints are intentionally
staged. The blocking gate is:

```
cargo clippy --workspace --all-targets --no-default-features --features cpu
cargo xtask check-lint-policy
cargo xtask check-no-panic-family
cargo xtask check-file-policy
cargo xtask policy-report
```

When the planned 1.94/1.95 flips land and the warn ladder is cleared, the
posture upgrades to `-D warnings` per repo decision.
