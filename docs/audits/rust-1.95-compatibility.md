# Rust 1.95 Compatibility Probe

Date: 2026-05-12

Branch: `probe/rust-1.95-compat`

Base: `origin/main` at `5a80644be448be96540f40df8b3479263bbf1fc0`

## Scope

This audit probes current `main` under Rust 1.95.0 before the declared
workspace MSRV, toolchain, Clippy MSRV, lint policy, or release version changes.

This PR intentionally does not change:

- `Cargo.toml`
- `rust-toolchain.toml`
- `clippy.toml`
- GitHub Actions workflows
- Rust source
- lint activation policy
- release version metadata

## Toolchain

Command:

```bash
rtk rustup toolchain install 1.95.0 --component rustfmt --component clippy --component rust-analyzer
```

Result: passed. The installed compiler reported Rust 1.95.0
(`59807616e 2026-04-14`).

## Local Environment Notes

The default Cargo cache was noisy on this Windows host because other Codex and
rust-analyzer processes were active. To avoid treating shared cache locks as
Rust 1.95 compatibility failures, the successful probe slices used isolated
paths:

```text
CARGO_HOME=E:\cargo-home\BitNet-rust-195-compat
CARGO_TARGET_DIR=E:\cargo-targets\BitNet-rust-195-compat-isolated
```

The main checkout on `D:` was not used for this branch. The probe ran from the
isolated worktree at `E:\Code\Rust\BitNet-rust-195-compat`.

## Results

| Command | Result | Notes |
| --- | --- | --- |
| `rtk cargo +1.95.0 fmt --all -- --check` | blocked | Windows command-line/path expansion failed with `The filename or extension is too long. (os error 206)`. This is the same local `cargo fmt --all` failure mode seen before this PR and is not a Rust source incompatibility. |
| `rtk cargo +1.95.0 check -p bitnet-common --locked --no-default-features` | passed | Completed with the isolated Cargo paths. |
| `rtk cargo +1.95.0 check -p bitnet-kernels --locked --lib --no-default-features --features cpu` | passed | Completed with the isolated Cargo paths. |
| `rtk cargo +1.95.0 check --locked --workspace --all-targets --no-default-features` | blocked locally | No Rust 1.95 source diagnostic was observed before the Windows native build path stalled. A direct `rtk rustup run 1.95.0 cargo check --locked --workspace --all-targets --no-default-features` run exceeded 2 hours while compiling `sentencepiece-sys`, `bitnet-py`, and related Python/SPM native dependencies. |
| `rtk cargo +1.95.0 check --locked --workspace --all-targets --features cpu` | not run | Deferred because the no-default full-workspace gate did not clear the native Python/SPM build precondition. |
| `rtk cargo +1.95.0 clippy --locked --workspace --all-targets --no-default-features -- -D warnings` | not run | Deferred behind the full-workspace check blocker. |
| `rtk cargo +1.95.0 clippy --locked --workspace --all-targets --features cpu -- -D warnings` | not run | Deferred behind the full-workspace check blocker. |
| `rtk cargo +1.95.0 run --locked -p xtask --no-default-features -- policy-report --report-dir target/bitnet/reports` | not run | Deferred because `xtask` depends on `bitnet-tokenizers` with the `spm` feature, which uses the same `sentencepiece-sys` native path. |

## Findings

No completed probe slice exposed a Rust 1.95 language, standard library, or
dependency compatibility failure.

The remaining blocker is validation infrastructure, not a confirmed code
compatibility defect: full `--workspace --all-targets` validation on this
Windows host pulls in the native SentencePiece/Python binding path and did not
complete locally within the available runtime. The next MSRV bump PR must not
claim the full Rust 1.95 matrix is green until the full gates complete in CI or
on a host with the native Python/SPM build path known-good.

## Carry-Forward For PR 3

- Keep the PR 3 MSRV/toolchain bump separate from lint activation, release
  version changes, no-panic baselines, and Rust 1.95 API cleanup.
- Re-run the full required Rust 1.95 check/clippy matrix from a clean checkout
  with a known-good native SentencePiece/Python build environment.
- Treat the Windows `cargo fmt --all` path-length failure as an environment/tool
  invocation issue unless it reproduces as a rustfmt formatting diagnostic.
- Do not use this audit as evidence that the full workspace matrix is green.
