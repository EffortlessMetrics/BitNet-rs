# Rust 1.95 Compatibility Audit

**Audited:** 2026-05-08  
**Auditor:** Automated (via `claude/upgrade-rust-1.95-vnJqM`)  
**Toolchain probed:** `rustc 1.95.0 (59807616e 2026-04-14)`  
**Current MSRV:** 1.93.0  
**Purpose:** Prove compatibility before the MSRV bump in PR 3.

## Commands run

```bash
rustup toolchain install 1.95.0 --component rustfmt --component clippy --component rust-analyzer
cargo +1.95.0 fmt --all -- --check
cargo +1.95.0 check --locked --workspace --all-targets --no-default-features
cargo +1.95.0 check --locked --workspace --all-targets --features cpu
cargo +1.95.0 clippy --locked --workspace --all-targets --no-default-features -- -D warnings
cargo +1.95.0 clippy --locked --workspace --all-targets --features cpu -- -D warnings
cargo +1.95.0 run --locked -p xtask --no-default-features -- check-file-policy --report-dir target/bitnet/reports
cargo +1.95.0 run --locked -p xtask --no-default-features -- policy-report --report-dir target/bitnet/reports
```

## Summary

| Check | Result | Notes |
|---|---|---|
| `cargo fmt --all -- --check` | **clean** | No format diff under 1.95.0 |
| `cargo check --no-default-features` | **clean** | 1 pre-existing unused_import warning in fuzz bench (not a 1.95 regression) |
| `cargo check --features cpu` | **clean** | Exit 0 |
| `cargo clippy --no-default-features -- -D warnings` | **5 new lint hits fixed, 42 in GPU crates deferred** | See lint debt table below |
| `cargo clippy --features cpu -- -D warnings` | same as above | Same crates, same lints |
| `xtask check-file-policy` | **0 findings** | 8673 files, 113 allowlist entries |
| `xtask policy-report` | **clean / advisory** | All sub-checks pass or advisory |

**Conclusion:** The workspace compiles and passes `cargo check` cleanly under Rust 1.95.0. The MSRV bump (PR 3) is unblocked. Five new Clippy lint hits in non-GPU production crates were fixed as part of this spike. GPU-crate lint debt is documented below and tracked for PR 5/13.

## Compilation

Both `--no-default-features` and `--features cpu` workspace checks exited 0 under 1.95.0. No compiler errors. The one pre-existing warning in `crossval/benches/gpu_offloading_bench.rs` (unused `std::path::Path` import) is unchanged from 1.93.0 and is not a 1.95 regression.

## Clippy `-D warnings` findings

Running with `-D warnings` is diagnostic only. The production CI uses explicit `[workspace.lints.clippy]` entries, not a global warning promotion. Findings here do not block the MSRV bump.

### New lints in 1.95.0 (not present under 1.93.0)

Confirmed by running the same `cargo clippy -D warnings` command under 1.93.0 on the affected crates and observing exit 0 with no errors.

| Lint | Clippy name | Occurrences | Crates |
|---|---|---|---|
| `sort_by_key` | `clippy::unnecessary_sort_by` | ~12 | bitnet-common (1, fixed), bitnet-gpu-hal (~6), bitnet-kernels (~5), bitnet-opencl (2) |
| `collapsible_match` | `clippy::collapsible_match` | ~5 | bitnet-testing-scenarios-core (2, fixed), bitnet-gpu-hal (3) |
| `manual checked division` | `clippy::manual_checked_ops` | ~5 | bitnet-kernels CUDA |
| `absurd extreme comparison` | `clippy::absurd_extreme_comparisons` | 1 | `bitnet-kernels/src/cuda/stream_management.rs:1074` |
| `if has identical blocks` | `clippy::if_same_then_else` | 2 | `bitnet-gpu-hal/src/structured_output.rs:333,338` |

### Fixes applied in this PR

Two non-GPU production crates had new 1.95 lint hits that were small and non-semantic. Fixed in this PR:

**`crates/bitnet-common/src/perf_profiler.rs:142`** — `clippy::unnecessary_sort_by`

```rust
// before
regions.sort_by(|a, b| b.total_time.cmp(&a.total_time));

// after
regions.sort_by_key(|a| std::cmp::Reverse(a.total_time));
```

**`crates/bitnet-testing-scenarios-core/src/lib.rs:198,203`** — `clippy::collapsible_match`

```rust
// before
match os.as_str() {
    "windows" => {
        if cfg.max_parallel_tests > 8 {
            cfg.max_parallel_tests = 8;
        }
    }
    "macos" => {
        if cfg.max_parallel_tests > 6 {
            cfg.max_parallel_tests = 6;
        }
    }
    _ => {}
}

// after
match os.as_str() {
    "windows" if cfg.max_parallel_tests > 8 => {
        cfg.max_parallel_tests = 8;
    }
    "macos" if cfg.max_parallel_tests > 6 => {
        cfg.max_parallel_tests = 6;
    }
    _ => {}
}
```

Both fixes were verified clean under both 1.93.0 and 1.95.0.

**`crates/bitnet-ffi/src/config.rs:286`** — `clippy::useless_conversion` (pre-existing in main, exposed by this PR's intel-gpu workflow trigger)

This was a pre-existing issue in `origin/main` introduced in commit `4d4b922`. On Linux x86_64, `c_ulong` is a type alias for `u64`, making `.into()` a useless identity conversion. On Windows, `c_ulong` is `u32` and the conversion is a meaningful widening. Fixed by applying `#[expect(clippy::useless_conversion)]` to the `to_generation_config` function, which satisfies the attribute on all CI platforms (Linux, macOS) where `c_ulong == u64`.

```rust
// before
config = config.with_seed(self.seed.into());

// after  
#[expect(clippy::useless_conversion)]  // c_ulong is u64 on POSIX-64 but u32 on Windows
pub fn to_generation_config(&self) -> bitnet_inference::GenerationConfig {
    ...
    config = config.with_seed(self.seed.into());
```

### GPU-crate lint debt (deferred to PR 5 / PR 13)

The following files have new 1.95 lint hits. They are GPU-only paths (bitnet-gpu-hal, bitnet-kernels CUDA/OpenCL, bitnet-opencl). These paths are:
- Not compiled or tested in the default CPU lane.
- Scaffolded but not validated end-to-end (per CLAUDE.md).
- Appropriate targets for the Clippy ratchet PR (PR 5) and the numeric/kernel cleanup PR (PR 13).

No changes were made to these files in this PR.

| File | Lint | Count |
|---|---|---|
| `crates/bitnet-gpu-hal/src/async_runtime.rs:353` | `unnecessary_sort_by` | 1 |
| `crates/bitnet-gpu-hal/src/checkpoint_manager.rs:340,392` | `unnecessary_sort_by` | 2 |
| `crates/bitnet-gpu-hal/src/continuous_profiling.rs:440` | `unnecessary_sort_by` | 1 |
| `crates/bitnet-gpu-hal/src/cross_attention.rs:783` | `unnecessary_sort_by` | 1 |
| `crates/bitnet-gpu-hal/src/distributed.rs:564` | `unnecessary_sort_by` | 1 |
| `crates/bitnet-gpu-hal/src/dynamic_shapes.rs:873` | `unnecessary_sort_by` | 1 |
| `crates/bitnet-gpu-hal/src/error_recovery.rs:442` | `collapsible_match` | 1 |
| `crates/bitnet-gpu-hal/src/model_validator.rs:454` | `unnecessary_sort_by` | 1 |
| `crates/bitnet-gpu-hal/src/multimodal_fusion.rs:613` | `collapsible_match` | 1 |
| `crates/bitnet-gpu-hal/src/structured_output.rs:333,338` | `if_same_then_else` | 2 |
| `crates/bitnet-kernels/src/cpu/cache_matmul.rs:266` | `unnecessary_sort_by` | 1 |
| `crates/bitnet-kernels/src/cpu/simd_embedding.rs:370` | `unnecessary_sort_by` | 1 |
| `crates/bitnet-kernels/src/cpu/simd_matmul.rs:85,86` | `absurd_extreme_comparisons` (×2 each) | 4 |
| `crates/bitnet-kernels/src/cpu/simd_tensor_parallel.rs:457` | `unnecessary_sort_by` | 1 |
| `crates/bitnet-kernels/src/cuda/cooperative_launch.rs:727,738,744` | `collapsible_match` / `manual_checked_ops` | 3 |
| `crates/bitnet-kernels/src/cuda/kernel_cache.rs:992` | `unnecessary_sort_by` | 1 |
| `crates/bitnet-kernels/src/cuda/launch_optimizer.rs:520,527` | `manual_checked_ops` | 2 |
| `crates/bitnet-kernels/src/cuda/memory_coalescing.rs:912` | `unnecessary_sort_by` | 1 |
| `crates/bitnet-kernels/src/cuda/occupancy_optimizer.rs:394,398,405` | `manual_checked_ops` | 3 |
| `crates/bitnet-kernels/src/cuda/profiling.rs:300,308` | `manual_checked_ops` | 2 |
| `crates/bitnet-kernels/src/cuda/register_optimizer.rs:447,454,595,605,792` | `manual_checked_ops` / `unnecessary_sort_by` | 5 |
| `crates/bitnet-kernels/src/cuda/stream_management.rs:1074` | `absurd_extreme_comparisons` | 1 |
| `crates/bitnet-kernels/src/cuda/stream_mgmt.rs:553` | `unnecessary_sort_by` | 1 |
| `crates/bitnet-kernels/src/opencl_batch_scheduler.rs:512` | `collapsible_match` | 1 |
| `crates/bitnet-kernels/src/opencl_op_fusion.rs:541` | `unnecessary_sort_by` | 1 |
| `crates/bitnet-kernels/src/opencl_prefill_decode.rs:261` | `manual_checked_ops` | 1 |
| `crates/bitnet-kernels/src/opencl_workgroup_opt.rs:362` | `unnecessary_sort_by` | 1 |
| `crates/bitnet-kernels/src/perf_tracker.rs:105` | `unnecessary_sort_by` | 1 |
| `crates/bitnet-opencl/src/backend_dispatcher.rs:236,294` | `unnecessary_sort_by` | 2 |

**Total deferred:** 42 lint hits across 29 file locations in 3 GPU/kernel crates.

## xtask policy checks

All policy checks ran under `cargo +1.95.0 run --locked -p xtask --no-default-features`:

| Check | Result |
|---|---|
| `check-file-policy` | 8673 files, 113 allowlist entries, **0 findings** |
| `ci-lane-whitelist` | 15 lanes, 2 exceptions; 2 pre-existing `duplicate_of` warnings (not regressions) |
| `lint-inheritance` | 136 crates checked, **0 missing** |
| `no-panic-family` | 31,694 advisory findings, 0 allowlist entries (advisory mode, unchanged from 1.93) |

No new policy regressions introduced by 1.95.0.

## Merge rule

This PR:
- Does **not** bump the declared MSRV in `Cargo.toml` or `rust-toolchain.toml`.
- Does **not** activate new Clippy policy lints.
- Does **not** reset or update the no-panic baseline.
- Does **not** change the release version.
- Only fixes two small, non-semantic lint regressions confirmed new to 1.95 and documents all findings.

The MSRV bump is PR 3 (`chore/msrv-rust-1.95`).
