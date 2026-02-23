# Workspace Hygiene: 2024 Edition + Dependency Unification

**Date:** 2025-11-14
**Branch:** `chore/workspace-hygiene-2024-edition`
**Type:** Infrastructure / Cross-cutting

---

## Summary

This commit performs workspace-wide hygiene to prepare for Rust 2024 and centralize dependency control:

1. **Workspace dependency normalization**
   - Introduces `serde_yaml` in `[workspace.dependencies]` as an alias for `serde_yaml_ng = 0.10.0`
   - Moves common crates (serde, serde_json, anyhow, thiserror, chrono, clap, reqwest, toml, tempfile, once_cell, etc.) to `workspace = true` in all members
   - Ensures all crates pull versions and features from root `Cargo.toml`

2. **Edition alignment**
   - Switches member crates from ad-hoc `edition = "2024"` to `edition.workspace = true`
   - Keeps workspace on 2024 edition, avoiding drift and one-off configurations

3. **BitNet tools and trace fixes**
   - `st-merge-ln-f16`: fix `serialize_to_file` call to pass `Some(meta)` by value rather than `&Some(meta)`
   - `bitnet-trace`: adapt env var tests to 2024's stricter env safety rules:
     - Wrap `env::set_var` / `env::remove_var` in `unsafe` blocks
     - Document safety via `ENV_LOCK` to prevent concurrent env access in tests

---

## Motivation

- **Remove duplicate dependency declarations** and version mismatches across the workspace
- **Prepare the entire workspace for Rust 2024 edition** without surprises
- **Keep YAML parsing using the safer `serde_yaml_ng` implementation** while exposing a stable `serde_yaml` surface
- **Fix small tools/trace issues** that 2024 and stricter lints surfaced

---

## Changes

### Files Modified (19)

**Root:**
- `Cargo.toml` - Added `[workspace.dependencies]` for common crates
- `Cargo.lock` - Updated dependency graph

**Crates:**
- `crates/bitnet-cli/Cargo.toml`
- `crates/bitnet-compat/Cargo.toml`
- `crates/bitnet-kernels/Cargo.toml`
- `crates/bitnet-models/Cargo.toml`
- `crates/bitnet-py/Cargo.toml`
- `crates/bitnet-st-tools/Cargo.toml`
- `crates/bitnet-trace/Cargo.toml`
- `crossval/Cargo.toml`
- `fuzz/Cargo.toml`
- `tests/Cargo.toml`
- `tools/migrate-gen-config/Cargo.toml`
- `xtask-build-helper/Cargo.toml`
- `xtask/Cargo.toml`

**Source:**
- `crates/bitnet-st-tools/src/bin/st-merge-ln-f16.rs` - Fixed `serialize_to_file` call
- `crates/bitnet-trace/src/lib.rs` - Added `unsafe` block for `env::remove_var`
- `crates/bitnet-trace/tests/integration_test.rs` - Added `unsafe` blocks with safety docs

---

## Validation

Local validation (GitHub Actions currently unavailable):

### Build
```bash
cargo check --workspace --no-default-features --features cpu
```
**Result:** ✅ PASS

### Clippy (strict)
```bash
cargo clippy --workspace --no-default-features --features cpu --all-targets -- -D warnings
```
**Result:** ✅ PASS

### Format
```bash
cargo fmt --all
```
**Result:** ✅ PASS (no changes)

### Targeted Tests
```bash
cargo test -p bitnet-compat --lib
cargo test -p bitnet-trace --lib
cargo test -p bitnet-trace --test integration_test
```
**Result:** ✅ PASS

### Dependency Tree Check
```bash
cargo tree -i serde_yaml_ng
```
**Result:** ✅ Only pulled via workspace alias `serde_yaml`

---

## Risk Assessment

**Risk Level:** Low

**Rationale:**
- Changes are **mechanical** (dependency unification, edition normalization)
- No runtime behavior changes except:
  - `st-merge-ln-f16` fix (correct signature match)
  - `bitnet-trace` env safety (correct 2024 semantics)
- All changes validated locally with strict lints
- Workspace-level changes ensure consistency across all crates

---

## Next Steps

1. **When GitHub Actions returns:**
   - Let CI validate full test matrix (CPU + GPU lanes)
   - Check for any feature-gated paths not covered by local validation

2. **Merge independently:**
   - This PR can merge **before** KV pool stack (518-521)
   - No dependencies on feature work

3. **Follow-up (future):**
   - Consider moving more common deps to `[workspace.dependencies]` (tokio, futures, etc.)
   - Audit for any remaining `edition = "2024"` stragglers

---

## PR Description Template

```markdown
## What

Workspace-wide hygiene pass to prepare for Rust 2024 and centralize dependency control:

1. **Workspace dependency normalization**
   - Introduces `serde_yaml` alias for `serde_yaml_ng = 0.10.0`
   - Moves common crates to `workspace = true` pattern
   - Single source of truth for versions and features

2. **Edition alignment**
   - All member crates now use `edition.workspace = true`
   - Workspace edition set to 2024

3. **BitNet tools and trace fixes**
   - `st-merge-ln-f16`: fix serialize_to_file signature
   - `bitnet-trace`: adapt to 2024 env safety requirements

## Why

- Eliminate duplicate dependency declarations
- Prepare entire workspace for Rust 2024 without surprises
- Maintain safer `serde_yaml_ng` implementation behind stable interface

## Validation

Local validation (GitHub Actions unavailable):

- ✅ `cargo check --workspace --no-default-features --features cpu`
- ✅ `cargo clippy --workspace --no-default-features --features cpu --all-targets -- -D warnings`
- ✅ `cargo fmt --all`
- ✅ Targeted tests: bitnet-compat, bitnet-trace (lib + integration)

All commands run on updated workspace; zero new warnings or test failures.

## Risk

**Low** - Mechanical changes only; no runtime behavior modifications except necessary 2024 compatibility fixes.
```

---

## Commit

```
commit 11644a91 (HEAD -> chore/workspace-hygiene-2024-edition)
Author: Steven Zimmerman <git@effortlesssteven.com>
Date:   Fri Nov 14 22:06:13 2025 -0500

    chore: update dependencies across multiple crates and improve workspace configuration

    19 files changed, 163 insertions(+), 136 deletions(-)
```
