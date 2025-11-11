# CI Guardrails

BitNet.rs implements a comprehensive guardrail system to ensure CI reproducibility, supply chain security, and honest compute validation. This document explains the philosophy, enforcement mechanisms, and developer workflows.

## Philosophy

BitNet.rs CI guardrails are designed around three core principles:

1. **Reproducibility**: Same inputs → same outputs (deterministic builds, pinned dependencies)
2. **Supply Chain Security**: No floating dependencies (SHA pins, locked cargo builds)
3. **Honest Compute**: Verify inference correctness (receipt verification, kernel ID validation)

## Guardrail Categories

### 1. SHA Pin Enforcement

**Why**: Prevent supply chain attacks from compromised GitHub Actions.

**Rule**: All GitHub Actions must use **40-hex SHA commits**, not floating tags like `@v4`, `@main`, or `@stable`.

**Enforcement**: `.github/workflows/guards.yml` validates all action references match:
```regex
uses:\s+[a-zA-Z0-9/_-]+@[0-9a-f]{40}
```

**Example**:
```yaml
# ❌ BAD - floating tag
- uses: actions/checkout@v4

# ✅ GOOD - pinned SHA
- uses: actions/checkout@b4ffde65f46336ab88eb53be808477a3936bae11  # v4.1.1
```

**Developer Workflow**:
```bash
# Repin all actions to latest SHAs
gh workflow run repin-actions.yml

# Local preflight check
make guards
# or
rg 'uses:\s+[a-zA-Z0-9/_-]+@(?![0-9a-f]{40}\b)' .github/workflows/
```

**See also**: `.github/workflows/repin-actions.yml` for automated SHA updates.

---

### 2. MSRV Single-Source

**Why**: Prevent MSRV drift across workflows (e.g., one workflow tests 1.89.0, another 1.90.0).

**Rule**: Minimum Supported Rust Version (MSRV) is **1.89.0** (Rust 2024 edition), sourced exclusively from `rust-toolchain.toml`.

**Enforcement**: Guards workflow validates:
- No hardcoded `1.90.0` or other versions in workflows
- All workflows use `rust-toolchain.toml` via `toolchain-file: rust-toolchain.toml`

**Example**:
```yaml
# ❌ BAD - hardcoded version
- uses: dtolnay/rust-toolchain@...
  with:
    toolchain: 1.89.0

# ✅ GOOD - single-source from file
- uses: dtolnay/rust-toolchain@...
  with:
    toolchain-file: rust-toolchain.toml
```

**Update Process**:
1. Update `rust-toolchain.toml`:
   ```toml
   [toolchain]
   channel = "1.89.0"
   ```
2. All workflows automatically inherit new MSRV
3. No workflow YAML changes required

---

### 3. `--locked` Flag Enforcement

**Why**: Ensure exact dependency versions used in CI (prevents supply chain drift from Cargo.lock changes).

**Rule**: All `cargo` and `cross` commands must include `--locked` flag (77+ invocations across workflows).

**Enforcement**: Guards workflow validates:
```bash
rg '\b(cargo|cross)\s+(build|test|bench|run|install|check|clippy)\b(?!.*--locked)' .github/workflows/
```

**Example**:
```yaml
# ❌ BAD - unlocked build
- run: cargo build --features cpu

# ✅ GOOD - locked build
- run: cargo build --locked --features cpu
```

**Rationale**: Without `--locked`, Cargo resolves dependencies from scratch, potentially using newer (untested) versions.

---

### 4. Receipt Workflow Hygiene

**Why**: Python/WASM bindings require dynamic libraries (libpython, wasm32 toolchain) that complicate CI builds. Receipt verification should test **core Rust inference**, not FFI bindings.

**Rule**: Receipt verification workflow must exclude `bitnet-py` and `bitnet-wasm` packages in CPU+GPU build lanes.

**Enforcement**: Guards workflow validates `.github/workflows/verify-receipts.yml` contains:
```yaml
--exclude bitnet-py --exclude bitnet-wasm
```
at least **2 occurrences** (CPU lane + GPU lane).

**Example**:
```yaml
# CPU receipt build
- run: |
    cargo build --locked --release \
      --no-default-features --features cpu \
      --exclude bitnet-py --exclude bitnet-wasm

# GPU receipt build
- run: |
    cargo build --locked --release \
      --no-default-features --features gpu \
      --exclude bitnet-py --exclude bitnet-wasm
```

**See also**: PR template checklist includes receipt hygiene reminder.

---

### 5. Runner Pinning

**Why**: Ensure deterministic CI environment (prevents Ubuntu 24.04 breaking changes from affecting builds).

**Rule**: Single-platform jobs use `ubuntu-22.04` runner (not `ubuntu-latest`).

**Example**:
```yaml
# Job name stays "ubuntu-latest" for branch protection compatibility
build-test (ubuntu-latest):
  runs-on: ubuntu-22.04  # Explicit pin
```

**Update Process**: When upgrading Ubuntu version, update all `runs-on` fields simultaneously.

---

### 6. Template Path Triggers

**Why**: Ensure PR/issue template changes trigger required CI checks (prevents "Expected" status blocking merges).

**Rule**: Both `guards.yml` and `ci-core.yml` must trigger on:
- `.github/pull_request_template.md`
- `.github/ISSUE_TEMPLATE/**`

**Example**:
```yaml
on:
  pull_request:
    paths:
      - '.github/workflows/**'
      - '.github/pull_request_template.md'
      - '.github/ISSUE_TEMPLATE/**'
      # ... other paths
```

---

## Guards Workflow Architecture

### Structure

The `.github/workflows/guards.yml` workflow runs **8 validation checks** on every PR:

1. **No floating action refs** (informational)
2. **40-hex SHA pins** (blocking)
3. **MSRV consistency** (blocking)
4. **`--locked` everywhere** (blocking)
5. **Receipt workflow excludes** (blocking)
6. **Dev-only flags** (informational - warns about `BITNET_FIX_LN_SCALE`, etc.)
7. **Bare `cfg(cuda)` in tests** (informational - should use `cfg(any(feature="gpu", feature="cuda"))`)
8. **CODEOWNERS team validation** (informational)

### Check Types

- **Blocking** (`set -e`): Failure prevents merge
- **Informational** (`set +e`): Prints warnings but doesn't block

### Local Preflight

Run guards locally before pushing:

```bash
# Option 1: Makefile target
make guards

# Option 2: Manual checks
rg 'uses:\s+[a-zA-Z0-9/_-]+@(?![0-9a-f]{40}\b)' .github/workflows/
rg '\b(cargo|cross)\s+(build|test|bench)\b(?!.*--locked)' .github/workflows/
rg '1\.90\.0' .github/workflows/  # Should be empty (use 1.89.0 from toolchain file)
```

---

## Receipt Verification

Receipt verification ensures **honest compute** by validating inference execution traces.

### Receipt Schema (v1.0.0)

```json
{
  "schema_version": "1.0.0",
  "model_path": "models/model.gguf",
  "backend": "cpu",
  "compute_path": "real",  // MUST be "real", not "mock"
  "kernels": ["i2s_cpu_dequantize_avx2", "gemm_cpu_f32"],  // Non-empty
  "metrics": {
    "tokens_per_second": 45.2,
    "latency_ms": 22.1
  }
}
```

### Validation Gates

1. **Schema validation**: Must be v1.0.0
2. **Compute path**: `compute_path == "real"` (no mock inference)
3. **Kernel hygiene**:
   - Non-empty `kernels` array
   - No empty string kernel IDs
   - Kernel ID length ≤ 128 chars
   - Kernel count ≤ 10,000
4. **Auto-GPU enforcement**: `backend == "cuda"` requires GPU kernels (`gemm_*`, `i2s_gpu_*`)
5. **CPU quantization**: `backend == "cpu"` requires quantized kernels (`i2s_*`, `tl1_*`, `tl2_*`)

### Local Receipt Generation

```bash
# Generate receipt (writes to ci/inference.json)
cargo run -p xtask -- benchmark --model models/model.gguf --tokens 128

# Verify receipt
cargo run -p xtask -- verify-receipt

# Verify with GPU kernel requirement
cargo run -p xtask -- verify-receipt --require-gpu-kernels
```

### CI Integration

Receipt verification runs on:
- **Label-gated**: PR has `receipts` label
- **Always on**: `main` and `develop` branches

---

## Nightly Guardrail Sweep

The `.github/workflows/guards-nightly.yml` workflow runs daily to detect guardrail drift from external changes (e.g., GitHub Action updates breaking SHA pins).

**Runs**: Daily at 02:00 UTC
**Alerts**: GitHub Issues on failure

---

## Developer Workflows

### Before Pushing

```bash
# Local preflight (mirrors CI guards)
make guards

# Format + lint
cargo fmt --all && cargo clippy --all-targets --all-features -- -D warnings
```

### Updating GitHub Actions

```bash
# Trigger automated repin workflow
gh workflow run repin-actions.yml

# Wait for PR creation
gh pr list --label "automation"

# Review and merge
gh pr view <NUMBER> && gh pr merge <NUMBER> --squash
```

### Upgrading MSRV

```bash
# 1. Update rust-toolchain.toml
echo '[toolchain]\nchannel = "1.90.0"' > rust-toolchain.toml

# 2. Verify all workflows inherit new MSRV
rg 'toolchain-file: rust-toolchain.toml' .github/workflows/

# 3. Commit and push (no workflow YAML changes needed)
git add rust-toolchain.toml
git commit -m "chore: bump MSRV to 1.90.0"
```

### Adding New Workflow

New workflows must:
1. Use 40-hex SHA pins for actions
2. Use `toolchain-file: rust-toolchain.toml` for Rust toolchain
3. Include `--locked` on all `cargo`/`cross` commands
4. Exclude `bitnet-py` and `bitnet-wasm` in receipt builds (if applicable)

**Template**:
```yaml
name: My New Workflow
on: [pull_request]
jobs:
  build:
    runs-on: ubuntu-22.04
    steps:
      - uses: actions/checkout@b4ffde65f46336ab88eb53be808477a3936bae11  # v4.1.1
      - uses: dtolnay/rust-toolchain@d8352f6b1d2e870bc5716e7a6d9b65c4cc244a1a
        with:
          toolchain-file: rust-toolchain.toml
      - run: cargo build --locked --features cpu
```

---

## Troubleshooting

### "Expected — Waiting for status to be reported"

**Symptom**: PR blocked by missing check.
**Cause**: Workflow path filters don't include modified files.
**Fix**: Add file patterns to workflow `on.pull_request.paths`.

**Example**: Template-only PR missing Guards check → add `.github/pull_request_template.md` to Guards workflow paths.

### Guards Workflow Fails: "Non-40-hex SHA detected"

**Symptom**: Guards check fails with action reference violation.
**Cause**: Workflow uses `@v4`, `@main`, or `@stable` tag instead of SHA.
**Fix**:
```bash
# Find violations
rg 'uses:\s+[a-zA-Z0-9/_-]+@(?![0-9a-f]{40}\b)' .github/workflows/

# Repin to SHA
gh workflow run repin-actions.yml
```

### Guards Workflow Fails: "--locked missing"

**Symptom**: Guards check fails with cargo/cross unlocked build.
**Cause**: Workflow invokes `cargo build` without `--locked` flag.
**Fix**: Add `--locked` to all cargo/cross commands:
```yaml
- run: cargo build --locked --features cpu
```

### Receipt Verification Fails: "Missing GPU kernels"

**Symptom**: Receipt verification fails with "Expected GPU kernels for CUDA backend".
**Cause**: Receipt shows `backend: "cuda"` but kernels array lacks GPU kernel IDs.
**Fix**: Ensure GPU inference actually runs GPU kernels (check for silent CPU fallback):
```bash
# Verify GPU availability
cargo run -p xtask -- gpu-preflight

# Check receipt kernel IDs
jq '.kernels' ci/inference.json
```

### MSRV Drift Detected

**Symptom**: Guards workflow fails with "Found 1.90.0 in workflows".
**Cause**: Workflow hardcodes MSRV instead of using `rust-toolchain.toml`.
**Fix**:
```yaml
# Before (BAD)
- uses: dtolnay/rust-toolchain@...
  with:
    toolchain: 1.89.0

# After (GOOD)
- uses: dtolnay/rust-toolchain@...
  with:
    toolchain-file: rust-toolchain.toml
```

---

## References

- **Guards Workflow**: `.github/workflows/guards.yml`
- **Repin Workflow**: `.github/workflows/repin-actions.yml`
- **Receipt Verification**: `.github/workflows/verify-receipts.yml`
- **CI Labels**: `docs/ci/labels.md`
- **Receipt Schema**: `docs/tdd/receipts/schema-v1.0.0.md`
- **Nightly Guards**: `.github/workflows/guards-nightly.yml`

---

## Summary

BitNet.rs guardrails ensure:
- ✅ No floating dependencies (SHA pins, `--locked`)
- ✅ No MSRV drift (single-source from `rust-toolchain.toml`)
- ✅ Honest compute (receipt verification)
- ✅ Reproducible CI (runner pinning)
- ✅ Template hygiene (path triggers)

**Local preflight**: `make guards`
**Automated repin**: `gh workflow run repin-actions.yml`
**Receipt verification**: `cargo run -p xtask -- verify-receipt`
