# Landing PR #452: Complete Checklist

## ✅ What's Done

All the heavy lifting is complete:

1. **✅ Implementation complete**
   - `xtask verify-receipt` command with strict gates
   - GPU kernel pattern detection (8 patterns + explicit CPU exclusion)
   - Schema v1.0 validation (accepts "1.0.0" and "1.0")
   - String typing enforcement for `kernels[]`
   - Unit tests for GPU kernel detection
   - Portable test infrastructure (no `.git` dependency)
   - Shell portability (printf, proper exit codes)

2. **✅ Documentation updated**
   - `CONTRIBUTING.md` includes verify-receipt workflow
   - `scripts/local_gates.sh` ready with receipt verification step
   - Roadmap issues drafted (6 follow-up issues)

3. **✅ Testing complete**
   - Format check: `cargo fmt --all`
   - Clippy clean: `cargo clippy --all-targets --all-features`
   - Unit tests pass: `cargo test --workspace --no-default-features --features cpu`
   - GPU kernel detection test validates positive/negative cases

## 📋 Next Steps (Copy-Paste Ready)

### 1. Post PR Comment

Copy this into PR #452 as a comment:

```markdown
Incorporated review feedback & hardening:

* **GPU kernels:** broadened detection (added `i2s_(quantize|dequantize)`, `cublas_*`, `cutlass_*`, `tl1/tl2_gpu_*`; explicitly exclude `i2s_cpu_*`)
* **Kernels typing:** all `kernels[]` entries must be strings
* **Schema const:** `RECEIPT_SCHEMA` alias; verification accepts `"1.0.0"` or `"1.0"`
* **Portable tests:** `xtask` tests resolve workspace without `.git` via `CARGO_WORKSPACE_DIR` or `[workspace]` in `Cargo.toml`
* **Shell portability:** `printf` instead of `echo -e`; `exit "$FAILED"` in `local_gates.sh`
* **Deps:** `once_cell` + `regex` centralize GPU patterns
* **Tests:** added unit test for `is_gpu_kernel_id()` (positive/negative cases)
* **Docs:** `CONTRIBUTING.md` updated with `verify-receipt` workflow

All tests pass locally; `fmt`/`clippy` clean. **Ready to merge.** ✅
```

### 2. Merge PR #452

Use GitHub UI to merge (squash or merge commit). If CI is noisy or incomplete, use maintainer override - this PR only adds tooling/tests/docs, no runtime hot paths.

### 3. Wire CI Gates

**File to modify:** Your main CI workflow (e.g., `.github/workflows/ci.yml` or similar)

**Add CPU verification:**
```yaml
- name: Verify CPU receipt (strict)
  # Temporary: only run if receipt exists (until microbench lands)
  if: hashFiles('ci/inference.json') != ''
  run: |
    cargo run -p xtask -- verify-receipt --path ci/inference.json
```

**Add GPU verification (optional):**
```yaml
- name: Verify GPU receipt (requires GPU kernels)
  if: matrix.backend == 'cuda' || matrix.backend == 'gpu'
  run: |
    cargo run -p xtask -- verify-receipt --path ci/inference.json --require-gpu-kernels
```

**Location:** Add after your inference benchmark step (when it exists), before artifact uploads.

### 4. Enable Branch Protection

1. Navigate to: **Settings → Branches → Branch protection rules**
2. Select `main` branch
3. Enable: **Require status checks to pass before merging**
4. Add required jobs:
   - Find the job that runs `verify-receipt` (e.g., "test-cpu")
   - Mark it as required
5. Save changes

**Note:** Keep this required once the CPU microbench is implemented (next issue).

### 5. Create Follow-Up Issues

Create these GitHub issues in order (copy from `.github/issues/issue-*.md`):

1. **Enforce quantized hot-path** (issue-quantized-hotpath.md)
   - Priority: High
   - Effort: ~1 day
   - Quick win with high value

2. **CPU microbench + receipt** (issue-cpu-microbench.md)
   - Priority: High ⚠️ (Unblocks CI gate)
   - Effort: ~2 days
   - Required to remove temporary `if: hashFiles()` guard in CI

3. **GPU microbench** (issue-gpu-microbench.md)
   - Priority: Medium
   - Effort: ~1 day
   - Completes receipt infrastructure

4. **Cross-validation harness** (issue-crossval-harness.md)
   - Priority: Medium
   - Effort: ~2 days
   - Improves accuracy confidence

5. **Fingerprint exceptions** (issue-fingerprint-exceptions.md)
   - Priority: Low
   - Effort: ~1 day
   - Future-proofing for fast GPUs

6. **Validation shared crate** (issue-validation-shared-crate.md)
   - Priority: Medium
   - Effort: ~2 days
   - Code quality improvement

## 🎯 Success Criteria

You'll know everything is working when:

1. **PR #452 merged:** Comment posted, PR merged to `main`
2. **CI wired:** Receipt verification step in CI workflow
3. **Branch protection enabled:** `verify-receipt` job is required
4. **Issues created:** 6 follow-up issues in GitHub Issues
5. **Developer gates work:** `./scripts/local_gates.sh` runs successfully

## 🔍 Verification (Local)

Test the complete flow locally:

```bash
# 1. Run local gates
./scripts/local_gates.sh

# Expected output:
# [1/5] Running format check... ✓
# [2/5] Running clippy... ✓
# [3/5] Running CPU test suite... ✓
# [4/5] Running tiny benchmark... ℹ Skipping (not yet implemented)
# [5/5] Verifying receipt... ℹ Skipping (ci/inference.json not found)
# ✓ All local quality gates passed!

# 2. Test verify-receipt command directly
cargo run -p xtask -- verify-receipt --help

# Expected output:
# Verify inference receipt against strict quality gates
# Usage: xtask verify-receipt [OPTIONS]
# Options:
#   --path <PATH>              Path to receipt JSON [default: ci/inference.json]
#   --require-gpu-kernels      Require at least one GPU kernel
#   -h, --help                 Print help
```

## 📚 Documentation

All documentation is ready:

- `.github/PR_452_READY_TO_MERGE.md` - This file + PR comment text
- `.github/CI_INTEGRATION.md` - CI integration guide with examples
- `.github/NEXT_ROADMAP_ISSUES.md` - Original roadmap (comprehensive version)
- `.github/issues/*.md` - 6 ready-to-create GitHub issues
- `CONTRIBUTING.md` - Already updated with verify-receipt workflow

## ⚡ Quick Commands

```bash
# Post PR comment (manual - copy from above)
# Merge PR #452 (manual - use GitHub UI)

# Create CI patch (example for ci.yml)
# Add the verification steps to your workflow file

# Create GitHub issues (manual - copy from .github/issues/*.md)
# Create each issue with the content from the markdown files

# Test local gates
./scripts/local_gates.sh

# Test verify-receipt
cargo run -p xtask -- verify-receipt --help
```

## 🚀 Why This Matters

Once complete, you'll have:

- **Enforceable honesty:** Receipts can't fake GPU compute without GPU kernels
- **CI enforcement:** Branch protection prevents merging without valid receipts
- **Developer confidence:** Local gates match CI requirements
- **Clear roadmap:** 6 issues define the path to complete receipt infrastructure

The keystone gate (PR #452) is ready. Time to lock it in! 🎯
