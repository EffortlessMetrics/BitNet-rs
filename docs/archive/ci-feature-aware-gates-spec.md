# CI Feature-Aware Gates Specification

**Issue**: #447 (AC8)
**Status**: Ready for Implementation
**Priority**: CI Improvements
**Date**: 2025-10-11
**Affected Files**: `.github/workflows/` (CI configuration)

---

## Executive Summary

Add feature-aware exploratory CI gates to systematically validate `--all-features` compilation across BitNet.rs workspace. Maintain strict required gates for CPU-only baseline while allowing exploratory gates to fail until Issue #447 fixes are complete. This provides visibility into all-features compatibility without blocking development.

**Key Changes**:
- Maintain strict required gates: `--no-default-features --features cpu`
- Add exploratory jobs: `--all-features` (allowed to fail initially)
- Promote exploratory to required after AC1-AC7 validated
- Zero impact on existing CI pass/fail criteria

---

## Acceptance Criteria

### AC8: Add feature-aware exploratory CI gates
**Test Tag**: `// AC8: Feature-aware CI gates`

**Requirements**:
- Maintain strict required gates (must pass):
  - `cargo clippy --workspace --no-default-features --features cpu -- -D warnings`
  - `cargo test --workspace --no-default-features --features cpu`
- Add exploratory jobs (allowed to fail until AC1-AC7 complete):
  - `cargo clippy --workspace --all-features -- -D warnings`
  - `cargo test --workspace --all-features`
- Promote exploratory to required after all fixes validated

**Validation Command**:
```bash
# Required gates (must pass immediately)
cargo clippy --workspace --no-default-features --features cpu -- -D warnings
cargo test --workspace --no-default-features --features cpu

# Exploratory gates (allowed to fail until Issue #447 merges)
cargo clippy --workspace --all-features -- -D warnings
cargo test --workspace --all-features
```

**Expected Output**:
- Required gates: Pass immediately
- Exploratory gates: May fail until AC1-AC7 complete, then must pass

**Evidence**: Spec lines 41-44 show CI gate strategy with strict required baseline and exploratory all-features validation

---

## Technical Design

### CI Workflow Structure

#### Required Gates (Always Enforced)

**File**: `.github/workflows/ci.yml` (existing workflow)

**Current Configuration**:
```yaml
# Lines 98-100 (existing check)
- name: Check formatting
  if: matrix.os == 'ubuntu-latest'
  run: cargo fmt --all -- --check

# Add after line 100:
- name: Clippy (CPU baseline - required)
  if: matrix.os == 'ubuntu-latest'
  run: cargo clippy --workspace --all-targets --no-default-features --features cpu -- -D warnings

# Add after clippy:
- name: Test (CPU baseline - required)
  if: matrix.os == 'ubuntu-latest'
  run: cargo test --workspace --no-default-features --features cpu
```

**Status**: `continue-on-error: false` (default) - **Must pass for CI green**

---

#### Exploratory Gates (Allowed to Fail)

**New File**: `.github/workflows/all-features-exploratory.yml`

```yaml
name: All-Features Exploratory Validation

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main, develop]
  workflow_dispatch:

env:
  CARGO_TERM_COLOR: always
  RUST_BACKTRACE: 1

jobs:
  clippy-all-features:
    name: Clippy (All Features - Exploratory)
    runs-on: ubuntu-latest
    continue-on-error: true  # ← ALLOW FAILURE until Issue #447 merges

    steps:
      - uses: actions/checkout@v4

      - name: Install Rust
        uses: dtolnay/rust-toolchain@stable

      - name: Cache cargo
        uses: actions/cache@v3
        with:
          path: |
            ~/.cargo/registry
            ~/.cargo/git
            target
          key: ${{ runner.os }}-cargo-all-features-${{ hashFiles('**/Cargo.lock') }}

      - name: Clippy (All Features)
        run: |
          echo "::notice::This gate is exploratory and allowed to fail until Issue #447 merges"
          cargo clippy --workspace --all-targets --all-features -- -D warnings

  test-all-features:
    name: Test (All Features - Exploratory)
    runs-on: ubuntu-latest
    continue-on-error: true  # ← ALLOW FAILURE until Issue #447 merges

    steps:
      - uses: actions/checkout@v4

      - name: Install Rust
        uses: dtolnay/rust-toolchain@stable

      - name: Cache cargo
        uses: actions/cache@v3
        with:
          path: |
            ~/.cargo/registry
            ~/.cargo/git
            target
          key: ${{ runner.os }}-cargo-all-features-${{ hashFiles('**/Cargo.lock') }}

      - name: Test (All Features)
        run: |
          echo "::notice::This gate is exploratory and allowed to fail until Issue #447 merges"
          cargo test --workspace --all-features

  # Summary job to track promotion readiness
  exploratory-summary:
    name: Exploratory Gates Summary
    runs-on: ubuntu-latest
    needs: [clippy-all-features, test-all-features]
    if: always()

    steps:
      - name: Check exploratory gate status
        run: |
          echo "## All-Features Exploratory Gates Status" >> $GITHUB_STEP_SUMMARY
          echo "" >> $GITHUB_STEP_SUMMARY

          if [[ "${{ needs.clippy-all-features.result }}" == "success" ]]; then
            echo "✅ Clippy (all-features): PASS" >> $GITHUB_STEP_SUMMARY
          else
            echo "⚠️ Clippy (all-features): FAIL (exploratory - not blocking)" >> $GITHUB_STEP_SUMMARY
          fi

          if [[ "${{ needs.test-all-features.result }}" == "success" ]]; then
            echo "✅ Test (all-features): PASS" >> $GITHUB_STEP_SUMMARY
          else
            echo "⚠️ Test (all-features): FAIL (exploratory - not blocking)" >> $GITHUB_STEP_SUMMARY
          fi

          echo "" >> $GITHUB_STEP_SUMMARY
          echo "**Note**: Exploratory gates will be promoted to required after Issue #447 merges." >> $GITHUB_STEP_SUMMARY

      - name: Notify promotion readiness
        if: needs.clippy-all-features.result == 'success' && needs.test-all-features.result == 'success'
        run: |
          echo "::notice::🎉 All exploratory gates passing! Ready to promote to required gates."
```

**Status**: `continue-on-error: true` - **Allowed to fail without blocking CI**

---

### Feature Combinations Matrix

| Feature Combination | Required/Exploratory | Status | Command |
|---------------------|---------------------|--------|---------|
| **No features** | Required | Must pass | `cargo clippy --workspace --no-default-features` |
| **CPU only** | Required | Must pass | `cargo clippy --workspace --no-default-features --features cpu` |
| **GPU only** | Exploratory | May fail | `cargo clippy --workspace --no-default-features --features gpu` |
| **All features** | Exploratory | May fail until #447 | `cargo clippy --workspace --all-features` |

**Rationale**: CPU baseline ensures BitNet.rs compiles on all platforms without external dependencies (CUDA, etc.).

---

### Promotion Strategy

#### Phase 1: Initial Deployment (Issue #447 PR)
- Deploy exploratory gates with `continue-on-error: true`
- Validate required gates still pass
- Observe exploratory gate failures (expected)

#### Phase 2: Fixes Complete (Issue #447 merges)
- AC1-AC3: OpenTelemetry OTLP migration completes
- AC4-AC5: Inference engine type exports complete
- AC6-AC7: Test infrastructure API updates complete
- Exploratory gates should now pass

#### Phase 3: Promotion (Separate PR after #447)
- Change `continue-on-error: true` → `continue-on-error: false`
- Update CI documentation to reflect required status
- Add branch protection rules requiring all-features gates

**Timeline**:
- **Day 0**: Deploy exploratory gates (Issue #447 PR)
- **Day 1-3**: Implement and merge AC1-AC8 fixes
- **Day 4-5**: Validate exploratory gates passing
- **Day 6**: Promote exploratory to required (separate PR)

---

## Validation Commands Matrix

| Gate | Command | Status | Blocking |
|------|---------|--------|----------|
| **Format** | `cargo fmt --all -- --check` | Required | ✅ Yes |
| **Clippy (CPU)** | `cargo clippy --workspace --no-default-features --features cpu -- -D warnings` | Required | ✅ Yes |
| **Test (CPU)** | `cargo test --workspace --no-default-features --features cpu` | Required | ✅ Yes |
| **Clippy (All)** | `cargo clippy --workspace --all-features -- -D warnings` | Exploratory → Required | ⚠️ No (until promoted) |
| **Test (All)** | `cargo test --workspace --all-features` | Exploratory → Required | ⚠️ No (until promoted) |

---

## Migration Checklist

### Phase 1: Deploy Exploratory Gates
- [ ] **AC8.1**: Create `.github/workflows/all-features-exploratory.yml` workflow
- [ ] **AC8.2**: Add `continue-on-error: true` to exploratory jobs
- [ ] **AC8.3**: Add summary job to track promotion readiness
- [ ] **AC8.4**: Validate required gates still pass

**Validation**:
```bash
# Simulate required gates (must pass)
cargo clippy --workspace --no-default-features --features cpu -- -D warnings
cargo test --workspace --no-default-features --features cpu

# Simulate exploratory gates (may fail)
cargo clippy --workspace --all-features -- -D warnings || echo "Expected failure before #447 fixes"
cargo test --workspace --all-features || echo "Expected failure before #447 fixes"
```

---

### Phase 2: Monitor Exploratory Gates
- [ ] **AC8.5**: Observe exploratory gate failures in CI
- [ ] **AC8.6**: Track AC1-AC7 fixes resolving failures
- [ ] **AC8.7**: Validate exploratory gates pass after all fixes merged

**Validation**:
```bash
# After AC1-AC3 (OTLP migration)
cargo check -p bitnet-server --no-default-features --features opentelemetry

# After AC4-AC5 (Inference types)
cargo check -p bitnet-inference --all-features

# After AC6-AC7 (Test infrastructure)
cargo test -p tests --no-run

# All together (exploratory gates)
cargo clippy --workspace --all-features -- -D warnings
cargo test --workspace --all-features
```

---

### Phase 3: Promote to Required Gates
- [ ] **AC8.8**: Change `continue-on-error: true` → `false` in exploratory workflow
- [ ] **AC8.9**: Update CI documentation to reflect required status
- [ ] **AC8.10**: Add branch protection rules (if applicable)

**Validation**:
```bash
# Verify all-features gates now block CI
# (Simulated - actual validation occurs in CI)
cargo clippy --workspace --all-features -- -D warnings
cargo test --workspace --all-features
```

---

## Rollback Strategy

### Rollback Steps

1. **Revert Exploratory Workflow**:
   ```bash
   git checkout .github/workflows/all-features-exploratory.yml
   # Or delete file if newly created:
   rm .github/workflows/all-features-exploratory.yml
   ```

2. **Restore Original CI Workflow** (if modified):
   ```bash
   git checkout .github/workflows/ci.yml
   ```

3. **Validate Rollback**:
   ```bash
   # Ensure required gates still work
   cargo clippy --workspace --no-default-features --features cpu -- -D warnings
   cargo test --workspace --no-default-features --features cpu
   ```

### Rollback Criteria
- Exploratory gates introduce CI instability
- Required gates incorrectly fail
- Workflow syntax errors
- Performance degradation (excessive CI time)

**Risk**: Low - exploratory gates are non-blocking by design

---

## CI Performance Impact

### Execution Time

**Baseline (Required Gates Only)**:
- Format check: ~5 seconds
- Clippy (CPU): ~3 minutes
- Test (CPU): ~5 minutes
- **Total**: ~8 minutes

**With Exploratory Gates**:
- Clippy (All Features): ~5 minutes (more dependencies)
- Test (All Features): ~8 minutes (more test configurations)
- **Total**: ~21 minutes (parallel execution)

**Mitigation**: Exploratory gates run in parallel, don't block required gates

---

### Caching Strategy

**Required Gates**:
```yaml
key: ${{ runner.os }}-cargo-cpu-${{ hashFiles('**/Cargo.lock') }}
```

**Exploratory Gates**:
```yaml
key: ${{ runner.os }}-cargo-all-features-${{ hashFiles('**/Cargo.lock') }}
```

**Rationale**: Separate cache keys prevent feature flag contamination

---

## Branch Protection Rules

### Current Rules (Unchanged)
- ✅ Required: CI workflow must pass
- ✅ Required: Format check must pass
- ✅ Required: Clippy (CPU) must pass
- ✅ Required: Test (CPU) must pass

### Proposed Rules (Post-Promotion)
- ✅ Required: All existing rules
- ✅ Required: Clippy (All Features) must pass
- ✅ Required: Test (All Features) must pass

**Implementation**: After Phase 3 (promotion) complete

---

## BitNet.rs Standards Compliance

### Feature Flag Discipline
✅ Required gates use `--no-default-features --features cpu`
✅ Exploratory gates use `--all-features` for comprehensive validation
✅ Default features remain empty (no hidden dependencies)

### Workspace Structure Alignment
✅ Workspace-wide validation with `--workspace` flag
✅ Crate-specific tests still pass (no regression)
✅ Cross-crate dependency compatibility validated

### Neural Network Development Patterns
✅ CPU baseline ensures inference works without GPU
✅ All-features validation catches integration issues
✅ Zero impact on existing inference algorithms

### TDD and Test Naming
✅ Exploratory gates provide early feedback on AC1-AC7 fixes
✅ Clear promotion criteria based on gate success
✅ Transparent status reporting in CI summaries

### GGUF Compatibility
✅ No impact (CI configuration only, no code changes)

---

## References

### GitHub Actions Documentation
- [Workflow Syntax](https://docs.github.com/en/actions/using-workflows/workflow-syntax-for-github-actions)
- [continue-on-error](https://docs.github.com/en/actions/using-workflows/workflow-syntax-for-github-actions#jobsjob_idcontinue-on-error)
- [Job Summaries](https://docs.github.com/en/actions/using-workflows/workflow-commands-for-github-actions#adding-a-job-summary)

### BitNet.rs Documentation
- `docs/development/build-commands.md` - Comprehensive build reference
- `docs/development/validation-framework.md` - Quality assurance
- `CLAUDE.md` - Feature flag discipline

### Related Issues
- Issue #447 - Compilation fixes across workspace crates
- PR #440 - GPU feature unification (CI patterns)

---

## Approval Checklist

Before implementation:
- [x] Acceptance criterion (AC8) clearly defined
- [x] Required vs exploratory gates distinguished
- [x] Promotion strategy documented with timeline
- [x] Validation commands specified
- [x] Rollback strategy documented
- [x] CI performance impact analyzed
- [x] Feature flag discipline maintained
- [x] Zero impact on existing required gates confirmed

**Status**: ✅ Ready for Implementation

**Next Steps**:
- **Phase 1**: Deploy exploratory gates in Issue #447 PR
- **Phase 2**: Monitor gates as AC1-AC7 fixes merge
- **Phase 3**: Promote to required gates in separate PR (post-#447)
