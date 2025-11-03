# #[ignore] Annotation Hygiene - Phase 1 Action Plan

**Date**: October 23, 2025
**Phase**: 1 of 5 (Issue-Blocked Tests)
**Estimated Time**: 2 hours
**Automation Rate**: 95% (high-confidence auto-annotation)

---

## Executive Summary

This action plan provides step-by-step instructions for **Phase 1** of the #[ignore] annotation hygiene migration. Phase 1 targets **46 issue-blocked tests** with **95% automation confidence** using the `auto-annotate-ignores.sh` script.

**Goal**: Reduce bare #[ignore] annotations from **134 (56.8%)** to **~91 (38.6%)** by annotating high-confidence issue-blocked tests.

---

## Table of Contents

1. [Pre-Execution Detection](#1-pre-execution-detection)
2. [Phase 1 Target Files](#2-phase-1-target-files)
3. [Execution Workflow](#3-execution-workflow)
4. [Manual Review Procedure](#4-manual-review-procedure)
5. [Post-Execution Verification](#5-post-execution-verification)
6. [CI Integration Check](#6-ci-integration-check)
7. [Rollback Procedure](#7-rollback-procedure)
8. [Success Criteria](#8-success-criteria)

---

## 1. Pre-Execution Detection

### 1.1 Current State Baseline

Run the hygiene checker to establish the current baseline:

```bash
# Navigate to BitNet.rs root
cd /home/steven/code/Rust/BitNet-rs

# Run full hygiene scan
MODE=full bash scripts/check-ignore-hygiene.sh > phase1-pre-baseline.txt

# Extract summary metrics
grep -E "Total #\[ignore\]|Bare \(no reason\)" phase1-pre-baseline.txt
```

**Expected Output**:
```
Total #[ignore] annotations: 236
Bare (no reason):            134 (56%)
```

### 1.2 Generate Suggestions Preview

Generate auto-annotation suggestions to preview changes:

```bash
# Generate suggestions file
MODE=suggest bash scripts/check-ignore-hygiene.sh

# Review suggestions file
cat ignore-suggestions.txt | head -100
```

**Validation**:
- ✅ File `ignore-suggestions.txt` created (~539 lines)
- ✅ Contains categorized suggestions with confidence scores
- ✅ High-confidence (≥70%) suggestions are reasonable

---

## 2. Phase 1 Target Files

### 2.1 High-Priority Files (≥70% Confidence)

These files contain **issue-blocked tests** with explicit issue references in comments, making auto-annotation **highly reliable**:

| File | Bare Count | Category | Confidence |
|------|-----------|----------|------------|
| `crates/bitnet-inference/tests/issue_254_ac3_deterministic_generation.rs` | 10 | slow + Issue #254 | 80% |
| `crates/bitnet-models/tests/gguf_weight_loading_property_tests.rs` | 9 | Issue #159 | 80% |
| `crates/bitnet-inference/tests/neural_network_test_scaffolding.rs` | 8 | Issue #248 | 75% |
| `crates/bitnet-inference/tests/ac3_autoregressive_generation.rs` | 6 | slow | 75% |
| `crates/bitnet-models/tests/gguf_weight_loading_property_tests_enhanced.rs` | 5 | Issue #159 | 80% |
| `crates/bitnet-inference/tests/issue_254_layer_norm_invariants.rs` | 4 | Issue #254 | 80% |
| `crates/bitnet-kernels/tests/issue_260_feature_gated_tests.rs` | 3 | Issue #260 | 80% |
| `crates/bitnet-inference/tests/issue_260_mock_elimination_inference_tests.rs` | 5 | Issue #260 | 80% |

**Total**: ~50 bare annotations (targets 46 from Phase 1 scope)

### 2.2 File List for Bulk Processing

Create a Phase 1 target list:

```bash
cat > phase1-files.txt <<'EOF'
crates/bitnet-inference/tests/issue_254_ac3_deterministic_generation.rs
crates/bitnet-models/tests/gguf_weight_loading_property_tests.rs
crates/bitnet-inference/tests/neural_network_test_scaffolding.rs
crates/bitnet-inference/tests/ac3_autoregressive_generation.rs
crates/bitnet-models/tests/gguf_weight_loading_property_tests_enhanced.rs
crates/bitnet-inference/tests/issue_254_layer_norm_invariants.rs
crates/bitnet-kernels/tests/issue_260_feature_gated_tests.rs
crates/bitnet-inference/tests/issue_260_mock_elimination_inference_tests.rs
crates/bitnet-inference/tests/issue_254_ac1_quantized_linear_no_fp32_staging.rs
crates/bitnet-inference/tests/issue_254_ac4_receipt_generation.rs
crates/bitnet-inference/tests/issue_254_ac6_determinism_integration.rs
EOF
```

---

## 3. Execution Workflow

### 3.1 Dry-Run Phase (Preview Changes)

**IMPORTANT**: Always run dry-run first to preview changes before applying.

#### 3.1.1 Single File Dry-Run (Test)

Test the automation script on a single high-confidence file:

```bash
# Dry-run on a single file
TARGET_FILE="crates/bitnet-inference/tests/issue_254_ac3_deterministic_generation.rs" \
  DRY_RUN=true bash scripts/auto-annotate-ignores.sh

# Expected output:
#   Processing: crates/bitnet-inference/tests/issue_254_ac3_deterministic_generation.rs
#   Line 23: Would annotate with "slow: ..." (confidence: 80%)
#   Line 27: Would annotate with "slow: ..." (confidence: 80%)
#   ...
#   📋 Would annotate 10 ignores
```

**Review Checklist**:
- [ ] Confidence scores are ≥70%
- [ ] Suggested annotations match file context (Issue #254 references)
- [ ] No false positives (annotations unrelated to ignore reason)

#### 3.1.2 Bulk Dry-Run (All Phase 1 Files)

Run dry-run on all Phase 1 files:

```bash
# Dry-run all Phase 1 files
while read -r file; do
  echo "=== Dry-run: $file ==="
  TARGET_FILE="$file" DRY_RUN=true bash scripts/auto-annotate-ignores.sh
  echo ""
done < phase1-files.txt > phase1-dryrun-output.txt

# Review dry-run output
less phase1-dryrun-output.txt
```

**Validation**:
- ✅ All files processed without errors
- ✅ Confidence scores are reasonable (≥70%)
- ✅ Total expected annotations: ~46-50

#### 3.1.3 Dry-Run Manual Review

Extract and review all suggested annotations:

```bash
# Extract all suggested annotations from dry-run
grep "Would annotate with" phase1-dryrun-output.txt | \
  awk -F'"' '{print $2}' | \
  sort | uniq -c | sort -rn

# Expected categories:
#   - Issue #254: ...
#   - Issue #159: ...
#   - Issue #248: ...
#   - Issue #260: ...
#   - slow: ...
```

**Manual Review Checklist**:
- [ ] Issue numbers match file context (e.g., `issue_254_*.rs` → `Issue #254`)
- [ ] Slow annotations have descriptive reasons
- [ ] No generic "FIXME: add reason" annotations (confidence <70%)

---

### 3.2 Apply Phase (Execute Changes)

**CRITICAL**: Only proceed if dry-run validation passes.

#### 3.2.1 Create Git Branch

Create a dedicated branch for Phase 1 changes:

```bash
# Create and switch to Phase 1 branch
git checkout -b ignore-hygiene-phase1

# Verify clean working directory
git status
```

#### 3.2.2 Execute Auto-Annotation

Apply auto-annotations to all Phase 1 files:

```bash
# Apply annotations (NO DRY-RUN)
while read -r file; do
  echo "=== Processing: $file ==="
  TARGET_FILE="$file" DRY_RUN=false bash scripts/auto-annotate-ignores.sh
  echo ""
done < phase1-files.txt > phase1-apply-output.txt

# Review application output
less phase1-apply-output.txt
```

**Expected Output (per file)**:
```
Processing: crates/bitnet-inference/tests/issue_254_ac3_deterministic_generation.rs
  Line 23: Annotating with "slow: ..." (confidence: 80%)
  Line 27: Annotating with "slow: ..." (confidence: 80%)
  ...
  ✅ Annotated 10 ignores and formatted
```

#### 3.2.3 Format Code

Run `rustfmt` to ensure consistent formatting:

```bash
# Format all modified files
cargo fmt --all

# Check formatting status
cargo fmt --all -- --check
```

---

## 4. Manual Review Procedure

### 4.1 Review Low-Confidence Annotations

The auto-annotation script **only applies annotations with ≥70% confidence**. Review files with low-confidence suggestions manually.

#### 4.1.1 Find Low-Confidence Cases

Extract low-confidence suggestions from dry-run output:

```bash
# Find low-confidence suggestions (<70%)
grep -E "LOW CONFIDENCE|confidence: [0-6][0-9]%" phase1-dryrun-output.txt | \
  grep -B1 "LOW CONFIDENCE"

# Review these files manually
```

#### 4.1.2 Manual Annotation Template

For low-confidence cases, manually add annotations using this template:

```rust
// BEFORE (bare #[ignore])
#[test]
#[ignore]
fn test_some_feature() { /* ... */ }

// AFTER (annotated with reason)
#[test]
#[ignore = "Issue #254: shape mismatch in layer-norm - awaiting fix"]
fn test_some_feature() { /* ... */ }
```

**Category Templates**:
- **Issue-blocked**: `Issue #NNN: <brief description>`
- **Slow tests**: `slow: <reason>, see <alternative>`
- **Requires model**: `requires: <resource>`
- **GPU tests**: `gpu: <requirement>`
- **Network**: `network: <dependency>`
- **TODO**: `TODO: <task>`

### 4.2 Verify Annotations Match Context

Manually review a sample of auto-annotated files to verify correctness:

```bash
# Review sample file (issue_254_ac3_deterministic_generation.rs)
git diff crates/bitnet-inference/tests/issue_254_ac3_deterministic_generation.rs | head -50

# Expected diff:
#   -    #[ignore]
#   +    #[ignore = "slow: 50-token generation, see tests/deterministic_sampling_unit.rs"]
```

**Validation Checklist**:
- [ ] Annotations reference correct issue numbers
- [ ] Slow test annotations include alternatives
- [ ] Requires annotations specify exact dependencies
- [ ] No placeholder text (e.g., "...") remains

### 4.3 Manual Refinement (If Needed)

If auto-annotations are too generic, refine them manually:

```bash
# Edit file with refined annotations
vim crates/bitnet-inference/tests/issue_254_ac3_deterministic_generation.rs

# Example refinement:
# BEFORE: #[ignore = "slow: ..."]
# AFTER:  #[ignore = "slow: 50-token generation (100+ forward passes), see tests/deterministic_sampling_unit.rs"]
```

---

## 5. Post-Execution Verification

### 5.1 Count Remaining Bare Ignores

Verify the reduction in bare #[ignore] annotations:

```bash
# Run hygiene check post-migration
MODE=full bash scripts/check-ignore-hygiene.sh > phase1-post-baseline.txt

# Extract metrics
grep -E "Total #\[ignore\]|Bare \(no reason\)" phase1-post-baseline.txt
```

**Expected Metrics**:
```
Total #[ignore] annotations: 236
Bare (no reason):            ~91 (38%)
```

**Success Criteria**:
- ✅ Bare annotations reduced from 134 to ~91 (43 annotations added)
- ✅ Bare percentage reduced from 56.8% to ~38.6%

### 5.2 Validate Annotation Syntax

Ensure all new annotations follow the proper format:

```bash
# Check for malformed annotations
rg '#\[ignore = "[^"]*"\]' --type rust crates/ tests/ xtask/ | \
  grep -v 'Issue #\|slow:\|requires:\|gpu:\|network:\|TODO:\|FLAKY:\|parity:\|quantization:' || \
  echo "✅ All annotations use valid prefixes"

# Check for placeholder text remaining
rg '#\[ignore = "[^"]*\.\.\.[^"]*"\]' --type rust crates/ tests/ xtask/ || \
  echo "✅ No placeholder text (...) remains"
```

### 5.3 Test Suite Validation

Ensure all tests still compile and pass (excluding ignored tests):

```bash
# Run test suite (non-ignored tests only)
cargo test --workspace --no-default-features --features cpu --lib 2>&1 | tee phase1-test-output.txt

# Check for compilation errors
grep -i "error\[E" phase1-test-output.txt || echo "✅ No compilation errors"

# Check test pass rate
grep "test result:" phase1-test-output.txt
```

**Expected Output**:
```
test result: ok. 152 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out
```

### 5.4 Git Diff Review

Review all changes before committing:

```bash
# Show summary of changed files
git status

# Review full diff
git diff | less

# Review only annotation changes
git diff | grep -A2 -B2 '#\[ignore'
```

**Review Checklist**:
- [ ] Only #[ignore] lines modified (no accidental code changes)
- [ ] All annotations have explicit reasons (no bare #[ignore] remains)
- [ ] Formatting is consistent (rustfmt applied)

---

## 6. CI Integration Check

### 6.1 Verify CI Job Configuration

Check if the `ignore-hygiene` CI job is configured:

```bash
# Check if CI job exists
grep -A10 "ignore-hygiene" .github/workflows/ci.yml

# Expected: CI job definition with MODE=diff enforcement
```

**If CI job is missing**, create it (defer to CI setup task):

```yaml
# .github/workflows/ci.yml (add this job)
ignore-hygiene:
  name: "#[ignore] Annotation Hygiene Check"
  runs-on: ubuntu-latest
  steps:
    - uses: actions/checkout@v4
    - name: Check ignore hygiene
      run: |
        MODE=diff bash scripts/check-ignore-hygiene.sh
      env:
        FAIL_ON_BARE: true
        MAX_BARE_PERCENT: 5
```

### 6.2 Local CI Simulation

Simulate the CI hygiene check locally:

```bash
# Run CI-style diff mode
MODE=diff FAIL_ON_BARE=false bash scripts/check-ignore-hygiene.sh

# Expected output (if no new bare ignores):
# ✅ No new bare #[ignore] annotations in this PR
```

---

## 7. Rollback Procedure

If issues are detected during verification, rollback changes:

```bash
# Discard all changes and return to main branch
git reset --hard HEAD
git checkout main
git branch -D ignore-hygiene-phase1

# Re-run baseline check
MODE=full bash scripts/check-ignore-hygiene.sh
```

**When to Rollback**:
- Test suite fails compilation
- Annotations are incorrect (wrong issue numbers, etc.)
- More than 5% of annotations are placeholder text

---

## 8. Success Criteria

### 8.1 Quantitative Metrics

- [ ] **Bare annotations reduced**: 134 → ~91 (43 annotations added)
- [ ] **Bare percentage reduced**: 56.8% → ~38.6%
- [ ] **High-confidence annotations**: ≥95% of Phase 1 targets annotated
- [ ] **Test suite passes**: No regressions (152+ tests passing)
- [ ] **No placeholder text**: Zero annotations with "..." remaining

### 8.2 Qualitative Checks

- [ ] **Annotations are accurate**: Issue numbers match file context
- [ ] **Annotations are descriptive**: Include actionable information
- [ ] **Slow test annotations**: Reference fast alternatives
- [ ] **Requires annotations**: Specify exact dependencies
- [ ] **No false positives**: All annotations match ignore reasons

### 8.3 Git Hygiene

- [ ] **Clean diff**: Only #[ignore] lines modified
- [ ] **Formatted code**: `cargo fmt --all -- --check` passes
- [ ] **Descriptive commit**: Clear message referencing Phase 1
- [ ] **Branch ready for PR**: `ignore-hygiene-phase1` branch created

---

## 9. Execution Timeline

### Step-by-Step Timeline (2 hours)

| Step | Duration | Task |
|------|----------|------|
| **Pre-Execution** | 15 min | Baseline detection, dry-run validation |
| **Dry-Run Review** | 20 min | Review suggestions, manual checklist |
| **Git Branch Setup** | 5 min | Create branch, verify clean state |
| **Auto-Annotation** | 10 min | Execute `auto-annotate-ignores.sh` |
| **Manual Review** | 30 min | Review low-confidence cases, refine annotations |
| **Formatting** | 5 min | Run `cargo fmt --all` |
| **Post-Verification** | 20 min | Count bare ignores, validate syntax, test suite |
| **Git Diff Review** | 10 min | Review changes, check for errors |
| **CI Simulation** | 5 min | Local CI check with MODE=diff |
| **Buffer** | 10 min | Contingency for issues |
| **Total** | **2 hours** | |

---

## 10. Copy-Paste Command Sequences

### Quick Execution (Experienced Users)

```bash
# === PRE-EXECUTION ===
cd /home/steven/code/Rust/BitNet-rs
MODE=full bash scripts/check-ignore-hygiene.sh > phase1-pre-baseline.txt
grep -E "Total #\[ignore\]|Bare \(no reason\)" phase1-pre-baseline.txt

# === CREATE PHASE 1 FILE LIST ===
cat > phase1-files.txt <<'EOF'
crates/bitnet-inference/tests/issue_254_ac3_deterministic_generation.rs
crates/bitnet-models/tests/gguf_weight_loading_property_tests.rs
crates/bitnet-inference/tests/neural_network_test_scaffolding.rs
crates/bitnet-inference/tests/ac3_autoregressive_generation.rs
crates/bitnet-models/tests/gguf_weight_loading_property_tests_enhanced.rs
crates/bitnet-inference/tests/issue_254_layer_norm_invariants.rs
crates/bitnet-kernels/tests/issue_260_feature_gated_tests.rs
crates/bitnet-inference/tests/issue_260_mock_elimination_inference_tests.rs
crates/bitnet-inference/tests/issue_254_ac1_quantized_linear_no_fp32_staging.rs
crates/bitnet-inference/tests/issue_254_ac4_receipt_generation.rs
crates/bitnet-inference/tests/issue_254_ac6_determinism_integration.rs
EOF

# === DRY-RUN (PREVIEW CHANGES) ===
while read -r file; do
  TARGET_FILE="$file" DRY_RUN=true bash scripts/auto-annotate-ignores.sh
done < phase1-files.txt > phase1-dryrun-output.txt
less phase1-dryrun-output.txt

# === MANUAL REVIEW OF DRY-RUN ===
grep "Would annotate with" phase1-dryrun-output.txt | \
  awk -F'"' '{print $2}' | sort | uniq -c | sort -rn

# === CREATE GIT BRANCH ===
git checkout -b ignore-hygiene-phase1
git status

# === APPLY AUTO-ANNOTATIONS ===
while read -r file; do
  TARGET_FILE="$file" DRY_RUN=false bash scripts/auto-annotate-ignores.sh
done < phase1-files.txt > phase1-apply-output.txt
less phase1-apply-output.txt

# === FORMAT CODE ===
cargo fmt --all
cargo fmt --all -- --check

# === POST-EXECUTION VERIFICATION ===
MODE=full bash scripts/check-ignore-hygiene.sh > phase1-post-baseline.txt
grep -E "Total #\[ignore\]|Bare \(no reason\)" phase1-post-baseline.txt

# === VALIDATE SYNTAX ===
rg '#\[ignore = "[^"]*\.\.\.[^"]*"\]' --type rust crates/ tests/ xtask/ || \
  echo "✅ No placeholder text (...) remains"

# === TEST SUITE ===
cargo test --workspace --no-default-features --features cpu --lib 2>&1 | tee phase1-test-output.txt
grep "test result:" phase1-test-output.txt

# === GIT DIFF REVIEW ===
git status
git diff | grep -A2 -B2 '#\[ignore' | less

# === LOCAL CI SIMULATION ===
MODE=diff FAIL_ON_BARE=false bash scripts/check-ignore-hygiene.sh

# === COMMIT (IF ALL CHECKS PASS) ===
git add -A
git commit -m "feat(tests): Phase 1 #[ignore] annotation hygiene - issue-blocked tests

- Auto-annotate 46 issue-blocked tests with explicit reasons
- Reduce bare #[ignore] from 134 (56.8%) to ~91 (38.6%)
- High-confidence (≥70%) auto-annotation using scripts/auto-annotate-ignores.sh
- Categories: Issue #254, #159, #248, #260, slow tests
- See IGNORE_HYGIENE_STATUS_REPORT.md for migration plan

Phase 1 of 5-week phased rollout (SPEC-2025-006)"
```

---

## 11. Expected Outcomes

### Before Phase 1

```
Total #[ignore] annotations: 236
Annotated (with reason):     102 (43%)
Bare (no reason):            134 (56%)
Status:                      Non-compliant (56% > 5% threshold)
```

### After Phase 1

```
Total #[ignore] annotations: 236
Annotated (with reason):     ~145 (61%)
Bare (no reason):            ~91 (38%)
Status:                      Improved (38% still > 5%, but 18% reduction)
Progress:                    Phase 1 of 5 complete (20% progress)
```

---

## 12. Next Steps (Post-Phase 1)

After Phase 1 completion and PR merge:

1. **Phase 2 (Week 2)**: Performance/slow tests (17 annotations, 85% automation)
2. **Phase 3 (Week 3)**: Network/external dependencies (13 annotations, 75% automation)
3. **Phase 4 (Week 4)**: Model/fixture dependencies (29 annotations, 70% automation)
4. **Phase 5 (Week 5)**: Quantization/parity/TODO (43 annotations, 60% automation)

**Final Target**: ≤5% bare annotations (≤12 remaining) by Week 5.

---

## 13. Troubleshooting

### Issue: Script Fails with "command not found"

**Cause**: Scripts not executable or missing dependencies.

**Solution**:
```bash
# Make scripts executable
chmod +x scripts/check-ignore-hygiene.sh
chmod +x scripts/auto-annotate-ignores.sh

# Install dependencies
sudo apt-get install ripgrep  # or brew install ripgrep on macOS
```

### Issue: Annotations Applied to Wrong Lines

**Cause**: File modified between dry-run and apply.

**Solution**:
```bash
# Rollback and re-run
git reset --hard HEAD
# Re-run dry-run validation before applying
```

### Issue: Test Suite Fails After Annotation

**Cause**: Syntax error in annotation string (unescaped quotes, etc.).

**Solution**:
```bash
# Find malformed annotations
rg '#\[ignore = "[^"]*"[^]]' --type rust crates/ tests/ xtask/

# Fix manually or rollback
git diff | grep -A2 '#\[ignore'
```

### Issue: Too Many Low-Confidence Cases

**Cause**: File context doesn't match auto-detection patterns.

**Solution**:
```bash
# Skip low-confidence files in Phase 1, defer to later phases
# Manually annotate high-priority files only
```

---

## 14. Document Closure

**Action Plan Status**: Ready for Immediate Execution
**Phase**: 1 of 5
**Estimated Completion**: 2 hours
**Automation Confidence**: 95%

**Pre-requisites**:
- [x] `scripts/check-ignore-hygiene.sh` present and executable
- [x] `scripts/auto-annotate-ignores.sh` present and executable
- [x] `scripts/ignore-taxonomy.json` v1.0.0 present
- [x] Clean working directory (no uncommitted changes)

**Next Action**: Execute Pre-Execution Detection (Section 1.1)

---

**Author**: BitNet.rs Automation Framework
**Last Updated**: October 23, 2025
**Version**: 1.0.0
**See Also**: `IGNORE_HYGIENE_STATUS_REPORT.md`, `SPEC-2025-006-ignore-annotation-automation.md`
