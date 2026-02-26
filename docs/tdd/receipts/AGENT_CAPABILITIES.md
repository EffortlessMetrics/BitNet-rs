# Agent Capabilities for Issue Resolution

**Date**: 2025-10-21
**Purpose**: Map available agents to fixable issues based on Agent A validation

## Current Issues (Agent A Findings)

### Issue #260: Two Failing Tests

From `docs/tdd/receipts/agent_a_scaffold_validation.md`:

1. **`test_cpu_simd_kernel_integration`** (line 182)
   - **Status**: FAILING ❌
   - **Error**: `Unimplemented: quantized_matmul not yet implemented`
   - **Root Cause**: Mock `TestKernel::quantized_matmul()` returns unimplemented error (line 74)

2. **`test_tl2_avx_optimization`** (line 317)
   - **Status**: FAILING ❌
   - **Error**: Expected 4096-entry table, got 65536
   - **Root Cause**: Mock `LookupTable` returns wrong size (line 102)

### Issue #254: Phantom Test Reference

**Correction**: Previous documentation incorrectly referenced `test_real_vs_mock_comparison` which doesn't exist.
- ❌ `test_real_vs_mock_comparison` — **Phantom test** (docs error)
- ✅ `test_real_vs_mock_inference_comparison` — **Actual test** exists in `test_real_vs_mock_comparison.rs`
- ✅ `test_real_transformer_forward_pass` — **Actual test** exists in `test_real_inference.rs`

**Status**: Only 1 documented scaffold test exists for Issue #254, not 2.

---

## Agents That Can Fix Issues

### 🔧 **impl-creator** — Make Failing Tests Pass

**Perfect for Issue #260 tests**

**Capability**: "Write minimal production code to make failing tests pass"

**Can Fix**:
1. ✅ `test_cpu_simd_kernel_integration`
   - Analyze the test requirements
   - Implement `quantized_matmul` in production code to satisfy test expectations
   - Make the test pass with minimal implementation

2. ✅ `test_tl2_avx_optimization`
   - Fix mock `LookupTable` to return 4096-entry table
   - Or implement real TL2 lookup table in production code

**Usage**:
```bash
# Let impl-creator analyze and fix the failing tests
Task tool → impl-creator:
"Analyze failing tests in crates/bitnet-kernels/tests/issue_260_feature_gated_tests.rs
and implement minimal production code to make test_cpu_simd_kernel_integration and
test_tl2_avx_optimization pass"
```

**Expected Outcome**:
- Two new/modified files with production implementations
- Tests change from FAILING → PASSING
- Minimal, focused code changes (TDD style)

---

### 📝 **test-creator** — Create Missing Tests

**Not needed for Issue #254** (tests already exist)

**Capability**: "Create comprehensive test scaffolding for features defined in specification files"

**Status**: Issue #254 tests already exist:
- `test_real_vs_mock_inference_comparison` — exists in `test_real_vs_mock_comparison.rs`
- `test_real_transformer_forward_pass` — exists in `test_real_inference.rs`

**Note**: Previous docs incorrectly claimed `test_real_vs_mock_comparison` was missing. The actual test has "inference" in the middle of the name.

**Expected Outcome**:
- New test files in `crates/bitnet-inference/tests/`
- Tests marked `#[ignore]` initially (TDD scaffold)
- Clear acceptance criteria mapped to test assertions

**Note**: Only useful if Issue #254 specs exist. If not, recommend removing doc references instead.

---

### 🧹 **doc-fixer** — Fix Documentation Issues

**Perfect for cleanup after Agent A findings**

**Capability**: "Remediate broken links, failing doctests, outdated examples"

**Can Fix**:
1. ✅ Remove references to non-existent Issue #254 tests from all docs
2. ✅ Update test counts in documentation (already done via receipts)
3. ✅ Fix any broken links discovered during validation

**Usage**:
```bash
Task tool → doc-fixer:
"Remove all references to test_real_vs_mock_comparison and test_real_transformer_forward_pass
from documentation files. These tests don't exist and are documentation artifacts per Agent A
validation."
```

**Expected Outcome**:
- Documentation matches reality
- No references to phantom tests
- Clean grep results for test names

---

### 🔍 **generative-code-reviewer** — Validate Implementations

**Perfect for after impl-creator fixes**

**Capability**: "Perform final code quality pass before implementation finalization"

**Can Fix**:
1. ✅ Review `impl-creator` implementations for quality
2. ✅ Check formatting, clippy lints, BitNet-rs standards
3. ✅ Validate neural network implementation patterns

**Usage**:
```bash
# After impl-creator makes tests pass
Task tool → generative-code-reviewer:
"Review the quantized_matmul implementation for code quality, performance patterns,
and BitNet-rs neural network standards compliance"
```

**Expected Outcome**:
- Code quality report
- Clippy/format violations identified
- Recommendations for improvements

---

### 🎯 **impl-finalizer** — Comprehensive Quality Review

**Perfect for validating the entire Issue #260 fix**

**Capability**: "Perform first full quality review ensuring tests pass, quality gates green, code meets standards"

**Can Fix**:
1. ✅ Validate all Issue #260 tests pass
2. ✅ Run quality gates (fmt, clippy, tests, build)
3. ✅ Ensure ready for refinement phase

**Usage**:
```bash
# After impl-creator + generative-code-reviewer
Task tool → impl-finalizer:
"Perform comprehensive quality review of Issue #260 implementation. Validate that
test_cpu_simd_kernel_integration and test_tl2_avx_optimization pass, quality gates green,
and code meets BitNet-rs standards."
```

**Expected Outcome**:
- Quality gate report (all green)
- Test execution receipt (both tests passing)
- Ready-for-refinement confirmation

---

### 🧪 **test-hardener** — Strengthen Test Suite

**Perfect for after tests pass**

**Capability**: "Improve test quality through mutation testing and fuzzing"

**Can Fix**:
1. ✅ Run mutation testing on Issue #260 implementations
2. ✅ Ensure tests actually catch bugs (not false positives)
3. ✅ Improve test coverage and edge cases

**Usage**:
```bash
# After impl-finalizer confirms all green
Task tool → test-hardener:
"Run mutation testing on quantized_matmul implementation to ensure
test_cpu_simd_kernel_integration catches real bugs"
```

**Expected Outcome**:
- Mutation test score (target ≥80%)
- Identified weak test cases
- Recommendations for additional assertions

---

## Recommended Workflow

### Option 1: Fix Issue #260 Tests (Make Them Pass)

```bash
1. impl-creator          → Implement quantized_matmul + fix TL2 table
2. generative-code-reviewer → Review code quality
3. impl-finalizer        → Validate everything works
4. test-hardener         → Ensure tests are robust
5. doc-updater           → Update docs to reflect working code
```

**Result**: Issue #260 tests go from FAILING → PASSING ✅

### Option 2: Clean Up Documentation (Remove Phantom Tests)

```bash
1. doc-fixer → Remove references to test_real_vs_mock_comparison and test_real_transformer_forward_pass
```

**Result**: Documentation matches reality ✅

### Option 3: Create Missing Tests (If Specs Exist)

```bash
1. Check if Issue #254 specs exist in docs/explanation/
2. IF specs exist:
   test-creator → Generate test scaffolding
3. ELSE:
   doc-fixer → Remove all Issue #254 references
```

**Result**: Either tests created OR documentation cleaned ✅

---

## Quick Wins

### Immediate Fix: Remove Phantom Tests from Docs

**Agent**: doc-fixer
**Time**: <5 min
**Impact**: Documentation accuracy ✅

```bash
Task tool → doc-fixer:
"Remove all documentation references to test_real_vs_mock_comparison and
test_real_transformer_forward_pass per Agent A validation showing these tests
don't exist"
```

### High-Value Fix: Make Issue #260 Tests Pass

**Agents**: impl-creator → generative-code-reviewer → impl-finalizer
**Time**: 15-30 min
**Impact**: Two failing tests → passing ✅

```bash
Task tool → impl-creator:
"Implement minimal production code to make test_cpu_simd_kernel_integration and
test_tl2_avx_optimization pass. Tests are in
crates/bitnet-kernels/tests/issue_260_feature_gated_tests.rs"
```

---

## Summary

**Total Issues Found**: 4
- **2 failing tests** (Issue #260) → **impl-creator can fix**
- **2 phantom tests** (Issue #254) → **doc-fixer can remove references**

**Best ROI**: Run `impl-creator` for Issue #260 to turn 2 failing tests into passing tests.

**Safest Fix**: Run `doc-fixer` to clean up phantom test references.

**Comprehensive Fix**: Run full workflow (impl-creator → code-reviewer → impl-finalizer → test-hardener) for production-ready Issue #260 resolution.
