# SPEC-2025-003: Classify and Plan 29 Unclassified Ignored Tests

**Status**: Draft
**Created**: 2025-10-23
**Priority**: P1
**Category**: Test Hygiene
**Related Issues**: #254, #260, #469
**Related PRs**: #475

---

## Executive Summary

Classify 29 unclassified ignored tests across the BitNet-rs test suite, document ownership and resolution plans, and establish test hygiene standards. This addresses technical debt from TDD scaffolding during MVP phase and provides clear roadmap for test enablement.

**Current State**: 135 `#[ignore]` markers across 46 files, with 29 tests lacking clear categorization or resolution plans.

**Target State**: All ignored tests classified with:
- **Owner**: Team/individual responsible for resolution
- **Reason**: Specific blocker (issue number, missing feature, performance concern)
- **Plan**: Resolution timeline and acceptance criteria

**Impact**:
- **Test Clarity**: Explicit documentation of test status and unblocking dependencies
- **Velocity**: Clear ownership enables parallel work on test enablement
- **CI Health**: Distinguishes between legitimate blockers and test hygiene issues

---

## Requirements Analysis

### Functional Requirements

1. **FR1: Test Classification**
   - Audit 29 unclassified ignored tests from `COMPREHENSIVE_TEST_AUDIT_REPORT.md`
   - Assign each test to one of 6 categories:
     - **Issue Blockers**: Tests blocked by #254, #260, #469
     - **TDD Placeholders**: Future feature scaffolding
     - **Performance/Slow**: Intentional slow test quarantine
     - **Network/External**: Resource dependency tests
     - **Missing Fixtures**: Requires real GGUF models
     - **Technical Debt**: Code quality improvements needed
   - Document classification in status table with rationale

2. **FR2: Ownership Assignment**
   - Assign owner for each unclassified test (individual or team)
   - Define clear acceptance criteria for test enablement
   - Establish priority (P0/P1/P2/P3) for each test

3. **FR3: Resolution Planning**
   - Create timeline estimates for each category
   - Document dependencies between tests (e.g., test X requires issue #254 resolution)
   - Identify quick wins (tests that can be enabled immediately)

### Non-Functional Requirements

1. **NFR1: Documentation Quality**
   - Status table must be machine-readable (Markdown table format)
   - Include file paths and line numbers for all tests
   - Link to blocking issues/PRs where applicable

2. **NFR2: Maintainability**
   - Establish process for future test classification
   - Document test hygiene standards in `docs/development/test-suite.md`
   - Create templates for adding new ignored tests

---

## Architecture Approach

### Test Audit Data Structure

**Status Table Schema**:
```markdown
| File | Test Name | Line | Category | Owner | Blocker | Priority | Plan | Est. |
|------|-----------|------|----------|-------|---------|----------|------|------|
| path/to/file.rs | test_name | 123 | Issue #254 | @owner | #254 | P1 | Unblock when #254 resolved | Q1 2026 |
```

**Categories**:
1. **Issue #254 Blockers**: Shape mismatch in layer-norm (3 tests currently classified)
2. **Issue #260 Blockers**: Mock elimination incomplete (11 tests currently classified)
3. **Issue #469 Blockers**: Tokenizer parity and FFI hygiene (affected cross-validation tests)
4. **TDD Placeholders**: Future feature scaffolding (44 tests currently classified)
5. **Performance/Slow**: Intentional slow test quarantine (3 tests currently classified)
6. **Network/External**: Resource dependencies (28 tests currently classified)
7. **Missing Fixtures**: Requires real GGUF models (NEW category for unclassified tests)
8. **Technical Debt**: Code quality improvements (NEW category for unclassified tests)

### Workspace Integration

**Documentation Structure**:
```
docs/development/
├── test-suite.md                # Add "Ignored Tests Management" section
└── test-hygiene-standards.md   # NEW: Test classification standards

ci/solutions/
└── IGNORED_TESTS_STATUS.md     # NEW: Machine-readable status table

.github/
└── PULL_REQUEST_TEMPLATE.md    # Add checklist: "New ignored tests classified?"
```

---

## Quantization Strategy

**Not Applicable**: This spec focuses on test hygiene and documentation, not quantization algorithms.

---

## GPU/CPU Implementation

**Not Applicable**: Test classification is backend-agnostic. However, we will document which tests are GPU-specific vs. CPU-specific for better test planning.

---

## GGUF Integration

**Not Applicable**: This spec focuses on test metadata and documentation. However, tests requiring GGUF fixtures will be classified in the "Missing Fixtures" category.

---

## Performance Specifications

**Not Applicable**: Performance is not a concern for documentation work. However, we will identify slow tests and document quarantine rationale.

---

## Cross-Validation Plan

### Test Audit Validation

**Audit Completeness Check**:
```bash
# 1. Count total ignored tests in codebase
grep -r "#\[ignore\]" --include="*.rs" | wc -l
# Expected: 135 (from audit report)

# 2. Count classified tests in status table
wc -l ci/solutions/IGNORED_TESTS_STATUS.md
# Expected: 135+ lines (header + 135 tests)

# 3. Count unclassified tests
grep "Unclassified" ci/solutions/IGNORED_TESTS_STATUS.md | wc -l
# Expected: 0 (all tests classified)
```

### Category Distribution Validation

**Expected Distribution** (from audit report):
- Issue #254 Blockers: 3 tests
- Issue #260 Blockers: 11 tests
- Issue #469 Blockers: ~20 tests (cross-validation + tokenizer)
- TDD Placeholders: 44 tests
- Performance/Slow: 3 tests
- Network/External: 28 tests
- Missing Fixtures: ~15 tests (estimated from unclassified)
- Technical Debt: ~11 tests (estimated from unclassified)

**Total**: 135 tests

---

## Feature Flag Analysis

**Not Applicable**: This spec focuses on test metadata documentation, not feature flag changes.

---

## Testing Strategy

### Audit Validation Tests

**Metadata Consistency Tests** (`tests/test_audit_validation.rs` - NEW):
```rust
#[test]
fn test_all_ignored_tests_classified() {
    // Parse IGNORED_TESTS_STATUS.md
    let status_table = std::fs::read_to_string("ci/solutions/IGNORED_TESTS_STATUS.md")
        .expect("Status table not found");

    // Ensure no "Unclassified" entries
    assert!(!status_table.contains("Unclassified"),
            "Found unclassified ignored tests");

    // Ensure all tests have owners
    assert!(!status_table.contains("| TBD |"),
            "Found tests without owners");
}

#[test]
fn test_ignored_test_count_matches_audit() {
    // Count #[ignore] markers in codebase
    let output = std::process::Command::new("grep")
        .args(&["-r", "#\\[ignore\\]", "--include=*.rs"])
        .output()
        .expect("grep failed");

    let actual_count = String::from_utf8_lossy(&output.stdout)
        .lines()
        .count();

    // Count entries in status table
    let status_table = std::fs::read_to_string("ci/solutions/IGNORED_TESTS_STATUS.md")
        .expect("Status table not found");
    let documented_count = status_table.lines()
        .filter(|line| line.starts_with("|") && !line.contains("File"))
        .count();

    assert_eq!(actual_count, documented_count,
               "Mismatch between actual ignored tests and documented tests");
}
```

### Integration Tests

**Not Applicable**: This spec produces documentation artifacts, not code changes requiring integration tests.

---

## Risk Mitigation

### Technical Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| **Audit drift** | Medium | Medium | Add CI check: `test_ignored_test_count_matches_audit` |
| **Stale status table** | Medium | Low | Require PR checklist: "Updated ignored tests status?" |
| **Incorrect categorization** | Low | Medium | Peer review for all classifications; link to blocking issues |
| **Ownership ambiguity** | Low | Medium | Assign to team (e.g., "Core Team") when individual unclear |

### Validation Commands

**Risk Validation**:
```bash
# 1. Verify audit completeness (drift detection)
cargo test --test test_audit_validation test_ignored_test_count_matches_audit

# 2. Check for unclassified tests
grep "Unclassified" ci/solutions/IGNORED_TESTS_STATUS.md
# Expected: no output (all tests classified)

# 3. Verify all tests have owners
grep "| TBD |" ci/solutions/IGNORED_TESTS_STATUS.md
# Expected: no output (all tests have owners)

# 4. Check category distribution
for category in "Issue #254" "Issue #260" "TDD Placeholder" "Performance" "Network" "Missing Fixtures" "Technical Debt"; do
  count=$(grep -c "$category" ci/solutions/IGNORED_TESTS_STATUS.md)
  echo "$category: $count tests"
done
```

---

## Success Criteria

### Measurable Acceptance Criteria

**AC1: All Ignored Tests Classified**
- ✅ 29 unclassified tests assigned to categories
- ✅ Status table contains 135+ entries (one per ignored test)
- ✅ No "Unclassified" entries in status table

**Validation**:
```bash
# Check total classified tests
wc -l ci/solutions/IGNORED_TESTS_STATUS.md | grep -E "13[5-9]|1[4-9][0-9]"
# Expected: 135+ lines (header + tests)

# Check for unclassified tests
grep "Unclassified" ci/solutions/IGNORED_TESTS_STATUS.md
# Expected: no output

# Run audit validation test
cargo test --test test_audit_validation test_all_ignored_tests_classified
# Expected: test passes
```

**AC2: Ownership and Plans Documented**
- ✅ Each test has assigned owner (individual or team)
- ✅ Each test has resolution plan with timeline estimate
- ✅ Blocking dependencies documented (issue numbers, PR links)

**Validation**:
```bash
# Check for missing owners
grep "| TBD |" ci/solutions/IGNORED_TESTS_STATUS.md
# Expected: no output

# Check for missing plans
grep "| TODO |" ci/solutions/IGNORED_TESTS_STATUS.md
# Expected: no output (or acceptable for TDD placeholders)

# Verify issue references
grep -oE "#[0-9]+" ci/solutions/IGNORED_TESTS_STATUS.md | sort -u
# Expected: #254, #260, #469 (known blockers)
```

**AC3: Test Hygiene Standards Documented**
- ✅ `docs/development/test-suite.md` updated with "Ignored Tests Management" section
- ✅ Template provided for adding new ignored tests
- ✅ CI check added to prevent undocumented ignored tests

**Validation**:
```bash
# Check documentation update
grep -A 10 "Ignored Tests Management" docs/development/test-suite.md
# Expected: section exists with classification guidelines

# Verify CI check exists
cargo test --test test_audit_validation
# Expected: test suite includes audit validation
```

**AC4: Quick Wins Identified**
- ✅ At least 5 tests identified for immediate enablement
- ✅ Quick wins prioritized in status table
- ✅ Implementation plan provided for quick wins

**Validation**:
```bash
# Check for quick wins
grep "Quick Win" ci/solutions/IGNORED_TESTS_STATUS.md | wc -l
# Expected: ≥5 tests

# Verify quick win priority
grep "Quick Win" ci/solutions/IGNORED_TESTS_STATUS.md | grep "P0"
# Expected: all quick wins marked P0 or P1
```

---

## Performance Thresholds

**Not Applicable**: This spec produces documentation artifacts with no runtime performance impact.

---

## Implementation Notes

### Status Table Format

**Template**:
```markdown
# BitNet-rs Ignored Tests Status Table

**Last Updated**: 2025-10-23
**Total Ignored Tests**: 135
**Unclassified Tests**: 0

## Summary by Category

| Category | Count | Owner | Est. Timeline |
|----------|-------|-------|---------------|
| Issue #254 Blockers | 3 | Core Team | Q1 2026 (issue resolution) |
| Issue #260 Blockers | 11 | Core Team | Q1 2026 (refactoring) |
| Issue #469 Blockers | 20 | Tokenizer Team | Q4 2025 |
| TDD Placeholders | 44 | Various | Q2-Q3 2026 (feature dev) |
| Performance/Slow | 3 | QA Team | Permanent (manual/nightly only) |
| Network/External | 28 | Infra Team | Q4 2025 (fixture mocks) |
| Missing Fixtures | 15 | Test Team | Q4 2025 (fixture creation) |
| Technical Debt | 11 | Core Team | Q1 2026 (cleanup) |

## Detailed Status Table

| File | Test Name | Line | Category | Owner | Blocker | Priority | Plan | Est. |
|------|-----------|------|----------|-------|---------|----------|------|------|
| crates/bitnet-inference/tests/simple_real_inference.rs | test_forward_pass_with_quantized_weights | 45 | Missing Fixtures | @test-team | Real GGUF with weights | P1 | Create fixture or use existing model | Q4 2025 |
| crates/bitnet-inference/tests/simple_real_inference.rs | test_attention_mechanism | 72 | Missing Fixtures | @test-team | Real GGUF with weights | P1 | Create fixture or use existing model | Q4 2025 |
| ... | ... | ... | ... | ... | ... | ... | ... | ... |
```

### Test Classification Guidelines

**Decision Tree**:
```
Is test blocked by open issue (#254, #260, #469)?
├─ YES → Category: "Issue #XXX Blocker"
└─ NO
   └─ Is test scaffolded with unimplemented!()?
      ├─ YES → Category: "TDD Placeholder"
      └─ NO
         └─ Does test run >30 seconds?
            ├─ YES → Category: "Performance/Slow"
            └─ NO
               └─ Does test require network/external resources?
                  ├─ YES → Category: "Network/External"
                  └─ NO
                     └─ Does test require real GGUF model?
                        ├─ YES → Category: "Missing Fixtures"
                        └─ NO → Category: "Technical Debt" (needs investigation)
```

### Ignored Test Template

**For New Ignored Tests**:
```rust
#[test]
#[ignore] // Category: <Issue #XXX | TDD Placeholder | Performance | Network | Missing Fixtures | Technical Debt>
//         Reason: <Specific blocker or rationale>
//         Owner: <@username or team>
//         Plan: <Resolution plan with timeline>
//         Tracked: <Issue number or status table reference>
fn test_new_functionality() {
    // Test implementation
}
```

**Example**:
```rust
#[test]
#[ignore] // Category: Missing Fixtures
//         Reason: Requires real GGUF with QK256 weights
//         Owner: @test-team
//         Plan: Create fixture via ci/fixtures/qk256/ workflow
//         Tracked: SPEC-2025-002 (persistent fixtures)
fn test_qk256_full_inference() {
    // Test implementation
}
```

---

## BitNet-rs Alignment

### TDD Practices

✅ **Alignment**: Explicit documentation of TDD scaffolding tests (44 placeholders)

### Feature-Gated Architecture

✅ **Alignment**: Status table distinguishes feature-gated tests (GPU-only, network-dependent)

### Workspace Structure

✅ **Alignment**: Documentation stored in `ci/solutions/` and `docs/development/`

### Cross-Platform Support

✅ **Alignment**: Test classification includes platform-specific notes (e.g., GPU tests)

---

## Neural Network References

**Not Applicable**: This spec focuses on test metadata and documentation, not neural network implementation.

---

## Related Documentation

- **Test Audit Report**: `COMPREHENSIVE_TEST_AUDIT_REPORT.md` (already exists)
- **Test Suite Guide**: `docs/development/test-suite.md` (needs update)
- **Issue #254**: Shape mismatch in layer-norm (blocker for 3 tests)
- **Issue #260**: Mock elimination incomplete (blocker for 11 tests)
- **Issue #469**: Tokenizer parity and FFI hygiene (blocker for ~20 tests)

---

## Implementation Checklist

**Phase 1: Audit Unclassified Tests** (2 hours)
- [ ] Review 29 unclassified tests from `COMPREHENSIVE_TEST_AUDIT_REPORT.md`
- [ ] Classify each test using decision tree
- [ ] Assign owner for each test (individual or team)
- [ ] Document blocker and resolution plan
- [ ] Estimate timeline for each category

**Phase 2: Create Status Table** (1 hour)
- [ ] Create `ci/solutions/IGNORED_TESTS_STATUS.md`
- [ ] Populate status table with all 135 ignored tests
- [ ] Include summary by category
- [ ] Add links to blocking issues/PRs
- [ ] Verify no "Unclassified" or "TBD" entries

**Phase 3: Documentation Updates** (1 hour)
- [ ] Add "Ignored Tests Management" section to `docs/development/test-suite.md`
- [ ] Document test classification guidelines
- [ ] Provide ignored test template for future use
- [ ] Update CLAUDE.md with status table reference

**Phase 4: CI Integration** (30 minutes)
- [ ] Create `tests/test_audit_validation.rs`
- [ ] Add `test_all_ignored_tests_classified` check
- [ ] Add `test_ignored_test_count_matches_audit` check
- [ ] Update PR template: Add "New ignored tests classified?" checklist

**Phase 5: Identify Quick Wins** (1 hour)
- [ ] Review status table for tests with no blockers
- [ ] Identify 5+ tests that can be enabled immediately
- [ ] Create implementation plan for quick wins
- [ ] Prioritize quick wins in status table

---

## Quick Wins Analysis

### Potential Quick Wins (To Be Identified During Implementation)

**Criteria for Quick Wins**:
1. No blocking issue dependencies
2. Test implementation complete (not TDD placeholder)
3. Minimal effort to enable (<2 hours work)
4. High value for CI health

**Example Quick Win Categories**:
- Tests ignored due to outdated comments (Issue #439 resolved)
- Tests waiting for fixtures (can use existing models)
- Tests with trivial assertion fixes
- Tests blocked by resolved issues

**Quick Win Template**:
```markdown
### Quick Win: <Test Name>

**File**: `<path/to/file.rs>`
**Line**: <line number>
**Category**: <original category>
**Effort**: <1-2 hours>
**Value**: <High/Medium>

**Current Blocker**: <original reason for ignore>
**Resolution**: <specific steps to enable>

**Implementation**:
1. <Step 1>
2. <Step 2>
3. Run test: `cargo test <test_name>`
4. Verify passes: Remove `#[ignore]` marker
```

---

## Status

**Current Phase**: Draft Specification
**Next Steps**: Review and approval → Implementation
**Estimated Implementation Time**: 6 hours (audit + documentation + CI integration)
**Risk Level**: Low (documentation only, no production code changes)

---

**Last Updated**: 2025-10-23
**Spec Author**: BitNet-rs Spec Analyzer Agent
**Review Status**: Pending
