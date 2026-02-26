# GitHub Issue Specifications - P1 Tasks

**Date**: 2025-10-23
**Scope**: Four P1 tasks for immediate implementation
**Related**: SPEC-2025-002 through SPEC-2025-005

---

## Overview

This document provides formatted GitHub issue templates for four P1 tasks identified from the comprehensive test audit and documentation review. Each issue is production-ready and can be copied directly to GitHub.

---

## Issue 1: Create Persistent GGUF Fixtures and Wire Integration Tests

**Title**: Create persistent GGUF fixtures and wire integration tests

**Labels**: `P1`, `test-infrastructure`, `enhancement`

**Related Spec**: `docs/explanation/specs/SPEC-2025-002-persistent-gguf-fixtures.md`

**Description**:

```markdown
## Summary

Migrate 12 fixture-based tests from in-memory generation to disk-based loading, using persistent GGUF fixtures in `ci/fixtures/qk256/`. This eliminates runtime generation overhead (200ms → <10ms) and ensures reproducible CI/CD builds.

## Current State

- Tests generate GGUF fixtures in-memory via `helpers::qk256_fixtures` module (~200ms overhead per suite)
- Fixtures already exist in `ci/fixtures/qk256/` from PR #475
- Tests still use in-memory generation instead of disk-based loading

## Target State

- 12 tests load fixtures from `ci/fixtures/qk256/` with SHA256 verification
- Fixture loading overhead <10ms per test (20× improvement)
- CI validates fixture integrity before test execution

## Acceptance Criteria

### AC1: Persistent Fixtures
- [ ] 3 GGUF fixtures exist in `ci/fixtures/qk256/`:
  - `qk256_4x256.gguf` (10,816 bytes)
  - `qk256_3x300.gguf` (10,696 bytes)
  - `bitnet32_2x64.gguf` (8,832 bytes)
- [ ] `SHA256SUMS` file present with correct checksums
- [ ] `README.md` documents fixture metadata and regeneration workflow

**Validation**:
```bash
ls -lh ci/fixtures/qk256/
sha256sum -c ci/fixtures/qk256/SHA256SUMS
```

### AC2: Test Migration
- [ ] 12 tests in `crates/bitnet-models/tests/qk256_dual_flavor_tests.rs` migrated
- [ ] All tests pass with `--features cpu,fixtures`
- [ ] Fixture loading overhead <10ms per test

**Validation**:
```bash
cargo test -p bitnet-models --test qk256_dual_flavor_tests \
  --no-default-features --features cpu,fixtures \
  -- --nocapture 2>&1 | tee /tmp/fixture-tests.log

# Expected: test result: ok. 12 passed; 0 failed
```

### AC3: Feature Gate Control
- [ ] Tests skip gracefully when `fixtures` feature disabled
- [ ] Clear error messages when fixture files missing
- [ ] In-memory fallback remains functional

**Validation**:
```bash
# Without fixtures feature (tests should skip or use fallback)
cargo test -p bitnet-models --test qk256_dual_flavor_tests \
  --no-default-features --features cpu

# With missing fixture (simulate file deletion)
mv ci/fixtures/qk256/qk256_4x256.gguf /tmp/
cargo test -p bitnet-models --test qk256_dual_flavor_tests \
  --no-default-features --features cpu,fixtures 2>&1 \
  | grep -E "(skipped|fixture not found)"
mv /tmp/qk256_4x256.gguf ci/fixtures/qk256/
```

### AC4: CI Integration
- [ ] CI runs fixture-based tests with integrity checks
- [ ] Fixture corruption fails CI builds
- [ ] Performance improvement measurable (200ms → <15ms)

**CI Workflow Addition**:
```yaml
- name: Verify fixture-based tests
  run: |
    cd ci/fixtures/qk256 && sha256sum -c SHA256SUMS
    cargo nextest run --profile ci -p bitnet-models \
      --no-default-features --features cpu,fixtures
```

## Implementation Guide

### Phase 1: Fixture Verification (5 minutes)
```bash
# Verify fixtures exist
ls -lh ci/fixtures/qk256/

# Run SHA256 verification
cd ci/fixtures/qk256 && sha256sum -c SHA256SUMS

# Validate GGUF format
cargo run -p bitnet-cli --features cpu,full-cli -- \
  compat-check ci/fixtures/qk256/*.gguf
```

### Phase 2: Test Migration (30 minutes)

**Add Helper Functions** (`crates/bitnet-models/tests/helpers/mod.rs`):
```rust
#[cfg(feature = "fixtures")]
pub fn load_fixture_path(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent().unwrap()
        .parent().unwrap()
        .join("ci/fixtures/qk256")
        .join(name)
}

#[cfg(feature = "fixtures")]
pub fn verify_fixture_integrity(path: &Path) -> Result<(), String> {
    use sha2::{Sha256, Digest};
    let mut hasher = Sha256::new();
    let bytes = std::fs::read(path).map_err(|e| e.to_string())?;
    hasher.update(&bytes);
    let hash = format!("{:x}", hasher.finalize());

    let sums_path = path.parent().unwrap().join("SHA256SUMS");
    let expected = std::fs::read_to_string(sums_path)
        .map_err(|e| e.to_string())?;

    if !expected.contains(&hash) {
        return Err(format!("SHA256 mismatch for {}", path.display()));
    }
    Ok(())
}
```

**Migrate Test Pattern**:
```rust
// Before (in-memory generation)
#[test]
#[cfg_attr(not(feature = "fixtures"), ignore)]
fn test_qk256_detection() {
    let fixture_bytes = helpers::qk256_fixtures::generate_qk256_4x256(42);
    let mut file = NamedTempFile::new().unwrap();
    file.write_all(&fixture_bytes).unwrap();
    // ... test logic
}

// After (disk-based loading)
#[test]
#[cfg(feature = "fixtures")]
fn test_qk256_detection() {
    let fixture_path = helpers::load_fixture_path("qk256_4x256.gguf");
    helpers::verify_fixture_integrity(&fixture_path).unwrap(); // Optional SHA256
    // ... test logic (use fixture_path directly)
}
```

### Phase 3: CI Integration (10 minutes)
- [ ] Update `.github/workflows/ci.yml`: Add SHA256 verification step
- [ ] Run CI tests: `cargo nextest run --profile ci --features cpu,fixtures`
- [ ] Verify performance improvement in CI logs

### Phase 4: Documentation (10 minutes)
- [ ] Update `docs/development/test-suite.md`: Add fixture loading section
- [ ] Update `CLAUDE.md`: Add fixture validation commands
- [ ] Review `ci/fixtures/qk256/README.md`: Ensure regeneration workflow documented

## Files to Modify

```
crates/bitnet-models/tests/
├── qk256_dual_flavor_tests.rs      # Migrate 12 tests
└── helpers/mod.rs                   # Add load_fixture_path(), verify_fixture_integrity()

.github/workflows/ci.yml             # Add SHA256 verification step
docs/development/test-suite.md       # Add fixture loading section
CLAUDE.md                            # Add validation commands
```

## Expected SHA256 Checksums

```
c1568a0a08e38ef2865ce0816bfd2c617e5589c113114cd731e4c5014b7fbb20  bitnet32_2x64.gguf
6e5a4f21607c0064affbcb86133627478eb34d812b59807a7123ff386c63bd3e  qk256_3x300.gguf
a41cc62c893bcf1d4c03c30ed3da12da03c339847c4d564e9e5794b5d4c6932a  qk256_4x256.gguf
```

## Estimated Effort

- Implementation: 1 hour
- Testing: 30 minutes
- Documentation: 15 minutes
- **Total**: ~2 hours

## Risk Assessment

**Risk Level**: Low
- Changes only affect test infrastructure
- No production code modified
- Version control safety net for fixture files
- Fixtures already validated in PR #475

## References

- **Detailed Spec**: `docs/explanation/specs/SPEC-2025-002-persistent-gguf-fixtures.md`
- **Fixture README**: `ci/fixtures/qk256/README.md`
- **Test Suite Guide**: `docs/development/test-suite.md`
```

---

## Issue 2: Classify and Plan 29 Unclassified Ignored Tests

**Title**: Classify and plan 29 unclassified ignored tests

**Labels**: `P1`, `test-hygiene`, `documentation`

**Related Spec**: `docs/explanation/specs/SPEC-2025-003-ignored-tests-audit.md`

**Description**:

```markdown
## Summary

Classify 29 unclassified ignored tests from comprehensive test audit, document ownership and resolution plans, and establish test hygiene standards. This provides clear roadmap for test enablement and distinguishes legitimate blockers from test debt.

## Current State

- 135 `#[ignore]` markers across 46 files
- 29 tests lack clear categorization or resolution plans
- No systematic tracking of ignored test status

## Target State

- All 135 ignored tests classified with owner, reason, and resolution plan
- Machine-readable status table in `ci/solutions/IGNORED_TESTS_STATUS.md`
- Test hygiene standards documented in `docs/development/test-suite.md`
- CI check prevents undocumented ignored tests

## Acceptance Criteria

### AC1: All Ignored Tests Classified
- [ ] 29 unclassified tests assigned to categories:
  - Issue #254 Blockers (shape mismatch)
  - Issue #260 Blockers (mock elimination)
  - Issue #469 Blockers (tokenizer parity)
  - TDD Placeholders (future features)
  - Performance/Slow (intentional quarantine)
  - Network/External (resource dependencies)
  - Missing Fixtures (needs GGUF models)
  - Technical Debt (code quality)
- [ ] Status table contains 135+ entries (one per ignored test)
- [ ] No "Unclassified" or "TBD" entries

**Validation**:
```bash
# Check total classified tests
wc -l ci/solutions/IGNORED_TESTS_STATUS.md | grep -E "13[5-9]|1[4-9][0-9]"

# Check for unclassified tests
grep "Unclassified" ci/solutions/IGNORED_TESTS_STATUS.md
# Expected: no output

# Check for missing owners
grep "| TBD |" ci/solutions/IGNORED_TESTS_STATUS.md
# Expected: no output
```

### AC2: Ownership and Plans Documented
- [ ] Each test has assigned owner (individual or team)
- [ ] Each test has resolution plan with timeline estimate
- [ ] Blocking dependencies documented (issue numbers, PR links)

**Validation**:
```bash
# Verify issue references
grep -oE "#[0-9]+" ci/solutions/IGNORED_TESTS_STATUS.md | sort -u
# Expected: #254, #260, #469 (known blockers)

# Check for missing plans
grep "| TODO |" ci/solutions/IGNORED_TESTS_STATUS.md
# Expected: no output (or acceptable for TDD placeholders)
```

### AC3: Test Hygiene Standards Documented
- [ ] `docs/development/test-suite.md` updated with "Ignored Tests Management" section
- [ ] Template provided for adding new ignored tests
- [ ] CI check added to prevent undocumented ignored tests

**Validation**:
```bash
# Check documentation update
grep -A 10 "Ignored Tests Management" docs/development/test-suite.md

# Verify CI check exists
cargo test --test test_audit_validation
# Expected: test suite includes audit validation
```

### AC4: Quick Wins Identified
- [ ] At least 5 tests identified for immediate enablement
- [ ] Quick wins prioritized in status table (P0/P1)
- [ ] Implementation plan provided for quick wins

**Validation**:
```bash
# Check for quick wins
grep "Quick Win" ci/solutions/IGNORED_TESTS_STATUS.md | wc -l
# Expected: ≥5 tests

# Verify quick win priority
grep "Quick Win" ci/solutions/IGNORED_TESTS_STATUS.md | grep -E "P[01]"
# Expected: all quick wins marked P0 or P1
```

## Implementation Guide

### Phase 1: Audit Unclassified Tests (2 hours)

**Review 29 Unclassified Tests**:
1. Open `COMPREHENSIVE_TEST_AUDIT_REPORT.md` section 1.7
2. For each test, apply decision tree:
   ```
   Blocked by #254/260/469? → Issue Blocker
   ├─ NO → Scaffolded with unimplemented!()? → TDD Placeholder
       ├─ NO → Runs >30s? → Performance/Slow
           ├─ NO → Needs network? → Network/External
               ├─ NO → Needs real GGUF? → Missing Fixtures
                   └─ NO → Technical Debt
   ```
3. Document blocker, owner, plan, timeline

**Example Classification**:
```markdown
| File | Test | Category | Owner | Blocker | Plan | Timeline |
|------|------|----------|-------|---------|------|----------|
| simple_real_inference.rs | test_forward_pass | Missing Fixtures | @test-team | Real GGUF | Create fixture | Q4 2025 |
```

### Phase 2: Create Status Table (1 hour)

**File**: `ci/solutions/IGNORED_TESTS_STATUS.md`

**Template**:
```markdown
# BitNet-rs Ignored Tests Status Table

**Last Updated**: 2025-10-23
**Total Ignored Tests**: 135
**Unclassified Tests**: 0

## Summary by Category

| Category | Count | Owner | Est. Timeline |
|----------|-------|-------|---------------|
| Issue #254 Blockers | 3 | Core Team | Q1 2026 |
| Issue #260 Blockers | 11 | Core Team | Q1 2026 |
| Issue #469 Blockers | 20 | Tokenizer Team | Q4 2025 |
| TDD Placeholders | 44 | Various | Q2-Q3 2026 |
| Performance/Slow | 3 | QA Team | Permanent |
| Network/External | 28 | Infra Team | Q4 2025 |
| Missing Fixtures | 15 | Test Team | Q4 2025 |
| Technical Debt | 11 | Core Team | Q1 2026 |

## Detailed Status Table

| File | Test Name | Line | Category | Owner | Blocker | Priority | Plan | Est. |
|------|-----------|------|----------|-------|---------|----------|------|------|
| ... | ... | ... | ... | ... | ... | ... | ... | ... |
```

### Phase 3: Documentation Updates (1 hour)

**Add to `docs/development/test-suite.md`**:
```markdown
## Ignored Tests Management

### Adding New Ignored Tests

When adding `#[ignore]` to a test, use this template:

```rust
#[test]
#[ignore] // Category: <Issue #XXX | TDD Placeholder | Performance | Network | Missing Fixtures>
//         Reason: <Specific blocker>
//         Owner: <@username or team>
//         Plan: <Resolution plan>
//         Tracked: <Status table reference>
fn test_new_functionality() { /* ... */ }
```

### Classification Categories

1. **Issue Blockers**: Tests blocked by open issues (#254, #260, #469)
2. **TDD Placeholders**: Future feature scaffolding
3. **Performance/Slow**: Intentional slow test quarantine (>30s)
4. **Network/External**: Tests requiring network/external resources
5. **Missing Fixtures**: Tests requiring real GGUF models
6. **Technical Debt**: Code quality improvements needed

### Status Tracking

All ignored tests are tracked in `ci/solutions/IGNORED_TESTS_STATUS.md`.
```

### Phase 4: CI Integration (30 minutes)

**Create `tests/test_audit_validation.rs`**:
```rust
#[test]
fn test_all_ignored_tests_classified() {
    let status_table = std::fs::read_to_string("ci/solutions/IGNORED_TESTS_STATUS.md")
        .expect("Status table not found");

    assert!(!status_table.contains("Unclassified"),
            "Found unclassified ignored tests");
    assert!(!status_table.contains("| TBD |"),
            "Found tests without owners");
}

#[test]
fn test_ignored_test_count_matches_audit() {
    let output = std::process::Command::new("grep")
        .args(&["-r", "#\\[ignore\\]", "--include=*.rs"])
        .output()
        .expect("grep failed");

    let actual_count = String::from_utf8_lossy(&output.stdout).lines().count();

    let status_table = std::fs::read_to_string("ci/solutions/IGNORED_TESTS_STATUS.md")
        .expect("Status table not found");
    let documented_count = status_table.lines()
        .filter(|line| line.starts_with("|") && !line.contains("File"))
        .count();

    assert_eq!(actual_count, documented_count,
               "Mismatch: {} actual vs {} documented", actual_count, documented_count);
}
```

### Phase 5: Identify Quick Wins (1 hour)

**Criteria**:
- No blocking issue dependencies
- Test implementation complete
- <2 hours effort to enable
- High value for CI health

**Example Quick Win**:
```markdown
### Quick Win: test_forward_pass_basic

**File**: `simple_real_inference.rs:45`
**Effort**: 1 hour
**Value**: High

**Current Blocker**: Missing GGUF fixture with weights
**Resolution**: Use existing `models/model.gguf` from CI

**Steps**:
1. Update test to use `BITNET_GGUF` env var
2. Remove `#[ignore]` marker
3. Run: `BITNET_GGUF=models/model.gguf cargo test test_forward_pass_basic`
```

## Files to Create/Modify

```
ci/solutions/
└── IGNORED_TESTS_STATUS.md          # NEW: Status table

docs/development/
└── test-suite.md                     # Update: Add "Ignored Tests Management"

tests/
└── test_audit_validation.rs          # NEW: CI validation

.github/
└── PULL_REQUEST_TEMPLATE.md         # Update: Add checklist item
```

## Estimated Effort

- Audit: 2 hours
- Status table: 1 hour
- Documentation: 1 hour
- CI integration: 30 minutes
- Quick wins: 1 hour
- **Total**: ~6 hours

## Risk Assessment

**Risk Level**: Low
- Documentation only (no production code)
- Clear audit trail via status table
- CI check prevents future drift

## References

- **Detailed Spec**: `docs/explanation/specs/SPEC-2025-003-ignored-tests-audit.md`
- **Test Audit Report**: `COMPREHENSIVE_TEST_AUDIT_REPORT.md`
- **Issue #254**: Shape mismatch blockers
- **Issue #260**: Mock elimination blockers
- **Issue #469**: Tokenizer parity blockers
```

---

## Issue 3: Consolidate ci/solutions/ Documentation Structure

**Title**: Consolidate ci/solutions/ documentation structure

**Labels**: `P1`, `documentation`, `tech-debt`

**Related Spec**: `docs/explanation/specs/SPEC-2025-004-docs-consolidation.md`

**Description**:

```markdown
## Summary

Consolidate redundant documentation in `ci/solutions/` by merging 3 small index files, deleting 2 duplicate summaries, and establishing single source of truth. This reduces documentation debt and improves navigation clarity.

## Current State

- 30+ markdown files with 4 overlapping navigation/summary documents
- `INDEX.md` duplicates content from `00_NAVIGATION_INDEX.md`
- `SOLUTION_SUMMARY.md` already merged to `SOLUTIONS_SUMMARY.md`
- `SUMMARY.md` content merged to `README.md`

## Target State

- Single master navigation: `00_NAVIGATION_INDEX.md`
- Single high-level summary: `SOLUTIONS_SUMMARY.md`
- Zero broken internal links (lychee validation)
- Clear entry point in `ci/README.md`

## Acceptance Criteria

### AC1: Index Files Consolidated
- [ ] `INDEX.md` content merged to `00_NAVIGATION_INDEX.md`
- [ ] `INDEX.md` deleted from repository
- [ ] No unique content lost in merge

**Validation**:
```bash
# Check INDEX.md deleted
[ ! -f ci/solutions/INDEX.md ] && echo "✓ Deleted" || echo "✗ Still exists"

# Check content merged
grep "Solution 1: Clippy" ci/solutions/00_NAVIGATION_INDEX.md
grep "Solution 2: Concurrent Load" ci/solutions/00_NAVIGATION_INDEX.md

# Check navigation breadcrumbs preserved
grep "Navigation:" ci/solutions/00_NAVIGATION_INDEX.md | head -1
```

### AC2: Duplicate Summaries Removed
- [ ] `SOLUTION_SUMMARY.md` deleted (content in `SOLUTIONS_SUMMARY.md`)
- [ ] `SUMMARY.md` deleted (content in `README.md`)
- [ ] No orphaned references to deleted files

**Validation**:
```bash
# Check files deleted
[ ! -f ci/solutions/SOLUTION_SUMMARY.md ] && echo "✓ Deleted"
[ ! -f ci/solutions/SUMMARY.md ] && echo "✓ Deleted"

# Check no orphaned references
grep -r "SOLUTION_SUMMARY.md" ci/solutions/*.md && echo "✗ Orphan" || echo "✓ Clean"
grep -r "SUMMARY.md" ci/solutions/*.md && echo "✗ Orphan" || echo "✓ Clean"
```

### AC3: Links Validated
- [ ] Lychee passes with 0 broken internal links
- [ ] All `INDEX.md` references updated to `00_NAVIGATION_INDEX.md`
- [ ] `ci/README.md` references master navigation

**Validation**:
```bash
# Run lychee link checker
lychee ci/solutions/*.md --offline --no-progress
# Expected: SUCCESS - 0 broken internal links

# Check ci/README.md navigation
grep "00_NAVIGATION_INDEX.md" ci/README.md

# Verify all INDEX.md references updated
grep -r "INDEX.md" ci/solutions/*.md | grep -v "00_NAVIGATION_INDEX.md"
# Expected: no output
```

### AC4: Documentation Structure Clear
- [ ] Single master navigation: `00_NAVIGATION_INDEX.md`
- [ ] Single summary: `SOLUTIONS_SUMMARY.md`
- [ ] Clear entry point in `ci/README.md`

**Validation**:
```bash
# Verify entry point documented
grep -A 3 "solutions/" ci/README.md
# Expected: Clear description of 00_NAVIGATION_INDEX.md
```

## Implementation Guide

### Phase 1: Content Audit (30 minutes)

**Verify Content Already Merged**:
```bash
# Check SOLUTION_SUMMARY.md vs SOLUTIONS_SUMMARY.md
diff ci/solutions/SOLUTION_SUMMARY.md ci/solutions/SOLUTIONS_SUMMARY.md
# Expected: Minimal diff (already merged per INDEX.md line 19)

# Check SUMMARY.md content in README.md
grep "^## Solution 1: Clippy" ci/solutions/README.md
# Expected: Content found (already merged per INDEX.md line 20)

# Identify unique content in INDEX.md
grep "^##" ci/solutions/INDEX.md > /tmp/index-sections.txt
while read -r section; do
  grep -F "$section" ci/solutions/00_NAVIGATION_INDEX.md >/dev/null \
    || echo "Missing: $section"
done < /tmp/index-sections.txt
```

### Phase 2: Merge INDEX.md Content (1 hour)

**Steps**:
1. Open `00_NAVIGATION_INDEX.md` in editor
2. Add new section: "## Solution Analysis Summaries"
3. Copy unique sections from `INDEX.md`:
   - Quick Navigation (lines 29-41)
   - Solution 1 analysis (lines 43-78)
   - Solution 2 analysis (lines 80-121)
   - Implementation workflows (lines 123-169)
   - QK256 documentation reference (lines 447-495)
4. Preserve breadcrumb navigation
5. Save and review diff

### Phase 3: Update Internal Links (30 minutes)

**Automated Link Replacement**:
```bash
# Replace INDEX.md → 00_NAVIGATION_INDEX.md
find ci/solutions -name "*.md" -exec sed -i 's|INDEX\.md|00_NAVIGATION_INDEX.md|g' {} \;

# Replace SOLUTION_SUMMARY.md → SOLUTIONS_SUMMARY.md (if any)
find ci/solutions -name "*.md" -exec sed -i 's|SOLUTION_SUMMARY\.md|SOLUTIONS_SUMMARY.md|g' {} \;

# Verify changes
git diff ci/solutions/*.md | grep "INDEX.md"
```

### Phase 4: Delete Duplicate Files (10 minutes)

**Final Verification Before Deletion**:
```bash
# Diff check (ensure no content loss)
git diff HEAD ci/solutions/00_NAVIGATION_INDEX.md | grep "^+" | wc -l
# Expected: >100 lines added (INDEX.md content merged)

# Delete duplicate files
git rm ci/solutions/INDEX.md
git rm ci/solutions/SOLUTION_SUMMARY.md
git rm ci/solutions/SUMMARY.md

# Verify deletions staged
git status ci/solutions/
```

### Phase 5: Link Validation (30 minutes)

**Before/After Lychee Check**:
```bash
# Run lychee before
lychee ci/solutions/*.md --exclude-path ci/solutions/_TEMPLATE.md \
  > /tmp/lychee-before.txt

# Run lychee after
lychee ci/solutions/*.md --exclude-path ci/solutions/_TEMPLATE.md \
  > /tmp/lychee-after.txt

# Compare results
diff /tmp/lychee-before.txt /tmp/lychee-after.txt
# Expected: Fewer or same errors (no new broken links)
```

### Phase 6: Update Entry Points (15 minutes)

**Update `ci/README.md`**:
```markdown
## Solutions Directory

For detailed analysis and implementation guides, see the master navigation:
- **[00_NAVIGATION_INDEX.md](solutions/00_NAVIGATION_INDEX.md)** - Complete index of all solutions
- **[SOLUTIONS_SUMMARY.md](solutions/SOLUTIONS_SUMMARY.md)** - High-level summary

Quick access:
- Clippy fixes: [CLIPPY_QUICK_REFERENCE.md](solutions/CLIPPY_QUICK_REFERENCE.md)
- QK256 analysis: [QK256_ANALYSIS_INDEX.md](solutions/QK256_ANALYSIS_INDEX.md)
```

### Phase 7: CI Integration (15 minutes)

**Create Validation Script** (`ci/scripts/validate-docs.sh`):
```bash
#!/bin/bash
set -e

echo "Validating ci/solutions/ documentation structure..."

# Check deleted files don't exist
for file in INDEX.md SOLUTION_SUMMARY.md SUMMARY.md; do
  if [ -f "ci/solutions/$file" ]; then
    echo "ERROR: Duplicate file still exists: $file"
    exit 1
  fi
done

# Run lychee link checker
lychee ci/solutions/*.md --exclude-path ci/solutions/_TEMPLATE.md \
  --offline --no-progress

echo "✓ Documentation structure validated"
```

**Add CI Job**:
```yaml
# .github/workflows/ci.yml
- name: Validate documentation structure
  run: |
    chmod +x ci/scripts/validate-docs.sh
    ./ci/scripts/validate-docs.sh
```

## Files to Modify/Delete

```
ci/solutions/
├── 00_NAVIGATION_INDEX.md           # Update: Merge INDEX.md content
├── INDEX.md                          # Delete: Content merged
├── SOLUTION_SUMMARY.md              # Delete: Already merged to SOLUTIONS_SUMMARY.md
├── SUMMARY.md                        # Delete: Content in README.md
├── README.md                         # Update: Fix links
├── QK256_ANALYSIS_INDEX.md          # Update: Fix links
├── SOLUTIONS_SUMMARY.md              # Update: Fix links (if needed)
└── ... (other files)                 # Update: Fix INDEX.md references

ci/
├── README.md                         # Update: Reference master navigation
└── scripts/
    └── validate-docs.sh              # NEW: Link validation script

.github/workflows/
└── ci.yml                            # Update: Add docs validation job
```

## Estimated Effort

- Content audit: 30 minutes
- Merge content: 1 hour
- Update links: 30 minutes
- Delete files: 10 minutes
- Link validation: 30 minutes
- Entry points: 15 minutes
- CI integration: 15 minutes
- **Total**: ~3.5 hours

## Risk Assessment

**Risk Level**: Low
- Documentation only (no code changes)
- Version control safety net
- Lychee validation prevents broken links

## References

- **Detailed Spec**: `docs/explanation/specs/SPEC-2025-004-docs-consolidation.md`
- **Lychee Config**: `.lychee.toml`
- **Current Master Nav**: `ci/solutions/00_NAVIGATION_INDEX.md`
```

---

## Issue 4: Document Environment Testing Standards with EnvGuard Guide

**Title**: Document environment testing standards with EnvGuard guide

**Labels**: `P1`, `documentation`, `test-standards`, `critical`

**Related Spec**: `docs/explanation/specs/SPEC-2025-005-envguard-testing-guide.md`

**Description**:

```markdown
## Summary

Document environment variable testing standards with comprehensive EnvGuard guide in `docs/development/test-suite.md`. This addresses **critical P0 test hygiene gap**: 39 tests mutate environment variables WITHOUT proper `#[serial(bitnet_env)]` guards, creating race conditions in parallel test execution.

## Current State

- EnvGuard pattern exists and works (7/7 tests passing from PR #475)
- Brief mention in `CLAUDE.md` (lines 345-360)
- **39 tests mutate env vars without `#[serial]` protection** (from audit report)
- No comprehensive developer guide for pattern usage

## Target State

- Comprehensive EnvGuard guide in `docs/development/test-suite.md`
- All env-mutating tests use `#[serial(bitnet_env)]` pattern
- Clear examples and anti-patterns documented
- CI check enforces pattern usage

## Acceptance Criteria

### AC1: EnvGuard Guide Section Added
- [ ] Section added to `docs/development/test-suite.md`
- [ ] ≥2000 words covering patterns, anti-patterns, examples
- [ ] ≥10 code examples (correct patterns + anti-patterns)

**Validation**:
```bash
# Check section exists
grep -A 10 "Environment Variable Testing with EnvGuard" docs/development/test-suite.md

# Count words in section
awk '/Environment Variable Testing with EnvGuard/,/^## [^E]/' \
  docs/development/test-suite.md | wc -w
# Expected: ≥2000 words

# Count code examples
awk '/Environment Variable Testing with EnvGuard/,/^## [^E]/' \
  docs/development/test-suite.md | grep -c '```rust'
# Expected: ≥10 code examples
```

### AC2: Cross-Links Established
- [ ] Link from `ci/README.md` to test-suite.md#envguard
- [ ] Link from `CLAUDE.md` to test-suite.md#envguard
- [ ] PR template checklist: "Env-mutating tests use EnvGuard?"

**Validation**:
```bash
# Check ci/README.md link
grep "test-suite.md.*EnvGuard" ci/README.md

# Check CLAUDE.md reference
grep "test-suite.md" CLAUDE.md

# Verify PR template
grep -i "envguard\|environment.*test" .github/PULL_REQUEST_TEMPLATE.md
```

### AC3: Pattern Examples Clear
- [ ] Correct pattern: `EnvGuard` + `#[serial(bitnet_env)]`
- [ ] Anti-patterns: Missing #[serial], manual set_var, guard dropped early
- [ ] Real-world examples from BitNet-rs codebase
- [ ] Common pitfalls section

**Validation**:
```bash
# Check correct pattern documented
grep -A 10 "Correct Pattern" docs/development/test-suite.md | grep "#\[serial(bitnet_env)\]"

# Check anti-patterns
grep -A 5 "ANTI-PATTERN\|WRONG:" docs/development/test-suite.md | wc -l
# Expected: ≥3 anti-pattern examples

# Check real-world examples
grep -A 5 "Real-World Examples" docs/development/test-suite.md
```

### AC4: CI Validation Documented
- [ ] Commands for running env-aware tests
- [ ] Nextest profile configuration explained
- [ ] Audit commands for detecting non-compliant tests

**Validation**:
```bash
# Check CI commands
grep "cargo test.*serial" docs/development/test-suite.md
grep "cargo nextest" docs/development/test-suite.md

# Check audit commands
grep "grep.*set_var" docs/development/test-suite.md
```

## Implementation Guide

### Phase 1: Draft EnvGuard Guide Section (2 hours)

**Section Structure**:
```markdown
## Environment Variable Testing with EnvGuard

### Overview
- Explain race condition problem in parallel tests

### The Problem: Race Conditions
- Anti-pattern example (without EnvGuard)
- Issues: flaky failures, test pollution, non-determinism

### The Solution: EnvGuard + #[serial]
- Correct pattern example
- Benefits: automatic cleanup, serial execution, no pollution

### Implementation Guide
- Step 1: Add dependencies
- Step 2: Use EnvGuard pattern
- Step 3: Multiple environment variables

### Common Pitfalls
- Pitfall 1: Missing #[serial] attribute
- Pitfall 2: Manual env mutation
- Pitfall 3: Forgetting guard lifetime

### Running EnvGuard Tests
- Commands for serial execution
- Nextest configuration

### CI Configuration
- Nextest profile settings
- CI workflow examples

### When to Use EnvGuard
- Use cases
- Don't use cases

### BitNet-rs Environment Variables
- List of common env vars requiring EnvGuard

### EnvGuard Implementation
- Reference to source code

### Real-World Examples
- Example 1: Deterministic inference
- Example 2: Strict mode validation
- Example 3: GPU fallback

### Validation Checklist
- Steps for adding env-mutating tests
```

### Phase 2: Add Code Examples (1 hour)

**Correct Pattern**:
```rust
use serial_test::serial;
use tests::helpers::env_guard::EnvGuard;

#[test]
#[serial(bitnet_env)]  // Critical: prevents parallel execution
fn test_determinism() {
    let _guard = EnvGuard::new("BITNET_DETERMINISTIC", "1");
    // Test logic - env automatically restored on drop
}
```

**Anti-Pattern: Missing #[serial]**:
```rust
// WRONG: EnvGuard without #[serial] - still has race condition!
#[test]
fn test_determinism() {
    let _guard = EnvGuard::new("BITNET_DETERMINISTIC", "1");
    // Can still run in parallel with other env tests - BUG!
}
```

**Anti-Pattern: Manual set_var**:
```rust
// WRONG: Manual set_var without cleanup
#[test]
#[serial(bitnet_env)]
fn test_determinism() {
    std::env::set_var("BITNET_DETERMINISTIC", "1");
    // If test panics, env not restored - BUG!
}
```

**Real-World Example**:
```rust
#[test]
#[serial(bitnet_env)]
fn test_deterministic_inference_with_seed() {
    let _det_guard = EnvGuard::new("BITNET_DETERMINISTIC", "1");
    let _seed_guard = EnvGuard::new("BITNET_SEED", "42");
    let _threads_guard = EnvGuard::new("RAYON_NUM_THREADS", "1");

    let result1 = run_inference();
    let result2 = run_inference();
    assert_eq!(result1, result2);
}
```

### Phase 3: Cross-Link Integration (30 minutes)

**Update `ci/README.md`**:
```markdown
## Testing Standards

For environment variable testing patterns, see:
- [EnvGuard Guide](../docs/development/test-suite.md#environment-variable-testing-with-envguard)
```

**Update `CLAUDE.md`** (expand existing section at lines 345-360):
```markdown
### Test Isolation

**EnvGuard Pattern**: Use `#[serial(bitnet_env)]` for tests that mutate environment variables.

For comprehensive guide, see: [Test Suite Guide - EnvGuard](docs/development/test-suite.md#environment-variable-testing-with-envguard)

```rust
// Brief example...
```
```

**Update `.github/PULL_REQUEST_TEMPLATE.md`**:
```markdown
## Testing Checklist

- [ ] All tests pass locally
- [ ] New env-mutating tests use `EnvGuard` + `#[serial(bitnet_env)]`
- [ ] No race conditions in parallel execution
```

### Phase 4: CI Validation Section (30 minutes)

**Document Commands**:
```bash
# Run all tests (env-aware tests execute serially)
cargo test --workspace --no-default-features --features cpu

# Run only env-aware tests
cargo test --workspace --no-default-features --features cpu -- serial

# Nextest (recommended)
cargo nextest run --profile ci --workspace --no-default-features --features cpu
```

**Audit Non-Compliant Tests**:
```bash
# Find all tests using std::env::set_var
grep -rn "std::env::set_var" --include="*.rs" crates/*/tests/

# Check which have #[serial(bitnet_env)]
grep -B 3 "std::env::set_var" crates/*/tests/*.rs | grep "#\[serial"

# Identify non-compliant (manual review of diff)
```

### Phase 5: Review and Validation (1 hour)

**Checklist**:
- [ ] Self-review guide as first-time developer
- [ ] Verify all code examples compile
- [ ] Test cross-links (click-through)
- [ ] Run audit commands
- [ ] Update word/example counts
- [ ] Commit guide section

## Files to Modify

```
docs/development/
└── test-suite.md                     # Update: Add EnvGuard section

ci/
└── README.md                         # Update: Add EnvGuard link

CLAUDE.md                             # Update: Expand EnvGuard reference

.github/
└── PULL_REQUEST_TEMPLATE.md         # Update: Add checklist item
```

## Optional: Migrate Non-Compliant Tests

**If time permits** (additional 3-5 hours):
```bash
# Audit non-compliant tests
grep -rn "std::env::set_var" crates/*/tests/ > /tmp/env-mutations.txt

# For each test (39 total from audit):
# 1. Add `#[serial(bitnet_env)]`
# 2. Replace `std::env::set_var` → `EnvGuard::new`
# 3. Run test to verify
# 4. Commit migration
```

## Estimated Effort

- Guide writing: 2 hours
- Code examples: 1 hour
- Cross-linking: 30 minutes
- CI validation: 30 minutes
- Review: 1 hour
- **Total**: ~5 hours

**Optional Migration**: +3-5 hours (39 non-compliant tests)

## Risk Assessment

**Risk Level**: Medium (documentation: Low, optional migration: Medium)
- Documentation changes: Low risk
- Test migration: Medium risk (must verify no race conditions)

**Critical Issue**: 39 tests currently have race condition risk - documentation addresses awareness, migration fixes root cause.

## References

- **Detailed Spec**: `docs/explanation/specs/SPEC-2025-005-envguard-testing-guide.md`
- **EnvGuard Implementation**: `tests/helpers/env_guard.rs`
- **Environment Variables**: `docs/environment-variables.md`
- **Test Audit Report**: `COMPREHENSIVE_TEST_AUDIT_REPORT.md` (section 2.1)
```

---

## Summary Table

| Issue | Title | Effort | Priority | Risk | Dependencies |
|-------|-------|--------|----------|------|--------------|
| 1 | Persistent GGUF fixtures | 2h | P1 | Low | None (fixtures exist) |
| 2 | Classify ignored tests | 6h | P1 | Low | Test audit complete |
| 3 | Docs consolidation | 3.5h | P1 | Low | None |
| 4 | EnvGuard guide | 5h (+3-5h optional) | P1 | Medium | EnvGuard exists |

**Total Estimated Effort**: 16.5 hours (core) + 3-5 hours (optional migration)

---

## Next Steps

1. **Copy Issue Templates**: Copy markdown from each issue section to GitHub
2. **Create Issues**: Create 4 GitHub issues with appropriate labels
3. **Assign Ownership**: Assign to appropriate team members
4. **Track Progress**: Link issues to project board/milestone
5. **Implementation**: Follow implementation guides in each spec

---

**Last Updated**: 2025-10-23
**Created By**: BitNet-rs Spec Analyzer Agent
**Status**: Ready for GitHub Issue Creation
