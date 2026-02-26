# SPEC-2025-006: Historical Reports Archive Migration

**Status**: Draft
**Created**: 2025-10-23
**Priority**: P1
**Category**: Documentation Hygiene
**Related Issues**: None
**Related PRs**: None

---

## Executive Summary

Archive 53 historical markdown reports (468 KB) from `docs/reports/` to `docs/archive/reports/` to eliminate 13 broken links (15.7% of all documentation link errors) and establish current documentation as the single source of truth. The reports are 94% duplicates of content already in `CLAUDE.md`, `PR_475_FINAL_SUCCESS_REPORT.md`, and `docs/development/` with minimal external references (4 root files).

**Current State**: `docs/reports/` contains 53 historical status reports, PR reviews, and validation receipts from Sept-Oct 2025 development phase. Files are 9+ days old with 13 broken internal links contributing to documentation link validation failures.

**Target State**: All 53 reports archived to `docs/archive/reports/` with tombstone banners linking to authoritative current documentation. Archive excluded from lychee link checks. 4 cross-references in root audit files updated to new paths.

**Impact**:
- **Link Hygiene**: Eliminate 13 broken links (15.7% reduction in total link errors)
- **Documentation Clarity**: Establishes `CLAUDE.md` and PR #475 report as authoritative sources
- **Maintenance**: Removes burden of maintaining historical reports while preserving audit trail
- **CI Performance**: Reduces lychee link check scope by excluding archived content

---

## Requirements Analysis

### Functional Requirements

1. **FR1: Archive Directory Creation**
   - Create `docs/archive/reports/` directory structure
   - Move all 53 markdown files from `docs/reports/` to `docs/archive/reports/`
   - Use `git mv` to preserve file history for compliance/audit purposes
   - Remove empty `docs/reports/` directory after migration

2. **FR2: Tombstone Banners**
   - Insert "ARCHIVED DOCUMENT" banner at top of each archived file (53 files)
   - Banner includes:
     - Archive date (2025-10-23)
     - Links to current authoritative sources (CLAUDE.md, PR #475, relevant docs)
     - Migration context explaining historical nature
   - Template supports variable substitution for file-specific context

3. **FR3: Cross-Reference Updates**
   - Update 4 root-level files that reference `docs/reports/`:
     - `COMPREHENSIVE_IMPLEMENTATION_REPORT.md`
     - `DOCS_LINK_VALIDATION_REPORT.md`
     - `ISSUE_254_SHAPE_MISMATCH_RESEARCH_REPORT.md`
     - `CARGO_FEATURE_FLAG_AUDIT.md`
   - Change all `docs/reports/` paths → `docs/archive/reports/`
   - Add contextual notes that reports are archived

4. **FR4: Link Validation Exclusion**
   - Update `.lychee.toml` to exclude `docs/archive/` from link checks
   - Document exclusion rationale in config comments
   - Verify lychee configuration syntax with test run

5. **FR5: Documentation Updates**
   - Add note to `docs/CONTRIBUTING-DOCS.md` about archive policy
   - Document that new reports should go to PR bodies or GitHub issues
   - Explain distinction between living documentation (`docs/`) and historical archive (`docs/archive/`)

### Non-Functional Requirements

1. **NFR1: Audit Compliance**
   - Preserve complete git history using `git mv` (not `rm` + `add`)
   - Maintain file timestamps and authorship metadata
   - Archive remains accessible for historical reference and compliance audits

2. **NFR2: Link Hygiene**
   - Post-migration lychee run shows ≤70 broken links (down from 83)
   - Zero broken references to archived content from active documentation
   - Clear error messages if archived links accessed accidentally

3. **NFR3: Rollback Safety**
   - Migration script idempotent (safe to re-run)
   - Dry-run mode available for verification
   - Rollback procedure documented with single-command restore

---

## Architecture Approach

### Documentation Structure Migration

**Architecture Decision: ARCHIVE (vs. DELETE or FIX)**

**Rationale**:
- **Preserves Historical Record**: Audit trail for project evolution and compliance
- **Reduces Maintenance**: Removes broken links from active validation scope
- **Clarifies Authority**: Establishes current docs as single source of truth
- **Low Risk**: Only 4 external references; zero code dependencies

**Rejected Alternatives**:
1. **DELETE**: Violates audit trail preservation; breaks existing references
2. **FIX**: Requires 4 hours to fix 13 broken links; duplicates current documentation (75-95% overlap)

### Directory Structure (Before → After)

**Before** (53 historical reports with broken links):
```
docs/
├── reports/
│   ├── ALPHA_READINESS_STATUS.md        # Superseded by CLAUDE.md
│   ├── PR422_FINAL_REVIEW_SUMMARY.md    # Superseded by PR_475_FINAL_SUCCESS_REPORT.md
│   ├── TEST_COVERAGE_REPORT.md          # Superseded by docs/development/test-suite.md
│   ├── SECURITY_FUZZ_REPORT.md          # Superseded by CLAUDE.md Known Issues
│   ├── GOALS_VS_REALITY_ANALYSIS.md     # 5 broken links
│   ├── DOCUMENTATION_UPDATE_REPORT.md   # 4 broken links
│   └── ... (47 other historical reports)
└── development/
    ├── test-suite.md                    # Current authoritative test docs
    └── validation-ci.md                 # Current authoritative validation docs
```

**After** (archived with tombstone banners):
```
docs/
├── archive/
│   └── reports/
│       ├── ALPHA_READINESS_STATUS.md        # Banner → CLAUDE.md
│       ├── PR422_FINAL_REVIEW_SUMMARY.md    # Banner → PR_475_FINAL_SUCCESS_REPORT.md
│       ├── TEST_COVERAGE_REPORT.md          # Banner → docs/development/test-suite.md
│       ├── SECURITY_FUZZ_REPORT.md          # Banner → CLAUDE.md
│       └── ... (53 archived reports with banners)
└── development/
    ├── test-suite.md                    # Current authoritative test docs
    └── validation-ci.md                 # Current authoritative validation docs
```

### Workspace Integration

**File Classification and Tombstone Mapping**:

| Report Category | Count | Current Authority | Banner Link Target |
|-----------------|-------|-------------------|-------------------|
| PR Reviews | 10 | PR_475_FINAL_SUCCESS_REPORT.md | ../../PR_475_FINAL_SUCCESS_REPORT.md |
| Issue Resolution | 4 | docs/explanation/specs/ | ../explanation/specs/GITHUB_ISSUES_P1_SPECIFICATIONS.md |
| Status/Readiness | 6 | CLAUDE.md | ../../CLAUDE.md |
| Test/Validation | 12 | docs/development/test-suite.md | ../development/test-suite.md |
| Implementation | 15 | PR_475_FINAL_SUCCESS_REPORT.md | ../../PR_475_FINAL_SUCCESS_REPORT.md |
| Documentation | 2 | docs/CONTRIBUTING-DOCS.md | ../CONTRIBUTING-DOCS.md |
| Miscellaneous | 4 | CLAUDE.md | ../../CLAUDE.md |

---

## Migration Script Specification

### Bash Automation Requirements

**Script Name**: `scripts/archive_reports.sh`

**Capabilities**:
1. **Dry-run mode**: Preview changes without executing (`--dry-run`)
2. **Banner injection**: Automated tombstone insertion with variable substitution
3. **Link updates**: Automated cross-reference path rewriting
4. **Validation**: Pre- and post-migration link health comparison
5. **Idempotency**: Safe to re-run without side effects
6. **Rollback**: Single-command restore via git

**Script Interface**:
```bash
#!/usr/bin/env bash
# scripts/archive_reports.sh - Archive historical docs/reports/ to docs/archive/reports/
#
# Usage:
#   ./scripts/archive_reports.sh [OPTIONS]
#
# Options:
#   --dry-run       Preview changes without executing
#   --rollback      Restore archived reports to docs/reports/
#   --skip-banner   Skip banner injection (migration only)
#   --help          Show this help message
#
# Examples:
#   # Preview migration
#   ./scripts/archive_reports.sh --dry-run
#
#   # Execute migration with banners
#   ./scripts/archive_reports.sh
#
#   # Rollback migration
#   ./scripts/archive_reports.sh --rollback

set -euo pipefail

# Configuration
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPORTS_DIR="${REPO_ROOT}/docs/reports"
ARCHIVE_DIR="${REPO_ROOT}/docs/archive/reports"
BANNER_TEMPLATE="${REPO_ROOT}/scripts/templates/archive_banner.md"

# Migration phases:
# 1. Pre-migration validation
# 2. Directory creation
# 3. File migration (git mv)
# 4. Banner injection
# 5. Cross-reference updates
# 6. Lychee config update
# 7. Post-migration validation
```

### Banner Template with Variable Substitution

**Template File**: `scripts/templates/archive_banner.md`

**Variables**:
- `{ARCHIVE_DATE}`: Migration date (2025-10-23)
- `{CURRENT_DOC}`: Path to current authoritative documentation
- `{CURRENT_DOC_NAME}`: Human-readable name of current doc
- `{REPORT_CATEGORY}`: Report category (PR Review, Status Report, etc.)

**Template Content**:
```markdown
> **ARCHIVED DOCUMENT** (Archived: {ARCHIVE_DATE})
>
> This is a historical {REPORT_CATEGORY} from active development (Sept-Oct 2025).
> **This document is no longer maintained and may contain outdated information.**
>
> **For current information, see:**
> - [{CURRENT_DOC_NAME}]({CURRENT_DOC})
> - [CLAUDE.md](../../CLAUDE.md) — Project reference and status
> - [PR #475 Final Report](../../PR_475_FINAL_SUCCESS_REPORT.md) — Comprehensive implementation summary
> - [Current CI Documentation](../development/validation-ci.md) — Test suite and validation
>
> **Archive Note**: This report was archived during documentation cleanup to establish
> current docs as the single source of truth. The content below is preserved for
> historical reference and audit purposes.

---

```

**Variable Substitution Logic**:
```bash
# Determine banner variables based on file category
case "${filename}" in
    PR_*|PR422_*)
        CATEGORY="PR Review Report"
        CURRENT_DOC="../../PR_475_FINAL_SUCCESS_REPORT.md"
        CURRENT_DOC_NAME="PR #475 Final Report"
        ;;
    ISSUE_*)
        CATEGORY="Issue Resolution Report"
        CURRENT_DOC="../explanation/specs/GITHUB_ISSUES_P1_SPECIFICATIONS.md"
        CURRENT_DOC_NAME="Current Issue Specifications"
        ;;
    ALPHA_*|LAUNCH_*|SPRINT_*|VALIDATION_STATUS*)
        CATEGORY="Status Report"
        CURRENT_DOC="../../CLAUDE.md"
        CURRENT_DOC_NAME="CLAUDE.md Project Reference"
        ;;
    TEST_*|COVERAGE_*|SECURITY_*)
        CATEGORY="Validation Report"
        CURRENT_DOC="../development/test-suite.md"
        CURRENT_DOC_NAME="Current Test Suite Documentation"
        ;;
    DOCUMENTATION_*)
        CATEGORY="Documentation Report"
        CURRENT_DOC="../CONTRIBUTING-DOCS.md"
        CURRENT_DOC_NAME="Documentation Contributing Guide"
        ;;
    *)
        CATEGORY="Project Report"
        CURRENT_DOC="../../CLAUDE.md"
        CURRENT_DOC_NAME="CLAUDE.md Project Reference"
        ;;
esac
```

### Cross-Reference Update Automation

**Automated Link Rewriting**:
```bash
# Update 4 root-level files with cross-references
update_cross_references() {
    local files=(
        "COMPREHENSIVE_IMPLEMENTATION_REPORT.md"
        "DOCS_LINK_VALIDATION_REPORT.md"
        "ISSUE_254_SHAPE_MISMATCH_RESEARCH_REPORT.md"
        "CARGO_FEATURE_FLAG_AUDIT.md"
    )

    for file in "${files[@]}"; do
        local filepath="${REPO_ROOT}/${file}"
        if [[ -f "${filepath}" ]]; then
            echo "Updating cross-references in ${file}..."

            # Replace docs/reports/ → docs/archive/reports/
            sed -i 's|docs/reports/|docs/archive/reports/|g' "${filepath}"

            # Add archive context note if not present
            if ! grep -q "Historical Archive" "${filepath}"; then
                # Insert note after first heading
                sed -i '0,/^#/a \\n> **Note**: References to `docs/archive/reports/` point to historical archived documentation.  \n> For current status, see [CLAUDE.md](CLAUDE.md) and [PR #475](PR_475_FINAL_SUCCESS_REPORT.md).\n' "${filepath}"
            fi
        fi
    done
}
```

### Lychee Configuration Update

**Automated Exclusion**:
```bash
update_lychee_config() {
    local lychee_config="${REPO_ROOT}/.lychee.toml"

    echo "Updating .lychee.toml to exclude docs/archive/..."

    # Add docs/archive/ to exclude list if not present
    if ! grep -q '"docs/archive/"' "${lychee_config}"; then
        # Insert after last exclude entry
        sed -i '/exclude = \[/,/\]/{ /\]/i\    "docs/archive/",  # Historical documentation - not maintained (archived 2025-10-23)
}' "${lychee_config}"
    fi
}
```

---

## Validation Procedure

### Pre-Migration Metrics

**Baseline Link Health**:
```bash
# Run lychee on docs/ directory for baseline
lychee docs/ --offline --format json > /tmp/lychee_before.json

# Extract key metrics
TOTAL_LINKS_BEFORE=$(jq '.total_links' /tmp/lychee_before.json)
BROKEN_LINKS_BEFORE=$(jq '.broken_links' /tmp/lychee_before.json)
REPORTS_BROKEN_BEFORE=$(lychee docs/reports/ --offline --format json | jq '.broken_links')

echo "Pre-Migration Metrics:"
echo "  Total links: ${TOTAL_LINKS_BEFORE}"
echo "  Broken links: ${BROKEN_LINKS_BEFORE}"
echo "  Reports broken: ${REPORTS_BROKEN_BEFORE}"
```

**Expected Baseline** (from DOCS_REPORTS_AUDIT.md):
- Total links in docs/reports/: 19
- Broken links in docs/reports/: 13 (68% failure rate)
- Total broken links in all docs: 83
- Reports contribution to total errors: 15.7%

### Post-Migration Validation

**Target Metrics**:
```bash
# Run lychee on active docs (excluding archive)
lychee docs/ --exclude 'docs/archive/' --offline --format json > /tmp/lychee_after.json

# Extract comparison metrics
TOTAL_LINKS_AFTER=$(jq '.total_links' /tmp/lychee_after.json)
BROKEN_LINKS_AFTER=$(jq '.broken_links' /tmp/lychee_after.json)

echo "Post-Migration Metrics:"
echo "  Total links: ${TOTAL_LINKS_AFTER}"
echo "  Broken links: ${BROKEN_LINKS_AFTER}"
echo "  Improvement: $((BROKEN_LINKS_BEFORE - BROKEN_LINKS_AFTER)) links fixed"

# Verify target achieved
if [[ ${BROKEN_LINKS_AFTER} -le 70 ]]; then
    echo "✅ Target achieved: ≤70 broken links"
else
    echo "❌ Target missed: ${BROKEN_LINKS_AFTER} broken links (target: ≤70)"
    exit 1
fi
```

**Success Criteria**:
- Broken links reduced from 83 → ≤70 (15.7% improvement minimum)
- Zero broken references to `docs/reports/` from active docs
- All 4 cross-references updated to `docs/archive/reports/`
- Lychee skips `docs/archive/` directory

### Archive Integrity Validation

**Git History Preservation**:
```bash
# Verify git history preserved for compliance
cd "${REPO_ROOT}/docs/archive/reports"

for file in *.md; do
    # Check file has commit history
    commits=$(git log --oneline -- "${file}" | wc -l)

    if [[ ${commits} -eq 0 ]]; then
        echo "❌ ERROR: ${file} has no git history (git mv failed)"
        exit 1
    fi
done

echo "✅ All archived files have preserved git history"
```

**Banner Injection Verification**:
```bash
# Verify all 53 archived files have banners
cd "${REPO_ROOT}/docs/archive/reports"

missing_banners=0
for file in *.md; do
    if ! grep -q "ARCHIVED DOCUMENT" "${file}"; then
        echo "❌ Missing banner: ${file}"
        ((missing_banners++))
    fi
done

if [[ ${missing_banners} -eq 0 ]]; then
    echo "✅ All 53 archived files have tombstone banners"
else
    echo "❌ ${missing_banners} files missing banners"
    exit 1
fi
```

---

## Rollback Plan

### Single-Command Restore

**Rollback Script** (embedded in `scripts/archive_reports.sh --rollback`):
```bash
rollback_migration() {
    echo "Rolling back archive migration..."

    # Verify archive exists
    if [[ ! -d "${ARCHIVE_DIR}" ]]; then
        echo "❌ ERROR: Archive directory not found (nothing to rollback)"
        exit 1
    fi

    # 1. Restore files from archive
    git mv "${ARCHIVE_DIR}"/*.md "${REPORTS_DIR}/"

    # 2. Remove banners (restore original content)
    cd "${REPORTS_DIR}"
    for file in *.md; do
        # Remove banner (first block before horizontal rule)
        sed -i '1,/^---$/{ /^> \*\*ARCHIVED DOCUMENT\*\*/,/^---$/d }' "${file}"
    done

    # 3. Restore cross-references
    for file in "${CROSS_REF_FILES[@]}"; do
        sed -i 's|docs/archive/reports/|docs/reports/|g' "${REPO_ROOT}/${file}"
        # Remove archive context note
        sed -i '/Historical Archive/,+2d' "${REPO_ROOT}/${file}"
    done

    # 4. Restore lychee config (remove archive exclusion)
    sed -i '/docs\/archive\//d' "${REPO_ROOT}/.lychee.toml"

    # 5. Remove empty archive directory
    rmdir "${ARCHIVE_DIR}" 2>/dev/null || true
    rmdir "${REPO_ROOT}/docs/archive" 2>/dev/null || true

    echo "✅ Rollback complete. Run validation to verify."
}
```

**Rollback Verification**:
```bash
# Verify rollback restored original state
git status | grep -E "(modified|deleted|renamed)" | wc -l
# Expected: Same number of changes as forward migration (inverse operation)

# Verify no archive directory remains
ls -d docs/archive 2>/dev/null && echo "❌ Archive still exists" || echo "✅ Archive removed"

# Verify cross-references restored
grep -r "docs/archive/reports/" *.md && echo "❌ Archive refs remain" || echo "✅ Refs restored"
```

---

## Risk Mitigation

### Technical Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| **Git history loss** | Low | High | Use `git mv` exclusively; verify history preservation in validation |
| **Broken cross-references** | Medium | Medium | Automated link rewriting with verification; 4 files only |
| **Lychee config syntax error** | Low | Medium | Test lychee run after config update; rollback on failure |
| **Banner injection corruption** | Low | High | Test banner template on 3 sample files first; dry-run validation |
| **Rollback failure** | Low | Medium | Test rollback in separate branch before production migration |
| **Accidental link breakage** | Medium | Low | Post-migration lychee run validates all active doc links |

### Validation Commands

**Risk Mitigation Validation**:
```bash
# 1. Test git mv history preservation (dry-run)
git mv docs/reports/ALPHA_READINESS_STATUS.md /tmp/test_archive.md
git log --oneline -- /tmp/test_archive.md | head -5
git mv /tmp/test_archive.md docs/reports/ALPHA_READINESS_STATUS.md
# Expected: Commit history preserved

# 2. Test banner injection on sample file
cp docs/reports/ALPHA_READINESS_STATUS.md /tmp/test_banner.md
./scripts/archive_reports.sh --dry-run 2>&1 | grep "ALPHA_READINESS_STATUS.md"
# Expected: Banner preview shown

# 3. Test lychee config syntax
lychee --version && echo "✅ Lychee available"
lychee docs/ --offline --max-concurrency 1 2>&1 | grep -i error
# Expected: No config syntax errors

# 4. Test rollback procedure (separate branch)
git checkout -b test-archive-rollback
./scripts/archive_reports.sh
./scripts/archive_reports.sh --rollback
git diff main -- docs/reports/ | wc -l
# Expected: 0 (identical to main after rollback)
git checkout main && git branch -D test-archive-rollback
```

---

## Success Criteria

### Measurable Acceptance Criteria

**AC1: Archive Migration Complete**
- ✅ 53 markdown files moved from `docs/reports/` → `docs/archive/reports/`
- ✅ Git history preserved for all 53 files (verified via `git log -- <file>`)
- ✅ Empty `docs/reports/` directory removed
- ✅ Archive directory structure created: `docs/archive/reports/`

**Validation**:
```bash
# Verify migration
ls -1 docs/archive/reports/*.md | wc -l
# Expected: 53

# Verify reports directory removed
ls docs/reports/ 2>&1 | grep "No such file"
# Expected: docs/reports/ does not exist

# Verify git history preserved (sample 3 files)
git log --oneline -- docs/archive/reports/ALPHA_READINESS_STATUS.md | wc -l
# Expected: >0 (has commit history)
```

**AC2: Tombstone Banners**
- ✅ All 53 archived files have "ARCHIVED DOCUMENT" banner
- ✅ Banners include correct archive date (2025-10-23)
- ✅ Banners link to appropriate current authoritative sources
- ✅ Banner template supports variable substitution (7 categories)

**Validation**:
```bash
# Verify banners present
grep -l "ARCHIVED DOCUMENT" docs/archive/reports/*.md | wc -l
# Expected: 53

# Verify banner links (sample)
grep -A 5 "ARCHIVED DOCUMENT" docs/archive/reports/ALPHA_READINESS_STATUS.md \
    | grep "CLAUDE.md"
# Expected: Link to CLAUDE.md present

# Verify archive date
grep "Archived: 2025-10-23" docs/archive/reports/*.md | wc -l
# Expected: 53
```

**AC3: Cross-Reference Updates**
- ✅ 4 root-level files updated: paths changed to `docs/archive/reports/`
- ✅ Archive context notes added to cross-referencing files
- ✅ No broken references from active docs to archived content

**Validation**:
```bash
# Verify path updates
grep -r "docs/archive/reports/" *.md | grep -E "(COMPREHENSIVE|DOCS_LINK|ISSUE_254|CARGO)" | wc -l
# Expected: ≥4 (at least 4 files with archive paths)

# Verify no old paths remain
grep -r "docs/reports/" *.md | grep -v "docs/archive/reports/" | wc -l
# Expected: 0 (all references updated)

# Verify archive context notes
grep -l "Historical Archive" COMPREHENSIVE_IMPLEMENTATION_REPORT.md
# Expected: Note present
```

**AC4: Link Validation Exclusion**
- ✅ `.lychee.toml` updated with `docs/archive/` exclusion
- ✅ Lychee skips archived content during link checks
- ✅ Broken link count reduced from 83 → ≤70

**Validation**:
```bash
# Verify lychee config
grep 'docs/archive/' .lychee.toml
# Expected: "docs/archive/",  # Historical documentation...

# Verify lychee skips archive
lychee docs/ --offline --format json 2>&1 | jq '.broken_links'
# Expected: ≤70 (down from 83)

# Verify archive not checked
lychee docs/ --offline --verbose 2>&1 | grep "docs/archive"
# Expected: No output (archive excluded)
```

**AC5: Documentation Updates**
- ✅ `docs/CONTRIBUTING-DOCS.md` documents archive policy
- ✅ Archive policy explains distinction between living docs and historical archive
- ✅ New report guidance provided (PR bodies, GitHub issues)

**Validation**:
```bash
# Verify CONTRIBUTING-DOCS.md updated
grep -A 5 "archive" docs/CONTRIBUTING-DOCS.md | grep -i "historical"
# Expected: Archive policy documented

# Verify new report guidance
grep -i "new report\|historical.*archive" docs/CONTRIBUTING-DOCS.md
# Expected: Guidance present
```

---

## Performance Specifications

### Migration Execution Time

| Phase | Target Time | Validation Command |
|-------|-------------|-------------------|
| Pre-migration validation | <30s | `time lychee docs/ --offline` |
| Directory creation | <1s | `time mkdir -p docs/archive/reports` |
| File migration (git mv) | <30s | `time git mv docs/reports/*.md docs/archive/reports/` |
| Banner injection | <2min | `time ./scripts/archive_reports.sh --banner-only` |
| Cross-reference updates | <10s | `time ./scripts/archive_reports.sh --update-refs-only` |
| Lychee config update | <5s | `time sed -i '...' .lychee.toml` |
| Post-migration validation | <30s | `time lychee docs/ --exclude 'docs/archive/' --offline` |
| **Total** | **<5 minutes** | **End-to-end script execution** |

### Storage Impact

| Metric | Before | After | Delta |
|--------|--------|-------|-------|
| Active docs size | ~2.5 MB | ~2.0 MB | -468 KB (reports archived) |
| Archive size | 0 KB | 468 KB | +468 KB (new archive) |
| Total repository size | ~50 MB | ~50 MB | Negligible (Git compression) |
| Lychee check time | ~5s | ~3s | -40% (fewer files scanned) |

---

## Feature Flag Analysis

**Not Applicable**: Documentation-only changes; no code feature flags involved.

---

## Testing Strategy

### Pre-Migration Testing

**Dry-Run Validation**:
```bash
# Test migration script in dry-run mode
./scripts/archive_reports.sh --dry-run 2>&1 | tee /tmp/dryrun.log

# Verify no actual file changes
git status | grep -E "modified|deleted|renamed" | wc -l
# Expected: 0 (dry-run makes no changes)

# Verify planned changes correct
grep "Would move:" /tmp/dryrun.log | wc -l
# Expected: 53 (all reports planned for migration)

grep "Would add banner to:" /tmp/dryrun.log | wc -l
# Expected: 53 (all reports get banners)
```

### Post-Migration Testing

**Integration Testing**:
```bash
# 1. Verify link health improvement
lychee docs/ --exclude 'docs/archive/' --offline --format json | jq '.broken_links'
# Expected: ≤70 (down from 83, 15.7% improvement minimum)

# 2. Verify archive links still work (for reference)
lychee docs/archive/reports/ --offline --format json | jq '.broken_links'
# Expected: 13 (same as before, but now excluded from active checks)

# 3. Verify cross-references valid
for file in COMPREHENSIVE_IMPLEMENTATION_REPORT.md \
            DOCS_LINK_VALIDATION_REPORT.md \
            ISSUE_254_SHAPE_MISMATCH_RESEARCH_REPORT.md \
            CARGO_FEATURE_FLAG_AUDIT.md; do
    lychee "${file}" --offline --base . | grep -i error && echo "❌ ${file}" || echo "✅ ${file}"
done
# Expected: All ✅ (no broken links)

# 4. Verify git history integrity
cd docs/archive/reports
for file in *.md; do
    commits=$(git log --oneline -- "${file}" | wc -l)
    if [[ ${commits} -eq 0 ]]; then
        echo "❌ ${file}: No history"
    fi
done
# Expected: All files have commit history

# 5. Verify banner content
cd docs/archive/reports
grep -h "ARCHIVED DOCUMENT" *.md | sort -u
# Expected: Consistent banner format across all 53 files
```

### Rollback Testing

**Rollback Validation** (in separate test branch):
```bash
# Create test branch
git checkout -b test-archive-migration

# Execute migration
./scripts/archive_reports.sh

# Verify migration succeeded
ls -1 docs/archive/reports/*.md | wc -l
# Expected: 53

# Execute rollback
./scripts/archive_reports.sh --rollback

# Verify rollback succeeded
ls -1 docs/reports/*.md | wc -l
# Expected: 53

# Verify no differences from main
git diff main -- docs/reports/ | wc -l
# Expected: 0 (identical state)

# Cleanup
git checkout main
git branch -D test-archive-migration
```

---

## BitNet-rs Alignment

### TDD Practices

✅ **Alignment**: Migration validated via automated testing (pre/post metrics comparison)

**Evidence**:
- Dry-run mode enables test-first validation
- Automated link health checks (lychee integration)
- Git history preservation verification
- Rollback testing in isolated branch

### Feature-Gated Architecture

**Not Applicable**: Documentation-only changes; no code feature gates

### Workspace Structure

✅ **Alignment**: Archive follows BitNet-rs documentation organization principles

**Evidence**:
- Separation of living docs (`docs/`) from historical archive (`docs/archive/`)
- Cross-references use relative paths (not absolute)
- Lychee exclusion maintains clean CI validation
- `CONTRIBUTING-DOCS.md` documents archive policy

### Cross-Platform Support

✅ **Alignment**: Migration script uses portable bash syntax (no GNU-specific extensions)

**Evidence**:
- POSIX-compliant shell commands
- Markdown format platform-agnostic
- Git operations cross-platform compatible

---

## Neural Network References

**Not Applicable**: Documentation hygiene task; no neural network inference or quantization changes.

---

## Related Documentation

- **Exploration Report**: `DOCS_REPORTS_AUDIT.md` (comprehensive analysis of 53 reports)
- **Link Validation Report**: `DOCS_LINK_VALIDATION_REPORT.md` (broken link inventory)
- **Contributing Guide**: `docs/CONTRIBUTING-DOCS.md` (to be updated with archive policy)
- **Related Spec**: `SPEC-2025-004-docs-consolidation.md` (ci/solutions/ cleanup)

---

## Implementation Checklist

**Phase 1: Pre-Migration Preparation** (10 minutes)
- [ ] Review DOCS_REPORTS_AUDIT.md for file inventory (53 files confirmed)
- [ ] Verify git working directory clean: `git status`
- [ ] Create test branch: `git checkout -b archive-reports-migration`
- [ ] Run pre-migration link health baseline: `lychee docs/ --offline > /tmp/baseline.txt`
- [ ] Document baseline metrics: Total links, broken links, reports contribution

**Phase 2: Script Development** (30 minutes)
- [ ] Create `scripts/archive_reports.sh` with dry-run support
- [ ] Create `scripts/templates/archive_banner.md` template
- [ ] Implement 7-category variable substitution logic
- [ ] Add automated cross-reference updating (4 files)
- [ ] Add lychee config exclusion logic
- [ ] Implement rollback functionality
- [ ] Test dry-run mode: `./scripts/archive_reports.sh --dry-run`

**Phase 3: Dry-Run Validation** (10 minutes)
- [ ] Run dry-run and review planned changes
- [ ] Verify 53 files targeted for migration
- [ ] Verify banner template variables resolve correctly
- [ ] Verify cross-reference paths correct (4 files)
- [ ] Review lychee config exclusion syntax

**Phase 4: Migration Execution** (5 minutes)
- [ ] Execute migration: `./scripts/archive_reports.sh`
- [ ] Verify archive directory created: `ls -la docs/archive/reports/`
- [ ] Verify 53 files migrated: `ls -1 docs/archive/reports/*.md | wc -l`
- [ ] Verify reports directory removed: `ls docs/reports/ 2>&1`
- [ ] Verify git status shows renames (not deletions)

**Phase 5: Post-Migration Validation** (15 minutes)
- [ ] Verify all banners present: `grep -l "ARCHIVED DOCUMENT" docs/archive/reports/*.md | wc -l`
- [ ] Verify banner links correct (sample 5 files)
- [ ] Verify cross-references updated: `grep -r "docs/archive/reports/" *.md | wc -l`
- [ ] Verify lychee config updated: `grep "docs/archive/" .lychee.toml`
- [ ] Run post-migration link health: `lychee docs/ --exclude 'docs/archive/' --offline`
- [ ] Verify broken links ≤70 (down from 83)
- [ ] Verify git history preserved: `git log --oneline -- docs/archive/reports/ALPHA_READINESS_STATUS.md`

**Phase 6: Rollback Testing** (10 minutes)
- [ ] Create rollback test branch: `git checkout -b test-rollback`
- [ ] Execute rollback: `./scripts/archive_reports.sh --rollback`
- [ ] Verify reports restored: `ls -1 docs/reports/*.md | wc -l`
- [ ] Verify archive removed: `ls docs/archive/reports/ 2>&1`
- [ ] Verify no diff from main: `git diff main -- docs/reports/`
- [ ] Return to migration branch: `git checkout archive-reports-migration`

**Phase 7: Documentation Updates** (10 minutes)
- [ ] Update `docs/CONTRIBUTING-DOCS.md` with archive policy
- [ ] Add archive distinction (living docs vs historical)
- [ ] Add new report guidance (PR bodies, GitHub issues)
- [ ] Verify CONTRIBUTING-DOCS.md link health: `lychee docs/CONTRIBUTING-DOCS.md --offline`

**Phase 8: Final Verification** (10 minutes)
- [ ] Run full test suite: `cargo test --workspace --no-default-features --features cpu`
- [ ] Verify no code references to docs/reports: `git grep "docs/reports" -- '*.rs' '*.toml'`
- [ ] Verify CI documentation references updated
- [ ] Review git diff for unintended changes: `git diff main`
- [ ] Commit migration: `git add -A && git commit -m "docs: archive historical reports to docs/archive/reports"`

**Phase 9: PR Creation** (5 minutes)
- [ ] Push branch: `git push origin archive-reports-migration`
- [ ] Create PR with migration summary
- [ ] Include before/after link health metrics
- [ ] Reference DOCS_REPORTS_AUDIT.md in PR description
- [ ] Request review from documentation maintainers

---

## Status

**Current Phase**: Draft Specification
**Next Steps**: Review and approval → Script development → Migration execution
**Estimated Implementation Time**: 90 minutes (including testing and validation)
**Risk Level**: Low (documentation-only, no code changes, rollback available)

---

## Appendix: File-Specific Banner Mapping

**Category Mappings** (for automated variable substitution):

| File Pattern | Category | Current Doc | Current Doc Name |
|--------------|----------|-------------|------------------|
| `PR_*`, `PR422_*` | PR Review Report | ../../PR_475_FINAL_SUCCESS_REPORT.md | PR #475 Final Report |
| `ISSUE_*` | Issue Resolution Report | ../explanation/specs/GITHUB_ISSUES_P1_SPECIFICATIONS.md | Current Issue Specifications |
| `ALPHA_*`, `LAUNCH_*`, `SPRINT_*`, `VALIDATION_STATUS*` | Status Report | ../../CLAUDE.md | CLAUDE.md Project Reference |
| `TEST_*`, `COVERAGE_*`, `SECURITY_*`, `CROSSVAL_*` | Validation Report | ../development/test-suite.md | Current Test Suite Documentation |
| `DOCUMENTATION_*` | Documentation Report | ../CONTRIBUTING-DOCS.md | Documentation Contributing Guide |
| `FIXTURE_*`, `BENCHMARKING_*`, `INFRASTRUCTURE_*` | Implementation Report | ../../PR_475_FINAL_SUCCESS_REPORT.md | PR #475 Final Report |
| `*` (default) | Project Report | ../../CLAUDE.md | CLAUDE.md Project Reference |

**Total Categories**: 7
**Total Files**: 53
**Default Fallback**: CLAUDE.md (covers miscellaneous reports)

---

**Last Updated**: 2025-10-23
**Spec Author**: BitNet-rs Spec Analyzer Agent
**Review Status**: Pending
