# Issue: Documentation Consolidation and Navigation Improvement

## Context

BitNet.rs documentation has grown organically to 292 files across the workspace, with particularly fragmented structure in `ci/solutions/` (35 files, including ~8-10 redundant index/summary files). This creates navigation friction and maintenance overhead for developers seeking specific CI/test guidance.

Following PR #475 (comprehensive integration), we need to consolidate documentation into a streamlined, navigable structure with single sources of truth for each topic.

**Affected Components:**
- `ci/solutions/` - 35 files requiring consolidation and categorization
- `docs/development/` - Cross-references to CI solutions
- `README.md`, `CLAUDE.md` - Top-level navigation updates
- `.lychee.toml` - Link validation configuration

**Inference Pipeline Impact:**
- Documentation only - no direct inference pipeline changes
- Indirect impact: Faster developer onboarding improves contribution quality

**Performance Implications:**
- Navigation time: Target < 30 seconds to find any CI solution (from README)
- Maintenance overhead: Single source of truth reduces documentation drift
- CI stability: Zero broken internal links (verified by lychee)

## User Story

As a documentation reader, I need streamlined navigation with fewer index files so that finding information is faster and documentation maintenance is easier.

## Acceptance Criteria

AC1: Reduce `ci/solutions/` file count from 35 files to ~15 files (57% reduction via consolidation)
AC2: Merge redundant indexes into category-specific `README.md` files (e.g., `qk256/README.md`, `clippy/README.md`)
AC3: Create single authoritative entry point: `ci/solutions/README.md` (consolidates `INDEX.md`, `00_NAVIGATION_INDEX.md`, `QUICK_REFERENCE.md`)
AC4: Update all cross-references to new structure and verify with lychee link checker (100% internal links pass)
AC5: Archive outdated summaries to `ci/solutions/archive/` with metadata headers (archival date, reason)
AC6: Ensure zero orphaned docs - all markdown files reachable from top-level README or explicit archive markers
AC7: Add `.lychee.toml` configuration for automated link validation with CI enforcement
AC8: Verify navigation time < 30 seconds to find any solution (user acceptance testing with 3+ developers)

## Technical Implementation Notes

- **Affected crates**: N/A (documentation only - no code changes)
- **Pipeline stages**: N/A (documentation only)
- **Performance considerations**:
  - Navigation time: < 30 seconds to find any CI solution (from README)
  - Link validation: 100% of internal links must resolve (verified by lychee)
  - Maintenance overhead: Single source of truth for each topic (no duplication)
- **Quantization requirements**: N/A (documentation only)
- **Cross-validation**: N/A (documentation only)
- **Feature flags**: N/A (documentation only)
- **GGUF compatibility**: N/A (documentation only)
- **Testing strategy**:
  - Link validation: `lychee --config .lychee.toml docs/ ci/ README.md CLAUDE.md`
  - Orphan detection: Script to verify all markdown files referenced from top-level docs
  - Navigation testing: User acceptance with 3+ developers timing solution discovery
  - CI integration: Add lychee check to `.github/workflows/ci.yml`

**Proposed Structure:**
```
ci/solutions/
├── README.md                        # AC3: Main entry point (consolidates INDEX.md, 00_NAVIGATION_INDEX.md)
│   ├── Quick Reference (consolidates QUICK_REFERENCE.md)
│   ├── Solutions Index (alphabetical)
│   └── Navigation Guide
├── IMPLEMENTATION_GUIDE.md          # Consolidates IMPLEMENTATION_SUMMARY.md, SOLUTION_SUMMARY.md
├── qk256/
│   ├── README.md                    # AC2: Consolidates QK256_ANALYSIS_INDEX.md
│   ├── test_failure_analysis.md     # Renames QK256_TEST_FAILURE_ANALYSIS_INDEX.md
│   ├── property_test_analysis.md    # Keeps QK256_PROPERTY_TEST_ANALYSIS_INDEX.md
│   └── tolerance_strategy.md        # Keeps QK256_TOLERANCE_STRATEGY.md
├── clippy/
│   ├── README.md                    # AC2: Consolidates CLIPPY_LINT_FIXES.md + CLIPPY_QUICK_REFERENCE.md
│   └── fixes.md                     # Detailed fix guide
├── gguf/
│   ├── README.md                    # AC2: Consolidates GGUF_SHAPE_VALIDATION_INDEX.md
│   └── shape_validation_fix.md      # Keeps gguf_shape_validation_fix.md
├── receipts/
│   ├── README.md                    # AC2: Consolidates INDEX_RECEIPT_ANALYSIS.md + README_RECEIPT_ANALYSIS.md
│   └── test_quick_reference.md      # Keeps RECEIPT_TEST_QUICK_REFERENCE.md
├── batch_prefill/
│   ├── README.md                    # AC2: Consolidates BATCH_PREFILL_INDEX.md
│   └── perf_quarantine.md           # Keeps batch_prefill_perf_quarantine.md
├── archive/
│   ├── SUMMARY.md                   # AC5: Old summaries with archival metadata
│   ├── SOLUTIONS_SUMMARY.md
│   └── ANALYSIS_SUMMARY.md
└── _TEMPLATE.md                     # Template for new solution docs
```

**Validation Commands:**
```bash
# AC4: Validate all markdown links
lychee --config .lychee.toml docs/ ci/ README.md CLAUDE.md

# AC6: Check for orphaned markdown files
find docs/ ci/ -name "*.md" -type f | while read -r file; do
  if ! grep -r "$(basename "$file")" docs/ ci/ README.md CLAUDE.md >/dev/null; then
    echo "⚠️  Orphaned: $file"
  fi
done

# AC1: Verify file count reduction
echo "Before: 35 files in ci/solutions/"
echo "After: $(find ci/solutions/ -name "*.md" -type f | wc -l) files"

# AC8: Navigation time testing (manual user acceptance)
# Ask 3+ developers to find specific solutions and measure time
```

**Lychee Configuration (AC7):**
```toml
# .lychee.toml
[cache]
max_age = "1d"

[include]
include_verbatim = true

[accept]
accept = [200, 204, 301, 302, 403, 429]

[exclude]
exclude_path = ["target/", "vendor/", ".git/"]

# Exclude external links that require authentication
exclude = [
  "https://github.com/.*/issues/.*",
  "https://github.com/.*/pull/.*"
]
```

**Estimate**: 4-5 hours

---

<!-- gates:start -->
| Gate | Status | Evidence |
|------|--------|----------|
| spec | ✅ pass | Feature spec created in docs/explanation/specs/SPEC-2025-001-next-steps-priority-implementation.md (Story 3) |
| format | pending | Markdownlint validation with markdownlint-cli2 |
| clippy | pending | N/A (documentation only) |
| tests | pending | Link validation with lychee --config .lychee.toml |
| build | pending | N/A (documentation only) |
| features | pending | N/A (documentation only) |
| benchmarks | pending | Navigation time testing (target: < 30 seconds) |
| docs | ✅ pass | This is the documentation consolidation task itself |
<!-- gates:end -->

<!-- hoplog:start -->
### Hop log
- Created feature spec: Story 3 in docs/explanation/specs/SPEC-2025-001-next-steps-priority-implementation.md
<!-- hoplog:end -->

<!-- decision:start -->
**State:** in-progress
**Why:** Feature spec created and validated, ready for implementation
**Next:** NEXT → implementation with lychee validation workflow (AC1-AC8)
<!-- decision:end -->
