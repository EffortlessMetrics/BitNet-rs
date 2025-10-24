# Documentation Link Validation Report

> **Note**: References to `docs/archive/reports/` point to historical archived documentation.
> For current status, see [CLAUDE.md](CLAUDE.md) and [PR #475](PR_475_FINAL_SUCCESS_REPORT.md).


> **Note**: References to `docs/archive/reports/` point to historical archived documentation.
> For current status, see [CLAUDE.md](CLAUDE.md) and [PR #475](PR_475_FINAL_SUCCESS_REPORT.md).


**Date**: 2025-10-23
**Validator**: lychee v0.21.0
**Scope**: ci/solutions/, docs/, README.md, CLAUDE.md, PR_475_FINAL_SUCCESS_REPORT.md

## Executive Summary

**Status**: ❌ **83 broken links found** across documentation

| Metric | Count | Percentage |
|--------|-------|------------|
| Total Links Checked | 723 | 100% |
| ✅ Successful | 504 | 69.7% |
| 🚫 **Errors** | **83** | **11.5%** |
| 👻 Excluded (External) | 136 | 18.8% |
| ⏳ Timeouts | 0 | 0% |

## Critical Findings

### 1. **Most Common Issue: Incorrect PR Report Path** (32 occurrences)

**Issue**: All ci/solutions/ files link to `ci/PR_475_FINAL_SUCCESS_REPORT.md`
**Reality**: File is located at `/home/steven/code/Rust/BitNet-rs/PR_475_FINAL_SUCCESS_REPORT.md` (root directory)

**Affected Files** (32 total):
- ci/solutions/00_NAVIGATION_INDEX.md
- ci/solutions/_TEMPLATE.md
- ci/solutions/ANALYSIS_SUMMARY.md
- ci/solutions/BATCH_PREFILL_INDEX.md
- ci/solutions/batch_prefill_perf_quarantine.md
- ci/solutions/CLIPPY_LINT_FIXES.md
- ci/solutions/CLIPPY_QUICK_REFERENCE.md
- ci/solutions/concurrent_load_perf_quarantine.md
- ci/solutions/CONCURRENT_LOAD_QUARANTINE_QUICK_REF.md
- ci/solutions/docs_code_example_fixes.md
- ci/solutions/ffi_build_hygiene_fixes.md
- ci/solutions/general_docs_scaffolding.md
- ci/solutions/gguf_shape_validation_fix.md
- ci/solutions/GGUF_SHAPE_VALIDATION_INDEX.md
- ci/solutions/IMPLEMENTATION_SUMMARY.md
- ci/solutions/INDEX.md
- ci/solutions/INDEX_RECEIPT_ANALYSIS.md
- ci/solutions/QK256_ANALYSIS_INDEX.md
- ci/solutions/qk256_docs_completion.md
- ci/solutions/qk256_property_test_analysis.md
- ci/solutions/qk256_struct_creation_analysis.md
- ci/solutions/qk256_test_failure_quickref.md
- ci/solutions/QK256_TOLERANCE_STRATEGY.md
- ci/solutions/QUICK_REFERENCE.md
- ci/solutions/README.md
- ci/solutions/README_RECEIPT_ANALYSIS.md
- ci/solutions/RECEIPT_TEST_QUICK_REFERENCE.md
- ci/solutions/RECEIPT_TEST_REFACTOR.md
- ci/solutions/RELATED_DOCS_ADDED.md
- ci/solutions/SOLUTIONS_SUMMARY.md
- ci/solutions/STOP_SEQUENCE_VERIFICATION.md
- ci/solutions/_TEMPLATE.md (also has placeholder links)

**Fix**: Update all links from `ci/PR_475_FINAL_SUCCESS_REPORT.md` to `../PR_475_FINAL_SUCCESS_REPORT.md`

---

### 2. **Missing Documentation Files in docs/development/** (7 files)

These files are referenced but don't exist:

| Missing File | Referenced By | Suggested Fix |
|--------------|---------------|---------------|
| `docs/development/performance-benchmarking.md` | build-commands.md, validation-framework.md | Update links to point to existing performance docs or create file |
| `docs/development/VALIDATION.md` | validation-framework.md | Update to `validation-framework.md` or create file |
| `docs/development/VALIDATION_QUICK_START.md` | validation-framework.md | Update to `validation-ci.md` or create quickstart |
| `docs/development/concurrency-caps.md` | test-suite.md | Remove link or create file |
| `docs/development/performance-tracking.md` | test-suite.md | Update to `validation-ci.md` or create file |
| `docs/development/streaming-api.md` | test-suite.md | Remove link or document streaming API |

---

### 3. **Missing Documentation Files in docs/reference/** (5 files)

| Missing File | Referenced By | Suggested Fix |
|--------------|---------------|---------------|
| `docs/reference/cli-reference.md` | use-qk256-models.md | Create CLI reference or update to CLAUDE.md |
| `docs/reference/prompt-templates.md` | troubleshoot-intelligibility.md | Create prompt template docs or update to CLAUDE.md |
| `docs/reference/model-validation.md` | correction-policy.md | Create or update to `validation-gates.md` |
| `docs/reference/COMPATIBILITY.md` | MIGRATION.md | Create or update to `api-compatibility.md` |
| `docs/reference/LICENSE` | INSTALLATION.md | Update to root `LICENSE` file |
| `docs/reference/getting-started.md` | api-compatibility.md | Update to `docs/getting-started.md` |
| `docs/reference/reference.md` | _TEMPLATE.md | Template placeholder - ignore |

---

### 4. **Missing Documentation Files in docs/howto/** (1 file)

| Missing File | Referenced By | Suggested Fix |
|--------------|---------------|---------------|
| `docs/howto/strict-mode-validation-workflows.md` | use-qk256-models.md | Create workflow docs or update to validation-ci.md |

---

### 5. **Missing Documentation Files in docs/troubleshooting/** (1 file)

| Missing File | Referenced By | Suggested Fix |
|--------------|---------------|---------------|
| `docs/troubleshooting/layernorm-issues.md` | correction-policy.md | Create troubleshooting doc for LayerNorm issues |

---

### 6. **Stale docs/archive/reports/ Links** (13 files)

The `docs/archive/reports/` directory contains outdated reports with broken internal links:

**Affected Files**:
- docs/archive/reports/DOCUMENTATION_UPDATE_REPORT.md (4 broken links)
- docs/archive/reports/GOALS_VS_REALITY_ANALYSIS.md (5 broken links)
- docs/archive/reports/PR_TEMPLATE_ISSUE_159.md (7 broken links)

**Missing Targets**:
- `docs/archive/reports/README.md`
- `docs/archive/reports/VALIDATION.md`
- `docs/archive/reports/benchmark_results.json`
- `docs/archive/reports/benchmark_comparison.py`
- `docs/archive/reports/PERFORMANCE_COMPARISON.md`
- `docs/archive/reports/docs/RESOURCE_MANAGEMENT_TESTING.md`
- `docs/archive/reports/docs/testing/` (entire directory)
- `docs/archive/reports/docs/development/gpu-development.md`
- `docs/archive/reports/docs/explanation/` (multiple files)
- `docs/archive/reports/docs/reference/quantization-support.md`

**Fix**: These are likely historical reports. Options:
1. Archive to `docs/archive/reports/` and update links
2. Delete outdated reports
3. Update reports to reference current documentation structure

---

### 7. **Fast-Feedback Documentation Links** (4 files)

File: `docs/fast-feedback.md` references missing files:

| Missing File | Suggested Fix |
|--------------|---------------|
| `docs/testing-framework.md` | Update to `docs/development/test-suite.md` |
| `docs/incremental-testing.md` | Update to `docs/development/test-suite.md` |
| `docs/performance-optimization.md` | Update to performance-related docs or create |
| `docs/ci-integration.md` | Update to `docs/development/ci-integration.md` |

---

### 8. **Incorrect QK256 Documentation Paths** (3 files)

File: `ci/solutions/qk256_docs_completion.md` links to non-existent paths:

| Broken Link | Correct Path |
|-------------|--------------|
| `ci/solutions/docs/explanation/i2s-dual-flavor.md` | `docs/explanation/i2s-dual-flavor.md` |
| `ci/solutions/docs/howto/use-qk256-models.md` | `docs/howto/use-qk256-models.md` |

File: `docs/explanation/specs/issue-469-mvp-sprint-polish-spec.md` has incorrect nested paths:

| Broken Link | Correct Path |
|-------------|--------------|
| `docs/explanation/specs/docs/explanation/i2s-dual-flavor.md` | `docs/explanation/i2s-dual-flavor.md` |
| `docs/explanation/specs/docs/howto/use-qk256-models.md` | `docs/howto/use-qk256-models.md` |
| `docs/explanation/specs/docs/quickstart.md` | `docs/quickstart.md` |
| `docs/explanation/specs/explanation/i2s-dual-flavor.md` | `docs/explanation/i2s-dual-flavor.md` |
| `docs/explanation/specs/howto/use-qk256-models.md` | `docs/howto/use-qk256-models.md` |
| `docs/explanation/specs/howto/validate-models.md` | `docs/howto/validate-models.md` |

---

### 9. **Anchor Link Errors** (1 file)

File: `docs/howto/troubleshoot-intelligibility.md`

| Broken Link | Issue |
|-------------|-------|
| `docs/CLAUDE.md#inference-usage` | File exists at root: `CLAUDE.md#inference-usage` |

**Fix**: Update to `../../CLAUDE.md#inference-usage`

---

### 10. **Template Placeholder Links** (5 links in _TEMPLATE.md)

File: `ci/solutions/_TEMPLATE.md` contains placeholder links (expected):

- `ci/solutions/related_solution_1.md`
- `ci/solutions/related_solution_2.md`
- `docs/explanation/topic.md`
- `docs/reference/reference.md`

**Action**: No fix needed - these are intentional template placeholders

---

### 11. **Missing Root Files** (2 files)

| Missing File | Referenced By | Suggested Fix |
|--------------|---------------|---------------|
| `GOALS_VS_REALITY_ANALYSIS.md` | migration-faq.md, api-compatibility.md | Update to existing analysis docs or remove links |
| `docs/examples` | getting-started.md | Update to `examples/` directory in root |

---

### 12. **Other Missing Files** (3 files)

| Missing File | Referenced By | Suggested Fix |
|--------------|---------------|---------------|
| `docs/GOALS_VS_REALITY_ANALYSIS.md` | api-compatibility.md | Remove or update to current docs |
| `docs/explanation/docs/baselines` | issue-465-implementation-spec.md | Update to `docs/baselines/` |

---

## Priority Recommendations

### High Priority (Breaking Navigation)

1. **Fix PR Report Links** (32 files): Update all `ci/solutions/*.md` files to use correct path `../PR_475_FINAL_SUCCESS_REPORT.md`

2. **Fix QK256 Documentation Paths** (2 files):
   - `ci/solutions/qk256_docs_completion.md`
   - `docs/explanation/specs/issue-469-mvp-sprint-polish-spec.md`

3. **Create Missing Core Reference Docs** (3 files):
   - `docs/reference/cli-reference.md`
   - `docs/reference/prompt-templates.md`
   - `docs/howto/strict-mode-validation-workflows.md`

### Medium Priority (Navigation Improvements)

4. **Update development/* Links** (3 files):
   - Decide whether to create `performance-benchmarking.md` or update existing links
   - Consolidate validation docs references
   - Remove or create streaming-api.md, concurrency-caps.md, performance-tracking.md

5. **Fix Anchor Link** (1 file):
   - `docs/howto/troubleshoot-intelligibility.md` → Update CLAUDE.md anchor

6. **Clean Up fast-feedback.md** (4 links):
   - Update to current test suite and CI documentation

### Low Priority (Historical Cleanup)

7. **Archive or Update docs/archive/reports/** (13 broken links):
   - These appear to be historical reports from previous issues
   - Consider archiving to `docs/archive/reports/` or removing entirely

8. **Fix Missing Root Files** (2 files):
   - Update `GOALS_VS_REALITY_ANALYSIS.md` references or remove
   - Fix `docs/examples` → `examples/` directory reference

---

## Detailed Breakdown by Category

### Category A: Path Correction (No File Creation)

**Count**: 35 errors
**Effort**: Low (find-and-replace)

Examples:
- `ci/PR_475_FINAL_SUCCESS_REPORT.md` → `../PR_475_FINAL_SUCCESS_REPORT.md` (32x)
- `docs/CLAUDE.md#inference-usage` → `../../CLAUDE.md#inference-usage` (1x)
- `ci/solutions/docs/explanation/i2s-dual-flavor.md` → `docs/explanation/i2s-dual-flavor.md` (2x)

### Category B: Missing Documentation (Needs Creation)

**Count**: 18 errors
**Effort**: Medium-High (requires writing documentation)

Core missing files:
- `docs/reference/cli-reference.md`
- `docs/reference/prompt-templates.md`
- `docs/howto/strict-mode-validation-workflows.md`
- `docs/troubleshooting/layernorm-issues.md`
- `docs/development/performance-benchmarking.md`

### Category C: Stale References (Needs Decision)

**Count**: 25 errors
**Effort**: Low (delete or update links)

Mostly in:
- `docs/archive/reports/` (13 broken links - historical)
- `docs/fast-feedback.md` (4 broken links)
- Template placeholders (5 links - intentional)

### Category D: Template Placeholders (Intentional)

**Count**: 5 errors
**Effort**: None (working as intended)

File: `ci/solutions/_TEMPLATE.md`

---

## Suggested Quick Fixes

### Fix 1: PR Report Path (Affects 32 files)

```bash
# Update all ci/solutions/*.md files
find ci/solutions -name "*.md" -type f -exec sed -i 's|ci/PR_475_FINAL_SUCCESS_REPORT.md|../PR_475_FINAL_SUCCESS_REPORT.md|g' {} +
```

### Fix 2: QK256 Documentation Paths (Affects 2 files)

**File**: `ci/solutions/qk256_docs_completion.md`

```bash
sed -i 's|ci/solutions/docs/explanation/i2s-dual-flavor.md|docs/explanation/i2s-dual-flavor.md|g' ci/solutions/qk256_docs_completion.md
sed -i 's|ci/solutions/docs/howto/use-qk256-models.md|docs/howto/use-qk256-models.md|g' ci/solutions/qk256_docs_completion.md
```

**File**: `docs/explanation/specs/issue-469-mvp-sprint-polish-spec.md`

```bash
sed -i 's|docs/explanation/specs/docs/|docs/|g' docs/explanation/specs/issue-469-mvp-sprint-polish-spec.md
sed -i 's|docs/explanation/specs/explanation/|docs/explanation/|g' docs/explanation/specs/issue-469-mvp-sprint-polish-spec.md
sed -i 's|docs/explanation/specs/howto/|docs/howto/|g' docs/explanation/specs/issue-469-mvp-sprint-polish-spec.md
```

### Fix 3: CLAUDE.md Anchor (Affects 1 file)

**File**: `docs/howto/troubleshoot-intelligibility.md`

```bash
sed -i 's|docs/CLAUDE.md#inference-usage|../../CLAUDE.md#inference-usage|g' docs/howto/troubleshoot-intelligibility.md
```

### Fix 4: Fast-Feedback Links (Affects 1 file)

**File**: `docs/fast-feedback.md`

```bash
sed -i 's|docs/testing-framework.md|docs/development/test-suite.md|g' docs/fast-feedback.md
sed -i 's|docs/incremental-testing.md|docs/development/test-suite.md|g' docs/fast-feedback.md
sed -i 's|docs/ci-integration.md|docs/development/ci-integration.md|g' docs/fast-feedback.md
# performance-optimization.md - needs manual review
```

---

## Files Requiring Manual Review

### High Priority

1. **docs/development/build-commands.md** (line 244)
   - Link: `performance-benchmarking.md`
   - Decision: Create file or update to existing performance docs?

2. **docs/development/validation-framework.md**
   - Links: `VALIDATION.md`, `VALIDATION_QUICK_START.md`, `performance-benchmarking.md`
   - Decision: Consolidate validation docs?

3. **docs/development/test-suite.md**
   - Links: `concurrency-caps.md`, `performance-tracking.md`, `streaming-api.md`
   - Decision: Create these files or remove references?

4. **docs/howto/use-qk256-models.md**
   - Links: `strict-mode-validation-workflows.md`, `cli-reference.md`
   - Decision: Create missing howto and reference docs?

### Medium Priority

5. **docs/reference/** directory
   - Missing: `cli-reference.md`, `prompt-templates.md`, `model-validation.md`
   - Decision: Create comprehensive reference documentation?

6. **docs/troubleshooting/** directory
   - Missing: `layernorm-issues.md`
   - Decision: Document LayerNorm validation troubleshooting?

### Low Priority

7. **docs/archive/reports/** directory (13 broken links)
   - Decision: Archive, delete, or update historical reports?

---

## Summary Statistics

| Category | Count | % of Errors |
|----------|-------|-------------|
| Path Corrections (Easy) | 35 | 42.2% |
| Missing Documentation | 18 | 21.7% |
| Stale References | 25 | 30.1% |
| Template Placeholders | 5 | 6.0% |
| **Total Errors** | **83** | **100%** |

---

## Next Steps

### Immediate Actions

1. ✅ **Apply Fix 1**: Update PR report links (32 files) - ~2 minutes
2. ✅ **Apply Fix 2**: Fix QK256 documentation paths (2 files) - ~1 minute
3. ✅ **Apply Fix 3**: Fix CLAUDE.md anchor (1 file) - ~30 seconds
4. ✅ **Apply Fix 4**: Update fast-feedback.md links (1 file) - ~1 minute

**Total Immediate Fixes**: 36 errors resolved in ~5 minutes

### Short-Term Actions (Next PR/Issue)

5. **Create Core Reference Docs** (~2-4 hours):
   - `docs/reference/cli-reference.md`
   - `docs/reference/prompt-templates.md`
   - `docs/howto/strict-mode-validation-workflows.md`
   - `docs/troubleshooting/layernorm-issues.md`

6. **Consolidate Validation Docs** (~1-2 hours):
   - Decide on validation doc structure
   - Update or remove VALIDATION.md, VALIDATION_QUICK_START.md references

### Long-Term Actions (Future Cleanup)

7. **Archive Historical Reports** (~30 minutes):
   - Move `docs/archive/reports/` to `docs/archive/reports/`
   - Update or remove references

8. **Create Development Guides** (as needed):
   - `performance-benchmarking.md`
   - `streaming-api.md`
   - `concurrency-caps.md`
   - `performance-tracking.md`

---

## Validation Methodology

**Tool**: lychee v0.21.0 (offline mode)
**Configuration**: `.lychee.toml`
- Checked: Internal file links only (offline mode)
- Excluded: External URLs, build artifacts, dependencies
- Accepted: HTTP 200, 429 status codes

**Command**:
```bash
lychee --config .lychee.toml --format markdown \
  ci/solutions/ docs/ README.md CLAUDE.md PR_475_FINAL_SUCCESS_REPORT.md
```

**Validation Date**: 2025-10-23
**Total Links Checked**: 723
**Success Rate**: 69.7% (504/723)
**Error Rate**: 11.5% (83/723)

---

## Appendix: Full Error List

<details>
<summary>Click to expand complete list of 83 broken links</summary>

### ci/solutions/ (32 errors - all PR report path)

1. ci/solutions/00_NAVIGATION_INDEX.md → `ci/PR_475_FINAL_SUCCESS_REPORT.md`
2. ci/solutions/_TEMPLATE.md → `ci/PR_475_FINAL_SUCCESS_REPORT.md`
3. ci/solutions/_TEMPLATE.md → `ci/solutions/related_solution_1.md` (placeholder)
4. ci/solutions/_TEMPLATE.md → `ci/solutions/related_solution_2.md` (placeholder)
5. ci/solutions/_TEMPLATE.md → `docs/explanation/topic.md` (placeholder)
6. ci/solutions/_TEMPLATE.md → `docs/reference/reference.md` (placeholder)
7. ci/solutions/ANALYSIS_SUMMARY.md → `ci/PR_475_FINAL_SUCCESS_REPORT.md`
8. ci/solutions/BATCH_PREFILL_INDEX.md → `ci/PR_475_FINAL_SUCCESS_REPORT.md`
9. ci/solutions/batch_prefill_perf_quarantine.md → `ci/PR_475_FINAL_SUCCESS_REPORT.md`
10. ci/solutions/CLIPPY_LINT_FIXES.md → `ci/PR_475_FINAL_SUCCESS_REPORT.md`
11. ci/solutions/CLIPPY_QUICK_REFERENCE.md → `ci/PR_475_FINAL_SUCCESS_REPORT.md`
12. ci/solutions/concurrent_load_perf_quarantine.md → `ci/PR_475_FINAL_SUCCESS_REPORT.md`
13. ci/solutions/CONCURRENT_LOAD_QUARANTINE_QUICK_REF.md → `ci/PR_475_FINAL_SUCCESS_REPORT.md`
14. ci/solutions/docs_code_example_fixes.md → `ci/PR_475_FINAL_SUCCESS_REPORT.md`
15. ci/solutions/ffi_build_hygiene_fixes.md → `ci/PR_475_FINAL_SUCCESS_REPORT.md`
16. ci/solutions/general_docs_scaffolding.md → `ci/PR_475_FINAL_SUCCESS_REPORT.md`
17. ci/solutions/gguf_shape_validation_fix.md → `ci/PR_475_FINAL_SUCCESS_REPORT.md`
18. ci/solutions/GGUF_SHAPE_VALIDATION_INDEX.md → `ci/PR_475_FINAL_SUCCESS_REPORT.md`
19. ci/solutions/IMPLEMENTATION_SUMMARY.md → `ci/PR_475_FINAL_SUCCESS_REPORT.md`
20. ci/solutions/INDEX.md → `ci/PR_475_FINAL_SUCCESS_REPORT.md`
21. ci/solutions/INDEX_RECEIPT_ANALYSIS.md → `ci/PR_475_FINAL_SUCCESS_REPORT.md`
22. ci/solutions/QK256_ANALYSIS_INDEX.md → `ci/PR_475_FINAL_SUCCESS_REPORT.md`
23. ci/solutions/qk256_docs_completion.md → `ci/PR_475_FINAL_SUCCESS_REPORT.md`
24. ci/solutions/qk256_docs_completion.md → `ci/solutions/docs/explanation/i2s-dual-flavor.md`
25. ci/solutions/qk256_docs_completion.md → `ci/solutions/docs/howto/use-qk256-models.md`
26. ci/solutions/qk256_property_test_analysis.md → `ci/PR_475_FINAL_SUCCESS_REPORT.md`
27. ci/solutions/qk256_struct_creation_analysis.md → `ci/PR_475_FINAL_SUCCESS_REPORT.md`
28. ci/solutions/qk256_test_failure_quickref.md → `ci/PR_475_FINAL_SUCCESS_REPORT.md`
29. ci/solutions/QK256_TOLERANCE_STRATEGY.md → `ci/PR_475_FINAL_SUCCESS_REPORT.md`
30. ci/solutions/QUICK_REFERENCE.md → `ci/PR_475_FINAL_SUCCESS_REPORT.md`
31. ci/solutions/README.md → `ci/PR_475_FINAL_SUCCESS_REPORT.md`
32. ci/solutions/README_RECEIPT_ANALYSIS.md → `ci/PR_475_FINAL_SUCCESS_REPORT.md`
33. ci/solutions/RECEIPT_TEST_QUICK_REFERENCE.md → `ci/PR_475_FINAL_SUCCESS_REPORT.md`
34. ci/solutions/RECEIPT_TEST_REFACTOR.md → `ci/PR_475_FINAL_SUCCESS_REPORT.md`
35. ci/solutions/RELATED_DOCS_ADDED.md → `ci/PR_475_FINAL_SUCCESS_REPORT.md`
36. ci/solutions/SOLUTIONS_SUMMARY.md → `ci/PR_475_FINAL_SUCCESS_REPORT.md`
37. ci/solutions/STOP_SEQUENCE_VERIFICATION.md → `ci/PR_475_FINAL_SUCCESS_REPORT.md`

### docs/development/ (7 errors)

38. docs/development/build-commands.md → `performance-benchmarking.md`
39. docs/development/validation-framework.md → `performance-benchmarking.md`
40. docs/development/validation-framework.md → `VALIDATION.md`
41. docs/development/validation-framework.md → `VALIDATION_QUICK_START.md`
42. docs/development/test-suite.md → `concurrency-caps.md`
43. docs/development/test-suite.md → `performance-tracking.md`
44. docs/development/test-suite.md → `streaming-api.md`

### docs/reference/ (7 errors)

45. docs/howto/use-qk256-models.md → `docs/reference/cli-reference.md`
46. docs/howto/troubleshoot-intelligibility.md → `docs/reference/prompt-templates.md`
47. docs/explanation/correction-policy.md → `docs/reference/model-validation.md`
48. docs/reference/MIGRATION.md → `docs/reference/COMPATIBILITY.md`
49. docs/reference/INSTALLATION.md → `docs/reference/LICENSE`
50. docs/reference/api-compatibility.md → `docs/reference/getting-started.md`
51. ci/solutions/_TEMPLATE.md → `docs/reference/reference.md` (placeholder)

### docs/howto/ (1 error)

52. docs/howto/use-qk256-models.md → `strict-mode-validation-workflows.md`

### docs/troubleshooting/ (1 error)

53. docs/explanation/correction-policy.md → `docs/troubleshooting/layernorm-issues.md`

### docs/archive/reports/ (13 errors - historical)

54. docs/archive/reports/GOALS_VS_REALITY_ANALYSIS.md → `docs/archive/reports/README.md`
55. docs/archive/reports/GOALS_VS_REALITY_ANALYSIS.md → `docs/archive/reports/VALIDATION.md`
56. docs/archive/reports/GOALS_VS_REALITY_ANALYSIS.md → `benchmark_results.json`
57. docs/archive/reports/GOALS_VS_REALITY_ANALYSIS.md → `benchmark_comparison.py`
58. docs/archive/reports/GOALS_VS_REALITY_ANALYSIS.md → `PERFORMANCE_COMPARISON.md`
59. docs/archive/reports/DOCUMENTATION_UPDATE_REPORT.md → `docs/RESOURCE_MANAGEMENT_TESTING.md`
60. docs/archive/reports/DOCUMENTATION_UPDATE_REPORT.md → `docs/testing/README.md`
61. docs/archive/reports/DOCUMENTATION_UPDATE_REPORT.md → `docs/testing/test-authoring-guide.md`
62. docs/archive/reports/DOCUMENTATION_UPDATE_REPORT.md → `docs/testing/framework-overview.md`
63. docs/archive/reports/PR_TEMPLATE_ISSUE_159.md → `docs/architecture-overview.md`
64. docs/archive/reports/PR_TEMPLATE_ISSUE_159.md → `docs/development/gpu-development.md`
65. docs/archive/reports/PR_TEMPLATE_ISSUE_159.md → `docs/explanation/gguf-weight-loading.md`
66. docs/archive/reports/PR_TEMPLATE_ISSUE_159.md → `docs/explanation/gguf-weight-loading-performance-validation.md`
67. docs/archive/reports/PR_TEMPLATE_ISSUE_159.md → `docs/explanation/gguf-weight-loading-integration-testing.md`
68. docs/archive/reports/PR_TEMPLATE_ISSUE_159.md → `docs/explanation/gguf-weight-loading-api-contracts.md`
69. docs/archive/reports/PR_TEMPLATE_ISSUE_159.md → `docs/reference/quantization-support.md`

### docs/ (other) (10 errors)

70. docs/fast-feedback.md → `docs/testing-framework.md`
71. docs/fast-feedback.md → `docs/incremental-testing.md`
72. docs/fast-feedback.md → `docs/performance-optimization.md`
73. docs/fast-feedback.md → `docs/ci-integration.md`
74. docs/getting-started.md → `docs/examples`
75. docs/howto/troubleshoot-intelligibility.md → `docs/CLAUDE.md#inference-usage`
76. docs/explanation/issue-465-implementation-spec.md → `docs/explanation/docs/baselines`
77. docs/migration-faq.md → `GOALS_VS_REALITY_ANALYSIS.md`
78. docs/reference/api-compatibility.md → `docs/GOALS_VS_REALITY_ANALYSIS.md`

### docs/explanation/specs/ (6 errors - nested paths)

79. docs/explanation/specs/issue-469-mvp-sprint-polish-spec.md → `docs/explanation/specs/docs/explanation/i2s-dual-flavor.md`
80. docs/explanation/specs/issue-469-mvp-sprint-polish-spec.md → `docs/explanation/specs/docs/howto/use-qk256-models.md`
81. docs/explanation/specs/issue-469-mvp-sprint-polish-spec.md → `docs/explanation/specs/docs/quickstart.md`
82. docs/explanation/specs/issue-469-mvp-sprint-polish-spec.md → `docs/explanation/specs/explanation/i2s-dual-flavor.md`
83. docs/explanation/specs/issue-469-mvp-sprint-polish-spec.md → `docs/explanation/specs/howto/use-qk256-models.md`
84. docs/explanation/specs/issue-469-mvp-sprint-polish-spec.md → `docs/explanation/specs/howto/validate-models.md`

</details>

---

**Report Generated**: 2025-10-23
**Validator**: Claude Code (BitNet.rs Documentation Gate)
**Next Actions**: Apply immediate fixes (36 errors), create missing docs (18 errors), clean up stale references (25 errors)
