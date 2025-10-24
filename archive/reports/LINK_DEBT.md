# Documentation Link Debt Tracking

**Purpose**: Track and triage broken documentation links for systematic resolution.
**Last Updated**: 2025-10-23
**Total Debt**: 83 broken links

## Status Overview

| Category | Count | Priority | Owner | Status |
|----------|-------|----------|-------|--------|
| PR #475 Path Fix | 32 | **P0** | @infra | 🔴 Fix Now |
| Missing dev/ docs | 7 | **P1** | @docs | 🟡 Create or Redirect |
| Missing reference/ docs | 7 | **P2** | @docs | 🟡 Create or Redirect |
| CI solutions/ internal | 32 | **P1** | @ci | 🟡 Fix with PR #475 |
| Archive stubs | 5 | **P3** | @docs | 🟢 Archive or Remove |

---

## P0 - Fix Now (32 links)

### 1. PR #475 Report Path Correction

**Issue**: All `ci/solutions/*.md` files reference incorrect path for final success report.

**Incorrect Path**: `ci/PR_475_FINAL_SUCCESS_REPORT.md`
**Correct Path**: `../PR_475_FINAL_SUCCESS_REPORT.md` (relative from ci/solutions/)
**Absolute Path**: `/home/steven/code/Rust/BitNet-rs/PR_475_FINAL_SUCCESS_REPORT.md`

**Affected Files** (32):
```
ci/solutions/00_NAVIGATION_INDEX.md
ci/solutions/_TEMPLATE.md
ci/solutions/ANALYSIS_SUMMARY.md
ci/solutions/BATCH_PREFILL_INDEX.md
ci/solutions/batch_prefill_perf_quarantine.md
ci/solutions/CLIPPY_LINT_FIXES.md
ci/solutions/CLIPPY_QUICK_REFERENCE.md
ci/solutions/concurrent_load_perf_quarantine.md
ci/solutions/CONCURRENT_LOAD_QUARANTINE_QUICK_REF.md
ci/solutions/docs_code_example_fixes.md
ci/solutions/ffi_build_hygiene_fixes.md
ci/solutions/general_docs_scaffolding.md
ci/solutions/gguf_shape_validation_fix.md
ci/solutions/GGUF_SHAPE_VALIDATION_INDEX.md
ci/solutions/IMPLEMENTATION_SUMMARY.md
ci/solutions/INDEX.md
ci/solutions/INDEX_RECEIPT_ANALYSIS.md
ci/solutions/QK256_ANALYSIS_INDEX.md
ci/solutions/qk256_docs_completion.md
ci/solutions/qk256_property_test_analysis.md
ci/solutions/qk256_struct_creation_analysis.md
ci/solutions/qk256_test_failure_quickref.md
ci/solutions/QK256_TOLERANCE_STRATEGY.md
ci/solutions/QUICK_REFERENCE.md
ci/solutions/README.md
ci/solutions/README_RECEIPT_ANALYSIS.md
ci/solutions/RECEIPT_TEST_QUICK_REFERENCE.md
ci/solutions/RECEIPT_TEST_REFACTOR.md
ci/solutions/RELATED_DOCS_ADDED.md
ci/solutions/SOLUTIONS_SUMMARY.md
ci/solutions/STOP_SEQUENCE_VERIFICATION.md
ci/solutions/_TEMPLATE.md
```

**Bulk Fix Command**:
```bash
# Replace incorrect path with correct relative path
find ci/solutions/ -name '*.md' -type f -exec sed -i 's|ci/PR_475_FINAL_SUCCESS_REPORT\.md|../PR_475_FINAL_SUCCESS_REPORT.md|g' {} +
```

**Validation**:
```bash
# Verify all references are now correct
rg -F 'ci/PR_475_FINAL_SUCCESS_REPORT.md' ci/solutions/
# Should return no results

# Verify correct references exist
rg -F '../PR_475_FINAL_SUCCESS_REPORT.md' ci/solutions/ | wc -l
# Should return 32 (or number of files that reference it)
```

---

## P1 - Create or Redirect (14 links)

### 2. Missing docs/development/ Files (7 links)

| Missing File | Referenced By | Action |
|--------------|---------------|--------|
| `docs/development/performance-benchmarking.md` | build-commands.md, validation-framework.md | ➡️ Update links to `docs/performance-benchmarking.md` (root docs/) |
| `docs/development/VALIDATION.md` | validation-framework.md | ➡️ Update to `validation-framework.md` (self-reference fix) |
| `docs/development/VALIDATION_QUICK_START.md` | validation-framework.md | ➡️ Update to `validation-ci.md` or add quickstart section |
| `docs/development/concurrency-caps.md` | test-suite.md | ❌ Remove link (concept documented in test-suite.md itself) |
| `docs/development/performance-tracking.md` | test-suite.md | ➡️ Update to `../performance-benchmarking.md` |
| `docs/development/streaming-api.md` | test-suite.md | 📝 Create stub or remove link (streaming not yet implemented) |
| `docs/development/cli-usage.md` | Multiple | ➡️ Update to `../CLI_USAGE.md` (root docs/) |

### 3. Missing docs/reference/ Files (7 links)

| Missing File | Referenced By | Action |
|--------------|---------------|--------|
| `docs/reference/cli-reference.md` | use-qk256-models.md | ➡️ Update to `../../CLAUDE.md` (CLI commands section) |
| `docs/reference/prompt-templates.md` | troubleshoot-intelligibility.md | ➡️ Update to `../../CLAUDE.md` (Prompt Templates section) |
| `docs/reference/model-validation.md` | correction-policy.md | ➡️ Update to `validation-gates.md` |
| `docs/reference/COMPATIBILITY.md` | MIGRATION.md | ➡️ Update to `api-compatibility.md` or `../../COMPATIBILITY.md` |
| `docs/reference/LICENSE` | INSTALLATION.md | ➡️ Update to `../../LICENSE` (root) |
| `docs/reference/getting-started.md` | api-compatibility.md | ➡️ Update to `../getting-started.md` |
| `docs/reference/reference.md` | _TEMPLATE.md | ⚠️ Template placeholder - ignore or fix template |

---

## P2 - Low Priority (5 links)

### 4. Archived Documentation Stubs

These are references to historical documentation that should be in `docs/archive/`:

| File | Status | Action |
|------|--------|--------|
| `docs/reports/ISSUE_249_ANALYSIS.md` | Archived | ✅ Already excluded via `.lychee.toml` |
| `docs/reports/SPEC_*.md` | Archived | ✅ Already excluded via `.lychee.toml` |
| `docs/archive/**/*.md` | Historical | ✅ Already excluded via `.lychee.toml` |

**No action required** - already excluded from link checking.

---

## P3 - Informational (0 links)

### 5. External Links (Excluded)

External URLs are excluded via `.lychee.toml` offline mode:
- GitHub URLs (PR #475, issues, etc.)
- Documentation sites
- Package registries

**Total excluded**: 136 external links (not counted in debt)

---

## Resolution Plan

### Phase 1: Immediate Fixes (P0)
- [ ] Run bulk sed command to fix 32 PR #475 path references
- [ ] Verify with ripgrep
- [ ] Commit with message: `docs: fix PR_475 report path in ci/solutions/ (32 files)`

### Phase 2: Documentation Redirects (P1)
- [ ] Update development/ missing links (7 files)
- [ ] Update reference/ missing links (7 files)
- [ ] Run lychee validation
- [ ] Commit with message: `docs: redirect missing doc links to existing files`

### Phase 3: CI Integration
- [ ] Ensure `.lychee.toml` excludes `docs/archive/` (already done)
- [ ] CI link-check job should be **green** after Phase 1+2
- [ ] Add link-check to PR gate (non-blocking initially)

---

## Validation Commands

```bash
# Check current broken link count
lychee --config .lychee.toml "**/*.md" 2>&1 | grep "🚫"

# After Phase 1 - should reduce by 32
lychee --config .lychee.toml "ci/solutions/**/*.md" 2>&1 | grep "Error"

# After Phase 2 - should be near zero
lychee --config .lychee.toml "docs/**/*.md" "README.md" "CLAUDE.md" 2>&1 | grep "Error"

# Final validation
lychee --config .lychee.toml "**/*.md" --accept 200,429 --no-progress
```

---

## Owner Assignment

| Area | Owner | GitHub Handle |
|------|-------|---------------|
| ci/solutions/ fixes | CI/Infra Team | @ci-team |
| docs/development/ | Documentation Team | @docs-team |
| docs/reference/ | API Documentation | @api-docs |
| Archive management | Maintenance | @maintenance |

---

## Progress Tracking

- **2025-10-23**: Initial debt audit complete (83 links)
- **Next Review**: After Phase 1 completion
- **Target**: < 10 broken links by 2025-10-30

---

## Notes

- **Archive Strategy**: `docs/archive/` already excluded via `.lychee.toml:15-17`
- **CI Status**: Link-check job in `.github/workflows/ci.yml` (non-blocking observer)
- **Upstream Report**: See `DOCS_LINK_VALIDATION_REPORT.md` for full lychee output

---

**End of Link Debt Report**
