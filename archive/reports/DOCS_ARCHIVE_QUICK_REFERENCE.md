# Docs/Reports Archive Migration - Quick Reference

**Status**: Ready to Execute  
**Files to Archive**: 53 markdown files (460 KB)  
**Migration Time**: 15-20 minutes  
**Risk Level**: LOW  

---

## One-Liner Execution

```bash
./scripts/archive_reports.sh --dry-run && ./scripts/archive_reports.sh
```

---

## What's Happening

**Before**:
- `docs/reports/` contains 53 historical reports (Sept-Oct 2025)
- Root files reference these reports
- Lychee checks broken links in archived files

**After**:
- `docs/archive/reports/` contains same 53 files with tombstone banners
- Each file links to current authoritative documentation
- Lychee excludes archive from link checks

---

## The 5-Minute Workflow

```bash
# 1. Preview (30 seconds)
./scripts/archive_reports.sh --dry-run

# 2. Execute (2 minutes)
./scripts/archive_reports.sh

# 3. Verify (3 minutes)
git status
git diff | head -100
ls -la docs/archive/reports/ | head -5

# 4. Commit (1 minute)
git add -A
git commit -m "feat(docs): archive legacy reports to docs/archive/reports/"

# 5. Confirm (optional)
git log --oneline -n 5
```

---

## File Categories & Routing

| Type | Count | Routes To |
|------|-------|-----------|
| PR Review Reports | 8 | PR #475 Final Report |
| Issue Resolution Reports | 3 | Current Issue Specs |
| Status & Launch Reports | 6 | CLAUDE.md |
| Validation & Test Reports | 9 | test-suite.md |
| Cross-Validation Reports | 4 | test-suite.md |
| Documentation Reports | 2 | CONTRIBUTING-DOCS.md |
| Implementation Reports | 21 | PR #475 Final Report |
| Utility Reports | 1 | CLAUDE.md |
| **TOTAL** | **53** | **5 destinations** |

---

## Banner Preview

```markdown
> **ARCHIVED DOCUMENT** (Archived: 2025-10-23)
>
> This is a historical [CATEGORY] from active development (Sept-Oct 2025).
> **This document is no longer maintained and may contain outdated information.**
>
> **For current information, see:**
> - [CURRENT_DOC_NAME](CURRENT_DOC)
> - [CLAUDE.md](../../CLAUDE.md)
> - [PR #475 Final Report](../../PR_475_FINAL_SUCCESS_REPORT.md)
> - [Current CI Documentation](../development/validation-ci.md)
>
> ---
```

---

## Updated Cross-References

These 4 files will be updated automatically:

1. `COMPREHENSIVE_IMPLEMENTATION_REPORT.md`
2. `DOCS_LINK_VALIDATION_REPORT.md`
3. `ISSUE_254_SHAPE_MISMATCH_RESEARCH_REPORT.md`
4. `CARGO_FEATURE_FLAG_AUDIT.md`

---

## Rollback (If Needed)

```bash
./scripts/archive_reports.sh --rollback
```

This will:
- Move files back to `docs/reports/`
- Remove banners
- Restore cross-references
- Remove archive directory

---

## What's Safe

✅ Code dependencies: 0 (safe to migrate)  
✅ Git history: Preserved via `git mv`  
✅ External links: Unaffected  
✅ Active development: No blockers  

---

## What's New

✅ Archive directory: `docs/archive/reports/`  
✅ Tombstone banners: 53 files  
✅ Lychee exclusions: Updated  
✅ Cross-references: Updated (4 files)  

---

## Key Facts

- **53 files** → 460 KB
- **4 destinations** for current docs
- **0 code changes** needed
- **15-20 minutes** total time
- **100% reversible** with rollback

---

## Documentation

| Document | Purpose |
|----------|---------|
| DOCS_ARCHIVE_STATUS_REPORT.md | Comprehensive reference (643 lines) |
| DOCS_ARCHIVE_MIGRATION_INDEX.md | Navigation guide (481 lines) |
| DOCS_REPORTS_AUDIT.md | Why archival is needed |
| This file | Quick reference |

---

## Success = All of These

- [x] 53 files in `docs/archive/reports/`
- [x] Git history preserved
- [x] Banners injected (53)
- [x] Cross-refs updated (4)
- [x] Lychee config updated
- [x] `docs/reports/` removed
- [x] `git status` clean

---

## Current Documentation (Where to Find Info)

| Topic | Location |
|-------|----------|
| Project Status | `CLAUDE.md` |
| Implementation | `PR_475_FINAL_SUCCESS_REPORT.md` |
| Issues | `docs/explanation/specs/GITHUB_ISSUES_P1_SPECIFICATIONS.md` |
| Tests | `docs/development/test-suite.md` |
| Docs Practices | `docs/CONTRIBUTING-DOCS.md` |

---

## Ready?

```bash
./scripts/archive_reports.sh --dry-run
./scripts/archive_reports.sh
```

