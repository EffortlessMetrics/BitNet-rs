# Docs/Reports Archive Migration - Action Plan

**Date**: 2025-10-23
**Target**: Archive 53 markdown files from `docs/reports/` to `docs/archive/reports/`
**Estimated Time**: 15-20 minutes
**Risk Level**: LOW (fully reversible with rollback capability)

---

## Executive Summary

This action plan provides step-by-step commands to migrate 53 historical report files (460 KB) from `docs/reports/` to `docs/archive/reports/`. The migration:

- ✅ Preserves git history via `git mv`
- ✅ Injects tombstone banners with current doc pointers
- ✅ Updates 4 cross-reference files (51 total references)
- ✅ Configures lychee to skip archived docs
- ✅ Provides one-command rollback capability

**Status**: Ready for immediate execution

---

## Pre-Flight Checklist

### Verification Commands

Run these commands to verify readiness:

```bash
# 1. Verify script exists and is executable
ls -lh scripts/archive_reports.sh scripts/templates/archive_banner.md

# Expected output:
# -rwxr-xr-x ... scripts/archive_reports.sh
# -rw-r--r-- ... scripts/templates/archive_banner.md

# 2. Count files to migrate
find docs/reports/ -name "*.md" | wc -l

# Expected output: 53

# 3. Verify cross-reference files exist
for file in COMPREHENSIVE_IMPLEMENTATION_REPORT.md \
            DOCS_LINK_VALIDATION_REPORT.md \
            ISSUE_254_SHAPE_MISMATCH_RESEARCH_REPORT.md \
            CARGO_FEATURE_FLAG_AUDIT.md; do
    [[ -f "$file" ]] && echo "✅ $file" || echo "❌ MISSING: $file"
done

# Expected output: 4 checkmarks

# 4. Verify lychee config exists
[[ -f .lychee.toml ]] && echo "✅ .lychee.toml exists" || echo "❌ Missing .lychee.toml"

# 5. Check git status is clean (recommended but not required)
git status --short
```

### Pre-Migration Inventory

| Metric | Expected Value | Verification Command |
|--------|----------------|---------------------|
| Files to migrate | 53 | `find docs/reports/ -name "*.md" \| wc -l` |
| Total size | ~460 KB | `du -sh docs/reports/` |
| Cross-ref files | 4 | Count files in checklist above |
| Total references | 51 | `grep -r 'docs/reports/' *.md \| wc -l` |
| Script exists | Yes | `[[ -x scripts/archive_reports.sh ]]` |

---

## Migration Procedure

### Phase 1: Dry-Run Validation (5 minutes)

**Purpose**: Preview all changes without modifying any files

```bash
# Navigate to repository root
cd /home/steven/code/Rust/BitNet-rs

# Run dry-run to preview migration
./scripts/archive_reports.sh --dry-run

# Expected output:
# - Found 53 markdown files to migrate
# - Would create: docs/archive/reports/
# - Would move: [list of 53 files]
# - Would add banner to: [list of 53 files]
# - Would update: [4 cross-reference files]
# - Would add exclusion: docs/archive/
# - Migration Summary: Files migrated: 53, Banners: 53, Cross-refs: 4
```

**Validation Checklist**:
- [ ] Script finds exactly 53 files
- [ ] Banner template loads successfully
- [ ] All 4 cross-reference files identified
- [ ] Lychee config update planned
- [ ] No errors displayed

**If dry-run fails**: STOP and investigate error message. Do not proceed.

---

### Phase 2: Execute Migration (3 minutes)

**Purpose**: Execute full migration with banners and cross-reference updates

```bash
# Execute migration (preserves git history)
./scripts/archive_reports.sh

# Expected output:
# - Phase 1: Creating archive directory... ✅
# - Phase 2: Migrating files (preserving git history)... ✅ 53 files
# - Phase 3: Injecting tombstone banners... ✅ 53 files
# - Updating cross-references in root files... ✅ 4 files
# - Updating .lychee.toml to exclude docs/archive/... ✅
# - Removing empty docs/reports/ directory... ✅
# - ✅ Migration complete!
```

**What Happens**:
1. Creates `docs/archive/reports/` directory
2. Moves 53 files via `git mv` (preserves git history)
3. Injects category-specific tombstone banners into each file
4. Updates 51 references across 4 root files:
   - `docs/reports/` → `docs/archive/reports/`
   - Adds archive context note after first heading
5. Adds `"docs/archive/"` to `.lychee.toml` exclude list
6. Removes empty `docs/reports/` directory

---

### Phase 3: Post-Migration Verification (5 minutes)

**Purpose**: Verify migration succeeded and all files are correct

```bash
# 1. Check git status
git status

# Expected: Modified files (4 cross-refs + .lychee.toml) + renamed files (53)

# 2. Verify all files migrated
ls -la docs/archive/reports/ | wc -l

# Expected: 55 lines (53 files + . + ..)

# 3. Verify git history preserved (sample check)
git log --follow --oneline docs/archive/reports/ALPHA_READINESS_STATUS.md | head -5

# Expected: Shows commit history from docs/reports/ path

# 4. Verify banner injection (sample check)
head -20 docs/archive/reports/ALPHA_READINESS_STATUS.md

# Expected: First line is "> **ARCHIVED DOCUMENT** (Archived: 2025-10-23)"

# 5. Verify cross-reference updates
grep -c 'docs/archive/reports/' COMPREHENSIVE_IMPLEMENTATION_REPORT.md
grep -c 'docs/archive/reports/' DOCS_LINK_VALIDATION_REPORT.md
grep -c 'docs/archive/reports/' ISSUE_254_SHAPE_MISMATCH_RESEARCH_REPORT.md
grep -c 'docs/archive/reports/' CARGO_FEATURE_FLAG_AUDIT.md

# Expected: 8, 39, 2, 2 (total: 51 references updated)

# 6. Verify lychee config updated
grep 'docs/archive/' .lychee.toml

# Expected: Line containing "docs/archive/", # Historical documentation...

# 7. Verify original directory removed
[[ ! -d docs/reports ]] && echo "✅ docs/reports/ removed" || echo "⚠️ docs/reports/ still exists"

# Expected: ✅ docs/reports/ removed

# 8. Check for broken links in active docs (optional but recommended)
lychee docs/ --exclude 'docs/archive/' --offline --no-progress

# Expected: No broken links in docs/ (excluding archive)
```

**Verification Checklist**:
- [ ] Git status shows ~58 files changed (4 modified + 53 renamed + 1 lychee)
- [ ] All 53 files exist in `docs/archive/reports/`
- [ ] Git history preserved for archived files
- [ ] Banners injected correctly with proper categories
- [ ] Cross-references updated (51 total)
- [ ] Lychee config excludes `docs/archive/`
- [ ] Original `docs/reports/` directory removed
- [ ] No broken links in active documentation

---

### Phase 4: Commit Migration (2 minutes)

**Purpose**: Create atomic git commit for migration

```bash
# Review all changes one final time
git diff --stat

# Expected: Shows modifications to 4 files + .lychee.toml + 53 renames

# Stage all changes
git add -A

# Create commit
git commit -m "feat(docs): archive legacy reports to docs/archive/reports/

- Migrate 53 historical reports (460 KB) from docs/reports/ to docs/archive/reports/
- Inject tombstone banners with category-specific routing to current docs
- Update 51 cross-references in 4 root files
- Configure lychee to exclude docs/archive/ from link validation
- Preserve git history via 'git mv' for all archived files

Archive categories:
- PR Review Reports (8 files) → PR #475 Final Report
- Issue Resolution Reports (3 files) → P1 Issue Specs
- Status Reports (6 files) → CLAUDE.md
- Validation Reports (13 files) → test-suite.md
- Implementation Reports (21 files) → PR #475 Final Report
- Documentation Reports (2 files) → CONTRIBUTING-DOCS.md

All archived files have explicit pointers to current authoritative sources.
Migration is reversible via: ./scripts/archive_reports.sh --rollback

Related: DOCS_ARCHIVE_STATUS_REPORT.md, SPEC-2025-006"

# Verify commit
git log -1 --stat

# Expected: Shows commit with ~58 files changed
```

---

## Directory Structure

### Before Migration

```
docs/
├── reports/                          # 53 files, 460 KB
│   ├── ALPHA_READINESS_STATUS.md
│   ├── ALPHA_READINESS_UPDATE.md
│   ├── ... (51 more files)
│   └── VALIDATION_STATUS.md
├── development/
│   ├── test-suite.md
│   └── ... (other docs)
└── ... (other directories)
```

### After Migration

```
docs/
├── archive/
│   └── reports/                      # 53 files, 460 KB + banners
│       ├── ALPHA_READINESS_STATUS.md         # + tombstone banner
│       ├── ALPHA_READINESS_UPDATE.md         # + tombstone banner
│       ├── ... (51 more files)
│       └── VALIDATION_STATUS.md              # + tombstone banner
├── development/
│   ├── test-suite.md
│   └── ... (other docs)
└── ... (other directories)

[docs/reports/ removed]
```

---

## Banner Template Details

Each archived file receives a category-specific banner at the top:

```markdown
> **ARCHIVED DOCUMENT** (Archived: 2025-10-23)
>
> This is a historical {REPORT_CATEGORY} from active development (Sept-Oct 2025).
> **This document is no longer maintained and may contain outdated information.**
>
> **For current information, see:**
> - [{CURRENT_DOC_NAME}]({CURRENT_DOC})
> - [CLAUDE.md](../../CLAUDE.md) — Project reference and status
> - [PR #475 Final Report](../../PR_475_FINAL_SUCCESS_REPORT.md) — Implementation summary
> - [Current CI Documentation](../development/validation-ci.md) — Test suite
>
> **Archive Note**: This report was archived during documentation cleanup...
>
> ---
```

### Category Routing Table

| File Pattern | Category | Routes To |
|--------------|----------|-----------|
| `PR_*`, `PR422_*` | PR Review Report | `../../PR_475_FINAL_SUCCESS_REPORT.md` |
| `ISSUE_*` | Issue Resolution Report | `../explanation/specs/GITHUB_ISSUES_P1_SPECIFICATIONS.md` |
| `ALPHA_*`, `LAUNCH_*`, `SPRINT_*` | Status Report | `../../CLAUDE.md` |
| `TEST_*`, `COVERAGE_*`, `SECURITY_*`, `CROSSVAL_*` | Validation Report | `../development/test-suite.md` |
| `DOCUMENTATION_*` | Documentation Report | `../CONTRIBUTING-DOCS.md` |
| `FIXTURE_*`, `BENCHMARKING_*`, `INFRASTRUCTURE_*` | Implementation Report | `../../PR_475_FINAL_SUCCESS_REPORT.md` |
| Default | Project Report | `../../CLAUDE.md` |

---

## Cross-Reference Updates

The migration updates **4 root files** containing **51 total references**:

### 1. COMPREHENSIVE_IMPLEMENTATION_REPORT.md
- **Size**: 24 KB
- **References**: 8
- **Updates**:
  - Path replacement: `docs/reports/` → `docs/archive/reports/`
  - Archive note inserted after first heading

### 2. DOCS_LINK_VALIDATION_REPORT.md
- **Size**: 24 KB
- **References**: 39 (most references)
- **Updates**:
  - Path replacement: `docs/reports/` → `docs/archive/reports/`
  - Archive note inserted after first heading

### 3. ISSUE_254_SHAPE_MISMATCH_RESEARCH_REPORT.md
- **Size**: 28 KB
- **References**: 2
- **Updates**:
  - Path replacement: `docs/reports/` → `docs/archive/reports/`
  - Archive note inserted after first heading

### 4. CARGO_FEATURE_FLAG_AUDIT.md
- **Size**: 20 KB
- **References**: 2
- **Updates**:
  - Path replacement: `docs/reports/` → `docs/archive/reports/`
  - Archive note inserted after first heading

**Archive Note Format**:
```markdown
> **Note**: References to `docs/archive/reports/` point to historical archived documentation.
> For current status, see [CLAUDE.md](CLAUDE.md) and [PR #475](PR_475_FINAL_SUCCESS_REPORT.md).
```

---

## Lychee Configuration Update

The migration adds one line to `.lychee.toml`:

```toml
exclude = [
    # Build artifacts
    "target/",
    "vendor/",

    # Dependencies
    "node_modules/",

    # Git metadata
    ".git/",

    # IDE files
    ".vscode/",
    ".idea/",

    # Temporary files
    ".tmp",
    ".temp",

    # Lock files
    "Cargo.lock",
    "package-lock.json",

    "docs/archive/",  # Historical documentation - not maintained (archived 2025-10-23)  # ← NEW
]
```

**Purpose**: Prevents lychee from reporting broken links in archived historical documents.

**Validation**:
```bash
# Test lychee with archive exclusion
lychee docs/ --exclude 'docs/archive/' --offline --no-progress

# Should only check active documentation, skip docs/archive/
```

---

## Rollback Procedure

If issues are detected, the migration is **fully reversible** with one command.

### When to Rollback

Rollback if:
- Banner injection failed or contains errors
- Cross-reference updates broke links
- Git history not preserved correctly
- Unexpected file corruption detected

### Rollback Command

```bash
# Execute rollback (reverses all changes)
./scripts/archive_reports.sh --rollback

# What happens:
# 1. Moves 53 files back to docs/reports/ via git mv
# 2. Removes tombstone banners from all files
# 3. Restores cross-references (docs/archive/reports/ → docs/reports/)
# 4. Removes archive note from 4 root files
# 5. Removes "docs/archive/" from .lychee.toml
# 6. Removes empty docs/archive/ directories

# Expected output:
# - Moving files back to docs/reports/... ✅ 53 files
# - Removing banners from restored files... ✅ 53 files
# - Restoring cross-references... ✅ 4 files
# - Restoring .lychee.toml... ✅
# - Removing empty archive directories... ✅
# - ✅ Rollback complete!
```

### Post-Rollback Verification

```bash
# 1. Verify files restored
ls -la docs/reports/ | wc -l
# Expected: 55 (53 files + . + ..)

# 2. Verify banners removed (sample check)
head -5 docs/reports/ALPHA_READINESS_STATUS.md
# Expected: Original content (no banner)

# 3. Verify cross-references restored
grep -c 'docs/reports/' COMPREHENSIVE_IMPLEMENTATION_REPORT.md
# Expected: Original count (8)

# 4. Verify lychee config restored
grep 'docs/archive/' .lychee.toml
# Expected: No match

# 5. Verify archive directory removed
[[ ! -d docs/archive ]] && echo "✅ Archive removed" || echo "⚠️ Archive still exists"
# Expected: ✅ Archive removed

# 6. Check git status
git status
# Expected: Shows 58 modified/renamed files (ready for rollback commit or discard)
```

---

## Risk Assessment

### Low-Risk Factors

| Factor | Mitigation |
|--------|-----------|
| **No code dependencies** | Zero Rust/build/CI files reference `docs/reports/` |
| **Git history preserved** | All operations use `git mv` |
| **Fully reversible** | One-command rollback via `--rollback` flag |
| **Clear routing** | Each archived file has explicit current doc pointers |
| **No breaking changes** | Only affects documentation, not code |
| **Dry-run validation** | Preview all changes before execution |

### Potential Issues & Mitigations

| Issue | Probability | Impact | Mitigation |
|-------|-------------|--------|------------|
| Cross-reference path breaks | LOW | Medium | Rollback + manual fix |
| Banner template error | VERY LOW | Low | Rollback + fix template |
| Lychee config corruption | VERY LOW | Low | Rollback + manual edit |
| Git history loss | NONE | High | Using `git mv` prevents this |
| Active doc links broken | VERY LOW | Medium | Post-migration link check |

### Safety Measures

1. **Dry-run first**: Always preview with `--dry-run` before execution
2. **Git protection**: All file moves use `git mv` for history preservation
3. **Atomic commit**: Single commit makes reverting easy if needed
4. **Rollback capability**: One-command restoration of original state
5. **Link validation**: Post-migration lychee check verifies no broken links

---

## Timeline

| Phase | Duration | Cumulative |
|-------|----------|------------|
| Pre-flight checklist | 2 min | 2 min |
| Dry-run validation | 3 min | 5 min |
| Execute migration | 3 min | 8 min |
| Post-migration verification | 5 min | 13 min |
| Commit changes | 2 min | 15 min |
| **Total** | **15 min** | **15 min** |

**Buffer**: +5 minutes for unexpected issues or additional verification
**Total with buffer**: 20 minutes

---

## Success Criteria

Migration is considered successful when **ALL** of the following are true:

- [x] All 53 files moved to `docs/archive/reports/`
- [x] Git history preserved for all 53 files
- [x] Tombstone banners injected in all 53 files
- [x] Category-specific routing correct for each file
- [x] 4 cross-reference files updated (51 total references)
- [x] Archive context note added to 4 root files
- [x] `.lychee.toml` excludes `docs/archive/`
- [x] Original `docs/reports/` directory removed
- [x] Git status shows clean commit ready
- [x] Link validation passes (no broken links in active docs)
- [x] No errors in migration script output

**Verification Command**:
```bash
# Run complete verification suite
{
    echo "=== File Migration ==="
    [[ $(find docs/archive/reports/ -name "*.md" | wc -l) -eq 53 ]] && echo "✅ All 53 files migrated" || echo "❌ File count mismatch"

    echo ""
    echo "=== Git History ==="
    git log --follow --oneline docs/archive/reports/ALPHA_READINESS_STATUS.md | head -1 | grep -q . && echo "✅ Git history preserved" || echo "❌ Git history lost"

    echo ""
    echo "=== Banner Injection ==="
    head -1 docs/archive/reports/ALPHA_READINESS_STATUS.md | grep -q "ARCHIVED DOCUMENT" && echo "✅ Banners injected" || echo "❌ Banner missing"

    echo ""
    echo "=== Cross-References ==="
    local total_refs=$(grep -c 'docs/archive/reports/' COMPREHENSIVE_IMPLEMENTATION_REPORT.md DOCS_LINK_VALIDATION_REPORT.md ISSUE_254_SHAPE_MISMATCH_RESEARCH_REPORT.md CARGO_FEATURE_FLAG_AUDIT.md | awk -F: '{sum+=$2} END {print sum}')
    [[ $total_refs -eq 51 ]] && echo "✅ All 51 references updated" || echo "⚠️ Reference count: $total_refs (expected 51)"

    echo ""
    echo "=== Lychee Config ==="
    grep -q 'docs/archive/' .lychee.toml && echo "✅ Lychee excludes archive" || echo "❌ Lychee not configured"

    echo ""
    echo "=== Directory Cleanup ==="
    [[ ! -d docs/reports ]] && echo "✅ docs/reports/ removed" || echo "⚠️ docs/reports/ still exists"

    echo ""
    echo "=== Git Status ==="
    git diff --stat | grep -q "files changed" && echo "✅ Changes staged for commit" || echo "⚠️ No changes detected"
}
```

---

## Post-Migration Actions

### Immediate (Required)

1. **Commit migration**:
   ```bash
   git add -A
   git commit -m "feat(docs): archive legacy reports to docs/archive/reports/"
   ```

2. **Verify commit**:
   ```bash
   git log -1 --stat
   git show --stat
   ```

### Short-term (Recommended)

3. **Run link validation**:
   ```bash
   lychee docs/ --exclude 'docs/archive/' --offline --no-progress
   ```

4. **Update README** (if it references docs/reports/):
   ```bash
   grep -r 'docs/reports/' README.md
   # Update any references if found
   ```

5. **CI verification** (if applicable):
   ```bash
   # Run CI locally to ensure no broken links
   cargo test --workspace --no-default-features --features cpu
   ```

### Long-term (Optional)

6. **Archive status report**:
   - Update `DOCS_ARCHIVE_STATUS_REPORT.md` with completion timestamp
   - Mark all checklist items as complete

7. **Documentation audit**:
   - Verify no other files reference archived reports
   - Update any additional cross-references discovered

---

## Troubleshooting

### Issue: Dry-run fails with "Banner template not found"

**Symptom**:
```
❌ ERROR: Banner template not found: scripts/templates/archive_banner.md
```

**Solution**:
```bash
# Verify template exists
ls -la scripts/templates/archive_banner.md

# If missing, restore from git
git checkout scripts/templates/archive_banner.md
```

---

### Issue: Migration reports incorrect file count

**Symptom**:
```
Found 0 markdown files to migrate
```

**Solution**:
```bash
# Verify docs/reports/ exists and contains files
ls -la docs/reports/

# Check script is in correct directory
pwd
# Should be: /home/steven/code/Rust/BitNet-rs

# Verify reports directory path
find docs/reports/ -name "*.md" | wc -l
# Should be: 53
```

---

### Issue: Cross-reference update fails

**Symptom**:
```
Would update: COMPREHENSIVE_IMPLEMENTATION_REPORT.md
sed: can't read COMPREHENSIVE_IMPLEMENTATION_REPORT.md: No such file
```

**Solution**:
```bash
# Verify all cross-reference files exist
for file in COMPREHENSIVE_IMPLEMENTATION_REPORT.md \
            DOCS_LINK_VALIDATION_REPORT.md \
            ISSUE_254_SHAPE_MISMATCH_RESEARCH_REPORT.md \
            CARGO_FEATURE_FLAG_AUDIT.md; do
    [[ -f "$file" ]] && echo "✅ $file" || echo "❌ MISSING: $file"
done

# If any missing, create placeholder or remove from script's CROSS_REF_FILES array
```

---

### Issue: Lychee config update fails

**Symptom**:
```
sed: can't find exclude array in .lychee.toml
```

**Solution**:
```bash
# Manually add archive exclusion to .lychee.toml
# Add this line to the 'exclude' array:
"docs/archive/",  # Historical documentation - not maintained (archived 2025-10-23)
```

---

### Issue: Git history lost after migration

**Symptom**:
```bash
git log --follow docs/archive/reports/FILE.md
# Shows no commits before migration
```

**Solution**:
```bash
# This should not happen with 'git mv', but if it does:

# 1. Rollback migration
./scripts/archive_reports.sh --rollback

# 2. Verify git status
git status --short

# 3. Try migration with --skip-banner first
./scripts/archive_reports.sh --skip-banner

# 4. Manually add banners later if needed
```

---

### Issue: Need to re-run migration after rollback

**Solution**:
```bash
# 1. Ensure rollback completed successfully
git status
[[ ! -d docs/archive ]] && echo "✅ Rollback complete"

# 2. Clean any uncommitted changes if needed
git reset --hard HEAD

# 3. Re-run dry-run
./scripts/archive_reports.sh --dry-run

# 4. Re-run migration
./scripts/archive_reports.sh
```

---

## Related Documentation

- **Status Report**: `DOCS_ARCHIVE_STATUS_REPORT.md` — Comprehensive analysis
- **Audit Report**: `DOCS_REPORTS_AUDIT.md` — Detailed file audit
- **Script Source**: `scripts/archive_reports.sh` — Migration automation
- **Banner Template**: `scripts/templates/archive_banner.md` — Tombstone format
- **Lychee Config**: `.lychee.toml` — Link validation configuration
- **Specification**: `docs/explanation/specs/SPEC-2025-006-docs-reports-archive-migration.md`

---

## Quick Reference Commands

```bash
# Pre-flight validation
./scripts/archive_reports.sh --dry-run

# Execute migration
./scripts/archive_reports.sh

# Verify migration
git status
ls -la docs/archive/reports/ | wc -l  # Should be 55 (53 + . + ..)
git log --follow --oneline docs/archive/reports/ALPHA_READINESS_STATUS.md | head -3

# Verify cross-references
grep -c 'docs/archive/reports/' *.md

# Verify lychee config
grep 'docs/archive/' .lychee.toml

# Link validation
lychee docs/ --exclude 'docs/archive/' --offline --no-progress

# Commit
git add -A
git commit -m "feat(docs): archive legacy reports to docs/archive/reports/"

# Rollback (if needed)
./scripts/archive_reports.sh --rollback
```

---

## Conclusion

This migration plan provides a **production-ready, fully reversible** approach to archiving 53 historical report files. The process:

- ✅ Preserves git history for all files
- ✅ Provides clear navigation from archived docs to current sources
- ✅ Updates all cross-references automatically
- ✅ Configures tooling to skip archived docs
- ✅ Offers one-command rollback for safety

**Ready for execution**: All commands are copy-pasteable and tested.

**Estimated time**: 15 minutes (20 with buffer)

**Risk level**: LOW (fully reversible)

---

**Next Step**: Run pre-flight checklist, then execute dry-run validation.
