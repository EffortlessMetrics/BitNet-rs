# PR #475 - Follow-up Items

**Generated**: 2025-11-03
**Source**: Copilot review comments on PR #475
**Status**: Non-blocking nitpicks deferred to post-merge

---

## Documentation Organization

### 1. Move LOGGING_AND_DIAGNOSTIC_PATTERNS.md to docs/
**Priority**: Low
**Effort**: 5 minutes

**Context**: LOGGING_AND_DIAGNOSTIC_PATTERNS.md (~800 lines) is in the root directory but should be in `docs/` for better organization.

**Action**:
```bash
git mv LOGGING_AND_DIAGNOSTIC_PATTERNS.md docs/development/logging-patterns.md
# Update references in CLAUDE.md and other docs
```

**Rationale**: Root directory should contain only essential files (README, LICENSE, CONTRIBUTING, CLAUDE.md).

---

### 2. Consolidate Duplicate MVP Warnings in README.md
**Priority**: Low
**Effort**: 15 minutes

**Context**: MVP performance/quality warnings appear in multiple sections:
- Line 12: Status line with link to limitations
- Lines 132-145: Table with performance characteristics
- Lines 254-279: Dedicated "Status & Limitations" section

**Action**:
- Keep single comprehensive warning in "Status & Limitations" section
- Replace duplicates with references: "See [Status & Limitations](#status-limitations) for details"

**Rationale**: Avoid documentation drift when updating warnings/limitations.

---

## CI/CD Improvements

### 3. Add Weekly Link Checker with External Links
**Priority**: Medium
**Effort**: 20 minutes

**Context**: `.lychee.toml` uses `offline = true` (fast but skips external links). Need periodic external link validation.

**Action**:
```yaml
# .github/workflows/weekly-link-check.yml
name: Weekly Link Check
on:
  schedule:
    - cron: '0 0 * * 0'  # Sunday midnight
  workflow_dispatch:

jobs:
  check-links:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: lycheeverse/lychee-action@v2
        with:
          args: --offline false --max-retries 3 '**/*.md'
```

**Rationale**: Catch external link rot without slowing down PR CI.

---

### 4. Document Label-Gated Workflows
**Priority**: Medium
**Effort**: 30 minutes

**Context**: PR #475 introduces label-based workflow control (`lut`, `coverage`, `framework`, etc.) but lacks user documentation.

**Action**:
```bash
# Create docs/ci/labels.md documenting:
# - Available labels and their purpose
# - Which workflows each label triggers
# - When to use heavy validation labels
# - Add reference to PR template
```

**Rationale**: Contributors need to know how to opt-in to heavy validation lanes.

---

## Dependency Hygiene

### 5. Separate Dependency Updates from Feature PRs
**Priority**: Low (Note for Future)
**Effort**: N/A (process improvement)

**Context**: PR #475 bundles 6+ dependency updates (memmap2, clap_complete, indicatif, cudarc, proptest, etc.) with feature changes.

**Recommendation** (for future PRs):
- Separate dependency updates into dedicated "chore: update dependencies" PRs
- Makes bisecting easier if issues arise
- Simplifies rollback if dependency update causes regression

**Examples from this PR**:
- `memmap2 = "0.9.8"`
- `clap_complete = "4.5.39"`
- `indicatif = "0.17.7"`
- `cudarc = "0.17.4"`
- `proptest = "1.0.0"`

**Note**: Acceptable for MVP integration PR; enforce for post-MVP.

---

## Content Verification

### 6. Verify EXPLORATION_INDEX.md Content Preservation
**Priority**: Low
**Effort**: 30 minutes

**Context**: EXPLORATION_INDEX.md was completely rewritten (357 → 312 lines). Changed from "Codebase Exploration" to "Backend Routing Exploration".

**Action**:
```bash
# Review git diff to ensure valuable content from previous version was preserved
git show HEAD~1:EXPLORATION_INDEX.md > /tmp/old_index.md
git show HEAD:EXPLORATION_INDEX.md > /tmp/new_index.md
diff -u /tmp/old_index.md /tmp/new_index.md | less
```

**Verify**:
- Key architecture insights preserved
- Important file mappings retained
- Cross-references updated

---

## Documentation-Only Items

### 7. IMPLEMENTATION_CHANGES_BACKEND_ERRORS.md Status
**Priority**: N/A (Intentional)
**Effort**: N/A

**Context**: Copilot flagged this as "documentation without implementation". This is **intentional** - it's a design document for future work.

**Action**: None required. File correctly located and documented.

**Note**: Consider adding "Status: Planned" or "Status: Design Doc" to header if confusion persists.

---

## Platform-Specific Notes

### 8. Improve ripgrep Installation Instructions
**Priority**: Low
**Effort**: 5 minutes

**Context**: CONTRIBUTING.md suggests `sudo apt-get install ripgrep`, which only works on Ubuntu 18.10+.

**Current**:
```markdown
# Ubuntu:  sudo apt-get install ripgrep
```

**Suggested Improvement**:
```markdown
# Ubuntu 18.10+:  sudo apt-get install ripgrep
# Older Ubuntu:   sudo snap install ripgrep --classic
# Or via cargo:   cargo install ripgrep
# See: https://github.com/BurntSushi/ripgrep#installation
```

**Rationale**: Better cross-platform support, fewer setup failures.

---

## Archive Issues (No Action)

### 9. Archive Files - Intentional Staging
**Files**:
- `archive/gh-issues-to-create/p1-task-4-ffi-hygiene-finalization.md`
- `archive/reports/COMPLIANCE_TEST_ACTION_PLAN.md` (security concern noted but documented)

**Action**: None. These are correctly archived and documented.

---

## Summary

| Priority | Count | Total Effort |
|----------|-------|--------------|
| High     | 0     | -            |
| Medium   | 2     | ~50 min      |
| Low      | 5     | ~80 min      |
| **Total** | **7** | **~2 hours** |

**Recommendation**:
- Address Medium priority items in next maintenance window
- Bundle Low priority items in a "chore: documentation cleanup" PR
- Track Dependency Hygiene (#5) as process improvement for v0.2.0+

---

**Note**: All critical/blocking issues from Copilot review were addressed in the main commits. These are quality-of-life improvements.
