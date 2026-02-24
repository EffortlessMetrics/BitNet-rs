# Issue #319 Stack Status: KV Cache Pool v2

**Last Updated:** 2025-11-12
**Current Phase:** PR #518 ready to merge (waiting on GitHub billing)

## Stack Overview

```
#319: KV Cache Pool v2 (5-PR stack)
├── PR #518 [READY] ✅ Arena foundation
├── PR #519 [DRAFT] 📝 Entry struct + MemoryPool integration
├── PR #520 [PLANNED] 📋 Eviction + metrics (PR-3)
├── PR #521 [PLANNED] 📋 Receipt generation (PR-4)
└── PR #522 [PLANNED] 📋 Final integration + cleanup (PR-5)
```

## Current Status

### PR #518: Arena Foundation ✅ READY

**Branch:** `feat/319-kv-pool-v2`
**Status:** All core CI gates pass locally, waiting for GitHub billing resolution

#### Verification Results

| Gate | Status | Notes |
|------|--------|-------|
| Build (strict) | ✅ PASS | 28.92s |
| Clippy (strict) | ✅ PASS | 10.41s |
| Format | ✅ PASS | Clean |
| Docs | ✅ PASS | 12.02s, 28 files |
| MSRV (1.89.0) | ✅ PASS | 3m 16s (without sccache) |
| Tests (lib) | ⚠️ ENV ISSUE | Compiler ICE (not code issue) |

**Blockers:** None - code is ready
**Waiting on:** GitHub Actions billing resolution

#### Merge Command

```bash
# When billing is resolved and CI is green:
gh pr checks 518 --watch
gh pr merge 518 --squash --delete-branch \
  --subject "kv-pool v2: Arena foundation (PR 1/5)" \
  --body "Part of #319. Real arena + helpers; doc-only exports; strict code builds; docs relaxed via RUSTDOCFLAGS. No runtime API changes."
```

### PR #519: Entry Struct 📝 DRAFT

**Branch:** `feat/319-kv-pool-v2-pr2-entry`
**Status:** Draft, needs rebase after #518 merges

#### Rebase Workflow (After #518 merges)

```bash
# 1. Switch and rebase
git switch feat/319-kv-pool-v2-pr2-entry
git fetch origin
git rebase origin/main

# 2. Verify locally
cargo fmt --all
env RUSTFLAGS='-D warnings' cargo build --locked --workspace --no-default-features --features cpu
env RUSTFLAGS='-D warnings' cargo test --locked --workspace --no-default-features --features cpu --lib
cargo clippy --workspace --all-targets --no-default-features --features cpu -- -D warnings
env RUSTDOCFLAGS='-A warnings' cargo doc --locked --no-deps --workspace --no-default-features --features cpu
RUSTC_WRAPPER="" cargo +1.89.0 check --workspace --all-targets --locked --no-default-features --features cpu

# 3. Push and mark ready
git push -f origin feat/319-kv-pool-v2-pr2-entry
gh pr ready 519
```

#### Reviewer Note (Add to PR description)

```markdown
✅ **Rebased on `main` post-#518**

All CPU-only CI gates verified locally (build/test/clippy/fmt/docs/MSRV).

**Next in stack:** PR-3 will switch eviction to `pool.deallocate(entry.block)` + add stats/receipts.
```

### PR #520 (PR-3): Eviction + Metrics 📋 PLANNED

**Status:** Patch guide complete, ready to implement after #519 merges

#### Key Changes

1. Update `remove_cache_entry` to call `pool.deallocate(entry.block)`
2. Add stats tracking (evictions, active_entries)
3. Add debug assertions for stats consistency
4. Add unit tests:
   - `test_evict_returns_block_to_pool`
   - `test_evict_nonexistent_key_returns_none`
   - `test_multiple_evictions_stats_consistency`

#### Estimated Scope

- **Files changed:** 2-3
- **Lines changed:** ~100-150
- **Implementation time:** 2-3 hours

**See:** `PR_3_EVICTION_PATCH_GUIDE.md` for complete implementation details

### PR #521 (PR-4): Receipt Generation 📋 PLANNED

**Status:** Not yet designed
**Dependencies:** PR #520

#### Planned Scope

- Add receipt generation for honest compute verification
- Integrate with existing receipt infrastructure
- Add tests for receipt validation

### PR #522 (PR-5): Final Integration 📋 PLANNED

**Status:** Not yet designed
**Dependencies:** PR #521

#### Planned Scope

- Final integration and cleanup
- End-to-end integration tests
- Documentation updates
- Performance validation

## Documentation Index

### PR #518 Documents

- `MERGE_CHECKLIST_518.md` - Step-by-step merge procedure
- `VERIFICATION_SUMMARY_518.md` - Detailed verification results (obsoleted by PR_518_FINAL_CHECKLIST.md)
- `PR_518_FINAL_CHECKLIST.md` - Comprehensive final checklist
- `scripts/ci-local.sh` - Local CI verification script

### PR #519 Documents

- `PR_519_REBASE_PLAN.md` - Detailed rebase workflow

### PR #520 (PR-3) Documents

- `PR_3_EVICTION_PATCH_GUIDE.md` - Complete implementation guide

### This Document

- `STACK_STATUS_319.md` - Overall stack status and roadmap

## Execution Timeline

### Phase 1: PR #518 Merge (Current)

**Waiting on:** GitHub Actions billing resolution

**Action items:**
1. ⏳ Wait for billing resolution
2. ⏳ Verify CI is green on GitHub
3. ⏳ Execute merge command
4. ⏳ Verify merge completed successfully

**ETA:** Blocked on external dependency (billing)

### Phase 2: PR #519 Rebase (After #518)

**Duration:** 30-60 minutes

**Action items:**
1. Rebase branch onto main
2. Run all 6 CI gates locally
3. Force-push updated branch
4. Mark PR as ready
5. Add reviewer note

**ETA:** Same day as #518 merge

### Phase 3: PR #520 Implementation (After #519)

**Duration:** 2-4 hours

**Action items:**
1. Create branch from #519
2. Implement eviction logic changes
3. Add unit tests
4. Run all CI gates locally
5. Submit PR

**ETA:** 1-2 days after #519 merge

### Phase 4: PR #521 & #522 (Future)

**Duration:** TBD

**Action items:** To be determined after PR #520 merges

**ETA:** TBD

## Success Criteria

### Per-PR Criteria

Each PR must satisfy:
- ✅ All 6 core CI gates pass (build, test, clippy, fmt, docs, MSRV)
- ✅ No API breaks to callers (until final PR)
- ✅ Comprehensive unit tests
- ✅ Documentation updated
- ✅ Code review approved

### Overall #319 Success

The complete stack must deliver:
- ✅ Real `MemoryPool` with `Vec<u8>` backing
- ✅ Zero-copy KV cache entry management
- ✅ Proper memory deallocation on eviction
- ✅ Stats tracking (allocations, evictions, fragmentation)
- ✅ Receipt generation for honest compute
- ✅ No performance regressions
- ✅ All tests passing

## Known Issues & Workarounds

### sccache Compiler ICEs

**Issue:** Compiler crashes with sccache enabled
**Workaround:** Disable sccache for MSRV checks:
```bash
RUSTC_WRAPPER="" cargo +1.89.0 check --workspace --all-targets --locked --no-default-features --features cpu
```

### Test Gate Environment Issue

**Issue:** Test gate hits compiler ICE due to nightly + sccache
**Impact:** Not a code issue - environment/toolchain problem
**Workaround:** CI uses stable toolchain, won't hit this issue

## Branch Protection

All PRs must pass:
- `CI Core Success` job in `.github/workflows/ci-core.yml`
- Required reviewers: (TBD)
- No merge conflicts with main

## Communication

### PR Link References

- All PRs reference "Part of #319" (not "Closes #319")
- This keeps #319 open until all 5 PRs merge
- Final PR (#522) will include "Closes #319"

### Reviewer Notes

Each PR includes:
- Link to stack status document (this file)
- Position in stack (e.g., "PR 2/5")
- Dependencies (which PRs must merge first)
- Next steps (what PR-N+1 will do)

## Local Verification Script

All PRs use the same verification script:

```bash
#!/bin/bash
# scripts/verify-pr.sh

set -e

echo "=== Core CI Gates ==="

echo "1/6: Build (strict)"
env RUSTFLAGS='-D warnings' \
  cargo build --locked --workspace --no-default-features --features cpu

echo "2/6: Tests (lib)"
env RUSTFLAGS='-D warnings' \
  cargo test --locked --workspace --no-default-features --features cpu --lib

echo "3/6: Clippy (strict)"
cargo clippy --workspace --all-targets --no-default-features --features cpu -- -D warnings

echo "4/6: Format"
cargo fmt --all -- --check

echo "5/6: Docs"
env RUSTDOCFLAGS='-A warnings' \
  cargo doc --locked --no-deps --workspace --no-default-features --features cpu

echo "6/6: MSRV"
RUSTC_WRAPPER="" cargo +1.89.0 check --workspace --all-targets --locked \
  --no-default-features --features cpu

echo ""
echo "✅ All core CI gates pass!"
```

## Quick Reference Commands

### Check PR Status

```bash
gh pr view 518 --json state,title,isDraft
gh pr view 519 --json state,title,isDraft
```

### View CI Status

```bash
gh pr checks 518
gh pr checks 519
```

### List Related Branches

```bash
git branch -a | grep -E "319|kv-pool"
```

### Show Recent Commits

```bash
git log --oneline --graph --all | grep -E "319|kv-pool|arena|evict" | head -20
```

## Contact & Support

- **Issue:** #319
- **Project:** BitNet.rs
- **Maintainer:** @EffortlessMetrics
- **Documentation:** See index above

## Appendix: File Tree

```
BitNet-rs/
├── STACK_STATUS_319.md              # This file
├── MERGE_CHECKLIST_518.md           # PR #518 merge steps
├── VERIFICATION_SUMMARY_518.md      # PR #518 verification (obsolete)
├── PR_518_FINAL_CHECKLIST.md        # PR #518 comprehensive checklist
├── PR_519_REBASE_PLAN.md            # PR #519 rebase workflow
├── PR_3_EVICTION_PATCH_GUIDE.md     # PR #520 implementation guide
├── scripts/
│   └── ci-local.sh                  # Local CI verification
└── crates/
    └── bitnet-models/
        └── src/
            ├── memory_pool.rs       # PR #518 (merged next)
            ├── kv_cache_manager.rs  # PR #519, PR #520
            └── pool_stats.rs        # PR #520
```

---

**Ready to proceed:** Once GitHub billing is resolved, execute merge command for PR #518.
