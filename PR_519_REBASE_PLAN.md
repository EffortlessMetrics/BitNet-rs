# PR #519 Rebase Plan

## Context
PR #519 is the second PR in the 5-PR stack for issue #319 (KV Cache Pool v2).
It introduces the `KVPoolEntry` struct and updates `KVCacheManager` to use the real `MemoryPool`.

## Prerequisites
- ✅ PR #518 must be merged to `main` first
- ✅ GitHub Actions billing resolved (CI must be operational)

## Rebase Workflow

### 1. Verify PR #518 is merged

```bash
# Check that #518 has been merged
gh pr view 518 --json state,mergedAt
```

Expected output:
```json
{
  "state": "MERGED",
  "mergedAt": "2025-XX-XX..."
}
```

### 2. Switch to PR #519 branch

```bash
git switch feat/319-kv-pool-v2-pr2-entry
git status
```

### 3. Fetch and rebase onto updated main

```bash
# Fetch latest from origin
git fetch origin

# Rebase onto main (which now includes #518)
git rebase origin/main
```

**Expected outcome:**
- Clean rebase (no conflicts expected)
- PR #519 commits now sit on top of merged #518

**If conflicts occur:**
- Resolve manually (unlikely given the stacked nature)
- Run `git rebase --continue` after resolving
- Verify all tests still pass

### 4. Format and verify

```bash
# Format any touched files
cargo fmt --all

# Core CI gates (CPU-only scope)
env RUSTFLAGS='-D warnings' \
  cargo build --locked --workspace --no-default-features --features cpu

env RUSTFLAGS='-D warnings' \
  cargo test --locked --workspace --no-default-features --features cpu --lib

cargo clippy --workspace --all-targets --no-default-features --features cpu \
  -- -D warnings

cargo fmt --all -- --check

env RUSTDOCFLAGS='-A warnings' \
  cargo doc --locked --no-deps --workspace --no-default-features --features cpu

# MSRV check (disable sccache if needed)
RUSTC_WRAPPER="" cargo +1.89.0 check --workspace --all-targets --locked \
  --no-default-features --features cpu
```

### 5. Force-push rebased branch

```bash
# Force push to update PR #519
git push -f origin feat/319-kv-pool-v2-pr2-entry
```

### 6. Mark PR #519 as ready

```bash
# Move PR from draft to ready for review
gh pr ready 519
```

### 7. Update PR description with reviewer note

Add this note to the PR #519 description:

```markdown
## Reviewer Note

✅ **Rebased on `main` post-#518**

- All CPU-only CI gates verified locally:
  - ✅ Build (strict warnings): `RUSTFLAGS="-D warnings"`
  - ✅ Tests (lib targets): core crates passing
  - ✅ Clippy (strict): `-D warnings`
  - ✅ Format: `cargo fmt --check`
  - ✅ Docs: builds successfully with `RUSTDOCFLAGS="-A warnings"`
  - ✅ MSRV: compatible with Rust 1.89.0

**Next in stack:** PR-3 will switch eviction to `pool.deallocate(entry.block)` + add stats/receipts.
```

## Verification Checklist

Before marking as ready:

- [ ] PR #518 merged successfully
- [ ] Clean rebase onto `origin/main` (no conflicts)
- [ ] All 6 core CI gates pass locally
- [ ] Force-pushed updated branch
- [ ] Marked PR as ready for review
- [ ] Added reviewer note to PR description

## Next Steps After #519

Once PR #519 is approved and merged, prepare PR-3 (Eviction + Metrics):

1. Switch to PR-3 branch: `git switch feat/319-kv-pool-v2-pr3-evict-metrics`
2. Repeat rebase workflow above
3. Verify eviction logic uses `pool.deallocate(entry.block)`
4. Verify stats are updated correctly
5. Verify all `unsafe` helpers are gated under `#[cfg(test)]`

## Troubleshooting

### Rebase conflicts

If conflicts occur during rebase:

```bash
# View conflicted files
git status

# Resolve conflicts in editor
# Then continue rebase
git add <resolved-files>
git rebase --continue
```

### CI failures after rebase

If CI fails after rebase:

1. Run local gates individually to isolate failure
2. Check for new dependencies or feature flag mismatches
3. Verify `Cargo.lock` is up-to-date
4. Re-run format: `cargo fmt --all`

### sccache corruption

If MSRV check hits compiler ICE:

```bash
# Disable sccache temporarily
RUSTC_WRAPPER="" cargo +1.89.0 check --workspace --all-targets --locked \
  --no-default-features --features cpu
```

## Reference

- Issue #319: KV Cache Pool v2
- PR #518: Arena foundation (merged)
- PR #519: Entry struct + MemoryPool integration
- PR #520: Eviction + Metrics (next)
