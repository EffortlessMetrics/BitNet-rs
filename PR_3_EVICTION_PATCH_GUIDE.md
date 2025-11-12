# PR-3: Eviction + Metrics Patch Guide

## Overview

**Sequence:** PR-3 (follows PR #518 and PR #519)
**Goal:** Update eviction logic to use `pool.deallocate()`, add stats tracking, gate unsafe helpers

## Prerequisites

- ✅ PR #518 merged (Arena foundation)
- ✅ PR #519 merged (KVPoolEntry + MemoryPool integration)

## Changes Required

### 1. Update Eviction Logic

**File:** `crates/bitnet-models/src/kv_cache_manager.rs`

**Current code (from PR #519):**
```rust
impl KVCacheManager {
    pub fn remove_cache_entry(&mut self, cache_key: &CacheKey) -> Option<KVPoolEntry> {
        if let Some(entry) = self.cache.remove(cache_key) {
            // TODO: Deallocate block back to pool
            // Currently just removes from cache without returning memory
            Some(entry)
        } else {
            None
        }
    }
}
```

**New code (PR-3):**
```rust
impl KVCacheManager {
    pub fn remove_cache_entry(&mut self, cache_key: &CacheKey) -> Option<KVPoolEntry> {
        if let Some(entry) = self.cache.remove(cache_key) {
            // Deallocate block back to pool (no fabricated offsets)
            self.pool.deallocate(entry.block);

            // Update stats atomically
            self.stats.evictions += 1;
            self.stats.active_entries = self.stats.active_entries.saturating_sub(1);

            // Consistency assertion
            debug_assert_eq!(
                self.stats.active_entries as usize,
                self.cache.len(),
                "Stats mismatch: active_entries={} != cache.len()={}",
                self.stats.active_entries,
                self.cache.len()
            );

            Some(entry)
        } else {
            None
        }
    }
}
```

**Key changes:**
1. Call `self.pool.deallocate(entry.block)` to return memory
2. Increment `evictions` counter
3. Decrement `active_entries` using `saturating_sub` (prevents underflow)
4. Add debug assertion for stats consistency

### 2. Add Unit Test for Eviction

**File:** `crates/bitnet-models/src/kv_cache_manager.rs` (tests module)

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_evict_returns_block_to_pool() {
        // Arrange: Create manager with small pool
        let mut manager = KVCacheManager::new(1024);
        let key1 = CacheKey::new("test", 1);
        let key2 = CacheKey::new("test2", 2);

        // Act: Allocate first entry
        let entry1 = manager.allocate_entry(&key1, 64).expect("Should allocate");
        let block_id = entry1.block;

        // Verify initial state
        assert_eq!(manager.stats.active_entries, 1);
        assert_eq!(manager.stats.evictions, 0);

        // Act: Evict first entry
        let evicted = manager.remove_cache_entry(&key1);
        assert!(evicted.is_some(), "Should return evicted entry");

        // Verify stats after eviction
        assert_eq!(manager.stats.active_entries, 0);
        assert_eq!(manager.stats.evictions, 1);

        // Act: Allocate second entry (should reuse same block)
        let entry2 = manager.allocate_entry(&key2, 64).expect("Should reallocate");

        // Assert: Same block ID proves memory was returned to pool
        assert_eq!(
            entry2.block, block_id,
            "Block should be reused after eviction"
        );

        // Verify final stats
        assert_eq!(manager.stats.active_entries, 1);
        assert_eq!(manager.stats.evictions, 1);
    }

    #[test]
    fn test_evict_nonexistent_key_returns_none() {
        let mut manager = KVCacheManager::new(1024);
        let key = CacheKey::new("nonexistent", 999);

        let result = manager.remove_cache_entry(&key);
        assert!(result.is_none(), "Should return None for nonexistent key");

        // Stats should be unchanged
        assert_eq!(manager.stats.evictions, 0);
        assert_eq!(manager.stats.active_entries, 0);
    }

    #[test]
    fn test_multiple_evictions_stats_consistency() {
        let mut manager = KVCacheManager::new(2048);

        // Allocate multiple entries
        let keys: Vec<_> = (0..5)
            .map(|i| CacheKey::new("test", i))
            .collect();

        for key in &keys {
            manager.allocate_entry(key, 128).expect("Should allocate");
        }

        assert_eq!(manager.stats.active_entries, 5);

        // Evict all entries
        for (i, key) in keys.iter().enumerate() {
            manager.remove_cache_entry(key);
            assert_eq!(manager.stats.evictions, (i + 1) as u64);
            assert_eq!(manager.stats.active_entries, (4 - i) as u64);
        }

        // Final state: all evicted
        assert_eq!(manager.stats.evictions, 5);
        assert_eq!(manager.stats.active_entries, 0);
        assert_eq!(manager.cache.len(), 0);
    }
}
```

### 3. Verify Unsafe Helpers Are Gated

**File:** `crates/bitnet-models/src/memory_pool.rs`

Confirm these helpers are already gated in PR #518:

```rust
impl MemoryPool {
    /// # Safety
    /// - `offset` must be < `self.total_size`
    /// - Returned pointer is valid for reads only during `&self` lifetime
    #[cfg(test)]  // ✅ Already gated in PR #518
    pub unsafe fn as_ptr_at(&self, offset: usize) -> *const u8 {
        debug_assert!(offset < self.total_size, "Offset out of bounds");
        self.memory.as_ptr().add(offset)
    }

    /// # Safety
    /// - `offset` must be < `self.total_size`
    /// - Returned pointer is valid for writes only during `&mut self` lifetime
    #[cfg(test)]  // ✅ Already gated in PR #518
    pub unsafe fn as_mut_ptr_at(&mut self, offset: usize) -> *mut u8 {
        debug_assert!(offset < self.total_size, "Offset out of bounds");
        self.memory.as_mut_ptr().add(offset)
    }
}
```

**Action:** No changes needed if already gated in PR #518.

### 4. Add Stats Assertions Throughout

**File:** `crates/bitnet-models/src/kv_cache_manager.rs`

Add consistency checks in key methods:

```rust
impl KVCacheManager {
    pub fn allocate_entry(&mut self, key: &CacheKey, size: usize) -> Result<KVPoolEntry, String> {
        // ... existing allocation logic ...

        // Update stats
        self.stats.active_entries += 1;
        self.stats.allocations += 1;

        // Consistency check
        debug_assert_eq!(
            self.stats.active_entries as usize,
            self.cache.len(),
            "Stats divergence after allocation"
        );

        Ok(entry)
    }

    pub fn get_stats(&self) -> &PoolStats {
        // Verify consistency before returning stats
        debug_assert_eq!(
            self.stats.active_entries as usize,
            self.cache.len(),
            "Stats inconsistent when queried"
        );
        &self.stats
    }
}
```

### 5. Update PoolStats Structure (if needed)

**File:** `crates/bitnet-models/src/pool_stats.rs` (or within kv_cache_manager.rs)

Ensure stats structure has required fields:

```rust
#[derive(Debug, Clone, Default)]
pub struct PoolStats {
    pub total_capacity: usize,
    pub used_memory: usize,
    pub active_entries: u64,
    pub allocations: u64,
    pub evictions: u64,      // ← Ensure this exists
    pub fragmentation: f64,
}
```

## Testing Strategy

### Unit Tests (Already Covered Above)

1. `test_evict_returns_block_to_pool` - Verify memory reuse
2. `test_evict_nonexistent_key_returns_none` - Edge case handling
3. `test_multiple_evictions_stats_consistency` - Stats accuracy

### Integration Tests (Optional for PR-3)

**File:** `crates/bitnet-models/tests/kv_cache_integration.rs`

```rust
#[test]
fn test_allocate_evict_reallocate_cycle() {
    let mut manager = KVCacheManager::new(512);

    // Phase 1: Fill pool
    let keys: Vec<_> = (0..8).map(|i| CacheKey::new("fill", i)).collect();
    for key in &keys {
        manager.allocate_entry(key, 64).expect("Should allocate");
    }

    // Phase 2: Evict half
    for key in &keys[0..4] {
        manager.remove_cache_entry(key);
    }

    // Phase 3: Reallocate in freed space
    let new_keys: Vec<_> = (0..4).map(|i| CacheKey::new("new", i)).collect();
    for key in &new_keys {
        manager.allocate_entry(key, 64).expect("Should reallocate");
    }

    // Verify final stats
    assert_eq!(manager.stats.allocations, 12); // 8 + 4
    assert_eq!(manager.stats.evictions, 4);
    assert_eq!(manager.stats.active_entries, 8); // 4 old + 4 new
}
```

## Validation Checklist

Before submitting PR-3:

- [ ] `remove_cache_entry` calls `pool.deallocate(entry.block)`
- [ ] Stats are updated correctly (evictions, active_entries)
- [ ] Debug assertions verify stats consistency
- [ ] Unit tests cover eviction + reallocation
- [ ] Unit tests cover edge cases (nonexistent key, multiple evictions)
- [ ] Unsafe helpers are gated under `#[cfg(test)]` (verify from PR #518)
- [ ] All core CI gates pass locally

## Local Verification Commands

```bash
# Format
cargo fmt --all

# Build (strict)
env RUSTFLAGS='-D warnings' \
  cargo build --locked --workspace --no-default-features --features cpu

# Test (lib only)
env RUSTFLAGS='-D warnings' \
  cargo test --locked --workspace --no-default-features --features cpu --lib

# Clippy (strict)
cargo clippy --workspace --all-targets --no-default-features --features cpu \
  -- -D warnings

# Docs
env RUSTDOCFLAGS='-A warnings' \
  cargo doc --locked --no-deps --workspace --no-default-features --features cpu

# MSRV
RUSTC_WRAPPER="" cargo +1.89.0 check --workspace --all-targets --locked \
  --no-default-features --features cpu
```

## PR-3 Metadata

**Title:** `kv-pool v2: Eviction + metrics (PR 3/5)`

**Description:**
```markdown
Part of #319. Updates eviction logic to use `pool.deallocate()` and tracks stats correctly.

## Changes
- ✅ `remove_cache_entry` now calls `pool.deallocate(entry.block)`
- ✅ Stats updated on eviction (evictions++, active_entries--)
- ✅ Debug assertions verify stats consistency
- ✅ Comprehensive unit tests (eviction + reallocation)
- ✅ Unsafe helpers gated under `#[cfg(test)]`

## Testing
- Unit tests: `test_evict_returns_block_to_pool`, `test_evict_nonexistent_key_returns_none`, `test_multiple_evictions_stats_consistency`
- All core CI gates pass locally

## Next in Stack
- PR-4: Add receipt generation for honest compute
- PR-5: Final integration + cleanup
```

**Labels:**
- `enhancement`
- `kv-cache`
- `part-of-#319`

## Diff Scope Estimate

**Files changed:** 2-3
- `crates/bitnet-models/src/kv_cache_manager.rs` (main changes)
- `crates/bitnet-models/src/pool_stats.rs` (if separate file)
- `crates/bitnet-models/tests/kv_cache_integration.rs` (optional integration tests)

**Lines changed:** ~100-150
- Eviction logic: ~15 lines
- Unit tests: ~80-100 lines
- Stats assertions: ~10 lines
- Integration tests (optional): ~30-40 lines

## Dependencies

**Depends on:**
- PR #518 (merged) - Arena foundation
- PR #519 (merged) - KVPoolEntry + MemoryPool integration

**Blocks:**
- PR-4 (Receipt generation)
- PR-5 (Final integration)

## Review Focus Areas

1. **Correctness:** Does `deallocate` properly return memory to the pool?
2. **Stats accuracy:** Are evictions/active_entries updated atomically and consistently?
3. **Test coverage:** Do tests prove memory is actually reused after eviction?
4. **Edge cases:** How does eviction handle nonexistent keys, empty pools, etc.?
5. **Safety:** Are all unsafe operations properly gated and documented?

## Known Limitations (Post-PR-3)

After PR-3 merges, the following will still be TODO:

1. **Receipt generation** - PR-4 will add honest compute receipts
2. **LRU policy** - Current free-list is FIFO; LRU eviction is out of scope
3. **Fragmentation metrics** - Basic fragmentation tracking exists, but advanced defrag is future work
4. **Concurrency** - Single-threaded only; concurrent access requires locks (future)

These are intentional scope limits for the MVP.

## Reference

- Issue: #319 (KV Cache Pool v2)
- Previous PRs: #518 (Arena), #519 (Entry struct)
- Related docs: `docs/architecture/kv-cache-pool.md`
