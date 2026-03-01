#![no_main]

use arbitrary::Arbitrary;
use bitnet_inference::prefix_cache::{EvictionPolicy, PrefixCache, PrefixCacheConfig};
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
struct PrefixCacheInput {
    max_entries: u8,
    max_memory: u8,
    min_prefix_len: u8,
    eviction_policy: u8,
    ops: Vec<CacheOp>,
}

#[derive(Arbitrary, Debug)]
enum CacheOp {
    Insert { tokens: Vec<u16>, state_len: u8 },
    Lookup { tokens: Vec<u16> },
    Invalidate { tokens: Vec<u16> },
    Evict,
    Clear,
    Stats,
}

fuzz_target!(|input: PrefixCacheInput| {
    let max_entries = (input.max_entries as usize % 16) + 1;
    let max_memory = (input.max_memory as usize % 64 + 1) * 64; // 64..4160 bytes
    let min_prefix_len = (input.min_prefix_len as usize % 8) + 1;
    let policy = match input.eviction_policy % 4 {
        0 => EvictionPolicy::LRU,
        1 => EvictionPolicy::LFU,
        2 => EvictionPolicy::FIFO,
        _ => EvictionPolicy::TTL,
    };

    let cfg = PrefixCacheConfig {
        max_entries,
        max_memory_bytes: max_memory,
        eviction_policy: policy,
        min_prefix_length: min_prefix_len,
        ttl_seconds: 3600,
    };

    let mut cache = PrefixCache::new(cfg);

    // Invariant 1: Fresh cache is empty.
    assert!(cache.is_empty());
    assert_eq!(cache.len(), 0);

    for op in input.ops.into_iter().take(256) {
        match op {
            CacheOp::Insert { tokens, state_len } => {
                let toks: Vec<u32> = tokens.iter().take(64).map(|&t| t as u32).collect();
                let state = vec![0xABu8; state_len as usize];
                let prev_len = cache.len();

                match cache.insert(&toks, state) {
                    Ok(()) => {
                        // Invariant 2: After successful insert, cache is non-empty.
                        assert!(!cache.is_empty());
                        // Invariant 3: Entry count does not exceed max_entries.
                        assert!(cache.len() <= max_entries);
                    }
                    Err(_) => {
                        // Short prefix or capacity exhaustion — no panic is the invariant.
                        let _ = prev_len;
                    }
                }
            }
            CacheOp::Lookup { tokens } => {
                let toks: Vec<u32> = tokens.iter().take(64).map(|&t| t as u32).collect();
                if let Some((matched_len, entry)) = cache.lookup(&toks) {
                    // Invariant 4: Matched length <= query length.
                    assert!(
                        matched_len <= toks.len(),
                        "matched {matched_len} > query len {}",
                        toks.len()
                    );
                    // Invariant 5: Entry prefix length >= matched length.
                    assert!(
                        entry.token_prefix.len() >= matched_len,
                        "prefix len {} < matched {matched_len}",
                        entry.token_prefix.len()
                    );
                    // Invariant 6: Entry prefix is a prefix of the query.
                    let prefix = &entry.token_prefix[..matched_len];
                    assert_eq!(prefix, &toks[..matched_len], "matched prefix doesn't match query");
                }
            }
            CacheOp::Invalidate { tokens } => {
                let toks: Vec<u32> = tokens.iter().take(64).map(|&t| t as u32).collect();
                let prev_len = cache.len();
                cache.invalidate(&toks);
                // Invariant 7: Invalidation never increases cache size.
                assert!(cache.len() <= prev_len);
            }
            CacheOp::Evict => {
                let prev_len = cache.len();
                let evicted = cache.evict();
                if evicted {
                    // Invariant 8: Successful eviction decreases size by 1.
                    assert_eq!(cache.len(), prev_len - 1);
                } else {
                    // Invariant 9: Failed eviction means cache was empty.
                    assert!(cache.is_empty());
                }
            }
            CacheOp::Clear => {
                cache.clear();
                // Invariant 10: After clear, cache is empty.
                assert!(cache.is_empty());
                assert_eq!(cache.len(), 0);
                assert_eq!(cache.stats().memory_usage, 0);
            }
            CacheOp::Stats => {
                let stats = cache.stats();
                // Invariant 11: Hit rate and miss rate are bounded.
                assert!(stats.hit_rate >= 0.0 && stats.hit_rate <= 1.0);
                assert!(stats.miss_rate >= 0.0 && stats.miss_rate <= 1.0);
                // Invariant 12: Memory usage does not exceed limit.
                assert!(
                    stats.memory_usage <= max_memory,
                    "memory {} > max {}",
                    stats.memory_usage,
                    max_memory
                );
            }
        }
    }

    // Final invariant: cache len never exceeds max_entries.
    assert!(cache.len() <= max_entries);
});
