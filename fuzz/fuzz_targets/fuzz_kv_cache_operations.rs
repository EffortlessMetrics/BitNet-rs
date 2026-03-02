#![no_main]

use arbitrary::Arbitrary;
use bitnet_inference::cache::{CacheConfig, EvictionPolicy, KVCache};
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
struct KvCacheOpsInput {
    eviction_policy: u8,
    max_size_kb: u8,
    max_seq_len: u8,
    ops: Vec<CacheOp>,
}

#[derive(Arbitrary, Debug)]
enum CacheOp {
    Store { layer: u8, position: u8, kv_len: u8 },
    Get { layer: u8, position: u8 },
    Contains { layer: u8, position: u8 },
    Clear,
    ClearLayer { layer: u8 },
    CheckStats,
    CheckSize,
}

fuzz_target!(|input: KvCacheOpsInput| {
    let eviction_policy = match input.eviction_policy % 3 {
        0 => EvictionPolicy::LRU,
        1 => EvictionPolicy::FIFO,
        _ => EvictionPolicy::LFU,
    };
    let max_size = ((input.max_size_kb as usize) + 1) * 1024;
    let max_seq = (input.max_seq_len as usize % 64) + 1;

    let config = CacheConfig {
        max_size_bytes: max_size,
        max_sequence_length: max_seq,
        enable_compression: false,
        eviction_policy,
        block_size: 64,
    };

    let mut cache = match KVCache::new(config) {
        Ok(c) => c,
        Err(_) => return,
    };

    // Invariant 1: Fresh cache has size 0.
    assert_eq!(cache.size(), 0, "fresh cache should have size 0");

    for op in input.ops.into_iter().take(256) {
        match op {
            CacheOp::Store { layer, position, kv_len } => {
                let layer = layer as usize % 8;
                let position = position as usize % max_seq;
                let len = (kv_len as usize % 32) + 1;
                let key = vec![1.0f32; len];
                let value = vec![2.0f32; len];
                // store must not panic; errors are acceptable.
                let _ = cache.store(layer, position, key, value);
            }
            CacheOp::Get { layer, position } => {
                let layer = layer as usize % 8;
                let position = position as usize % max_seq;
                // get must not panic.
                let _ = cache.get(layer, position);
            }
            CacheOp::Contains { layer, position } => {
                let layer = layer as usize % 8;
                let position = position as usize % max_seq;
                let _ = cache.contains(layer, position);
            }
            CacheOp::Clear => {
                cache.clear();
                // Invariant 2: After clear, size is 0.
                assert_eq!(cache.size(), 0, "cache size should be 0 after clear");
            }
            CacheOp::ClearLayer { layer } => {
                let layer = layer as usize % 8;
                cache.clear_layer(layer);
            }
            CacheOp::CheckStats => {
                let stats = cache.stats();
                // Invariant 3: Stats fields must be sensible.
                assert!(stats.hit_rate.is_finite(), "hit_rate must be finite");
                assert!(stats.memory_efficiency.is_finite(), "memory_efficiency must be finite");
            }
            CacheOp::CheckSize => {
                let _ = cache.size();
                let pct = cache.usage_percent();
                assert!(pct.is_finite(), "usage_percent must be finite");
            }
        }
    }

    // Invariant 4: Store then get returns the data.
    cache.clear();
    let k = vec![3.14f32; 4];
    let v = vec![2.72f32; 4];
    if cache.store(0, 0, k.clone(), v.clone()).is_ok() {
        if let Some((got_k, got_v)) = cache.get(0, 0) {
            assert_eq!(got_k, &k, "retrieved key must match stored key");
            assert_eq!(got_v, &v, "retrieved value must match stored value");
        }
    }

    // Invariant 5: contains returns true for stored entry.
    if cache.contains(0, 0) {
        assert!(cache.get(0, 0).is_some(), "contains=true but get=None");
    }
});
