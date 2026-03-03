#![no_main]

use arbitrary::Arbitrary;
use bitnet_inference::cache::{CacheConfig, EvictionPolicy, KVCache};
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
struct KvCacheInput {
    max_seq_len: u8,
    block_size: u8,
    eviction_mode: u8,
    ops: Vec<KvOp>,
}

#[derive(Arbitrary, Debug)]
enum KvOp {
    Store { layer: u8, position: u8, key_val: u8 },
    Get { layer: u8, position: u8 },
    Contains { layer: u8, position: u8 },
    ClearLayer { layer: u8 },
    ClearAll,
    CheckStats,
}

fuzz_target!(|input: KvCacheInput| {
    let max_seq_len = (input.max_seq_len as usize % 64) + 1;
    let block_size = (input.block_size as usize % 32) + 1;

    let eviction_policy = match input.eviction_mode % 3 {
        0 => EvictionPolicy::LRU,
        1 => EvictionPolicy::FIFO,
        _ => EvictionPolicy::LFU,
    };

    let config = CacheConfig {
        max_size_bytes: 1024 * 1024, // 1 MiB budget
        max_sequence_length: max_seq_len,
        enable_compression: false,
        eviction_policy,
        block_size,
    };

    let cache = KVCache::new(config);
    let mut cache = match cache {
        Ok(c) => c,
        Err(_) => return,
    };

    // Key/value vector dimension (small for speed).
    let dim = 8;

    for op in input.ops.iter().take(256) {
        match op {
            KvOp::Store { layer, position, key_val } => {
                let layer = *layer as usize % 4;
                let position = *position as usize % max_seq_len;
                let val = *key_val as f32;
                let key = vec![val; dim];
                let value = vec![val + 1.0; dim];
                let _ = cache.store(layer, position, key, value);
            }
            KvOp::Get { layer, position } => {
                let layer = *layer as usize % 4;
                let position = *position as usize % max_seq_len;
                let _ = cache.get(layer, position);
            }
            KvOp::Contains { layer, position } => {
                let layer = *layer as usize % 4;
                let position = *position as usize % max_seq_len;
                let _ = cache.contains(layer, position);
            }
            KvOp::ClearLayer { layer } => {
                let layer = *layer as usize % 4;
                cache.clear_layer(layer);
            }
            KvOp::ClearAll => {
                cache.clear();
                assert_eq!(cache.size(), 0, "size should be 0 after clear");
            }
            KvOp::CheckStats => {
                let stats = cache.stats();
                // hits + misses >= 0 (just ensure no panic).
                let _ = format!("{stats:?}");
            }
        }
    }

    // After all ops, get/contains must not panic for any layer/position.
    for layer in 0..4 {
        for pos in 0..max_seq_len.min(8) {
            let _ = cache.get(layer, pos);
            let _ = cache.contains(layer, pos);
        }
    }

    // clear must reset size to 0.
    cache.clear();
    assert_eq!(cache.size(), 0, "size should be 0 after final clear");
});
