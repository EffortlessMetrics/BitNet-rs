#![no_main]

use arbitrary::Arbitrary;
use bitnet_inference::cache::{CacheConfig, EvictionPolicy, KVCache};
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
struct CacheInput {
    max_size_bytes: u32,
    max_seq_len: u16,
    enable_compression: bool,
    eviction_idx: u8,
    block_size: u16,
    ops: Vec<CacheOp>,
}

#[derive(Arbitrary, Debug)]
enum CacheOp {
    Store { layer: u8, position: u16, kv_len: u8 },
    Get { layer: u8, position: u16 },
    Contains { layer: u8, position: u16 },
    ClearLayer { layer: u8 },
    ClearAll,
    Stats,
}

fuzz_target!(|input: CacheInput| {
    let policies = [EvictionPolicy::LRU, EvictionPolicy::FIFO, EvictionPolicy::LFU];
    let policy = policies[input.eviction_idx as usize % policies.len()];

    // Bound sizes to prevent OOM.
    let max_size = ((input.max_size_bytes as usize) % (4 * 1024 * 1024)).max(1024);
    let block_size = ((input.block_size as usize) % 256).max(1);

    let config = CacheConfig {
        max_size_bytes: max_size,
        max_sequence_length: (input.max_seq_len as usize % 1024).max(1),
        enable_compression: input.enable_compression,
        eviction_policy: policy,
        block_size,
    };

    let mut cache = match KVCache::new(config) {
        Ok(c) => c,
        Err(_) => return,
    };

    // Execute operations capped at 256 to prevent timeout.
    for op in input.ops.iter().take(256) {
        match op {
            CacheOp::Store { layer, position, kv_len } => {
                let l = *layer as usize % 32;
                let p = *position as usize;
                let len = (*kv_len as usize % 64).max(1);
                let key = vec![0.5f32; len];
                let value = vec![-0.5f32; len];
                let _ = cache.store(l, p, key, value);
            }
            CacheOp::Get { layer, position } => {
                let l = *layer as usize % 32;
                let p = *position as usize;
                let _ = cache.get(l, p);
            }
            CacheOp::Contains { layer, position } => {
                let l = *layer as usize % 32;
                let p = *position as usize;
                let _ = cache.contains(l, p);
            }
            CacheOp::ClearLayer { layer } => {
                let l = *layer as usize % 32;
                cache.clear_layer(l);
            }
            CacheOp::ClearAll => {
                cache.clear();
            }
            CacheOp::Stats => {
                let stats = cache.stats();
                let _ = format!("{stats:?}");
                let _ = cache.size();
                let _ = cache.usage_percent();
            }
        }
    }

    // After all ops, stats must not panic.
    let _ = cache.stats();
    let _ = cache.size();
    let _ = cache.usage_percent();

    // Verify invariants.
    assert!(cache.usage_percent() <= 100.1, "Usage cannot exceed 100%");
});
