#![no_main]

use arbitrary::Arbitrary;
use bitnet_inference::cache::{CacheConfig, EvictionPolicy, KVCache};
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug, Clone, Copy)]
enum FuzzOp {
    Store { layer: u8, position: u8, dim: u8 },
    Get { layer: u8, position: u8 },
    Contains { layer: u8, position: u8 },
    ClearLayer { layer: u8 },
    ClearAll,
    Stats,
}

#[derive(Arbitrary, Debug)]
struct KvCacheInput {
    /// Maximum cache size (clamped to small range for fuzzing).
    max_size_kb: u8,
    /// Maximum sequence length.
    max_seq_len: u8,
    /// Eviction policy selector.
    eviction_selector: u8,
    /// Block size.
    block_size_hint: u8,
    /// Sequence of operations to perform.
    ops: Vec<FuzzOp>,
    /// Raw data to fill key/value tensors.
    fill_data: Vec<f32>,
}

const MAX_OPS: usize = 128;

fuzz_target!(|input: KvCacheInput| {
    let max_size = ((input.max_size_kb as usize) + 1) * 1024;
    let max_seq = (input.max_seq_len as usize % 256) + 1;
    let block_size = (input.block_size_hint as usize % 128) + 1;

    let eviction = match input.eviction_selector % 3 {
        0 => EvictionPolicy::LRU,
        1 => EvictionPolicy::FIFO,
        _ => EvictionPolicy::LFU,
    };

    let config = CacheConfig {
        max_size_bytes: max_size,
        max_sequence_length: max_seq,
        enable_compression: false,
        eviction_policy: eviction,
        block_size,
    };

    let mut cache = match KVCache::new(config) {
        Ok(c) => c,
        Err(_) => return,
    };

    let fill: Vec<f32> = input
        .fill_data
        .iter()
        .take(512)
        .map(|&x| if x.is_nan() || x.is_infinite() { 0.0 } else { x })
        .collect();

    for op in input.ops.iter().take(MAX_OPS) {
        match *op {
            FuzzOp::Store { layer, position, dim } => {
                let d = (dim as usize % 32) + 1;
                let key: Vec<f32> = fill.iter().copied().cycle().take(d).collect();
                let value: Vec<f32> = fill.iter().copied().cycle().take(d).collect();
                let _ = cache.store(layer as usize, position as usize, key, value);
            }
            FuzzOp::Get { layer, position } => {
                let result = cache.get(layer as usize, position as usize);
                if let Some((k, v)) = result {
                    assert!(!k.is_empty(), "cached key must not be empty");
                    assert!(!v.is_empty(), "cached value must not be empty");
                }
            }
            FuzzOp::Contains { layer, position } => {
                let _ = cache.contains(layer as usize, position as usize);
            }
            FuzzOp::ClearLayer { layer } => {
                cache.clear_layer(layer as usize);
            }
            FuzzOp::ClearAll => {
                cache.clear();
            }
            FuzzOp::Stats => {
                let stats = cache.stats();
                assert!(stats.hit_rate >= 0.0 && stats.hit_rate <= 1.0);
                assert!(stats.memory_efficiency >= 0.0);
            }
        }
    }

    // Final invariant: size never exceeds configured maximum.
    assert!(cache.size() <= max_size, "cache size {} exceeds max {}", cache.size(), max_size);
});
