#![no_main]

use arbitrary::Arbitrary;
use bitnet_kernels::cuda::memory_pool::{MemoryPool, MemoryPoolConfig};
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
struct MemoryPoolInput {
    /// Encoded allocator choice: 0=BestFit, 1=Buddy, 2=Slab.
    allocator_byte: u8,
    /// Fuzzed initial pool size (mapped to realistic range).
    initial_size_byte: u8,
    /// Sequence of operations: (op_type, size_byte).
    ops: Vec<(u8, u8)>,
}

fuzz_target!(|input: MemoryPoolInput| {
    // Realistic pool sizes: 1 KiB – 4 MiB.
    let initial_size = ((input.initial_size_byte as usize) % 4096 + 1) * 1024;
    let config = MemoryPoolConfig {
        initial_size,
        max_size: initial_size * 4,
        block_size: 256,
        alignment: 256,
    };

    let mut pool = match input.allocator_byte % 3 {
        0 => MemoryPool::with_best_fit(config),
        1 => MemoryPool::with_buddy(config),
        _ => MemoryPool::with_slab(config, 4096),
    };
    let mut pool = match pool {
        Ok(p) => p,
        Err(_) => return,
    };

    let mut live_ids = Vec::new();

    for &(op, size_byte) in input.ops.iter().take(128) {
        match op % 5 {
            // Allocate
            0 | 1 => {
                let size = (size_byte as usize % 4096) + 1;
                if let Ok(block) = pool.allocate(size) {
                    live_ids.push(block.id);
                }
            }
            // Deallocate
            2 => {
                if let Some(id) = live_ids.pop() {
                    let _ = pool.deallocate(id);
                }
            }
            // Touch (LRU update)
            3 => {
                if let Some(&id) = live_ids.last() {
                    let _ = pool.touch(id);
                }
            }
            // Evict LRU
            _ => {
                if let Ok((evicted_id, _size)) = pool.evict_least_recently_used() {
                    live_ids.retain(|id| *id != evicted_id);
                }
            }
        }
    }

    // Invariant: pressure is in [0, 1].
    let pressure = pool.memory_pressure();
    assert!((0.0..=1.0).contains(&pressure), "memory_pressure out of range: {pressure}");

    // Invariant: peak_usage >= current usage.
    let stats = pool.memory_usage();
    assert!(
        pool.peak_usage() >= stats.used,
        "peak_usage {} < current used {}",
        pool.peak_usage(),
        stats.used,
    );

    // Defragment must not panic.
    pool.defragment();

    // Reset must not panic.
    pool.reset();
    let stats_after = pool.memory_usage();
    assert_eq!(stats_after.used, 0, "used should be 0 after reset");
});
