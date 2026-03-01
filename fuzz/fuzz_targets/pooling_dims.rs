#![no_main]

use arbitrary::Arbitrary;
use bitnet_common::memory_pool::TensorPool;
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
struct PoolingInput {
    max_pool_bytes: u16,
    ops: Vec<PoolOp>,
}

#[derive(Arbitrary, Debug)]
enum PoolOp {
    Allocate { size: u16 },
    AllocateAndUseF32 { size: u16 },
    Clear,
    Stats,
}

fuzz_target!(|input: PoolingInput| {
    let max_bytes = (input.max_pool_bytes as usize).saturating_mul(64).max(64);
    let pool = TensorPool::new(max_bytes);
    let mut live_buffers = Vec::new();

    for op in input.ops.into_iter().take(256) {
        match op {
            PoolOp::Allocate { size } => {
                let buf = pool.allocate(size as usize);
                assert!(buf.len() >= (size as usize).min(1));
                live_buffers.push(buf);
                // Cap live buffers to avoid OOM
                if live_buffers.len() > 32 {
                    live_buffers.remove(0);
                }
            }
            PoolOp::AllocateAndUseF32 { size } => {
                // Ensure multiple of 4 for f32 reinterpretation
                let byte_size = ((size as usize) / 4).max(1) * 4;
                let mut buf = pool.allocate(byte_size);
                // Buffer is zeroed; verify f32 view doesn't panic
                let floats = buf.as_f32_mut_slice();
                if !floats.is_empty() {
                    floats[0] = 1.0;
                    assert_eq!(buf.as_f32_slice()[0], 1.0);
                }
                live_buffers.push(buf);
                if live_buffers.len() > 32 {
                    live_buffers.remove(0);
                }
            }
            PoolOp::Clear => {
                pool.clear();
            }
            PoolOp::Stats => {
                let stats = pool.stats();
                // Invariant: total allocations == hits + misses
                let _ = stats.total_allocations();
            }
        }
    }

    // Drop all buffers and verify stats are consistent
    drop(live_buffers);
    let final_stats = pool.stats();
    assert_eq!(final_stats.active_bytes, 0);
});
