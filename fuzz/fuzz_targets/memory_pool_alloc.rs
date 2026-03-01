#![no_main]
use arbitrary::Arbitrary;
use bitnet_common::memory_pool::TensorPool;
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug, Clone, Copy)]
enum PoolOp {
    Allocate { size: u16 },
    DropOldest,
    DropNewest,
    Clear,
    Stats,
}

#[derive(Arbitrary, Debug)]
struct PoolInput {
    max_pool_bytes: u16,
    ops: Vec<PoolOp>,
}

fuzz_target!(|input: PoolInput| {
    let max_bytes = (input.max_pool_bytes as usize).saturating_mul(64);
    let pool = TensorPool::new(max_bytes);
    let mut live_buffers = Vec::new();

    for op in input.ops.into_iter().take(256) {
        match op {
            PoolOp::Allocate { size } => {
                let capped = size as usize % 4096;
                let buf = pool.allocate(capped);
                assert!(buf.len() >= capped.max(64));
                live_buffers.push(buf);
            }
            PoolOp::DropOldest => {
                if !live_buffers.is_empty() {
                    live_buffers.remove(0);
                }
            }
            PoolOp::DropNewest => {
                live_buffers.pop();
            }
            PoolOp::Clear => {
                pool.clear();
            }
            PoolOp::Stats => {
                let stats = pool.stats();
                assert!(stats.total_allocations() >= stats.hits);
            }
        }
    }

    let final_stats = pool.stats();
    drop(live_buffers);
    let after_drop = pool.stats();
    assert_eq!(after_drop.active_bytes, 0);
    assert!(after_drop.pooled_bytes <= max_bytes);
    assert!(final_stats.total_allocations() <= 256);
});
