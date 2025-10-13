# [Inference] Implement MemoryPool Block Allocation

## Problem Description

The `MemoryPool::allocate` method in `/home/steven/code/Rust/BitNet-rs/crates/bitnet-inference/src/cache.rs` only returns block indices instead of actual memory block references, limiting its usefulness for real memory management scenarios.

## Current Implementation
```rust
fn allocate(&mut self) -> Option<usize> {
    self.free_blocks.pop()
}
```

## Proposed Solution
Return actual memory block references with proper lifecycle management:

```rust
fn allocate(&mut self) -> Option<MemoryBlock> {
    if let Some(block_id) = self.free_blocks.pop() {
        self.blocks.get_mut(block_id).map(|data| MemoryBlock {
            id: block_id,
            data,
            size: self.block_size,
        })
    } else {
        None
    }
}

pub struct MemoryBlock {
    pub id: usize,
    pub data: &mut [f32],
    pub size: usize,
}
```

## Acceptance Criteria
- [ ] Return usable memory block references
- [ ] Proper memory lifecycle management
- [ ] Thread-safe allocation/deallocation
- [ ] Memory usage tracking and limits
- [ ] Integration with KV cache and tensor operations
