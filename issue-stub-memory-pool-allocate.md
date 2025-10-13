# [IMPLEMENTATION] Complete MemoryPool allocation system in KV cache

## Problem Description

The `MemoryPool::allocate` function in `crates/bitnet-inference/src/cache.rs` only returns block indices but doesn't provide access to actual memory blocks or properly manage memory allocation lifecycle.

## Environment
- **File**: `crates/bitnet-inference/src/cache.rs`
- **Component**: MemoryPool struct
- **Current State**: Incomplete allocation system

## Root Cause Analysis

Current implementation:
```rust
fn allocate(&mut self) -> Option<usize> {
    self.free_blocks.pop()  // Only returns index, no actual memory access
}
```

**Issues:**
1. Returns index but no access to memory block
2. Blocks vector never populated with actual memory
3. No memory initialization or lifecycle management
4. Deallocate function doesn't validate block state

## Proposed Solution

```rust
impl MemoryPool {
    fn new(block_size: usize, max_size: usize) -> Result<Self> {
        let num_blocks = max_size / (block_size * std::mem::size_of::<f32>());
        let mut blocks = Vec::with_capacity(num_blocks);

        // Pre-allocate and initialize memory blocks
        for _ in 0..num_blocks {
            blocks.push(vec![0.0; block_size]);
        }

        let free_blocks = (0..num_blocks).collect();
        Ok(Self { block_size, blocks, free_blocks })
    }

    fn allocate(&mut self) -> Option<MemoryBlock> {
        if let Some(block_id) = self.free_blocks.pop() {
            Some(MemoryBlock {
                id: block_id,
                data: &mut self.blocks[block_id],
            })
        } else {
            None
        }
    }

    fn deallocate(&mut self, block_id: usize) -> Result<()> {
        if block_id >= self.blocks.len() {
            return Err(Error::InvalidBlockId(block_id));
        }

        if self.free_blocks.contains(&block_id) {
            return Err(Error::DoubleDeallocation(block_id));
        }

        // Clear block data for security/consistency
        self.blocks[block_id].fill(0.0);
        self.free_blocks.push(block_id);
        Ok(())
    }

    fn get_block_mut(&mut self, block_id: usize) -> Option<&mut Vec<f32>> {
        if !self.free_blocks.contains(&block_id) {
            self.blocks.get_mut(block_id)
        } else {
            None
        }
    }
}

#[derive(Debug)]
pub struct MemoryBlock<'a> {
    pub id: usize,
    pub data: &'a mut Vec<f32>,
}
```

## Implementation Plan

### Phase 1: Core Memory Management (2 days)
- [ ] Implement proper memory block allocation and initialization
- [ ] Add block lifecycle management with validation
- [ ] Create memory block access patterns
- [ ] Add proper error handling for allocation failures

### Phase 2: Integration (1 day)
- [ ] Update KVCache to use new allocation system
- [ ] Add memory usage tracking and statistics
- [ ] Implement memory pressure handling
- [ ] Add comprehensive testing

## Acceptance Criteria
- [ ] Memory blocks properly allocated and accessible
- [ ] Allocation/deallocation lifecycle correctly managed
- [ ] Error handling for invalid operations
- [ ] Memory usage properly tracked
- [ ] All existing cache functionality preserved

**Labels**: `implementation`, `memory-management`, `P2-medium`
**Effort**: 3 days
