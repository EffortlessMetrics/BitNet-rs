#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;

/// Fuzz register allocation with random kernel configs, verifying that
/// allocation never exceeds capacity and spill counts are consistent.
#[derive(Arbitrary, Debug)]
struct RegisterInput {
    total_registers: u8,
    kernel_configs: Vec<KernelConfig>,
    ops: Vec<AllocOp>,
}

#[derive(Arbitrary, Debug)]
struct KernelConfig {
    live_ranges: Vec<LiveRange>,
    _priority: u8,
}

#[derive(Arbitrary, Debug)]
struct LiveRange {
    _start: u8,
    _end: u8,
    size: u8,
}

#[derive(Arbitrary, Debug)]
enum AllocOp {
    Allocate { _kernel_idx: u8, _range_idx: u8 },
    Free { reg_id: u8 },
    Spill { reg_id: u8 },
    Reload { reg_id: u8 },
    Compact,
    Snapshot,
}

struct RegisterAllocator {
    capacity: usize,
    allocated: Vec<(usize, usize)>, // (reg_id, size)
    spilled: Vec<(usize, usize)>,   // (reg_id, size)
    next_id: usize,
    total_allocs: usize,
    total_spills: usize,
    total_reloads: usize,
}

impl RegisterAllocator {
    fn new(capacity: usize) -> Self {
        Self {
            capacity,
            allocated: Vec::new(),
            spilled: Vec::new(),
            next_id: 0,
            total_allocs: 0,
            total_spills: 0,
            total_reloads: 0,
        }
    }

    fn used(&self) -> usize {
        self.allocated.iter().map(|(_, s)| *s).sum()
    }

    fn free_regs(&self) -> usize {
        self.capacity.saturating_sub(self.used())
    }

    fn allocate(&mut self, size: usize) -> Option<usize> {
        if size == 0 || size > self.capacity {
            return None;
        }
        if self.used() + size > self.capacity {
            return None;
        }
        let id = self.next_id;
        self.next_id += 1;
        self.allocated.push((id, size));
        self.total_allocs += 1;
        Some(id)
    }

    fn free(&mut self, reg_id: usize) -> bool {
        if let Some(pos) = self.allocated.iter().position(|(id, _)| *id == reg_id) {
            self.allocated.swap_remove(pos);
            true
        } else {
            false
        }
    }

    fn spill(&mut self, reg_id: usize) -> bool {
        if let Some(pos) = self.allocated.iter().position(|(id, _)| *id == reg_id) {
            let entry = self.allocated.swap_remove(pos);
            self.spilled.push(entry);
            self.total_spills += 1;
            true
        } else {
            false
        }
    }

    fn reload(&mut self, reg_id: usize) -> bool {
        if let Some(pos) = self.spilled.iter().position(|(id, _)| *id == reg_id) {
            let (id, size) = self.spilled.swap_remove(pos);
            if self.used() + size > self.capacity {
                // Cannot reload — put back in spill
                self.spilled.push((id, size));
                return false;
            }
            self.allocated.push((id, size));
            self.total_reloads += 1;
            true
        } else {
            false
        }
    }

    fn compact(&mut self) {
        // Sort by ID to simulate defragmentation
        self.allocated.sort_by_key(|(id, _)| *id);
    }
}

fuzz_target!(|input: RegisterInput| {
    let capacity = (input.total_registers as usize % 64) + 4;
    let mut alloc = RegisterAllocator::new(capacity);

    // Invariant 1: Fresh allocator has full capacity.
    assert_eq!(alloc.free_regs(), capacity);
    assert_eq!(alloc.used(), 0);

    let mut live_ids: Vec<usize> = Vec::new();

    // Build allocation sizes from kernel configs.
    let sizes: Vec<usize> = input
        .kernel_configs
        .iter()
        .flat_map(|kc| {
            kc.live_ranges.iter().map(|lr| {
                let size = (lr.size as usize % 8) + 1;
                size
            })
        })
        .collect();

    let mut size_idx = 0;

    for op in input.ops.iter().take(256) {
        match op {
            AllocOp::Allocate { .. } => {
                let size = if size_idx < sizes.len() {
                    let s = sizes[size_idx];
                    size_idx += 1;
                    s
                } else {
                    1
                };
                if let Some(id) = alloc.allocate(size) {
                    live_ids.push(id);
                    // Invariant 2: used() never exceeds capacity.
                    assert!(
                        alloc.used() <= capacity,
                        "used={} > capacity={}",
                        alloc.used(),
                        capacity
                    );
                }
            }
            AllocOp::Free { reg_id } => {
                if !live_ids.is_empty() {
                    let idx = *reg_id as usize % live_ids.len();
                    let id = live_ids.swap_remove(idx);
                    let prev_used = alloc.used();
                    if alloc.free(id) {
                        // Invariant 3: Free decreases usage.
                        assert!(alloc.used() < prev_used);
                    }
                }
            }
            AllocOp::Spill { reg_id } => {
                if !live_ids.is_empty() {
                    let idx = *reg_id as usize % live_ids.len();
                    let id = live_ids[idx];
                    let prev_used = alloc.used();
                    if alloc.spill(id) {
                        live_ids.swap_remove(idx);
                        // Invariant 4: Spill frees register space.
                        assert!(alloc.used() < prev_used);
                    }
                }
            }
            AllocOp::Reload { reg_id } => {
                if !alloc.spilled.is_empty() {
                    let idx = *reg_id as usize % alloc.spilled.len();
                    let id = alloc.spilled[idx].0;
                    if alloc.reload(id) {
                        live_ids.push(id);
                    }
                }
            }
            AllocOp::Compact => {
                let used_before = alloc.used();
                alloc.compact();
                // Invariant 5: Compact doesn't change usage.
                assert_eq!(alloc.used(), used_before);
            }
            AllocOp::Snapshot => {
                // Invariant 6: used + free == capacity always.
                assert_eq!(
                    alloc.used() + alloc.free_regs(),
                    capacity,
                    "used={} free={} cap={}",
                    alloc.used(),
                    alloc.free_regs(),
                    capacity
                );
            }
        }
    }

    // Invariant 7: Spill count >= reload count (can't reload what wasn't spilled).
    assert!(
        alloc.total_spills >= alloc.total_reloads,
        "spills={} < reloads={}",
        alloc.total_spills,
        alloc.total_reloads
    );

    // Invariant 8: Total allocated tracked correctly.
    assert!(alloc.total_allocs >= live_ids.len());
});
