//! GPU memory defragmentation and pool management.
//!
//! Provides a [`MemoryPool`] with compaction support, fragmentation detection,
//! usage metrics, and background defragmentation triggered by a configurable
//! fragmentation threshold.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

// ---------------------------------------------------------------------------
// Block metadata
// ---------------------------------------------------------------------------

/// State of a memory block inside the pool.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlockState {
    Free,
    Allocated,
}

/// A contiguous region within the pool.
#[derive(Debug, Clone)]
pub struct MemoryBlock {
    /// Byte offset from the start of the pool.
    pub offset: usize,
    /// Size of this block in bytes.
    pub size: usize,
    /// Whether the block is free or allocated.
    pub state: BlockState,
    /// Optional label for debugging.
    pub label: Option<String>,
}

// ---------------------------------------------------------------------------
// Usage report
// ---------------------------------------------------------------------------

/// Snapshot of memory pool utilisation.
#[derive(Debug, Clone)]
pub struct MemoryUsageReport {
    /// Total pool capacity in bytes.
    pub total_bytes: usize,
    /// Bytes currently in use (allocated blocks).
    pub used_bytes: usize,
    /// Bytes that are free but non-contiguous (fragmented).
    pub fragmented_bytes: usize,
    /// Peak allocation observed since pool creation.
    pub peak_bytes: usize,
    /// Number of allocated blocks.
    pub allocated_blocks: usize,
    /// Number of free blocks.
    pub free_blocks: usize,
    /// Fragmentation ratio (0.0 = none, 1.0 = fully fragmented).
    pub fragmentation_ratio: f64,
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the memory pool.
#[derive(Debug, Clone)]
pub struct MemoryPoolConfig {
    /// Total pool capacity in bytes.
    pub capacity_bytes: usize,
    /// Fragmentation ratio above which defrag is triggered (0.0–1.0).
    pub defrag_threshold: f64,
    /// Minimum free block size to keep; smaller free blocks are
    /// eagerly merged with neighbours.
    pub min_free_block_bytes: usize,
}

impl Default for MemoryPoolConfig {
    fn default() -> Self {
        Self {
            capacity_bytes: 1024 * 1024 * 1024, // 1 GiB
            defrag_threshold: 0.3,
            min_free_block_bytes: 4096,
        }
    }
}

impl MemoryPoolConfig {
    pub fn validate(&self) -> Result<(), String> {
        if self.capacity_bytes == 0 {
            return Err("capacity_bytes must be > 0".into());
        }
        if !(0.0..=1.0).contains(&self.defrag_threshold) {
            return Err("defrag_threshold must be in [0.0, 1.0]".into());
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// MemoryPool
// ---------------------------------------------------------------------------

/// A simulated GPU memory pool with allocation, deallocation, compaction,
/// fragmentation detection, and usage reporting.
pub struct MemoryPool {
    config: MemoryPoolConfig,
    /// Ordered map from offset → block.
    blocks: Arc<Mutex<BTreeMap<usize, MemoryBlock>>>,
    peak_bytes: Arc<Mutex<usize>>,
    defrag_running: AtomicBool,
}

impl MemoryPool {
    /// Create a new pool with the given configuration.
    pub fn new(config: MemoryPoolConfig) -> Result<Self, String> {
        config.validate()?;

        let mut blocks = BTreeMap::new();
        blocks.insert(
            0,
            MemoryBlock {
                offset: 0,
                size: config.capacity_bytes,
                state: BlockState::Free,
                label: None,
            },
        );

        Ok(Self {
            config,
            blocks: Arc::new(Mutex::new(blocks)),
            peak_bytes: Arc::new(Mutex::new(0)),
            defrag_running: AtomicBool::new(false),
        })
    }

    // -- Allocation ----------------------------------------------------------

    /// Allocate `size` bytes using first-fit strategy. Returns the offset.
    pub fn allocate(&self, size: usize, label: Option<String>) -> Result<usize, String> {
        if size == 0 {
            return Err("cannot allocate 0 bytes".into());
        }
        let mut blocks = self.blocks.lock().unwrap();

        // First-fit: find first free block large enough.
        let free_offset = blocks
            .iter()
            .find(|(_, b)| b.state == BlockState::Free && b.size >= size)
            .map(|(&off, _)| off);

        let offset = free_offset.ok_or("out of memory")?;
        let free_block = blocks.remove(&offset).unwrap();

        // Insert allocated block.
        blocks.insert(
            offset,
            MemoryBlock {
                offset,
                size,
                state: BlockState::Allocated,
                label,
            },
        );

        // If there is leftover space, insert a free block after.
        if free_block.size > size {
            let remainder = free_block.size - size;
            blocks.insert(
                offset + size,
                MemoryBlock {
                    offset: offset + size,
                    size: remainder,
                    state: BlockState::Free,
                    label: None,
                },
            );
        }

        // Update peak.
        let used = Self::used_bytes_inner(&blocks);
        let mut peak = self.peak_bytes.lock().unwrap();
        if used > *peak {
            *peak = used;
        }

        Ok(offset)
    }

    /// Free the block at `offset`.
    pub fn deallocate(&self, offset: usize) -> Result<(), String> {
        let mut blocks = self.blocks.lock().unwrap();
        let block = blocks
            .get_mut(&offset)
            .ok_or_else(|| format!("no block at offset {offset}"))?;
        if block.state == BlockState::Free {
            return Err("block is already free".into());
        }
        block.state = BlockState::Free;
        block.label = None;

        // Merge adjacent free blocks.
        Self::merge_free_neighbours(&mut blocks, offset);
        Ok(())
    }

    // -- Merge helper --------------------------------------------------------

    fn merge_free_neighbours(blocks: &mut BTreeMap<usize, MemoryBlock>, offset: usize) {
        // Merge with next block if free.
        let end = {
            let b = &blocks[&offset];
            b.offset + b.size
        };
        if let Some(next) = blocks.get(&end).cloned() {
            if next.state == BlockState::Free {
                blocks.get_mut(&offset).unwrap().size += next.size;
                blocks.remove(&end);
            }
        }

        // Merge with previous block if free.
        let prev_offset = blocks
            .range(..offset)
            .next_back()
            .map(|(&o, _)| o);
        if let Some(po) = prev_offset {
            let prev = blocks.get(&po).unwrap().clone();
            if prev.state == BlockState::Free && prev.offset + prev.size == offset {
                let cur = blocks.remove(&offset).unwrap();
                blocks.get_mut(&po).unwrap().size += cur.size;
            }
        }
    }

    // -- Metrics -------------------------------------------------------------

    fn used_bytes_inner(blocks: &BTreeMap<usize, MemoryBlock>) -> usize {
        blocks
            .values()
            .filter(|b| b.state == BlockState::Allocated)
            .map(|b| b.size)
            .sum()
    }

    fn free_blocks_inner(blocks: &BTreeMap<usize, MemoryBlock>) -> Vec<&MemoryBlock> {
        blocks.values().filter(|b| b.state == BlockState::Free).collect()
    }

    /// Compute fragmentation ratio.
    ///
    /// Defined as `1 - (largest_free_block / total_free_bytes)`.  When all
    /// free memory is contiguous the ratio is 0; when every free byte is in a
    /// separate block it approaches 1.
    pub fn fragmentation_ratio(&self) -> f64 {
        let blocks = self.blocks.lock().unwrap();
        let free: Vec<_> = Self::free_blocks_inner(&blocks);
        let total_free: usize = free.iter().map(|b| b.size).sum();
        if total_free == 0 {
            return 0.0;
        }
        let max_free = free.iter().map(|b| b.size).max().unwrap_or(0);
        1.0 - (max_free as f64 / total_free as f64)
    }

    /// Build a usage report.
    pub fn usage_report(&self) -> MemoryUsageReport {
        let blocks = self.blocks.lock().unwrap();
        let used_bytes = Self::used_bytes_inner(&blocks);
        let free_blocks: Vec<_> = Self::free_blocks_inner(&blocks);
        let total_free: usize = free_blocks.iter().map(|b| b.size).sum();
        let max_free = free_blocks.iter().map(|b| b.size).max().unwrap_or(0);
        let frag_ratio = if total_free > 0 {
            1.0 - (max_free as f64 / total_free as f64)
        } else {
            0.0
        };
        let fragmented_bytes = if total_free > 0 {
            total_free - max_free
        } else {
            0
        };
        let peak = *self.peak_bytes.lock().unwrap();

        MemoryUsageReport {
            total_bytes: self.config.capacity_bytes,
            used_bytes,
            fragmented_bytes,
            peak_bytes: peak,
            allocated_blocks: blocks.values().filter(|b| b.state == BlockState::Allocated).count(),
            free_blocks: free_blocks.len(),
            fragmentation_ratio: frag_ratio,
        }
    }

    // -- Compaction / defragmentation ----------------------------------------

    /// Compact all allocated blocks toward the beginning of the pool,
    /// coalescing free space at the end.
    ///
    /// Returns the number of blocks relocated.
    pub fn compact(&self) -> usize {
        let mut blocks = self.blocks.lock().unwrap();
        let allocated: Vec<MemoryBlock> = blocks
            .values()
            .filter(|b| b.state == BlockState::Allocated)
            .cloned()
            .collect();

        blocks.clear();
        let mut cursor = 0usize;
        let mut relocated = 0usize;

        for mut blk in allocated {
            if blk.offset != cursor {
                relocated += 1;
            }
            blk.offset = cursor;
            blocks.insert(cursor, blk.clone());
            cursor += blk.size;
        }

        // Remaining space becomes one large free block.
        if cursor < self.config.capacity_bytes {
            blocks.insert(
                cursor,
                MemoryBlock {
                    offset: cursor,
                    size: self.config.capacity_bytes - cursor,
                    state: BlockState::Free,
                    label: None,
                },
            );
        }

        relocated
    }

    /// Check fragmentation and compact if above the configured threshold.
    ///
    /// Returns `true` if defragmentation was performed.
    pub fn maybe_defrag(&self) -> bool {
        if self
            .defrag_running
            .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
            .is_err()
        {
            return false; // already running
        }

        let should = self.fragmentation_ratio() >= self.config.defrag_threshold;
        if should {
            self.compact();
        }
        self.defrag_running.store(false, Ordering::SeqCst);
        should
    }

    /// Return true while a defragmentation pass is in progress.
    pub fn is_defrag_running(&self) -> bool {
        self.defrag_running.load(Ordering::SeqCst)
    }

    /// Total pool capacity.
    pub fn capacity(&self) -> usize {
        self.config.capacity_bytes
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn pool_1k() -> MemoryPool {
        MemoryPool::new(MemoryPoolConfig {
            capacity_bytes: 1024,
            defrag_threshold: 0.3,
            min_free_block_bytes: 1,
        })
        .unwrap()
    }

    // -- Config validation ---------------------------------------------------

    #[test]
    fn test_config_default_valid() {
        assert!(MemoryPoolConfig::default().validate().is_ok());
    }

    #[test]
    fn test_config_zero_capacity_rejected() {
        let mut cfg = MemoryPoolConfig::default();
        cfg.capacity_bytes = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_bad_threshold_rejected() {
        let mut cfg = MemoryPoolConfig::default();
        cfg.defrag_threshold = 1.5;
        assert!(cfg.validate().is_err());
    }

    // -- Allocation / deallocation -------------------------------------------

    #[test]
    fn test_allocate_and_report() {
        let pool = pool_1k();
        let off = pool.allocate(256, Some("a".into())).unwrap();
        assert_eq!(off, 0);
        let report = pool.usage_report();
        assert_eq!(report.used_bytes, 256);
        assert_eq!(report.allocated_blocks, 1);
        assert_eq!(report.total_bytes, 1024);
    }

    #[test]
    fn test_allocate_zero_rejected() {
        let pool = pool_1k();
        assert!(pool.allocate(0, None).is_err());
    }

    #[test]
    fn test_allocate_oom() {
        let pool = pool_1k();
        assert!(pool.allocate(2048, None).is_err());
    }

    #[test]
    fn test_deallocate_and_merge() {
        let pool = pool_1k();
        let o1 = pool.allocate(128, None).unwrap();
        let o2 = pool.allocate(128, None).unwrap();
        pool.deallocate(o1).unwrap();
        pool.deallocate(o2).unwrap();
        // After freeing both adjacent blocks they should merge into one.
        let report = pool.usage_report();
        assert_eq!(report.free_blocks, 1);
        assert_eq!(report.used_bytes, 0);
    }

    #[test]
    fn test_double_free_rejected() {
        let pool = pool_1k();
        let o = pool.allocate(64, None).unwrap();
        pool.deallocate(o).unwrap();
        assert!(pool.deallocate(o).is_err());
    }

    // -- Fragmentation detection ---------------------------------------------

    #[test]
    fn test_fragmentation_ratio_zero_when_contiguous() {
        let pool = pool_1k();
        assert_eq!(pool.fragmentation_ratio(), 0.0);
    }

    #[test]
    fn test_fragmentation_ratio_increases() {
        let pool = pool_1k();
        let o1 = pool.allocate(128, None).unwrap();
        let _o2 = pool.allocate(128, None).unwrap();
        let o3 = pool.allocate(128, None).unwrap();
        // Free alternating blocks → two non-contiguous free regions.
        pool.deallocate(o1).unwrap();
        pool.deallocate(o3).unwrap();
        let ratio = pool.fragmentation_ratio();
        assert!(ratio > 0.0, "expected fragmentation, got {ratio}");
    }

    // -- Compaction ----------------------------------------------------------

    #[test]
    fn test_compact_coalesces_free_space() {
        let pool = pool_1k();
        let o1 = pool.allocate(100, None).unwrap();
        let _o2 = pool.allocate(200, None).unwrap();
        let _o3 = pool.allocate(100, None).unwrap();
        pool.deallocate(o1).unwrap();
        // Before compaction: hole at the front.
        let relocated = pool.compact();
        assert!(relocated > 0);
        let report = pool.usage_report();
        // After compaction: only 1 free block at the end.
        assert_eq!(report.free_blocks, 1);
    }

    // -- Threshold-triggered defrag ------------------------------------------

    #[test]
    fn test_maybe_defrag_triggers_above_threshold() {
        let pool = MemoryPool::new(MemoryPoolConfig {
            capacity_bytes: 1024,
            defrag_threshold: 0.01, // very low threshold → always trigger
            min_free_block_bytes: 1,
        })
        .unwrap();
        let o1 = pool.allocate(100, None).unwrap();
        let _o2 = pool.allocate(100, None).unwrap();
        let o3 = pool.allocate(100, None).unwrap();
        pool.deallocate(o1).unwrap();
        pool.deallocate(o3).unwrap();
        // Fragmentation > 0.01 → should trigger.
        assert!(pool.maybe_defrag());
    }

    #[test]
    fn test_maybe_defrag_skips_below_threshold() {
        let pool = MemoryPool::new(MemoryPoolConfig {
            capacity_bytes: 1024,
            defrag_threshold: 0.99,
            min_free_block_bytes: 1,
        })
        .unwrap();
        // No allocations → 0 fragmentation → should not trigger.
        assert!(!pool.maybe_defrag());
    }

    // -- Peak tracking -------------------------------------------------------

    #[test]
    fn test_peak_tracking() {
        let pool = pool_1k();
        pool.allocate(400, None).unwrap();
        let o2 = pool.allocate(300, None).unwrap();
        pool.deallocate(o2).unwrap();
        let report = pool.usage_report();
        assert_eq!(report.peak_bytes, 700);
        assert_eq!(report.used_bytes, 400);
    }
}
