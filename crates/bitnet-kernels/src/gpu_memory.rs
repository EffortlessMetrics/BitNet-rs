//! CUDA memory management for GPU buffer allocation and transfers.
//!
//! Provides a simple arena-style memory pool that tracks GPU buffer allocations
//! without requiring actual CUDA runtime calls. This enables development and
//! testing of memory management logic on CPU-only machines.
//!
//! # Components
//!
//! - [`GpuBuffer`] — metadata for a single GPU allocation (size, alignment, device).
//! - [`MemoryPool`] — arena allocator that hands out [`BufferHandle`] tokens.
//! - [`PoolStats`] — snapshot of pool utilisation and fragmentation.
//! - [`TransferPlan`] — describes a host↔device or device↔device copy.

use bitnet_common::{Device, KernelError, Result};

// ── Buffer handle ────────────────────────────────────────────────────

/// Opaque handle returned by [`MemoryPool::allocate`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BufferHandle(u64);

impl BufferHandle {
    /// Returns the raw numeric id of this handle.
    #[inline]
    pub fn id(&self) -> u64 {
        self.0
    }
}

// ── GpuBuffer ────────────────────────────────────────────────────────

/// Metadata for a single GPU allocation.
#[derive(Debug, Clone)]
pub struct GpuBuffer {
    /// Unique handle for this buffer.
    pub handle: BufferHandle,
    /// Size in bytes.
    pub size: usize,
    /// Alignment in bytes (must be a power of two).
    pub alignment: usize,
    /// Device the buffer resides on.
    pub device_id: Device,
    /// Offset within the pool's arena.
    offset: usize,
}

impl GpuBuffer {
    /// Returns the arena offset for this buffer.
    #[inline]
    pub fn offset(&self) -> usize {
        self.offset
    }
}

// ── PoolStats ────────────────────────────────────────────────────────

/// Snapshot of memory-pool utilisation.
#[derive(Debug, Clone, PartialEq)]
pub struct PoolStats {
    /// Total capacity in bytes.
    pub total_bytes: usize,
    /// Bytes currently allocated.
    pub used_bytes: usize,
    /// Bytes available for new allocations.
    pub free_bytes: usize,
    /// Number of live allocations.
    pub num_allocations: usize,
    /// Fragmentation ratio in `[0.0, 1.0]`.
    ///
    /// Defined as `1 − (largest_contiguous_free / free_bytes)`.
    /// A value of `0.0` means all free space is contiguous.
    pub fragmentation: f64,
}

// ── MemoryPool ───────────────────────────────────────────────────────

/// A simple arena allocator that simulates GPU memory management.
///
/// Allocations are placed using a first-fit strategy over a list of free
/// regions. This is intentionally simple — production GPU allocators use
/// slab/buddy schemes, but first-fit is sufficient for scaffolding and
/// correctness testing.
#[derive(Debug)]
pub struct MemoryPool {
    capacity: usize,
    device: Device,
    next_handle: u64,
    /// Live allocations keyed by handle id.
    allocations: std::collections::HashMap<u64, GpuBuffer>,
    /// Sorted list of free regions `(offset, size)`.
    free_regions: Vec<(usize, usize)>,
}

impl MemoryPool {
    /// Creates a new pool with the given total `capacity` on `device`.
    pub fn new(capacity: usize, device: Device) -> Self {
        Self {
            capacity,
            device,
            next_handle: 1,
            allocations: std::collections::HashMap::new(),
            free_regions: vec![(0, capacity)],
        }
    }

    /// Allocates `size` bytes with the given alignment.
    ///
    /// Returns a [`BufferHandle`] on success. Fails with
    /// [`KernelError::GpuError`] when the pool cannot satisfy the request.
    pub fn allocate(&mut self, size: usize, align: usize) -> Result<BufferHandle> {
        if size == 0 {
            return Err(KernelError::InvalidArguments {
                reason: "allocation size must be > 0".into(),
            }
            .into());
        }
        if !align.is_power_of_two() {
            return Err(KernelError::InvalidArguments {
                reason: format!("alignment must be a power of two, got {align}"),
            }
            .into());
        }

        // First-fit over free regions.
        let mut found: Option<(usize, usize, usize)> = None; // (index, aligned_offset, padded_size)
        for (i, &(offset, region_size)) in self.free_regions.iter().enumerate() {
            let aligned_offset = align_up(offset, align);
            let padding = aligned_offset - offset;
            let needed = padding + size;
            if region_size >= needed {
                found = Some((i, aligned_offset, needed));
                break;
            }
        }

        let (idx, aligned_offset, needed) = found.ok_or_else(|| KernelError::GpuError {
            reason: format!(
                "out of GPU memory: requested {size} bytes (align {align}), pool has {} free",
                self.free_bytes(),
            ),
        })?;

        // Carve the allocation out of the free region.
        let (region_offset, region_size) = self.free_regions[idx];
        self.free_regions.remove(idx);

        // Left fragment (alignment padding becomes a free region only when > 0
        // and distinct from the aligned start).
        let left_size = aligned_offset - region_offset;
        if left_size > 0 {
            self.free_regions.push((region_offset, left_size));
        }
        // Right fragment.
        let right_offset = region_offset + needed;
        let right_size = region_size - needed;
        if right_size > 0 {
            self.free_regions.push((right_offset, right_size));
        }
        self.free_regions.sort_by_key(|&(off, _)| off);

        let handle = BufferHandle(self.next_handle);
        self.next_handle += 1;

        self.allocations.insert(
            handle.0,
            GpuBuffer {
                handle,
                size,
                alignment: align,
                device_id: self.device,
                offset: aligned_offset,
            },
        );

        Ok(handle)
    }

    /// Releases the allocation identified by `handle`.
    ///
    /// The freed region is merged with adjacent free regions to reduce
    /// fragmentation.
    pub fn deallocate(&mut self, handle: BufferHandle) {
        let Some(buf) = self.allocations.remove(&handle.0) else {
            return; // double-free is a no-op
        };

        // The region to return spans from the aligned offset for `size` bytes,
        // but we also need to account for any padding that was consumed.
        // We stored the aligned offset in the buffer; the actual consumed range
        // is [aligned_offset, aligned_offset + size).
        let free_offset = buf.offset;
        let free_size = buf.size;
        self.free_regions.push((free_offset, free_size));
        self.free_regions.sort_by_key(|&(off, _)| off);
        self.coalesce_free_regions();
    }

    /// Returns a snapshot of pool statistics.
    pub fn stats(&self) -> PoolStats {
        let used = self.used_bytes();
        let free = self.free_bytes();
        let largest_free = self.free_regions.iter().map(|&(_, s)| s).max().unwrap_or(0);
        let fragmentation = if free == 0 { 0.0 } else { 1.0 - (largest_free as f64 / free as f64) };
        PoolStats {
            total_bytes: self.capacity,
            used_bytes: used,
            free_bytes: free,
            num_allocations: self.allocations.len(),
            fragmentation,
        }
    }

    /// Number of live allocations.
    #[inline]
    pub fn num_allocations(&self) -> usize {
        self.allocations.len()
    }

    /// Looks up a buffer by its handle.
    pub fn get(&self, handle: BufferHandle) -> Option<&GpuBuffer> {
        self.allocations.get(&handle.0)
    }

    // ── internal helpers ─────────────────────────────────────────────

    fn used_bytes(&self) -> usize {
        self.allocations.values().map(|b| b.size).sum()
    }

    fn free_bytes(&self) -> usize {
        self.free_regions.iter().map(|&(_, s)| s).sum()
    }

    /// Merge adjacent free regions.
    fn coalesce_free_regions(&mut self) {
        if self.free_regions.len() < 2 {
            return;
        }
        let mut merged: Vec<(usize, usize)> = Vec::with_capacity(self.free_regions.len());
        merged.push(self.free_regions[0]);
        for &(offset, size) in &self.free_regions[1..] {
            let last = merged.last_mut().expect("merged is non-empty");
            if last.0 + last.1 == offset {
                last.1 += size;
            } else {
                merged.push((offset, size));
            }
        }
        self.free_regions = merged;
    }
}

// ── TransferPlan ─────────────────────────────────────────────────────

/// Direction of a memory transfer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransferKind {
    /// Host (CPU) → Device (GPU).
    HostToDevice,
    /// Device (GPU) → Host (CPU).
    DeviceToHost,
    /// Device → Device (peer-to-peer or same-device copy).
    DeviceToDevice,
    /// No transfer needed (same location).
    NoOp,
}

/// Describes a planned memory transfer between devices.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TransferPlan {
    /// Source device.
    pub src: Device,
    /// Destination device.
    pub dst: Device,
    /// Number of bytes to transfer.
    pub size: usize,
    /// Transfer direction.
    pub kind: TransferKind,
}

/// Builds a [`TransferPlan`] for moving `size` bytes from `src` to `dst`.
pub fn transfer_plan(src: Device, dst: Device, size: usize) -> TransferPlan {
    let kind = match (&src, &dst) {
        _ if src == dst => TransferKind::NoOp,
        (Device::Cpu, Device::Cuda(_) | Device::Hip(_) | Device::Metal | Device::OpenCL(_)) => {
            TransferKind::HostToDevice
        }
        (Device::Cuda(_) | Device::Hip(_) | Device::Metal | Device::OpenCL(_), Device::Cpu) => {
            TransferKind::DeviceToHost
        }
        _ => TransferKind::DeviceToDevice,
    };
    TransferPlan { src, dst, size, kind }
}

// ── helpers ──────────────────────────────────────────────────────────

/// Round `offset` up to the next multiple of `align`.
#[inline]
fn align_up(offset: usize, align: usize) -> usize {
    (offset + align - 1) & !(align - 1)
}

// ── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn pool(cap: usize) -> MemoryPool {
        MemoryPool::new(cap, Device::Cuda(0))
    }

    // ── basic allocation ────────────────────────────────────────────

    #[test]
    fn test_single_allocation() {
        let mut p = pool(1024);
        let h = p.allocate(256, 1).unwrap();
        assert_eq!(p.num_allocations(), 1);
        let buf = p.get(h).unwrap();
        assert_eq!(buf.size, 256);
        assert_eq!(buf.device_id, Device::Cuda(0));
    }

    #[test]
    fn test_allocation_returns_unique_handles() {
        let mut p = pool(1024);
        let h1 = p.allocate(64, 1).unwrap();
        let h2 = p.allocate(64, 1).unwrap();
        assert_ne!(h1, h2);
    }

    #[test]
    fn test_allocate_zero_size_fails() {
        let mut p = pool(1024);
        assert!(p.allocate(0, 1).is_err());
    }

    #[test]
    fn test_allocate_bad_alignment_fails() {
        let mut p = pool(1024);
        assert!(p.allocate(64, 3).is_err()); // 3 is not a power of two
    }

    #[test]
    fn test_allocate_exact_capacity() {
        let mut p = pool(512);
        let h = p.allocate(512, 1).unwrap();
        assert_eq!(p.stats().free_bytes, 0);
        assert_eq!(p.get(h).unwrap().size, 512);
    }

    // ── deallocation ────────────────────────────────────────────────

    #[test]
    fn test_deallocate_frees_space() {
        let mut p = pool(1024);
        let h = p.allocate(256, 1).unwrap();
        assert_eq!(p.stats().used_bytes, 256);
        p.deallocate(h);
        assert_eq!(p.stats().used_bytes, 0);
        assert_eq!(p.num_allocations(), 0);
    }

    #[test]
    fn test_double_deallocate_is_noop() {
        let mut p = pool(1024);
        let h = p.allocate(128, 1).unwrap();
        p.deallocate(h);
        p.deallocate(h); // should not panic
        assert_eq!(p.num_allocations(), 0);
    }

    #[test]
    fn test_deallocate_unknown_handle_is_noop() {
        let mut p = pool(1024);
        p.deallocate(BufferHandle(999));
        assert_eq!(p.num_allocations(), 0);
    }

    // ── pool exhaustion ─────────────────────────────────────────────

    #[test]
    fn test_pool_exhaustion() {
        let mut p = pool(256);
        let _h1 = p.allocate(256, 1).unwrap();
        assert!(p.allocate(1, 1).is_err());
    }

    #[test]
    fn test_pool_exhaustion_after_many_allocations() {
        let mut p = pool(128);
        for _ in 0..8 {
            p.allocate(16, 1).unwrap();
        }
        assert!(p.allocate(1, 1).is_err());
    }

    #[test]
    fn test_reuse_after_deallocation() {
        let mut p = pool(256);
        let h = p.allocate(256, 1).unwrap();
        assert!(p.allocate(1, 1).is_err());
        p.deallocate(h);
        p.allocate(256, 1).unwrap(); // should succeed now
    }

    // ── alignment ───────────────────────────────────────────────────

    #[test]
    fn test_aligned_allocation() {
        let mut p = pool(4096);
        // First alloc at offset 0, size 1 — puts next free region at offset 1.
        let _h1 = p.allocate(1, 1).unwrap();
        // Second alloc with 256-byte alignment should land at offset 256.
        let h2 = p.allocate(64, 256).unwrap();
        let buf = p.get(h2).unwrap();
        assert_eq!(buf.offset() % 256, 0);
    }

    #[test]
    fn test_alignment_power_of_two_variants() {
        let mut p = pool(8192);
        for align in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024] {
            let h = p.allocate(1, align).unwrap();
            let buf = p.get(h).unwrap();
            assert_eq!(buf.offset() % align, 0, "failed for alignment {align}");
        }
    }

    // ── varying sizes ───────────────────────────────────────────────

    #[test]
    fn test_mixed_sizes() {
        let mut p = pool(1024 * 1024);
        let sizes = [1, 7, 16, 63, 128, 255, 512, 1023, 4096, 65536];
        let mut handles = Vec::new();
        for &s in &sizes {
            handles.push(p.allocate(s, 1).unwrap());
        }
        assert_eq!(p.num_allocations(), sizes.len());
        let total_alloc: usize = sizes.iter().sum();
        assert_eq!(p.stats().used_bytes, total_alloc);
    }

    #[test]
    fn test_large_then_small_allocations() {
        let mut p = pool(1024);
        let h = p.allocate(512, 1).unwrap();
        let _h2 = p.allocate(256, 1).unwrap();
        p.deallocate(h);
        // 512 bytes freed; allocate 3 × 128 in the gap.
        for _ in 0..3 {
            p.allocate(128, 1).unwrap();
        }
    }

    // ── fragmentation ───────────────────────────────────────────────

    #[test]
    fn test_no_fragmentation_when_empty() {
        let p = pool(1024);
        assert_eq!(p.stats().fragmentation, 0.0);
    }

    #[test]
    fn test_fragmentation_after_interleaved_free() {
        let mut p = pool(1024);
        // Allocate 4 blocks of 256 bytes.
        let h1 = p.allocate(256, 1).unwrap();
        let _h2 = p.allocate(256, 1).unwrap();
        let h3 = p.allocate(256, 1).unwrap();
        let _h4 = p.allocate(256, 1).unwrap();
        // Free alternating blocks → two non-contiguous free regions.
        p.deallocate(h1);
        p.deallocate(h3);
        let stats = p.stats();
        assert_eq!(stats.free_bytes, 512);
        assert!(stats.fragmentation > 0.0, "expected fragmentation");
    }

    #[test]
    fn test_fragmentation_goes_to_zero_after_full_free() {
        let mut p = pool(1024);
        let h1 = p.allocate(256, 1).unwrap();
        let h2 = p.allocate(256, 1).unwrap();
        p.deallocate(h1);
        p.deallocate(h2);
        // After all frees and coalescing, fragmentation should be 0.
        assert_eq!(p.stats().fragmentation, 0.0);
    }

    // ── stats invariant: used + free = total ────────────────────────

    #[test]
    fn test_stats_invariant_empty_pool() {
        let p = pool(4096);
        let s = p.stats();
        assert_eq!(s.used_bytes + s.free_bytes, s.total_bytes);
    }

    #[test]
    fn test_stats_invariant_after_alloc_dealloc() {
        let mut p = pool(4096);
        let mut handles = Vec::new();
        // Allocate several buffers.
        for size in [100, 200, 300, 400] {
            handles.push(p.allocate(size, 1).unwrap());
        }
        let s = p.stats();
        assert_eq!(s.used_bytes + s.free_bytes, s.total_bytes);
        // Deallocate half.
        p.deallocate(handles[1]);
        p.deallocate(handles[3]);
        let s = p.stats();
        assert_eq!(s.used_bytes + s.free_bytes, s.total_bytes);
    }

    #[test]
    fn test_stats_invariant_full_pool() {
        let mut p = pool(512);
        let _h = p.allocate(512, 1).unwrap();
        let s = p.stats();
        assert_eq!(s.used_bytes, 512);
        assert_eq!(s.free_bytes, 0);
        assert_eq!(s.used_bytes + s.free_bytes, s.total_bytes);
    }

    // ── transfer plan ───────────────────────────────────────────────

    #[test]
    fn test_transfer_plan_host_to_device() {
        let plan = transfer_plan(Device::Cpu, Device::Cuda(0), 1024);
        assert_eq!(plan.kind, TransferKind::HostToDevice);
        assert_eq!(plan.size, 1024);
    }

    #[test]
    fn test_transfer_plan_device_to_host() {
        let plan = transfer_plan(Device::Cuda(0), Device::Cpu, 512);
        assert_eq!(plan.kind, TransferKind::DeviceToHost);
    }

    #[test]
    fn test_transfer_plan_device_to_device() {
        let plan = transfer_plan(Device::Cuda(0), Device::Cuda(1), 256);
        assert_eq!(plan.kind, TransferKind::DeviceToDevice);
    }

    #[test]
    fn test_transfer_plan_same_device_noop() {
        let plan = transfer_plan(Device::Cuda(0), Device::Cuda(0), 128);
        assert_eq!(plan.kind, TransferKind::NoOp);
    }

    #[test]
    fn test_transfer_plan_cpu_to_cpu_noop() {
        let plan = transfer_plan(Device::Cpu, Device::Cpu, 64);
        assert_eq!(plan.kind, TransferKind::NoOp);
    }

    #[test]
    fn test_transfer_plan_metal_variants() {
        let h2d = transfer_plan(Device::Cpu, Device::Metal, 32);
        assert_eq!(h2d.kind, TransferKind::HostToDevice);
        let d2h = transfer_plan(Device::Metal, Device::Cpu, 32);
        assert_eq!(d2h.kind, TransferKind::DeviceToHost);
    }

    // ── coalescing ──────────────────────────────────────────────────

    #[test]
    fn test_coalescing_adjacent_frees() {
        let mut p = pool(1024);
        let h1 = p.allocate(256, 1).unwrap();
        let h2 = p.allocate(256, 1).unwrap();
        let h3 = p.allocate(256, 1).unwrap();
        // Free in order — should coalesce into one big free region.
        p.deallocate(h1);
        p.deallocate(h2);
        p.deallocate(h3);
        assert_eq!(p.stats().fragmentation, 0.0);
        // Can re-allocate the full contiguous block.
        p.allocate(768, 1).unwrap();
    }

    // ── property-style: many random-ish ops ─────────────────────────

    #[test]
    fn test_bulk_alloc_dealloc_invariant() {
        let mut p = pool(1024 * 1024);
        let mut handles = Vec::new();
        // Allocate 100 buffers of varying sizes.
        for i in 0..100 {
            let size = (i % 17 + 1) * 64; // 64..1088 stepping
            handles.push(p.allocate(size, 1).unwrap());
        }
        let s = p.stats();
        assert_eq!(s.num_allocations, 100);
        assert_eq!(s.used_bytes + s.free_bytes, s.total_bytes);
        // Free every other one.
        for i in (0..100).step_by(2) {
            p.deallocate(handles[i]);
        }
        let s = p.stats();
        assert_eq!(s.num_allocations, 50);
        assert_eq!(s.used_bytes + s.free_bytes, s.total_bytes);
    }

    // ── align_up helper ─────────────────────────────────────────────

    #[test]
    fn test_align_up() {
        assert_eq!(align_up(0, 16), 0);
        assert_eq!(align_up(1, 16), 16);
        assert_eq!(align_up(15, 16), 16);
        assert_eq!(align_up(16, 16), 16);
        assert_eq!(align_up(17, 16), 32);
    }
}
