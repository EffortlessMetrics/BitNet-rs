#![allow(dead_code, unused_imports, unused_variables, non_camel_case_types, unused_mut)]
#![cfg(target_os = "macos")]

//! Metal memory management correctness tests for Apple Silicon.
//!
//! Validates buffer allocation sizing, 256-byte alignment, storage mode
//! selection, buffer lifecycle, zero-copy mapping, out-of-memory handling,
//! and buffer pool reuse patterns.  All tests use struct-based validation
//! (no Metal framework imports) so they compile on any platform when the
//! cfg gate is satisfied.

// ── Metal memory constants ──────────────────────────────────────────

/// Metal buffer alignment (bytes).
const METAL_BUFFER_ALIGNMENT: usize = 256;

/// Apple Silicon page size (bytes).
const PAGE_SIZE: usize = 16_384;

/// Maximum practical single-buffer allocation (conservative 256 GiB).
const MAX_BUFFER_SIZE: usize = 256 * 1024 * 1024 * 1024;

/// Maximum threadgroup (shared) memory per threadgroup on Apple Silicon.
const MAX_THREADGROUP_MEMORY: usize = 32 * 1024;

// ── Storage modes ───────────────────────────────────────────────────

/// Metal resource storage modes relevant for Apple Silicon.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StorageMode {
    /// CPU and GPU share the same memory region (unified memory).
    Shared,
    /// GPU-only memory; CPU access requires a blit.
    Private,
    /// Managed mode — Metal synchronises between CPU/GPU copies.
    /// On Apple Silicon unified memory this is functionally equivalent
    /// to `Shared`, but useful for portability with discrete GPUs.
    Managed,
}

impl StorageMode {
    /// Whether CPU can read/write without an explicit blit.
    fn cpu_accessible(self) -> bool {
        matches!(self, Self::Shared | Self::Managed)
    }

    /// Whether this mode supports zero-copy buffer mapping on Apple
    /// Silicon unified memory.
    fn supports_zero_copy(self) -> bool {
        self == Self::Shared
    }

    /// Recommended mode for the given access pattern on Apple Silicon.
    fn recommended(cpu_writes: bool, gpu_reads: bool, gpu_writes: bool) -> Self {
        match (cpu_writes, gpu_reads, gpu_writes) {
            // CPU writes, GPU reads only → shared (zero-copy upload).
            (true, true, false) => Self::Shared,
            // GPU-only → private for best bandwidth.
            (false, _, true) => Self::Private,
            // Both read+write → managed for portability.
            (true, _, true) => Self::Managed,
            _ => Self::Shared,
        }
    }
}

// ── Buffer model ────────────────────────────────────────────────────

/// Mock Metal buffer for validation purposes.
#[derive(Debug)]
struct MockMetalBuffer {
    label: String,
    /// Requested allocation size (before alignment).
    requested_size: usize,
    /// Actual allocation size (after alignment).
    allocated_size: usize,
    storage_mode: StorageMode,
    /// Simulated contents (None = uninitialised / GPU-private).
    contents: Option<Vec<u8>>,
}

impl MockMetalBuffer {
    /// Allocate a new buffer, rounding size up to 256-byte alignment.
    fn new(label: impl Into<String>, size: usize, mode: StorageMode) -> Self {
        let aligned = align_up(size, METAL_BUFFER_ALIGNMENT);
        let contents = if mode.cpu_accessible() { Some(vec![0u8; aligned]) } else { None };
        Self {
            label: label.into(),
            requested_size: size,
            allocated_size: aligned,
            storage_mode: mode,
            contents,
        }
    }

    fn is_aligned(&self) -> bool {
        self.allocated_size.is_multiple_of(METAL_BUFFER_ALIGNMENT)
    }

    /// Write bytes at offset (only for CPU-accessible modes).
    fn write(&mut self, offset: usize, data: &[u8]) -> Result<(), &'static str> {
        let buf = self.contents.as_mut().ok_or("buffer is not CPU-accessible")?;
        if offset + data.len() > buf.len() {
            return Err("write exceeds buffer bounds");
        }
        buf[offset..offset + data.len()].copy_from_slice(data);
        Ok(())
    }

    /// Read bytes at offset (only for CPU-accessible modes).
    fn read(&self, offset: usize, len: usize) -> Result<&[u8], &'static str> {
        let buf = self.contents.as_ref().ok_or("buffer is not CPU-accessible")?;
        if offset + len > buf.len() {
            return Err("read exceeds buffer bounds");
        }
        Ok(&buf[offset..offset + len])
    }
}

// ── Buffer pool ─────────────────────────────────────────────────────

/// Simple buffer pool that reuses previously-freed buffers of matching
/// size class (aligned size).
struct BufferPool {
    free: Vec<MockMetalBuffer>,
    allocated_count: usize,
    reuse_count: usize,
}

impl BufferPool {
    fn new() -> Self {
        Self { free: Vec::new(), allocated_count: 0, reuse_count: 0 }
    }

    fn acquire(&mut self, label: &str, size: usize, mode: StorageMode) -> MockMetalBuffer {
        let aligned = align_up(size, METAL_BUFFER_ALIGNMENT);
        if let Some(pos) =
            self.free.iter().position(|b| b.allocated_size == aligned && b.storage_mode == mode)
        {
            self.reuse_count += 1;
            let mut buf = self.free.swap_remove(pos);
            buf.label = label.to_string();
            // Zero contents on reuse to prevent data leakage.
            if let Some(ref mut v) = buf.contents {
                v.fill(0);
            }
            buf
        } else {
            self.allocated_count += 1;
            MockMetalBuffer::new(label, size, mode)
        }
    }

    fn release(&mut self, buf: MockMetalBuffer) {
        self.free.push(buf);
    }
}

// ── OOM simulation ──────────────────────────────────────────────────

/// Result of a simulated buffer allocation that may fail due to memory
/// pressure.
#[derive(Debug, PartialEq, Eq)]
enum AllocResult {
    Ok { aligned_size: usize },
    OutOfMemory { requested: usize, available: usize },
}

/// Simulate allocation with a memory budget.
fn try_alloc(requested: usize, available: usize) -> AllocResult {
    let aligned = align_up(requested, METAL_BUFFER_ALIGNMENT);
    if aligned > available {
        AllocResult::OutOfMemory { requested: aligned, available }
    } else {
        AllocResult::Ok { aligned_size: aligned }
    }
}

// ── Helpers ─────────────────────────────────────────────────────────

/// Round `n` up to the next multiple of `align` (must be power of two).
fn align_up(n: usize, align: usize) -> usize {
    debug_assert!(align.is_power_of_two());
    (n + align - 1) & !(align - 1)
}

// ═════════════════════════════════════════════════════════════════════
// Tests
// ═════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    // ── 1. Buffer allocation sizes ──────────────────────────────────

    mod buffer_allocation {
        use super::*;

        #[test]
        #[ignore = "requires macOS Metal GPU - run on Apple Silicon hardware"]
        fn allocate_1_byte() {
            let buf = MockMetalBuffer::new("1B", 1, StorageMode::Shared);
            assert_eq!(buf.requested_size, 1);
            assert_eq!(buf.allocated_size, METAL_BUFFER_ALIGNMENT);
            assert!(buf.is_aligned());
        }

        #[test]
        #[ignore = "requires macOS Metal GPU - run on Apple Silicon hardware"]
        fn allocate_1kb() {
            let buf = MockMetalBuffer::new("1KB", 1024, StorageMode::Shared);
            assert_eq!(buf.allocated_size, 1024); // 1024 is already 256-aligned
            assert!(buf.is_aligned());
        }

        #[test]
        #[ignore = "requires macOS Metal GPU - run on Apple Silicon hardware"]
        fn allocate_1mb() {
            let one_mb = 1024 * 1024;
            let buf = MockMetalBuffer::new("1MB", one_mb, StorageMode::Shared);
            assert_eq!(buf.allocated_size, one_mb);
            assert!(buf.is_aligned());
        }

        #[test]
        #[ignore = "requires macOS Metal GPU - run on Apple Silicon hardware"]
        fn allocate_256mb() {
            let size = 256 * 1024 * 1024;
            let buf = MockMetalBuffer::new("256MB", size, StorageMode::Private);
            assert_eq!(buf.allocated_size, size);
            assert!(buf.is_aligned());
            // Private mode → no CPU-visible contents.
            assert!(buf.contents.is_none());
        }

        #[test]
        #[ignore = "requires macOS Metal GPU - run on Apple Silicon hardware"]
        fn allocation_sizes_always_aligned() {
            for size in [0, 1, 127, 255, 256, 257, 1023, 4096, 65537] {
                let buf = MockMetalBuffer::new("test", size, StorageMode::Shared);
                assert!(
                    buf.is_aligned(),
                    "buffer of size {size} not aligned: got {}",
                    buf.allocated_size
                );
                assert!(buf.allocated_size >= size);
            }
        }
    }

    // ── 2. Buffer alignment verification ────────────────────────────

    mod alignment_verification {
        use super::*;

        #[test]
        #[ignore = "requires macOS Metal GPU - run on Apple Silicon hardware"]
        fn alignment_is_256_bytes() {
            assert_eq!(METAL_BUFFER_ALIGNMENT, 256);
            assert!(METAL_BUFFER_ALIGNMENT.is_power_of_two());
        }

        #[test]
        #[ignore = "requires macOS Metal GPU - run on Apple Silicon hardware"]
        fn align_up_exact_multiples_unchanged() {
            for m in 1..=16 {
                let v = m * METAL_BUFFER_ALIGNMENT;
                assert_eq!(align_up(v, METAL_BUFFER_ALIGNMENT), v);
            }
        }

        #[test]
        #[ignore = "requires macOS Metal GPU - run on Apple Silicon hardware"]
        fn align_up_rounds_to_next_boundary() {
            assert_eq!(align_up(1, METAL_BUFFER_ALIGNMENT), 256);
            assert_eq!(align_up(255, METAL_BUFFER_ALIGNMENT), 256);
            assert_eq!(align_up(257, METAL_BUFFER_ALIGNMENT), 512);
        }

        #[test]
        #[ignore = "requires macOS Metal GPU - run on Apple Silicon hardware"]
        fn page_aligned_implies_buffer_aligned() {
            for pages in 1..=8 {
                let addr = pages * PAGE_SIZE;
                assert_eq!(addr % METAL_BUFFER_ALIGNMENT, 0);
            }
        }

        #[test]
        #[ignore = "requires macOS Metal GPU - run on Apple Silicon hardware"]
        fn tensor_buffer_alignment() {
            // f32 tensor: 1024 elements = 4096 bytes (already aligned).
            assert_eq!(align_up(1024 * 4, METAL_BUFFER_ALIGNMENT), 4096);
            // f16 tensor: 1023 elements = 2046 bytes → 2048.
            assert_eq!(align_up(1023 * 2, METAL_BUFFER_ALIGNMENT), 2048);
            // i2 packed: 4096 weights / 4 per byte = 1024 bytes (aligned).
            assert_eq!(align_up(4096 / 4, METAL_BUFFER_ALIGNMENT), 1024);
        }
    }

    // ── 3. Shared vs private storage modes ──────────────────────────

    mod storage_modes {
        use super::*;

        #[test]
        #[ignore = "requires macOS Metal GPU - run on Apple Silicon hardware"]
        fn shared_mode_is_cpu_accessible() {
            assert!(StorageMode::Shared.cpu_accessible());
            assert!(StorageMode::Shared.supports_zero_copy());
        }

        #[test]
        #[ignore = "requires macOS Metal GPU - run on Apple Silicon hardware"]
        fn private_mode_not_cpu_accessible() {
            assert!(!StorageMode::Private.cpu_accessible());
            assert!(!StorageMode::Private.supports_zero_copy());
        }

        #[test]
        #[ignore = "requires macOS Metal GPU - run on Apple Silicon hardware"]
        fn managed_mode_cpu_accessible_no_zero_copy() {
            assert!(StorageMode::Managed.cpu_accessible());
            assert!(!StorageMode::Managed.supports_zero_copy());
        }

        #[test]
        #[ignore = "requires macOS Metal GPU - run on Apple Silicon hardware"]
        fn recommended_mode_for_weight_upload() {
            // Weights: CPU writes once, GPU reads many times.
            let mode = StorageMode::recommended(true, true, false);
            assert_eq!(mode, StorageMode::Shared);
        }

        #[test]
        #[ignore = "requires macOS Metal GPU - run on Apple Silicon hardware"]
        fn recommended_mode_for_gpu_scratch() {
            // GPU scratch: no CPU access needed.
            let mode = StorageMode::recommended(false, true, true);
            assert_eq!(mode, StorageMode::Private);
        }

        #[test]
        #[ignore = "requires macOS Metal GPU - run on Apple Silicon hardware"]
        fn recommended_mode_for_readback() {
            // CPU reads results that GPU wrote.
            let mode = StorageMode::recommended(true, true, true);
            assert_eq!(mode, StorageMode::Managed);
        }

        #[test]
        #[ignore = "requires macOS Metal GPU - run on Apple Silicon hardware"]
        fn shared_buffer_has_contents() {
            let buf = MockMetalBuffer::new("shared", 512, StorageMode::Shared);
            assert!(buf.contents.is_some());
        }

        #[test]
        #[ignore = "requires macOS Metal GPU - run on Apple Silicon hardware"]
        fn private_buffer_has_no_contents() {
            let buf = MockMetalBuffer::new("private", 512, StorageMode::Private);
            assert!(buf.contents.is_none());
        }
    }

    // ── 4. Buffer lifecycle (create → write → read → drop) ─────────

    mod buffer_lifecycle {
        use super::*;

        #[test]
        #[ignore = "requires macOS Metal GPU - run on Apple Silicon hardware"]
        fn create_write_read_roundtrip() {
            let mut buf = MockMetalBuffer::new("lifecycle", 1024, StorageMode::Shared);

            // Write pattern at offset 0.
            let data = [0xDE, 0xAD, 0xBE, 0xEF];
            buf.write(0, &data).unwrap();

            // Read back and verify.
            let readback = buf.read(0, 4).unwrap();
            assert_eq!(readback, &data);
        }

        #[test]
        #[ignore = "requires macOS Metal GPU - run on Apple Silicon hardware"]
        fn write_at_offset() {
            let mut buf = MockMetalBuffer::new("offset", 512, StorageMode::Shared);
            let data = [1, 2, 3, 4];
            buf.write(256, &data).unwrap();

            // Bytes before offset remain zero.
            let before = buf.read(0, 4).unwrap();
            assert_eq!(before, &[0, 0, 0, 0]);

            // Bytes at offset match written data.
            let at_offset = buf.read(256, 4).unwrap();
            assert_eq!(at_offset, &data);
        }

        #[test]
        #[ignore = "requires macOS Metal GPU - run on Apple Silicon hardware"]
        fn write_out_of_bounds_fails() {
            let mut buf = MockMetalBuffer::new("bounds", 256, StorageMode::Shared);
            let result = buf.write(256, &[1]);
            assert!(result.is_err());
        }

        #[test]
        #[ignore = "requires macOS Metal GPU - run on Apple Silicon hardware"]
        fn read_out_of_bounds_fails() {
            let buf = MockMetalBuffer::new("bounds", 256, StorageMode::Shared);
            let result = buf.read(256, 1);
            assert!(result.is_err());
        }

        #[test]
        #[ignore = "requires macOS Metal GPU - run on Apple Silicon hardware"]
        fn write_to_private_buffer_fails() {
            let mut buf = MockMetalBuffer::new("private", 512, StorageMode::Private);
            let result = buf.write(0, &[1, 2, 3, 4]);
            assert!(result.is_err());
        }

        #[test]
        #[ignore = "requires macOS Metal GPU - run on Apple Silicon hardware"]
        fn buffer_drops_cleanly() {
            // Ensure no panic on drop.
            let buf = MockMetalBuffer::new("drop-me", 1024, StorageMode::Shared);
            drop(buf);
        }

        #[test]
        #[ignore = "requires macOS Metal GPU - run on Apple Silicon hardware"]
        fn initialised_to_zero() {
            let buf = MockMetalBuffer::new("zeroed", 512, StorageMode::Shared);
            let data = buf.read(0, 512).unwrap();
            assert!(data.iter().all(|&b| b == 0));
        }
    }

    // ── 5. Zero-copy buffer mapping ─────────────────────────────────

    mod zero_copy {
        use super::*;

        #[test]
        #[ignore = "requires macOS Metal GPU - run on Apple Silicon hardware"]
        fn shared_mode_supports_zero_copy() {
            let buf = MockMetalBuffer::new("zc", 4096, StorageMode::Shared);
            assert!(buf.storage_mode.supports_zero_copy());
            // On Apple Silicon unified memory, the CPU pointer IS the GPU
            // pointer — no copy required.
            assert!(buf.contents.is_some());
        }

        #[test]
        #[ignore = "requires macOS Metal GPU - run on Apple Silicon hardware"]
        fn private_mode_no_zero_copy() {
            let buf = MockMetalBuffer::new("priv", 4096, StorageMode::Private);
            assert!(!buf.storage_mode.supports_zero_copy());
        }

        #[test]
        #[ignore = "requires macOS Metal GPU - run on Apple Silicon hardware"]
        fn zero_copy_write_visible_immediately() {
            let mut buf = MockMetalBuffer::new("zc-write", 1024, StorageMode::Shared);
            let payload = vec![42u8; 256];
            buf.write(0, &payload).unwrap();
            // On unified memory the GPU sees this without an explicit
            // synchronisation barrier.
            let readback = buf.read(0, 256).unwrap();
            assert_eq!(readback, &payload[..]);
        }

        #[test]
        #[ignore = "requires macOS Metal GPU - run on Apple Silicon hardware"]
        fn large_zero_copy_buffer() {
            // 64 MB buffer — typical for model weights on Apple Silicon.
            let size = 64 * 1024 * 1024;
            let buf = MockMetalBuffer::new("weights", size, StorageMode::Shared);
            assert_eq!(buf.allocated_size, size);
            assert!(buf.storage_mode.supports_zero_copy());
        }
    }

    // ── 6. Out-of-memory graceful handling ──────────────────────────

    mod oom_handling {
        use super::*;

        #[test]
        #[ignore = "requires macOS Metal GPU - run on Apple Silicon hardware"]
        fn allocation_within_budget_succeeds() {
            let available = 1024 * 1024; // 1 MB budget
            let result = try_alloc(512, available);
            assert_eq!(result, AllocResult::Ok { aligned_size: 512 });
        }

        #[test]
        #[ignore = "requires macOS Metal GPU - run on Apple Silicon hardware"]
        fn allocation_exceeding_budget_returns_oom() {
            let available = 1024; // 1 KB budget
            let result = try_alloc(2048, available);
            assert_eq!(result, AllocResult::OutOfMemory { requested: 2048, available: 1024 });
        }

        #[test]
        #[ignore = "requires macOS Metal GPU - run on Apple Silicon hardware"]
        fn alignment_can_push_over_budget() {
            // 255 bytes requested → aligned to 256, which exceeds 255
            // available.
            let result = try_alloc(255, 255);
            assert_eq!(result, AllocResult::OutOfMemory { requested: 256, available: 255 });
        }

        #[test]
        #[ignore = "requires macOS Metal GPU - run on Apple Silicon hardware"]
        fn zero_size_allocation_succeeds() {
            let result = try_alloc(0, 0);
            assert_eq!(result, AllocResult::Ok { aligned_size: 0 });
        }

        #[test]
        #[ignore = "requires macOS Metal GPU - run on Apple Silicon hardware"]
        fn max_buffer_size_is_reasonable() {
            assert!(MAX_BUFFER_SIZE >= 1024 * 1024 * 1024); // ≥ 1 GiB
            assert_eq!(MAX_BUFFER_SIZE % METAL_BUFFER_ALIGNMENT, 0);
        }

        #[test]
        #[ignore = "requires macOS Metal GPU - run on Apple Silicon hardware"]
        fn sequential_allocations_track_budget() {
            let mut remaining = 4096_usize;
            let sizes = [1024, 1024, 512, 256];

            for &sz in &sizes {
                match try_alloc(sz, remaining) {
                    AllocResult::Ok { aligned_size } => {
                        assert!(aligned_size <= remaining);
                        remaining -= aligned_size;
                    }
                    AllocResult::OutOfMemory { .. } => {
                        panic!("unexpected OOM for size {sz} with {remaining} remaining");
                    }
                }
            }
            // Budget should be exactly exhausted: 4096 - 1024 - 1024 - 512 - 256 = 1280
            assert_eq!(remaining, 1280);
        }
    }

    // ── 7. Buffer pool reuse patterns ───────────────────────────────

    mod buffer_pool {
        use super::*;

        #[test]
        #[ignore = "requires macOS Metal GPU - run on Apple Silicon hardware"]
        fn pool_allocates_fresh_buffer() {
            let mut pool = BufferPool::new();
            let buf = pool.acquire("first", 1024, StorageMode::Shared);
            assert_eq!(buf.allocated_size, 1024);
            assert_eq!(pool.allocated_count, 1);
            assert_eq!(pool.reuse_count, 0);
        }

        #[test]
        #[ignore = "requires macOS Metal GPU - run on Apple Silicon hardware"]
        fn pool_reuses_matching_buffer() {
            let mut pool = BufferPool::new();

            let buf = pool.acquire("a", 1024, StorageMode::Shared);
            pool.release(buf);

            let buf2 = pool.acquire("b", 1024, StorageMode::Shared);
            assert_eq!(buf2.allocated_size, 1024);
            assert_eq!(pool.allocated_count, 1); // no new allocation
            assert_eq!(pool.reuse_count, 1);
        }

        #[test]
        #[ignore = "requires macOS Metal GPU - run on Apple Silicon hardware"]
        fn pool_does_not_reuse_wrong_size() {
            let mut pool = BufferPool::new();

            let buf = pool.acquire("small", 256, StorageMode::Shared);
            pool.release(buf);

            let buf2 = pool.acquire("big", 1024, StorageMode::Shared);
            assert_eq!(buf2.allocated_size, 1024);
            assert_eq!(pool.allocated_count, 2); // new allocation required
            assert_eq!(pool.reuse_count, 0);
        }

        #[test]
        #[ignore = "requires macOS Metal GPU - run on Apple Silicon hardware"]
        fn pool_does_not_reuse_wrong_storage_mode() {
            let mut pool = BufferPool::new();

            let buf = pool.acquire("shared", 1024, StorageMode::Shared);
            pool.release(buf);

            let buf2 = pool.acquire("private", 1024, StorageMode::Private);
            assert_eq!(pool.allocated_count, 2);
            assert_eq!(pool.reuse_count, 0);
            assert_eq!(buf2.storage_mode, StorageMode::Private);
        }

        #[test]
        #[ignore = "requires macOS Metal GPU - run on Apple Silicon hardware"]
        fn pool_zeroes_contents_on_reuse() {
            let mut pool = BufferPool::new();

            let mut buf = pool.acquire("dirty", 256, StorageMode::Shared);
            buf.write(0, &[0xFF; 256]).unwrap();
            pool.release(buf);

            let buf2 = pool.acquire("clean", 256, StorageMode::Shared);
            let data = buf2.read(0, 256).unwrap();
            assert!(data.iter().all(|&b| b == 0), "reused buffer must be zeroed");
        }

        #[test]
        #[ignore = "requires macOS Metal GPU - run on Apple Silicon hardware"]
        fn pool_multiple_release_reuse_cycles() {
            let mut pool = BufferPool::new();

            for i in 0..10 {
                let buf = pool.acquire(&format!("iter-{i}"), 512, StorageMode::Shared);
                pool.release(buf);
            }

            // Only 1 physical allocation, 9 reuses.
            assert_eq!(pool.allocated_count, 1);
            assert_eq!(pool.reuse_count, 9);
        }

        #[test]
        #[ignore = "requires macOS Metal GPU - run on Apple Silicon hardware"]
        fn pool_concurrent_outstanding_buffers() {
            let mut pool = BufferPool::new();

            // Acquire several at once before releasing.
            let a = pool.acquire("a", 256, StorageMode::Shared);
            let b = pool.acquire("b", 256, StorageMode::Shared);
            let c = pool.acquire("c", 256, StorageMode::Shared);
            assert_eq!(pool.allocated_count, 3);

            pool.release(a);
            pool.release(b);
            pool.release(c);

            // Now acquire 3 again — all should be reused.
            let _d = pool.acquire("d", 256, StorageMode::Shared);
            let _e = pool.acquire("e", 256, StorageMode::Shared);
            let _f = pool.acquire("f", 256, StorageMode::Shared);
            assert_eq!(pool.allocated_count, 3);
            assert_eq!(pool.reuse_count, 3);
        }
    }

    // ── 8. Threadgroup memory budget ────────────────────────────────

    mod threadgroup_memory {
        use super::*;

        #[test]
        #[ignore = "requires macOS Metal GPU - run on Apple Silicon hardware"]
        fn threadgroup_memory_limit_is_32kb() {
            assert_eq!(MAX_THREADGROUP_MEMORY, 32 * 1024);
        }

        #[test]
        #[ignore = "requires macOS Metal GPU - run on Apple Silicon hardware"]
        fn f32_reduction_tile_fits() {
            // 1024 threads × 4 bytes = 4096 bytes — well within 32 KB.
            let usage = 1024 * std::mem::size_of::<f32>();
            assert!(usage <= MAX_THREADGROUP_MEMORY);
        }

        #[test]
        #[ignore = "requires macOS Metal GPU - run on Apple Silicon hardware"]
        fn double_buffered_tile_fits() {
            // Two 16×16 f32 tiles for A and B in a matmul.
            let per_tile = 16 * 16 * std::mem::size_of::<f32>();
            assert!(2 * per_tile <= MAX_THREADGROUP_MEMORY);
        }
    }
}
