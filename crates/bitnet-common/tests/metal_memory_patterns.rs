//! Metal / Apple Silicon memory management pattern tests.
//!
//! These tests validate memory allocation strategies, alignment requirements,
//! buffer reuse, and RAII cleanup patterns that underpin Metal/Apple Silicon
//! unified-memory inference — all **without** requiring actual Metal hardware.

use bitnet_common::memory_pool::{PoolStats, TensorPool};
use bitnet_common::types::Device;
use std::thread;

// ── Constants modelling Metal constraints ────────────────────────────

/// Metal requires 256-byte buffer alignment for `MTLBuffer` storage.
const METAL_BUFFER_ALIGNMENT: usize = 256;

/// Page size on Apple Silicon (16 KiB).
const APPLE_SILICON_PAGE_SIZE: usize = 16384;

/// Minimum `TensorPool` bucket size.
const MIN_BUCKET: usize = 64;

// ── Helpers ─────────────────────────────────────────────────────────

/// Round `size` up to the nearest multiple of `alignment`.
fn align_up(size: usize, alignment: usize) -> usize {
    assert!(alignment.is_power_of_two(), "alignment must be power-of-two");
    (size + alignment - 1) & !(alignment - 1)
}

/// Simulate a device memory report.
struct DeviceMemoryReport {
    #[allow(dead_code)]
    device: Device,
    total_bytes: usize,
    available_bytes: usize,
}

impl DeviceMemoryReport {
    fn utilization(&self) -> f64 {
        if self.total_bytes == 0 {
            return 0.0;
        }
        1.0 - (self.available_bytes as f64 / self.total_bytes as f64)
    }
}

// ═══════════════════════════════════════════════════════════════════
// 1. Unified Memory Model Validation
// ═══════════════════════════════════════════════════════════════════

#[test]
fn metal_device_variant_exists() {
    let dev = Device::Metal;
    assert!(!dev.is_cpu());
    assert!(!dev.is_cuda());
    assert!(!dev.is_opencl());
    assert!(!dev.is_hip());
    assert!(!dev.is_npu());
}

#[test]
fn metal_device_debug_format() {
    let dev = Device::Metal;
    let dbg = format!("{dev:?}");
    assert!(dbg.contains("Metal"), "debug should mention Metal: {dbg}");
}

#[test]
fn metal_device_serialization_roundtrip() {
    let dev = Device::Metal;
    let json = serde_json::to_string(&dev).unwrap();
    let back: Device = serde_json::from_str(&json).unwrap();
    assert_eq!(dev, back);
}

#[test]
fn metal_to_candle_does_not_panic() {
    // On non-macOS or without real Metal, this should fall back to Cpu.
    let dev = Device::Metal;
    let candle = dev.to_candle().unwrap();
    // Regardless of platform the call must succeed.
    let _ = format!("{candle:?}");
}

#[test]
fn unified_memory_same_pool_for_cpu_and_device() {
    // Unified memory means one pool serves both CPU and GPU allocations.
    let pool = TensorPool::new(1 << 20);

    let cpu_buf = pool.allocate(1024);
    drop(cpu_buf);

    // "Device" allocation from the same pool reuses the buffer.
    let device_buf = pool.allocate(1024);
    let stats = pool.stats();
    assert_eq!(stats.hits, 1, "unified pool should reuse across CPU/device");
    drop(device_buf);
}

#[test]
fn unified_memory_zero_copy_read_after_write() {
    let pool = TensorPool::new(4096);
    let mut buf = pool.allocate(256);

    // Simulate CPU write.
    buf[0] = 0xAB;
    buf[255] = 0xCD;

    // Unified memory: GPU can read the same bytes without a copy.
    assert_eq!(buf[0], 0xAB);
    assert_eq!(buf[255], 0xCD);
}

// ═══════════════════════════════════════════════════════════════════
// 2. Buffer Alignment (256-byte for Metal)
// ═══════════════════════════════════════════════════════════════════

#[test]
fn align_up_to_metal_boundary() {
    assert_eq!(align_up(1, METAL_BUFFER_ALIGNMENT), 256);
    assert_eq!(align_up(256, METAL_BUFFER_ALIGNMENT), 256);
    assert_eq!(align_up(257, METAL_BUFFER_ALIGNMENT), 512);
    assert_eq!(align_up(0, METAL_BUFFER_ALIGNMENT), 0);
}

#[test]
fn pool_bucket_satisfies_metal_alignment_for_large_allocs() {
    let pool = TensorPool::new(1 << 20);
    // Any request >= 256 lands in a power-of-two bucket that is a multiple
    // of 256, because 256 is itself a power of two.
    for &req in &[256, 300, 512, 1000, 4096] {
        let buf = pool.allocate(req);
        assert!(
            buf.len().is_multiple_of(METAL_BUFFER_ALIGNMENT),
            "bucket size {} not aligned to {METAL_BUFFER_ALIGNMENT} (req={req})",
            buf.len()
        );
    }
}

#[test]
fn small_allocation_below_metal_alignment() {
    let pool = TensorPool::new(4096);
    let buf = pool.allocate(100);
    // Bucket is 128, which is < 256. Caller must pad externally for Metal.
    assert_eq!(buf.len(), 128);
    assert!(buf.len() < METAL_BUFFER_ALIGNMENT);
}

#[test]
fn align_up_page_boundary() {
    let size = 17000;
    let aligned = align_up(size, APPLE_SILICON_PAGE_SIZE);
    assert_eq!(aligned, 2 * APPLE_SILICON_PAGE_SIZE);
    assert!(aligned.is_multiple_of(APPLE_SILICON_PAGE_SIZE));
}

// ═══════════════════════════════════════════════════════════════════
// 3. Memory Pool Allocation Patterns (various sizes)
// ═══════════════════════════════════════════════════════════════════

#[test]
fn pool_tiny_allocation() {
    let pool = TensorPool::new(4096);
    let buf = pool.allocate(1);
    assert_eq!(buf.len(), MIN_BUCKET);
}

#[test]
fn pool_medium_allocation() {
    let pool = TensorPool::new(1 << 20);
    let buf = pool.allocate(50_000);
    assert!(buf.len() >= 50_000);
    assert!(buf.len().is_power_of_two());
}

#[test]
fn pool_large_allocation_1mb() {
    let pool = TensorPool::new(4 << 20);
    let buf = pool.allocate(1 << 20);
    assert_eq!(buf.len(), 1 << 20);
}

#[test]
fn pool_exact_power_of_two_sizes() {
    let pool = TensorPool::new(1 << 24);
    for exp in 6..20 {
        let size = 1usize << exp;
        let buf = pool.allocate(size);
        assert_eq!(buf.len(), size, "exact p2 {size} should not round up");
        drop(buf);
    }
}

#[test]
fn pool_mixed_sizes_independent_buckets() {
    let pool = TensorPool::new(1 << 20);
    let sizes: Vec<usize> = vec![64, 128, 256, 512, 1024];
    // Allocate then free all sizes.
    for &s in &sizes {
        let buf = pool.allocate(s);
        drop(buf);
    }
    // Re-allocate each: all should be hits.
    for &s in &sizes {
        let _buf = pool.allocate(s);
    }
    let stats = pool.stats();
    assert_eq!(stats.hits, sizes.len() as u64);
    assert_eq!(stats.misses, sizes.len() as u64);
}

// ═══════════════════════════════════════════════════════════════════
// 4. Power-of-Two Bucket Strategy
// ═══════════════════════════════════════════════════════════════════

#[test]
fn bucket_rounding_series() {
    let pool = TensorPool::new(1 << 20);
    let cases: Vec<(usize, usize)> = vec![
        (0, 64),
        (1, 64),
        (63, 64),
        (64, 64),
        (65, 128),
        (128, 128),
        (129, 256),
        (255, 256),
        (256, 256),
        (257, 512),
        (1000, 1024),
        (1025, 2048),
    ];
    for (req, expected) in cases {
        let buf = pool.allocate(req);
        assert_eq!(buf.len(), expected, "allocate({req}) → expected {expected}, got {}", buf.len());
        drop(buf);
    }
}

#[test]
fn all_buckets_are_power_of_two() {
    let pool = TensorPool::new(1 << 20);
    for req in [33, 77, 150, 300, 999, 5000, 100_000] {
        let buf = pool.allocate(req);
        assert!(buf.len().is_power_of_two(), "bucket {req} → {} not p2", buf.len());
        drop(buf);
    }
}

#[test]
fn bucket_fragmentation_bounded() {
    // Internal fragmentation is at most 50 % for any request > MIN_BUCKET.
    let pool = TensorPool::new(1 << 20);
    for req in (65..2048).step_by(37) {
        let buf = pool.allocate(req);
        let waste = buf.len() - req;
        assert!(
            waste < req,
            "fragmentation too high: req={req}, bucket={}, waste={waste}",
            buf.len()
        );
        drop(buf);
    }
}

// ═══════════════════════════════════════════════════════════════════
// 5. Buffer Reuse and RAII Cleanup
// ═══════════════════════════════════════════════════════════════════

#[test]
fn raii_drop_returns_buffer_to_pool() {
    let pool = TensorPool::new(4096);
    {
        let _buf = pool.allocate(256);
        assert_eq!(pool.stats().active_bytes, 256);
    }
    assert_eq!(pool.stats().active_bytes, 0);
    assert_eq!(pool.stats().pooled_bytes, 256);
}

#[test]
fn raii_multiple_buffers_returned_on_scope_exit() {
    let pool = TensorPool::new(1 << 20);
    {
        let _a = pool.allocate(128);
        let _b = pool.allocate(256);
        let _c = pool.allocate(512);
        assert_eq!(pool.stats().active_bytes, 128 + 256 + 512);
    }
    let stats = pool.stats();
    assert_eq!(stats.active_bytes, 0);
    assert_eq!(stats.pooled_bytes, 128 + 256 + 512);
}

#[test]
fn reuse_cycle_10_iterations() {
    let pool = TensorPool::new(4096);
    for _ in 0..10 {
        let buf = pool.allocate(256);
        drop(buf);
    }
    let stats = pool.stats();
    assert_eq!(stats.misses, 1);
    assert_eq!(stats.hits, 9);
}

#[test]
fn recycled_buffer_is_zeroed() {
    let pool = TensorPool::new(4096);
    let mut buf = pool.allocate(64);
    buf.iter_mut().for_each(|b| *b = 0xFF);
    drop(buf);

    let buf2 = pool.allocate(64);
    assert!(buf2.iter().all(|&b| b == 0), "recycled buffer must be zeroed");
}

#[test]
fn f32_reinterpret_after_reuse() {
    let pool = TensorPool::new(4096);
    let mut buf = pool.allocate(256);
    {
        let floats = buf.as_f32_mut_slice();
        floats[0] = 3.14;
    }
    drop(buf);

    let buf2 = pool.allocate(256);
    // Buffer was zeroed on recycle.
    assert_eq!(buf2.as_f32_slice()[0], 0.0);
}

// ═══════════════════════════════════════════════════════════════════
// 6. Zero-Copy Memory Mapping Patterns
// ═══════════════════════════════════════════════════════════════════

#[test]
fn zero_copy_deref_read() {
    let pool = TensorPool::new(4096);
    let mut buf = pool.allocate(128);
    buf[42] = 0xAA;

    // Deref gives &[u8] — zero-copy read access.
    let slice: &[u8] = &buf;
    assert_eq!(slice[42], 0xAA);
}

#[test]
fn zero_copy_deref_mut_write() {
    let pool = TensorPool::new(4096);
    let mut buf = pool.allocate(128);

    // DerefMut gives &mut [u8] — zero-copy write.
    let slice: &mut [u8] = &mut buf;
    slice[0] = 0x01;
    assert_eq!(buf[0], 0x01);
}

#[test]
fn zero_copy_f32_view() {
    let pool = TensorPool::new(4096);
    let mut buf = pool.allocate(64);
    let floats = buf.as_f32_mut_slice();
    assert_eq!(floats.len(), 64 / 4);
    floats[0] = 1.0;
    floats[15] = -1.0;

    let read = buf.as_f32_slice();
    assert_eq!(read[0], 1.0);
    assert_eq!(read[15], -1.0);
}

#[test]
fn zero_copy_no_allocation_on_view() {
    let pool = TensorPool::new(4096);
    let buf = pool.allocate(256);
    let before = pool.stats().total_allocations();

    // Taking a slice view should not cause another pool allocation.
    let _slice: &[u8] = &buf;
    let _f32s = buf.as_f32_slice();

    assert_eq!(pool.stats().total_allocations(), before);
}

// ═══════════════════════════════════════════════════════════════════
// 7. Memory Pressure Handling
// ═══════════════════════════════════════════════════════════════════

#[test]
fn pool_evicts_when_capacity_exceeded() {
    // Pool can hold 256 bytes.
    let pool = TensorPool::new(256);
    let a = pool.allocate(128);
    let b = pool.allocate(256);
    drop(a); // 128 bytes pooled
    drop(b); // 256 bytes would exceed 256 cap → evicted

    let stats = pool.stats();
    assert_eq!(stats.pooled_bytes, 128, "only the first buffer fits");
}

#[test]
fn pool_clear_under_pressure() {
    let pool = TensorPool::new(1 << 20);
    for _ in 0..100 {
        let buf = pool.allocate(4096);
        drop(buf);
    }
    assert!(pool.stats().pooled_bytes > 0);

    pool.clear();
    assert_eq!(pool.stats().pooled_bytes, 0);
}

#[test]
fn allocation_succeeds_after_clear() {
    let pool = TensorPool::new(4096);
    let buf = pool.allocate(128);
    drop(buf);
    pool.clear();

    // Next allocation is a miss — no pooled buffers.
    let _buf2 = pool.allocate(128);
    let stats = pool.stats();
    assert_eq!(stats.misses, 2);
    assert_eq!(stats.hits, 0);
}

#[test]
fn pressure_multiple_concurrent_threads() {
    let pool = TensorPool::new(4096);
    let handles: Vec<_> = (0..4)
        .map(|_| {
            let p = pool.clone();
            thread::spawn(move || {
                for _ in 0..50 {
                    let mut buf = p.allocate(256);
                    buf[0] = 1;
                    drop(buf);
                }
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }

    let stats = pool.stats();
    assert_eq!(stats.total_allocations(), 200);
    assert_eq!(stats.active_bytes, 0);
}

#[test]
fn zero_capacity_pool_never_caches() {
    let pool = TensorPool::new(0);
    let buf = pool.allocate(64);
    drop(buf);

    let _buf2 = pool.allocate(64);
    let stats = pool.stats();
    assert_eq!(stats.hits, 0, "zero-cap pool should never reuse");
    assert_eq!(stats.misses, 2);
    assert_eq!(stats.pooled_bytes, 0);
}

// ═══════════════════════════════════════════════════════════════════
// 8. Device Memory Reporting
// ═══════════════════════════════════════════════════════════════════

#[test]
fn memory_report_metal_device() {
    let report = DeviceMemoryReport {
        device: Device::Metal,
        total_bytes: 16 * 1024 * 1024 * 1024, // 16 GiB
        available_bytes: 12 * 1024 * 1024 * 1024,
    };
    assert!(report.utilization() > 0.0 && report.utilization() < 1.0);
    assert!((report.utilization() - 0.25).abs() < 0.01);
}

#[test]
fn memory_report_full_utilization() {
    let report =
        DeviceMemoryReport { device: Device::Metal, total_bytes: 1024, available_bytes: 0 };
    assert!((report.utilization() - 1.0).abs() < f64::EPSILON);
}

#[test]
fn memory_report_zero_total() {
    let report = DeviceMemoryReport { device: Device::Metal, total_bytes: 0, available_bytes: 0 };
    assert_eq!(report.utilization(), 0.0);
}

#[test]
fn pool_stats_reflect_active_and_pooled() {
    let pool = TensorPool::new(1 << 20);
    let a = pool.allocate(256);
    let b = pool.allocate(512);

    let stats = pool.stats();
    assert_eq!(stats.active_bytes, 256 + 512);
    assert_eq!(stats.pooled_bytes, 0);

    drop(a);
    let stats = pool.stats();
    assert_eq!(stats.active_bytes, 512);
    assert_eq!(stats.pooled_bytes, 256);

    drop(b);
    let stats = pool.stats();
    assert_eq!(stats.active_bytes, 0);
    assert_eq!(stats.pooled_bytes, 256 + 512);
}

#[test]
fn pool_stats_default_is_empty() {
    let stats = PoolStats::default();
    assert_eq!(stats.hits, 0);
    assert_eq!(stats.misses, 0);
    assert_eq!(stats.pooled_bytes, 0);
    assert_eq!(stats.active_bytes, 0);
    assert_eq!(stats.total_allocations(), 0);
}

#[test]
fn shared_pool_stats_consistent_across_clones() {
    let pool = TensorPool::new(1 << 20);
    let clone = pool.clone();

    let _buf = pool.allocate(256);
    // Stats visible via both handles.
    assert_eq!(pool.stats().misses, 1);
    assert_eq!(clone.stats().misses, 1);
}

// ═══════════════════════════════════════════════════════════════════
// Additional coverage: Device ordering, metal page multiples
// ═══════════════════════════════════════════════════════════════════

#[test]
fn device_metal_ordering_is_consistent() {
    let cpu = Device::Cpu;
    let metal = Device::Metal;
    // Device derives Ord — just verify no panic and determinism.
    let cmp1 = cpu.cmp(&metal);
    let cmp2 = cpu.cmp(&metal);
    assert_eq!(cmp1, cmp2);
}

#[test]
fn metal_aligned_sizes_for_typical_tensors() {
    // Typical hidden sizes in transformer models.
    for &hidden in &[768, 1024, 2048, 4096] {
        let bytes = hidden * 4; // f32
        let aligned = align_up(bytes, METAL_BUFFER_ALIGNMENT);
        assert!(aligned.is_multiple_of(METAL_BUFFER_ALIGNMENT));
        assert!(aligned >= bytes);
    }
}

#[test]
fn pool_allocate_page_aligned_request() {
    let pool = TensorPool::new(1 << 20);
    let buf = pool.allocate(APPLE_SILICON_PAGE_SIZE);
    assert_eq!(buf.len(), APPLE_SILICON_PAGE_SIZE);
    assert!(buf.len().is_power_of_two());
}
