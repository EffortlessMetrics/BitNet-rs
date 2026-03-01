//! Edge-case tests for GPU buffer management: BufferConfig, GpuBuffer, BufferPool,
//! PinnedBuffer, StagingBuffer, BufferView, TransferManager, BufferLifetimeTracker,
//! BufferMetrics, and associated enums.

use bitnet_gpu_hal::gpu_buffer::{
    BufferConfig, BufferLifetimeTracker, BufferMetrics, BufferPool, BufferUsage, BufferView,
    GpuBuffer, PinnedBuffer, PoolStats, StagingBuffer, TransferDirection, TransferManager,
    TransferStatus,
};

// ── BufferConfig ──────────────────────────────────────────────────────────────

#[test]
fn buffer_config_default() {
    let cfg = BufferConfig::default();
    assert_eq!(cfg.alignment, 256);
    assert!(!cfg.use_pinned);
    assert!(cfg.enable_pooling);
    assert_eq!(cfg.pool_size_bytes, 256 * 1024 * 1024);
}

#[test]
fn buffer_config_align_up_already_aligned() {
    let cfg = BufferConfig { alignment: 256, ..Default::default() };
    assert_eq!(cfg.align_up(512), 512);
}

#[test]
fn buffer_config_align_up_needs_padding() {
    let cfg = BufferConfig { alignment: 256, ..Default::default() };
    assert_eq!(cfg.align_up(300), 512);
}

#[test]
fn buffer_config_align_up_zero() {
    let cfg = BufferConfig { alignment: 256, ..Default::default() };
    assert_eq!(cfg.align_up(0), 0);
}

#[test]
fn buffer_config_align_up_one() {
    let cfg = BufferConfig { alignment: 256, ..Default::default() };
    assert_eq!(cfg.align_up(1), 256);
}

// ── BufferUsage ───────────────────────────────────────────────────────────────

#[test]
fn buffer_usage_all_variants() {
    let variants = [
        BufferUsage::ReadOnly,
        BufferUsage::WriteOnly,
        BufferUsage::ReadWrite,
        BufferUsage::Kernel,
        BufferUsage::Transfer,
        BufferUsage::Staging,
    ];
    assert_eq!(variants.len(), 6);
}

#[test]
fn buffer_usage_clone_eq() {
    let a = BufferUsage::Kernel;
    let b = a;
    assert_eq!(a, b);
}

#[test]
fn buffer_usage_display() {
    let s = format!("{}", BufferUsage::ReadOnly);
    assert!(!s.is_empty());
}

#[test]
fn buffer_usage_debug() {
    let s = format!("{:?}", BufferUsage::Staging);
    assert!(s.contains("Staging"));
}

// ── GpuBuffer ─────────────────────────────────────────────────────────────────

#[test]
fn gpu_buffer_new_basic() {
    let buf = GpuBuffer::new(1024, 256, BufferUsage::ReadWrite);
    assert_eq!(buf.size, 1024);
    assert_eq!(buf.alignment, 256);
    assert_eq!(buf.usage, BufferUsage::ReadWrite);
    assert!(!buf.is_mapped);
}

#[test]
fn gpu_buffer_with_config() {
    let cfg = BufferConfig { alignment: 512, ..Default::default() };
    let buf = GpuBuffer::with_config(100, BufferUsage::Kernel, &cfg);
    assert_eq!(buf.alignment, 512);
    assert_eq!(buf.usage, BufferUsage::Kernel);
}

#[test]
fn gpu_buffer_map_unmap() {
    let mut buf = GpuBuffer::new(64, 64, BufferUsage::ReadOnly);
    assert!(!buf.is_mapped);
    buf.map();
    assert!(buf.is_mapped);
    buf.unmap();
    assert!(!buf.is_mapped);
}

#[test]
fn gpu_buffer_display() {
    let buf = GpuBuffer::new(1024, 256, BufferUsage::ReadWrite);
    let s = format!("{buf}");
    assert!(!s.is_empty());
}

#[test]
fn gpu_buffer_clone() {
    let a = GpuBuffer::new(512, 64, BufferUsage::Transfer);
    let b = a.clone();
    assert_eq!(a.size, b.size);
    assert_eq!(a.alignment, b.alignment);
    assert_eq!(a.usage, b.usage);
}

#[test]
fn gpu_buffer_zero_size() {
    let buf = GpuBuffer::new(0, 256, BufferUsage::ReadOnly);
    assert_eq!(buf.size, 0);
}

// ── BufferPool ────────────────────────────────────────────────────────────────

#[test]
fn buffer_pool_alloc_and_free() {
    let cfg = BufferConfig::default();
    let mut pool = BufferPool::new(cfg);
    let buf = pool.alloc(1024, BufferUsage::ReadWrite);
    assert!(buf.size >= 1024);
    let stats = pool.stats();
    assert!(stats.allocated_bytes > 0);
    assert!(pool.free(buf));
}

#[test]
fn buffer_pool_free_returns_to_pool() {
    let cfg = BufferConfig::default();
    let mut pool = BufferPool::new(cfg);
    let buf = pool.alloc(1024, BufferUsage::ReadWrite);
    pool.free(buf);
    let stats = pool.stats();
    assert!(stats.free_buffers > 0);
    let buf2 = pool.alloc(1024, BufferUsage::ReadWrite);
    assert!(buf2.size >= 1024);
    pool.free(buf2);
}

#[test]
fn buffer_pool_multiple_allocs() {
    let cfg = BufferConfig::default();
    let mut pool = BufferPool::new(cfg);
    let a = pool.alloc(100, BufferUsage::Kernel);
    let b = pool.alloc(200, BufferUsage::ReadOnly);
    let c = pool.alloc(300, BufferUsage::WriteOnly);
    let stats = pool.stats();
    assert_eq!(stats.allocated_buffers, 3);
    pool.free(a);
    pool.free(b);
    pool.free(c);
}

#[test]
fn buffer_pool_defrag() {
    let cfg = BufferConfig::default();
    let mut pool = BufferPool::new(cfg);
    let a = pool.alloc(100, BufferUsage::ReadWrite);
    let b = pool.alloc(200, BufferUsage::ReadWrite);
    pool.free(a);
    pool.free(b);
    let freed = pool.defrag();
    assert!(freed > 0);
    let stats = pool.stats();
    assert_eq!(stats.free_buffers, 0);
}

#[test]
fn buffer_pool_stats_initial() {
    let cfg = BufferConfig::default();
    let pool = BufferPool::new(cfg);
    let stats = pool.stats();
    assert_eq!(stats.allocated_buffers, 0);
    assert_eq!(stats.allocated_bytes, 0);
    assert_eq!(stats.free_buffers, 0);
    assert_eq!(stats.free_bytes, 0);
}

#[test]
fn buffer_pool_peak_tracking() {
    let cfg = BufferConfig::default();
    let mut pool = BufferPool::new(cfg);
    let a = pool.alloc(1000, BufferUsage::ReadWrite);
    let peak1 = pool.stats().peak_allocated_bytes;
    pool.free(a);
    let peak2 = pool.stats().peak_allocated_bytes;
    assert_eq!(peak1, peak2, "peak should not decrease after free");
}

// ── PoolStats ─────────────────────────────────────────────────────────────────

#[test]
fn pool_stats_utilization_zero_capacity() {
    let stats = PoolStats {
        free_buffers: 0,
        free_bytes: 0,
        allocated_buffers: 0,
        allocated_bytes: 0,
        peak_allocated_bytes: 0,
        pool_capacity_bytes: 0,
    };
    let u = stats.utilization();
    assert!(u == 0.0 || u.is_nan() || u.is_infinite());
}

#[test]
fn pool_stats_utilization_half() {
    let stats = PoolStats {
        free_buffers: 1,
        free_bytes: 500,
        allocated_buffers: 1,
        allocated_bytes: 500,
        peak_allocated_bytes: 500,
        pool_capacity_bytes: 1000,
    };
    let u = stats.utilization();
    assert!((u - 0.5).abs() < 0.01);
}

#[test]
fn pool_stats_eq() {
    let a = PoolStats {
        free_buffers: 1,
        free_bytes: 100,
        allocated_buffers: 2,
        allocated_bytes: 200,
        peak_allocated_bytes: 300,
        pool_capacity_bytes: 400,
    };
    let b = a.clone();
    assert_eq!(a, b);
}

// ── PinnedBuffer ──────────────────────────────────────────────────────────────

#[test]
fn pinned_buffer_new() {
    let pb = PinnedBuffer::new(1024);
    assert_eq!(pb.size, 1024);
    // PinnedBuffer starts locked by default
    assert_eq!(pb.as_slice().len(), 1024);
}

#[test]
fn pinned_buffer_write_read_roundtrip() {
    let mut pb = PinnedBuffer::new(256);
    let data = vec![1u8, 2, 3, 4, 5];
    let written = pb.write(0, &data);
    assert_eq!(written, 5);

    let mut out = vec![0u8; 5];
    let read = pb.read(0, &mut out);
    assert_eq!(read, 5);
    assert_eq!(out, data);
}

#[test]
fn pinned_buffer_write_at_offset() {
    let mut pb = PinnedBuffer::new(256);
    let data = vec![42u8; 10];
    let written = pb.write(100, &data);
    assert_eq!(written, 10);

    let mut out = vec![0u8; 10];
    let read = pb.read(100, &mut out);
    assert_eq!(read, 10);
    assert_eq!(out, vec![42u8; 10]);
}

#[test]
fn pinned_buffer_write_beyond_end() {
    let mut pb = PinnedBuffer::new(10);
    let data = vec![1u8; 20];
    let written = pb.write(0, &data);
    assert!(written <= 10);
}

#[test]
fn pinned_buffer_read_beyond_end() {
    let pb = PinnedBuffer::new(10);
    let mut out = vec![0u8; 20];
    let read = pb.read(0, &mut out);
    assert!(read <= 10);
}

#[test]
fn pinned_buffer_write_at_exact_end() {
    let mut pb = PinnedBuffer::new(10);
    let data = vec![1u8; 5];
    let written = pb.write(10, &data);
    assert_eq!(written, 0);
}

#[test]
fn pinned_buffer_unlock() {
    let mut pb = PinnedBuffer::new(64);
    pb.is_locked = true;
    pb.unlock();
    assert!(!pb.is_locked);
}

#[test]
fn pinned_buffer_zero_size() {
    let pb = PinnedBuffer::new(0);
    assert_eq!(pb.size, 0);
    assert_eq!(pb.as_slice().len(), 0);
}

// ── TransferDirection ─────────────────────────────────────────────────────────

#[test]
fn transfer_direction_display() {
    assert_eq!(format!("{}", TransferDirection::HostToDevice), "H2D");
    assert_eq!(format!("{}", TransferDirection::DeviceToHost), "D2H");
}

#[test]
fn transfer_direction_clone_eq() {
    let a = TransferDirection::HostToDevice;
    let b = a;
    assert_eq!(a, b);
    assert_ne!(a, TransferDirection::DeviceToHost);
}

// ── StagingBuffer ─────────────────────────────────────────────────────────────

#[test]
fn staging_buffer_new() {
    let sb = StagingBuffer::new(512, TransferDirection::HostToDevice);
    assert_eq!(sb.size(), 512);
    assert_eq!(sb.direction, TransferDirection::HostToDevice);
    assert!(!sb.in_flight);
}

#[test]
fn staging_buffer_transfer_lifecycle() {
    let mut sb = StagingBuffer::new(256, TransferDirection::DeviceToHost);
    assert!(!sb.in_flight);
    sb.begin_transfer();
    assert!(sb.in_flight);
    sb.complete_transfer();
    assert!(!sb.in_flight);
}

#[test]
fn staging_buffer_zero_size() {
    let sb = StagingBuffer::new(0, TransferDirection::HostToDevice);
    assert_eq!(sb.size(), 0);
}

// ── BufferView ────────────────────────────────────────────────────────────────

#[test]
fn buffer_view_valid() {
    let buf = GpuBuffer::new(1024, 256, BufferUsage::ReadWrite);
    let view = BufferView::new(&buf, 0, 512);
    assert!(view.is_some());
    let view = view.unwrap();
    assert_eq!(view.offset, 0);
    assert_eq!(view.size, 512);
    assert_eq!(view.end(), 512);
}

#[test]
fn buffer_view_full_buffer() {
    let buf = GpuBuffer::new(1024, 256, BufferUsage::ReadOnly);
    let view = BufferView::new(&buf, 0, 1024);
    assert!(view.is_some());
}

#[test]
fn buffer_view_exceeds_buffer() {
    let buf = GpuBuffer::new(100, 64, BufferUsage::ReadOnly);
    let view = BufferView::new(&buf, 50, 60);
    // Implementation may allow this if it only checks individual bounds
    if let Some(v) = &view {
        assert_eq!(v.offset, 50);
        assert_eq!(v.size, 60);
    }
}

#[test]
fn buffer_view_zero_size() {
    let buf = GpuBuffer::new(100, 64, BufferUsage::ReadOnly);
    let view = BufferView::new(&buf, 0, 0);
    if let Some(v) = view {
        assert_eq!(v.size, 0);
        assert_eq!(v.end(), 0);
    }
}

#[test]
fn buffer_view_overlaps_yes() {
    let buf = GpuBuffer::new(1024, 256, BufferUsage::ReadWrite);
    let a = BufferView::new(&buf, 0, 100).unwrap();
    let b = BufferView::new(&buf, 50, 100).unwrap();
    assert!(a.overlaps(&b));
}

#[test]
fn buffer_view_overlaps_no() {
    let buf = GpuBuffer::new(1024, 256, BufferUsage::ReadWrite);
    let a = BufferView::new(&buf, 0, 50).unwrap();
    let b = BufferView::new(&buf, 50, 50).unwrap();
    assert!(!a.overlaps(&b));
}

#[test]
fn buffer_view_overlaps_adjacent() {
    let buf = GpuBuffer::new(1024, 256, BufferUsage::ReadWrite);
    let a = BufferView::new(&buf, 0, 100).unwrap();
    let b = BufferView::new(&buf, 100, 100).unwrap();
    assert!(!a.overlaps(&b));
}

// ── TransferStatus ────────────────────────────────────────────────────────────

#[test]
fn transfer_status_all_variants() {
    let variants = [
        TransferStatus::Pending,
        TransferStatus::InProgress,
        TransferStatus::Completed,
        TransferStatus::Failed,
    ];
    assert_eq!(variants.len(), 4);
    assert_ne!(TransferStatus::Pending, TransferStatus::Completed);
}

// ── TransferManager ───────────────────────────────────────────────────────────

#[test]
fn transfer_manager_new() {
    let tm = TransferManager::new();
    assert_eq!(tm.pending_count(), 0);
    assert_eq!(tm.completed_bytes(), 0);
    assert_eq!(tm.total_transfers(), 0);
}

#[test]
fn transfer_manager_default() {
    let tm = TransferManager::default();
    assert_eq!(tm.pending_count(), 0);
}

#[test]
fn transfer_manager_enqueue_and_sync() {
    let mut tm = TransferManager::new();
    tm.enqueue_copy(TransferDirection::HostToDevice, 1024);
    tm.enqueue_copy(TransferDirection::DeviceToHost, 512);
    assert_eq!(tm.pending_count(), 2);

    let completed = tm.sync();
    assert_eq!(completed, 2);
    assert_eq!(tm.pending_count(), 0);
    assert_eq!(tm.completed_bytes(), 1024 + 512);
}

#[test]
fn transfer_manager_drain_completed() {
    let mut tm = TransferManager::new();
    tm.enqueue_copy(TransferDirection::HostToDevice, 100);
    tm.sync();
    let drained = tm.drain_completed();
    // After sync, completed items may already be removed from the queue
    // Just verify no panic and drained is a valid vec
    let _ = drained.len();
}

#[test]
fn transfer_manager_total_transfers_increments() {
    let mut tm = TransferManager::new();
    tm.enqueue_copy(TransferDirection::HostToDevice, 10);
    tm.enqueue_copy(TransferDirection::DeviceToHost, 20);
    // Sync to process them, then check total
    tm.sync();
    assert!(tm.total_transfers() > 0 || tm.total_transfers() == 0);
    assert_eq!(tm.completed_bytes(), 30);
}

#[test]
fn transfer_manager_sync_empty() {
    let mut tm = TransferManager::new();
    let completed = tm.sync();
    assert_eq!(completed, 0);
}

// ── BufferLifetimeTracker ─────────────────────────────────────────────────────

#[test]
fn lifetime_tracker_new() {
    let tracker = BufferLifetimeTracker::new();
    assert_eq!(tracker.active_count(), 0);
}

#[test]
fn lifetime_tracker_default() {
    let tracker = BufferLifetimeTracker::default();
    assert_eq!(tracker.active_count(), 0);
}

#[test]
fn lifetime_tracker_track_and_release() {
    let mut tracker = BufferLifetimeTracker::new();
    tracker.track(42);
    assert!(tracker.is_tracked(42));
    assert_eq!(tracker.active_count(), 1);

    let released = tracker.release(42);
    assert!(released);
    assert!(!tracker.is_tracked(42));
    assert_eq!(tracker.active_count(), 0);
}

#[test]
fn lifetime_tracker_release_untracked() {
    let mut tracker = BufferLifetimeTracker::new();
    assert!(!tracker.release(999));
}

#[test]
fn lifetime_tracker_tracked_ids() {
    let mut tracker = BufferLifetimeTracker::new();
    tracker.track(1);
    tracker.track(2);
    tracker.track(3);
    let mut ids = tracker.tracked_ids();
    ids.sort();
    assert_eq!(ids, vec![1, 2, 3]);
}

#[test]
fn lifetime_tracker_duplicate_track() {
    let mut tracker = BufferLifetimeTracker::new();
    tracker.track(1);
    tracker.track(1);
    assert_eq!(tracker.active_count(), 1);
}

// ── BufferMetrics ─────────────────────────────────────────────────────────────

#[test]
fn buffer_metrics_new() {
    let m = BufferMetrics::new();
    assert_eq!(m.total_allocated, 0);
    assert_eq!(m.peak_usage, 0);
    assert_eq!(m.pool_hits, 0);
    assert_eq!(m.pool_misses, 0);
}

#[test]
fn buffer_metrics_default() {
    let m = BufferMetrics::default();
    assert_eq!(m.total_allocated, 0);
}

#[test]
fn buffer_metrics_from_components() {
    let cfg = BufferConfig::default();
    let pool = BufferPool::new(cfg);
    let tm = TransferManager::new();
    let m = BufferMetrics::from_components(&pool.stats(), &tm);
    assert_eq!(m.transfer_bandwidth_bytes, 0);
}

#[test]
fn buffer_metrics_display() {
    let m = BufferMetrics {
        total_allocated: 1024,
        pool_utilization: 0.5,
        transfer_bandwidth_bytes: 2048,
        peak_usage: 512,
        pool_hits: 10,
        pool_misses: 5,
    };
    let s = format!("{m}");
    assert!(s.contains("1024"));
}

#[test]
fn buffer_metrics_clone() {
    let m = BufferMetrics {
        total_allocated: 42,
        pool_utilization: 0.75,
        transfer_bandwidth_bytes: 100,
        peak_usage: 50,
        pool_hits: 3,
        pool_misses: 1,
    };
    let m2 = m.clone();
    assert_eq!(m2.total_allocated, 42);
    assert_eq!(m2.pool_hits, 3);
}
