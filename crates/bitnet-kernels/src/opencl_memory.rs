//! OpenCL memory transfer tracking and optimization.
//!
//! Provides memory transfer recording, GPU memory budget management,
//! buffer pooling with size bucketing, and optimization suggestions
//! for OpenCL workloads targeting Intel Arc GPUs.

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Core enums
// ---------------------------------------------------------------------------

/// Location of a memory region.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MemoryLocation {
    Host,
    Device,
    Mapped,
}

/// Direction of a memory transfer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TransferDirection {
    HostToDevice,
    DeviceToHost,
    DeviceToDevice,
}

/// Strategy for allocating device buffers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BufferAllocationStrategy {
    /// Allocate on first write.
    AllocateOnWrite,
    /// Pre-allocate before any transfer.
    PreAllocate,
    /// Use pinned (page-locked) host memory for DMA.
    PinnedMemory,
    /// Use zero-copy shared memory (e.g. integrated GPU).
    ZeroCopy,
}

// ---------------------------------------------------------------------------
// MemoryRegion
// ---------------------------------------------------------------------------

/// Tracks a single allocated buffer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemoryRegion {
    pub id: u64,
    pub size: usize,
    pub location: MemoryLocation,
}

impl MemoryRegion {
    pub fn new(id: u64, size: usize, location: MemoryLocation) -> Self {
        Self { id, size, location }
    }
}

// ---------------------------------------------------------------------------
// TransferRecord / TransferStats
// ---------------------------------------------------------------------------

/// A single recorded transfer event.
#[derive(Debug, Clone)]
pub struct TransferRecord {
    pub direction: TransferDirection,
    pub bytes: usize,
    pub timestamp_ns: u64,
    pub duration_ns: u64,
}

impl TransferRecord {
    /// Bandwidth in GB/s for this transfer, or 0.0 if duration is zero.
    pub fn bandwidth_gbps(&self) -> f64 {
        if self.duration_ns == 0 {
            return 0.0;
        }
        (self.bytes as f64) / (self.duration_ns as f64) // bytes/ns == GB/s
    }
}

/// Aggregate statistics over a set of transfers.
#[derive(Debug, Clone, Default)]
pub struct TransferStats {
    pub total_bytes: usize,
    pub count: usize,
    pub total_duration_ns: u64,
    pub peak_bandwidth_gbps: f64,
}

impl TransferStats {
    /// Average bandwidth in GB/s, or 0.0 when empty.
    pub fn avg_bandwidth_gbps(&self) -> f64 {
        if self.total_duration_ns == 0 {
            return 0.0;
        }
        (self.total_bytes as f64) / (self.total_duration_ns as f64)
    }
}

// ---------------------------------------------------------------------------
// MemoryTransferTracker
// ---------------------------------------------------------------------------

/// Tracks all memory transfers and derives statistics.
#[derive(Debug, Default)]
pub struct MemoryTransferTracker {
    records: Vec<TransferRecord>,
    next_timestamp: u64,
}

impl MemoryTransferTracker {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a transfer event.
    pub fn record_transfer(
        &mut self,
        direction: TransferDirection,
        bytes: usize,
        duration_ns: u64,
    ) {
        let timestamp_ns = self.next_timestamp;
        self.next_timestamp = timestamp_ns.saturating_add(duration_ns);
        let record = TransferRecord { direction, bytes, timestamp_ns, duration_ns };
        self.records.push(record);
    }

    pub fn total_bytes_transferred(&self) -> usize {
        self.records.iter().map(|r| r.bytes).sum()
    }

    pub fn transfer_count(&self) -> usize {
        self.records.len()
    }

    /// Stats filtered by direction.
    fn stats_for(&self, dir: TransferDirection) -> TransferStats {
        let mut stats = TransferStats::default();
        for r in &self.records {
            if r.direction == dir {
                stats.total_bytes += r.bytes;
                stats.count += 1;
                stats.total_duration_ns += r.duration_ns;
                let bw = r.bandwidth_gbps();
                if bw > stats.peak_bandwidth_gbps {
                    stats.peak_bandwidth_gbps = bw;
                }
            }
        }
        stats
    }

    pub fn host_to_device_stats(&self) -> TransferStats {
        self.stats_for(TransferDirection::HostToDevice)
    }

    pub fn device_to_host_stats(&self) -> TransferStats {
        self.stats_for(TransferDirection::DeviceToHost)
    }

    pub fn device_to_device_stats(&self) -> TransferStats {
        self.stats_for(TransferDirection::DeviceToDevice)
    }

    /// Average bandwidth across all recorded transfers (GB/s).
    pub fn average_bandwidth_gbps(&self) -> f64 {
        let total_dur: u64 = self.records.iter().map(|r| r.duration_ns).sum();
        if total_dur == 0 {
            return 0.0;
        }
        (self.total_bytes_transferred() as f64) / (total_dur as f64)
    }

    /// Peak bandwidth across any single transfer (GB/s).
    pub fn peak_bandwidth_gbps(&self) -> f64 {
        self.records.iter().map(|r| r.bandwidth_gbps()).fold(0.0_f64, f64::max)
    }

    /// Suggest optimizations based on recorded transfer patterns.
    pub fn suggest_optimizations(&self) -> Vec<String> {
        let mut suggestions = Vec::new();

        // Detect many small transfers (< 4 KB)
        let small_count = self.records.iter().filter(|r| r.bytes < 4096).count();
        if small_count > 10 {
            suggestions.push(format!(
                "Batch {} small transfers (<4 KB) into fewer large transfers",
                small_count
            ));
        }

        // Detect low bandwidth utilisation vs A770 theoretical peak (560 GB/s)
        let avg = self.average_bandwidth_gbps();
        if !self.records.is_empty() && avg > 0.0 && avg < 10.0 {
            suggestions
                .push("Average bandwidth <10 GB/s — consider pinned memory or zero-copy".into());
        }

        // Detect excessive H→D traffic
        let h2d = self.host_to_device_stats();
        let d2h = self.device_to_host_stats();
        if h2d.count > 0 && d2h.count > 0 && h2d.total_bytes > d2h.total_bytes * 4 {
            suggestions.push(
                "Host-to-device traffic is 4×+ device-to-host — keep data on device longer".into(),
            );
        }

        // Recommend pre-allocation when many allocations detected
        if self.records.len() > 100 {
            suggestions
                .push("Over 100 transfers recorded — consider pre-allocating buffers".into());
        }

        // Detect redundant round-trips (H→D immediately followed by D→H of same size)
        let mut prev: Option<&TransferRecord> = None;
        let mut round_trips = 0usize;
        for r in &self.records {
            if let Some(p) = prev
                && p.direction == TransferDirection::HostToDevice
                && r.direction == TransferDirection::DeviceToHost
                && p.bytes == r.bytes
            {
                round_trips += 1;
            }
            prev = Some(r);
        }
        if round_trips > 2 {
            suggestions.push(format!(
                "Detected {} likely redundant H2D→D2H round-trips of same size",
                round_trips
            ));
        }

        suggestions
    }
}

// ---------------------------------------------------------------------------
// MemoryBudget
// ---------------------------------------------------------------------------

/// Manages a GPU memory budget.
#[derive(Debug, Clone)]
pub struct MemoryBudget {
    pub total_bytes: u64,
    pub used_bytes: u64,
    pub reserved_bytes: u64,
}

impl MemoryBudget {
    pub fn new(total_bytes: u64) -> Self {
        Self { total_bytes, used_bytes: 0, reserved_bytes: 0 }
    }

    /// Create a budget with some bytes reserved for the system/driver.
    pub fn with_reserved(total_bytes: u64, reserved_bytes: u64) -> Self {
        Self { total_bytes, used_bytes: 0, reserved_bytes }
    }

    /// Attempt to allocate `bytes`. Fails if not enough free space.
    pub fn allocate(&mut self, bytes: u64) -> Result<(), String> {
        if !self.can_allocate(bytes) {
            return Err(format!(
                "Cannot allocate {} bytes: only {} free",
                bytes,
                self.free_bytes()
            ));
        }
        self.used_bytes += bytes;
        Ok(())
    }

    /// Free previously allocated bytes. Saturates at zero.
    pub fn free(&mut self, bytes: u64) {
        self.used_bytes = self.used_bytes.saturating_sub(bytes);
    }

    /// Fraction of total memory currently in use (0.0–1.0).
    pub fn utilization(&self) -> f64 {
        if self.total_bytes == 0 {
            return 0.0;
        }
        self.used_bytes as f64 / self.total_bytes as f64
    }

    /// Bytes available for allocation.
    pub fn free_bytes(&self) -> u64 {
        self.total_bytes.saturating_sub(self.used_bytes + self.reserved_bytes)
    }

    pub fn can_allocate(&self, bytes: u64) -> bool {
        bytes <= self.free_bytes()
    }
}

// ---------------------------------------------------------------------------
// MemoryPool
// ---------------------------------------------------------------------------

/// Simple pool for reusing GPU buffers by size bucket.
#[derive(Debug)]
pub struct MemoryPool {
    /// Buckets keyed by (rounded-up) size, values are lists of pool entry ids.
    buckets: HashMap<usize, Vec<usize>>,
    max_entries: usize,
    total_entries: usize,
    next_id: usize,
    hits: usize,
    misses: usize,
}

/// Statistics for a `MemoryPool`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PoolStats {
    pub total_entries: usize,
    pub bucket_count: usize,
    pub hits: usize,
    pub misses: usize,
}

impl MemoryPool {
    pub fn new(max_entries: usize) -> Self {
        Self {
            buckets: HashMap::new(),
            max_entries,
            total_entries: 0,
            next_id: 0,
            hits: 0,
            misses: 0,
        }
    }

    /// Round size up to nearest power-of-two bucket (minimum 256 bytes).
    fn bucket_size(size: usize) -> usize {
        let min = 256;
        let s = size.max(min);
        s.next_power_of_two()
    }

    /// Try to get a pooled buffer of at least `size` bytes.
    /// Returns `Some(pool_entry_id)` on hit, `None` on miss.
    pub fn get(&mut self, size: usize) -> Option<usize> {
        let bucket = Self::bucket_size(size);
        if let Some(ids) = self.buckets.get_mut(&bucket)
            && let Some(id) = ids.pop()
        {
            self.total_entries -= 1;
            self.hits += 1;
            return Some(id);
        }
        self.misses += 1;
        None
    }

    /// Return a buffer of `size` bytes to the pool. Returns the pool entry id.
    pub fn put(&mut self, size: usize) -> usize {
        let bucket = Self::bucket_size(size);
        let id = self.next_id;
        self.next_id += 1;

        // Evict oldest entry in the same bucket if at capacity.
        if self.total_entries >= self.max_entries {
            // Try to evict from the largest bucket to free space.
            if let Some((&evict_bucket, _)) =
                self.buckets.iter().filter(|(_, v)| !v.is_empty()).max_by_key(|&(&k, _)| k)
                && let Some(ids) = self.buckets.get_mut(&evict_bucket)
            {
                ids.remove(0);
                self.total_entries -= 1;
            }
        }

        self.buckets.entry(bucket).or_default().push(id);
        self.total_entries += 1;
        id
    }

    /// Clear all pooled entries.
    pub fn clear(&mut self) {
        self.buckets.clear();
        self.total_entries = 0;
    }

    pub fn stats(&self) -> PoolStats {
        PoolStats {
            total_entries: self.total_entries,
            bucket_count: self.buckets.len(),
            hits: self.hits,
            misses: self.misses,
        }
    }
}

// ---------------------------------------------------------------------------
// TransferOptimizer
// ---------------------------------------------------------------------------

/// Analyses a tracker and produces actionable optimization suggestions.
pub struct TransferOptimizer;

impl TransferOptimizer {
    /// Suggest a `BufferAllocationStrategy` based on transfer patterns.
    pub fn recommend_strategy(tracker: &MemoryTransferTracker) -> BufferAllocationStrategy {
        let avg_bw = tracker.average_bandwidth_gbps();
        let count = tracker.transfer_count();
        let small = tracker.records.iter().filter(|r| r.bytes < 4096).count();

        if count == 0 {
            return BufferAllocationStrategy::AllocateOnWrite;
        }

        // Lots of small transfers → pinned memory is most beneficial
        if small as f64 / count as f64 > 0.5 {
            return BufferAllocationStrategy::PinnedMemory;
        }

        // Low bandwidth → try zero-copy
        if avg_bw > 0.0 && avg_bw < 5.0 {
            return BufferAllocationStrategy::ZeroCopy;
        }

        // Many transfers → pre-allocate
        if count > 50 {
            return BufferAllocationStrategy::PreAllocate;
        }

        BufferAllocationStrategy::AllocateOnWrite
    }

    /// Estimate bytes wasted due to pool bucketing for a given set of sizes.
    pub fn estimate_pool_waste(sizes: &[usize]) -> usize {
        sizes.iter().map(|&s| MemoryPool::bucket_size(s).saturating_sub(s)).sum()
    }
}

// ---------------------------------------------------------------------------
// IntelArcMemoryProfile
// ---------------------------------------------------------------------------

/// Memory characteristics for the Intel Arc A770 (16 GB GDDR6, 560 GB/s).
#[derive(Debug, Clone)]
pub struct IntelArcMemoryProfile {
    /// Total VRAM in bytes (16 GiB).
    pub total_vram_bytes: u64,
    /// Theoretical peak bandwidth in GB/s.
    pub peak_bandwidth_gbps: f64,
    /// Memory bus width in bits.
    pub bus_width_bits: u32,
    /// GDDR6 effective clock in MHz.
    pub effective_clock_mhz: u32,
    /// Recommended minimum transfer size for good throughput.
    pub min_efficient_transfer_bytes: usize,
}

impl Default for IntelArcMemoryProfile {
    fn default() -> Self {
        Self::a770()
    }
}

impl IntelArcMemoryProfile {
    /// Intel Arc A770 16 GB reference profile.
    pub fn a770() -> Self {
        Self {
            total_vram_bytes: 16 * 1024 * 1024 * 1024, // 16 GiB
            peak_bandwidth_gbps: 560.0,
            bus_width_bits: 256,
            effective_clock_mhz: 17_500,
            min_efficient_transfer_bytes: 64 * 1024, // 64 KB
        }
    }

    /// Returns the budget for this profile (reserving 512 MB for driver).
    pub fn budget(&self) -> MemoryBudget {
        let reserved = 512 * 1024 * 1024; // 512 MB
        MemoryBudget::with_reserved(self.total_vram_bytes, reserved)
    }

    /// Estimated transfer time in nanoseconds at theoretical peak bandwidth.
    pub fn estimated_transfer_ns(&self, bytes: usize) -> u64 {
        if self.peak_bandwidth_gbps == 0.0 {
            return 0;
        }
        // bandwidth in bytes/ns = peak_bandwidth_gbps (since 1 GB/s = 1 byte/ns)
        let ns = bytes as f64 / self.peak_bandwidth_gbps;
        ns.ceil() as u64
    }

    /// Whether a transfer size is considered efficient.
    pub fn is_efficient_transfer(&self, bytes: usize) -> bool {
        bytes >= self.min_efficient_transfer_bytes
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ===== MemoryRegion =====

    #[test]
    fn region_new() {
        let r = MemoryRegion::new(1, 1024, MemoryLocation::Device);
        assert_eq!(r.id, 1);
        assert_eq!(r.size, 1024);
        assert_eq!(r.location, MemoryLocation::Device);
    }

    #[test]
    fn region_clone_eq() {
        let a = MemoryRegion::new(2, 512, MemoryLocation::Host);
        let b = a.clone();
        assert_eq!(a, b);
    }

    #[test]
    fn region_mapped_location() {
        let r = MemoryRegion::new(3, 0, MemoryLocation::Mapped);
        assert_eq!(r.location, MemoryLocation::Mapped);
    }

    // ===== TransferDirection =====

    #[test]
    fn direction_equality() {
        assert_eq!(TransferDirection::HostToDevice, TransferDirection::HostToDevice);
        assert_ne!(TransferDirection::HostToDevice, TransferDirection::DeviceToHost);
    }

    // ===== TransferRecord =====

    #[test]
    fn record_bandwidth_normal() {
        let r = TransferRecord {
            direction: TransferDirection::HostToDevice,
            bytes: 1_000_000_000, // 1 GB
            timestamp_ns: 0,
            duration_ns: 1_000_000_000, // 1 s
        };
        let bw = r.bandwidth_gbps();
        assert!((bw - 1.0).abs() < 1e-9);
    }

    #[test]
    fn record_bandwidth_zero_duration() {
        let r = TransferRecord {
            direction: TransferDirection::DeviceToHost,
            bytes: 100,
            timestamp_ns: 0,
            duration_ns: 0,
        };
        assert_eq!(r.bandwidth_gbps(), 0.0);
    }

    #[test]
    fn record_bandwidth_small_transfer() {
        let r = TransferRecord {
            direction: TransferDirection::HostToDevice,
            bytes: 256,
            timestamp_ns: 0,
            duration_ns: 1,
        };
        assert!((r.bandwidth_gbps() - 256.0).abs() < 1e-9);
    }

    // ===== TransferStats =====

    #[test]
    fn stats_default_is_empty() {
        let s = TransferStats::default();
        assert_eq!(s.total_bytes, 0);
        assert_eq!(s.count, 0);
        assert_eq!(s.avg_bandwidth_gbps(), 0.0);
    }

    #[test]
    fn stats_avg_bandwidth() {
        let s = TransferStats {
            total_bytes: 2_000_000_000,
            count: 2,
            total_duration_ns: 1_000_000_000,
            peak_bandwidth_gbps: 3.0,
        };
        assert!((s.avg_bandwidth_gbps() - 2.0).abs() < 1e-9);
    }

    // ===== MemoryTransferTracker =====

    #[test]
    fn tracker_new_is_empty() {
        let t = MemoryTransferTracker::new();
        assert_eq!(t.transfer_count(), 0);
        assert_eq!(t.total_bytes_transferred(), 0);
    }

    #[test]
    fn tracker_record_single() {
        let mut t = MemoryTransferTracker::new();
        t.record_transfer(TransferDirection::HostToDevice, 1024, 100);
        assert_eq!(t.transfer_count(), 1);
        assert_eq!(t.total_bytes_transferred(), 1024);
    }

    #[test]
    fn tracker_record_multiple() {
        let mut t = MemoryTransferTracker::new();
        t.record_transfer(TransferDirection::HostToDevice, 500, 50);
        t.record_transfer(TransferDirection::DeviceToHost, 300, 30);
        t.record_transfer(TransferDirection::DeviceToDevice, 200, 20);
        assert_eq!(t.transfer_count(), 3);
        assert_eq!(t.total_bytes_transferred(), 1000);
    }

    #[test]
    fn tracker_h2d_stats() {
        let mut t = MemoryTransferTracker::new();
        t.record_transfer(TransferDirection::HostToDevice, 1000, 100);
        t.record_transfer(TransferDirection::HostToDevice, 2000, 200);
        t.record_transfer(TransferDirection::DeviceToHost, 500, 50);
        let s = t.host_to_device_stats();
        assert_eq!(s.count, 2);
        assert_eq!(s.total_bytes, 3000);
        assert_eq!(s.total_duration_ns, 300);
    }

    #[test]
    fn tracker_d2h_stats() {
        let mut t = MemoryTransferTracker::new();
        t.record_transfer(TransferDirection::DeviceToHost, 800, 80);
        let s = t.device_to_host_stats();
        assert_eq!(s.count, 1);
        assert_eq!(s.total_bytes, 800);
    }

    #[test]
    fn tracker_d2d_stats() {
        let mut t = MemoryTransferTracker::new();
        t.record_transfer(TransferDirection::DeviceToDevice, 4096, 10);
        let s = t.device_to_device_stats();
        assert_eq!(s.count, 1);
        assert_eq!(s.total_bytes, 4096);
    }

    #[test]
    fn tracker_average_bandwidth() {
        let mut t = MemoryTransferTracker::new();
        // 2 GB in 1 s => 2 GB/s
        t.record_transfer(TransferDirection::HostToDevice, 2_000_000_000, 1_000_000_000);
        assert!((t.average_bandwidth_gbps() - 2.0).abs() < 1e-9);
    }

    #[test]
    fn tracker_average_bandwidth_empty() {
        let t = MemoryTransferTracker::new();
        assert_eq!(t.average_bandwidth_gbps(), 0.0);
    }

    #[test]
    fn tracker_peak_bandwidth() {
        let mut t = MemoryTransferTracker::new();
        t.record_transfer(TransferDirection::HostToDevice, 100, 100); // 1 GB/s
        t.record_transfer(TransferDirection::HostToDevice, 500, 100); // 5 GB/s
        t.record_transfer(TransferDirection::HostToDevice, 200, 100); // 2 GB/s
        assert!((t.peak_bandwidth_gbps() - 5.0).abs() < 1e-9);
    }

    #[test]
    fn tracker_peak_bandwidth_empty() {
        let t = MemoryTransferTracker::new();
        assert_eq!(t.peak_bandwidth_gbps(), 0.0);
    }

    #[test]
    fn tracker_zero_byte_transfer() {
        let mut t = MemoryTransferTracker::new();
        t.record_transfer(TransferDirection::HostToDevice, 0, 100);
        assert_eq!(t.total_bytes_transferred(), 0);
        assert_eq!(t.transfer_count(), 1);
    }

    #[test]
    fn tracker_zero_duration_transfer() {
        let mut t = MemoryTransferTracker::new();
        t.record_transfer(TransferDirection::HostToDevice, 1024, 0);
        assert_eq!(t.total_bytes_transferred(), 1024);
        // peak bandwidth should be 0 for zero-duration
        assert_eq!(t.peak_bandwidth_gbps(), 0.0);
    }

    #[test]
    fn tracker_stats_peak_bandwidth_tracks_max() {
        let mut t = MemoryTransferTracker::new();
        t.record_transfer(TransferDirection::HostToDevice, 100, 10); // 10 GB/s
        t.record_transfer(TransferDirection::HostToDevice, 100, 5); // 20 GB/s
        let s = t.host_to_device_stats();
        assert!((s.peak_bandwidth_gbps - 20.0).abs() < 1e-9);
    }

    #[test]
    fn tracker_timestamps_advance() {
        let mut t = MemoryTransferTracker::new();
        t.record_transfer(TransferDirection::HostToDevice, 100, 50);
        t.record_transfer(TransferDirection::HostToDevice, 200, 30);
        assert_eq!(t.records[0].timestamp_ns, 0);
        assert_eq!(t.records[1].timestamp_ns, 50);
    }

    // ===== suggest_optimizations =====

    #[test]
    fn suggest_batch_small_transfers() {
        let mut t = MemoryTransferTracker::new();
        for _ in 0..15 {
            t.record_transfer(TransferDirection::HostToDevice, 128, 10);
        }
        let suggestions = t.suggest_optimizations();
        assert!(suggestions.iter().any(|s| s.contains("small transfers")));
    }

    #[test]
    fn suggest_low_bandwidth() {
        let mut t = MemoryTransferTracker::new();
        // 100 bytes in 100 ns = 1 GB/s (low)
        t.record_transfer(TransferDirection::HostToDevice, 100, 100);
        let suggestions = t.suggest_optimizations();
        assert!(suggestions.iter().any(|s| s.contains("pinned memory")));
    }

    #[test]
    fn suggest_excessive_h2d() {
        let mut t = MemoryTransferTracker::new();
        t.record_transfer(TransferDirection::HostToDevice, 50_000, 100);
        t.record_transfer(TransferDirection::DeviceToHost, 10_000, 100);
        let suggestions = t.suggest_optimizations();
        assert!(suggestions.iter().any(|s| s.contains("keep data on device")));
    }

    #[test]
    fn suggest_pre_allocate_many_transfers() {
        let mut t = MemoryTransferTracker::new();
        for _ in 0..110 {
            t.record_transfer(TransferDirection::HostToDevice, 10_000, 10);
        }
        let suggestions = t.suggest_optimizations();
        assert!(suggestions.iter().any(|s| s.contains("pre-allocating")));
    }

    #[test]
    fn suggest_round_trip_detection() {
        let mut t = MemoryTransferTracker::new();
        for _ in 0..5 {
            t.record_transfer(TransferDirection::HostToDevice, 4096, 10);
            t.record_transfer(TransferDirection::DeviceToHost, 4096, 10);
        }
        let suggestions = t.suggest_optimizations();
        assert!(suggestions.iter().any(|s| s.contains("round-trips")));
    }

    #[test]
    fn suggest_no_suggestions_for_healthy_pattern() {
        let mut t = MemoryTransferTracker::new();
        // A few large, fast transfers
        for _ in 0..5 {
            t.record_transfer(TransferDirection::HostToDevice, 1_000_000, 10);
        }
        let suggestions = t.suggest_optimizations();
        assert!(suggestions.is_empty());
    }

    // ===== MemoryBudget =====

    #[test]
    fn budget_new() {
        let b = MemoryBudget::new(1_000_000);
        assert_eq!(b.total_bytes, 1_000_000);
        assert_eq!(b.used_bytes, 0);
        assert_eq!(b.free_bytes(), 1_000_000);
    }

    #[test]
    fn budget_allocate_success() {
        let mut b = MemoryBudget::new(1000);
        assert!(b.allocate(500).is_ok());
        assert_eq!(b.used_bytes, 500);
        assert_eq!(b.free_bytes(), 500);
    }

    #[test]
    fn budget_allocate_exact() {
        let mut b = MemoryBudget::new(1000);
        assert!(b.allocate(1000).is_ok());
        assert_eq!(b.free_bytes(), 0);
    }

    #[test]
    fn budget_allocate_overflow() {
        let mut b = MemoryBudget::new(1000);
        let err = b.allocate(1001);
        assert!(err.is_err());
        assert_eq!(b.used_bytes, 0);
    }

    #[test]
    fn budget_allocate_multiple() {
        let mut b = MemoryBudget::new(1000);
        assert!(b.allocate(300).is_ok());
        assert!(b.allocate(300).is_ok());
        assert!(b.allocate(300).is_ok());
        assert!(b.allocate(200).is_err());
        assert_eq!(b.used_bytes, 900);
    }

    #[test]
    fn budget_free() {
        let mut b = MemoryBudget::new(1000);
        b.allocate(600).unwrap();
        b.free(200);
        assert_eq!(b.used_bytes, 400);
        assert_eq!(b.free_bytes(), 600);
    }

    #[test]
    fn budget_free_saturates() {
        let mut b = MemoryBudget::new(1000);
        b.free(500); // freeing more than used
        assert_eq!(b.used_bytes, 0);
    }

    #[test]
    fn budget_utilization_empty() {
        let b = MemoryBudget::new(1000);
        assert!((b.utilization() - 0.0).abs() < 1e-9);
    }

    #[test]
    fn budget_utilization_half() {
        let mut b = MemoryBudget::new(1000);
        b.allocate(500).unwrap();
        assert!((b.utilization() - 0.5).abs() < 1e-9);
    }

    #[test]
    fn budget_utilization_full() {
        let mut b = MemoryBudget::new(1000);
        b.allocate(1000).unwrap();
        assert!((b.utilization() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn budget_utilization_zero_total() {
        let b = MemoryBudget::new(0);
        assert_eq!(b.utilization(), 0.0);
    }

    #[test]
    fn budget_can_allocate() {
        let mut b = MemoryBudget::new(1000);
        assert!(b.can_allocate(1000));
        assert!(!b.can_allocate(1001));
        b.allocate(600).unwrap();
        assert!(b.can_allocate(400));
        assert!(!b.can_allocate(401));
    }

    #[test]
    fn budget_can_allocate_zero() {
        let b = MemoryBudget::new(1000);
        assert!(b.can_allocate(0));
    }

    #[test]
    fn budget_with_reserved() {
        let b = MemoryBudget::with_reserved(1000, 200);
        assert_eq!(b.free_bytes(), 800);
        assert!(b.can_allocate(800));
        assert!(!b.can_allocate(801));
    }

    #[test]
    fn budget_reserved_affects_allocation() {
        let mut b = MemoryBudget::with_reserved(1000, 300);
        assert!(b.allocate(700).is_ok());
        assert!(b.allocate(1).is_err());
    }

    // ===== MemoryPool =====

    #[test]
    fn pool_new() {
        let p = MemoryPool::new(16);
        assert_eq!(p.stats().total_entries, 0);
    }

    #[test]
    fn pool_put_and_get() {
        let mut p = MemoryPool::new(16);
        let id = p.put(1024);
        let got = p.get(1024);
        assert_eq!(got, Some(id));
    }

    #[test]
    fn pool_get_miss() {
        let mut p = MemoryPool::new(16);
        assert_eq!(p.get(1024), None);
    }

    #[test]
    fn pool_size_bucketing() {
        let mut p = MemoryPool::new(16);
        // 1000 rounds up to 1024
        p.put(1000);
        // Requesting 900 also rounds to 1024 → hit
        assert!(p.get(900).is_some());
    }

    #[test]
    fn pool_size_bucketing_different_bucket() {
        let mut p = MemoryPool::new(16);
        p.put(1000); // bucket 1024
        // Requesting 2000 rounds to 2048 → miss
        assert!(p.get(2000).is_none());
    }

    #[test]
    fn pool_multiple_same_bucket() {
        let mut p = MemoryPool::new(16);
        let id1 = p.put(600);
        let id2 = p.put(700);
        // Both in bucket 1024 (next_power_of_two), LIFO order
        assert_eq!(p.get(600), Some(id2));
        assert_eq!(p.get(600), Some(id1));
        assert_eq!(p.get(600), None);
    }

    #[test]
    fn pool_clear() {
        let mut p = MemoryPool::new(16);
        p.put(1024);
        p.put(2048);
        p.clear();
        assert_eq!(p.stats().total_entries, 0);
        assert_eq!(p.get(1024), None);
    }

    #[test]
    fn pool_stats_hits_misses() {
        let mut p = MemoryPool::new(16);
        p.get(100); // miss
        p.put(100);
        p.get(100); // hit
        let s = p.stats();
        assert_eq!(s.hits, 1);
        assert_eq!(s.misses, 1);
    }

    #[test]
    fn pool_eviction_at_capacity() {
        let mut p = MemoryPool::new(2);
        p.put(1024);
        p.put(2048);
        // Pool full, adding another should evict
        p.put(4096);
        assert_eq!(p.stats().total_entries, 2);
    }

    #[test]
    fn pool_min_bucket_256() {
        let mut p = MemoryPool::new(16);
        p.put(1);
        // Bucket is 256
        assert!(p.get(1).is_some());
        assert!(p.get(200).is_none()); // already consumed
    }

    #[test]
    fn pool_stats_bucket_count() {
        let mut p = MemoryPool::new(16);
        p.put(100); // bucket 256
        p.put(500); // bucket 1024
        p.put(3000); // bucket 4096
        assert_eq!(p.stats().bucket_count, 3);
    }

    #[test]
    fn pool_zero_size() {
        let mut p = MemoryPool::new(16);
        p.put(0);
        assert!(p.get(0).is_some());
    }

    // ===== BufferAllocationStrategy =====

    #[test]
    fn strategy_enum_equality() {
        assert_eq!(BufferAllocationStrategy::PinnedMemory, BufferAllocationStrategy::PinnedMemory);
        assert_ne!(BufferAllocationStrategy::ZeroCopy, BufferAllocationStrategy::PreAllocate);
    }

    #[test]
    fn strategy_all_variants() {
        let strategies = [
            BufferAllocationStrategy::AllocateOnWrite,
            BufferAllocationStrategy::PreAllocate,
            BufferAllocationStrategy::PinnedMemory,
            BufferAllocationStrategy::ZeroCopy,
        ];
        assert_eq!(strategies.len(), 4);
    }

    // ===== TransferOptimizer =====

    #[test]
    fn optimizer_empty_tracker() {
        let t = MemoryTransferTracker::new();
        let strat = TransferOptimizer::recommend_strategy(&t);
        assert_eq!(strat, BufferAllocationStrategy::AllocateOnWrite);
    }

    #[test]
    fn optimizer_many_small_suggests_pinned() {
        let mut t = MemoryTransferTracker::new();
        for _ in 0..20 {
            t.record_transfer(TransferDirection::HostToDevice, 128, 10);
        }
        let strat = TransferOptimizer::recommend_strategy(&t);
        assert_eq!(strat, BufferAllocationStrategy::PinnedMemory);
    }

    #[test]
    fn optimizer_low_bandwidth_suggests_zerocopy() {
        let mut t = MemoryTransferTracker::new();
        // All large transfers but slow (1 GB/s)
        for _ in 0..10 {
            t.record_transfer(TransferDirection::HostToDevice, 1_000_000, 1_000_000);
        }
        let strat = TransferOptimizer::recommend_strategy(&t);
        assert_eq!(strat, BufferAllocationStrategy::ZeroCopy);
    }

    #[test]
    fn optimizer_many_transfers_suggests_preallocate() {
        let mut t = MemoryTransferTracker::new();
        for _ in 0..60 {
            // Large, fast → not small, not low bw, but > 50 count
            t.record_transfer(TransferDirection::HostToDevice, 1_000_000, 10);
        }
        let strat = TransferOptimizer::recommend_strategy(&t);
        assert_eq!(strat, BufferAllocationStrategy::PreAllocate);
    }

    #[test]
    fn optimizer_pool_waste_estimate() {
        let sizes = vec![100, 500, 1000];
        let waste = TransferOptimizer::estimate_pool_waste(&sizes);
        // 100→256 (+156), 500→512 (+12), 1000→1024 (+24) = 192
        assert_eq!(waste, 156 + 12 + 24);
    }

    #[test]
    fn optimizer_pool_waste_exact_powers() {
        let sizes = vec![256, 512, 1024];
        let waste = TransferOptimizer::estimate_pool_waste(&sizes);
        assert_eq!(waste, 0);
    }

    #[test]
    fn optimizer_pool_waste_empty() {
        let waste = TransferOptimizer::estimate_pool_waste(&[]);
        assert_eq!(waste, 0);
    }

    // ===== IntelArcMemoryProfile =====

    #[test]
    fn a770_vram() {
        let p = IntelArcMemoryProfile::a770();
        assert_eq!(p.total_vram_bytes, 16 * 1024 * 1024 * 1024);
    }

    #[test]
    fn a770_peak_bandwidth() {
        let p = IntelArcMemoryProfile::a770();
        assert!((p.peak_bandwidth_gbps - 560.0).abs() < 1e-9);
    }

    #[test]
    fn a770_bus_width() {
        let p = IntelArcMemoryProfile::a770();
        assert_eq!(p.bus_width_bits, 256);
    }

    #[test]
    fn a770_default_is_a770() {
        let d = IntelArcMemoryProfile::default();
        let a = IntelArcMemoryProfile::a770();
        assert_eq!(d.total_vram_bytes, a.total_vram_bytes);
        assert!((d.peak_bandwidth_gbps - a.peak_bandwidth_gbps).abs() < 1e-9);
    }

    #[test]
    fn a770_budget_reserves_512mb() {
        let p = IntelArcMemoryProfile::a770();
        let b = p.budget();
        let expected_free = p.total_vram_bytes - 512 * 1024 * 1024;
        assert_eq!(b.free_bytes(), expected_free);
    }

    #[test]
    fn a770_estimated_transfer_time() {
        let p = IntelArcMemoryProfile::a770();
        // 560 GB in 1s → 1 GB ≈ 1.786 ms = 1_785_714 ns
        let ns = p.estimated_transfer_ns(1_000_000_000);
        // Should be roughly 1_785_714 ns
        assert!(ns > 1_700_000);
        assert!(ns < 1_900_000);
    }

    #[test]
    fn a770_estimated_transfer_zero() {
        let p = IntelArcMemoryProfile::a770();
        assert_eq!(p.estimated_transfer_ns(0), 0);
    }

    #[test]
    fn a770_efficient_transfer_large() {
        let p = IntelArcMemoryProfile::a770();
        assert!(p.is_efficient_transfer(64 * 1024));
        assert!(p.is_efficient_transfer(1_000_000));
    }

    #[test]
    fn a770_efficient_transfer_small() {
        let p = IntelArcMemoryProfile::a770();
        assert!(!p.is_efficient_transfer(1024));
        assert!(!p.is_efficient_transfer(0));
    }

    // ===== Edge cases =====

    #[test]
    fn tracker_large_byte_count() {
        let mut t = MemoryTransferTracker::new();
        t.record_transfer(TransferDirection::HostToDevice, usize::MAX, 1);
        assert_eq!(t.total_bytes_transferred(), usize::MAX);
    }

    #[test]
    fn budget_max_u64() {
        let b = MemoryBudget::new(u64::MAX);
        assert!(b.can_allocate(u64::MAX));
    }

    #[test]
    fn pool_sequential_ids() {
        let mut p = MemoryPool::new(16);
        let a = p.put(100);
        let b = p.put(200);
        let c = p.put(300);
        assert_eq!(a, 0);
        assert_eq!(b, 1);
        assert_eq!(c, 2);
    }

    #[test]
    fn pool_get_after_clear_returns_none() {
        let mut p = MemoryPool::new(16);
        p.put(1024);
        p.clear();
        assert!(p.get(1024).is_none());
    }

    #[test]
    fn tracker_mixed_directions_stats_independent() {
        let mut t = MemoryTransferTracker::new();
        t.record_transfer(TransferDirection::HostToDevice, 100, 10);
        t.record_transfer(TransferDirection::DeviceToHost, 200, 20);
        t.record_transfer(TransferDirection::DeviceToDevice, 300, 30);

        assert_eq!(t.host_to_device_stats().total_bytes, 100);
        assert_eq!(t.device_to_host_stats().total_bytes, 200);
        assert_eq!(t.device_to_device_stats().total_bytes, 300);
    }

    #[test]
    fn budget_allocate_then_free_then_allocate() {
        let mut b = MemoryBudget::new(1000);
        b.allocate(800).unwrap();
        b.free(500);
        assert!(b.allocate(500).is_ok());
        assert_eq!(b.used_bytes, 800);
    }

    #[test]
    fn pool_stats_after_operations() {
        let mut p = MemoryPool::new(16);
        p.put(100);
        p.put(200);
        p.get(100); // hit
        p.get(999); // miss (different bucket, empty)
        let s = p.stats();
        assert_eq!(s.total_entries, 1); // one left
        assert_eq!(s.hits, 1);
        assert_eq!(s.misses, 1);
    }
}
