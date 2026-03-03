//! Memory statistics tracking.

/// A snapshot of memory pool statistics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PoolStats {
    /// Total bytes allocated (cumulative).
    pub allocated_bytes: u64,
    /// Total bytes freed (cumulative).
    pub freed_bytes: u64,
    /// Peak bytes in use at any point.
    pub peak_bytes: u64,
    /// Total number of allocation operations.
    pub allocation_count: u64,
    /// Total number of deallocation operations.
    pub deallocation_count: u64,
    /// Total pool capacity in bytes.
    pub capacity_bytes: u64,
}

impl PoolStats {
    /// Bytes currently in use (allocated − freed).
    #[must_use]
    pub const fn in_use_bytes(&self) -> u64 {
        self.allocated_bytes.saturating_sub(self.freed_bytes)
    }

    /// Fragmentation ratio in `[0.0, 1.0]`.
    ///
    /// Defined as `1.0 − (in_use / capacity)`. Returns `0.0` when nothing is
    /// allocated or when capacity is zero.
    #[must_use]
    #[expect(clippy::cast_precision_loss)]
    pub fn fragmentation(&self) -> f64 {
        if self.capacity_bytes == 0 {
            return 0.0;
        }
        let in_use = self.in_use_bytes();
        if in_use == 0 {
            return 0.0;
        }
        1.0 - (in_use as f64 / self.capacity_bytes as f64)
    }

    /// Utilisation ratio in `[0.0, 1.0]`.
    ///
    /// Defined as `in_use / capacity`. Returns `0.0` when capacity is zero.
    #[must_use]
    #[expect(clippy::cast_precision_loss)]
    pub fn utilisation(&self) -> f64 {
        if self.capacity_bytes == 0 {
            return 0.0;
        }
        self.in_use_bytes() as f64 / self.capacity_bytes as f64
    }
}
