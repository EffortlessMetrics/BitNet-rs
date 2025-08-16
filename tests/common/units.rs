/// Canonical MB→bytes multiplier used everywhere we translate disk sizes.
pub const BYTES_PER_MB: u64 = 1_048_576;

/// Canonical GB→bytes multiplier (1024 * MB).
pub const BYTES_PER_GB: u64 = 1024 * BYTES_PER_MB;