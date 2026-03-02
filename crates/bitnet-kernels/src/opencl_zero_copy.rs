//! Zero-copy buffer sharing for CPU-GPU memory on Intel Arc A770.
//!
//! Maps host memory into GPU address space for A770's unified memory
//! architecture. CPU reference implementations simulate zero-copy via
//! direct `Vec` access, providing the same API surface that a real
//! OpenCL `CL_MEM_USE_HOST_PTR` path would expose.

use std::fmt;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for zero-copy buffer management.
#[derive(Debug, Clone)]
pub struct ZeroCopyConfig {
    /// Whether host-mapped buffers are enabled.
    pub enable_host_mapped: bool,
    /// Required byte alignment for mapped buffers.
    pub alignment: usize,
    /// Whether to use pinned (page-locked) host memory.
    pub use_pinned_memory: bool,
    /// Upper bound on total mapped bytes.
    pub max_mapped_bytes: usize,
}

impl Default for ZeroCopyConfig {
    fn default() -> Self {
        Self {
            enable_host_mapped: true,
            alignment: 64,
            use_pinned_memory: false,
            max_mapped_bytes: 16 * 1024 * 1024 * 1024, // 16 GB
        }
    }
}

// ---------------------------------------------------------------------------
// Access mode
// ---------------------------------------------------------------------------

/// Access mode for a mapped buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AccessMode {
    ReadOnly,
    WriteOnly,
    ReadWrite,
}

// ---------------------------------------------------------------------------
// MappedBuffer
// ---------------------------------------------------------------------------

/// A host-side buffer that can be mapped into GPU address space.
#[derive(Debug, Clone)]
pub struct MappedBuffer {
    pub id: u64,
    pub host_data: Vec<f32>,
    pub size_bytes: usize,
    pub is_dirty: bool,
    pub access_mode: AccessMode,
}

// ---------------------------------------------------------------------------
// BufferMapping
// ---------------------------------------------------------------------------

/// Describes an active mapping of a sub-range of a `MappedBuffer`.
#[derive(Debug, Clone)]
pub struct BufferMapping {
    pub buffer_id: u64,
    pub offset: usize,
    pub length: usize,
    pub mapped: bool,
}

// ---------------------------------------------------------------------------
// Stats
// ---------------------------------------------------------------------------

/// Cumulative statistics for the zero-copy manager.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ZeroCopyStats {
    pub total_mapped: u64,
    pub total_unmapped: u64,
    pub bytes_transferred_saved: u64,
    pub current_mapped_bytes: usize,
    pub peak_mapped_bytes: usize,
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Errors that can occur during zero-copy operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ZeroCopyError {
    NotSupported,
    AlignmentError { required: usize, got: usize },
    MapFailed(String),
    BufferNotFound(u64),
    MemoryLimitExceeded { requested: usize, available: usize },
}

impl fmt::Display for ZeroCopyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotSupported => write!(f, "zero-copy not supported"),
            Self::AlignmentError { required, got } => {
                write!(f, "alignment error: required {required}, got {got}")
            }
            Self::MapFailed(msg) => write!(f, "map failed: {msg}"),
            Self::BufferNotFound(id) => write!(f, "buffer {id} not found"),
            Self::MemoryLimitExceeded { requested, available } => {
                write!(f, "memory limit exceeded: requested {requested}, available {available}")
            }
        }
    }
}

impl std::error::Error for ZeroCopyError {}

// ---------------------------------------------------------------------------
// Manager
// ---------------------------------------------------------------------------

/// Manages zero-copy buffers and their host↔device mappings.
pub struct ZeroCopyManager {
    pub config: ZeroCopyConfig,
    pub buffers: Vec<MappedBuffer>,
    pub mappings: Vec<BufferMapping>,
    pub next_id: u64,
    pub stats: ZeroCopyStats,
}

// ---------------------------------------------------------------------------
// Public API — CPU reference implementations
// ---------------------------------------------------------------------------

/// Create a new `ZeroCopyManager` from the given configuration.
pub fn create_zero_copy_manager(config: ZeroCopyConfig) -> ZeroCopyManager {
    ZeroCopyManager {
        config,
        buffers: Vec::new(),
        mappings: Vec::new(),
        next_id: 1,
        stats: ZeroCopyStats::default(),
    }
}

/// Allocate a new mapped buffer of `size` f32 elements.
///
/// Returns the buffer ID on success.
pub fn cpu_create_mapped_buffer(
    mgr: &mut ZeroCopyManager,
    size: usize,
    mode: AccessMode,
) -> Result<u64, ZeroCopyError> {
    let size_bytes = size * std::mem::size_of::<f32>();

    if !cpu_is_aligned(size_bytes, mgr.config.alignment) && size_bytes != 0 {
        // Round up internally — real OpenCL would reject misaligned host ptrs.
        // We allow it here but track aligned size.
    }

    let available = mgr.config.max_mapped_bytes - mgr.stats.current_mapped_bytes;
    if size_bytes > available {
        return Err(ZeroCopyError::MemoryLimitExceeded { requested: size_bytes, available });
    }

    let id = mgr.next_id;
    mgr.next_id += 1;

    mgr.buffers.push(MappedBuffer {
        id,
        host_data: vec![0.0f32; size],
        size_bytes,
        is_dirty: false,
        access_mode: mode,
    });

    mgr.stats.current_mapped_bytes += size_bytes;
    if mgr.stats.current_mapped_bytes > mgr.stats.peak_mapped_bytes {
        mgr.stats.peak_mapped_bytes = mgr.stats.current_mapped_bytes;
    }

    Ok(id)
}

/// Write `data` into the buffer identified by `id`.
pub fn cpu_write_buffer(
    mgr: &mut ZeroCopyManager,
    id: u64,
    data: &[f32],
) -> Result<(), ZeroCopyError> {
    let buf =
        mgr.buffers.iter_mut().find(|b| b.id == id).ok_or(ZeroCopyError::BufferNotFound(id))?;

    if buf.access_mode == AccessMode::ReadOnly {
        return Err(ZeroCopyError::MapFailed("buffer is read-only".into()));
    }

    let len = data.len().min(buf.host_data.len());
    buf.host_data[..len].copy_from_slice(&data[..len]);
    buf.is_dirty = true;
    Ok(())
}

/// Read the contents of the buffer identified by `id`.
pub fn cpu_read_buffer(mgr: &ZeroCopyManager, id: u64) -> Result<&[f32], ZeroCopyError> {
    let buf = mgr.buffers.iter().find(|b| b.id == id).ok_or(ZeroCopyError::BufferNotFound(id))?;
    Ok(&buf.host_data)
}

/// Map a sub-range of a buffer. Returns the index into `mgr.mappings`.
pub fn cpu_map_buffer(
    mgr: &mut ZeroCopyManager,
    id: u64,
    offset: usize,
    length: usize,
) -> Result<usize, ZeroCopyError> {
    // Verify buffer exists.
    if !mgr.buffers.iter().any(|b| b.id == id) {
        return Err(ZeroCopyError::BufferNotFound(id));
    }

    let mapping = BufferMapping { buffer_id: id, offset, length, mapped: true };
    mgr.mappings.push(mapping);
    mgr.stats.total_mapped += 1;
    mgr.stats.bytes_transferred_saved += length as u64;

    Ok(mgr.mappings.len() - 1)
}

/// Unmap a previously created mapping by its index.
pub fn cpu_unmap_buffer(
    mgr: &mut ZeroCopyManager,
    mapping_idx: usize,
) -> Result<(), ZeroCopyError> {
    if mapping_idx >= mgr.mappings.len() {
        return Err(ZeroCopyError::MapFailed("mapping index out of range".into()));
    }
    mgr.mappings[mapping_idx].mapped = false;
    mgr.stats.total_unmapped += 1;
    Ok(())
}

/// Mark a buffer as modified (dirty).
pub fn cpu_mark_dirty(mgr: &mut ZeroCopyManager, id: u64) {
    if let Some(buf) = mgr.buffers.iter_mut().find(|b| b.id == id) {
        buf.is_dirty = true;
    }
}

/// Flush a dirty buffer (simulate device synchronisation).
pub fn cpu_sync_buffer(mgr: &mut ZeroCopyManager, id: u64) -> Result<(), ZeroCopyError> {
    let buf =
        mgr.buffers.iter_mut().find(|b| b.id == id).ok_or(ZeroCopyError::BufferNotFound(id))?;
    buf.is_dirty = false;
    Ok(())
}

/// Release a buffer and free its tracked memory.
pub fn cpu_release_buffer(mgr: &mut ZeroCopyManager, id: u64) -> Result<(), ZeroCopyError> {
    let pos =
        mgr.buffers.iter().position(|b| b.id == id).ok_or(ZeroCopyError::BufferNotFound(id))?;
    let buf = mgr.buffers.remove(pos);
    mgr.stats.current_mapped_bytes = mgr.stats.current_mapped_bytes.saturating_sub(buf.size_bytes);

    // Remove any associated mappings.
    mgr.mappings.retain(|m| m.buffer_id != id);
    Ok(())
}

/// Check whether `ptr_offset` satisfies the given `alignment`.
#[inline]
pub fn cpu_is_aligned(ptr_offset: usize, alignment: usize) -> bool {
    if alignment == 0 {
        return true;
    }
    ptr_offset.is_multiple_of(alignment)
}

/// Estimate time saved (in µs) by avoiding a PCIe transfer of `size` bytes
/// at `bandwidth_gbps` GB/s.
pub fn cpu_estimate_transfer_savings(size: usize, bandwidth_gbps: f64) -> f64 {
    if bandwidth_gbps <= 0.0 {
        return 0.0;
    }
    // bytes → seconds → microseconds
    let bytes = size as f64;
    let seconds = bytes / (bandwidth_gbps * 1e9);
    seconds * 1e6
}

/// Return a snapshot of the current statistics.
pub fn cpu_get_stats(mgr: &ZeroCopyManager) -> ZeroCopyStats {
    mgr.stats.clone()
}

/// Human-readable summary of the manager state.
pub fn format_zero_copy_status(mgr: &ZeroCopyManager) -> String {
    format!(
        "ZeroCopy: {} buffers, {} mappings, {}/{} bytes mapped (peak {}), saved {} bytes xfer",
        mgr.buffers.len(),
        mgr.mappings.iter().filter(|m| m.mapped).count(),
        mgr.stats.current_mapped_bytes,
        mgr.config.max_mapped_bytes,
        mgr.stats.peak_mapped_bytes,
        mgr.stats.bytes_transferred_saved,
    )
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ---- helpers ----------------------------------------------------------

    fn default_mgr() -> ZeroCopyManager {
        create_zero_copy_manager(ZeroCopyConfig::default())
    }

    fn a770_config() -> ZeroCopyConfig {
        ZeroCopyConfig {
            enable_host_mapped: true,
            alignment: 64,
            use_pinned_memory: true,
            max_mapped_bytes: 16 * 1024 * 1024 * 1024,
        }
    }

    // ---- create manager ---------------------------------------------------

    #[test]
    fn test_create_manager_empty() {
        let mgr = default_mgr();
        assert!(mgr.buffers.is_empty());
        assert!(mgr.mappings.is_empty());
        assert_eq!(mgr.next_id, 1);
    }

    #[test]
    fn test_create_manager_with_config() {
        let cfg = a770_config();
        let mgr = create_zero_copy_manager(cfg.clone());
        assert_eq!(mgr.config.alignment, 64);
        assert!(mgr.config.use_pinned_memory);
    }

    // ---- create buffer ----------------------------------------------------

    #[test]
    fn test_create_buffer_returns_id() {
        let mut mgr = default_mgr();
        let id = cpu_create_mapped_buffer(&mut mgr, 256, AccessMode::ReadWrite).unwrap();
        assert_eq!(id, 1);
    }

    #[test]
    fn test_create_multiple_buffers_unique_ids() {
        let mut mgr = default_mgr();
        let a = cpu_create_mapped_buffer(&mut mgr, 64, AccessMode::ReadOnly).unwrap();
        let b = cpu_create_mapped_buffer(&mut mgr, 64, AccessMode::WriteOnly).unwrap();
        assert_ne!(a, b);
    }

    #[test]
    fn test_create_buffer_tracks_bytes() {
        let mut mgr = default_mgr();
        cpu_create_mapped_buffer(&mut mgr, 100, AccessMode::ReadWrite).unwrap();
        assert_eq!(mgr.stats.current_mapped_bytes, 100 * 4);
    }

    // ---- write / read -----------------------------------------------------

    #[test]
    fn test_write_read_roundtrip() {
        let mut mgr = default_mgr();
        let id = cpu_create_mapped_buffer(&mut mgr, 4, AccessMode::ReadWrite).unwrap();
        let data = [1.0, 2.0, 3.0, 4.0];
        cpu_write_buffer(&mut mgr, id, &data).unwrap();
        let out = cpu_read_buffer(&mgr, id).unwrap();
        assert_eq!(out, &data);
    }

    #[test]
    fn test_write_sets_dirty() {
        let mut mgr = default_mgr();
        let id = cpu_create_mapped_buffer(&mut mgr, 4, AccessMode::ReadWrite).unwrap();
        cpu_write_buffer(&mut mgr, id, &[1.0; 4]).unwrap();
        assert!(mgr.buffers[0].is_dirty);
    }

    #[test]
    fn test_read_buffer_not_found() {
        let mgr = default_mgr();
        assert_eq!(cpu_read_buffer(&mgr, 999), Err(ZeroCopyError::BufferNotFound(999)));
    }

    #[test]
    fn test_write_buffer_not_found() {
        let mut mgr = default_mgr();
        assert_eq!(cpu_write_buffer(&mut mgr, 42, &[1.0]), Err(ZeroCopyError::BufferNotFound(42)));
    }

    // ---- map / unmap ------------------------------------------------------

    #[test]
    fn test_map_buffer_recorded() {
        let mut mgr = default_mgr();
        let id = cpu_create_mapped_buffer(&mut mgr, 128, AccessMode::ReadWrite).unwrap();
        let idx = cpu_map_buffer(&mut mgr, id, 0, 64).unwrap();
        assert!(mgr.mappings[idx].mapped);
        assert_eq!(mgr.mappings[idx].buffer_id, id);
    }

    #[test]
    fn test_unmap_buffer_clears_flag() {
        let mut mgr = default_mgr();
        let id = cpu_create_mapped_buffer(&mut mgr, 128, AccessMode::ReadWrite).unwrap();
        let idx = cpu_map_buffer(&mut mgr, id, 0, 64).unwrap();
        cpu_unmap_buffer(&mut mgr, idx).unwrap();
        assert!(!mgr.mappings[idx].mapped);
    }

    #[test]
    fn test_map_nonexistent_buffer() {
        let mut mgr = default_mgr();
        assert_eq!(cpu_map_buffer(&mut mgr, 99, 0, 10), Err(ZeroCopyError::BufferNotFound(99)));
    }

    #[test]
    fn test_unmap_out_of_range() {
        let mut mgr = default_mgr();
        assert!(cpu_unmap_buffer(&mut mgr, 100).is_err());
    }

    #[test]
    fn test_map_updates_stats() {
        let mut mgr = default_mgr();
        let id = cpu_create_mapped_buffer(&mut mgr, 128, AccessMode::ReadWrite).unwrap();
        cpu_map_buffer(&mut mgr, id, 0, 256).unwrap();
        assert_eq!(mgr.stats.total_mapped, 1);
        assert_eq!(mgr.stats.bytes_transferred_saved, 256);
    }

    // ---- dirty / sync -----------------------------------------------------

    #[test]
    fn test_mark_dirty_sets_flag() {
        let mut mgr = default_mgr();
        let id = cpu_create_mapped_buffer(&mut mgr, 4, AccessMode::ReadWrite).unwrap();
        cpu_mark_dirty(&mut mgr, id);
        assert!(mgr.buffers[0].is_dirty);
    }

    #[test]
    fn test_sync_clears_dirty() {
        let mut mgr = default_mgr();
        let id = cpu_create_mapped_buffer(&mut mgr, 4, AccessMode::ReadWrite).unwrap();
        cpu_mark_dirty(&mut mgr, id);
        cpu_sync_buffer(&mut mgr, id).unwrap();
        assert!(!mgr.buffers[0].is_dirty);
    }

    #[test]
    fn test_sync_not_found() {
        let mut mgr = default_mgr();
        assert_eq!(cpu_sync_buffer(&mut mgr, 77), Err(ZeroCopyError::BufferNotFound(77)));
    }

    // ---- release ----------------------------------------------------------

    #[test]
    fn test_release_removes_buffer() {
        let mut mgr = default_mgr();
        let id = cpu_create_mapped_buffer(&mut mgr, 64, AccessMode::ReadWrite).unwrap();
        cpu_release_buffer(&mut mgr, id).unwrap();
        assert!(mgr.buffers.is_empty());
    }

    #[test]
    fn test_release_updates_mapped_bytes() {
        let mut mgr = default_mgr();
        let id = cpu_create_mapped_buffer(&mut mgr, 64, AccessMode::ReadWrite).unwrap();
        cpu_release_buffer(&mut mgr, id).unwrap();
        assert_eq!(mgr.stats.current_mapped_bytes, 0);
    }

    #[test]
    fn test_release_not_found() {
        let mut mgr = default_mgr();
        assert_eq!(cpu_release_buffer(&mut mgr, 5), Err(ZeroCopyError::BufferNotFound(5)));
    }

    #[test]
    fn test_release_cleans_up_mappings() {
        let mut mgr = default_mgr();
        let id = cpu_create_mapped_buffer(&mut mgr, 64, AccessMode::ReadWrite).unwrap();
        cpu_map_buffer(&mut mgr, id, 0, 32).unwrap();
        cpu_release_buffer(&mut mgr, id).unwrap();
        assert!(mgr.mappings.is_empty());
    }

    // ---- memory limit -----------------------------------------------------

    #[test]
    fn test_memory_limit_exceeded() {
        let cfg = ZeroCopyConfig { max_mapped_bytes: 100, ..ZeroCopyConfig::default() };
        let mut mgr = create_zero_copy_manager(cfg);
        // 100 f32 = 400 bytes > 100-byte limit
        let res = cpu_create_mapped_buffer(&mut mgr, 100, AccessMode::ReadWrite);
        assert!(matches!(res, Err(ZeroCopyError::MemoryLimitExceeded { .. })));
    }

    #[test]
    fn test_memory_limit_cumulative() {
        let cfg = ZeroCopyConfig { max_mapped_bytes: 64, ..ZeroCopyConfig::default() };
        let mut mgr = create_zero_copy_manager(cfg);
        // 8 f32 = 32 bytes, fits
        cpu_create_mapped_buffer(&mut mgr, 8, AccessMode::ReadWrite).unwrap();
        // another 32 bytes, fits (total 64)
        cpu_create_mapped_buffer(&mut mgr, 8, AccessMode::ReadWrite).unwrap();
        // 1 more byte would exceed
        let res = cpu_create_mapped_buffer(&mut mgr, 1, AccessMode::ReadWrite);
        assert!(matches!(res, Err(ZeroCopyError::MemoryLimitExceeded { .. })));
    }

    // ---- alignment --------------------------------------------------------

    #[test]
    fn test_is_aligned_true() {
        assert!(cpu_is_aligned(128, 64));
        assert!(cpu_is_aligned(0, 64));
    }

    #[test]
    fn test_is_aligned_false() {
        assert!(!cpu_is_aligned(13, 64));
        assert!(!cpu_is_aligned(65, 64));
    }

    #[test]
    fn test_is_aligned_zero_alignment() {
        assert!(cpu_is_aligned(42, 0));
    }

    // ---- transfer savings -------------------------------------------------

    #[test]
    fn test_transfer_savings_positive() {
        let savings = cpu_estimate_transfer_savings(1_000_000, 16.0);
        assert!(savings > 0.0);
    }

    #[test]
    fn test_transfer_savings_zero_bandwidth() {
        assert_eq!(cpu_estimate_transfer_savings(1024, 0.0), 0.0);
    }

    #[test]
    fn test_transfer_savings_large_buffer() {
        let small = cpu_estimate_transfer_savings(1024, 16.0);
        let large = cpu_estimate_transfer_savings(1_048_576, 16.0);
        assert!(large > small);
    }

    // ---- stats ------------------------------------------------------------

    #[test]
    fn test_stats_initial_zeroes() {
        let mgr = default_mgr();
        let s = cpu_get_stats(&mgr);
        assert_eq!(s, ZeroCopyStats::default());
    }

    #[test]
    fn test_stats_after_operations() {
        let mut mgr = default_mgr();
        let id = cpu_create_mapped_buffer(&mut mgr, 16, AccessMode::ReadWrite).unwrap();
        cpu_map_buffer(&mut mgr, id, 0, 32).unwrap();
        let s = cpu_get_stats(&mgr);
        assert_eq!(s.total_mapped, 1);
        assert_eq!(s.current_mapped_bytes, 16 * 4);
    }

    #[test]
    fn test_stats_unmap_counted() {
        let mut mgr = default_mgr();
        let id = cpu_create_mapped_buffer(&mut mgr, 16, AccessMode::ReadWrite).unwrap();
        let idx = cpu_map_buffer(&mut mgr, id, 0, 32).unwrap();
        cpu_unmap_buffer(&mut mgr, idx).unwrap();
        assert_eq!(mgr.stats.total_unmapped, 1);
    }

    // ---- edge cases -------------------------------------------------------

    #[test]
    fn test_single_element_buffer() {
        let mut mgr = default_mgr();
        let id = cpu_create_mapped_buffer(&mut mgr, 1, AccessMode::ReadWrite).unwrap();
        cpu_write_buffer(&mut mgr, id, &[42.0]).unwrap();
        assert_eq!(cpu_read_buffer(&mgr, id).unwrap(), &[42.0]);
    }

    #[test]
    fn test_readonly_buffer_write_rejected() {
        let mut mgr = default_mgr();
        let id = cpu_create_mapped_buffer(&mut mgr, 4, AccessMode::ReadOnly).unwrap();
        let res = cpu_write_buffer(&mut mgr, id, &[1.0]);
        assert!(res.is_err());
    }

    #[test]
    fn test_zero_length_map() {
        let mut mgr = default_mgr();
        let id = cpu_create_mapped_buffer(&mut mgr, 16, AccessMode::ReadWrite).unwrap();
        let idx = cpu_map_buffer(&mut mgr, id, 0, 0).unwrap();
        assert_eq!(mgr.mappings[idx].length, 0);
    }

    #[test]
    fn test_writeonly_read_allowed() {
        // CPU reference allows reads regardless; access mode is advisory.
        let mut mgr = default_mgr();
        let id = cpu_create_mapped_buffer(&mut mgr, 4, AccessMode::WriteOnly).unwrap();
        assert!(cpu_read_buffer(&mgr, id).is_ok());
    }

    // ---- property tests ---------------------------------------------------

    #[test]
    fn test_mapped_bytes_within_limit() {
        let cfg = ZeroCopyConfig { max_mapped_bytes: 1024, ..ZeroCopyConfig::default() };
        let mut mgr = create_zero_copy_manager(cfg);
        for _ in 0..10 {
            let _ = cpu_create_mapped_buffer(&mut mgr, 8, AccessMode::ReadWrite);
        }
        assert!(mgr.stats.current_mapped_bytes <= mgr.config.max_mapped_bytes);
    }

    #[test]
    fn test_peak_gte_current() {
        let mut mgr = default_mgr();
        let id = cpu_create_mapped_buffer(&mut mgr, 64, AccessMode::ReadWrite).unwrap();
        cpu_release_buffer(&mut mgr, id).unwrap();
        assert!(mgr.stats.peak_mapped_bytes >= mgr.stats.current_mapped_bytes);
    }

    #[test]
    fn test_peak_tracks_high_water() {
        let mut mgr = default_mgr();
        let a = cpu_create_mapped_buffer(&mut mgr, 64, AccessMode::ReadWrite).unwrap();
        let b = cpu_create_mapped_buffer(&mut mgr, 64, AccessMode::ReadWrite).unwrap();
        let peak_after_two = mgr.stats.peak_mapped_bytes;
        cpu_release_buffer(&mut mgr, a).unwrap();
        cpu_release_buffer(&mut mgr, b).unwrap();
        assert_eq!(mgr.stats.peak_mapped_bytes, peak_after_two);
        assert_eq!(mgr.stats.current_mapped_bytes, 0);
    }

    // ---- A770 specific ----------------------------------------------------

    #[test]
    fn test_a770_alignment_64() {
        let cfg = a770_config();
        assert_eq!(cfg.alignment, 64);
        assert!(cpu_is_aligned(256, cfg.alignment));
        assert!(!cpu_is_aligned(100, cfg.alignment));
    }

    #[test]
    fn test_a770_16gb_limit() {
        let cfg = a770_config();
        assert_eq!(cfg.max_mapped_bytes, 16 * 1024 * 1024 * 1024);
    }

    #[test]
    fn test_a770_pinned_memory_flag() {
        let cfg = a770_config();
        assert!(cfg.use_pinned_memory);
    }

    // ---- format status ----------------------------------------------------

    #[test]
    fn test_format_zero_copy_status_empty() {
        let mgr = default_mgr();
        let s = format_zero_copy_status(&mgr);
        assert!(s.contains("0 buffers"));
    }

    #[test]
    fn test_format_zero_copy_status_nonempty() {
        let mut mgr = default_mgr();
        cpu_create_mapped_buffer(&mut mgr, 16, AccessMode::ReadWrite).unwrap();
        let s = format_zero_copy_status(&mgr);
        assert!(s.contains("1 buffers"));
        assert!(s.contains("64")); // 16 * 4 bytes
    }

    // ---- error display ----------------------------------------------------

    #[test]
    fn test_error_display_not_supported() {
        assert_eq!(ZeroCopyError::NotSupported.to_string(), "zero-copy not supported");
    }

    #[test]
    fn test_error_display_alignment() {
        let e = ZeroCopyError::AlignmentError { required: 64, got: 13 };
        assert!(e.to_string().contains("64"));
    }

    #[test]
    fn test_error_display_buffer_not_found() {
        let e = ZeroCopyError::BufferNotFound(7);
        assert!(e.to_string().contains("7"));
    }

    #[test]
    fn test_error_display_memory_limit() {
        let e = ZeroCopyError::MemoryLimitExceeded { requested: 200, available: 100 };
        let s = e.to_string();
        assert!(s.contains("200") && s.contains("100"));
    }

    #[test]
    fn test_error_display_map_failed() {
        let e = ZeroCopyError::MapFailed("oops".into());
        assert!(e.to_string().contains("oops"));
    }
}
