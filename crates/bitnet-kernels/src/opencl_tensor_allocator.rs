//! GPU tensor memory allocator with slab allocation and free-list caching.
//!
//! Provides CPU-reference implementations for managing GPU-side tensor memory
//! on Intel Arc A770 (16 GB VRAM, 64-byte alignment).

use std::collections::HashMap;
use std::fmt;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// Tensor data type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TensorDType {
    F32,
    F16,
    I8,
    Ternary,
}

impl TensorDType {
    /// Bytes per element (Ternary uses 1 byte per element for simplicity).
    pub fn element_size(self) -> usize {
        match self {
            TensorDType::F32 => 4,
            TensorDType::F16 => 2,
            TensorDType::I8 => 1,
            // 2-bit ternary packed 4 per byte, but we allocate per-element
            // with ceil-division so the minimum granularity is 1 byte.
            TensorDType::Ternary => 0, // handled specially in size computation
        }
    }
}

/// Describes a tensor to be allocated.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TensorDescriptor {
    pub shape: Vec<usize>,
    pub dtype: TensorDType,
    pub alignment: usize,
    pub name: Option<String>,
}

/// A tensor that has been placed inside a slab.
#[derive(Debug, Clone)]
pub struct AllocatedTensor {
    pub id: u64,
    pub descriptor: TensorDescriptor,
    pub offset: usize,
    pub size_bytes: usize,
    pub slab_id: usize,
}

/// A contiguous region of (emulated) GPU memory.
#[derive(Debug)]
pub struct Slab {
    pub id: usize,
    pub total_size: usize,
    pub used_size: usize,
    /// Free intervals stored as (offset, length).
    pub free_list: Vec<(usize, usize)>,
    /// IDs of tensors currently residing in this slab.
    pub tensors: Vec<u64>,
}

/// Configuration for [`TensorAllocator`].
#[derive(Debug, Clone)]
pub struct AllocatorConfig {
    pub initial_slab_size: usize,
    pub max_slab_count: usize,
    pub growth_factor: f32,
    pub cache_enabled: bool,
    pub default_alignment: usize,
}

impl Default for AllocatorConfig {
    fn default() -> Self {
        Self {
            initial_slab_size: 256 * 1024 * 1024, // 256 MiB
            max_slab_count: 64,
            growth_factor: 1.5,
            cache_enabled: true,
            default_alignment: 64, // A770 preferred alignment
        }
    }
}

/// Cumulative allocator statistics.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct AllocatorStats {
    pub total_allocated: usize,
    pub total_freed: usize,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub slab_count: usize,
    pub peak_usage: usize,
}

/// Allocation errors.
#[derive(Debug, Clone, PartialEq)]
pub enum AllocError {
    OutOfMemory { requested: usize, available: usize },
    InvalidTensor(u64),
    SlabLimitReached,
    FragmentationTooHigh(f32),
}

impl fmt::Display for AllocError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AllocError::OutOfMemory { requested, available } => {
                write!(f, "out of memory: requested {requested} bytes, {available} available")
            }
            AllocError::InvalidTensor(id) => write!(f, "invalid tensor id: {id}"),
            AllocError::SlabLimitReached => write!(f, "maximum slab count reached"),
            AllocError::FragmentationTooHigh(frag) => {
                write!(f, "fragmentation too high: {frag:.2}")
            }
        }
    }
}

impl std::error::Error for AllocError {}

/// The main allocator that owns slabs and a tensor cache.
pub struct TensorAllocator {
    pub config: AllocatorConfig,
    pub slabs: Vec<Slab>,
    pub tensor_map: HashMap<u64, AllocatedTensor>,
    /// Cache of freed tensors: (descriptor, slab_id, offset).
    pub cache: Vec<(TensorDescriptor, usize, usize)>,
    pub next_id: u64,
    pub stats: AllocatorStats,
}

// ---------------------------------------------------------------------------
// Public API — free functions (CPU reference implementations)
// ---------------------------------------------------------------------------

/// Create a new, empty [`TensorAllocator`].
pub fn create_tensor_allocator(config: AllocatorConfig) -> TensorAllocator {
    TensorAllocator {
        config,
        slabs: Vec::new(),
        tensor_map: HashMap::new(),
        cache: Vec::new(),
        next_id: 1,
        stats: AllocatorStats::default(),
    }
}

/// Compute the byte size for a tensor descriptor.
pub fn cpu_tensor_size_bytes(desc: &TensorDescriptor) -> usize {
    let num_elements: usize = desc.shape.iter().copied().product();
    if num_elements == 0 {
        return 0;
    }
    match desc.dtype {
        TensorDType::Ternary => {
            // 2-bit ternary: 4 elements per byte, round up.
            num_elements.div_ceil(4)
        }
        other => num_elements * other.element_size(),
    }
}

/// Allocate a tensor, using the cache when enabled.
pub fn cpu_alloc_tensor(
    allocator: &mut TensorAllocator,
    desc: TensorDescriptor,
) -> Result<u64, AllocError> {
    // Try cache first.
    if allocator.config.cache_enabled
        && let Some(id) = cpu_alloc_from_cache(allocator, &desc)
    {
        return Ok(id);
    }

    allocator.stats.cache_misses += 1;

    let size = cpu_tensor_size_bytes(&desc);
    let alignment = if desc.alignment == 0 { allocator.config.default_alignment } else { desc.alignment };

    // Try existing slabs.
    for slab_idx in 0..allocator.slabs.len() {
        if let Some((free_idx, offset)) =
            cpu_find_free_block(&allocator.slabs[slab_idx], size, alignment)
        {
            return Ok(place_tensor(allocator, slab_idx, free_idx, offset, size, desc));
        }
    }

    // Need a new slab.
    let min_size = size.max(allocator.config.initial_slab_size);
    let slab_idx = cpu_create_slab(allocator, min_size)?;

    let (free_idx, offset) =
        cpu_find_free_block(&allocator.slabs[slab_idx], size, alignment).expect(
            "freshly created slab must have space",
        );

    Ok(place_tensor(allocator, slab_idx, free_idx, offset, size, desc))
}

/// Free a tensor and optionally cache it.
pub fn cpu_free_tensor(
    allocator: &mut TensorAllocator,
    tensor_id: u64,
) -> Result<(), AllocError> {
    let tensor =
        allocator.tensor_map.remove(&tensor_id).ok_or(AllocError::InvalidTensor(tensor_id))?;

    let slab = &mut allocator.slabs[tensor.slab_id];
    slab.used_size = slab.used_size.saturating_sub(tensor.size_bytes);
    slab.tensors.retain(|&id| id != tensor_id);
    slab.free_list.push((tensor.offset, tensor.size_bytes));
    slab.free_list.sort_by_key(|&(off, _)| off);

    allocator.stats.total_freed += tensor.size_bytes;

    if allocator.config.cache_enabled {
        allocator.cache.push((tensor.descriptor, tensor.slab_id, tensor.offset));
    }

    Ok(())
}

/// Try to satisfy an allocation from the cache.
pub fn cpu_alloc_from_cache(
    allocator: &mut TensorAllocator,
    desc: &TensorDescriptor,
) -> Option<u64> {
    let pos = allocator.cache.iter().position(|(cached, _, _)| {
        cached.dtype == desc.dtype
            && cached.shape == desc.shape
    })?;

    let (cached_desc, slab_id, offset) = allocator.cache.remove(pos);
    let size = cpu_tensor_size_bytes(&cached_desc);

    // Remove the region from the slab free list (it was re-added on free).
    let slab = &mut allocator.slabs[slab_id];
    slab.free_list.retain(|&(o, _)| o != offset);
    slab.used_size += size;
    slab.tensors.push(allocator.next_id);

    let id = allocator.next_id;
    allocator.next_id += 1;
    allocator.stats.cache_hits += 1;
    allocator.stats.total_allocated += size;

    let current_usage: usize = allocator.slabs.iter().map(|s| s.used_size).sum();
    allocator.stats.peak_usage = allocator.stats.peak_usage.max(current_usage);

    allocator.tensor_map.insert(
        id,
        AllocatedTensor {
            id,
            descriptor: desc.clone(),
            offset,
            size_bytes: size,
            slab_id,
        },
    );

    Some(id)
}

/// Create a new slab of at least `min_size` bytes.
pub fn cpu_create_slab(
    allocator: &mut TensorAllocator,
    min_size: usize,
) -> Result<usize, AllocError> {
    if allocator.slabs.len() >= allocator.config.max_slab_count {
        return Err(AllocError::SlabLimitReached);
    }

    // Grow slabs geometrically.
    let grown = if allocator.slabs.is_empty() {
        allocator.config.initial_slab_size
    } else {
        let last = allocator.slabs.last().unwrap().total_size;
        (last as f32 * allocator.config.growth_factor) as usize
    };
    let total_size = min_size.max(grown);

    let id = allocator.slabs.len();
    allocator.slabs.push(Slab {
        id,
        total_size,
        used_size: 0,
        free_list: vec![(0, total_size)],
        tensors: Vec::new(),
    });
    allocator.stats.slab_count = allocator.slabs.len();

    Ok(id)
}

/// First-fit search in a slab's free list, respecting alignment.
///
/// Returns `(free_list_index, aligned_offset)`.
pub fn cpu_find_free_block(
    slab: &Slab,
    size: usize,
    alignment: usize,
) -> Option<(usize, usize)> {
    let align = if alignment == 0 { 1 } else { alignment };
    for (idx, &(offset, length)) in slab.free_list.iter().enumerate() {
        let aligned = (offset + align - 1) & !(align - 1);
        let padding = aligned - offset;
        if length >= size + padding {
            return Some((idx, aligned));
        }
    }
    None
}

/// Merge adjacent free blocks in a slab. Returns the number of merges.
pub fn cpu_coalesce_free_blocks(slab: &mut Slab) -> usize {
    if slab.free_list.len() < 2 {
        return 0;
    }
    slab.free_list.sort_by_key(|&(off, _)| off);

    let mut merged = 0usize;
    let mut i = 0;
    while i + 1 < slab.free_list.len() {
        let (off_a, len_a) = slab.free_list[i];
        let (off_b, len_b) = slab.free_list[i + 1];
        if off_a + len_a == off_b {
            slab.free_list[i] = (off_a, len_a + len_b);
            slab.free_list.remove(i + 1);
            merged += 1;
        } else {
            i += 1;
        }
    }
    merged
}

/// Look up an allocated tensor by ID.
pub fn cpu_get_tensor_info(allocator: &TensorAllocator, id: u64) -> Option<&AllocatedTensor> {
    allocator.tensor_map.get(&id)
}

/// Memory utilisation in `[0.0, 1.0]`.
pub fn cpu_compute_utilization(allocator: &TensorAllocator) -> f32 {
    let total: usize = allocator.slabs.iter().map(|s| s.total_size).sum();
    if total == 0 {
        return 0.0;
    }
    let used: usize = allocator.slabs.iter().map(|s| s.used_size).sum();
    used as f32 / total as f32
}

/// Free everything and reset the allocator to its initial state.
pub fn cpu_reset_allocator(allocator: &mut TensorAllocator) {
    allocator.slabs.clear();
    allocator.tensor_map.clear();
    allocator.cache.clear();
    allocator.next_id = 1;
    allocator.stats = AllocatorStats::default();
}

/// Return a snapshot of the allocator statistics.
pub fn cpu_get_stats(allocator: &TensorAllocator) -> AllocatorStats {
    allocator.stats.clone()
}

/// Human-readable summary of allocator state.
pub fn format_allocator_status(allocator: &TensorAllocator) -> String {
    let util = cpu_compute_utilization(allocator);
    let total: usize = allocator.slabs.iter().map(|s| s.total_size).sum();
    let used: usize = allocator.slabs.iter().map(|s| s.used_size).sum();
    format!(
        "TensorAllocator: slabs={} tensors={} used={} total={} util={:.1}% \
         cache={} hits={} misses={} peak={}",
        allocator.slabs.len(),
        allocator.tensor_map.len(),
        used,
        total,
        util * 100.0,
        allocator.cache.len(),
        allocator.stats.cache_hits,
        allocator.stats.cache_misses,
        allocator.stats.peak_usage,
    )
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Place a tensor into `slab_idx` using the free-list entry at `free_idx`.
fn place_tensor(
    allocator: &mut TensorAllocator,
    slab_idx: usize,
    free_idx: usize,
    aligned_offset: usize,
    size: usize,
    desc: TensorDescriptor,
) -> u64 {
    let slab = &mut allocator.slabs[slab_idx];
    let (orig_off, orig_len) = slab.free_list[free_idx];
    let padding = aligned_offset - orig_off;

    // Split the free block.
    slab.free_list.remove(free_idx);
    if padding > 0 {
        slab.free_list.push((orig_off, padding));
    }
    let remainder = orig_len - padding - size;
    if remainder > 0 {
        slab.free_list.push((aligned_offset + size, remainder));
    }
    slab.free_list.sort_by_key(|&(off, _)| off);

    slab.used_size += size;
    let id = allocator.next_id;
    allocator.next_id += 1;
    slab.tensors.push(id);

    allocator.stats.total_allocated += size;
    let current_usage: usize = allocator.slabs.iter().map(|s| s.used_size).sum();
    allocator.stats.peak_usage = allocator.stats.peak_usage.max(current_usage);

    allocator.tensor_map.insert(
        id,
        AllocatedTensor {
            id,
            descriptor: desc,
            offset: aligned_offset,
            size_bytes: size,
            slab_id: slab_idx,
        },
    );

    id
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn default_config() -> AllocatorConfig {
        AllocatorConfig {
            initial_slab_size: 4096,
            max_slab_count: 8,
            growth_factor: 2.0,
            cache_enabled: true,
            default_alignment: 64,
        }
    }

    fn desc_f32(shape: &[usize]) -> TensorDescriptor {
        TensorDescriptor {
            shape: shape.to_vec(),
            dtype: TensorDType::F32,
            alignment: 64,
            name: None,
        }
    }

    fn desc_f16(shape: &[usize]) -> TensorDescriptor {
        TensorDescriptor {
            shape: shape.to_vec(),
            dtype: TensorDType::F16,
            alignment: 64,
            name: None,
        }
    }

    fn desc_i8(shape: &[usize]) -> TensorDescriptor {
        TensorDescriptor {
            shape: shape.to_vec(),
            dtype: TensorDType::I8,
            alignment: 64,
            name: None,
        }
    }

    fn desc_ternary(shape: &[usize]) -> TensorDescriptor {
        TensorDescriptor {
            shape: shape.to_vec(),
            dtype: TensorDType::Ternary,
            alignment: 64,
            name: None,
        }
    }

    // -----------------------------------------------------------------------
    // Basic creation
    // -----------------------------------------------------------------------

    #[test]
    fn test_create_allocator_empty() {
        let alloc = create_tensor_allocator(default_config());
        assert!(alloc.slabs.is_empty());
        assert!(alloc.tensor_map.is_empty());
        assert!(alloc.cache.is_empty());
        assert_eq!(alloc.next_id, 1);
    }

    #[test]
    fn test_create_allocator_default_config() {
        let alloc = create_tensor_allocator(AllocatorConfig::default());
        assert_eq!(alloc.config.initial_slab_size, 256 * 1024 * 1024);
        assert_eq!(alloc.config.default_alignment, 64);
    }

    #[test]
    fn test_create_allocator_stats_zeroed() {
        let alloc = create_tensor_allocator(default_config());
        assert_eq!(alloc.stats, AllocatorStats::default());
    }

    // -----------------------------------------------------------------------
    // Tensor size computation
    // -----------------------------------------------------------------------

    #[test]
    fn test_tensor_size_f32() {
        let d = desc_f32(&[2, 3, 4]);
        assert_eq!(cpu_tensor_size_bytes(&d), 2 * 3 * 4 * 4);
    }

    #[test]
    fn test_tensor_size_f16() {
        let d = desc_f16(&[10, 20]);
        assert_eq!(cpu_tensor_size_bytes(&d), 10 * 20 * 2);
    }

    #[test]
    fn test_tensor_size_i8() {
        let d = desc_i8(&[100]);
        assert_eq!(cpu_tensor_size_bytes(&d), 100);
    }

    #[test]
    fn test_tensor_size_ternary() {
        // 10 elements → ceil(10/4) = 3 bytes
        let d = desc_ternary(&[10]);
        assert_eq!(cpu_tensor_size_bytes(&d), 3);
    }

    #[test]
    fn test_tensor_size_ternary_exact() {
        // 8 elements → 8/4 = 2 bytes exactly
        let d = desc_ternary(&[8]);
        assert_eq!(cpu_tensor_size_bytes(&d), 2);
    }

    #[test]
    fn test_tensor_size_empty_shape() {
        let d = TensorDescriptor {
            shape: vec![0],
            dtype: TensorDType::F32,
            alignment: 64,
            name: None,
        };
        assert_eq!(cpu_tensor_size_bytes(&d), 0);
    }

    #[test]
    fn test_tensor_size_single_element() {
        let d = desc_f32(&[1]);
        assert_eq!(cpu_tensor_size_bytes(&d), 4);
    }

    // -----------------------------------------------------------------------
    // Allocation basics
    // -----------------------------------------------------------------------

    #[test]
    fn test_alloc_tensor_creates_slab() {
        let mut alloc = create_tensor_allocator(default_config());
        let id = cpu_alloc_tensor(&mut alloc, desc_f32(&[16])).unwrap();
        assert!(id > 0);
        assert_eq!(alloc.slabs.len(), 1);
    }

    #[test]
    fn test_alloc_tensor_records_in_map() {
        let mut alloc = create_tensor_allocator(default_config());
        let id = cpu_alloc_tensor(&mut alloc, desc_f32(&[8])).unwrap();
        assert!(cpu_get_tensor_info(&alloc, id).is_some());
    }

    #[test]
    fn test_alloc_multiple_unique_ids() {
        let mut alloc = create_tensor_allocator(default_config());
        let id1 = cpu_alloc_tensor(&mut alloc, desc_f32(&[4])).unwrap();
        let id2 = cpu_alloc_tensor(&mut alloc, desc_f16(&[4])).unwrap();
        let id3 = cpu_alloc_tensor(&mut alloc, desc_i8(&[4])).unwrap();
        assert_ne!(id1, id2);
        assert_ne!(id2, id3);
        assert_ne!(id1, id3);
    }

    #[test]
    fn test_alloc_tensor_size_recorded() {
        let mut alloc = create_tensor_allocator(default_config());
        let id = cpu_alloc_tensor(&mut alloc, desc_f32(&[16])).unwrap();
        let info = cpu_get_tensor_info(&alloc, id).unwrap();
        assert_eq!(info.size_bytes, 16 * 4);
    }

    #[test]
    fn test_alloc_tensor_alignment() {
        let mut alloc = create_tensor_allocator(default_config());
        let id = cpu_alloc_tensor(&mut alloc, desc_f32(&[4])).unwrap();
        let info = cpu_get_tensor_info(&alloc, id).unwrap();
        assert_eq!(info.offset % 64, 0);
    }

    #[test]
    fn test_alloc_updates_stats() {
        let mut alloc = create_tensor_allocator(default_config());
        cpu_alloc_tensor(&mut alloc, desc_f32(&[4])).unwrap();
        assert!(alloc.stats.total_allocated > 0);
        assert!(alloc.stats.peak_usage > 0);
    }

    // -----------------------------------------------------------------------
    // Free + cache
    // -----------------------------------------------------------------------

    #[test]
    fn test_free_tensor_removes_from_map() {
        let mut alloc = create_tensor_allocator(default_config());
        let id = cpu_alloc_tensor(&mut alloc, desc_f32(&[8])).unwrap();
        cpu_free_tensor(&mut alloc, id).unwrap();
        assert!(cpu_get_tensor_info(&alloc, id).is_none());
    }

    #[test]
    fn test_free_tensor_adds_to_cache() {
        let mut alloc = create_tensor_allocator(default_config());
        let id = cpu_alloc_tensor(&mut alloc, desc_f32(&[8])).unwrap();
        cpu_free_tensor(&mut alloc, id).unwrap();
        assert_eq!(alloc.cache.len(), 1);
    }

    #[test]
    fn test_free_invalid_tensor() {
        let mut alloc = create_tensor_allocator(default_config());
        let err = cpu_free_tensor(&mut alloc, 9999).unwrap_err();
        assert_eq!(err, AllocError::InvalidTensor(9999));
    }

    #[test]
    fn test_free_updates_stats() {
        let mut alloc = create_tensor_allocator(default_config());
        let id = cpu_alloc_tensor(&mut alloc, desc_f32(&[8])).unwrap();
        cpu_free_tensor(&mut alloc, id).unwrap();
        assert!(alloc.stats.total_freed > 0);
    }

    #[test]
    fn test_cache_hit_reuse() {
        let mut alloc = create_tensor_allocator(default_config());
        let id1 = cpu_alloc_tensor(&mut alloc, desc_f32(&[8])).unwrap();
        let info1 = cpu_get_tensor_info(&alloc, id1).unwrap().offset;
        cpu_free_tensor(&mut alloc, id1).unwrap();

        // Allocate same shape+dtype — should get a cache hit.
        let id2 = cpu_alloc_tensor(&mut alloc, desc_f32(&[8])).unwrap();
        let info2 = cpu_get_tensor_info(&alloc, id2).unwrap().offset;
        assert_eq!(info1, info2);
        assert_eq!(alloc.stats.cache_hits, 1);
    }

    #[test]
    fn test_cache_miss_different_dtype() {
        let mut alloc = create_tensor_allocator(default_config());
        let id1 = cpu_alloc_tensor(&mut alloc, desc_f32(&[8])).unwrap();
        cpu_free_tensor(&mut alloc, id1).unwrap();

        // Different dtype — cache miss.
        let _id2 = cpu_alloc_tensor(&mut alloc, desc_f16(&[8])).unwrap();
        assert_eq!(alloc.stats.cache_hits, 0);
    }

    #[test]
    fn test_cache_miss_different_shape() {
        let mut alloc = create_tensor_allocator(default_config());
        let id1 = cpu_alloc_tensor(&mut alloc, desc_f32(&[8])).unwrap();
        cpu_free_tensor(&mut alloc, id1).unwrap();

        let _id2 = cpu_alloc_tensor(&mut alloc, desc_f32(&[16])).unwrap();
        assert_eq!(alloc.stats.cache_hits, 0);
    }

    #[test]
    fn test_cache_disabled() {
        let mut cfg = default_config();
        cfg.cache_enabled = false;
        let mut alloc = create_tensor_allocator(cfg);
        let id = cpu_alloc_tensor(&mut alloc, desc_f32(&[8])).unwrap();
        cpu_free_tensor(&mut alloc, id).unwrap();
        assert!(alloc.cache.is_empty());
    }

    // -----------------------------------------------------------------------
    // Slab creation & growth
    // -----------------------------------------------------------------------

    #[test]
    fn test_slab_creation_correct_size() {
        let mut alloc = create_tensor_allocator(default_config());
        let idx = cpu_create_slab(&mut alloc, 1024).unwrap();
        assert!(alloc.slabs[idx].total_size >= 1024);
    }

    #[test]
    fn test_slab_initial_free_list() {
        let mut alloc = create_tensor_allocator(default_config());
        let idx = cpu_create_slab(&mut alloc, 1024).unwrap();
        assert_eq!(alloc.slabs[idx].free_list.len(), 1);
        let (off, len) = alloc.slabs[idx].free_list[0];
        assert_eq!(off, 0);
        assert_eq!(len, alloc.slabs[idx].total_size);
    }

    #[test]
    fn test_growth_factor_increases_slab_size() {
        let mut alloc = create_tensor_allocator(default_config());
        cpu_create_slab(&mut alloc, 100).unwrap();
        let first_size = alloc.slabs[0].total_size;
        cpu_create_slab(&mut alloc, 100).unwrap();
        let second_size = alloc.slabs[1].total_size;
        assert!(second_size >= (first_size as f32 * alloc.config.growth_factor) as usize);
    }

    #[test]
    fn test_slab_limit_reached() {
        let mut cfg = default_config();
        cfg.max_slab_count = 2;
        let mut alloc = create_tensor_allocator(cfg);
        cpu_create_slab(&mut alloc, 64).unwrap();
        cpu_create_slab(&mut alloc, 64).unwrap();
        let err = cpu_create_slab(&mut alloc, 64).unwrap_err();
        assert_eq!(err, AllocError::SlabLimitReached);
    }

    // -----------------------------------------------------------------------
    // Free-block finding & coalescing
    // -----------------------------------------------------------------------

    #[test]
    fn test_find_free_block_first_fit() {
        let slab = Slab {
            id: 0,
            total_size: 1024,
            used_size: 0,
            free_list: vec![(0, 128), (256, 512)],
            tensors: vec![],
        };
        // 64 bytes fits in first block.
        let (idx, offset) = cpu_find_free_block(&slab, 64, 1).unwrap();
        assert_eq!(idx, 0);
        assert_eq!(offset, 0);
    }

    #[test]
    fn test_find_free_block_skip_too_small() {
        let slab = Slab {
            id: 0,
            total_size: 1024,
            used_size: 0,
            free_list: vec![(0, 32), (256, 512)],
            tensors: vec![],
        };
        let (idx, _offset) = cpu_find_free_block(&slab, 64, 1).unwrap();
        assert_eq!(idx, 1);
    }

    #[test]
    fn test_find_free_block_none() {
        let slab = Slab {
            id: 0,
            total_size: 64,
            used_size: 64,
            free_list: vec![],
            tensors: vec![],
        };
        assert!(cpu_find_free_block(&slab, 64, 1).is_none());
    }

    #[test]
    fn test_find_free_block_alignment() {
        let slab = Slab {
            id: 0,
            total_size: 1024,
            used_size: 0,
            free_list: vec![(1, 256)],
            tensors: vec![],
        };
        let (_, offset) = cpu_find_free_block(&slab, 64, 64).unwrap();
        assert_eq!(offset % 64, 0);
        assert!(offset >= 1);
    }

    #[test]
    fn test_coalesce_adjacent_blocks() {
        let mut slab = Slab {
            id: 0,
            total_size: 1024,
            used_size: 0,
            free_list: vec![(0, 128), (128, 128), (256, 256)],
            tensors: vec![],
        };
        let merges = cpu_coalesce_free_blocks(&mut slab);
        assert_eq!(merges, 2);
        assert_eq!(slab.free_list.len(), 1);
        assert_eq!(slab.free_list[0], (0, 512));
    }

    #[test]
    fn test_coalesce_non_adjacent() {
        let mut slab = Slab {
            id: 0,
            total_size: 1024,
            used_size: 0,
            free_list: vec![(0, 64), (128, 64)],
            tensors: vec![],
        };
        let merges = cpu_coalesce_free_blocks(&mut slab);
        assert_eq!(merges, 0);
        assert_eq!(slab.free_list.len(), 2);
    }

    #[test]
    fn test_coalesce_empty() {
        let mut slab = Slab {
            id: 0,
            total_size: 1024,
            used_size: 0,
            free_list: vec![],
            tensors: vec![],
        };
        assert_eq!(cpu_coalesce_free_blocks(&mut slab), 0);
    }

    // -----------------------------------------------------------------------
    // Utilization
    // -----------------------------------------------------------------------

    #[test]
    fn test_utilization_empty() {
        let alloc = create_tensor_allocator(default_config());
        assert_eq!(cpu_compute_utilization(&alloc), 0.0);
    }

    #[test]
    fn test_utilization_after_alloc() {
        let mut alloc = create_tensor_allocator(default_config());
        cpu_alloc_tensor(&mut alloc, desc_f32(&[16])).unwrap();
        let u = cpu_compute_utilization(&alloc);
        assert!(u > 0.0);
        assert!(u <= 1.0);
    }

    #[test]
    fn test_utilization_bounded() {
        let mut alloc = create_tensor_allocator(default_config());
        // Fill with many allocations.
        for _ in 0..20 {
            let _ = cpu_alloc_tensor(&mut alloc, desc_f32(&[64]));
        }
        let u = cpu_compute_utilization(&alloc);
        assert!((0.0..=1.0).contains(&u));
    }

    // -----------------------------------------------------------------------
    // Reset
    // -----------------------------------------------------------------------

    #[test]
    fn test_reset_clears_everything() {
        let mut alloc = create_tensor_allocator(default_config());
        cpu_alloc_tensor(&mut alloc, desc_f32(&[8])).unwrap();
        cpu_reset_allocator(&mut alloc);
        assert!(alloc.slabs.is_empty());
        assert!(alloc.tensor_map.is_empty());
        assert!(alloc.cache.is_empty());
        assert_eq!(alloc.next_id, 1);
        assert_eq!(alloc.stats, AllocatorStats::default());
    }

    // -----------------------------------------------------------------------
    // Stats
    // -----------------------------------------------------------------------

    #[test]
    fn test_stats_slab_count() {
        let mut alloc = create_tensor_allocator(default_config());
        cpu_alloc_tensor(&mut alloc, desc_f32(&[8])).unwrap();
        let s = cpu_get_stats(&alloc);
        assert_eq!(s.slab_count, 1);
    }

    #[test]
    fn test_stats_peak_usage() {
        let mut alloc = create_tensor_allocator(default_config());
        let id = cpu_alloc_tensor(&mut alloc, desc_f32(&[8])).unwrap();
        let peak1 = cpu_get_stats(&alloc).peak_usage;
        cpu_free_tensor(&mut alloc, id).unwrap();
        let peak2 = cpu_get_stats(&alloc).peak_usage;
        // Peak should not decrease after free.
        assert_eq!(peak1, peak2);
    }

    // -----------------------------------------------------------------------
    // OOM
    // -----------------------------------------------------------------------

    #[test]
    fn test_oom_when_slab_limit_reached() {
        let mut cfg = default_config();
        cfg.max_slab_count = 1;
        cfg.initial_slab_size = 64;
        let mut alloc = create_tensor_allocator(cfg);

        // First alloc creates the slab; keep allocating until full.
        let mut ids = Vec::new();
        loop {
            match cpu_alloc_tensor(&mut alloc, desc_f32(&[16])) {
                Ok(id) => ids.push(id),
                Err(AllocError::SlabLimitReached) => break,
                Err(e) => panic!("unexpected error: {e:?}"),
            }
        }
        assert!(!ids.is_empty());
    }

    // -----------------------------------------------------------------------
    // Edge cases
    // -----------------------------------------------------------------------

    #[test]
    fn test_single_element_tensor() {
        let mut alloc = create_tensor_allocator(default_config());
        let id = cpu_alloc_tensor(&mut alloc, desc_f32(&[1])).unwrap();
        let info = cpu_get_tensor_info(&alloc, id).unwrap();
        assert_eq!(info.size_bytes, 4);
    }

    #[test]
    fn test_named_tensor() {
        let mut alloc = create_tensor_allocator(default_config());
        let desc = TensorDescriptor {
            shape: vec![4],
            dtype: TensorDType::F32,
            alignment: 64,
            name: Some("weights".to_string()),
        };
        let id = cpu_alloc_tensor(&mut alloc, desc).unwrap();
        let info = cpu_get_tensor_info(&alloc, id).unwrap();
        assert_eq!(info.descriptor.name.as_deref(), Some("weights"));
    }

    #[test]
    fn test_alloc_free_alloc_same_size() {
        let mut alloc = create_tensor_allocator(default_config());
        let id1 = cpu_alloc_tensor(&mut alloc, desc_f32(&[8])).unwrap();
        let size1 = cpu_get_tensor_info(&alloc, id1).unwrap().size_bytes;
        cpu_free_tensor(&mut alloc, id1).unwrap();
        let id2 = cpu_alloc_tensor(&mut alloc, desc_f32(&[8])).unwrap();
        let size2 = cpu_get_tensor_info(&alloc, id2).unwrap().size_bytes;
        assert_eq!(size1, size2);
    }

    // -----------------------------------------------------------------------
    // A770 specific
    // -----------------------------------------------------------------------

    #[test]
    fn test_a770_16gb_limit() {
        let cfg = AllocatorConfig {
            initial_slab_size: 16 * 1024 * 1024 * 1024, // 16 GiB
            max_slab_count: 1,
            growth_factor: 1.0,
            cache_enabled: true,
            default_alignment: 64,
        };
        let mut alloc = create_tensor_allocator(cfg);
        // Should be able to create one 16 GiB slab.
        let idx = cpu_create_slab(&mut alloc, 1).unwrap();
        assert_eq!(alloc.slabs[idx].total_size, 16 * 1024 * 1024 * 1024);
    }

    #[test]
    fn test_a770_64_byte_alignment() {
        let cfg = AllocatorConfig {
            initial_slab_size: 4096,
            max_slab_count: 4,
            growth_factor: 1.5,
            cache_enabled: true,
            default_alignment: 64,
        };
        let mut alloc = create_tensor_allocator(cfg);
        for _ in 0..10 {
            let id = cpu_alloc_tensor(&mut alloc, desc_f32(&[7])).unwrap();
            let info = cpu_get_tensor_info(&alloc, id).unwrap();
            assert_eq!(info.offset % 64, 0, "offset {} not 64-byte aligned", info.offset);
        }
    }

    // -----------------------------------------------------------------------
    // Format / display
    // -----------------------------------------------------------------------

    #[test]
    fn test_format_allocator_status() {
        let alloc = create_tensor_allocator(default_config());
        let s = format_allocator_status(&alloc);
        assert!(s.contains("TensorAllocator:"));
        assert!(s.contains("slabs=0"));
    }

    #[test]
    fn test_format_after_allocs() {
        let mut alloc = create_tensor_allocator(default_config());
        cpu_alloc_tensor(&mut alloc, desc_f32(&[8])).unwrap();
        let s = format_allocator_status(&alloc);
        assert!(s.contains("slabs=1"));
        assert!(s.contains("tensors=1"));
    }

    // -----------------------------------------------------------------------
    // AllocError display
    // -----------------------------------------------------------------------

    #[test]
    fn test_alloc_error_display_oom() {
        let e = AllocError::OutOfMemory { requested: 1024, available: 512 };
        let s = format!("{e}");
        assert!(s.contains("1024"));
        assert!(s.contains("512"));
    }

    #[test]
    fn test_alloc_error_display_invalid() {
        let e = AllocError::InvalidTensor(42);
        assert!(format!("{e}").contains("42"));
    }

    #[test]
    fn test_alloc_error_display_slab_limit() {
        let e = AllocError::SlabLimitReached;
        assert!(format!("{e}").contains("slab"));
    }

    #[test]
    fn test_alloc_error_display_fragmentation() {
        let e = AllocError::FragmentationTooHigh(0.85);
        assert!(format!("{e}").contains("0.85"));
    }

    // -----------------------------------------------------------------------
    // Multiple dtype allocations in one allocator
    // -----------------------------------------------------------------------

    #[test]
    fn test_mixed_dtype_allocations() {
        let mut alloc = create_tensor_allocator(default_config());
        let a = cpu_alloc_tensor(&mut alloc, desc_f32(&[16])).unwrap();
        let b = cpu_alloc_tensor(&mut alloc, desc_f16(&[16])).unwrap();
        let c = cpu_alloc_tensor(&mut alloc, desc_i8(&[16])).unwrap();
        let d = cpu_alloc_tensor(&mut alloc, desc_ternary(&[16])).unwrap();
        assert_eq!(alloc.tensor_map.len(), 4);
        for id in [a, b, c, d] {
            assert!(cpu_get_tensor_info(&alloc, id).is_some());
        }
    }
}
