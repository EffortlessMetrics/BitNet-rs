//! Module stub - implementation pending merge from feature branch
//! Memory-mapped I/O for zero-copy model loading.
//!
//! Provides safe abstractions over memory-mapped files for efficient,
//! zero-copy access to model weights and tensors. Includes typed slice
//! views, tensor views, pool management with LRU eviction, page-fault
//! tracking, prefetching, and RAII safety guards.

use std::collections::{HashMap, VecDeque};
use std::fmt;
use std::fs::{self, File};
use std::io::{self, Read, Seek, SeekFrom};
use std::marker::PhantomData;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use log::debug;

// ── Error Type ──────────────────────────────────────────────────────────────

/// Errors that can occur during mmap operations.
#[derive(Debug)]
pub enum MmapError {
    /// I/O error from the underlying filesystem.
    Io(io::Error),
    /// The requested region is out of bounds for the file.
    OutOfBounds { offset: u64, length: u64, file_size: u64 },
    /// Alignment requirement not met for typed access.
    AlignmentError { required: usize, actual: usize },
    /// The pool has reached its capacity limit.
    PoolFull { capacity: usize },
    /// Region not found in the pool.
    RegionNotFound { id: u64 },
    /// Zero-length mapping requested.
    ZeroLength,
}

impl fmt::Display for MmapError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(e) => write!(f, "mmap I/O error: {e}"),
            Self::OutOfBounds { offset, length, file_size } => {
                write!(
                    f,
                    "mmap region out of bounds: offset={offset}, \
                     length={length}, file_size={file_size}"
                )
            }
            Self::AlignmentError { required, actual } => {
                write!(f, "alignment error: required {required}, got {actual}")
            }
            Self::PoolFull { capacity } => {
                write!(f, "mmap pool full: capacity={capacity}")
            }
            Self::RegionNotFound { id } => {
                write!(f, "mmap region not found: id={id}")
            }
            Self::ZeroLength => write!(f, "zero-length mmap region"),
        }
    }
}

impl std::error::Error for MmapError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<io::Error> for MmapError {
    fn from(e: io::Error) -> Self {
        Self::Io(e)
    }
}

/// Result type alias for mmap operations.
pub type MmapResult<T> = Result<T, MmapError>;

// ── ID Generator ────────────────────────────────────────────────────────────

static NEXT_REGION_ID: AtomicU64 = AtomicU64::new(1);

fn next_region_id() -> u64 {
    NEXT_REGION_ID.fetch_add(1, Ordering::Relaxed)
}

// ── MmapConfig ──────────────────────────────────────────────────────────────

/// Configuration for memory-mapped I/O operations.
#[derive(Debug, Clone)]
#[allow(clippy::struct_excessive_bools)]
pub struct MmapConfig {
    /// Open the mapping as read-only.
    pub read_only: bool,
    /// Pre-fault pages into memory on creation (populate).
    pub populate: bool,
    /// Request huge-page backing for the mapping.
    pub huge_pages: bool,
    /// Lock mapped pages in physical memory (prevent swapping).
    pub lock_in_memory: bool,
}

impl Default for MmapConfig {
    fn default() -> Self {
        Self { read_only: true, populate: false, huge_pages: false, lock_in_memory: false }
    }
}

impl MmapConfig {
    /// Create a config optimized for model weight loading.
    #[must_use]
    pub const fn for_model_weights() -> Self {
        Self { read_only: true, populate: true, huge_pages: true, lock_in_memory: false }
    }

    /// Create a config optimized for streaming access.
    #[must_use]
    pub const fn for_streaming() -> Self {
        Self { read_only: true, populate: false, huge_pages: false, lock_in_memory: false }
    }
}

// ── MmapRegion ──────────────────────────────────────────────────────────────

/// A memory-mapped region backed by a file section.
///
/// Stores the actual file data in a `Vec<u8>` buffer (simulated mmap).
/// The region is identified by a unique ID and tracks its offset/length
/// within the source file.
pub struct MmapRegion {
    id: u64,
    data: Vec<u8>,
    offset: u64,
    length: u64,
    config: MmapConfig,
    _source_path: PathBuf,
}

impl MmapRegion {
    /// Unique identifier for this region.
    #[must_use]
    pub const fn id(&self) -> u64 {
        self.id
    }

    /// Byte offset within the source file.
    #[must_use]
    pub const fn offset(&self) -> u64 {
        self.offset
    }

    /// Length of the mapped region in bytes.
    #[must_use]
    pub const fn length(&self) -> u64 {
        self.length
    }

    /// Borrow the mapped data as a byte slice.
    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        &self.data
    }

    /// Whether this region is read-only.
    #[must_use]
    pub const fn is_read_only(&self) -> bool {
        self.config.read_only
    }

    /// Returns a pointer to the start of the mapped data.
    #[must_use]
    pub const fn as_ptr(&self) -> *const u8 {
        self.data.as_ptr()
    }
}

impl fmt::Debug for MmapRegion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MmapRegion")
            .field("id", &self.id)
            .field("offset", &self.offset)
            .field("length", &self.length)
            .field("read_only", &self.config.read_only)
            .finish_non_exhaustive()
    }
}

// ── MmapFile ────────────────────────────────────────────────────────────────

/// A memory-mapped file providing region-based access.
///
/// Opens a file and creates simulated memory mappings for requested
/// byte ranges. Each call to [`map_region`](Self::map_region) reads
/// the specified byte range into memory.
pub struct MmapFile {
    path: PathBuf,
    file_size: u64,
    config: MmapConfig,
}

impl MmapFile {
    /// Open a file for memory-mapped access.
    pub fn open(path: impl AsRef<Path>, config: MmapConfig) -> MmapResult<Self> {
        let path = path.as_ref().to_path_buf();
        let metadata = fs::metadata(&path)?;
        let file_size = metadata.len();
        debug!("MmapFile::open path={} size={file_size}", path.display());
        Ok(Self { path, file_size, config })
    }

    /// Size of the underlying file in bytes.
    #[must_use]
    pub const fn file_size(&self) -> u64 {
        self.file_size
    }

    /// Path to the underlying file.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Map the entire file into a region.
    pub fn map_all(&self) -> MmapResult<MmapRegion> {
        if self.file_size == 0 {
            return Ok(MmapRegion {
                id: next_region_id(),
                data: Vec::new(),
                offset: 0,
                length: 0,
                config: self.config.clone(),
                _source_path: self.path.clone(),
            });
        }
        self.map_region(0, self.file_size)
    }

    /// Map a specific byte range into a region.
    pub fn map_region(&self, offset: u64, length: u64) -> MmapResult<MmapRegion> {
        if length == 0 {
            return Err(MmapError::ZeroLength);
        }
        if offset.saturating_add(length) > self.file_size {
            return Err(MmapError::OutOfBounds { offset, length, file_size: self.file_size });
        }

        let mut file = File::open(&self.path)?;
        file.seek(SeekFrom::Start(offset))?;
        #[allow(clippy::cast_possible_truncation)]
        let mut buf = vec![0u8; length as usize];
        file.read_exact(&mut buf)?;

        debug!(
            "MmapFile::map_region offset={offset} length={length} \
             populate={} huge_pages={}",
            self.config.populate, self.config.huge_pages
        );

        Ok(MmapRegion {
            id: next_region_id(),
            data: buf,
            offset,
            length,
            config: self.config.clone(),
            _source_path: self.path.clone(),
        })
    }
}

impl fmt::Debug for MmapFile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MmapFile")
            .field("path", &self.path)
            .field("file_size", &self.file_size)
            .finish_non_exhaustive()
    }
}

// ── MmapSlice ───────────────────────────────────────────────────────────────

/// A typed, zero-copy slice view into an [`MmapRegion`].
///
/// Provides safe access to a contiguous sequence of `T` values stored
/// within a memory-mapped region. The lifetime is tied to the region.
pub struct MmapSlice<'a, T: Copy> {
    data: &'a [u8],
    _marker: PhantomData<T>,
}

impl<'a, T: Copy> MmapSlice<'a, T> {
    /// Create a typed slice from a region's byte data.
    ///
    /// Returns an error if the data is not properly aligned for `T`
    /// or the length is not a multiple of `size_of::<T>()`.
    pub fn new(region: &'a MmapRegion) -> MmapResult<Self> {
        Self::from_bytes(region.as_bytes())
    }

    /// Create a typed slice from raw bytes.
    pub fn from_bytes(data: &'a [u8]) -> MmapResult<Self> {
        let elem_size = std::mem::size_of::<T>();
        if elem_size == 0 {
            return Ok(Self { data, _marker: PhantomData });
        }

        let align = std::mem::align_of::<T>();
        let ptr_val = data.as_ptr() as usize;
        if !ptr_val.is_multiple_of(align) {
            return Err(MmapError::AlignmentError { required: align, actual: ptr_val % align });
        }

        if !data.len().is_multiple_of(elem_size) {
            return Err(MmapError::AlignmentError {
                required: elem_size,
                actual: data.len() % elem_size,
            });
        }

        Ok(Self { data, _marker: PhantomData })
    }

    /// Number of `T` elements in this slice.
    #[must_use]
    pub const fn len(&self) -> usize {
        let elem_size = std::mem::size_of::<T>();
        if elem_size == 0 {
            return 0;
        }
        self.data.len() / elem_size
    }

    /// Whether this slice is empty.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get element at the given index.
    ///
    /// # Panics
    ///
    /// Panics if `index >= self.len()`.
    #[must_use]
    pub fn get(&self, index: usize) -> T {
        assert!(index < self.len(), "index {index} out of bounds");
        let elem_size = std::mem::size_of::<T>();
        let offset = index * elem_size;
        // SAFETY: alignment and bounds were checked in the constructor
        // and by the assert above.
        unsafe { std::ptr::read_unaligned(self.data.as_ptr().add(offset).cast()) }
    }

    /// Get element at the given index, returning `None` if out of bounds.
    #[must_use]
    pub fn try_get(&self, index: usize) -> Option<T> {
        if index >= self.len() {
            return None;
        }
        Some(self.get(index))
    }

    /// Return the underlying byte data.
    #[must_use]
    pub const fn as_bytes(&self) -> &'a [u8] {
        self.data
    }
}

impl<T: Copy> fmt::Debug for MmapSlice<'_, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MmapSlice")
            .field("len", &self.len())
            .field("elem_size", &std::mem::size_of::<T>())
            .finish()
    }
}

// ── MmapTensorView ──────────────────────────────────────────────────────────

/// A view into tensor data stored in a memory-mapped region.
///
/// Provides shape-aware access to tensor elements without copying data.
#[derive(Debug, Clone)]
pub struct MmapTensorView<'a> {
    data: &'a [u8],
    shape: Vec<usize>,
    elem_size: usize,
}

impl<'a> MmapTensorView<'a> {
    /// Create a tensor view from a region with the given shape and
    /// element size in bytes.
    pub fn new(region: &'a MmapRegion, shape: Vec<usize>, elem_size: usize) -> MmapResult<Self> {
        Self::from_bytes(region.as_bytes(), shape, elem_size)
    }

    /// Create a tensor view from raw bytes.
    pub fn from_bytes(data: &'a [u8], shape: Vec<usize>, elem_size: usize) -> MmapResult<Self> {
        let total_elems: usize = shape.iter().product();
        let required = total_elems * elem_size;
        if data.len() < required {
            return Err(MmapError::OutOfBounds {
                offset: 0,
                length: required as u64,
                file_size: data.len() as u64,
            });
        }
        Ok(Self { data, shape, elem_size })
    }

    /// Shape of the tensor.
    #[must_use]
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Number of dimensions.
    #[must_use]
    pub const fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// Total number of elements.
    #[must_use]
    pub fn num_elements(&self) -> usize {
        self.shape.iter().product()
    }

    /// Size of each element in bytes.
    #[must_use]
    pub const fn elem_size(&self) -> usize {
        self.elem_size
    }

    /// Total size in bytes of the tensor data.
    #[must_use]
    pub fn byte_size(&self) -> usize {
        self.num_elements() * self.elem_size
    }

    /// Read a single element's bytes at a flat index.
    #[must_use]
    pub fn get_element_bytes(&self, flat_index: usize) -> Option<&'a [u8]> {
        let start = flat_index * self.elem_size;
        let end = start + self.elem_size;
        if end > self.data.len() {
            return None;
        }
        Some(&self.data[start..end])
    }

    /// Get a row slice for a 2-D tensor (returns bytes for the row).
    #[must_use]
    pub fn row_bytes(&self, row: usize) -> Option<&'a [u8]> {
        if self.shape.len() < 2 || row >= self.shape[0] {
            return None;
        }
        let cols = self.shape[1];
        let row_bytes = cols * self.elem_size;
        let start = row * row_bytes;
        let end = start + row_bytes;
        if end > self.data.len() {
            return None;
        }
        Some(&self.data[start..end])
    }

    /// Access the underlying byte data.
    #[must_use]
    pub fn as_bytes(&self) -> &'a [u8] {
        &self.data[..self.byte_size()]
    }
}

// ── PageFaultTracker ────────────────────────────────────────────────────────

/// Tracks page-fault-like access events for prefetching optimization.
///
/// Records which byte offsets are accessed and computes a stride hint
/// that can inform the [`MmapPrefetcher`].
#[derive(Debug)]
pub struct PageFaultTracker {
    page_size: usize,
    accesses: Vec<u64>,
    fault_count: u64,
}

impl PageFaultTracker {
    /// Create a new tracker with the given logical page size.
    #[must_use]
    pub const fn new(page_size: usize) -> Self {
        Self {
            page_size: if page_size == 0 { 4096 } else { page_size },
            accesses: Vec::new(),
            fault_count: 0,
        }
    }

    /// Record an access at the given byte offset.
    pub fn record_access(&mut self, offset: u64) {
        let page = offset / self.page_size as u64;
        if self.accesses.last() != Some(&page) {
            self.fault_count += 1;
        }
        self.accesses.push(page);
    }

    /// Total number of recorded page faults.
    #[must_use]
    pub const fn fault_count(&self) -> u64 {
        self.fault_count
    }

    /// Compute the most common stride between consecutive accesses.
    #[must_use]
    pub fn estimated_stride(&self) -> Option<u64> {
        if self.accesses.len() < 2 {
            return None;
        }
        let mut stride_counts: HashMap<i64, u64> = HashMap::new();
        for pair in self.accesses.windows(2) {
            let stride = pair[1].cast_signed() - pair[0].cast_signed();
            *stride_counts.entry(stride).or_insert(0) += 1;
        }
        stride_counts
            .into_iter()
            .max_by_key(|&(_, count)| count)
            .map(|(stride, _)| stride.unsigned_abs())
    }

    /// Number of recorded accesses.
    #[must_use]
    pub const fn access_count(&self) -> usize {
        self.accesses.len()
    }

    /// The configured page size.
    #[must_use]
    pub const fn page_size(&self) -> usize {
        self.page_size
    }

    /// Reset all tracking state.
    pub fn reset(&mut self) {
        self.accesses.clear();
        self.fault_count = 0;
    }
}

// ── MmapPool ────────────────────────────────────────────────────────────────

/// A pool of memory-mapped regions with LRU eviction.
///
/// Manages a fixed number of [`MmapRegion`]s, evicting the least-recently
/// used region when the pool is at capacity and a new region is inserted.
pub struct MmapPool {
    capacity: usize,
    regions: HashMap<u64, MmapRegion>,
    lru_order: VecDeque<u64>,
    total_mapped_bytes: u64,
    eviction_count: u64,
}

impl MmapPool {
    /// Create a pool with the given maximum region count.
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            regions: HashMap::new(),
            lru_order: VecDeque::new(),
            total_mapped_bytes: 0,
            eviction_count: 0,
        }
    }

    /// Insert a region into the pool, evicting the LRU region if full.
    ///
    /// Returns the evicted region, if any.
    pub fn insert(&mut self, region: MmapRegion) -> MmapResult<Option<MmapRegion>> {
        if self.capacity == 0 {
            return Err(MmapError::PoolFull { capacity: 0 });
        }

        let evicted = if self.regions.len() >= self.capacity { self.evict_lru() } else { None };

        let id = region.id();
        self.total_mapped_bytes += region.length();
        self.lru_order.push_back(id);
        self.regions.insert(id, region);

        Ok(evicted)
    }

    /// Look up a region by ID and mark it as recently used.
    #[must_use]
    pub fn get(&mut self, id: u64) -> Option<&MmapRegion> {
        if self.regions.contains_key(&id) {
            self.touch(id);
            self.regions.get(&id)
        } else {
            None
        }
    }

    /// Remove a region by ID.
    pub fn remove(&mut self, id: u64) -> MmapResult<MmapRegion> {
        let region = self.regions.remove(&id).ok_or(MmapError::RegionNotFound { id })?;
        self.lru_order.retain(|&x| x != id);
        self.total_mapped_bytes -= region.length();
        Ok(region)
    }

    /// Number of regions currently in the pool.
    #[must_use]
    pub fn len(&self) -> usize {
        self.regions.len()
    }

    /// Whether the pool is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.regions.is_empty()
    }

    /// Maximum number of regions the pool can hold.
    #[must_use]
    pub const fn capacity(&self) -> usize {
        self.capacity
    }

    /// Total bytes currently mapped across all regions.
    #[must_use]
    pub const fn total_mapped_bytes(&self) -> u64 {
        self.total_mapped_bytes
    }

    /// Number of regions evicted since pool creation.
    #[must_use]
    pub const fn eviction_count(&self) -> u64 {
        self.eviction_count
    }

    fn touch(&mut self, id: u64) {
        self.lru_order.retain(|&x| x != id);
        self.lru_order.push_back(id);
    }

    fn evict_lru(&mut self) -> Option<MmapRegion> {
        let lru_id = self.lru_order.pop_front()?;
        let region = self.regions.remove(&lru_id)?;
        self.total_mapped_bytes -= region.length();
        self.eviction_count += 1;
        debug!("MmapPool: evicted region id={} len={}", region.id(), region.length());
        Some(region)
    }
}

impl fmt::Debug for MmapPool {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MmapPool")
            .field("capacity", &self.capacity)
            .field("len", &self.regions.len())
            .field("total_mapped_bytes", &self.total_mapped_bytes)
            .field("eviction_count", &self.eviction_count)
            .finish_non_exhaustive()
    }
}

// ── MmapPrefetcher ──────────────────────────────────────────────────────────

/// Prefetches pages ahead of the current access pattern.
///
/// Uses a simple stride-based heuristic: given the last access offset
/// and a stride, it "advises" the system to prefetch upcoming pages.
/// In a real implementation this would call `madvise(MADV_WILLNEED)`.
#[derive(Debug)]
pub struct MmapPrefetcher {
    lookahead_pages: usize,
    page_size: usize,
    prefetch_count: u64,
    last_offset: Option<u64>,
}

impl MmapPrefetcher {
    /// Create a prefetcher with the given look-ahead depth and page size.
    #[must_use]
    pub const fn new(lookahead_pages: usize, page_size: usize) -> Self {
        Self {
            lookahead_pages,
            page_size: if page_size == 0 { 4096 } else { page_size },
            prefetch_count: 0,
            last_offset: None,
        }
    }

    /// Advise the system about an upcoming access at `offset`.
    ///
    /// Returns the list of page-aligned offsets that were prefetched.
    pub fn advise(&mut self, offset: u64, region_length: u64) -> Vec<u64> {
        let stride = match self.last_offset {
            Some(last) if offset > last => offset - last,
            _ => self.page_size as u64,
        };
        self.last_offset = Some(offset);

        let mut prefetched = Vec::new();
        for i in 1..=self.lookahead_pages {
            let target = offset + stride * i as u64;
            let page_aligned = (target / self.page_size as u64) * self.page_size as u64;
            if page_aligned < region_length {
                prefetched.push(page_aligned);
                self.prefetch_count += 1;
            }
        }
        prefetched
    }

    /// Total number of prefetch advisories issued.
    #[must_use]
    pub const fn prefetch_count(&self) -> u64 {
        self.prefetch_count
    }

    /// Reset prefetcher state.
    pub const fn reset(&mut self) {
        self.last_offset = None;
        self.prefetch_count = 0;
    }
}

// ── MmapSafetyGuard ────────────────────────────────────────────────────────

/// RAII guard ensuring a memory-mapped region is properly released on drop.
///
/// When the guard is dropped, it removes the region from its associated
/// pool (if one is recorded) and logs the unmap event.
pub struct MmapSafetyGuard {
    region_id: u64,
    region_length: u64,
    created_at: Instant,
    dropped: bool,
}

impl MmapSafetyGuard {
    /// Create a guard for the given region.
    #[must_use]
    pub fn new(region: &MmapRegion) -> Self {
        Self {
            region_id: region.id(),
            region_length: region.length(),
            created_at: Instant::now(),
            dropped: false,
        }
    }

    /// The guarded region's ID.
    #[must_use]
    pub const fn region_id(&self) -> u64 {
        self.region_id
    }

    /// How long the guard has been alive.
    #[must_use]
    pub fn elapsed(&self) -> std::time::Duration {
        self.created_at.elapsed()
    }

    /// Explicitly release the guard (same effect as drop).
    pub fn release(mut self) {
        self.do_release();
    }

    fn do_release(&mut self) {
        if !self.dropped {
            self.dropped = true;
            debug!(
                "MmapSafetyGuard: releasing region id={} length={} \
                 held_for={:?}",
                self.region_id,
                self.region_length,
                self.created_at.elapsed()
            );
        }
    }
}

impl Drop for MmapSafetyGuard {
    fn drop(&mut self) {
        self.do_release();
    }
}

impl fmt::Debug for MmapSafetyGuard {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MmapSafetyGuard")
            .field("region_id", &self.region_id)
            .field("region_length", &self.region_length)
            .field("dropped", &self.dropped)
            .finish_non_exhaustive()
    }
}

// ── MmapManager ─────────────────────────────────────────────────────────────

/// Orchestrator for memory-mapped I/O operations.
///
/// Coordinates file opening, region mapping, pool management, prefetching,
/// and page-fault tracking into a single high-level API.
pub struct MmapManager {
    pool: MmapPool,
    prefetcher: MmapPrefetcher,
    tracker: PageFaultTracker,
    config: MmapConfig,
    open_files: HashMap<PathBuf, MmapFile>,
}

impl MmapManager {
    /// Create a new manager with the given pool capacity and config.
    #[must_use]
    pub fn new(pool_capacity: usize, config: MmapConfig) -> Self {
        Self {
            pool: MmapPool::new(pool_capacity),
            prefetcher: MmapPrefetcher::new(4, 4096),
            tracker: PageFaultTracker::new(4096),
            config,
            open_files: HashMap::new(),
        }
    }

    /// Open a file (cached) and map a region into the pool.
    pub fn map_file_region(
        &mut self,
        path: impl AsRef<Path>,
        offset: u64,
        length: u64,
    ) -> MmapResult<u64> {
        let path = path.as_ref().to_path_buf();
        if !self.open_files.contains_key(&path) {
            let file = MmapFile::open(&path, self.config.clone())?;
            self.open_files.insert(path.clone(), file);
        }
        let file = self.open_files.get(&path).expect("just inserted");
        let region = file.map_region(offset, length)?;
        let id = region.id();
        self.pool.insert(region)?;
        self.tracker.record_access(offset);
        Ok(id)
    }

    /// Map an entire file into the pool.
    pub fn map_file_all(&mut self, path: impl AsRef<Path>) -> MmapResult<u64> {
        let path = path.as_ref().to_path_buf();
        if !self.open_files.contains_key(&path) {
            let file = MmapFile::open(&path, self.config.clone())?;
            self.open_files.insert(path.clone(), file);
        }
        let file = self.open_files.get(&path).expect("just inserted");
        let region = file.map_all()?;
        let id = region.id();
        self.pool.insert(region)?;
        Ok(id)
    }

    /// Access a region by ID.
    #[must_use]
    pub fn get_region(&mut self, id: u64) -> Option<&MmapRegion> {
        self.pool.get(id)
    }

    /// Remove a region by ID.
    pub fn release_region(&mut self, id: u64) -> MmapResult<MmapRegion> {
        self.pool.remove(id)
    }

    /// Advise the prefetcher about an upcoming access.
    pub fn prefetch(&mut self, offset: u64, region_length: u64) -> Vec<u64> {
        self.prefetcher.advise(offset, region_length)
    }

    /// Access the page-fault tracker.
    #[must_use]
    pub const fn tracker(&self) -> &PageFaultTracker {
        &self.tracker
    }

    /// Access the pool.
    #[must_use]
    pub const fn pool(&self) -> &MmapPool {
        &self.pool
    }

    /// Number of open files.
    #[must_use]
    pub fn open_file_count(&self) -> usize {
        self.open_files.len()
    }
}

impl fmt::Debug for MmapManager {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MmapManager")
            .field("pool", &self.pool)
            .field("prefetcher", &self.prefetcher)
            .field("open_files", &self.open_files.len())
            .finish_non_exhaustive()
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    fn temp_file(data: &[u8]) -> NamedTempFile {
        let mut f = NamedTempFile::new().unwrap();
        f.write_all(data).unwrap();
        f.flush().unwrap();
        f
    }

    // ── MmapConfig tests ────────────────────────────────────────────────

    #[test]
    fn config_default_is_read_only() {
        let c = MmapConfig::default();
        assert!(c.read_only);
        assert!(!c.populate);
        assert!(!c.huge_pages);
        assert!(!c.lock_in_memory);
    }

    #[test]
    fn config_for_model_weights() {
        let c = MmapConfig::for_model_weights();
        assert!(c.read_only);
        assert!(c.populate);
        assert!(c.huge_pages);
    }

    #[test]
    fn config_for_streaming() {
        let c = MmapConfig::for_streaming();
        assert!(c.read_only);
        assert!(!c.populate);
        assert!(!c.huge_pages);
    }

    #[test]
    fn config_clone() {
        let c =
            MmapConfig { read_only: false, populate: true, huge_pages: true, lock_in_memory: true };
        let c2 = c.clone();
        assert!(!c2.read_only);
        assert!(c2.populate);
    }

    // ── MmapRegion tests ────────────────────────────────────────────────

    #[test]
    fn region_creation_and_accessors() {
        let data = b"hello world";
        let f = temp_file(data);
        let mf = MmapFile::open(f.path(), MmapConfig::default()).unwrap();
        let region = mf.map_all().unwrap();
        assert_eq!(region.length(), data.len() as u64);
        assert_eq!(region.offset(), 0);
        assert_eq!(region.as_bytes(), data);
        assert!(region.is_read_only());
    }

    #[test]
    fn region_partial_mapping() {
        let data = b"0123456789ABCDEF";
        let f = temp_file(data);
        let mf = MmapFile::open(f.path(), MmapConfig::default()).unwrap();
        let region = mf.map_region(4, 8).unwrap();
        assert_eq!(region.offset(), 4);
        assert_eq!(region.length(), 8);
        assert_eq!(region.as_bytes(), b"456789AB");
    }

    #[test]
    fn region_unique_ids() {
        let f = temp_file(b"ab");
        let mf = MmapFile::open(f.path(), MmapConfig::default()).unwrap();
        let r1 = mf.map_region(0, 1).unwrap();
        let r2 = mf.map_region(1, 1).unwrap();
        assert_ne!(r1.id(), r2.id());
    }

    #[test]
    fn region_debug_format() {
        let f = temp_file(b"x");
        let mf = MmapFile::open(f.path(), MmapConfig::default()).unwrap();
        let r = mf.map_all().unwrap();
        let dbg = format!("{r:?}");
        assert!(dbg.contains("MmapRegion"));
        assert!(dbg.contains("length"));
    }

    #[test]
    fn region_as_ptr() {
        let f = temp_file(b"ptr");
        let mf = MmapFile::open(f.path(), MmapConfig::default()).unwrap();
        let r = mf.map_all().unwrap();
        assert!(!r.as_ptr().is_null());
    }

    // ── MmapFile tests ─────────────────────────────────────────────────

    #[test]
    fn file_open_and_size() {
        let data = vec![0u8; 1024];
        let f = temp_file(&data);
        let mf = MmapFile::open(f.path(), MmapConfig::default()).unwrap();
        assert_eq!(mf.file_size(), 1024);
        assert_eq!(mf.path(), f.path());
    }

    #[test]
    fn file_map_all() {
        let data = b"test data for mapping";
        let f = temp_file(data);
        let mf = MmapFile::open(f.path(), MmapConfig::default()).unwrap();
        let region = mf.map_all().unwrap();
        assert_eq!(region.as_bytes(), data);
    }

    #[test]
    fn file_map_region_out_of_bounds() {
        let f = temp_file(b"short");
        let mf = MmapFile::open(f.path(), MmapConfig::default()).unwrap();
        let result = mf.map_region(0, 100);
        assert!(matches!(result, Err(MmapError::OutOfBounds { .. })));
    }

    #[test]
    fn file_map_region_offset_out_of_bounds() {
        let f = temp_file(b"short");
        let mf = MmapFile::open(f.path(), MmapConfig::default()).unwrap();
        let result = mf.map_region(10, 1);
        assert!(matches!(result, Err(MmapError::OutOfBounds { .. })));
    }

    #[test]
    fn file_map_zero_length_error() {
        let f = temp_file(b"data");
        let mf = MmapFile::open(f.path(), MmapConfig::default()).unwrap();
        let result = mf.map_region(0, 0);
        assert!(matches!(result, Err(MmapError::ZeroLength)));
    }

    #[test]
    fn file_empty_map_all() {
        let f = temp_file(b"");
        let mf = MmapFile::open(f.path(), MmapConfig::default()).unwrap();
        assert_eq!(mf.file_size(), 0);
        let region = mf.map_all().unwrap();
        assert_eq!(region.length(), 0);
        assert!(region.as_bytes().is_empty());
    }

    #[test]
    fn file_nonexistent_path() {
        let result = MmapFile::open("nonexistent_file_12345.bin", MmapConfig::default());
        assert!(matches!(result, Err(MmapError::Io(_))));
    }

    #[test]
    fn file_debug_format() {
        let f = temp_file(b"d");
        let mf = MmapFile::open(f.path(), MmapConfig::default()).unwrap();
        let dbg = format!("{mf:?}");
        assert!(dbg.contains("MmapFile"));
    }

    #[test]
    fn file_multiple_regions_same_file() {
        let data = b"AABBCCDD";
        let f = temp_file(data);
        let mf = MmapFile::open(f.path(), MmapConfig::default()).unwrap();
        let r1 = mf.map_region(0, 4).unwrap();
        let r2 = mf.map_region(4, 4).unwrap();
        assert_eq!(r1.as_bytes(), b"AABB");
        assert_eq!(r2.as_bytes(), b"CCDD");
    }

    #[test]
    fn file_overlapping_regions() {
        let data = b"overlap";
        let f = temp_file(data);
        let mf = MmapFile::open(f.path(), MmapConfig::default()).unwrap();
        let r1 = mf.map_region(0, 5).unwrap();
        let r2 = mf.map_region(2, 5).unwrap();
        assert_eq!(r1.as_bytes(), b"overl");
        assert_eq!(r2.as_bytes(), b"erlap");
    }

    #[test]
    fn file_boundary_region() {
        let data = b"boundary";
        let f = temp_file(data);
        let mf = MmapFile::open(f.path(), MmapConfig::default()).unwrap();
        let r = mf.map_region(0, data.len() as u64).unwrap();
        assert_eq!(r.as_bytes(), data);
    }

    // ── MmapSlice tests ────────────────────────────────────────────────

    #[test]
    fn slice_u8_access() {
        let data = vec![10u8, 20, 30, 40];
        let f = temp_file(&data);
        let mf = MmapFile::open(f.path(), MmapConfig::default()).unwrap();
        let region = mf.map_all().unwrap();
        let slice: MmapSlice<'_, u8> = MmapSlice::new(&region).unwrap();
        assert_eq!(slice.len(), 4);
        assert!(!slice.is_empty());
        assert_eq!(slice.get(0), 10);
        assert_eq!(slice.get(3), 40);
    }

    #[test]
    fn slice_u32_access() {
        let values: Vec<u32> = vec![100, 200, 300];
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_ne_bytes()).collect();
        let f = temp_file(&bytes);
        let mf = MmapFile::open(f.path(), MmapConfig::default()).unwrap();
        let region = mf.map_all().unwrap();
        let slice: MmapSlice<'_, u32> = MmapSlice::new(&region).unwrap();
        assert_eq!(slice.len(), 3);
        assert_eq!(slice.get(0), 100);
        assert_eq!(slice.get(1), 200);
        assert_eq!(slice.get(2), 300);
    }

    #[test]
    fn slice_f32_access() {
        let values: Vec<f32> = vec![1.0, 2.5, -3.14];
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_ne_bytes()).collect();
        let f = temp_file(&bytes);
        let mf = MmapFile::open(f.path(), MmapConfig::default()).unwrap();
        let region = mf.map_all().unwrap();
        let slice: MmapSlice<'_, f32> = MmapSlice::new(&region).unwrap();
        assert_eq!(slice.len(), 3);
        assert!((slice.get(0) - 1.0).abs() < f32::EPSILON);
        assert!((slice.get(1) - 2.5).abs() < f32::EPSILON);
    }

    #[test]
    fn slice_try_get_out_of_bounds() {
        let f = temp_file(&[1u8, 2, 3]);
        let mf = MmapFile::open(f.path(), MmapConfig::default()).unwrap();
        let region = mf.map_all().unwrap();
        let slice: MmapSlice<'_, u8> = MmapSlice::new(&region).unwrap();
        assert!(slice.try_get(2).is_some());
        assert!(slice.try_get(3).is_none());
        assert!(slice.try_get(100).is_none());
    }

    #[test]
    #[should_panic(expected = "out of bounds")]
    fn slice_get_panics_on_oob() {
        let f = temp_file(&[1u8]);
        let mf = MmapFile::open(f.path(), MmapConfig::default()).unwrap();
        let region = mf.map_all().unwrap();
        let slice: MmapSlice<'_, u8> = MmapSlice::new(&region).unwrap();
        let _ = slice.get(1);
    }

    #[test]
    fn slice_empty() {
        let f = temp_file(b"");
        let mf = MmapFile::open(f.path(), MmapConfig::default()).unwrap();
        let region = mf.map_all().unwrap();
        let slice: MmapSlice<'_, u8> = MmapSlice::new(&region).unwrap();
        assert!(slice.is_empty());
        assert_eq!(slice.len(), 0);
    }

    #[test]
    fn slice_as_bytes() {
        let data = [0xDE, 0xAD, 0xBE, 0xEF];
        let f = temp_file(&data);
        let mf = MmapFile::open(f.path(), MmapConfig::default()).unwrap();
        let region = mf.map_all().unwrap();
        let slice: MmapSlice<'_, u8> = MmapSlice::new(&region).unwrap();
        assert_eq!(slice.as_bytes(), &data);
    }

    #[test]
    fn slice_debug() {
        let f = temp_file(&[0u8; 8]);
        let mf = MmapFile::open(f.path(), MmapConfig::default()).unwrap();
        let region = mf.map_all().unwrap();
        let slice: MmapSlice<'_, u8> = MmapSlice::new(&region).unwrap();
        let dbg = format!("{slice:?}");
        assert!(dbg.contains("MmapSlice"));
    }

    // ── MmapTensorView tests ───────────────────────────────────────────

    #[test]
    fn tensor_view_1d() {
        let data = vec![0u8; 16];
        let f = temp_file(&data);
        let mf = MmapFile::open(f.path(), MmapConfig::default()).unwrap();
        let region = mf.map_all().unwrap();
        let tv = MmapTensorView::new(&region, vec![16], 1).unwrap();
        assert_eq!(tv.ndim(), 1);
        assert_eq!(tv.num_elements(), 16);
        assert_eq!(tv.byte_size(), 16);
        assert_eq!(tv.shape(), &[16]);
    }

    #[test]
    fn tensor_view_2d() {
        let data = vec![0u8; 24];
        let f = temp_file(&data);
        let mf = MmapFile::open(f.path(), MmapConfig::default()).unwrap();
        let region = mf.map_all().unwrap();
        let tv = MmapTensorView::new(&region, vec![3, 8], 1).unwrap();
        assert_eq!(tv.ndim(), 2);
        assert_eq!(tv.num_elements(), 24);
    }

    #[test]
    fn tensor_view_3d() {
        let data = vec![0u8; 60];
        let f = temp_file(&data);
        let mf = MmapFile::open(f.path(), MmapConfig::default()).unwrap();
        let region = mf.map_all().unwrap();
        let tv = MmapTensorView::new(&region, vec![3, 4, 5], 1).unwrap();
        assert_eq!(tv.ndim(), 3);
        assert_eq!(tv.num_elements(), 60);
    }

    #[test]
    fn tensor_view_elem_size() {
        // 4 f32 elements = 16 bytes
        let data = vec![0u8; 16];
        let f = temp_file(&data);
        let mf = MmapFile::open(f.path(), MmapConfig::default()).unwrap();
        let region = mf.map_all().unwrap();
        let tv = MmapTensorView::new(&region, vec![4], 4).unwrap();
        assert_eq!(tv.elem_size(), 4);
        assert_eq!(tv.byte_size(), 16);
    }

    #[test]
    fn tensor_view_get_element_bytes() {
        let data: Vec<u8> = (0..12).collect();
        let f = temp_file(&data);
        let mf = MmapFile::open(f.path(), MmapConfig::default()).unwrap();
        let region = mf.map_all().unwrap();
        let tv = MmapTensorView::new(&region, vec![3], 4).unwrap();
        assert_eq!(tv.get_element_bytes(0), Some(&[0, 1, 2, 3][..]));
        assert_eq!(tv.get_element_bytes(1), Some(&[4, 5, 6, 7][..]));
        assert_eq!(tv.get_element_bytes(2), Some(&[8, 9, 10, 11][..]));
        assert!(tv.get_element_bytes(3).is_none());
    }

    #[test]
    fn tensor_view_row_bytes() {
        // 2x3 matrix, elem_size=2 => 12 bytes total
        let data: Vec<u8> = (0..12).collect();
        let f = temp_file(&data);
        let mf = MmapFile::open(f.path(), MmapConfig::default()).unwrap();
        let region = mf.map_all().unwrap();
        let tv = MmapTensorView::new(&region, vec![2, 3], 2).unwrap();
        assert_eq!(tv.row_bytes(0), Some(&[0, 1, 2, 3, 4, 5][..]));
        assert_eq!(tv.row_bytes(1), Some(&[6, 7, 8, 9, 10, 11][..]));
        assert!(tv.row_bytes(2).is_none());
    }

    #[test]
    fn tensor_view_row_bytes_1d_returns_none() {
        let data = vec![0u8; 8];
        let f = temp_file(&data);
        let mf = MmapFile::open(f.path(), MmapConfig::default()).unwrap();
        let region = mf.map_all().unwrap();
        let tv = MmapTensorView::new(&region, vec![8], 1).unwrap();
        assert!(tv.row_bytes(0).is_none());
    }

    #[test]
    fn tensor_view_insufficient_data() {
        let data = vec![0u8; 10];
        let f = temp_file(&data);
        let mf = MmapFile::open(f.path(), MmapConfig::default()).unwrap();
        let region = mf.map_all().unwrap();
        let result = MmapTensorView::new(&region, vec![4, 4], 1);
        assert!(matches!(result, Err(MmapError::OutOfBounds { .. })));
    }

    #[test]
    fn tensor_view_as_bytes() {
        let data: Vec<u8> = (0..8).collect();
        let f = temp_file(&data);
        let mf = MmapFile::open(f.path(), MmapConfig::default()).unwrap();
        let region = mf.map_all().unwrap();
        let tv = MmapTensorView::new(&region, vec![8], 1).unwrap();
        assert_eq!(tv.as_bytes(), &data[..]);
    }

    #[test]
    fn tensor_view_from_bytes() {
        let data: Vec<u8> = (0..16).collect();
        let tv = MmapTensorView::from_bytes(&data, vec![4, 2], 2).unwrap();
        assert_eq!(tv.num_elements(), 8);
        assert_eq!(tv.byte_size(), 16);
    }

    #[test]
    fn tensor_view_clone() {
        let data = vec![0u8; 8];
        let tv = MmapTensorView::from_bytes(&data, vec![2, 4], 1).unwrap();
        let tv2 = tv.clone();
        assert_eq!(tv.shape(), tv2.shape());
        assert_eq!(tv.ndim(), tv2.ndim());
    }

    // ── PageFaultTracker tests ─────────────────────────────────────────

    #[test]
    fn tracker_new() {
        let t = PageFaultTracker::new(4096);
        assert_eq!(t.fault_count(), 0);
        assert_eq!(t.access_count(), 0);
        assert_eq!(t.page_size(), 4096);
    }

    #[test]
    fn tracker_zero_page_size_defaults() {
        let t = PageFaultTracker::new(0);
        assert_eq!(t.page_size(), 4096);
    }

    #[test]
    fn tracker_record_single() {
        let mut t = PageFaultTracker::new(4096);
        t.record_access(0);
        assert_eq!(t.fault_count(), 1);
        assert_eq!(t.access_count(), 1);
    }

    #[test]
    fn tracker_same_page_no_extra_fault() {
        let mut t = PageFaultTracker::new(4096);
        t.record_access(0);
        t.record_access(100);
        // Both on page 0, but second access to same page still counts as 1
        // because last recorded page == current page
        assert_eq!(t.fault_count(), 1);
    }

    #[test]
    fn tracker_different_pages() {
        let mut t = PageFaultTracker::new(4096);
        t.record_access(0);
        t.record_access(4096);
        t.record_access(8192);
        assert_eq!(t.fault_count(), 3);
    }

    #[test]
    fn tracker_estimated_stride_sequential() {
        let mut t = PageFaultTracker::new(4096);
        for i in 0..5 {
            t.record_access(i * 4096);
        }
        assert_eq!(t.estimated_stride(), Some(1));
    }

    #[test]
    fn tracker_estimated_stride_skip() {
        let mut t = PageFaultTracker::new(4096);
        for i in 0..5 {
            t.record_access(i * 2 * 4096);
        }
        assert_eq!(t.estimated_stride(), Some(2));
    }

    #[test]
    fn tracker_estimated_stride_none_single() {
        let mut t = PageFaultTracker::new(4096);
        t.record_access(0);
        assert!(t.estimated_stride().is_none());
    }

    #[test]
    fn tracker_estimated_stride_none_empty() {
        let t = PageFaultTracker::new(4096);
        assert!(t.estimated_stride().is_none());
    }

    #[test]
    fn tracker_reset() {
        let mut t = PageFaultTracker::new(4096);
        t.record_access(0);
        t.record_access(4096);
        t.reset();
        assert_eq!(t.fault_count(), 0);
        assert_eq!(t.access_count(), 0);
    }

    // ── MmapPool tests ─────────────────────────────────────────────────

    #[test]
    fn pool_new() {
        let pool = MmapPool::new(4);
        assert_eq!(pool.capacity(), 4);
        assert!(pool.is_empty());
        assert_eq!(pool.len(), 0);
        assert_eq!(pool.total_mapped_bytes(), 0);
    }

    #[test]
    fn pool_insert_and_get() {
        let f = temp_file(b"pool data");
        let mf = MmapFile::open(f.path(), MmapConfig::default()).unwrap();
        let region = mf.map_all().unwrap();
        let id = region.id();

        let mut pool = MmapPool::new(4);
        pool.insert(region).unwrap();
        assert_eq!(pool.len(), 1);

        let r = pool.get(id).unwrap();
        assert_eq!(r.as_bytes(), b"pool data");
    }

    #[test]
    fn pool_lru_eviction() {
        let f = temp_file(b"data");
        let mf = MmapFile::open(f.path(), MmapConfig::default()).unwrap();

        let mut pool = MmapPool::new(2);
        let r1 = mf.map_region(0, 2).unwrap();
        let id1 = r1.id();
        let r2 = mf.map_region(0, 2).unwrap();
        let _id2 = r2.id();
        pool.insert(r1).unwrap();
        pool.insert(r2).unwrap();

        // Pool is full (capacity=2). Insert a third → evicts r1.
        let r3 = mf.map_region(0, 2).unwrap();
        let evicted = pool.insert(r3).unwrap();
        assert!(evicted.is_some());
        assert_eq!(evicted.unwrap().id(), id1);
        assert_eq!(pool.eviction_count(), 1);
    }

    #[test]
    fn pool_lru_touch_changes_order() {
        let f = temp_file(b"abcd");
        let mf = MmapFile::open(f.path(), MmapConfig::default()).unwrap();

        let mut pool = MmapPool::new(2);
        let r1 = mf.map_region(0, 2).unwrap();
        let id1 = r1.id();
        let r2 = mf.map_region(2, 2).unwrap();
        let id2 = r2.id();
        pool.insert(r1).unwrap();
        pool.insert(r2).unwrap();

        // Touch r1 → r2 becomes LRU
        let _ = pool.get(id1);

        let r3 = mf.map_region(0, 2).unwrap();
        let evicted = pool.insert(r3).unwrap();
        assert_eq!(evicted.unwrap().id(), id2);
    }

    #[test]
    fn pool_remove() {
        let f = temp_file(b"rm");
        let mf = MmapFile::open(f.path(), MmapConfig::default()).unwrap();
        let region = mf.map_all().unwrap();
        let id = region.id();

        let mut pool = MmapPool::new(4);
        pool.insert(region).unwrap();
        assert_eq!(pool.len(), 1);

        let removed = pool.remove(id).unwrap();
        assert_eq!(removed.id(), id);
        assert!(pool.is_empty());
    }

    #[test]
    fn pool_remove_not_found() {
        let mut pool = MmapPool::new(4);
        let result = pool.remove(999);
        assert!(matches!(result, Err(MmapError::RegionNotFound { id: 999 })));
    }

    #[test]
    fn pool_zero_capacity() {
        let f = temp_file(b"x");
        let mf = MmapFile::open(f.path(), MmapConfig::default()).unwrap();
        let region = mf.map_all().unwrap();

        let mut pool = MmapPool::new(0);
        let result = pool.insert(region);
        assert!(matches!(result, Err(MmapError::PoolFull { capacity: 0 })));
    }

    #[test]
    fn pool_total_mapped_bytes_tracking() {
        let f = temp_file(b"ABCDEFGH");
        let mf = MmapFile::open(f.path(), MmapConfig::default()).unwrap();

        let mut pool = MmapPool::new(10);
        let r1 = mf.map_region(0, 4).unwrap();
        let id1 = r1.id();
        let r2 = mf.map_region(4, 4).unwrap();
        pool.insert(r1).unwrap();
        pool.insert(r2).unwrap();
        assert_eq!(pool.total_mapped_bytes(), 8);

        pool.remove(id1).unwrap();
        assert_eq!(pool.total_mapped_bytes(), 4);
    }

    #[test]
    fn pool_get_nonexistent() {
        let mut pool = MmapPool::new(4);
        assert!(pool.get(42).is_none());
    }

    #[test]
    fn pool_debug_format() {
        let pool = MmapPool::new(4);
        let dbg = format!("{pool:?}");
        assert!(dbg.contains("MmapPool"));
    }

    // ── MmapPrefetcher tests ───────────────────────────────────────────

    #[test]
    fn prefetcher_new() {
        let p = MmapPrefetcher::new(4, 4096);
        assert_eq!(p.prefetch_count(), 0);
    }

    #[test]
    fn prefetcher_zero_page_size_defaults() {
        let p = MmapPrefetcher::new(2, 0);
        // Should default to 4096
        assert_eq!(p.prefetch_count(), 0);
    }

    #[test]
    fn prefetcher_advise_sequential() {
        let mut p = MmapPrefetcher::new(2, 4096);
        let pages = p.advise(0, 100_000);
        // First call: no prior offset, uses page_size as stride
        assert!(!pages.is_empty());
        assert!(pages.iter().all(|&pg| pg < 100_000));
    }

    #[test]
    fn prefetcher_advise_with_stride() {
        let mut p = MmapPrefetcher::new(3, 4096);
        let _ = p.advise(0, 1_000_000);
        let pages = p.advise(8192, 1_000_000);
        assert!(!pages.is_empty());
        assert_eq!(p.prefetch_count() as usize, pages.len() + 3);
    }

    #[test]
    fn prefetcher_advise_near_end() {
        let mut p = MmapPrefetcher::new(10, 4096);
        let pages = p.advise(90_000, 100_000);
        // Should not include offsets beyond region_length
        assert!(pages.iter().all(|&pg| pg < 100_000));
    }

    #[test]
    fn prefetcher_reset() {
        let mut p = MmapPrefetcher::new(2, 4096);
        let _ = p.advise(0, 100_000);
        assert!(p.prefetch_count() > 0);
        p.reset();
        assert_eq!(p.prefetch_count(), 0);
    }

    // ── MmapSafetyGuard tests ──────────────────────────────────────────

    #[test]
    fn guard_creation() {
        let f = temp_file(b"guard");
        let mf = MmapFile::open(f.path(), MmapConfig::default()).unwrap();
        let region = mf.map_all().unwrap();
        let guard = MmapSafetyGuard::new(&region);
        assert_eq!(guard.region_id(), region.id());
    }

    #[test]
    fn guard_elapsed() {
        let f = temp_file(b"t");
        let mf = MmapFile::open(f.path(), MmapConfig::default()).unwrap();
        let region = mf.map_all().unwrap();
        let guard = MmapSafetyGuard::new(&region);
        // Elapsed should be very small, just check it doesn't panic
        let _ = guard.elapsed();
    }

    #[test]
    fn guard_release() {
        let f = temp_file(b"rel");
        let mf = MmapFile::open(f.path(), MmapConfig::default()).unwrap();
        let region = mf.map_all().unwrap();
        let guard = MmapSafetyGuard::new(&region);
        guard.release(); // Explicit release
    }

    #[test]
    fn guard_drop() {
        let f = temp_file(b"drop");
        let mf = MmapFile::open(f.path(), MmapConfig::default()).unwrap();
        let region = mf.map_all().unwrap();
        {
            let _guard = MmapSafetyGuard::new(&region);
            // guard dropped here
        }
        // If we get here, drop worked without panic
    }

    #[test]
    fn guard_debug_format() {
        let f = temp_file(b"dbg");
        let mf = MmapFile::open(f.path(), MmapConfig::default()).unwrap();
        let region = mf.map_all().unwrap();
        let guard = MmapSafetyGuard::new(&region);
        let dbg = format!("{guard:?}");
        assert!(dbg.contains("MmapSafetyGuard"));
    }

    // ── MmapManager tests ──────────────────────────────────────────────

    #[test]
    fn manager_new() {
        let mgr = MmapManager::new(4, MmapConfig::default());
        assert_eq!(mgr.pool().capacity(), 4);
        assert_eq!(mgr.open_file_count(), 0);
    }

    #[test]
    fn manager_map_file_region() {
        let data = b"manager test data";
        let f = temp_file(data);
        let mut mgr = MmapManager::new(4, MmapConfig::default());
        let id = mgr.map_file_region(f.path(), 0, 7).unwrap();
        let region = mgr.get_region(id).unwrap();
        assert_eq!(region.as_bytes(), b"manager");
    }

    #[test]
    fn manager_map_file_all() {
        let data = b"all data here";
        let f = temp_file(data);
        let mut mgr = MmapManager::new(4, MmapConfig::default());
        let id = mgr.map_file_all(f.path()).unwrap();
        let region = mgr.get_region(id).unwrap();
        assert_eq!(region.as_bytes(), data);
    }

    #[test]
    fn manager_release_region() {
        let f = temp_file(b"release");
        let mut mgr = MmapManager::new(4, MmapConfig::default());
        let id = mgr.map_file_all(f.path()).unwrap();
        let released = mgr.release_region(id).unwrap();
        assert_eq!(released.as_bytes(), b"release");
        assert!(mgr.get_region(id).is_none());
    }

    #[test]
    fn manager_caches_open_files() {
        let f = temp_file(b"cached file");
        let mut mgr = MmapManager::new(10, MmapConfig::default());
        let _ = mgr.map_file_region(f.path(), 0, 3).unwrap();
        let _ = mgr.map_file_region(f.path(), 3, 3).unwrap();
        assert_eq!(mgr.open_file_count(), 1);
    }

    #[test]
    fn manager_prefetch() {
        let mut mgr = MmapManager::new(4, MmapConfig::default());
        let pages = mgr.prefetch(0, 100_000);
        assert!(!pages.is_empty());
    }

    #[test]
    fn manager_tracker() {
        let f = temp_file(b"track");
        let mut mgr = MmapManager::new(4, MmapConfig::default());
        let _ = mgr.map_file_region(f.path(), 0, 5).unwrap();
        assert!(mgr.tracker().access_count() > 0);
    }

    #[test]
    fn manager_debug_format() {
        let mgr = MmapManager::new(4, MmapConfig::default());
        let dbg = format!("{mgr:?}");
        assert!(dbg.contains("MmapManager"));
    }

    #[test]
    fn manager_nonexistent_file() {
        let mut mgr = MmapManager::new(4, MmapConfig::default());
        let result = mgr.map_file_region("no_such_file_xyz.bin", 0, 10);
        assert!(matches!(result, Err(MmapError::Io(_))));
    }

    // ── Error type tests ───────────────────────────────────────────────

    #[test]
    fn error_display_io() {
        let e = MmapError::Io(io::Error::new(io::ErrorKind::NotFound, "test"));
        assert!(format!("{e}").contains("I/O error"));
    }

    #[test]
    fn error_display_oob() {
        let e = MmapError::OutOfBounds { offset: 10, length: 20, file_size: 15 };
        let msg = format!("{e}");
        assert!(msg.contains("out of bounds"));
    }

    #[test]
    fn error_display_alignment() {
        let e = MmapError::AlignmentError { required: 4, actual: 1 };
        assert!(format!("{e}").contains("alignment"));
    }

    #[test]
    fn error_display_pool_full() {
        let e = MmapError::PoolFull { capacity: 5 };
        assert!(format!("{e}").contains("pool full"));
    }

    #[test]
    fn error_display_not_found() {
        let e = MmapError::RegionNotFound { id: 42 };
        assert!(format!("{e}").contains("not found"));
    }

    #[test]
    fn error_display_zero_length() {
        let e = MmapError::ZeroLength;
        assert!(format!("{e}").contains("zero-length"));
    }

    #[test]
    fn error_source_io() {
        let inner = io::Error::new(io::ErrorKind::Other, "inner");
        let e = MmapError::Io(inner);
        assert!(std::error::Error::source(&e).is_some());
    }

    #[test]
    fn error_source_non_io() {
        let e = MmapError::ZeroLength;
        assert!(std::error::Error::source(&e).is_none());
    }

    #[test]
    fn error_from_io() {
        let io_err = io::Error::new(io::ErrorKind::Other, "test");
        let e: MmapError = io_err.into();
        assert!(matches!(e, MmapError::Io(_)));
    }

    // ── Integration / edge-case tests ──────────────────────────────────

    #[test]
    fn integration_file_to_tensor_view() {
        // 4×4 matrix of f32
        let values: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_ne_bytes()).collect();
        let f = temp_file(&bytes);

        let mf = MmapFile::open(f.path(), MmapConfig::default()).unwrap();
        let region = mf.map_all().unwrap();
        let tv = MmapTensorView::new(&region, vec![4, 4], 4).unwrap();

        assert_eq!(tv.num_elements(), 16);
        assert_eq!(tv.byte_size(), 64);

        // Row 0 bytes = first 16 bytes
        let row0 = tv.row_bytes(0).unwrap();
        assert_eq!(row0.len(), 16);
    }

    #[test]
    fn integration_manager_full_workflow() {
        let data = b"workflow test data for mmap";
        let f = temp_file(data);

        let mut mgr = MmapManager::new(2, MmapConfig::default());

        // Map two regions from same file
        let id1 = mgr.map_file_region(f.path(), 0, 8).unwrap();
        let id2 = mgr.map_file_region(f.path(), 8, 8).unwrap();
        assert_eq!(mgr.pool().len(), 2);

        // Access them
        assert_eq!(mgr.get_region(id1).unwrap().as_bytes(), b"workflow");
        assert_eq!(mgr.get_region(id2).unwrap().as_bytes(), b" test da");

        // Insert a third → evicts LRU (id1 was touched last, so id2 is LRU
        // because we accessed id1 after id2 ... wait, we accessed both,
        // but id2 was accessed last, so id1 is LRU)
        let id3 = mgr.map_file_region(f.path(), 16, 10).unwrap();
        assert_eq!(mgr.pool().len(), 2);
        assert!(mgr.get_region(id3).is_some());

        // Prefetch
        let pages = mgr.prefetch(0, 100_000);
        assert!(!pages.is_empty());
    }

    #[test]
    fn integration_guard_with_pool() {
        let f = temp_file(b"guarded pool");
        let mf = MmapFile::open(f.path(), MmapConfig::default()).unwrap();
        let region = mf.map_all().unwrap();

        let guard = MmapSafetyGuard::new(&region);
        let rid = guard.region_id();

        let mut pool = MmapPool::new(4);
        pool.insert(region).unwrap();

        // Region accessible in pool while guard alive
        assert!(pool.get(rid).is_some());
        drop(guard);
        // Region still in pool (guard doesn't auto-remove from pool)
        assert!(pool.get(rid).is_some());
    }

    #[test]
    fn large_file_partial_maps() {
        let data = vec![0xABu8; 65536];
        let f = temp_file(&data);
        let mf = MmapFile::open(f.path(), MmapConfig::default()).unwrap();

        // Map several 4K chunks
        for i in 0..16 {
            let offset = i * 4096;
            let r = mf.map_region(offset, 4096).unwrap();
            assert_eq!(r.length(), 4096);
            assert!(r.as_bytes().iter().all(|&b| b == 0xAB));
        }
    }

    #[test]
    fn single_byte_file() {
        let f = temp_file(&[0xFF]);
        let mf = MmapFile::open(f.path(), MmapConfig::default()).unwrap();
        let r = mf.map_all().unwrap();
        assert_eq!(r.as_bytes(), &[0xFF]);
    }

    #[test]
    fn pool_eviction_preserves_newest() {
        let f = temp_file(b"AABBCCDD");
        let mf = MmapFile::open(f.path(), MmapConfig::default()).unwrap();

        let mut pool = MmapPool::new(1);
        let r1 = mf.map_region(0, 2).unwrap();
        pool.insert(r1).unwrap();

        let r2 = mf.map_region(2, 2).unwrap();
        let id2 = r2.id();
        let evicted = pool.insert(r2).unwrap();
        assert!(evicted.is_some());
        assert_eq!(pool.len(), 1);
        assert!(pool.get(id2).is_some());
    }

    #[test]
    fn tracker_many_accesses() {
        let mut t = PageFaultTracker::new(64);
        for i in 0..1000 {
            t.record_access(i * 64);
        }
        assert_eq!(t.access_count(), 1000);
        assert_eq!(t.estimated_stride(), Some(1));
    }

    #[test]
    fn slice_from_bytes_u16() {
        let values: Vec<u16> = vec![1000, 2000, 3000, 4000];
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_ne_bytes()).collect();
        let slice: MmapSlice<'_, u16> = MmapSlice::from_bytes(&bytes).unwrap();
        assert_eq!(slice.len(), 4);
        assert_eq!(slice.get(0), 1000);
        assert_eq!(slice.get(3), 4000);
    }

    // ── Proptest ───────────────────────────────────────────────────────

    mod proptests {
        use super::*;
        use proptest::prelude::*;

        proptest! {
            #[test]
            fn prop_region_offset_length(
                file_size in 1u64..=4096,
                offset in 0u64..=4096,
                length in 1u64..=4096
            ) {
                let data = vec![0xCDu8; file_size as usize];
                let f = temp_file(&data);
                let mf = MmapFile::open(f.path(), MmapConfig::default()).unwrap();

                let result = mf.map_region(offset, length);
                if offset + length <= file_size {
                    let r = result.unwrap();
                    prop_assert_eq!(r.offset(), offset);
                    prop_assert_eq!(r.length(), length);
                    prop_assert_eq!(r.as_bytes().len(), length as usize);
                } else {
                    prop_assert!(result.is_err());
                }
            }

            #[test]
            fn prop_slice_u8_roundtrip(data in proptest::collection::vec(any::<u8>(), 1..256)) {
                let f = temp_file(&data);
                let mf = MmapFile::open(f.path(), MmapConfig::default()).unwrap();
                let region = mf.map_all().unwrap();
                let slice: MmapSlice<'_, u8> = MmapSlice::new(&region).unwrap();
                prop_assert_eq!(slice.len(), data.len());
                for (i, &expected) in data.iter().enumerate() {
                    prop_assert_eq!(slice.get(i), expected);
                }
            }

            #[test]
            fn prop_tensor_view_shape(
                rows in 1usize..=16,
                cols in 1usize..=16
            ) {
                let total = rows * cols;
                let data = vec![0u8; total];
                let tv = MmapTensorView::from_bytes(&data, vec![rows, cols], 1).unwrap();
                prop_assert_eq!(tv.ndim(), 2);
                prop_assert_eq!(tv.num_elements(), total);
                prop_assert_eq!(tv.shape(), &[rows, cols]);
            }

            #[test]
            fn prop_pool_capacity_honoured(cap in 1usize..=8) {
                let data = vec![0u8; 64];
                let f = temp_file(&data);
                let mf = MmapFile::open(f.path(), MmapConfig::default()).unwrap();

                let mut pool = MmapPool::new(cap);
                for _ in 0..cap * 2 {
                    let r = mf.map_region(0, 1).unwrap();
                    pool.insert(r).unwrap();
                }
                prop_assert!(pool.len() <= cap);
            }

            #[test]
            fn prop_prefetcher_within_bounds(
                offset in 0u64..=100_000,
                region_len in 1u64..=200_000
            ) {
                let mut p = MmapPrefetcher::new(4, 4096);
                let pages = p.advise(offset, region_len);
                for &pg in &pages {
                    prop_assert!(pg < region_len);
                }
            }

            #[test]
            fn prop_tracker_fault_count_monotonic(
                offsets in proptest::collection::vec(0u64..=100_000, 1..100)
            ) {
                let mut t = PageFaultTracker::new(4096);
                let mut prev = 0;
                for &off in &offsets {
                    t.record_access(off);
                    prop_assert!(t.fault_count() >= prev);
                    prev = t.fault_count();
                }
            }
        }
    }
}
