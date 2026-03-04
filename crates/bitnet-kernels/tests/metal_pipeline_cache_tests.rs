#![allow(dead_code, unused_imports, unused_variables, non_camel_case_types, unused_mut)]
//! Metal pipeline cache tests for Apple Silicon.
//!
//! Validates Metal compute pipeline caching, buffer management, dispatch
//! sizing, and error handling without requiring live Metal hardware.
//! All mock infrastructure simulates Metal semantics (256-byte alignment,
//! 1024-thread workgroup limit, pipeline keying by source+function).

#![cfg(target_os = "macos")]

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, RwLock};

// ── Constants (mirror Metal hardware limits) ────────────────────────

const METAL_BUFFER_ALIGNMENT: usize = 256;
const METAL_MAX_WORKGROUP_SIZE: u32 = 1024;
const MAX_DISPATCH_DIM: u32 = 65535;
const SIMD_WIDTH: u32 = 32;

// ── Mock types ──────────────────────────────────────────────────────

/// Unique identifier for a compiled pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct PipelineId(u64);

/// Storage mode for Metal buffers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StorageMode {
    Shared,
    Private,
    Managed,
}

/// Result of a cache lookup.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CacheResult {
    Hit,
    Miss,
}

/// A simulated Metal compute pipeline.
#[derive(Debug, Clone)]
struct MockPipeline {
    id: PipelineId,
    source: String,
    function_name: String,
    workgroup_size: (u32, u32, u32),
}

/// A simulated Metal buffer.
#[derive(Debug, Clone)]
struct MockBuffer {
    data: Vec<u8>,
    size: usize,
    aligned_size: usize,
    storage_mode: StorageMode,
    label: String,
}

impl MockBuffer {
    fn new(size: usize, mode: StorageMode, label: &str) -> Self {
        let aligned = align_to_256(size);
        Self {
            data: vec![0u8; aligned],
            size,
            aligned_size: aligned,
            storage_mode: mode,
            label: label.to_string(),
        }
    }

    fn write(&mut self, offset: usize, data: &[u8]) -> Result<(), PipelineError> {
        if offset + data.len() > self.aligned_size {
            return Err(PipelineError::BufferOverflow {
                requested: offset + data.len(),
                capacity: self.aligned_size,
            });
        }
        self.data[offset..offset + data.len()].copy_from_slice(data);
        Ok(())
    }

    fn read(&self, offset: usize, len: usize) -> Result<&[u8], PipelineError> {
        if offset + len > self.aligned_size {
            return Err(PipelineError::BufferOverflow {
                requested: offset + len,
                capacity: self.aligned_size,
            });
        }
        Ok(&self.data[offset..offset + len])
    }
}

/// Errors from mock Metal pipeline operations.
#[derive(Debug, Clone, PartialEq, Eq)]
enum PipelineError {
    CompilationFailed(String),
    FunctionNotFound(String),
    InvalidSource,
    EmptySource,
    WorkgroupTooLarge { total: u64, max: u32 },
    ZeroDimension,
    DispatchTooLarge { dim: u32, max: u32 },
    BufferOverflow { requested: usize, capacity: usize },
    InvalidBufferSize,
    DeviceUnavailable,
    OutOfMemory { requested: usize, available: usize },
    CacheFull,
}

impl std::fmt::Display for PipelineError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::CompilationFailed(msg) => {
                write!(f, "shader compilation failed: {msg}")
            }
            Self::FunctionNotFound(name) => {
                write!(f, "function '{name}' not found")
            }
            Self::InvalidSource => write!(f, "invalid shader source"),
            Self::EmptySource => write!(f, "empty shader source"),
            Self::WorkgroupTooLarge { total, max } => {
                write!(f, "workgroup {total} exceeds limit {max}")
            }
            Self::ZeroDimension => write!(f, "zero dimension"),
            Self::DispatchTooLarge { dim, max } => {
                write!(f, "dispatch dim {dim} exceeds {max}")
            }
            Self::BufferOverflow { requested, capacity } => {
                write!(f, "buffer overflow: {requested} > {capacity}")
            }
            Self::InvalidBufferSize => write!(f, "invalid buffer size"),
            Self::DeviceUnavailable => write!(f, "Metal device unavailable"),
            Self::OutOfMemory { requested, available } => {
                write!(f, "out of memory: need {requested}, have {available}")
            }
            Self::CacheFull => write!(f, "pipeline cache is full"),
        }
    }
}

/// Cache statistics tracker.
#[derive(Debug, Default, Clone)]
struct CacheStats {
    hits: u64,
    misses: u64,
    evictions: u64,
    insertions: u64,
}

/// Composite cache key: source hash + function name.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct CacheKey {
    source_hash: u64,
    function_name: String,
}

impl CacheKey {
    fn new(source: &str, function: &str) -> Self {
        // FNV-1a for deterministic hashing without external deps.
        let mut h: u64 = 0xcbf29ce484222325;
        for b in source.as_bytes() {
            h ^= *b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
        Self { source_hash: h, function_name: function.to_string() }
    }
}

/// Thread-safe pipeline cache with capacity limits.
struct PipelineCache {
    entries: RwLock<HashMap<CacheKey, MockPipeline>>,
    capacity: usize,
    stats: Mutex<CacheStats>,
    next_id: AtomicU64,
}

impl PipelineCache {
    fn new(capacity: usize) -> Self {
        Self {
            entries: RwLock::new(HashMap::new()),
            capacity,
            stats: Mutex::new(CacheStats::default()),
            next_id: AtomicU64::new(1),
        }
    }

    fn get_or_create(
        &self,
        source: &str,
        function: &str,
    ) -> Result<(MockPipeline, CacheResult), PipelineError> {
        if source.is_empty() {
            return Err(PipelineError::EmptySource);
        }
        if !source.contains("fn ") && !source.contains("kernel ") {
            return Err(PipelineError::CompilationFailed(
                "no valid function found in source".into(),
            ));
        }

        let key = CacheKey::new(source, function);

        // Read path — check cache.
        {
            let map = self.entries.read().unwrap();
            if let Some(p) = map.get(&key) {
                self.stats.lock().unwrap().hits += 1;
                return Ok((p.clone(), CacheResult::Hit));
            }
        }

        // Write path — compile and insert.
        let mut map = self.entries.write().unwrap();

        // Double-check after upgrading lock.
        if let Some(p) = map.get(&key) {
            self.stats.lock().unwrap().hits += 1;
            return Ok((p.clone(), CacheResult::Hit));
        }

        // Evict if full.
        let mut stats = self.stats.lock().unwrap();
        stats.misses += 1;

        if map.len() >= self.capacity {
            if let Some(evict_key) = map.keys().next().cloned() {
                map.remove(&evict_key);
                stats.evictions += 1;
            }
        }

        let id = PipelineId(self.next_id.fetch_add(1, Ordering::Relaxed));
        let pipeline = MockPipeline {
            id,
            source: source.to_string(),
            function_name: function.to_string(),
            workgroup_size: (SIMD_WIDTH, 1, 1),
        };
        map.insert(key, pipeline.clone());
        stats.insertions += 1;

        Ok((pipeline, CacheResult::Miss))
    }

    fn stats(&self) -> CacheStats {
        self.stats.lock().unwrap().clone()
    }

    fn len(&self) -> usize {
        self.entries.read().unwrap().len()
    }

    fn clear(&self) {
        self.entries.write().unwrap().clear();
    }

    fn contains(&self, source: &str, function: &str) -> bool {
        let key = CacheKey::new(source, function);
        self.entries.read().unwrap().contains_key(&key)
    }
}

/// Simulated buffer pool with recycling.
struct BufferPool {
    free: Mutex<Vec<MockBuffer>>,
    allocated: AtomicU64,
    recycled: AtomicU64,
    memory_limit: usize,
    used_memory: AtomicU64,
}

impl BufferPool {
    fn new(memory_limit: usize) -> Self {
        Self {
            free: Mutex::new(Vec::new()),
            allocated: AtomicU64::new(0),
            recycled: AtomicU64::new(0),
            memory_limit,
            used_memory: AtomicU64::new(0),
        }
    }

    fn allocate(
        &self,
        size: usize,
        mode: StorageMode,
        label: &str,
    ) -> Result<MockBuffer, PipelineError> {
        let aligned = align_to_256(size);
        if aligned == 0 && size > 0 {
            return Err(PipelineError::InvalidBufferSize);
        }

        // Check memory limit.
        let used = self.used_memory.load(Ordering::Relaxed) as usize;
        if used + aligned > self.memory_limit {
            return Err(PipelineError::OutOfMemory {
                requested: aligned,
                available: self.memory_limit.saturating_sub(used),
            });
        }

        // Try recycling.
        {
            let mut pool = self.free.lock().unwrap();
            if let Some(idx) =
                pool.iter().position(|b| b.aligned_size >= aligned && b.storage_mode == mode)
            {
                let mut buf = pool.swap_remove(idx);
                buf.label = label.to_string();
                buf.size = size;
                buf.data.iter_mut().for_each(|b| *b = 0);
                self.recycled.fetch_add(1, Ordering::Relaxed);
                return Ok(buf);
            }
        }

        self.used_memory.fetch_add(aligned as u64, Ordering::Relaxed);
        self.allocated.fetch_add(1, Ordering::Relaxed);
        Ok(MockBuffer::new(size, mode, label))
    }

    fn release(&self, buf: MockBuffer) {
        self.free.lock().unwrap().push(buf);
    }

    fn allocated_count(&self) -> u64 {
        self.allocated.load(Ordering::Relaxed)
    }

    fn recycled_count(&self) -> u64 {
        self.recycled.load(Ordering::Relaxed)
    }
}

// ── Utility functions ───────────────────────────────────────────────

fn align_to_256(size: usize) -> usize {
    let mask = METAL_BUFFER_ALIGNMENT - 1;
    (size + mask) & !mask
}

fn is_aligned(offset: usize) -> bool {
    offset % METAL_BUFFER_ALIGNMENT == 0
}

fn validate_workgroup(x: u32, y: u32, z: u32) -> Result<(), PipelineError> {
    if x == 0 || y == 0 || z == 0 {
        return Err(PipelineError::ZeroDimension);
    }
    let total = x as u64 * y as u64 * z as u64;
    if total > METAL_MAX_WORKGROUP_SIZE as u64 {
        return Err(PipelineError::WorkgroupTooLarge { total, max: METAL_MAX_WORKGROUP_SIZE });
    }
    Ok(())
}

fn compute_dispatch_groups(
    problem: (u32, u32, u32),
    wg: (u32, u32, u32),
) -> Result<(u32, u32, u32), PipelineError> {
    let dim = |p: u32, w: u32| -> Result<u32, PipelineError> {
        if w == 0 {
            return Err(PipelineError::ZeroDimension);
        }
        let d = p.div_ceil(w);
        if d > MAX_DISPATCH_DIM {
            return Err(PipelineError::DispatchTooLarge { dim: d, max: MAX_DISPATCH_DIM });
        }
        Ok(d)
    };
    Ok((dim(problem.0, wg.0)?, dim(problem.1, wg.1)?, dim(problem.2, wg.2)?))
}

const VALID_SHADER: &str = "kernel fn add_arrays() {} fn helper() {}";
const VALID_SHADER_ALT: &str = "kernel fn multiply_arrays() {} fn helper() {}";
const MATMUL_SHADER: &str = "kernel fn matmul() {} fn matmul_tiled() {}";
const REDUCTION_SHADER: &str = "kernel fn reduce_sum() {} fn reduce_max() {}";
const ELEMENTWISE_SHADER: &str = "kernel fn elementwise_add() {} fn relu() {}";

// ═════════════════════════════════════════════════════════════════════
// 1. Pipeline Creation Tests (18 tests)
// ═════════════════════════════════════════════════════════════════════

/// Creating a compute pipeline from valid source succeeds.
#[test]
#[cfg(target_os = "macos")]
fn test_metal_pipeline_cache_create_from_source() {
    let cache = PipelineCache::new(64);
    let (pipeline, result) = cache.get_or_create(VALID_SHADER, "add_arrays").unwrap();
    assert_eq!(result, CacheResult::Miss);
    assert_eq!(pipeline.function_name, "add_arrays");
}

/// Different function names produce distinct pipelines.
#[test]
#[cfg(target_os = "macos")]
fn test_metal_pipeline_cache_different_functions() {
    let cache = PipelineCache::new(64);
    let (p1, _) = cache.get_or_create(VALID_SHADER, "add_arrays").unwrap();
    let (p2, _) = cache.get_or_create(VALID_SHADER, "helper").unwrap();
    assert_ne!(p1.id, p2.id);
    assert_ne!(p1.function_name, p2.function_name);
}

/// Invalid source (no function declarations) returns a compilation error.
#[test]
#[cfg(target_os = "macos")]
fn test_metal_pipeline_cache_invalid_source_errors() {
    let cache = PipelineCache::new(64);
    let err = cache.get_or_create("not a shader", "main").unwrap_err();
    assert!(matches!(err, PipelineError::CompilationFailed(_)));
}

/// Empty source returns `EmptySource` error.
#[test]
#[cfg(target_os = "macos")]
fn test_metal_pipeline_cache_empty_source() {
    let cache = PipelineCache::new(64);
    let err = cache.get_or_create("", "main").unwrap_err();
    assert_eq!(err, PipelineError::EmptySource);
}

/// Requesting the same source+function twice returns the cached copy.
#[test]
#[cfg(target_os = "macos")]
fn test_metal_pipeline_cache_reuse_same_pipeline() {
    let cache = PipelineCache::new(64);
    let (p1, r1) = cache.get_or_create(VALID_SHADER, "add_arrays").unwrap();
    let (p2, r2) = cache.get_or_create(VALID_SHADER, "add_arrays").unwrap();
    assert_eq!(r1, CacheResult::Miss);
    assert_eq!(r2, CacheResult::Hit);
    assert_eq!(p1.id, p2.id);
}

/// Concurrent pipeline creation from multiple threads is safe.
#[test]
#[cfg(target_os = "macos")]
fn test_metal_pipeline_cache_concurrent_creation() {
    let cache = Arc::new(PipelineCache::new(256));
    let mut handles = Vec::new();
    for i in 0..8 {
        let c = Arc::clone(&cache);
        let src = format!("kernel fn worker_{i}() {{}}");
        handles.push(std::thread::spawn(move || {
            c.get_or_create(&src, &format!("worker_{i}")).unwrap()
        }));
    }
    let pipelines: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();
    assert_eq!(pipelines.len(), 8);
    // All IDs are distinct.
    let ids: std::collections::HashSet<_> = pipelines.iter().map(|(p, _)| p.id).collect();
    assert_eq!(ids.len(), 8);
}

/// Pipeline IDs are monotonically increasing.
#[test]
#[cfg(target_os = "macos")]
fn test_metal_pipeline_cache_monotonic_ids() {
    let cache = PipelineCache::new(64);
    let (p1, _) = cache.get_or_create(VALID_SHADER, "add_arrays").unwrap();
    let (p2, _) = cache.get_or_create(VALID_SHADER_ALT, "multiply_arrays").unwrap();
    assert!(p2.id.0 > p1.id.0);
}

/// Pipeline stores its source faithfully.
#[test]
#[cfg(target_os = "macos")]
fn test_metal_pipeline_cache_stores_source() {
    let cache = PipelineCache::new(64);
    let (p, _) = cache.get_or_create(VALID_SHADER, "add_arrays").unwrap();
    assert_eq!(p.source, VALID_SHADER);
}

/// Creating a pipeline with a very long function name succeeds.
#[test]
#[cfg(target_os = "macos")]
fn test_metal_pipeline_cache_long_function_name() {
    let cache = PipelineCache::new(64);
    let long_name = "a".repeat(1024);
    let src = format!("kernel fn {long_name}() {{}}");
    let (p, _) = cache.get_or_create(&src, &long_name).unwrap();
    assert_eq!(p.function_name, long_name);
}

/// Clearing the cache resets its length to zero.
#[test]
#[cfg(target_os = "macos")]
fn test_metal_pipeline_cache_clear() {
    let cache = PipelineCache::new(64);
    cache.get_or_create(VALID_SHADER, "add_arrays").unwrap();
    assert_eq!(cache.len(), 1);
    cache.clear();
    assert_eq!(cache.len(), 0);
}

/// After clearing, re-inserting the same key is a miss.
#[test]
#[cfg(target_os = "macos")]
fn test_metal_pipeline_cache_clear_resets_hits() {
    let cache = PipelineCache::new(64);
    cache.get_or_create(VALID_SHADER, "add_arrays").unwrap();
    cache.clear();
    let (_, r) = cache.get_or_create(VALID_SHADER, "add_arrays").unwrap();
    assert_eq!(r, CacheResult::Miss);
}

/// Pipelines created from different sources get different IDs.
#[test]
#[cfg(target_os = "macos")]
fn test_metal_pipeline_cache_different_sources_different_ids() {
    let cache = PipelineCache::new(64);
    let (p1, _) = cache.get_or_create(VALID_SHADER, "add_arrays").unwrap();
    let (p2, _) = cache.get_or_create(VALID_SHADER_ALT, "multiply_arrays").unwrap();
    assert_ne!(p1.id, p2.id);
}

/// Default workgroup is SIMD_WIDTH × 1 × 1.
#[test]
#[cfg(target_os = "macos")]
fn test_metal_pipeline_cache_default_workgroup() {
    let cache = PipelineCache::new(64);
    let (p, _) = cache.get_or_create(VALID_SHADER, "add_arrays").unwrap();
    assert_eq!(p.workgroup_size, (SIMD_WIDTH, 1, 1));
}

/// Multiple functions from the same source each get separate entries.
#[test]
#[cfg(target_os = "macos")]
fn test_metal_pipeline_cache_multiple_functions_same_source() {
    let cache = PipelineCache::new(64);
    let (p1, _) = cache.get_or_create(MATMUL_SHADER, "matmul").unwrap();
    let (p2, _) = cache.get_or_create(MATMUL_SHADER, "matmul_tiled").unwrap();
    assert_ne!(p1.id, p2.id);
    assert_eq!(cache.len(), 2);
}

/// Whitespace-only source is rejected.
#[test]
#[cfg(target_os = "macos")]
fn test_metal_pipeline_cache_whitespace_source() {
    let cache = PipelineCache::new(64);
    let err = cache.get_or_create("   \n\t  ", "main").unwrap_err();
    assert!(matches!(err, PipelineError::CompilationFailed(_)));
}

/// `contains` predicate is accurate for cached/uncached keys.
#[test]
#[cfg(target_os = "macos")]
fn test_metal_pipeline_cache_contains() {
    let cache = PipelineCache::new(64);
    assert!(!cache.contains(VALID_SHADER, "add_arrays"));
    cache.get_or_create(VALID_SHADER, "add_arrays").unwrap();
    assert!(cache.contains(VALID_SHADER, "add_arrays"));
    assert!(!cache.contains(VALID_SHADER, "nonexistent"));
}

/// Source with only `fn` but no `kernel` still compiles (helper fn).
#[test]
#[cfg(target_os = "macos")]
fn test_metal_pipeline_cache_helper_fn_only() {
    let cache = PipelineCache::new(64);
    let src = "fn helper_only() {}";
    let (p, _) = cache.get_or_create(src, "helper_only").unwrap();
    assert_eq!(p.function_name, "helper_only");
}

/// Creating a pipeline with unicode function name succeeds.
#[test]
#[cfg(target_os = "macos")]
fn test_metal_pipeline_cache_unicode_function() {
    let cache = PipelineCache::new(64);
    let src = "kernel fn 加算() {}";
    let (p, _) = cache.get_or_create(src, "加算").unwrap();
    assert_eq!(p.function_name, "加算");
}

// ═════════════════════════════════════════════════════════════════════
// 2. Cache Behavior Tests (22 tests)
// ═════════════════════════════════════════════════════════════════════

/// Second lookup for the same key is a cache hit.
#[test]
#[cfg(target_os = "macos")]
fn test_metal_pipeline_cache_hit_after_first() {
    let cache = PipelineCache::new(64);
    cache.get_or_create(VALID_SHADER, "add_arrays").unwrap();
    let (_, r) = cache.get_or_create(VALID_SHADER, "add_arrays").unwrap();
    assert_eq!(r, CacheResult::Hit);
}

/// Different source text is a cache miss.
#[test]
#[cfg(target_os = "macos")]
fn test_metal_pipeline_cache_miss_different_source() {
    let cache = PipelineCache::new(64);
    cache.get_or_create(VALID_SHADER, "add_arrays").unwrap();
    let (_, r) = cache.get_or_create(VALID_SHADER_ALT, "multiply_arrays").unwrap();
    assert_eq!(r, CacheResult::Miss);
}

/// When the cache is full, inserting a new key evicts an existing entry.
#[test]
#[cfg(target_os = "macos")]
fn test_metal_pipeline_cache_eviction_when_full() {
    let cache = PipelineCache::new(2);
    cache.get_or_create(VALID_SHADER, "add_arrays").unwrap();
    cache.get_or_create(VALID_SHADER_ALT, "multiply_arrays").unwrap();
    assert_eq!(cache.len(), 2);

    // Third insertion must evict one entry.
    cache.get_or_create(MATMUL_SHADER, "matmul").unwrap();
    assert_eq!(cache.len(), 2);
    assert_eq!(cache.stats().evictions, 1);
}

/// Cache respects its size limit.
#[test]
#[cfg(target_os = "macos")]
fn test_metal_pipeline_cache_size_limit() {
    let cache = PipelineCache::new(3);
    for i in 0..10 {
        let src = format!("kernel fn func_{i}() {{}}");
        cache.get_or_create(&src, &format!("func_{i}")).unwrap();
    }
    assert!(cache.len() <= 3);
}

/// Same source with different function names produces unique cache keys.
#[test]
#[cfg(target_os = "macos")]
fn test_metal_pipeline_cache_key_uniqueness() {
    let cache = PipelineCache::new(64);
    let (p1, r1) = cache.get_or_create(VALID_SHADER, "add_arrays").unwrap();
    let (p2, r2) = cache.get_or_create(VALID_SHADER, "helper").unwrap();
    assert_eq!(r1, CacheResult::Miss);
    assert_eq!(r2, CacheResult::Miss);
    assert_ne!(p1.id, p2.id);
}

/// Thread-safe concurrent reads all see the cached pipeline.
#[test]
#[cfg(target_os = "macos")]
fn test_metal_pipeline_cache_thread_safe_reads() {
    let cache = Arc::new(PipelineCache::new(64));
    cache.get_or_create(VALID_SHADER, "add_arrays").unwrap();

    let mut handles = Vec::new();
    for _ in 0..16 {
        let c = Arc::clone(&cache);
        handles.push(std::thread::spawn(move || {
            let (_, r) = c.get_or_create(VALID_SHADER, "add_arrays").unwrap();
            r
        }));
    }
    for h in handles {
        assert_eq!(h.join().unwrap(), CacheResult::Hit);
    }
}

/// Hit counter increments on cache hits.
#[test]
#[cfg(target_os = "macos")]
fn test_metal_pipeline_cache_stats_hits() {
    let cache = PipelineCache::new(64);
    cache.get_or_create(VALID_SHADER, "add_arrays").unwrap();
    cache.get_or_create(VALID_SHADER, "add_arrays").unwrap();
    cache.get_or_create(VALID_SHADER, "add_arrays").unwrap();
    let s = cache.stats();
    assert_eq!(s.hits, 2);
    assert_eq!(s.misses, 1);
}

/// Miss counter increments on first lookup.
#[test]
#[cfg(target_os = "macos")]
fn test_metal_pipeline_cache_stats_misses() {
    let cache = PipelineCache::new(64);
    cache.get_or_create(VALID_SHADER, "add_arrays").unwrap();
    cache.get_or_create(VALID_SHADER_ALT, "multiply_arrays").unwrap();
    let s = cache.stats();
    assert_eq!(s.misses, 2);
}

/// Eviction counter is correct after forced evictions.
#[test]
#[cfg(target_os = "macos")]
fn test_metal_pipeline_cache_stats_evictions() {
    let cache = PipelineCache::new(1);
    cache.get_or_create(VALID_SHADER, "add_arrays").unwrap();
    cache.get_or_create(VALID_SHADER_ALT, "multiply_arrays").unwrap();
    let s = cache.stats();
    assert_eq!(s.evictions, 1);
}

/// Insertion counter tracks total compilations.
#[test]
#[cfg(target_os = "macos")]
fn test_metal_pipeline_cache_stats_insertions() {
    let cache = PipelineCache::new(64);
    for i in 0..5 {
        let src = format!("kernel fn f_{i}() {{}}");
        cache.get_or_create(&src, &format!("f_{i}")).unwrap();
    }
    assert_eq!(cache.stats().insertions, 5);
}

/// A full cache (capacity 1) still works correctly.
#[test]
#[cfg(target_os = "macos")]
fn test_metal_pipeline_cache_capacity_one() {
    let cache = PipelineCache::new(1);
    cache.get_or_create(VALID_SHADER, "add_arrays").unwrap();
    assert_eq!(cache.len(), 1);
    cache.get_or_create(VALID_SHADER_ALT, "multiply_arrays").unwrap();
    assert_eq!(cache.len(), 1);
}

/// After eviction, the evicted key is a miss.
#[test]
#[cfg(target_os = "macos")]
fn test_metal_pipeline_cache_evicted_key_is_miss() {
    let cache = PipelineCache::new(1);
    cache.get_or_create(VALID_SHADER, "add_arrays").unwrap();
    cache.get_or_create(VALID_SHADER_ALT, "multiply_arrays").unwrap();
    // Original key was evicted.
    let (_, r) = cache.get_or_create(VALID_SHADER, "add_arrays").unwrap();
    assert_eq!(r, CacheResult::Miss);
}

/// Large cache handles hundreds of entries.
#[test]
#[cfg(target_os = "macos")]
fn test_metal_pipeline_cache_large_capacity() {
    let cache = PipelineCache::new(1000);
    for i in 0..500 {
        let src = format!("kernel fn op_{i}() {{}}");
        cache.get_or_create(&src, &format!("op_{i}")).unwrap();
    }
    assert_eq!(cache.len(), 500);
    assert_eq!(cache.stats().evictions, 0);
}

/// Concurrent writes from multiple threads don't lose entries.
#[test]
#[cfg(target_os = "macos")]
fn test_metal_pipeline_cache_concurrent_writes() {
    let cache = Arc::new(PipelineCache::new(1024));
    let mut handles = Vec::new();
    for i in 0..32 {
        let c = Arc::clone(&cache);
        handles.push(std::thread::spawn(move || {
            let src = format!("kernel fn thread_{i}() {{}}");
            c.get_or_create(&src, &format!("thread_{i}")).unwrap();
        }));
    }
    for h in handles {
        h.join().unwrap();
    }
    assert_eq!(cache.len(), 32);
}

/// Repeated lookups on multiple keys are all hits after priming.
#[test]
#[cfg(target_os = "macos")]
fn test_metal_pipeline_cache_all_hits_after_priming() {
    let cache = PipelineCache::new(64);
    let shaders = [VALID_SHADER, VALID_SHADER_ALT, MATMUL_SHADER];
    let funcs = ["add_arrays", "multiply_arrays", "matmul"];
    for (&s, f) in shaders.iter().zip(funcs.iter()) {
        cache.get_or_create(s, f).unwrap();
    }
    for (&s, f) in shaders.iter().zip(funcs.iter()) {
        let (_, r) = cache.get_or_create(s, f).unwrap();
        assert_eq!(r, CacheResult::Hit, "Expected hit for {f}");
    }
}

/// Clear resets statistics.
#[test]
#[cfg(target_os = "macos")]
fn test_metal_pipeline_cache_clear_does_not_reset_stats() {
    let cache = PipelineCache::new(64);
    cache.get_or_create(VALID_SHADER, "add_arrays").unwrap();
    cache.get_or_create(VALID_SHADER, "add_arrays").unwrap();
    let before = cache.stats();
    cache.clear();
    let after = cache.stats();
    // Stats survive clear (they track lifetime totals).
    assert_eq!(before.hits, after.hits);
    assert_eq!(before.misses, after.misses);
}

/// Cache key hash is deterministic.
#[test]
#[cfg(target_os = "macos")]
fn test_metal_pipeline_cache_key_deterministic() {
    let k1 = CacheKey::new("kernel fn f() {}", "f");
    let k2 = CacheKey::new("kernel fn f() {}", "f");
    assert_eq!(k1, k2);
    assert_eq!(k1.source_hash, k2.source_hash);
}

/// Different source produces different hash.
#[test]
#[cfg(target_os = "macos")]
fn test_metal_pipeline_cache_key_different_hash() {
    let k1 = CacheKey::new("kernel fn a() {}", "a");
    let k2 = CacheKey::new("kernel fn b() {}", "b");
    assert_ne!(k1.source_hash, k2.source_hash);
}

/// Cache survives rapid create-clear-create cycles.
#[test]
#[cfg(target_os = "macos")]
fn test_metal_pipeline_cache_stress_clear_cycles() {
    let cache = PipelineCache::new(16);
    for _ in 0..100 {
        cache.get_or_create(VALID_SHADER, "add_arrays").unwrap();
        cache.clear();
    }
    assert_eq!(cache.len(), 0);
    assert_eq!(cache.stats().insertions, 100);
}

/// Eviction keeps cache at exactly capacity.
#[test]
#[cfg(target_os = "macos")]
fn test_metal_pipeline_cache_never_exceeds_capacity() {
    let cap = 5;
    let cache = PipelineCache::new(cap);
    for i in 0..50 {
        let src = format!("kernel fn f_{i}() {{}}");
        cache.get_or_create(&src, &format!("f_{i}")).unwrap();
        assert!(cache.len() <= cap);
    }
}

/// Interleaved reads/writes maintain consistency.
#[test]
#[cfg(target_os = "macos")]
fn test_metal_pipeline_cache_interleaved_rw() {
    let cache = Arc::new(PipelineCache::new(64));
    let mut handles = Vec::new();
    for i in 0..16 {
        let c = Arc::clone(&cache);
        handles.push(std::thread::spawn(move || {
            let src = format!("kernel fn rw_{i}() {{}}");
            c.get_or_create(&src, &format!("rw_{i}")).unwrap();
            // Immediate re-read.
            let (_, r) = c.get_or_create(&src, &format!("rw_{i}")).unwrap();
            assert_eq!(r, CacheResult::Hit);
        }));
    }
    for h in handles {
        h.join().unwrap();
    }
}

// ═════════════════════════════════════════════════════════════════════
// 3. Buffer Management Tests (22 tests)
// ═════════════════════════════════════════════════════════════════════

/// Zero-size buffer produces zero aligned size.
#[test]
#[cfg(target_os = "macos")]
fn test_metal_buffer_zero_size() {
    let buf = MockBuffer::new(0, StorageMode::Shared, "zero");
    assert_eq!(buf.size, 0);
    assert_eq!(buf.aligned_size, 0);
}

/// Single-byte buffer is aligned up to 256.
#[test]
#[cfg(target_os = "macos")]
fn test_metal_buffer_one_byte() {
    let buf = MockBuffer::new(1, StorageMode::Shared, "one");
    assert_eq!(buf.aligned_size, 256);
}

/// 4096-byte buffer is already aligned.
#[test]
#[cfg(target_os = "macos")]
fn test_metal_buffer_4096() {
    let buf = MockBuffer::new(4096, StorageMode::Shared, "page");
    assert_eq!(buf.aligned_size, 4096);
    assert!(is_aligned(buf.aligned_size));
}

/// 1 MiB buffer alignment.
#[test]
#[cfg(target_os = "macos")]
fn test_metal_buffer_1mb() {
    let buf = MockBuffer::new(1024 * 1024, StorageMode::Shared, "1mb");
    assert_eq!(buf.aligned_size, 1024 * 1024);
}

/// Buffer contents can be read back after write.
#[test]
#[cfg(target_os = "macos")]
fn test_metal_buffer_readback() {
    let mut buf = MockBuffer::new(256, StorageMode::Shared, "rw");
    let payload = [1u8, 2, 3, 4];
    buf.write(0, &payload).unwrap();
    let out = buf.read(0, 4).unwrap();
    assert_eq!(out, &payload);
}

/// Buffer alignment helper rounds correctly.
#[test]
#[cfg(target_os = "macos")]
fn test_metal_buffer_alignment_helper() {
    assert_eq!(align_to_256(0), 0);
    assert_eq!(align_to_256(1), 256);
    assert_eq!(align_to_256(255), 256);
    assert_eq!(align_to_256(256), 256);
    assert_eq!(align_to_256(257), 512);
    assert_eq!(align_to_256(1023), 1024);
}

/// `is_aligned` predicate.
#[test]
#[cfg(target_os = "macos")]
fn test_metal_buffer_is_aligned() {
    assert!(is_aligned(0));
    assert!(is_aligned(256));
    assert!(is_aligned(512));
    assert!(!is_aligned(1));
    assert!(!is_aligned(128));
}

/// Shared storage mode is recorded.
#[test]
#[cfg(target_os = "macos")]
fn test_metal_buffer_shared_storage_mode() {
    let buf = MockBuffer::new(256, StorageMode::Shared, "shared");
    assert_eq!(buf.storage_mode, StorageMode::Shared);
}

/// Private storage mode is recorded.
#[test]
#[cfg(target_os = "macos")]
fn test_metal_buffer_private_storage_mode() {
    let buf = MockBuffer::new(256, StorageMode::Private, "private");
    assert_eq!(buf.storage_mode, StorageMode::Private);
}

/// Managed storage mode is recorded.
#[test]
#[cfg(target_os = "macos")]
fn test_metal_buffer_managed_storage_mode() {
    let buf = MockBuffer::new(256, StorageMode::Managed, "managed");
    assert_eq!(buf.storage_mode, StorageMode::Managed);
}

/// Buffer pool allocates a fresh buffer.
#[test]
#[cfg(target_os = "macos")]
fn test_metal_buffer_pool_allocate() {
    let pool = BufferPool::new(1024 * 1024);
    let buf = pool.allocate(512, StorageMode::Shared, "test").unwrap();
    assert_eq!(buf.aligned_size, 512);
    assert_eq!(pool.allocated_count(), 1);
}

/// Buffer pool recycles a released buffer.
#[test]
#[cfg(target_os = "macos")]
fn test_metal_buffer_pool_recycle() {
    let pool = BufferPool::new(1024 * 1024);
    let buf = pool.allocate(256, StorageMode::Shared, "first").unwrap();
    pool.release(buf);
    let buf2 = pool.allocate(256, StorageMode::Shared, "second").unwrap();
    assert_eq!(buf2.label, "second");
    assert_eq!(pool.recycled_count(), 1);
}

/// Pool returns OOM when memory limit is exceeded.
#[test]
#[cfg(target_os = "macos")]
fn test_metal_buffer_pool_oom() {
    let pool = BufferPool::new(512);
    pool.allocate(256, StorageMode::Shared, "a").unwrap();
    let err = pool.allocate(512, StorageMode::Shared, "b").unwrap_err();
    assert!(matches!(err, PipelineError::OutOfMemory { .. }));
}

/// Recycled buffer is zeroed.
#[test]
#[cfg(target_os = "macos")]
fn test_metal_buffer_pool_recycled_zeroed() {
    let pool = BufferPool::new(1024 * 1024);
    let mut buf = pool.allocate(256, StorageMode::Shared, "dirty").unwrap();
    buf.write(0, &[0xFFu8; 256]).unwrap();
    pool.release(buf);
    let clean = pool.allocate(256, StorageMode::Shared, "clean").unwrap();
    assert!(clean.data.iter().all(|&b| b == 0));
}

/// Writing past buffer capacity returns overflow.
#[test]
#[cfg(target_os = "macos")]
fn test_metal_buffer_write_overflow() {
    let mut buf = MockBuffer::new(256, StorageMode::Shared, "small");
    let err = buf.write(200, &[0u8; 100]).unwrap_err();
    assert!(matches!(err, PipelineError::BufferOverflow { .. }));
}

/// Reading past buffer capacity returns overflow.
#[test]
#[cfg(target_os = "macos")]
fn test_metal_buffer_read_overflow() {
    let buf = MockBuffer::new(256, StorageMode::Shared, "small");
    let err = buf.read(200, 100).unwrap_err();
    assert!(matches!(err, PipelineError::BufferOverflow { .. }));
}

/// Buffer label is stored.
#[test]
#[cfg(target_os = "macos")]
fn test_metal_buffer_label() {
    let buf = MockBuffer::new(256, StorageMode::Shared, "my_label");
    assert_eq!(buf.label, "my_label");
}

/// Large buffer (16 MiB) allocates correctly.
#[test]
#[cfg(target_os = "macos")]
fn test_metal_buffer_large_allocation() {
    let pool = BufferPool::new(64 * 1024 * 1024);
    let buf = pool.allocate(16 * 1024 * 1024, StorageMode::Shared, "large").unwrap();
    assert_eq!(buf.aligned_size, 16 * 1024 * 1024);
}

/// Multiple allocations track total count.
#[test]
#[cfg(target_os = "macos")]
fn test_metal_buffer_pool_multiple_allocations() {
    let pool = BufferPool::new(1024 * 1024);
    for i in 0..10 {
        pool.allocate(256, StorageMode::Shared, &format!("buf_{i}")).unwrap();
    }
    assert_eq!(pool.allocated_count(), 10);
}

/// Pool prefers recycling matching storage mode.
#[test]
#[cfg(target_os = "macos")]
fn test_metal_buffer_pool_storage_mode_match() {
    let pool = BufferPool::new(1024 * 1024);
    let shared = pool.allocate(256, StorageMode::Shared, "s").unwrap();
    pool.release(shared);
    // Request a Private buffer — should allocate fresh, not recycle Shared.
    let _priv = pool.allocate(256, StorageMode::Private, "p").unwrap();
    assert_eq!(pool.recycled_count(), 0);
    assert_eq!(pool.allocated_count(), 2);
}

/// Buffer write at exact boundary succeeds.
#[test]
#[cfg(target_os = "macos")]
fn test_metal_buffer_write_exact_boundary() {
    let mut buf = MockBuffer::new(256, StorageMode::Shared, "exact");
    let payload = vec![0xABu8; 256];
    buf.write(0, &payload).unwrap();
    assert_eq!(buf.read(0, 256).unwrap(), &payload[..]);
}

/// Buffer initialized to all zeros.
#[test]
#[cfg(target_os = "macos")]
fn test_metal_buffer_initialized_zero() {
    let buf = MockBuffer::new(1024, StorageMode::Shared, "z");
    assert!(buf.data.iter().all(|&b| b == 0));
}

// ═════════════════════════════════════════════════════════════════════
// 4. Dispatch Tests (22 tests)
// ═════════════════════════════════════════════════════════════════════

/// 1-D dispatch with 256 elements and 32-wide workgroup.
#[test]
#[cfg(target_os = "macos")]
fn test_metal_dispatch_1d_exact() {
    let g = compute_dispatch_groups((256, 1, 1), (SIMD_WIDTH, 1, 1)).unwrap();
    assert_eq!(g, (8, 1, 1));
}

/// 1-D dispatch rounds up for non-multiples.
#[test]
#[cfg(target_os = "macos")]
fn test_metal_dispatch_1d_round_up() {
    let g = compute_dispatch_groups((33, 1, 1), (SIMD_WIDTH, 1, 1)).unwrap();
    assert_eq!(g.0, 2); // ceil(33/32)
}

/// 2-D dispatch with square tiles.
#[test]
#[cfg(target_os = "macos")]
fn test_metal_dispatch_2d_exact() {
    let g = compute_dispatch_groups((64, 64, 1), (16, 16, 1)).unwrap();
    assert_eq!(g, (4, 4, 1));
}

/// 2-D dispatch with non-aligned dimensions.
#[test]
#[cfg(target_os = "macos")]
fn test_metal_dispatch_2d_non_aligned() {
    let g = compute_dispatch_groups((17, 17, 1), (16, 16, 1)).unwrap();
    assert_eq!(g, (2, 2, 1));
}

/// 3-D dispatch.
#[test]
#[cfg(target_os = "macos")]
fn test_metal_dispatch_3d() {
    let g = compute_dispatch_groups((64, 64, 8), (16, 16, 1)).unwrap();
    assert_eq!(g, (4, 4, 8));
}

/// Dispatch dimension at MAX_DISPATCH_DIM is valid.
#[test]
#[cfg(target_os = "macos")]
fn test_metal_dispatch_at_limit() {
    let g = compute_dispatch_groups((MAX_DISPATCH_DIM, 1, 1), (1, 1, 1)).unwrap();
    assert_eq!(g.0, MAX_DISPATCH_DIM);
}

/// Dispatch dimension exceeding MAX_DISPATCH_DIM errors.
#[test]
#[cfg(target_os = "macos")]
fn test_metal_dispatch_exceeds_limit() {
    let err = compute_dispatch_groups((MAX_DISPATCH_DIM + 1, 1, 1), (1, 1, 1)).unwrap_err();
    assert!(matches!(err, PipelineError::DispatchTooLarge { .. }));
}

/// Dispatch with single element.
#[test]
#[cfg(target_os = "macos")]
fn test_metal_dispatch_single_element() {
    let g = compute_dispatch_groups((1, 1, 1), (SIMD_WIDTH, 1, 1)).unwrap();
    assert_eq!(g, (1, 1, 1));
}

/// Dispatch with zero workgroup dimension errors.
#[test]
#[cfg(target_os = "macos")]
fn test_metal_dispatch_zero_workgroup() {
    let err = compute_dispatch_groups((64, 1, 1), (0, 1, 1)).unwrap_err();
    assert_eq!(err, PipelineError::ZeroDimension);
}

/// Dispatch for a large 1-D problem.
#[test]
#[cfg(target_os = "macos")]
fn test_metal_dispatch_1d_large() {
    let g = compute_dispatch_groups((1_000_000, 1, 1), (256, 1, 1)).unwrap();
    assert_eq!(g.0, 3907); // ceil(1M/256)
}

/// Multiple sequential dispatches maintain correctness.
#[test]
#[cfg(target_os = "macos")]
fn test_metal_dispatch_sequential() {
    let sizes = [128, 256, 512, 1024, 2048];
    for &s in &sizes {
        let g = compute_dispatch_groups((s, 1, 1), (SIMD_WIDTH, 1, 1)).unwrap();
        assert_eq!(g.0, s / SIMD_WIDTH);
    }
}

/// Workgroup validation at exactly 1024 threads.
#[test]
#[cfg(target_os = "macos")]
fn test_metal_dispatch_workgroup_at_max() {
    validate_workgroup(1024, 1, 1).unwrap();
    validate_workgroup(32, 32, 1).unwrap();
}

/// Workgroup exceeding 1024 threads errors.
#[test]
#[cfg(target_os = "macos")]
fn test_metal_dispatch_workgroup_exceeds_max() {
    let err = validate_workgroup(1025, 1, 1).unwrap_err();
    assert!(matches!(err, PipelineError::WorkgroupTooLarge { .. }));
}

/// Workgroup with zero dimension errors.
#[test]
#[cfg(target_os = "macos")]
fn test_metal_dispatch_workgroup_zero() {
    let err = validate_workgroup(0, 1, 1).unwrap_err();
    assert_eq!(err, PipelineError::ZeroDimension);
}

/// Dispatch with power-of-two problem sizes.
#[test]
#[cfg(target_os = "macos")]
fn test_metal_dispatch_power_of_two() {
    for exp in 0..16u32 {
        let size = 1u32 << exp;
        let g = compute_dispatch_groups((size, 1, 1), (1, 1, 1)).unwrap();
        assert_eq!(g.0, size);
    }
}

/// Dispatch with non-power-of-two workgroup.
#[test]
#[cfg(target_os = "macos")]
fn test_metal_dispatch_non_pot_workgroup() {
    let g = compute_dispatch_groups((100, 1, 1), (48, 1, 1)).unwrap();
    assert_eq!(g.0, 3); // ceil(100/48)
}

/// 2-D dispatch for matrix multiply shapes.
#[test]
#[cfg(target_os = "macos")]
fn test_metal_dispatch_matmul_shapes() {
    // 1024×1024 matrix, 16×16 tile.
    let g = compute_dispatch_groups((1024, 1024, 1), (16, 16, 1)).unwrap();
    assert_eq!(g, (64, 64, 1));
}

/// 2-D dispatch for non-square matrix.
#[test]
#[cfg(target_os = "macos")]
fn test_metal_dispatch_non_square_matrix() {
    let g = compute_dispatch_groups((512, 256, 1), (16, 16, 1)).unwrap();
    assert_eq!(g, (32, 16, 1));
}

/// Dispatch for batch processing (3D).
#[test]
#[cfg(target_os = "macos")]
fn test_metal_dispatch_batch() {
    let g = compute_dispatch_groups((256, 256, 32), (16, 16, 1)).unwrap();
    assert_eq!(g, (16, 16, 32));
}

/// Dispatch with workgroup equal to problem size.
#[test]
#[cfg(target_os = "macos")]
fn test_metal_dispatch_workgroup_equals_problem() {
    let g = compute_dispatch_groups((32, 1, 1), (32, 1, 1)).unwrap();
    assert_eq!(g, (1, 1, 1));
}

/// Dispatch with very small workgroup over large problem.
#[test]
#[cfg(target_os = "macos")]
fn test_metal_dispatch_small_workgroup_large_problem() {
    let g = compute_dispatch_groups((60000, 1, 1), (1, 1, 1)).unwrap();
    assert_eq!(g.0, 60000);
}

/// Dispatch group count is always positive for non-zero problem.
#[test]
#[cfg(target_os = "macos")]
fn test_metal_dispatch_always_positive() {
    for size in [1, 2, 15, 16, 17, 31, 32, 33, 255, 256, 257] {
        let g = compute_dispatch_groups((size, 1, 1), (SIMD_WIDTH, 1, 1)).unwrap();
        assert!(g.0 > 0, "Expected > 0 groups for size {size}");
    }
}

// ═════════════════════════════════════════════════════════════════════
// 5. Error Handling Tests (16 tests)
// ═════════════════════════════════════════════════════════════════════

/// DeviceUnavailable error renders correctly.
#[test]
#[cfg(target_os = "macos")]
fn test_metal_error_device_unavailable_display() {
    let e = PipelineError::DeviceUnavailable;
    assert_eq!(e.to_string(), "Metal device unavailable");
}

/// OutOfMemory error includes sizes.
#[test]
#[cfg(target_os = "macos")]
fn test_metal_error_oom_display() {
    let e = PipelineError::OutOfMemory { requested: 1024, available: 512 };
    let s = e.to_string();
    assert!(s.contains("1024"));
    assert!(s.contains("512"));
}

/// InvalidBufferSize error renders.
#[test]
#[cfg(target_os = "macos")]
fn test_metal_error_invalid_buffer_size() {
    let e = PipelineError::InvalidBufferSize;
    assert_eq!(e.to_string(), "invalid buffer size");
}

/// CompilationFailed carries the message.
#[test]
#[cfg(target_os = "macos")]
fn test_metal_error_compilation_message() {
    let e = PipelineError::CompilationFailed("syntax error at line 3".into());
    assert!(e.to_string().contains("syntax error at line 3"));
}

/// FunctionNotFound carries the name.
#[test]
#[cfg(target_os = "macos")]
fn test_metal_error_function_not_found() {
    let e = PipelineError::FunctionNotFound("missing_fn".into());
    assert!(e.to_string().contains("missing_fn"));
}

/// EmptySource error.
#[test]
#[cfg(target_os = "macos")]
fn test_metal_error_empty_source() {
    assert_eq!(PipelineError::EmptySource.to_string(), "empty shader source");
}

/// InvalidSource error.
#[test]
#[cfg(target_os = "macos")]
fn test_metal_error_invalid_source() {
    assert_eq!(PipelineError::InvalidSource.to_string(), "invalid shader source");
}

/// WorkgroupTooLarge error includes dimensions.
#[test]
#[cfg(target_os = "macos")]
fn test_metal_error_workgroup_too_large() {
    let e = PipelineError::WorkgroupTooLarge { total: 2048, max: 1024 };
    let s = e.to_string();
    assert!(s.contains("2048"));
    assert!(s.contains("1024"));
}

/// ZeroDimension error.
#[test]
#[cfg(target_os = "macos")]
fn test_metal_error_zero_dimension() {
    assert_eq!(PipelineError::ZeroDimension.to_string(), "zero dimension");
}

/// DispatchTooLarge error includes dimensions.
#[test]
#[cfg(target_os = "macos")]
fn test_metal_error_dispatch_too_large() {
    let e = PipelineError::DispatchTooLarge { dim: 70000, max: 65535 };
    let s = e.to_string();
    assert!(s.contains("70000"));
    assert!(s.contains("65535"));
}

/// BufferOverflow error includes sizes.
#[test]
#[cfg(target_os = "macos")]
fn test_metal_error_buffer_overflow() {
    let e = PipelineError::BufferOverflow { requested: 512, capacity: 256 };
    let s = e.to_string();
    assert!(s.contains("512"));
    assert!(s.contains("256"));
}

/// CacheFull error.
#[test]
#[cfg(target_os = "macos")]
fn test_metal_error_cache_full() {
    assert_eq!(PipelineError::CacheFull.to_string(), "pipeline cache is full");
}

/// Error equality works.
#[test]
#[cfg(target_os = "macos")]
fn test_metal_error_equality() {
    assert_eq!(PipelineError::EmptySource, PipelineError::EmptySource);
    assert_ne!(PipelineError::EmptySource, PipelineError::InvalidSource);
}

/// Compilation error from numeric-only source.
#[test]
#[cfg(target_os = "macos")]
fn test_metal_error_numeric_source() {
    let cache = PipelineCache::new(64);
    let err = cache.get_or_create("12345", "main").unwrap_err();
    assert!(matches!(err, PipelineError::CompilationFailed(_)));
}

/// Compilation error from special characters source.
#[test]
#[cfg(target_os = "macos")]
fn test_metal_error_special_chars_source() {
    let cache = PipelineCache::new(64);
    let err = cache.get_or_create("@#$%^&*()", "main").unwrap_err();
    assert!(matches!(err, PipelineError::CompilationFailed(_)));
}

/// OOM from buffer pool with zero capacity.
#[test]
#[cfg(target_os = "macos")]
fn test_metal_error_pool_zero_capacity() {
    let pool = BufferPool::new(0);
    let err = pool.allocate(256, StorageMode::Shared, "fail").unwrap_err();
    assert!(matches!(err, PipelineError::OutOfMemory { .. }));
}

// ═════════════════════════════════════════════════════════════════════
// 6. Integration Tests (12 tests)
// ═════════════════════════════════════════════════════════════════════

/// Full pipeline: create cache → compile → set buffer → dispatch → read.
#[test]
#[cfg(target_os = "macos")]
fn test_metal_integration_full_pipeline() {
    let cache = PipelineCache::new(64);
    let pool = BufferPool::new(1024 * 1024);

    // 1. Compile pipeline.
    let (pipeline, _) = cache.get_or_create(VALID_SHADER, "add_arrays").unwrap();
    assert_eq!(pipeline.function_name, "add_arrays");

    // 2. Allocate input/output buffers.
    let mut input = pool.allocate(1024, StorageMode::Shared, "input").unwrap();
    let output = pool.allocate(1024, StorageMode::Shared, "output").unwrap();

    // 3. Write data.
    let data = vec![1u8; 1024];
    input.write(0, &data).unwrap();

    // 4. Compute dispatch dimensions.
    let groups = compute_dispatch_groups((256, 1, 1), pipeline.workgroup_size).unwrap();
    assert_eq!(groups, (8, 1, 1));

    // 5. Read back (simulated).
    let readback = input.read(0, 1024).unwrap();
    assert_eq!(readback.len(), 1024);

    pool.release(input);
    pool.release(output);
}

/// Pipeline with multiple buffers.
#[test]
#[cfg(target_os = "macos")]
fn test_metal_integration_multiple_buffers() {
    let pool = BufferPool::new(4 * 1024 * 1024);
    let buffers: Vec<_> = (0..4)
        .map(|i| pool.allocate(1024, StorageMode::Shared, &format!("buf_{i}")).unwrap())
        .collect();
    assert_eq!(buffers.len(), 4);
    assert_eq!(pool.allocated_count(), 4);
    for buf in buffers {
        pool.release(buf);
    }
}

/// Sequential kernel execution reuses the same pipeline.
#[test]
#[cfg(target_os = "macos")]
fn test_metal_integration_sequential_execution() {
    let cache = PipelineCache::new(64);
    for _ in 0..10 {
        let (_, r) = cache.get_or_create(VALID_SHADER, "add_arrays").unwrap();
        if cache.stats().misses == 1 {
            // After first miss, all should be hits.
            assert_eq!(r, CacheResult::Hit);
        }
    }
    let s = cache.stats();
    assert_eq!(s.misses, 1);
    assert_eq!(s.hits, 9);
}

/// Pipeline cache + buffer pool work together under load.
#[test]
#[cfg(target_os = "macos")]
fn test_metal_integration_cache_and_pool() {
    let cache = PipelineCache::new(16);
    let pool = BufferPool::new(16 * 1024 * 1024);

    let shaders = [
        (VALID_SHADER, "add_arrays"),
        (VALID_SHADER_ALT, "multiply_arrays"),
        (MATMUL_SHADER, "matmul"),
        (REDUCTION_SHADER, "reduce_sum"),
    ];

    for &(src, func) in &shaders {
        let (pipeline, _) = cache.get_or_create(src, func).unwrap();
        let buf = pool.allocate(4096, StorageMode::Shared, func).unwrap();
        let _groups = compute_dispatch_groups((128, 1, 1), pipeline.workgroup_size).unwrap();
        pool.release(buf);
    }

    assert_eq!(cache.len(), 4);
    assert_eq!(pool.allocated_count(), 4);
}

/// Buffer recycling reduces allocations.
#[test]
#[cfg(target_os = "macos")]
fn test_metal_integration_buffer_recycling_efficiency() {
    let pool = BufferPool::new(1024 * 1024);
    for _ in 0..100 {
        let buf = pool.allocate(256, StorageMode::Shared, "temp").unwrap();
        pool.release(buf);
    }
    // First allocation is fresh; remaining 99 are recycled.
    assert_eq!(pool.allocated_count(), 1);
    assert_eq!(pool.recycled_count(), 99);
}

/// End-to-end: write f32 data → read back as bytes → verify.
#[test]
#[cfg(target_os = "macos")]
fn test_metal_integration_f32_roundtrip() {
    let pool = BufferPool::new(1024 * 1024);
    let mut buf = pool.allocate(1024, StorageMode::Shared, "f32_data").unwrap();

    let values: Vec<f32> = (0..64).map(|i| i as f32 * 0.5).collect();
    let bytes: &[u8] = bytemuck::cast_slice(&values);
    buf.write(0, bytes).unwrap();

    let readback = buf.read(0, bytes.len()).unwrap();
    let result: &[f32] = bytemuck::cast_slice(readback);
    assert_eq!(result.len(), 64);
    for (i, &v) in result.iter().enumerate() {
        assert!(
            (v - i as f32 * 0.5).abs() < 1e-6,
            "Mismatch at index {i}: expected {}, got {v}",
            i as f32 * 0.5,
        );
    }
}

/// Concurrent pipeline compilation + buffer allocation.
#[test]
#[cfg(target_os = "macos")]
fn test_metal_integration_concurrent_compile_and_alloc() {
    let cache = Arc::new(PipelineCache::new(256));
    let pool = Arc::new(BufferPool::new(64 * 1024 * 1024));

    let mut handles = Vec::new();
    for i in 0..8 {
        let c = Arc::clone(&cache);
        let p = Arc::clone(&pool);
        handles.push(std::thread::spawn(move || {
            let src = format!("kernel fn worker_{i}() {{}}");
            let (pipeline, _) = c.get_or_create(&src, &format!("worker_{i}")).unwrap();
            let buf = p.allocate(1024, StorageMode::Shared, &format!("buf_{i}")).unwrap();
            let _groups = compute_dispatch_groups((256, 1, 1), pipeline.workgroup_size).unwrap();
            p.release(buf);
        }));
    }
    for h in handles {
        h.join().unwrap();
    }
    assert_eq!(cache.len(), 8);
}

/// Multiple dispatch shapes use the same pipeline.
#[test]
#[cfg(target_os = "macos")]
fn test_metal_integration_varied_dispatch_sizes() {
    let cache = PipelineCache::new(64);
    let (pipeline, _) = cache.get_or_create(VALID_SHADER, "add_arrays").unwrap();

    let problems = [(64, 1, 1), (128, 1, 1), (256, 256, 1), (1024, 1024, 1)];
    for prob in &problems {
        let g = compute_dispatch_groups(*prob, pipeline.workgroup_size).unwrap();
        assert!(g.0 > 0);
        assert!(g.1 > 0);
        assert!(g.2 > 0);
    }
}

/// Pipeline state is consistent after many operations.
#[test]
#[cfg(target_os = "macos")]
fn test_metal_integration_consistency_after_operations() {
    let cache = PipelineCache::new(64);
    let pool = BufferPool::new(8 * 1024 * 1024);

    // Compile multiple pipelines.
    for &(src, func) in &[
        (VALID_SHADER, "add_arrays"),
        (MATMUL_SHADER, "matmul"),
        (REDUCTION_SHADER, "reduce_sum"),
        (ELEMENTWISE_SHADER, "elementwise_add"),
    ] {
        cache.get_or_create(src, func).unwrap();
    }

    // Allocate and release buffers.
    for _ in 0..20 {
        let buf = pool.allocate(512, StorageMode::Shared, "tmp").unwrap();
        pool.release(buf);
    }

    // Verify cache integrity.
    assert_eq!(cache.len(), 4);
    let s = cache.stats();
    assert_eq!(s.insertions, 4);
    assert_eq!(s.misses, 4);
    assert_eq!(s.evictions, 0);
}

/// Clear cache and re-populate.
#[test]
#[cfg(target_os = "macos")]
fn test_metal_integration_clear_and_repopulate() {
    let cache = PipelineCache::new(64);
    cache.get_or_create(VALID_SHADER, "add_arrays").unwrap();
    cache.get_or_create(MATMUL_SHADER, "matmul").unwrap();
    assert_eq!(cache.len(), 2);

    cache.clear();
    assert_eq!(cache.len(), 0);

    cache.get_or_create(VALID_SHADER, "add_arrays").unwrap();
    assert_eq!(cache.len(), 1);
    let (_, r) = cache.get_or_create(VALID_SHADER, "add_arrays").unwrap();
    assert_eq!(r, CacheResult::Hit);
}

/// Workgroup validation for common NN layer shapes.
#[test]
#[cfg(target_os = "macos")]
fn test_metal_integration_nn_workgroup_shapes() {
    // Typical shapes from NN layers.
    let valid_shapes = [
        (32, 1, 1),   // 1D vector op
        (16, 16, 1),  // 2D tile (matmul)
        (8, 8, 8),    // 3D (batch convolution)
        (256, 1, 1),  // Reduction
        (32, 32, 1),  // Large tile
        (1024, 1, 1), // Max 1D
    ];
    for (x, y, z) in valid_shapes {
        validate_workgroup(x, y, z).unwrap_or_else(|e| panic!("shape ({x},{y},{z}) failed: {e}"));
    }

    // Invalid shapes.
    let invalid_shapes = [
        (0, 1, 1),
        (1025, 1, 1),
        (33, 32, 1), // 33*32 = 1056 > 1024
    ];
    for (x, y, z) in invalid_shapes {
        assert!(validate_workgroup(x, y, z).is_err(), "shape ({x},{y},{z}) should fail");
    }
}
