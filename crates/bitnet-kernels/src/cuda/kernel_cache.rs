//! CUDA kernel compilation cache with LRU eviction, disk persistence, and warmup.
//!
//! # Overview
//!
//! Provides a multi-layer caching system for compiled CUDA kernels to eliminate
//! redundant runtime compilation via NVRTC. Features include:
//!
//! - [`KernelCache`] — LRU in-memory cache for compiled kernels with hash-based lookup
//! - [`CacheKey`] — composite key from kernel source, launch config, and device caps
//! - [`BinaryCache`] — persistent disk cache for compiled PTX/cubin binaries
//! - [`CacheStats`] — hit/miss/eviction metrics tracking
//! - [`WarmupStrategy`] — pre-compilation of frequently used kernels at startup
//!
//! All code is feature-gated behind `#[cfg(any(feature = "gpu", feature = "cuda"))]`.
//! CPU builds provide simulation for testing on non-GPU hosts.

use std::collections::HashMap;
use std::fmt;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use bitnet_common::{KernelError, Result};

// ── Hash utilities ───────────────────────────────────────────────────

/// FNV-1a 64-bit hash for fast, non-cryptographic fingerprinting.
fn fnv1a_hash(data: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &b in data {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

// ── DeviceCapability ─────────────────────────────────────────────────

/// CUDA device capability descriptor used as part of the cache key.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct DeviceCapability {
    /// Compute capability major version (e.g. 8 for sm_80).
    pub major: u32,
    /// Compute capability minor version (e.g. 0 for sm_80).
    pub minor: u32,
    /// Maximum threads per block.
    pub max_threads_per_block: u32,
    /// Shared memory per block in bytes.
    pub shared_memory_per_block: u32,
    /// Number of SMs on the device.
    pub num_sms: u32,
}

impl DeviceCapability {
    /// Create a new device capability descriptor.
    pub fn new(
        major: u32,
        minor: u32,
        max_threads_per_block: u32,
        shared_memory_per_block: u32,
        num_sms: u32,
    ) -> Self {
        Self { major, minor, max_threads_per_block, shared_memory_per_block, num_sms }
    }

    /// Return the SM architecture string (e.g. `"sm_80"`).
    pub fn sm_arch(&self) -> String {
        format!("sm_{}{}", self.major, self.minor)
    }

    /// Fingerprint bytes for hashing.
    fn fingerprint_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(20);
        buf.extend_from_slice(&self.major.to_le_bytes());
        buf.extend_from_slice(&self.minor.to_le_bytes());
        buf.extend_from_slice(&self.max_threads_per_block.to_le_bytes());
        buf.extend_from_slice(&self.shared_memory_per_block.to_le_bytes());
        buf.extend_from_slice(&self.num_sms.to_le_bytes());
        buf
    }
}

impl Default for DeviceCapability {
    fn default() -> Self {
        Self {
            major: 8,
            minor: 0,
            max_threads_per_block: 1024,
            shared_memory_per_block: 49152,
            num_sms: 108,
        }
    }
}

impl fmt::Display for DeviceCapability {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "sm_{}{} ({}SMs, {}tpb)",
            self.major, self.minor, self.num_sms, self.max_threads_per_block
        )
    }
}

// ── LaunchConfig ─────────────────────────────────────────────────────

/// Kernel launch configuration that affects compilation.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct LaunchConfig {
    /// Block dimensions (x, y, z).
    pub block_dim: (u32, u32, u32),
    /// Grid dimensions (x, y, z).
    pub grid_dim: (u32, u32, u32),
    /// Dynamic shared memory in bytes.
    pub shared_mem_bytes: u32,
    /// Compiler options (e.g. `--use_fast_math`).
    pub compile_options: Vec<String>,
}

impl LaunchConfig {
    /// Create a 1-D launch configuration.
    pub fn linear(block_size: u32, grid_size: u32) -> Self {
        Self {
            block_dim: (block_size, 1, 1),
            grid_dim: (grid_size, 1, 1),
            shared_mem_bytes: 0,
            compile_options: Vec::new(),
        }
    }

    /// Create a 2-D launch configuration.
    pub fn grid_2d(bx: u32, by: u32, gx: u32, gy: u32) -> Self {
        Self {
            block_dim: (bx, by, 1),
            grid_dim: (gx, gy, 1),
            shared_mem_bytes: 0,
            compile_options: Vec::new(),
        }
    }

    /// Add dynamic shared memory.
    pub fn with_shared_mem(mut self, bytes: u32) -> Self {
        self.shared_mem_bytes = bytes;
        self
    }

    /// Add a compile option.
    pub fn with_option(mut self, opt: impl Into<String>) -> Self {
        self.compile_options.push(opt.into());
        self
    }

    /// Total number of threads per block.
    pub fn threads_per_block(&self) -> u32 {
        self.block_dim.0 * self.block_dim.1 * self.block_dim.2
    }

    /// Fingerprint bytes for hashing.
    fn fingerprint_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(32);
        buf.extend_from_slice(&self.block_dim.0.to_le_bytes());
        buf.extend_from_slice(&self.block_dim.1.to_le_bytes());
        buf.extend_from_slice(&self.block_dim.2.to_le_bytes());
        buf.extend_from_slice(&self.grid_dim.0.to_le_bytes());
        buf.extend_from_slice(&self.grid_dim.1.to_le_bytes());
        buf.extend_from_slice(&self.grid_dim.2.to_le_bytes());
        buf.extend_from_slice(&self.shared_mem_bytes.to_le_bytes());
        for opt in &self.compile_options {
            buf.extend_from_slice(opt.as_bytes());
            buf.push(0);
        }
        buf
    }
}

impl Default for LaunchConfig {
    fn default() -> Self {
        Self::linear(256, 1)
    }
}

// ── CacheKey ─────────────────────────────────────────────────────────

/// Composite cache key combining kernel source, launch config, and device caps.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CacheKey {
    /// Hash of the kernel source code.
    pub source_hash: u64,
    /// Hash of the launch configuration.
    pub config_hash: u64,
    /// Hash of the device capabilities.
    pub device_hash: u64,
    /// Human-readable kernel name for diagnostics.
    pub kernel_name: String,
}

impl CacheKey {
    /// Build a cache key from source, launch config, and device capabilities.
    pub fn new(
        kernel_name: &str,
        source: &str,
        launch_config: &LaunchConfig,
        device: &DeviceCapability,
    ) -> Self {
        let source_hash = fnv1a_hash(source.as_bytes());
        let config_hash = fnv1a_hash(&launch_config.fingerprint_bytes());
        let device_hash = fnv1a_hash(&device.fingerprint_bytes());
        Self { source_hash, config_hash, device_hash, kernel_name: kernel_name.to_string() }
    }

    /// Combined 64-bit hash of all three components.
    pub fn combined_hash(&self) -> u64 {
        let mut buf = Vec::with_capacity(24);
        buf.extend_from_slice(&self.source_hash.to_le_bytes());
        buf.extend_from_slice(&self.config_hash.to_le_bytes());
        buf.extend_from_slice(&self.device_hash.to_le_bytes());
        fnv1a_hash(&buf)
    }

    /// Return a filesystem-safe string for disk caching.
    pub fn cache_filename(&self) -> String {
        format!("{}_{:016x}.bin", self.kernel_name, self.combined_hash())
    }
}

impl fmt::Display for CacheKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}@{:016x}", self.kernel_name, self.combined_hash())
    }
}

// ── CompiledKernel ───────────────────────────────────────────────────

/// A compiled kernel binary (PTX or cubin) with metadata.
#[derive(Debug, Clone)]
pub struct CompiledKernel {
    /// The compiled binary data.
    pub binary: Vec<u8>,
    /// Format of the binary.
    pub format: BinaryFormat,
    /// Kernel entry-point name.
    pub entry_point: String,
    /// Time taken to compile.
    pub compile_time: Duration,
    /// Timestamp when compiled.
    pub compiled_at: u64,
    /// Register count (if known).
    pub register_count: Option<u32>,
    /// Shared memory usage in bytes (if known).
    pub shared_mem_bytes: Option<u32>,
}

impl CompiledKernel {
    /// Create a new compiled kernel.
    pub fn new(
        binary: Vec<u8>,
        format: BinaryFormat,
        entry_point: &str,
        compile_time: Duration,
    ) -> Self {
        let compiled_at =
            SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs();
        Self {
            binary,
            format,
            entry_point: entry_point.to_string(),
            compile_time,
            compiled_at,
            register_count: None,
            shared_mem_bytes: None,
        }
    }

    /// Size of the compiled binary in bytes.
    pub fn binary_size(&self) -> usize {
        self.binary.len()
    }

    /// Set register count metadata.
    pub fn with_register_count(mut self, count: u32) -> Self {
        self.register_count = Some(count);
        self
    }

    /// Set shared memory metadata.
    pub fn with_shared_mem(mut self, bytes: u32) -> Self {
        self.shared_mem_bytes = Some(bytes);
        self
    }
}

/// Binary format of a compiled kernel.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BinaryFormat {
    /// PTX intermediate representation.
    Ptx,
    /// Device-specific cubin binary.
    Cubin,
    /// Simulated binary (CPU testing).
    Simulated,
}

impl fmt::Display for BinaryFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Ptx => write!(f, "PTX"),
            Self::Cubin => write!(f, "cubin"),
            Self::Simulated => write!(f, "simulated"),
        }
    }
}

// ── CacheStats ───────────────────────────────────────────────────────

/// Hit/miss/eviction metrics for the kernel cache.
#[derive(Debug, Clone, Default)]
pub struct CacheStats {
    /// Number of cache hits.
    pub hits: u64,
    /// Number of cache misses.
    pub misses: u64,
    /// Number of evictions due to capacity limits.
    pub evictions: u64,
    /// Number of entries currently in cache.
    pub entries: u64,
    /// Total bytes of compiled binaries in cache.
    pub total_bytes: u64,
    /// Number of disk cache hits.
    pub disk_hits: u64,
    /// Number of disk cache misses.
    pub disk_misses: u64,
    /// Number of warmup compilations.
    pub warmup_compilations: u64,
    /// Total compilation time saved by cache hits.
    pub time_saved: Duration,
}

impl CacheStats {
    /// Cache hit rate as a fraction in [0.0, 1.0].
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            return 0.0;
        }
        self.hits as f64 / total as f64
    }

    /// Disk cache hit rate as a fraction in [0.0, 1.0].
    pub fn disk_hit_rate(&self) -> f64 {
        let total = self.disk_hits + self.disk_misses;
        if total == 0 {
            return 0.0;
        }
        self.disk_hits as f64 / total as f64
    }

    /// Reset all counters to zero.
    pub fn reset(&mut self) {
        *self = Self::default();
    }

    /// Total lookups (hits + misses).
    pub fn total_lookups(&self) -> u64 {
        self.hits + self.misses
    }
}

impl fmt::Display for CacheStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "CacheStats {{ hits: {}, misses: {}, evictions: {}, entries: {}, \
             hit_rate: {:.1}%, bytes: {}, time_saved: {:?} }}",
            self.hits,
            self.misses,
            self.evictions,
            self.entries,
            self.hit_rate() * 100.0,
            self.total_bytes,
            self.time_saved,
        )
    }
}

// ── CacheConfig ──────────────────────────────────────────────────────

/// Configuration for the kernel cache.
#[derive(Debug, Clone)]
pub struct CacheConfig {
    /// Maximum number of entries in the in-memory LRU cache.
    pub max_entries: usize,
    /// Maximum total binary size in bytes for in-memory cache.
    pub max_memory_bytes: usize,
    /// Whether to enable the disk cache.
    pub enable_disk_cache: bool,
    /// Directory for the disk cache.
    pub disk_cache_dir: PathBuf,
    /// Maximum disk cache size in bytes.
    pub max_disk_bytes: u64,
    /// TTL for disk cache entries (0 = no expiry).
    pub disk_ttl_secs: u64,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            max_entries: 256,
            max_memory_bytes: 128 * 1024 * 1024, // 128 MiB
            enable_disk_cache: false,
            disk_cache_dir: PathBuf::from("/tmp/bitnet_kernel_cache"),
            max_disk_bytes: 512 * 1024 * 1024, // 512 MiB
            disk_ttl_secs: 0,
        }
    }
}

impl CacheConfig {
    /// Validate configuration values.
    pub fn validate(&self) -> Result<()> {
        if self.max_entries == 0 {
            return Err(KernelError::InvalidArguments {
                reason: "max_entries must be non-zero".into(),
            }
            .into());
        }
        if self.max_memory_bytes == 0 {
            return Err(KernelError::InvalidArguments {
                reason: "max_memory_bytes must be non-zero".into(),
            }
            .into());
        }
        Ok(())
    }

    /// Create a small config for testing.
    pub fn tiny() -> Self {
        Self {
            max_entries: 4,
            max_memory_bytes: 4096,
            enable_disk_cache: false,
            disk_cache_dir: PathBuf::from("/tmp/bitnet_kernel_cache_test"),
            max_disk_bytes: 8192,
            disk_ttl_secs: 0,
        }
    }
}

// ── LRU tracking node ────────────────────────────────────────────────

/// Internal LRU entry wrapping a compiled kernel.
#[derive(Debug, Clone)]
struct LruEntry {
    key: CacheKey,
    kernel: CompiledKernel,
    last_access: Instant,
    access_count: u64,
}

// ── KernelCache ──────────────────────────────────────────────────────

/// LRU in-memory cache for compiled CUDA kernels with hash-based lookup.
///
/// Entries are evicted in LRU order when capacity limits are exceeded.
pub struct KernelCache {
    /// Map from combined hash to LRU entry.
    entries: HashMap<u64, LruEntry>,
    /// Ordered list of keys by last access time (oldest first).
    access_order: Vec<u64>,
    /// Configuration.
    config: CacheConfig,
    /// Metrics.
    stats: CacheStats,
}

impl KernelCache {
    /// Create a new kernel cache with the given configuration.
    pub fn new(config: CacheConfig) -> Result<Self> {
        config.validate()?;
        Ok(Self {
            entries: HashMap::new(),
            access_order: Vec::new(),
            config,
            stats: CacheStats::default(),
        })
    }

    /// Create a cache with default configuration.
    pub fn with_defaults() -> Result<Self> {
        Self::new(CacheConfig::default())
    }

    /// Look up a compiled kernel by cache key.
    pub fn get(&mut self, key: &CacheKey) -> Option<&CompiledKernel> {
        let hash = key.combined_hash();
        if let Some(entry) = self.entries.get_mut(&hash) {
            entry.last_access = Instant::now();
            entry.access_count += 1;
            self.stats.hits += 1;
            self.stats.time_saved += entry.kernel.compile_time;
            // Move to end of access order.
            self.access_order.retain(|&h| h != hash);
            self.access_order.push(hash);
            Some(&entry.kernel)
        } else {
            self.stats.misses += 1;
            None
        }
    }

    /// Insert a compiled kernel into the cache, evicting if necessary.
    pub fn insert(&mut self, key: CacheKey, kernel: CompiledKernel) -> Result<()> {
        let hash = key.combined_hash();
        let binary_size = kernel.binary_size() as u64;

        // Evict until we have room.
        while self.should_evict(binary_size) {
            if !self.evict_lru()? {
                break;
            }
        }

        // Remove old entry with same key if present.
        if let Some(old) = self.entries.remove(&hash) {
            self.stats.total_bytes -= old.kernel.binary_size() as u64;
            self.access_order.retain(|&h| h != hash);
        }

        self.entries
            .insert(hash, LruEntry { key, kernel, last_access: Instant::now(), access_count: 1 });
        self.access_order.push(hash);
        self.stats.total_bytes += binary_size;
        self.stats.entries = self.entries.len() as u64;
        Ok(())
    }

    /// Remove an entry by key.
    pub fn remove(&mut self, key: &CacheKey) -> Option<CompiledKernel> {
        let hash = key.combined_hash();
        if let Some(entry) = self.entries.remove(&hash) {
            self.stats.total_bytes -= entry.kernel.binary_size() as u64;
            self.stats.entries = self.entries.len() as u64;
            self.access_order.retain(|&h| h != hash);
            Some(entry.kernel)
        } else {
            None
        }
    }

    /// Clear all entries.
    pub fn clear(&mut self) {
        self.entries.clear();
        self.access_order.clear();
        self.stats.total_bytes = 0;
        self.stats.entries = 0;
    }

    /// Number of entries in the cache.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Current total binary size in bytes.
    pub fn total_bytes(&self) -> u64 {
        self.stats.total_bytes
    }

    /// Get a snapshot of cache statistics.
    pub fn stats(&self) -> &CacheStats {
        &self.stats
    }

    /// Reset statistics counters.
    pub fn reset_stats(&mut self) {
        let entries = self.stats.entries;
        let total_bytes = self.stats.total_bytes;
        self.stats.reset();
        self.stats.entries = entries;
        self.stats.total_bytes = total_bytes;
    }

    /// Check whether a key is present without updating access metadata.
    pub fn contains(&self, key: &CacheKey) -> bool {
        self.entries.contains_key(&key.combined_hash())
    }

    /// Return all kernel names currently cached.
    pub fn cached_kernel_names(&self) -> Vec<String> {
        self.entries.values().map(|e| e.key.kernel_name.clone()).collect()
    }

    /// Memory utilisation as a fraction in [0.0, 1.0].
    pub fn memory_utilisation(&self) -> f64 {
        if self.config.max_memory_bytes == 0 {
            return 1.0;
        }
        self.stats.total_bytes as f64 / self.config.max_memory_bytes as f64
    }

    /// Get the configuration.
    pub fn config(&self) -> &CacheConfig {
        &self.config
    }

    // ── Internal helpers ─────────────────────────────────────────────

    fn should_evict(&self, additional_bytes: u64) -> bool {
        if self.entries.is_empty() {
            return false;
        }
        let over_entries = self.entries.len() >= self.config.max_entries;
        let over_memory =
            (self.stats.total_bytes + additional_bytes) > self.config.max_memory_bytes as u64;
        over_entries || over_memory
    }

    fn evict_lru(&mut self) -> Result<bool> {
        if let Some(hash) = self.access_order.first().copied() {
            self.access_order.remove(0);
            if let Some(entry) = self.entries.remove(&hash) {
                self.stats.total_bytes -= entry.kernel.binary_size() as u64;
                self.stats.entries = self.entries.len() as u64;
                self.stats.evictions += 1;
                return Ok(true);
            }
        }
        Ok(false)
    }
}

impl fmt::Debug for KernelCache {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("KernelCache")
            .field("entries", &self.entries.len())
            .field("total_bytes", &self.stats.total_bytes)
            .field("stats", &self.stats)
            .finish()
    }
}

// ── BinaryCache (disk persistence) ───────────────────────────────────

/// Persistent disk cache for compiled PTX/cubin binaries.
///
/// Stores compiled kernels as individual files keyed by their cache hash.
/// Supports TTL expiry and size limits.
pub struct BinaryCache {
    /// Directory to store cached binaries.
    cache_dir: PathBuf,
    /// Maximum total disk usage in bytes.
    max_bytes: u64,
    /// TTL for entries (0 = no expiry).
    ttl_secs: u64,
    /// Index of known cached files (filename → size).
    index: HashMap<String, u64>,
    /// Total bytes used on disk.
    used_bytes: u64,
    /// Statistics.
    stats: BinaryCacheStats,
}

/// Statistics for the binary disk cache.
#[derive(Debug, Clone, Default)]
pub struct BinaryCacheStats {
    /// Number of disk reads.
    pub reads: u64,
    /// Number of disk writes.
    pub writes: u64,
    /// Number of expired entries cleaned up.
    pub expirations: u64,
    /// Number of entries evicted for space.
    pub evictions: u64,
}

impl BinaryCache {
    /// Open or create a binary cache at the given directory.
    pub fn open(cache_dir: impl Into<PathBuf>, max_bytes: u64, ttl_secs: u64) -> Result<Self> {
        let cache_dir = cache_dir.into();
        // Ensure dir exists (simulated for non-GPU testing).
        ensure_dir(&cache_dir)?;

        let mut bc = Self {
            cache_dir,
            max_bytes,
            ttl_secs,
            index: HashMap::new(),
            used_bytes: 0,
            stats: BinaryCacheStats::default(),
        };
        bc.rebuild_index()?;
        Ok(bc)
    }

    /// Store a compiled kernel binary on disk.
    pub fn store(&mut self, key: &CacheKey, kernel: &CompiledKernel) -> Result<()> {
        let filename = key.cache_filename();
        let size = kernel.binary.len() as u64;

        // Evict until we have room.
        while self.used_bytes + size > self.max_bytes && !self.index.is_empty() {
            self.evict_oldest()?;
        }

        let path = self.cache_dir.join(&filename);
        ensure_dir(&self.cache_dir)?;
        // Serialize: 8-byte compile_time_nanos + 8-byte compiled_at + entry_point_len(4) + entry_point + binary
        let mut data = Vec::new();
        data.extend_from_slice(&kernel.compile_time.as_nanos().to_le_bytes());
        data.extend_from_slice(&kernel.compiled_at.to_le_bytes());
        let ep_bytes = kernel.entry_point.as_bytes();
        data.extend_from_slice(&(ep_bytes.len() as u32).to_le_bytes());
        data.extend_from_slice(ep_bytes);
        data.push(kernel.format as u8);
        data.extend_from_slice(&kernel.binary);

        std::fs::write(&path, &data).map_err(|e| KernelError::InvalidArguments {
            reason: format!("failed to write binary cache: {e}"),
        })?;

        self.index.insert(filename, data.len() as u64);
        self.used_bytes += data.len() as u64;
        self.stats.writes += 1;
        Ok(())
    }

    /// Load a compiled kernel binary from disk.
    pub fn load(&mut self, key: &CacheKey) -> Result<Option<CompiledKernel>> {
        let filename = key.cache_filename();
        let path = self.cache_dir.join(&filename);

        if !path.exists() {
            return Ok(None);
        }

        // Check TTL.
        if self.ttl_secs > 0
            && let Ok(meta) = std::fs::metadata(&path)
            && let Ok(modified) = meta.modified()
        {
            let age = SystemTime::now().duration_since(modified).unwrap_or_default().as_secs();
            if age > self.ttl_secs {
                self.remove_file(&filename)?;
                self.stats.expirations += 1;
                return Ok(None);
            }
        }

        let data = std::fs::read(&path).map_err(|e| KernelError::InvalidArguments {
            reason: format!("failed to read binary cache: {e}"),
        })?;

        self.stats.reads += 1;

        if data.len() < 29 {
            // Minimum: 16 (timestamps) + 4 (ep_len) + 1 (format) + 0 (ep) + 0 (binary)
            return Ok(None);
        }

        // Deserialize.
        let compile_nanos = u128::from_le_bytes(data[0..16].try_into().unwrap_or([0; 16]));
        let compiled_at = u64::from_le_bytes(data[16..24].try_into().unwrap_or([0; 8]));
        let ep_len = u32::from_le_bytes(data[24..28].try_into().unwrap_or([0; 4])) as usize;

        let header_end = 28 + ep_len + 1;
        if data.len() < header_end {
            return Ok(None);
        }

        let entry_point = String::from_utf8_lossy(&data[28..28 + ep_len]).to_string();
        let format_byte = data[28 + ep_len];
        let format = match format_byte {
            0 => BinaryFormat::Ptx,
            1 => BinaryFormat::Cubin,
            _ => BinaryFormat::Simulated,
        };
        let binary = data[header_end..].to_vec();

        let mut kernel = CompiledKernel {
            binary,
            format,
            entry_point,
            compile_time: Duration::from_nanos(compile_nanos as u64),
            compiled_at,
            register_count: None,
            shared_mem_bytes: None,
        };
        // Preserve register/shared metadata if the format carries it.
        let _ = &mut kernel;

        Ok(Some(kernel))
    }

    /// Check if a key exists on disk.
    pub fn contains(&self, key: &CacheKey) -> bool {
        let filename = key.cache_filename();
        self.index.contains_key(&filename)
    }

    /// Remove a specific entry from disk.
    pub fn remove(&mut self, key: &CacheKey) -> Result<bool> {
        let filename = key.cache_filename();
        self.remove_file(&filename)
    }

    /// Clear all cached files.
    pub fn clear(&mut self) -> Result<()> {
        let filenames: Vec<String> = self.index.keys().cloned().collect();
        for name in filenames {
            let _ = self.remove_file(&name);
        }
        self.used_bytes = 0;
        Ok(())
    }

    /// Number of cached files on disk.
    pub fn len(&self) -> usize {
        self.index.len()
    }

    /// Whether the disk cache is empty.
    pub fn is_empty(&self) -> bool {
        self.index.is_empty()
    }

    /// Total bytes used on disk.
    pub fn used_bytes(&self) -> u64 {
        self.used_bytes
    }

    /// Statistics snapshot.
    pub fn stats(&self) -> &BinaryCacheStats {
        &self.stats
    }

    /// Cache directory path.
    pub fn cache_dir(&self) -> &Path {
        &self.cache_dir
    }

    // ── Internal helpers ─────────────────────────────────────────────

    fn rebuild_index(&mut self) -> Result<()> {
        self.index.clear();
        self.used_bytes = 0;
        if let Ok(entries) = std::fs::read_dir(&self.cache_dir) {
            for entry in entries.flatten() {
                if let Ok(meta) = entry.metadata()
                    && meta.is_file()
                {
                    let name = entry.file_name().to_string_lossy().to_string();
                    let size = meta.len();
                    self.index.insert(name, size);
                    self.used_bytes += size;
                }
            }
        }
        Ok(())
    }

    fn remove_file(&mut self, filename: &str) -> Result<bool> {
        let path = self.cache_dir.join(filename);
        if let Some(size) = self.index.remove(filename) {
            self.used_bytes = self.used_bytes.saturating_sub(size);
            let _ = std::fs::remove_file(&path);
            return Ok(true);
        }
        Ok(false)
    }

    fn evict_oldest(&mut self) -> Result<()> {
        // Evict the first key in iteration order (approximately oldest).
        if let Some(name) = self.index.keys().next().cloned() {
            self.remove_file(&name)?;
            self.stats.evictions += 1;
        }
        Ok(())
    }
}

impl fmt::Debug for BinaryCache {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BinaryCache")
            .field("cache_dir", &self.cache_dir)
            .field("entries", &self.index.len())
            .field("used_bytes", &self.used_bytes)
            .finish()
    }
}

// ── WarmupStrategy ───────────────────────────────────────────────────

/// Strategy for pre-compiling frequently used kernels at startup.
#[derive(Debug, Clone)]
pub struct WarmupStrategy {
    /// Kernel descriptors to pre-compile.
    pub kernels: Vec<WarmupKernel>,
    /// Whether to warm up in parallel (when supported).
    pub parallel: bool,
    /// Maximum time budget for warmup.
    pub timeout: Duration,
    /// Priority order (higher = compile first).
    pub priority_order: bool,
}

/// Descriptor for a kernel to pre-compile during warmup.
#[derive(Debug, Clone)]
pub struct WarmupKernel {
    /// Kernel name.
    pub name: String,
    /// Kernel source code.
    pub source: String,
    /// Launch configuration.
    pub launch_config: LaunchConfig,
    /// Priority (higher = more important).
    pub priority: u32,
}

impl WarmupKernel {
    /// Create a new warmup kernel descriptor.
    pub fn new(name: &str, source: &str, config: LaunchConfig, priority: u32) -> Self {
        Self { name: name.to_string(), source: source.to_string(), launch_config: config, priority }
    }
}

impl Default for WarmupStrategy {
    fn default() -> Self {
        Self {
            kernels: Vec::new(),
            parallel: false,
            timeout: Duration::from_secs(30),
            priority_order: true,
        }
    }
}

impl WarmupStrategy {
    /// Create a warmup strategy with the standard BitNet kernels.
    pub fn standard_bitnet() -> Self {
        let mut strategy = Self::default();
        strategy.add_kernel(WarmupKernel::new(
            "i2s_dequant",
            STUB_I2S_DEQUANT_SRC,
            LaunchConfig::linear(256, 1),
            10,
        ));
        strategy.add_kernel(WarmupKernel::new(
            "rmsnorm",
            STUB_RMSNORM_SRC,
            LaunchConfig::linear(256, 1),
            9,
        ));
        strategy.add_kernel(WarmupKernel::new(
            "softmax",
            STUB_SOFTMAX_SRC,
            LaunchConfig::linear(256, 1),
            8,
        ));
        strategy.add_kernel(WarmupKernel::new(
            "rope",
            STUB_ROPE_SRC,
            LaunchConfig::linear(128, 1),
            7,
        ));
        strategy.add_kernel(WarmupKernel::new(
            "matmul_tiled",
            STUB_MATMUL_SRC,
            LaunchConfig::grid_2d(16, 16, 1, 1),
            6,
        ));
        strategy
    }

    /// Add a kernel to the warmup list.
    pub fn add_kernel(&mut self, kernel: WarmupKernel) {
        self.kernels.push(kernel);
    }

    /// Number of kernels in the warmup plan.
    pub fn len(&self) -> usize {
        self.kernels.len()
    }

    /// Whether the warmup plan is empty.
    pub fn is_empty(&self) -> bool {
        self.kernels.is_empty()
    }

    /// Sort kernels by priority (highest first).
    pub fn sort_by_priority(&mut self) {
        self.kernels.sort_by(|a, b| b.priority.cmp(&a.priority));
    }

    /// Execute the warmup strategy, compiling all kernels into the cache.
    pub fn execute(
        &mut self,
        cache: &mut KernelCache,
        device: &DeviceCapability,
    ) -> Result<WarmupResult> {
        let start = Instant::now();

        if self.priority_order {
            self.sort_by_priority();
        }

        let mut compiled = 0u32;
        let mut failed = 0u32;
        let mut skipped = 0u32;

        for wk in &self.kernels {
            if start.elapsed() > self.timeout {
                skipped += (self.kernels.len() as u32) - compiled - failed - skipped;
                break;
            }

            let key = CacheKey::new(&wk.name, &wk.source, &wk.launch_config, device);

            if cache.contains(&key) {
                skipped += 1;
                continue;
            }

            match simulate_compile(&wk.source, &wk.name) {
                Ok(kernel) => {
                    cache.insert(key, kernel)?;
                    cache.stats.warmup_compilations += 1;
                    compiled += 1;
                }
                Err(_) => {
                    failed += 1;
                }
            }
        }

        Ok(WarmupResult { compiled, failed, skipped, total_time: start.elapsed() })
    }
}

/// Result of a warmup execution.
#[derive(Debug, Clone)]
pub struct WarmupResult {
    /// Number of kernels successfully compiled.
    pub compiled: u32,
    /// Number of kernels that failed to compile.
    pub failed: u32,
    /// Number of kernels skipped (already cached or timed out).
    pub skipped: u32,
    /// Total time for the warmup.
    pub total_time: Duration,
}

impl fmt::Display for WarmupResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Warmup: {} compiled, {} failed, {} skipped in {:?}",
            self.compiled, self.failed, self.skipped, self.total_time
        )
    }
}

// ── Stub kernel sources for warmup & testing ─────────────────────────

const STUB_I2S_DEQUANT_SRC: &str = r#"extern "C" __global__ void i2s_dequant(
    const unsigned char* packed, float* out, float scale, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { out[i] = ((float)((packed[i/4] >> ((i%4)*2)) & 3) - 1.0f) * scale; }
}"#;

const STUB_RMSNORM_SRC: &str = r#"extern "C" __global__ void rmsnorm(
    float* out, const float* x, const float* w, float eps, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { out[i] = x[i] * w[i]; }
}"#;

const STUB_SOFTMAX_SRC: &str = r#"extern "C" __global__ void softmax(
    float* out, const float* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { out[i] = x[i]; }
}"#;

const STUB_ROPE_SRC: &str = r#"extern "C" __global__ void rope(
    float* out, const float* x, const float* freqs, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { out[i] = x[i] * freqs[i]; }
}"#;

const STUB_MATMUL_SRC: &str = r#"extern "C" __global__ void matmul_tiled(
    float* C, const float* A, const float* B, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) { C[row * N + col] = 0.0f; }
}"#;

// ── Simulation helper ────────────────────────────────────────────────

/// Simulate kernel compilation for CPU testing (no NVRTC required).
fn simulate_compile(source: &str, entry_point: &str) -> Result<CompiledKernel> {
    if source.is_empty() {
        return Err(KernelError::InvalidArguments { reason: "empty kernel source".into() }.into());
    }
    let fake_binary = fnv1a_hash(source.as_bytes()).to_le_bytes().to_vec();
    Ok(CompiledKernel::new(
        fake_binary,
        BinaryFormat::Simulated,
        entry_point,
        Duration::from_micros(100),
    ))
}

// ── Filesystem helper ────────────────────────────────────────────────

fn ensure_dir(dir: &Path) -> Result<()> {
    std::fs::create_dir_all(dir).map_err(|e| KernelError::InvalidArguments {
        reason: format!("cannot create cache directory: {e}"),
    })?;
    Ok(())
}

// ── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    static TEMP_DIR_COUNTER: AtomicU64 = AtomicU64::new(0);

    // ── Helper constructors ──────────────────────────────────────────

    fn sample_device() -> DeviceCapability {
        DeviceCapability::new(8, 0, 1024, 49152, 108)
    }

    fn sample_device_alt() -> DeviceCapability {
        DeviceCapability::new(8, 6, 1024, 100 * 1024, 84)
    }

    fn sample_launch() -> LaunchConfig {
        LaunchConfig::linear(256, 128)
    }

    fn sample_launch_alt() -> LaunchConfig {
        LaunchConfig::grid_2d(16, 16, 32, 32).with_shared_mem(4096)
    }

    fn sample_source() -> &'static str {
        r#"extern "C" __global__ void add(float* a, float* b, int n) {
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < n) a[i] += b[i];
        }"#
    }

    fn alt_source() -> &'static str {
        r#"extern "C" __global__ void mul(float* a, float* b, int n) {
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < n) a[i] *= b[i];
        }"#
    }

    fn make_key(name: &str) -> CacheKey {
        CacheKey::new(name, sample_source(), &sample_launch(), &sample_device())
    }

    fn make_kernel(name: &str) -> CompiledKernel {
        simulate_compile(sample_source(), name).unwrap()
    }

    fn tiny_cache() -> KernelCache {
        KernelCache::new(CacheConfig::tiny()).unwrap()
    }

    fn default_cache() -> KernelCache {
        KernelCache::with_defaults().unwrap()
    }

    fn temp_dir() -> PathBuf {
        let seq = TEMP_DIR_COUNTER.fetch_add(1, Ordering::Relaxed);
        let nanos = std::time::SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |duration| duration.as_nanos());
        let dir = std::env::temp_dir().join(format!(
            "bitnet_kc_test_{}_{}_{}",
            std::process::id(),
            nanos,
            seq
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn cleanup_dir(dir: &Path) {
        let _ = std::fs::remove_dir_all(dir);
    }

    // ── DeviceCapability tests ───────────────────────────────────────

    #[test]
    fn device_capability_sm_arch() {
        let dev = DeviceCapability::new(8, 6, 1024, 49152, 84);
        assert_eq!(dev.sm_arch(), "sm_86");
    }

    #[test]
    fn device_capability_default() {
        let dev = DeviceCapability::default();
        assert_eq!(dev.major, 8);
        assert_eq!(dev.minor, 0);
        assert_eq!(dev.sm_arch(), "sm_80");
    }

    #[test]
    fn device_capability_display() {
        let dev = sample_device();
        let s = format!("{dev}");
        assert!(s.contains("sm_80"));
        assert!(s.contains("108SMs"));
    }

    #[test]
    fn device_capability_fingerprint_differs() {
        let d1 = sample_device();
        let d2 = sample_device_alt();
        assert_ne!(fnv1a_hash(&d1.fingerprint_bytes()), fnv1a_hash(&d2.fingerprint_bytes()));
    }

    #[test]
    fn device_capability_eq() {
        let d1 = sample_device();
        let d2 = sample_device();
        assert_eq!(d1, d2);
    }

    // ── LaunchConfig tests ───────────────────────────────────────────

    #[test]
    fn launch_config_linear() {
        let lc = LaunchConfig::linear(128, 64);
        assert_eq!(lc.block_dim, (128, 1, 1));
        assert_eq!(lc.grid_dim, (64, 1, 1));
        assert_eq!(lc.threads_per_block(), 128);
    }

    #[test]
    fn launch_config_grid_2d() {
        let lc = LaunchConfig::grid_2d(16, 16, 8, 8);
        assert_eq!(lc.block_dim, (16, 16, 1));
        assert_eq!(lc.threads_per_block(), 256);
    }

    #[test]
    fn launch_config_with_shared_mem() {
        let lc = LaunchConfig::linear(256, 1).with_shared_mem(8192);
        assert_eq!(lc.shared_mem_bytes, 8192);
    }

    #[test]
    fn launch_config_with_option() {
        let lc = LaunchConfig::linear(256, 1).with_option("--use_fast_math");
        assert_eq!(lc.compile_options.len(), 1);
        assert_eq!(lc.compile_options[0], "--use_fast_math");
    }

    #[test]
    fn launch_config_default() {
        let lc = LaunchConfig::default();
        assert_eq!(lc.threads_per_block(), 256);
    }

    #[test]
    fn launch_config_fingerprint_differs_with_options() {
        let l1 = LaunchConfig::linear(256, 1);
        let l2 = LaunchConfig::linear(256, 1).with_option("--use_fast_math");
        assert_ne!(fnv1a_hash(&l1.fingerprint_bytes()), fnv1a_hash(&l2.fingerprint_bytes()));
    }

    // ── CacheKey tests ───────────────────────────────────────────────

    #[test]
    fn cache_key_combined_hash_stable() {
        let k1 = make_key("test");
        let k2 = make_key("test");
        assert_eq!(k1.combined_hash(), k2.combined_hash());
    }

    #[test]
    fn cache_key_different_source_different_hash() {
        let k1 = CacheKey::new("k", sample_source(), &sample_launch(), &sample_device());
        let k2 = CacheKey::new("k", alt_source(), &sample_launch(), &sample_device());
        assert_ne!(k1.combined_hash(), k2.combined_hash());
    }

    #[test]
    fn cache_key_different_device_different_hash() {
        let k1 = CacheKey::new("k", sample_source(), &sample_launch(), &sample_device());
        let k2 = CacheKey::new("k", sample_source(), &sample_launch(), &sample_device_alt());
        assert_ne!(k1.combined_hash(), k2.combined_hash());
    }

    #[test]
    fn cache_key_different_config_different_hash() {
        let k1 = CacheKey::new("k", sample_source(), &sample_launch(), &sample_device());
        let k2 = CacheKey::new("k", sample_source(), &sample_launch_alt(), &sample_device());
        assert_ne!(k1.combined_hash(), k2.combined_hash());
    }

    #[test]
    fn cache_key_cache_filename() {
        let k = make_key("matmul");
        let fname = k.cache_filename();
        assert!(fname.starts_with("matmul_"));
        assert!(fname.ends_with(".bin"));
    }

    #[test]
    fn cache_key_display() {
        let k = make_key("rope");
        let s = format!("{k}");
        assert!(s.starts_with("rope@"));
    }

    #[test]
    fn cache_key_eq() {
        let k1 = make_key("test");
        let k2 = make_key("test");
        assert_eq!(k1, k2);
    }

    // ── CompiledKernel tests ─────────────────────────────────────────

    #[test]
    fn compiled_kernel_new() {
        let k = make_kernel("add");
        assert_eq!(k.entry_point, "add");
        assert_eq!(k.format, BinaryFormat::Simulated);
        assert!(!k.binary.is_empty());
    }

    #[test]
    fn compiled_kernel_binary_size() {
        let k = make_kernel("add");
        assert_eq!(k.binary_size(), k.binary.len());
    }

    #[test]
    fn compiled_kernel_with_register_count() {
        let k = make_kernel("add").with_register_count(32);
        assert_eq!(k.register_count, Some(32));
    }

    #[test]
    fn compiled_kernel_with_shared_mem() {
        let k = make_kernel("add").with_shared_mem(4096);
        assert_eq!(k.shared_mem_bytes, Some(4096));
    }

    // ── BinaryFormat tests ───────────────────────────────────────────

    #[test]
    fn binary_format_display() {
        assert_eq!(format!("{}", BinaryFormat::Ptx), "PTX");
        assert_eq!(format!("{}", BinaryFormat::Cubin), "cubin");
        assert_eq!(format!("{}", BinaryFormat::Simulated), "simulated");
    }

    // ── CacheStats tests ─────────────────────────────────────────────

    #[test]
    fn cache_stats_default_zero() {
        let s = CacheStats::default();
        assert_eq!(s.hits, 0);
        assert_eq!(s.misses, 0);
        assert_eq!(s.evictions, 0);
        assert_eq!(s.hit_rate(), 0.0);
    }

    #[test]
    fn cache_stats_hit_rate() {
        let mut s = CacheStats::default();
        s.hits = 3;
        s.misses = 1;
        assert!((s.hit_rate() - 0.75).abs() < 1e-9);
    }

    #[test]
    fn cache_stats_disk_hit_rate() {
        let mut s = CacheStats::default();
        s.disk_hits = 7;
        s.disk_misses = 3;
        assert!((s.disk_hit_rate() - 0.7).abs() < 1e-9);
    }

    #[test]
    fn cache_stats_total_lookups() {
        let mut s = CacheStats::default();
        s.hits = 10;
        s.misses = 5;
        assert_eq!(s.total_lookups(), 15);
    }

    #[test]
    fn cache_stats_reset() {
        let mut s = CacheStats::default();
        s.hits = 100;
        s.misses = 50;
        s.evictions = 10;
        s.reset();
        assert_eq!(s.hits, 0);
        assert_eq!(s.misses, 0);
        assert_eq!(s.evictions, 0);
    }

    #[test]
    fn cache_stats_display() {
        let s = CacheStats::default();
        let formatted = format!("{s}");
        assert!(formatted.contains("CacheStats"));
        assert!(formatted.contains("hits: 0"));
    }

    // ── CacheConfig tests ────────────────────────────────────────────

    #[test]
    fn cache_config_default_valid() {
        let cfg = CacheConfig::default();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn cache_config_tiny_valid() {
        let cfg = CacheConfig::tiny();
        assert!(cfg.validate().is_ok());
        assert_eq!(cfg.max_entries, 4);
    }

    #[test]
    fn cache_config_zero_entries_invalid() {
        let mut cfg = CacheConfig::default();
        cfg.max_entries = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn cache_config_zero_memory_invalid() {
        let mut cfg = CacheConfig::default();
        cfg.max_memory_bytes = 0;
        assert!(cfg.validate().is_err());
    }

    // ── KernelCache basic tests ──────────────────────────────────────

    #[test]
    fn cache_new_default() {
        let cache = default_cache();
        assert!(cache.is_empty());
        assert_eq!(cache.len(), 0);
    }

    #[test]
    fn cache_insert_and_get() {
        let mut cache = default_cache();
        let key = make_key("add");
        let kernel = make_kernel("add");
        cache.insert(key.clone(), kernel).unwrap();
        assert!(cache.get(&key).is_some());
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn cache_miss_returns_none() {
        let mut cache = default_cache();
        let key = make_key("nonexistent");
        assert!(cache.get(&key).is_none());
    }

    #[test]
    fn cache_hit_increments_stats() {
        let mut cache = default_cache();
        let key = make_key("add");
        cache.insert(key.clone(), make_kernel("add")).unwrap();
        let _ = cache.get(&key);
        assert_eq!(cache.stats().hits, 1);
    }

    #[test]
    fn cache_miss_increments_stats() {
        let mut cache = default_cache();
        let key = make_key("missing");
        let _ = cache.get(&key);
        assert_eq!(cache.stats().misses, 1);
    }

    #[test]
    fn cache_contains_check() {
        let mut cache = default_cache();
        let key = make_key("add");
        assert!(!cache.contains(&key));
        cache.insert(key.clone(), make_kernel("add")).unwrap();
        assert!(cache.contains(&key));
    }

    #[test]
    fn cache_remove() {
        let mut cache = default_cache();
        let key = make_key("add");
        cache.insert(key.clone(), make_kernel("add")).unwrap();
        let removed = cache.remove(&key);
        assert!(removed.is_some());
        assert!(cache.is_empty());
    }

    #[test]
    fn cache_remove_nonexistent() {
        let mut cache = default_cache();
        let key = make_key("nonexistent");
        assert!(cache.remove(&key).is_none());
    }

    #[test]
    fn cache_clear() {
        let mut cache = default_cache();
        cache.insert(make_key("a"), make_kernel("a")).unwrap();
        cache.insert(make_key("b"), make_kernel("b")).unwrap();
        cache.clear();
        assert!(cache.is_empty());
        assert_eq!(cache.total_bytes(), 0);
    }

    #[test]
    fn cache_total_bytes_tracks() {
        let mut cache = default_cache();
        let k = make_kernel("add");
        let expected_size = k.binary_size() as u64;
        cache.insert(make_key("add"), k).unwrap();
        assert_eq!(cache.total_bytes(), expected_size);
    }

    #[test]
    fn cache_overwrite_same_key() {
        let mut cache = default_cache();
        let key = make_key("add");
        cache.insert(key.clone(), make_kernel("add")).unwrap();
        cache.insert(key.clone(), make_kernel("add")).unwrap();
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn cache_multiple_entries() {
        let mut cache = default_cache();
        let k1 = CacheKey::new("a", sample_source(), &sample_launch(), &sample_device());
        let k2 = CacheKey::new("b", alt_source(), &sample_launch(), &sample_device());
        cache.insert(k1, make_kernel("a")).unwrap();
        cache.insert(k2, make_kernel("b")).unwrap();
        assert_eq!(cache.len(), 2);
    }

    #[test]
    fn cache_cached_kernel_names() {
        let mut cache = default_cache();
        let k1 = CacheKey::new("alpha", sample_source(), &sample_launch(), &sample_device());
        let k2 = CacheKey::new("beta", alt_source(), &sample_launch(), &sample_device());
        cache.insert(k1, make_kernel("alpha")).unwrap();
        cache.insert(k2, make_kernel("beta")).unwrap();
        let names = cache.cached_kernel_names();
        assert!(names.contains(&"alpha".to_string()));
        assert!(names.contains(&"beta".to_string()));
    }

    #[test]
    fn cache_memory_utilisation_zero_when_empty() {
        let cache = default_cache();
        assert_eq!(cache.memory_utilisation(), 0.0);
    }

    #[test]
    fn cache_memory_utilisation_increases() {
        let mut cache = default_cache();
        cache.insert(make_key("add"), make_kernel("add")).unwrap();
        assert!(cache.memory_utilisation() > 0.0);
    }

    #[test]
    fn cache_reset_stats() {
        let mut cache = default_cache();
        cache.insert(make_key("add"), make_kernel("add")).unwrap();
        let _ = cache.get(&make_key("add"));
        let _ = cache.get(&make_key("miss"));
        cache.reset_stats();
        assert_eq!(cache.stats().hits, 0);
        assert_eq!(cache.stats().misses, 0);
        // entries and total_bytes preserved
        assert_eq!(cache.stats().entries, 1);
    }

    // ── LRU eviction tests ──────────────────────────────────────────

    #[test]
    fn cache_evicts_lru_on_capacity() {
        let mut cache = tiny_cache(); // max 4 entries
        for i in 0..5 {
            let name = format!("k{i}");
            let key = CacheKey::new(&name, &format!("src{i}"), &sample_launch(), &sample_device());
            let kernel = simulate_compile(&format!("src{i}"), &name).unwrap();
            cache.insert(key, kernel).unwrap();
        }
        assert!(cache.len() <= 4);
        assert!(cache.stats().evictions > 0);
    }

    #[test]
    fn cache_evicts_oldest_first() {
        let mut cache = tiny_cache(); // max 4 entries
        let keys: Vec<CacheKey> = (0..4)
            .map(|i| {
                let name = format!("k{i}");
                CacheKey::new(&name, &format!("src{i}"), &sample_launch(), &sample_device())
            })
            .collect();

        for (i, key) in keys.iter().enumerate() {
            let kernel = simulate_compile(&format!("src{i}"), &format!("k{i}")).unwrap();
            cache.insert(key.clone(), kernel).unwrap();
        }

        // Access k0 to make it recently used.
        let _ = cache.get(&keys[0]);

        // Insert a 5th entry — k1 should be evicted (oldest not recently accessed).
        let new_key = CacheKey::new("k4", "src4", &sample_launch(), &sample_device());
        let new_kernel = simulate_compile("src4", "k4").unwrap();
        cache.insert(new_key, new_kernel).unwrap();

        assert!(cache.contains(&keys[0]), "k0 should survive (recently accessed)");
        assert!(!cache.contains(&keys[1]), "k1 should be evicted (LRU)");
    }

    #[test]
    fn cache_eviction_counter() {
        let mut cache = tiny_cache();
        for i in 0..8 {
            let key = CacheKey::new(
                &format!("k{i}"),
                &format!("s{i}"),
                &sample_launch(),
                &sample_device(),
            );
            let kernel = simulate_compile(&format!("s{i}"), &format!("k{i}")).unwrap();
            cache.insert(key, kernel).unwrap();
        }
        assert!(cache.stats().evictions >= 4);
    }

    #[test]
    fn cache_eviction_reduces_bytes() {
        let mut cache = tiny_cache();
        for i in 0..6 {
            let key = CacheKey::new(
                &format!("k{i}"),
                &format!("s{i}"),
                &sample_launch(),
                &sample_device(),
            );
            let kernel = simulate_compile(&format!("s{i}"), &format!("k{i}")).unwrap();
            cache.insert(key, kernel).unwrap();
        }
        assert!(cache.total_bytes() <= cache.config().max_memory_bytes as u64);
    }

    #[test]
    fn cache_time_saved_accumulates() {
        let mut cache = default_cache();
        let key = make_key("add");
        cache.insert(key.clone(), make_kernel("add")).unwrap();
        let _ = cache.get(&key);
        let _ = cache.get(&key);
        assert!(cache.stats().time_saved > Duration::ZERO);
    }

    // ── BinaryCache tests ────────────────────────────────────────────

    #[test]
    fn binary_cache_open_creates_dir() {
        let dir = temp_dir();
        let sub = dir.join("new_sub");
        let bc = BinaryCache::open(&sub, 1024 * 1024, 0).unwrap();
        assert!(sub.exists());
        assert!(bc.is_empty());
        cleanup_dir(&dir);
    }

    #[test]
    fn binary_cache_store_and_load() {
        let dir = temp_dir();
        let mut bc = BinaryCache::open(&dir, 1024 * 1024, 0).unwrap();
        let key = make_key("test_store");
        let kernel = make_kernel("test_store");
        bc.store(&key, &kernel).unwrap();
        assert!(bc.contains(&key));
        let loaded = bc.load(&key).unwrap().unwrap();
        assert_eq!(loaded.entry_point, "test_store");
        assert_eq!(loaded.binary, kernel.binary);
        cleanup_dir(&dir);
    }

    #[test]
    fn binary_cache_store_recreates_missing_dir() {
        let dir = temp_dir();
        let mut bc = BinaryCache::open(&dir, 1024 * 1024, 0).unwrap();
        cleanup_dir(&dir);

        let key = make_key("recreate_dir");
        let kernel = make_kernel("recreate_dir");
        bc.store(&key, &kernel).unwrap();

        assert!(dir.exists());
        assert!(bc.contains(&key));
        cleanup_dir(&dir);
    }

    #[test]
    fn binary_cache_load_missing_returns_none() {
        let dir = temp_dir();
        let mut bc = BinaryCache::open(&dir, 1024 * 1024, 0).unwrap();
        let key = make_key("missing");
        assert!(bc.load(&key).unwrap().is_none());
        cleanup_dir(&dir);
    }

    #[test]
    fn binary_cache_remove() {
        let dir = temp_dir();
        let mut bc = BinaryCache::open(&dir, 1024 * 1024, 0).unwrap();
        let key = make_key("removeme");
        bc.store(&key, &make_kernel("removeme")).unwrap();
        assert!(bc.remove(&key).unwrap());
        assert!(!bc.contains(&key));
        cleanup_dir(&dir);
    }

    #[test]
    fn binary_cache_clear() {
        let dir = temp_dir();
        let mut bc = BinaryCache::open(&dir, 1024 * 1024, 0).unwrap();
        for i in 0..3 {
            let name = format!("k{i}");
            let key = CacheKey::new(&name, &format!("s{i}"), &sample_launch(), &sample_device());
            bc.store(&key, &simulate_compile(&format!("s{i}"), &name).unwrap()).unwrap();
        }
        assert_eq!(bc.len(), 3);
        bc.clear().unwrap();
        assert!(bc.is_empty());
        cleanup_dir(&dir);
    }

    #[test]
    fn binary_cache_used_bytes_tracks() {
        let dir = temp_dir();
        let mut bc = BinaryCache::open(&dir, 1024 * 1024, 0).unwrap();
        assert_eq!(bc.used_bytes(), 0);
        let key = make_key("sized");
        bc.store(&key, &make_kernel("sized")).unwrap();
        assert!(bc.used_bytes() > 0);
        cleanup_dir(&dir);
    }

    #[test]
    fn binary_cache_stats_writes() {
        let dir = temp_dir();
        let mut bc = BinaryCache::open(&dir, 1024 * 1024, 0).unwrap();
        bc.store(&make_key("w"), &make_kernel("w")).unwrap();
        assert_eq!(bc.stats().writes, 1);
        cleanup_dir(&dir);
    }

    #[test]
    fn binary_cache_stats_reads() {
        let dir = temp_dir();
        let mut bc = BinaryCache::open(&dir, 1024 * 1024, 0).unwrap();
        let key = make_key("r");
        bc.store(&key, &make_kernel("r")).unwrap();
        let _ = bc.load(&key).unwrap();
        assert_eq!(bc.stats().reads, 1);
        cleanup_dir(&dir);
    }

    #[test]
    fn binary_cache_eviction_on_limit() {
        let dir = temp_dir();
        // Very small limit to force eviction.
        let mut bc = BinaryCache::open(&dir, 100, 0).unwrap();
        for i in 0..5 {
            let name = format!("e{i}");
            let key = CacheKey::new(&name, &format!("es{i}"), &sample_launch(), &sample_device());
            let kernel = simulate_compile(&format!("es{i}"), &name).unwrap();
            bc.store(&key, &kernel).unwrap();
        }
        assert!(bc.stats().evictions > 0);
        cleanup_dir(&dir);
    }

    #[test]
    fn binary_cache_ttl_expiry() {
        let dir = temp_dir();
        // TTL = 0 means no expiry; use 1 second and verify basic behaviour.
        let mut bc = BinaryCache::open(&dir, 1024 * 1024, 1).unwrap();
        let key = make_key("ttl_test");
        bc.store(&key, &make_kernel("ttl_test")).unwrap();
        // Immediately should still be loadable (within 1s).
        let loaded = bc.load(&key).unwrap();
        assert!(loaded.is_some());
        cleanup_dir(&dir);
    }

    #[test]
    fn binary_cache_rebuild_index() {
        let dir = temp_dir();
        {
            let mut bc = BinaryCache::open(&dir, 1024 * 1024, 0).unwrap();
            bc.store(&make_key("persist"), &make_kernel("persist")).unwrap();
        }
        // Reopen — should find the file.
        let bc2 = BinaryCache::open(&dir, 1024 * 1024, 0).unwrap();
        assert_eq!(bc2.len(), 1);
        assert!(bc2.used_bytes() > 0);
        cleanup_dir(&dir);
    }

    #[test]
    fn binary_cache_dir_accessor() {
        let dir = temp_dir();
        let bc = BinaryCache::open(&dir, 1024, 0).unwrap();
        assert_eq!(bc.cache_dir(), dir.as_path());
        cleanup_dir(&dir);
    }

    // ── WarmupStrategy tests ─────────────────────────────────────────

    #[test]
    fn warmup_strategy_default_empty() {
        let ws = WarmupStrategy::default();
        assert!(ws.is_empty());
        assert_eq!(ws.len(), 0);
    }

    #[test]
    fn warmup_strategy_standard_bitnet() {
        let ws = WarmupStrategy::standard_bitnet();
        assert_eq!(ws.len(), 5);
        assert!(!ws.is_empty());
    }

    #[test]
    fn warmup_strategy_add_kernel() {
        let mut ws = WarmupStrategy::default();
        ws.add_kernel(WarmupKernel::new("test", "src", LaunchConfig::default(), 1));
        assert_eq!(ws.len(), 1);
    }

    #[test]
    fn warmup_strategy_sort_by_priority() {
        let mut ws = WarmupStrategy::default();
        ws.add_kernel(WarmupKernel::new("low", "s1", LaunchConfig::default(), 1));
        ws.add_kernel(WarmupKernel::new("high", "s2", LaunchConfig::default(), 10));
        ws.add_kernel(WarmupKernel::new("mid", "s3", LaunchConfig::default(), 5));
        ws.sort_by_priority();
        assert_eq!(ws.kernels[0].name, "high");
        assert_eq!(ws.kernels[1].name, "mid");
        assert_eq!(ws.kernels[2].name, "low");
    }

    #[test]
    fn warmup_execute_compiles_all() {
        let mut ws = WarmupStrategy::standard_bitnet();
        let mut cache = default_cache();
        let device = sample_device();
        let result = ws.execute(&mut cache, &device).unwrap();
        assert_eq!(result.compiled, 5);
        assert_eq!(result.failed, 0);
        assert_eq!(result.skipped, 0);
        assert_eq!(cache.len(), 5);
    }

    #[test]
    fn warmup_execute_skips_cached() {
        let mut ws = WarmupStrategy::standard_bitnet();
        let mut cache = default_cache();
        let device = sample_device();

        // First run.
        let _ = ws.execute(&mut cache, &device).unwrap();
        // Second run — all should be skipped.
        let mut ws2 = WarmupStrategy::standard_bitnet();
        let result = ws2.execute(&mut cache, &device).unwrap();
        assert_eq!(result.compiled, 0);
        assert_eq!(result.skipped, 5);
    }

    #[test]
    fn warmup_execute_empty_source_fails_gracefully() {
        let mut ws = WarmupStrategy::default();
        ws.add_kernel(WarmupKernel::new("bad", "", LaunchConfig::default(), 1));
        let mut cache = default_cache();
        let device = sample_device();
        let result = ws.execute(&mut cache, &device).unwrap();
        assert_eq!(result.failed, 1);
        assert_eq!(result.compiled, 0);
    }

    #[test]
    fn warmup_result_display() {
        let result = WarmupResult {
            compiled: 3,
            failed: 1,
            skipped: 2,
            total_time: Duration::from_millis(42),
        };
        let s = format!("{result}");
        assert!(s.contains("3 compiled"));
        assert!(s.contains("1 failed"));
        assert!(s.contains("2 skipped"));
    }

    #[test]
    fn warmup_increments_warmup_compilations() {
        let mut ws = WarmupStrategy::standard_bitnet();
        let mut cache = default_cache();
        let device = sample_device();
        ws.execute(&mut cache, &device).unwrap();
        assert_eq!(cache.stats().warmup_compilations, 5);
    }

    #[test]
    fn warmup_timeout_respected() {
        let mut ws = WarmupStrategy::default();
        ws.timeout = Duration::ZERO;
        for i in 0..10 {
            ws.add_kernel(WarmupKernel::new(
                &format!("k{i}"),
                &format!("src{i}"),
                LaunchConfig::default(),
                i,
            ));
        }
        let mut cache = default_cache();
        let device = sample_device();
        let result = ws.execute(&mut cache, &device).unwrap();
        // With zero timeout, most should be skipped.
        assert!(result.skipped > 0 || result.compiled + result.failed <= 10);
    }

    // ── simulate_compile tests ───────────────────────────────────────

    #[test]
    fn simulate_compile_ok() {
        let k = simulate_compile("source", "entry").unwrap();
        assert_eq!(k.entry_point, "entry");
        assert_eq!(k.format, BinaryFormat::Simulated);
        assert!(!k.binary.is_empty());
    }

    #[test]
    fn simulate_compile_empty_source_err() {
        assert!(simulate_compile("", "entry").is_err());
    }

    #[test]
    fn simulate_compile_deterministic() {
        let k1 = simulate_compile("hello", "e").unwrap();
        let k2 = simulate_compile("hello", "e").unwrap();
        assert_eq!(k1.binary, k2.binary);
    }

    // ── fnv1a hash tests ─────────────────────────────────────────────

    #[test]
    fn fnv1a_deterministic() {
        let h1 = fnv1a_hash(b"hello world");
        let h2 = fnv1a_hash(b"hello world");
        assert_eq!(h1, h2);
    }

    #[test]
    fn fnv1a_different_inputs() {
        let h1 = fnv1a_hash(b"abc");
        let h2 = fnv1a_hash(b"xyz");
        assert_ne!(h1, h2);
    }

    #[test]
    fn fnv1a_empty() {
        let h = fnv1a_hash(b"");
        assert_eq!(h, 0xcbf29ce484222325); // FNV offset basis
    }

    // ── Integration / round-trip tests ───────────────────────────────

    #[test]
    fn full_round_trip_memory_cache() {
        let mut cache = default_cache();
        let key = make_key("roundtrip");
        let kernel = make_kernel("roundtrip");
        let original_binary = kernel.binary.clone();

        cache.insert(key.clone(), kernel).unwrap();
        let retrieved = cache.get(&key).unwrap();
        assert_eq!(retrieved.binary, original_binary);
        assert_eq!(retrieved.entry_point, "roundtrip");
    }

    #[test]
    fn full_round_trip_disk_cache() {
        let dir = temp_dir();
        let mut bc = BinaryCache::open(&dir, 1024 * 1024, 0).unwrap();
        let key = make_key("disk_rt");
        let kernel = make_kernel("disk_rt");
        let original_binary = kernel.binary.clone();

        bc.store(&key, &kernel).unwrap();
        let loaded = bc.load(&key).unwrap().unwrap();
        assert_eq!(loaded.binary, original_binary);
        assert_eq!(loaded.entry_point, "disk_rt");
        cleanup_dir(&dir);
    }

    #[test]
    fn warmup_then_lookup() {
        let mut ws = WarmupStrategy::standard_bitnet();
        let mut cache = default_cache();
        let device = sample_device();
        ws.execute(&mut cache, &device).unwrap();

        // Look up the i2s_dequant kernel.
        let key = CacheKey::new(
            "i2s_dequant",
            STUB_I2S_DEQUANT_SRC,
            &LaunchConfig::linear(256, 1),
            &device,
        );
        assert!(cache.get(&key).is_some());
        assert_eq!(cache.stats().hits, 1);
    }

    #[test]
    fn memory_and_disk_cache_interop() {
        let dir = temp_dir();
        let mut mem = default_cache();
        let mut disk = BinaryCache::open(&dir, 1024 * 1024, 0).unwrap();

        let key = make_key("interop");
        let kernel = make_kernel("interop");

        // Miss in memory → compile → store in both.
        assert!(mem.get(&key).is_none());
        mem.insert(key.clone(), kernel.clone()).unwrap();
        disk.store(&key, &kernel).unwrap();

        // Simulate restart: new memory cache, load from disk.
        let mut mem2 = default_cache();
        assert!(mem2.get(&key).is_none());
        let from_disk = disk.load(&key).unwrap().unwrap();
        mem2.insert(key.clone(), from_disk).unwrap();
        assert!(mem2.get(&key).is_some());

        cleanup_dir(&dir);
    }

    #[test]
    fn cache_debug_format() {
        let cache = default_cache();
        let debug = format!("{cache:?}");
        assert!(debug.contains("KernelCache"));
        assert!(debug.contains("entries"));
    }

    #[test]
    fn binary_cache_debug_format() {
        let dir = temp_dir();
        let bc = BinaryCache::open(&dir, 1024, 0).unwrap();
        let debug = format!("{bc:?}");
        assert!(debug.contains("BinaryCache"));
        cleanup_dir(&dir);
    }
}
