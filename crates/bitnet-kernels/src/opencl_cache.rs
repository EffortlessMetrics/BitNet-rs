//! OpenCL kernel binary cache for fast startup.
//!
//! Caches compiled OpenCL kernel binaries in memory and/or on disk so that
//! repeated launches skip the expensive online compilation step. A composite
//! cache key — source hash, device name, build options, and OpenCL version —
//! ensures binaries are only reused when the compilation environment matches.

use std::collections::{HashMap, VecDeque};
use std::fmt;
use std::hash::{Hash, Hasher};
use std::path::PathBuf;
use std::sync::RwLock;
use std::time::{Duration, SystemTime};

// ---------------------------------------------------------------------------
// Source hasher — deterministic SHA-256-style hash of kernel source
// ---------------------------------------------------------------------------

/// Computes a deterministic 64-bit hash of kernel source code for cache keys.
///
/// Uses the standard `DefaultHasher` (SipHash) for speed; collisions are
/// acceptable because the full key also includes device name and build options.
#[derive(Debug, Clone)]
pub struct SourceHasher;

impl SourceHasher {
    /// Hash kernel source text into a deterministic `u64`.
    pub fn hash_source(source: &str) -> u64 {
        use std::hash::DefaultHasher;
        let mut hasher = DefaultHasher::new();
        source.hash(&mut hasher);
        hasher.finish()
    }

    /// Hash raw bytes (e.g. pre-processed source).
    pub fn hash_bytes(data: &[u8]) -> u64 {
        use std::hash::DefaultHasher;
        let mut hasher = DefaultHasher::new();
        data.hash(&mut hasher);
        hasher.finish()
    }
}

// ---------------------------------------------------------------------------
// KernelCacheKey
// ---------------------------------------------------------------------------

/// Composite key that uniquely identifies a compiled kernel binary.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct KernelCacheKey {
    /// Hash of the kernel source code.
    pub source_hash: u64,
    /// OpenCL device name string (e.g. "Intel(R) Arc(TM) A770").
    pub device_name: String,
    /// Build options passed to `clBuildProgram` (e.g. "-cl-mad-enable").
    pub build_options: String,
    /// OpenCL version string (e.g. "OpenCL 3.0").
    pub opencl_version: String,
}

impl KernelCacheKey {
    pub fn new(
        source_hash: u64,
        device_name: impl Into<String>,
        build_options: impl Into<String>,
        opencl_version: impl Into<String>,
    ) -> Self {
        Self {
            source_hash,
            device_name: device_name.into(),
            build_options: build_options.into(),
            opencl_version: opencl_version.into(),
        }
    }

    /// Build a key from raw source text plus device metadata.
    pub fn from_source(
        source: &str,
        device_name: impl Into<String>,
        build_options: impl Into<String>,
        opencl_version: impl Into<String>,
    ) -> Self {
        Self::new(SourceHasher::hash_source(source), device_name, build_options, opencl_version)
    }

    /// Deterministic filename suitable for disk caching.
    pub fn cache_filename(&self) -> String {
        format!("{:016x}.bin", self.source_hash)
    }
}

impl fmt::Display for KernelCacheKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "CacheKey(hash={:016x}, device={}, opts={}, ocl={})",
            self.source_hash, self.device_name, self.build_options, self.opencl_version,
        )
    }
}

// ---------------------------------------------------------------------------
// KernelCacheEntry
// ---------------------------------------------------------------------------

/// A cached kernel binary together with metadata.
#[derive(Debug, Clone)]
pub struct KernelCacheEntry {
    /// The compiled binary bytes.
    pub binary: Vec<u8>,
    /// Build log emitted by the OpenCL compiler.
    pub build_log: String,
    /// How long the original compilation took.
    pub compilation_time: Duration,
    /// Device the binary was compiled for.
    pub device_info: String,
    /// When this entry was created.
    pub created_at: SystemTime,
    /// When this entry was last accessed.
    pub last_accessed: SystemTime,
    /// Number of times this entry has been accessed.
    pub access_count: u64,
}

impl KernelCacheEntry {
    pub fn new(
        binary: Vec<u8>,
        build_log: String,
        compilation_time: Duration,
        device_info: String,
    ) -> Self {
        let now = SystemTime::now();
        Self {
            binary,
            build_log,
            compilation_time,
            device_info,
            created_at: now,
            last_accessed: now,
            access_count: 0,
        }
    }

    /// Size of the binary in bytes.
    pub fn binary_size(&self) -> usize {
        self.binary.len()
    }

    /// Returns `true` if this entry has expired according to `ttl`.
    pub fn is_expired(&self, ttl: Duration) -> bool {
        self.created_at.elapsed().map(|age| age > ttl).unwrap_or(false)
    }

    /// Touch the entry (update last-accessed time and bump counter).
    pub fn touch(&mut self) {
        self.last_accessed = SystemTime::now();
        self.access_count += 1;
    }
}

// ---------------------------------------------------------------------------
// CachePolicy / CacheEvictionStrategy
// ---------------------------------------------------------------------------

/// Where cached entries are stored.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CachePolicy {
    /// Caching is disabled.
    NoCache,
    /// Keep entries only in memory.
    MemoryOnly,
    /// Keep entries only on disk.
    DiskOnly,
    /// Keep entries in both memory and on disk.
    MemoryAndDisk,
}

/// Strategy used to evict entries when the cache is full.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheEvictionStrategy {
    /// Least Recently Used.
    Lru,
    /// Least Frequently Used.
    Lfu,
    /// First In, First Out.
    Fifo,
    /// Prefer evicting the largest entry first.
    SizeWeighted,
}

// ---------------------------------------------------------------------------
// CacheConfig
// ---------------------------------------------------------------------------

/// Configuration for the kernel cache.
#[derive(Debug, Clone)]
pub struct CacheConfig {
    /// Maximum number of entries in the cache.
    pub max_entries: usize,
    /// Maximum aggregate size of all cached binaries (bytes).
    pub max_total_size: usize,
    /// Directory for disk-backed entries (when policy includes disk).
    pub cache_dir: PathBuf,
    /// Time-to-live for cache entries. `None` means entries never expire.
    pub ttl: Option<Duration>,
    /// Caching policy.
    pub policy: CachePolicy,
    /// Eviction strategy.
    pub eviction_strategy: CacheEvictionStrategy,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            max_entries: 256,
            max_total_size: 256 * 1024 * 1024, // 256 MiB
            cache_dir: PathBuf::from(".opencl_cache"),
            ttl: None,
            policy: CachePolicy::MemoryOnly,
            eviction_strategy: CacheEvictionStrategy::Lru,
        }
    }
}

impl CacheConfig {
    /// Validate the configuration and return a human-readable error if invalid.
    pub fn validate(&self) -> Result<(), String> {
        if self.max_entries == 0 {
            return Err("max_entries must be > 0".into());
        }
        if self.max_total_size == 0 {
            return Err("max_total_size must be > 0".into());
        }
        if matches!(self.policy, CachePolicy::DiskOnly | CachePolicy::MemoryAndDisk)
            && self.cache_dir.as_os_str().is_empty()
        {
            return Err("cache_dir must be non-empty for disk-backed policies".into());
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// CacheStats
// ---------------------------------------------------------------------------

/// Runtime statistics for the kernel cache.
#[derive(Debug, Clone, Default)]
pub struct CacheStats {
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
    pub stores: u64,
    pub total_binary_size: usize,
    pub entry_count: usize,
}

impl CacheStats {
    /// Hit rate as a fraction in `[0.0, 1.0]`.
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            return 0.0;
        }
        self.hits as f64 / total as f64
    }
}

impl fmt::Display for CacheStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "CacheStats(hits={}, misses={}, evictions={}, entries={}, size={}B, \
             hit_rate={:.1}%)",
            self.hits,
            self.misses,
            self.evictions,
            self.entry_count,
            self.total_binary_size,
            self.hit_rate() * 100.0,
        )
    }
}

// ---------------------------------------------------------------------------
// CacheSerializer — serialize / deserialize entries for disk storage
// ---------------------------------------------------------------------------

/// Simple binary serialization for [`KernelCacheEntry`].
///
/// Wire format (little-endian):
/// ```text
/// [4B magic][4B version]
/// [8B binary_len][binary_bytes…]
/// [8B log_len][log_bytes…]
/// [8B compilation_time_ns]
/// [8B device_info_len][device_info_bytes…]
/// [8B created_at_secs]
/// ```
pub struct CacheSerializer;

impl CacheSerializer {
    const MAGIC: [u8; 4] = *b"BKCL"; // BitNet Kernel CL
    const VERSION: u32 = 1;

    pub fn serialize(entry: &KernelCacheEntry) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(&Self::MAGIC);
        buf.extend_from_slice(&Self::VERSION.to_le_bytes());

        // binary
        buf.extend_from_slice(&(entry.binary.len() as u64).to_le_bytes());
        buf.extend_from_slice(&entry.binary);

        // build_log
        let log_bytes = entry.build_log.as_bytes();
        buf.extend_from_slice(&(log_bytes.len() as u64).to_le_bytes());
        buf.extend_from_slice(log_bytes);

        // compilation_time
        buf.extend_from_slice(&entry.compilation_time.as_nanos().to_le_bytes());

        // device_info
        let info_bytes = entry.device_info.as_bytes();
        buf.extend_from_slice(&(info_bytes.len() as u64).to_le_bytes());
        buf.extend_from_slice(info_bytes);

        // created_at
        let secs =
            entry.created_at.duration_since(SystemTime::UNIX_EPOCH).unwrap_or_default().as_secs();
        buf.extend_from_slice(&secs.to_le_bytes());

        buf
    }

    pub fn deserialize(data: &[u8]) -> Result<KernelCacheEntry, String> {
        let mut pos = 0usize;

        let read = |pos: &mut usize, n: usize| -> Result<&[u8], String> {
            if *pos + n > data.len() {
                return Err("unexpected EOF in cache entry".into());
            }
            let slice = &data[*pos..*pos + n];
            *pos += n;
            Ok(slice)
        };

        let read_u64 = |pos: &mut usize| -> Result<u64, String> {
            let b = read(pos, 8)?;
            Ok(u64::from_le_bytes(b.try_into().unwrap()))
        };

        let read_u128 = |pos: &mut usize| -> Result<u128, String> {
            let b = read(pos, 16)?;
            Ok(u128::from_le_bytes(b.try_into().unwrap()))
        };

        // magic
        let magic = read(&mut pos, 4)?;
        if magic != Self::MAGIC {
            return Err("invalid cache magic".into());
        }

        // version
        let ver_bytes = read(&mut pos, 4)?;
        let version = u32::from_le_bytes(ver_bytes.try_into().unwrap());
        if version != Self::VERSION {
            return Err(format!("unsupported cache version {version}"));
        }

        // binary
        let bin_len = read_u64(&mut pos)? as usize;
        let binary = read(&mut pos, bin_len)?.to_vec();

        // build_log
        let log_len = read_u64(&mut pos)? as usize;
        let log_bytes = read(&mut pos, log_len)?;
        let build_log =
            String::from_utf8(log_bytes.to_vec()).map_err(|e| format!("bad log UTF-8: {e}"))?;

        // compilation_time
        let comp_ns = read_u128(&mut pos)?;
        let compilation_time = Duration::from_nanos(comp_ns as u64);

        // device_info
        let info_len = read_u64(&mut pos)? as usize;
        let info_bytes = read(&mut pos, info_len)?;
        let device_info = String::from_utf8(info_bytes.to_vec())
            .map_err(|e| format!("bad device_info UTF-8: {e}"))?;

        // created_at
        let secs = read_u64(&mut pos)?;
        let created_at = SystemTime::UNIX_EPOCH + Duration::from_secs(secs);

        Ok(KernelCacheEntry {
            binary,
            build_log,
            compilation_time,
            device_info,
            created_at,
            last_accessed: SystemTime::now(),
            access_count: 0,
        })
    }
}

// ---------------------------------------------------------------------------
// KernelCacheManager
// ---------------------------------------------------------------------------

/// High-level thread-safe API for storing and retrieving cached kernel binaries.
pub struct KernelCacheManager {
    config: CacheConfig,
    inner: RwLock<CacheInner>,
}

struct CacheInner {
    entries: HashMap<KernelCacheKey, KernelCacheEntry>,
    /// Insertion order (for FIFO) and access order (for LRU).
    order: VecDeque<KernelCacheKey>,
    stats: CacheStats,
}

impl KernelCacheManager {
    pub fn new(config: CacheConfig) -> Self {
        Self {
            config,
            inner: RwLock::new(CacheInner {
                entries: HashMap::new(),
                order: VecDeque::new(),
                stats: CacheStats::default(),
            }),
        }
    }

    /// Look up a cached entry. Returns a cloned entry on hit.
    pub fn lookup(&self, key: &KernelCacheKey) -> Option<KernelCacheEntry> {
        if self.config.policy == CachePolicy::NoCache {
            let mut inner = self.inner.write().unwrap();
            inner.stats.misses += 1;
            return None;
        }

        // Try memory first.
        let mut inner = self.inner.write().unwrap();
        if let Some(entry) = inner.entries.get_mut(key) {
            // Check TTL.
            if let Some(ttl) = self.config.ttl
                && entry.is_expired(ttl)
            {
                let binary_size = entry.binary_size();
                inner.entries.remove(key);
                inner.order.retain(|k| k != key);
                inner.stats.entry_count = inner.entries.len();
                inner.stats.total_binary_size =
                    inner.stats.total_binary_size.saturating_sub(binary_size);
                inner.stats.misses += 1;
                return None;
            }
            entry.touch();
            let entry = entry.clone();
            // Move to back of order queue for LRU.
            inner.order.retain(|k| k != key);
            inner.order.push_back(key.clone());
            inner.stats.hits += 1;
            return Some(entry);
        }

        // Try loading from disk if configured.
        if matches!(self.config.policy, CachePolicy::DiskOnly | CachePolicy::MemoryAndDisk)
            && let Some(entry) = self.load_from_disk(key)
        {
            // Promote into memory when MemoryAndDisk.
            if self.config.policy == CachePolicy::MemoryAndDisk {
                inner.stats.total_binary_size += entry.binary_size();
                inner.order.push_back(key.clone());
                inner.entries.insert(key.clone(), entry.clone());
                inner.stats.entry_count = inner.entries.len();
            }
            inner.stats.hits += 1;
            return Some(entry);
        }

        inner.stats.misses += 1;
        None
    }

    /// Store a compiled binary in the cache.
    pub fn store(&self, key: KernelCacheKey, binary: Vec<u8>, build_log: String) {
        if self.config.policy == CachePolicy::NoCache {
            return;
        }

        let entry = KernelCacheEntry::new(binary, build_log, Duration::ZERO, String::new());

        let mut inner = self.inner.write().unwrap();

        // Evict if necessary.
        while inner.entries.len() >= self.config.max_entries
            || (inner.stats.total_binary_size + entry.binary_size() > self.config.max_total_size
                && !inner.entries.is_empty())
        {
            if let Some(evict_key) = self.select_eviction_candidate(&inner) {
                if let Some(removed) = inner.entries.remove(&evict_key) {
                    inner.stats.total_binary_size =
                        inner.stats.total_binary_size.saturating_sub(removed.binary_size());
                    inner.order.retain(|k| k != &evict_key);
                    inner.stats.evictions += 1;
                }
            } else {
                break;
            }
        }

        // Store in memory (unless DiskOnly).
        if self.config.policy != CachePolicy::DiskOnly {
            inner.stats.total_binary_size += entry.binary_size();
            inner.order.push_back(key.clone());
            inner.entries.insert(key.clone(), entry.clone());
            inner.stats.entry_count = inner.entries.len();
        }
        inner.stats.stores += 1;

        // Persist to disk when configured.
        if matches!(self.config.policy, CachePolicy::DiskOnly | CachePolicy::MemoryAndDisk) {
            self.save_to_disk(&key, &entry);
        }
    }

    /// Store a compiled binary with full metadata.
    pub fn store_with_metadata(
        &self,
        key: KernelCacheKey,
        binary: Vec<u8>,
        build_log: String,
        compilation_time: Duration,
        device_info: String,
    ) {
        if self.config.policy == CachePolicy::NoCache {
            return;
        }

        let entry = KernelCacheEntry::new(binary, build_log, compilation_time, device_info);

        let mut inner = self.inner.write().unwrap();

        while inner.entries.len() >= self.config.max_entries
            || (inner.stats.total_binary_size + entry.binary_size() > self.config.max_total_size
                && !inner.entries.is_empty())
        {
            if let Some(evict_key) = self.select_eviction_candidate(&inner) {
                if let Some(removed) = inner.entries.remove(&evict_key) {
                    inner.stats.total_binary_size =
                        inner.stats.total_binary_size.saturating_sub(removed.binary_size());
                    inner.order.retain(|k| k != &evict_key);
                    inner.stats.evictions += 1;
                }
            } else {
                break;
            }
        }

        if self.config.policy != CachePolicy::DiskOnly {
            inner.stats.total_binary_size += entry.binary_size();
            inner.order.push_back(key.clone());
            inner.entries.insert(key.clone(), entry.clone());
            inner.stats.entry_count = inner.entries.len();
        }
        inner.stats.stores += 1;

        if matches!(self.config.policy, CachePolicy::DiskOnly | CachePolicy::MemoryAndDisk) {
            self.save_to_disk(&key, &entry);
        }
    }

    /// Remove a specific entry from the cache.
    pub fn invalidate(&self, key: &KernelCacheKey) {
        let mut inner = self.inner.write().unwrap();
        if let Some(removed) = inner.entries.remove(key) {
            inner.stats.total_binary_size =
                inner.stats.total_binary_size.saturating_sub(removed.binary_size());
            inner.order.retain(|k| k != key);
            inner.stats.entry_count = inner.entries.len();
        }
        // Also remove from disk.
        if matches!(self.config.policy, CachePolicy::DiskOnly | CachePolicy::MemoryAndDisk) {
            let path = self.config.cache_dir.join(key.cache_filename());
            let _ = std::fs::remove_file(path);
        }
    }

    /// Remove all entries from the cache.
    pub fn clear(&self) {
        let mut inner = self.inner.write().unwrap();
        inner.entries.clear();
        inner.order.clear();
        inner.stats.total_binary_size = 0;
        inner.stats.entry_count = 0;
    }

    /// Return a snapshot of the current cache statistics.
    pub fn stats(&self) -> CacheStats {
        self.inner.read().unwrap().stats.clone()
    }

    /// Pre-populate the memory cache from disk entries. Returns the number of
    /// entries loaded.
    pub fn warm_up(&self) -> Result<usize, String> {
        if !matches!(self.config.policy, CachePolicy::DiskOnly | CachePolicy::MemoryAndDisk) {
            return Ok(0);
        }

        let dir = &self.config.cache_dir;
        if !dir.exists() {
            return Ok(0);
        }

        let entries = std::fs::read_dir(dir).map_err(|e| format!("cannot read cache dir: {e}"))?;

        let mut loaded = 0usize;
        for entry in entries {
            let entry = entry.map_err(|e| format!("dir entry error: {e}"))?;
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) != Some("bin") {
                continue;
            }
            let data =
                std::fs::read(&path).map_err(|e| format!("cannot read {}: {e}", path.display()))?;
            if let Ok(cache_entry) = CacheSerializer::deserialize(&data) {
                // Reconstruct a minimal key from the filename hash.
                let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or_default();
                let hash = u64::from_str_radix(stem, 16).unwrap_or(0);
                let key = KernelCacheKey::new(hash, cache_entry.device_info.clone(), "", "");

                let mut inner = self.inner.write().unwrap();
                if inner.entries.len() < self.config.max_entries {
                    inner.stats.total_binary_size += cache_entry.binary_size();
                    inner.order.push_back(key.clone());
                    inner.entries.insert(key, cache_entry);
                    inner.stats.entry_count = inner.entries.len();
                    loaded += 1;
                }
            }
        }
        Ok(loaded)
    }

    /// The current number of entries in the memory cache.
    pub fn len(&self) -> usize {
        self.inner.read().unwrap().entries.len()
    }

    /// Whether the memory cache is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Return a reference to the config.
    pub fn config(&self) -> &CacheConfig {
        &self.config
    }

    // -- private helpers ----------------------------------------------------

    fn select_eviction_candidate(&self, inner: &CacheInner) -> Option<KernelCacheKey> {
        match self.config.eviction_strategy {
            CacheEvictionStrategy::Lru | CacheEvictionStrategy::Fifo => {
                inner.order.front().cloned()
            }
            CacheEvictionStrategy::Lfu => {
                inner.entries.iter().min_by_key(|(_, e)| e.access_count).map(|(k, _)| k.clone())
            }
            CacheEvictionStrategy::SizeWeighted => {
                inner.entries.iter().max_by_key(|(_, e)| e.binary_size()).map(|(k, _)| k.clone())
            }
        }
    }

    fn save_to_disk(&self, key: &KernelCacheKey, entry: &KernelCacheEntry) {
        let dir = &self.config.cache_dir;
        let _ = std::fs::create_dir_all(dir);
        let path = dir.join(key.cache_filename());
        let data = CacheSerializer::serialize(entry);
        let _ = std::fs::write(path, data);
    }

    fn load_from_disk(&self, key: &KernelCacheKey) -> Option<KernelCacheEntry> {
        let path = self.config.cache_dir.join(key.cache_filename());
        let data = std::fs::read(path).ok()?;
        CacheSerializer::deserialize(&data).ok()
    }
}

impl fmt::Debug for KernelCacheManager {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let inner = self.inner.read().unwrap();
        f.debug_struct("KernelCacheManager")
            .field("config", &self.config)
            .field("entries", &inner.entries.len())
            .field("stats", &inner.stats)
            .finish()
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    // -- helpers ------------------------------------------------------------

    fn make_key(id: u64) -> KernelCacheKey {
        KernelCacheKey::new(id, "TestDevice", "-O2", "OpenCL 3.0")
    }

    fn make_binary(size: usize) -> Vec<u8> {
        vec![0xAB; size]
    }

    fn default_manager(max_entries: usize) -> KernelCacheManager {
        let config = CacheConfig {
            max_entries,
            max_total_size: 1024 * 1024,
            policy: CachePolicy::MemoryOnly,
            ..Default::default()
        };
        KernelCacheManager::new(config)
    }

    // -----------------------------------------------------------------------
    // KernelCacheKey tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_key_equality_same_fields() {
        let a = make_key(1);
        let b = make_key(1);
        assert_eq!(a, b);
    }

    #[test]
    fn test_key_inequality_different_hash() {
        assert_ne!(make_key(1), make_key(2));
    }

    #[test]
    fn test_key_inequality_different_device() {
        let a = KernelCacheKey::new(1, "DevA", "", "");
        let b = KernelCacheKey::new(1, "DevB", "", "");
        assert_ne!(a, b);
    }

    #[test]
    fn test_key_inequality_different_build_opts() {
        let a = KernelCacheKey::new(1, "Dev", "-O0", "3.0");
        let b = KernelCacheKey::new(1, "Dev", "-O2", "3.0");
        assert_ne!(a, b);
    }

    #[test]
    fn test_key_inequality_different_ocl_version() {
        let a = KernelCacheKey::new(1, "Dev", "", "OpenCL 2.0");
        let b = KernelCacheKey::new(1, "Dev", "", "OpenCL 3.0");
        assert_ne!(a, b);
    }

    #[test]
    fn test_key_hash_consistency() {
        use std::collections::hash_map::DefaultHasher;
        let key = make_key(42);
        let h1 = {
            let mut h = DefaultHasher::new();
            key.hash(&mut h);
            h.finish()
        };
        let h2 = {
            let mut h = DefaultHasher::new();
            key.hash(&mut h);
            h.finish()
        };
        assert_eq!(h1, h2);
    }

    #[test]
    fn test_key_ordering() {
        let a = make_key(1);
        let b = make_key(2);
        assert!(a < b);
    }

    #[test]
    fn test_key_display() {
        let k = make_key(255);
        let s = format!("{k}");
        assert!(s.contains("00000000000000ff"));
        assert!(s.contains("TestDevice"));
    }

    #[test]
    fn test_key_from_source() {
        let k = KernelCacheKey::from_source("__kernel void f() {}", "Dev", "", "3.0");
        assert_ne!(k.source_hash, 0);
    }

    #[test]
    fn test_key_cache_filename() {
        let k = make_key(0xDEAD);
        assert_eq!(k.cache_filename(), "000000000000dead.bin");
    }

    // -----------------------------------------------------------------------
    // SourceHasher tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_source_hasher_deterministic() {
        let h1 = SourceHasher::hash_source("kernel void f(){}");
        let h2 = SourceHasher::hash_source("kernel void f(){}");
        assert_eq!(h1, h2);
    }

    #[test]
    fn test_source_hasher_different_sources() {
        let h1 = SourceHasher::hash_source("kernel void f(){}");
        let h2 = SourceHasher::hash_source("kernel void g(){}");
        assert_ne!(h1, h2);
    }

    #[test]
    fn test_source_hasher_bytes_deterministic() {
        let data = b"hello opencl";
        assert_eq!(SourceHasher::hash_bytes(data), SourceHasher::hash_bytes(data),);
    }

    #[test]
    fn test_source_hasher_bytes_different() {
        assert_ne!(SourceHasher::hash_bytes(b"aaa"), SourceHasher::hash_bytes(b"bbb"),);
    }

    #[test]
    fn test_source_hasher_empty_string() {
        // Must not panic.
        let _ = SourceHasher::hash_source("");
    }

    // -----------------------------------------------------------------------
    // KernelCacheEntry tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_entry_binary_size() {
        let entry =
            KernelCacheEntry::new(vec![0; 128], String::new(), Duration::ZERO, String::new());
        assert_eq!(entry.binary_size(), 128);
    }

    #[test]
    fn test_entry_touch_increments_count() {
        let mut entry = KernelCacheEntry::new(vec![], String::new(), Duration::ZERO, String::new());
        assert_eq!(entry.access_count, 0);
        entry.touch();
        assert_eq!(entry.access_count, 1);
        entry.touch();
        assert_eq!(entry.access_count, 2);
    }

    #[test]
    fn test_entry_not_expired_without_ttl() {
        let entry = KernelCacheEntry::new(vec![], String::new(), Duration::ZERO, String::new());
        // A very generous TTL should not be expired.
        assert!(!entry.is_expired(Duration::from_secs(3600)));
    }

    #[test]
    fn test_entry_expired_with_zero_ttl() {
        let mut entry = KernelCacheEntry::new(vec![], String::new(), Duration::ZERO, String::new());
        // Backdate creation.
        entry.created_at = SystemTime::now() - Duration::from_secs(10);
        assert!(entry.is_expired(Duration::from_secs(1)));
    }

    // -----------------------------------------------------------------------
    // CacheConfig validation
    // -----------------------------------------------------------------------

    #[test]
    fn test_config_default_valid() {
        assert!(CacheConfig::default().validate().is_ok());
    }

    #[test]
    fn test_config_zero_max_entries() {
        let mut cfg = CacheConfig::default();
        cfg.max_entries = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_zero_max_size() {
        let mut cfg = CacheConfig::default();
        cfg.max_total_size = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_disk_empty_dir() {
        let cfg = CacheConfig {
            policy: CachePolicy::DiskOnly,
            cache_dir: PathBuf::from(""),
            ..Default::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_memory_only_empty_dir_ok() {
        let cfg = CacheConfig {
            policy: CachePolicy::MemoryOnly,
            cache_dir: PathBuf::from(""),
            ..Default::default()
        };
        assert!(cfg.validate().is_ok());
    }

    // -----------------------------------------------------------------------
    // CacheStats
    // -----------------------------------------------------------------------

    #[test]
    fn test_stats_hit_rate_no_lookups() {
        let stats = CacheStats::default();
        assert_eq!(stats.hit_rate(), 0.0);
    }

    #[test]
    fn test_stats_hit_rate_all_hits() {
        let stats = CacheStats { hits: 10, misses: 0, ..Default::default() };
        assert!((stats.hit_rate() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_stats_hit_rate_half() {
        let stats = CacheStats { hits: 5, misses: 5, ..Default::default() };
        assert!((stats.hit_rate() - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_stats_display() {
        let stats = CacheStats { hits: 3, misses: 1, ..Default::default() };
        let s = format!("{stats}");
        assert!(s.contains("hits=3"));
        assert!(s.contains("misses=1"));
    }

    // -----------------------------------------------------------------------
    // Store and lookup round-trip
    // -----------------------------------------------------------------------

    #[test]
    fn test_store_and_lookup() {
        let mgr = default_manager(8);
        let key = make_key(1);
        mgr.store(key.clone(), make_binary(64), "ok".into());
        let entry = mgr.lookup(&key).expect("should hit");
        assert_eq!(entry.binary, make_binary(64));
        assert_eq!(entry.build_log, "ok");
    }

    #[test]
    fn test_lookup_miss() {
        let mgr = default_manager(8);
        assert!(mgr.lookup(&make_key(999)).is_none());
    }

    #[test]
    fn test_store_overwrites_existing() {
        let mgr = default_manager(8);
        let key = make_key(1);
        mgr.store(key.clone(), vec![1], "first".into());
        mgr.store(key.clone(), vec![2], "second".into());
        let entry = mgr.lookup(&key).unwrap();
        assert_eq!(entry.binary, vec![2]);
    }

    // -----------------------------------------------------------------------
    // Eviction tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_lru_eviction_removes_oldest() {
        let mgr = default_manager(2);
        mgr.store(make_key(1), make_binary(16), String::new());
        mgr.store(make_key(2), make_binary(16), String::new());
        // Access key 1 to make it recent.
        mgr.lookup(&make_key(1));
        // Inserting key 3 should evict key 2 (least recently used).
        mgr.store(make_key(3), make_binary(16), String::new());
        assert!(mgr.lookup(&make_key(2)).is_none(), "key 2 should be evicted");
        assert!(mgr.lookup(&make_key(1)).is_some(), "key 1 should survive");
    }

    #[test]
    fn test_fifo_eviction() {
        let config = CacheConfig {
            max_entries: 2,
            max_total_size: 1024 * 1024,
            policy: CachePolicy::MemoryOnly,
            eviction_strategy: CacheEvictionStrategy::Fifo,
            ..Default::default()
        };
        let mgr = KernelCacheManager::new(config);
        mgr.store(make_key(1), make_binary(16), String::new());
        mgr.store(make_key(2), make_binary(16), String::new());
        mgr.store(make_key(3), make_binary(16), String::new());
        assert!(mgr.lookup(&make_key(1)).is_none(), "FIFO: key 1 evicted");
        assert!(mgr.lookup(&make_key(3)).is_some());
    }

    #[test]
    fn test_lfu_eviction() {
        let config = CacheConfig {
            max_entries: 2,
            max_total_size: 1024 * 1024,
            policy: CachePolicy::MemoryOnly,
            eviction_strategy: CacheEvictionStrategy::Lfu,
            ..Default::default()
        };
        let mgr = KernelCacheManager::new(config);
        mgr.store(make_key(1), make_binary(16), String::new());
        mgr.store(make_key(2), make_binary(16), String::new());
        // Access key 1 several times.
        for _ in 0..5 {
            mgr.lookup(&make_key(1));
        }
        // Key 2 has fewer accesses → should be evicted.
        mgr.store(make_key(3), make_binary(16), String::new());
        assert!(mgr.lookup(&make_key(1)).is_some(), "LFU: key 1 kept");
    }

    #[test]
    fn test_size_weighted_eviction() {
        let config = CacheConfig {
            max_entries: 2,
            max_total_size: 1024 * 1024,
            policy: CachePolicy::MemoryOnly,
            eviction_strategy: CacheEvictionStrategy::SizeWeighted,
            ..Default::default()
        };
        let mgr = KernelCacheManager::new(config);
        mgr.store(make_key(1), make_binary(1000), String::new()); // large
        mgr.store(make_key(2), make_binary(10), String::new()); // small
        mgr.store(make_key(3), make_binary(10), String::new());
        // Largest entry (key 1) should be evicted first.
        assert!(mgr.lookup(&make_key(1)).is_none(), "SizeWeighted: largest evicted");
        assert!(mgr.lookup(&make_key(2)).is_some());
    }

    #[test]
    fn test_eviction_stats_counted() {
        let mgr = default_manager(1);
        mgr.store(make_key(1), make_binary(8), String::new());
        mgr.store(make_key(2), make_binary(8), String::new());
        assert!(mgr.stats().evictions >= 1);
    }

    #[test]
    fn test_max_total_size_triggers_eviction() {
        let config = CacheConfig {
            max_entries: 100,
            max_total_size: 32,
            policy: CachePolicy::MemoryOnly,
            ..Default::default()
        };
        let mgr = KernelCacheManager::new(config);
        mgr.store(make_key(1), make_binary(20), String::new());
        mgr.store(make_key(2), make_binary(20), String::new());
        // Total would be 40 > 32 so key 1 should be evicted.
        assert!(mgr.lookup(&make_key(1)).is_none());
    }

    // -----------------------------------------------------------------------
    // Policy tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_no_cache_never_stores() {
        let config = CacheConfig { policy: CachePolicy::NoCache, ..Default::default() };
        let mgr = KernelCacheManager::new(config);
        mgr.store(make_key(1), make_binary(8), String::new());
        assert!(mgr.lookup(&make_key(1)).is_none());
        assert_eq!(mgr.len(), 0);
    }

    #[test]
    fn test_memory_only_no_disk_files() {
        let tmp = tempfile::tempdir().unwrap();
        let config = CacheConfig {
            policy: CachePolicy::MemoryOnly,
            cache_dir: tmp.path().to_path_buf(),
            ..Default::default()
        };
        let mgr = KernelCacheManager::new(config);
        mgr.store(make_key(1), make_binary(8), String::new());
        // No files should appear on disk.
        let count = std::fs::read_dir(tmp.path()).unwrap().count();
        assert_eq!(count, 0, "MemoryOnly must not write to disk");
    }

    #[test]
    fn test_disk_only_writes_file() {
        let tmp = tempfile::tempdir().unwrap();
        let config = CacheConfig {
            policy: CachePolicy::DiskOnly,
            cache_dir: tmp.path().to_path_buf(),
            ..Default::default()
        };
        let mgr = KernelCacheManager::new(config);
        mgr.store(make_key(1), make_binary(8), "log".into());
        let files: Vec<_> = std::fs::read_dir(tmp.path()).unwrap().filter_map(|e| e.ok()).collect();
        assert_eq!(files.len(), 1, "DiskOnly should write exactly one file");
        // Memory cache should remain empty for DiskOnly.
        assert_eq!(mgr.len(), 0);
    }

    #[test]
    fn test_disk_lookup_round_trip() {
        let tmp = tempfile::tempdir().unwrap();
        let config = CacheConfig {
            policy: CachePolicy::DiskOnly,
            cache_dir: tmp.path().to_path_buf(),
            ..Default::default()
        };
        let mgr = KernelCacheManager::new(config);
        let key = make_key(42);
        mgr.store(key.clone(), make_binary(16), "build ok".into());
        let entry = mgr.lookup(&key).expect("disk lookup should hit");
        assert_eq!(entry.binary, make_binary(16));
    }

    #[test]
    fn test_memory_and_disk_stores_both() {
        let tmp = tempfile::tempdir().unwrap();
        let config = CacheConfig {
            policy: CachePolicy::MemoryAndDisk,
            cache_dir: tmp.path().to_path_buf(),
            ..Default::default()
        };
        let mgr = KernelCacheManager::new(config);
        mgr.store(make_key(1), make_binary(8), String::new());
        assert_eq!(mgr.len(), 1, "memory should have the entry");
        let disk_count = std::fs::read_dir(tmp.path()).unwrap().count();
        assert_eq!(disk_count, 1, "disk should have the entry");
    }

    // -----------------------------------------------------------------------
    // Invalidate / clear
    // -----------------------------------------------------------------------

    #[test]
    fn test_invalidate_removes_entry() {
        let mgr = default_manager(8);
        let key = make_key(1);
        mgr.store(key.clone(), make_binary(8), String::new());
        mgr.invalidate(&key);
        assert!(mgr.lookup(&key).is_none());
    }

    #[test]
    fn test_invalidate_nonexistent_is_noop() {
        let mgr = default_manager(8);
        mgr.invalidate(&make_key(999)); // must not panic
    }

    #[test]
    fn test_clear_empties_cache() {
        let mgr = default_manager(8);
        for i in 0..5 {
            mgr.store(make_key(i), make_binary(8), String::new());
        }
        assert_eq!(mgr.len(), 5);
        mgr.clear();
        assert_eq!(mgr.len(), 0);
        assert_eq!(mgr.stats().total_binary_size, 0);
    }

    // -----------------------------------------------------------------------
    // Stats tracking
    // -----------------------------------------------------------------------

    #[test]
    fn test_stats_miss_counted() {
        let mgr = default_manager(8);
        mgr.lookup(&make_key(1));
        mgr.lookup(&make_key(2));
        assert_eq!(mgr.stats().misses, 2);
        assert_eq!(mgr.stats().hits, 0);
    }

    #[test]
    fn test_stats_hit_counted() {
        let mgr = default_manager(8);
        let key = make_key(1);
        mgr.store(key.clone(), make_binary(8), String::new());
        mgr.lookup(&key);
        mgr.lookup(&key);
        assert_eq!(mgr.stats().hits, 2);
    }

    #[test]
    fn test_stats_stores_counted() {
        let mgr = default_manager(8);
        mgr.store(make_key(1), make_binary(8), String::new());
        mgr.store(make_key(2), make_binary(8), String::new());
        assert_eq!(mgr.stats().stores, 2);
    }

    #[test]
    fn test_stats_total_binary_size() {
        let mgr = default_manager(8);
        mgr.store(make_key(1), make_binary(100), String::new());
        mgr.store(make_key(2), make_binary(200), String::new());
        assert_eq!(mgr.stats().total_binary_size, 300);
    }

    // -----------------------------------------------------------------------
    // CacheSerializer round-trip
    // -----------------------------------------------------------------------

    #[test]
    fn test_serializer_round_trip() {
        let entry = KernelCacheEntry::new(
            vec![1, 2, 3, 4],
            "build ok".into(),
            Duration::from_millis(42),
            "Intel Arc".into(),
        );
        let data = CacheSerializer::serialize(&entry);
        let restored = CacheSerializer::deserialize(&data).unwrap();
        assert_eq!(restored.binary, entry.binary);
        assert_eq!(restored.build_log, entry.build_log);
        assert_eq!(restored.device_info, entry.device_info);
    }

    #[test]
    fn test_serializer_bad_magic() {
        let mut data = CacheSerializer::serialize(&KernelCacheEntry::new(
            vec![],
            String::new(),
            Duration::ZERO,
            String::new(),
        ));
        data[0] = 0xFF;
        assert!(CacheSerializer::deserialize(&data).is_err());
    }

    #[test]
    fn test_serializer_bad_version() {
        let mut data = CacheSerializer::serialize(&KernelCacheEntry::new(
            vec![],
            String::new(),
            Duration::ZERO,
            String::new(),
        ));
        // Version is at bytes 4..8, set to 99.
        data[4] = 99;
        data[5] = 0;
        data[6] = 0;
        data[7] = 0;
        assert!(CacheSerializer::deserialize(&data).is_err());
    }

    #[test]
    fn test_serializer_truncated_data() {
        let data = CacheSerializer::serialize(&KernelCacheEntry::new(
            vec![1, 2, 3],
            "log".into(),
            Duration::ZERO,
            String::new(),
        ));
        // Cut data in half.
        let truncated = &data[..data.len() / 2];
        assert!(CacheSerializer::deserialize(truncated).is_err());
    }

    // -----------------------------------------------------------------------
    // Warm-up
    // -----------------------------------------------------------------------

    #[test]
    fn test_warm_up_memory_only_returns_zero() {
        let mgr = default_manager(8);
        assert_eq!(mgr.warm_up().unwrap(), 0);
    }

    #[test]
    fn test_warm_up_loads_from_disk() {
        let tmp = tempfile::tempdir().unwrap();
        let config = CacheConfig {
            policy: CachePolicy::MemoryAndDisk,
            cache_dir: tmp.path().to_path_buf(),
            ..Default::default()
        };

        // Write entries with one manager.
        let mgr1 = KernelCacheManager::new(config.clone());
        mgr1.store(make_key(1), make_binary(8), String::new());
        mgr1.store(make_key(2), make_binary(16), String::new());

        // Create a fresh manager and warm up.
        let mgr2 = KernelCacheManager::new(config);
        let loaded = mgr2.warm_up().unwrap();
        assert_eq!(loaded, 2);
        assert_eq!(mgr2.len(), 2);
    }

    // -----------------------------------------------------------------------
    // Edge cases
    // -----------------------------------------------------------------------

    #[test]
    fn test_empty_binary() {
        let mgr = default_manager(8);
        let key = make_key(1);
        mgr.store(key.clone(), vec![], String::new());
        let entry = mgr.lookup(&key).unwrap();
        assert!(entry.binary.is_empty());
    }

    #[test]
    fn test_very_large_entry_evicts_all() {
        let config = CacheConfig {
            max_entries: 10,
            max_total_size: 100,
            policy: CachePolicy::MemoryOnly,
            ..Default::default()
        };
        let mgr = KernelCacheManager::new(config);
        for i in 0..5 {
            mgr.store(make_key(i), make_binary(10), String::new());
        }
        // A single large entry that exceeds max_total_size by itself.
        mgr.store(make_key(99), make_binary(90), String::new());
        // The cache should have evicted older entries to make room.
        assert!(mgr.len() <= 2);
    }

    #[test]
    fn test_is_empty_on_new_cache() {
        let mgr = default_manager(8);
        assert!(mgr.is_empty());
    }

    #[test]
    fn test_len_tracks_entries() {
        let mgr = default_manager(8);
        mgr.store(make_key(1), make_binary(8), String::new());
        mgr.store(make_key(2), make_binary(8), String::new());
        assert_eq!(mgr.len(), 2);
    }

    #[test]
    fn test_store_with_metadata() {
        let mgr = default_manager(8);
        let key = make_key(1);
        mgr.store_with_metadata(
            key.clone(),
            make_binary(32),
            "ok".into(),
            Duration::from_millis(100),
            "Intel Arc A770".into(),
        );
        let entry = mgr.lookup(&key).unwrap();
        assert_eq!(entry.binary.len(), 32);
        assert_eq!(entry.device_info, "Intel Arc A770");
    }

    #[test]
    fn test_no_cache_miss_counted_in_stats() {
        let config = CacheConfig { policy: CachePolicy::NoCache, ..Default::default() };
        let mgr = KernelCacheManager::new(config);
        mgr.lookup(&make_key(1));
        assert_eq!(mgr.stats().misses, 1);
    }

    #[test]
    fn test_invalidate_updates_size() {
        let mgr = default_manager(8);
        let key = make_key(1);
        mgr.store(key.clone(), make_binary(50), String::new());
        assert_eq!(mgr.stats().total_binary_size, 50);
        mgr.invalidate(&key);
        assert_eq!(mgr.stats().total_binary_size, 0);
    }

    #[test]
    fn test_debug_format() {
        let mgr = default_manager(4);
        let dbg = format!("{mgr:?}");
        assert!(dbg.contains("KernelCacheManager"));
    }

    #[test]
    fn test_invalidate_disk_entry() {
        let tmp = tempfile::tempdir().unwrap();
        let config = CacheConfig {
            policy: CachePolicy::MemoryAndDisk,
            cache_dir: tmp.path().to_path_buf(),
            ..Default::default()
        };
        let mgr = KernelCacheManager::new(config);
        let key = make_key(1);
        mgr.store(key.clone(), make_binary(8), String::new());
        assert_eq!(std::fs::read_dir(tmp.path()).unwrap().count(), 1);
        mgr.invalidate(&key);
        assert_eq!(std::fs::read_dir(tmp.path()).unwrap().count(), 0);
    }
}
