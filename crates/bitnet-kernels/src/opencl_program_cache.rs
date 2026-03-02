//! OpenCL program binary persistence for fast kernel loading on Intel Arc A770.
//!
//! Provides disk-backed caching of compiled OpenCL program binaries so that
//! subsequent launches skip expensive online compilation. A deterministic
//! [`CacheKey`] built from kernel source, device fingerprint, and compile flags
//! ensures binaries are only reused when the compilation environment matches.
//!
//! # A770-specific considerations
//!
//! - [`DeviceFingerprint`] captures driver version, device name, and PCI ID so
//!   that a driver update automatically invalidates stale binaries.
//! - TTL-based expiration (configurable via [`CacheConfig::ttl_days`]) guards
//!   against silent binary-format changes across driver releases.
//! - Default cache directory: `~/.cache/bitnet-rs/opencl-binaries/`

use std::collections::{HashMap, VecDeque};
use std::fmt;
use std::hash::{Hash, Hasher};
use std::path::PathBuf;
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant, SystemTime};

// ---------------------------------------------------------------------------
// DeviceFingerprint
// ---------------------------------------------------------------------------

/// Fingerprint that uniquely identifies an OpenCL device + driver combination.
///
/// Binary compatibility can break across driver updates, so the fingerprint
/// includes the driver version string alongside hardware identifiers.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct DeviceFingerprint {
    /// Device name (e.g. "Intel(R) Arc(TM) A770 Graphics").
    pub device_name: String,
    /// Driver version string (e.g. "23.35.27191.42").
    pub driver_version: String,
    /// PCI device ID (e.g. "0x56a0") or empty when unavailable.
    pub pci_id: String,
}

impl DeviceFingerprint {
    pub fn new(
        device_name: impl Into<String>,
        driver_version: impl Into<String>,
        pci_id: impl Into<String>,
    ) -> Self {
        Self {
            device_name: device_name.into(),
            driver_version: driver_version.into(),
            pci_id: pci_id.into(),
        }
    }

    /// Compute a deterministic 64-bit hash of this fingerprint.
    pub fn hash_u64(&self) -> u64 {
        use std::hash::DefaultHasher;
        let mut h = DefaultHasher::new();
        self.hash(&mut h);
        h.finish()
    }
}

impl fmt::Display for DeviceFingerprint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}|drv={}|pci={}", self.device_name, self.driver_version, self.pci_id)
    }
}

// ---------------------------------------------------------------------------
// ProgramBinary
// ---------------------------------------------------------------------------

/// Compiled OpenCL program binary blob with provenance metadata.
#[derive(Debug, Clone)]
pub struct ProgramBinary {
    /// Raw binary bytes obtained from `clGetProgramInfo(CL_PROGRAM_BINARIES)`.
    pub data: Vec<u8>,
    /// Hash of the kernel source that produced this binary.
    pub source_hash: u64,
    /// Fingerprint of the device the binary was compiled for.
    pub device_fingerprint: DeviceFingerprint,
    /// Compile options passed to `clBuildProgram`.
    pub compile_options: String,
    /// How long compilation took.
    pub compilation_time: Duration,
}

impl ProgramBinary {
    pub fn new(
        data: Vec<u8>,
        source_hash: u64,
        device_fingerprint: DeviceFingerprint,
        compile_options: impl Into<String>,
        compilation_time: Duration,
    ) -> Self {
        Self {
            data,
            source_hash,
            device_fingerprint,
            compile_options: compile_options.into(),
            compilation_time,
        }
    }

    /// Size of the binary blob in bytes.
    pub fn size(&self) -> usize {
        self.data.len()
    }
}

// ---------------------------------------------------------------------------
// CacheKey
// ---------------------------------------------------------------------------

/// Deterministic key for looking up a cached program binary.
///
/// Constructed from the kernel source hash, device fingerprint hash, and
/// compile flags so that any change in source, device, or options yields a
/// distinct key.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct CacheKey {
    /// 64-bit hash of the kernel source text.
    pub source_hash: u64,
    /// 64-bit hash of the [`DeviceFingerprint`].
    pub device_hash: u64,
    /// Normalised compile options string.
    pub compile_options: String,
}

impl CacheKey {
    pub fn new(source_hash: u64, device_hash: u64, compile_options: impl Into<String>) -> Self {
        Self { source_hash, device_hash, compile_options: compile_options.into() }
    }

    /// Build a key from raw source text and a device fingerprint.
    pub fn from_source(
        source: &str,
        fingerprint: &DeviceFingerprint,
        compile_options: impl Into<String>,
    ) -> Self {
        Self::new(hash_bytes(source.as_bytes()), fingerprint.hash_u64(), compile_options)
    }

    /// Deterministic filename for disk persistence.
    pub fn filename(&self) -> String {
        let opts_hash = hash_bytes(self.compile_options.as_bytes());
        format!("{:016x}_{:016x}_{:016x}.bin", self.source_hash, self.device_hash, opts_hash)
    }
}

impl fmt::Display for CacheKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "CacheKey(src={:016x}, dev={:016x}, opts={})",
            self.source_hash,
            self.device_hash,
            if self.compile_options.is_empty() { "<none>" } else { &self.compile_options },
        )
    }
}

// ---------------------------------------------------------------------------
// Compression
// ---------------------------------------------------------------------------

/// Compression algorithm for cached binaries on disk.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Compression {
    /// No compression — store raw bytes.
    None,
    /// Zstd compression (placeholder — actual zstd requires external crate).
    /// When selected without the zstd runtime, falls back to [`Compression::None`].
    Zstd,
}

impl Compression {
    /// Compress `data` according to the selected algorithm.
    /// The CPU reference implementation uses identity for both variants.
    pub fn compress(&self, data: &[u8]) -> Vec<u8> {
        match self {
            Self::None | Self::Zstd => data.to_vec(),
        }
    }

    /// Decompress `data` according to the selected algorithm.
    pub fn decompress(&self, data: &[u8]) -> Result<Vec<u8>, String> {
        match self {
            Self::None | Self::Zstd => Ok(data.to_vec()),
        }
    }
}

// ---------------------------------------------------------------------------
// CacheConfig
// ---------------------------------------------------------------------------

/// Configuration for the program binary cache.
#[derive(Debug, Clone)]
pub struct CacheConfig {
    /// Directory for cached binaries on disk.
    pub cache_dir: PathBuf,
    /// Maximum aggregate size of all cached binaries (bytes).
    pub max_size_bytes: usize,
    /// Maximum number of cache entries.
    pub max_entries: usize,
    /// Time-to-live in days. Entries older than this are considered stale.
    /// `None` means entries never expire.
    pub ttl_days: Option<u32>,
    /// Compression algorithm for on-disk storage.
    pub compression: Compression,
    /// Eviction policy when the cache is full.
    pub eviction: CacheEviction,
}

impl Default for CacheConfig {
    fn default() -> Self {
        let cache_dir = dirs_fallback().join("opencl-binaries");
        Self {
            cache_dir,
            max_size_bytes: 512 * 1024 * 1024, // 512 MiB
            max_entries: 512,
            ttl_days: Some(30),
            compression: Compression::None,
            eviction: CacheEviction::Lru,
        }
    }
}

impl CacheConfig {
    /// Validate the configuration.
    pub fn validate(&self) -> Result<(), String> {
        if self.max_size_bytes == 0 {
            return Err("max_size_bytes must be > 0".into());
        }
        if self.max_entries == 0 {
            return Err("max_entries must be > 0".into());
        }
        Ok(())
    }

    /// TTL as a [`Duration`], or `None` if no TTL is configured.
    pub fn ttl_duration(&self) -> Option<Duration> {
        self.ttl_days.map(|d| Duration::from_secs(u64::from(d) * 86_400))
    }
}

/// Return a sensible fallback cache root.
fn dirs_fallback() -> PathBuf {
    if let Ok(home) = std::env::var("HOME") {
        PathBuf::from(home).join(".cache").join("bitnet-rs")
    } else if let Ok(home) = std::env::var("USERPROFILE") {
        PathBuf::from(home).join(".cache").join("bitnet-rs")
    } else {
        PathBuf::from(".cache").join("bitnet-rs")
    }
}

// ---------------------------------------------------------------------------
// CacheEviction
// ---------------------------------------------------------------------------

/// Eviction policy for the binary cache.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheEviction {
    /// Least Recently Used — evict the entry that was accessed longest ago.
    Lru,
    /// Least Frequently Used — evict the entry with the fewest accesses.
    Lfu,
    /// Size-based — evict the largest entry first.
    SizeBased,
}

// ---------------------------------------------------------------------------
// CacheEntry
// ---------------------------------------------------------------------------

/// A stored cache entry: binary data plus bookkeeping metadata.
#[derive(Debug, Clone)]
pub struct CacheEntry {
    /// The compiled binary (possibly compressed on disk).
    pub binary: ProgramBinary,
    /// When the entry was first stored.
    pub created_at: SystemTime,
    /// When the entry was last accessed (read hit).
    pub last_accessed: SystemTime,
    /// Number of cache hits on this entry.
    pub hit_count: u64,
}

impl CacheEntry {
    pub fn new(binary: ProgramBinary) -> Self {
        let now = SystemTime::now();
        Self { binary, created_at: now, last_accessed: now, hit_count: 0 }
    }

    /// Total size of the cached binary blob.
    pub fn size(&self) -> usize {
        self.binary.size()
    }

    /// Mark the entry as accessed.
    pub fn touch(&mut self) {
        self.last_accessed = SystemTime::now();
        self.hit_count += 1;
    }

    /// Returns `true` if this entry has expired given `ttl`.
    pub fn is_expired(&self, ttl: Duration) -> bool {
        self.created_at.elapsed().map(|age| age > ttl).unwrap_or(false)
    }
}

// ---------------------------------------------------------------------------
// CacheStats
// ---------------------------------------------------------------------------

/// Runtime statistics for the program binary cache.
#[derive(Debug, Clone, Default)]
pub struct CacheStats {
    /// Number of cache hits.
    pub hits: u64,
    /// Number of cache misses.
    pub misses: u64,
    /// Number of entries evicted.
    pub evictions: u64,
    /// Total size of all stored binaries (bytes).
    pub total_size: usize,
    /// Number of entries currently stored.
    pub entry_count: usize,
    /// Cumulative time saved by cache hits (sum of original compilation times).
    pub total_time_saved: Duration,
}

impl CacheStats {
    /// Hit rate as a fraction in `[0.0, 1.0]`.
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 { 0.0 } else { self.hits as f64 / total as f64 }
    }

    /// Average load time saved per cache hit.
    pub fn avg_load_time_saved(&self) -> Duration {
        if self.hits == 0 { Duration::ZERO } else { self.total_time_saved / self.hits as u32 }
    }
}

impl fmt::Display for CacheStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "CacheStats(hits={}, misses={}, evictions={}, entries={}, \
             size={}B, hit_rate={:.1}%, avg_saved={:?})",
            self.hits,
            self.misses,
            self.evictions,
            self.entry_count,
            self.total_size,
            self.hit_rate() * 100.0,
            self.avg_load_time_saved(),
        )
    }
}

// ---------------------------------------------------------------------------
// WarmupStrategy
// ---------------------------------------------------------------------------

/// Strategy for pre-compiling commonly used kernels at startup.
#[derive(Debug, Clone)]
pub enum WarmupStrategy {
    /// No warmup — compile on first use.
    None,
    /// Warm up the N most frequently used kernels from cache history.
    MostFrequent(usize),
    /// Warm up a specific set of kernel sources + options.
    Explicit(Vec<WarmupSpec>),
    /// Warm up all entries found in the cache directory.
    All,
}

/// Specification for a single kernel to pre-compile during warmup.
#[derive(Debug, Clone)]
pub struct WarmupSpec {
    /// Kernel source code.
    pub source: String,
    /// Compile options.
    pub compile_options: String,
}

impl WarmupSpec {
    pub fn new(source: impl Into<String>, compile_options: impl Into<String>) -> Self {
        Self { source: source.into(), compile_options: compile_options.into() }
    }
}

/// Result of a warmup pass.
#[derive(Debug, Clone, Default)]
pub struct WarmupResult {
    /// Number of kernels loaded from cache.
    pub loaded: usize,
    /// Number of kernels that needed recompilation.
    pub compiled: usize,
    /// Number of failures.
    pub failed: usize,
    /// Total wall time for the warmup.
    pub elapsed: Duration,
}

// ---------------------------------------------------------------------------
// BinaryCache — the main thread-safe cache
// ---------------------------------------------------------------------------

/// Thread-safe disk-backed cache for compiled OpenCL program binaries.
///
/// The cache maps [`CacheKey`] → [`CacheEntry`] with configurable eviction,
/// TTL, and optional compression. A CPU reference implementation keeps entries
/// in an in-memory [`HashMap`]; disk persistence is layered on top.
pub struct BinaryCache {
    config: CacheConfig,
    inner: RwLock<CacheInner>,
}

struct CacheInner {
    entries: HashMap<CacheKey, CacheEntry>,
    /// Access / insertion order for LRU and FIFO eviction.
    order: VecDeque<CacheKey>,
    stats: CacheStats,
}

impl BinaryCache {
    /// Create a new cache with the given configuration.
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

    /// Create a cache with default configuration.
    pub fn with_defaults() -> Self {
        Self::new(CacheConfig::default())
    }

    /// Look up a cached binary. Returns a cloned entry on hit, `None` on miss.
    pub fn lookup(&self, key: &CacheKey) -> Option<CacheEntry> {
        let mut inner = self.inner.write().unwrap();

        // Check in-memory first.
        if let Some(entry) = inner.entries.get_mut(key) {
            // Check TTL.
            if let Some(ttl) = self.config.ttl_duration()
                && entry.is_expired(ttl)
            {
                let sz = entry.size();
                inner.entries.remove(key);
                inner.order.retain(|k| k != key);
                inner.stats.total_size = inner.stats.total_size.saturating_sub(sz);
                inner.stats.entry_count = inner.entries.len();
                inner.stats.misses += 1;
                return None;
            }
            entry.touch();
            let cloned = entry.clone();
            // Move to back for LRU.
            inner.order.retain(|k| k != key);
            inner.order.push_back(key.clone());
            inner.stats.hits += 1;
            inner.stats.total_time_saved += cloned.binary.compilation_time;
            return Some(cloned);
        }

        // Try disk.
        if let Some(mut entry) = self.load_from_disk(key) {
            entry.touch();
            let cloned = entry.clone();
            inner.stats.total_size += entry.size();
            inner.order.push_back(key.clone());
            inner.entries.insert(key.clone(), entry);
            inner.stats.entry_count = inner.entries.len();
            inner.stats.hits += 1;
            inner.stats.total_time_saved += cloned.binary.compilation_time;
            return Some(cloned);
        }

        inner.stats.misses += 1;
        None
    }

    /// Store a program binary in the cache.
    pub fn store(&self, key: CacheKey, binary: ProgramBinary) {
        let entry = CacheEntry::new(binary);
        let entry_size = entry.size();

        let mut inner = self.inner.write().unwrap();

        // Evict until we have room.
        while inner.entries.len() >= self.config.max_entries
            || (inner.stats.total_size + entry_size > self.config.max_size_bytes
                && !inner.entries.is_empty())
        {
            if let Some(evict_key) = self.pick_eviction(&inner) {
                if let Some(removed) = inner.entries.remove(&evict_key) {
                    inner.stats.total_size = inner.stats.total_size.saturating_sub(removed.size());
                    inner.order.retain(|k| k != &evict_key);
                    inner.stats.evictions += 1;
                    // Remove from disk too.
                    self.remove_from_disk(&evict_key);
                }
            } else {
                break;
            }
        }

        inner.stats.total_size += entry_size;
        inner.order.push_back(key.clone());
        inner.entries.insert(key.clone(), entry.clone());
        inner.stats.entry_count = inner.entries.len();

        // Persist to disk.
        self.save_to_disk(&key, &entry);
    }

    /// Remove a specific entry from the cache.
    pub fn invalidate(&self, key: &CacheKey) {
        let mut inner = self.inner.write().unwrap();
        if let Some(removed) = inner.entries.remove(key) {
            inner.stats.total_size = inner.stats.total_size.saturating_sub(removed.size());
            inner.order.retain(|k| k != key);
            inner.stats.entry_count = inner.entries.len();
        }
        self.remove_from_disk(key);
    }

    /// Remove all entries.
    pub fn clear(&self) {
        let mut inner = self.inner.write().unwrap();
        inner.entries.clear();
        inner.order.clear();
        inner.stats.total_size = 0;
        inner.stats.entry_count = 0;
    }

    /// Snapshot of current statistics.
    pub fn stats(&self) -> CacheStats {
        self.inner.read().unwrap().stats.clone()
    }

    /// Number of entries in memory.
    pub fn len(&self) -> usize {
        self.inner.read().unwrap().entries.len()
    }

    /// Whether the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Reference to the configuration.
    pub fn config(&self) -> &CacheConfig {
        &self.config
    }

    /// Run a warmup pass according to the given strategy.
    ///
    /// The `compile_fn` callback is invoked for each kernel that needs
    /// (re)compilation. It receives `(source, compile_options)` and should
    /// return the compiled [`ProgramBinary`].
    pub fn warmup<F>(
        &self,
        strategy: &WarmupStrategy,
        fingerprint: &DeviceFingerprint,
        compile_fn: F,
    ) -> WarmupResult
    where
        F: Fn(&str, &str) -> Result<ProgramBinary, String>,
    {
        let start = Instant::now();
        let mut result = WarmupResult::default();

        let specs: Vec<WarmupSpec> = match strategy {
            WarmupStrategy::None => return result,
            WarmupStrategy::Explicit(specs) => specs.clone(),
            WarmupStrategy::MostFrequent(n) => {
                let inner = self.inner.read().unwrap();
                let mut by_hits: Vec<_> =
                    inner.entries.iter().map(|(k, e)| (k.clone(), e.hit_count)).collect();
                by_hits.sort_by(|a, b| b.1.cmp(&a.1));
                by_hits
                    .into_iter()
                    .take(*n)
                    .map(|(k, _)| WarmupSpec {
                        source: format!("<cached:{:016x}>", k.source_hash),
                        compile_options: k.compile_options.clone(),
                    })
                    .collect()
            }
            WarmupStrategy::All => {
                let inner = self.inner.read().unwrap();
                inner
                    .entries
                    .keys()
                    .map(|k| WarmupSpec {
                        source: format!("<cached:{:016x}>", k.source_hash),
                        compile_options: k.compile_options.clone(),
                    })
                    .collect()
            }
        };

        for spec in &specs {
            let key = CacheKey::from_source(&spec.source, fingerprint, &spec.compile_options);
            if self.lookup(&key).is_some() {
                result.loaded += 1;
            } else {
                match compile_fn(&spec.source, &spec.compile_options) {
                    Ok(binary) => {
                        self.store(key, binary);
                        result.compiled += 1;
                    }
                    Err(_) => {
                        result.failed += 1;
                    }
                }
            }
        }

        result.elapsed = start.elapsed();
        result
    }

    /// Pre-populate from disk entries that pass TTL validation.
    /// Returns the count of entries loaded.
    pub fn load_disk_entries(&self) -> Result<usize, String> {
        let dir = &self.config.cache_dir;
        if !dir.exists() {
            return Ok(0);
        }
        let rd = std::fs::read_dir(dir).map_err(|e| format!("read dir: {e}"))?;
        let mut loaded = 0usize;
        for de in rd {
            let de = de.map_err(|e| format!("dir entry: {e}"))?;
            let path = de.path();
            if path.extension().and_then(|e| e.to_str()) != Some("bin") {
                continue;
            }
            let raw = match std::fs::read(&path) {
                Ok(d) => d,
                Err(_) => continue,
            };
            let data = match self.config.compression.decompress(&raw) {
                Ok(d) => d,
                Err(_) => continue, // corruption recovery: skip bad entries
            };
            if let Ok(entry) = EntrySerializer::deserialize(&data) {
                if let Some(ttl) = self.config.ttl_duration()
                    && entry.is_expired(ttl)
                {
                    let _ = std::fs::remove_file(&path);
                    continue;
                }
                let key = CacheKey::new(
                    entry.binary.source_hash,
                    entry.binary.device_fingerprint.hash_u64(),
                    &entry.binary.compile_options,
                );
                let mut inner = self.inner.write().unwrap();
                if inner.entries.len() < self.config.max_entries {
                    inner.stats.total_size += entry.size();
                    inner.order.push_back(key.clone());
                    inner.entries.insert(key, entry);
                    inner.stats.entry_count = inner.entries.len();
                    loaded += 1;
                }
            }
            // else: corruption → skip silently (recovery)
        }
        Ok(loaded)
    }

    // -- private helpers ---------------------------------------------------

    fn pick_eviction(&self, inner: &CacheInner) -> Option<CacheKey> {
        match self.config.eviction {
            CacheEviction::Lru => inner.order.front().cloned(),
            CacheEviction::Lfu => {
                inner.entries.iter().min_by_key(|(_, e)| e.hit_count).map(|(k, _)| k.clone())
            }
            CacheEviction::SizeBased => {
                inner.entries.iter().max_by_key(|(_, e)| e.size()).map(|(k, _)| k.clone())
            }
        }
    }

    fn save_to_disk(&self, key: &CacheKey, entry: &CacheEntry) {
        let dir = &self.config.cache_dir;
        let _ = std::fs::create_dir_all(dir);
        let data = EntrySerializer::serialize(entry);
        let compressed = self.config.compression.compress(&data);
        let path = dir.join(key.filename());
        let _ = std::fs::write(path, compressed);
    }

    fn load_from_disk(&self, key: &CacheKey) -> Option<CacheEntry> {
        let path = self.config.cache_dir.join(key.filename());
        let raw = std::fs::read(path).ok()?;
        let data = self.config.compression.decompress(&raw).ok()?;
        let entry = EntrySerializer::deserialize(&data).ok()?;
        // TTL check
        if let Some(ttl) = self.config.ttl_duration()
            && entry.is_expired(ttl)
        {
            return None;
        }
        Some(entry)
    }

    fn remove_from_disk(&self, key: &CacheKey) {
        let path = self.config.cache_dir.join(key.filename());
        let _ = std::fs::remove_file(path);
    }
}

impl fmt::Debug for BinaryCache {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let inner = self.inner.read().unwrap();
        f.debug_struct("BinaryCache")
            .field("config", &self.config)
            .field("entries", &inner.entries.len())
            .field("stats", &inner.stats)
            .finish()
    }
}

// ---------------------------------------------------------------------------
// CpuReferenceCache — simple in-memory HashMap for testing
// ---------------------------------------------------------------------------

/// Minimal in-memory cache backed by a [`HashMap`], intended as a CPU
/// reference implementation for testing and validation.
pub struct CpuReferenceCache {
    entries: Arc<RwLock<HashMap<CacheKey, ProgramBinary>>>,
}

impl CpuReferenceCache {
    pub fn new() -> Self {
        Self { entries: Arc::new(RwLock::new(HashMap::new())) }
    }

    pub fn get(&self, key: &CacheKey) -> Option<ProgramBinary> {
        self.entries.read().unwrap().get(key).cloned()
    }

    pub fn put(&self, key: CacheKey, binary: ProgramBinary) {
        self.entries.write().unwrap().insert(key, binary);
    }

    pub fn remove(&self, key: &CacheKey) -> bool {
        self.entries.write().unwrap().remove(key).is_some()
    }

    pub fn len(&self) -> usize {
        self.entries.read().unwrap().len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn clear(&self) {
        self.entries.write().unwrap().clear();
    }
}

impl Default for CpuReferenceCache {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// EntrySerializer — binary (de)serialization
// ---------------------------------------------------------------------------

/// Binary serializer for [`CacheEntry`] disk persistence.
///
/// Wire format (little-endian):
/// ```text
/// [4B magic "BPBC"][4B version]
/// [8B binary_len][binary_bytes…]
/// [8B source_hash]
/// [8B device_name_len][device_name…]
/// [8B driver_version_len][driver_version…]
/// [8B pci_id_len][pci_id…]
/// [8B compile_options_len][compile_options…]
/// [8B compilation_time_ns]
/// [8B created_at_secs]
/// [8B hit_count]
/// ```
pub struct EntrySerializer;

impl EntrySerializer {
    const MAGIC: [u8; 4] = *b"BPBC"; // BitNet Program Binary Cache
    const VERSION: u32 = 1;

    pub fn serialize(entry: &CacheEntry) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(&Self::MAGIC);
        buf.extend_from_slice(&Self::VERSION.to_le_bytes());

        // binary data
        Self::write_bytes(&mut buf, &entry.binary.data);
        // source_hash
        buf.extend_from_slice(&entry.binary.source_hash.to_le_bytes());
        // device fingerprint fields
        Self::write_str(&mut buf, &entry.binary.device_fingerprint.device_name);
        Self::write_str(&mut buf, &entry.binary.device_fingerprint.driver_version);
        Self::write_str(&mut buf, &entry.binary.device_fingerprint.pci_id);
        // compile options
        Self::write_str(&mut buf, &entry.binary.compile_options);
        // compilation_time as nanos (u64)
        buf.extend_from_slice(&(entry.binary.compilation_time.as_nanos() as u64).to_le_bytes());
        // created_at
        let secs =
            entry.created_at.duration_since(SystemTime::UNIX_EPOCH).unwrap_or_default().as_secs();
        buf.extend_from_slice(&secs.to_le_bytes());
        // hit_count
        buf.extend_from_slice(&entry.hit_count.to_le_bytes());

        buf
    }

    pub fn deserialize(data: &[u8]) -> Result<CacheEntry, String> {
        let mut pos = 0usize;

        // magic
        let magic = Self::read_slice(data, &mut pos, 4)?;
        if magic != Self::MAGIC {
            return Err("invalid magic".into());
        }
        // version
        let ver = u32::from_le_bytes(Self::read_slice(data, &mut pos, 4)?.try_into().unwrap());
        if ver != Self::VERSION {
            return Err(format!("unsupported version {ver}"));
        }

        let binary_data = Self::read_bytes(data, &mut pos)?;
        let source_hash = Self::read_u64(data, &mut pos)?;
        let device_name = Self::read_string(data, &mut pos)?;
        let driver_version = Self::read_string(data, &mut pos)?;
        let pci_id = Self::read_string(data, &mut pos)?;
        let compile_options = Self::read_string(data, &mut pos)?;
        let comp_ns = Self::read_u64(data, &mut pos)?;
        let created_secs = Self::read_u64(data, &mut pos)?;
        let hit_count = Self::read_u64(data, &mut pos)?;

        let fingerprint = DeviceFingerprint::new(device_name, driver_version, pci_id);
        let binary = ProgramBinary::new(
            binary_data,
            source_hash,
            fingerprint,
            compile_options,
            Duration::from_nanos(comp_ns),
        );
        let created_at = SystemTime::UNIX_EPOCH + Duration::from_secs(created_secs);

        Ok(CacheEntry { binary, created_at, last_accessed: SystemTime::now(), hit_count })
    }

    fn write_bytes(buf: &mut Vec<u8>, data: &[u8]) {
        buf.extend_from_slice(&(data.len() as u64).to_le_bytes());
        buf.extend_from_slice(data);
    }

    fn write_str(buf: &mut Vec<u8>, s: &str) {
        Self::write_bytes(buf, s.as_bytes());
    }

    fn read_slice<'a>(data: &'a [u8], pos: &mut usize, n: usize) -> Result<&'a [u8], String> {
        if *pos + n > data.len() {
            return Err("unexpected EOF".into());
        }
        let s = &data[*pos..*pos + n];
        *pos += n;
        Ok(s)
    }

    fn read_u64(data: &[u8], pos: &mut usize) -> Result<u64, String> {
        let b = Self::read_slice(data, pos, 8)?;
        Ok(u64::from_le_bytes(b.try_into().unwrap()))
    }

    fn read_bytes(data: &[u8], pos: &mut usize) -> Result<Vec<u8>, String> {
        let len = Self::read_u64(data, pos)? as usize;
        Ok(Self::read_slice(data, pos, len)?.to_vec())
    }

    fn read_string(data: &[u8], pos: &mut usize) -> Result<String, String> {
        let bytes = Self::read_bytes(data, pos)?;
        String::from_utf8(bytes).map_err(|e| format!("invalid UTF-8: {e}"))
    }
}

// ---------------------------------------------------------------------------
// Utility
// ---------------------------------------------------------------------------

/// Deterministic 64-bit hash of a byte slice (SipHash via DefaultHasher).
pub fn hash_bytes(data: &[u8]) -> u64 {
    use std::hash::DefaultHasher;
    let mut h = DefaultHasher::new();
    data.hash(&mut h);
    h.finish()
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;
    use std::time::Duration;

    // -- test helpers -------------------------------------------------------

    fn test_fingerprint() -> DeviceFingerprint {
        DeviceFingerprint::new("Intel(R) Arc(TM) A770 Graphics", "23.35.27191.42", "0x56a0")
    }

    fn alt_fingerprint() -> DeviceFingerprint {
        DeviceFingerprint::new("NVIDIA RTX 4090", "545.23.08", "0x2684")
    }

    fn make_binary(size: usize) -> ProgramBinary {
        ProgramBinary::new(
            vec![0xAB; size],
            42,
            test_fingerprint(),
            "-cl-mad-enable",
            Duration::from_millis(100),
        )
    }

    fn make_key(id: u64) -> CacheKey {
        CacheKey::new(id, test_fingerprint().hash_u64(), "-cl-mad-enable")
    }

    fn temp_cache(max_entries: usize, max_size: usize) -> BinaryCache {
        let dir = std::env::temp_dir().join("bitnet-rs-test-progcache").join(format!(
            "{:x}",
            hash_bytes(std::thread::current().name().unwrap_or("t").as_bytes(),)
        ));
        let _ = std::fs::remove_dir_all(&dir);
        let config = CacheConfig {
            cache_dir: dir,
            max_size_bytes: max_size,
            max_entries,
            ttl_days: None,
            compression: Compression::None,
            eviction: CacheEviction::Lru,
        };
        BinaryCache::new(config)
    }

    fn temp_cache_with_ttl(ttl_days: u32) -> BinaryCache {
        let dir = std::env::temp_dir().join("bitnet-rs-test-progcache-ttl").join(format!(
            "{:x}",
            hash_bytes(std::thread::current().name().unwrap_or("t").as_bytes(),)
        ));
        let _ = std::fs::remove_dir_all(&dir);
        let config = CacheConfig {
            cache_dir: dir,
            max_size_bytes: 1024 * 1024,
            max_entries: 64,
            ttl_days: Some(ttl_days),
            compression: Compression::None,
            eviction: CacheEviction::Lru,
        };
        BinaryCache::new(config)
    }

    // -----------------------------------------------------------------------
    // DeviceFingerprint
    // -----------------------------------------------------------------------

    #[test]
    fn fingerprint_equality() {
        let a = test_fingerprint();
        let b = test_fingerprint();
        assert_eq!(a, b);
    }

    #[test]
    fn fingerprint_inequality_driver() {
        let a = test_fingerprint();
        let mut b = test_fingerprint();
        b.driver_version = "99.99.99".into();
        assert_ne!(a, b);
    }

    #[test]
    fn fingerprint_inequality_pci_id() {
        let a = test_fingerprint();
        let mut b = test_fingerprint();
        b.pci_id = "0xBEEF".into();
        assert_ne!(a, b);
    }

    #[test]
    fn fingerprint_hash_deterministic() {
        let fp = test_fingerprint();
        assert_eq!(fp.hash_u64(), fp.hash_u64());
    }

    #[test]
    fn fingerprint_hash_varies_with_driver() {
        let a = test_fingerprint();
        let mut b = test_fingerprint();
        b.driver_version = "0.0.1".into();
        assert_ne!(a.hash_u64(), b.hash_u64());
    }

    #[test]
    fn fingerprint_display() {
        let fp = test_fingerprint();
        let s = format!("{fp}");
        assert!(s.contains("A770"));
        assert!(s.contains("23.35.27191.42"));
        assert!(s.contains("0x56a0"));
    }

    // -----------------------------------------------------------------------
    // CacheKey
    // -----------------------------------------------------------------------

    #[test]
    fn key_equality() {
        assert_eq!(make_key(1), make_key(1));
    }

    #[test]
    fn key_inequality_source_hash() {
        assert_ne!(make_key(1), make_key(2));
    }

    #[test]
    fn key_inequality_device_hash() {
        let a = CacheKey::new(1, 100, "");
        let b = CacheKey::new(1, 200, "");
        assert_ne!(a, b);
    }

    #[test]
    fn key_inequality_options() {
        let a = CacheKey::new(1, 1, "-O0");
        let b = CacheKey::new(1, 1, "-O2");
        assert_ne!(a, b);
    }

    #[test]
    fn key_from_source_deterministic() {
        let fp = test_fingerprint();
        let k1 = CacheKey::from_source("kernel void foo(){}", &fp, "-O2");
        let k2 = CacheKey::from_source("kernel void foo(){}", &fp, "-O2");
        assert_eq!(k1, k2);
    }

    #[test]
    fn key_from_source_different_source() {
        let fp = test_fingerprint();
        let k1 = CacheKey::from_source("kernel void foo(){}", &fp, "");
        let k2 = CacheKey::from_source("kernel void bar(){}", &fp, "");
        assert_ne!(k1, k2);
    }

    #[test]
    fn key_from_source_different_device() {
        let k1 = CacheKey::from_source("src", &test_fingerprint(), "");
        let k2 = CacheKey::from_source("src", &alt_fingerprint(), "");
        assert_ne!(k1, k2);
    }

    #[test]
    fn key_filename_deterministic() {
        let k = make_key(42);
        assert_eq!(k.filename(), k.filename());
    }

    #[test]
    fn key_filename_ends_with_bin() {
        assert!(make_key(1).filename().ends_with(".bin"));
    }

    #[test]
    fn key_display() {
        let s = format!("{}", make_key(255));
        assert!(s.contains("00000000000000ff"));
    }

    // -----------------------------------------------------------------------
    // CacheConfig
    // -----------------------------------------------------------------------

    #[test]
    fn config_default_valid() {
        assert!(CacheConfig::default().validate().is_ok());
    }

    #[test]
    fn config_zero_entries_invalid() {
        let mut c = CacheConfig::default();
        c.max_entries = 0;
        assert!(c.validate().is_err());
    }

    #[test]
    fn config_zero_size_invalid() {
        let mut c = CacheConfig::default();
        c.max_size_bytes = 0;
        assert!(c.validate().is_err());
    }

    #[test]
    fn config_ttl_duration_some() {
        let mut c = CacheConfig::default();
        c.ttl_days = Some(7);
        assert_eq!(c.ttl_duration(), Some(Duration::from_secs(7 * 86_400)));
    }

    #[test]
    fn config_ttl_duration_none() {
        let mut c = CacheConfig::default();
        c.ttl_days = None;
        assert_eq!(c.ttl_duration(), None);
    }

    #[test]
    fn config_default_cache_dir() {
        let c = CacheConfig::default();
        let dir_str = c.cache_dir.to_string_lossy();
        assert!(dir_str.contains("opencl-binaries"), "dir={dir_str}");
    }

    // -----------------------------------------------------------------------
    // ProgramBinary
    // -----------------------------------------------------------------------

    #[test]
    fn program_binary_size() {
        let b = make_binary(256);
        assert_eq!(b.size(), 256);
    }

    #[test]
    fn program_binary_empty() {
        let b = ProgramBinary::new(vec![], 0, test_fingerprint(), "", Duration::ZERO);
        assert_eq!(b.size(), 0);
    }

    // -----------------------------------------------------------------------
    // CacheEntry
    // -----------------------------------------------------------------------

    #[test]
    fn entry_touch_increments_hit_count() {
        let mut e = CacheEntry::new(make_binary(64));
        assert_eq!(e.hit_count, 0);
        e.touch();
        assert_eq!(e.hit_count, 1);
        e.touch();
        assert_eq!(e.hit_count, 2);
    }

    #[test]
    fn entry_not_expired_when_fresh() {
        let e = CacheEntry::new(make_binary(64));
        assert!(!e.is_expired(Duration::from_secs(3600)));
    }

    // -----------------------------------------------------------------------
    // Compression
    // -----------------------------------------------------------------------

    #[test]
    fn compression_none_roundtrip() {
        let data = b"hello world";
        let c = Compression::None;
        let compressed = c.compress(data);
        let decompressed = c.decompress(&compressed).unwrap();
        assert_eq!(decompressed, data);
    }

    #[test]
    fn compression_zstd_fallback_roundtrip() {
        let data = b"test binary data";
        let c = Compression::Zstd;
        let compressed = c.compress(data);
        let decompressed = c.decompress(&compressed).unwrap();
        assert_eq!(decompressed, data);
    }

    // -----------------------------------------------------------------------
    // EntrySerializer
    // -----------------------------------------------------------------------

    #[test]
    fn serializer_roundtrip() {
        let entry = CacheEntry::new(make_binary(128));
        let data = EntrySerializer::serialize(&entry);
        let restored = EntrySerializer::deserialize(&data).unwrap();
        assert_eq!(restored.binary.data, entry.binary.data);
        assert_eq!(restored.binary.source_hash, entry.binary.source_hash);
        assert_eq!(restored.binary.device_fingerprint, entry.binary.device_fingerprint,);
        assert_eq!(restored.binary.compile_options, entry.binary.compile_options);
    }

    #[test]
    fn serializer_roundtrip_empty_binary() {
        let binary = ProgramBinary::new(vec![], 0, test_fingerprint(), "", Duration::ZERO);
        let entry = CacheEntry::new(binary);
        let data = EntrySerializer::serialize(&entry);
        let restored = EntrySerializer::deserialize(&data).unwrap();
        assert!(restored.binary.data.is_empty());
    }

    #[test]
    fn serializer_invalid_magic() {
        let mut data = EntrySerializer::serialize(&CacheEntry::new(make_binary(8)));
        data[0] = b'X';
        assert!(EntrySerializer::deserialize(&data).is_err());
    }

    #[test]
    fn serializer_invalid_version() {
        let mut data = EntrySerializer::serialize(&CacheEntry::new(make_binary(8)));
        // Overwrite version bytes (offset 4..8) with 99
        data[4] = 99;
        data[5] = 0;
        data[6] = 0;
        data[7] = 0;
        assert!(EntrySerializer::deserialize(&data).is_err());
    }

    #[test]
    fn serializer_truncated_data() {
        let data = EntrySerializer::serialize(&CacheEntry::new(make_binary(64)));
        assert!(EntrySerializer::deserialize(&data[..10]).is_err());
    }

    #[test]
    fn serializer_preserves_hit_count() {
        let mut entry = CacheEntry::new(make_binary(32));
        entry.hit_count = 42;
        let data = EntrySerializer::serialize(&entry);
        let restored = EntrySerializer::deserialize(&data).unwrap();
        assert_eq!(restored.hit_count, 42);
    }

    // -----------------------------------------------------------------------
    // BinaryCache — miss then hit
    // -----------------------------------------------------------------------

    #[test]
    fn cache_miss_then_store_then_hit() {
        let cache = temp_cache(64, 1024 * 1024);
        let key = make_key(1);

        // Miss.
        assert!(cache.lookup(&key).is_none());
        assert_eq!(cache.stats().misses, 1);

        // Store.
        cache.store(key.clone(), make_binary(128));

        // Hit.
        let entry = cache.lookup(&key).unwrap();
        assert_eq!(entry.binary.data.len(), 128);
        let s = cache.stats();
        assert_eq!(s.hits, 1);
        assert_eq!(s.misses, 1);
    }

    #[test]
    fn cache_stores_update_entry_count() {
        let cache = temp_cache(64, 1024 * 1024);
        assert_eq!(cache.len(), 0);
        assert!(cache.is_empty());

        cache.store(make_key(1), make_binary(64));
        assert_eq!(cache.len(), 1);
        assert!(!cache.is_empty());

        cache.store(make_key(2), make_binary(64));
        assert_eq!(cache.len(), 2);
    }

    #[test]
    fn cache_invalidate_removes_entry() {
        let cache = temp_cache(64, 1024 * 1024);
        let key = make_key(1);
        cache.store(key.clone(), make_binary(64));
        assert_eq!(cache.len(), 1);

        cache.invalidate(&key);
        assert_eq!(cache.len(), 0);
        assert!(cache.lookup(&key).is_none());
    }

    #[test]
    fn cache_clear_removes_all() {
        let cache = temp_cache(64, 1024 * 1024);
        for i in 0..5 {
            cache.store(make_key(i), make_binary(32));
        }
        assert_eq!(cache.len(), 5);
        cache.clear();
        assert_eq!(cache.len(), 0);
    }

    // -----------------------------------------------------------------------
    // Eviction — LRU
    // -----------------------------------------------------------------------

    #[test]
    fn eviction_lru_basic() {
        let cache = temp_cache(3, 1024 * 1024);
        // Fill to capacity.
        cache.store(make_key(1), make_binary(32));
        cache.store(make_key(2), make_binary(32));
        cache.store(make_key(3), make_binary(32));
        assert_eq!(cache.len(), 3);

        // Inserting a 4th evicts the LRU (key 1).
        cache.store(make_key(4), make_binary(32));
        assert_eq!(cache.len(), 3);
        assert!(cache.lookup(&make_key(1)).is_none()); // evicted
        assert!(cache.lookup(&make_key(4)).is_some()); // present
    }

    #[test]
    fn eviction_lru_access_refreshes() {
        let cache = temp_cache(3, 1024 * 1024);
        cache.store(make_key(1), make_binary(32));
        cache.store(make_key(2), make_binary(32));
        cache.store(make_key(3), make_binary(32));

        // Access key 1 to refresh it.
        cache.lookup(&make_key(1));

        // Now key 2 is the LRU candidate.
        cache.store(make_key(4), make_binary(32));
        assert!(cache.lookup(&make_key(2)).is_none()); // evicted
        assert!(cache.lookup(&make_key(1)).is_some()); // refreshed, still present
    }

    #[test]
    fn eviction_count_tracked() {
        let cache = temp_cache(2, 1024 * 1024);
        cache.store(make_key(1), make_binary(32));
        cache.store(make_key(2), make_binary(32));
        cache.store(make_key(3), make_binary(32)); // evicts 1
        assert_eq!(cache.stats().evictions, 1);
    }

    // -----------------------------------------------------------------------
    // Eviction — LFU
    // -----------------------------------------------------------------------

    #[test]
    fn eviction_lfu() {
        let dir = std::env::temp_dir().join("bitnet-rs-test-lfu");
        let _ = std::fs::remove_dir_all(&dir);
        let config = CacheConfig {
            cache_dir: dir,
            max_size_bytes: 1024 * 1024,
            max_entries: 3,
            ttl_days: None,
            compression: Compression::None,
            eviction: CacheEviction::Lfu,
        };
        let cache = BinaryCache::new(config);

        cache.store(make_key(1), make_binary(32));
        cache.store(make_key(2), make_binary(32));
        cache.store(make_key(3), make_binary(32));

        // Access 1 and 3 so they have higher hit counts.
        cache.lookup(&make_key(1));
        cache.lookup(&make_key(1));
        cache.lookup(&make_key(3));

        // key 2 has lowest hit count (0) — should be evicted.
        cache.store(make_key(4), make_binary(32));
        assert!(cache.lookup(&make_key(2)).is_none());
        assert!(cache.lookup(&make_key(1)).is_some());
    }

    // -----------------------------------------------------------------------
    // Eviction — size-based
    // -----------------------------------------------------------------------

    #[test]
    fn eviction_size_based() {
        let dir = std::env::temp_dir().join("bitnet-rs-test-sizebased");
        let _ = std::fs::remove_dir_all(&dir);
        let config = CacheConfig {
            cache_dir: dir,
            max_size_bytes: 1024 * 1024,
            max_entries: 3,
            ttl_days: None,
            compression: Compression::None,
            eviction: CacheEviction::SizeBased,
        };
        let cache = BinaryCache::new(config);

        cache.store(make_key(1), make_binary(100)); // small
        cache.store(make_key(2), make_binary(500)); // largest
        cache.store(make_key(3), make_binary(200)); // medium

        // Inserting 4th should evict the largest (key 2).
        cache.store(make_key(4), make_binary(50));
        assert!(cache.lookup(&make_key(2)).is_none()); // evicted
        assert!(cache.lookup(&make_key(1)).is_some());
        assert!(cache.lookup(&make_key(3)).is_some());
    }

    // -----------------------------------------------------------------------
    // Eviction — size limit
    // -----------------------------------------------------------------------

    #[test]
    fn eviction_by_total_size() {
        // max_size_bytes = 200, so 3 × 100 byte entries won't fit.
        let cache = temp_cache(100, 200);
        cache.store(make_key(1), make_binary(100));
        cache.store(make_key(2), make_binary(100));
        assert_eq!(cache.len(), 2);

        // This should trigger size-based eviction.
        cache.store(make_key(3), make_binary(100));
        assert!(cache.len() <= 2);
        assert!(cache.stats().evictions >= 1);
    }

    // -----------------------------------------------------------------------
    // TTL expiration
    // -----------------------------------------------------------------------

    #[test]
    fn ttl_expired_entry_not_returned() {
        let dir = std::env::temp_dir().join("bitnet-rs-test-ttl-expired");
        let _ = std::fs::remove_dir_all(&dir);
        let config = CacheConfig {
            cache_dir: dir,
            max_size_bytes: 1024 * 1024,
            max_entries: 64,
            ttl_days: Some(0), // 0 days → immediately expired
            compression: Compression::None,
            eviction: CacheEviction::Lru,
        };
        let cache = BinaryCache::new(config);
        let key = make_key(1);
        cache.store(key.clone(), make_binary(64));

        // Entry is immediately expired (TTL = 0 days).
        assert!(cache.lookup(&key).is_none());
    }

    #[test]
    fn ttl_fresh_entry_returned() {
        let cache = temp_cache_with_ttl(365); // 1 year
        let key = make_key(1);
        cache.store(key.clone(), make_binary(64));
        assert!(cache.lookup(&key).is_some());
    }

    // -----------------------------------------------------------------------
    // Cache corruption recovery
    // -----------------------------------------------------------------------

    #[test]
    fn disk_corruption_returns_none() {
        let dir = std::env::temp_dir().join("bitnet-rs-test-corrupt");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();

        let config = CacheConfig {
            cache_dir: dir.clone(),
            max_size_bytes: 1024 * 1024,
            max_entries: 64,
            ttl_days: None,
            compression: Compression::None,
            eviction: CacheEviction::Lru,
        };
        let cache = BinaryCache::new(config);
        let key = make_key(99);

        // Write garbage to the expected file path.
        let path = dir.join(key.filename());
        std::fs::write(path, b"not a valid binary cache entry").unwrap();

        // Lookup should handle corruption gracefully.
        assert!(cache.lookup(&key).is_none());
    }

    #[test]
    fn disk_corruption_during_load_entries() {
        let dir = std::env::temp_dir().join("bitnet-rs-test-corrupt-load");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();

        // Write a corrupt .bin file.
        std::fs::write(dir.join("deadbeef.bin"), b"garbage").unwrap();

        let config = CacheConfig {
            cache_dir: dir,
            max_size_bytes: 1024 * 1024,
            max_entries: 64,
            ttl_days: None,
            compression: Compression::None,
            eviction: CacheEviction::Lru,
        };
        let cache = BinaryCache::new(config);

        // Should not fail, just skip corrupt entries.
        let loaded = cache.load_disk_entries().unwrap();
        assert_eq!(loaded, 0);
    }

    // -----------------------------------------------------------------------
    // Disk persistence roundtrip
    // -----------------------------------------------------------------------

    #[test]
    fn disk_store_and_reload() {
        let dir = std::env::temp_dir().join("bitnet-rs-test-disk-rt");
        let _ = std::fs::remove_dir_all(&dir);

        let config = CacheConfig {
            cache_dir: dir.clone(),
            max_size_bytes: 1024 * 1024,
            max_entries: 64,
            ttl_days: None,
            compression: Compression::None,
            eviction: CacheEviction::Lru,
        };

        let key = make_key(7);
        let binary_data = vec![1, 2, 3, 4, 5];

        // Store in one cache instance.
        {
            let cache = BinaryCache::new(config.clone());
            let binary = ProgramBinary::new(
                binary_data.clone(),
                7,
                test_fingerprint(),
                "-cl-mad-enable",
                Duration::from_millis(50),
            );
            cache.store(key.clone(), binary);
        }

        // Load from a fresh instance (disk only).
        {
            let cache = BinaryCache::new(config);
            // Clear memory so we test disk path.
            cache.clear();
            let entry = cache.lookup(&key).unwrap();
            assert_eq!(entry.binary.data, binary_data);
        }
    }

    // -----------------------------------------------------------------------
    // Concurrent access safety
    // -----------------------------------------------------------------------

    #[test]
    fn concurrent_readers() {
        let cache = Arc::new(temp_cache(64, 1024 * 1024));
        let key = make_key(1);
        cache.store(key.clone(), make_binary(128));

        let mut handles = Vec::new();
        for _ in 0..8 {
            let c = Arc::clone(&cache);
            let k = key.clone();
            handles.push(thread::spawn(move || {
                for _ in 0..100 {
                    let _ = c.lookup(&k);
                }
            }));
        }
        for h in handles {
            h.join().unwrap();
        }

        let s = cache.stats();
        // 1 initial miss + 800 reads, but some go through disk path.
        assert!(s.hits >= 800, "hits={}", s.hits);
    }

    #[test]
    fn concurrent_writers_and_readers() {
        let cache = Arc::new(temp_cache(64, 1024 * 1024));
        let mut handles = Vec::new();

        // Writers.
        for i in 0..4u64 {
            let c = Arc::clone(&cache);
            handles.push(thread::spawn(move || {
                for j in 0..50 {
                    let key = make_key(i * 1000 + j);
                    c.store(key, make_binary(32));
                }
            }));
        }

        // Readers.
        for i in 0..4u64 {
            let c = Arc::clone(&cache);
            handles.push(thread::spawn(move || {
                for j in 0..50 {
                    let key = make_key(i * 1000 + j);
                    let _ = c.lookup(&key);
                }
            }));
        }

        for h in handles {
            h.join().unwrap();
        }

        // Just verify no panic / deadlock.
        assert!(cache.len() <= 64);
    }

    // -----------------------------------------------------------------------
    // Stats accuracy
    // -----------------------------------------------------------------------

    #[test]
    fn stats_initial_zero() {
        let cache = temp_cache(64, 1024 * 1024);
        let s = cache.stats();
        assert_eq!(s.hits, 0);
        assert_eq!(s.misses, 0);
        assert_eq!(s.evictions, 0);
        assert_eq!(s.total_size, 0);
        assert_eq!(s.entry_count, 0);
    }

    #[test]
    fn stats_hit_rate_empty() {
        let s = CacheStats::default();
        assert_eq!(s.hit_rate(), 0.0);
    }

    #[test]
    fn stats_hit_rate_all_hits() {
        let cache = temp_cache(64, 1024 * 1024);
        cache.store(make_key(1), make_binary(32));
        cache.lookup(&make_key(1));
        cache.lookup(&make_key(1));
        let s = cache.stats();
        assert_eq!(s.hits, 2);
        assert_eq!(s.misses, 0);
        assert!((s.hit_rate() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn stats_hit_rate_mixed() {
        let cache = temp_cache(64, 1024 * 1024);
        cache.lookup(&make_key(1)); // miss
        cache.store(make_key(1), make_binary(32));
        cache.lookup(&make_key(1)); // hit
        let s = cache.stats();
        assert_eq!(s.hits, 1);
        assert_eq!(s.misses, 1);
        assert!((s.hit_rate() - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn stats_total_size_accurate() {
        let cache = temp_cache(64, 1024 * 1024);
        cache.store(make_key(1), make_binary(100));
        cache.store(make_key(2), make_binary(200));
        assert_eq!(cache.stats().total_size, 300);
    }

    #[test]
    fn stats_total_size_after_invalidate() {
        let cache = temp_cache(64, 1024 * 1024);
        cache.store(make_key(1), make_binary(100));
        cache.store(make_key(2), make_binary(200));
        cache.invalidate(&make_key(1));
        assert_eq!(cache.stats().total_size, 200);
    }

    #[test]
    fn stats_time_saved_accumulates() {
        let cache = temp_cache(64, 1024 * 1024);
        let binary =
            ProgramBinary::new(vec![0; 32], 1, test_fingerprint(), "", Duration::from_millis(50));
        let key = make_key(1);
        cache.store(key.clone(), binary);

        cache.lookup(&key); // 50ms saved
        cache.lookup(&key); // 50ms saved
        let s = cache.stats();
        assert_eq!(s.total_time_saved, Duration::from_millis(100));
    }

    #[test]
    fn stats_avg_load_time_saved() {
        let mut s = CacheStats::default();
        s.hits = 4;
        s.total_time_saved = Duration::from_millis(200);
        assert_eq!(s.avg_load_time_saved(), Duration::from_millis(50));
    }

    #[test]
    fn stats_avg_load_time_saved_zero_hits() {
        let s = CacheStats::default();
        assert_eq!(s.avg_load_time_saved(), Duration::ZERO);
    }

    #[test]
    fn stats_display() {
        let s = CacheStats { hits: 10, misses: 5, ..Default::default() };
        let disp = format!("{s}");
        assert!(disp.contains("hits=10"));
        assert!(disp.contains("misses=5"));
    }

    // -----------------------------------------------------------------------
    // Warmup
    // -----------------------------------------------------------------------

    #[test]
    fn warmup_none_is_noop() {
        let cache = temp_cache(64, 1024 * 1024);
        let result = cache.warmup(&WarmupStrategy::None, &test_fingerprint(), |_, _| {
            panic!("should not be called");
        });
        assert_eq!(result.loaded, 0);
        assert_eq!(result.compiled, 0);
    }

    #[test]
    fn warmup_explicit_compiles() {
        let cache = temp_cache(64, 1024 * 1024);
        let specs = vec![
            WarmupSpec::new("kernel void a(){}", ""),
            WarmupSpec::new("kernel void b(){}", "-O2"),
        ];
        let fp = test_fingerprint();
        let result = cache.warmup(&WarmupStrategy::Explicit(specs), &fp, |src, opts| {
            Ok(ProgramBinary::new(
                src.as_bytes().to_vec(),
                hash_bytes(src.as_bytes()),
                fp.clone(),
                opts,
                Duration::from_millis(10),
            ))
        });
        assert_eq!(result.compiled, 2);
        assert_eq!(result.failed, 0);
        assert_eq!(cache.len(), 2);
    }

    #[test]
    fn warmup_explicit_loads_cached() {
        let cache = temp_cache(64, 1024 * 1024);
        let fp = test_fingerprint();
        let src = "kernel void cached(){}";

        // Pre-populate.
        let key = CacheKey::from_source(src, &fp, "");
        cache.store(
            key,
            ProgramBinary::new(
                vec![0xCC; 16],
                hash_bytes(src.as_bytes()),
                fp.clone(),
                "",
                Duration::from_millis(5),
            ),
        );

        let specs = vec![WarmupSpec::new(src, "")];
        let result = cache.warmup(&WarmupStrategy::Explicit(specs), &fp, |_, _| {
            panic!("should not compile");
        });
        assert_eq!(result.loaded, 1);
        assert_eq!(result.compiled, 0);
    }

    #[test]
    fn warmup_handles_compile_failure() {
        let cache = temp_cache(64, 1024 * 1024);
        let specs = vec![WarmupSpec::new("bad kernel", "")];
        let result = cache.warmup(&WarmupStrategy::Explicit(specs), &test_fingerprint(), |_, _| {
            Err("compile error".into())
        });
        assert_eq!(result.failed, 1);
        assert_eq!(result.compiled, 0);
    }

    // -----------------------------------------------------------------------
    // Edge cases
    // -----------------------------------------------------------------------

    #[test]
    fn store_empty_binary() {
        let cache = temp_cache(64, 1024 * 1024);
        let key = make_key(1);
        let binary = ProgramBinary::new(vec![], 1, test_fingerprint(), "", Duration::ZERO);
        cache.store(key.clone(), binary);
        let entry = cache.lookup(&key).unwrap();
        assert!(entry.binary.data.is_empty());
    }

    #[test]
    fn store_large_binary() {
        let cache = temp_cache(64, 64 * 1024 * 1024);
        let key = make_key(1);
        cache.store(key.clone(), make_binary(1024 * 1024)); // 1 MiB
        let entry = cache.lookup(&key).unwrap();
        assert_eq!(entry.binary.data.len(), 1024 * 1024);
    }

    #[test]
    fn cache_full_single_entry() {
        // max_entries = 1
        let cache = temp_cache(1, 1024 * 1024);
        cache.store(make_key(1), make_binary(32));
        cache.store(make_key(2), make_binary(32));
        assert_eq!(cache.len(), 1);
        assert!(cache.lookup(&make_key(1)).is_none());
        assert!(cache.lookup(&make_key(2)).is_some());
    }

    #[test]
    fn store_overwrite_same_key() {
        let cache = temp_cache(64, 1024 * 1024);
        let key = make_key(1);
        cache.store(key.clone(), make_binary(32));
        cache.store(key.clone(), make_binary(64));
        let entry = cache.lookup(&key).unwrap();
        assert_eq!(entry.binary.data.len(), 64);
    }

    #[test]
    fn invalidate_nonexistent_key() {
        let cache = temp_cache(64, 1024 * 1024);
        cache.invalidate(&make_key(999)); // should not panic
    }

    #[test]
    fn lookup_on_empty_cache() {
        let cache = temp_cache(64, 1024 * 1024);
        assert!(cache.lookup(&make_key(1)).is_none());
        assert_eq!(cache.stats().misses, 1);
    }

    // -----------------------------------------------------------------------
    // CpuReferenceCache
    // -----------------------------------------------------------------------

    #[test]
    fn cpu_ref_cache_basic() {
        let cache = CpuReferenceCache::new();
        let key = make_key(1);
        assert!(cache.get(&key).is_none());
        assert!(cache.is_empty());

        cache.put(key.clone(), make_binary(64));
        assert_eq!(cache.len(), 1);
        assert!(!cache.is_empty());

        let b = cache.get(&key).unwrap();
        assert_eq!(b.data.len(), 64);
    }

    #[test]
    fn cpu_ref_cache_remove() {
        let cache = CpuReferenceCache::new();
        let key = make_key(1);
        cache.put(key.clone(), make_binary(32));
        assert!(cache.remove(&key));
        assert!(!cache.remove(&key));
        assert!(cache.is_empty());
    }

    #[test]
    fn cpu_ref_cache_clear() {
        let cache = CpuReferenceCache::new();
        for i in 0..5 {
            cache.put(make_key(i), make_binary(16));
        }
        assert_eq!(cache.len(), 5);
        cache.clear();
        assert!(cache.is_empty());
    }

    #[test]
    fn cpu_ref_cache_default() {
        let cache = CpuReferenceCache::default();
        assert!(cache.is_empty());
    }

    // -----------------------------------------------------------------------
    // hash_bytes utility
    // -----------------------------------------------------------------------

    #[test]
    fn hash_bytes_deterministic() {
        let h1 = hash_bytes(b"hello");
        let h2 = hash_bytes(b"hello");
        assert_eq!(h1, h2);
    }

    #[test]
    fn hash_bytes_different_input() {
        assert_ne!(hash_bytes(b"hello"), hash_bytes(b"world"));
    }

    #[test]
    fn hash_bytes_empty() {
        // Should not panic.
        let _ = hash_bytes(b"");
    }

    // -----------------------------------------------------------------------
    // CacheEviction enum coverage
    // -----------------------------------------------------------------------

    #[test]
    fn eviction_variants_debug() {
        assert_eq!(format!("{:?}", CacheEviction::Lru), "Lru");
        assert_eq!(format!("{:?}", CacheEviction::Lfu), "Lfu");
        assert_eq!(format!("{:?}", CacheEviction::SizeBased), "SizeBased");
    }

    #[test]
    fn eviction_variants_eq() {
        assert_eq!(CacheEviction::Lru, CacheEviction::Lru);
        assert_ne!(CacheEviction::Lru, CacheEviction::Lfu);
    }

    // -----------------------------------------------------------------------
    // WarmupStrategy coverage
    // -----------------------------------------------------------------------

    #[test]
    fn warmup_spec_new() {
        let spec = WarmupSpec::new("src", "-O2");
        assert_eq!(spec.source, "src");
        assert_eq!(spec.compile_options, "-O2");
    }

    #[test]
    fn warmup_result_default() {
        let r = WarmupResult::default();
        assert_eq!(r.loaded, 0);
        assert_eq!(r.compiled, 0);
        assert_eq!(r.failed, 0);
    }

    // -----------------------------------------------------------------------
    // Property-like: store then load = identity
    // -----------------------------------------------------------------------

    #[test]
    fn property_store_load_identity_small() {
        let cache = temp_cache(256, 16 * 1024 * 1024);
        let fp = test_fingerprint();
        for i in 0u64..50 {
            let data: Vec<u8> = (0..((i + 1) * 7)).map(|b| (b % 256) as u8).collect();
            let binary = ProgramBinary::new(
                data.clone(),
                i,
                fp.clone(),
                format!("-DX={i}"),
                Duration::from_micros(i * 10),
            );
            let key = CacheKey::new(i, fp.hash_u64(), format!("-DX={i}"));
            cache.store(key.clone(), binary);

            let entry = cache.lookup(&key).unwrap();
            assert_eq!(entry.binary.data, data, "mismatch at i={i}");
            assert_eq!(entry.binary.source_hash, i);
            assert_eq!(entry.binary.device_fingerprint, fp);
        }
    }

    #[test]
    fn property_serializer_roundtrip_various_sizes() {
        let fp = test_fingerprint();
        for size in [0, 1, 7, 128, 1024, 8192] {
            let binary = ProgramBinary::new(
                vec![0xDE; size],
                size as u64,
                fp.clone(),
                "-cl-fast-relaxed-math",
                Duration::from_millis(size as u64),
            );
            let entry = CacheEntry::new(binary);
            let data = EntrySerializer::serialize(&entry);
            let restored = EntrySerializer::deserialize(&data).unwrap();
            assert_eq!(restored.binary.data.len(), size, "size={size}");
            assert_eq!(restored.binary.source_hash, size as u64);
        }
    }

    // -----------------------------------------------------------------------
    // load_disk_entries
    // -----------------------------------------------------------------------

    #[test]
    fn load_disk_entries_empty_dir() {
        let dir = std::env::temp_dir().join("bitnet-rs-test-load-empty");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let config = CacheConfig {
            cache_dir: dir,
            max_size_bytes: 1024 * 1024,
            max_entries: 64,
            ttl_days: None,
            compression: Compression::None,
            eviction: CacheEviction::Lru,
        };
        let cache = BinaryCache::new(config);
        assert_eq!(cache.load_disk_entries().unwrap(), 0);
    }

    #[test]
    fn load_disk_entries_nonexistent_dir() {
        let dir = std::env::temp_dir().join("bitnet-rs-test-load-nodir-progcache");
        let _ = std::fs::remove_dir_all(&dir);
        let config = CacheConfig {
            cache_dir: dir,
            max_size_bytes: 1024 * 1024,
            max_entries: 64,
            ttl_days: None,
            compression: Compression::None,
            eviction: CacheEviction::Lru,
        };
        let cache = BinaryCache::new(config);
        assert_eq!(cache.load_disk_entries().unwrap(), 0);
    }

    #[test]
    fn load_disk_entries_skips_non_bin() {
        let dir = std::env::temp_dir().join("bitnet-rs-test-load-nonbin");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("readme.txt"), b"not a cache file").unwrap();
        let config = CacheConfig {
            cache_dir: dir,
            max_size_bytes: 1024 * 1024,
            max_entries: 64,
            ttl_days: None,
            compression: Compression::None,
            eviction: CacheEviction::Lru,
        };
        let cache = BinaryCache::new(config);
        assert_eq!(cache.load_disk_entries().unwrap(), 0);
    }

    // -----------------------------------------------------------------------
    // BinaryCache Debug
    // -----------------------------------------------------------------------

    #[test]
    fn binary_cache_debug() {
        let cache = temp_cache(8, 1024);
        let dbg = format!("{cache:?}");
        assert!(dbg.contains("BinaryCache"));
    }
}
