//! SPIR-V pre-compilation and caching pipeline for OpenCL kernels.
//!
//! Provides filesystem-backed caching of compiled OpenCL kernel binaries
//! (SPIR-V or device-specific) for Intel Arc A770 and other Intel GPUs.
//! Cache keys include a SHA-256 hash of the kernel source, compile options,
//! and a device fingerprint so binaries are only reused when the compilation
//! environment matches exactly.

use std::collections::HashMap;
use std::fmt;
use std::fs;
use std::io;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::SystemTime;

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Errors that can occur during cache operations.
#[derive(Debug)]
pub enum CacheError {
    /// An I/O error occurred (e.g. creating directories, reading/writing files).
    Io(io::Error),
    /// The cache directory could not be created or is not writable.
    InvalidCacheDir(String),
    /// A serialization or deserialization error.
    Serialization(String),
}

impl fmt::Display for CacheError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CacheError::Io(e) => write!(f, "cache I/O error: {e}"),
            CacheError::InvalidCacheDir(msg) => write!(f, "invalid cache directory: {msg}"),
            CacheError::Serialization(msg) => write!(f, "serialization error: {msg}"),
        }
    }
}

impl std::error::Error for CacheError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            CacheError::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<io::Error> for CacheError {
    fn from(e: io::Error) -> Self {
        CacheError::Io(e)
    }
}

/// Errors from the compilation pipeline.
#[derive(Debug)]
pub enum CompileError {
    /// Source code was empty or invalid.
    EmptySource,
    /// The underlying OpenCL compilation failed.
    CompilationFailed(String),
    /// A cache error during cache-through compilation.
    Cache(CacheError),
}

impl fmt::Display for CompileError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CompileError::EmptySource => write!(f, "kernel source is empty"),
            CompileError::CompilationFailed(msg) => write!(f, "compilation failed: {msg}"),
            CompileError::Cache(e) => write!(f, "cache error during compilation: {e}"),
        }
    }
}

impl std::error::Error for CompileError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            CompileError::Cache(e) => Some(e),
            _ => None,
        }
    }
}

impl From<CacheError> for CompileError {
    fn from(e: CacheError) -> Self {
        CompileError::Cache(e)
    }
}

// ---------------------------------------------------------------------------
// SpirvCacheConfig
// ---------------------------------------------------------------------------

/// Configuration for the SPIR-V compilation cache.
#[derive(Debug, Clone)]
pub struct SpirvCacheConfig {
    /// Directory where cached binaries are stored on disk.
    pub cache_dir: PathBuf,
    /// Maximum total cache size in megabytes (default: 256 MiB).
    pub max_cache_size_mb: usize,
    /// Device fingerprint string: `"{vendor_id}:{device_id}:{driver_version}"`.
    pub device_fingerprint: String,
}

impl Default for SpirvCacheConfig {
    fn default() -> Self {
        let cache_dir = dirs_fallback_cache_dir();
        Self {
            cache_dir,
            max_cache_size_mb: 256,
            device_fingerprint: String::from("unknown:unknown:unknown"),
        }
    }
}

/// Best-effort default cache directory (`~/.cache/bitnet/spirv/`).
fn dirs_fallback_cache_dir() -> PathBuf {
    #[cfg(target_os = "windows")]
    {
        if let Ok(local) = std::env::var("LOCALAPPDATA") {
            return PathBuf::from(local).join("bitnet").join("spirv");
        }
    }
    if let Ok(home) = std::env::var("HOME") {
        return PathBuf::from(home).join(".cache").join("bitnet").join("spirv");
    }
    PathBuf::from(".cache").join("bitnet").join("spirv")
}

// ---------------------------------------------------------------------------
// CacheKey
// ---------------------------------------------------------------------------

/// A composite key that uniquely identifies a compiled kernel binary.
///
/// Two compilations produce the same `CacheKey` if and only if the kernel
/// source, compile options, **and** target device are identical.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CacheKey {
    /// SHA-256 digest of the kernel source code.
    pub kernel_source_hash: [u8; 32],
    /// Compile options passed to the OpenCL compiler.
    pub compile_options: String,
    /// Device fingerprint (`vendor_id:device_id:driver_version`).
    pub device_fingerprint: String,
}

impl CacheKey {
    /// Create a new cache key from raw components.
    pub fn new(
        kernel_source_hash: [u8; 32],
        compile_options: impl Into<String>,
        device_fingerprint: impl Into<String>,
    ) -> Self {
        Self {
            kernel_source_hash,
            compile_options: compile_options.into(),
            device_fingerprint: device_fingerprint.into(),
        }
    }

    /// Derive a cache key by hashing `source` with SHA-256.
    pub fn from_source(
        source: &str,
        compile_options: impl Into<String>,
        device_fingerprint: impl Into<String>,
    ) -> Self {
        Self::new(sha256_hash(source.as_bytes()), compile_options, device_fingerprint)
    }

    /// Hex-encoded string suitable for use as a filename.
    pub fn hex(&self) -> String {
        let mut s = String::with_capacity(32 * 2);
        for byte in &self.kernel_source_hash {
            s.push_str(&format!("{byte:02x}"));
        }
        s
    }

    /// Filename for on-disk storage: `"{hex}_{options_hash}.bin"`.
    fn cache_filename(&self) -> String {
        // Include a short hash of compile_options + fingerprint to avoid collisions.
        let combo = format!("{}|{}", self.compile_options, self.device_fingerprint);
        let extra = sha256_hash(combo.as_bytes());
        let short: String = extra[..4].iter().map(|b| format!("{b:02x}")).collect();
        format!("{}_{short}.bin", self.hex())
    }
}

impl fmt::Display for CacheKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "CacheKey(hash={}, opts={:?}, dev={:?})",
            self.hex(),
            self.compile_options,
            self.device_fingerprint,
        )
    }
}

// ---------------------------------------------------------------------------
// CacheStats
// ---------------------------------------------------------------------------

/// Runtime statistics for the SPIR-V cache.
#[derive(Debug, Default)]
pub struct CacheStats {
    /// Number of cache hits.
    pub hits: AtomicU64,
    /// Number of cache misses.
    pub misses: AtomicU64,
    /// Total size of cached binaries in bytes.
    pub total_size_bytes: AtomicU64,
    /// Number of entries currently in the cache.
    pub entry_count: AtomicU64,
}

impl CacheStats {
    /// Cache hit rate in `[0.0, 1.0]`. Returns `0.0` when there have been no lookups.
    pub fn hit_rate(&self) -> f64 {
        let h = self.hits.load(Ordering::Relaxed);
        let m = self.misses.load(Ordering::Relaxed);
        let total = h + m;
        if total == 0 {
            return 0.0;
        }
        h as f64 / total as f64
    }

    /// Create a snapshot with plain `u64` values.
    pub fn snapshot(&self) -> CacheStatsSnapshot {
        CacheStatsSnapshot {
            hits: self.hits.load(Ordering::Relaxed),
            misses: self.misses.load(Ordering::Relaxed),
            total_size_bytes: self.total_size_bytes.load(Ordering::Relaxed),
            entry_count: self.entry_count.load(Ordering::Relaxed) as usize,
        }
    }
}

/// A plain-data snapshot of [`CacheStats`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CacheStatsSnapshot {
    pub hits: u64,
    pub misses: u64,
    pub total_size_bytes: u64,
    pub entry_count: usize,
}

impl CacheStatsSnapshot {
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            return 0.0;
        }
        self.hits as f64 / total as f64
    }
}

impl fmt::Display for CacheStatsSnapshot {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "SpirvCacheStats(hits={}, misses={}, entries={}, size={}B, hit_rate={:.1}%)",
            self.hits,
            self.misses,
            self.entry_count,
            self.total_size_bytes,
            self.hit_rate() * 100.0,
        )
    }
}

// ---------------------------------------------------------------------------
// SpirvCache
// ---------------------------------------------------------------------------

/// Metadata for an in-memory cache entry (the binary lives on disk).
#[derive(Debug, Clone)]
struct CacheEntryMeta {
    /// Path to the binary on disk.
    path: PathBuf,
    /// Size of the binary in bytes.
    size: u64,
    /// Last access time (for LRU eviction).
    last_accessed: SystemTime,
}

/// Filesystem-backed SPIR-V compilation cache with LRU eviction.
pub struct SpirvCache {
    config: SpirvCacheConfig,
    /// In-memory index: CacheKey → metadata (binary read from disk on lookup).
    index: HashMap<CacheKey, CacheEntryMeta>,
    /// LRU order (front = oldest).
    lru_order: Vec<CacheKey>,
    stats: CacheStats,
}

impl SpirvCache {
    /// Create a new cache with the given configuration.
    ///
    /// The cache directory is created on first `store` if it does not exist.
    pub fn new(config: SpirvCacheConfig) -> Self {
        let mut cache = Self {
            config,
            index: HashMap::new(),
            lru_order: Vec::new(),
            stats: CacheStats::default(),
        };
        // Load existing entries from disk.
        cache.load_index();
        cache
    }

    /// Look up a cached binary by key. Returns `None` on cache miss.
    pub fn lookup(&mut self, key: &CacheKey) -> Option<Vec<u8>> {
        if let Some(meta) = self.index.get_mut(key) {
            meta.last_accessed = SystemTime::now();
            // Move to back of LRU list.
            self.lru_order.retain(|k| k != key);
            self.lru_order.push(key.clone());
            // Read from disk.
            match fs::read(&meta.path) {
                Ok(data) => {
                    self.stats.hits.fetch_add(1, Ordering::Relaxed);
                    Some(data)
                }
                Err(_) => {
                    // Stale entry — remove from index.
                    let meta = self.index.remove(key).unwrap();
                    self.stats
                        .total_size_bytes
                        .fetch_sub(meta.size, Ordering::Relaxed);
                    self.stats.entry_count.fetch_sub(1, Ordering::Relaxed);
                    self.lru_order.retain(|k| k != key);
                    self.stats.misses.fetch_add(1, Ordering::Relaxed);
                    None
                }
            }
        } else {
            self.stats.misses.fetch_add(1, Ordering::Relaxed);
            None
        }
    }

    /// Store a compiled binary in the cache.
    pub fn store(&mut self, key: &CacheKey, binary: &[u8]) -> Result<(), CacheError> {
        let max_bytes = (self.config.max_cache_size_mb as u64) * 1024 * 1024;

        // Evict until there is room.
        while self.current_size() + binary.len() as u64 > max_bytes && !self.lru_order.is_empty() {
            self.evict_lru();
        }

        // Ensure cache dir exists.
        fs::create_dir_all(&self.config.cache_dir).map_err(|e| {
            CacheError::InvalidCacheDir(format!(
                "cannot create {}: {e}",
                self.config.cache_dir.display()
            ))
        })?;

        let filename = key.cache_filename();
        let path = self.config.cache_dir.join(&filename);
        fs::write(&path, binary)?;

        let size = binary.len() as u64;

        // Update index.
        if let Some(old) = self.index.insert(
            key.clone(),
            CacheEntryMeta {
                path,
                size,
                last_accessed: SystemTime::now(),
            },
        ) {
            self.stats
                .total_size_bytes
                .fetch_sub(old.size, Ordering::Relaxed);
        } else {
            self.stats.entry_count.fetch_add(1, Ordering::Relaxed);
        }

        self.stats
            .total_size_bytes
            .fetch_add(size, Ordering::Relaxed);

        // Update LRU.
        self.lru_order.retain(|k| k != key);
        self.lru_order.push(key.clone());

        Ok(())
    }

    /// Evict the least-recently-used entry.
    pub fn evict_lru(&mut self) {
        if let Some(oldest_key) = self.lru_order.first().cloned() {
            self.remove_entry(&oldest_key);
        }
    }

    /// Invalidate (remove) a specific entry.
    pub fn invalidate(&mut self, key: &CacheKey) {
        self.remove_entry(key);
    }

    /// Remove all entries and delete the cache directory contents.
    pub fn clear(&mut self) {
        let keys: Vec<CacheKey> = self.index.keys().cloned().collect();
        for key in keys {
            self.remove_entry(&key);
        }
    }

    /// Return a snapshot of cache statistics.
    pub fn stats(&self) -> CacheStatsSnapshot {
        self.stats.snapshot()
    }

    // -- internal helpers ---------------------------------------------------

    fn current_size(&self) -> u64 {
        self.stats.total_size_bytes.load(Ordering::Relaxed)
    }

    fn remove_entry(&mut self, key: &CacheKey) {
        if let Some(meta) = self.index.remove(key) {
            let _ = fs::remove_file(&meta.path);
            self.stats
                .total_size_bytes
                .fetch_sub(meta.size, Ordering::Relaxed);
            self.stats.entry_count.fetch_sub(1, Ordering::Relaxed);
        }
        self.lru_order.retain(|k| k != key);
    }

    /// Scan the cache directory and rebuild the in-memory index.
    fn load_index(&mut self) {
        let dir = &self.config.cache_dir;
        if !dir.is_dir() {
            return;
        }
        let Ok(entries) = fs::read_dir(dir) else {
            return;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) != Some("bin") {
                continue;
            }
            let Ok(metadata) = entry.metadata() else {
                continue;
            };
            let size = metadata.len();
            let last_accessed = metadata.modified().unwrap_or(SystemTime::UNIX_EPOCH);

            // We cannot reconstruct the full CacheKey from the filename alone,
            // so we use the filename as a synthetic key. Real lookups will
            // populate proper keys via `store`.
            let fname = path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("unknown")
                .to_string();
            let synthetic_key = CacheKey {
                kernel_source_hash: filename_to_synthetic_hash(&fname),
                compile_options: String::new(),
                device_fingerprint: self.config.device_fingerprint.clone(),
            };

            self.index.insert(
                synthetic_key.clone(),
                CacheEntryMeta {
                    path,
                    size,
                    last_accessed,
                },
            );
            self.lru_order.push(synthetic_key);
            self.stats
                .total_size_bytes
                .fetch_add(size, Ordering::Relaxed);
            self.stats.entry_count.fetch_add(1, Ordering::Relaxed);
        }
    }
}

// ---------------------------------------------------------------------------
// CompilationPipeline
// ---------------------------------------------------------------------------

/// Stub compilation pipeline for OpenCL → SPIR-V / device binary.
///
/// In a real implementation, `compile_cl_to_binary` would invoke
/// `clCreateProgramWithSource` + `clBuildProgram` + `clGetProgramInfo(CL_PROGRAM_BINARIES)`.
pub struct CompilationPipeline;

impl CompilationPipeline {
    /// Compile OpenCL source to a device binary.
    ///
    /// **Stub**: returns a deterministic pseudo-binary derived from the source
    /// and options so that tests can exercise the caching layer without a real
    /// OpenCL runtime.
    pub fn compile_cl_to_binary(source: &str, options: &str) -> Result<Vec<u8>, CompileError> {
        if source.is_empty() {
            return Err(CompileError::EmptySource);
        }
        // Deterministic stub: SHA-256 of source+options padded to 256 bytes.
        let digest = sha256_hash(format!("{source}{options}").as_bytes());
        let mut binary = Vec::with_capacity(256);
        // SPIR-V magic number (little-endian)
        binary.extend_from_slice(&[0x03, 0x02, 0x23, 0x07]);
        binary.extend_from_slice(&digest);
        binary.resize(256, 0xAB);
        Ok(binary)
    }

    /// Compile with cache-through: return cached binary on hit, else compile
    /// and store the result.
    pub fn compile_with_cache(
        source: &str,
        options: &str,
        cache: &mut SpirvCache,
    ) -> Result<Vec<u8>, CompileError> {
        let key = CacheKey::from_source(source, options, &cache.config.device_fingerprint);

        if let Some(binary) = cache.lookup(&key) {
            return Ok(binary);
        }

        let binary = Self::compile_cl_to_binary(source, options)?;
        cache.store(&key, &binary)?;
        Ok(binary)
    }
}

// ---------------------------------------------------------------------------
// SHA-256 (minimal, dependency-free)
// ---------------------------------------------------------------------------

/// Minimal SHA-256 implementation (no external crate dependency).
///
/// Uses the standard NIST algorithm. This keeps the kernel crate free of
/// extra dependencies — production code in other crates already uses `sha2`.
fn sha256_hash(data: &[u8]) -> [u8; 32] {
    // Initial hash values (first 32 bits of fractional parts of square roots of first 8 primes)
    let mut h: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
        0x5be0cd19,
    ];

    // Round constants
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
        0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
        0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
        0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
        0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
        0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2,
    ];

    // Pre-processing: pad the message
    let bit_len = (data.len() as u64) * 8;
    let mut msg = data.to_vec();
    msg.push(0x80);
    while (msg.len() % 64) != 56 {
        msg.push(0x00);
    }
    msg.extend_from_slice(&bit_len.to_be_bytes());

    // Process each 512-bit (64-byte) block
    for chunk in msg.chunks_exact(64) {
        let mut w = [0u32; 64];
        for i in 0..16 {
            w[i] = u32::from_be_bytes([
                chunk[4 * i],
                chunk[4 * i + 1],
                chunk[4 * i + 2],
                chunk[4 * i + 3],
            ]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }

        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut hh] = h;

        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & g);
            let temp1 = hh
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(K[i])
                .wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let temp2 = s0.wrapping_add(maj);

            hh = g;
            g = f;
            f = e;
            e = d.wrapping_add(temp1);
            d = c;
            c = b;
            b = a;
            a = temp1.wrapping_add(temp2);
        }

        h[0] = h[0].wrapping_add(a);
        h[1] = h[1].wrapping_add(b);
        h[2] = h[2].wrapping_add(c);
        h[3] = h[3].wrapping_add(d);
        h[4] = h[4].wrapping_add(e);
        h[5] = h[5].wrapping_add(f);
        h[6] = h[6].wrapping_add(g);
        h[7] = h[7].wrapping_add(hh);
    }

    let mut digest = [0u8; 32];
    for (i, val) in h.iter().enumerate() {
        digest[i * 4..i * 4 + 4].copy_from_slice(&val.to_be_bytes());
    }
    digest
}

/// Convert a filename stem to a synthetic 32-byte hash for index loading.
fn filename_to_synthetic_hash(name: &str) -> [u8; 32] {
    sha256_hash(name.as_bytes())
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;
    use std::thread;
    use tempfile::TempDir;

    fn test_config(dir: &Path) -> SpirvCacheConfig {
        SpirvCacheConfig {
            cache_dir: dir.to_path_buf(),
            max_cache_size_mb: 1, // 1 MiB for tests
            device_fingerprint: "8086:56a0:23.3.1".into(),
        }
    }

    // -- CacheKey tests -----------------------------------------------------

    #[test]
    fn cache_key_deterministic() {
        let k1 = CacheKey::from_source("kernel_a", "-cl-mad-enable", "dev1");
        let k2 = CacheKey::from_source("kernel_a", "-cl-mad-enable", "dev1");
        assert_eq!(k1, k2);
        assert_eq!(k1.hex(), k2.hex());
    }

    #[test]
    fn different_sources_produce_different_keys() {
        let k1 = CacheKey::from_source("kernel_a", "", "dev1");
        let k2 = CacheKey::from_source("kernel_b", "", "dev1");
        assert_ne!(k1, k2);
        assert_ne!(k1.hex(), k2.hex());
    }

    #[test]
    fn same_source_different_options_produce_different_keys() {
        let k1 = CacheKey::from_source("kernel_a", "-cl-mad-enable", "dev1");
        let k2 = CacheKey::from_source("kernel_a", "-cl-fast-relaxed-math", "dev1");
        assert_ne!(k1, k2);
    }

    #[test]
    fn same_source_different_device_produce_different_keys() {
        let k1 = CacheKey::from_source("kernel_a", "", "8086:56a0:23.3.1");
        let k2 = CacheKey::from_source("kernel_a", "", "8086:56a1:24.0.0");
        assert_ne!(k1, k2);
    }

    #[test]
    fn cache_key_display_contains_hex() {
        let k = CacheKey::from_source("hello", "", "d");
        let display = format!("{k}");
        assert!(display.contains(&k.hex()));
        assert!(display.starts_with("CacheKey("));
    }

    #[test]
    fn cache_key_hex_is_64_chars() {
        let k = CacheKey::from_source("test", "", "d");
        assert_eq!(k.hex().len(), 64); // 32 bytes × 2 hex chars
    }

    #[test]
    fn cache_key_filename_is_stable() {
        let k = CacheKey::from_source("src", "-O2", "dev");
        let f1 = k.cache_filename();
        let f2 = k.cache_filename();
        assert_eq!(f1, f2);
        assert!(f1.ends_with(".bin"));
    }

    // -- CacheStats tests ---------------------------------------------------

    #[test]
    fn hit_rate_zero_when_no_lookups() {
        let stats = CacheStats::default();
        assert_eq!(stats.hit_rate(), 0.0);
    }

    #[test]
    fn hit_rate_calculation() {
        let stats = CacheStats::default();
        stats.hits.store(3, Ordering::Relaxed);
        stats.misses.store(1, Ordering::Relaxed);
        assert!((stats.hit_rate() - 0.75).abs() < f64::EPSILON);
    }

    #[test]
    fn stats_snapshot_matches() {
        let stats = CacheStats::default();
        stats.hits.store(10, Ordering::Relaxed);
        stats.misses.store(5, Ordering::Relaxed);
        stats.total_size_bytes.store(4096, Ordering::Relaxed);
        stats.entry_count.store(2, Ordering::Relaxed);
        let snap = stats.snapshot();
        assert_eq!(snap.hits, 10);
        assert_eq!(snap.misses, 5);
        assert_eq!(snap.total_size_bytes, 4096);
        assert_eq!(snap.entry_count, 2);
        assert!((snap.hit_rate() - 10.0 / 15.0).abs() < 1e-10);
    }

    #[test]
    fn stats_snapshot_display() {
        let snap = CacheStatsSnapshot {
            hits: 7,
            misses: 3,
            total_size_bytes: 1024,
            entry_count: 2,
        };
        let s = format!("{snap}");
        assert!(s.contains("hits=7"));
        assert!(s.contains("misses=3"));
        assert!(s.contains("70.0%"));
    }

    // -- SpirvCache store/lookup tests --------------------------------------

    #[test]
    fn store_and_lookup_roundtrip() {
        let tmp = TempDir::new().unwrap();
        let mut cache = SpirvCache::new(test_config(tmp.path()));
        let key = CacheKey::from_source("kernel", "", "dev");
        let data = vec![0xDE, 0xAD, 0xBE, 0xEF];
        cache.store(&key, &data).unwrap();
        let got = cache.lookup(&key).unwrap();
        assert_eq!(got, data);
    }

    #[test]
    fn cache_miss_returns_none() {
        let tmp = TempDir::new().unwrap();
        let mut cache = SpirvCache::new(test_config(tmp.path()));
        let key = CacheKey::from_source("missing", "", "dev");
        assert!(cache.lookup(&key).is_none());
    }

    #[test]
    fn cache_stats_track_hit_and_miss() {
        let tmp = TempDir::new().unwrap();
        let mut cache = SpirvCache::new(test_config(tmp.path()));
        let key = CacheKey::from_source("k", "", "dev");

        cache.lookup(&key); // miss
        cache.store(&key, &[1, 2, 3]).unwrap();
        cache.lookup(&key); // hit

        let snap = cache.stats();
        assert_eq!(snap.hits, 1);
        assert_eq!(snap.misses, 1);
        assert!((snap.hit_rate() - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn cache_eviction_when_full() {
        let tmp = TempDir::new().unwrap();
        // Tiny cache: 1 MiB
        let mut cache = SpirvCache::new(test_config(tmp.path()));

        // Store entries until we exceed 1 MiB
        let big_data = vec![0u8; 512 * 1024]; // 512 KiB each
        let k1 = CacheKey::from_source("first", "", "dev");
        let k2 = CacheKey::from_source("second", "", "dev");
        let k3 = CacheKey::from_source("third", "", "dev");

        cache.store(&k1, &big_data).unwrap();
        cache.store(&k2, &big_data).unwrap();
        // Now at 1 MiB — storing k3 should evict k1 (LRU).
        cache.store(&k3, &big_data).unwrap();

        assert!(cache.lookup(&k1).is_none(), "k1 should have been evicted");
        assert!(cache.lookup(&k2).is_some(), "k2 should still be present");
        assert!(cache.lookup(&k3).is_some(), "k3 should still be present");
    }

    #[test]
    fn cache_invalidation() {
        let tmp = TempDir::new().unwrap();
        let mut cache = SpirvCache::new(test_config(tmp.path()));
        let key = CacheKey::from_source("k", "", "dev");
        cache.store(&key, &[1, 2]).unwrap();
        assert!(cache.lookup(&key).is_some());
        cache.invalidate(&key);
        assert!(cache.lookup(&key).is_none());
    }

    #[test]
    fn cache_clear_removes_all() {
        let tmp = TempDir::new().unwrap();
        let mut cache = SpirvCache::new(test_config(tmp.path()));
        for i in 0..5 {
            let key = CacheKey::from_source(&format!("k{i}"), "", "dev");
            cache.store(&key, &[i as u8; 4]).unwrap();
        }
        assert_eq!(cache.stats().entry_count, 5);
        cache.clear();
        assert_eq!(cache.stats().entry_count, 0);
        assert_eq!(cache.stats().total_size_bytes, 0);
    }

    #[test]
    fn filesystem_persistence() {
        let tmp = TempDir::new().unwrap();
        let key = CacheKey::from_source("persist", "", "dev");
        let data = vec![0xCA, 0xFE];

        // Store in one cache instance.
        {
            let mut cache = SpirvCache::new(test_config(tmp.path()));
            cache.store(&key, &data).unwrap();
        }

        // Verify files exist on disk.
        let entries: Vec<_> = fs::read_dir(tmp.path()).unwrap().collect();
        assert!(!entries.is_empty(), "cache dir should contain files");

        // A fresh cache should load the index from disk.
        let cache2 = SpirvCache::new(test_config(tmp.path()));
        assert!(cache2.stats().entry_count > 0, "index should reload from disk");
    }

    #[test]
    fn invalid_cache_directory_on_store() {
        let cfg = SpirvCacheConfig {
            // Attempt to write to an invalid path.
            cache_dir: PathBuf::from(if cfg!(windows) {
                "Z:\\nonexistent\\deeply\\nested"
            } else {
                "/proc/nonexistent/deeply/nested"
            }),
            max_cache_size_mb: 1,
            device_fingerprint: "dev".into(),
        };
        let mut cache = SpirvCache::new(cfg);
        let key = CacheKey::from_source("k", "", "dev");
        let result = cache.store(&key, &[1]);
        assert!(result.is_err());
    }

    #[test]
    fn concurrent_access_safety() {
        // Ensure independent cache instances on the same directory don't corrupt data.
        let tmp = TempDir::new().unwrap();
        let dir = tmp.path().to_path_buf();
        let handles: Vec<_> = (0..4)
            .map(|i| {
                let d = dir.clone();
                thread::spawn(move || {
                    let mut cache = SpirvCache::new(SpirvCacheConfig {
                        cache_dir: d,
                        max_cache_size_mb: 10,
                        device_fingerprint: "dev".into(),
                    });
                    let key = CacheKey::from_source(&format!("thread_{i}"), "", "dev");
                    cache.store(&key, &vec![i as u8; 64]).unwrap();
                    let got = cache.lookup(&key).unwrap();
                    assert_eq!(got.len(), 64);
                    assert!(got.iter().all(|&b| b == i as u8));
                })
            })
            .collect();
        for h in handles {
            h.join().unwrap();
        }
    }

    // -- CompilationPipeline tests ------------------------------------------

    #[test]
    fn compile_empty_source_returns_error() {
        let result = CompilationPipeline::compile_cl_to_binary("", "");
        assert!(matches!(result, Err(CompileError::EmptySource)));
    }

    #[test]
    fn compile_stub_produces_deterministic_binary() {
        let a = CompilationPipeline::compile_cl_to_binary("hello", "-O2").unwrap();
        let b = CompilationPipeline::compile_cl_to_binary("hello", "-O2").unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn compile_stub_different_source_different_binary() {
        let a = CompilationPipeline::compile_cl_to_binary("kernel_a", "").unwrap();
        let b = CompilationPipeline::compile_cl_to_binary("kernel_b", "").unwrap();
        assert_ne!(a, b);
    }

    #[test]
    fn compile_stub_starts_with_spirv_magic() {
        let bin = CompilationPipeline::compile_cl_to_binary("test", "").unwrap();
        assert_eq!(&bin[..4], &[0x03, 0x02, 0x23, 0x07]);
    }

    #[test]
    fn compile_with_cache_misses_then_hits() {
        let tmp = TempDir::new().unwrap();
        let mut cache = SpirvCache::new(test_config(tmp.path()));

        let bin1 =
            CompilationPipeline::compile_with_cache("my_kernel", "-cl-mad-enable", &mut cache)
                .unwrap();
        assert_eq!(cache.stats().misses, 1);
        assert_eq!(cache.stats().hits, 0);

        let bin2 =
            CompilationPipeline::compile_with_cache("my_kernel", "-cl-mad-enable", &mut cache)
                .unwrap();
        assert_eq!(cache.stats().hits, 1);
        assert_eq!(bin1, bin2);
    }

    #[test]
    fn compile_with_cache_empty_source_error() {
        let tmp = TempDir::new().unwrap();
        let mut cache = SpirvCache::new(test_config(tmp.path()));
        let result = CompilationPipeline::compile_with_cache("", "", &mut cache);
        assert!(result.is_err());
    }

    // -- SHA-256 sanity tests -----------------------------------------------

    #[test]
    fn sha256_empty_input() {
        // NIST test vector for empty string
        let digest = sha256_hash(b"");
        let hex: String = digest.iter().map(|b| format!("{b:02x}")).collect();
        assert_eq!(
            hex,
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
    }

    #[test]
    fn sha256_known_vector() {
        // NIST test vector for "abc"
        let digest = sha256_hash(b"abc");
        let hex: String = digest.iter().map(|b| format!("{b:02x}")).collect();
        assert_eq!(
            hex,
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    // -- Default config tests -----------------------------------------------

    #[test]
    fn default_config_has_sensible_values() {
        let cfg = SpirvCacheConfig::default();
        assert_eq!(cfg.max_cache_size_mb, 256);
        assert!(cfg.cache_dir.to_str().unwrap().contains("spirv"));
    }

    // -- Edge-case tests ----------------------------------------------------

    #[test]
    fn store_overwrite_same_key() {
        let tmp = TempDir::new().unwrap();
        let mut cache = SpirvCache::new(test_config(tmp.path()));
        let key = CacheKey::from_source("k", "", "dev");
        cache.store(&key, &[1, 2, 3]).unwrap();
        cache.store(&key, &[4, 5, 6, 7]).unwrap();
        let got = cache.lookup(&key).unwrap();
        assert_eq!(got, vec![4, 5, 6, 7]);
        // Entry count should still be 1.
        assert_eq!(cache.stats().entry_count, 1);
    }

    #[test]
    fn invalidate_nonexistent_key_is_noop() {
        let tmp = TempDir::new().unwrap();
        let mut cache = SpirvCache::new(test_config(tmp.path()));
        let key = CacheKey::from_source("nope", "", "dev");
        cache.invalidate(&key); // should not panic
        assert_eq!(cache.stats().entry_count, 0);
    }

    #[test]
    fn cache_error_display() {
        let e = CacheError::InvalidCacheDir("not writable".into());
        let s = format!("{e}");
        assert!(s.contains("not writable"));
    }

    #[test]
    fn compile_error_display() {
        let e = CompileError::EmptySource;
        let s = format!("{e}");
        assert!(s.contains("empty"));
    }

    #[test]
    fn evict_lru_on_empty_cache_is_noop() {
        let tmp = TempDir::new().unwrap();
        let mut cache = SpirvCache::new(test_config(tmp.path()));
        cache.evict_lru(); // should not panic
    }

    #[test]
    fn lru_order_updated_on_lookup() {
        let tmp = TempDir::new().unwrap();
        let mut cfg = test_config(tmp.path());
        cfg.max_cache_size_mb = 1;
        let mut cache = SpirvCache::new(cfg);

        let big = vec![0u8; 400 * 1024]; // 400 KiB
        let k1 = CacheKey::from_source("a", "", "dev");
        let k2 = CacheKey::from_source("b", "", "dev");
        let k3 = CacheKey::from_source("c", "", "dev");

        cache.store(&k1, &big).unwrap();
        cache.store(&k2, &big).unwrap();
        // Touch k1 to move it to back of LRU.
        cache.lookup(&k1);
        // Now store k3 — should evict k2 (now oldest), not k1.
        cache.store(&k3, &big).unwrap();

        assert!(cache.lookup(&k1).is_some(), "k1 was accessed recently, should survive");
        assert!(cache.lookup(&k2).is_none(), "k2 should have been evicted");
    }
}
