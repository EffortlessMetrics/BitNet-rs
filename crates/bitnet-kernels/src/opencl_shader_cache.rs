//! Compiled OpenCL shader/program cache for Intel Arc A770 and similar devices.
//!
//! Persists compiled OpenCL binaries keyed by source hash and build options to
//! avoid recompilation across launches. Supports LRU, LFU, FIFO, and
//! size-weighted eviction policies with serialization for on-disk persistence.

use std::collections::HashMap;
use std::fmt;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// Composite key identifying a compiled program binary.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CacheKey {
    pub source_hash: u64,
    pub build_options: String,
    pub device_name: String,
}

/// A cached compiled program binary with metadata.
#[derive(Debug, Clone)]
pub struct CachedProgram {
    pub key: CacheKey,
    pub binary: Vec<u8>,
    pub compiled_at_ns: u64,
    pub hit_count: u64,
    pub size_bytes: usize,
    /// Monotonic insertion order used by FIFO eviction.
    insert_seq: u64,
    /// Monotonic access timestamp used by LRU eviction.
    last_access_seq: u64,
}

/// Eviction policy for the shader cache.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvictionPolicy {
    LRU,
    LFU,
    FIFO,
    SizeWeighted,
}

/// Configuration for [`ShaderCache`].
#[derive(Debug, Clone)]
pub struct CacheConfig {
    pub max_entries: usize,
    pub max_total_bytes: usize,
    pub eviction_policy: EvictionPolicy,
    pub persist_path: Option<String>,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            max_entries: 256,
            max_total_bytes: 256 * 1024 * 1024, // 256 MiB
            eviction_policy: EvictionPolicy::LRU,
            persist_path: None,
        }
    }
}

/// Aggregate cache statistics.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CacheStats {
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
    pub total_compile_saved_us: u64,
    pub current_entries: usize,
    pub current_bytes: usize,
}

/// Errors returned by cache operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CacheError {
    EntryNotFound,
    CacheFull,
    SerializationError(String),
    IoError(String),
    InvalidBinary,
}

impl fmt::Display for CacheError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EntryNotFound => write!(f, "entry not found"),
            Self::CacheFull => write!(f, "cache full"),
            Self::SerializationError(e) => write!(f, "serialization error: {e}"),
            Self::IoError(e) => write!(f, "I/O error: {e}"),
            Self::InvalidBinary => write!(f, "invalid binary"),
        }
    }
}

impl std::error::Error for CacheError {}

/// Compiled shader/program cache.
pub struct ShaderCache {
    pub entries: HashMap<u64, CachedProgram>,
    pub config: CacheConfig,
    pub stats: CacheStats,
    /// Monotonic counter for insertion ordering (FIFO).
    seq_counter: u64,
    /// Monotonic counter for access ordering (LRU).
    access_counter: u64,
    /// Running total of stored bytes.
    total_bytes: usize,
}

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

/// Create a new [`ShaderCache`] with the given configuration.
pub fn create_shader_cache(config: CacheConfig) -> ShaderCache {
    ShaderCache {
        entries: HashMap::new(),
        config,
        stats: CacheStats::default(),
        seq_counter: 0,
        access_counter: 0,
        total_bytes: 0,
    }
}

// ---------------------------------------------------------------------------
// Hashing
// ---------------------------------------------------------------------------

/// FNV-1a hash of `source` concatenated with `options`.
pub fn cpu_hash_source(source: &str, options: &str) -> u64 {
    const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const FNV_PRIME: u64 = 0x0100_0000_01b3;

    let mut hash = FNV_OFFSET;
    for byte in source.as_bytes().iter().chain(options.as_bytes()) {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

// ---------------------------------------------------------------------------
// Lookup / Store
// ---------------------------------------------------------------------------

/// Look up a compiled program. Returns `None` on miss; updates stats on hit.
pub fn cpu_cache_lookup<'a>(
    cache: &'a mut ShaderCache,
    key: &CacheKey,
) -> Option<&'a CachedProgram> {
    let hash =
        cpu_hash_source(&format!("{}{}", key.source_hash, key.device_name), &key.build_options);
    if cache.entries.contains_key(&hash) {
        cache.stats.hits += 1;
        cache.access_counter += 1;
        let entry = cache.entries.get_mut(&hash).unwrap();
        entry.hit_count += 1;
        entry.last_access_seq = cache.access_counter;
        // Re-borrow immutably after mutation.
        return cache.entries.get(&hash);
    }
    cache.stats.misses += 1;
    None
}

/// Store a compiled binary in the cache, evicting if necessary.
pub fn cpu_cache_store(
    cache: &mut ShaderCache,
    key: CacheKey,
    binary: Vec<u8>,
) -> Result<(), CacheError> {
    let size = binary.len();

    // Single entry too large for the cache — reject immediately.
    if size > cache.config.max_total_bytes && cache.config.max_total_bytes > 0 {
        return Err(CacheError::CacheFull);
    }

    // Evict until we have room.
    while cache.entries.len() >= cache.config.max_entries && !cache.entries.is_empty() {
        evict_one(cache);
    }
    while cache.total_bytes + size > cache.config.max_total_bytes
        && !cache.entries.is_empty()
        && cache.config.max_total_bytes > 0
    {
        evict_one(cache);
    }

    cache.seq_counter += 1;
    cache.access_counter += 1;
    let hash =
        cpu_hash_source(&format!("{}{}", key.source_hash, key.device_name), &key.build_options);
    let program = CachedProgram {
        key,
        binary,
        compiled_at_ns: cache.seq_counter, // monotonic stand-in when no real clock
        hit_count: 0,
        size_bytes: size,
        insert_seq: cache.seq_counter,
        last_access_seq: cache.access_counter,
    };
    cache.entries.insert(hash, program);
    cache.total_bytes += size;
    cache.stats.current_entries = cache.entries.len();
    cache.stats.current_bytes = cache.total_bytes;
    Ok(())
}

// ---------------------------------------------------------------------------
// Eviction helpers
// ---------------------------------------------------------------------------

fn evict_one(cache: &mut ShaderCache) {
    match cache.config.eviction_policy {
        EvictionPolicy::LRU => {
            cpu_evict_lru(cache);
        }
        EvictionPolicy::LFU => {
            cpu_evict_lfu(cache);
        }
        EvictionPolicy::FIFO => {
            cpu_evict_fifo(cache);
        }
        EvictionPolicy::SizeWeighted => {
            cpu_evict_size_weighted(cache);
        }
    }
}

/// Evict the least-recently-used entry.
pub fn cpu_evict_lru(cache: &mut ShaderCache) -> Option<CachedProgram> {
    let victim = cache.entries.iter().min_by_key(|(_, p)| p.last_access_seq).map(|(&k, _)| k);
    if let Some(k) = victim {
        let evicted = cache.entries.remove(&k).unwrap();
        cache.total_bytes = cache.total_bytes.saturating_sub(evicted.size_bytes);
        cache.stats.evictions += 1;
        cache.stats.current_entries = cache.entries.len();
        cache.stats.current_bytes = cache.total_bytes;
        return Some(evicted);
    }
    None
}

/// Evict the least-frequently-used entry.
pub fn cpu_evict_lfu(cache: &mut ShaderCache) -> Option<CachedProgram> {
    let victim = cache.entries.iter().min_by_key(|(_, p)| p.hit_count).map(|(&k, _)| k);
    if let Some(k) = victim {
        let evicted = cache.entries.remove(&k).unwrap();
        cache.total_bytes = cache.total_bytes.saturating_sub(evicted.size_bytes);
        cache.stats.evictions += 1;
        cache.stats.current_entries = cache.entries.len();
        cache.stats.current_bytes = cache.total_bytes;
        return Some(evicted);
    }
    None
}

/// Evict the oldest (first inserted) entry.
pub fn cpu_evict_fifo(cache: &mut ShaderCache) -> Option<CachedProgram> {
    let victim = cache.entries.iter().min_by_key(|(_, p)| p.insert_seq).map(|(&k, _)| k);
    if let Some(k) = victim {
        let evicted = cache.entries.remove(&k).unwrap();
        cache.total_bytes = cache.total_bytes.saturating_sub(evicted.size_bytes);
        cache.stats.evictions += 1;
        cache.stats.current_entries = cache.entries.len();
        cache.stats.current_bytes = cache.total_bytes;
        return Some(evicted);
    }
    None
}

/// Evict the entry with the largest binary (size-weighted policy).
fn cpu_evict_size_weighted(cache: &mut ShaderCache) -> Option<CachedProgram> {
    let victim = cache.entries.iter().max_by_key(|(_, p)| p.size_bytes).map(|(&k, _)| k);
    if let Some(k) = victim {
        let evicted = cache.entries.remove(&k).unwrap();
        cache.total_bytes = cache.total_bytes.saturating_sub(evicted.size_bytes);
        cache.stats.evictions += 1;
        cache.stats.current_entries = cache.entries.len();
        cache.stats.current_bytes = cache.total_bytes;
        return Some(evicted);
    }
    None
}

// ---------------------------------------------------------------------------
// Invalidation / clear
// ---------------------------------------------------------------------------

/// Remove a specific entry. Returns `true` if it existed.
pub fn cpu_invalidate(cache: &mut ShaderCache, key: &CacheKey) -> bool {
    let hash =
        cpu_hash_source(&format!("{}{}", key.source_hash, key.device_name), &key.build_options);
    if let Some(evicted) = cache.entries.remove(&hash) {
        cache.total_bytes = cache.total_bytes.saturating_sub(evicted.size_bytes);
        cache.stats.current_entries = cache.entries.len();
        cache.stats.current_bytes = cache.total_bytes;
        return true;
    }
    false
}

/// Clear the entire cache.
pub fn cpu_clear_cache(cache: &mut ShaderCache) {
    cache.entries.clear();
    cache.total_bytes = 0;
    cache.stats.current_entries = 0;
    cache.stats.current_bytes = 0;
}

// ---------------------------------------------------------------------------
// Stats
// ---------------------------------------------------------------------------

/// Return a snapshot of the current cache statistics.
pub fn cpu_get_stats(cache: &ShaderCache) -> CacheStats {
    cache.stats.clone()
}

/// Human-readable summary of cache statistics.
pub fn format_cache_stats(stats: &CacheStats) -> String {
    let total = stats.hits + stats.misses;
    let hit_rate = if total > 0 { stats.hits as f64 / total as f64 * 100.0 } else { 0.0 };
    format!(
        "ShaderCache: entries={} bytes={} hits={} misses={} hit_rate={:.1}% evictions={} saved={}µs",
        stats.current_entries,
        stats.current_bytes,
        stats.hits,
        stats.misses,
        hit_rate,
        stats.evictions,
        stats.total_compile_saved_us,
    )
}

// ---------------------------------------------------------------------------
// Serialization (simple binary format)
// ---------------------------------------------------------------------------

/// Serialize the cache to a byte vector.
///
/// Format (all little-endian):
///   magic: 4 bytes  "BSCV"
///   version: u32
///   entry_count: u64
///   for each entry:
///     source_hash: u64
///     build_options_len: u32, build_options: [u8]
///     device_name_len: u32, device_name: [u8]
///     binary_len: u64, binary: [u8]
///     compiled_at_ns: u64
///     hit_count: u64
pub fn cpu_serialize_cache(cache: &ShaderCache) -> Vec<u8> {
    let mut buf: Vec<u8> = Vec::new();
    // Magic + version
    buf.extend_from_slice(b"BSCV");
    buf.extend_from_slice(&1u32.to_le_bytes());
    buf.extend_from_slice(&(cache.entries.len() as u64).to_le_bytes());

    for entry in cache.entries.values() {
        buf.extend_from_slice(&entry.key.source_hash.to_le_bytes());

        let opts = entry.key.build_options.as_bytes();
        buf.extend_from_slice(&(opts.len() as u32).to_le_bytes());
        buf.extend_from_slice(opts);

        let dev = entry.key.device_name.as_bytes();
        buf.extend_from_slice(&(dev.len() as u32).to_le_bytes());
        buf.extend_from_slice(dev);

        buf.extend_from_slice(&(entry.binary.len() as u64).to_le_bytes());
        buf.extend_from_slice(&entry.binary);

        buf.extend_from_slice(&entry.compiled_at_ns.to_le_bytes());
        buf.extend_from_slice(&entry.hit_count.to_le_bytes());
    }
    buf
}

/// Deserialize a cache from a byte slice produced by [`cpu_serialize_cache`].
pub fn cpu_deserialize_cache(data: &[u8], config: CacheConfig) -> Result<ShaderCache, CacheError> {
    let mut cache = create_shader_cache(config);
    let mut pos: usize = 0;

    let read_u32 = |pos: &mut usize, data: &[u8]| -> Result<u32, CacheError> {
        if *pos + 4 > data.len() {
            return Err(CacheError::SerializationError("unexpected EOF".into()));
        }
        let val = u32::from_le_bytes(data[*pos..*pos + 4].try_into().unwrap());
        *pos += 4;
        Ok(val)
    };

    let read_u64 = |pos: &mut usize, data: &[u8]| -> Result<u64, CacheError> {
        if *pos + 8 > data.len() {
            return Err(CacheError::SerializationError("unexpected EOF".into()));
        }
        let val = u64::from_le_bytes(data[*pos..*pos + 8].try_into().unwrap());
        *pos += 8;
        Ok(val)
    };

    let read_bytes = |pos: &mut usize, data: &[u8], len: usize| -> Result<Vec<u8>, CacheError> {
        if *pos + len > data.len() {
            return Err(CacheError::SerializationError("unexpected EOF".into()));
        }
        let val = data[*pos..*pos + len].to_vec();
        *pos += len;
        Ok(val)
    };

    // Magic
    if data.len() < 4 || &data[0..4] != b"BSCV" {
        return Err(CacheError::SerializationError("bad magic".into()));
    }
    // Skip past the 4-byte magic header.
    pos += 4;
    let version = read_u32(&mut pos, data)?;
    if version != 1 {
        return Err(CacheError::SerializationError(format!("unsupported version {version}")));
    }
    let count = read_u64(&mut pos, data)? as usize;

    for _ in 0..count {
        let source_hash = read_u64(&mut pos, data)?;

        let opts_len = read_u32(&mut pos, data)? as usize;
        let opts_bytes = read_bytes(&mut pos, data, opts_len)?;
        let build_options = String::from_utf8(opts_bytes)
            .map_err(|e| CacheError::SerializationError(e.to_string()))?;

        let dev_len = read_u32(&mut pos, data)? as usize;
        let dev_bytes = read_bytes(&mut pos, data, dev_len)?;
        let device_name = String::from_utf8(dev_bytes)
            .map_err(|e| CacheError::SerializationError(e.to_string()))?;

        let bin_len = read_u64(&mut pos, data)? as usize;
        let binary = read_bytes(&mut pos, data, bin_len)?;

        let compiled_at_ns = read_u64(&mut pos, data)?;
        let hit_count = read_u64(&mut pos, data)?;

        let key = CacheKey { source_hash, build_options, device_name };
        let size = binary.len();
        let hash =
            cpu_hash_source(&format!("{}{}", key.source_hash, key.device_name), &key.build_options);

        cache.seq_counter += 1;
        cache.access_counter += 1;
        let program = CachedProgram {
            key,
            binary,
            compiled_at_ns,
            hit_count,
            size_bytes: size,
            insert_seq: cache.seq_counter,
            last_access_seq: cache.access_counter,
        };
        cache.entries.insert(hash, program);
        cache.total_bytes += size;
    }
    cache.stats.current_entries = cache.entries.len();
    cache.stats.current_bytes = cache.total_bytes;
    Ok(cache)
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn default_config() -> CacheConfig {
        CacheConfig { max_entries: 64, max_total_bytes: 1024 * 1024, ..CacheConfig::default() }
    }

    fn make_key(id: u64) -> CacheKey {
        CacheKey {
            source_hash: id,
            build_options: format!("-O{id}"),
            device_name: "Intel_Arc_A770".into(),
        }
    }

    fn make_binary(len: usize) -> Vec<u8> {
        (0..len).map(|i| (i & 0xFF) as u8).collect()
    }

    // -- hash tests --

    #[test]
    fn hash_same_input_same_hash() {
        let h1 = cpu_hash_source("kernel void f(){}", "-O2");
        let h2 = cpu_hash_source("kernel void f(){}", "-O2");
        assert_eq!(h1, h2);
    }

    #[test]
    fn hash_different_source_different_hash() {
        let h1 = cpu_hash_source("kernel void f(){}", "-O2");
        let h2 = cpu_hash_source("kernel void g(){}", "-O2");
        assert_ne!(h1, h2);
    }

    #[test]
    fn hash_different_options_different_hash() {
        let h1 = cpu_hash_source("kernel void f(){}", "-O2");
        let h2 = cpu_hash_source("kernel void f(){}", "-O0");
        assert_ne!(h1, h2);
    }

    #[test]
    fn hash_empty_source() {
        let h1 = cpu_hash_source("", "");
        let h2 = cpu_hash_source("", "");
        assert_eq!(h1, h2);
    }

    #[test]
    fn hash_empty_options_not_equal_nonempty() {
        let h1 = cpu_hash_source("src", "");
        let h2 = cpu_hash_source("src", "-O2");
        assert_ne!(h1, h2);
    }

    // -- store / lookup --

    #[test]
    fn store_and_lookup_roundtrip() {
        let mut cache = create_shader_cache(default_config());
        let key = make_key(1);
        let bin = make_binary(64);
        cpu_cache_store(&mut cache, key.clone(), bin.clone()).unwrap();
        let found = cpu_cache_lookup(&mut cache, &key).unwrap();
        assert_eq!(found.binary, bin);
    }

    #[test]
    fn lookup_miss_returns_none() {
        let mut cache = create_shader_cache(default_config());
        let key = make_key(42);
        assert!(cpu_cache_lookup(&mut cache, &key).is_none());
    }

    #[test]
    fn cache_hit_increments_hit_count() {
        let mut cache = create_shader_cache(default_config());
        let key = make_key(1);
        cpu_cache_store(&mut cache, key.clone(), make_binary(8)).unwrap();
        cpu_cache_lookup(&mut cache, &key);
        cpu_cache_lookup(&mut cache, &key);
        let entry = cpu_cache_lookup(&mut cache, &key).unwrap();
        assert_eq!(entry.hit_count, 3);
    }

    #[test]
    fn cache_miss_increments_miss_count() {
        let mut cache = create_shader_cache(default_config());
        cpu_cache_lookup(&mut cache, &make_key(1));
        cpu_cache_lookup(&mut cache, &make_key(2));
        assert_eq!(cache.stats.misses, 2);
    }

    #[test]
    fn cache_hit_increments_hit_stat() {
        let mut cache = create_shader_cache(default_config());
        let key = make_key(1);
        cpu_cache_store(&mut cache, key.clone(), make_binary(8)).unwrap();
        cpu_cache_lookup(&mut cache, &key);
        assert_eq!(cache.stats.hits, 1);
    }

    // -- LRU eviction --

    #[test]
    fn lru_eviction_evicts_least_recently_used() {
        let config = CacheConfig {
            max_entries: 2,
            eviction_policy: EvictionPolicy::LRU,
            ..default_config()
        };
        let mut cache = create_shader_cache(config);
        let k1 = make_key(1);
        let k2 = make_key(2);
        let k3 = make_key(3);
        cpu_cache_store(&mut cache, k1.clone(), make_binary(8)).unwrap();
        cpu_cache_store(&mut cache, k2.clone(), make_binary(8)).unwrap();
        // Access k1 so k2 becomes LRU.
        cpu_cache_lookup(&mut cache, &k1);
        cpu_cache_store(&mut cache, k3.clone(), make_binary(8)).unwrap();
        assert!(cpu_cache_lookup(&mut cache, &k2).is_none());
        assert!(cpu_cache_lookup(&mut cache, &k1).is_some());
    }

    #[test]
    fn lru_eviction_returns_evicted_entry() {
        let config = CacheConfig {
            max_entries: 1,
            eviction_policy: EvictionPolicy::LRU,
            ..default_config()
        };
        let mut cache = create_shader_cache(config);
        cpu_cache_store(&mut cache, make_key(1), make_binary(8)).unwrap();
        let evicted = cpu_evict_lru(&mut cache);
        assert!(evicted.is_some());
        assert!(cache.entries.is_empty());
    }

    #[test]
    fn lru_eviction_on_empty_cache() {
        let mut cache = create_shader_cache(default_config());
        assert!(cpu_evict_lru(&mut cache).is_none());
    }

    // -- LFU eviction --

    #[test]
    fn lfu_eviction_evicts_least_frequently_used() {
        let config = CacheConfig {
            max_entries: 2,
            eviction_policy: EvictionPolicy::LFU,
            ..default_config()
        };
        let mut cache = create_shader_cache(config);
        let k1 = make_key(1);
        let k2 = make_key(2);
        let k3 = make_key(3);
        cpu_cache_store(&mut cache, k1.clone(), make_binary(8)).unwrap();
        cpu_cache_store(&mut cache, k2.clone(), make_binary(8)).unwrap();
        // Hit k1 twice so k2 (hit_count=0) is least frequent.
        cpu_cache_lookup(&mut cache, &k1);
        cpu_cache_lookup(&mut cache, &k1);
        cpu_cache_store(&mut cache, k3.clone(), make_binary(8)).unwrap();
        assert!(cpu_cache_lookup(&mut cache, &k2).is_none());
        assert!(cpu_cache_lookup(&mut cache, &k1).is_some());
    }

    #[test]
    fn lfu_eviction_returns_evicted_entry() {
        let config = CacheConfig {
            max_entries: 1,
            eviction_policy: EvictionPolicy::LFU,
            ..default_config()
        };
        let mut cache = create_shader_cache(config);
        cpu_cache_store(&mut cache, make_key(1), make_binary(8)).unwrap();
        let evicted = cpu_evict_lfu(&mut cache);
        assert!(evicted.is_some());
    }

    #[test]
    fn lfu_eviction_on_empty_cache() {
        let mut cache = create_shader_cache(default_config());
        assert!(cpu_evict_lfu(&mut cache).is_none());
    }

    // -- FIFO eviction --

    #[test]
    fn fifo_eviction_evicts_oldest() {
        let config = CacheConfig {
            max_entries: 2,
            eviction_policy: EvictionPolicy::FIFO,
            ..default_config()
        };
        let mut cache = create_shader_cache(config);
        let k1 = make_key(1);
        let k2 = make_key(2);
        let k3 = make_key(3);
        cpu_cache_store(&mut cache, k1.clone(), make_binary(8)).unwrap();
        cpu_cache_store(&mut cache, k2.clone(), make_binary(8)).unwrap();
        cpu_cache_store(&mut cache, k3.clone(), make_binary(8)).unwrap();
        // k1 is oldest → evicted
        assert!(cpu_cache_lookup(&mut cache, &k1).is_none());
        assert!(cpu_cache_lookup(&mut cache, &k2).is_some());
    }

    #[test]
    fn fifo_eviction_returns_evicted_entry() {
        let config = CacheConfig {
            max_entries: 1,
            eviction_policy: EvictionPolicy::FIFO,
            ..default_config()
        };
        let mut cache = create_shader_cache(config);
        cpu_cache_store(&mut cache, make_key(1), make_binary(8)).unwrap();
        let evicted = cpu_evict_fifo(&mut cache);
        assert!(evicted.is_some());
    }

    #[test]
    fn fifo_eviction_on_empty_cache() {
        let mut cache = create_shader_cache(default_config());
        assert!(cpu_evict_fifo(&mut cache).is_none());
    }

    // -- size-weighted eviction --

    #[test]
    fn size_weighted_evicts_largest() {
        let config = CacheConfig {
            max_entries: 2,
            eviction_policy: EvictionPolicy::SizeWeighted,
            ..default_config()
        };
        let mut cache = create_shader_cache(config);
        let k1 = make_key(1);
        let k2 = make_key(2);
        let k3 = make_key(3);
        cpu_cache_store(&mut cache, k1.clone(), make_binary(128)).unwrap();
        cpu_cache_store(&mut cache, k2.clone(), make_binary(8)).unwrap();
        cpu_cache_store(&mut cache, k3.clone(), make_binary(16)).unwrap();
        // k1 is largest → evicted
        assert!(cpu_cache_lookup(&mut cache, &k1).is_none());
        assert!(cpu_cache_lookup(&mut cache, &k2).is_some());
    }

    // -- max entries / max bytes triggers --

    #[test]
    fn max_entries_triggers_eviction() {
        let config = CacheConfig {
            max_entries: 3,
            eviction_policy: EvictionPolicy::FIFO,
            ..default_config()
        };
        let mut cache = create_shader_cache(config);
        for i in 0..5 {
            cpu_cache_store(&mut cache, make_key(i), make_binary(8)).unwrap();
        }
        assert_eq!(cache.entries.len(), 3);
        assert!(cache.stats.evictions >= 2);
    }

    #[test]
    fn max_bytes_triggers_eviction() {
        let config = CacheConfig {
            max_entries: 100,
            max_total_bytes: 24,
            eviction_policy: EvictionPolicy::FIFO,
            ..default_config()
        };
        let mut cache = create_shader_cache(config);
        for i in 0..5 {
            cpu_cache_store(&mut cache, make_key(i), make_binary(8)).unwrap();
        }
        assert!(cache.total_bytes <= 24);
        assert!(cache.stats.evictions > 0);
    }

    #[test]
    fn store_rejects_oversized_binary() {
        let config = CacheConfig {
            max_entries: 10,
            max_total_bytes: 4,
            eviction_policy: EvictionPolicy::LRU,
            ..default_config()
        };
        let mut cache = create_shader_cache(config);
        let res = cpu_cache_store(&mut cache, make_key(1), make_binary(100));
        assert_eq!(res, Err(CacheError::CacheFull));
    }

    // -- invalidate / clear --

    #[test]
    fn invalidate_removes_entry() {
        let mut cache = create_shader_cache(default_config());
        let key = make_key(1);
        cpu_cache_store(&mut cache, key.clone(), make_binary(8)).unwrap();
        assert!(cpu_invalidate(&mut cache, &key));
        assert!(cpu_cache_lookup(&mut cache, &key).is_none());
    }

    #[test]
    fn invalidate_nonexistent_returns_false() {
        let mut cache = create_shader_cache(default_config());
        assert!(!cpu_invalidate(&mut cache, &make_key(99)));
    }

    #[test]
    fn clear_removes_all() {
        let mut cache = create_shader_cache(default_config());
        for i in 0..5 {
            cpu_cache_store(&mut cache, make_key(i), make_binary(8)).unwrap();
        }
        cpu_clear_cache(&mut cache);
        assert!(cache.entries.is_empty());
        assert_eq!(cache.stats.current_entries, 0);
        assert_eq!(cache.stats.current_bytes, 0);
    }

    // -- stats --

    #[test]
    fn stats_correct_hit_miss_counts() {
        let mut cache = create_shader_cache(default_config());
        let key = make_key(1);
        cpu_cache_store(&mut cache, key.clone(), make_binary(8)).unwrap();
        cpu_cache_lookup(&mut cache, &make_key(99)); // miss
        cpu_cache_lookup(&mut cache, &key); // hit
        let stats = cpu_get_stats(&cache);
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 1);
    }

    #[test]
    fn stats_current_entries_and_bytes() {
        let mut cache = create_shader_cache(default_config());
        cpu_cache_store(&mut cache, make_key(1), make_binary(16)).unwrap();
        cpu_cache_store(&mut cache, make_key(2), make_binary(32)).unwrap();
        let stats = cpu_get_stats(&cache);
        assert_eq!(stats.current_entries, 2);
        assert_eq!(stats.current_bytes, 48);
    }

    #[test]
    fn stats_eviction_count() {
        let config = CacheConfig {
            max_entries: 1,
            eviction_policy: EvictionPolicy::LRU,
            ..default_config()
        };
        let mut cache = create_shader_cache(config);
        cpu_cache_store(&mut cache, make_key(1), make_binary(8)).unwrap();
        cpu_cache_store(&mut cache, make_key(2), make_binary(8)).unwrap();
        assert_eq!(cpu_get_stats(&cache).evictions, 1);
    }

    // -- serialize / deserialize --

    #[test]
    fn serialize_deserialize_roundtrip() {
        let mut cache = create_shader_cache(default_config());
        cpu_cache_store(&mut cache, make_key(1), make_binary(16)).unwrap();
        cpu_cache_store(&mut cache, make_key(2), make_binary(32)).unwrap();
        let data = cpu_serialize_cache(&cache);
        let restored = cpu_deserialize_cache(&data, default_config()).unwrap();
        assert_eq!(restored.entries.len(), 2);
        assert_eq!(restored.stats.current_bytes, 48);
    }

    #[test]
    fn serialize_empty_cache() {
        let cache = create_shader_cache(default_config());
        let data = cpu_serialize_cache(&cache);
        let restored = cpu_deserialize_cache(&data, default_config()).unwrap();
        assert!(restored.entries.is_empty());
    }

    #[test]
    fn deserialize_invalid_magic() {
        let data = b"XXXX\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00";
        let res = cpu_deserialize_cache(data, default_config());
        assert!(matches!(res, Err(CacheError::SerializationError(_))));
    }

    #[test]
    fn deserialize_unsupported_version() {
        let mut data = b"BSCV".to_vec();
        data.extend_from_slice(&99u32.to_le_bytes());
        data.extend_from_slice(&0u64.to_le_bytes());
        let res = cpu_deserialize_cache(&data, default_config());
        assert!(matches!(res, Err(CacheError::SerializationError(_))));
    }

    #[test]
    fn deserialize_truncated_data() {
        let data = b"BSCV\x01\x00\x00\x00";
        let res = cpu_deserialize_cache(data, default_config());
        assert!(matches!(res, Err(CacheError::SerializationError(_))));
    }

    #[test]
    fn serialize_preserves_binary_content() {
        let mut cache = create_shader_cache(default_config());
        let bin = vec![0xDE, 0xAD, 0xBE, 0xEF];
        let key = make_key(7);
        cpu_cache_store(&mut cache, key.clone(), bin.clone()).unwrap();
        let data = cpu_serialize_cache(&cache);
        let restored = cpu_deserialize_cache(&data, default_config()).unwrap();
        let hash =
            cpu_hash_source(&format!("{}{}", key.source_hash, key.device_name), &key.build_options);
        assert_eq!(restored.entries[&hash].binary, bin);
    }

    // -- edge cases --

    #[test]
    fn single_entry_cache() {
        let config = CacheConfig { max_entries: 1, ..default_config() };
        let mut cache = create_shader_cache(config);
        cpu_cache_store(&mut cache, make_key(1), make_binary(8)).unwrap();
        assert_eq!(cache.entries.len(), 1);
        cpu_cache_store(&mut cache, make_key(2), make_binary(8)).unwrap();
        assert_eq!(cache.entries.len(), 1);
    }

    #[test]
    fn zero_byte_binary() {
        let mut cache = create_shader_cache(default_config());
        let key = make_key(1);
        cpu_cache_store(&mut cache, key.clone(), vec![]).unwrap();
        let found = cpu_cache_lookup(&mut cache, &key).unwrap();
        assert!(found.binary.is_empty());
        assert_eq!(found.size_bytes, 0);
    }

    #[test]
    fn large_binary_stored_correctly() {
        let mut cache = create_shader_cache(default_config());
        let key = make_key(1);
        let bin = make_binary(65_536);
        cpu_cache_store(&mut cache, key.clone(), bin.clone()).unwrap();
        let found = cpu_cache_lookup(&mut cache, &key).unwrap();
        assert_eq!(found.binary.len(), 65_536);
        assert_eq!(found.binary, bin);
    }

    // -- property tests --

    #[test]
    fn hit_rate_property() {
        let mut cache = create_shader_cache(default_config());
        let key = make_key(1);
        cpu_cache_store(&mut cache, key.clone(), make_binary(8)).unwrap();
        // 3 hits, 2 misses
        cpu_cache_lookup(&mut cache, &key);
        cpu_cache_lookup(&mut cache, &key);
        cpu_cache_lookup(&mut cache, &key);
        cpu_cache_lookup(&mut cache, &make_key(90));
        cpu_cache_lookup(&mut cache, &make_key(91));
        let stats = cpu_get_stats(&cache);
        let total = stats.hits + stats.misses;
        let rate = stats.hits as f64 / total as f64;
        assert!((rate - 0.6).abs() < 1e-9);
    }

    #[test]
    fn eviction_preserves_max_entries_invariant() {
        let config = CacheConfig {
            max_entries: 4,
            eviction_policy: EvictionPolicy::LRU,
            ..default_config()
        };
        let mut cache = create_shader_cache(config);
        for i in 0..20 {
            cpu_cache_store(&mut cache, make_key(i), make_binary(8)).unwrap();
            assert!(cache.entries.len() <= 4);
        }
    }

    #[test]
    fn format_cache_stats_contains_hit_rate() {
        let stats = CacheStats { hits: 7, misses: 3, ..Default::default() };
        let formatted = format_cache_stats(&stats);
        assert!(formatted.contains("70.0%"));
    }

    #[test]
    fn format_cache_stats_zero_requests() {
        let stats = CacheStats::default();
        let formatted = format_cache_stats(&stats);
        assert!(formatted.contains("0.0%"));
    }

    #[test]
    fn different_device_names_produce_different_entries() {
        let mut cache = create_shader_cache(default_config());
        let k1 = CacheKey { source_hash: 1, build_options: "".into(), device_name: "A770".into() };
        let k2 = CacheKey { source_hash: 1, build_options: "".into(), device_name: "A750".into() };
        cpu_cache_store(&mut cache, k1.clone(), vec![1]).unwrap();
        cpu_cache_store(&mut cache, k2.clone(), vec![2]).unwrap();
        assert_eq!(cache.entries.len(), 2);
        assert_eq!(cpu_cache_lookup(&mut cache, &k1).unwrap().binary, vec![1]);
        assert_eq!(cpu_cache_lookup(&mut cache, &k2).unwrap().binary, vec![2]);
    }

    #[test]
    fn invalidate_updates_byte_count() {
        let mut cache = create_shader_cache(default_config());
        let key = make_key(1);
        cpu_cache_store(&mut cache, key.clone(), make_binary(64)).unwrap();
        assert_eq!(cache.stats.current_bytes, 64);
        cpu_invalidate(&mut cache, &key);
        assert_eq!(cache.stats.current_bytes, 0);
    }

    #[test]
    fn clear_after_evictions_resets_bytes() {
        let config = CacheConfig {
            max_entries: 2,
            eviction_policy: EvictionPolicy::LRU,
            ..default_config()
        };
        let mut cache = create_shader_cache(config);
        for i in 0..5 {
            cpu_cache_store(&mut cache, make_key(i), make_binary(16)).unwrap();
        }
        cpu_clear_cache(&mut cache);
        assert_eq!(cache.total_bytes, 0);
        assert_eq!(cache.stats.current_bytes, 0);
    }

    #[test]
    fn create_shader_cache_default_config() {
        let cache = create_shader_cache(CacheConfig::default());
        assert_eq!(cache.entries.len(), 0);
        assert_eq!(cache.config.max_entries, 256);
    }

    #[test]
    fn cache_config_default_values() {
        let cfg = CacheConfig::default();
        assert_eq!(cfg.max_entries, 256);
        assert_eq!(cfg.max_total_bytes, 256 * 1024 * 1024);
        assert_eq!(cfg.eviction_policy, EvictionPolicy::LRU);
        assert!(cfg.persist_path.is_none());
    }

    #[test]
    fn cache_error_display() {
        assert_eq!(format!("{}", CacheError::EntryNotFound), "entry not found");
        assert_eq!(format!("{}", CacheError::CacheFull), "cache full");
        assert_eq!(format!("{}", CacheError::InvalidBinary), "invalid binary");
        assert!(format!("{}", CacheError::SerializationError("x".into())).contains("x"));
        assert!(format!("{}", CacheError::IoError("y".into())).contains("y"));
    }
}
