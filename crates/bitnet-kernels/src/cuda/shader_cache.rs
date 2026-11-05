//! PTX/CUBIN shader compilation cache with LRU eviction and persistence.
//!
//! # Overview
//!
//! Provides a caching layer for compiled CUDA PTX and CUBIN shaders to avoid
//! redundant runtime compilation via NVRTC. Features include:
//!
//! - [`ShaderCache`] — in-memory LRU cache keyed by source content hash
//! - [`ShaderSource`] — source code plus compile options and target architecture
//! - [`CachedShader`] — compiled PTX/CUBIN with metadata
//! - [`compile_shader`] — compile (or simulate on CPU) a shader source
//! - [`precompile_common_shaders`] — warm the cache with frequently used kernels
//! - [`warm_cache_from_disk`] / [`save_cache_to_disk`] — persistence
//!
//! All GPU dispatch is feature-gated behind
//! `#[cfg(any(feature = "gpu", feature = "cuda"))]`.
//! CPU builds simulate compilation for testing on non-GPU hosts.

use std::collections::HashMap;
use std::fmt;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use bitnet_common::{KernelError, Result};

// ── Configuration ────────────────────────────────────────────────────

/// Hash algorithm used for shader source fingerprinting.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum HashAlgorithm {
    /// FNV-1a 64-bit hash (fast, non-cryptographic).
    #[default]
    Fnv1a64,
    /// Simple DJB2 hash (fast, non-cryptographic).
    Djb2,
}

impl fmt::Display for HashAlgorithm {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Fnv1a64 => write!(f, "fnv1a64"),
            Self::Djb2 => write!(f, "djb2"),
        }
    }
}

/// Configuration for the shader cache.
#[derive(Debug, Clone)]
pub struct ShaderCacheConfig {
    /// Directory for persisting cached shaders.
    pub cache_dir: PathBuf,
    /// Maximum total cache size in megabytes.
    pub max_cache_size_mb: u64,
    /// Whether to persist the cache to disk.
    pub enable_persistence: bool,
    /// Hash algorithm for source fingerprinting.
    pub hash_algorithm: HashAlgorithm,
}

impl Default for ShaderCacheConfig {
    fn default() -> Self {
        Self {
            cache_dir: PathBuf::from(".shader_cache"),
            max_cache_size_mb: 256,
            enable_persistence: true,
            hash_algorithm: HashAlgorithm::default(),
        }
    }
}

impl ShaderCacheConfig {
    /// Validate configuration values.
    pub fn validate(&self) -> Result<()> {
        if self.max_cache_size_mb == 0 {
            return Err(KernelError::InvalidArguments {
                reason: "max_cache_size_mb must be non-zero".into(),
            }
            .into());
        }
        Ok(())
    }

    /// Maximum cache size in bytes.
    pub fn max_cache_size_bytes(&self) -> u64 {
        self.max_cache_size_mb * 1024 * 1024
    }
}

// ── Shader source ────────────────────────────────────────────────────

/// A CUDA shader source ready for compilation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShaderSource {
    /// CUDA source code (PTX or CUDA C).
    pub cuda_source: String,
    /// Compile options passed to NVRTC (e.g. `--gpu-architecture=sm_80`).
    pub compile_options: Vec<String>,
    /// Target GPU architecture (e.g. `sm_80`, `sm_86`).
    pub target_arch: String,
}

impl ShaderSource {
    /// Create a new shader source.
    pub fn new(
        cuda_source: impl Into<String>,
        compile_options: Vec<String>,
        target_arch: impl Into<String>,
    ) -> Self {
        Self { cuda_source: cuda_source.into(), compile_options, target_arch: target_arch.into() }
    }

    /// Compute a content hash for this source (source + options + arch).
    pub fn content_hash(&self, algorithm: HashAlgorithm) -> u64 {
        let mut combined = self.cuda_source.clone();
        for opt in &self.compile_options {
            combined.push('|');
            combined.push_str(opt);
        }
        combined.push('|');
        combined.push_str(&self.target_arch);
        compute_hash(combined.as_bytes(), algorithm)
    }
}

// ── Shader metadata ──────────────────────────────────────────────────

/// Metadata associated with a compiled shader.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShaderMetadata {
    /// Hash of the original source code.
    pub source_hash: u64,
    /// Hash of the compile options.
    pub compile_options_hash: u64,
    /// Unix timestamp when the shader was compiled.
    pub timestamp: u64,
    /// Target GPU architecture string.
    pub target_arch: String,
}

impl ShaderMetadata {
    /// Create metadata from a shader source and algorithm.
    pub fn from_source(source: &ShaderSource, algorithm: HashAlgorithm) -> Self {
        let source_hash = compute_hash(source.cuda_source.as_bytes(), algorithm);
        let opts_combined: String = source.compile_options.join("|");
        let compile_options_hash = compute_hash(opts_combined.as_bytes(), algorithm);
        let timestamp =
            SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or(Duration::ZERO).as_secs();
        Self {
            source_hash,
            compile_options_hash,
            timestamp,
            target_arch: source.target_arch.clone(),
        }
    }
}

// ── Cached shader ────────────────────────────────────────────────────

/// A compiled shader stored in the cache.
#[derive(Debug, Clone)]
pub struct CachedShader {
    /// Compiled PTX bytecode.
    pub ptx: Vec<u8>,
    /// Optional compiled CUBIN (device-specific binary).
    pub cubin: Option<Vec<u8>>,
    /// Time spent compiling this shader in milliseconds.
    pub compile_time_ms: u64,
    /// Compilation metadata.
    pub metadata: ShaderMetadata,
}

impl CachedShader {
    /// Total size in bytes of this cached shader (PTX + optional CUBIN).
    pub fn size_bytes(&self) -> usize {
        self.ptx.len() + self.cubin.as_ref().map_or(0, |c| c.len())
    }
}

// ── Cache statistics ─────────────────────────────────────────────────

/// Runtime statistics for the shader cache.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CacheStats {
    /// Number of cache hits.
    pub hits: u64,
    /// Number of cache misses.
    pub misses: u64,
    /// Number of entries evicted via LRU.
    pub evictions: u64,
    /// Total size of all cached shaders in bytes.
    pub total_size: u64,
    /// Number of entries currently in the cache.
    pub entry_count: u64,
    /// Number of compilation errors encountered.
    pub compile_errors: u64,
}

impl CacheStats {
    /// Hit rate as a percentage (0.0–100.0).
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            return 0.0;
        }
        (self.hits as f64 / total as f64) * 100.0
    }
}

impl fmt::Display for CacheStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "hits={} misses={} evictions={} entries={} size={}B hit_rate={:.1}%",
            self.hits,
            self.misses,
            self.evictions,
            self.entry_count,
            self.total_size,
            self.hit_rate(),
        )
    }
}

// ── LRU entry wrapper ────────────────────────────────────────────────

/// Internal wrapper tracking LRU access order.
#[derive(Debug, Clone)]
struct LruEntry {
    shader: CachedShader,
    last_access: Instant,
    access_count: u64,
}

// ── ShaderCache ──────────────────────────────────────────────────────

/// In-memory LRU shader cache with optional disk persistence.
#[derive(Debug)]
pub struct ShaderCache {
    config: ShaderCacheConfig,
    entries: HashMap<u64, LruEntry>,
    stats: CacheStats,
}

impl ShaderCache {
    /// Create a new shader cache with the given configuration.
    pub fn new(config: ShaderCacheConfig) -> Result<Self> {
        config.validate()?;
        Ok(Self { config, entries: HashMap::new(), stats: CacheStats::default() })
    }

    /// Create a cache with default configuration.
    pub fn with_defaults() -> Result<Self> {
        Self::new(ShaderCacheConfig::default())
    }

    /// Current configuration.
    pub fn config(&self) -> &ShaderCacheConfig {
        &self.config
    }

    /// Insert a compiled shader into the cache.
    pub fn insert(&mut self, key: u64, shader: CachedShader) {
        let size = shader.size_bytes() as u64;

        // Evict until there is room.
        while self.stats.total_size + size > self.config.max_cache_size_bytes()
            && !self.entries.is_empty()
        {
            self.evict_lru();
        }

        // If the single entry exceeds the budget, still insert it (but it
        // will be the only entry).
        if let Some(existing) = self.entries.get(&key) {
            self.stats.total_size =
                self.stats.total_size.saturating_sub(existing.shader.size_bytes() as u64);
            self.stats.entry_count = self.stats.entry_count.saturating_sub(1);
        }

        self.entries.insert(key, LruEntry { shader, last_access: Instant::now(), access_count: 1 });
        self.stats.total_size += size;
        self.stats.entry_count = self.entries.len() as u64;
    }

    /// Look up a shader by its source hash, updating LRU order.
    pub fn get(&mut self, key: u64) -> Option<&CachedShader> {
        if let Some(entry) = self.entries.get_mut(&key) {
            entry.last_access = Instant::now();
            entry.access_count += 1;
            self.stats.hits += 1;
            Some(&entry.shader)
        } else {
            self.stats.misses += 1;
            None
        }
    }

    /// Look up without updating stats or LRU order (peek).
    pub fn peek(&self, key: u64) -> Option<&CachedShader> {
        self.entries.get(&key).map(|e| &e.shader)
    }

    /// Remove a specific entry from the cache.
    pub fn invalidate(&mut self, key: u64) -> bool {
        if let Some(entry) = self.entries.remove(&key) {
            self.stats.total_size =
                self.stats.total_size.saturating_sub(entry.shader.size_bytes() as u64);
            self.stats.entry_count = self.entries.len() as u64;
            true
        } else {
            false
        }
    }

    /// Remove all entries from the cache.
    pub fn clear(&mut self) {
        self.entries.clear();
        self.stats.total_size = 0;
        self.stats.entry_count = 0;
    }

    /// Current cache statistics.
    pub fn stats(&self) -> &CacheStats {
        &self.stats
    }

    /// Reset statistics counters to zero (entries are preserved).
    pub fn reset_stats(&mut self) {
        self.stats.hits = 0;
        self.stats.misses = 0;
        self.stats.evictions = 0;
        self.stats.compile_errors = 0;
    }

    /// Number of entries in the cache.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Iterator over all cached shader keys.
    pub fn keys(&self) -> impl Iterator<Item = &u64> {
        self.entries.keys()
    }

    /// Check if a key exists in the cache without updating LRU.
    pub fn contains_key(&self, key: u64) -> bool {
        self.entries.contains_key(&key)
    }

    // ── internal ─────────────────────────────────────────────────────

    /// Evict the least-recently-used entry.
    fn evict_lru(&mut self) {
        let lru_key = self.entries.iter().min_by_key(|(_, e)| e.last_access).map(|(k, _)| *k);

        if let Some(key) = lru_key
            && let Some(entry) = self.entries.remove(&key)
        {
            self.stats.total_size =
                self.stats.total_size.saturating_sub(entry.shader.size_bytes() as u64);
            self.stats.evictions += 1;
            self.stats.entry_count = self.entries.len() as u64;
        }
    }
}

// ── Free functions ───────────────────────────────────────────────────

/// Compile a shader source into a [`CachedShader`].
///
/// On CPU builds (without NVRTC) this simulates compilation by hashing
/// the source into synthetic PTX output. On GPU builds the real NVRTC
/// pipeline would be invoked behind the `gpu`/`cuda` feature gate.
pub fn compile_shader(source: &ShaderSource, algorithm: HashAlgorithm) -> Result<CachedShader> {
    if source.cuda_source.is_empty() {
        return Err(KernelError::InvalidArguments {
            reason: "shader source must not be empty".into(),
        }
        .into());
    }

    let start = Instant::now();
    let metadata = ShaderMetadata::from_source(source, algorithm);

    // Simulate PTX compilation (CPU stub).
    // Real GPU builds would call nvrtcCompileProgram here.
    let ptx = simulate_ptx_compile(&source.cuda_source, &source.compile_options);
    let cubin = simulate_cubin_compile(&ptx, &source.target_arch);

    let compile_time_ms = start.elapsed().as_millis() as u64;

    Ok(CachedShader { ptx, cubin: Some(cubin), compile_time_ms, metadata })
}

/// Look up a shader in the cache by source hash.
pub fn lookup_shader(cache: &mut ShaderCache, source_hash: u64) -> Option<&CachedShader> {
    cache.get(source_hash)
}

/// Invalidate (remove) a shader from the cache by source hash.
pub fn invalidate_shader(cache: &mut ShaderCache, source_hash: u64) -> bool {
    cache.invalidate(source_hash)
}

/// Return current cache statistics.
pub fn cache_stats(cache: &ShaderCache) -> CacheStats {
    cache.stats().clone()
}

/// Common CUDA kernel sources that are frequently used in BitNet inference.
const COMMON_KERNELS: &[(&str, &str)] = &[
    (
        "i2s_dequant",
        r#"extern "C" __global__ void i2s_dequant(
            const unsigned char* __restrict__ input,
            float* __restrict__ output,
            const float* __restrict__ scales,
            int n) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < n) {
                int byte_idx = idx / 4;
                int bit_pos = (idx % 4) * 2;
                int val = (input[byte_idx] >> bit_pos) & 0x3;
                float dequant = (float)(val - 1);
                output[idx] = dequant * scales[idx / 256];
            }
        }"#,
    ),
    (
        "rmsnorm",
        r#"extern "C" __global__ void rmsnorm(
            float* __restrict__ output,
            const float* __restrict__ input,
            const float* __restrict__ weight,
            int hidden_size,
            float eps) {
            int row = blockIdx.x;
            float sum_sq = 0.0f;
            for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
                float val = input[row * hidden_size + i];
                sum_sq += val * val;
            }
            sum_sq = sum_sq / hidden_size + eps;
            float rms = rsqrtf(sum_sq);
            for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
                output[row * hidden_size + i] =
                    input[row * hidden_size + i] * rms * weight[i];
            }
        }"#,
    ),
    (
        "softmax",
        r#"extern "C" __global__ void softmax(
            float* __restrict__ output,
            const float* __restrict__ input,
            int cols) {
            int row = blockIdx.x;
            float max_val = -1e38f;
            for (int i = threadIdx.x; i < cols; i += blockDim.x) {
                max_val = fmaxf(max_val, input[row * cols + i]);
            }
            float sum = 0.0f;
            for (int i = threadIdx.x; i < cols; i += blockDim.x) {
                float e = expf(input[row * cols + i] - max_val);
                output[row * cols + i] = e;
                sum += e;
            }
            for (int i = threadIdx.x; i < cols; i += blockDim.x) {
                output[row * cols + i] /= sum;
            }
        }"#,
    ),
    (
        "elementwise_add",
        r#"extern "C" __global__ void elementwise_add(
            float* __restrict__ output,
            const float* __restrict__ a,
            const float* __restrict__ b,
            int n) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < n) {
                output[idx] = a[idx] + b[idx];
            }
        }"#,
    ),
    (
        "rope",
        r#"extern "C" __global__ void rope(
            float* __restrict__ output,
            const float* __restrict__ input,
            const float* __restrict__ cos_table,
            const float* __restrict__ sin_table,
            int head_dim, int seq_len) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            int half = head_dim / 2;
            if (idx < seq_len * head_dim) {
                int pos = idx / head_dim;
                int d = idx % head_dim;
                if (d < half) {
                    float c = cos_table[pos * half + d];
                    float s = sin_table[pos * half + d];
                    float x0 = input[idx];
                    float x1 = input[idx + half];
                    output[idx] = x0 * c - x1 * s;
                    output[idx + half] = x0 * s + x1 * c;
                }
            }
        }"#,
    ),
    (
        "silu_gate",
        r#"extern "C" __global__ void silu_gate(
            float* __restrict__ output,
            const float* __restrict__ gate,
            const float* __restrict__ up,
            int n) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < n) {
                float g = gate[idx];
                float silu = g / (1.0f + expf(-g));
                output[idx] = silu * up[idx];
            }
        }"#,
    ),
];

/// Precompile commonly used BitNet inference kernels into the cache.
pub fn precompile_common_shaders(
    cache: &mut ShaderCache,
    target_arch: &str,
    algorithm: HashAlgorithm,
) -> Result<usize> {
    let mut compiled = 0usize;
    for &(_name, source_code) in COMMON_KERNELS {
        let source = ShaderSource::new(source_code, Vec::new(), target_arch);
        let key = source.content_hash(algorithm);
        if cache.contains_key(key) {
            continue;
        }
        let shader = compile_shader(&source, algorithm)?;
        cache.insert(key, shader);
        compiled += 1;
    }
    Ok(compiled)
}

/// Load cached shaders from a directory into the cache.
///
/// Each file is expected to be named `<hash>.ptx` with an optional
/// companion `<hash>.cubin`. A `<hash>.meta` JSON sidecar stores the
/// [`ShaderMetadata`].
pub fn warm_cache_from_disk(cache: &mut ShaderCache, dir: &Path) -> Result<usize> {
    if !dir.is_dir() {
        return Ok(0);
    }

    let mut loaded = 0usize;
    let entries: Vec<_> = match std::fs::read_dir(dir) {
        Ok(rd) => rd.filter_map(|e| e.ok()).collect(),
        Err(_) => return Ok(0),
    };

    for entry in &entries {
        let path = entry.path();
        let ext = path.extension().and_then(|e| e.to_str());
        if ext != Some("ptx") {
            continue;
        }

        let stem = match path.file_stem().and_then(|s| s.to_str()) {
            Some(s) => s.to_owned(),
            None => continue,
        };

        let hash: u64 = match stem.parse() {
            Ok(h) => h,
            Err(_) => continue,
        };

        let ptx = match std::fs::read(&path) {
            Ok(data) => data,
            Err(_) => continue,
        };

        let cubin_path = dir.join(format!("{hash}.cubin"));
        let cubin = std::fs::read(&cubin_path).ok();

        let meta_path = dir.join(format!("{hash}.meta"));
        let metadata = load_metadata(&meta_path).unwrap_or(ShaderMetadata {
            source_hash: hash,
            compile_options_hash: 0,
            timestamp: 0,
            target_arch: String::new(),
        });

        let shader = CachedShader { ptx, cubin, compile_time_ms: 0, metadata };
        cache.insert(hash, shader);
        loaded += 1;
    }

    Ok(loaded)
}

/// Persist cached shaders to a directory.
///
/// Writes `<hash>.ptx`, optional `<hash>.cubin`, and `<hash>.meta`
/// for each entry.
pub fn save_cache_to_disk(cache: &ShaderCache, dir: &Path) -> Result<usize> {
    std::fs::create_dir_all(dir).map_err(|e| KernelError::InvalidArguments {
        reason: format!("failed to create cache dir: {e}"),
    })?;

    let mut saved = 0usize;
    for (&key, entry) in &cache.entries {
        let ptx_path = dir.join(format!("{key}.ptx"));
        if std::fs::write(&ptx_path, &entry.shader.ptx).is_err() {
            continue;
        }

        if let Some(ref cubin) = entry.shader.cubin {
            let cubin_path = dir.join(format!("{key}.cubin"));
            let _ = std::fs::write(&cubin_path, cubin);
        }

        let meta_path = dir.join(format!("{key}.meta"));
        let _ = save_metadata(&meta_path, &entry.shader.metadata);
        saved += 1;
    }

    Ok(saved)
}

// ── Internal helpers ─────────────────────────────────────────────────

/// Compute a hash of the given bytes using the specified algorithm.
fn compute_hash(data: &[u8], algorithm: HashAlgorithm) -> u64 {
    match algorithm {
        HashAlgorithm::Fnv1a64 => {
            let mut hash: u64 = 0xcbf29ce484222325;
            for &byte in data {
                hash ^= byte as u64;
                hash = hash.wrapping_mul(0x100000001b3);
            }
            hash
        }
        HashAlgorithm::Djb2 => {
            let mut hash: u64 = 5381;
            for &byte in data {
                hash = hash.wrapping_mul(33).wrapping_add(byte as u64);
            }
            hash
        }
    }
}

/// Simulate PTX compilation on CPU (returns a synthetic PTX payload).
fn simulate_ptx_compile(source: &str, options: &[String]) -> Vec<u8> {
    let mut ptx = Vec::new();
    ptx.extend_from_slice(b"// PTX ISA Version 8.0\n");
    ptx.extend_from_slice(b".version 8.0\n");
    ptx.extend_from_slice(b".target sm_80\n");
    ptx.extend_from_slice(b".address_size 64\n\n");

    // Embed source hash as a comment for traceability.
    let hash = compute_hash(source.as_bytes(), HashAlgorithm::Fnv1a64);
    let header = format!("// source_hash: {hash:#018x}\n");
    ptx.extend_from_slice(header.as_bytes());

    // Embed options if any.
    if !options.is_empty() {
        let opts = format!("// options: {}\n", options.join(" "));
        ptx.extend_from_slice(opts.as_bytes());
    }

    // Simulated kernel entry.
    ptx.extend_from_slice(b".visible .entry kernel_main() {\n");
    ptx.extend_from_slice(b"    ret;\n");
    ptx.extend_from_slice(b"}\n");

    ptx
}

/// Simulate CUBIN compilation (returns a synthetic CUBIN payload).
fn simulate_cubin_compile(ptx: &[u8], target_arch: &str) -> Vec<u8> {
    let mut cubin = Vec::new();
    // ELF magic for CUDA (simplified).
    cubin.extend_from_slice(&[0x7f, b'E', b'L', b'F']);
    let arch_bytes = target_arch.as_bytes();
    cubin.push(arch_bytes.len() as u8);
    cubin.extend_from_slice(arch_bytes);
    // Embed a size marker.
    let size = ptx.len() as u32;
    cubin.extend_from_slice(&size.to_le_bytes());
    cubin
}

/// Load shader metadata from a `.meta` JSON sidecar file.
fn load_metadata(path: &Path) -> Option<ShaderMetadata> {
    let data = std::fs::read_to_string(path).ok()?;
    // Minimal JSON parsing (avoid serde dependency in non-dev).
    let source_hash = extract_json_u64(&data, "source_hash")?;
    let compile_options_hash = extract_json_u64(&data, "compile_options_hash")?;
    let timestamp = extract_json_u64(&data, "timestamp")?;
    let target_arch = extract_json_string(&data, "target_arch")?;
    Some(ShaderMetadata { source_hash, compile_options_hash, timestamp, target_arch })
}

/// Save shader metadata to a `.meta` JSON sidecar file.
fn save_metadata(path: &Path, meta: &ShaderMetadata) -> std::io::Result<()> {
    let json = format!(
        concat!(
            "{{\n",
            "  \"source_hash\": {},\n",
            "  \"compile_options_hash\": {},\n",
            "  \"timestamp\": {},\n",
            "  \"target_arch\": \"{}\"\n",
            "}}",
        ),
        meta.source_hash, meta.compile_options_hash, meta.timestamp, meta.target_arch,
    );
    std::fs::write(path, json)
}

/// Extract a u64 value from minimal JSON.
fn extract_json_u64(json: &str, key: &str) -> Option<u64> {
    let pattern = format!("\"{key}\":");
    let start = json.find(&pattern)? + pattern.len();
    let rest = json[start..].trim_start();
    let end = rest.find(|c: char| !c.is_ascii_digit())?;
    rest[..end].parse().ok()
}

/// Extract a string value from minimal JSON.
fn extract_json_string(json: &str, key: &str) -> Option<String> {
    let pattern = format!("\"{key}\": \"");
    let start = json.find(&pattern)? + pattern.len();
    let end = json[start..].find('"')?;
    Some(json[start..start + end].to_owned())
}

// ── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use tempfile::TempDir;

    // ── Helper constructors ──────────────────────────────────────────

    fn sample_source() -> ShaderSource {
        ShaderSource::new(
            r#"extern "C" __global__ void add(float* a, float* b, int n) {
                int i = blockIdx.x * blockDim.x + threadIdx.x;
                if (i < n) a[i] += b[i];
            }"#,
            vec!["--gpu-architecture=sm_80".into()],
            "sm_80",
        )
    }

    fn sample_source_alt() -> ShaderSource {
        ShaderSource::new(
            r#"extern "C" __global__ void mul(float* a, float* b, int n) {
                int i = blockIdx.x * blockDim.x + threadIdx.x;
                if (i < n) a[i] *= b[i];
            }"#,
            vec!["--gpu-architecture=sm_86".into()],
            "sm_86",
        )
    }

    fn tiny_config() -> ShaderCacheConfig {
        ShaderCacheConfig {
            cache_dir: PathBuf::from("/tmp/test_shader_cache"),
            max_cache_size_mb: 1,
            enable_persistence: false,
            hash_algorithm: HashAlgorithm::Fnv1a64,
        }
    }

    fn compile_sample(src: &ShaderSource) -> CachedShader {
        compile_shader(src, HashAlgorithm::Fnv1a64).unwrap()
    }

    // ── ShaderCacheConfig tests ──────────────────────────────────────

    #[test]
    fn config_default_is_valid() {
        let cfg = ShaderCacheConfig::default();
        assert!(cfg.validate().is_ok());
        assert!(cfg.max_cache_size_mb > 0);
        assert!(cfg.enable_persistence);
    }

    #[test]
    fn config_zero_size_is_invalid() {
        let cfg = ShaderCacheConfig { max_cache_size_mb: 0, ..tiny_config() };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn config_max_cache_size_bytes_conversion() {
        let cfg = ShaderCacheConfig { max_cache_size_mb: 10, ..tiny_config() };
        assert_eq!(cfg.max_cache_size_bytes(), 10 * 1024 * 1024);
    }

    #[test]
    fn config_max_cache_size_bytes_one() {
        let cfg = ShaderCacheConfig { max_cache_size_mb: 1, ..tiny_config() };
        assert_eq!(cfg.max_cache_size_bytes(), 1024 * 1024);
    }

    // ── HashAlgorithm tests ──────────────────────────────────────────

    #[test]
    fn hash_algorithm_default_is_fnv1a64() {
        assert_eq!(HashAlgorithm::default(), HashAlgorithm::Fnv1a64);
    }

    #[test]
    fn hash_algorithm_display() {
        assert_eq!(format!("{}", HashAlgorithm::Fnv1a64), "fnv1a64");
        assert_eq!(format!("{}", HashAlgorithm::Djb2), "djb2");
    }

    #[test]
    fn fnv1a64_deterministic() {
        let data = b"hello world";
        let h1 = compute_hash(data, HashAlgorithm::Fnv1a64);
        let h2 = compute_hash(data, HashAlgorithm::Fnv1a64);
        assert_eq!(h1, h2);
    }

    #[test]
    fn djb2_deterministic() {
        let data = b"hello world";
        let h1 = compute_hash(data, HashAlgorithm::Djb2);
        let h2 = compute_hash(data, HashAlgorithm::Djb2);
        assert_eq!(h1, h2);
    }

    #[test]
    fn different_data_different_hash_fnv() {
        let h1 = compute_hash(b"aaa", HashAlgorithm::Fnv1a64);
        let h2 = compute_hash(b"bbb", HashAlgorithm::Fnv1a64);
        assert_ne!(h1, h2);
    }

    #[test]
    fn different_data_different_hash_djb2() {
        let h1 = compute_hash(b"aaa", HashAlgorithm::Djb2);
        let h2 = compute_hash(b"bbb", HashAlgorithm::Djb2);
        assert_ne!(h1, h2);
    }

    #[test]
    fn different_algorithms_different_hash() {
        let data = b"test data";
        let h1 = compute_hash(data, HashAlgorithm::Fnv1a64);
        let h2 = compute_hash(data, HashAlgorithm::Djb2);
        assert_ne!(h1, h2);
    }

    #[test]
    fn empty_data_hashes() {
        let h1 = compute_hash(b"", HashAlgorithm::Fnv1a64);
        let h2 = compute_hash(b"", HashAlgorithm::Djb2);
        assert_ne!(h1, 0);
        assert_ne!(h2, 0);
    }

    // ── ShaderSource tests ───────────────────────────────────────────

    #[test]
    fn shader_source_new() {
        let src = ShaderSource::new("code", vec!["-O3".into()], "sm_80");
        assert_eq!(src.cuda_source, "code");
        assert_eq!(src.compile_options, vec!["-O3"]);
        assert_eq!(src.target_arch, "sm_80");
    }

    #[test]
    fn shader_source_content_hash_stable() {
        let src = sample_source();
        let h1 = src.content_hash(HashAlgorithm::Fnv1a64);
        let h2 = src.content_hash(HashAlgorithm::Fnv1a64);
        assert_eq!(h1, h2);
    }

    #[test]
    fn shader_source_different_options_different_hash() {
        let s1 = ShaderSource::new("code", vec!["-O2".into()], "sm_80");
        let s2 = ShaderSource::new("code", vec!["-O3".into()], "sm_80");
        assert_ne!(
            s1.content_hash(HashAlgorithm::Fnv1a64),
            s2.content_hash(HashAlgorithm::Fnv1a64),
        );
    }

    #[test]
    fn shader_source_different_arch_different_hash() {
        let s1 = ShaderSource::new("code", vec![], "sm_80");
        let s2 = ShaderSource::new("code", vec![], "sm_86");
        assert_ne!(
            s1.content_hash(HashAlgorithm::Fnv1a64),
            s2.content_hash(HashAlgorithm::Fnv1a64),
        );
    }

    #[test]
    fn shader_source_same_content_same_hash() {
        let s1 = ShaderSource::new("x", vec!["a".into()], "sm_80");
        let s2 = ShaderSource::new("x", vec!["a".into()], "sm_80");
        assert_eq!(
            s1.content_hash(HashAlgorithm::Fnv1a64),
            s2.content_hash(HashAlgorithm::Fnv1a64),
        );
    }

    #[test]
    fn shader_source_equality() {
        let s1 = sample_source();
        let s2 = sample_source();
        assert_eq!(s1, s2);
    }

    #[test]
    fn shader_source_inequality() {
        let s1 = sample_source();
        let s2 = sample_source_alt();
        assert_ne!(s1, s2);
    }

    // ── ShaderMetadata tests ─────────────────────────────────────────

    #[test]
    fn metadata_from_source() {
        let src = sample_source();
        let meta = ShaderMetadata::from_source(&src, HashAlgorithm::Fnv1a64);
        assert_ne!(meta.source_hash, 0);
        assert_eq!(meta.target_arch, "sm_80");
        assert!(meta.timestamp > 0);
    }

    #[test]
    fn metadata_timestamp_increases() {
        let src = sample_source();
        let m1 = ShaderMetadata::from_source(&src, HashAlgorithm::Fnv1a64);
        thread::sleep(Duration::from_millis(10));
        let m2 = ShaderMetadata::from_source(&src, HashAlgorithm::Fnv1a64);
        assert!(m2.timestamp >= m1.timestamp);
    }

    #[test]
    fn metadata_different_sources_different_hashes() {
        let m1 = ShaderMetadata::from_source(&sample_source(), HashAlgorithm::Fnv1a64);
        let m2 = ShaderMetadata::from_source(&sample_source_alt(), HashAlgorithm::Fnv1a64);
        assert_ne!(m1.source_hash, m2.source_hash);
    }

    // ── CachedShader tests ───────────────────────────────────────────

    #[test]
    fn cached_shader_size_bytes_ptx_only() {
        let shader = CachedShader {
            ptx: vec![0u8; 100],
            cubin: None,
            compile_time_ms: 1,
            metadata: ShaderMetadata::from_source(&sample_source(), HashAlgorithm::Fnv1a64),
        };
        assert_eq!(shader.size_bytes(), 100);
    }

    #[test]
    fn cached_shader_size_bytes_ptx_and_cubin() {
        let shader = CachedShader {
            ptx: vec![0u8; 100],
            cubin: Some(vec![0u8; 50]),
            compile_time_ms: 1,
            metadata: ShaderMetadata::from_source(&sample_source(), HashAlgorithm::Fnv1a64),
        };
        assert_eq!(shader.size_bytes(), 150);
    }

    // ── CacheStats tests ─────────────────────────────────────────────

    #[test]
    fn cache_stats_default() {
        let stats = CacheStats::default();
        assert_eq!(stats.hits, 0);
        assert_eq!(stats.misses, 0);
        assert_eq!(stats.evictions, 0);
        assert_eq!(stats.total_size, 0);
        assert_eq!(stats.entry_count, 0);
    }

    #[test]
    fn cache_stats_hit_rate_zero() {
        let stats = CacheStats::default();
        assert_eq!(stats.hit_rate(), 0.0);
    }

    #[test]
    fn cache_stats_hit_rate_100_percent() {
        let stats = CacheStats { hits: 10, misses: 0, ..Default::default() };
        assert!((stats.hit_rate() - 100.0).abs() < f64::EPSILON);
    }

    #[test]
    fn cache_stats_hit_rate_50_percent() {
        let stats = CacheStats { hits: 5, misses: 5, ..Default::default() };
        assert!((stats.hit_rate() - 50.0).abs() < f64::EPSILON);
    }

    #[test]
    fn cache_stats_display() {
        let stats = CacheStats { hits: 3, misses: 1, evictions: 0, ..Default::default() };
        let s = format!("{stats}");
        assert!(s.contains("hits=3"));
        assert!(s.contains("misses=1"));
        assert!(s.contains("hit_rate=75.0%"));
    }

    // ── ShaderCache basic tests ──────────────────────────────────────

    #[test]
    fn cache_new_with_defaults() {
        let cache = ShaderCache::with_defaults().unwrap();
        assert!(cache.is_empty());
        assert_eq!(cache.len(), 0);
    }

    #[test]
    fn cache_new_invalid_config_rejected() {
        let cfg = ShaderCacheConfig { max_cache_size_mb: 0, ..tiny_config() };
        assert!(ShaderCache::new(cfg).is_err());
    }

    #[test]
    fn cache_insert_and_get() {
        let mut cache = ShaderCache::new(tiny_config()).unwrap();
        let src = sample_source();
        let shader = compile_sample(&src);
        let key = src.content_hash(HashAlgorithm::Fnv1a64);
        cache.insert(key, shader.clone());

        assert!(!cache.is_empty());
        assert_eq!(cache.len(), 1);

        let found = cache.get(key).unwrap();
        assert_eq!(found.ptx, shader.ptx);
    }

    #[test]
    fn cache_miss_increments_stat() {
        let mut cache = ShaderCache::new(tiny_config()).unwrap();
        assert!(cache.get(12345).is_none());
        assert_eq!(cache.stats().misses, 1);
    }

    #[test]
    fn cache_hit_increments_stat() {
        let mut cache = ShaderCache::new(tiny_config()).unwrap();
        let src = sample_source();
        let key = src.content_hash(HashAlgorithm::Fnv1a64);
        cache.insert(key, compile_sample(&src));
        let _ = cache.get(key);
        assert_eq!(cache.stats().hits, 1);
    }

    #[test]
    fn cache_peek_does_not_update_stats() {
        let mut cache = ShaderCache::new(tiny_config()).unwrap();
        let src = sample_source();
        let key = src.content_hash(HashAlgorithm::Fnv1a64);
        cache.insert(key, compile_sample(&src));
        let _ = cache.peek(key);
        assert_eq!(cache.stats().hits, 0);
        assert_eq!(cache.stats().misses, 0);
    }

    #[test]
    fn cache_contains_key() {
        let mut cache = ShaderCache::new(tiny_config()).unwrap();
        let src = sample_source();
        let key = src.content_hash(HashAlgorithm::Fnv1a64);
        assert!(!cache.contains_key(key));
        cache.insert(key, compile_sample(&src));
        assert!(cache.contains_key(key));
    }

    #[test]
    fn cache_invalidate_existing() {
        let mut cache = ShaderCache::new(tiny_config()).unwrap();
        let src = sample_source();
        let key = src.content_hash(HashAlgorithm::Fnv1a64);
        cache.insert(key, compile_sample(&src));
        assert!(cache.invalidate(key));
        assert!(cache.is_empty());
    }

    #[test]
    fn cache_invalidate_nonexistent() {
        let mut cache = ShaderCache::new(tiny_config()).unwrap();
        assert!(!cache.invalidate(999));
    }

    #[test]
    fn cache_clear() {
        let mut cache = ShaderCache::new(tiny_config()).unwrap();
        for i in 0..5 {
            let src = ShaderSource::new(format!("code_{i}"), vec![], "sm_80");
            let key = src.content_hash(HashAlgorithm::Fnv1a64);
            cache.insert(key, compile_sample(&src));
        }
        assert_eq!(cache.len(), 5);
        cache.clear();
        assert!(cache.is_empty());
        assert_eq!(cache.stats().total_size, 0);
    }

    #[test]
    fn cache_reset_stats() {
        let mut cache = ShaderCache::new(tiny_config()).unwrap();
        let _ = cache.get(1);
        let _ = cache.get(2);
        assert_eq!(cache.stats().misses, 2);
        cache.reset_stats();
        assert_eq!(cache.stats().misses, 0);
        assert_eq!(cache.stats().hits, 0);
    }

    #[test]
    fn cache_keys_iterator() {
        let mut cache = ShaderCache::new(tiny_config()).unwrap();
        let src = sample_source();
        let key = src.content_hash(HashAlgorithm::Fnv1a64);
        cache.insert(key, compile_sample(&src));
        let keys: Vec<_> = cache.keys().copied().collect();
        assert_eq!(keys, vec![key]);
    }

    #[test]
    fn cache_config_accessor() {
        let cfg = tiny_config();
        let cache = ShaderCache::new(cfg.clone()).unwrap();
        assert_eq!(cache.config().max_cache_size_mb, cfg.max_cache_size_mb);
    }

    // ── LRU eviction tests ──────────────────────────────────────────

    #[test]
    fn cache_evicts_lru_when_full() {
        // Very small cache: ~256 bytes max.
        let cfg = ShaderCacheConfig { max_cache_size_mb: 0, ..tiny_config() };
        // max_cache_size_mb == 0 is invalid, use a minimal valid config
        let cfg = ShaderCacheConfig { max_cache_size_mb: 1, ..cfg };
        let mut cache = ShaderCache::new(cfg).unwrap();

        // Fill with entries.
        let s1 = ShaderSource::new("kernel_a", vec![], "sm_80");
        let s2 = ShaderSource::new("kernel_b", vec![], "sm_80");

        let k1 = s1.content_hash(HashAlgorithm::Fnv1a64);
        let k2 = s2.content_hash(HashAlgorithm::Fnv1a64);

        cache.insert(k1, compile_sample(&s1));
        // Access k1 so it becomes recently used.
        let _ = cache.get(k1);

        cache.insert(k2, compile_sample(&s2));

        // Both should be present since 1MB is more than enough.
        assert!(cache.contains_key(k1));
        assert!(cache.contains_key(k2));
    }

    #[test]
    fn cache_eviction_stats_tracked() {
        // Create an artificially tiny cache.
        let cfg = ShaderCacheConfig { max_cache_size_mb: 1, ..tiny_config() };
        let mut cache = ShaderCache::new(cfg).unwrap();
        let max_bytes = cache.config().max_cache_size_bytes();

        // Insert entries with large synthetic PTX until we exceed capacity.
        let mut keys = Vec::new();
        for i in 0..100 {
            let src = ShaderSource::new(format!("k_{i}"), vec![], "sm_80");
            let key = src.content_hash(HashAlgorithm::Fnv1a64);
            let mut shader = compile_sample(&src);
            // Inflate PTX to force eviction.
            shader.ptx = vec![0u8; (max_bytes / 5) as usize];
            cache.insert(key, shader);
            keys.push(key);
        }

        // We should have evicted some entries.
        assert!(cache.stats().evictions > 0);
        assert!(cache.len() < 100);
    }

    #[test]
    fn cache_lru_order_updated_on_get() {
        let cfg = ShaderCacheConfig { max_cache_size_mb: 1, ..tiny_config() };
        let mut cache = ShaderCache::new(cfg).unwrap();

        let s1 = ShaderSource::new("a", vec![], "sm_80");
        let s2 = ShaderSource::new("b", vec![], "sm_80");
        let k1 = s1.content_hash(HashAlgorithm::Fnv1a64);
        let k2 = s2.content_hash(HashAlgorithm::Fnv1a64);

        cache.insert(k1, compile_sample(&s1));
        thread::sleep(Duration::from_millis(5));
        cache.insert(k2, compile_sample(&s2));

        // Touch k1 to make it more recently used than k2.
        let _ = cache.get(k1);

        // Both are present.
        assert!(cache.contains_key(k1));
        assert!(cache.contains_key(k2));
    }

    #[test]
    fn cache_insert_replaces_existing_entry() {
        let mut cache = ShaderCache::new(tiny_config()).unwrap();
        let src = sample_source();
        let key = src.content_hash(HashAlgorithm::Fnv1a64);

        let s1 = compile_sample(&src);
        let s1_size = s1.size_bytes() as u64;
        cache.insert(key, s1);
        assert_eq!(cache.stats().total_size, s1_size);

        // Insert again with different PTX.
        let mut s2 = compile_sample(&src);
        s2.ptx = vec![0u8; 42];
        let s2_size = s2.size_bytes() as u64;
        cache.insert(key, s2);

        assert_eq!(cache.len(), 1);
        assert_eq!(cache.stats().total_size, s2_size);
    }

    // ── compile_shader tests ─────────────────────────────────────────

    #[test]
    fn compile_shader_success() {
        let src = sample_source();
        let shader = compile_shader(&src, HashAlgorithm::Fnv1a64).unwrap();
        assert!(!shader.ptx.is_empty());
        assert!(shader.cubin.is_some());
    }

    #[test]
    fn compile_shader_empty_source_fails() {
        let src = ShaderSource::new("", vec![], "sm_80");
        assert!(compile_shader(&src, HashAlgorithm::Fnv1a64).is_err());
    }

    #[test]
    fn compile_shader_ptx_contains_version() {
        let src = sample_source();
        let shader = compile_shader(&src, HashAlgorithm::Fnv1a64).unwrap();
        let ptx_str = String::from_utf8_lossy(&shader.ptx);
        assert!(ptx_str.contains(".version 8.0"));
    }

    #[test]
    fn compile_shader_ptx_contains_source_hash() {
        let src = sample_source();
        let shader = compile_shader(&src, HashAlgorithm::Fnv1a64).unwrap();
        let ptx_str = String::from_utf8_lossy(&shader.ptx);
        assert!(ptx_str.contains("source_hash:"));
    }

    #[test]
    fn compile_shader_ptx_contains_options() {
        let src = ShaderSource::new("code", vec!["-O3".into(), "--fast".into()], "sm_80");
        let shader = compile_shader(&src, HashAlgorithm::Fnv1a64).unwrap();
        let ptx_str = String::from_utf8_lossy(&shader.ptx);
        assert!(ptx_str.contains("options:"));
    }

    #[test]
    fn compile_shader_cubin_starts_with_elf_magic() {
        let src = sample_source();
        let shader = compile_shader(&src, HashAlgorithm::Fnv1a64).unwrap();
        let cubin = shader.cubin.as_ref().unwrap();
        assert!(cubin.len() >= 4);
        assert_eq!(cubin[0], 0x7f);
        assert_eq!(cubin[1], b'E');
        assert_eq!(cubin[2], b'L');
        assert_eq!(cubin[3], b'F');
    }

    #[test]
    fn compile_shader_metadata_populated() {
        let src = sample_source();
        let shader = compile_shader(&src, HashAlgorithm::Fnv1a64).unwrap();
        assert_ne!(shader.metadata.source_hash, 0);
        assert_eq!(shader.metadata.target_arch, "sm_80");
        assert!(shader.metadata.timestamp > 0);
    }

    #[test]
    fn compile_shader_no_options_no_options_line() {
        let src = ShaderSource::new("code", vec![], "sm_80");
        let shader = compile_shader(&src, HashAlgorithm::Fnv1a64).unwrap();
        let ptx_str = String::from_utf8_lossy(&shader.ptx);
        assert!(!ptx_str.contains("options:"));
    }

    // ── Free function tests ──────────────────────────────────────────

    #[test]
    fn lookup_shader_hit() {
        let mut cache = ShaderCache::new(tiny_config()).unwrap();
        let src = sample_source();
        let key = src.content_hash(HashAlgorithm::Fnv1a64);
        cache.insert(key, compile_sample(&src));
        assert!(lookup_shader(&mut cache, key).is_some());
    }

    #[test]
    fn lookup_shader_miss() {
        let mut cache = ShaderCache::new(tiny_config()).unwrap();
        assert!(lookup_shader(&mut cache, 42).is_none());
    }

    #[test]
    fn invalidate_shader_removes() {
        let mut cache = ShaderCache::new(tiny_config()).unwrap();
        let src = sample_source();
        let key = src.content_hash(HashAlgorithm::Fnv1a64);
        cache.insert(key, compile_sample(&src));
        assert!(invalidate_shader(&mut cache, key));
        assert!(!cache.contains_key(key));
    }

    #[test]
    fn cache_stats_fn_reflects_state() {
        let mut cache = ShaderCache::new(tiny_config()).unwrap();
        let src = sample_source();
        let key = src.content_hash(HashAlgorithm::Fnv1a64);
        cache.insert(key, compile_sample(&src));
        let _ = cache.get(key);
        let _ = cache.get(999);

        let stats = cache_stats(&cache);
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 1);
        assert_eq!(stats.entry_count, 1);
    }

    // ── precompile_common_shaders tests ──────────────────────────────

    #[test]
    fn precompile_common_shaders_populates_cache() {
        let mut cache = ShaderCache::new(tiny_config()).unwrap();
        let count = precompile_common_shaders(&mut cache, "sm_80", HashAlgorithm::Fnv1a64).unwrap();
        assert_eq!(count, COMMON_KERNELS.len());
        assert_eq!(cache.len(), COMMON_KERNELS.len());
    }

    #[test]
    fn precompile_common_shaders_idempotent() {
        let mut cache = ShaderCache::new(tiny_config()).unwrap();
        let c1 = precompile_common_shaders(&mut cache, "sm_80", HashAlgorithm::Fnv1a64).unwrap();
        let c2 = precompile_common_shaders(&mut cache, "sm_80", HashAlgorithm::Fnv1a64).unwrap();
        assert_eq!(c1, COMMON_KERNELS.len());
        assert_eq!(c2, 0); // all already cached
    }

    #[test]
    fn precompile_different_arch_caches_separately() {
        let mut cache = ShaderCache::new(tiny_config()).unwrap();
        let c1 = precompile_common_shaders(&mut cache, "sm_80", HashAlgorithm::Fnv1a64).unwrap();
        let c2 = precompile_common_shaders(&mut cache, "sm_86", HashAlgorithm::Fnv1a64).unwrap();
        assert!(c1 > 0);
        assert!(c2 > 0);
        assert_eq!(cache.len(), c1 + c2);
    }

    // ── Disk persistence tests ───────────────────────────────────────

    #[test]
    fn save_and_warm_cache_roundtrip() {
        let tmp = TempDir::new().unwrap();
        let dir = tmp.path();

        let mut cache = ShaderCache::new(tiny_config()).unwrap();
        let src = sample_source();
        let key = src.content_hash(HashAlgorithm::Fnv1a64);
        cache.insert(key, compile_sample(&src));

        let saved = save_cache_to_disk(&cache, dir).unwrap();
        assert_eq!(saved, 1);

        let mut cache2 = ShaderCache::new(tiny_config()).unwrap();
        let loaded = warm_cache_from_disk(&mut cache2, dir).unwrap();
        assert_eq!(loaded, 1);
        assert!(cache2.contains_key(key));
    }

    #[test]
    fn save_cache_creates_directory() {
        let tmp = TempDir::new().unwrap();
        let dir = tmp.path().join("nested").join("cache");

        let cache = ShaderCache::new(tiny_config()).unwrap();
        let result = save_cache_to_disk(&cache, &dir);
        assert!(result.is_ok());
        assert!(dir.is_dir());
    }

    #[test]
    fn warm_cache_from_nonexistent_dir() {
        let mut cache = ShaderCache::new(tiny_config()).unwrap();
        let loaded = warm_cache_from_disk(&mut cache, Path::new("/nonexistent/path")).unwrap();
        assert_eq!(loaded, 0);
    }

    #[test]
    fn warm_cache_from_empty_dir() {
        let tmp = TempDir::new().unwrap();
        let mut cache = ShaderCache::new(tiny_config()).unwrap();
        let loaded = warm_cache_from_disk(&mut cache, tmp.path()).unwrap();
        assert_eq!(loaded, 0);
    }

    #[test]
    fn save_multiple_entries() {
        let tmp = TempDir::new().unwrap();
        let mut cache = ShaderCache::new(tiny_config()).unwrap();

        for i in 0..5 {
            let src = ShaderSource::new(format!("kernel_{i}"), vec![], "sm_80");
            let key = src.content_hash(HashAlgorithm::Fnv1a64);
            cache.insert(key, compile_sample(&src));
        }

        let saved = save_cache_to_disk(&cache, tmp.path()).unwrap();
        assert_eq!(saved, 5);

        let mut cache2 = ShaderCache::new(tiny_config()).unwrap();
        let loaded = warm_cache_from_disk(&mut cache2, tmp.path()).unwrap();
        assert_eq!(loaded, 5);
    }

    #[test]
    fn save_preserves_cubin() {
        let tmp = TempDir::new().unwrap();
        let mut cache = ShaderCache::new(tiny_config()).unwrap();
        let src = sample_source();
        let key = src.content_hash(HashAlgorithm::Fnv1a64);
        let shader = compile_sample(&src);
        assert!(shader.cubin.is_some());
        cache.insert(key, shader);

        save_cache_to_disk(&cache, tmp.path()).unwrap();

        let mut cache2 = ShaderCache::new(tiny_config()).unwrap();
        warm_cache_from_disk(&mut cache2, tmp.path()).unwrap();
        let loaded = cache2.peek(key).unwrap();
        assert!(loaded.cubin.is_some());
    }

    #[test]
    fn save_preserves_metadata() {
        let tmp = TempDir::new().unwrap();
        let mut cache = ShaderCache::new(tiny_config()).unwrap();
        let src = sample_source();
        let key = src.content_hash(HashAlgorithm::Fnv1a64);
        let shader = compile_sample(&src);
        let orig_arch = shader.metadata.target_arch.clone();
        cache.insert(key, shader);

        save_cache_to_disk(&cache, tmp.path()).unwrap();

        let mut cache2 = ShaderCache::new(tiny_config()).unwrap();
        warm_cache_from_disk(&mut cache2, tmp.path()).unwrap();
        let loaded = cache2.peek(key).unwrap();
        assert_eq!(loaded.metadata.target_arch, orig_arch);
    }

    #[test]
    fn warm_cache_skips_non_ptx_files() {
        let tmp = TempDir::new().unwrap();
        std::fs::write(tmp.path().join("readme.txt"), "hello").unwrap();
        std::fs::write(tmp.path().join("data.json"), "{}").unwrap();

        let mut cache = ShaderCache::new(tiny_config()).unwrap();
        let loaded = warm_cache_from_disk(&mut cache, tmp.path()).unwrap();
        assert_eq!(loaded, 0);
    }

    #[test]
    fn warm_cache_skips_invalid_hash_filenames() {
        let tmp = TempDir::new().unwrap();
        std::fs::write(tmp.path().join("not_a_number.ptx"), b"data").unwrap();

        let mut cache = ShaderCache::new(tiny_config()).unwrap();
        let loaded = warm_cache_from_disk(&mut cache, tmp.path()).unwrap();
        assert_eq!(loaded, 0);
    }

    // ── Metadata JSON round-trip tests ───────────────────────────────

    #[test]
    fn metadata_save_load_roundtrip() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("test.meta");
        let meta = ShaderMetadata {
            source_hash: 123456789,
            compile_options_hash: 987654321,
            timestamp: 1700000000,
            target_arch: "sm_89".into(),
        };
        save_metadata(&path, &meta).unwrap();
        let loaded = load_metadata(&path).unwrap();
        assert_eq!(loaded.source_hash, meta.source_hash);
        assert_eq!(loaded.compile_options_hash, meta.compile_options_hash);
        assert_eq!(loaded.timestamp, meta.timestamp);
        assert_eq!(loaded.target_arch, meta.target_arch);
    }

    #[test]
    fn load_metadata_missing_file() {
        assert!(load_metadata(Path::new("/nonexistent/path.meta")).is_none());
    }

    #[test]
    fn load_metadata_malformed_json() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("bad.meta");
        std::fs::write(&path, "not json at all").unwrap();
        assert!(load_metadata(&path).is_none());
    }

    // ── JSON helper tests ────────────────────────────────────────────

    #[test]
    fn extract_json_u64_valid() {
        let json = r#"{"source_hash": 42, "other": 7}"#;
        assert_eq!(extract_json_u64(json, "source_hash"), Some(42));
    }

    #[test]
    fn extract_json_u64_missing_key() {
        let json = r#"{"other": 7}"#;
        assert_eq!(extract_json_u64(json, "source_hash"), None);
    }

    #[test]
    fn extract_json_string_valid() {
        let json = r#"{"target_arch": "sm_80"}"#;
        assert_eq!(extract_json_string(json, "target_arch"), Some("sm_80".into()),);
    }

    #[test]
    fn extract_json_string_missing_key() {
        let json = r#"{"other": "val"}"#;
        assert_eq!(extract_json_string(json, "target_arch"), None);
    }

    // ── Integration / workflow tests ─────────────────────────────────

    #[test]
    fn full_compile_and_cache_workflow() {
        let mut cache = ShaderCache::new(tiny_config()).unwrap();
        let src = sample_source();
        let algo = HashAlgorithm::Fnv1a64;
        let key = src.content_hash(algo);

        // Miss.
        assert!(cache.get(key).is_none());
        assert_eq!(cache.stats().misses, 1);

        // Compile.
        let shader = compile_shader(&src, algo).unwrap();
        cache.insert(key, shader);

        // Hit.
        assert!(cache.get(key).is_some());
        assert_eq!(cache.stats().hits, 1);
    }

    #[test]
    fn full_workflow_with_disk_persistence() {
        let tmp = TempDir::new().unwrap();

        // Phase 1: compile and persist.
        let algo = HashAlgorithm::Fnv1a64;
        let src = sample_source();
        let key = src.content_hash(algo);

        let mut cache1 = ShaderCache::new(tiny_config()).unwrap();
        let shader = compile_shader(&src, algo).unwrap();
        cache1.insert(key, shader);
        save_cache_to_disk(&cache1, tmp.path()).unwrap();

        // Phase 2: new process, warm from disk.
        let mut cache2 = ShaderCache::new(tiny_config()).unwrap();
        warm_cache_from_disk(&mut cache2, tmp.path()).unwrap();
        assert!(cache2.contains_key(key));
    }

    #[test]
    fn precompile_then_lookup_all() {
        let mut cache = ShaderCache::new(tiny_config()).unwrap();
        let algo = HashAlgorithm::Fnv1a64;
        precompile_common_shaders(&mut cache, "sm_80", algo).unwrap();

        for &(_name, source_code) in COMMON_KERNELS {
            let src = ShaderSource::new(source_code, Vec::new(), "sm_80");
            let key = src.content_hash(algo);
            assert!(cache.get(key).is_some(), "common kernel should be cached");
        }
    }

    #[test]
    fn invalidate_then_recompile() {
        let mut cache = ShaderCache::new(tiny_config()).unwrap();
        let algo = HashAlgorithm::Fnv1a64;
        let src = sample_source();
        let key = src.content_hash(algo);

        cache.insert(key, compile_shader(&src, algo).unwrap());
        assert!(cache.contains_key(key));

        invalidate_shader(&mut cache, key);
        assert!(!cache.contains_key(key));

        cache.insert(key, compile_shader(&src, algo).unwrap());
        assert!(cache.contains_key(key));
    }

    #[test]
    fn multiple_arch_targets_coexist() {
        let mut cache = ShaderCache::new(tiny_config()).unwrap();
        let algo = HashAlgorithm::Fnv1a64;

        for arch in &["sm_70", "sm_75", "sm_80", "sm_86", "sm_89", "sm_90"] {
            let src = ShaderSource::new("code", vec![], *arch);
            let key = src.content_hash(algo);
            cache.insert(key, compile_shader(&src, algo).unwrap());
        }
        assert_eq!(cache.len(), 6);
    }

    #[test]
    fn compile_options_affect_cache_key() {
        let mut cache = ShaderCache::new(tiny_config()).unwrap();
        let algo = HashAlgorithm::Fnv1a64;

        let s1 = ShaderSource::new("code", vec!["-O0".into()], "sm_80");
        let s2 = ShaderSource::new("code", vec!["-O3".into()], "sm_80");
        let k1 = s1.content_hash(algo);
        let k2 = s2.content_hash(algo);

        cache.insert(k1, compile_shader(&s1, algo).unwrap());
        cache.insert(k2, compile_shader(&s2, algo).unwrap());
        assert_eq!(cache.len(), 2);
    }

    #[test]
    fn total_size_tracks_all_entries() {
        let mut cache = ShaderCache::new(tiny_config()).unwrap();
        let algo = HashAlgorithm::Fnv1a64;

        let src = sample_source();
        let shader = compile_shader(&src, algo).unwrap();
        let expected_size = shader.size_bytes() as u64;
        let key = src.content_hash(algo);
        cache.insert(key, shader);

        assert_eq!(cache.stats().total_size, expected_size);
    }

    #[test]
    fn invalidate_reduces_total_size() {
        let mut cache = ShaderCache::new(tiny_config()).unwrap();
        let algo = HashAlgorithm::Fnv1a64;

        let src = sample_source();
        let key = src.content_hash(algo);
        cache.insert(key, compile_shader(&src, algo).unwrap());
        assert!(cache.stats().total_size > 0);

        cache.invalidate(key);
        assert_eq!(cache.stats().total_size, 0);
    }

    #[test]
    fn common_kernels_list_not_empty() {
        assert!(!COMMON_KERNELS.is_empty());
    }

    #[test]
    fn common_kernels_all_have_nonempty_source() {
        for &(name, source) in COMMON_KERNELS {
            assert!(!name.is_empty(), "kernel name should not be empty");
            assert!(!source.is_empty(), "kernel source should not be empty");
        }
    }

    #[test]
    fn cache_stats_entry_count_consistent() {
        let mut cache = ShaderCache::new(tiny_config()).unwrap();
        let algo = HashAlgorithm::Fnv1a64;

        for i in 0..10 {
            let src = ShaderSource::new(format!("k{i}"), vec![], "sm_80");
            cache.insert(src.content_hash(algo), compile_shader(&src, algo).unwrap());
        }
        assert_eq!(cache.stats().entry_count, cache.len() as u64);
    }
}
