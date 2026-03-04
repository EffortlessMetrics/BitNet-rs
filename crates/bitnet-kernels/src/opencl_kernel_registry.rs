//! Centralized registry for OpenCL kernels with metadata, versioning,
//! and dispatch.
//!
//! Provides [`KernelRegistry`] for registering / looking-up kernel
//! entries, [`CompiledKernelCache`] for caching compiled programs, and
//! [`KernelDispatcher`] for selecting and executing kernels by name
//! with runtime parameters.

use std::collections::HashMap;
use std::fmt;

// ── KernelVersion ───────────────────────────────────────────────────

/// Semantic version for a kernel source.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct KernelVersion {
    pub major: u32,
    pub minor: u32,
    pub patch: u32,
}

impl KernelVersion {
    pub fn new(major: u32, minor: u32, patch: u32) -> Self {
        Self { major, minor, patch }
    }
}

impl fmt::Display for KernelVersion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}.{}.{}", self.major, self.minor, self.patch)
    }
}

impl PartialOrd for KernelVersion {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for KernelVersion {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.major
            .cmp(&other.major)
            .then(self.minor.cmp(&other.minor))
            .then(self.patch.cmp(&other.patch))
    }
}

// ── KernelCategory ──────────────────────────────────────────────────

/// Broad category of an OpenCL kernel.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum KernelCategory {
    Matmul,
    Attention,
    Norm,
    Activation,
    Reduce,
    Quantize,
    Custom,
}

impl fmt::Display for KernelCategory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Matmul => write!(f, "Matmul"),
            Self::Attention => write!(f, "Attention"),
            Self::Norm => write!(f, "Norm"),
            Self::Activation => write!(f, "Activation"),
            Self::Reduce => write!(f, "Reduce"),
            Self::Quantize => write!(f, "Quantize"),
            Self::Custom => write!(f, "Custom"),
        }
    }
}

// ── KernelMetadata ──────────────────────────────────────────────────

/// Static metadata about a kernel.
#[derive(Debug, Clone)]
pub struct KernelMetadata {
    /// Which category this kernel belongs to.
    pub category: KernelCategory,
    /// Descriptive input type names (e.g. `["f32", "i8"]`).
    pub input_types: Vec<String>,
    /// Descriptive output type names.
    pub output_types: Vec<String>,
    /// Estimated FLOPs for a reference problem size (0 = unknown).
    pub flop_estimate: u64,
    /// Estimated memory traffic in bytes (0 = unknown).
    pub memory_estimate: u64,
}

// ── KernelEntry ─────────────────────────────────────────────────────

/// A single kernel registration in the registry.
#[derive(Debug, Clone)]
pub struct KernelEntry {
    /// Kernel name (unique key).
    pub name: String,
    /// OpenCL C source code.
    pub source: String,
    /// Compiler build options (e.g. `"-DTILE=16"`).
    pub build_options: String,
    /// Recommended work-group size (0 = let the driver decide).
    pub workgroup_hint: usize,
    /// Required OpenCL extensions (e.g. `"cl_khr_fp16"`).
    pub required_extensions: Vec<String>,
    /// Semantic version of this kernel source.
    pub version: KernelVersion,
    /// Metadata.
    pub metadata: KernelMetadata,
}

// ── KernelRegistry ──────────────────────────────────────────────────

/// Error type returned by registry operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RegistryError {
    /// The kernel name is already registered.
    DuplicateName(String),
    /// No kernel with the given name was found.
    NotFound(String),
    /// A required extension is not available.
    MissingExtension { kernel: String, extension: String },
    /// The kernel source failed to compile.
    CompileFailed { kernel: String, reason: String },
}

impl fmt::Display for RegistryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DuplicateName(n) => write!(f, "kernel '{n}' is already registered"),
            Self::NotFound(n) => write!(f, "kernel '{n}' not found"),
            Self::MissingExtension { kernel, extension } => {
                write!(f, "kernel '{kernel}' requires extension '{extension}'")
            }
            Self::CompileFailed { kernel, reason } => {
                write!(f, "kernel '{kernel}' compile failed: {reason}")
            }
        }
    }
}

impl std::error::Error for RegistryError {}

/// HashMap-based kernel registry with register / lookup / list / remove.
pub struct KernelRegistry {
    entries: HashMap<String, KernelEntry>,
}

impl KernelRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self { entries: HashMap::new() }
    }

    /// Register a kernel. Returns `Err` if `entry.name` already exists.
    pub fn register(&mut self, entry: KernelEntry) -> Result<(), RegistryError> {
        if self.entries.contains_key(&entry.name) {
            return Err(RegistryError::DuplicateName(entry.name.clone()));
        }
        self.entries.insert(entry.name.clone(), entry);
        Ok(())
    }

    /// Look up a kernel by name.
    pub fn lookup(&self, name: &str) -> Result<&KernelEntry, RegistryError> {
        self.entries.get(name).ok_or_else(|| RegistryError::NotFound(name.to_string()))
    }

    /// List all registered kernel names (sorted for determinism).
    pub fn list(&self) -> Vec<&str> {
        let mut names: Vec<&str> = self.entries.keys().map(|s| s.as_str()).collect();
        names.sort_unstable();
        names
    }

    /// List kernels belonging to a specific category.
    pub fn list_by_category(&self, category: KernelCategory) -> Vec<&KernelEntry> {
        let mut out: Vec<&KernelEntry> =
            self.entries.values().filter(|e| e.metadata.category == category).collect();
        out.sort_by(|a, b| a.name.cmp(&b.name));
        out
    }

    /// Remove a kernel by name. Returns `Err(NotFound)` if absent.
    pub fn remove(&mut self, name: &str) -> Result<KernelEntry, RegistryError> {
        self.entries.remove(name).ok_or_else(|| RegistryError::NotFound(name.to_string()))
    }

    /// Replace an existing kernel. The new entry must have the same
    /// name as an already-registered kernel, otherwise `NotFound`.
    pub fn replace(&mut self, entry: KernelEntry) -> Result<KernelEntry, RegistryError> {
        if !self.entries.contains_key(&entry.name) {
            return Err(RegistryError::NotFound(entry.name.clone()));
        }
        // unwrap is safe — we just checked contains_key
        let old = self.entries.insert(entry.name.clone(), entry).unwrap();
        Ok(old)
    }

    /// Total number of registered kernels.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the registry is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Collect [`RegistryStats`] for the current state.
    pub fn stats(&self) -> RegistryStats {
        let mut by_category: HashMap<KernelCategory, usize> = HashMap::new();
        for entry in self.entries.values() {
            *by_category.entry(entry.metadata.category).or_insert(0) += 1;
        }
        RegistryStats {
            total_kernels: self.entries.len(),
            by_category,
            compile_cache_hit_rate: 0.0,
        }
    }
}

impl Default for KernelRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// ── CompiledKernelCache ─────────────────────────────────────────────

/// Cache key for a compiled kernel: (name, device_id, build_options).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CompileCacheKey {
    pub name: String,
    pub device_id: String,
    pub options: String,
}

/// Opaque handle representing a compiled kernel program.
#[derive(Debug, Clone)]
pub struct CompiledKernel {
    /// The key that produced this entry.
    pub key: CompileCacheKey,
    /// Binary blob (would be real program bytes with an OpenCL runtime).
    pub binary: Vec<u8>,
}

/// Caches compiled kernel programs keyed by
/// `(kernel_name, device_id, build_options)`.
pub struct CompiledKernelCache {
    cache: HashMap<CompileCacheKey, CompiledKernel>,
    hits: u64,
    misses: u64,
}

impl CompiledKernelCache {
    pub fn new() -> Self {
        Self { cache: HashMap::new(), hits: 0, misses: 0 }
    }

    /// Try to retrieve a previously compiled kernel.
    pub fn get(&mut self, key: &CompileCacheKey) -> Option<&CompiledKernel> {
        if self.cache.contains_key(key) {
            self.hits += 1;
            self.cache.get(key)
        } else {
            self.misses += 1;
            None
        }
    }

    /// Insert a compiled kernel into the cache.
    pub fn insert(&mut self, compiled: CompiledKernel) {
        self.cache.insert(compiled.key.clone(), compiled);
    }

    /// Total entries in the cache.
    pub fn len(&self) -> usize {
        self.cache.len()
    }

    pub fn is_empty(&self) -> bool {
        self.cache.is_empty()
    }

    /// Cache hit count.
    pub fn hits(&self) -> u64 {
        self.hits
    }

    /// Cache miss count.
    pub fn misses(&self) -> u64 {
        self.misses
    }

    /// Hit rate as a fraction in `[0.0, 1.0]`. Returns 0.0 when no
    /// lookups have been performed.
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 { 0.0 } else { self.hits as f64 / total as f64 }
    }

    /// Remove all cached entries and reset counters.
    pub fn clear(&mut self) {
        self.cache.clear();
        self.hits = 0;
        self.misses = 0;
    }
}

impl Default for CompiledKernelCache {
    fn default() -> Self {
        Self::new()
    }
}

// ── RegistryStats ───────────────────────────────────────────────────

/// Snapshot of registry statistics.
#[derive(Debug, Clone)]
pub struct RegistryStats {
    pub total_kernels: usize,
    pub by_category: HashMap<KernelCategory, usize>,
    pub compile_cache_hit_rate: f64,
}

impl RegistryStats {
    /// Build stats combining registry and cache state.
    pub fn from_registry_and_cache(registry: &KernelRegistry, cache: &CompiledKernelCache) -> Self {
        let mut s = registry.stats();
        s.compile_cache_hit_rate = cache.hit_rate();
        s
    }
}

// ── KernelDispatcher ────────────────────────────────────────────────

/// Runtime parameters passed to a dispatched kernel.
#[derive(Debug, Clone)]
pub struct DispatchParams {
    /// Number of elements in the problem (drives global work size).
    pub elements: usize,
    /// Optional override for the work-group size.
    pub workgroup_override: Option<usize>,
    /// Extra build options appended at dispatch time.
    pub extra_options: String,
    /// Device identifier for the compile cache.
    pub device_id: String,
}

impl Default for DispatchParams {
    fn default() -> Self {
        Self {
            elements: 0,
            workgroup_override: None,
            extra_options: String::new(),
            device_id: "cpu-ref".to_string(),
        }
    }
}

/// Result of dispatching a kernel (CPU reference path).
#[derive(Debug)]
pub struct DispatchResult {
    /// Name of the kernel that was dispatched.
    pub kernel_name: String,
    /// Output buffer produced by the CPU reference implementation.
    pub output: Vec<f32>,
    /// Whether the compiled kernel was served from cache.
    pub cache_hit: bool,
}

/// Selects and dispatches a kernel from the registry.
pub struct KernelDispatcher {
    registry: KernelRegistry,
    cache: CompiledKernelCache,
}

impl KernelDispatcher {
    pub fn new(registry: KernelRegistry) -> Self {
        Self { registry, cache: CompiledKernelCache::new() }
    }

    /// Dispatch a kernel by name using the CPU reference path.
    pub fn dispatch(
        &mut self,
        name: &str,
        input: &[f32],
        params: &DispatchParams,
    ) -> Result<DispatchResult, RegistryError> {
        let entry = self.registry.lookup(name)?;

        let build_opts = if params.extra_options.is_empty() {
            entry.build_options.clone()
        } else {
            format!("{} {}", entry.build_options, params.extra_options)
        };

        let key = CompileCacheKey {
            name: name.to_string(),
            device_id: params.device_id.clone(),
            options: build_opts,
        };

        let cache_hit = self.cache.get(&key).is_some();
        if !cache_hit {
            // "Compile" — in CPU-ref mode we just store a dummy binary.
            let compiled =
                CompiledKernel { key: key.clone(), binary: entry.source.as_bytes().to_vec() };
            self.cache.insert(compiled);
        }

        let category = entry.metadata.category;
        let output = cpu_reference_dispatch(category, input, params);

        Ok(DispatchResult { kernel_name: name.to_string(), output, cache_hit })
    }

    /// Access the underlying registry.
    pub fn registry(&self) -> &KernelRegistry {
        &self.registry
    }

    /// Access the underlying compiled-kernel cache.
    pub fn cache(&self) -> &CompiledKernelCache {
        &self.cache
    }

    /// Mutable access to the registry (e.g. to register more kernels).
    pub fn registry_mut(&mut self) -> &mut KernelRegistry {
        &mut self.registry
    }

    /// Collect combined stats.
    pub fn stats(&self) -> RegistryStats {
        RegistryStats::from_registry_and_cache(&self.registry, &self.cache)
    }
}

// ── CPU reference implementations ───────────────────────────────────

/// Trivial CPU reference implementations used when no OpenCL runtime
/// is available. These are *not* optimised — they exist purely for
/// correctness testing and fallback.
fn cpu_reference_dispatch(
    category: KernelCategory,
    input: &[f32],
    params: &DispatchParams,
) -> Vec<f32> {
    match category {
        KernelCategory::Matmul => cpu_ref_matmul(input, params.elements),
        KernelCategory::Activation => cpu_ref_activation(input),
        KernelCategory::Norm => cpu_ref_norm(input),
        KernelCategory::Reduce => cpu_ref_reduce(input),
        KernelCategory::Quantize => cpu_ref_quantize(input),
        KernelCategory::Attention | KernelCategory::Custom => {
            // Pass-through for categories without a simple scalar ref.
            input.to_vec()
        }
    }
}

/// CPU reference: element-wise multiply (placeholder for matmul).
fn cpu_ref_matmul(input: &[f32], _elements: usize) -> Vec<f32> {
    input.iter().map(|&v| v * v).collect()
}

/// CPU reference: ReLU activation.
fn cpu_ref_activation(input: &[f32]) -> Vec<f32> {
    input.iter().map(|&v| v.max(0.0)).collect()
}

/// CPU reference: L2 normalisation.
fn cpu_ref_norm(input: &[f32]) -> Vec<f32> {
    let sq_sum: f64 = input.iter().map(|&v| (v as f64) * (v as f64)).sum();
    let norm = (sq_sum + 1e-12).sqrt();
    input.iter().map(|&v| (v as f64 / norm) as f32).collect()
}

/// CPU reference: sum reduction → single-element output.
fn cpu_ref_reduce(input: &[f32]) -> Vec<f32> {
    let sum: f64 = input.iter().map(|&v| v as f64).sum();
    vec![sum as f32]
}

/// CPU reference: fake "quantize" — clamp to [-1, 1].
fn cpu_ref_quantize(input: &[f32]) -> Vec<f32> {
    input.iter().map(|&v| v.clamp(-1.0, 1.0)).collect()
}

// ── Helper to build KernelEntry concisely in tests ──────────────────

/// Build a minimal [`KernelEntry`] for testing.
pub fn make_entry(name: &str, category: KernelCategory) -> KernelEntry {
    KernelEntry {
        name: name.to_string(),
        source: format!("__kernel void {name}() {{}}"),
        build_options: String::new(),
        workgroup_hint: 256,
        required_extensions: vec![],
        version: KernelVersion::new(1, 0, 0),
        metadata: KernelMetadata {
            category,
            input_types: vec!["f32".to_string()],
            output_types: vec!["f32".to_string()],
            flop_estimate: 0,
            memory_estimate: 0,
        },
    }
}

// ═══════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    // ── helper ──────────────────────────────────────────────────────

    fn entry(name: &str, cat: KernelCategory) -> KernelEntry {
        make_entry(name, cat)
    }

    fn entry_v(name: &str, cat: KernelCategory, v: (u32, u32, u32)) -> KernelEntry {
        let mut e = entry(name, cat);
        e.version = KernelVersion::new(v.0, v.1, v.2);
        e
    }

    fn default_params() -> DispatchParams {
        DispatchParams::default()
    }

    // ── Register & lookup ───────────────────────────────────────────

    #[test]
    fn register_and_lookup() {
        let mut reg = KernelRegistry::new();
        reg.register(entry("matmul", KernelCategory::Matmul)).unwrap();
        let e = reg.lookup("matmul").unwrap();
        assert_eq!(e.name, "matmul");
    }

    #[test]
    fn lookup_returns_correct_category() {
        let mut reg = KernelRegistry::new();
        reg.register(entry("softmax", KernelCategory::Activation)).unwrap();
        let e = reg.lookup("softmax").unwrap();
        assert_eq!(e.metadata.category, KernelCategory::Activation);
    }

    #[test]
    fn lookup_returns_correct_source() {
        let mut reg = KernelRegistry::new();
        reg.register(entry("relu", KernelCategory::Activation)).unwrap();
        let e = reg.lookup("relu").unwrap();
        assert!(e.source.contains("relu"));
    }

    #[test]
    fn lookup_unknown_kernel_errors() {
        let reg = KernelRegistry::new();
        let err = reg.lookup("nope").unwrap_err();
        assert_eq!(err, RegistryError::NotFound("nope".to_string()));
    }

    // ── Duplicate name ──────────────────────────────────────────────

    #[test]
    fn register_duplicate_errors() {
        let mut reg = KernelRegistry::new();
        reg.register(entry("dup", KernelCategory::Matmul)).unwrap();
        let res = reg.register(entry("dup", KernelCategory::Norm));
        assert_eq!(res, Err(RegistryError::DuplicateName("dup".to_string())));
    }

    // ── Empty registry ──────────────────────────────────────────────

    #[test]
    fn empty_registry_is_empty() {
        let reg = KernelRegistry::new();
        assert!(reg.is_empty());
        assert_eq!(reg.len(), 0);
    }

    #[test]
    fn empty_registry_list_empty() {
        let reg = KernelRegistry::new();
        assert!(reg.list().is_empty());
    }

    // ── List all ────────────────────────────────────────────────────

    #[test]
    fn list_returns_sorted_names() {
        let mut reg = KernelRegistry::new();
        reg.register(entry("z_kern", KernelCategory::Reduce)).unwrap();
        reg.register(entry("a_kern", KernelCategory::Matmul)).unwrap();
        reg.register(entry("m_kern", KernelCategory::Norm)).unwrap();
        assert_eq!(reg.list(), vec!["a_kern", "m_kern", "z_kern"]);
    }

    #[test]
    fn list_length_matches_len() {
        let mut reg = KernelRegistry::new();
        for i in 0..5 {
            reg.register(entry(&format!("k{i}"), KernelCategory::Custom)).unwrap();
        }
        assert_eq!(reg.list().len(), reg.len());
    }

    // ── Category filtering ──────────────────────────────────────────

    #[test]
    fn list_by_category_only_matching() {
        let mut reg = KernelRegistry::new();
        reg.register(entry("mm1", KernelCategory::Matmul)).unwrap();
        reg.register(entry("mm2", KernelCategory::Matmul)).unwrap();
        reg.register(entry("act", KernelCategory::Activation)).unwrap();

        let matmuls = reg.list_by_category(KernelCategory::Matmul);
        assert_eq!(matmuls.len(), 2);
        for e in &matmuls {
            assert_eq!(e.metadata.category, KernelCategory::Matmul);
        }
    }

    #[test]
    fn list_by_category_empty_when_none() {
        let mut reg = KernelRegistry::new();
        reg.register(entry("mm1", KernelCategory::Matmul)).unwrap();
        assert!(reg.list_by_category(KernelCategory::Norm).is_empty());
    }

    #[test]
    fn list_by_category_sorted_by_name() {
        let mut reg = KernelRegistry::new();
        reg.register(entry("z_norm", KernelCategory::Norm)).unwrap();
        reg.register(entry("a_norm", KernelCategory::Norm)).unwrap();
        let norms = reg.list_by_category(KernelCategory::Norm);
        assert_eq!(norms[0].name, "a_norm");
        assert_eq!(norms[1].name, "z_norm");
    }

    // ── Remove / replace ────────────────────────────────────────────

    #[test]
    fn remove_existing_kernel() {
        let mut reg = KernelRegistry::new();
        reg.register(entry("rm_me", KernelCategory::Reduce)).unwrap();
        let removed = reg.remove("rm_me").unwrap();
        assert_eq!(removed.name, "rm_me");
        assert!(reg.lookup("rm_me").is_err());
    }

    #[test]
    fn remove_unknown_kernel_errors() {
        let mut reg = KernelRegistry::new();
        let err = reg.remove("ghost").unwrap_err();
        assert_eq!(err, RegistryError::NotFound("ghost".to_string()));
    }

    #[test]
    fn replace_existing_kernel() {
        let mut reg = KernelRegistry::new();
        let v1 = entry_v("kern", KernelCategory::Matmul, (1, 0, 0));
        reg.register(v1).unwrap();

        let mut v2 = entry_v("kern", KernelCategory::Matmul, (2, 0, 0));
        v2.source = "__kernel void kern() { /* v2 */ }".to_string();
        let old = reg.replace(v2).unwrap();
        assert_eq!(old.version, KernelVersion::new(1, 0, 0));

        let cur = reg.lookup("kern").unwrap();
        assert_eq!(cur.version, KernelVersion::new(2, 0, 0));
        assert!(cur.source.contains("v2"));
    }

    #[test]
    fn replace_missing_kernel_errors() {
        let mut reg = KernelRegistry::new();
        let e = entry("missing", KernelCategory::Matmul);
        let err = reg.replace(e).unwrap_err();
        assert_eq!(err, RegistryError::NotFound("missing".to_string()));
    }

    // ── Version comparison ──────────────────────────────────────────

    #[test]
    fn version_display() {
        let v = KernelVersion::new(1, 2, 3);
        assert_eq!(v.to_string(), "1.2.3");
    }

    #[test]
    fn version_ord_major() {
        let a = KernelVersion::new(1, 0, 0);
        let b = KernelVersion::new(2, 0, 0);
        assert!(a < b);
    }

    #[test]
    fn version_ord_minor() {
        let a = KernelVersion::new(1, 1, 0);
        let b = KernelVersion::new(1, 2, 0);
        assert!(a < b);
    }

    #[test]
    fn version_ord_patch() {
        let a = KernelVersion::new(1, 0, 1);
        let b = KernelVersion::new(1, 0, 2);
        assert!(a < b);
    }

    #[test]
    fn version_eq() {
        let a = KernelVersion::new(3, 2, 1);
        let b = KernelVersion::new(3, 2, 1);
        assert_eq!(a, b);
    }

    #[test]
    fn version_ne() {
        assert_ne!(KernelVersion::new(1, 0, 0), KernelVersion::new(0, 1, 0));
    }

    #[test]
    fn version_sorting() {
        let mut versions = vec![
            KernelVersion::new(2, 0, 0),
            KernelVersion::new(1, 1, 0),
            KernelVersion::new(1, 0, 1),
        ];
        versions.sort();
        assert_eq!(
            versions,
            vec![
                KernelVersion::new(1, 0, 1),
                KernelVersion::new(1, 1, 0),
                KernelVersion::new(2, 0, 0),
            ]
        );
    }

    // ── CompiledKernelCache ─────────────────────────────────────────

    #[test]
    fn cache_miss_increments_counter() {
        let mut cache = CompiledKernelCache::new();
        let key = CompileCacheKey { name: "k".into(), device_id: "d".into(), options: "".into() };
        assert!(cache.get(&key).is_none());
        assert_eq!(cache.misses(), 1);
        assert_eq!(cache.hits(), 0);
    }

    #[test]
    fn cache_hit_increments_counter() {
        let mut cache = CompiledKernelCache::new();
        let key = CompileCacheKey { name: "k".into(), device_id: "d".into(), options: "".into() };
        cache.insert(CompiledKernel { key: key.clone(), binary: vec![1] });
        assert!(cache.get(&key).is_some());
        assert_eq!(cache.hits(), 1);
        assert_eq!(cache.misses(), 0);
    }

    #[test]
    fn cache_hit_rate_zero_when_empty() {
        let cache = CompiledKernelCache::new();
        assert_eq!(cache.hit_rate(), 0.0);
    }

    #[test]
    fn cache_hit_rate_one_after_all_hits() {
        let mut cache = CompiledKernelCache::new();
        let key = CompileCacheKey { name: "k".into(), device_id: "d".into(), options: "".into() };
        cache.insert(CompiledKernel { key: key.clone(), binary: vec![] });
        cache.get(&key);
        cache.get(&key);
        assert!((cache.hit_rate() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn cache_hit_rate_half() {
        let mut cache = CompiledKernelCache::new();
        let key = CompileCacheKey { name: "k".into(), device_id: "d".into(), options: "".into() };
        // miss
        cache.get(&key);
        // insert + hit
        cache.insert(CompiledKernel { key: key.clone(), binary: vec![] });
        cache.get(&key);
        assert!((cache.hit_rate() - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn cache_clear_resets() {
        let mut cache = CompiledKernelCache::new();
        let key = CompileCacheKey { name: "k".into(), device_id: "d".into(), options: "".into() };
        cache.insert(CompiledKernel { key: key.clone(), binary: vec![42] });
        cache.get(&key);
        cache.clear();
        assert!(cache.is_empty());
        assert_eq!(cache.hits(), 0);
        assert_eq!(cache.misses(), 0);
    }

    #[test]
    fn cache_different_devices_are_separate() {
        let mut cache = CompiledKernelCache::new();
        let k1 = CompileCacheKey { name: "k".into(), device_id: "gpu0".into(), options: "".into() };
        let k2 = CompileCacheKey { name: "k".into(), device_id: "gpu1".into(), options: "".into() };
        cache.insert(CompiledKernel { key: k1.clone(), binary: vec![1] });
        assert!(cache.get(&k1).is_some());
        assert!(cache.get(&k2).is_none());
    }

    #[test]
    fn cache_different_options_are_separate() {
        let mut cache = CompiledKernelCache::new();
        let k1 = CompileCacheKey {
            name: "k".into(),
            device_id: "d".into(),
            options: "-DTILE=16".into(),
        };
        let k2 = CompileCacheKey {
            name: "k".into(),
            device_id: "d".into(),
            options: "-DTILE=32".into(),
        };
        cache.insert(CompiledKernel { key: k1.clone(), binary: vec![1] });
        assert!(cache.get(&k1).is_some());
        assert!(cache.get(&k2).is_none());
    }

    // ── Metadata validation ─────────────────────────────────────────

    #[test]
    fn metadata_category_roundtrip() {
        let m = KernelMetadata {
            category: KernelCategory::Quantize,
            input_types: vec!["f32".into()],
            output_types: vec!["i8".into()],
            flop_estimate: 1024,
            memory_estimate: 512,
        };
        assert_eq!(m.category, KernelCategory::Quantize);
        assert_eq!(m.flop_estimate, 1024);
        assert_eq!(m.memory_estimate, 512);
    }

    #[test]
    fn metadata_input_output_types() {
        let m = KernelMetadata {
            category: KernelCategory::Matmul,
            input_types: vec!["f32".into(), "i8".into()],
            output_types: vec!["f32".into()],
            flop_estimate: 0,
            memory_estimate: 0,
        };
        assert_eq!(m.input_types.len(), 2);
        assert_eq!(m.output_types.len(), 1);
    }

    // ── Dispatcher ──────────────────────────────────────────────────

    #[test]
    fn dispatcher_selects_correct_kernel() {
        let mut reg = KernelRegistry::new();
        reg.register(entry("relu", KernelCategory::Activation)).unwrap();
        reg.register(entry("matmul", KernelCategory::Matmul)).unwrap();

        let mut disp = KernelDispatcher::new(reg);
        let res = disp.dispatch("relu", &[1.0, -2.0, 3.0], &default_params()).unwrap();
        assert_eq!(res.kernel_name, "relu");
    }

    #[test]
    fn dispatcher_activation_relu_output() {
        let mut reg = KernelRegistry::new();
        reg.register(entry("relu", KernelCategory::Activation)).unwrap();

        let mut disp = KernelDispatcher::new(reg);
        let res = disp.dispatch("relu", &[-1.0, 0.0, 3.5], &default_params()).unwrap();
        assert_eq!(res.output, vec![0.0, 0.0, 3.5]);
    }

    #[test]
    fn dispatcher_matmul_output() {
        let mut reg = KernelRegistry::new();
        reg.register(entry("mm", KernelCategory::Matmul)).unwrap();

        let mut disp = KernelDispatcher::new(reg);
        let res = disp.dispatch("mm", &[2.0, 3.0], &default_params()).unwrap();
        assert_eq!(res.output, vec![4.0, 9.0]);
    }

    #[test]
    fn dispatcher_unknown_kernel_errors() {
        let reg = KernelRegistry::new();
        let mut disp = KernelDispatcher::new(reg);
        assert!(disp.dispatch("nope", &[1.0], &default_params()).is_err());
    }

    #[test]
    fn dispatcher_cache_miss_then_hit() {
        let mut reg = KernelRegistry::new();
        reg.register(entry("k", KernelCategory::Reduce)).unwrap();
        let mut disp = KernelDispatcher::new(reg);

        let r1 = disp.dispatch("k", &[1.0, 2.0], &default_params()).unwrap();
        assert!(!r1.cache_hit);

        let r2 = disp.dispatch("k", &[1.0, 2.0], &default_params()).unwrap();
        assert!(r2.cache_hit);
    }

    #[test]
    fn dispatcher_reduce_output() {
        let mut reg = KernelRegistry::new();
        reg.register(entry("sum", KernelCategory::Reduce)).unwrap();
        let mut disp = KernelDispatcher::new(reg);
        let res = disp.dispatch("sum", &[1.0, 2.0, 3.0], &default_params()).unwrap();
        assert_eq!(res.output.len(), 1);
        assert!((res.output[0] - 6.0).abs() < 1e-6);
    }

    #[test]
    fn dispatcher_quantize_clamps() {
        let mut reg = KernelRegistry::new();
        reg.register(entry("quant", KernelCategory::Quantize)).unwrap();
        let mut disp = KernelDispatcher::new(reg);
        let res = disp.dispatch("quant", &[-5.0, 0.5, 10.0], &default_params()).unwrap();
        assert_eq!(res.output, vec![-1.0, 0.5, 1.0]);
    }

    #[test]
    fn dispatcher_norm_output_unit_length() {
        let mut reg = KernelRegistry::new();
        reg.register(entry("norm", KernelCategory::Norm)).unwrap();
        let mut disp = KernelDispatcher::new(reg);
        let res = disp.dispatch("norm", &[3.0, 4.0], &default_params()).unwrap();
        let sq: f64 = res.output.iter().map(|v| (*v as f64).powi(2)).sum();
        assert!((sq - 1.0).abs() < 1e-5);
    }

    #[test]
    fn dispatcher_custom_passthrough() {
        let mut reg = KernelRegistry::new();
        reg.register(entry("pass", KernelCategory::Custom)).unwrap();
        let mut disp = KernelDispatcher::new(reg);
        let res = disp.dispatch("pass", &[7.0, 8.0], &default_params()).unwrap();
        assert_eq!(res.output, vec![7.0, 8.0]);
    }

    #[test]
    fn dispatcher_attention_passthrough() {
        let mut reg = KernelRegistry::new();
        reg.register(entry("attn", KernelCategory::Attention)).unwrap();
        let mut disp = KernelDispatcher::new(reg);
        let res = disp.dispatch("attn", &[1.0, 2.0], &default_params()).unwrap();
        assert_eq!(res.output, vec![1.0, 2.0]);
    }

    // ── Stats ───────────────────────────────────────────────────────

    #[test]
    fn stats_total_kernels() {
        let mut reg = KernelRegistry::new();
        reg.register(entry("a", KernelCategory::Matmul)).unwrap();
        reg.register(entry("b", KernelCategory::Norm)).unwrap();
        let s = reg.stats();
        assert_eq!(s.total_kernels, 2);
    }

    #[test]
    fn stats_by_category() {
        let mut reg = KernelRegistry::new();
        reg.register(entry("m1", KernelCategory::Matmul)).unwrap();
        reg.register(entry("m2", KernelCategory::Matmul)).unwrap();
        reg.register(entry("n1", KernelCategory::Norm)).unwrap();
        let s = reg.stats();
        assert_eq!(s.by_category[&KernelCategory::Matmul], 2);
        assert_eq!(s.by_category[&KernelCategory::Norm], 1);
    }

    #[test]
    fn stats_combined_with_cache() {
        let mut reg = KernelRegistry::new();
        reg.register(entry("k", KernelCategory::Reduce)).unwrap();

        let mut cache = CompiledKernelCache::new();
        let key = CompileCacheKey { name: "k".into(), device_id: "d".into(), options: "".into() };
        cache.insert(CompiledKernel { key: key.clone(), binary: vec![] });
        cache.get(&key); // hit

        let s = RegistryStats::from_registry_and_cache(&reg, &cache);
        assert_eq!(s.total_kernels, 1);
        assert!((s.compile_cache_hit_rate - 1.0).abs() < f64::EPSILON);
    }

    // ── KernelCategory Display ──────────────────────────────────────

    #[test]
    fn category_display_matmul() {
        assert_eq!(KernelCategory::Matmul.to_string(), "Matmul");
    }

    #[test]
    fn category_display_attention() {
        assert_eq!(KernelCategory::Attention.to_string(), "Attention");
    }

    #[test]
    fn category_display_norm() {
        assert_eq!(KernelCategory::Norm.to_string(), "Norm");
    }

    #[test]
    fn category_display_activation() {
        assert_eq!(KernelCategory::Activation.to_string(), "Activation");
    }

    #[test]
    fn category_display_reduce() {
        assert_eq!(KernelCategory::Reduce.to_string(), "Reduce");
    }

    #[test]
    fn category_display_quantize() {
        assert_eq!(KernelCategory::Quantize.to_string(), "Quantize");
    }

    #[test]
    fn category_display_custom() {
        assert_eq!(KernelCategory::Custom.to_string(), "Custom");
    }

    // ── RegistryError Display ───────────────────────────────────────

    #[test]
    fn error_display_duplicate() {
        let e = RegistryError::DuplicateName("dup".into());
        assert!(e.to_string().contains("dup"));
        assert!(e.to_string().contains("already registered"));
    }

    #[test]
    fn error_display_not_found() {
        let e = RegistryError::NotFound("x".into());
        assert!(e.to_string().contains("not found"));
    }

    #[test]
    fn error_display_missing_ext() {
        let e =
            RegistryError::MissingExtension { kernel: "k".into(), extension: "cl_khr_fp16".into() };
        let s = e.to_string();
        assert!(s.contains("cl_khr_fp16"));
    }

    #[test]
    fn error_display_compile_failed() {
        let e = RegistryError::CompileFailed { kernel: "k".into(), reason: "syntax".into() };
        assert!(e.to_string().contains("syntax"));
    }

    #[test]
    fn error_is_std_error() {
        let e: Box<dyn std::error::Error> = Box::new(RegistryError::NotFound("x".into()));
        assert!(!e.to_string().is_empty());
    }

    // ── Property-style tests ────────────────────────────────────────

    #[test]
    fn registered_kernel_always_found() {
        let mut reg = KernelRegistry::new();
        let names: Vec<String> = (0..20).map(|i| format!("kern_{i}")).collect();
        for n in &names {
            reg.register(entry(n, KernelCategory::Matmul)).unwrap();
        }
        for n in &names {
            assert!(reg.lookup(n).is_ok());
        }
    }

    #[test]
    fn remove_then_reregister_ok() {
        let mut reg = KernelRegistry::new();
        reg.register(entry("tmp", KernelCategory::Reduce)).unwrap();
        reg.remove("tmp").unwrap();
        // Should succeed after removal.
        reg.register(entry("tmp", KernelCategory::Reduce)).unwrap();
        assert!(reg.lookup("tmp").is_ok());
    }

    #[test]
    fn len_tracks_mutations() {
        let mut reg = KernelRegistry::new();
        assert_eq!(reg.len(), 0);
        reg.register(entry("a", KernelCategory::Matmul)).unwrap();
        assert_eq!(reg.len(), 1);
        reg.register(entry("b", KernelCategory::Matmul)).unwrap();
        assert_eq!(reg.len(), 2);
        reg.remove("a").unwrap();
        assert_eq!(reg.len(), 1);
    }

    // ── Default impls ───────────────────────────────────────────────

    #[test]
    fn registry_default_is_empty() {
        let reg = KernelRegistry::default();
        assert!(reg.is_empty());
    }

    #[test]
    fn cache_default_is_empty() {
        let cache = CompiledKernelCache::default();
        assert!(cache.is_empty());
    }

    // ── Edge cases ──────────────────────────────────────────────────

    #[test]
    fn dispatch_empty_input() {
        let mut reg = KernelRegistry::new();
        reg.register(entry("relu", KernelCategory::Activation)).unwrap();
        let mut disp = KernelDispatcher::new(reg);
        let res = disp.dispatch("relu", &[], &default_params()).unwrap();
        assert!(res.output.is_empty());
    }

    #[test]
    fn dispatch_with_extra_options() {
        let mut reg = KernelRegistry::new();
        let mut e = entry("mm", KernelCategory::Matmul);
        e.build_options = "-DTILE=16".to_string();
        reg.register(e).unwrap();

        let mut disp = KernelDispatcher::new(reg);
        let params = DispatchParams { extra_options: "-DUNROLL=4".into(), ..Default::default() };
        let r1 = disp.dispatch("mm", &[2.0], &params).unwrap();
        assert!(!r1.cache_hit);

        // Same params → cache hit
        let r2 = disp.dispatch("mm", &[2.0], &params).unwrap();
        assert!(r2.cache_hit);
    }

    #[test]
    fn version_zero_zero_zero() {
        let v = KernelVersion::new(0, 0, 0);
        assert_eq!(v.to_string(), "0.0.0");
    }

    #[test]
    fn entry_with_extensions() {
        let mut e = entry("fp16_mm", KernelCategory::Matmul);
        e.required_extensions = vec!["cl_khr_fp16".to_string()];
        assert_eq!(e.required_extensions.len(), 1);
    }

    #[test]
    fn dispatcher_stats_reflect_state() {
        let mut reg = KernelRegistry::new();
        reg.register(entry("k", KernelCategory::Reduce)).unwrap();
        let mut disp = KernelDispatcher::new(reg);
        disp.dispatch("k", &[1.0], &default_params()).unwrap();
        disp.dispatch("k", &[1.0], &default_params()).unwrap();

        let s = disp.stats();
        assert_eq!(s.total_kernels, 1);
        // 1 miss then 1 hit → 50%
        assert!((s.compile_cache_hit_rate - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn workgroup_hint_default_256() {
        let e = entry("k", KernelCategory::Matmul);
        assert_eq!(e.workgroup_hint, 256);
    }

    #[test]
    fn cache_len_after_inserts() {
        let mut cache = CompiledKernelCache::new();
        for i in 0..3 {
            let key = CompileCacheKey {
                name: format!("k{i}"),
                device_id: "d".into(),
                options: "".into(),
            };
            cache.insert(CompiledKernel { key, binary: vec![] });
        }
        assert_eq!(cache.len(), 3);
    }
}
