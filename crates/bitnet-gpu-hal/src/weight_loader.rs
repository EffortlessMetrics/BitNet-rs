//! Weight loading infrastructure for GPU inference.
//!
//! Provides lazy loading, prefetch scheduling, multi-shard parallel loading,
//! format detection, and conversion utilities for model weight tensors.

use std::collections::HashMap;
use std::fmt;
use std::time::{Duration, Instant};

// ── Weight format ───────────────────────────────────────────────────────────

/// Supported weight serialisation formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WeightFormat {
    /// GGUF (GGML Universal File) format.
    GGUF,
    /// `SafeTensors` (`HuggingFace`) format.
    SafeTensors,
    /// `PyTorch` pickle-based `.bin` / `.pt` format.
    PyTorch,
    /// ONNX protobuf format.
    ONNX,
    /// `NumPy` `.npy` / `.npz` format.
    NumPy,
    /// Raw binary blob with external metadata.
    Raw,
}

impl fmt::Display for WeightFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::GGUF => write!(f, "GGUF"),
            Self::SafeTensors => write!(f, "SafeTensors"),
            Self::PyTorch => write!(f, "PyTorch"),
            Self::ONNX => write!(f, "ONNX"),
            Self::NumPy => write!(f, "NumPy"),
            Self::Raw => write!(f, "Raw"),
        }
    }
}

impl WeightFormat {
    /// Infer format from a file extension.
    pub fn from_extension(ext: &str) -> Option<Self> {
        match ext.to_ascii_lowercase().as_str() {
            "gguf" => Some(Self::GGUF),
            "safetensors" => Some(Self::SafeTensors),
            "bin" | "pt" | "pth" => Some(Self::PyTorch),
            "onnx" => Some(Self::ONNX),
            "npy" | "npz" => Some(Self::NumPy),
            "raw" => Some(Self::Raw),
            _ => None,
        }
    }

    /// Canonical file extension for this format.
    pub const fn extension(&self) -> &'static str {
        match self {
            Self::GGUF => "gguf",
            Self::SafeTensors => "safetensors",
            Self::PyTorch => "bin",
            Self::ONNX => "onnx",
            Self::NumPy => "npy",
            Self::Raw => "raw",
        }
    }
}

// ── Data type ───────────────────────────────────────────────────────────────

/// Element data types for tensor storage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DataType {
    F32,
    F16,
    BF16,
    I8,
    I2,
    I4,
    U8,
    Bool,
}

impl DataType {
    /// Size of a single element in bytes (packed types return sub-byte as 1).
    pub const fn element_size_bytes(&self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 | Self::BF16 => 2,
            // Sub-byte types return 1 as the minimum addressable unit.
            Self::I8 | Self::U8 | Self::Bool | Self::I2 | Self::I4 => 1,
        }
    }
}

// ── Loader configuration ────────────────────────────────────────────────────

/// Configuration for the weight loading pipeline.
#[derive(Debug, Clone)]
#[allow(clippy::struct_excessive_bools)]
pub struct LoaderConfig {
    /// Expected on-disk format.
    pub format: WeightFormat,
    /// Use memory-mapped I/O when possible.
    pub memory_map: bool,
    /// Prefetch tensors that are likely needed soon.
    pub prefetch: bool,
    /// Load independent tensors in parallel.
    pub parallel_load: bool,
    /// Verify embedded checksums (if format supports them).
    pub verify_checksums: bool,
    /// Maximum number of parallel I/O operations.
    pub max_io_parallelism: usize,
}

impl Default for LoaderConfig {
    fn default() -> Self {
        Self {
            format: WeightFormat::GGUF,
            memory_map: true,
            prefetch: true,
            parallel_load: true,
            verify_checksums: true,
            max_io_parallelism: 4,
        }
    }
}

// ── Tensor metadata ─────────────────────────────────────────────────────────

/// Metadata describing a single tensor in a weight file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorInfo {
    /// Fully-qualified tensor name (e.g. `"layers.0.attention.wq.weight"`).
    pub name: String,
    /// Dimension sizes in row-major order.
    pub shape: Vec<usize>,
    /// Element data type.
    pub dtype: DataType,
    /// Byte offset from the start of the data section.
    pub offset: u64,
    /// Total size in bytes on disk.
    pub size_bytes: u64,
}

impl TensorInfo {
    /// Number of logical elements in the tensor.
    pub fn num_elements(&self) -> usize {
        self.shape.iter().copied().product()
    }

    /// Number of dimensions.
    pub const fn ndim(&self) -> usize {
        self.shape.len()
    }
}

// ── Weight map ──────────────────────────────────────────────────────────────

/// Maps tensor names to their metadata and (optionally loaded) data.
#[derive(Debug, Clone)]
pub struct WeightMap {
    /// Ordered list of tensor metadata.
    tensors: Vec<TensorInfo>,
    /// Name → index lookup.
    name_index: HashMap<String, usize>,
    /// Loaded data keyed by tensor name.
    data: HashMap<String, Vec<u8>>,
    /// Source format.
    format: WeightFormat,
}

impl WeightMap {
    /// Create a new weight map for the given format with pre-scanned metadata.
    pub fn new(format: WeightFormat, tensors: Vec<TensorInfo>) -> Self {
        let name_index = tensors.iter().enumerate().map(|(i, t)| (t.name.clone(), i)).collect();
        Self { tensors, name_index, data: HashMap::new(), format }
    }

    /// Number of tensors in the map.
    pub const fn len(&self) -> usize {
        self.tensors.len()
    }

    /// Whether the map contains zero tensors.
    pub const fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }

    /// Look up tensor metadata by name.
    pub fn get_info(&self, name: &str) -> Option<&TensorInfo> {
        self.name_index.get(name).map(|&i| &self.tensors[i])
    }

    /// Iterator over all tensor metadata entries.
    pub fn tensor_infos(&self) -> &[TensorInfo] {
        &self.tensors
    }

    /// Source format of this weight map.
    pub const fn format(&self) -> WeightFormat {
        self.format
    }

    /// Store loaded bytes for a tensor.
    pub fn insert_data(&mut self, name: &str, bytes: Vec<u8>) {
        self.data.insert(name.to_string(), bytes);
    }

    /// Retrieve loaded bytes for a tensor.
    pub fn get_data(&self, name: &str) -> Option<&[u8]> {
        self.data.get(name).map(Vec::as_slice)
    }

    /// Whether the raw bytes for a given tensor have been loaded.
    pub fn is_loaded(&self, name: &str) -> bool {
        self.data.contains_key(name)
    }

    /// Number of tensors whose data has been loaded.
    pub fn loaded_count(&self) -> usize {
        self.data.len()
    }

    /// Total bytes loaded across all tensors.
    pub fn total_loaded_bytes(&self) -> usize {
        self.data.values().map(Vec::len).sum()
    }

    /// List tensor names that have not yet been loaded.
    pub fn pending_names(&self) -> Vec<&str> {
        self.tensors
            .iter()
            .filter(|t| !self.data.contains_key(&t.name))
            .map(|t| t.name.as_str())
            .collect()
    }
}

// ── Lazy loader ─────────────────────────────────────────────────────────────

/// Load state for a single tensor slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LoadState {
    /// Metadata known but data not yet requested.
    NotLoaded,
    /// Data has been requested but not yet available.
    Loading,
    /// Data is resident in memory.
    Loaded,
    /// Load failed.
    Failed,
}

/// Lazy loader that materialises tensor data on first access.
#[derive(Debug)]
pub struct LazyLoader {
    config: LoaderConfig,
    weight_map: WeightMap,
    states: HashMap<String, LoadState>,
    access_order: Vec<String>,
}

impl LazyLoader {
    /// Create a lazy loader over the given weight map.
    pub fn new(config: LoaderConfig, weight_map: WeightMap) -> Self {
        let states = weight_map
            .tensor_infos()
            .iter()
            .map(|t| (t.name.clone(), LoadState::NotLoaded))
            .collect();
        Self { config, weight_map, states, access_order: Vec::new() }
    }

    /// Current load state for a tensor.
    pub fn state(&self, name: &str) -> Option<LoadState> {
        self.states.get(name).copied()
    }

    /// Record an access (simulates triggering a lazy load).
    pub fn request_load(&mut self, name: &str) -> bool {
        if !self.states.contains_key(name) {
            return false;
        }
        let state = self.states.get_mut(name).unwrap();
        if *state == LoadState::NotLoaded {
            *state = LoadState::Loading;
            self.access_order.push(name.to_string());
        }
        true
    }

    /// Mark a tensor as loaded, storing its data.
    pub fn complete_load(&mut self, name: &str, data: Vec<u8>) -> bool {
        match self.states.get_mut(name) {
            Some(state) if *state == LoadState::Loading => {
                self.weight_map.insert_data(name, data);
                *state = LoadState::Loaded;
                true
            }
            _ => false,
        }
    }

    /// Mark a tensor load as failed.
    pub fn fail_load(&mut self, name: &str) -> bool {
        match self.states.get_mut(name) {
            Some(state) if *state == LoadState::Loading => {
                *state = LoadState::Failed;
                true
            }
            _ => false,
        }
    }

    /// Retrieve loaded data for a tensor.
    pub fn get_data(&self, name: &str) -> Option<&[u8]> {
        self.weight_map.get_data(name)
    }

    /// Number of tensors in each load state.
    pub fn state_counts(&self) -> HashMap<LoadState, usize> {
        let mut counts = HashMap::new();
        for &s in self.states.values() {
            *counts.entry(s).or_insert(0) += 1;
        }
        counts
    }

    /// Access order (order in which tensors were first requested).
    pub fn access_order(&self) -> &[String] {
        &self.access_order
    }

    /// Reference to the underlying config.
    pub const fn config(&self) -> &LoaderConfig {
        &self.config
    }

    /// Reference to the underlying weight map.
    pub const fn weight_map(&self) -> &WeightMap {
        &self.weight_map
    }
}

// ── Prefetch scheduler ──────────────────────────────────────────────────────

/// Strategy for deciding which tensors to prefetch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrefetchStrategy {
    /// Prefetch the next N tensors in declaration order.
    Sequential,
    /// Prefetch tensors that share a layer index with the current request.
    LayerLocal,
    /// No prefetching.
    Disabled,
}

/// Schedules prefetching of upcoming tensors.
#[derive(Debug)]
pub struct PrefetchScheduler {
    strategy: PrefetchStrategy,
    window_size: usize,
    tensor_names: Vec<String>,
    prefetched: HashMap<String, bool>,
    hits: u64,
    misses: u64,
}

impl PrefetchScheduler {
    /// Create a new scheduler over the ordered tensor names.
    pub fn new(strategy: PrefetchStrategy, window_size: usize, tensor_names: Vec<String>) -> Self {
        Self { strategy, window_size, tensor_names, prefetched: HashMap::new(), hits: 0, misses: 0 }
    }

    /// Determine which tensors should be prefetched given the last accessed
    /// tensor. Returns names of tensors to prefetch.
    pub fn next_prefetch(&self, current_name: &str) -> Vec<String> {
        match self.strategy {
            PrefetchStrategy::Disabled => Vec::new(),
            PrefetchStrategy::Sequential => {
                let pos = self.tensor_names.iter().position(|n| n == current_name);
                pos.map_or_else(Vec::new, |idx| {
                    self.tensor_names[idx + 1..].iter().take(self.window_size).cloned().collect()
                })
            }
            PrefetchStrategy::LayerLocal => {
                let layer = extract_layer_index(current_name);
                layer.map_or_else(Vec::new, |layer_idx| {
                    self.tensor_names
                        .iter()
                        .filter(|n| {
                            n.as_str() != current_name
                                && extract_layer_index(n) == Some(layer_idx)
                                && !self.prefetched.contains_key(n.as_str())
                        })
                        .take(self.window_size)
                        .cloned()
                        .collect()
                })
            }
        }
    }

    /// Record that a tensor was prefetched.
    pub fn mark_prefetched(&mut self, name: &str) {
        self.prefetched.insert(name.to_string(), true);
    }

    /// Record a prefetch hit (requested tensor was already prefetched).
    pub const fn record_hit(&mut self) {
        self.hits += 1;
    }

    /// Record a prefetch miss.
    pub const fn record_miss(&mut self) {
        self.misses += 1;
    }

    /// Total prefetch hits.
    pub const fn hits(&self) -> u64 {
        self.hits
    }

    /// Total prefetch misses.
    pub const fn misses(&self) -> u64 {
        self.misses
    }

    /// Prefetch accuracy as a ratio in `[0.0, 1.0]`.
    pub fn accuracy(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            return 0.0;
        }
        #[allow(clippy::cast_precision_loss)]
        let acc = self.hits as f64 / total as f64;
        acc
    }

    /// Active strategy.
    pub const fn strategy(&self) -> PrefetchStrategy {
        self.strategy
    }

    /// Window size.
    pub const fn window_size(&self) -> usize {
        self.window_size
    }
}

/// Extract a layer index from a tensor name like `"layers.5.attention.wq"`.
fn extract_layer_index(name: &str) -> Option<usize> {
    for part in name.split('.') {
        if let Ok(idx) = part.parse::<usize>() {
            return Some(idx);
        }
    }
    None
}

// ── Shard info ──────────────────────────────────────────────────────────────

/// Describes one shard in a sharded weight set.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WeightShardInfo {
    /// Zero-based shard index.
    pub shard_idx: usize,
    /// Total number of shards.
    pub total_shards: usize,
    /// Byte size of this shard on disk.
    pub shard_size_bytes: u64,
    /// File path or identifier for the shard.
    pub path: String,
    /// Tensor names contained in this shard.
    pub tensor_names: Vec<String>,
}

impl WeightShardInfo {
    /// Whether this is the last shard.
    pub const fn is_last(&self) -> bool {
        self.shard_idx + 1 == self.total_shards
    }
}

// ── Sharded loader ──────────────────────────────────────────────────────────

/// Loads weights from multiple shard files.
#[derive(Debug)]
pub struct ShardedLoader {
    shards: Vec<WeightShardInfo>,
    loaded_shards: Vec<bool>,
    total_bytes: u64,
    loaded_bytes: u64,
}

impl ShardedLoader {
    /// Create a sharded loader from shard metadata.
    ///
    /// Returns `None` if `shards` is empty or shard indices are inconsistent.
    pub fn new(shards: Vec<WeightShardInfo>) -> Option<Self> {
        if shards.is_empty() {
            return None;
        }
        let expected_total = shards[0].total_shards;
        if shards.len() != expected_total {
            return None;
        }
        for (i, s) in shards.iter().enumerate() {
            if s.shard_idx != i || s.total_shards != expected_total {
                return None;
            }
        }
        let total_bytes = shards.iter().map(|s| s.shard_size_bytes).sum();
        let loaded_shards = vec![false; shards.len()];
        Some(Self { shards, loaded_shards, total_bytes, loaded_bytes: 0 })
    }

    /// Number of shards.
    pub const fn num_shards(&self) -> usize {
        self.shards.len()
    }

    /// Total bytes across all shards.
    pub const fn total_bytes(&self) -> u64 {
        self.total_bytes
    }

    /// Bytes loaded so far.
    pub const fn loaded_bytes(&self) -> u64 {
        self.loaded_bytes
    }

    /// Loading progress as a ratio in `[0.0, 1.0]`.
    pub fn progress(&self) -> f64 {
        if self.total_bytes == 0 {
            return 1.0;
        }
        #[allow(clippy::cast_precision_loss)]
        let p = self.loaded_bytes as f64 / self.total_bytes as f64;
        p
    }

    /// Mark a shard as loaded.
    pub fn mark_loaded(&mut self, shard_idx: usize) -> bool {
        if shard_idx >= self.shards.len() || self.loaded_shards[shard_idx] {
            return false;
        }
        self.loaded_shards[shard_idx] = true;
        self.loaded_bytes += self.shards[shard_idx].shard_size_bytes;
        true
    }

    /// Whether all shards have been loaded.
    pub fn all_loaded(&self) -> bool {
        self.loaded_shards.iter().all(|&b| b)
    }

    /// Indices of shards not yet loaded.
    pub fn pending_shards(&self) -> Vec<usize> {
        self.loaded_shards
            .iter()
            .enumerate()
            .filter(|(_, loaded)| !**loaded)
            .map(|(i, _)| i)
            .collect()
    }

    /// Shard info by index.
    pub fn shard(&self, idx: usize) -> Option<&WeightShardInfo> {
        self.shards.get(idx)
    }

    /// Tensor names across all shards (ordered by shard index).
    pub fn all_tensor_names(&self) -> Vec<&str> {
        self.shards.iter().flat_map(|s| s.tensor_names.iter().map(String::as_str)).collect()
    }
}

// ── Weight converter ────────────────────────────────────────────────────────

/// Describes a format conversion operation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConversionRequest {
    /// Source format.
    pub from: WeightFormat,
    /// Target format.
    pub to: WeightFormat,
    /// Optional data-type cast (e.g. F32 → F16).
    pub cast_dtype: Option<DataType>,
}

/// Result of a format conversion.
#[derive(Debug, Clone)]
pub struct ConversionResult {
    /// Whether the conversion succeeded.
    pub success: bool,
    /// Number of tensors converted.
    pub tensors_converted: usize,
    /// Total bytes written.
    pub bytes_written: u64,
    /// Wall-clock duration.
    pub duration: Duration,
    /// Per-tensor errors (empty on full success).
    pub errors: Vec<String>,
}

/// Converts between weight formats.
#[derive(Debug)]
pub struct WeightConverter {
    supported_pairs: Vec<(WeightFormat, WeightFormat)>,
}

impl WeightConverter {
    /// Create a converter with the default supported format pairs.
    pub fn new() -> Self {
        Self {
            supported_pairs: vec![
                (WeightFormat::SafeTensors, WeightFormat::GGUF),
                (WeightFormat::PyTorch, WeightFormat::GGUF),
                (WeightFormat::PyTorch, WeightFormat::SafeTensors),
                (WeightFormat::NumPy, WeightFormat::SafeTensors),
                (WeightFormat::ONNX, WeightFormat::SafeTensors),
                (WeightFormat::Raw, WeightFormat::GGUF),
            ],
        }
    }

    /// Whether a given conversion pair is supported.
    pub fn is_supported(&self, from: WeightFormat, to: WeightFormat) -> bool {
        self.supported_pairs.contains(&(from, to))
    }

    /// List all supported conversion pairs.
    pub fn supported_conversions(&self) -> &[(WeightFormat, WeightFormat)] {
        &self.supported_pairs
    }

    /// Perform a dry-run conversion, validating the request without writing.
    pub fn validate(&self, request: &ConversionRequest) -> Result<(), String> {
        if request.from == request.to {
            return Err("source and target formats are identical".to_string());
        }
        if !self.is_supported(request.from, request.to) {
            return Err(format!(
                "conversion from {} to {} is not supported",
                request.from, request.to
            ));
        }
        Ok(())
    }

    /// Simulate a conversion and return metrics.
    ///
    /// In a real implementation this would read/write files; here it returns
    /// a synthetic result for testing the pipeline plumbing.
    pub fn convert(
        &self,
        request: &ConversionRequest,
        tensor_count: usize,
        total_bytes: u64,
    ) -> ConversionResult {
        let start = Instant::now();
        if let Err(e) = self.validate(request) {
            return ConversionResult {
                success: false,
                tensors_converted: 0,
                bytes_written: 0,
                duration: start.elapsed(),
                errors: vec![e],
            };
        }
        ConversionResult {
            success: true,
            tensors_converted: tensor_count,
            bytes_written: total_bytes,
            duration: start.elapsed(),
            errors: Vec::new(),
        }
    }
}

impl Default for WeightConverter {
    fn default() -> Self {
        Self::new()
    }
}

// ── Metrics ─────────────────────────────────────────────────────────────────

/// Aggregate metrics collected during weight loading.
#[derive(Debug, Clone)]
pub struct WeightLoaderMetrics {
    /// Wall-clock time spent loading.
    pub load_duration: Duration,
    /// Total bytes loaded from disk.
    pub bytes_loaded: u64,
    /// Number of cache hits (e.g. mmap page already resident).
    pub cache_hits: u64,
    /// Number of cache misses.
    pub cache_misses: u64,
    /// Prefetch accuracy ratio `[0.0, 1.0]`.
    pub prefetch_accuracy: f64,
    /// Number of tensors loaded.
    pub tensors_loaded: usize,
    /// Number of shards loaded (0 if not sharded).
    pub shards_loaded: usize,
}

impl WeightLoaderMetrics {
    /// Create zeroed metrics.
    pub const fn new() -> Self {
        Self {
            load_duration: Duration::ZERO,
            bytes_loaded: 0,
            cache_hits: 0,
            cache_misses: 0,
            prefetch_accuracy: 0.0,
            tensors_loaded: 0,
            shards_loaded: 0,
        }
    }

    /// Effective throughput in bytes per second.
    pub fn throughput_bytes_per_sec(&self) -> f64 {
        let secs = self.load_duration.as_secs_f64();
        if secs == 0.0 {
            return 0.0;
        }
        #[allow(clippy::cast_precision_loss)]
        let bps = self.bytes_loaded as f64 / secs;
        bps
    }

    /// Cache hit ratio in `[0.0, 1.0]`.
    pub fn cache_hit_ratio(&self) -> f64 {
        let total = self.cache_hits + self.cache_misses;
        if total == 0 {
            return 0.0;
        }
        #[allow(clippy::cast_precision_loss)]
        let ratio = self.cache_hits as f64 / total as f64;
        ratio
    }
}

impl Default for WeightLoaderMetrics {
    fn default() -> Self {
        Self::new()
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;

    // ── helpers ─────────────────────────────────────────────────────────────

    fn sample_tensor_info(name: &str, shape: Vec<usize>, dtype: DataType) -> TensorInfo {
        let num: u64 = shape.iter().copied().product::<usize>() as u64;
        let elem = dtype.element_size_bytes() as u64;
        TensorInfo { name: name.to_string(), shape, dtype, offset: 0, size_bytes: num * elem }
    }

    fn three_layer_tensors() -> Vec<TensorInfo> {
        vec![
            sample_tensor_info("layers.0.attn.wq", vec![4096, 4096], DataType::F16),
            sample_tensor_info("layers.0.attn.wk", vec![4096, 4096], DataType::F16),
            sample_tensor_info("layers.1.attn.wq", vec![4096, 4096], DataType::F16),
            sample_tensor_info("layers.1.attn.wk", vec![4096, 4096], DataType::F16),
            sample_tensor_info("layers.2.attn.wq", vec![4096, 4096], DataType::F16),
        ]
    }

    fn make_weight_map() -> WeightMap {
        WeightMap::new(WeightFormat::GGUF, three_layer_tensors())
    }

    fn make_shard(idx: usize, total: usize, size: u64) -> WeightShardInfo {
        WeightShardInfo {
            shard_idx: idx,
            total_shards: total,
            shard_size_bytes: size,
            path: format!("model-{idx:05}-of-{total:05}.safetensors"),
            tensor_names: vec![format!("shard_{idx}_t0"), format!("shard_{idx}_t1")],
        }
    }

    // ── WeightFormat ────────────────────────────────────────────────────────

    #[test]
    fn format_display() {
        assert_eq!(WeightFormat::GGUF.to_string(), "GGUF");
        assert_eq!(WeightFormat::SafeTensors.to_string(), "SafeTensors");
        assert_eq!(WeightFormat::PyTorch.to_string(), "PyTorch");
        assert_eq!(WeightFormat::ONNX.to_string(), "ONNX");
        assert_eq!(WeightFormat::NumPy.to_string(), "NumPy");
        assert_eq!(WeightFormat::Raw.to_string(), "Raw");
    }

    #[test]
    fn format_from_extension_known() {
        assert_eq!(WeightFormat::from_extension("gguf"), Some(WeightFormat::GGUF));
        assert_eq!(WeightFormat::from_extension("safetensors"), Some(WeightFormat::SafeTensors));
        assert_eq!(WeightFormat::from_extension("bin"), Some(WeightFormat::PyTorch));
        assert_eq!(WeightFormat::from_extension("pt"), Some(WeightFormat::PyTorch));
        assert_eq!(WeightFormat::from_extension("pth"), Some(WeightFormat::PyTorch));
        assert_eq!(WeightFormat::from_extension("onnx"), Some(WeightFormat::ONNX));
        assert_eq!(WeightFormat::from_extension("npy"), Some(WeightFormat::NumPy));
        assert_eq!(WeightFormat::from_extension("npz"), Some(WeightFormat::NumPy));
        assert_eq!(WeightFormat::from_extension("raw"), Some(WeightFormat::Raw));
    }

    #[test]
    fn format_from_extension_unknown() {
        assert_eq!(WeightFormat::from_extension("txt"), None);
        assert_eq!(WeightFormat::from_extension(""), None);
    }

    #[test]
    fn format_from_extension_case_insensitive() {
        assert_eq!(WeightFormat::from_extension("GGUF"), Some(WeightFormat::GGUF));
        assert_eq!(WeightFormat::from_extension("SafeTensors"), Some(WeightFormat::SafeTensors));
    }

    #[test]
    fn format_extension_roundtrip() {
        for fmt in [
            WeightFormat::GGUF,
            WeightFormat::SafeTensors,
            WeightFormat::PyTorch,
            WeightFormat::ONNX,
            WeightFormat::NumPy,
            WeightFormat::Raw,
        ] {
            assert_eq!(WeightFormat::from_extension(fmt.extension()), Some(fmt));
        }
    }

    #[test]
    fn format_eq_and_hash() {
        let mut set = std::collections::HashSet::new();
        set.insert(WeightFormat::GGUF);
        set.insert(WeightFormat::GGUF);
        assert_eq!(set.len(), 1);
    }

    // ── DataType ────────────────────────────────────────────────────────────

    #[test]
    fn dtype_element_sizes() {
        assert_eq!(DataType::F32.element_size_bytes(), 4);
        assert_eq!(DataType::F16.element_size_bytes(), 2);
        assert_eq!(DataType::BF16.element_size_bytes(), 2);
        assert_eq!(DataType::I8.element_size_bytes(), 1);
        assert_eq!(DataType::U8.element_size_bytes(), 1);
        assert_eq!(DataType::Bool.element_size_bytes(), 1);
        assert_eq!(DataType::I2.element_size_bytes(), 1);
        assert_eq!(DataType::I4.element_size_bytes(), 1);
    }

    // ── LoaderConfig ────────────────────────────────────────────────────────

    #[test]
    fn default_config_values() {
        let cfg = LoaderConfig::default();
        assert_eq!(cfg.format, WeightFormat::GGUF);
        assert!(cfg.memory_map);
        assert!(cfg.prefetch);
        assert!(cfg.parallel_load);
        assert!(cfg.verify_checksums);
        assert_eq!(cfg.max_io_parallelism, 4);
    }

    #[test]
    fn config_custom_values() {
        let cfg = LoaderConfig {
            format: WeightFormat::SafeTensors,
            memory_map: false,
            prefetch: false,
            parallel_load: false,
            verify_checksums: false,
            max_io_parallelism: 1,
        };
        assert_eq!(cfg.format, WeightFormat::SafeTensors);
        assert!(!cfg.memory_map);
        assert_eq!(cfg.max_io_parallelism, 1);
    }

    // ── TensorInfo ──────────────────────────────────────────────────────────

    #[test]
    fn tensor_info_num_elements() {
        let t = sample_tensor_info("t", vec![2, 3, 4], DataType::F32);
        assert_eq!(t.num_elements(), 24);
    }

    #[test]
    fn tensor_info_ndim() {
        let t = sample_tensor_info("t", vec![10, 20], DataType::F16);
        assert_eq!(t.ndim(), 2);
    }

    #[test]
    fn tensor_info_scalar() {
        let t = sample_tensor_info("t", vec![1], DataType::F32);
        assert_eq!(t.num_elements(), 1);
        assert_eq!(t.ndim(), 1);
    }

    #[test]
    fn tensor_info_size_bytes() {
        let t = sample_tensor_info("t", vec![100], DataType::F32);
        assert_eq!(t.size_bytes, 400);
    }

    #[test]
    fn tensor_info_equality() {
        let a = sample_tensor_info("t", vec![4], DataType::I8);
        let b = sample_tensor_info("t", vec![4], DataType::I8);
        assert_eq!(a, b);
    }

    // ── WeightMap ───────────────────────────────────────────────────────────

    #[test]
    fn weight_map_len() {
        let wm = make_weight_map();
        assert_eq!(wm.len(), 5);
        assert!(!wm.is_empty());
    }

    #[test]
    fn weight_map_empty() {
        let wm = WeightMap::new(WeightFormat::Raw, vec![]);
        assert!(wm.is_empty());
        assert_eq!(wm.len(), 0);
    }

    #[test]
    fn weight_map_get_info_found() {
        let wm = make_weight_map();
        let info = wm.get_info("layers.0.attn.wq").unwrap();
        assert_eq!(info.shape, vec![4096, 4096]);
    }

    #[test]
    fn weight_map_get_info_not_found() {
        let wm = make_weight_map();
        assert!(wm.get_info("nonexistent").is_none());
    }

    #[test]
    fn weight_map_format() {
        let wm = make_weight_map();
        assert_eq!(wm.format(), WeightFormat::GGUF);
    }

    #[test]
    fn weight_map_insert_and_get_data() {
        let mut wm = make_weight_map();
        wm.insert_data("layers.0.attn.wq", vec![1, 2, 3, 4]);
        let data = wm.get_data("layers.0.attn.wq").unwrap();
        assert_eq!(data, &[1, 2, 3, 4]);
    }

    #[test]
    fn weight_map_is_loaded() {
        let mut wm = make_weight_map();
        assert!(!wm.is_loaded("layers.0.attn.wq"));
        wm.insert_data("layers.0.attn.wq", vec![0]);
        assert!(wm.is_loaded("layers.0.attn.wq"));
    }

    #[test]
    fn weight_map_loaded_count() {
        let mut wm = make_weight_map();
        assert_eq!(wm.loaded_count(), 0);
        wm.insert_data("layers.0.attn.wq", vec![0; 8]);
        wm.insert_data("layers.1.attn.wq", vec![0; 4]);
        assert_eq!(wm.loaded_count(), 2);
    }

    #[test]
    fn weight_map_total_loaded_bytes() {
        let mut wm = make_weight_map();
        wm.insert_data("layers.0.attn.wq", vec![0; 100]);
        wm.insert_data("layers.1.attn.wq", vec![0; 200]);
        assert_eq!(wm.total_loaded_bytes(), 300);
    }

    #[test]
    fn weight_map_pending_names() {
        let mut wm = make_weight_map();
        assert_eq!(wm.pending_names().len(), 5);
        wm.insert_data("layers.0.attn.wq", vec![]);
        wm.insert_data("layers.0.attn.wk", vec![]);
        assert_eq!(wm.pending_names().len(), 3);
    }

    #[test]
    fn weight_map_tensor_infos_order() {
        let wm = make_weight_map();
        let names: Vec<&str> = wm.tensor_infos().iter().map(|t| t.name.as_str()).collect();
        assert_eq!(names[0], "layers.0.attn.wq");
        assert_eq!(names[4], "layers.2.attn.wq");
    }

    // ── LoadState ───────────────────────────────────────────────────────────

    #[test]
    fn load_state_equality() {
        assert_eq!(LoadState::NotLoaded, LoadState::NotLoaded);
        assert_ne!(LoadState::Loaded, LoadState::Failed);
    }

    // ── LazyLoader ──────────────────────────────────────────────────────────

    #[test]
    fn lazy_initial_states_not_loaded() {
        let ll = LazyLoader::new(LoaderConfig::default(), make_weight_map());
        assert_eq!(ll.state("layers.0.attn.wq"), Some(LoadState::NotLoaded));
    }

    #[test]
    fn lazy_request_load_transitions() {
        let mut ll = LazyLoader::new(LoaderConfig::default(), make_weight_map());
        assert!(ll.request_load("layers.0.attn.wq"));
        assert_eq!(ll.state("layers.0.attn.wq"), Some(LoadState::Loading));
    }

    #[test]
    fn lazy_request_load_unknown_tensor() {
        let mut ll = LazyLoader::new(LoaderConfig::default(), make_weight_map());
        assert!(!ll.request_load("no_such_tensor"));
    }

    #[test]
    fn lazy_complete_load() {
        let mut ll = LazyLoader::new(LoaderConfig::default(), make_weight_map());
        ll.request_load("layers.0.attn.wq");
        assert!(ll.complete_load("layers.0.attn.wq", vec![42; 8]));
        assert_eq!(ll.state("layers.0.attn.wq"), Some(LoadState::Loaded));
        assert_eq!(ll.get_data("layers.0.attn.wq"), Some([42u8; 8].as_slice()));
    }

    #[test]
    fn lazy_complete_load_without_request_fails() {
        let mut ll = LazyLoader::new(LoaderConfig::default(), make_weight_map());
        assert!(!ll.complete_load("layers.0.attn.wq", vec![0]));
    }

    #[test]
    fn lazy_fail_load() {
        let mut ll = LazyLoader::new(LoaderConfig::default(), make_weight_map());
        ll.request_load("layers.0.attn.wq");
        assert!(ll.fail_load("layers.0.attn.wq"));
        assert_eq!(ll.state("layers.0.attn.wq"), Some(LoadState::Failed));
    }

    #[test]
    fn lazy_fail_without_request_fails() {
        let mut ll = LazyLoader::new(LoaderConfig::default(), make_weight_map());
        assert!(!ll.fail_load("layers.0.attn.wq"));
    }

    #[test]
    fn lazy_state_counts() {
        let mut ll = LazyLoader::new(LoaderConfig::default(), make_weight_map());
        ll.request_load("layers.0.attn.wq");
        ll.complete_load("layers.0.attn.wq", vec![0]);
        ll.request_load("layers.0.attn.wk");
        let counts = ll.state_counts();
        assert_eq!(counts[&LoadState::Loaded], 1);
        assert_eq!(counts[&LoadState::Loading], 1);
        assert_eq!(counts[&LoadState::NotLoaded], 3);
    }

    #[test]
    fn lazy_access_order() {
        let mut ll = LazyLoader::new(LoaderConfig::default(), make_weight_map());
        ll.request_load("layers.1.attn.wq");
        ll.request_load("layers.0.attn.wq");
        // Re-requesting should not duplicate
        ll.request_load("layers.1.attn.wq");
        assert_eq!(ll.access_order(), &["layers.1.attn.wq", "layers.0.attn.wq"]);
    }

    #[test]
    fn lazy_config_accessible() {
        let cfg = LoaderConfig { max_io_parallelism: 8, ..LoaderConfig::default() };
        let ll = LazyLoader::new(cfg, make_weight_map());
        assert_eq!(ll.config().max_io_parallelism, 8);
    }

    #[test]
    fn lazy_weight_map_accessible() {
        let ll = LazyLoader::new(LoaderConfig::default(), make_weight_map());
        assert_eq!(ll.weight_map().len(), 5);
    }

    // ── PrefetchScheduler ───────────────────────────────────────────────────

    #[test]
    fn prefetch_sequential_window() {
        let names: Vec<String> = three_layer_tensors().iter().map(|t| t.name.clone()).collect();
        let ps = PrefetchScheduler::new(PrefetchStrategy::Sequential, 2, names);
        let next = ps.next_prefetch("layers.0.attn.wq");
        assert_eq!(next, vec!["layers.0.attn.wk", "layers.1.attn.wq"]);
    }

    #[test]
    fn prefetch_sequential_at_end() {
        let names: Vec<String> = three_layer_tensors().iter().map(|t| t.name.clone()).collect();
        let ps = PrefetchScheduler::new(PrefetchStrategy::Sequential, 3, names);
        let next = ps.next_prefetch("layers.2.attn.wq");
        assert!(next.is_empty());
    }

    #[test]
    fn prefetch_sequential_unknown_tensor() {
        let names: Vec<String> = three_layer_tensors().iter().map(|t| t.name.clone()).collect();
        let ps = PrefetchScheduler::new(PrefetchStrategy::Sequential, 2, names);
        assert!(ps.next_prefetch("nonexistent").is_empty());
    }

    #[test]
    fn prefetch_disabled() {
        let names: Vec<String> = three_layer_tensors().iter().map(|t| t.name.clone()).collect();
        let ps = PrefetchScheduler::new(PrefetchStrategy::Disabled, 2, names);
        assert!(ps.next_prefetch("layers.0.attn.wq").is_empty());
    }

    #[test]
    fn prefetch_layer_local() {
        let names: Vec<String> = three_layer_tensors().iter().map(|t| t.name.clone()).collect();
        let ps = PrefetchScheduler::new(PrefetchStrategy::LayerLocal, 10, names);
        let next = ps.next_prefetch("layers.0.attn.wq");
        assert_eq!(next, vec!["layers.0.attn.wk"]);
    }

    #[test]
    fn prefetch_layer_local_skips_prefetched() {
        let names: Vec<String> = three_layer_tensors().iter().map(|t| t.name.clone()).collect();
        let mut ps = PrefetchScheduler::new(PrefetchStrategy::LayerLocal, 10, names);
        ps.mark_prefetched("layers.0.attn.wk");
        let next = ps.next_prefetch("layers.0.attn.wq");
        assert!(next.is_empty());
    }

    #[test]
    fn prefetch_layer_local_no_layer_index() {
        let names = vec!["embed_tokens.weight".to_string()];
        let ps = PrefetchScheduler::new(PrefetchStrategy::LayerLocal, 5, names);
        assert!(ps.next_prefetch("embed_tokens.weight").is_empty());
    }

    #[test]
    fn prefetch_accuracy_zero() {
        let ps = PrefetchScheduler::new(PrefetchStrategy::Sequential, 2, vec![]);
        assert_eq!(ps.accuracy(), 0.0);
    }

    #[test]
    fn prefetch_accuracy_all_hits() {
        let mut ps = PrefetchScheduler::new(PrefetchStrategy::Sequential, 2, vec![]);
        ps.record_hit();
        ps.record_hit();
        ps.record_hit();
        assert!((ps.accuracy() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn prefetch_accuracy_mixed() {
        let mut ps = PrefetchScheduler::new(PrefetchStrategy::Sequential, 2, vec![]);
        ps.record_hit();
        ps.record_miss();
        assert!((ps.accuracy() - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn prefetch_hits_misses_count() {
        let mut ps = PrefetchScheduler::new(PrefetchStrategy::Sequential, 2, vec![]);
        ps.record_hit();
        ps.record_hit();
        ps.record_miss();
        assert_eq!(ps.hits(), 2);
        assert_eq!(ps.misses(), 1);
    }

    #[test]
    fn prefetch_strategy_accessor() {
        let ps = PrefetchScheduler::new(PrefetchStrategy::LayerLocal, 4, vec![]);
        assert_eq!(ps.strategy(), PrefetchStrategy::LayerLocal);
        assert_eq!(ps.window_size(), 4);
    }

    // ── extract_layer_index ─────────────────────────────────────────────────

    #[test]
    fn extract_layer_index_found() {
        assert_eq!(extract_layer_index("layers.5.attention.wq"), Some(5));
    }

    #[test]
    fn extract_layer_index_first_numeric() {
        assert_eq!(extract_layer_index("model.layers.12.mlp.gate"), Some(12));
    }

    #[test]
    fn extract_layer_index_none() {
        assert_eq!(extract_layer_index("embed_tokens.weight"), None);
    }

    #[test]
    fn extract_layer_index_zero() {
        assert_eq!(extract_layer_index("layers.0.wq"), Some(0));
    }

    // ── WeightShardInfo ─────────────────────────────────────────────────────

    #[test]
    fn shard_is_last() {
        let s = make_shard(2, 3, 100);
        assert!(s.is_last());
    }

    #[test]
    fn shard_is_not_last() {
        let s = make_shard(0, 3, 100);
        assert!(!s.is_last());
    }

    // ── ShardedLoader ───────────────────────────────────────────────────────

    #[test]
    fn sharded_loader_creation() {
        let shards = vec![make_shard(0, 3, 100), make_shard(1, 3, 200), make_shard(2, 3, 300)];
        let sl = ShardedLoader::new(shards).unwrap();
        assert_eq!(sl.num_shards(), 3);
        assert_eq!(sl.total_bytes(), 600);
    }

    #[test]
    fn sharded_loader_empty_fails() {
        assert!(ShardedLoader::new(vec![]).is_none());
    }

    #[test]
    fn sharded_loader_mismatched_count_fails() {
        let shards = vec![make_shard(0, 3, 100), make_shard(1, 3, 200)];
        assert!(ShardedLoader::new(shards).is_none());
    }

    #[test]
    fn sharded_loader_wrong_index_order_fails() {
        let mut shards = vec![make_shard(0, 2, 100), make_shard(1, 2, 200)];
        shards.swap(0, 1);
        assert!(ShardedLoader::new(shards).is_none());
    }

    #[test]
    fn sharded_loader_progress() {
        let shards = vec![make_shard(0, 2, 100), make_shard(1, 2, 100)];
        let mut sl = ShardedLoader::new(shards).unwrap();
        assert!((sl.progress() - 0.0).abs() < f64::EPSILON);
        sl.mark_loaded(0);
        assert!((sl.progress() - 0.5).abs() < f64::EPSILON);
        sl.mark_loaded(1);
        assert!((sl.progress() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn sharded_loader_zero_bytes_progress() {
        let mut s0 = make_shard(0, 1, 0);
        s0.shard_size_bytes = 0;
        let sl = ShardedLoader::new(vec![s0]).unwrap();
        assert!((sl.progress() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn sharded_loader_mark_loaded() {
        let shards = vec![make_shard(0, 2, 50), make_shard(1, 2, 50)];
        let mut sl = ShardedLoader::new(shards).unwrap();
        assert!(sl.mark_loaded(0));
        assert_eq!(sl.loaded_bytes(), 50);
        // Double-mark returns false
        assert!(!sl.mark_loaded(0));
    }

    #[test]
    fn sharded_loader_mark_out_of_range() {
        let shards = vec![make_shard(0, 1, 100)];
        let mut sl = ShardedLoader::new(shards).unwrap();
        assert!(!sl.mark_loaded(5));
    }

    #[test]
    fn sharded_loader_all_loaded() {
        let shards = vec![make_shard(0, 2, 50), make_shard(1, 2, 50)];
        let mut sl = ShardedLoader::new(shards).unwrap();
        assert!(!sl.all_loaded());
        sl.mark_loaded(0);
        sl.mark_loaded(1);
        assert!(sl.all_loaded());
    }

    #[test]
    fn sharded_loader_pending() {
        let shards = vec![make_shard(0, 3, 10), make_shard(1, 3, 10), make_shard(2, 3, 10)];
        let mut sl = ShardedLoader::new(shards).unwrap();
        sl.mark_loaded(1);
        assert_eq!(sl.pending_shards(), vec![0, 2]);
    }

    #[test]
    fn sharded_loader_shard_accessor() {
        let shards = vec![make_shard(0, 1, 42)];
        let sl = ShardedLoader::new(shards).unwrap();
        assert_eq!(sl.shard(0).unwrap().shard_size_bytes, 42);
        assert!(sl.shard(1).is_none());
    }

    #[test]
    fn sharded_loader_all_tensor_names() {
        let shards = vec![make_shard(0, 2, 10), make_shard(1, 2, 10)];
        let sl = ShardedLoader::new(shards).unwrap();
        let names = sl.all_tensor_names();
        assert_eq!(names.len(), 4);
        assert_eq!(names[0], "shard_0_t0");
        assert_eq!(names[3], "shard_1_t1");
    }

    // ── WeightConverter ─────────────────────────────────────────────────────

    #[test]
    fn converter_supported_pair() {
        let wc = WeightConverter::new();
        assert!(wc.is_supported(WeightFormat::SafeTensors, WeightFormat::GGUF));
        assert!(wc.is_supported(WeightFormat::PyTorch, WeightFormat::GGUF));
    }

    #[test]
    fn converter_unsupported_pair() {
        let wc = WeightConverter::new();
        assert!(!wc.is_supported(WeightFormat::GGUF, WeightFormat::PyTorch));
    }

    #[test]
    fn converter_validate_same_format() {
        let wc = WeightConverter::new();
        let req = ConversionRequest {
            from: WeightFormat::GGUF,
            to: WeightFormat::GGUF,
            cast_dtype: None,
        };
        assert!(wc.validate(&req).is_err());
    }

    #[test]
    fn converter_validate_unsupported() {
        let wc = WeightConverter::new();
        let req = ConversionRequest {
            from: WeightFormat::GGUF,
            to: WeightFormat::NumPy,
            cast_dtype: None,
        };
        let err = wc.validate(&req).unwrap_err();
        assert!(err.contains("not supported"));
    }

    #[test]
    fn converter_validate_ok() {
        let wc = WeightConverter::new();
        let req = ConversionRequest {
            from: WeightFormat::SafeTensors,
            to: WeightFormat::GGUF,
            cast_dtype: None,
        };
        assert!(wc.validate(&req).is_ok());
    }

    #[test]
    fn converter_convert_success() {
        let wc = WeightConverter::new();
        let req = ConversionRequest {
            from: WeightFormat::PyTorch,
            to: WeightFormat::GGUF,
            cast_dtype: None,
        };
        let result = wc.convert(&req, 10, 1024);
        assert!(result.success);
        assert_eq!(result.tensors_converted, 10);
        assert_eq!(result.bytes_written, 1024);
        assert!(result.errors.is_empty());
    }

    #[test]
    fn converter_convert_failure() {
        let wc = WeightConverter::new();
        let req = ConversionRequest {
            from: WeightFormat::GGUF,
            to: WeightFormat::GGUF,
            cast_dtype: None,
        };
        let result = wc.convert(&req, 10, 1024);
        assert!(!result.success);
        assert_eq!(result.tensors_converted, 0);
        assert!(!result.errors.is_empty());
    }

    #[test]
    fn converter_supported_conversions_list() {
        let wc = WeightConverter::new();
        assert_eq!(wc.supported_conversions().len(), 6);
    }

    #[test]
    fn converter_default_same_as_new() {
        let a = WeightConverter::new();
        let b = WeightConverter::default();
        assert_eq!(a.supported_conversions().len(), b.supported_conversions().len());
    }

    // ── WeightLoaderMetrics ─────────────────────────────────────────────────

    #[test]
    fn metrics_new_zeroed() {
        let m = WeightLoaderMetrics::new();
        assert_eq!(m.bytes_loaded, 0);
        assert_eq!(m.cache_hits, 0);
        assert_eq!(m.tensors_loaded, 0);
        assert_eq!(m.shards_loaded, 0);
    }

    #[test]
    fn metrics_default_equals_new() {
        let a = WeightLoaderMetrics::new();
        let b = WeightLoaderMetrics::default();
        assert_eq!(a.bytes_loaded, b.bytes_loaded);
    }

    #[test]
    fn metrics_throughput_nonzero_duration() {
        let m = WeightLoaderMetrics {
            load_duration: Duration::from_secs(2),
            bytes_loaded: 1_000_000,
            ..WeightLoaderMetrics::new()
        };
        assert!((m.throughput_bytes_per_sec() - 500_000.0).abs() < 1.0);
    }

    #[test]
    fn metrics_throughput_zero_duration() {
        let m = WeightLoaderMetrics::new();
        assert_eq!(m.throughput_bytes_per_sec(), 0.0);
    }

    #[test]
    fn metrics_cache_hit_ratio() {
        let m =
            WeightLoaderMetrics { cache_hits: 3, cache_misses: 1, ..WeightLoaderMetrics::new() };
        assert!((m.cache_hit_ratio() - 0.75).abs() < f64::EPSILON);
    }

    #[test]
    fn metrics_cache_hit_ratio_zero() {
        let m = WeightLoaderMetrics::new();
        assert_eq!(m.cache_hit_ratio(), 0.0);
    }

    #[test]
    fn metrics_cache_hit_ratio_all_hits() {
        let m =
            WeightLoaderMetrics { cache_hits: 10, cache_misses: 0, ..WeightLoaderMetrics::new() };
        assert!((m.cache_hit_ratio() - 1.0).abs() < f64::EPSILON);
    }

    // ── proptest ────────────────────────────────────────────────────────────

    proptest::proptest! {
        #[test]
        fn format_extension_always_parses(
            fmt in proptest::sample::select(vec![
                WeightFormat::GGUF,
                WeightFormat::SafeTensors,
                WeightFormat::PyTorch,
                WeightFormat::ONNX,
                WeightFormat::NumPy,
                WeightFormat::Raw,
            ])
        ) {
            let ext = fmt.extension();
            proptest::prop_assert!(WeightFormat::from_extension(ext).is_some());
        }

        #[test]
        fn tensor_info_num_elements_matches_shape(
            d0 in 1usize..64,
            d1 in 1usize..64,
        ) {
            let t = sample_tensor_info("t", vec![d0, d1], DataType::F32);
            proptest::prop_assert_eq!(t.num_elements(), d0 * d1);
        }

        #[test]
        fn sharded_loader_progress_monotonic(
            n in 2usize..8,
        ) {
            let shards: Vec<WeightShardInfo> =
                (0..n).map(|i| make_shard(i, n, 100)).collect();
            let mut sl = ShardedLoader::new(shards).unwrap();
            let mut prev = 0.0f64;
            for i in 0..n {
                sl.mark_loaded(i);
                let p = sl.progress();
                proptest::prop_assert!(p >= prev);
                prev = p;
            }
            proptest::prop_assert!((prev - 1.0).abs() < f64::EPSILON);
        }

        #[test]
        fn prefetch_accuracy_bounded(
            hits in 0u64..100,
            misses in 0u64..100,
        ) {
            let mut ps = PrefetchScheduler::new(
                PrefetchStrategy::Sequential, 2, vec![],
            );
            for _ in 0..hits { ps.record_hit(); }
            for _ in 0..misses { ps.record_miss(); }
            let acc = ps.accuracy();
            proptest::prop_assert!((0.0..=1.0).contains(&acc));
        }

        #[test]
        fn metrics_cache_ratio_bounded(
            hits in 0u64..1000,
            misses in 0u64..1000,
        ) {
            let m = WeightLoaderMetrics {
                cache_hits: hits,
                cache_misses: misses,
                ..WeightLoaderMetrics::new()
            };
            let r = m.cache_hit_ratio();
            proptest::prop_assert!((0.0..=1.0).contains(&r));
        }
    }
}
