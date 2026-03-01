//! Weight loading pipeline for GPU inference.
//!
//! Provides [`WeightLoadingEngine`] for loading model weights from multiple
//! formats (GGUF, `SafeTensors`, `PyTorch`, `NumPy`, Raw) with lazy loading,
//! sharded file support, dtype casting, validation, caching, and progress
//! tracking.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;
use std::io;
use std::path::{Path, PathBuf};

// -- Weight format ------------------------------------------------------------

/// Serialisation format of the weight file(s).
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum WeightFormat {
    /// GGUF quantised format (default for `BitNet` models).
    #[default]
    Gguf,
    /// Hugging Face `SafeTensors` format.
    SafeTensors,
    /// `PyTorch` `.pt` / `.bin` format.
    PyTorch,
    /// `NumPy` `.npy` / `.npz` format.
    NumPy,
    /// Raw binary blob with explicit layout.
    Raw,
}

impl fmt::Display for WeightFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Gguf => write!(f, "gguf"),
            Self::SafeTensors => write!(f, "safetensors"),
            Self::PyTorch => write!(f, "pytorch"),
            Self::NumPy => write!(f, "numpy"),
            Self::Raw => write!(f, "raw"),
        }
    }
}

impl WeightFormat {
    /// Detect format from a file extension.
    #[must_use]
    pub fn from_extension(path: &Path) -> Option<Self> {
        match path.extension()?.to_str()? {
            "gguf" => Some(Self::Gguf),
            "safetensors" => Some(Self::SafeTensors),
            "pt" | "bin" | "pth" => Some(Self::PyTorch),
            "npy" | "npz" => Some(Self::NumPy),
            "raw" => Some(Self::Raw),
            _ => None,
        }
    }
}

// -- Data types ---------------------------------------------------------------

/// Numeric data type of tensor elements.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DType {
    /// 32-bit floating point.
    #[default]
    F32,
    /// 16-bit floating point.
    F16,
    /// Brain floating point (16-bit).
    BF16,
    /// 8-bit integer (signed).
    I8,
    /// 2-bit integer (signed) -- `BitNet` ternary.
    I2,
    /// 4-bit integer (unsigned, quantised).
    U4,
}

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::F32 => write!(f, "f32"),
            Self::F16 => write!(f, "f16"),
            Self::BF16 => write!(f, "bf16"),
            Self::I8 => write!(f, "i8"),
            Self::I2 => write!(f, "i2"),
            Self::U4 => write!(f, "u4"),
        }
    }
}

impl DType {
    /// Size of a single element in bytes (approximate for sub-byte types).
    #[must_use]
    pub const fn element_size_bytes(self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 | Self::BF16 => 2,
            Self::I8 | Self::I2 | Self::U4 => 1,
        }
    }
}

// -- Shard strategy -----------------------------------------------------------

/// How weights are distributed across shard files.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ShardStrategy {
    /// All weights in a single file.
    #[default]
    Single,
    /// Sharded by transformer layer (one file per layer group).
    ByLayer,
    /// Sharded by tensor size (large tensors split across files).
    BySize,
    /// Custom sharding with an explicit mapping.
    Custom,
}

impl fmt::Display for ShardStrategy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Single => write!(f, "single"),
            Self::ByLayer => write!(f, "by_layer"),
            Self::BySize => write!(f, "by_size"),
            Self::Custom => write!(f, "custom"),
        }
    }
}

// -- Errors -------------------------------------------------------------------

/// Errors produced by weight loading operations.
#[derive(Debug)]
pub enum WeightLoadError {
    /// An I/O error occurred.
    Io(io::Error),
    /// The weight file format is unsupported or unrecognised.
    UnsupportedFormat(String),
    /// A requested weight was not found in the file(s).
    WeightNotFound(String),
    /// Shape mismatch between expected and actual tensor dimensions.
    ShapeMismatch {
        /// Name of the tensor.
        name: String,
        /// Expected shape.
        expected: Vec<usize>,
        /// Actual shape found.
        actual: Vec<usize>,
    },
    /// Weight data contains NaN or infinity values.
    InvalidValues(String),
    /// Shard file is missing or corrupt.
    ShardError(String),
    /// Dtype cast is not supported.
    CastError(String),
    /// Configuration is invalid.
    InvalidConfig(String),
}

impl fmt::Display for WeightLoadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(e) => write!(f, "I/O error: {e}"),
            Self::UnsupportedFormat(s) => write!(f, "unsupported format: {s}"),
            Self::WeightNotFound(n) => write!(f, "weight not found: {n}"),
            Self::ShapeMismatch { name, expected, actual } => {
                write!(f, "shape mismatch for {name}: expected {expected:?}, got {actual:?}")
            }
            Self::InvalidValues(s) => write!(f, "invalid values: {s}"),
            Self::ShardError(s) => write!(f, "shard error: {s}"),
            Self::CastError(s) => write!(f, "cast error: {s}"),
            Self::InvalidConfig(s) => write!(f, "invalid config: {s}"),
        }
    }
}

impl std::error::Error for WeightLoadError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<io::Error> for WeightLoadError {
    fn from(e: io::Error) -> Self {
        Self::Io(e)
    }
}

// -- Load configuration -------------------------------------------------------

/// Configuration for the weight loading pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadConfig {
    /// Weight serialisation format.
    pub format: WeightFormat,
    /// Target device identifier (e.g. `"cpu"`, `"cuda:0"`).
    pub device: String,
    /// Optional dtype to cast all weights into after loading.
    pub dtype_cast: Option<DType>,
    /// Enable lazy (on-demand) weight loading via memory mapping.
    pub lazy_load: bool,
    /// Sharding strategy when loading from multiple files.
    pub shard_strategy: ShardStrategy,
    /// Maximum number of weights to keep in memory cache.
    pub cache_capacity: usize,
    /// Whether to validate weights after loading.
    pub validate: bool,
}

impl Default for LoadConfig {
    fn default() -> Self {
        Self {
            format: WeightFormat::default(),
            device: "cpu".to_string(),
            dtype_cast: None,
            lazy_load: false,
            shard_strategy: ShardStrategy::default(),
            cache_capacity: 256,
            validate: true,
        }
    }
}

impl LoadConfig {
    /// Create a new configuration with the given format and device.
    #[must_use]
    pub fn new(format: WeightFormat, device: impl Into<String>) -> Self {
        Self { format, device: device.into(), ..Self::default() }
    }

    /// Enable lazy loading.
    #[must_use]
    pub const fn with_lazy_load(mut self, lazy: bool) -> Self {
        self.lazy_load = lazy;
        self
    }

    /// Set a dtype cast for all loaded weights.
    #[must_use]
    pub const fn with_dtype_cast(mut self, dtype: DType) -> Self {
        self.dtype_cast = Some(dtype);
        self
    }

    /// Set the shard strategy.
    #[must_use]
    pub const fn with_shard_strategy(mut self, strategy: ShardStrategy) -> Self {
        self.shard_strategy = strategy;
        self
    }

    /// Set the cache capacity.
    #[must_use]
    pub const fn with_cache_capacity(mut self, cap: usize) -> Self {
        self.cache_capacity = cap;
        self
    }

    /// Set whether to validate weights after loading.
    #[must_use]
    pub const fn with_validation(mut self, validate: bool) -> Self {
        self.validate = validate;
        self
    }
}

// -- Tensor metadata ----------------------------------------------------------

/// Metadata describing a single tensor in a weight file.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TensorMeta {
    /// Tensor name (e.g. `"model.layers.0.self_attn.q_proj.weight"`).
    pub name: String,
    /// Shape dimensions.
    pub shape: Vec<usize>,
    /// Element data type.
    pub dtype: DType,
    /// Byte offset within the file.
    pub offset: u64,
    /// Size in bytes.
    pub size_bytes: u64,
    /// Index of the shard file containing this tensor (0 for single-file).
    pub shard_index: usize,
}

impl TensorMeta {
    /// Create new tensor metadata.
    #[must_use]
    pub fn new(
        name: impl Into<String>,
        shape: Vec<usize>,
        dtype: DType,
        offset: u64,
        size_bytes: u64,
    ) -> Self {
        Self { name: name.into(), shape, dtype, offset, size_bytes, shard_index: 0 }
    }

    /// Total number of elements in the tensor.
    #[must_use]
    pub fn num_elements(&self) -> usize {
        self.shape.iter().product()
    }

    /// Set the shard index.
    #[must_use]
    pub const fn with_shard_index(mut self, idx: usize) -> Self {
        self.shard_index = idx;
        self
    }
}

// -- Weight map ---------------------------------------------------------------

/// Maps weight names to their tensor metadata.
///
/// Acts as a table of contents for a model's weight file(s), enabling lookup
/// by name without reading the actual tensor data.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct WeightMap {
    entries: Vec<TensorMeta>,
    index: HashMap<String, usize>,
    total_bytes: u64,
}

impl WeightMap {
    /// Create an empty weight map.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert a tensor entry.
    pub fn insert(&mut self, meta: TensorMeta) {
        let idx = self.entries.len();
        self.total_bytes += meta.size_bytes;
        self.index.insert(meta.name.clone(), idx);
        self.entries.push(meta);
    }

    /// Look up tensor metadata by name.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<&TensorMeta> {
        self.index.get(name).map(|&i| &self.entries[i])
    }

    /// Check whether a weight exists.
    #[must_use]
    pub fn contains(&self, name: &str) -> bool {
        self.index.contains_key(name)
    }

    /// Number of tensors in the map.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the map is empty.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Total size of all tensors in bytes.
    #[must_use]
    pub const fn total_bytes(&self) -> u64 {
        self.total_bytes
    }

    /// Iterate over all tensor names.
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.entries.iter().map(|e| e.name.as_str())
    }

    /// Iterate over all entries.
    pub fn iter(&self) -> impl Iterator<Item = &TensorMeta> {
        self.entries.iter()
    }

    /// Return all tensors belonging to a specific shard index.
    #[must_use]
    pub fn tensors_in_shard(&self, shard: usize) -> Vec<&TensorMeta> {
        self.entries.iter().filter(|e| e.shard_index == shard).collect()
    }

    /// Build a weight map from a list of tensor metadata entries.
    #[must_use]
    pub fn from_entries(entries: Vec<TensorMeta>) -> Self {
        let mut map = Self::new();
        for entry in entries {
            map.insert(entry);
        }
        map
    }
}

// -- Loaded tensor ------------------------------------------------------------

/// A loaded weight tensor with its data and metadata.
#[derive(Debug, Clone, PartialEq)]
pub struct LoadedTensor {
    /// Tensor metadata.
    pub meta: TensorMeta,
    /// Raw tensor data as `f32` values (CPU reference representation).
    pub data: Vec<f32>,
}

impl LoadedTensor {
    /// Create a loaded tensor from metadata and data.
    #[must_use]
    pub const fn new(meta: TensorMeta, data: Vec<f32>) -> Self {
        Self { meta, data }
    }

    /// Number of elements.
    #[must_use]
    pub const fn num_elements(&self) -> usize {
        self.data.len()
    }

    /// Whether the data contains any NaN values.
    #[must_use]
    pub fn has_nan(&self) -> bool {
        self.data.iter().any(|v| v.is_nan())
    }

    /// Whether the data contains any infinity values.
    #[must_use]
    pub fn has_inf(&self) -> bool {
        self.data.iter().any(|v| v.is_infinite())
    }
}

// -- Lazy loader --------------------------------------------------------------

/// Lazy weight loader that reads tensors on demand.
///
/// Instead of loading the entire model into memory, [`LazyLoader`] reads the
/// weight map (table of contents) eagerly and then loads individual tensors
/// only when requested, using byte-offset seeking.
#[derive(Debug, Clone)]
pub struct LazyLoader {
    path: PathBuf,
    weight_map: WeightMap,
    loaded: HashMap<String, LoadedTensor>,
    format: WeightFormat,
}

impl LazyLoader {
    /// Create a lazy loader for the given file.
    ///
    /// The weight map must be pre-built (e.g. by parsing the file header).
    #[must_use]
    pub fn new(path: PathBuf, weight_map: WeightMap, format: WeightFormat) -> Self {
        Self { path, weight_map, loaded: HashMap::new(), format }
    }

    /// Path to the backing file.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// The underlying weight map.
    #[must_use]
    pub const fn weight_map(&self) -> &WeightMap {
        &self.weight_map
    }

    /// Format of the backing file.
    #[must_use]
    pub const fn format(&self) -> WeightFormat {
        self.format
    }

    /// Whether a tensor has already been loaded into memory.
    #[must_use]
    pub fn is_loaded(&self, name: &str) -> bool {
        self.loaded.contains_key(name)
    }

    /// Number of tensors currently resident in memory.
    #[must_use]
    pub fn resident_count(&self) -> usize {
        self.loaded.len()
    }

    /// Load a tensor by name.
    ///
    /// Returns a cached copy if already loaded, otherwise performs a
    /// simulated read from disk (CPU reference: generates deterministic
    /// data from the tensor metadata).
    pub fn load(&mut self, name: &str) -> Result<&LoadedTensor, WeightLoadError> {
        if self.loaded.contains_key(name) {
            return Ok(&self.loaded[name]);
        }

        let meta = self
            .weight_map
            .get(name)
            .ok_or_else(|| WeightLoadError::WeightNotFound(name.to_string()))?
            .clone();

        let num_elements = meta.num_elements();
        // CPU reference: deterministic data from offset.
        let data: Vec<f32> =
            (0..num_elements).map(|i| generate_deterministic_value(meta.offset, i)).collect();

        let tensor = LoadedTensor::new(meta, data);
        self.loaded.insert(name.to_string(), tensor);
        Ok(&self.loaded[name])
    }

    /// Evict a tensor from memory.
    pub fn evict(&mut self, name: &str) -> bool {
        self.loaded.remove(name).is_some()
    }

    /// Evict all loaded tensors.
    pub fn evict_all(&mut self) {
        self.loaded.clear();
    }
}

// -- Sharded loader -----------------------------------------------------------

/// Loads weights distributed across multiple shard files.
///
/// Each shard is a separate file containing a subset of the model's tensors.
/// The [`ShardedLoader`] maintains a unified [`WeightMap`] that tracks which
/// shard holds each tensor.
#[derive(Debug, Clone)]
pub struct ShardedLoader {
    shard_paths: Vec<PathBuf>,
    weight_map: WeightMap,
    format: WeightFormat,
    strategy: ShardStrategy,
    loaded: HashMap<String, LoadedTensor>,
}

impl ShardedLoader {
    /// Create a sharded loader.
    #[must_use]
    pub fn new(
        shard_paths: Vec<PathBuf>,
        weight_map: WeightMap,
        format: WeightFormat,
        strategy: ShardStrategy,
    ) -> Self {
        Self { shard_paths, weight_map, format, strategy, loaded: HashMap::new() }
    }

    /// Number of shards.
    #[must_use]
    pub const fn num_shards(&self) -> usize {
        self.shard_paths.len()
    }

    /// The unified weight map.
    #[must_use]
    pub const fn weight_map(&self) -> &WeightMap {
        &self.weight_map
    }

    /// Format of the shard files.
    #[must_use]
    pub const fn format(&self) -> WeightFormat {
        self.format
    }

    /// The sharding strategy.
    #[must_use]
    pub const fn strategy(&self) -> ShardStrategy {
        self.strategy
    }

    /// Path to a specific shard.
    #[must_use]
    pub fn shard_path(&self, index: usize) -> Option<&Path> {
        self.shard_paths.get(index).map(PathBuf::as_path)
    }

    /// Load a tensor by name from the appropriate shard.
    pub fn load(&mut self, name: &str) -> Result<&LoadedTensor, WeightLoadError> {
        if self.loaded.contains_key(name) {
            return Ok(&self.loaded[name]);
        }

        let meta = self
            .weight_map
            .get(name)
            .ok_or_else(|| WeightLoadError::WeightNotFound(name.to_string()))?
            .clone();

        if meta.shard_index >= self.shard_paths.len() {
            return Err(WeightLoadError::ShardError(format!(
                "tensor {name} references shard {} but only {} shards available",
                meta.shard_index,
                self.shard_paths.len()
            )));
        }

        let num_elements = meta.num_elements();
        // CPU reference: deterministic data from shard index + offset.
        let shard_base = (meta.shard_index as u64).saturating_mul(1000);
        let base_offset = shard_base.saturating_add(meta.offset);
        let data: Vec<f32> =
            (0..num_elements).map(|i| generate_deterministic_value(base_offset, i)).collect();

        let tensor = LoadedTensor::new(meta, data);
        self.loaded.insert(name.to_string(), tensor);
        Ok(&self.loaded[name])
    }

    /// Number of tensors currently loaded.
    #[must_use]
    pub fn loaded_count(&self) -> usize {
        self.loaded.len()
    }
}

/// Generate a deterministic `f32` value for CPU reference data.
#[allow(clippy::cast_precision_loss)]
fn generate_deterministic_value(base_offset: u64, index: usize) -> f32 {
    (base_offset as f32).mul_add(1.0, index as f32) * 0.001
}

// -- Weight transformer -------------------------------------------------------

/// Operation to apply to a weight tensor during loading.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TransformOp {
    /// Transpose the last two dimensions.
    Transpose,
    /// Reshape to the given dimensions.
    Reshape(Vec<usize>),
    /// Cast to a different dtype (represented as target dtype).
    Cast(DType),
    /// Scale all values by a constant factor.
    Scale(f64),
    /// Permute dimensions according to the given order.
    Permute(Vec<usize>),
}

impl fmt::Display for TransformOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Transpose => write!(f, "transpose"),
            Self::Reshape(shape) => write!(f, "reshape({shape:?})"),
            Self::Cast(dtype) => write!(f, "cast({dtype})"),
            Self::Scale(s) => write!(f, "scale({s})"),
            Self::Permute(order) => write!(f, "permute({order:?})"),
        }
    }
}

/// Transforms weight tensors during the loading pipeline.
///
/// A sequence of [`TransformOp`]s is applied to each tensor as it is loaded.
/// Operations are applied in order, and the shape/metadata are updated
/// accordingly.
#[derive(Debug, Clone, Default)]
pub struct WeightTransformer {
    global_ops: Vec<TransformOp>,
    per_tensor_ops: HashMap<String, Vec<TransformOp>>,
}

impl WeightTransformer {
    /// Create an empty transformer (no-op).
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a global transform applied to all tensors.
    pub fn add_global(&mut self, op: TransformOp) {
        self.global_ops.push(op);
    }

    /// Add a transform for a specific tensor name.
    pub fn add_for_tensor(&mut self, name: impl Into<String>, op: TransformOp) {
        self.per_tensor_ops.entry(name.into()).or_default().push(op);
    }

    /// Number of global transforms registered.
    #[must_use]
    pub const fn num_global_ops(&self) -> usize {
        self.global_ops.len()
    }

    /// Number of tensors with per-tensor transforms.
    #[must_use]
    pub fn num_tensor_specific(&self) -> usize {
        self.per_tensor_ops.len()
    }

    /// Get the list of transforms for a given tensor.
    #[must_use]
    pub fn ops_for(&self, name: &str) -> Vec<&TransformOp> {
        let mut ops: Vec<&TransformOp> = self.global_ops.iter().collect();
        if let Some(extra) = self.per_tensor_ops.get(name) {
            ops.extend(extra.iter());
        }
        ops
    }

    /// Apply all registered transforms to a loaded tensor.
    pub fn apply(&self, tensor: &mut LoadedTensor) -> Result<(), WeightLoadError> {
        let ops = self.ops_for(&tensor.meta.name);
        for op in ops {
            apply_single_op(op, tensor)?;
        }
        Ok(())
    }
}

/// Apply a single transform operation to a tensor.
fn apply_single_op(op: &TransformOp, tensor: &mut LoadedTensor) -> Result<(), WeightLoadError> {
    match op {
        TransformOp::Transpose => {
            let shape = &tensor.meta.shape;
            if shape.len() < 2 {
                return Err(WeightLoadError::ShapeMismatch {
                    name: tensor.meta.name.clone(),
                    expected: vec![0, 0],
                    actual: shape.clone(),
                });
            }
            let rows = shape[shape.len() - 2];
            let cols = shape[shape.len() - 1];
            let mut transposed = vec![0.0f32; tensor.data.len()];
            for r in 0..rows {
                for c in 0..cols {
                    transposed[c * rows + r] = tensor.data[r * cols + c];
                }
            }
            tensor.data = transposed;
            let len = tensor.meta.shape.len();
            tensor.meta.shape[len - 2] = cols;
            tensor.meta.shape[len - 1] = rows;
        }
        TransformOp::Reshape(target) => {
            let current_elems: usize = tensor.meta.shape.iter().product();
            let target_elems: usize = target.iter().product();
            if current_elems != target_elems {
                return Err(WeightLoadError::ShapeMismatch {
                    name: tensor.meta.name.clone(),
                    expected: target.clone(),
                    actual: tensor.meta.shape.clone(),
                });
            }
            tensor.meta.shape.clone_from(target);
        }
        TransformOp::Cast(dtype) => {
            // CPU reference: data stays as f32, only metadata updates.
            tensor.meta.dtype = *dtype;
        }
        TransformOp::Scale(factor) => {
            #[allow(clippy::cast_possible_truncation)]
            let f = *factor as f32;
            for v in &mut tensor.data {
                *v *= f;
            }
        }
        TransformOp::Permute(order) => {
            if order.len() != tensor.meta.shape.len() {
                return Err(WeightLoadError::ShapeMismatch {
                    name: tensor.meta.name.clone(),
                    expected: (0..order.len()).collect(),
                    actual: tensor.meta.shape.clone(),
                });
            }
            let new_shape: Vec<usize> = order.iter().map(|&i| tensor.meta.shape[i]).collect();
            tensor.meta.shape = new_shape;
            // CPU reference: permute is metadata-only.
        }
    }
    Ok(())
}

// -- Weight validator ---------------------------------------------------------

/// Validation check that can be applied to loaded weights.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ValidationCheck {
    /// Reject tensors containing NaN values.
    NoNaN,
    /// Reject tensors containing infinity values.
    NoInf,
    /// Reject tensors where all values are zero.
    NonZero,
    /// Reject tensors whose values fall outside `[-bound, bound]`.
    RangeBound,
    /// Verify shape matches an expected shape.
    ShapeMatch,
}

/// Result of validating a single tensor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValidationResult {
    /// Tensor name.
    pub name: String,
    /// Whether validation passed.
    pub passed: bool,
    /// Per-check results: (check, passed, message).
    pub checks: Vec<(ValidationCheck, bool, String)>,
}

impl ValidationResult {
    /// Create a passing result with no issues.
    #[must_use]
    pub fn pass(name: impl Into<String>) -> Self {
        Self { name: name.into(), passed: true, checks: Vec::new() }
    }
}

/// Validates loaded weight tensors for correctness.
///
/// Runs configurable checks (NaN, Inf, range, shape) against each tensor
/// and collects results.
#[derive(Debug, Clone)]
pub struct WeightValidator {
    checks: Vec<ValidationCheck>,
    range_bound: f32,
    expected_shapes: HashMap<String, Vec<usize>>,
}

impl Default for WeightValidator {
    fn default() -> Self {
        Self {
            checks: vec![ValidationCheck::NoNaN, ValidationCheck::NoInf],
            range_bound: 1e6,
            expected_shapes: HashMap::new(),
        }
    }
}

impl WeightValidator {
    /// Create a validator with default checks (`NoNaN`, `NoInf`).
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a validator with a custom set of checks.
    #[must_use]
    pub fn with_checks(checks: Vec<ValidationCheck>) -> Self {
        Self { checks, ..Self::default() }
    }

    /// Set the range bound for `RangeBound` checks.
    #[must_use]
    pub const fn with_range_bound(mut self, bound: f32) -> Self {
        self.range_bound = bound;
        self
    }

    /// Register an expected shape for a tensor.
    pub fn expect_shape(&mut self, name: impl Into<String>, shape: Vec<usize>) {
        self.expected_shapes.insert(name.into(), shape);
    }

    /// Number of registered checks.
    #[must_use]
    pub const fn num_checks(&self) -> usize {
        self.checks.len()
    }

    /// The current range bound.
    #[must_use]
    pub const fn range_bound(&self) -> f32 {
        self.range_bound
    }

    /// Validate a single tensor. Returns a [`ValidationResult`].
    #[must_use]
    pub fn validate(&self, tensor: &LoadedTensor) -> ValidationResult {
        let mut result =
            ValidationResult { name: tensor.meta.name.clone(), passed: true, checks: Vec::new() };

        for &check in &self.checks {
            let (ok, msg) = run_check(check, tensor, self.range_bound, &self.expected_shapes);
            if !ok {
                result.passed = false;
            }
            result.checks.push((check, ok, msg));
        }

        result
    }

    /// Validate multiple tensors, returning all results.
    #[must_use]
    pub fn validate_all(&self, tensors: &[LoadedTensor]) -> Vec<ValidationResult> {
        tensors.iter().map(|t| self.validate(t)).collect()
    }
}

/// Run a single validation check.
fn run_check(
    check: ValidationCheck,
    tensor: &LoadedTensor,
    range_bound: f32,
    expected_shapes: &HashMap<String, Vec<usize>>,
) -> (bool, String) {
    match check {
        ValidationCheck::NoNaN => {
            let ok = !tensor.has_nan();
            (ok, if ok { "no NaN".into() } else { "contains NaN".into() })
        }
        ValidationCheck::NoInf => {
            let ok = !tensor.has_inf();
            (ok, if ok { "no Inf".into() } else { "contains Inf".into() })
        }
        ValidationCheck::NonZero => {
            let ok = tensor.data.iter().any(|&v| v != 0.0);
            (ok, if ok { "non-zero".into() } else { "all zeros".into() })
        }
        ValidationCheck::RangeBound => {
            let ok = tensor.data.iter().all(|&v| v.abs() <= range_bound);
            (
                ok,
                if ok {
                    format!("within [-{range_bound}, {range_bound}]")
                } else {
                    format!("values outside [-{range_bound}, {range_bound}]")
                },
            )
        }
        ValidationCheck::ShapeMatch => expected_shapes.get(&tensor.meta.name).map_or_else(
            || (true, "no expected shape registered".into()),
            |expected| {
                let ok = tensor.meta.shape == *expected;
                (
                    ok,
                    if ok {
                        "shape matches".into()
                    } else {
                        format!(
                            "shape mismatch: expected {expected:?}, got {:?}",
                            tensor.meta.shape
                        )
                    },
                )
            },
        ),
    }
}

// -- Load progress ------------------------------------------------------------

/// Tracks progress of a model weight loading operation.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LoadProgress {
    /// Total number of tensors to load.
    pub total_tensors: usize,
    /// Number of tensors loaded so far.
    pub loaded_tensors: usize,
    /// Total bytes to load.
    pub total_bytes: u64,
    /// Bytes loaded so far.
    pub loaded_bytes: u64,
    /// Name of the tensor currently being loaded.
    pub current_tensor: Option<String>,
    /// Names of tensors that failed to load.
    pub failed: Vec<String>,
    /// Whether loading is complete.
    pub completed: bool,
}

impl LoadProgress {
    /// Create a new progress tracker for the given totals.
    #[must_use]
    pub const fn new(total_tensors: usize, total_bytes: u64) -> Self {
        Self {
            total_tensors,
            loaded_tensors: 0,
            total_bytes,
            loaded_bytes: 0,
            current_tensor: None,
            failed: Vec::new(),
            completed: false,
        }
    }

    /// Progress as a fraction in `[0.0, 1.0]`.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn fraction(&self) -> f64 {
        if self.total_tensors == 0 {
            return 0.0;
        }
        self.loaded_tensors as f64 / self.total_tensors as f64
    }

    /// Progress as a percentage string.
    #[must_use]
    pub fn percent_string(&self) -> String {
        format!("{:.1}%", self.fraction() * 100.0)
    }

    /// Byte-level progress fraction.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn byte_fraction(&self) -> f64 {
        if self.total_bytes == 0 {
            return 0.0;
        }
        self.loaded_bytes as f64 / self.total_bytes as f64
    }

    /// Record that a tensor was loaded successfully.
    pub fn record_loaded(&mut self, name: &str, size_bytes: u64) {
        self.loaded_tensors += 1;
        self.loaded_bytes += size_bytes;
        self.current_tensor = Some(name.to_string());
        if self.loaded_tensors >= self.total_tensors {
            self.completed = true;
            self.current_tensor = None;
        }
    }

    /// Record that a tensor failed to load.
    pub fn record_failed(&mut self, name: &str) {
        self.failed.push(name.to_string());
        self.loaded_tensors += 1;
        if self.loaded_tensors >= self.total_tensors {
            self.completed = true;
            self.current_tensor = None;
        }
    }

    /// Whether any tensors failed to load.
    #[must_use]
    pub const fn has_failures(&self) -> bool {
        !self.failed.is_empty()
    }

    /// Number of failures.
    #[must_use]
    pub const fn failure_count(&self) -> usize {
        self.failed.len()
    }

    /// Whether all tensors have been processed (loaded or failed).
    #[must_use]
    pub const fn is_complete(&self) -> bool {
        self.completed
    }
}

impl fmt::Display for LoadProgress {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "[{}/{}] {} ({} bytes / {} bytes)",
            self.loaded_tensors,
            self.total_tensors,
            self.percent_string(),
            self.loaded_bytes,
            self.total_bytes,
        )
    }
}

// -- Weight cache -------------------------------------------------------------

/// Simple LRU-style cache for frequently-accessed weight tensors.
///
/// Keeps at most `capacity` tensors resident in memory. When the cache is
/// full the least-recently-inserted entry is evicted.
#[derive(Debug, Clone)]
pub struct WeightCache {
    capacity: usize,
    entries: HashMap<String, LoadedTensor>,
    order: Vec<String>,
    hits: u64,
    misses: u64,
}

impl WeightCache {
    /// Create a cache with the given capacity.
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        Self { capacity, entries: HashMap::new(), order: Vec::new(), hits: 0, misses: 0 }
    }

    /// Maximum number of cached tensors.
    #[must_use]
    pub const fn capacity(&self) -> usize {
        self.capacity
    }

    /// Number of tensors currently cached.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the cache is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Total cache hits.
    #[must_use]
    pub const fn hits(&self) -> u64 {
        self.hits
    }

    /// Total cache misses.
    #[must_use]
    pub const fn misses(&self) -> u64 {
        self.misses
    }

    /// Cache hit rate as a fraction in `[0.0, 1.0]`.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            return 0.0;
        }
        self.hits as f64 / total as f64
    }

    /// Look up a tensor in the cache.
    pub fn get(&mut self, name: &str) -> Option<&LoadedTensor> {
        if self.entries.contains_key(name) {
            self.hits += 1;
            self.entries.get(name)
        } else {
            self.misses += 1;
            None
        }
    }

    /// Insert a tensor into the cache, evicting the oldest if full.
    pub fn insert(&mut self, tensor: LoadedTensor) {
        let name = tensor.meta.name.clone();

        if self.entries.contains_key(&name) {
            self.order.retain(|n| n != &name);
        } else if self.entries.len() >= self.capacity
            && self.capacity > 0
            && let Some(oldest) = self.order.first().cloned()
        {
            self.entries.remove(&oldest);
            self.order.remove(0);
        }

        self.order.push(name.clone());
        self.entries.insert(name, tensor);
    }

    /// Remove a tensor from the cache.
    pub fn evict(&mut self, name: &str) -> bool {
        if self.entries.remove(name).is_some() {
            self.order.retain(|n| n != name);
            true
        } else {
            false
        }
    }

    /// Clear the cache.
    pub fn clear(&mut self) {
        self.entries.clear();
        self.order.clear();
    }

    /// Reset hit/miss counters.
    pub const fn reset_stats(&mut self) {
        self.hits = 0;
        self.misses = 0;
    }

    /// Check whether a tensor is cached (without updating stats).
    #[must_use]
    pub fn contains(&self, name: &str) -> bool {
        self.entries.contains_key(name)
    }
}

// -- Weight loading engine ----------------------------------------------------

/// Unified weight loading pipeline.
///
/// Orchestrates format detection, lazy/eager loading, sharded file assembly,
/// dtype casting, weight transformation, validation, caching, and progress
/// tracking into a single API.
#[derive(Debug, Clone)]
pub struct WeightLoadingEngine {
    config: LoadConfig,
    weight_map: WeightMap,
    transformer: WeightTransformer,
    validator: WeightValidator,
    cache: WeightCache,
    progress: LoadProgress,
    tensors: HashMap<String, LoadedTensor>,
}

impl WeightLoadingEngine {
    /// Create a new engine with the given configuration and weight map.
    #[must_use]
    pub fn new(config: LoadConfig, weight_map: WeightMap) -> Self {
        let total_tensors = weight_map.len();
        let total_bytes = weight_map.total_bytes();
        let cache_cap = config.cache_capacity;
        Self {
            config,
            weight_map,
            transformer: WeightTransformer::new(),
            validator: WeightValidator::new(),
            cache: WeightCache::new(cache_cap),
            progress: LoadProgress::new(total_tensors, total_bytes),
            tensors: HashMap::new(),
        }
    }

    /// The pipeline configuration.
    #[must_use]
    pub const fn config(&self) -> &LoadConfig {
        &self.config
    }

    /// The weight map.
    #[must_use]
    pub const fn weight_map(&self) -> &WeightMap {
        &self.weight_map
    }

    /// Current progress.
    #[must_use]
    pub const fn progress(&self) -> &LoadProgress {
        &self.progress
    }

    /// Mutable access to the transformer for registering transforms.
    pub const fn transformer_mut(&mut self) -> &mut WeightTransformer {
        &mut self.transformer
    }

    /// Mutable access to the validator for registering expected shapes.
    pub const fn validator_mut(&mut self) -> &mut WeightValidator {
        &mut self.validator
    }

    /// Number of tensors loaded and validated.
    #[must_use]
    pub fn loaded_count(&self) -> usize {
        self.tensors.len()
    }

    /// Get a loaded tensor by name.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<&LoadedTensor> {
        self.tensors.get(name)
    }

    /// Load a single tensor by name.
    ///
    /// CPU reference: generates deterministic data, applies transforms,
    /// runs validation, and caches the result.
    pub fn load_tensor(&mut self, name: &str) -> Result<&LoadedTensor, WeightLoadError> {
        if self.tensors.contains_key(name) {
            return Ok(&self.tensors[name]);
        }

        let meta = self
            .weight_map
            .get(name)
            .ok_or_else(|| WeightLoadError::WeightNotFound(name.to_string()))?
            .clone();

        self.progress.current_tensor = Some(name.to_string());

        let num_elements = meta.num_elements();
        let data: Vec<f32> =
            (0..num_elements).map(|i| generate_deterministic_value(meta.offset, i)).collect();

        let mut tensor = LoadedTensor::new(meta, data);

        self.transformer.apply(&mut tensor)?;

        if self.config.validate {
            let result = self.validator.validate(&tensor);
            if !result.passed {
                self.progress.record_failed(name);
                let reasons: Vec<String> = result
                    .checks
                    .iter()
                    .filter(|(_, ok, _)| !ok)
                    .map(|(_, _, msg)| msg.clone())
                    .collect();
                return Err(WeightLoadError::InvalidValues(format!(
                    "{name}: {}",
                    reasons.join(", ")
                )));
            }
        }

        self.progress.record_loaded(name, tensor.meta.size_bytes);
        self.cache.insert(tensor.clone());
        self.tensors.insert(name.to_string(), tensor);
        Ok(&self.tensors[name])
    }

    /// Load all tensors in the weight map.
    pub fn load_all(&mut self) -> Result<usize, WeightLoadError> {
        let names: Vec<String> = self.weight_map.names().map(String::from).collect();
        let mut count = 0;
        for name in &names {
            self.load_tensor(name)?;
            count += 1;
        }
        Ok(count)
    }

    /// Check whether a tensor has been loaded.
    #[must_use]
    pub fn is_loaded(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }

    /// Evict a tensor from the engine (and cache).
    pub fn evict(&mut self, name: &str) -> bool {
        self.cache.evict(name);
        self.tensors.remove(name).is_some()
    }

    /// Total bytes loaded so far.
    #[must_use]
    pub const fn loaded_bytes(&self) -> u64 {
        self.progress.loaded_bytes
    }
}

// -- Tests --------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- Helpers --

    fn sample_meta(name: &str, shape: Vec<usize>) -> TensorMeta {
        let elems: usize = shape.iter().product();
        let size = (elems * 4) as u64;
        TensorMeta::new(name, shape, DType::F32, 0, size)
    }

    fn sample_meta_at(name: &str, shape: Vec<usize>, offset: u64) -> TensorMeta {
        let elems: usize = shape.iter().product();
        let size = (elems * 4) as u64;
        TensorMeta::new(name, shape, DType::F32, offset, size)
    }

    fn sample_tensor(name: &str, shape: Vec<usize>, data: Vec<f32>) -> LoadedTensor {
        let meta = sample_meta(name, shape);
        LoadedTensor::new(meta, data)
    }

    fn sample_weight_map(n: usize) -> WeightMap {
        let mut wm = WeightMap::new();
        for i in 0..n {
            let name = format!("layer.{i}.weight");
            let meta = sample_meta_at(&name, vec![4, 4], (i * 100) as u64);
            wm.insert(meta);
        }
        wm
    }

    // -- WeightFormat tests --

    #[test]
    fn weight_format_default() {
        assert_eq!(WeightFormat::default(), WeightFormat::Gguf);
    }

    #[test]
    fn weight_format_display() {
        assert_eq!(WeightFormat::Gguf.to_string(), "gguf");
        assert_eq!(WeightFormat::SafeTensors.to_string(), "safetensors");
        assert_eq!(WeightFormat::PyTorch.to_string(), "pytorch");
        assert_eq!(WeightFormat::NumPy.to_string(), "numpy");
        assert_eq!(WeightFormat::Raw.to_string(), "raw");
    }

    #[test]
    fn weight_format_from_extension_gguf() {
        let fmt = WeightFormat::from_extension(Path::new("model.gguf"));
        assert_eq!(fmt, Some(WeightFormat::Gguf));
    }

    #[test]
    fn weight_format_from_extension_safetensors() {
        let fmt = WeightFormat::from_extension(Path::new("model.safetensors"));
        assert_eq!(fmt, Some(WeightFormat::SafeTensors));
    }

    #[test]
    fn weight_format_from_extension_pytorch_pt() {
        assert_eq!(
            WeightFormat::from_extension(Path::new("model.pt")),
            Some(WeightFormat::PyTorch)
        );
    }

    #[test]
    fn weight_format_from_extension_pytorch_bin() {
        assert_eq!(
            WeightFormat::from_extension(Path::new("model.bin")),
            Some(WeightFormat::PyTorch)
        );
    }

    #[test]
    fn weight_format_from_extension_pytorch_pth() {
        assert_eq!(
            WeightFormat::from_extension(Path::new("weights.pth")),
            Some(WeightFormat::PyTorch)
        );
    }

    #[test]
    fn weight_format_from_extension_numpy_npy() {
        assert_eq!(WeightFormat::from_extension(Path::new("data.npy")), Some(WeightFormat::NumPy));
    }

    #[test]
    fn weight_format_from_extension_numpy_npz() {
        assert_eq!(WeightFormat::from_extension(Path::new("data.npz")), Some(WeightFormat::NumPy));
    }

    #[test]
    fn weight_format_from_extension_raw() {
        assert_eq!(WeightFormat::from_extension(Path::new("blob.raw")), Some(WeightFormat::Raw));
    }

    #[test]
    fn weight_format_from_extension_unknown() {
        assert_eq!(WeightFormat::from_extension(Path::new("file.txt")), None);
    }

    #[test]
    fn weight_format_from_extension_no_ext() {
        assert_eq!(WeightFormat::from_extension(Path::new("model")), None);
    }

    #[test]
    fn weight_format_serde_roundtrip() {
        for fmt in [
            WeightFormat::Gguf,
            WeightFormat::SafeTensors,
            WeightFormat::PyTorch,
            WeightFormat::NumPy,
            WeightFormat::Raw,
        ] {
            let json = serde_json::to_string(&fmt).unwrap();
            let back: WeightFormat = serde_json::from_str(&json).unwrap();
            assert_eq!(fmt, back);
        }
    }

    #[test]
    fn weight_format_equality() {
        assert_eq!(WeightFormat::Gguf, WeightFormat::Gguf);
        assert_ne!(WeightFormat::Gguf, WeightFormat::SafeTensors);
    }

    // -- DType tests --

    #[test]
    fn dtype_default() {
        assert_eq!(DType::default(), DType::F32);
    }

    #[test]
    fn dtype_display() {
        assert_eq!(DType::F32.to_string(), "f32");
        assert_eq!(DType::F16.to_string(), "f16");
        assert_eq!(DType::BF16.to_string(), "bf16");
        assert_eq!(DType::I8.to_string(), "i8");
        assert_eq!(DType::I2.to_string(), "i2");
        assert_eq!(DType::U4.to_string(), "u4");
    }

    #[test]
    fn dtype_element_size() {
        assert_eq!(DType::F32.element_size_bytes(), 4);
        assert_eq!(DType::F16.element_size_bytes(), 2);
        assert_eq!(DType::BF16.element_size_bytes(), 2);
        assert_eq!(DType::I8.element_size_bytes(), 1);
        assert_eq!(DType::I2.element_size_bytes(), 1);
        assert_eq!(DType::U4.element_size_bytes(), 1);
    }

    #[test]
    fn dtype_serde_roundtrip() {
        for dtype in [DType::F32, DType::F16, DType::BF16, DType::I8, DType::I2, DType::U4] {
            let json = serde_json::to_string(&dtype).unwrap();
            let back: DType = serde_json::from_str(&json).unwrap();
            assert_eq!(dtype, back);
        }
    }

    // -- ShardStrategy tests --

    #[test]
    fn shard_strategy_default() {
        assert_eq!(ShardStrategy::default(), ShardStrategy::Single);
    }

    #[test]
    fn shard_strategy_display() {
        assert_eq!(ShardStrategy::Single.to_string(), "single");
        assert_eq!(ShardStrategy::ByLayer.to_string(), "by_layer");
        assert_eq!(ShardStrategy::BySize.to_string(), "by_size");
        assert_eq!(ShardStrategy::Custom.to_string(), "custom");
    }

    #[test]
    fn shard_strategy_serde_roundtrip() {
        for s in [
            ShardStrategy::Single,
            ShardStrategy::ByLayer,
            ShardStrategy::BySize,
            ShardStrategy::Custom,
        ] {
            let json = serde_json::to_string(&s).unwrap();
            let back: ShardStrategy = serde_json::from_str(&json).unwrap();
            assert_eq!(s, back);
        }
    }

    // -- LoadConfig tests --

    #[test]
    fn load_config_default() {
        let cfg = LoadConfig::default();
        assert_eq!(cfg.format, WeightFormat::Gguf);
        assert_eq!(cfg.device, "cpu");
        assert!(cfg.dtype_cast.is_none());
        assert!(!cfg.lazy_load);
        assert_eq!(cfg.shard_strategy, ShardStrategy::Single);
        assert_eq!(cfg.cache_capacity, 256);
        assert!(cfg.validate);
    }

    #[test]
    fn load_config_new() {
        let cfg = LoadConfig::new(WeightFormat::SafeTensors, "cuda:0");
        assert_eq!(cfg.format, WeightFormat::SafeTensors);
        assert_eq!(cfg.device, "cuda:0");
    }

    #[test]
    fn load_config_builder_methods() {
        let cfg = LoadConfig::new(WeightFormat::Gguf, "cpu")
            .with_lazy_load(true)
            .with_dtype_cast(DType::F16)
            .with_shard_strategy(ShardStrategy::ByLayer)
            .with_cache_capacity(512)
            .with_validation(false);
        assert!(cfg.lazy_load);
        assert_eq!(cfg.dtype_cast, Some(DType::F16));
        assert_eq!(cfg.shard_strategy, ShardStrategy::ByLayer);
        assert_eq!(cfg.cache_capacity, 512);
        assert!(!cfg.validate);
    }

    #[test]
    fn load_config_serde_roundtrip() {
        let cfg = LoadConfig::new(WeightFormat::SafeTensors, "cuda:0")
            .with_lazy_load(true)
            .with_dtype_cast(DType::BF16);
        let json = serde_json::to_string(&cfg).unwrap();
        let back: LoadConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(back.format, cfg.format);
        assert_eq!(back.device, cfg.device);
        assert_eq!(back.dtype_cast, cfg.dtype_cast);
        assert_eq!(back.lazy_load, cfg.lazy_load);
    }

    // -- TensorMeta tests --

    #[test]
    fn tensor_meta_new() {
        let meta = TensorMeta::new("weight", vec![3, 4], DType::F32, 100, 48);
        assert_eq!(meta.name, "weight");
        assert_eq!(meta.shape, vec![3, 4]);
        assert_eq!(meta.dtype, DType::F32);
        assert_eq!(meta.offset, 100);
        assert_eq!(meta.size_bytes, 48);
        assert_eq!(meta.shard_index, 0);
    }

    #[test]
    fn tensor_meta_num_elements() {
        let meta = TensorMeta::new("w", vec![3, 4, 5], DType::F32, 0, 240);
        assert_eq!(meta.num_elements(), 60);
    }

    #[test]
    fn tensor_meta_with_shard_index() {
        let meta = TensorMeta::new("w", vec![2, 2], DType::F32, 0, 16).with_shard_index(3);
        assert_eq!(meta.shard_index, 3);
    }

    #[test]
    fn tensor_meta_serde_roundtrip() {
        let meta = TensorMeta::new("layer.0.weight", vec![4, 4], DType::F16, 64, 32);
        let json = serde_json::to_string(&meta).unwrap();
        let back: TensorMeta = serde_json::from_str(&json).unwrap();
        assert_eq!(meta, back);
    }

    #[test]
    fn tensor_meta_scalar_shape() {
        let meta = TensorMeta::new("bias", vec![1], DType::F32, 0, 4);
        assert_eq!(meta.num_elements(), 1);
    }

    // -- WeightMap tests --

    #[test]
    fn weight_map_empty() {
        let wm = WeightMap::new();
        assert!(wm.is_empty());
        assert_eq!(wm.len(), 0);
        assert_eq!(wm.total_bytes(), 0);
    }

    #[test]
    fn weight_map_insert_and_get() {
        let mut wm = WeightMap::new();
        wm.insert(sample_meta("layer.0.weight", vec![4, 4]));
        assert_eq!(wm.len(), 1);
        assert!(wm.contains("layer.0.weight"));
        assert!(!wm.contains("layer.1.weight"));
        let got = wm.get("layer.0.weight").unwrap();
        assert_eq!(got.shape, vec![4, 4]);
    }

    #[test]
    fn weight_map_total_bytes() {
        let mut wm = WeightMap::new();
        wm.insert(TensorMeta::new("a", vec![4], DType::F32, 0, 16));
        wm.insert(TensorMeta::new("b", vec![8], DType::F32, 16, 32));
        assert_eq!(wm.total_bytes(), 48);
    }

    #[test]
    fn weight_map_names() {
        let wm = sample_weight_map(3);
        let names: Vec<&str> = wm.names().collect();
        assert_eq!(names.len(), 3);
        assert!(names.contains(&"layer.0.weight"));
        assert!(names.contains(&"layer.1.weight"));
        assert!(names.contains(&"layer.2.weight"));
    }

    #[test]
    fn weight_map_iter() {
        let wm = sample_weight_map(2);
        assert_eq!(wm.iter().count(), 2);
    }

    #[test]
    fn weight_map_tensors_in_shard() {
        let mut wm = WeightMap::new();
        wm.insert(TensorMeta::new("a", vec![4], DType::F32, 0, 16).with_shard_index(0));
        wm.insert(TensorMeta::new("b", vec![4], DType::F32, 0, 16).with_shard_index(1));
        wm.insert(TensorMeta::new("c", vec![4], DType::F32, 0, 16).with_shard_index(0));
        assert_eq!(wm.tensors_in_shard(0).len(), 2);
        assert_eq!(wm.tensors_in_shard(1).len(), 1);
        assert_eq!(wm.tensors_in_shard(2).len(), 0);
    }

    #[test]
    fn weight_map_from_entries() {
        let entries = vec![sample_meta("a", vec![2, 2]), sample_meta("b", vec![3, 3])];
        let wm = WeightMap::from_entries(entries);
        assert_eq!(wm.len(), 2);
        assert!(wm.contains("a"));
        assert!(wm.contains("b"));
    }

    #[test]
    fn weight_map_get_missing() {
        let wm = WeightMap::new();
        assert!(wm.get("nonexistent").is_none());
    }

    // -- LoadedTensor tests --

    #[test]
    fn loaded_tensor_basic() {
        let t = sample_tensor("w", vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(t.num_elements(), 6);
        assert!(!t.has_nan());
        assert!(!t.has_inf());
    }

    #[test]
    fn loaded_tensor_has_nan() {
        let t = sample_tensor("w", vec![3], vec![1.0, f32::NAN, 3.0]);
        assert!(t.has_nan());
    }

    #[test]
    fn loaded_tensor_has_inf() {
        let t = sample_tensor("w", vec![2], vec![f32::INFINITY, 1.0]);
        assert!(t.has_inf());
    }

    #[test]
    fn loaded_tensor_neg_inf() {
        let t = sample_tensor("w", vec![2], vec![f32::NEG_INFINITY, 0.0]);
        assert!(t.has_inf());
    }

    #[test]
    fn loaded_tensor_clean() {
        let t = sample_tensor("w", vec![4], vec![0.0, 0.5, -0.5, 1.0]);
        assert!(!t.has_nan());
        assert!(!t.has_inf());
    }

    // -- LazyLoader tests --

    #[test]
    fn lazy_loader_new() {
        let wm = sample_weight_map(3);
        let loader = LazyLoader::new(PathBuf::from("model.gguf"), wm, WeightFormat::Gguf);
        assert_eq!(loader.path(), Path::new("model.gguf"));
        assert_eq!(loader.format(), WeightFormat::Gguf);
        assert_eq!(loader.resident_count(), 0);
        assert_eq!(loader.weight_map().len(), 3);
    }

    #[test]
    fn lazy_loader_load() {
        let wm = sample_weight_map(2);
        let mut loader = LazyLoader::new(PathBuf::from("model.gguf"), wm, WeightFormat::Gguf);
        let tensor = loader.load("layer.0.weight").unwrap();
        assert_eq!(tensor.num_elements(), 16);
        assert!(loader.is_loaded("layer.0.weight"));
        assert!(!loader.is_loaded("layer.1.weight"));
        assert_eq!(loader.resident_count(), 1);
    }

    #[test]
    fn lazy_loader_load_cached() {
        let wm = sample_weight_map(1);
        let mut loader = LazyLoader::new(PathBuf::from("m.gguf"), wm, WeightFormat::Gguf);
        loader.load("layer.0.weight").unwrap();
        let t = loader.load("layer.0.weight").unwrap();
        assert_eq!(t.num_elements(), 16);
        assert_eq!(loader.resident_count(), 1);
    }

    #[test]
    fn lazy_loader_load_not_found() {
        let wm = sample_weight_map(1);
        let mut loader = LazyLoader::new(PathBuf::from("m.gguf"), wm, WeightFormat::Gguf);
        assert!(loader.load("nonexistent").is_err());
    }

    #[test]
    fn lazy_loader_evict() {
        let wm = sample_weight_map(2);
        let mut loader = LazyLoader::new(PathBuf::from("m.gguf"), wm, WeightFormat::Gguf);
        loader.load("layer.0.weight").unwrap();
        assert!(loader.evict("layer.0.weight"));
        assert!(!loader.is_loaded("layer.0.weight"));
        assert_eq!(loader.resident_count(), 0);
    }

    #[test]
    fn lazy_loader_evict_nonexistent() {
        let wm = sample_weight_map(1);
        let mut loader = LazyLoader::new(PathBuf::from("m.gguf"), wm, WeightFormat::Gguf);
        assert!(!loader.evict("nonexistent"));
    }

    #[test]
    fn lazy_loader_evict_all() {
        let wm = sample_weight_map(3);
        let mut loader = LazyLoader::new(PathBuf::from("m.gguf"), wm, WeightFormat::Gguf);
        loader.load("layer.0.weight").unwrap();
        loader.load("layer.1.weight").unwrap();
        loader.evict_all();
        assert_eq!(loader.resident_count(), 0);
    }

    #[test]
    fn lazy_loader_data_deterministic() {
        let wm = sample_weight_map(1);
        let mut loader1 = LazyLoader::new(PathBuf::from("m.gguf"), wm.clone(), WeightFormat::Gguf);
        let mut loader2 = LazyLoader::new(PathBuf::from("m.gguf"), wm, WeightFormat::Gguf);
        let t1 = loader1.load("layer.0.weight").unwrap().data.clone();
        let t2 = loader2.load("layer.0.weight").unwrap().data.clone();
        assert_eq!(t1, t2);
    }

    // -- ShardedLoader tests --

    #[test]
    fn sharded_loader_new() {
        let mut wm = WeightMap::new();
        wm.insert(TensorMeta::new("a", vec![4, 4], DType::F32, 0, 64).with_shard_index(0));
        wm.insert(TensorMeta::new("b", vec![4, 4], DType::F32, 0, 64).with_shard_index(1));
        let loader = ShardedLoader::new(
            vec![PathBuf::from("shard0.gguf"), PathBuf::from("shard1.gguf")],
            wm,
            WeightFormat::Gguf,
            ShardStrategy::ByLayer,
        );
        assert_eq!(loader.num_shards(), 2);
        assert_eq!(loader.strategy(), ShardStrategy::ByLayer);
        assert_eq!(loader.format(), WeightFormat::Gguf);
        assert_eq!(loader.loaded_count(), 0);
    }

    #[test]
    fn sharded_loader_load() {
        let mut wm = WeightMap::new();
        wm.insert(TensorMeta::new("a", vec![2, 2], DType::F32, 0, 16).with_shard_index(0));
        wm.insert(TensorMeta::new("b", vec![2, 2], DType::F32, 0, 16).with_shard_index(1));
        let mut loader = ShardedLoader::new(
            vec![PathBuf::from("s0.gguf"), PathBuf::from("s1.gguf")],
            wm,
            WeightFormat::Gguf,
            ShardStrategy::ByLayer,
        );
        let t = loader.load("a").unwrap();
        assert_eq!(t.num_elements(), 4);
        assert_eq!(loader.loaded_count(), 1);
    }

    #[test]
    fn sharded_loader_load_from_second_shard() {
        let mut wm = WeightMap::new();
        wm.insert(TensorMeta::new("a", vec![2, 2], DType::F32, 0, 16).with_shard_index(0));
        wm.insert(TensorMeta::new("b", vec![2, 2], DType::F32, 0, 16).with_shard_index(1));
        let mut loader = ShardedLoader::new(
            vec![PathBuf::from("s0.gguf"), PathBuf::from("s1.gguf")],
            wm,
            WeightFormat::Gguf,
            ShardStrategy::ByLayer,
        );
        let t = loader.load("b").unwrap();
        assert_eq!(t.num_elements(), 4);
    }

    #[test]
    fn sharded_loader_not_found() {
        let wm = WeightMap::new();
        let mut loader = ShardedLoader::new(
            vec![PathBuf::from("s0.gguf")],
            wm,
            WeightFormat::Gguf,
            ShardStrategy::Single,
        );
        assert!(loader.load("missing").is_err());
    }

    #[test]
    fn sharded_loader_invalid_shard_index() {
        let mut wm = WeightMap::new();
        wm.insert(TensorMeta::new("a", vec![2, 2], DType::F32, 0, 16).with_shard_index(5));
        let mut loader = ShardedLoader::new(
            vec![PathBuf::from("s0.gguf")],
            wm,
            WeightFormat::Gguf,
            ShardStrategy::Single,
        );
        let err = loader.load("a").unwrap_err();
        assert!(err.to_string().contains("shard"));
    }

    #[test]
    fn sharded_loader_shard_path() {
        let loader = ShardedLoader::new(
            vec![PathBuf::from("a.gguf"), PathBuf::from("b.gguf")],
            WeightMap::new(),
            WeightFormat::Gguf,
            ShardStrategy::Single,
        );
        assert_eq!(loader.shard_path(0), Some(Path::new("a.gguf")));
        assert_eq!(loader.shard_path(1), Some(Path::new("b.gguf")));
        assert_eq!(loader.shard_path(2), None);
    }

    #[test]
    fn sharded_loader_cached_load() {
        let mut wm = WeightMap::new();
        wm.insert(TensorMeta::new("a", vec![2, 2], DType::F32, 0, 16).with_shard_index(0));
        let mut loader = ShardedLoader::new(
            vec![PathBuf::from("s0.gguf")],
            wm,
            WeightFormat::Gguf,
            ShardStrategy::Single,
        );
        loader.load("a").unwrap();
        let t = loader.load("a").unwrap();
        assert_eq!(t.num_elements(), 4);
        assert_eq!(loader.loaded_count(), 1);
    }

    // -- TransformOp tests --

    #[test]
    fn transform_op_display() {
        assert_eq!(TransformOp::Transpose.to_string(), "transpose");
        assert_eq!(TransformOp::Reshape(vec![2, 3]).to_string(), "reshape([2, 3])");
        assert_eq!(TransformOp::Cast(DType::F16).to_string(), "cast(f16)");
        assert_eq!(TransformOp::Scale(2.0).to_string(), "scale(2)");
        assert_eq!(TransformOp::Permute(vec![1, 0]).to_string(), "permute([1, 0])");
    }

    // -- WeightTransformer tests --

    #[test]
    fn transformer_empty() {
        let tr = WeightTransformer::new();
        assert_eq!(tr.num_global_ops(), 0);
        assert_eq!(tr.num_tensor_specific(), 0);
    }

    #[test]
    fn transformer_global_ops() {
        let mut tr = WeightTransformer::new();
        tr.add_global(TransformOp::Scale(2.0));
        tr.add_global(TransformOp::Cast(DType::F16));
        assert_eq!(tr.num_global_ops(), 2);
    }

    #[test]
    fn transformer_per_tensor_ops() {
        let mut tr = WeightTransformer::new();
        tr.add_for_tensor("layer.0.weight", TransformOp::Transpose);
        tr.add_for_tensor("layer.1.weight", TransformOp::Scale(0.5));
        assert_eq!(tr.num_tensor_specific(), 2);
    }

    #[test]
    fn transformer_ops_for_combines() {
        let mut tr = WeightTransformer::new();
        tr.add_global(TransformOp::Scale(2.0));
        tr.add_for_tensor("w", TransformOp::Transpose);
        let ops = tr.ops_for("w");
        assert_eq!(ops.len(), 2);
    }

    #[test]
    fn transformer_ops_for_global_only() {
        let mut tr = WeightTransformer::new();
        tr.add_global(TransformOp::Cast(DType::F16));
        let ops = tr.ops_for("any_tensor");
        assert_eq!(ops.len(), 1);
    }

    #[test]
    fn transformer_apply_scale() {
        let mut tr = WeightTransformer::new();
        tr.add_global(TransformOp::Scale(2.0));
        let mut tensor = sample_tensor("w", vec![3], vec![1.0, 2.0, 3.0]);
        tr.apply(&mut tensor).unwrap();
        assert_eq!(tensor.data, vec![2.0, 4.0, 6.0]);
    }

    #[test]
    fn transformer_apply_transpose_2d() {
        let mut tr = WeightTransformer::new();
        tr.add_global(TransformOp::Transpose);
        let mut tensor = sample_tensor("w", vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        tr.apply(&mut tensor).unwrap();
        assert_eq!(tensor.meta.shape, vec![3, 2]);
        assert_eq!(tensor.data, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn transformer_apply_transpose_1d_fails() {
        let mut tr = WeightTransformer::new();
        tr.add_global(TransformOp::Transpose);
        let mut tensor = sample_tensor("w", vec![4], vec![1.0, 2.0, 3.0, 4.0]);
        assert!(tr.apply(&mut tensor).is_err());
    }

    #[test]
    fn transformer_apply_reshape() {
        let mut tr = WeightTransformer::new();
        tr.add_global(TransformOp::Reshape(vec![3, 2]));
        let mut tensor = sample_tensor("w", vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        tr.apply(&mut tensor).unwrap();
        assert_eq!(tensor.meta.shape, vec![3, 2]);
    }

    #[test]
    fn transformer_apply_reshape_mismatch() {
        let mut tr = WeightTransformer::new();
        tr.add_global(TransformOp::Reshape(vec![5, 5]));
        let mut tensor = sample_tensor("w", vec![4], vec![1.0, 2.0, 3.0, 4.0]);
        assert!(tr.apply(&mut tensor).is_err());
    }

    #[test]
    fn transformer_apply_cast() {
        let mut tr = WeightTransformer::new();
        tr.add_global(TransformOp::Cast(DType::F16));
        let mut tensor = sample_tensor("w", vec![2], vec![1.0, 2.0]);
        tr.apply(&mut tensor).unwrap();
        assert_eq!(tensor.meta.dtype, DType::F16);
        assert_eq!(tensor.data, vec![1.0, 2.0]);
    }

    #[test]
    fn transformer_apply_permute() {
        let mut tr = WeightTransformer::new();
        tr.add_global(TransformOp::Permute(vec![1, 0]));
        let mut tensor = sample_tensor("w", vec![2, 3], vec![1.0; 6]);
        tr.apply(&mut tensor).unwrap();
        assert_eq!(tensor.meta.shape, vec![3, 2]);
    }

    #[test]
    fn transformer_apply_permute_wrong_dims() {
        let mut tr = WeightTransformer::new();
        tr.add_global(TransformOp::Permute(vec![0, 1, 2]));
        let mut tensor = sample_tensor("w", vec![2, 3], vec![1.0; 6]);
        assert!(tr.apply(&mut tensor).is_err());
    }

    #[test]
    fn transformer_chain_scale_then_cast() {
        let mut tr = WeightTransformer::new();
        tr.add_global(TransformOp::Scale(0.5));
        tr.add_global(TransformOp::Cast(DType::BF16));
        let mut tensor = sample_tensor("w", vec![2], vec![4.0, 6.0]);
        tr.apply(&mut tensor).unwrap();
        assert_eq!(tensor.data, vec![2.0, 3.0]);
        assert_eq!(tensor.meta.dtype, DType::BF16);
    }

    // -- WeightValidator tests --

    #[test]
    fn validator_default() {
        let v = WeightValidator::new();
        assert_eq!(v.num_checks(), 2);
    }

    #[test]
    fn validator_with_checks() {
        let v = WeightValidator::with_checks(vec![
            ValidationCheck::NoNaN,
            ValidationCheck::NoInf,
            ValidationCheck::NonZero,
            ValidationCheck::RangeBound,
        ]);
        assert_eq!(v.num_checks(), 4);
    }

    #[test]
    fn validator_range_bound() {
        let v = WeightValidator::new().with_range_bound(100.0);
        assert!((f64::from(v.range_bound()) - 100.0).abs() < f64::EPSILON);
    }

    #[test]
    fn validator_pass_clean_tensor() {
        let v = WeightValidator::new();
        let t = sample_tensor("w", vec![4], vec![0.1, 0.2, 0.3, 0.4]);
        assert!(v.validate(&t).passed);
    }

    #[test]
    fn validator_fail_nan() {
        let v = WeightValidator::new();
        let t = sample_tensor("w", vec![3], vec![1.0, f32::NAN, 3.0]);
        assert!(!v.validate(&t).passed);
    }

    #[test]
    fn validator_fail_inf() {
        let v = WeightValidator::new();
        let t = sample_tensor("w", vec![2], vec![f32::INFINITY, 1.0]);
        assert!(!v.validate(&t).passed);
    }

    #[test]
    fn validator_non_zero_pass() {
        let v = WeightValidator::with_checks(vec![ValidationCheck::NonZero]);
        let t = sample_tensor("w", vec![3], vec![0.0, 0.0, 1.0]);
        assert!(v.validate(&t).passed);
    }

    #[test]
    fn validator_non_zero_fail() {
        let v = WeightValidator::with_checks(vec![ValidationCheck::NonZero]);
        let t = sample_tensor("w", vec![3], vec![0.0, 0.0, 0.0]);
        assert!(!v.validate(&t).passed);
    }

    #[test]
    fn validator_range_bound_pass() {
        let v =
            WeightValidator::with_checks(vec![ValidationCheck::RangeBound]).with_range_bound(10.0);
        let t = sample_tensor("w", vec![3], vec![-9.0, 0.0, 9.0]);
        assert!(v.validate(&t).passed);
    }

    #[test]
    fn validator_range_bound_fail() {
        let v =
            WeightValidator::with_checks(vec![ValidationCheck::RangeBound]).with_range_bound(1.0);
        let t = sample_tensor("w", vec![2], vec![0.5, 2.0]);
        assert!(!v.validate(&t).passed);
    }

    #[test]
    fn validator_shape_match_pass() {
        let mut v = WeightValidator::with_checks(vec![ValidationCheck::ShapeMatch]);
        v.expect_shape("w", vec![2, 3]);
        let t = sample_tensor("w", vec![2, 3], vec![0.0; 6]);
        assert!(v.validate(&t).passed);
    }

    #[test]
    fn validator_shape_match_fail() {
        let mut v = WeightValidator::with_checks(vec![ValidationCheck::ShapeMatch]);
        v.expect_shape("w", vec![3, 3]);
        let t = sample_tensor("w", vec![2, 3], vec![0.0; 6]);
        assert!(!v.validate(&t).passed);
    }

    #[test]
    fn validator_shape_match_no_expected() {
        let v = WeightValidator::with_checks(vec![ValidationCheck::ShapeMatch]);
        let t = sample_tensor("w", vec![2, 3], vec![0.0; 6]);
        assert!(v.validate(&t).passed);
    }

    #[test]
    fn validator_validate_all() {
        let v = WeightValidator::new();
        let tensors = vec![
            sample_tensor("a", vec![2], vec![1.0, 2.0]),
            sample_tensor("b", vec![2], vec![3.0, f32::NAN]),
        ];
        let results = v.validate_all(&tensors);
        assert_eq!(results.len(), 2);
        assert!(results[0].passed);
        assert!(!results[1].passed);
    }

    #[test]
    fn validator_result_pass_constructor() {
        let r = ValidationResult::pass("test_tensor");
        assert!(r.passed);
        assert_eq!(r.name, "test_tensor");
        assert!(r.checks.is_empty());
    }

    #[test]
    fn validator_multiple_failures() {
        let v = WeightValidator::with_checks(vec![ValidationCheck::NoNaN, ValidationCheck::NoInf]);
        let t = sample_tensor("w", vec![2], vec![f32::NAN, f32::INFINITY]);
        let result = v.validate(&t);
        assert!(!result.passed);
        let failed_count = result.checks.iter().filter(|(_, ok, _)| !ok).count();
        assert_eq!(failed_count, 2);
    }

    // -- LoadProgress tests --

    #[test]
    fn progress_new() {
        let p = LoadProgress::new(10, 1000);
        assert_eq!(p.total_tensors, 10);
        assert_eq!(p.total_bytes, 1000);
        assert_eq!(p.loaded_tensors, 0);
        assert!(!p.is_complete());
    }

    #[test]
    fn progress_fraction_empty() {
        let p = LoadProgress::new(0, 0);
        assert!((p.fraction()).abs() < f64::EPSILON);
        assert!((p.byte_fraction()).abs() < f64::EPSILON);
    }

    #[test]
    fn progress_fraction_half() {
        let mut p = LoadProgress::new(4, 400);
        p.record_loaded("a", 100);
        p.record_loaded("b", 100);
        assert!((p.fraction() - 0.5).abs() < f64::EPSILON);
        assert!((p.byte_fraction() - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn progress_complete() {
        let mut p = LoadProgress::new(2, 200);
        p.record_loaded("a", 100);
        assert!(!p.is_complete());
        p.record_loaded("b", 100);
        assert!(p.is_complete());
    }

    #[test]
    fn progress_record_failed() {
        let mut p = LoadProgress::new(2, 200);
        p.record_loaded("a", 100);
        p.record_failed("b");
        assert!(p.is_complete());
        assert!(p.has_failures());
        assert_eq!(p.failure_count(), 1);
    }

    #[test]
    fn progress_percent_string() {
        let mut p = LoadProgress::new(4, 400);
        p.record_loaded("a", 100);
        assert_eq!(p.percent_string(), "25.0%");
    }

    #[test]
    fn progress_display() {
        let p = LoadProgress::new(10, 5000);
        let s = p.to_string();
        assert!(s.contains("[0/10]"));
        assert!(s.contains("0.0%"));
    }

    #[test]
    fn progress_serde_roundtrip() {
        let p = LoadProgress::new(5, 500);
        let json = serde_json::to_string(&p).unwrap();
        let back: LoadProgress = serde_json::from_str(&json).unwrap();
        assert_eq!(back.total_tensors, 5);
        assert_eq!(back.total_bytes, 500);
    }

    #[test]
    fn progress_current_tensor_updates() {
        let mut p = LoadProgress::new(3, 300);
        assert!(p.current_tensor.is_none());
        p.record_loaded("first", 100);
        assert_eq!(p.current_tensor.as_deref(), Some("first"));
        p.record_loaded("second", 100);
        assert_eq!(p.current_tensor.as_deref(), Some("second"));
        p.record_loaded("third", 100);
        assert!(p.current_tensor.is_none());
    }

    // -- WeightCache tests --

    #[test]
    fn cache_new() {
        let c = WeightCache::new(10);
        assert_eq!(c.capacity(), 10);
        assert!(c.is_empty());
        assert_eq!(c.len(), 0);
    }

    #[test]
    fn cache_insert_and_get() {
        let mut c = WeightCache::new(10);
        c.insert(sample_tensor("w", vec![2], vec![1.0, 2.0]));
        assert_eq!(c.len(), 1);
        assert!(c.contains("w"));
        let got = c.get("w").unwrap();
        assert_eq!(got.data, vec![1.0, 2.0]);
    }

    #[test]
    fn cache_miss() {
        let mut c = WeightCache::new(10);
        assert!(c.get("missing").is_none());
        assert_eq!(c.misses(), 1);
    }

    #[test]
    fn cache_hit_rate() {
        let mut c = WeightCache::new(10);
        c.insert(sample_tensor("w", vec![2], vec![1.0, 2.0]));
        c.get("w"); // hit
        c.get("x"); // miss
        assert!((c.hit_rate() - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn cache_eviction_lru() {
        let mut c = WeightCache::new(2);
        c.insert(sample_tensor("a", vec![1], vec![1.0]));
        c.insert(sample_tensor("b", vec![1], vec![2.0]));
        c.insert(sample_tensor("c", vec![1], vec![3.0]));
        assert!(!c.contains("a"));
        assert!(c.contains("b"));
        assert!(c.contains("c"));
        assert_eq!(c.len(), 2);
    }

    #[test]
    fn cache_explicit_evict() {
        let mut c = WeightCache::new(10);
        c.insert(sample_tensor("w", vec![2], vec![1.0, 2.0]));
        assert!(c.evict("w"));
        assert!(!c.contains("w"));
        assert!(c.is_empty());
    }

    #[test]
    fn cache_evict_nonexistent() {
        let mut c = WeightCache::new(10);
        assert!(!c.evict("missing"));
    }

    #[test]
    fn cache_clear() {
        let mut c = WeightCache::new(10);
        c.insert(sample_tensor("a", vec![1], vec![1.0]));
        c.insert(sample_tensor("b", vec![1], vec![2.0]));
        c.clear();
        assert!(c.is_empty());
    }

    #[test]
    fn cache_reset_stats() {
        let mut c = WeightCache::new(10);
        c.insert(sample_tensor("w", vec![1], vec![1.0]));
        c.get("w");
        c.get("x");
        assert!(c.hits() > 0);
        c.reset_stats();
        assert_eq!(c.hits(), 0);
        assert_eq!(c.misses(), 0);
    }

    #[test]
    fn cache_reinsert_updates() {
        let mut c = WeightCache::new(10);
        c.insert(sample_tensor("w", vec![1], vec![1.0]));
        c.insert(sample_tensor("w", vec![1], vec![99.0]));
        assert_eq!(c.len(), 1);
        let t = c.get("w").unwrap();
        assert_eq!(t.data, vec![99.0]);
    }

    #[test]
    fn cache_hit_rate_empty() {
        let c = WeightCache::new(10);
        assert!((c.hit_rate()).abs() < f64::EPSILON);
    }

    // -- WeightLoadingEngine tests --

    #[test]
    fn engine_new() {
        let wm = sample_weight_map(5);
        let engine = WeightLoadingEngine::new(LoadConfig::default(), wm);
        assert_eq!(engine.loaded_count(), 0);
        assert_eq!(engine.weight_map().len(), 5);
        assert!(!engine.progress().is_complete());
    }

    #[test]
    fn engine_load_single_tensor() {
        let wm = sample_weight_map(3);
        let mut engine = WeightLoadingEngine::new(LoadConfig::default(), wm);
        engine.load_tensor("layer.0.weight").unwrap();
        assert_eq!(engine.loaded_count(), 1);
        assert!(engine.is_loaded("layer.0.weight"));
    }

    #[test]
    fn engine_load_all() {
        let wm = sample_weight_map(4);
        let mut engine = WeightLoadingEngine::new(LoadConfig::default(), wm);
        let count = engine.load_all().unwrap();
        assert_eq!(count, 4);
        assert_eq!(engine.loaded_count(), 4);
        assert!(engine.progress().is_complete());
    }

    #[test]
    fn engine_load_tensor_not_found() {
        let wm = sample_weight_map(1);
        let mut engine = WeightLoadingEngine::new(LoadConfig::default(), wm);
        assert!(engine.load_tensor("nonexistent").is_err());
    }

    #[test]
    fn engine_load_tensor_cached() {
        let wm = sample_weight_map(1);
        let mut engine = WeightLoadingEngine::new(LoadConfig::default(), wm);
        engine.load_tensor("layer.0.weight").unwrap();
        let t = engine.load_tensor("layer.0.weight").unwrap();
        assert_eq!(t.num_elements(), 16);
    }

    #[test]
    fn engine_evict() {
        let wm = sample_weight_map(2);
        let mut engine = WeightLoadingEngine::new(LoadConfig::default(), wm);
        engine.load_tensor("layer.0.weight").unwrap();
        assert!(engine.evict("layer.0.weight"));
        assert!(!engine.is_loaded("layer.0.weight"));
    }

    #[test]
    fn engine_evict_nonexistent() {
        let wm = sample_weight_map(1);
        let mut engine = WeightLoadingEngine::new(LoadConfig::default(), wm);
        assert!(!engine.evict("missing"));
    }

    #[test]
    fn engine_progress_tracking() {
        let wm = sample_weight_map(3);
        let mut engine = WeightLoadingEngine::new(LoadConfig::default(), wm);
        assert_eq!(engine.progress().loaded_tensors, 0);
        engine.load_tensor("layer.0.weight").unwrap();
        assert_eq!(engine.progress().loaded_tensors, 1);
        engine.load_tensor("layer.1.weight").unwrap();
        assert_eq!(engine.progress().loaded_tensors, 2);
    }

    #[test]
    fn engine_with_transform() {
        let wm = sample_weight_map(1);
        let mut engine = WeightLoadingEngine::new(LoadConfig::default(), wm);
        engine.transformer_mut().add_global(TransformOp::Scale(1000.0));
        engine.load_tensor("layer.0.weight").unwrap();
        let t = engine.get("layer.0.weight").unwrap();
        assert!(t.data.iter().all(|&v| v.abs() < 200.0));
    }

    #[test]
    fn engine_with_validation_disabled() {
        let mut wm = WeightMap::new();
        wm.insert(sample_meta("nan_tensor", vec![2]));
        let cfg = LoadConfig::default().with_validation(false);
        let mut engine = WeightLoadingEngine::new(cfg, wm);
        engine.load_tensor("nan_tensor").unwrap();
        assert!(engine.is_loaded("nan_tensor"));
    }

    #[test]
    fn engine_loaded_bytes() {
        let wm = sample_weight_map(2);
        let mut engine = WeightLoadingEngine::new(LoadConfig::default(), wm);
        engine.load_tensor("layer.0.weight").unwrap();
        assert!(engine.loaded_bytes() > 0);
    }

    #[test]
    fn engine_get_returns_none_for_unloaded() {
        let wm = sample_weight_map(2);
        let engine = WeightLoadingEngine::new(LoadConfig::default(), wm);
        assert!(engine.get("layer.0.weight").is_none());
    }

    #[test]
    fn engine_config_accessor() {
        let cfg = LoadConfig::new(WeightFormat::SafeTensors, "cuda:0");
        let engine = WeightLoadingEngine::new(cfg, WeightMap::new());
        assert_eq!(engine.config().format, WeightFormat::SafeTensors);
        assert_eq!(engine.config().device, "cuda:0");
    }

    // -- Error display tests --

    #[test]
    fn error_display_io() {
        let err = WeightLoadError::Io(io::Error::new(io::ErrorKind::NotFound, "gone"));
        assert!(err.to_string().contains("I/O error"));
    }

    #[test]
    fn error_display_unsupported_format() {
        let err = WeightLoadError::UnsupportedFormat("xyz".into());
        assert!(err.to_string().contains("unsupported format"));
    }

    #[test]
    fn error_display_weight_not_found() {
        let err = WeightLoadError::WeightNotFound("layer.99".into());
        assert!(err.to_string().contains("weight not found"));
    }

    #[test]
    fn error_display_shape_mismatch() {
        let err = WeightLoadError::ShapeMismatch {
            name: "w".into(),
            expected: vec![2, 3],
            actual: vec![3, 2],
        };
        assert!(err.to_string().contains("shape mismatch"));
    }

    #[test]
    fn error_display_invalid_values() {
        let err = WeightLoadError::InvalidValues("has NaN".into());
        assert!(err.to_string().contains("invalid values"));
    }

    #[test]
    fn error_display_shard_error() {
        let err = WeightLoadError::ShardError("missing shard 2".into());
        assert!(err.to_string().contains("shard error"));
    }

    #[test]
    fn error_display_cast_error() {
        let err = WeightLoadError::CastError("u4->f32".into());
        assert!(err.to_string().contains("cast error"));
    }

    #[test]
    fn error_display_invalid_config() {
        let err = WeightLoadError::InvalidConfig("bad".into());
        assert!(err.to_string().contains("invalid config"));
    }

    #[test]
    fn error_from_io() {
        let io_err = io::Error::new(io::ErrorKind::PermissionDenied, "no");
        let err: WeightLoadError = io_err.into();
        assert!(matches!(err, WeightLoadError::Io(_)));
    }

    #[test]
    fn error_source_io() {
        use std::error::Error;
        let io_err = io::Error::other("test");
        let err = WeightLoadError::Io(io_err);
        assert!(err.source().is_some());
    }

    #[test]
    fn error_source_non_io() {
        use std::error::Error;
        let err = WeightLoadError::WeightNotFound("x".into());
        assert!(err.source().is_none());
    }
}
