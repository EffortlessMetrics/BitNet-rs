//! Module stub - implementation pending merge from feature branch
//! GGUF model file loader with lazy tensor loading and device placement.
//!
//! Provides a complete pipeline for loading GGUF model files:
//! validate header → read metadata → detect architecture → map weights → load tensors.

use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

// ── Constants ───────────────────────────────────────────────────────────────

/// GGUF magic number: "GGUF" in little-endian.
pub const GGUF_MAGIC: u32 = 0x4647_5547;

/// Minimum supported GGUF version.
pub const MIN_GGUF_VERSION: u32 = 2;

/// Maximum supported GGUF version.
pub const MAX_GGUF_VERSION: u32 = 3;

// ── Errors ──────────────────────────────────────────────────────────────────

/// Errors that can occur during GGUF loading.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GGUFError {
    /// Invalid magic number in header.
    InvalidMagic(u32),
    /// Unsupported GGUF version.
    UnsupportedVersion(u32),
    /// Metadata key not found.
    MetadataKeyNotFound(String),
    /// Metadata type mismatch.
    MetadataTypeMismatch { key: String, expected: String, got: String },
    /// Invalid tensor descriptor.
    InvalidTensorDescriptor(String),
    /// Architecture not recognized.
    UnknownArchitecture(String),
    /// Weight mapping failed.
    WeightMappingError(String),
    /// Tensor load failure.
    TensorLoadError(String),
    /// I/O or mmap error.
    IoError(String),
    /// Allocation failure.
    AllocationError(String),
}

impl fmt::Display for GGUFError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidMagic(m) => {
                write!(f, "invalid GGUF magic: 0x{m:08X}, expected 0x{GGUF_MAGIC:08X}")
            }
            Self::UnsupportedVersion(v) => {
                write!(
                    f,
                    "unsupported GGUF version {v} (supported: {MIN_GGUF_VERSION}-{MAX_GGUF_VERSION})"
                )
            }
            Self::MetadataKeyNotFound(k) => write!(f, "metadata key not found: {k}"),
            Self::MetadataTypeMismatch { key, expected, got } => {
                write!(f, "metadata type mismatch for '{key}': expected {expected}, got {got}")
            }
            Self::InvalidTensorDescriptor(msg) => write!(f, "invalid tensor descriptor: {msg}"),
            Self::UnknownArchitecture(a) => write!(f, "unknown architecture: {a}"),
            Self::WeightMappingError(msg) => write!(f, "weight mapping error: {msg}"),
            Self::TensorLoadError(msg) => write!(f, "tensor load error: {msg}"),
            Self::IoError(msg) => write!(f, "I/O error: {msg}"),
            Self::AllocationError(msg) => write!(f, "allocation error: {msg}"),
        }
    }
}

impl std::error::Error for GGUFError {}

pub type Result<T> = std::result::Result<T, GGUFError>;

// ── Enums ───────────────────────────────────────────────────────────────────

/// GGUF metadata value types.
#[derive(Debug, Clone, PartialEq)]
pub enum MetadataValue {
    /// Unsigned 8-bit integer.
    UInt8(u8),
    /// Signed 8-bit integer.
    Int8(i8),
    /// Unsigned 16-bit integer.
    UInt16(u16),
    /// Signed 16-bit integer.
    Int16(i16),
    /// Unsigned 32-bit integer.
    UInt32(u32),
    /// Signed 32-bit integer.
    Int32(i32),
    /// Unsigned 64-bit integer.
    UInt64(u64),
    /// Signed 64-bit integer.
    Int64(i64),
    /// 32-bit float.
    Float32(f32),
    /// 64-bit float.
    Float64(f64),
    /// Boolean value.
    Bool(bool),
    /// UTF-8 string.
    String(String),
    /// Array of metadata values.
    Array(Vec<Self>),
}

impl MetadataValue {
    /// Returns a human-readable type name.
    pub const fn type_name(&self) -> &'static str {
        match self {
            Self::UInt8(_) => "uint8",
            Self::Int8(_) => "int8",
            Self::UInt16(_) => "uint16",
            Self::Int16(_) => "int16",
            Self::UInt32(_) => "uint32",
            Self::Int32(_) => "int32",
            Self::UInt64(_) => "uint64",
            Self::Int64(_) => "int64",
            Self::Float32(_) => "float32",
            Self::Float64(_) => "float64",
            Self::Bool(_) => "bool",
            Self::String(_) => "string",
            Self::Array(_) => "array",
        }
    }

    /// Attempt to extract as string.
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(s) => Some(s),
            _ => None,
        }
    }

    /// Attempt to extract as u32.
    pub const fn as_u32(&self) -> Option<u32> {
        match self {
            Self::UInt32(v) => Some(*v),
            _ => None,
        }
    }

    /// Attempt to extract as u64.
    pub const fn as_u64(&self) -> Option<u64> {
        match self {
            Self::UInt64(v) => Some(*v),
            _ => None,
        }
    }

    /// Attempt to extract as i32.
    pub const fn as_i32(&self) -> Option<i32> {
        match self {
            Self::Int32(v) => Some(*v),
            _ => None,
        }
    }

    /// Attempt to extract as f32.
    pub const fn as_f32(&self) -> Option<f32> {
        match self {
            Self::Float32(v) => Some(*v),
            _ => None,
        }
    }

    /// Attempt to extract as bool.
    pub const fn as_bool(&self) -> Option<bool> {
        match self {
            Self::Bool(v) => Some(*v),
            _ => None,
        }
    }

    /// Attempt to extract as array.
    pub fn as_array(&self) -> Option<&[Self]> {
        match self {
            Self::Array(v) => Some(v),
            _ => None,
        }
    }
}

/// Quantization data types for tensors.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TensorDType {
    /// 1.58-bit ternary (`BitNet` `I2_S`).
    I2S,
    /// 4-bit quantization (GGML `Q4_0`).
    Q4_0,
    /// 8-bit quantization (GGML `Q8_0`).
    Q8_0,
    /// 16-bit float.
    F16,
    /// 32-bit float.
    F32,
}

impl TensorDType {
    /// Bytes per element (approximate for sub-byte types; rounded up).
    pub const fn bytes_per_element(self) -> f64 {
        match self {
            Self::I2S => 0.25, // 2 bits
            Self::Q4_0 => 0.5, // 4 bits
            Self::Q8_0 => 1.0, // 8 bits
            Self::F16 => 2.0,
            Self::F32 => 4.0,
        }
    }

    /// Returns the human-readable name.
    pub const fn name(self) -> &'static str {
        match self {
            Self::I2S => "I2_S",
            Self::Q4_0 => "Q4_0",
            Self::Q8_0 => "Q8_0",
            Self::F16 => "F16",
            Self::F32 => "F32",
        }
    }
}

impl fmt::Display for TensorDType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// Target device for tensor placement.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DeviceTarget {
    /// Host CPU memory.
    Cpu,
    /// GPU device memory (indexed).
    Gpu(u32),
}

impl fmt::Display for DeviceTarget {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cpu => write!(f, "CPU"),
            Self::Gpu(idx) => write!(f, "GPU:{idx}"),
        }
    }
}

/// Validation strictness levels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValidationLevel {
    /// Skip all validation.
    None,
    /// Validate header and basic structure only.
    Basic,
    /// Full validation including tensor checksums.
    Full,
}

/// Strategy for placing tensors across devices.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PlacementStrategy {
    /// All tensors on CPU.
    AllCpu,
    /// All tensors on specified GPU.
    AllGpu(u32),
    /// Automatically split across CPU and GPU based on memory.
    Auto,
    /// Custom per-layer placement.
    Custom(HashMap<String, DeviceTarget>),
}

/// Recognized model architectures.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ModelArchitecture {
    Llama,
    BitNet,
    Gpt2,
    Falcon,
    Mpt,
    Bloom,
    Phi,
    Gemma,
    Qwen,
    Mistral,
}

impl ModelArchitecture {
    /// Parse architecture string from GGUF metadata.
    pub fn from_str_value(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "llama" => Some(Self::Llama),
            "bitnet" => Some(Self::BitNet),
            "gpt2" => Some(Self::Gpt2),
            "falcon" => Some(Self::Falcon),
            "mpt" => Some(Self::Mpt),
            "bloom" => Some(Self::Bloom),
            "phi" | "phi2" | "phi3" => Some(Self::Phi),
            "gemma" => Some(Self::Gemma),
            "qwen" | "qwen2" => Some(Self::Qwen),
            "mistral" => Some(Self::Mistral),
            _ => None,
        }
    }

    /// Returns the architecture name.
    pub const fn name(self) -> &'static str {
        match self {
            Self::Llama => "llama",
            Self::BitNet => "bitnet",
            Self::Gpt2 => "gpt2",
            Self::Falcon => "falcon",
            Self::Mpt => "mpt",
            Self::Bloom => "bloom",
            Self::Phi => "phi",
            Self::Gemma => "gemma",
            Self::Qwen => "qwen",
            Self::Mistral => "mistral",
        }
    }
}

impl fmt::Display for ModelArchitecture {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

// ── Configuration ───────────────────────────────────────────────────────────

/// Configuration for the GGUF model loader.
#[derive(Debug, Clone)]
pub struct GGUFLoaderConfig {
    /// Whether to use memory-mapped I/O.
    pub mmap_enabled: bool,
    /// Validation strictness level.
    pub validation_level: ValidationLevel,
    /// Whether to load tensors lazily (on first access).
    pub lazy_loading: bool,
    /// Device placement strategy.
    pub placement_strategy: PlacementStrategy,
    /// Maximum memory budget for GPU tensors (bytes, 0 = unlimited).
    pub gpu_memory_budget: u64,
    /// Whether to prefetch tensor data during loading.
    pub prefetch: bool,
}

impl Default for GGUFLoaderConfig {
    fn default() -> Self {
        Self {
            mmap_enabled: true,
            validation_level: ValidationLevel::Basic,
            lazy_loading: true,
            placement_strategy: PlacementStrategy::AllCpu,
            gpu_memory_budget: 0,
            prefetch: false,
        }
    }
}

// ── Header ──────────────────────────────────────────────────────────────────

/// Parsed and validated GGUF file header.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GGUFHeader {
    /// Magic number (must be `GGUF_MAGIC`).
    pub magic: u32,
    /// Format version.
    pub version: u32,
    /// Number of tensors in the file.
    pub tensor_count: u64,
    /// Number of metadata key-value pairs.
    pub metadata_kv_count: u64,
}

impl GGUFHeader {
    /// Create a new header with validation.
    #[allow(clippy::manual_range_contains)]
    pub const fn new(
        magic: u32,
        version: u32,
        tensor_count: u64,
        metadata_kv_count: u64,
    ) -> Result<Self> {
        if magic != GGUF_MAGIC {
            return Err(GGUFError::InvalidMagic(magic));
        }
        if version < MIN_GGUF_VERSION || version > MAX_GGUF_VERSION {
            return Err(GGUFError::UnsupportedVersion(version));
        }
        Ok(Self { magic, version, tensor_count, metadata_kv_count })
    }

    /// Parse header from a raw byte slice (at least 24 bytes).
    pub fn from_bytes(data: &[u8]) -> Result<Self> {
        if data.len() < 24 {
            return Err(GGUFError::IoError("header too short: need at least 24 bytes".into()));
        }
        let magic = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
        let version = u32::from_le_bytes([data[4], data[5], data[6], data[7]]);
        let tensor_count = u64::from_le_bytes([
            data[8], data[9], data[10], data[11], data[12], data[13], data[14], data[15],
        ]);
        let metadata_kv_count = u64::from_le_bytes([
            data[16], data[17], data[18], data[19], data[20], data[21], data[22], data[23],
        ]);
        Self::new(magic, version, tensor_count, metadata_kv_count)
    }

    /// Serialize header to bytes (little-endian).
    pub fn to_bytes(&self) -> [u8; 24] {
        let mut buf = [0u8; 24];
        buf[0..4].copy_from_slice(&self.magic.to_le_bytes());
        buf[4..8].copy_from_slice(&self.version.to_le_bytes());
        buf[8..16].copy_from_slice(&self.tensor_count.to_le_bytes());
        buf[16..24].copy_from_slice(&self.metadata_kv_count.to_le_bytes());
        buf
    }
}

// ── Metadata Reader ─────────────────────────────────────────────────────────

/// Reads and stores GGUF key-value metadata.
#[derive(Debug, Clone, Default)]
pub struct GGUFMetadataReader {
    entries: HashMap<String, MetadataValue>,
}

impl GGUFMetadataReader {
    /// Create a new empty metadata reader.
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert a metadata key-value pair.
    pub fn insert(&mut self, key: impl Into<String>, value: MetadataValue) {
        self.entries.insert(key.into(), value);
    }

    /// Get a metadata value by key.
    pub fn get(&self, key: &str) -> Option<&MetadataValue> {
        self.entries.get(key)
    }

    /// Get a required metadata value by key.
    pub fn get_required(&self, key: &str) -> Result<&MetadataValue> {
        self.entries.get(key).ok_or_else(|| GGUFError::MetadataKeyNotFound(key.into()))
    }

    /// Get a string value by key.
    pub fn get_string(&self, key: &str) -> Result<&str> {
        let val = self.get_required(key)?;
        val.as_str().ok_or_else(|| GGUFError::MetadataTypeMismatch {
            key: key.into(),
            expected: "string".into(),
            got: val.type_name().into(),
        })
    }

    /// Get a u32 value by key.
    pub fn get_u32(&self, key: &str) -> Result<u32> {
        let val = self.get_required(key)?;
        val.as_u32().ok_or_else(|| GGUFError::MetadataTypeMismatch {
            key: key.into(),
            expected: "uint32".into(),
            got: val.type_name().into(),
        })
    }

    /// Get an f32 value by key.
    pub fn get_f32(&self, key: &str) -> Result<f32> {
        let val = self.get_required(key)?;
        val.as_f32().ok_or_else(|| GGUFError::MetadataTypeMismatch {
            key: key.into(),
            expected: "float32".into(),
            got: val.type_name().into(),
        })
    }

    /// Returns the number of metadata entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns `true` if there are no metadata entries.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Returns an iterator over all key-value pairs.
    pub fn iter(&self) -> impl Iterator<Item = (&String, &MetadataValue)> {
        self.entries.iter()
    }

    /// Returns all keys.
    pub fn keys(&self) -> impl Iterator<Item = &String> {
        self.entries.keys()
    }
}

// ── Tensor Descriptor ───────────────────────────────────────────────────────

/// Describes a tensor's location and layout within a GGUF file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorDescriptor {
    /// Tensor name (e.g., `blk.0.attn_q.weight`).
    pub name: String,
    /// Shape dimensions (e.g., `[4096, 4096]`).
    pub shape: Vec<u64>,
    /// Data type.
    pub dtype: TensorDType,
    /// Byte offset from the start of the tensor data section.
    pub offset: u64,
    /// Size in bytes.
    pub size: u64,
}

impl TensorDescriptor {
    /// Create a new tensor descriptor with validation.
    pub fn new(
        name: impl Into<String>,
        shape: Vec<u64>,
        dtype: TensorDType,
        offset: u64,
    ) -> Result<Self> {
        let name = name.into();
        if name.is_empty() {
            return Err(GGUFError::InvalidTensorDescriptor("tensor name cannot be empty".into()));
        }
        if shape.is_empty() {
            return Err(GGUFError::InvalidTensorDescriptor(format!(
                "tensor '{name}' has empty shape"
            )));
        }
        let num_elements: u64 = shape.iter().product();
        if num_elements == 0 {
            return Err(GGUFError::InvalidTensorDescriptor(format!(
                "tensor '{name}' has zero elements"
            )));
        }
        #[allow(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            clippy::cast_precision_loss
        )]
        let size = (num_elements as f64 * dtype.bytes_per_element()).ceil() as u64;
        Ok(Self { name, shape, dtype, offset, size })
    }

    /// Total number of elements.
    pub fn num_elements(&self) -> u64 {
        self.shape.iter().product()
    }

    /// Number of dimensions.
    pub const fn ndim(&self) -> usize {
        self.shape.len()
    }
}

impl fmt::Display for TensorDescriptor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}[{:?}, {}] @ offset {}", self.name, self.shape, self.dtype, self.offset)
    }
}

// ── Tensor Allocator ────────────────────────────────────────────────────────

/// Allocation record for a tensor placement.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorAllocation {
    /// The tensor descriptor.
    pub descriptor: TensorDescriptor,
    /// Target device.
    pub device: DeviceTarget,
    /// Whether the data has been loaded.
    pub loaded: bool,
}

/// Allocates tensors to target devices based on placement strategy.
#[derive(Debug)]
pub struct TensorAllocator {
    strategy: PlacementStrategy,
    gpu_memory_budget: u64,
    gpu_memory_used: u64,
    allocations: Vec<TensorAllocation>,
}

impl TensorAllocator {
    /// Create a new allocator with the given placement strategy.
    pub const fn new(strategy: PlacementStrategy, gpu_memory_budget: u64) -> Self {
        Self { strategy, gpu_memory_budget, gpu_memory_used: 0, allocations: Vec::new() }
    }

    /// Determine the device for a tensor and register the allocation.
    pub fn allocate(&mut self, descriptor: TensorDescriptor) -> Result<DeviceTarget> {
        let device = self.determine_device(&descriptor);
        if let DeviceTarget::Gpu(_) = device {
            if self.gpu_memory_budget > 0
                && self.gpu_memory_used + descriptor.size > self.gpu_memory_budget
            {
                // Fall back to CPU if GPU budget exhausted.
                let alloc =
                    TensorAllocation { descriptor, device: DeviceTarget::Cpu, loaded: false };
                self.allocations.push(alloc);
                return Ok(DeviceTarget::Cpu);
            }
            self.gpu_memory_used += descriptor.size;
        }
        let alloc = TensorAllocation { descriptor, device, loaded: false };
        self.allocations.push(alloc);
        Ok(device)
    }

    fn determine_device(&self, descriptor: &TensorDescriptor) -> DeviceTarget {
        match &self.strategy {
            PlacementStrategy::AllCpu => DeviceTarget::Cpu,
            PlacementStrategy::AllGpu(idx) => DeviceTarget::Gpu(*idx),
            PlacementStrategy::Auto => {
                // Heuristic: large tensors go to GPU, small ones stay on CPU.
                if descriptor.size >= 1024 * 1024 {
                    DeviceTarget::Gpu(0)
                } else {
                    DeviceTarget::Cpu
                }
            }
            PlacementStrategy::Custom(map) => {
                map.get(&descriptor.name).copied().unwrap_or(DeviceTarget::Cpu)
            }
        }
    }

    /// Returns all registered allocations.
    pub fn allocations(&self) -> &[TensorAllocation] {
        &self.allocations
    }

    /// Returns the number of allocations.
    pub const fn len(&self) -> usize {
        self.allocations.len()
    }

    /// Returns `true` if no allocations have been made.
    pub const fn is_empty(&self) -> bool {
        self.allocations.is_empty()
    }

    /// Returns GPU memory used in bytes.
    pub const fn gpu_memory_used(&self) -> u64 {
        self.gpu_memory_used
    }
}

// ── Architecture Detector ───────────────────────────────────────────────────

/// Detects model architecture from GGUF metadata.
pub struct ModelArchitectureDetector;

impl ModelArchitectureDetector {
    /// Detect architecture from metadata.
    pub fn detect(metadata: &GGUFMetadataReader) -> Result<ModelArchitecture> {
        // Standard key: "general.architecture"
        if let Ok(arch_str) = metadata.get_string("general.architecture") {
            if let Some(arch) = ModelArchitecture::from_str_value(arch_str) {
                return Ok(arch);
            }
            return Err(GGUFError::UnknownArchitecture(arch_str.to_string()));
        }

        // Fallback: scan for architecture-specific metadata prefixes.
        let known_prefixes = [
            ("llama.", ModelArchitecture::Llama),
            ("bitnet.", ModelArchitecture::BitNet),
            ("gpt2.", ModelArchitecture::Gpt2),
            ("falcon.", ModelArchitecture::Falcon),
            ("mpt.", ModelArchitecture::Mpt),
            ("bloom.", ModelArchitecture::Bloom),
            ("phi.", ModelArchitecture::Phi),
            ("gemma.", ModelArchitecture::Gemma),
            ("qwen.", ModelArchitecture::Qwen),
            ("mistral.", ModelArchitecture::Mistral),
        ];

        for key in metadata.keys() {
            for &(prefix, arch) in &known_prefixes {
                if key.starts_with(prefix) {
                    return Ok(arch);
                }
            }
        }

        Err(GGUFError::UnknownArchitecture("no architecture metadata found".into()))
    }
}

// ── Weight Mapper ───────────────────────────────────────────────────────────

/// A mapped weight associating a GGUF tensor name to a model layer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MappedWeight {
    /// Original GGUF tensor name.
    pub gguf_name: String,
    /// Layer index (None for non-layer tensors like embeddings).
    pub layer_index: Option<usize>,
    /// Component within the layer (e.g., `attn_q`, `ffn_up`).
    pub component: String,
}

/// Maps GGUF tensor names to model layer structure.
pub struct WeightMapper {
    architecture: ModelArchitecture,
}

impl WeightMapper {
    /// Create a mapper for the given architecture.
    pub const fn new(architecture: ModelArchitecture) -> Self {
        Self { architecture }
    }

    /// Returns the architecture this mapper targets.
    pub const fn architecture(&self) -> ModelArchitecture {
        self.architecture
    }

    /// Map a GGUF tensor name to a model component.
    pub fn map_tensor(&self, name: &str) -> Result<MappedWeight> {
        // Common patterns: "blk.{N}.{component}.weight", "token_embd.weight", "output_norm.weight"
        if let Some(mapped) = Self::try_map_block_tensor(name) {
            return Ok(mapped);
        }
        if let Some(mapped) = Self::try_map_global_tensor(name) {
            return Ok(mapped);
        }
        Err(GGUFError::WeightMappingError(format!(
            "cannot map tensor '{name}' for architecture '{}'",
            self.architecture
        )))
    }

    fn try_map_block_tensor(name: &str) -> Option<MappedWeight> {
        // Pattern: "blk.{N}.{component}.weight" or "blk.{N}.{component}"
        if !name.starts_with("blk.") {
            return None;
        }
        let rest = &name[4..]; // skip "blk."
        let dot_pos = rest.find('.')?;
        let layer_str = &rest[..dot_pos];
        let layer_index: usize = layer_str.parse().ok()?;
        let component_part = &rest[dot_pos + 1..];
        // Strip trailing ".weight" or ".bias" if present.
        let component = component_part
            .strip_suffix(".weight")
            .or_else(|| component_part.strip_suffix(".bias"))
            .unwrap_or(component_part);

        Some(MappedWeight {
            gguf_name: name.to_string(),
            layer_index: Some(layer_index),
            component: component.to_string(),
        })
    }

    fn try_map_global_tensor(name: &str) -> Option<MappedWeight> {
        let global_components = [
            "token_embd.weight",
            "token_embd",
            "output.weight",
            "output",
            "output_norm.weight",
            "output_norm",
        ];
        for &pattern in &global_components {
            if name == pattern {
                let component = pattern.strip_suffix(".weight").unwrap_or(pattern);
                return Some(MappedWeight {
                    gguf_name: name.to_string(),
                    layer_index: None,
                    component: component.to_string(),
                });
            }
        }
        None
    }
}

// ── Lazy Tensor Loader ──────────────────────────────────────────────────────

/// A lazy tensor handle that defers data loading until first access.
#[derive(Debug)]
pub struct LazyTensor {
    /// Tensor descriptor.
    pub descriptor: TensorDescriptor,
    /// Target device for this tensor.
    pub device: DeviceTarget,
    /// Whether the tensor data has been materialized.
    materialized: AtomicBool,
    /// Raw tensor data (populated on first access).
    data: std::sync::Mutex<Option<Vec<u8>>>,
}

impl LazyTensor {
    /// Create a new lazy tensor (not yet materialized).
    pub const fn new(descriptor: TensorDescriptor, device: DeviceTarget) -> Self {
        Self {
            descriptor,
            device,
            materialized: AtomicBool::new(false),
            data: std::sync::Mutex::new(None),
        }
    }

    /// Returns `true` if the tensor data has been loaded.
    pub fn is_materialized(&self) -> bool {
        self.materialized.load(Ordering::Acquire)
    }

    /// Materialize the tensor with the given data.
    pub fn materialize(&self, raw_data: Vec<u8>) -> Result<()> {
        if raw_data.len() as u64 != self.descriptor.size {
            return Err(GGUFError::TensorLoadError(format!(
                "data size mismatch for '{}': expected {} bytes, got {}",
                self.descriptor.name,
                self.descriptor.size,
                raw_data.len()
            )));
        }
        let mut guard = self.data.lock().map_err(|e| GGUFError::TensorLoadError(e.to_string()))?;
        *guard = Some(raw_data);
        self.materialized.store(true, Ordering::Release);
        drop(guard);
        Ok(())
    }

    /// Access the raw tensor data, returning `None` if not yet materialized.
    pub fn data(&self) -> Option<Vec<u8>> {
        let guard = self.data.lock().ok()?;
        guard.clone()
    }
}

/// Loads tensors on-demand from a GGUF data source.
pub struct LazyTensorLoader {
    tensors: HashMap<String, Arc<LazyTensor>>,
    /// Simulated backing store for tests / in-memory loading.
    backing_data: HashMap<String, Vec<u8>>,
}

impl LazyTensorLoader {
    /// Create a loader with pre-registered tensor descriptors.
    pub fn new(descriptors: Vec<(TensorDescriptor, DeviceTarget)>) -> Self {
        let mut tensors = HashMap::new();
        for (desc, device) in descriptors {
            let name = desc.name.clone();
            tensors.insert(name, Arc::new(LazyTensor::new(desc, device)));
        }
        Self { tensors, backing_data: HashMap::new() }
    }

    /// Register backing data for a tensor (used for testing / in-memory loading).
    pub fn register_backing_data(&mut self, name: &str, data: Vec<u8>) {
        self.backing_data.insert(name.to_string(), data);
    }

    /// Get a tensor handle (lazy, not necessarily materialized).
    pub fn get(&self, name: &str) -> Option<Arc<LazyTensor>> {
        self.tensors.get(name).cloned()
    }

    /// Load a tensor's data from the backing store.
    pub fn load_tensor(&self, name: &str) -> Result<Arc<LazyTensor>> {
        let tensor = self
            .tensors
            .get(name)
            .ok_or_else(|| GGUFError::TensorLoadError(format!("tensor '{name}' not found")))?;

        if tensor.is_materialized() {
            return Ok(tensor.clone());
        }

        let data = self
            .backing_data
            .get(name)
            .ok_or_else(|| {
                GGUFError::TensorLoadError(format!("no backing data for tensor '{name}'"))
            })?
            .clone();

        tensor.materialize(data)?;
        Ok(tensor.clone())
    }

    /// Returns the number of registered tensors.
    pub fn tensor_count(&self) -> usize {
        self.tensors.len()
    }

    /// Returns how many tensors are materialized.
    pub fn materialized_count(&self) -> usize {
        self.tensors.values().filter(|t| t.is_materialized()).count()
    }

    /// Returns all tensor names.
    pub fn tensor_names(&self) -> Vec<String> {
        self.tensors.keys().cloned().collect()
    }
}

// ── Load Progress ───────────────────────────────────────────────────────────

/// Reports loading progress.
#[derive(Debug, Clone)]
pub struct LoadProgress {
    /// Total number of tensors to load.
    pub total_tensors: u64,
    /// Number of tensors loaded so far.
    pub tensors_loaded: u64,
    /// Total bytes to load.
    pub total_bytes: u64,
    /// Bytes loaded so far.
    pub bytes_loaded: u64,
    /// Time when loading started.
    start_time: Instant,
}

impl LoadProgress {
    /// Create a new progress tracker.
    pub fn new(total_tensors: u64, total_bytes: u64) -> Self {
        Self {
            total_tensors,
            tensors_loaded: 0,
            total_bytes,
            bytes_loaded: 0,
            start_time: Instant::now(),
        }
    }

    /// Record progress for one tensor.
    pub const fn record_tensor(&mut self, bytes: u64) {
        self.tensors_loaded += 1;
        self.bytes_loaded += bytes;
    }

    /// Fraction complete (0.0 – 1.0).
    #[allow(clippy::cast_precision_loss)]
    pub fn fraction(&self) -> f64 {
        if self.total_bytes == 0 {
            return if self.total_tensors == 0 { 1.0 } else { 0.0 };
        }
        self.bytes_loaded as f64 / self.total_bytes as f64
    }

    /// Percentage complete (0 – 100).
    pub fn percent(&self) -> f64 {
        self.fraction() * 100.0
    }

    /// Elapsed time since loading started.
    pub fn elapsed(&self) -> Duration {
        self.start_time.elapsed()
    }

    /// Estimated time remaining (returns `None` if no progress yet).
    pub fn eta(&self) -> Option<Duration> {
        let frac = self.fraction();
        if frac <= 0.0 || frac >= 1.0 {
            return None;
        }
        let elapsed = self.elapsed().as_secs_f64();
        let total_est = elapsed / frac;
        let remaining = total_est - elapsed;
        Some(Duration::from_secs_f64(remaining))
    }

    /// Returns `true` if loading is complete.
    pub const fn is_complete(&self) -> bool {
        self.tensors_loaded >= self.total_tensors
    }
}

impl fmt::Display for LoadProgress {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "[{}/{}] {:.1}% ({}/{} bytes)",
            self.tensors_loaded,
            self.total_tensors,
            self.percent(),
            self.bytes_loaded,
            self.total_bytes,
        )
    }
}

// ── GGUF Model Loader (Orchestrator) ────────────────────────────────────────

/// Orchestrates the full GGUF loading pipeline.
///
/// Pipeline: validate header → read metadata → detect architecture →
/// map weights → allocate tensors → lazy-load on access.
pub struct GGUFModelLoader {
    config: GGUFLoaderConfig,
    header: Option<GGUFHeader>,
    metadata: GGUFMetadataReader,
    architecture: Option<ModelArchitecture>,
    weight_mapper: Option<WeightMapper>,
    allocator: Option<TensorAllocator>,
    loader: Option<LazyTensorLoader>,
    progress: Option<LoadProgress>,
}

impl GGUFModelLoader {
    /// Create a new loader with the given configuration.
    pub fn new(config: GGUFLoaderConfig) -> Self {
        Self {
            config,
            header: None,
            metadata: GGUFMetadataReader::new(),
            architecture: None,
            weight_mapper: None,
            allocator: None,
            loader: None,
            progress: None,
        }
    }

    /// Create a loader with default configuration.
    pub fn with_defaults() -> Self {
        Self::new(GGUFLoaderConfig::default())
    }

    /// Step 1: Validate and parse the header.
    pub fn validate_header(&mut self, data: &[u8]) -> Result<&GGUFHeader> {
        let header = GGUFHeader::from_bytes(data)?;
        log::info!(
            "GGUF v{} header: {} tensors, {} metadata entries",
            header.version,
            header.tensor_count,
            header.metadata_kv_count,
        );
        self.header = Some(header);
        Ok(self.header.as_ref().unwrap())
    }

    /// Step 2: Provide pre-parsed metadata.
    pub fn set_metadata(&mut self, metadata: GGUFMetadataReader) {
        self.metadata = metadata;
    }

    /// Step 3: Detect model architecture.
    pub fn detect_architecture(&mut self) -> Result<ModelArchitecture> {
        let arch = ModelArchitectureDetector::detect(&self.metadata)?;
        log::info!("detected architecture: {arch}");
        self.architecture = Some(arch);
        self.weight_mapper = Some(WeightMapper::new(arch));
        Ok(arch)
    }

    /// Step 4: Map and allocate tensors.
    pub fn prepare_tensors(
        &mut self,
        descriptors: Vec<TensorDescriptor>,
    ) -> Result<Vec<TensorAllocation>> {
        let mut allocator = TensorAllocator::new(
            self.config.placement_strategy.clone(),
            self.config.gpu_memory_budget,
        );

        let total_bytes: u64 = descriptors.iter().map(|d| d.size).sum();
        let total_tensors = descriptors.len() as u64;

        let mut lazy_pairs = Vec::new();

        for desc in descriptors {
            let device = allocator.allocate(desc.clone())?;
            lazy_pairs.push((desc, device));
        }

        let allocations: Vec<TensorAllocation> = allocator.allocations().to_vec();
        self.allocator = Some(allocator);

        if self.config.lazy_loading {
            self.loader = Some(LazyTensorLoader::new(lazy_pairs));
        }

        self.progress = Some(LoadProgress::new(total_tensors, total_bytes));

        Ok(allocations)
    }

    /// Returns a reference to the current header (if validated).
    pub const fn header(&self) -> Option<&GGUFHeader> {
        self.header.as_ref()
    }

    /// Returns the detected architecture (if any).
    pub const fn architecture(&self) -> Option<ModelArchitecture> {
        self.architecture
    }

    /// Returns a reference to the metadata reader.
    pub const fn metadata(&self) -> &GGUFMetadataReader {
        &self.metadata
    }

    /// Returns a reference to the tensor loader (if initialized).
    pub const fn tensor_loader(&self) -> Option<&LazyTensorLoader> {
        self.loader.as_ref()
    }

    /// Returns a reference to the progress tracker (if initialized).
    pub const fn progress(&self) -> Option<&LoadProgress> {
        self.progress.as_ref()
    }

    /// Returns the configuration.
    pub const fn config(&self) -> &GGUFLoaderConfig {
        &self.config
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Header Tests ────────────────────────────────────────────────────

    #[test]
    fn test_header_valid_magic() {
        let h = GGUFHeader::new(GGUF_MAGIC, 3, 10, 5).unwrap();
        assert_eq!(h.magic, GGUF_MAGIC);
        assert_eq!(h.version, 3);
        assert_eq!(h.tensor_count, 10);
        assert_eq!(h.metadata_kv_count, 5);
    }

    #[test]
    fn test_header_invalid_magic_zero() {
        let err = GGUFHeader::new(0, 3, 10, 5).unwrap_err();
        assert_eq!(err, GGUFError::InvalidMagic(0));
    }

    #[test]
    fn test_header_invalid_magic_random() {
        let err = GGUFHeader::new(0xDEAD_BEEF, 3, 1, 1).unwrap_err();
        assert_eq!(err, GGUFError::InvalidMagic(0xDEAD_BEEF));
    }

    #[test]
    fn test_header_invalid_magic_reversed() {
        // "FUGG" instead of "GGUF"
        let err = GGUFHeader::new(0x4755_4646, 3, 1, 1).unwrap_err();
        assert!(matches!(err, GGUFError::InvalidMagic(_)));
    }

    #[test]
    fn test_header_invalid_magic_off_by_one() {
        let err = GGUFHeader::new(GGUF_MAGIC + 1, 2, 1, 1).unwrap_err();
        assert!(matches!(err, GGUFError::InvalidMagic(_)));
    }

    #[test]
    fn test_header_version_2() {
        let h = GGUFHeader::new(GGUF_MAGIC, 2, 0, 0).unwrap();
        assert_eq!(h.version, 2);
    }

    #[test]
    fn test_header_version_3() {
        let h = GGUFHeader::new(GGUF_MAGIC, 3, 0, 0).unwrap();
        assert_eq!(h.version, 3);
    }

    #[test]
    fn test_header_version_1_unsupported() {
        let err = GGUFHeader::new(GGUF_MAGIC, 1, 0, 0).unwrap_err();
        assert_eq!(err, GGUFError::UnsupportedVersion(1));
    }

    #[test]
    fn test_header_version_0_unsupported() {
        let err = GGUFHeader::new(GGUF_MAGIC, 0, 0, 0).unwrap_err();
        assert_eq!(err, GGUFError::UnsupportedVersion(0));
    }

    #[test]
    fn test_header_version_4_unsupported() {
        let err = GGUFHeader::new(GGUF_MAGIC, 4, 0, 0).unwrap_err();
        assert_eq!(err, GGUFError::UnsupportedVersion(4));
    }

    #[test]
    fn test_header_version_255_unsupported() {
        let err = GGUFHeader::new(GGUF_MAGIC, 255, 0, 0).unwrap_err();
        assert_eq!(err, GGUFError::UnsupportedVersion(255));
    }

    #[test]
    fn test_header_from_bytes_roundtrip() {
        let h = GGUFHeader::new(GGUF_MAGIC, 3, 42, 7).unwrap();
        let bytes = h.to_bytes();
        let h2 = GGUFHeader::from_bytes(&bytes).unwrap();
        assert_eq!(h, h2);
    }

    #[test]
    fn test_header_from_bytes_too_short() {
        let err = GGUFHeader::from_bytes(&[0u8; 20]).unwrap_err();
        assert!(matches!(err, GGUFError::IoError(_)));
    }

    #[test]
    fn test_header_from_bytes_exact_24() {
        let h = GGUFHeader::new(GGUF_MAGIC, 2, 1, 1).unwrap();
        let bytes = h.to_bytes();
        assert_eq!(bytes.len(), 24);
        let h2 = GGUFHeader::from_bytes(&bytes).unwrap();
        assert_eq!(h, h2);
    }

    #[test]
    fn test_header_from_bytes_extra_data() {
        let h = GGUFHeader::new(GGUF_MAGIC, 3, 100, 50).unwrap();
        let mut bytes = h.to_bytes().to_vec();
        bytes.extend_from_slice(&[0xFF; 100]);
        let h2 = GGUFHeader::from_bytes(&bytes).unwrap();
        assert_eq!(h, h2);
    }

    #[test]
    fn test_header_zero_tensors_and_metadata() {
        let h = GGUFHeader::new(GGUF_MAGIC, 3, 0, 0).unwrap();
        assert_eq!(h.tensor_count, 0);
        assert_eq!(h.metadata_kv_count, 0);
    }

    #[test]
    fn test_header_large_tensor_count() {
        let h = GGUFHeader::new(GGUF_MAGIC, 3, u64::MAX, 0).unwrap();
        assert_eq!(h.tensor_count, u64::MAX);
    }

    #[test]
    fn test_header_large_metadata_count() {
        let h = GGUFHeader::new(GGUF_MAGIC, 3, 0, u64::MAX).unwrap();
        assert_eq!(h.metadata_kv_count, u64::MAX);
    }

    // ── MetadataValue Type Tests ────────────────────────────────────────

    #[test]
    fn test_metadata_value_uint8() {
        let v = MetadataValue::UInt8(42);
        assert_eq!(v.type_name(), "uint8");
        assert!(v.as_str().is_none());
    }

    #[test]
    fn test_metadata_value_int8() {
        let v = MetadataValue::Int8(-1);
        assert_eq!(v.type_name(), "int8");
    }

    #[test]
    fn test_metadata_value_uint16() {
        let v = MetadataValue::UInt16(1000);
        assert_eq!(v.type_name(), "uint16");
    }

    #[test]
    fn test_metadata_value_int16() {
        let v = MetadataValue::Int16(-500);
        assert_eq!(v.type_name(), "int16");
    }

    #[test]
    fn test_metadata_value_uint32() {
        let v = MetadataValue::UInt32(100_000);
        assert_eq!(v.type_name(), "uint32");
        assert_eq!(v.as_u32(), Some(100_000));
    }

    #[test]
    fn test_metadata_value_int32() {
        let v = MetadataValue::Int32(-42);
        assert_eq!(v.type_name(), "int32");
        assert_eq!(v.as_i32(), Some(-42));
    }

    #[test]
    fn test_metadata_value_uint64() {
        let v = MetadataValue::UInt64(u64::MAX);
        assert_eq!(v.type_name(), "uint64");
        assert_eq!(v.as_u64(), Some(u64::MAX));
    }

    #[test]
    fn test_metadata_value_int64() {
        let v = MetadataValue::Int64(i64::MIN);
        assert_eq!(v.type_name(), "int64");
    }

    #[test]
    fn test_metadata_value_float32() {
        let v = MetadataValue::Float32(3.14);
        assert_eq!(v.type_name(), "float32");
        assert!((v.as_f32().unwrap() - 3.14).abs() < 1e-5);
    }

    #[test]
    fn test_metadata_value_float64() {
        let v = MetadataValue::Float64(2.718281828);
        assert_eq!(v.type_name(), "float64");
    }

    #[test]
    fn test_metadata_value_bool_true() {
        let v = MetadataValue::Bool(true);
        assert_eq!(v.type_name(), "bool");
        assert_eq!(v.as_bool(), Some(true));
    }

    #[test]
    fn test_metadata_value_bool_false() {
        let v = MetadataValue::Bool(false);
        assert_eq!(v.as_bool(), Some(false));
    }

    #[test]
    fn test_metadata_value_string() {
        let v = MetadataValue::String("hello".into());
        assert_eq!(v.type_name(), "string");
        assert_eq!(v.as_str(), Some("hello"));
    }

    #[test]
    fn test_metadata_value_string_empty() {
        let v = MetadataValue::String(String::new());
        assert_eq!(v.as_str(), Some(""));
    }

    #[test]
    fn test_metadata_value_array_empty() {
        let v = MetadataValue::Array(vec![]);
        assert_eq!(v.type_name(), "array");
        assert_eq!(v.as_array().unwrap().len(), 0);
    }

    #[test]
    fn test_metadata_value_array_mixed() {
        let v = MetadataValue::Array(vec![
            MetadataValue::UInt32(1),
            MetadataValue::String("two".into()),
            MetadataValue::Float32(3.0),
        ]);
        assert_eq!(v.as_array().unwrap().len(), 3);
    }

    #[test]
    fn test_metadata_value_nested_array() {
        let inner = MetadataValue::Array(vec![MetadataValue::Int32(99)]);
        let outer = MetadataValue::Array(vec![inner]);
        let arr = outer.as_array().unwrap();
        assert_eq!(arr.len(), 1);
        assert!(arr[0].as_array().is_some());
    }

    #[test]
    fn test_metadata_value_type_mismatch_accessors() {
        let v = MetadataValue::UInt32(5);
        assert!(v.as_str().is_none());
        assert!(v.as_f32().is_none());
        assert!(v.as_bool().is_none());
        assert!(v.as_array().is_none());
        assert!(v.as_u64().is_none());
    }

    // ── Metadata Reader Tests ───────────────────────────────────────────

    #[test]
    fn test_metadata_reader_empty() {
        let reader = GGUFMetadataReader::new();
        assert!(reader.is_empty());
        assert_eq!(reader.len(), 0);
    }

    #[test]
    fn test_metadata_reader_insert_and_get() {
        let mut reader = GGUFMetadataReader::new();
        reader.insert("general.architecture", MetadataValue::String("llama".into()));
        assert_eq!(reader.len(), 1);
        assert!(!reader.is_empty());
        assert_eq!(reader.get("general.architecture").unwrap().as_str(), Some("llama"));
    }

    #[test]
    fn test_metadata_reader_get_required_missing() {
        let reader = GGUFMetadataReader::new();
        let err = reader.get_required("missing").unwrap_err();
        assert_eq!(err, GGUFError::MetadataKeyNotFound("missing".into()));
    }

    #[test]
    fn test_metadata_reader_get_string_type_mismatch() {
        let mut reader = GGUFMetadataReader::new();
        reader.insert("key", MetadataValue::UInt32(42));
        let err = reader.get_string("key").unwrap_err();
        assert!(matches!(err, GGUFError::MetadataTypeMismatch { .. }));
    }

    #[test]
    fn test_metadata_reader_get_u32() {
        let mut reader = GGUFMetadataReader::new();
        reader.insert("llama.block_count", MetadataValue::UInt32(32));
        assert_eq!(reader.get_u32("llama.block_count").unwrap(), 32);
    }

    #[test]
    fn test_metadata_reader_get_u32_type_mismatch() {
        let mut reader = GGUFMetadataReader::new();
        reader.insert("key", MetadataValue::String("oops".into()));
        let err = reader.get_u32("key").unwrap_err();
        assert!(matches!(err, GGUFError::MetadataTypeMismatch { .. }));
    }

    #[test]
    fn test_metadata_reader_get_f32() {
        let mut reader = GGUFMetadataReader::new();
        reader.insert("rope.freq_base", MetadataValue::Float32(10000.0));
        assert!((reader.get_f32("rope.freq_base").unwrap() - 10000.0).abs() < 1e-5);
    }

    #[test]
    fn test_metadata_reader_overwrite() {
        let mut reader = GGUFMetadataReader::new();
        reader.insert("key", MetadataValue::UInt32(1));
        reader.insert("key", MetadataValue::UInt32(2));
        assert_eq!(reader.get_u32("key").unwrap(), 2);
        assert_eq!(reader.len(), 1);
    }

    #[test]
    fn test_metadata_reader_multiple_entries() {
        let mut reader = GGUFMetadataReader::new();
        reader.insert("a", MetadataValue::UInt32(1));
        reader.insert("b", MetadataValue::String("hello".into()));
        reader.insert("c", MetadataValue::Float32(3.14));
        assert_eq!(reader.len(), 3);
    }

    #[test]
    fn test_metadata_reader_keys() {
        let mut reader = GGUFMetadataReader::new();
        reader.insert("alpha", MetadataValue::Bool(true));
        reader.insert("beta", MetadataValue::Bool(false));
        let mut keys: Vec<_> = reader.keys().cloned().collect();
        keys.sort();
        assert_eq!(keys, vec!["alpha", "beta"]);
    }

    #[test]
    fn test_metadata_reader_iter() {
        let mut reader = GGUFMetadataReader::new();
        reader.insert("x", MetadataValue::Int32(10));
        let items: Vec<_> = reader.iter().collect();
        assert_eq!(items.len(), 1);
    }

    // ── TensorDType Tests ───────────────────────────────────────────────

    #[test]
    fn test_tensor_dtype_i2s_bytes() {
        assert!((TensorDType::I2S.bytes_per_element() - 0.25).abs() < 1e-10);
    }

    #[test]
    fn test_tensor_dtype_q4_0_bytes() {
        assert!((TensorDType::Q4_0.bytes_per_element() - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_tensor_dtype_q8_0_bytes() {
        assert!((TensorDType::Q8_0.bytes_per_element() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_tensor_dtype_f16_bytes() {
        assert!((TensorDType::F16.bytes_per_element() - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_tensor_dtype_f32_bytes() {
        assert!((TensorDType::F32.bytes_per_element() - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_tensor_dtype_names() {
        assert_eq!(TensorDType::I2S.name(), "I2_S");
        assert_eq!(TensorDType::Q4_0.name(), "Q4_0");
        assert_eq!(TensorDType::Q8_0.name(), "Q8_0");
        assert_eq!(TensorDType::F16.name(), "F16");
        assert_eq!(TensorDType::F32.name(), "F32");
    }

    #[test]
    fn test_tensor_dtype_display() {
        assert_eq!(format!("{}", TensorDType::I2S), "I2_S");
        assert_eq!(format!("{}", TensorDType::F32), "F32");
    }

    // ── TensorDescriptor Tests ──────────────────────────────────────────

    #[test]
    fn test_tensor_descriptor_valid() {
        let d = TensorDescriptor::new("blk.0.attn_q.weight", vec![4096, 4096], TensorDType::F32, 0)
            .unwrap();
        assert_eq!(d.name, "blk.0.attn_q.weight");
        assert_eq!(d.shape, vec![4096, 4096]);
        assert_eq!(d.dtype, TensorDType::F32);
        assert_eq!(d.offset, 0);
        assert_eq!(d.num_elements(), 4096 * 4096);
        assert_eq!(d.ndim(), 2);
        assert_eq!(d.size, 4096 * 4096 * 4); // F32 = 4 bytes
    }

    #[test]
    fn test_tensor_descriptor_empty_name() {
        let err = TensorDescriptor::new("", vec![1], TensorDType::F32, 0).unwrap_err();
        assert!(matches!(err, GGUFError::InvalidTensorDescriptor(_)));
    }

    #[test]
    fn test_tensor_descriptor_empty_shape() {
        let err = TensorDescriptor::new("test", vec![], TensorDType::F32, 0).unwrap_err();
        assert!(matches!(err, GGUFError::InvalidTensorDescriptor(_)));
    }

    #[test]
    fn test_tensor_descriptor_zero_dim() {
        let err = TensorDescriptor::new("test", vec![0, 4096], TensorDType::F32, 0).unwrap_err();
        assert!(matches!(err, GGUFError::InvalidTensorDescriptor(_)));
    }

    #[test]
    fn test_tensor_descriptor_1d() {
        let d = TensorDescriptor::new("bias", vec![4096], TensorDType::F32, 100).unwrap();
        assert_eq!(d.ndim(), 1);
        assert_eq!(d.num_elements(), 4096);
        assert_eq!(d.offset, 100);
    }

    #[test]
    fn test_tensor_descriptor_3d() {
        let d = TensorDescriptor::new("tensor3d", vec![2, 3, 4], TensorDType::F16, 0).unwrap();
        assert_eq!(d.ndim(), 3);
        assert_eq!(d.num_elements(), 24);
        assert_eq!(d.size, 48); // 24 * 2 bytes
    }

    #[test]
    fn test_tensor_descriptor_i2s_size() {
        let d = TensorDescriptor::new("ternary", vec![256], TensorDType::I2S, 0).unwrap();
        assert_eq!(d.num_elements(), 256);
        assert_eq!(d.size, 64); // 256 * 0.25 = 64
    }

    #[test]
    fn test_tensor_descriptor_q4_0_size() {
        let d = TensorDescriptor::new("q4", vec![1024], TensorDType::Q4_0, 0).unwrap();
        assert_eq!(d.size, 512); // 1024 * 0.5
    }

    #[test]
    fn test_tensor_descriptor_display() {
        let d = TensorDescriptor::new("test", vec![10, 20], TensorDType::F32, 0).unwrap();
        let s = format!("{d}");
        assert!(s.contains("test"));
        assert!(s.contains("F32"));
    }

    #[test]
    fn test_tensor_descriptor_single_element() {
        let d = TensorDescriptor::new("scalar", vec![1], TensorDType::F32, 0).unwrap();
        assert_eq!(d.num_elements(), 1);
        assert_eq!(d.size, 4);
    }

    // ── TensorAllocator Tests ───────────────────────────────────────────

    #[test]
    fn test_allocator_all_cpu() {
        let mut alloc = TensorAllocator::new(PlacementStrategy::AllCpu, 0);
        let d = TensorDescriptor::new("t1", vec![100], TensorDType::F32, 0).unwrap();
        let device = alloc.allocate(d).unwrap();
        assert_eq!(device, DeviceTarget::Cpu);
        assert_eq!(alloc.len(), 1);
    }

    #[test]
    fn test_allocator_all_gpu() {
        let mut alloc = TensorAllocator::new(PlacementStrategy::AllGpu(0), 0);
        let d = TensorDescriptor::new("t1", vec![100], TensorDType::F32, 0).unwrap();
        let device = alloc.allocate(d).unwrap();
        assert_eq!(device, DeviceTarget::Gpu(0));
    }

    #[test]
    fn test_allocator_all_gpu_specific_device() {
        let mut alloc = TensorAllocator::new(PlacementStrategy::AllGpu(2), 0);
        let d = TensorDescriptor::new("t1", vec![100], TensorDType::F32, 0).unwrap();
        let device = alloc.allocate(d).unwrap();
        assert_eq!(device, DeviceTarget::Gpu(2));
    }

    #[test]
    fn test_allocator_auto_small_tensor_cpu() {
        let mut alloc = TensorAllocator::new(PlacementStrategy::Auto, 0);
        let d = TensorDescriptor::new("small", vec![100], TensorDType::F32, 0).unwrap();
        let device = alloc.allocate(d).unwrap();
        assert_eq!(device, DeviceTarget::Cpu);
    }

    #[test]
    fn test_allocator_auto_large_tensor_gpu() {
        let mut alloc = TensorAllocator::new(PlacementStrategy::Auto, 0);
        // 1M elements * 4 bytes = 4MB > 1MB threshold
        let d = TensorDescriptor::new("large", vec![1_000_000], TensorDType::F32, 0).unwrap();
        let device = alloc.allocate(d).unwrap();
        assert_eq!(device, DeviceTarget::Gpu(0));
    }

    #[test]
    fn test_allocator_custom_placement() {
        let mut custom = HashMap::new();
        custom.insert("on_gpu".to_string(), DeviceTarget::Gpu(1));
        let mut alloc = TensorAllocator::new(PlacementStrategy::Custom(custom), 0);

        let d1 = TensorDescriptor::new("on_gpu", vec![100], TensorDType::F32, 0).unwrap();
        let d2 = TensorDescriptor::new("on_cpu", vec![100], TensorDType::F32, 0).unwrap();

        assert_eq!(alloc.allocate(d1).unwrap(), DeviceTarget::Gpu(1));
        assert_eq!(alloc.allocate(d2).unwrap(), DeviceTarget::Cpu); // default fallback
    }

    #[test]
    fn test_allocator_gpu_memory_budget_fallback() {
        let mut alloc = TensorAllocator::new(PlacementStrategy::AllGpu(0), 500);
        let d1 = TensorDescriptor::new("fits", vec![100], TensorDType::F32, 0).unwrap(); // 400 bytes
        let d2 = TensorDescriptor::new("exceeds", vec![100], TensorDType::F32, 0).unwrap(); // 400 more

        let dev1 = alloc.allocate(d1).unwrap();
        assert_eq!(dev1, DeviceTarget::Gpu(0));
        assert_eq!(alloc.gpu_memory_used(), 400);

        let dev2 = alloc.allocate(d2).unwrap();
        assert_eq!(dev2, DeviceTarget::Cpu); // fell back
    }

    #[test]
    fn test_allocator_empty() {
        let alloc = TensorAllocator::new(PlacementStrategy::AllCpu, 0);
        assert!(alloc.is_empty());
        assert_eq!(alloc.len(), 0);
        assert_eq!(alloc.gpu_memory_used(), 0);
    }

    #[test]
    fn test_allocator_multiple_tensors() {
        let mut alloc = TensorAllocator::new(PlacementStrategy::AllCpu, 0);
        for i in 0..10 {
            let d = TensorDescriptor::new(format!("t{i}"), vec![100], TensorDType::F32, 0).unwrap();
            alloc.allocate(d).unwrap();
        }
        assert_eq!(alloc.len(), 10);
    }

    // ── Architecture Detection Tests ────────────────────────────────────

    #[test]
    fn test_detect_arch_llama() {
        let mut meta = GGUFMetadataReader::new();
        meta.insert("general.architecture", MetadataValue::String("llama".into()));
        let arch = ModelArchitectureDetector::detect(&meta).unwrap();
        assert_eq!(arch, ModelArchitecture::Llama);
    }

    #[test]
    fn test_detect_arch_bitnet() {
        let mut meta = GGUFMetadataReader::new();
        meta.insert("general.architecture", MetadataValue::String("bitnet".into()));
        assert_eq!(ModelArchitectureDetector::detect(&meta).unwrap(), ModelArchitecture::BitNet);
    }

    #[test]
    fn test_detect_arch_gpt2() {
        let mut meta = GGUFMetadataReader::new();
        meta.insert("general.architecture", MetadataValue::String("gpt2".into()));
        assert_eq!(ModelArchitectureDetector::detect(&meta).unwrap(), ModelArchitecture::Gpt2);
    }

    #[test]
    fn test_detect_arch_falcon() {
        let mut meta = GGUFMetadataReader::new();
        meta.insert("general.architecture", MetadataValue::String("falcon".into()));
        assert_eq!(ModelArchitectureDetector::detect(&meta).unwrap(), ModelArchitecture::Falcon);
    }

    #[test]
    fn test_detect_arch_phi_variants() {
        for name in ["phi", "phi2", "phi3"] {
            let mut meta = GGUFMetadataReader::new();
            meta.insert("general.architecture", MetadataValue::String(name.into()));
            assert_eq!(ModelArchitectureDetector::detect(&meta).unwrap(), ModelArchitecture::Phi);
        }
    }

    #[test]
    fn test_detect_arch_qwen_variants() {
        for name in ["qwen", "qwen2"] {
            let mut meta = GGUFMetadataReader::new();
            meta.insert("general.architecture", MetadataValue::String(name.into()));
            assert_eq!(ModelArchitectureDetector::detect(&meta).unwrap(), ModelArchitecture::Qwen);
        }
    }

    #[test]
    fn test_detect_arch_case_insensitive() {
        let mut meta = GGUFMetadataReader::new();
        meta.insert("general.architecture", MetadataValue::String("LLAMA".into()));
        assert_eq!(ModelArchitectureDetector::detect(&meta).unwrap(), ModelArchitecture::Llama);
    }

    #[test]
    fn test_detect_arch_unknown() {
        let mut meta = GGUFMetadataReader::new();
        meta.insert("general.architecture", MetadataValue::String("transformer_xl".into()));
        let err = ModelArchitectureDetector::detect(&meta).unwrap_err();
        assert!(matches!(err, GGUFError::UnknownArchitecture(_)));
    }

    #[test]
    fn test_detect_arch_fallback_by_prefix() {
        let mut meta = GGUFMetadataReader::new();
        meta.insert("llama.block_count", MetadataValue::UInt32(32));
        assert_eq!(ModelArchitectureDetector::detect(&meta).unwrap(), ModelArchitecture::Llama);
    }

    #[test]
    fn test_detect_arch_no_metadata() {
        let meta = GGUFMetadataReader::new();
        let err = ModelArchitectureDetector::detect(&meta).unwrap_err();
        assert!(matches!(err, GGUFError::UnknownArchitecture(_)));
    }

    #[test]
    fn test_detect_arch_gemma() {
        let mut meta = GGUFMetadataReader::new();
        meta.insert("general.architecture", MetadataValue::String("gemma".into()));
        assert_eq!(ModelArchitectureDetector::detect(&meta).unwrap(), ModelArchitecture::Gemma);
    }

    #[test]
    fn test_detect_arch_mistral() {
        let mut meta = GGUFMetadataReader::new();
        meta.insert("general.architecture", MetadataValue::String("mistral".into()));
        assert_eq!(ModelArchitectureDetector::detect(&meta).unwrap(), ModelArchitecture::Mistral);
    }

    #[test]
    fn test_architecture_names() {
        assert_eq!(ModelArchitecture::Llama.name(), "llama");
        assert_eq!(ModelArchitecture::BitNet.name(), "bitnet");
        assert_eq!(ModelArchitecture::Gpt2.name(), "gpt2");
        assert_eq!(ModelArchitecture::Bloom.name(), "bloom");
        assert_eq!(ModelArchitecture::Mpt.name(), "mpt");
    }

    #[test]
    fn test_architecture_display() {
        assert_eq!(format!("{}", ModelArchitecture::Llama), "llama");
    }

    // ── Weight Mapper Tests ─────────────────────────────────────────────

    #[test]
    fn test_weight_mapper_block_tensor() {
        let mapper = WeightMapper::new(ModelArchitecture::Llama);
        let mapped = mapper.map_tensor("blk.0.attn_q.weight").unwrap();
        assert_eq!(mapped.layer_index, Some(0));
        assert_eq!(mapped.component, "attn_q");
    }

    #[test]
    fn test_weight_mapper_block_bias() {
        let mapper = WeightMapper::new(ModelArchitecture::Llama);
        let mapped = mapper.map_tensor("blk.5.attn_v.bias").unwrap();
        assert_eq!(mapped.layer_index, Some(5));
        assert_eq!(mapped.component, "attn_v");
    }

    #[test]
    fn test_weight_mapper_deep_layer() {
        let mapper = WeightMapper::new(ModelArchitecture::BitNet);
        let mapped = mapper.map_tensor("blk.31.ffn_up.weight").unwrap();
        assert_eq!(mapped.layer_index, Some(31));
        assert_eq!(mapped.component, "ffn_up");
    }

    #[test]
    fn test_weight_mapper_token_embedding() {
        let mapper = WeightMapper::new(ModelArchitecture::Llama);
        let mapped = mapper.map_tensor("token_embd.weight").unwrap();
        assert_eq!(mapped.layer_index, None);
        assert_eq!(mapped.component, "token_embd");
    }

    #[test]
    fn test_weight_mapper_output_weight() {
        let mapper = WeightMapper::new(ModelArchitecture::Llama);
        let mapped = mapper.map_tensor("output.weight").unwrap();
        assert_eq!(mapped.layer_index, None);
        assert_eq!(mapped.component, "output");
    }

    #[test]
    fn test_weight_mapper_output_norm() {
        let mapper = WeightMapper::new(ModelArchitecture::Llama);
        let mapped = mapper.map_tensor("output_norm.weight").unwrap();
        assert_eq!(mapped.layer_index, None);
        assert_eq!(mapped.component, "output_norm");
    }

    #[test]
    fn test_weight_mapper_unknown_tensor() {
        let mapper = WeightMapper::new(ModelArchitecture::Llama);
        let err = mapper.map_tensor("some_random_tensor").unwrap_err();
        assert!(matches!(err, GGUFError::WeightMappingError(_)));
    }

    #[test]
    fn test_weight_mapper_architecture() {
        let mapper = WeightMapper::new(ModelArchitecture::BitNet);
        assert_eq!(mapper.architecture(), ModelArchitecture::BitNet);
    }

    #[test]
    fn test_weight_mapper_multiple_layers() {
        let mapper = WeightMapper::new(ModelArchitecture::Llama);
        for i in 0..32 {
            let name = format!("blk.{i}.attn_q.weight");
            let mapped = mapper.map_tensor(&name).unwrap();
            assert_eq!(mapped.layer_index, Some(i));
        }
    }

    // ── Lazy Tensor Loader Tests ────────────────────────────────────────

    #[test]
    fn test_lazy_tensor_not_materialized() {
        let desc = TensorDescriptor::new("t", vec![4], TensorDType::F32, 0).unwrap();
        let tensor = LazyTensor::new(desc, DeviceTarget::Cpu);
        assert!(!tensor.is_materialized());
        assert!(tensor.data().is_none());
    }

    #[test]
    fn test_lazy_tensor_materialize() {
        let desc = TensorDescriptor::new("t", vec![4], TensorDType::F32, 0).unwrap();
        let tensor = LazyTensor::new(desc, DeviceTarget::Cpu);
        let data = vec![0u8; 16]; // 4 * 4 bytes
        tensor.materialize(data.clone()).unwrap();
        assert!(tensor.is_materialized());
        assert_eq!(tensor.data().unwrap(), data);
    }

    #[test]
    fn test_lazy_tensor_materialize_size_mismatch() {
        let desc = TensorDescriptor::new("t", vec![4], TensorDType::F32, 0).unwrap();
        let tensor = LazyTensor::new(desc, DeviceTarget::Cpu);
        let err = tensor.materialize(vec![0u8; 10]).unwrap_err();
        assert!(matches!(err, GGUFError::TensorLoadError(_)));
    }

    #[test]
    fn test_lazy_loader_not_loaded_until_accessed() {
        let desc = TensorDescriptor::new("t1", vec![4], TensorDType::F32, 0).unwrap();
        let loader = LazyTensorLoader::new(vec![(desc, DeviceTarget::Cpu)]);
        assert_eq!(loader.tensor_count(), 1);
        assert_eq!(loader.materialized_count(), 0);

        let handle = loader.get("t1").unwrap();
        assert!(!handle.is_materialized());
    }

    #[test]
    fn test_lazy_loader_load_tensor() {
        let desc = TensorDescriptor::new("t1", vec![4], TensorDType::F32, 0).unwrap();
        let mut loader = LazyTensorLoader::new(vec![(desc, DeviceTarget::Cpu)]);
        loader.register_backing_data("t1", vec![0u8; 16]);

        let handle = loader.load_tensor("t1").unwrap();
        assert!(handle.is_materialized());
        assert_eq!(loader.materialized_count(), 1);
    }

    #[test]
    fn test_lazy_loader_already_materialized() {
        let desc = TensorDescriptor::new("t1", vec![4], TensorDType::F32, 0).unwrap();
        let mut loader = LazyTensorLoader::new(vec![(desc, DeviceTarget::Cpu)]);
        loader.register_backing_data("t1", vec![0u8; 16]);

        loader.load_tensor("t1").unwrap();
        // Second call should succeed without re-loading
        let handle = loader.load_tensor("t1").unwrap();
        assert!(handle.is_materialized());
    }

    #[test]
    fn test_lazy_loader_missing_tensor() {
        let loader = LazyTensorLoader::new(vec![]);
        let err = loader.load_tensor("nonexistent").unwrap_err();
        assert!(matches!(err, GGUFError::TensorLoadError(_)));
    }

    #[test]
    fn test_lazy_loader_no_backing_data() {
        let desc = TensorDescriptor::new("t1", vec![4], TensorDType::F32, 0).unwrap();
        let loader = LazyTensorLoader::new(vec![(desc, DeviceTarget::Cpu)]);
        let err = loader.load_tensor("t1").unwrap_err();
        assert!(matches!(err, GGUFError::TensorLoadError(_)));
    }

    #[test]
    fn test_lazy_loader_tensor_names() {
        let d1 = TensorDescriptor::new("alpha", vec![1], TensorDType::F32, 0).unwrap();
        let d2 = TensorDescriptor::new("beta", vec![1], TensorDType::F32, 0).unwrap();
        let loader =
            LazyTensorLoader::new(vec![(d1, DeviceTarget::Cpu), (d2, DeviceTarget::Gpu(0))]);
        let mut names = loader.tensor_names();
        names.sort();
        assert_eq!(names, vec!["alpha", "beta"]);
    }

    #[test]
    fn test_lazy_loader_multiple_tensors_selective_load() {
        let d1 = TensorDescriptor::new("a", vec![2], TensorDType::F32, 0).unwrap();
        let d2 = TensorDescriptor::new("b", vec![2], TensorDType::F32, 0).unwrap();
        let d3 = TensorDescriptor::new("c", vec![2], TensorDType::F32, 0).unwrap();

        let mut loader = LazyTensorLoader::new(vec![
            (d1, DeviceTarget::Cpu),
            (d2, DeviceTarget::Cpu),
            (d3, DeviceTarget::Cpu),
        ]);
        loader.register_backing_data("a", vec![0u8; 8]);
        loader.register_backing_data("b", vec![0u8; 8]);
        loader.register_backing_data("c", vec![0u8; 8]);

        loader.load_tensor("b").unwrap();
        assert_eq!(loader.materialized_count(), 1);

        assert!(!loader.get("a").unwrap().is_materialized());
        assert!(loader.get("b").unwrap().is_materialized());
        assert!(!loader.get("c").unwrap().is_materialized());
    }

    // ── Load Progress Tests ─────────────────────────────────────────────

    #[test]
    fn test_progress_initial_state() {
        let p = LoadProgress::new(10, 1000);
        assert_eq!(p.tensors_loaded, 0);
        assert_eq!(p.bytes_loaded, 0);
        assert!((p.fraction() - 0.0).abs() < 1e-10);
        assert!((p.percent() - 0.0).abs() < 1e-10);
        assert!(!p.is_complete());
    }

    #[test]
    fn test_progress_record_tensor() {
        let mut p = LoadProgress::new(2, 200);
        p.record_tensor(100);
        assert_eq!(p.tensors_loaded, 1);
        assert_eq!(p.bytes_loaded, 100);
        assert!((p.fraction() - 0.5).abs() < 1e-10);
        assert!((p.percent() - 50.0).abs() < 1e-10);
    }

    #[test]
    fn test_progress_complete() {
        let mut p = LoadProgress::new(2, 200);
        p.record_tensor(100);
        p.record_tensor(100);
        assert!(p.is_complete());
        assert!((p.fraction() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_progress_eta_no_progress() {
        let p = LoadProgress::new(10, 1000);
        assert!(p.eta().is_none());
    }

    #[test]
    fn test_progress_eta_when_complete() {
        let mut p = LoadProgress::new(1, 100);
        p.record_tensor(100);
        assert!(p.eta().is_none());
    }

    #[test]
    fn test_progress_zero_bytes() {
        let p = LoadProgress::new(0, 0);
        assert!((p.fraction() - 1.0).abs() < 1e-10);
        assert!(p.is_complete());
    }

    #[test]
    fn test_progress_display() {
        let mut p = LoadProgress::new(5, 500);
        p.record_tensor(100);
        let s = format!("{p}");
        assert!(s.contains("1/5"));
        assert!(s.contains("100/500"));
    }

    #[test]
    fn test_progress_elapsed() {
        let p = LoadProgress::new(1, 100);
        // elapsed should be very small (just created)
        assert!(p.elapsed().as_secs() < 2);
    }

    // ── Device Target Tests ─────────────────────────────────────────────

    #[test]
    fn test_device_target_display() {
        assert_eq!(format!("{}", DeviceTarget::Cpu), "CPU");
        assert_eq!(format!("{}", DeviceTarget::Gpu(0)), "GPU:0");
        assert_eq!(format!("{}", DeviceTarget::Gpu(3)), "GPU:3");
    }

    #[test]
    fn test_device_target_equality() {
        assert_eq!(DeviceTarget::Cpu, DeviceTarget::Cpu);
        assert_eq!(DeviceTarget::Gpu(0), DeviceTarget::Gpu(0));
        assert_ne!(DeviceTarget::Cpu, DeviceTarget::Gpu(0));
        assert_ne!(DeviceTarget::Gpu(0), DeviceTarget::Gpu(1));
    }

    // ── GGUFLoaderConfig Tests ──────────────────────────────────────────

    #[test]
    fn test_loader_config_defaults() {
        let config = GGUFLoaderConfig::default();
        assert!(config.mmap_enabled);
        assert_eq!(config.validation_level, ValidationLevel::Basic);
        assert!(config.lazy_loading);
        assert_eq!(config.placement_strategy, PlacementStrategy::AllCpu);
        assert_eq!(config.gpu_memory_budget, 0);
        assert!(!config.prefetch);
    }

    // ── GGUFModelLoader Orchestrator Tests ──────────────────────────────

    #[test]
    fn test_model_loader_creation() {
        let loader = GGUFModelLoader::with_defaults();
        assert!(loader.header().is_none());
        assert!(loader.architecture().is_none());
        assert!(loader.tensor_loader().is_none());
        assert!(loader.progress().is_none());
        assert!(loader.config().mmap_enabled);
    }

    #[test]
    fn test_model_loader_validate_header() {
        let mut loader = GGUFModelLoader::with_defaults();
        let header = GGUFHeader::new(GGUF_MAGIC, 3, 10, 5).unwrap();
        let bytes = header.to_bytes();
        loader.validate_header(&bytes).unwrap();
        assert!(loader.header().is_some());
        assert_eq!(loader.header().unwrap().tensor_count, 10);
    }

    #[test]
    fn test_model_loader_validate_header_invalid() {
        let mut loader = GGUFModelLoader::with_defaults();
        let mut bytes = [0u8; 24];
        bytes[0..4].copy_from_slice(&0xDEADBEEFu32.to_le_bytes());
        let err = loader.validate_header(&bytes).unwrap_err();
        assert!(matches!(err, GGUFError::InvalidMagic(_)));
    }

    #[test]
    fn test_model_loader_detect_architecture() {
        let mut loader = GGUFModelLoader::with_defaults();
        let mut meta = GGUFMetadataReader::new();
        meta.insert("general.architecture", MetadataValue::String("bitnet".into()));
        loader.set_metadata(meta);
        let arch = loader.detect_architecture().unwrap();
        assert_eq!(arch, ModelArchitecture::BitNet);
        assert_eq!(loader.architecture(), Some(ModelArchitecture::BitNet));
    }

    #[test]
    fn test_model_loader_prepare_tensors() {
        let config = GGUFLoaderConfig {
            lazy_loading: true,
            placement_strategy: PlacementStrategy::AllCpu,
            ..GGUFLoaderConfig::default()
        };
        let mut loader = GGUFModelLoader::new(config);

        let descs = vec![
            TensorDescriptor::new("t1", vec![100], TensorDType::F32, 0).unwrap(),
            TensorDescriptor::new("t2", vec![200], TensorDType::F16, 400).unwrap(),
        ];
        let allocs = loader.prepare_tensors(descs).unwrap();
        assert_eq!(allocs.len(), 2);
        assert!(loader.tensor_loader().is_some());
        assert!(loader.progress().is_some());
        assert_eq!(loader.progress().unwrap().total_tensors, 2);
    }

    #[test]
    fn test_model_loader_full_pipeline() {
        let config = GGUFLoaderConfig {
            lazy_loading: true,
            placement_strategy: PlacementStrategy::AllCpu,
            ..GGUFLoaderConfig::default()
        };
        let mut loader = GGUFModelLoader::new(config);

        // Step 1: header
        let header = GGUFHeader::new(GGUF_MAGIC, 3, 2, 1).unwrap();
        loader.validate_header(&header.to_bytes()).unwrap();

        // Step 2: metadata
        let mut meta = GGUFMetadataReader::new();
        meta.insert("general.architecture", MetadataValue::String("llama".into()));
        loader.set_metadata(meta);

        // Step 3: detect architecture
        let arch = loader.detect_architecture().unwrap();
        assert_eq!(arch, ModelArchitecture::Llama);

        // Step 4: prepare tensors
        let descs = vec![
            TensorDescriptor::new("token_embd.weight", vec![4096], TensorDType::F32, 0).unwrap(),
            TensorDescriptor::new("blk.0.attn_q.weight", vec![4096, 4096], TensorDType::I2S, 16384)
                .unwrap(),
        ];
        let allocs = loader.prepare_tensors(descs).unwrap();
        assert_eq!(allocs.len(), 2);
        assert_eq!(loader.tensor_loader().unwrap().tensor_count(), 2);
    }

    #[test]
    fn test_model_loader_gpu_placement() {
        let config = GGUFLoaderConfig {
            placement_strategy: PlacementStrategy::AllGpu(0),
            lazy_loading: true,
            ..GGUFLoaderConfig::default()
        };
        let mut loader = GGUFModelLoader::new(config);
        let descs = vec![TensorDescriptor::new("t1", vec![100], TensorDType::F32, 0).unwrap()];
        let allocs = loader.prepare_tensors(descs).unwrap();
        assert_eq!(allocs[0].device, DeviceTarget::Gpu(0));
    }

    #[test]
    fn test_model_loader_empty_model() {
        let mut loader = GGUFModelLoader::with_defaults();
        let header = GGUFHeader::new(GGUF_MAGIC, 3, 0, 0).unwrap();
        loader.validate_header(&header.to_bytes()).unwrap();
        let allocs = loader.prepare_tensors(vec![]).unwrap();
        assert!(allocs.is_empty());
    }

    #[test]
    fn test_model_loader_single_tensor() {
        let mut loader = GGUFModelLoader::with_defaults();
        let descs =
            vec![TensorDescriptor::new("only_tensor", vec![1], TensorDType::F32, 0).unwrap()];
        let allocs = loader.prepare_tensors(descs).unwrap();
        assert_eq!(allocs.len(), 1);
    }

    // ── Error Display Tests ─────────────────────────────────────────────

    #[test]
    fn test_error_display_invalid_magic() {
        let err = GGUFError::InvalidMagic(0xDEADBEEF);
        let s = format!("{err}");
        assert!(s.contains("DEADBEEF"));
        assert!(s.contains("invalid GGUF magic"));
    }

    #[test]
    fn test_error_display_unsupported_version() {
        let err = GGUFError::UnsupportedVersion(99);
        let s = format!("{err}");
        assert!(s.contains("99"));
    }

    #[test]
    fn test_error_display_metadata_key_not_found() {
        let err = GGUFError::MetadataKeyNotFound("my.key".into());
        assert!(format!("{err}").contains("my.key"));
    }

    #[test]
    fn test_error_is_std_error() {
        let err: Box<dyn std::error::Error> = Box::new(GGUFError::IoError("test".into()));
        assert!(format!("{err}").contains("test"));
    }

    // ── Proptest ────────────────────────────────────────────────────────

    mod proptests {
        use super::super::*;
        use proptest::prelude::*;

        fn arb_dtype() -> impl Strategy<Value = TensorDType> {
            prop_oneof![
                Just(TensorDType::I2S),
                Just(TensorDType::Q4_0),
                Just(TensorDType::Q8_0),
                Just(TensorDType::F16),
                Just(TensorDType::F32),
            ]
        }

        proptest! {
            #[test]
            fn prop_header_roundtrip(version in 2u32..=3, tc in 0u64..10000, mc in 0u64..10000) {
                let h = GGUFHeader::new(GGUF_MAGIC, version, tc, mc).unwrap();
                let bytes = h.to_bytes();
                let h2 = GGUFHeader::from_bytes(&bytes).unwrap();
                prop_assert_eq!(h, h2);
            }

            #[test]
            fn prop_header_invalid_magic(magic in 0u32..0x4647_5547u32) {
                // any magic < GGUF_MAGIC is invalid (GGUF_MAGIC excluded)
                let result = GGUFHeader::new(magic, 3, 0, 0);
                prop_assert!(result.is_err());
            }

            #[test]
            fn prop_header_invalid_version(version in 4u32..1000) {
                let result = GGUFHeader::new(GGUF_MAGIC, version, 0, 0);
                prop_assert!(result.is_err());
            }

            #[test]
            fn prop_tensor_descriptor_valid(
                dim1 in 1u64..1000,
                dim2 in 1u64..1000,
                offset in 0u64..1_000_000,
                dtype in arb_dtype(),
            ) {
                let name = format!("tensor_{dim1}_{dim2}");
                let d = TensorDescriptor::new(name, vec![dim1, dim2], dtype, offset).unwrap();
                prop_assert_eq!(d.num_elements(), dim1 * dim2);
                prop_assert_eq!(d.ndim(), 2);
                prop_assert!(d.size > 0);
            }

            #[test]
            fn prop_tensor_descriptor_size_consistent(
                elements in 1u64..10000,
                dtype in arb_dtype(),
            ) {
                let d = TensorDescriptor::new("t", vec![elements], dtype, 0).unwrap();
                let expected = (elements as f64 * dtype.bytes_per_element()).ceil() as u64;
                prop_assert_eq!(d.size, expected);
            }

            #[test]
            fn prop_metadata_insert_retrieve(key in "[a-z.]{1,30}", val in 0u32..u32::MAX) {
                let mut reader = GGUFMetadataReader::new();
                reader.insert(key.clone(), MetadataValue::UInt32(val));
                prop_assert_eq!(reader.get_u32(&key).unwrap(), val);
            }

            #[test]
            fn prop_progress_fraction_bounded(total in 1u64..10000, loaded in 0u64..10000) {
                let mut p = LoadProgress::new(total, total * 100);
                for _ in 0..loaded.min(total) {
                    p.record_tensor(100);
                }
                let frac = p.fraction();
                prop_assert!(frac >= 0.0);
                prop_assert!(frac <= 1.0);
            }

            #[test]
            fn prop_allocator_all_cpu_always_cpu(count in 1usize..20) {
                let mut alloc = TensorAllocator::new(PlacementStrategy::AllCpu, 0);
                for i in 0..count {
                    let d = TensorDescriptor::new(format!("t{i}"), vec![10], TensorDType::F32, 0).unwrap();
                    let dev = alloc.allocate(d).unwrap();
                    prop_assert_eq!(dev, DeviceTarget::Cpu);
                }
            }

            #[test]
            fn prop_weight_mapper_block_tensors(layer in 0usize..100) {
                let mapper = WeightMapper::new(ModelArchitecture::Llama);
                let name = format!("blk.{layer}.attn_q.weight");
                let mapped = mapper.map_tensor(&name).unwrap();
                prop_assert_eq!(mapped.layer_index, Some(layer));
                prop_assert_eq!(mapped.component.as_str(), "attn_q");
            }
        }
    }
}
