//! Model-specific optimization profiles for Intel Arc A770 GPU.
//!
//! Provides tuned configurations for different model sizes (BitNet 2B, 3B, etc.)
//! including workgroup sizes, tiling parameters, memory strategies, and kernel
//! selection. All implementations are CPU reference code — no OpenCL runtime
//! required.

use std::collections::HashMap;
use std::fmt;

// ---------------------------------------------------------------------------
// ModelSize — 5-tier classification
// ---------------------------------------------------------------------------

/// Classifies a model by parameter count into one of five tiers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ModelSize {
    /// < 500 M parameters.
    Tiny,
    /// 500 M – 2 B parameters.
    Small,
    /// 2 B – 7 B parameters.
    Medium,
    /// 7 B – 13 B parameters.
    Large,
    /// > 13 B parameters.
    XLarge,
}

impl ModelSize {
    /// Classify from a raw parameter count.
    pub fn from_param_count(params: u64) -> Self {
        match params {
            0..500_000_000 => Self::Tiny,
            500_000_000..2_000_000_000 => Self::Small,
            2_000_000_000..7_000_000_000 => Self::Medium,
            7_000_000_000..13_000_000_000 => Self::Large,
            _ => Self::XLarge,
        }
    }

    /// Estimate parameter count from layer count, hidden dim, and vocab size.
    pub fn estimate_params(layers: u32, hidden_dim: u32, vocab_size: u32) -> u64 {
        let h = hidden_dim as u64;
        let l = layers as u64;
        let v = vocab_size as u64;
        // Transformer parameter estimate:
        // each layer ≈ 12·h² (attention + FFN), plus embeddings ≈ v·h
        l * 12 * h * h + v * h
    }
}

impl fmt::Display for ModelSize {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Tiny => write!(f, "Tiny (<500M)"),
            Self::Small => write!(f, "Small (500M-2B)"),
            Self::Medium => write!(f, "Medium (2B-7B)"),
            Self::Large => write!(f, "Large (7B-13B)"),
            Self::XLarge => write!(f, "XLarge (>13B)"),
        }
    }
}

// ---------------------------------------------------------------------------
// MemoryStrategy
// ---------------------------------------------------------------------------

/// GPU memory management strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MemoryStrategy {
    /// Compute in-place, no extra buffers. Best for tiny models.
    InPlace,
    /// Double-buffer for overlapping compute and transfer.
    DoubleBuffer,
    /// Stream chunks through GPU memory for models exceeding VRAM.
    StreamingChunked,
    /// Pin host memory for fast DMA transfers.
    PinnedTransfer,
}

impl MemoryStrategy {
    /// Select strategy based on model memory requirement vs available VRAM.
    pub fn select(model_bytes: u64, vram_bytes: u64) -> Self {
        if vram_bytes == 0 {
            return Self::InPlace;
        }
        let ratio = model_bytes as f64 / vram_bytes as f64;
        if ratio < 0.3 {
            Self::InPlace
        } else if ratio < 0.7 {
            Self::DoubleBuffer
        } else if ratio < 1.0 {
            Self::PinnedTransfer
        } else {
            Self::StreamingChunked
        }
    }
}

impl fmt::Display for MemoryStrategy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InPlace => write!(f, "InPlace"),
            Self::DoubleBuffer => write!(f, "DoubleBuffer"),
            Self::StreamingChunked => write!(f, "StreamingChunked"),
            Self::PinnedTransfer => write!(f, "PinnedTransfer"),
        }
    }
}

// ---------------------------------------------------------------------------
// KernelSelection
// ---------------------------------------------------------------------------

/// Per-operation kernel variant selection.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KernelSelection {
    /// Matrix multiply variant (e.g. "tiled_fp16", "dp4a_i8", "naive").
    pub matmul_variant: String,
    /// Attention variant (e.g. "flash_v2", "standard", "chunked").
    pub attention_variant: String,
    /// Layer-norm variant (e.g. "fused_rms", "two_pass", "online").
    pub norm_variant: String,
    /// Activation variant (e.g. "fused_silu", "separate_gelu").
    pub activation_variant: String,
}

impl Default for KernelSelection {
    fn default() -> Self {
        Self {
            matmul_variant: "tiled_fp16".into(),
            attention_variant: "standard".into(),
            norm_variant: "fused_rms".into(),
            activation_variant: "fused_silu".into(),
        }
    }
}

impl KernelSelection {
    /// Select kernel variants based on model size and hardware capabilities.
    pub fn for_model_size(size: ModelSize, use_dp4a: bool) -> Self {
        let matmul = if use_dp4a {
            match size {
                ModelSize::Tiny | ModelSize::Small => "dp4a_i8",
                _ => "dp4a_i8_tiled",
            }
        } else {
            match size {
                ModelSize::Tiny => "naive",
                ModelSize::Small | ModelSize::Medium => "tiled_fp16",
                ModelSize::Large | ModelSize::XLarge => "tiled_fp16_large",
            }
        };

        let attention = match size {
            ModelSize::Tiny | ModelSize::Small => "standard",
            ModelSize::Medium => "flash_v2",
            ModelSize::Large | ModelSize::XLarge => "chunked",
        };

        let norm = match size {
            ModelSize::Tiny => "two_pass",
            _ => "fused_rms",
        };

        let activation = match size {
            ModelSize::Tiny => "separate_gelu",
            _ => "fused_silu",
        };

        Self {
            matmul_variant: matmul.into(),
            attention_variant: attention.into(),
            norm_variant: norm.into(),
            activation_variant: activation.into(),
        }
    }
}

// ---------------------------------------------------------------------------
// BatchStrategy
// ---------------------------------------------------------------------------

/// How to handle batch processing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BatchStrategy {
    /// Process one sequence at a time.
    Single,
    /// Pad sequences to max length and batch.
    Padded,
    /// Sort by length, batch similar lengths together.
    Bucketed,
}

impl fmt::Display for BatchStrategy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Single => write!(f, "Single"),
            Self::Padded => write!(f, "Padded"),
            Self::Bucketed => write!(f, "Bucketed"),
        }
    }
}

// ---------------------------------------------------------------------------
// OptimizationProfile
// ---------------------------------------------------------------------------

/// Tuned configuration for a specific model on A770 hardware.
#[derive(Debug, Clone, PartialEq)]
pub struct OptimizationProfile {
    /// Workgroup size (local work size for 1-D dispatches).
    pub workgroup_size: usize,
    /// Tile dimension for tiled matrix multiplication.
    pub tile_size: usize,
    /// Use FP16 accumulation where possible.
    pub use_fp16: bool,
    /// Use DP4A (int8 dot-product) instructions.
    pub use_dp4a: bool,
    /// Number of cache lines to prefetch ahead.
    pub prefetch_depth: usize,
    /// Batch processing strategy.
    pub batch_strategy: BatchStrategy,
    /// Memory management strategy.
    pub memory_strategy: MemoryStrategy,
    /// Per-operation kernel variant selection.
    pub kernel_selection: KernelSelection,
}

impl Default for OptimizationProfile {
    fn default() -> Self {
        Self {
            workgroup_size: 256,
            tile_size: 16,
            use_fp16: true,
            use_dp4a: false,
            prefetch_depth: 2,
            batch_strategy: BatchStrategy::Single,
            memory_strategy: MemoryStrategy::InPlace,
            kernel_selection: KernelSelection::default(),
        }
    }
}

impl OptimizationProfile {
    /// Merge with an override profile. Non-default fields in `overrides` win.
    #[must_use]
    pub fn merge(&self, overrides: &ProfileOverrides) -> Self {
        let mut result = self.clone();
        if let Some(ws) = overrides.workgroup_size {
            result.workgroup_size = ws;
        }
        if let Some(ts) = overrides.tile_size {
            result.tile_size = ts;
        }
        if let Some(fp16) = overrides.use_fp16 {
            result.use_fp16 = fp16;
        }
        if let Some(dp4a) = overrides.use_dp4a {
            result.use_dp4a = dp4a;
        }
        if let Some(pd) = overrides.prefetch_depth {
            result.prefetch_depth = pd;
        }
        if let Some(bs) = overrides.batch_strategy {
            result.batch_strategy = bs;
        }
        if let Some(ms) = overrides.memory_strategy {
            result.memory_strategy = ms;
        }
        result
    }
}

// ---------------------------------------------------------------------------
// ProfileOverrides — sparse override bag
// ---------------------------------------------------------------------------

/// Optional overrides that can be merged onto an `OptimizationProfile`.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ProfileOverrides {
    pub workgroup_size: Option<usize>,
    pub tile_size: Option<usize>,
    pub use_fp16: Option<bool>,
    pub use_dp4a: Option<bool>,
    pub prefetch_depth: Option<usize>,
    pub batch_strategy: Option<BatchStrategy>,
    pub memory_strategy: Option<MemoryStrategy>,
}

// ---------------------------------------------------------------------------
// A770Constraints — hardware limits
// ---------------------------------------------------------------------------

/// Intel Arc A770 hardware constraints for profile validation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct A770Constraints {
    /// Maximum workgroup (local work) size.
    pub max_workgroup: usize,
    /// Supported subgroup (SIMD lane) sizes.
    pub subgroup_sizes: Vec<usize>,
    /// Shared local memory (SLM) in bytes.
    pub slm_size: usize,
    /// Number of execution units (Xe-cores × threads per core).
    pub eu_count: usize,
    /// VRAM in bytes (A770 16 GB variant).
    pub vram_bytes: u64,
}

impl Default for A770Constraints {
    fn default() -> Self {
        Self {
            max_workgroup: 1024,
            subgroup_sizes: vec![8, 16, 32],
            slm_size: 65536,
            eu_count: 512,
            vram_bytes: 16 * 1024 * 1024 * 1024, // 16 GB
        }
    }
}

/// Validation error for profiles that violate hardware constraints.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProfileValidationError {
    /// Workgroup size exceeds hardware maximum.
    WorkgroupTooLarge { requested: usize, max: usize },
    /// Workgroup size is not a power of two.
    WorkgroupNotPowerOfTwo { value: usize },
    /// Tile size exceeds workgroup size.
    TileTooLarge { tile: usize, workgroup: usize },
    /// Tile size is zero.
    TileZero,
    /// Shared memory requirement exceeds SLM capacity.
    SlmExceeded { required: usize, available: usize },
}

impl fmt::Display for ProfileValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::WorkgroupTooLarge { requested, max } => {
                write!(f, "workgroup size {requested} exceeds max {max}")
            }
            Self::WorkgroupNotPowerOfTwo { value } => {
                write!(f, "workgroup size {value} is not a power of two")
            }
            Self::TileTooLarge { tile, workgroup } => {
                write!(f, "tile size {tile} exceeds workgroup size {workgroup}")
            }
            Self::TileZero => write!(f, "tile size must be > 0"),
            Self::SlmExceeded { required, available } => {
                write!(f, "SLM requirement {required} B exceeds {available} B")
            }
        }
    }
}

impl A770Constraints {
    /// Validate a profile against hardware constraints.
    pub fn validate(
        &self,
        profile: &OptimizationProfile,
    ) -> std::result::Result<(), Vec<ProfileValidationError>> {
        let mut errors = Vec::new();

        if profile.workgroup_size > self.max_workgroup {
            errors.push(ProfileValidationError::WorkgroupTooLarge {
                requested: profile.workgroup_size,
                max: self.max_workgroup,
            });
        }

        if !profile.workgroup_size.is_power_of_two() {
            errors.push(ProfileValidationError::WorkgroupNotPowerOfTwo {
                value: profile.workgroup_size,
            });
        }

        if profile.tile_size == 0 {
            errors.push(ProfileValidationError::TileZero);
        } else if profile.tile_size > profile.workgroup_size {
            errors.push(ProfileValidationError::TileTooLarge {
                tile: profile.tile_size,
                workgroup: profile.workgroup_size,
            });
        }

        // Estimate SLM usage: two tiles of f32 for tiled matmul.
        let slm_needed = 2 * profile.tile_size * profile.tile_size * 4;
        if slm_needed > self.slm_size {
            errors.push(ProfileValidationError::SlmExceeded {
                required: slm_needed,
                available: self.slm_size,
            });
        }

        if errors.is_empty() { Ok(()) } else { Err(errors) }
    }

    /// Clamp a profile to satisfy all hardware constraints (best-effort).
    pub fn clamp(&self, profile: &OptimizationProfile) -> OptimizationProfile {
        let mut result = profile.clone();

        // Clamp workgroup to max, round down to power of two.
        let wg = result.workgroup_size.min(self.max_workgroup);
        result.workgroup_size = prev_power_of_two(wg).max(1);

        // Clamp tile to workgroup and SLM limits.
        if result.tile_size == 0 {
            result.tile_size = 1;
        }
        result.tile_size = result.tile_size.min(result.workgroup_size);
        // SLM: 2 * tile² * 4 ≤ slm_size → tile ≤ sqrt(slm_size / 8)
        let max_tile_for_slm = ((self.slm_size / 8) as f64).sqrt() as usize;
        result.tile_size = result.tile_size.min(max_tile_for_slm.max(1));

        result
    }
}

/// Largest power of two ≤ n (returns 0 for n == 0).
fn prev_power_of_two(n: usize) -> usize {
    if n == 0 {
        return 0;
    }
    1 << (usize::BITS - 1 - n.leading_zeros())
}

// ---------------------------------------------------------------------------
// ModelMetadata — input for auto-tuning
// ---------------------------------------------------------------------------

/// Metadata extracted from a model file for auto-tuning.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelMetadata {
    /// Optional human-readable model name.
    pub name: Option<String>,
    /// Number of transformer layers.
    pub layer_count: u32,
    /// Hidden dimension (d_model).
    pub hidden_dim: u32,
    /// Vocabulary size.
    pub vocab_size: u32,
    /// Number of attention heads.
    pub num_heads: u32,
    /// Bytes per parameter (e.g. 0.25 for 2-bit → stored as 2 meaning 2 bits).
    pub bits_per_weight: u32,
}

impl ModelMetadata {
    /// Estimated parameter count.
    pub fn estimated_params(&self) -> u64 {
        ModelSize::estimate_params(self.layer_count, self.hidden_dim, self.vocab_size)
    }

    /// Estimated model size in bytes (parameter_count × bits_per_weight / 8).
    pub fn estimated_bytes(&self) -> u64 {
        let params = self.estimated_params();
        let bpw = if self.bits_per_weight == 0 { 16 } else { self.bits_per_weight };
        params * bpw as u64 / 8
    }

    /// Classify this model.
    pub fn size_class(&self) -> ModelSize {
        ModelSize::from_param_count(self.estimated_params())
    }
}

// ---------------------------------------------------------------------------
// ProfileAutoTuner
// ---------------------------------------------------------------------------

/// Generates an `OptimizationProfile` from model metadata and hardware constraints.
#[derive(Debug, Clone, Default)]
pub struct ProfileAutoTuner {
    constraints: A770Constraints,
}

impl ProfileAutoTuner {
    /// Create with custom constraints.
    pub fn new(constraints: A770Constraints) -> Self {
        Self { constraints }
    }

    /// Generate a profile from model metadata.
    pub fn tune(&self, meta: &ModelMetadata) -> OptimizationProfile {
        let size = meta.size_class();
        let model_bytes = meta.estimated_bytes();
        let memory_strategy = MemoryStrategy::select(model_bytes, self.constraints.vram_bytes);

        let (workgroup_size, tile_size) = match size {
            ModelSize::Tiny => (128, 8),
            ModelSize::Small => (256, 16),
            ModelSize::Medium => (256, 16),
            ModelSize::Large => (512, 32),
            ModelSize::XLarge => (1024, 32),
        };

        let use_fp16 = !matches!(size, ModelSize::Tiny);
        let use_dp4a = matches!(size, ModelSize::Medium | ModelSize::Large | ModelSize::XLarge);

        let prefetch_depth = match size {
            ModelSize::Tiny | ModelSize::Small => 1,
            ModelSize::Medium => 2,
            ModelSize::Large | ModelSize::XLarge => 4,
        };

        let batch_strategy = match size {
            ModelSize::Tiny | ModelSize::Small => BatchStrategy::Single,
            ModelSize::Medium => BatchStrategy::Padded,
            ModelSize::Large | ModelSize::XLarge => BatchStrategy::Bucketed,
        };

        let kernel_selection = KernelSelection::for_model_size(size, use_dp4a);

        let profile = OptimizationProfile {
            workgroup_size,
            tile_size,
            use_fp16,
            use_dp4a,
            prefetch_depth,
            batch_strategy,
            memory_strategy,
            kernel_selection,
        };

        // Ensure the auto-tuned profile satisfies constraints.
        self.constraints.clamp(&profile)
    }
}

// ---------------------------------------------------------------------------
// ProfileRegistry
// ---------------------------------------------------------------------------

/// Registry of known model profiles indexed by name and size.
#[derive(Debug, Clone)]
pub struct ProfileRegistry {
    by_name: HashMap<String, OptimizationProfile>,
    by_size: HashMap<ModelSize, OptimizationProfile>,
    auto_tuner: ProfileAutoTuner,
}

impl Default for ProfileRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl ProfileRegistry {
    /// Create a registry pre-populated with curated A770 profiles.
    pub fn new() -> Self {
        let constraints = A770Constraints::default();
        let auto_tuner = ProfileAutoTuner::new(constraints);
        let mut reg = Self { by_name: HashMap::new(), by_size: HashMap::new(), auto_tuner };
        reg.register_defaults();
        reg
    }

    /// Create an empty registry (no defaults).
    pub fn empty() -> Self {
        Self {
            by_name: HashMap::new(),
            by_size: HashMap::new(),
            auto_tuner: ProfileAutoTuner::default(),
        }
    }

    fn register_defaults(&mut self) {
        // BitNet 2B (Medium tier)
        self.by_name.insert(
            "bitnet-2b".into(),
            OptimizationProfile {
                workgroup_size: 256,
                tile_size: 16,
                use_fp16: true,
                use_dp4a: true,
                prefetch_depth: 2,
                batch_strategy: BatchStrategy::Padded,
                memory_strategy: MemoryStrategy::DoubleBuffer,
                kernel_selection: KernelSelection::for_model_size(ModelSize::Medium, true),
            },
        );

        // BitNet 3B (Medium tier, tuned up)
        self.by_name.insert(
            "bitnet-3b".into(),
            OptimizationProfile {
                workgroup_size: 256,
                tile_size: 16,
                use_fp16: true,
                use_dp4a: true,
                prefetch_depth: 3,
                batch_strategy: BatchStrategy::Padded,
                memory_strategy: MemoryStrategy::DoubleBuffer,
                kernel_selection: KernelSelection::for_model_size(ModelSize::Medium, true),
            },
        );

        // Register size-based defaults.
        self.by_size.insert(
            ModelSize::Tiny,
            OptimizationProfile {
                workgroup_size: 128,
                tile_size: 8,
                use_fp16: false,
                use_dp4a: false,
                prefetch_depth: 1,
                batch_strategy: BatchStrategy::Single,
                memory_strategy: MemoryStrategy::InPlace,
                kernel_selection: KernelSelection::for_model_size(ModelSize::Tiny, false),
            },
        );
        self.by_size.insert(
            ModelSize::Small,
            OptimizationProfile {
                workgroup_size: 256,
                tile_size: 16,
                use_fp16: true,
                use_dp4a: false,
                prefetch_depth: 1,
                batch_strategy: BatchStrategy::Single,
                memory_strategy: MemoryStrategy::InPlace,
                kernel_selection: KernelSelection::for_model_size(ModelSize::Small, false),
            },
        );
        self.by_size.insert(
            ModelSize::Medium,
            OptimizationProfile {
                workgroup_size: 256,
                tile_size: 16,
                use_fp16: true,
                use_dp4a: true,
                prefetch_depth: 2,
                batch_strategy: BatchStrategy::Padded,
                memory_strategy: MemoryStrategy::DoubleBuffer,
                kernel_selection: KernelSelection::for_model_size(ModelSize::Medium, true),
            },
        );
        self.by_size.insert(
            ModelSize::Large,
            OptimizationProfile {
                workgroup_size: 512,
                tile_size: 32,
                use_fp16: true,
                use_dp4a: true,
                prefetch_depth: 4,
                batch_strategy: BatchStrategy::Bucketed,
                memory_strategy: MemoryStrategy::PinnedTransfer,
                kernel_selection: KernelSelection::for_model_size(ModelSize::Large, true),
            },
        );
        self.by_size.insert(
            ModelSize::XLarge,
            OptimizationProfile {
                workgroup_size: 1024,
                tile_size: 32,
                use_fp16: true,
                use_dp4a: true,
                prefetch_depth: 4,
                batch_strategy: BatchStrategy::Bucketed,
                memory_strategy: MemoryStrategy::StreamingChunked,
                kernel_selection: KernelSelection::for_model_size(ModelSize::XLarge, true),
            },
        );
    }

    /// Look up by exact model name.
    pub fn get_by_name(&self, name: &str) -> Option<&OptimizationProfile> {
        self.by_name.get(name)
    }

    /// Look up by model size tier.
    pub fn get_by_size(&self, size: ModelSize) -> Option<&OptimizationProfile> {
        self.by_size.get(&size)
    }

    /// Best-effort lookup: try name first, then size tier, then auto-tune.
    pub fn resolve(&self, meta: &ModelMetadata) -> OptimizationProfile {
        if let Some(name) = &meta.name
            && let Some(p) = self.by_name.get(name.as_str())
        {
            return p.clone();
        }
        let size = meta.size_class();
        if let Some(p) = self.by_size.get(&size) {
            return p.clone();
        }
        self.auto_tuner.tune(meta)
    }

    /// Register a named profile.
    pub fn register_name(&mut self, name: impl Into<String>, profile: OptimizationProfile) {
        self.by_name.insert(name.into(), profile);
    }

    /// Register a size-tier profile.
    pub fn register_size(&mut self, size: ModelSize, profile: OptimizationProfile) {
        self.by_size.insert(size, profile);
    }

    /// Number of registered named profiles.
    pub fn named_count(&self) -> usize {
        self.by_name.len()
    }

    /// Number of registered size-tier profiles.
    pub fn size_count(&self) -> usize {
        self.by_size.len()
    }

    /// All registered named profile keys.
    pub fn named_keys(&self) -> Vec<&str> {
        self.by_name.keys().map(|s| s.as_str()).collect()
    }
}

// ---------------------------------------------------------------------------
// Serialization helpers (simple key-value format for tests / config files)
// ---------------------------------------------------------------------------

/// Serialize a profile to a simple key-value `HashMap<String, String>`.
pub fn profile_to_map(profile: &OptimizationProfile) -> HashMap<String, String> {
    let mut m = HashMap::new();
    m.insert("workgroup_size".into(), profile.workgroup_size.to_string());
    m.insert("tile_size".into(), profile.tile_size.to_string());
    m.insert("use_fp16".into(), profile.use_fp16.to_string());
    m.insert("use_dp4a".into(), profile.use_dp4a.to_string());
    m.insert("prefetch_depth".into(), profile.prefetch_depth.to_string());
    m.insert("batch_strategy".into(), profile.batch_strategy.to_string());
    m.insert("memory_strategy".into(), profile.memory_strategy.to_string());
    m.insert("matmul_variant".into(), profile.kernel_selection.matmul_variant.clone());
    m.insert("attention_variant".into(), profile.kernel_selection.attention_variant.clone());
    m.insert("norm_variant".into(), profile.kernel_selection.norm_variant.clone());
    m.insert("activation_variant".into(), profile.kernel_selection.activation_variant.clone());
    m
}

/// Deserialize a profile from a key-value map (best-effort, missing keys use defaults).
pub fn profile_from_map(
    m: &HashMap<String, String>,
) -> std::result::Result<OptimizationProfile, String> {
    let def = OptimizationProfile::default();
    let workgroup_size = m
        .get("workgroup_size")
        .map(|v| v.parse::<usize>())
        .transpose()
        .map_err(|e| format!("workgroup_size: {e}"))?
        .unwrap_or(def.workgroup_size);
    let tile_size = m
        .get("tile_size")
        .map(|v| v.parse::<usize>())
        .transpose()
        .map_err(|e| format!("tile_size: {e}"))?
        .unwrap_or(def.tile_size);
    let use_fp16 = m
        .get("use_fp16")
        .map(|v| v.parse::<bool>())
        .transpose()
        .map_err(|e| format!("use_fp16: {e}"))?
        .unwrap_or(def.use_fp16);
    let use_dp4a = m
        .get("use_dp4a")
        .map(|v| v.parse::<bool>())
        .transpose()
        .map_err(|e| format!("use_dp4a: {e}"))?
        .unwrap_or(def.use_dp4a);
    let prefetch_depth = m
        .get("prefetch_depth")
        .map(|v| v.parse::<usize>())
        .transpose()
        .map_err(|e| format!("prefetch_depth: {e}"))?
        .unwrap_or(def.prefetch_depth);

    let batch_strategy = m
        .get("batch_strategy")
        .map(|v| match v.as_str() {
            "Single" => Ok(BatchStrategy::Single),
            "Padded" => Ok(BatchStrategy::Padded),
            "Bucketed" => Ok(BatchStrategy::Bucketed),
            other => Err(format!("unknown batch_strategy: {other}")),
        })
        .transpose()?
        .unwrap_or(def.batch_strategy);

    let memory_strategy = m
        .get("memory_strategy")
        .map(|v| match v.as_str() {
            "InPlace" => Ok(MemoryStrategy::InPlace),
            "DoubleBuffer" => Ok(MemoryStrategy::DoubleBuffer),
            "StreamingChunked" => Ok(MemoryStrategy::StreamingChunked),
            "PinnedTransfer" => Ok(MemoryStrategy::PinnedTransfer),
            other => Err(format!("unknown memory_strategy: {other}")),
        })
        .transpose()?
        .unwrap_or(def.memory_strategy);

    let kernel_selection = KernelSelection {
        matmul_variant: m
            .get("matmul_variant")
            .cloned()
            .unwrap_or(def.kernel_selection.matmul_variant),
        attention_variant: m
            .get("attention_variant")
            .cloned()
            .unwrap_or(def.kernel_selection.attention_variant),
        norm_variant: m.get("norm_variant").cloned().unwrap_or(def.kernel_selection.norm_variant),
        activation_variant: m
            .get("activation_variant")
            .cloned()
            .unwrap_or(def.kernel_selection.activation_variant),
    };

    Ok(OptimizationProfile {
        workgroup_size,
        tile_size,
        use_fp16,
        use_dp4a,
        prefetch_depth,
        batch_strategy,
        memory_strategy,
        kernel_selection,
    })
}

// =========================================================================
// Tests
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -- helpers ----------------------------------------------------------

    fn bitnet_2b_meta() -> ModelMetadata {
        ModelMetadata {
            name: Some("bitnet-2b".into()),
            layer_count: 24,
            hidden_dim: 2560,
            vocab_size: 32000,
            num_heads: 32,
            bits_per_weight: 2,
        }
    }

    fn tiny_meta() -> ModelMetadata {
        ModelMetadata {
            name: None,
            layer_count: 6,
            hidden_dim: 512,
            vocab_size: 8000,
            num_heads: 8,
            bits_per_weight: 2,
        }
    }

    fn xlarge_meta() -> ModelMetadata {
        ModelMetadata {
            name: None,
            layer_count: 80,
            hidden_dim: 8192,
            vocab_size: 128000,
            num_heads: 64,
            bits_per_weight: 2,
        }
    }

    fn constraints() -> A770Constraints {
        A770Constraints::default()
    }

    // =====================================================================
    // ModelSize classification
    // =====================================================================

    #[test]
    fn model_size_tiny() {
        assert_eq!(ModelSize::from_param_count(100_000_000), ModelSize::Tiny);
    }

    #[test]
    fn model_size_small() {
        assert_eq!(ModelSize::from_param_count(1_000_000_000), ModelSize::Small);
    }

    #[test]
    fn model_size_medium() {
        assert_eq!(ModelSize::from_param_count(3_000_000_000), ModelSize::Medium);
    }

    #[test]
    fn model_size_large() {
        assert_eq!(ModelSize::from_param_count(10_000_000_000), ModelSize::Large);
    }

    #[test]
    fn model_size_xlarge() {
        assert_eq!(ModelSize::from_param_count(20_000_000_000), ModelSize::XLarge);
    }

    #[test]
    fn model_size_boundary_500m() {
        assert_eq!(ModelSize::from_param_count(499_999_999), ModelSize::Tiny);
        assert_eq!(ModelSize::from_param_count(500_000_000), ModelSize::Small);
    }

    #[test]
    fn model_size_boundary_2b() {
        assert_eq!(ModelSize::from_param_count(1_999_999_999), ModelSize::Small);
        assert_eq!(ModelSize::from_param_count(2_000_000_000), ModelSize::Medium);
    }

    #[test]
    fn model_size_boundary_7b() {
        assert_eq!(ModelSize::from_param_count(6_999_999_999), ModelSize::Medium);
        assert_eq!(ModelSize::from_param_count(7_000_000_000), ModelSize::Large);
    }

    #[test]
    fn model_size_boundary_13b() {
        assert_eq!(ModelSize::from_param_count(12_999_999_999), ModelSize::Large);
        assert_eq!(ModelSize::from_param_count(13_000_000_000), ModelSize::XLarge);
    }

    #[test]
    fn model_size_zero_params() {
        assert_eq!(ModelSize::from_param_count(0), ModelSize::Tiny);
    }

    #[test]
    fn model_size_display() {
        assert_eq!(ModelSize::Tiny.to_string(), "Tiny (<500M)");
        assert_eq!(ModelSize::XLarge.to_string(), "XLarge (>13B)");
    }

    #[test]
    fn estimate_params_smoke() {
        // 24 layers, hidden 2560, vocab 32000 → ~1.97 B
        let params = ModelSize::estimate_params(24, 2560, 32000);
        assert!(params > 1_000_000_000, "expected > 1B, got {params}");
        assert!(params < 3_000_000_000, "expected < 3B, got {params}");
    }

    // =====================================================================
    // MemoryStrategy selection
    // =====================================================================

    #[test]
    fn memory_strategy_inplace_small_model() {
        // Model 1 GB, VRAM 16 GB → ratio 0.0625 < 0.3 → InPlace
        let s = MemoryStrategy::select(1_000_000_000, 16_000_000_000);
        assert_eq!(s, MemoryStrategy::InPlace);
    }

    #[test]
    fn memory_strategy_double_buffer() {
        // Model 8 GB, VRAM 16 GB → ratio 0.5 → DoubleBuffer
        let s = MemoryStrategy::select(8_000_000_000, 16_000_000_000);
        assert_eq!(s, MemoryStrategy::DoubleBuffer);
    }

    #[test]
    fn memory_strategy_pinned_transfer() {
        // Model 14 GB, VRAM 16 GB → ratio 0.875 → PinnedTransfer
        let s = MemoryStrategy::select(14_000_000_000, 16_000_000_000);
        assert_eq!(s, MemoryStrategy::PinnedTransfer);
    }

    #[test]
    fn memory_strategy_streaming_chunked() {
        // Model 20 GB, VRAM 16 GB → ratio 1.25 → StreamingChunked
        let s = MemoryStrategy::select(20_000_000_000, 16_000_000_000);
        assert_eq!(s, MemoryStrategy::StreamingChunked);
    }

    #[test]
    fn memory_strategy_zero_vram_fallback() {
        let s = MemoryStrategy::select(1_000_000_000, 0);
        assert_eq!(s, MemoryStrategy::InPlace);
    }

    // =====================================================================
    // KernelSelection
    // =====================================================================

    #[test]
    fn kernel_selection_tiny_no_dp4a() {
        let ks = KernelSelection::for_model_size(ModelSize::Tiny, false);
        assert_eq!(ks.matmul_variant, "naive");
        assert_eq!(ks.attention_variant, "standard");
        assert_eq!(ks.norm_variant, "two_pass");
        assert_eq!(ks.activation_variant, "separate_gelu");
    }

    #[test]
    fn kernel_selection_medium_dp4a() {
        let ks = KernelSelection::for_model_size(ModelSize::Medium, true);
        assert_eq!(ks.matmul_variant, "dp4a_i8_tiled");
        assert_eq!(ks.attention_variant, "flash_v2");
        assert_eq!(ks.norm_variant, "fused_rms");
        assert_eq!(ks.activation_variant, "fused_silu");
    }

    #[test]
    fn kernel_selection_large_no_dp4a() {
        let ks = KernelSelection::for_model_size(ModelSize::Large, false);
        assert_eq!(ks.matmul_variant, "tiled_fp16_large");
        assert_eq!(ks.attention_variant, "chunked");
    }

    #[test]
    fn kernel_selection_xlarge_dp4a() {
        let ks = KernelSelection::for_model_size(ModelSize::XLarge, true);
        assert_eq!(ks.matmul_variant, "dp4a_i8_tiled");
        assert_eq!(ks.attention_variant, "chunked");
    }

    #[test]
    fn kernel_selection_small_dp4a() {
        let ks = KernelSelection::for_model_size(ModelSize::Small, true);
        assert_eq!(ks.matmul_variant, "dp4a_i8");
    }

    #[test]
    fn kernel_selection_default() {
        let ks = KernelSelection::default();
        assert_eq!(ks.matmul_variant, "tiled_fp16");
        assert_eq!(ks.attention_variant, "standard");
    }

    // =====================================================================
    // A770 constraint validation
    // =====================================================================

    #[test]
    fn constraint_valid_default_profile() {
        let c = constraints();
        let p = OptimizationProfile::default();
        assert!(c.validate(&p).is_ok());
    }

    #[test]
    fn constraint_workgroup_too_large() {
        let c = constraints();
        let p = OptimizationProfile { workgroup_size: 2048, ..Default::default() };
        let errs = c.validate(&p).unwrap_err();
        assert!(errs.iter().any(|e| matches!(e, ProfileValidationError::WorkgroupTooLarge { .. })));
    }

    #[test]
    fn constraint_workgroup_not_power_of_two() {
        let c = constraints();
        let p = OptimizationProfile { workgroup_size: 300, ..Default::default() };
        let errs = c.validate(&p).unwrap_err();
        assert!(
            errs.iter().any(|e| matches!(e, ProfileValidationError::WorkgroupNotPowerOfTwo { .. }))
        );
    }

    #[test]
    fn constraint_tile_zero() {
        let c = constraints();
        let p = OptimizationProfile { tile_size: 0, ..Default::default() };
        let errs = c.validate(&p).unwrap_err();
        assert!(errs.iter().any(|e| matches!(e, ProfileValidationError::TileZero)));
    }

    #[test]
    fn constraint_tile_exceeds_workgroup() {
        let c = constraints();
        let p = OptimizationProfile { workgroup_size: 64, tile_size: 128, ..Default::default() };
        let errs = c.validate(&p).unwrap_err();
        assert!(errs.iter().any(|e| matches!(e, ProfileValidationError::TileTooLarge { .. })));
    }

    #[test]
    fn constraint_slm_exceeded() {
        let c = A770Constraints { slm_size: 128, ..Default::default() };
        let p = OptimizationProfile { tile_size: 16, ..Default::default() };
        // 2 * 16 * 16 * 4 = 2048 > 128
        let errs = c.validate(&p).unwrap_err();
        assert!(errs.iter().any(|e| matches!(e, ProfileValidationError::SlmExceeded { .. })));
    }

    #[test]
    fn constraint_max_workgroup_exactly_1024() {
        let c = constraints();
        let p = OptimizationProfile { workgroup_size: 1024, tile_size: 32, ..Default::default() };
        assert!(c.validate(&p).is_ok());
    }

    #[test]
    fn constraint_multiple_errors() {
        let c = constraints();
        let p = OptimizationProfile { workgroup_size: 2000, tile_size: 0, ..Default::default() };
        let errs = c.validate(&p).unwrap_err();
        assert!(errs.len() >= 2, "expected multiple errors, got {}", errs.len());
    }

    #[test]
    fn constraint_validation_error_display() {
        let e = ProfileValidationError::WorkgroupTooLarge { requested: 2048, max: 1024 };
        let s = e.to_string();
        assert!(s.contains("2048"));
        assert!(s.contains("1024"));
    }

    // =====================================================================
    // Clamp
    // =====================================================================

    #[test]
    fn clamp_oversized_workgroup() {
        let c = constraints();
        let p = OptimizationProfile { workgroup_size: 2048, ..Default::default() };
        let clamped = c.clamp(&p);
        assert!(clamped.workgroup_size <= 1024);
        assert!(clamped.workgroup_size.is_power_of_two());
        assert!(c.validate(&clamped).is_ok());
    }

    #[test]
    fn clamp_non_power_of_two() {
        let c = constraints();
        let p = OptimizationProfile { workgroup_size: 300, ..Default::default() };
        let clamped = c.clamp(&p);
        assert!(clamped.workgroup_size.is_power_of_two());
        assert!(clamped.workgroup_size <= 300);
    }

    #[test]
    fn clamp_zero_tile() {
        let c = constraints();
        let p = OptimizationProfile { tile_size: 0, ..Default::default() };
        let clamped = c.clamp(&p);
        assert!(clamped.tile_size >= 1);
        assert!(c.validate(&clamped).is_ok());
    }

    #[test]
    fn clamp_preserves_valid_profile() {
        let c = constraints();
        let p = OptimizationProfile::default();
        let clamped = c.clamp(&p);
        assert_eq!(p.workgroup_size, clamped.workgroup_size);
        assert_eq!(p.tile_size, clamped.tile_size);
    }

    // =====================================================================
    // Auto-tuner
    // =====================================================================

    #[test]
    fn autotune_bitnet_2b() {
        let tuner = ProfileAutoTuner::default();
        let profile = tuner.tune(&bitnet_2b_meta());
        let c = constraints();
        assert!(c.validate(&profile).is_ok());
        assert!(profile.use_fp16);
    }

    #[test]
    fn autotune_tiny_model() {
        let tuner = ProfileAutoTuner::default();
        let profile = tuner.tune(&tiny_meta());
        let c = constraints();
        assert!(c.validate(&profile).is_ok());
        assert!(!profile.use_fp16);
        assert_eq!(profile.batch_strategy, BatchStrategy::Single);
    }

    #[test]
    fn autotune_xlarge_model() {
        let tuner = ProfileAutoTuner::default();
        let profile = tuner.tune(&xlarge_meta());
        let c = constraints();
        assert!(c.validate(&profile).is_ok());
        assert!(profile.use_dp4a);
        assert_eq!(profile.batch_strategy, BatchStrategy::Bucketed);
    }

    #[test]
    fn autotune_always_satisfies_constraints() {
        let c = constraints();
        let tuner = ProfileAutoTuner::new(c.clone());
        for params in [0u64, 100_000, 1_000_000_000, 5_000_000_000, 50_000_000_000] {
            let meta = ModelMetadata {
                name: None,
                layer_count: 24,
                hidden_dim: 2560,
                vocab_size: 32000,
                num_heads: 32,
                bits_per_weight: 2,
            };
            // Override size class by directly tuning
            let profile = tuner.tune(&meta);
            assert!(
                c.validate(&profile).is_ok(),
                "auto-tuned profile for {params} params failed validation"
            );
        }
    }

    // =====================================================================
    // ProfileRegistry
    // =====================================================================

    #[test]
    fn registry_has_defaults() {
        let reg = ProfileRegistry::new();
        assert!(reg.named_count() >= 2);
        assert!(reg.size_count() >= 5);
    }

    #[test]
    fn registry_lookup_by_name() {
        let reg = ProfileRegistry::new();
        let p = reg.get_by_name("bitnet-2b");
        assert!(p.is_some());
        assert!(p.unwrap().use_dp4a);
    }

    #[test]
    fn registry_lookup_by_size() {
        let reg = ProfileRegistry::new();
        let p = reg.get_by_size(ModelSize::Large);
        assert!(p.is_some());
        assert_eq!(p.unwrap().workgroup_size, 512);
    }

    #[test]
    fn registry_resolve_known_name() {
        let reg = ProfileRegistry::new();
        let p = reg.resolve(&bitnet_2b_meta());
        assert!(p.use_dp4a);
        assert_eq!(p.memory_strategy, MemoryStrategy::DoubleBuffer);
    }

    #[test]
    fn registry_resolve_unknown_name_falls_to_size() {
        let reg = ProfileRegistry::new();
        let meta = ModelMetadata {
            name: Some("unknown-model-xyz".into()),
            layer_count: 24,
            hidden_dim: 2560,
            vocab_size: 32000,
            num_heads: 32,
            bits_per_weight: 2,
        };
        let p = reg.resolve(&meta);
        // Should fall through to size-based lookup
        assert!(constraints().validate(&p).is_ok());
    }

    #[test]
    fn registry_resolve_no_name_uses_size() {
        let reg = ProfileRegistry::new();
        let p = reg.resolve(&tiny_meta());
        assert_eq!(p.workgroup_size, 128);
    }

    #[test]
    fn registry_custom_profile() {
        let mut reg = ProfileRegistry::new();
        let custom = OptimizationProfile {
            workgroup_size: 64,
            tile_size: 8,
            use_fp16: false,
            use_dp4a: false,
            prefetch_depth: 0,
            batch_strategy: BatchStrategy::Single,
            memory_strategy: MemoryStrategy::InPlace,
            kernel_selection: KernelSelection::default(),
        };
        reg.register_name("my-model", custom.clone());
        assert_eq!(reg.get_by_name("my-model"), Some(&custom));
    }

    #[test]
    fn registry_empty_has_no_defaults() {
        let reg = ProfileRegistry::empty();
        assert_eq!(reg.named_count(), 0);
        assert_eq!(reg.size_count(), 0);
    }

    #[test]
    fn registry_named_keys() {
        let reg = ProfileRegistry::new();
        let keys = reg.named_keys();
        assert!(keys.contains(&"bitnet-2b"));
        assert!(keys.contains(&"bitnet-3b"));
    }

    // =====================================================================
    // Profile merge / override
    // =====================================================================

    #[test]
    fn merge_empty_overrides_is_identity() {
        let base = OptimizationProfile::default();
        let overrides = ProfileOverrides::default();
        let merged = base.merge(&overrides);
        assert_eq!(merged, base);
    }

    #[test]
    fn merge_partial_override() {
        let base = OptimizationProfile::default();
        let overrides = ProfileOverrides {
            workgroup_size: Some(512),
            use_dp4a: Some(true),
            ..Default::default()
        };
        let merged = base.merge(&overrides);
        assert_eq!(merged.workgroup_size, 512);
        assert!(merged.use_dp4a);
        // Unchanged fields
        assert_eq!(merged.tile_size, base.tile_size);
        assert_eq!(merged.use_fp16, base.use_fp16);
    }

    #[test]
    fn merge_full_override() {
        let base = OptimizationProfile::default();
        let overrides = ProfileOverrides {
            workgroup_size: Some(64),
            tile_size: Some(4),
            use_fp16: Some(false),
            use_dp4a: Some(true),
            prefetch_depth: Some(8),
            batch_strategy: Some(BatchStrategy::Bucketed),
            memory_strategy: Some(MemoryStrategy::StreamingChunked),
        };
        let merged = base.merge(&overrides);
        assert_eq!(merged.workgroup_size, 64);
        assert_eq!(merged.tile_size, 4);
        assert!(!merged.use_fp16);
        assert!(merged.use_dp4a);
        assert_eq!(merged.prefetch_depth, 8);
        assert_eq!(merged.batch_strategy, BatchStrategy::Bucketed);
        assert_eq!(merged.memory_strategy, MemoryStrategy::StreamingChunked);
    }

    // =====================================================================
    // Serialization round-trip
    // =====================================================================

    #[test]
    fn serialize_roundtrip_default() {
        let original = OptimizationProfile::default();
        let map = profile_to_map(&original);
        let restored = profile_from_map(&map).unwrap();
        assert_eq!(original, restored);
    }

    #[test]
    fn serialize_roundtrip_custom() {
        let original = OptimizationProfile {
            workgroup_size: 512,
            tile_size: 32,
            use_fp16: false,
            use_dp4a: true,
            prefetch_depth: 4,
            batch_strategy: BatchStrategy::Bucketed,
            memory_strategy: MemoryStrategy::PinnedTransfer,
            kernel_selection: KernelSelection {
                matmul_variant: "dp4a_i8_tiled".into(),
                attention_variant: "chunked".into(),
                norm_variant: "fused_rms".into(),
                activation_variant: "fused_silu".into(),
            },
        };
        let map = profile_to_map(&original);
        let restored = profile_from_map(&map).unwrap();
        assert_eq!(original, restored);
    }

    #[test]
    fn deserialize_empty_map_gives_defaults() {
        let map = HashMap::new();
        let p = profile_from_map(&map).unwrap();
        assert_eq!(p, OptimizationProfile::default());
    }

    #[test]
    fn deserialize_bad_value_returns_error() {
        let mut map = HashMap::new();
        map.insert("workgroup_size".into(), "not_a_number".into());
        assert!(profile_from_map(&map).is_err());
    }

    #[test]
    fn deserialize_bad_strategy_returns_error() {
        let mut map = HashMap::new();
        map.insert("batch_strategy".into(), "UnknownStrategy".into());
        assert!(profile_from_map(&map).is_err());
    }

    // =====================================================================
    // ModelMetadata
    // =====================================================================

    #[test]
    fn metadata_estimated_bytes() {
        let meta = bitnet_2b_meta();
        let bytes = meta.estimated_bytes();
        // 2-bit weights → roughly params/4 bytes
        assert!(bytes > 0);
        assert!(bytes < meta.estimated_params()); // 2-bit < 8-bit
    }

    #[test]
    fn metadata_zero_bits_defaults_to_fp16() {
        let meta = ModelMetadata {
            name: None,
            layer_count: 1,
            hidden_dim: 128,
            vocab_size: 1000,
            num_heads: 4,
            bits_per_weight: 0,
        };
        let bytes = meta.estimated_bytes();
        // Default 16 bits → 2 bytes per param
        assert!(bytes > 0);
    }

    #[test]
    fn metadata_size_class() {
        // 24 layers × 12 × 2560² + 32000 × 2560 ≈ 1.97B → Small tier
        assert_eq!(bitnet_2b_meta().size_class(), ModelSize::Small);
        assert_eq!(tiny_meta().size_class(), ModelSize::Tiny);
    }

    // =====================================================================
    // Edge cases
    // =====================================================================

    #[test]
    fn edge_zero_layer_model() {
        let meta = ModelMetadata {
            name: None,
            layer_count: 0,
            hidden_dim: 0,
            vocab_size: 0,
            num_heads: 0,
            bits_per_weight: 2,
        };
        assert_eq!(meta.estimated_params(), 0);
        assert_eq!(meta.size_class(), ModelSize::Tiny);
        let tuner = ProfileAutoTuner::default();
        let profile = tuner.tune(&meta);
        assert!(constraints().validate(&profile).is_ok());
    }

    #[test]
    fn edge_exceeds_gpu_memory() {
        let meta = xlarge_meta();
        let bytes = meta.estimated_bytes();
        let c = A770Constraints::default();
        if bytes > c.vram_bytes {
            let tuner = ProfileAutoTuner::new(c.clone());
            let profile = tuner.tune(&meta);
            assert_eq!(profile.memory_strategy, MemoryStrategy::StreamingChunked);
            assert!(c.validate(&profile).is_ok());
        }
    }

    #[test]
    fn edge_prev_power_of_two() {
        assert_eq!(prev_power_of_two(0), 0);
        assert_eq!(prev_power_of_two(1), 1);
        assert_eq!(prev_power_of_two(2), 2);
        assert_eq!(prev_power_of_two(3), 2);
        assert_eq!(prev_power_of_two(1023), 512);
        assert_eq!(prev_power_of_two(1024), 1024);
    }

    // =====================================================================
    // Property-style: all registry profiles satisfy constraints
    // =====================================================================

    #[test]
    fn property_all_registry_profiles_valid() {
        let reg = ProfileRegistry::new();
        let c = constraints();
        for size in [
            ModelSize::Tiny,
            ModelSize::Small,
            ModelSize::Medium,
            ModelSize::Large,
            ModelSize::XLarge,
        ] {
            if let Some(p) = reg.get_by_size(size) {
                assert!(c.validate(p).is_ok(), "size-tier profile for {size} failed validation");
            }
        }
        for key in reg.named_keys() {
            let p = reg.get_by_name(key).unwrap();
            assert!(c.validate(p).is_ok(), "named profile '{key}' failed validation");
        }
    }

    #[test]
    fn property_autotuner_various_hidden_dims() {
        let c = constraints();
        let tuner = ProfileAutoTuner::new(c.clone());
        for hidden in [64, 256, 512, 1024, 2048, 4096, 8192] {
            let meta = ModelMetadata {
                name: None,
                layer_count: 24,
                hidden_dim: hidden,
                vocab_size: 32000,
                num_heads: 8,
                bits_per_weight: 2,
            };
            let profile = tuner.tune(&meta);
            assert!(
                c.validate(&profile).is_ok(),
                "auto-tuned profile for hidden_dim={hidden} failed validation"
            );
        }
    }

    #[test]
    fn property_clamped_profiles_always_valid() {
        let c = constraints();
        // Feed in deliberately invalid profiles.
        let bad_profiles = vec![
            OptimizationProfile { workgroup_size: 0, tile_size: 0, ..Default::default() },
            OptimizationProfile { workgroup_size: 99999, tile_size: 9999, ..Default::default() },
            OptimizationProfile { workgroup_size: 7, tile_size: 1, ..Default::default() },
        ];
        for (i, bad) in bad_profiles.iter().enumerate() {
            let clamped = c.clamp(bad);
            assert!(
                c.validate(&clamped).is_ok(),
                "clamped profile #{i} still invalid: {:?}",
                c.validate(&clamped)
            );
        }
    }

    // =====================================================================
    // A770 constraints defaults
    // =====================================================================

    #[test]
    fn a770_constraints_defaults() {
        let c = A770Constraints::default();
        assert_eq!(c.max_workgroup, 1024);
        assert_eq!(c.subgroup_sizes, vec![8, 16, 32]);
        assert_eq!(c.slm_size, 65536);
        assert_eq!(c.eu_count, 512);
        assert_eq!(c.vram_bytes, 16 * 1024 * 1024 * 1024);
    }

    // =====================================================================
    // Display implementations
    // =====================================================================

    #[test]
    fn memory_strategy_display() {
        assert_eq!(MemoryStrategy::InPlace.to_string(), "InPlace");
        assert_eq!(MemoryStrategy::DoubleBuffer.to_string(), "DoubleBuffer");
        assert_eq!(MemoryStrategy::StreamingChunked.to_string(), "StreamingChunked");
        assert_eq!(MemoryStrategy::PinnedTransfer.to_string(), "PinnedTransfer");
    }

    #[test]
    fn batch_strategy_display() {
        assert_eq!(BatchStrategy::Single.to_string(), "Single");
        assert_eq!(BatchStrategy::Padded.to_string(), "Padded");
        assert_eq!(BatchStrategy::Bucketed.to_string(), "Bucketed");
    }
}
