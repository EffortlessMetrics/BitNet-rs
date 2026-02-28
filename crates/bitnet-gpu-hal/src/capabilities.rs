//! Backend capability matrix and feature negotiation for GPU backends.
//!
//! Provides [`BackendCapabilityMatrix`] for querying, comparing, and
//! negotiating capabilities across heterogeneous compute backends.

use std::fmt;

// ── Backend identifiers ────────────────────────────────────────────

/// Identifies a compute backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BackendId {
    Cpu,
    Cuda,
    OpenCL,
    Vulkan,
    Metal,
    Rocm,
    WebGpu,
    LevelZero,
}

impl fmt::Display for BackendId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            Self::Cpu => "CPU",
            Self::Cuda => "CUDA",
            Self::OpenCL => "OpenCL",
            Self::Vulkan => "Vulkan",
            Self::Metal => "Metal",
            Self::Rocm => "ROCm",
            Self::WebGpu => "WebGPU",
            Self::LevelZero => "Level Zero",
        };
        f.write_str(name)
    }
}

/// All backend variants in declaration order.
pub const ALL_BACKENDS: &[BackendId] = &[
    BackendId::Cpu,
    BackendId::Cuda,
    BackendId::OpenCL,
    BackendId::Vulkan,
    BackendId::Metal,
    BackendId::Rocm,
    BackendId::WebGpu,
    BackendId::LevelZero,
];

// ── Feature flags (bitfield) ──────────────────────────────────────

/// Bitfield of optional hardware / driver features.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct FeatureFlags {
    bits: u64,
}

impl FeatureFlags {
    pub const NONE: Self = Self { bits: 0 };
    pub const KERNEL_FUSION: Self = Self { bits: 1 << 0 };
    pub const ASYNC_COPY: Self = Self { bits: 1 << 1 };
    pub const TENSOR_CORES: Self = Self { bits: 1 << 2 };
    pub const RAY_TRACING: Self = Self { bits: 1 << 3 };
    pub const COOPERATIVE_GROUPS: Self = Self { bits: 1 << 4 };
    pub const SUBGROUP_OPS: Self = Self { bits: 1 << 5 };
    pub const DYNAMIC_PARALLELISM: Self = Self { bits: 1 << 6 };
    pub const SPARSE_OPERATIONS: Self = Self { bits: 1 << 7 };
    pub const UNIFIED_MEMORY: Self = Self { bits: 1 << 8 };
    pub const HARDWARE_SCHEDULING: Self = Self { bits: 1 << 9 };
    pub const BINDLESS_RESOURCES: Self = Self { bits: 1 << 10 };
    pub const MESH_SHADERS: Self = Self { bits: 1 << 11 };

    /// Create flags from a raw u64 value.
    #[inline]
    pub const fn from_bits(bits: u64) -> Self {
        Self { bits }
    }

    /// Return the raw bits.
    #[inline]
    pub const fn bits(self) -> u64 {
        self.bits
    }

    /// True when *all* bits in `other` are set in `self`.
    #[inline]
    pub const fn contains(self, other: Self) -> bool {
        (self.bits & other.bits) == other.bits
    }

    /// True when *any* bit in `other` is set in `self`.
    #[inline]
    pub const fn intersects(self, other: Self) -> bool {
        (self.bits & other.bits) != 0
    }

    /// True when no bits are set.
    #[inline]
    pub const fn is_empty(self) -> bool {
        self.bits == 0
    }

    /// Number of set bits.
    #[inline]
    pub const fn count(self) -> u32 {
        self.bits.count_ones()
    }

    /// Union of two flag sets.
    #[inline]
    pub const fn union(self, other: Self) -> Self {
        Self { bits: self.bits | other.bits }
    }

    /// Intersection of two flag sets.
    #[inline]
    pub const fn intersection(self, other: Self) -> Self {
        Self { bits: self.bits & other.bits }
    }

    /// Bits in `self` that are not in `other`.
    #[inline]
    pub const fn difference(self, other: Self) -> Self {
        Self { bits: self.bits & !other.bits }
    }
}

impl std::ops::BitOr for FeatureFlags {
    type Output = Self;
    #[inline]
    fn bitor(self, rhs: Self) -> Self {
        self.union(rhs)
    }
}

impl std::ops::BitAnd for FeatureFlags {
    type Output = Self;
    #[inline]
    fn bitand(self, rhs: Self) -> Self {
        self.intersection(rhs)
    }
}

impl std::ops::BitOrAssign for FeatureFlags {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        self.bits |= rhs.bits;
    }
}

impl fmt::Display for FeatureFlags {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        const NAMES: &[(FeatureFlags, &str)] = &[
            (FeatureFlags::KERNEL_FUSION, "kernel_fusion"),
            (FeatureFlags::ASYNC_COPY, "async_copy"),
            (FeatureFlags::TENSOR_CORES, "tensor_cores"),
            (FeatureFlags::RAY_TRACING, "ray_tracing"),
            (FeatureFlags::COOPERATIVE_GROUPS, "cooperative_groups"),
            (FeatureFlags::SUBGROUP_OPS, "subgroup_ops"),
            (FeatureFlags::DYNAMIC_PARALLELISM, "dynamic_parallelism"),
            (FeatureFlags::SPARSE_OPERATIONS, "sparse_operations"),
            (FeatureFlags::UNIFIED_MEMORY, "unified_memory"),
            (FeatureFlags::HARDWARE_SCHEDULING, "hardware_scheduling"),
            (FeatureFlags::BINDLESS_RESOURCES, "bindless_resources"),
            (FeatureFlags::MESH_SHADERS, "mesh_shaders"),
        ];
        let mut first = true;
        for &(flag, name) in NAMES {
            if self.contains(flag) {
                if !first {
                    f.write_str(" | ")?;
                }
                f.write_str(name)?;
                first = false;
            }
        }
        if first {
            f.write_str("(none)")?;
        }
        Ok(())
    }
}

// ── Capability structs ────────────────────────────────────────────

/// Compute capabilities of a backend / device.
#[derive(Debug, Clone)]
pub struct ComputeCapabilities {
    pub max_workgroup_size: [u32; 3],
    pub max_workgroups: [u32; 3],
    pub subgroup_sizes: Vec<u32>,
    pub supports_fp16: bool,
    pub supports_bf16: bool,
    pub supports_int8: bool,
    pub supports_int4: bool,
    pub supports_atomic_float: bool,
    pub shader_model: Option<String>,
}

/// Memory capabilities of a backend / device.
#[derive(Debug, Clone)]
pub struct MemoryCapabilities {
    pub total_memory: u64,
    pub max_buffer_size: u64,
    pub max_shared_memory: u32,
    pub supports_unified_memory: bool,
    pub supports_pinned_memory: bool,
    pub memory_bus_width: u32,
    pub bandwidth_gbps: f64,
}

/// Quantisation / accumulation format support.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FormatSupport {
    pub ternary_i2s: bool,
    pub qk256: bool,
    pub fp16_accumulate: bool,
    pub bf16_accumulate: bool,
    pub int8_dot_product: bool,
}

/// Hard device limits.
#[derive(Debug, Clone)]
pub struct DeviceLimits {
    pub max_compute_units: u32,
    pub max_clock_mhz: u32,
    pub max_alloc_size: u64,
    pub preferred_vector_width_float: u32,
    pub warp_size: u32,
}

/// Full capability snapshot for one backend.
#[derive(Debug, Clone)]
pub struct BackendCapabilities {
    pub backend: BackendId,
    pub compute: ComputeCapabilities,
    pub memory: MemoryCapabilities,
    pub formats: FormatSupport,
    pub features: FeatureFlags,
    pub limits: DeviceLimits,
}

// ── Feature negotiation ───────────────────────────────────────────

/// Describes a set of requirements that a workload needs.
#[derive(Debug, Clone)]
pub struct FeatureRequirement {
    pub name: String,
    pub required_features: FeatureFlags,
    pub min_memory: u64,
    pub min_compute_units: u32,
    pub requires_fp16: bool,
    pub requires_int8: bool,
}

impl FeatureRequirement {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            required_features: FeatureFlags::NONE,
            min_memory: 0,
            min_compute_units: 0,
            requires_fp16: false,
            requires_int8: false,
        }
    }

    pub fn with_features(mut self, flags: FeatureFlags) -> Self {
        self.required_features = flags;
        self
    }

    pub fn with_min_memory(mut self, bytes: u64) -> Self {
        self.min_memory = bytes;
        self
    }

    pub fn with_min_compute_units(mut self, units: u32) -> Self {
        self.min_compute_units = units;
        self
    }

    pub fn with_fp16(mut self) -> Self {
        self.requires_fp16 = true;
        self
    }

    pub fn with_int8(mut self) -> Self {
        self.requires_int8 = true;
        self
    }
}

/// Result of negotiating a single backend against requirements.
#[derive(Debug, Clone)]
pub struct NegotiationResult {
    pub backend: BackendId,
    pub supported_features: FeatureFlags,
    pub unsupported: Vec<String>,
    pub warnings: Vec<String>,
    pub score: f64,
}

/// Side-by-side comparison of two backends.
#[derive(Debug, Clone)]
pub struct Comparison {
    pub a: BackendId,
    pub b: BackendId,
    pub a_score: f64,
    pub b_score: f64,
    pub a_only_features: FeatureFlags,
    pub b_only_features: FeatureFlags,
    pub shared_features: FeatureFlags,
    pub memory_ratio: f64,
    pub compute_ratio: f64,
}

// ── Capability matrix ─────────────────────────────────────────────

/// Collection of [`BackendCapabilities`] with negotiation helpers.
#[derive(Debug, Clone, Default)]
pub struct BackendCapabilityMatrix {
    backends: Vec<BackendCapabilities>,
}

impl BackendCapabilityMatrix {
    /// Empty matrix.
    pub fn new() -> Self {
        Self { backends: Vec::new() }
    }

    /// Pre-populated with sensible defaults for every backend.
    pub fn with_defaults() -> Self {
        let mut m = Self::new();
        for &id in ALL_BACKENDS {
            m.add_backend(default_capabilities(id));
        }
        m
    }

    /// Register (or replace) a backend.
    pub fn add_backend(&mut self, caps: BackendCapabilities) {
        self.remove_backend(caps.backend);
        self.backends.push(caps);
    }

    /// Remove a backend by id. Returns `true` if it was present.
    pub fn remove_backend(&mut self, id: BackendId) -> bool {
        let before = self.backends.len();
        self.backends.retain(|b| b.backend != id);
        self.backends.len() < before
    }

    /// Number of registered backends.
    pub fn len(&self) -> usize {
        self.backends.len()
    }

    /// True when no backends are registered.
    pub fn is_empty(&self) -> bool {
        self.backends.is_empty()
    }

    /// Look up a single backend.
    pub fn get(&self, id: BackendId) -> Option<&BackendCapabilities> {
        self.backends.iter().find(|b| b.backend == id)
    }

    /// All registered backends.
    pub fn backends(&self) -> &[BackendCapabilities] {
        &self.backends
    }

    /// Check whether `backend` advertises `feature`.
    pub fn supports(&self, backend: BackendId, feature: FeatureFlags) -> bool {
        self.get(backend).is_some_and(|b| b.features.contains(feature))
    }

    /// Rank every registered backend against `requirements`.
    /// Results are sorted best-first (descending score).
    pub fn negotiate(&self, requirements: &FeatureRequirement) -> Vec<NegotiationResult> {
        let mut results: Vec<NegotiationResult> =
            self.backends.iter().map(|b| negotiate_one(b, requirements)).collect();
        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        results
    }

    /// Best-scoring backend, or `None` if the matrix is empty.
    pub fn best_backend(&self, requirements: &FeatureRequirement) -> Option<NegotiationResult> {
        self.negotiate(requirements).into_iter().next()
    }

    /// Side-by-side comparison of two backends.
    pub fn compare_backends(&self, a: BackendId, b: BackendId) -> Option<Comparison> {
        let cap_a = self.get(a)?;
        let cap_b = self.get(b)?;
        Some(compare(cap_a, cap_b))
    }
}

// ── Default capabilities per backend ──────────────────────────────

/// Sensible defaults for a given backend.
pub fn default_capabilities(backend: BackendId) -> BackendCapabilities {
    match backend {
        BackendId::Cpu => cpu_defaults(),
        BackendId::Cuda => cuda_defaults(),
        BackendId::OpenCL => opencl_defaults(),
        BackendId::Vulkan => vulkan_defaults(),
        BackendId::Metal => metal_defaults(),
        BackendId::Rocm => rocm_defaults(),
        BackendId::WebGpu => webgpu_defaults(),
        BackendId::LevelZero => level_zero_defaults(),
    }
}

fn cpu_defaults() -> BackendCapabilities {
    BackendCapabilities {
        backend: BackendId::Cpu,
        compute: ComputeCapabilities {
            max_workgroup_size: [1, 1, 1],
            max_workgroups: [1, 1, 1],
            subgroup_sizes: vec![1],
            supports_fp16: false,
            supports_bf16: false,
            supports_int8: true,
            supports_int4: false,
            supports_atomic_float: false,
            shader_model: None,
        },
        memory: MemoryCapabilities {
            total_memory: 16 * GB,
            max_buffer_size: 16 * GB,
            max_shared_memory: 0,
            supports_unified_memory: true,
            supports_pinned_memory: false,
            memory_bus_width: 128,
            bandwidth_gbps: 50.0,
        },
        formats: FormatSupport {
            ternary_i2s: true,
            qk256: true,
            fp16_accumulate: false,
            bf16_accumulate: false,
            int8_dot_product: true,
        },
        features: FeatureFlags::NONE,
        limits: DeviceLimits {
            max_compute_units: 16,
            max_clock_mhz: 4000,
            max_alloc_size: 16 * GB,
            preferred_vector_width_float: 8,
            warp_size: 1,
        },
    }
}

fn cuda_defaults() -> BackendCapabilities {
    BackendCapabilities {
        backend: BackendId::Cuda,
        compute: ComputeCapabilities {
            max_workgroup_size: [1024, 1024, 64],
            max_workgroups: [2_147_483_647, 65535, 65535],
            subgroup_sizes: vec![32],
            supports_fp16: true,
            supports_bf16: true,
            supports_int8: true,
            supports_int4: true,
            supports_atomic_float: true,
            shader_model: Some("sm_86".into()),
        },
        memory: MemoryCapabilities {
            total_memory: 24 * GB,
            max_buffer_size: 24 * GB,
            max_shared_memory: 49152,
            supports_unified_memory: true,
            supports_pinned_memory: true,
            memory_bus_width: 384,
            bandwidth_gbps: 936.0,
        },
        formats: FormatSupport {
            ternary_i2s: true,
            qk256: true,
            fp16_accumulate: true,
            bf16_accumulate: true,
            int8_dot_product: true,
        },
        features: FeatureFlags::KERNEL_FUSION
            .union(FeatureFlags::ASYNC_COPY)
            .union(FeatureFlags::TENSOR_CORES)
            .union(FeatureFlags::COOPERATIVE_GROUPS)
            .union(FeatureFlags::DYNAMIC_PARALLELISM)
            .union(FeatureFlags::UNIFIED_MEMORY)
            .union(FeatureFlags::HARDWARE_SCHEDULING),
        limits: DeviceLimits {
            max_compute_units: 84,
            max_clock_mhz: 1695,
            max_alloc_size: 24 * GB,
            preferred_vector_width_float: 1,
            warp_size: 32,
        },
    }
}

fn opencl_defaults() -> BackendCapabilities {
    BackendCapabilities {
        backend: BackendId::OpenCL,
        compute: ComputeCapabilities {
            max_workgroup_size: [1024, 1024, 64],
            max_workgroups: [65535, 65535, 65535],
            subgroup_sizes: vec![16, 32],
            supports_fp16: true,
            supports_bf16: false,
            supports_int8: true,
            supports_int4: false,
            supports_atomic_float: false,
            shader_model: None,
        },
        memory: MemoryCapabilities {
            total_memory: 8 * GB,
            max_buffer_size: 4 * GB,
            max_shared_memory: 32768,
            supports_unified_memory: false,
            supports_pinned_memory: true,
            memory_bus_width: 256,
            bandwidth_gbps: 448.0,
        },
        formats: FormatSupport {
            ternary_i2s: true,
            qk256: true,
            fp16_accumulate: true,
            bf16_accumulate: false,
            int8_dot_product: true,
        },
        features: FeatureFlags::SUBGROUP_OPS.union(FeatureFlags::ASYNC_COPY),
        limits: DeviceLimits {
            max_compute_units: 64,
            max_clock_mhz: 1500,
            max_alloc_size: 4 * GB,
            preferred_vector_width_float: 4,
            warp_size: 32,
        },
    }
}

fn vulkan_defaults() -> BackendCapabilities {
    BackendCapabilities {
        backend: BackendId::Vulkan,
        compute: ComputeCapabilities {
            max_workgroup_size: [1024, 1024, 64],
            max_workgroups: [65535, 65535, 65535],
            subgroup_sizes: vec![32, 64],
            supports_fp16: true,
            supports_bf16: false,
            supports_int8: true,
            supports_int4: false,
            supports_atomic_float: false,
            shader_model: Some("spirv_1_5".into()),
        },
        memory: MemoryCapabilities {
            total_memory: 8 * GB,
            max_buffer_size: 4 * GB,
            max_shared_memory: 32768,
            supports_unified_memory: false,
            supports_pinned_memory: false,
            memory_bus_width: 256,
            bandwidth_gbps: 448.0,
        },
        formats: FormatSupport {
            ternary_i2s: true,
            qk256: true,
            fp16_accumulate: true,
            bf16_accumulate: false,
            int8_dot_product: true,
        },
        features: FeatureFlags::SUBGROUP_OPS
            .union(FeatureFlags::BINDLESS_RESOURCES)
            .union(FeatureFlags::MESH_SHADERS),
        limits: DeviceLimits {
            max_compute_units: 64,
            max_clock_mhz: 1500,
            max_alloc_size: 4 * GB,
            preferred_vector_width_float: 4,
            warp_size: 32,
        },
    }
}

fn metal_defaults() -> BackendCapabilities {
    BackendCapabilities {
        backend: BackendId::Metal,
        compute: ComputeCapabilities {
            max_workgroup_size: [1024, 1024, 1024],
            max_workgroups: [65535, 65535, 65535],
            subgroup_sizes: vec![32],
            supports_fp16: true,
            supports_bf16: true,
            supports_int8: true,
            supports_int4: false,
            supports_atomic_float: true,
            shader_model: Some("metal_3_0".into()),
        },
        memory: MemoryCapabilities {
            total_memory: 32 * GB,
            max_buffer_size: 32 * GB,
            max_shared_memory: 32768,
            supports_unified_memory: true,
            supports_pinned_memory: false,
            memory_bus_width: 256,
            bandwidth_gbps: 400.0,
        },
        formats: FormatSupport {
            ternary_i2s: true,
            qk256: true,
            fp16_accumulate: true,
            bf16_accumulate: true,
            int8_dot_product: true,
        },
        features: FeatureFlags::KERNEL_FUSION
            .union(FeatureFlags::ASYNC_COPY)
            .union(FeatureFlags::SUBGROUP_OPS)
            .union(FeatureFlags::UNIFIED_MEMORY)
            .union(FeatureFlags::MESH_SHADERS),
        limits: DeviceLimits {
            max_compute_units: 40,
            max_clock_mhz: 1398,
            max_alloc_size: 32 * GB,
            preferred_vector_width_float: 4,
            warp_size: 32,
        },
    }
}

fn rocm_defaults() -> BackendCapabilities {
    BackendCapabilities {
        backend: BackendId::Rocm,
        compute: ComputeCapabilities {
            max_workgroup_size: [1024, 1024, 1024],
            max_workgroups: [2_147_483_647, 65535, 65535],
            subgroup_sizes: vec![64],
            supports_fp16: true,
            supports_bf16: true,
            supports_int8: true,
            supports_int4: true,
            supports_atomic_float: true,
            shader_model: None,
        },
        memory: MemoryCapabilities {
            total_memory: 16 * GB,
            max_buffer_size: 16 * GB,
            max_shared_memory: 65536,
            supports_unified_memory: false,
            supports_pinned_memory: true,
            memory_bus_width: 256,
            bandwidth_gbps: 512.0,
        },
        formats: FormatSupport {
            ternary_i2s: true,
            qk256: true,
            fp16_accumulate: true,
            bf16_accumulate: true,
            int8_dot_product: true,
        },
        features: FeatureFlags::KERNEL_FUSION
            .union(FeatureFlags::ASYNC_COPY)
            .union(FeatureFlags::COOPERATIVE_GROUPS)
            .union(FeatureFlags::SUBGROUP_OPS)
            .union(FeatureFlags::DYNAMIC_PARALLELISM),
        limits: DeviceLimits {
            max_compute_units: 120,
            max_clock_mhz: 1700,
            max_alloc_size: 16 * GB,
            preferred_vector_width_float: 1,
            warp_size: 64,
        },
    }
}

fn webgpu_defaults() -> BackendCapabilities {
    BackendCapabilities {
        backend: BackendId::WebGpu,
        compute: ComputeCapabilities {
            max_workgroup_size: [256, 256, 64],
            max_workgroups: [65535, 65535, 65535],
            subgroup_sizes: vec![],
            supports_fp16: false,
            supports_bf16: false,
            supports_int8: false,
            supports_int4: false,
            supports_atomic_float: false,
            shader_model: Some("wgsl".into()),
        },
        memory: MemoryCapabilities {
            total_memory: 2 * GB,
            max_buffer_size: 256 * MB,
            max_shared_memory: 16384,
            supports_unified_memory: false,
            supports_pinned_memory: false,
            memory_bus_width: 128,
            bandwidth_gbps: 100.0,
        },
        formats: FormatSupport {
            ternary_i2s: false,
            qk256: false,
            fp16_accumulate: false,
            bf16_accumulate: false,
            int8_dot_product: false,
        },
        features: FeatureFlags::NONE,
        limits: DeviceLimits {
            max_compute_units: 16,
            max_clock_mhz: 1000,
            max_alloc_size: 256 * MB,
            preferred_vector_width_float: 4,
            warp_size: 1,
        },
    }
}

fn level_zero_defaults() -> BackendCapabilities {
    BackendCapabilities {
        backend: BackendId::LevelZero,
        compute: ComputeCapabilities {
            max_workgroup_size: [1024, 1024, 1024],
            max_workgroups: [2_147_483_647, 65535, 65535],
            subgroup_sizes: vec![8, 16, 32],
            supports_fp16: true,
            supports_bf16: true,
            supports_int8: true,
            supports_int4: true,
            supports_atomic_float: true,
            shader_model: Some("xe_hpg".into()),
        },
        memory: MemoryCapabilities {
            total_memory: 16 * GB,
            max_buffer_size: 16 * GB,
            max_shared_memory: 65536,
            supports_unified_memory: true,
            supports_pinned_memory: true,
            memory_bus_width: 256,
            bandwidth_gbps: 560.0,
        },
        formats: FormatSupport {
            ternary_i2s: true,
            qk256: true,
            fp16_accumulate: true,
            bf16_accumulate: true,
            int8_dot_product: true,
        },
        features: FeatureFlags::KERNEL_FUSION
            .union(FeatureFlags::ASYNC_COPY)
            .union(FeatureFlags::SUBGROUP_OPS)
            .union(FeatureFlags::UNIFIED_MEMORY)
            .union(FeatureFlags::HARDWARE_SCHEDULING),
        limits: DeviceLimits {
            max_compute_units: 512,
            max_clock_mhz: 2100,
            max_alloc_size: 16 * GB,
            preferred_vector_width_float: 8,
            warp_size: 16,
        },
    }
}

const GB: u64 = 1024 * 1024 * 1024;
const MB: u64 = 1024 * 1024;

// ── Scoring / negotiation internals ───────────────────────────────

fn negotiate_one(caps: &BackendCapabilities, req: &FeatureRequirement) -> NegotiationResult {
    let mut unsupported: Vec<String> = Vec::new();
    let mut warnings: Vec<String> = Vec::new();

    // Feature flag check
    let missing_features = req.required_features.difference(caps.features);
    if !missing_features.is_empty() {
        unsupported.push(format!("missing features: {}", missing_features));
    }

    // Memory check
    if caps.memory.total_memory < req.min_memory {
        unsupported.push(format!(
            "insufficient memory: {} < {}",
            caps.memory.total_memory, req.min_memory
        ));
    }

    // Compute units check
    if caps.limits.max_compute_units < req.min_compute_units {
        unsupported.push(format!(
            "insufficient compute units: {} < {}",
            caps.limits.max_compute_units, req.min_compute_units
        ));
    }

    // fp16 check
    if req.requires_fp16 && !caps.compute.supports_fp16 {
        unsupported.push("fp16 not supported".into());
    }

    // int8 check
    if req.requires_int8 && !caps.compute.supports_int8 {
        unsupported.push("int8 not supported".into());
    }

    // Warnings for tight fits
    if req.min_memory > 0
        && caps.memory.total_memory < req.min_memory * 2
        && caps.memory.total_memory >= req.min_memory
    {
        warnings.push("memory headroom is tight".into());
    }

    let score = compute_score(caps, req, &unsupported);

    NegotiationResult {
        backend: caps.backend,
        supported_features: caps.features.intersection(req.required_features),
        unsupported,
        warnings,
        score,
    }
}

/// Score: higher is better.  Penalised heavily for unsupported items.
fn compute_score(
    caps: &BackendCapabilities,
    req: &FeatureRequirement,
    unsupported: &[String],
) -> f64 {
    if !unsupported.is_empty() {
        return 0.0;
    }

    let mut score = 0.0;

    // Memory score (log-scaled, max 30 points)
    let mem_gb = caps.memory.total_memory as f64 / (1024.0 * 1024.0 * 1024.0);
    score += (mem_gb.ln_1p() * 10.0).min(30.0);

    // Compute units score (max 25 points)
    score += (caps.limits.max_compute_units as f64 / 10.0).min(25.0);

    // Bandwidth score (max 20 points)
    score += (caps.memory.bandwidth_gbps / 50.0).min(20.0);

    // Feature match bonus (max 15 points)
    if !req.required_features.is_empty() {
        let matched = caps.features.intersection(req.required_features).count() as f64;
        let total = req.required_features.count() as f64;
        score += (matched / total) * 15.0;
    } else {
        // Bonus proportional to total features
        score += (caps.features.count() as f64).min(15.0);
    }

    // Format support bonus (max 10 points)
    let mut fmt_score = 0.0;
    if caps.formats.ternary_i2s {
        fmt_score += 2.0;
    }
    if caps.formats.qk256 {
        fmt_score += 2.0;
    }
    if caps.formats.fp16_accumulate {
        fmt_score += 2.0;
    }
    if caps.formats.bf16_accumulate {
        fmt_score += 2.0;
    }
    if caps.formats.int8_dot_product {
        fmt_score += 2.0;
    }
    score += fmt_score;

    score
}

fn compare(a: &BackendCapabilities, b: &BackendCapabilities) -> Comparison {
    let dummy_req = FeatureRequirement::new("comparison");
    let a_score = compute_score(a, &dummy_req, &[]);
    let b_score = compute_score(b, &dummy_req, &[]);

    let a_only = a.features.difference(b.features);
    let b_only = b.features.difference(a.features);
    let shared = a.features.intersection(b.features);

    let memory_ratio = if b.memory.total_memory > 0 {
        a.memory.total_memory as f64 / b.memory.total_memory as f64
    } else {
        f64::INFINITY
    };

    let compute_ratio = if b.limits.max_compute_units > 0 {
        a.limits.max_compute_units as f64 / b.limits.max_compute_units as f64
    } else {
        f64::INFINITY
    };

    Comparison {
        a: a.backend,
        b: b.backend,
        a_score,
        b_score,
        a_only_features: a_only,
        b_only_features: b_only,
        shared_features: shared,
        memory_ratio,
        compute_ratio,
    }
}

// ── Tests ─────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Default capability tests ──────────────────────────────

    #[test]
    fn default_cpu_has_reasonable_memory() {
        let caps = default_capabilities(BackendId::Cpu);
        assert!(caps.memory.total_memory >= GB);
    }

    #[test]
    fn default_cuda_has_tensor_cores() {
        let caps = default_capabilities(BackendId::Cuda);
        assert!(caps.features.contains(FeatureFlags::TENSOR_CORES));
    }

    #[test]
    fn default_opencl_no_tensor_cores() {
        let caps = default_capabilities(BackendId::OpenCL);
        assert!(!caps.features.contains(FeatureFlags::TENSOR_CORES));
    }

    #[test]
    fn default_cuda_supports_fp16() {
        let caps = default_capabilities(BackendId::Cuda);
        assert!(caps.compute.supports_fp16);
    }

    #[test]
    fn default_cpu_no_fp16() {
        let caps = default_capabilities(BackendId::Cpu);
        assert!(!caps.compute.supports_fp16);
    }

    #[test]
    fn default_metal_has_unified_memory() {
        let caps = default_capabilities(BackendId::Metal);
        assert!(caps.memory.supports_unified_memory);
    }

    #[test]
    fn default_webgpu_conservative_limits() {
        let caps = default_capabilities(BackendId::WebGpu);
        assert!(caps.memory.total_memory <= 4 * GB);
        assert!(caps.memory.max_buffer_size <= GB);
        assert!(caps.limits.max_compute_units <= 32);
    }

    #[test]
    fn default_webgpu_no_advanced_formats() {
        let caps = default_capabilities(BackendId::WebGpu);
        assert!(!caps.formats.ternary_i2s);
        assert!(!caps.formats.qk256);
        assert!(!caps.formats.fp16_accumulate);
    }

    #[test]
    fn default_rocm_has_large_shared_memory() {
        let caps = default_capabilities(BackendId::Rocm);
        assert!(caps.compute.max_workgroup_size[0] >= 1024);
        assert!(caps.memory.max_shared_memory >= 65536);
    }

    #[test]
    fn default_level_zero_has_subgroup_ops() {
        let caps = default_capabilities(BackendId::LevelZero);
        assert!(caps.features.contains(FeatureFlags::SUBGROUP_OPS));
    }

    #[test]
    fn default_vulkan_has_spirv_shader_model() {
        let caps = default_capabilities(BackendId::Vulkan);
        assert!(caps.compute.shader_model.as_deref().unwrap().starts_with("spirv"));
    }

    #[test]
    fn all_backends_have_defaults() {
        for &id in ALL_BACKENDS {
            let caps = default_capabilities(id);
            assert_eq!(caps.backend, id);
        }
    }

    #[test]
    fn all_backends_count() {
        assert_eq!(ALL_BACKENDS.len(), 8);
    }

    #[test]
    fn cpu_always_supports_i2s_and_qk256() {
        let caps = default_capabilities(BackendId::Cpu);
        assert!(caps.formats.ternary_i2s);
        assert!(caps.formats.qk256);
    }

    #[test]
    fn cuda_warp_size_is_32() {
        let caps = default_capabilities(BackendId::Cuda);
        assert_eq!(caps.limits.warp_size, 32);
    }

    #[test]
    fn rocm_warp_size_is_64() {
        let caps = default_capabilities(BackendId::Rocm);
        assert_eq!(caps.limits.warp_size, 64);
    }

    // ── FeatureFlags bitwise tests ────────────────────────────

    #[test]
    fn feature_flags_empty() {
        assert!(FeatureFlags::NONE.is_empty());
        assert_eq!(FeatureFlags::NONE.count(), 0);
    }

    #[test]
    fn feature_flags_union() {
        let f = FeatureFlags::TENSOR_CORES | FeatureFlags::ASYNC_COPY;
        assert!(f.contains(FeatureFlags::TENSOR_CORES));
        assert!(f.contains(FeatureFlags::ASYNC_COPY));
        assert!(!f.contains(FeatureFlags::RAY_TRACING));
    }

    #[test]
    fn feature_flags_intersection() {
        let a = FeatureFlags::TENSOR_CORES | FeatureFlags::ASYNC_COPY | FeatureFlags::SUBGROUP_OPS;
        let b = FeatureFlags::TENSOR_CORES | FeatureFlags::RAY_TRACING;
        let c = a & b;
        assert!(c.contains(FeatureFlags::TENSOR_CORES));
        assert!(!c.contains(FeatureFlags::ASYNC_COPY));
        assert!(!c.contains(FeatureFlags::RAY_TRACING));
    }

    #[test]
    fn feature_flags_difference() {
        let a = FeatureFlags::TENSOR_CORES | FeatureFlags::ASYNC_COPY;
        let b = FeatureFlags::TENSOR_CORES;
        let d = a.difference(b);
        assert!(d.contains(FeatureFlags::ASYNC_COPY));
        assert!(!d.contains(FeatureFlags::TENSOR_CORES));
    }

    #[test]
    fn feature_flags_contains_self() {
        let f = FeatureFlags::KERNEL_FUSION;
        assert!(f.contains(f));
    }

    #[test]
    fn feature_flags_intersects() {
        let a = FeatureFlags::TENSOR_CORES | FeatureFlags::ASYNC_COPY;
        let b = FeatureFlags::ASYNC_COPY | FeatureFlags::RAY_TRACING;
        assert!(a.intersects(b));
    }

    #[test]
    fn feature_flags_no_intersect() {
        let a = FeatureFlags::TENSOR_CORES;
        let b = FeatureFlags::RAY_TRACING;
        assert!(!a.intersects(b));
    }

    #[test]
    fn feature_flags_count() {
        let f = FeatureFlags::TENSOR_CORES | FeatureFlags::ASYNC_COPY | FeatureFlags::KERNEL_FUSION;
        assert_eq!(f.count(), 3);
    }

    #[test]
    fn feature_flags_bitor_assign() {
        let mut f = FeatureFlags::TENSOR_CORES;
        f |= FeatureFlags::ASYNC_COPY;
        assert!(f.contains(FeatureFlags::TENSOR_CORES));
        assert!(f.contains(FeatureFlags::ASYNC_COPY));
    }

    #[test]
    fn feature_flags_from_bits_roundtrip() {
        let original = FeatureFlags::TENSOR_CORES | FeatureFlags::SUBGROUP_OPS;
        let restored = FeatureFlags::from_bits(original.bits());
        assert_eq!(original, restored);
    }

    #[test]
    fn feature_flags_display_single() {
        let f = FeatureFlags::TENSOR_CORES;
        assert_eq!(format!("{f}"), "tensor_cores");
    }

    #[test]
    fn feature_flags_display_multiple() {
        let f = FeatureFlags::KERNEL_FUSION | FeatureFlags::ASYNC_COPY;
        let s = format!("{f}");
        assert!(s.contains("kernel_fusion"));
        assert!(s.contains("async_copy"));
    }

    #[test]
    fn feature_flags_display_none() {
        assert_eq!(format!("{}", FeatureFlags::NONE), "(none)");
    }

    // ── Matrix basic operations ───────────────────────────────

    #[test]
    fn empty_matrix() {
        let m = BackendCapabilityMatrix::new();
        assert!(m.is_empty());
        assert_eq!(m.len(), 0);
    }

    #[test]
    fn with_defaults_populates_all() {
        let m = BackendCapabilityMatrix::with_defaults();
        assert_eq!(m.len(), ALL_BACKENDS.len());
    }

    #[test]
    fn add_and_get_backend() {
        let mut m = BackendCapabilityMatrix::new();
        m.add_backend(default_capabilities(BackendId::Cuda));
        assert!(m.get(BackendId::Cuda).is_some());
        assert!(m.get(BackendId::Cpu).is_none());
    }

    #[test]
    fn add_backend_replaces_existing() {
        let mut m = BackendCapabilityMatrix::new();
        m.add_backend(default_capabilities(BackendId::Cuda));
        let mut custom = default_capabilities(BackendId::Cuda);
        custom.memory.total_memory = 48 * GB;
        m.add_backend(custom);
        assert_eq!(m.len(), 1);
        assert_eq!(m.get(BackendId::Cuda).unwrap().memory.total_memory, 48 * GB);
    }

    #[test]
    fn remove_backend_returns_true() {
        let mut m = BackendCapabilityMatrix::with_defaults();
        assert!(m.remove_backend(BackendId::Cuda));
        assert!(m.get(BackendId::Cuda).is_none());
    }

    #[test]
    fn remove_absent_backend_returns_false() {
        let mut m = BackendCapabilityMatrix::new();
        assert!(!m.remove_backend(BackendId::Cuda));
    }

    #[test]
    fn supports_query() {
        let m = BackendCapabilityMatrix::with_defaults();
        assert!(m.supports(BackendId::Cuda, FeatureFlags::TENSOR_CORES));
        assert!(!m.supports(BackendId::OpenCL, FeatureFlags::TENSOR_CORES));
    }

    #[test]
    fn supports_missing_backend_returns_false() {
        let m = BackendCapabilityMatrix::new();
        assert!(!m.supports(BackendId::Cuda, FeatureFlags::TENSOR_CORES));
    }

    // ── Negotiation tests ─────────────────────────────────────

    #[test]
    fn negotiate_empty_matrix_returns_empty() {
        let m = BackendCapabilityMatrix::new();
        let req = FeatureRequirement::new("test");
        assert!(m.negotiate(&req).is_empty());
    }

    #[test]
    fn negotiate_single_backend_always_wins() {
        let mut m = BackendCapabilityMatrix::new();
        m.add_backend(default_capabilities(BackendId::Cpu));
        let req = FeatureRequirement::new("basic");
        let results = m.negotiate(&req);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].backend, BackendId::Cpu);
        assert!(results[0].score > 0.0);
    }

    #[test]
    fn negotiate_ranks_cuda_highest_for_fp16() {
        let m = BackendCapabilityMatrix::with_defaults();
        let req = FeatureRequirement::new("fp16_workload")
            .with_fp16()
            .with_features(FeatureFlags::TENSOR_CORES);
        let results = m.negotiate(&req);
        // Backends that don't support fp16 or tensor cores score 0
        let passing: Vec<_> = results.iter().filter(|r| r.score > 0.0).collect();
        assert!(!passing.is_empty());
        assert_eq!(passing[0].backend, BackendId::Cuda);
    }

    #[test]
    fn negotiate_memory_requirement_filters_small_gpus() {
        let m = BackendCapabilityMatrix::with_defaults();
        let req = FeatureRequirement::new("big_model").with_min_memory(20 * GB);
        let results = m.negotiate(&req);
        let passing: Vec<_> = results.iter().filter(|r| r.score > 0.0).collect();
        // Only backends with >= 20 GB pass
        for r in &passing {
            let caps = m.get(r.backend).unwrap();
            assert!(caps.memory.total_memory >= 20 * GB);
        }
    }

    #[test]
    fn negotiate_compute_units_filter() {
        let m = BackendCapabilityMatrix::with_defaults();
        let req = FeatureRequirement::new("high_cu").with_min_compute_units(100);
        let results = m.negotiate(&req);
        let passing: Vec<_> = results.iter().filter(|r| r.score > 0.0).collect();
        for r in &passing {
            let caps = m.get(r.backend).unwrap();
            assert!(caps.limits.max_compute_units >= 100);
        }
    }

    #[test]
    fn negotiate_unsupported_features_listed() {
        let mut m = BackendCapabilityMatrix::new();
        m.add_backend(default_capabilities(BackendId::Cpu));
        let req = FeatureRequirement::new("tensor_cores").with_features(FeatureFlags::TENSOR_CORES);
        let results = m.negotiate(&req);
        assert_eq!(results[0].backend, BackendId::Cpu);
        assert!(!results[0].unsupported.is_empty());
        assert_eq!(results[0].score, 0.0);
    }

    #[test]
    fn negotiate_warnings_for_tight_memory() {
        let mut m = BackendCapabilityMatrix::new();
        let mut caps = default_capabilities(BackendId::Cuda);
        caps.memory.total_memory = 10 * GB;
        m.add_backend(caps);
        let req = FeatureRequirement::new("tight").with_min_memory(6 * GB);
        let results = m.negotiate(&req);
        let cuda = &results[0];
        assert!(cuda.warnings.iter().any(|w| w.contains("tight")));
    }

    #[test]
    fn best_backend_empty_matrix_returns_none() {
        let m = BackendCapabilityMatrix::new();
        let req = FeatureRequirement::new("any");
        assert!(m.best_backend(&req).is_none());
    }

    #[test]
    fn best_backend_returns_highest_score() {
        let m = BackendCapabilityMatrix::with_defaults();
        let req = FeatureRequirement::new("general");
        let best = m.best_backend(&req).unwrap();
        let all = m.negotiate(&req);
        assert_eq!(best.backend, all[0].backend);
    }

    #[test]
    fn negotiation_result_supported_features_subset() {
        let m = BackendCapabilityMatrix::with_defaults();
        let req = FeatureRequirement::new("subset_check")
            .with_features(FeatureFlags::TENSOR_CORES | FeatureFlags::SUBGROUP_OPS);
        let results = m.negotiate(&req);
        for r in &results {
            // supported_features ⊆ required_features
            assert!(req.required_features.contains(r.supported_features));
        }
    }

    #[test]
    fn cpu_always_available_as_fallback() {
        let m = BackendCapabilityMatrix::with_defaults();
        let req = FeatureRequirement::new("minimal");
        let results = m.negotiate(&req);
        assert!(results.iter().any(|r| r.backend == BackendId::Cpu));
        let cpu = results.iter().find(|r| r.backend == BackendId::Cpu);
        assert!(cpu.unwrap().score > 0.0);
    }

    // ── Format support matching ───────────────────────────────

    #[test]
    fn cuda_supports_all_formats() {
        let caps = default_capabilities(BackendId::Cuda);
        assert!(caps.formats.ternary_i2s);
        assert!(caps.formats.qk256);
        assert!(caps.formats.fp16_accumulate);
        assert!(caps.formats.bf16_accumulate);
        assert!(caps.formats.int8_dot_product);
    }

    #[test]
    fn webgpu_supports_no_formats() {
        let caps = default_capabilities(BackendId::WebGpu);
        assert!(!caps.formats.ternary_i2s);
        assert!(!caps.formats.qk256);
        assert!(!caps.formats.fp16_accumulate);
        assert!(!caps.formats.bf16_accumulate);
        assert!(!caps.formats.int8_dot_product);
    }

    // ── Comparison tests ──────────────────────────────────────

    #[test]
    fn compare_cuda_vs_cpu() {
        let m = BackendCapabilityMatrix::with_defaults();
        let cmp = m.compare_backends(BackendId::Cuda, BackendId::Cpu).unwrap();
        assert_eq!(cmp.a, BackendId::Cuda);
        assert_eq!(cmp.b, BackendId::Cpu);
        assert!(cmp.a_score > cmp.b_score);
    }

    #[test]
    fn compare_memory_ratio() {
        let m = BackendCapabilityMatrix::with_defaults();
        let cmp = m.compare_backends(BackendId::Cuda, BackendId::Cpu).unwrap();
        let cuda_mem = m.get(BackendId::Cuda).unwrap().memory.total_memory as f64;
        let cpu_mem = m.get(BackendId::Cpu).unwrap().memory.total_memory as f64;
        let expected = cuda_mem / cpu_mem;
        assert!((cmp.memory_ratio - expected).abs() < 1e-10);
    }

    #[test]
    fn compare_compute_ratio() {
        let m = BackendCapabilityMatrix::with_defaults();
        let cmp = m.compare_backends(BackendId::Cuda, BackendId::Cpu).unwrap();
        let cuda_cu = m.get(BackendId::Cuda).unwrap().limits.max_compute_units as f64;
        let cpu_cu = m.get(BackendId::Cpu).unwrap().limits.max_compute_units as f64;
        let expected = cuda_cu / cpu_cu;
        assert!((cmp.compute_ratio - expected).abs() < 1e-10);
    }

    #[test]
    fn compare_shared_features() {
        let m = BackendCapabilityMatrix::with_defaults();
        let cmp = m.compare_backends(BackendId::Cuda, BackendId::Metal).unwrap();
        // Both have KERNEL_FUSION and ASYNC_COPY
        assert!(cmp.shared_features.contains(FeatureFlags::KERNEL_FUSION));
        assert!(cmp.shared_features.contains(FeatureFlags::ASYNC_COPY));
    }

    #[test]
    fn compare_a_only_features() {
        let m = BackendCapabilityMatrix::with_defaults();
        let cmp = m.compare_backends(BackendId::Cuda, BackendId::OpenCL).unwrap();
        // CUDA has tensor cores, OpenCL doesn't
        assert!(cmp.a_only_features.contains(FeatureFlags::TENSOR_CORES));
    }

    #[test]
    fn compare_missing_backend_returns_none() {
        let m = BackendCapabilityMatrix::new();
        assert!(m.compare_backends(BackendId::Cuda, BackendId::Cpu).is_none());
    }

    // ── Limits validation ─────────────────────────────────────

    #[test]
    fn all_defaults_have_positive_compute_units() {
        for &id in ALL_BACKENDS {
            let caps = default_capabilities(id);
            assert!(caps.limits.max_compute_units > 0, "{id} has 0 compute units");
        }
    }

    #[test]
    fn all_defaults_have_positive_memory() {
        for &id in ALL_BACKENDS {
            let caps = default_capabilities(id);
            assert!(caps.memory.total_memory > 0, "{id} has 0 memory");
        }
    }

    #[test]
    fn all_defaults_max_alloc_le_total_memory() {
        for &id in ALL_BACKENDS {
            let caps = default_capabilities(id);
            assert!(
                caps.limits.max_alloc_size <= caps.memory.total_memory,
                "{id}: max_alloc > total_memory"
            );
        }
    }

    // ── Score calculation ─────────────────────────────────────

    #[test]
    fn unsupported_requirement_gives_zero_score() {
        let mut m = BackendCapabilityMatrix::new();
        m.add_backend(default_capabilities(BackendId::Cpu));
        let req = FeatureRequirement::new("impossible").with_min_memory(1024 * GB);
        let results = m.negotiate(&req);
        assert_eq!(results[0].score, 0.0);
    }

    #[test]
    fn more_memory_yields_higher_score() {
        let mut small = default_capabilities(BackendId::Cuda);
        small.memory.total_memory = 8 * GB;

        let mut big = default_capabilities(BackendId::Cuda);
        big.backend = BackendId::Rocm; // different id to coexist
        big.memory.total_memory = 48 * GB;

        let mut m = BackendCapabilityMatrix::new();
        m.add_backend(small);
        m.add_backend(big);

        let req = FeatureRequirement::new("mem");
        let results = m.negotiate(&req);
        let rocm = results.iter().find(|r| r.backend == BackendId::Rocm).unwrap();
        let cuda = results.iter().find(|r| r.backend == BackendId::Cuda).unwrap();
        assert!(rocm.score > cuda.score);
    }

    #[test]
    fn score_incorporates_bandwidth() {
        let mut slow = default_capabilities(BackendId::Cuda);
        slow.memory.bandwidth_gbps = 100.0;

        let mut fast = default_capabilities(BackendId::Cuda);
        fast.backend = BackendId::Rocm;
        fast.memory.bandwidth_gbps = 900.0;

        let mut m = BackendCapabilityMatrix::new();
        m.add_backend(slow);
        m.add_backend(fast);

        let req = FeatureRequirement::new("bw");
        let results = m.negotiate(&req);
        let fast_r = results.iter().find(|r| r.backend == BackendId::Rocm).unwrap();
        let slow_r = results.iter().find(|r| r.backend == BackendId::Cuda).unwrap();
        assert!(fast_r.score > slow_r.score);
    }

    // ── Dynamic add / remove ──────────────────────────────────

    #[test]
    fn dynamic_add_then_negotiate() {
        let mut m = BackendCapabilityMatrix::new();
        m.add_backend(default_capabilities(BackendId::Cpu));
        let req = FeatureRequirement::new("dyn");
        assert_eq!(m.negotiate(&req).len(), 1);

        m.add_backend(default_capabilities(BackendId::Cuda));
        assert_eq!(m.negotiate(&req).len(), 2);
    }

    #[test]
    fn dynamic_remove_excludes_from_negotiation() {
        let mut m = BackendCapabilityMatrix::with_defaults();
        let req = FeatureRequirement::new("dyn");
        let before = m.negotiate(&req).len();
        m.remove_backend(BackendId::WebGpu);
        let after = m.negotiate(&req).len();
        assert_eq!(after, before - 1);
    }

    // ── BackendId display ─────────────────────────────────────

    #[test]
    fn backend_id_display() {
        assert_eq!(format!("{}", BackendId::Cuda), "CUDA");
        assert_eq!(format!("{}", BackendId::WebGpu), "WebGPU");
        assert_eq!(format!("{}", BackendId::LevelZero), "Level Zero");
    }

    // ── FeatureRequirement builder ────────────────────────────

    #[test]
    fn feature_requirement_builder() {
        let req = FeatureRequirement::new("test")
            .with_features(FeatureFlags::TENSOR_CORES)
            .with_min_memory(8 * GB)
            .with_min_compute_units(32)
            .with_fp16()
            .with_int8();
        assert_eq!(req.name, "test");
        assert!(req.required_features.contains(FeatureFlags::TENSOR_CORES));
        assert_eq!(req.min_memory, 8 * GB);
        assert_eq!(req.min_compute_units, 32);
        assert!(req.requires_fp16);
        assert!(req.requires_int8);
    }

    #[test]
    fn feature_requirement_defaults() {
        let req = FeatureRequirement::new("empty");
        assert!(req.required_features.is_empty());
        assert_eq!(req.min_memory, 0);
        assert_eq!(req.min_compute_units, 0);
        assert!(!req.requires_fp16);
        assert!(!req.requires_int8);
    }

    // ── Int8 requirement negotiation ──────────────────────────

    #[test]
    fn negotiate_int8_requirement() {
        let m = BackendCapabilityMatrix::with_defaults();
        let req = FeatureRequirement::new("int8").with_int8();
        let results = m.negotiate(&req);
        for r in &results {
            let caps = m.get(r.backend).unwrap();
            if caps.compute.supports_int8 {
                assert!(r.score > 0.0);
            } else {
                assert_eq!(r.score, 0.0);
            }
        }
    }

    // ── Backends accessor ─────────────────────────────────────

    #[test]
    fn backends_accessor_returns_all() {
        let m = BackendCapabilityMatrix::with_defaults();
        assert_eq!(m.backends().len(), ALL_BACKENDS.len());
    }

    // ── Level Zero specifics ──────────────────────────────────

    #[test]
    fn level_zero_has_multiple_subgroup_sizes() {
        let caps = default_capabilities(BackendId::LevelZero);
        assert!(caps.compute.subgroup_sizes.len() >= 2);
    }

    #[test]
    fn level_zero_supports_all_int_widths() {
        let caps = default_capabilities(BackendId::LevelZero);
        assert!(caps.compute.supports_int8);
        assert!(caps.compute.supports_int4);
    }
}
