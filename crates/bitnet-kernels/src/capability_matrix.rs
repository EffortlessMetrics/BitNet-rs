//! Device capability matrix for multi-backend hardware support.
//!
//! Maps hardware devices to supported operations and precision levels,
//! enabling runtime capability queries for backend selection.

use std::fmt;

// ---------------------------------------------------------------------------
// DeviceClass
// ---------------------------------------------------------------------------

/// Classification of compute hardware.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum DeviceClass {
    /// Intel Arc discrete GPU (DG2 / Alchemist)
    IntelArc,
    /// Intel Xe integrated / next-gen GPU
    IntelXe,
    /// NVIDIA CUDA-capable GPU
    NvidiaCuda,
    /// AMD ROCm/HIP GPU
    AmdRocm,
    /// Apple Metal GPU (M-series, A-series)
    AppleMetal,
    /// CPU with SIMD (AVX2/AVX-512/NEON)
    CpuSimd,
    /// CPU scalar fallback (no SIMD)
    CpuScalar,
    /// WebGPU (browser / wasm)
    WebGpu,
}

impl fmt::Display for DeviceClass {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::IntelArc => write!(f, "Intel Arc"),
            Self::IntelXe => write!(f, "Intel Xe"),
            Self::NvidiaCuda => write!(f, "NVIDIA CUDA"),
            Self::AmdRocm => write!(f, "AMD ROCm"),
            Self::AppleMetal => write!(f, "Apple Metal"),
            Self::CpuSimd => write!(f, "CPU SIMD"),
            Self::CpuScalar => write!(f, "CPU Scalar"),
            Self::WebGpu => write!(f, "WebGPU"),
        }
    }
}

impl DeviceClass {
    /// All known device classes.
    pub const ALL: &'static [DeviceClass] = &[
        DeviceClass::IntelArc,
        DeviceClass::IntelXe,
        DeviceClass::NvidiaCuda,
        DeviceClass::AmdRocm,
        DeviceClass::AppleMetal,
        DeviceClass::CpuSimd,
        DeviceClass::CpuScalar,
        DeviceClass::WebGpu,
    ];
}

// ---------------------------------------------------------------------------
// OperationCategory
// ---------------------------------------------------------------------------

/// Categories of compute operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum OperationCategory {
    /// Matrix multiplication / GEMM
    MatrixOps,
    /// LayerNorm, RMSNorm
    NormOps,
    /// Activation functions (SiLU, GELU, ReLU)
    ActivationOps,
    /// Rotary position encoding (RoPE)
    PositionEncoding,
    /// Quantized / low-bit compute (I2_S, QK256)
    QuantizedOps,
    /// Attention (flash, paged, grouped-query)
    AttentionOps,
}

impl fmt::Display for OperationCategory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MatrixOps => write!(f, "Matrix Ops"),
            Self::NormOps => write!(f, "Norm Ops"),
            Self::ActivationOps => write!(f, "Activation Ops"),
            Self::PositionEncoding => write!(f, "Position Encoding"),
            Self::QuantizedOps => write!(f, "Quantized Ops"),
            Self::AttentionOps => write!(f, "Attention Ops"),
        }
    }
}

impl OperationCategory {
    /// All known operation categories.
    pub const ALL: &'static [OperationCategory] = &[
        OperationCategory::MatrixOps,
        OperationCategory::NormOps,
        OperationCategory::ActivationOps,
        OperationCategory::PositionEncoding,
        OperationCategory::QuantizedOps,
        OperationCategory::AttentionOps,
    ];
}

// ---------------------------------------------------------------------------
// PrecisionSupport
// ---------------------------------------------------------------------------

/// Numeric precision / data type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum PrecisionSupport {
    FP32,
    FP16,
    BF16,
    INT8,
    INT4,
    /// 1.58-bit / ternary (BitNet I2_S)
    Binary,
}

impl fmt::Display for PrecisionSupport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::FP32 => write!(f, "FP32"),
            Self::FP16 => write!(f, "FP16"),
            Self::BF16 => write!(f, "BF16"),
            Self::INT8 => write!(f, "INT8"),
            Self::INT4 => write!(f, "INT4"),
            Self::Binary => write!(f, "Binary"),
        }
    }
}

impl PrecisionSupport {
    /// All known precision levels.
    pub const ALL: &'static [PrecisionSupport] = &[
        PrecisionSupport::FP32,
        PrecisionSupport::FP16,
        PrecisionSupport::BF16,
        PrecisionSupport::INT8,
        PrecisionSupport::INT4,
        PrecisionSupport::Binary,
    ];
}

// ---------------------------------------------------------------------------
// SupportLevel
// ---------------------------------------------------------------------------

/// How well a device supports a given operation × precision pair.
#[derive(Debug, Clone, PartialEq)]
pub enum SupportLevel {
    /// Fully supported with estimated efficiency in `[0.0, 1.0]`.
    Full(f64),
    /// Partially supported — reason describes the limitation.
    Partial(String),
    /// Supported via software emulation (slow).
    Emulated,
    /// Not supported at all.
    Unsupported,
}

impl SupportLevel {
    /// Returns `true` when the operation can be executed (Full, Partial, or Emulated).
    pub fn is_supported(&self) -> bool {
        !matches!(self, SupportLevel::Unsupported)
    }

    /// Returns the efficiency estimate, if available.
    pub fn efficiency(&self) -> Option<f64> {
        match self {
            SupportLevel::Full(e) => Some(*e),
            _ => None,
        }
    }
}

impl fmt::Display for SupportLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Full(e) => write!(f, "Full ({:.0}%)", e * 100.0),
            Self::Partial(reason) => write!(f, "Partial ({reason})"),
            Self::Emulated => write!(f, "Emulated"),
            Self::Unsupported => write!(f, "Unsupported"),
        }
    }
}

// ---------------------------------------------------------------------------
// CapabilityEntry
// ---------------------------------------------------------------------------

/// A single entry in the capability matrix.
#[derive(Debug, Clone, PartialEq)]
pub struct CapabilityEntry {
    pub operation: OperationCategory,
    pub precision: PrecisionSupport,
    pub support: SupportLevel,
}

impl CapabilityEntry {
    pub fn new(
        operation: OperationCategory,
        precision: PrecisionSupport,
        support: SupportLevel,
    ) -> Self {
        Self { operation, precision, support }
    }
}

// ---------------------------------------------------------------------------
// DeviceProfile
// ---------------------------------------------------------------------------

/// Pre-built profile for a known hardware device.
#[derive(Debug, Clone)]
pub struct DeviceProfile {
    pub device_class: DeviceClass,
    pub name: String,
    pub compute_units: u32,
    pub memory_gb: u32,
    pub capabilities: Vec<CapabilityEntry>,
}

impl DeviceProfile {
    /// Look up the support level for a specific operation + precision.
    pub fn lookup(
        &self,
        operation: OperationCategory,
        precision: PrecisionSupport,
    ) -> &SupportLevel {
        self.capabilities
            .iter()
            .find(|e| e.operation == operation && e.precision == precision)
            .map(|e| &e.support)
            .unwrap_or(&SupportLevel::Unsupported)
    }

    /// Returns all entries for a given operation category.
    pub fn entries_for_operation(&self, op: OperationCategory) -> Vec<&CapabilityEntry> {
        self.capabilities.iter().filter(|e| e.operation == op).collect()
    }

    /// Returns all entries for a given precision.
    pub fn entries_for_precision(&self, prec: PrecisionSupport) -> Vec<&CapabilityEntry> {
        self.capabilities.iter().filter(|e| e.precision == prec).collect()
    }

    /// Count of fully-supported operation×precision pairs.
    pub fn full_support_count(&self) -> usize {
        self.capabilities.iter().filter(|e| matches!(e.support, SupportLevel::Full(_))).count()
    }
}

// ---------------------------------------------------------------------------
// DeviceCapabilityMatrix
// ---------------------------------------------------------------------------

/// Complete capability matrix: device × operation × precision.
#[derive(Debug, Clone)]
pub struct DeviceCapabilityMatrix {
    profiles: Vec<DeviceProfile>,
}

impl DeviceCapabilityMatrix {
    /// Create an empty matrix.
    pub fn new() -> Self {
        Self { profiles: Vec::new() }
    }

    /// Create a matrix pre-loaded with all built-in profiles.
    pub fn with_builtin_profiles() -> Self {
        Self {
            profiles: vec![
                intel_arc_a770(),
                nvidia_rtx_4090(),
                apple_m1(),
                cpu_simd_avx2(),
                cpu_scalar(),
            ],
        }
    }

    /// Add a device profile.
    pub fn add_profile(&mut self, profile: DeviceProfile) {
        self.profiles.push(profile);
    }

    /// Retrieve a profile by device class.
    pub fn profile_for_class(&self, class: DeviceClass) -> Option<&DeviceProfile> {
        self.profiles.iter().find(|p| p.device_class == class)
    }

    /// Retrieve a profile by name (case-insensitive substring match).
    pub fn profile_by_name(&self, name: &str) -> Option<&DeviceProfile> {
        let needle = name.to_lowercase();
        self.profiles.iter().find(|p| p.name.to_lowercase().contains(&needle))
    }

    /// All registered profiles.
    pub fn profiles(&self) -> &[DeviceProfile] {
        &self.profiles
    }
}

impl Default for DeviceCapabilityMatrix {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// CapabilityQuery
// ---------------------------------------------------------------------------

/// Query interface for checking device capabilities.
pub struct CapabilityQuery<'a> {
    profile: &'a DeviceProfile,
}

impl<'a> CapabilityQuery<'a> {
    pub fn new(profile: &'a DeviceProfile) -> Self {
        Self { profile }
    }

    /// Check whether the device supports a specific operation at a given precision.
    pub fn supports(&self, op: OperationCategory, prec: PrecisionSupport) -> bool {
        self.profile.lookup(op, prec).is_supported()
    }

    /// Get the support level for a specific operation at a given precision.
    pub fn support_level(&self, op: OperationCategory, prec: PrecisionSupport) -> &SupportLevel {
        self.profile.lookup(op, prec)
    }

    /// Find the best precision for a given operation (highest efficiency).
    pub fn best_precision_for(&self, op: OperationCategory) -> Option<PrecisionSupport> {
        self.profile
            .entries_for_operation(op)
            .into_iter()
            .filter_map(|e| e.support.efficiency().map(|eff| (e.precision, eff)))
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(prec, _)| prec)
    }

    /// List all operations supported at a given precision.
    pub fn operations_at_precision(&self, prec: PrecisionSupport) -> Vec<OperationCategory> {
        self.profile
            .entries_for_precision(prec)
            .into_iter()
            .filter(|e| e.support.is_supported())
            .map(|e| e.operation)
            .collect()
    }
}

// ---------------------------------------------------------------------------
// CompatibilityReport
// ---------------------------------------------------------------------------

/// Summary report of device compatibility for an inference workload.
#[derive(Debug, Clone)]
pub struct CompatibilityReport {
    pub device_name: String,
    pub device_class: DeviceClass,
    pub supported_ops: Vec<(OperationCategory, PrecisionSupport, SupportLevel)>,
    pub unsupported_ops: Vec<(OperationCategory, PrecisionSupport)>,
    pub overall_ready: bool,
}

impl CompatibilityReport {
    /// Generate a report for the given profile against a set of required
    /// operation × precision pairs.
    pub fn generate(
        profile: &DeviceProfile,
        required: &[(OperationCategory, PrecisionSupport)],
    ) -> Self {
        let mut supported = Vec::new();
        let mut unsupported = Vec::new();

        for &(op, prec) in required {
            let level = profile.lookup(op, prec);
            if level.is_supported() {
                supported.push((op, prec, level.clone()));
            } else {
                unsupported.push((op, prec));
            }
        }

        let overall_ready = unsupported.is_empty();

        Self {
            device_name: profile.name.clone(),
            device_class: profile.device_class,
            supported_ops: supported,
            unsupported_ops: unsupported,
            overall_ready,
        }
    }

    /// Human-readable summary.
    pub fn summary(&self) -> String {
        let supported_count = self.supported_ops.len();
        let total = supported_count + self.unsupported_ops.len();
        let status = if self.overall_ready { "READY" } else { "INCOMPLETE" };
        format!(
            "{} ({}): {status} — {supported_count}/{total} required ops supported",
            self.device_name, self.device_class,
        )
    }
}

impl fmt::Display for CompatibilityReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "{}", self.summary())?;
        if !self.unsupported_ops.is_empty() {
            writeln!(f, "  Missing:")?;
            for (op, prec) in &self.unsupported_ops {
                writeln!(f, "    - {op} @ {prec}")?;
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Pre-built profiles
// ---------------------------------------------------------------------------

/// Intel Arc A770 (Alchemist, 32 Xe-cores, 16 GB GDDR6).
pub fn intel_arc_a770() -> DeviceProfile {
    DeviceProfile {
        device_class: DeviceClass::IntelArc,
        name: "Intel Arc A770".to_string(),
        compute_units: 32,
        memory_gb: 16,
        capabilities: vec![
            // FP32 — fully supported for all ops
            CapabilityEntry::new(
                OperationCategory::MatrixOps,
                PrecisionSupport::FP32,
                SupportLevel::Full(0.85),
            ),
            CapabilityEntry::new(
                OperationCategory::NormOps,
                PrecisionSupport::FP32,
                SupportLevel::Full(0.90),
            ),
            CapabilityEntry::new(
                OperationCategory::ActivationOps,
                PrecisionSupport::FP32,
                SupportLevel::Full(0.90),
            ),
            CapabilityEntry::new(
                OperationCategory::PositionEncoding,
                PrecisionSupport::FP32,
                SupportLevel::Full(0.88),
            ),
            CapabilityEntry::new(
                OperationCategory::AttentionOps,
                PrecisionSupport::FP32,
                SupportLevel::Full(0.82),
            ),
            // FP16 — XMX matrix engines
            CapabilityEntry::new(
                OperationCategory::MatrixOps,
                PrecisionSupport::FP16,
                SupportLevel::Full(0.95),
            ),
            CapabilityEntry::new(
                OperationCategory::NormOps,
                PrecisionSupport::FP16,
                SupportLevel::Full(0.92),
            ),
            CapabilityEntry::new(
                OperationCategory::AttentionOps,
                PrecisionSupport::FP16,
                SupportLevel::Full(0.90),
            ),
            // BF16
            CapabilityEntry::new(
                OperationCategory::MatrixOps,
                PrecisionSupport::BF16,
                SupportLevel::Full(0.93),
            ),
            // INT8 — DP4A
            CapabilityEntry::new(
                OperationCategory::QuantizedOps,
                PrecisionSupport::INT8,
                SupportLevel::Full(0.80),
            ),
            CapabilityEntry::new(
                OperationCategory::MatrixOps,
                PrecisionSupport::INT8,
                SupportLevel::Full(0.78),
            ),
            // INT4
            CapabilityEntry::new(
                OperationCategory::QuantizedOps,
                PrecisionSupport::INT4,
                SupportLevel::Partial("Emulated via INT8 path".to_string()),
            ),
            // Binary / I2_S — needs custom kernels
            CapabilityEntry::new(
                OperationCategory::QuantizedOps,
                PrecisionSupport::Binary,
                SupportLevel::Partial("Custom kernel required".to_string()),
            ),
        ],
    }
}

/// NVIDIA RTX 4090 (Ada Lovelace, 128 SMs, 24 GB GDDR6X).
pub fn nvidia_rtx_4090() -> DeviceProfile {
    DeviceProfile {
        device_class: DeviceClass::NvidiaCuda,
        name: "NVIDIA RTX 4090".to_string(),
        compute_units: 128,
        memory_gb: 24,
        capabilities: vec![
            // FP32
            CapabilityEntry::new(
                OperationCategory::MatrixOps,
                PrecisionSupport::FP32,
                SupportLevel::Full(0.90),
            ),
            CapabilityEntry::new(
                OperationCategory::NormOps,
                PrecisionSupport::FP32,
                SupportLevel::Full(0.95),
            ),
            CapabilityEntry::new(
                OperationCategory::ActivationOps,
                PrecisionSupport::FP32,
                SupportLevel::Full(0.95),
            ),
            CapabilityEntry::new(
                OperationCategory::PositionEncoding,
                PrecisionSupport::FP32,
                SupportLevel::Full(0.92),
            ),
            CapabilityEntry::new(
                OperationCategory::AttentionOps,
                PrecisionSupport::FP32,
                SupportLevel::Full(0.88),
            ),
            CapabilityEntry::new(
                OperationCategory::QuantizedOps,
                PrecisionSupport::FP32,
                SupportLevel::Full(0.85),
            ),
            // FP16 — Tensor Cores
            CapabilityEntry::new(
                OperationCategory::MatrixOps,
                PrecisionSupport::FP16,
                SupportLevel::Full(0.98),
            ),
            CapabilityEntry::new(
                OperationCategory::NormOps,
                PrecisionSupport::FP16,
                SupportLevel::Full(0.95),
            ),
            CapabilityEntry::new(
                OperationCategory::AttentionOps,
                PrecisionSupport::FP16,
                SupportLevel::Full(0.96),
            ),
            // BF16
            CapabilityEntry::new(
                OperationCategory::MatrixOps,
                PrecisionSupport::BF16,
                SupportLevel::Full(0.97),
            ),
            // INT8 — Tensor Cores
            CapabilityEntry::new(
                OperationCategory::QuantizedOps,
                PrecisionSupport::INT8,
                SupportLevel::Full(0.92),
            ),
            CapabilityEntry::new(
                OperationCategory::MatrixOps,
                PrecisionSupport::INT8,
                SupportLevel::Full(0.90),
            ),
            // INT4
            CapabilityEntry::new(
                OperationCategory::QuantizedOps,
                PrecisionSupport::INT4,
                SupportLevel::Full(0.88),
            ),
            // Binary / I2_S
            CapabilityEntry::new(
                OperationCategory::QuantizedOps,
                PrecisionSupport::Binary,
                SupportLevel::Full(0.75),
            ),
        ],
    }
}

/// Apple M1 (8-core GPU, 8/16 GB unified).
pub fn apple_m1() -> DeviceProfile {
    DeviceProfile {
        device_class: DeviceClass::AppleMetal,
        name: "Apple M1".to_string(),
        compute_units: 8,
        memory_gb: 16,
        capabilities: vec![
            // FP32
            CapabilityEntry::new(
                OperationCategory::MatrixOps,
                PrecisionSupport::FP32,
                SupportLevel::Full(0.80),
            ),
            CapabilityEntry::new(
                OperationCategory::NormOps,
                PrecisionSupport::FP32,
                SupportLevel::Full(0.85),
            ),
            CapabilityEntry::new(
                OperationCategory::ActivationOps,
                PrecisionSupport::FP32,
                SupportLevel::Full(0.85),
            ),
            CapabilityEntry::new(
                OperationCategory::PositionEncoding,
                PrecisionSupport::FP32,
                SupportLevel::Full(0.82),
            ),
            CapabilityEntry::new(
                OperationCategory::AttentionOps,
                PrecisionSupport::FP32,
                SupportLevel::Full(0.78),
            ),
            // FP16 — native on Metal
            CapabilityEntry::new(
                OperationCategory::MatrixOps,
                PrecisionSupport::FP16,
                SupportLevel::Full(0.92),
            ),
            CapabilityEntry::new(
                OperationCategory::NormOps,
                PrecisionSupport::FP16,
                SupportLevel::Full(0.90),
            ),
            CapabilityEntry::new(
                OperationCategory::AttentionOps,
                PrecisionSupport::FP16,
                SupportLevel::Full(0.88),
            ),
            // INT8 — emulated
            CapabilityEntry::new(
                OperationCategory::QuantizedOps,
                PrecisionSupport::INT8,
                SupportLevel::Emulated,
            ),
            // Binary — custom
            CapabilityEntry::new(
                OperationCategory::QuantizedOps,
                PrecisionSupport::Binary,
                SupportLevel::Partial("Requires Metal compute shader".to_string()),
            ),
        ],
    }
}

/// CPU with AVX2 SIMD.
pub fn cpu_simd_avx2() -> DeviceProfile {
    DeviceProfile {
        device_class: DeviceClass::CpuSimd,
        name: "CPU SIMD (AVX2)".to_string(),
        compute_units: 1,
        memory_gb: 0,
        capabilities: vec![
            // FP32 — good SIMD coverage
            CapabilityEntry::new(
                OperationCategory::MatrixOps,
                PrecisionSupport::FP32,
                SupportLevel::Full(0.70),
            ),
            CapabilityEntry::new(
                OperationCategory::NormOps,
                PrecisionSupport::FP32,
                SupportLevel::Full(0.75),
            ),
            CapabilityEntry::new(
                OperationCategory::ActivationOps,
                PrecisionSupport::FP32,
                SupportLevel::Full(0.75),
            ),
            CapabilityEntry::new(
                OperationCategory::PositionEncoding,
                PrecisionSupport::FP32,
                SupportLevel::Full(0.72),
            ),
            CapabilityEntry::new(
                OperationCategory::AttentionOps,
                PrecisionSupport::FP32,
                SupportLevel::Full(0.65),
            ),
            CapabilityEntry::new(
                OperationCategory::QuantizedOps,
                PrecisionSupport::FP32,
                SupportLevel::Full(0.68),
            ),
            // INT8 — SIMD accelerated
            CapabilityEntry::new(
                OperationCategory::QuantizedOps,
                PrecisionSupport::INT8,
                SupportLevel::Full(0.60),
            ),
            // Binary / I2_S — AVX2 nibble-LUT path
            CapabilityEntry::new(
                OperationCategory::QuantizedOps,
                PrecisionSupport::Binary,
                SupportLevel::Full(0.50),
            ),
            CapabilityEntry::new(
                OperationCategory::MatrixOps,
                PrecisionSupport::Binary,
                SupportLevel::Full(0.45),
            ),
        ],
    }
}

/// CPU scalar fallback (no SIMD).
pub fn cpu_scalar() -> DeviceProfile {
    DeviceProfile {
        device_class: DeviceClass::CpuScalar,
        name: "CPU Scalar".to_string(),
        compute_units: 1,
        memory_gb: 0,
        capabilities: vec![
            // FP32 — always available
            CapabilityEntry::new(
                OperationCategory::MatrixOps,
                PrecisionSupport::FP32,
                SupportLevel::Full(0.30),
            ),
            CapabilityEntry::new(
                OperationCategory::NormOps,
                PrecisionSupport::FP32,
                SupportLevel::Full(0.35),
            ),
            CapabilityEntry::new(
                OperationCategory::ActivationOps,
                PrecisionSupport::FP32,
                SupportLevel::Full(0.35),
            ),
            CapabilityEntry::new(
                OperationCategory::PositionEncoding,
                PrecisionSupport::FP32,
                SupportLevel::Full(0.32),
            ),
            CapabilityEntry::new(
                OperationCategory::AttentionOps,
                PrecisionSupport::FP32,
                SupportLevel::Full(0.25),
            ),
            CapabilityEntry::new(
                OperationCategory::QuantizedOps,
                PrecisionSupport::FP32,
                SupportLevel::Full(0.28),
            ),
            // Binary / I2_S — scalar path (very slow)
            CapabilityEntry::new(
                OperationCategory::QuantizedOps,
                PrecisionSupport::Binary,
                SupportLevel::Full(0.10),
            ),
        ],
    }
}

/// Returns the CPU scalar profile as a safe universal fallback.
pub fn fallback_profile() -> DeviceProfile {
    cpu_scalar()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // === DeviceClass tests ===

    #[test]
    fn device_class_display_intel_arc() {
        assert_eq!(DeviceClass::IntelArc.to_string(), "Intel Arc");
    }

    #[test]
    fn device_class_display_intel_xe() {
        assert_eq!(DeviceClass::IntelXe.to_string(), "Intel Xe");
    }

    #[test]
    fn device_class_display_nvidia_cuda() {
        assert_eq!(DeviceClass::NvidiaCuda.to_string(), "NVIDIA CUDA");
    }

    #[test]
    fn device_class_display_amd_rocm() {
        assert_eq!(DeviceClass::AmdRocm.to_string(), "AMD ROCm");
    }

    #[test]
    fn device_class_display_apple_metal() {
        assert_eq!(DeviceClass::AppleMetal.to_string(), "Apple Metal");
    }

    #[test]
    fn device_class_display_cpu_simd() {
        assert_eq!(DeviceClass::CpuSimd.to_string(), "CPU SIMD");
    }

    #[test]
    fn device_class_display_cpu_scalar() {
        assert_eq!(DeviceClass::CpuScalar.to_string(), "CPU Scalar");
    }

    #[test]
    fn device_class_display_webgpu() {
        assert_eq!(DeviceClass::WebGpu.to_string(), "WebGPU");
    }

    #[test]
    fn device_class_all_has_8_variants() {
        assert_eq!(DeviceClass::ALL.len(), 8);
    }

    #[test]
    fn device_class_equality() {
        assert_eq!(DeviceClass::IntelArc, DeviceClass::IntelArc);
        assert_ne!(DeviceClass::IntelArc, DeviceClass::NvidiaCuda);
    }

    #[test]
    fn device_class_clone() {
        let a = DeviceClass::IntelArc;
        let b = a;
        assert_eq!(a, b);
    }

    #[test]
    fn device_class_hash() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(DeviceClass::IntelArc);
        set.insert(DeviceClass::IntelArc);
        assert_eq!(set.len(), 1);
    }

    // === OperationCategory tests ===

    #[test]
    fn operation_category_all_has_6_variants() {
        assert_eq!(OperationCategory::ALL.len(), 6);
    }

    #[test]
    fn operation_category_display() {
        assert_eq!(OperationCategory::MatrixOps.to_string(), "Matrix Ops");
        assert_eq!(OperationCategory::NormOps.to_string(), "Norm Ops");
        assert_eq!(OperationCategory::ActivationOps.to_string(), "Activation Ops");
        assert_eq!(OperationCategory::PositionEncoding.to_string(), "Position Encoding");
        assert_eq!(OperationCategory::QuantizedOps.to_string(), "Quantized Ops");
        assert_eq!(OperationCategory::AttentionOps.to_string(), "Attention Ops");
    }

    // === PrecisionSupport tests ===

    #[test]
    fn precision_support_all_has_6_variants() {
        assert_eq!(PrecisionSupport::ALL.len(), 6);
    }

    #[test]
    fn precision_support_display() {
        assert_eq!(PrecisionSupport::FP32.to_string(), "FP32");
        assert_eq!(PrecisionSupport::FP16.to_string(), "FP16");
        assert_eq!(PrecisionSupport::BF16.to_string(), "BF16");
        assert_eq!(PrecisionSupport::INT8.to_string(), "INT8");
        assert_eq!(PrecisionSupport::INT4.to_string(), "INT4");
        assert_eq!(PrecisionSupport::Binary.to_string(), "Binary");
    }

    // === SupportLevel tests ===

    #[test]
    fn support_level_full_is_supported() {
        assert!(SupportLevel::Full(0.9).is_supported());
    }

    #[test]
    fn support_level_partial_is_supported() {
        assert!(SupportLevel::Partial("reason".to_string()).is_supported());
    }

    #[test]
    fn support_level_emulated_is_supported() {
        assert!(SupportLevel::Emulated.is_supported());
    }

    #[test]
    fn support_level_unsupported_is_not_supported() {
        assert!(!SupportLevel::Unsupported.is_supported());
    }

    #[test]
    fn support_level_efficiency_full() {
        assert_eq!(SupportLevel::Full(0.85).efficiency(), Some(0.85));
    }

    #[test]
    fn support_level_efficiency_partial_none() {
        assert_eq!(SupportLevel::Partial("x".to_string()).efficiency(), None);
    }

    #[test]
    fn support_level_efficiency_emulated_none() {
        assert_eq!(SupportLevel::Emulated.efficiency(), None);
    }

    #[test]
    fn support_level_efficiency_unsupported_none() {
        assert_eq!(SupportLevel::Unsupported.efficiency(), None);
    }

    #[test]
    fn support_level_display_full() {
        assert_eq!(SupportLevel::Full(0.85).to_string(), "Full (85%)");
    }

    #[test]
    fn support_level_display_partial() {
        let s = SupportLevel::Partial("needs work".to_string());
        assert_eq!(s.to_string(), "Partial (needs work)");
    }

    #[test]
    fn support_level_display_emulated() {
        assert_eq!(SupportLevel::Emulated.to_string(), "Emulated");
    }

    #[test]
    fn support_level_display_unsupported() {
        assert_eq!(SupportLevel::Unsupported.to_string(), "Unsupported");
    }

    #[test]
    fn support_level_efficiency_in_unit_range() {
        let profile = intel_arc_a770();
        for entry in &profile.capabilities {
            if let Some(e) = entry.support.efficiency() {
                assert!(
                    (0.0..=1.0).contains(&e),
                    "efficiency {e} out of [0,1] for {} @ {}",
                    entry.operation,
                    entry.precision,
                );
            }
        }
    }

    // === CapabilityEntry tests ===

    #[test]
    fn capability_entry_new() {
        let entry = CapabilityEntry::new(
            OperationCategory::MatrixOps,
            PrecisionSupport::FP32,
            SupportLevel::Full(0.9),
        );
        assert_eq!(entry.operation, OperationCategory::MatrixOps);
        assert_eq!(entry.precision, PrecisionSupport::FP32);
        assert!(matches!(entry.support, SupportLevel::Full(e) if (e - 0.9).abs() < f64::EPSILON));
    }

    // === DeviceProfile / pre-built profiles ===

    #[test]
    fn a770_device_class() {
        let p = intel_arc_a770();
        assert_eq!(p.device_class, DeviceClass::IntelArc);
    }

    #[test]
    fn a770_name() {
        assert_eq!(intel_arc_a770().name, "Intel Arc A770");
    }

    #[test]
    fn a770_compute_units() {
        assert_eq!(intel_arc_a770().compute_units, 32);
    }

    #[test]
    fn a770_memory() {
        assert_eq!(intel_arc_a770().memory_gb, 16);
    }

    #[test]
    fn a770_fp32_matmul_supported() {
        let p = intel_arc_a770();
        assert!(p.lookup(OperationCategory::MatrixOps, PrecisionSupport::FP32).is_supported());
    }

    #[test]
    fn a770_fp16_matmul_full() {
        let p = intel_arc_a770();
        let level = p.lookup(OperationCategory::MatrixOps, PrecisionSupport::FP16);
        assert!(matches!(level, SupportLevel::Full(e) if *e >= 0.9));
    }

    #[test]
    fn a770_int8_quantized() {
        let p = intel_arc_a770();
        assert!(p.lookup(OperationCategory::QuantizedOps, PrecisionSupport::INT8).is_supported());
    }

    #[test]
    fn a770_binary_partial() {
        let p = intel_arc_a770();
        let level = p.lookup(OperationCategory::QuantizedOps, PrecisionSupport::Binary);
        assert!(matches!(level, SupportLevel::Partial(_)));
    }

    #[test]
    fn a770_unsupported_lookup_returns_unsupported() {
        let p = intel_arc_a770();
        // WebGPU precision doesn't exist — should fall through to Unsupported
        let level = p.lookup(OperationCategory::ActivationOps, PrecisionSupport::INT4);
        assert!(matches!(level, SupportLevel::Unsupported));
    }

    #[test]
    fn a770_entries_for_fp32() {
        let p = intel_arc_a770();
        let entries = p.entries_for_precision(PrecisionSupport::FP32);
        assert!(entries.len() >= 4, "A770 should have ≥4 FP32 entries");
    }

    #[test]
    fn a770_full_support_count() {
        let p = intel_arc_a770();
        assert!(p.full_support_count() >= 8, "A770 should have ≥8 full-support entries");
    }

    #[test]
    fn a770_all_efficiency_values_in_range() {
        let p = intel_arc_a770();
        for entry in &p.capabilities {
            if let Some(e) = entry.support.efficiency() {
                assert!((0.0..=1.0).contains(&e));
            }
        }
    }

    #[test]
    fn rtx4090_device_class() {
        assert_eq!(nvidia_rtx_4090().device_class, DeviceClass::NvidiaCuda);
    }

    #[test]
    fn rtx4090_memory() {
        assert_eq!(nvidia_rtx_4090().memory_gb, 24);
    }

    #[test]
    fn rtx4090_binary_full() {
        let p = nvidia_rtx_4090();
        let level = p.lookup(OperationCategory::QuantizedOps, PrecisionSupport::Binary);
        assert!(matches!(level, SupportLevel::Full(_)));
    }

    #[test]
    fn apple_m1_profile() {
        let p = apple_m1();
        assert_eq!(p.device_class, DeviceClass::AppleMetal);
        assert_eq!(p.name, "Apple M1");
    }

    #[test]
    fn apple_m1_int8_emulated() {
        let p = apple_m1();
        let level = p.lookup(OperationCategory::QuantizedOps, PrecisionSupport::INT8);
        assert!(matches!(level, SupportLevel::Emulated));
    }

    #[test]
    fn cpu_simd_profile() {
        let p = cpu_simd_avx2();
        assert_eq!(p.device_class, DeviceClass::CpuSimd);
    }

    #[test]
    fn cpu_scalar_profile() {
        let p = cpu_scalar();
        assert_eq!(p.device_class, DeviceClass::CpuScalar);
    }

    #[test]
    fn cpu_scalar_fp32_matmul_low_efficiency() {
        let p = cpu_scalar();
        let level = p.lookup(OperationCategory::MatrixOps, PrecisionSupport::FP32);
        assert!(matches!(level, SupportLevel::Full(e) if *e <= 0.5));
    }

    #[test]
    fn fallback_profile_is_cpu_scalar() {
        let p = fallback_profile();
        assert_eq!(p.device_class, DeviceClass::CpuScalar);
    }

    // === DeviceCapabilityMatrix tests ===

    #[test]
    fn matrix_new_is_empty() {
        let m = DeviceCapabilityMatrix::new();
        assert!(m.profiles().is_empty());
    }

    #[test]
    fn matrix_default_is_empty() {
        let m = DeviceCapabilityMatrix::default();
        assert!(m.profiles().is_empty());
    }

    #[test]
    fn matrix_with_builtin_profiles() {
        let m = DeviceCapabilityMatrix::with_builtin_profiles();
        assert_eq!(m.profiles().len(), 5);
    }

    #[test]
    fn matrix_add_profile() {
        let mut m = DeviceCapabilityMatrix::new();
        m.add_profile(cpu_scalar());
        assert_eq!(m.profiles().len(), 1);
    }

    #[test]
    fn matrix_profile_for_class() {
        let m = DeviceCapabilityMatrix::with_builtin_profiles();
        let p = m.profile_for_class(DeviceClass::IntelArc);
        assert!(p.is_some());
        assert_eq!(p.unwrap().name, "Intel Arc A770");
    }

    #[test]
    fn matrix_profile_for_class_missing() {
        let m = DeviceCapabilityMatrix::with_builtin_profiles();
        assert!(m.profile_for_class(DeviceClass::WebGpu).is_none());
    }

    #[test]
    fn matrix_profile_by_name() {
        let m = DeviceCapabilityMatrix::with_builtin_profiles();
        let p = m.profile_by_name("a770").unwrap();
        assert_eq!(p.device_class, DeviceClass::IntelArc);
    }

    #[test]
    fn matrix_profile_by_name_case_insensitive() {
        let m = DeviceCapabilityMatrix::with_builtin_profiles();
        assert!(m.profile_by_name("RTX 4090").is_some());
        assert!(m.profile_by_name("rtx 4090").is_some());
    }

    #[test]
    fn matrix_profile_by_name_missing() {
        let m = DeviceCapabilityMatrix::with_builtin_profiles();
        assert!(m.profile_by_name("nonexistent").is_none());
    }

    // === CapabilityQuery tests ===

    #[test]
    fn query_supports_true() {
        let p = intel_arc_a770();
        let q = CapabilityQuery::new(&p);
        assert!(q.supports(OperationCategory::MatrixOps, PrecisionSupport::FP32));
    }

    #[test]
    fn query_supports_false() {
        let p = intel_arc_a770();
        let q = CapabilityQuery::new(&p);
        assert!(!q.supports(OperationCategory::ActivationOps, PrecisionSupport::INT4));
    }

    #[test]
    fn query_support_level() {
        let p = nvidia_rtx_4090();
        let q = CapabilityQuery::new(&p);
        let level = q.support_level(OperationCategory::MatrixOps, PrecisionSupport::FP16);
        assert!(matches!(level, SupportLevel::Full(e) if *e >= 0.95));
    }

    #[test]
    fn query_best_precision_for_matmul_a770() {
        let p = intel_arc_a770();
        let q = CapabilityQuery::new(&p);
        let best = q.best_precision_for(OperationCategory::MatrixOps);
        // FP16 has 0.95 efficiency — highest for A770 matmul
        assert_eq!(best, Some(PrecisionSupport::FP16));
    }

    #[test]
    fn query_best_precision_for_matmul_rtx4090() {
        let p = nvidia_rtx_4090();
        let q = CapabilityQuery::new(&p);
        let best = q.best_precision_for(OperationCategory::MatrixOps);
        // FP16 @ 0.98 or BF16 @ 0.97 — FP16 wins
        assert_eq!(best, Some(PrecisionSupport::FP16));
    }

    #[test]
    fn query_best_precision_no_entries() {
        let p = cpu_scalar();
        let q = CapabilityQuery::new(&p);
        // CPU scalar has no FP16 entries
        let best = q.best_precision_for(OperationCategory::NormOps);
        // Only FP32 is registered, so it should be FP32
        assert_eq!(best, Some(PrecisionSupport::FP32));
    }

    #[test]
    fn query_operations_at_fp32() {
        let p = intel_arc_a770();
        let q = CapabilityQuery::new(&p);
        let ops = q.operations_at_precision(PrecisionSupport::FP32);
        assert!(ops.contains(&OperationCategory::MatrixOps));
        assert!(ops.contains(&OperationCategory::NormOps));
    }

    #[test]
    fn query_operations_at_binary_a770() {
        let p = intel_arc_a770();
        let q = CapabilityQuery::new(&p);
        let ops = q.operations_at_precision(PrecisionSupport::Binary);
        assert!(ops.contains(&OperationCategory::QuantizedOps));
    }

    // === CompatibilityReport tests ===

    #[test]
    fn report_all_supported() {
        let p = nvidia_rtx_4090();
        let required = vec![
            (OperationCategory::MatrixOps, PrecisionSupport::FP32),
            (OperationCategory::NormOps, PrecisionSupport::FP32),
        ];
        let report = CompatibilityReport::generate(&p, &required);
        assert!(report.overall_ready);
        assert_eq!(report.supported_ops.len(), 2);
        assert!(report.unsupported_ops.is_empty());
    }

    #[test]
    fn report_some_unsupported() {
        let p = cpu_scalar();
        let required = vec![
            (OperationCategory::MatrixOps, PrecisionSupport::FP32),
            (OperationCategory::MatrixOps, PrecisionSupport::FP16),
        ];
        let report = CompatibilityReport::generate(&p, &required);
        assert!(!report.overall_ready);
        assert_eq!(report.unsupported_ops.len(), 1);
    }

    #[test]
    fn report_summary_ready() {
        let p = nvidia_rtx_4090();
        let required = vec![(OperationCategory::MatrixOps, PrecisionSupport::FP32)];
        let report = CompatibilityReport::generate(&p, &required);
        let s = report.summary();
        assert!(s.contains("READY"));
        assert!(s.contains("1/1"));
    }

    #[test]
    fn report_summary_incomplete() {
        let p = cpu_scalar();
        let required = vec![
            (OperationCategory::MatrixOps, PrecisionSupport::FP32),
            (OperationCategory::AttentionOps, PrecisionSupport::FP16),
        ];
        let report = CompatibilityReport::generate(&p, &required);
        let s = report.summary();
        assert!(s.contains("INCOMPLETE"));
    }

    #[test]
    fn report_display_shows_missing() {
        let p = cpu_scalar();
        let required = vec![(OperationCategory::MatrixOps, PrecisionSupport::FP16)];
        let report = CompatibilityReport::generate(&p, &required);
        let display = format!("{report}");
        assert!(display.contains("Missing"));
        assert!(display.contains("Matrix Ops"));
        assert!(display.contains("FP16"));
    }

    #[test]
    fn report_device_name() {
        let p = intel_arc_a770();
        let report = CompatibilityReport::generate(&p, &[]);
        assert_eq!(report.device_name, "Intel Arc A770");
        assert_eq!(report.device_class, DeviceClass::IntelArc);
    }

    // === Profile comparisons ===

    #[test]
    fn a770_vs_rtx4090_binary_support() {
        let a770 = intel_arc_a770();
        let rtx = nvidia_rtx_4090();
        let a = a770.lookup(OperationCategory::QuantizedOps, PrecisionSupport::Binary);
        let b = rtx.lookup(OperationCategory::QuantizedOps, PrecisionSupport::Binary);
        // A770 is Partial, RTX 4090 is Full
        assert!(matches!(a, SupportLevel::Partial(_)));
        assert!(matches!(b, SupportLevel::Full(_)));
    }

    #[test]
    fn rtx4090_more_full_support_than_scalar() {
        let rtx = nvidia_rtx_4090();
        let scalar = cpu_scalar();
        assert!(rtx.full_support_count() > scalar.full_support_count());
    }

    #[test]
    fn a770_higher_fp16_matmul_than_cpu() {
        let a770 = intel_arc_a770();
        let cpu = cpu_simd_avx2();
        let a_eff = a770
            .lookup(OperationCategory::MatrixOps, PrecisionSupport::FP16)
            .efficiency()
            .unwrap_or(0.0);
        let c_eff = cpu
            .lookup(OperationCategory::MatrixOps, PrecisionSupport::FP16)
            .efficiency()
            .unwrap_or(0.0);
        assert!(a_eff > c_eff);
    }

    // === Unknown device fallback ===

    #[test]
    fn unknown_device_falls_back_to_scalar() {
        let m = DeviceCapabilityMatrix::with_builtin_profiles();
        // WebGpu not in built-in profiles — use fallback
        let profile =
            m.profile_for_class(DeviceClass::WebGpu).cloned().unwrap_or_else(fallback_profile);
        assert_eq!(profile.device_class, DeviceClass::CpuScalar);
    }

    // === Comprehensive coverage ===

    #[test]
    fn all_builtin_profiles_have_capabilities() {
        let m = DeviceCapabilityMatrix::with_builtin_profiles();
        for p in m.profiles() {
            assert!(!p.capabilities.is_empty(), "{} has no capabilities", p.name);
        }
    }

    #[test]
    fn all_builtin_profiles_support_fp32_matmul() {
        let m = DeviceCapabilityMatrix::with_builtin_profiles();
        for p in m.profiles() {
            assert!(
                p.lookup(OperationCategory::MatrixOps, PrecisionSupport::FP32).is_supported(),
                "{} should support FP32 matmul",
                p.name,
            );
        }
    }

    #[test]
    fn all_efficiency_values_in_range_across_profiles() {
        let m = DeviceCapabilityMatrix::with_builtin_profiles();
        for p in m.profiles() {
            for entry in &p.capabilities {
                if let Some(e) = entry.support.efficiency() {
                    assert!(
                        (0.0..=1.0).contains(&e),
                        "{}: {} @ {} efficiency {e} out of [0,1]",
                        p.name,
                        entry.operation,
                        entry.precision,
                    );
                }
            }
        }
    }
}
