//! Dynamic kernel dispatch by GPU device capabilities.
//!
//! Provides [`KernelDispatcher`] which selects the optimal kernel variant at
//! runtime based on device capability queries (subgroup size, local memory,
//! max work-group size). Kernel variants are registered in a
//! [`KernelVariantRegistry`] and auto-selected via heuristic scoring.

use std::collections::HashMap;
use std::fmt;

// ---------------------------------------------------------------------------
// Device capability model
// ---------------------------------------------------------------------------

/// GPU device capabilities queried at runtime.
#[derive(Debug, Clone, PartialEq)]
pub struct GpuDeviceCapabilities {
    /// Preferred subgroup (wavefront / warp) size.
    pub subgroup_size: u32,
    /// Available local (shared) memory in bytes.
    pub local_memory_bytes: u64,
    /// Maximum work-group size (product of all dimensions).
    pub max_work_group_size: u32,
    /// Maximum number of compute units.
    pub max_compute_units: u32,
    /// Device vendor hint used for heuristic tie-breaking.
    pub vendor: GpuVendor,
    /// Whether the device supports sub-group shuffle operations.
    pub supports_subgroup_shuffle: bool,
}

impl GpuDeviceCapabilities {
    /// Create capabilities for a generic device with sensible defaults.
    pub fn generic() -> Self {
        Self {
            subgroup_size: 32,
            local_memory_bytes: 16_384,
            max_work_group_size: 256,
            max_compute_units: 8,
            vendor: GpuVendor::Unknown,
            supports_subgroup_shuffle: false,
        }
    }

    /// Create capabilities typical of an Intel Arc GPU.
    pub fn intel_arc() -> Self {
        Self {
            subgroup_size: 16,
            local_memory_bytes: 65_536,
            max_work_group_size: 1024,
            max_compute_units: 512,
            vendor: GpuVendor::Intel,
            supports_subgroup_shuffle: true,
        }
    }

    /// Create capabilities typical of an NVIDIA GPU.
    pub fn nvidia() -> Self {
        Self {
            subgroup_size: 32,
            local_memory_bytes: 49_152,
            max_work_group_size: 1024,
            max_compute_units: 128,
            vendor: GpuVendor::Nvidia,
            supports_subgroup_shuffle: true,
        }
    }
}

/// Known GPU vendor families for heuristic dispatch.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GpuVendor {
    Intel,
    Nvidia,
    Amd,
    Unknown,
}

impl fmt::Display for GpuVendor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GpuVendor::Intel => write!(f, "Intel"),
            GpuVendor::Nvidia => write!(f, "NVIDIA"),
            GpuVendor::Amd => write!(f, "AMD"),
            GpuVendor::Unknown => write!(f, "Unknown"),
        }
    }
}

// ---------------------------------------------------------------------------
// Kernel variant model
// ---------------------------------------------------------------------------

/// The category of a kernel implementation variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum KernelVariantKind {
    /// Simple scalar per-element kernel (always available).
    Scalar,
    /// Vectorized kernel using vector load/store (vec4/vec8).
    Vectorized,
    /// Sub-group (warp/wavefront) cooperative kernel.
    Subgroup,
    /// Tiled kernel using local memory for data reuse.
    Tiled,
}

impl fmt::Display for KernelVariantKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            KernelVariantKind::Scalar => write!(f, "scalar"),
            KernelVariantKind::Vectorized => write!(f, "vectorized"),
            KernelVariantKind::Subgroup => write!(f, "subgroup"),
            KernelVariantKind::Tiled => write!(f, "tiled"),
        }
    }
}

/// Requirements a kernel variant imposes on the device.
#[derive(Debug, Clone, PartialEq)]
pub struct KernelRequirements {
    /// Minimum subgroup size (0 = no requirement).
    pub min_subgroup_size: u32,
    /// Minimum local memory in bytes (0 = no requirement).
    pub min_local_memory_bytes: u64,
    /// Minimum work-group size (0 = no requirement).
    pub min_work_group_size: u32,
    /// Whether sub-group shuffle is required.
    pub requires_subgroup_shuffle: bool,
}

impl KernelRequirements {
    /// Requirements that any device satisfies.
    pub fn none() -> Self {
        Self {
            min_subgroup_size: 0,
            min_local_memory_bytes: 0,
            min_work_group_size: 0,
            requires_subgroup_shuffle: false,
        }
    }
}

/// A registered kernel variant with metadata for dispatch scoring.
#[derive(Debug, Clone)]
pub struct KernelVariant {
    /// Unique name, e.g. `"matmul_i2s_tiled_16x16"`.
    pub name: String,
    /// The broad category.
    pub kind: KernelVariantKind,
    /// Hardware requirements.
    pub requirements: KernelRequirements,
    /// Static priority boost (higher = preferred when requirements are met).
    pub priority: u32,
    /// Per-vendor priority overrides (additive).
    pub vendor_boosts: HashMap<GpuVendor, i32>,
}

impl KernelVariant {
    /// Create a new variant with the given name and kind.
    pub fn new(name: impl Into<String>, kind: KernelVariantKind) -> Self {
        Self {
            name: name.into(),
            kind,
            requirements: KernelRequirements::none(),
            priority: 0,
            vendor_boosts: HashMap::new(),
        }
    }

    /// Set hardware requirements.
    #[must_use]
    pub fn with_requirements(mut self, reqs: KernelRequirements) -> Self {
        self.requirements = reqs;
        self
    }

    /// Set base priority.
    #[must_use]
    pub fn with_priority(mut self, p: u32) -> Self {
        self.priority = p;
        self
    }

    /// Add a vendor-specific priority boost.
    #[must_use]
    pub fn with_vendor_boost(mut self, vendor: GpuVendor, boost: i32) -> Self {
        self.vendor_boosts.insert(vendor, boost);
        self
    }

    /// Check whether `caps` satisfies this variant's requirements.
    pub fn is_compatible(&self, caps: &GpuDeviceCapabilities) -> bool {
        let r = &self.requirements;
        caps.subgroup_size >= r.min_subgroup_size
            && caps.local_memory_bytes >= r.min_local_memory_bytes
            && caps.max_work_group_size >= r.min_work_group_size
            && (!r.requires_subgroup_shuffle || caps.supports_subgroup_shuffle)
    }

    /// Compute a heuristic score for a given device (higher = better).
    pub fn score(&self, caps: &GpuDeviceCapabilities) -> i64 {
        if !self.is_compatible(caps) {
            return i64::MIN;
        }
        let mut s = self.priority as i64;
        if let Some(&boost) = self.vendor_boosts.get(&caps.vendor) {
            s += boost as i64;
        }
        s += match self.kind {
            KernelVariantKind::Tiled => 300,
            KernelVariantKind::Subgroup => 200,
            KernelVariantKind::Vectorized => 100,
            KernelVariantKind::Scalar => 0,
        };
        s
    }
}

// ---------------------------------------------------------------------------
// Registry
// ---------------------------------------------------------------------------

/// Registry of kernel variants for a single operation (e.g. `matmul_i2s`).
#[derive(Debug, Clone, Default)]
pub struct KernelVariantRegistry {
    variants: Vec<KernelVariant>,
}

impl KernelVariantRegistry {
    pub fn new() -> Self {
        Self { variants: Vec::new() }
    }

    pub fn register(&mut self, variant: KernelVariant) {
        self.variants.push(variant);
    }

    pub fn variants(&self) -> &[KernelVariant] {
        &self.variants
    }

    pub fn len(&self) -> usize {
        self.variants.len()
    }

    pub fn is_empty(&self) -> bool {
        self.variants.is_empty()
    }

    /// Return all variants compatible with `caps`, sorted best-first.
    pub fn compatible_variants(
        &self,
        caps: &GpuDeviceCapabilities,
    ) -> Vec<&KernelVariant> {
        let mut compat: Vec<_> =
            self.variants.iter().filter(|v| v.is_compatible(caps)).collect();
        compat.sort_by(|a, b| b.score(caps).cmp(&a.score(caps)));
        compat
    }

    /// Select the single best variant for `caps`.
    pub fn best_variant(
        &self,
        caps: &GpuDeviceCapabilities,
    ) -> Option<&KernelVariant> {
        self.compatible_variants(caps).into_iter().next()
    }
}

// ---------------------------------------------------------------------------
// Dispatcher
// ---------------------------------------------------------------------------

/// Result of a dispatch decision.
#[derive(Debug, Clone)]
pub struct DispatchDecision {
    pub variant_name: String,
    pub kind: KernelVariantKind,
    pub score: i64,
}

/// Top-level dispatcher mapping operation names to registries.
#[derive(Debug, Clone, Default)]
pub struct KernelDispatcher {
    registries: HashMap<String, KernelVariantRegistry>,
    cached_caps: Option<GpuDeviceCapabilities>,
}

impl KernelDispatcher {
    pub fn new() -> Self {
        Self { registries: HashMap::new(), cached_caps: None }
    }

    pub fn with_capabilities(caps: GpuDeviceCapabilities) -> Self {
        Self { registries: HashMap::new(), cached_caps: Some(caps) }
    }

    pub fn set_capabilities(&mut self, caps: GpuDeviceCapabilities) {
        self.cached_caps = Some(caps);
    }

    pub fn capabilities(&self) -> Option<&GpuDeviceCapabilities> {
        self.cached_caps.as_ref()
    }

    pub fn register(
        &mut self,
        operation: impl Into<String>,
        variant: KernelVariant,
    ) {
        self.registries.entry(operation.into()).or_default().register(variant);
    }

    pub fn registry(&self, operation: &str) -> Option<&KernelVariantRegistry> {
        self.registries.get(operation)
    }

    pub fn operations(&self) -> Vec<&str> {
        self.registries.keys().map(|s| s.as_str()).collect()
    }

    pub fn dispatch(
        &self,
        operation: &str,
        caps: &GpuDeviceCapabilities,
    ) -> Option<DispatchDecision> {
        let registry = self.registries.get(operation)?;
        let best = registry.best_variant(caps)?;
        Some(DispatchDecision {
            variant_name: best.name.clone(),
            kind: best.kind,
            score: best.score(caps),
        })
    }

    pub fn dispatch_cached(
        &self,
        operation: &str,
    ) -> Option<DispatchDecision> {
        let caps = self.cached_caps.as_ref()?;
        self.dispatch(operation, caps)
    }

    pub fn dispatch_all(
        &self,
        caps: &GpuDeviceCapabilities,
    ) -> HashMap<String, DispatchDecision> {
        self.registries
            .keys()
            .filter_map(|op| self.dispatch(op, caps).map(|d| (op.clone(), d)))
            .collect()
    }
}

/// Pre-populate a dispatcher with standard BitNet kernel variants.
pub fn build_default_dispatcher() -> KernelDispatcher {
    let mut d = KernelDispatcher::new();

    // matmul_i2s
    d.register(
        "matmul_i2s",
        KernelVariant::new("matmul_i2s_scalar", KernelVariantKind::Scalar),
    );
    d.register(
        "matmul_i2s",
        KernelVariant::new("matmul_i2s_vec4", KernelVariantKind::Vectorized)
            .with_priority(10)
            .with_requirements(KernelRequirements {
                min_subgroup_size: 0,
                min_local_memory_bytes: 0,
                min_work_group_size: 64,
                requires_subgroup_shuffle: false,
            }),
    );
    d.register(
        "matmul_i2s",
        KernelVariant::new("matmul_i2s_subgroup", KernelVariantKind::Subgroup)
            .with_priority(20)
            .with_vendor_boost(GpuVendor::Intel, 50)
            .with_requirements(KernelRequirements {
                min_subgroup_size: 8,
                min_local_memory_bytes: 0,
                min_work_group_size: 64,
                requires_subgroup_shuffle: true,
            }),
    );
    d.register(
        "matmul_i2s",
        KernelVariant::new("matmul_i2s_tiled_16x16", KernelVariantKind::Tiled)
            .with_priority(30)
            .with_requirements(KernelRequirements {
                min_subgroup_size: 0,
                min_local_memory_bytes: 32_768,
                min_work_group_size: 256,
                requires_subgroup_shuffle: false,
            }),
    );

    // elementwise
    d.register(
        "elementwise",
        KernelVariant::new("elementwise_scalar", KernelVariantKind::Scalar),
    );
    d.register(
        "elementwise",
        KernelVariant::new("elementwise_vec4", KernelVariantKind::Vectorized)
            .with_priority(10),
    );

    // quantize_i2s
    d.register(
        "quantize_i2s",
        KernelVariant::new("quantize_i2s_scalar", KernelVariantKind::Scalar),
    );
    d.register(
        "quantize_i2s",
        KernelVariant::new("quantize_i2s_subgroup", KernelVariantKind::Subgroup)
            .with_priority(20)
            .with_vendor_boost(GpuVendor::Intel, 40)
            .with_requirements(KernelRequirements {
                min_subgroup_size: 8,
                min_local_memory_bytes: 0,
                min_work_group_size: 64,
                requires_subgroup_shuffle: true,
            }),
    );

    d
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_registry() -> KernelVariantRegistry {
        let mut reg = KernelVariantRegistry::new();
        reg.register(KernelVariant::new("scalar", KernelVariantKind::Scalar));
        reg.register(
            KernelVariant::new("vec4", KernelVariantKind::Vectorized)
                .with_priority(10)
                .with_requirements(KernelRequirements {
                    min_subgroup_size: 0,
                    min_local_memory_bytes: 0,
                    min_work_group_size: 64,
                    requires_subgroup_shuffle: false,
                }),
        );
        reg.register(
            KernelVariant::new("subgroup", KernelVariantKind::Subgroup)
                .with_priority(20)
                .with_vendor_boost(GpuVendor::Intel, 50)
                .with_requirements(KernelRequirements {
                    min_subgroup_size: 8,
                    min_local_memory_bytes: 0,
                    min_work_group_size: 64,
                    requires_subgroup_shuffle: true,
                }),
        );
        reg.register(
            KernelVariant::new("tiled", KernelVariantKind::Tiled)
                .with_priority(30)
                .with_requirements(KernelRequirements {
                    min_subgroup_size: 0,
                    min_local_memory_bytes: 32_768,
                    min_work_group_size: 256,
                    requires_subgroup_shuffle: false,
                }),
        );
        reg
    }

    #[test]
    fn scalar_always_compatible_with_generic_device() {
        let caps = GpuDeviceCapabilities::generic();
        let v = KernelVariant::new("s", KernelVariantKind::Scalar);
        assert!(v.is_compatible(&caps));
    }

    #[test]
    fn subgroup_rejected_without_shuffle_support() {
        let mut caps = GpuDeviceCapabilities::generic();
        caps.supports_subgroup_shuffle = false;
        let v = KernelVariant::new("sg", KernelVariantKind::Subgroup)
            .with_requirements(KernelRequirements {
                min_subgroup_size: 8,
                min_local_memory_bytes: 0,
                min_work_group_size: 64,
                requires_subgroup_shuffle: true,
            });
        assert!(!v.is_compatible(&caps));
    }

    #[test]
    fn tiled_rejected_with_insufficient_local_memory() {
        let mut caps = GpuDeviceCapabilities::generic();
        caps.local_memory_bytes = 1024;
        let v = KernelVariant::new("t", KernelVariantKind::Tiled)
            .with_requirements(KernelRequirements {
                min_subgroup_size: 0,
                min_local_memory_bytes: 32_768,
                min_work_group_size: 256,
                requires_subgroup_shuffle: false,
            });
        assert!(!v.is_compatible(&caps));
    }

    #[test]
    fn intel_arc_selects_tiled_when_capable() {
        let caps = GpuDeviceCapabilities::intel_arc();
        let reg = sample_registry();
        let best = reg.best_variant(&caps).unwrap();
        assert_eq!(best.name, "tiled");
    }

    #[test]
    fn empty_registry_returns_none() {
        let reg = KernelVariantRegistry::new();
        assert!(reg.best_variant(&GpuDeviceCapabilities::generic()).is_none());
        assert!(reg.is_empty());
    }

    #[test]
    fn dispatcher_dispatches_known_operation() {
        let mut d = KernelDispatcher::new();
        d.register(
            "matmul",
            KernelVariant::new("matmul_scalar", KernelVariantKind::Scalar),
        );
        let caps = GpuDeviceCapabilities::generic();
        let decision = d.dispatch("matmul", &caps).unwrap();
        assert_eq!(decision.variant_name, "matmul_scalar");
    }

    #[test]
    fn dispatcher_returns_none_for_unknown_op() {
        let d = KernelDispatcher::new();
        assert!(d
            .dispatch("nonexistent", &GpuDeviceCapabilities::generic())
            .is_none());
    }

    #[test]
    fn dispatch_cached_uses_stored_capabilities() {
        let mut d =
            KernelDispatcher::with_capabilities(GpuDeviceCapabilities::nvidia());
        d.register(
            "relu",
            KernelVariant::new("relu_scalar", KernelVariantKind::Scalar),
        );
        let dec = d.dispatch_cached("relu").unwrap();
        assert_eq!(dec.variant_name, "relu_scalar");
    }

    #[test]
    fn dispatch_all_returns_full_table() {
        let d = build_default_dispatcher();
        let caps = GpuDeviceCapabilities::nvidia();
        let table = d.dispatch_all(&caps);
        assert!(table.contains_key("matmul_i2s"));
        assert!(table.contains_key("elementwise"));
        assert!(table.contains_key("quantize_i2s"));
    }

    #[test]
    fn vendor_boost_changes_winner() {
        let mut reg = KernelVariantRegistry::new();
        reg.register(
            KernelVariant::new("generic_vec", KernelVariantKind::Vectorized)
                .with_priority(50),
        );
        reg.register(
            KernelVariant::new("intel_sg", KernelVariantKind::Vectorized)
                .with_priority(10)
                .with_vendor_boost(GpuVendor::Intel, 100),
        );

        let generic = GpuDeviceCapabilities::generic();
        assert_eq!(
            reg.best_variant(&generic).unwrap().name,
            "generic_vec"
        );

        let intel = GpuDeviceCapabilities::intel_arc();
        assert_eq!(reg.best_variant(&intel).unwrap().name, "intel_sg");
    }

    #[test]
    fn compatible_variants_sorted_descending() {
        let reg = sample_registry();
        let caps = GpuDeviceCapabilities::nvidia();
        let compat = reg.compatible_variants(&caps);
        let scores: Vec<i64> = compat.iter().map(|v| v.score(&caps)).collect();
        for w in scores.windows(2) {
            assert!(w[0] >= w[1], "not sorted: {} < {}", w[0], w[1]);
        }
    }

    #[test]
    fn variant_kind_display() {
        assert_eq!(KernelVariantKind::Scalar.to_string(), "scalar");
        assert_eq!(KernelVariantKind::Vectorized.to_string(), "vectorized");
        assert_eq!(KernelVariantKind::Subgroup.to_string(), "subgroup");
        assert_eq!(KernelVariantKind::Tiled.to_string(), "tiled");
    }

    #[test]
    fn default_dispatcher_has_standard_operations() {
        let d = build_default_dispatcher();
        let ops = d.operations();
        assert!(ops.contains(&"matmul_i2s"));
        assert!(ops.contains(&"elementwise"));
        assert!(ops.contains(&"quantize_i2s"));
    }

    #[test]
    fn gpu_vendor_display() {
        assert_eq!(GpuVendor::Intel.to_string(), "Intel");
        assert_eq!(GpuVendor::Nvidia.to_string(), "NVIDIA");
        assert_eq!(GpuVendor::Amd.to_string(), "AMD");
        assert_eq!(GpuVendor::Unknown.to_string(), "Unknown");
    }

    #[test]
    fn work_group_size_requirement_filters() {
        let mut caps = GpuDeviceCapabilities::generic();
        caps.max_work_group_size = 32;
        let v = KernelVariant::new("big", KernelVariantKind::Vectorized)
            .with_requirements(KernelRequirements {
                min_subgroup_size: 0,
                min_local_memory_bytes: 0,
                min_work_group_size: 256,
                requires_subgroup_shuffle: false,
            });
        assert!(!v.is_compatible(&caps));
    }
}
