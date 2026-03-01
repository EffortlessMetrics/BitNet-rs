//! GPU device capability query, validation, and selection for `OpenCL` backends.
//!
//! Provides [`GpuDeviceCapabilities`] for comprehensive hardware profiling,
//! [`DeviceCapabilityChecker`] for model-device compatibility validation, and
//! [`DeviceSelector`] for multi-GPU scoring and selection.
//!
//! All types support a mock mode via [`GpuDeviceCapabilities::mock`] so tests
//! can run without real GPU hardware.

use std::fmt;

use bitnet_common::kernel_registry::KernelBackend;

// ── GpuDeviceCapabilities ────────────────────────────────────────────────────

/// Comprehensive hardware profile for a single GPU device.
#[derive(Debug, Clone, PartialEq, Eq)]
#[allow(clippy::struct_excessive_bools)]
pub struct GpuDeviceCapabilities {
    /// Human-readable device name (e.g. "Intel Arc A770").
    pub name: String,
    /// Vendor string (e.g. "Intel", "NVIDIA", "AMD").
    pub vendor: String,
    /// Driver version string reported by the device.
    pub driver_version: String,
    /// Number of compute units (CUs / SMs / EUs).
    pub compute_units: u32,
    /// Maximum clock frequency in MHz.
    pub max_clock_mhz: u32,
    /// Total global memory in bytes.
    pub global_memory_bytes: u64,
    /// Per-workgroup local (shared) memory in bytes.
    pub local_memory_bytes: u64,
    /// Maximum work-group size (threads per group).
    pub max_work_group_size: usize,
    /// Maximum number of work-item dimensions.
    pub max_work_item_dims: u32,
    /// Maximum work-item sizes per dimension `[x, y, z]`.
    pub max_work_item_sizes: [usize; 3],
    /// Preferred vector width for `f32` operations.
    pub preferred_vector_width_float: u32,
    /// Device supports half-precision (FP16) arithmetic.
    pub supports_fp16: bool,
    /// Device supports double-precision (FP64) arithmetic.
    pub supports_fp64: bool,
    /// Device supports sub-group (warp/wavefront) operations.
    pub supports_subgroups: bool,
    /// Maximum sub-group size (0 if sub-groups unsupported).
    pub max_subgroup_size: u32,
    /// Device and host share a unified address space.
    pub supports_unified_memory: bool,
    /// Kernel backend this device is associated with.
    pub backend: KernelBackend,
}

impl GpuDeviceCapabilities {
    /// Create a mock device with reasonable defaults for testing.
    ///
    /// The mock represents a mid-range GPU with 8 GB VRAM, 32 compute
    /// units, FP16 support, and sub-group operations.
    pub fn mock() -> Self {
        Self {
            name: "Mock GPU Device".into(),
            vendor: "MockVendor".into(),
            driver_version: "1.0.0-mock".into(),
            compute_units: 32,
            max_clock_mhz: 1500,
            global_memory_bytes: 8 * 1024 * 1024 * 1024, // 8 GB
            local_memory_bytes: 64 * 1024,               // 64 KB
            max_work_group_size: 1024,
            max_work_item_dims: 3,
            max_work_item_sizes: [1024, 1024, 64],
            preferred_vector_width_float: 4,
            supports_fp16: true,
            supports_fp64: true,
            supports_subgroups: true,
            max_subgroup_size: 32,
            supports_unified_memory: false,
            backend: KernelBackend::OneApi,
        }
    }
}

/// Format a byte count into a human-readable string.
#[allow(clippy::cast_precision_loss)]
fn format_bytes(bytes: u64) -> String {
    const GB: u64 = 1024 * 1024 * 1024;
    const MB: u64 = 1024 * 1024;
    const KB: u64 = 1024;

    if bytes >= GB {
        format!("{:.2} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.2} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.2} KB", bytes as f64 / KB as f64)
    } else {
        format!("{bytes} B")
    }
}

/// Produce a human-readable summary of device capabilities.
pub fn format_device_info(caps: &GpuDeviceCapabilities) -> String {
    let mut lines = Vec::with_capacity(16);

    lines.push(format!("Device:          {}", caps.name));
    lines.push(format!("Vendor:          {}", caps.vendor));
    lines.push(format!("Driver:          {}", caps.driver_version));
    lines.push(format!("Backend:         {}", caps.backend));
    lines.push(format!("Compute units:   {}", caps.compute_units));
    lines.push(format!("Max clock:       {} MHz", caps.max_clock_mhz));
    lines.push(format!("Global memory:   {}", format_bytes(caps.global_memory_bytes)));
    lines.push(format!("Local memory:    {}", format_bytes(caps.local_memory_bytes)));
    lines.push(format!("Max workgroup:   {}", caps.max_work_group_size));
    lines.push(format!(
        "Work-item dims:  {} (max sizes: {:?})",
        caps.max_work_item_dims, caps.max_work_item_sizes
    ));
    lines.push(format!("Vec width (f32): {}", caps.preferred_vector_width_float));
    lines.push(format!("FP16:            {}", caps.supports_fp16));
    lines.push(format!("FP64:            {}", caps.supports_fp64));
    lines.push(format!(
        "Subgroups:       {} (max size: {})",
        caps.supports_subgroups, caps.max_subgroup_size
    ));
    lines.push(format!("Unified memory:  {}", caps.supports_unified_memory));

    lines.join("\n")
}

// ── ModelRequirements ────────────────────────────────────────────────────────

/// Minimum hardware requirements for running a particular model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelRequirements {
    /// Minimum global memory required in bytes.
    pub memory_bytes: u64,
    /// Model needs FP16 support.
    pub requires_fp16: bool,
    /// Model needs FP64 support.
    pub requires_fp64: bool,
    /// Model needs sub-group operations.
    pub requires_subgroups: bool,
    /// Minimum sub-group size required (0 = any).
    pub min_subgroup_size: u32,
    /// Minimum work-group size required.
    pub min_work_group_size: usize,
}

impl ModelRequirements {
    /// Minimal requirements: 1 GB memory, no special features.
    pub const fn minimal() -> Self {
        Self {
            memory_bytes: 1024 * 1024 * 1024, // 1 GB
            requires_fp16: false,
            requires_fp64: false,
            requires_subgroups: false,
            min_subgroup_size: 0,
            min_work_group_size: 64,
        }
    }
}

// ── CapabilityCheckResult ────────────────────────────────────────────────────

/// Outcome of checking a device against model requirements.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CapabilityCheckResult {
    /// Device meets all requirements.
    Compatible,
    /// Device is incompatible; contains human-readable reasons.
    Incompatible(Vec<String>),
}

impl CapabilityCheckResult {
    /// Returns `true` when the device is compatible.
    pub const fn is_compatible(&self) -> bool {
        matches!(self, Self::Compatible)
    }

    /// Returns the list of incompatibility reasons, or an empty slice.
    pub fn reasons(&self) -> &[String] {
        match self {
            Self::Compatible => &[],
            Self::Incompatible(reasons) => reasons,
        }
    }
}

// ── DeviceCapabilityChecker ──────────────────────────────────────────────────

/// Validates whether a [`GpuDeviceCapabilities`] satisfies a set of
/// [`ModelRequirements`].
pub struct DeviceCapabilityChecker;

impl DeviceCapabilityChecker {
    /// Check if `device` can run a model with the given `requirements`.
    pub fn check(
        device: &GpuDeviceCapabilities,
        requirements: &ModelRequirements,
    ) -> CapabilityCheckResult {
        let mut reasons = Vec::new();

        if device.global_memory_bytes < requirements.memory_bytes {
            reasons.push(format!(
                "insufficient memory: device has {}, model needs {}",
                format_bytes(device.global_memory_bytes),
                format_bytes(requirements.memory_bytes),
            ));
        }

        if requirements.requires_fp16 && !device.supports_fp16 {
            reasons.push("FP16 required but not supported".into());
        }

        if requirements.requires_fp64 && !device.supports_fp64 {
            reasons.push("FP64 required but not supported".into());
        }

        if requirements.requires_subgroups && !device.supports_subgroups {
            reasons.push("subgroup operations required but not supported".into());
        }

        if requirements.min_subgroup_size > 0
            && device.max_subgroup_size < requirements.min_subgroup_size
        {
            reasons.push(format!(
                "subgroup size too small: device max {}, model needs {}",
                device.max_subgroup_size, requirements.min_subgroup_size,
            ));
        }

        if device.max_work_group_size < requirements.min_work_group_size {
            reasons.push(format!(
                "work-group size too small: device max {}, model needs {}",
                device.max_work_group_size, requirements.min_work_group_size,
            ));
        }

        if reasons.is_empty() {
            CapabilityCheckResult::Compatible
        } else {
            CapabilityCheckResult::Incompatible(reasons)
        }
    }
}

// ── DeviceScore ──────────────────────────────────────────────────────────────

/// A scored device produced by [`DeviceSelector`].
#[derive(Debug, Clone, PartialEq)]
pub struct ScoredDevice {
    /// The device capabilities.
    pub device: GpuDeviceCapabilities,
    /// Composite score (higher is better).
    pub score: f64,
}

// ── DeviceSelectorError ──────────────────────────────────────────────────────

/// Errors returned by [`DeviceSelector`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DeviceSelectorError {
    /// No devices were provided.
    EmptyDeviceList,
    /// No device met the minimum requirements.
    NoCompatibleDevice(Vec<String>),
}

impl fmt::Display for DeviceSelectorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyDeviceList => {
                write!(f, "no devices provided for selection")
            }
            Self::NoCompatibleDevice(reasons) => {
                write!(f, "no compatible device found: {}", reasons.join("; "))
            }
        }
    }
}

impl std::error::Error for DeviceSelectorError {}

// ── DeviceSelector ───────────────────────────────────────────────────────────

/// Selects the best GPU from a list of candidates.
///
/// Scoring weights:
/// - Compute units: 40 %
/// - Global memory (GB): 30 %
/// - Clock speed (GHz): 10 %
/// - Feature bonuses (FP16, sub-groups, unified memory): 20 %
pub struct DeviceSelector;

impl DeviceSelector {
    /// Score a single device. Higher is better.
    #[allow(clippy::cast_precision_loss)]
    pub fn score(device: &GpuDeviceCapabilities) -> f64 {
        let cu_score = f64::from(device.compute_units) * 0.4;

        let mem_gb = device.global_memory_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
        let mem_score = mem_gb * 0.3;

        let clock_ghz = f64::from(device.max_clock_mhz) / 1000.0;
        let clock_score = clock_ghz * 0.1;

        let mut feature_score = 0.0;
        if device.supports_fp16 {
            feature_score += 0.08;
        }
        if device.supports_subgroups {
            feature_score += 0.07;
        }
        if device.supports_unified_memory {
            feature_score += 0.05;
        }

        cu_score + mem_score + clock_score + feature_score
    }

    /// Select the best device from `devices`, optionally filtering by
    /// `requirements`.
    ///
    /// Returns the highest-scored compatible device, or an error if no
    /// device qualifies.
    pub fn select(
        devices: Vec<GpuDeviceCapabilities>,
        requirements: Option<&ModelRequirements>,
    ) -> Result<ScoredDevice, DeviceSelectorError> {
        if devices.is_empty() {
            return Err(DeviceSelectorError::EmptyDeviceList);
        }

        let mut rejection_reasons: Vec<String> = Vec::new();
        let mut best: Option<ScoredDevice> = None;

        for device in devices {
            // Filter by requirements when provided.
            if let Some(reqs) = requirements {
                let result = DeviceCapabilityChecker::check(&device, reqs);
                if !result.is_compatible() {
                    rejection_reasons.push(format!(
                        "{}: {}",
                        device.name,
                        result.reasons().join(", ")
                    ));
                    continue;
                }
            }

            let score = Self::score(&device);
            let candidate = ScoredDevice { device, score };

            best = Some(match best {
                Some(current) if current.score >= candidate.score => current,
                _ => candidate,
            });
        }

        best.ok_or(DeviceSelectorError::NoCompatibleDevice(rejection_reasons))
    }

    /// Score and sort all devices (descending by score). No filtering.
    pub fn rank(devices: Vec<GpuDeviceCapabilities>) -> Vec<ScoredDevice> {
        let mut scored: Vec<ScoredDevice> = devices
            .into_iter()
            .map(|d| {
                let score = Self::score(&d);
                ScoredDevice { device: d, score }
            })
            .collect();

        scored.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));

        scored
    }
}
