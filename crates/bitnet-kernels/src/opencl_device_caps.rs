//! OpenCL device capability detection and feature gating for safe A770 usage.
//!
//! Provides comprehensive device profiling, extension detection, capability
//! gates, and a device selector for multi-platform environments. Compiles
//! unconditionally — no feature gates required.

use std::collections::BTreeSet;
use std::fmt;

// ---------------------------------------------------------------------------
// PlatformInfo
// ---------------------------------------------------------------------------

/// OpenCL platform details (vendor, version, profile).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlatformInfo {
    /// Platform name (e.g. "Intel(R) OpenCL HD Graphics").
    pub name: String,
    /// Vendor string.
    pub vendor: String,
    /// OpenCL version string (e.g. "OpenCL 3.0").
    pub version: String,
    /// Profile — `"FULL_PROFILE"` or `"EMBEDDED_PROFILE"`.
    pub profile: String,
}

impl PlatformInfo {
    /// Whether this is a full (non-embedded) profile.
    pub fn is_full_profile(&self) -> bool {
        self.profile == "FULL_PROFILE"
    }

    /// Parse major.minor from the version string.
    ///
    /// Expects a prefix like `"OpenCL 3.0 ..."` and returns `(3, 0)`.
    /// Returns `None` if the format is unrecognised.
    pub fn version_tuple(&self) -> Option<(u32, u32)> {
        let after = self.version.strip_prefix("OpenCL ")?;
        let dotted = after.split_whitespace().next()?;
        let mut parts = dotted.split('.');
        let major = parts.next()?.parse().ok()?;
        let minor = parts.next()?.parse().ok()?;
        Some((major, minor))
    }
}

impl fmt::Display for PlatformInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} ({}, {})", self.name, self.vendor, self.version)
    }
}

// ---------------------------------------------------------------------------
// MemoryInfo
// ---------------------------------------------------------------------------

/// Device memory characteristics.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemoryInfo {
    /// Total global memory in bytes.
    pub global_mem_bytes: u64,
    /// Total local (shared) memory in bytes.
    pub local_mem_bytes: u64,
    /// Maximum single-buffer allocation in bytes.
    pub max_alloc_bytes: u64,
    /// Global memory cache size in bytes (0 if none).
    pub cache_size_bytes: u64,
    /// Cache line size in bytes (0 if unknown).
    pub cache_line_bytes: u32,
}

impl MemoryInfo {
    /// Global memory in MiB.
    pub fn global_mem_mib(&self) -> f64 {
        self.global_mem_bytes as f64 / (1024.0 * 1024.0)
    }

    /// Global memory in GiB.
    pub fn global_mem_gib(&self) -> f64 {
        self.global_mem_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
    }

    /// Whether the device has at least `bytes` of global memory.
    pub fn has_global_mem(&self, bytes: u64) -> bool {
        self.global_mem_bytes >= bytes
    }

    /// Whether the device has at least `bytes` of local memory.
    pub fn has_local_mem(&self, bytes: u64) -> bool {
        self.local_mem_bytes >= bytes
    }
}

impl fmt::Display for MemoryInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "global={:.1} GiB, local={} KiB, max_alloc={:.1} GiB, \
             cache={} KiB, cache_line={} B",
            self.global_mem_gib(),
            self.local_mem_bytes / 1024,
            self.max_alloc_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
            self.cache_size_bytes / 1024,
            self.cache_line_bytes,
        )
    }
}

// ---------------------------------------------------------------------------
// ComputeInfo
// ---------------------------------------------------------------------------

/// Device compute characteristics.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ComputeInfo {
    /// Maximum work-group size.
    pub max_workgroup_size: usize,
    /// Maximum work-item dimensions (typically 3).
    pub max_workitem_dims: u32,
    /// Supported subgroup sizes (e.g. `[8, 16, 32]`).
    pub subgroup_sizes: Vec<u32>,
    /// Whether FP16 (half) is natively supported.
    pub fp16: bool,
    /// Whether FP64 (double) is natively supported.
    pub fp64: bool,
    /// Whether INT8 dot-product acceleration is available.
    pub int8_dot: bool,
    /// Number of compute units (Xe-cores on Intel Arc).
    pub compute_units: u32,
    /// Maximum clock frequency in MHz.
    pub max_clock_mhz: u32,
}

impl ComputeInfo {
    /// Whether the device supports a specific subgroup size.
    pub fn has_subgroup_size(&self, size: u32) -> bool {
        self.subgroup_sizes.contains(&size)
    }

    /// Preferred subgroup size (largest available, or 0).
    pub fn preferred_subgroup_size(&self) -> u32 {
        self.subgroup_sizes.iter().copied().max().unwrap_or(0)
    }
}

impl fmt::Display for ComputeInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "CUs={}, max_wg={}, subgroups={:?}, \
             fp16={}, fp64={}, int8_dot={}",
            self.compute_units,
            self.max_workgroup_size,
            self.subgroup_sizes,
            self.fp16,
            self.fp64,
            self.int8_dot,
        )
    }
}

// ---------------------------------------------------------------------------
// ExtensionSet
// ---------------------------------------------------------------------------

/// Detected OpenCL extensions stored as a sorted set.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExtensionSet {
    extensions: BTreeSet<String>,
}

impl ExtensionSet {
    /// Create an empty extension set.
    pub fn new() -> Self {
        Self { extensions: BTreeSet::new() }
    }

    /// Create from a space-separated extension string (OpenCL convention).
    pub fn from_extension_string(ext_str: &str) -> Self {
        let extensions = ext_str.split_whitespace().map(String::from).collect();
        Self { extensions }
    }

    /// Create from an iterator of extension names.
    pub fn from_names(iter: impl IntoIterator<Item = impl Into<String>>) -> Self {
        let extensions = iter.into_iter().map(Into::into).collect();
        Self { extensions }
    }

    /// Insert an extension.
    pub fn insert(&mut self, ext: impl Into<String>) {
        self.extensions.insert(ext.into());
    }

    /// Whether the set contains a specific extension.
    pub fn contains(&self, ext: &str) -> bool {
        self.extensions.contains(ext)
    }

    /// Number of extensions.
    pub fn len(&self) -> usize {
        self.extensions.len()
    }

    /// Whether there are no extensions.
    pub fn is_empty(&self) -> bool {
        self.extensions.is_empty()
    }

    /// Iterate over extensions in sorted order.
    pub fn iter(&self) -> impl Iterator<Item = &str> {
        self.extensions.iter().map(String::as_str)
    }

    // ── Well-known extension queries ────────────────────────────────

    pub fn has_khr_subgroups(&self) -> bool {
        self.contains("cl_khr_subgroups")
    }

    pub fn has_intel_subgroups(&self) -> bool {
        self.contains("cl_intel_subgroups")
    }

    pub fn has_intel_subgroups_short(&self) -> bool {
        self.contains("cl_intel_subgroups_short")
    }

    pub fn has_khr_fp16(&self) -> bool {
        self.contains("cl_khr_fp16")
    }

    pub fn has_khr_fp64(&self) -> bool {
        self.contains("cl_khr_fp64")
    }

    pub fn has_intel_unified_shared_memory(&self) -> bool {
        self.contains("cl_intel_unified_shared_memory")
    }
}

impl Default for ExtensionSet {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for ExtensionSet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let list: Vec<&str> = self.iter().collect();
        write!(f, "[{}]", list.join(", "))
    }
}

// ---------------------------------------------------------------------------
// DeviceCapabilities
// ---------------------------------------------------------------------------

/// Full device profile combining platform, memory, compute, and extensions.
#[derive(Debug, Clone)]
pub struct DeviceCapabilities {
    /// Human-readable device name.
    pub device_name: String,
    /// Vendor string.
    pub vendor: String,
    /// Platform information.
    pub platform: PlatformInfo,
    /// Memory characteristics.
    pub memory: MemoryInfo,
    /// Compute characteristics.
    pub compute: ComputeInfo,
    /// Detected extensions.
    pub extensions: ExtensionSet,
}

impl fmt::Display for DeviceCapabilities {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Device: {} ({})", self.device_name, self.vendor)?;
        writeln!(f, "Platform: {}", self.platform)?;
        writeln!(f, "Memory: {}", self.memory)?;
        writeln!(f, "Compute: {}", self.compute)?;
        write!(f, "Extensions: {}", self.extensions)
    }
}

// ---------------------------------------------------------------------------
// A770Profile — hard-coded known-good values
// ---------------------------------------------------------------------------

/// Hard-coded known-good capability profile for the Intel Arc A770.
///
/// Used for validation and testing against real hardware queries.
pub struct A770Profile;

impl A770Profile {
    /// Expected compute units (512 EUs = 32 Xe-cores × 16 EUs each,
    /// but OpenCL reports 512 EUs).
    pub const COMPUTE_UNITS: u32 = 512;

    /// Expected global memory: 16 GiB VRAM.
    pub const GLOBAL_MEM_BYTES: u64 = 16 * 1024 * 1024 * 1024;

    /// Expected shared local memory: 64 KiB per work-group.
    pub const LOCAL_MEM_BYTES: u64 = 64 * 1024;

    /// Maximum single-buffer allocation (typically ¼ of VRAM).
    pub const MAX_ALLOC_BYTES: u64 = 4 * 1024 * 1024 * 1024;

    /// Maximum work-group size.
    pub const MAX_WORKGROUP_SIZE: usize = 1024;

    /// Supported subgroup sizes on Xe-HPG.
    pub const SUBGROUP_SIZES: &[u32] = &[8, 16, 32];

    /// Maximum clock frequency (MHz) — reference value.
    pub const MAX_CLOCK_MHZ: u32 = 2100;

    /// Global memory cache size (2 MiB L2).
    pub const CACHE_SIZE_BYTES: u64 = 2 * 1024 * 1024;

    /// Cache line size (64 bytes).
    pub const CACHE_LINE_BYTES: u32 = 64;

    /// Required extensions for full BitNet acceleration.
    pub const REQUIRED_EXTENSIONS: &[&str] = &[
        "cl_khr_subgroups",
        "cl_intel_subgroups",
        "cl_intel_subgroups_short",
        "cl_khr_fp16",
        "cl_intel_unified_shared_memory",
    ];

    /// Build a [`DeviceCapabilities`] from the known A770 profile.
    pub fn capabilities() -> DeviceCapabilities {
        let extensions = ExtensionSet::from_names(Self::REQUIRED_EXTENSIONS.iter().copied());

        DeviceCapabilities {
            device_name: "Intel(R) Arc(TM) A770 Graphics".into(),
            vendor: "Intel(R) Corporation".into(),
            platform: PlatformInfo {
                name: "Intel(R) OpenCL Graphics".into(),
                vendor: "Intel(R) Corporation".into(),
                version: "OpenCL 3.0".into(),
                profile: "FULL_PROFILE".into(),
            },
            memory: MemoryInfo {
                global_mem_bytes: Self::GLOBAL_MEM_BYTES,
                local_mem_bytes: Self::LOCAL_MEM_BYTES,
                max_alloc_bytes: Self::MAX_ALLOC_BYTES,
                cache_size_bytes: Self::CACHE_SIZE_BYTES,
                cache_line_bytes: Self::CACHE_LINE_BYTES,
            },
            compute: ComputeInfo {
                max_workgroup_size: Self::MAX_WORKGROUP_SIZE,
                max_workitem_dims: 3,
                subgroup_sizes: Self::SUBGROUP_SIZES.to_vec(),
                fp16: true,
                fp64: false,
                int8_dot: true,
                compute_units: Self::COMPUTE_UNITS,
                max_clock_mhz: Self::MAX_CLOCK_MHZ,
            },
            extensions,
        }
    }
}

// ---------------------------------------------------------------------------
// CapabilityGate
// ---------------------------------------------------------------------------

/// Feature gate that checks device capabilities against requirements.
#[derive(Debug, Clone)]
pub struct CapabilityGate {
    /// Human-readable gate name.
    pub name: String,
    checks: Vec<GateCheck>,
}

#[derive(Debug, Clone)]
enum GateCheck {
    MinGlobalMem(u64),
    MinLocalMem(u64),
    MinComputeUnits(u32),
    RequiresFp16,
    RequiresFp64,
    RequiresInt8Dot,
    RequiresSubgroupSize(u32),
    RequiresExtension(String),
    MinWorkgroupSize(usize),
}

/// Result of evaluating a [`CapabilityGate`] against a device.
#[derive(Debug, Clone)]
pub struct GateResult {
    pub gate_name: String,
    pub passed: bool,
    pub failures: Vec<String>,
}

impl fmt::Display for GateResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.passed {
            write!(f, "PASS: {}", self.gate_name)
        } else {
            write!(f, "FAIL: {} — {}", self.gate_name, self.failures.join("; "))
        }
    }
}

impl CapabilityGate {
    /// Create a new gate with the given name.
    pub fn new(name: impl Into<String>) -> Self {
        Self { name: name.into(), checks: Vec::new() }
    }

    /// Require at least `bytes` of global memory.
    pub fn min_global_mem(mut self, bytes: u64) -> Self {
        self.checks.push(GateCheck::MinGlobalMem(bytes));
        self
    }

    /// Require at least `bytes` of local (shared) memory.
    pub fn min_local_mem(mut self, bytes: u64) -> Self {
        self.checks.push(GateCheck::MinLocalMem(bytes));
        self
    }

    /// Require at least `n` compute units.
    pub fn min_compute_units(mut self, n: u32) -> Self {
        self.checks.push(GateCheck::MinComputeUnits(n));
        self
    }

    /// Require FP16 support.
    pub fn requires_fp16(mut self) -> Self {
        self.checks.push(GateCheck::RequiresFp16);
        self
    }

    /// Require FP64 support.
    pub fn requires_fp64(mut self) -> Self {
        self.checks.push(GateCheck::RequiresFp64);
        self
    }

    /// Require INT8 dot-product acceleration.
    pub fn requires_int8_dot(mut self) -> Self {
        self.checks.push(GateCheck::RequiresInt8Dot);
        self
    }

    /// Require a specific subgroup size.
    pub fn requires_subgroup_size(mut self, size: u32) -> Self {
        self.checks.push(GateCheck::RequiresSubgroupSize(size));
        self
    }

    /// Require a specific OpenCL extension.
    pub fn requires_extension(mut self, ext: impl Into<String>) -> Self {
        self.checks.push(GateCheck::RequiresExtension(ext.into()));
        self
    }

    /// Require minimum work-group size.
    pub fn min_workgroup_size(mut self, size: usize) -> Self {
        self.checks.push(GateCheck::MinWorkgroupSize(size));
        self
    }

    /// Evaluate this gate against a device's capabilities.
    pub fn evaluate(&self, caps: &DeviceCapabilities) -> GateResult {
        let mut failures = Vec::new();

        for check in &self.checks {
            match check {
                GateCheck::MinGlobalMem(req) => {
                    if caps.memory.global_mem_bytes < *req {
                        failures.push(format!(
                            "global_mem: need {} B, have {} B",
                            req, caps.memory.global_mem_bytes
                        ));
                    }
                }
                GateCheck::MinLocalMem(req) => {
                    if caps.memory.local_mem_bytes < *req {
                        failures.push(format!(
                            "local_mem: need {} B, have {} B",
                            req, caps.memory.local_mem_bytes
                        ));
                    }
                }
                GateCheck::MinComputeUnits(req) => {
                    if caps.compute.compute_units < *req {
                        failures.push(format!(
                            "compute_units: need {}, have {}",
                            req, caps.compute.compute_units
                        ));
                    }
                }
                GateCheck::RequiresFp16 => {
                    if !caps.compute.fp16 {
                        failures.push("fp16 not supported".into());
                    }
                }
                GateCheck::RequiresFp64 => {
                    if !caps.compute.fp64 {
                        failures.push("fp64 not supported".into());
                    }
                }
                GateCheck::RequiresInt8Dot => {
                    if !caps.compute.int8_dot {
                        failures.push("int8_dot not supported".into());
                    }
                }
                GateCheck::RequiresSubgroupSize(size) => {
                    if !caps.compute.has_subgroup_size(*size) {
                        failures.push(format!(
                            "subgroup size {} not in {:?}",
                            size, caps.compute.subgroup_sizes
                        ));
                    }
                }
                GateCheck::RequiresExtension(ext) => {
                    if !caps.extensions.contains(ext) {
                        failures.push(format!("extension {ext} missing"));
                    }
                }
                GateCheck::MinWorkgroupSize(req) => {
                    if caps.compute.max_workgroup_size < *req {
                        failures.push(format!(
                            "max_workgroup: need {}, have {}",
                            req, caps.compute.max_workgroup_size
                        ));
                    }
                }
            }
        }

        GateResult { gate_name: self.name.clone(), passed: failures.is_empty(), failures }
    }
}

// ---------------------------------------------------------------------------
// DeviceSelector
// ---------------------------------------------------------------------------

/// Scoring criteria for device selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SelectionStrategy {
    /// Prefer the device with the most compute units.
    MaxComputeUnits,
    /// Prefer the device with the most global memory.
    MaxMemory,
    /// Prefer GPU over CPU, then by compute units.
    PreferGpu,
}

/// Selects the best device from a list of candidates.
pub struct DeviceSelector;

impl DeviceSelector {
    /// Select the best device according to the given strategy.
    ///
    /// Returns the index of the chosen device, or `None` if the
    /// slice is empty.
    pub fn select(devices: &[DeviceCapabilities], strategy: SelectionStrategy) -> Option<usize> {
        if devices.is_empty() {
            return None;
        }

        let score = |d: &DeviceCapabilities| -> (u64, u64) {
            match strategy {
                SelectionStrategy::MaxComputeUnits => {
                    (d.compute.compute_units as u64, d.memory.global_mem_bytes)
                }
                SelectionStrategy::MaxMemory => {
                    (d.memory.global_mem_bytes, d.compute.compute_units as u64)
                }
                SelectionStrategy::PreferGpu => {
                    let is_gpu = if d.vendor.to_lowercase().contains("intel")
                        || d.vendor.to_lowercase().contains("nvidia")
                        || d.vendor.to_lowercase().contains("amd")
                    {
                        // Heuristic: real GPU vendors. CPU reference
                        // devices use "BitNet-rs" as vendor.
                        1u64
                    } else {
                        0u64
                    };
                    // Primary: prefer GPU; secondary: compute units.
                    (is_gpu * 1_000_000 + d.compute.compute_units as u64, d.memory.global_mem_bytes)
                }
            }
        };

        let mut best_idx = 0;
        let mut best_score = score(&devices[0]);
        for (i, dev) in devices.iter().enumerate().skip(1) {
            let s = score(dev);
            if s > best_score {
                best_score = s;
                best_idx = i;
            }
        }
        Some(best_idx)
    }
}

// ---------------------------------------------------------------------------
// CapabilityReport
// ---------------------------------------------------------------------------

/// Human-readable device capability summary, also serialisable as JSON.
pub struct CapabilityReport<'a> {
    caps: &'a DeviceCapabilities,
}

impl<'a> CapabilityReport<'a> {
    pub fn new(caps: &'a DeviceCapabilities) -> Self {
        Self { caps }
    }

    /// Render as a human-readable multi-line string.
    pub fn to_text(&self) -> String {
        format!("{}", self.caps)
    }

    /// Render as a JSON string (hand-rolled, no serde dependency).
    pub fn to_json(&self) -> String {
        let c = self.caps;
        let ext_list: Vec<String> =
            c.extensions.iter().map(|e| format!("\"{}\"", escape_json(e))).collect();
        let sg_list: Vec<String> = c.compute.subgroup_sizes.iter().map(|s| s.to_string()).collect();

        format!(
            "{{\n\
             \x20 \"device_name\": \"{}\",\n\
             \x20 \"vendor\": \"{}\",\n\
             \x20 \"platform\": {{\n\
             \x20   \"name\": \"{}\",\n\
             \x20   \"vendor\": \"{}\",\n\
             \x20   \"version\": \"{}\",\n\
             \x20   \"profile\": \"{}\"\n\
             \x20 }},\n\
             \x20 \"memory\": {{\n\
             \x20   \"global_mem_bytes\": {},\n\
             \x20   \"local_mem_bytes\": {},\n\
             \x20   \"max_alloc_bytes\": {},\n\
             \x20   \"cache_size_bytes\": {},\n\
             \x20   \"cache_line_bytes\": {}\n\
             \x20 }},\n\
             \x20 \"compute\": {{\n\
             \x20   \"compute_units\": {},\n\
             \x20   \"max_workgroup_size\": {},\n\
             \x20   \"max_workitem_dims\": {},\n\
             \x20   \"subgroup_sizes\": [{}],\n\
             \x20   \"fp16\": {},\n\
             \x20   \"fp64\": {},\n\
             \x20   \"int8_dot\": {},\n\
             \x20   \"max_clock_mhz\": {}\n\
             \x20 }},\n\
             \x20 \"extensions\": [{}]\n\
             }}",
            escape_json(&c.device_name),
            escape_json(&c.vendor),
            escape_json(&c.platform.name),
            escape_json(&c.platform.vendor),
            escape_json(&c.platform.version),
            escape_json(&c.platform.profile),
            c.memory.global_mem_bytes,
            c.memory.local_mem_bytes,
            c.memory.max_alloc_bytes,
            c.memory.cache_size_bytes,
            c.memory.cache_line_bytes,
            c.compute.compute_units,
            c.compute.max_workgroup_size,
            c.compute.max_workitem_dims,
            sg_list.join(", "),
            c.compute.fp16,
            c.compute.fp64,
            c.compute.int8_dot,
            c.compute.max_clock_mhz,
            ext_list.join(", "),
        )
    }
}

// ---------------------------------------------------------------------------
// CPU reference / mock device
// ---------------------------------------------------------------------------

/// Build a CPU-reference [`DeviceCapabilities`] for testing purposes.
///
/// Mimics a host CPU with no GPU extensions.
pub fn cpu_reference_device() -> DeviceCapabilities {
    let cpus = std::thread::available_parallelism().map(|n| n.get() as u32).unwrap_or(4);

    DeviceCapabilities {
        device_name: "CPU Reference Device".into(),
        vendor: "BitNet-rs".into(),
        platform: PlatformInfo {
            name: "CPU Reference Platform".into(),
            vendor: "BitNet-rs".into(),
            version: "OpenCL 1.2".into(),
            profile: "FULL_PROFILE".into(),
        },
        memory: MemoryInfo {
            global_mem_bytes: 0,
            local_mem_bytes: 65536,
            max_alloc_bytes: 0,
            cache_size_bytes: 0,
            cache_line_bytes: 64,
        },
        compute: ComputeInfo {
            max_workgroup_size: 1024,
            max_workitem_dims: 3,
            subgroup_sizes: vec![],
            fp16: false,
            fp64: true,
            int8_dot: false,
            compute_units: cpus,
            max_clock_mhz: 0,
        },
        extensions: ExtensionSet::new(),
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Minimal JSON string escaping (backslash + double-quote).
fn escape_json(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"")
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ── A770 profile constants ──────────────────────────────────────

    #[test]
    fn a770_compute_units() {
        assert_eq!(A770Profile::COMPUTE_UNITS, 512);
    }

    #[test]
    fn a770_global_mem_16gib() {
        assert_eq!(A770Profile::GLOBAL_MEM_BYTES, 16 * 1024 * 1024 * 1024);
    }

    #[test]
    fn a770_local_mem_64kib() {
        assert_eq!(A770Profile::LOCAL_MEM_BYTES, 64 * 1024);
    }

    #[test]
    fn a770_max_alloc_4gib() {
        assert_eq!(A770Profile::MAX_ALLOC_BYTES, 4 * 1024 * 1024 * 1024);
    }

    #[test]
    fn a770_max_workgroup() {
        assert_eq!(A770Profile::MAX_WORKGROUP_SIZE, 1024);
    }

    #[test]
    fn a770_subgroup_sizes() {
        assert_eq!(A770Profile::SUBGROUP_SIZES, &[8, 16, 32]);
    }

    #[test]
    fn a770_max_clock() {
        assert_eq!(A770Profile::MAX_CLOCK_MHZ, 2100);
    }

    #[test]
    fn a770_cache_size() {
        assert_eq!(A770Profile::CACHE_SIZE_BYTES, 2 * 1024 * 1024);
    }

    #[test]
    fn a770_cache_line() {
        assert_eq!(A770Profile::CACHE_LINE_BYTES, 64);
    }

    // ── A770 capabilities builder ───────────────────────────────────

    #[test]
    fn a770_caps_device_name() {
        let caps = A770Profile::capabilities();
        assert!(caps.device_name.contains("A770"));
    }

    #[test]
    fn a770_caps_vendor_intel() {
        let caps = A770Profile::capabilities();
        assert!(caps.vendor.contains("Intel"));
    }

    #[test]
    fn a770_caps_full_profile() {
        let caps = A770Profile::capabilities();
        assert!(caps.platform.is_full_profile());
    }

    #[test]
    fn a770_caps_opencl_3() {
        let caps = A770Profile::capabilities();
        assert_eq!(caps.platform.version_tuple(), Some((3, 0)));
    }

    #[test]
    fn a770_caps_fp16_enabled() {
        let caps = A770Profile::capabilities();
        assert!(caps.compute.fp16);
    }

    #[test]
    fn a770_caps_fp64_disabled() {
        let caps = A770Profile::capabilities();
        assert!(!caps.compute.fp64);
    }

    #[test]
    fn a770_caps_int8_dot() {
        let caps = A770Profile::capabilities();
        assert!(caps.compute.int8_dot);
    }

    #[test]
    fn a770_caps_memory_matches_constants() {
        let caps = A770Profile::capabilities();
        assert_eq!(caps.memory.global_mem_bytes, A770Profile::GLOBAL_MEM_BYTES);
        assert_eq!(caps.memory.local_mem_bytes, A770Profile::LOCAL_MEM_BYTES);
    }

    #[test]
    fn a770_caps_compute_matches_constants() {
        let caps = A770Profile::capabilities();
        assert_eq!(caps.compute.compute_units, A770Profile::COMPUTE_UNITS);
        assert_eq!(caps.compute.max_workgroup_size, A770Profile::MAX_WORKGROUP_SIZE);
    }

    // ── MemoryInfo ──────────────────────────────────────────────────

    #[test]
    fn memory_info_gib_conversion() {
        let caps = A770Profile::capabilities();
        let gib = caps.memory.global_mem_gib();
        assert!((gib - 16.0).abs() < 0.001);
    }

    #[test]
    fn memory_info_mib_conversion() {
        let caps = A770Profile::capabilities();
        let mib = caps.memory.global_mem_mib();
        assert!((mib - 16384.0).abs() < 0.1);
    }

    #[test]
    fn memory_has_global_mem_pass() {
        let caps = A770Profile::capabilities();
        assert!(caps.memory.has_global_mem(8 * 1024 * 1024 * 1024));
    }

    #[test]
    fn memory_has_global_mem_fail() {
        let caps = A770Profile::capabilities();
        assert!(!caps.memory.has_global_mem(32 * 1024 * 1024 * 1024));
    }

    #[test]
    fn memory_has_local_mem_pass() {
        let caps = A770Profile::capabilities();
        assert!(caps.memory.has_local_mem(32 * 1024));
    }

    #[test]
    fn memory_has_local_mem_fail() {
        let caps = A770Profile::capabilities();
        assert!(!caps.memory.has_local_mem(128 * 1024));
    }

    #[test]
    fn memory_display() {
        let caps = A770Profile::capabilities();
        let s = format!("{}", caps.memory);
        assert!(s.contains("GiB"));
        assert!(s.contains("KiB"));
    }

    #[test]
    fn memory_zero_global() {
        let m = MemoryInfo {
            global_mem_bytes: 0,
            local_mem_bytes: 0,
            max_alloc_bytes: 0,
            cache_size_bytes: 0,
            cache_line_bytes: 0,
        };
        assert_eq!(m.global_mem_gib(), 0.0);
        assert!(!m.has_global_mem(1));
    }

    // ── ComputeInfo ─────────────────────────────────────────────────

    #[test]
    fn compute_has_subgroup_8() {
        let caps = A770Profile::capabilities();
        assert!(caps.compute.has_subgroup_size(8));
    }

    #[test]
    fn compute_has_subgroup_16() {
        let caps = A770Profile::capabilities();
        assert!(caps.compute.has_subgroup_size(16));
    }

    #[test]
    fn compute_has_subgroup_32() {
        let caps = A770Profile::capabilities();
        assert!(caps.compute.has_subgroup_size(32));
    }

    #[test]
    fn compute_no_subgroup_64() {
        let caps = A770Profile::capabilities();
        assert!(!caps.compute.has_subgroup_size(64));
    }

    #[test]
    fn compute_preferred_subgroup() {
        let caps = A770Profile::capabilities();
        assert_eq!(caps.compute.preferred_subgroup_size(), 32);
    }

    #[test]
    fn compute_preferred_subgroup_empty() {
        let ci = ComputeInfo {
            max_workgroup_size: 256,
            max_workitem_dims: 3,
            subgroup_sizes: vec![],
            fp16: false,
            fp64: false,
            int8_dot: false,
            compute_units: 1,
            max_clock_mhz: 0,
        };
        assert_eq!(ci.preferred_subgroup_size(), 0);
    }

    #[test]
    fn compute_display() {
        let caps = A770Profile::capabilities();
        let s = format!("{}", caps.compute);
        assert!(s.contains("512"));
        assert!(s.contains("fp16=true"));
    }

    // ── ExtensionSet ────────────────────────────────────────────────

    #[test]
    fn extension_set_empty() {
        let es = ExtensionSet::new();
        assert!(es.is_empty());
        assert_eq!(es.len(), 0);
    }

    #[test]
    fn extension_set_from_string() {
        let es = ExtensionSet::from_extension_string("cl_khr_subgroups cl_khr_fp16");
        assert_eq!(es.len(), 2);
        assert!(es.has_khr_subgroups());
        assert!(es.has_khr_fp16());
    }

    #[test]
    fn extension_set_insert() {
        let mut es = ExtensionSet::new();
        es.insert("cl_khr_fp64");
        assert!(es.has_khr_fp64());
        assert_eq!(es.len(), 1);
    }

    #[test]
    fn extension_set_dedup() {
        let es = ExtensionSet::from_extension_string("cl_khr_fp16 cl_khr_fp16 cl_khr_fp16");
        assert_eq!(es.len(), 1);
    }

    #[test]
    fn extension_set_iter_sorted() {
        let es = ExtensionSet::from_extension_string("cl_khr_fp64 cl_khr_fp16 cl_intel_subgroups");
        let names: Vec<&str> = es.iter().collect();
        assert_eq!(names, &["cl_intel_subgroups", "cl_khr_fp16", "cl_khr_fp64"]);
    }

    #[test]
    fn extension_a770_has_khr_subgroups() {
        let caps = A770Profile::capabilities();
        assert!(caps.extensions.has_khr_subgroups());
    }

    #[test]
    fn extension_a770_has_intel_subgroups() {
        let caps = A770Profile::capabilities();
        assert!(caps.extensions.has_intel_subgroups());
    }

    #[test]
    fn extension_a770_has_intel_usm() {
        let caps = A770Profile::capabilities();
        assert!(caps.extensions.has_intel_unified_shared_memory());
    }

    #[test]
    fn extension_set_display() {
        let es = ExtensionSet::from_extension_string("cl_khr_fp16");
        let s = format!("{es}");
        assert!(s.starts_with('['));
        assert!(s.ends_with(']'));
        assert!(s.contains("cl_khr_fp16"));
    }

    #[test]
    fn extension_set_default_empty() {
        let es = ExtensionSet::default();
        assert!(es.is_empty());
    }

    // ── PlatformInfo ────────────────────────────────────────────────

    #[test]
    fn platform_version_tuple_ok() {
        let p = PlatformInfo {
            name: "test".into(),
            vendor: "v".into(),
            version: "OpenCL 3.0 Intel".into(),
            profile: "FULL_PROFILE".into(),
        };
        assert_eq!(p.version_tuple(), Some((3, 0)));
    }

    #[test]
    fn platform_version_tuple_1_2() {
        let p = PlatformInfo {
            name: "t".into(),
            vendor: "v".into(),
            version: "OpenCL 1.2".into(),
            profile: "FULL_PROFILE".into(),
        };
        assert_eq!(p.version_tuple(), Some((1, 2)));
    }

    #[test]
    fn platform_version_tuple_bad() {
        let p = PlatformInfo {
            name: "t".into(),
            vendor: "v".into(),
            version: "not an opencl version".into(),
            profile: "FULL_PROFILE".into(),
        };
        assert_eq!(p.version_tuple(), None);
    }

    #[test]
    fn platform_is_full_profile() {
        let p = PlatformInfo {
            name: "t".into(),
            vendor: "v".into(),
            version: "OpenCL 1.0".into(),
            profile: "FULL_PROFILE".into(),
        };
        assert!(p.is_full_profile());
    }

    #[test]
    fn platform_not_full_profile() {
        let p = PlatformInfo {
            name: "t".into(),
            vendor: "v".into(),
            version: "OpenCL 1.0".into(),
            profile: "EMBEDDED_PROFILE".into(),
        };
        assert!(!p.is_full_profile());
    }

    #[test]
    fn platform_display() {
        let p = PlatformInfo {
            name: "TestPlatform".into(),
            vendor: "TestVendor".into(),
            version: "OpenCL 2.0".into(),
            profile: "FULL_PROFILE".into(),
        };
        let s = format!("{p}");
        assert!(s.contains("TestPlatform"));
        assert!(s.contains("TestVendor"));
    }

    #[test]
    fn platform_eq() {
        let a = PlatformInfo {
            name: "x".into(),
            vendor: "v".into(),
            version: "OpenCL 1.0".into(),
            profile: "FULL_PROFILE".into(),
        };
        let b = a.clone();
        assert_eq!(a, b);
    }

    // ── CapabilityGate ──────────────────────────────────────────────

    #[test]
    fn gate_empty_passes() {
        let gate = CapabilityGate::new("empty");
        let result = gate.evaluate(&A770Profile::capabilities());
        assert!(result.passed);
        assert!(result.failures.is_empty());
    }

    #[test]
    fn gate_fp16_pass_on_a770() {
        let gate = CapabilityGate::new("fp16").requires_fp16();
        let result = gate.evaluate(&A770Profile::capabilities());
        assert!(result.passed);
    }

    #[test]
    fn gate_fp64_fail_on_a770() {
        let gate = CapabilityGate::new("fp64").requires_fp64();
        let result = gate.evaluate(&A770Profile::capabilities());
        assert!(!result.passed);
        assert!(!result.failures.is_empty());
    }

    #[test]
    fn gate_subgroup_16_pass() {
        let gate = CapabilityGate::new("sg16").requires_subgroup_size(16);
        let result = gate.evaluate(&A770Profile::capabilities());
        assert!(result.passed);
    }

    #[test]
    fn gate_subgroup_64_fail() {
        let gate = CapabilityGate::new("sg64").requires_subgroup_size(64);
        let result = gate.evaluate(&A770Profile::capabilities());
        assert!(!result.passed);
    }

    #[test]
    fn gate_min_global_mem_pass() {
        let gate = CapabilityGate::new("mem").min_global_mem(8 * 1024 * 1024 * 1024);
        let result = gate.evaluate(&A770Profile::capabilities());
        assert!(result.passed);
    }

    #[test]
    fn gate_min_global_mem_fail() {
        let gate = CapabilityGate::new("mem").min_global_mem(32 * 1024 * 1024 * 1024);
        let result = gate.evaluate(&A770Profile::capabilities());
        assert!(!result.passed);
    }

    #[test]
    fn gate_min_local_mem_pass() {
        let gate = CapabilityGate::new("lmem").min_local_mem(32 * 1024);
        let result = gate.evaluate(&A770Profile::capabilities());
        assert!(result.passed);
    }

    #[test]
    fn gate_min_compute_units_pass() {
        let gate = CapabilityGate::new("cu").min_compute_units(256);
        let result = gate.evaluate(&A770Profile::capabilities());
        assert!(result.passed);
    }

    #[test]
    fn gate_min_compute_units_fail() {
        let gate = CapabilityGate::new("cu").min_compute_units(1024);
        let result = gate.evaluate(&A770Profile::capabilities());
        assert!(!result.passed);
    }

    #[test]
    fn gate_extension_pass() {
        let gate = CapabilityGate::new("ext").requires_extension("cl_khr_subgroups");
        let result = gate.evaluate(&A770Profile::capabilities());
        assert!(result.passed);
    }

    #[test]
    fn gate_extension_fail() {
        let gate = CapabilityGate::new("ext").requires_extension("cl_khr_gl_sharing");
        let result = gate.evaluate(&A770Profile::capabilities());
        assert!(!result.passed);
    }

    #[test]
    fn gate_int8_dot_pass() {
        let gate = CapabilityGate::new("i8").requires_int8_dot();
        let result = gate.evaluate(&A770Profile::capabilities());
        assert!(result.passed);
    }

    #[test]
    fn gate_workgroup_pass() {
        let gate = CapabilityGate::new("wg").min_workgroup_size(512);
        let result = gate.evaluate(&A770Profile::capabilities());
        assert!(result.passed);
    }

    #[test]
    fn gate_workgroup_fail() {
        let gate = CapabilityGate::new("wg").min_workgroup_size(2048);
        let result = gate.evaluate(&A770Profile::capabilities());
        assert!(!result.passed);
    }

    #[test]
    fn gate_multiple_checks_all_pass() {
        let gate = CapabilityGate::new("combo")
            .requires_fp16()
            .requires_subgroup_size(16)
            .min_compute_units(256)
            .requires_extension("cl_khr_subgroups");
        let result = gate.evaluate(&A770Profile::capabilities());
        assert!(result.passed);
    }

    #[test]
    fn gate_multiple_checks_partial_fail() {
        let gate = CapabilityGate::new("combo").requires_fp16().requires_fp64(); // A770 lacks fp64
        let result = gate.evaluate(&A770Profile::capabilities());
        assert!(!result.passed);
        assert_eq!(result.failures.len(), 1);
    }

    #[test]
    fn gate_result_display_pass() {
        let r = GateResult { gate_name: "test".into(), passed: true, failures: vec![] };
        assert!(format!("{r}").contains("PASS"));
    }

    #[test]
    fn gate_result_display_fail() {
        let r =
            GateResult { gate_name: "test".into(), passed: false, failures: vec!["oops".into()] };
        let s = format!("{r}");
        assert!(s.contains("FAIL"));
        assert!(s.contains("oops"));
    }

    // ── DeviceSelector ──────────────────────────────────────────────

    #[test]
    fn selector_empty_returns_none() {
        let result = DeviceSelector::select(&[], SelectionStrategy::MaxComputeUnits);
        assert!(result.is_none());
    }

    #[test]
    fn selector_single_device() {
        let devs = vec![A770Profile::capabilities()];
        let idx = DeviceSelector::select(&devs, SelectionStrategy::MaxComputeUnits);
        assert_eq!(idx, Some(0));
    }

    #[test]
    fn selector_picks_highest_compute_units() {
        let mut low = cpu_reference_device();
        low.compute.compute_units = 4;
        let high = A770Profile::capabilities(); // 512 CUs
        let devs = vec![low, high];
        let idx = DeviceSelector::select(&devs, SelectionStrategy::MaxComputeUnits);
        assert_eq!(idx, Some(1));
    }

    #[test]
    fn selector_picks_most_memory() {
        let mut small = cpu_reference_device();
        small.memory.global_mem_bytes = 1024;
        let big = A770Profile::capabilities(); // 16 GiB
        let devs = vec![small, big];
        let idx = DeviceSelector::select(&devs, SelectionStrategy::MaxMemory);
        assert_eq!(idx, Some(1));
    }

    #[test]
    fn selector_prefer_gpu_over_cpu() {
        let cpu = cpu_reference_device(); // vendor="BitNet-rs"
        let gpu = A770Profile::capabilities(); // vendor="Intel"
        let devs = vec![cpu, gpu];
        let idx = DeviceSelector::select(&devs, SelectionStrategy::PreferGpu);
        assert_eq!(idx, Some(1));
    }

    // ── CPU reference device ────────────────────────────────────────

    #[test]
    fn cpu_ref_vendor() {
        let d = cpu_reference_device();
        assert_eq!(d.vendor, "BitNet-rs");
    }

    #[test]
    fn cpu_ref_no_fp16() {
        let d = cpu_reference_device();
        assert!(!d.compute.fp16);
    }

    #[test]
    fn cpu_ref_has_fp64() {
        let d = cpu_reference_device();
        assert!(d.compute.fp64);
    }

    #[test]
    fn cpu_ref_no_extensions() {
        let d = cpu_reference_device();
        assert!(d.extensions.is_empty());
    }

    #[test]
    fn cpu_ref_compute_units_positive() {
        let d = cpu_reference_device();
        assert!(d.compute.compute_units > 0);
    }

    #[test]
    fn cpu_ref_full_profile() {
        let d = cpu_reference_device();
        assert!(d.platform.is_full_profile());
    }

    // ── CapabilityReport ────────────────────────────────────────────

    #[test]
    fn report_text_contains_device_name() {
        let caps = A770Profile::capabilities();
        let report = CapabilityReport::new(&caps);
        let text = report.to_text();
        assert!(text.contains("A770"));
    }

    #[test]
    fn report_json_starts_with_brace() {
        let caps = A770Profile::capabilities();
        let report = CapabilityReport::new(&caps);
        let json = report.to_json();
        assert!(json.starts_with('{'));
        assert!(json.ends_with('}'));
    }

    #[test]
    fn report_json_contains_device_name() {
        let caps = A770Profile::capabilities();
        let report = CapabilityReport::new(&caps);
        let json = report.to_json();
        assert!(json.contains("\"device_name\""));
        assert!(json.contains("A770"));
    }

    #[test]
    fn report_json_contains_memory() {
        let caps = A770Profile::capabilities();
        let report = CapabilityReport::new(&caps);
        let json = report.to_json();
        assert!(json.contains("\"global_mem_bytes\""));
        assert!(json.contains("\"local_mem_bytes\""));
    }

    #[test]
    fn report_json_contains_compute() {
        let caps = A770Profile::capabilities();
        let report = CapabilityReport::new(&caps);
        let json = report.to_json();
        assert!(json.contains("\"compute_units\": 512"));
        assert!(json.contains("\"subgroup_sizes\""));
    }

    #[test]
    fn report_json_contains_extensions() {
        let caps = A770Profile::capabilities();
        let report = CapabilityReport::new(&caps);
        let json = report.to_json();
        assert!(json.contains("\"extensions\""));
        assert!(json.contains("cl_khr_subgroups"));
    }

    #[test]
    fn report_json_cpu_ref_valid() {
        let caps = cpu_reference_device();
        let report = CapabilityReport::new(&caps);
        let json = report.to_json();
        assert!(json.starts_with('{'));
        assert!(json.ends_with('}'));
        assert!(json.contains("\"device_name\""));
    }

    // ── DeviceCapabilities Display ──────────────────────────────────

    #[test]
    fn device_caps_display() {
        let caps = A770Profile::capabilities();
        let s = format!("{caps}");
        assert!(s.contains("Device:"));
        assert!(s.contains("Memory:"));
        assert!(s.contains("Compute:"));
        assert!(s.contains("Extensions:"));
    }

    // ── Edge cases ──────────────────────────────────────────────────

    #[test]
    fn unknown_vendor_device() {
        let mut d = cpu_reference_device();
        d.vendor = "UnknownVendor".into();
        // Should still work with DeviceSelector
        let devs = vec![d];
        let idx = DeviceSelector::select(&devs, SelectionStrategy::PreferGpu);
        assert_eq!(idx, Some(0));
    }

    #[test]
    fn zero_memory_device() {
        let d = DeviceCapabilities {
            device_name: "null".into(),
            vendor: "none".into(),
            platform: PlatformInfo {
                name: "n".into(),
                vendor: "n".into(),
                version: "OpenCL 1.0".into(),
                profile: "FULL_PROFILE".into(),
            },
            memory: MemoryInfo {
                global_mem_bytes: 0,
                local_mem_bytes: 0,
                max_alloc_bytes: 0,
                cache_size_bytes: 0,
                cache_line_bytes: 0,
            },
            compute: ComputeInfo {
                max_workgroup_size: 1,
                max_workitem_dims: 1,
                subgroup_sizes: vec![],
                fp16: false,
                fp64: false,
                int8_dot: false,
                compute_units: 0,
                max_clock_mhz: 0,
            },
            extensions: ExtensionSet::new(),
        };
        let gate = CapabilityGate::new("basic").min_global_mem(1).requires_fp16();
        let result = gate.evaluate(&d);
        assert!(!result.passed);
        assert_eq!(result.failures.len(), 2);
    }

    #[test]
    fn extension_from_empty_string() {
        let es = ExtensionSet::from_extension_string("");
        assert!(es.is_empty());
    }

    #[test]
    fn extension_from_whitespace_only() {
        let es = ExtensionSet::from_extension_string("   ");
        assert!(es.is_empty());
    }

    #[test]
    fn escape_json_quotes() {
        assert_eq!(escape_json(r#"say "hello""#), r#"say \"hello\""#);
    }

    #[test]
    fn escape_json_backslash() {
        assert_eq!(escape_json(r"a\b"), r"a\\b");
    }
}
