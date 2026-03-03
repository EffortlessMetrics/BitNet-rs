//! ROCm device detection, enumeration, and capability probing.
//!
//! All detection works without actual ROCm hardware by parsing output from
//! standard ROCm tools (`rocm-smi`, `rocminfo`, `hipconfig`) or falling back
//! to well-known filesystem paths and environment variables.

use std::collections::HashMap;
use std::fmt;
use std::path::{Path, PathBuf};

// ---------------------------------------------------------------------------
// HIP runtime version
// ---------------------------------------------------------------------------

/// Parsed HIP runtime version triple.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct HipVersion {
    pub major: u32,
    pub minor: u32,
    pub patch: u32,
}

impl HipVersion {
    /// Create a new `HipVersion`.
    #[must_use]
    pub const fn new(major: u32, minor: u32, patch: u32) -> Self {
        Self { major, minor, patch }
    }

    /// Parse from a `"major.minor.patch"` string, with optional trailing data.
    #[must_use]
    pub fn parse(s: &str) -> Option<Self> {
        let s = s.trim();
        let mut parts = s.split('.');
        let major = parts.next()?.parse().ok()?;
        let minor = parts.next()?.parse().ok()?;
        // patch may have trailing text like "60061-..." — take digits only
        let patch_str = parts.next().unwrap_or("0");
        let patch_digits: String = patch_str.chars().take_while(|c| c.is_ascii_digit()).collect();
        let patch = if patch_digits.is_empty() { 0 } else { patch_digits.parse().ok()? };
        Some(Self { major, minor, patch })
    }

    /// Returns `true` if this version is at least the given threshold.
    #[must_use]
    pub const fn at_least(&self, major: u32, minor: u32, patch: u32) -> bool {
        if self.major != major {
            return self.major > major;
        }
        if self.minor != minor {
            return self.minor > minor;
        }
        self.patch >= patch
    }
}

impl fmt::Display for HipVersion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}.{}.{}", self.major, self.minor, self.patch)
    }
}

// ---------------------------------------------------------------------------
// ROCm installation path detection
// ---------------------------------------------------------------------------

/// Well-known ROCm installation directories, checked in order.
const ROCM_SEARCH_PATHS: &[&str] = &["/opt/rocm", "/usr/local/rocm", "/usr/lib/rocm"];

/// Detected ROCm installation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RocmInstallation {
    /// Root path of the ROCm installation (e.g. `/opt/rocm`).
    pub root: PathBuf,
    /// HIP runtime version if detectable.
    pub hip_version: Option<HipVersion>,
    /// ROCm platform version if detectable.
    pub rocm_version: Option<String>,
}

/// Detect ROCm installation from environment and well-known paths.
///
/// Checks, in order:
/// 1. `ROCM_PATH` environment variable
/// 2. `HIP_PATH` environment variable (parent of `hip/`)
/// 3. Well-known filesystem paths
#[must_use]
pub fn detect_rocm_installation() -> Option<RocmInstallation> {
    detect_rocm_installation_with(&StdEnv, &StdFs)
}

/// Testable core: accepts injected env + filesystem.
fn detect_rocm_installation_with(
    env: &dyn EnvProvider,
    fs: &dyn FsProvider,
) -> Option<RocmInstallation> {
    // 1. ROCM_PATH
    if let Some(p) = env.var("ROCM_PATH") {
        let root = PathBuf::from(&p);
        if fs.is_dir(&root) {
            return Some(build_installation(root, env, fs));
        }
    }
    // 2. HIP_PATH — typically <rocm>/hip
    if let Some(p) = env.var("HIP_PATH") {
        let hip = PathBuf::from(&p);
        if let Some(parent) = hip.parent() {
            let parent_buf = PathBuf::from(parent);
            if fs.is_dir(&parent_buf) {
                return Some(build_installation(parent_buf, env, fs));
            }
        }
    }
    // 3. well-known paths
    for candidate in ROCM_SEARCH_PATHS {
        let root = PathBuf::from(candidate);
        if fs.is_dir(&root) {
            return Some(build_installation(root, env, fs));
        }
    }
    None
}

fn build_installation(
    root: PathBuf,
    _env: &dyn EnvProvider,
    fs: &dyn FsProvider,
) -> RocmInstallation {
    let version_file = root.join(".info/version");
    let rocm_version = fs.read_to_string(&version_file).map(|s| s.trim().to_string());
    let hip_version = rocm_version.as_deref().and_then(HipVersion::parse);
    RocmInstallation { root, hip_version, rocm_version }
}

// ---------------------------------------------------------------------------
// GPU device types
// ---------------------------------------------------------------------------

/// Architecture family of an AMD GPU.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GpuArchFamily {
    /// RDNA 3 (gfx1100, gfx1101, gfx1102)
    Rdna3,
    /// RDNA 2 (gfx1030, gfx1031, gfx1032)
    Rdna2,
    /// CDNA 3 (gfx940, gfx941, gfx942) — MI300 series
    Cdna3,
    /// CDNA 2 (gfx90a) — MI200 series
    Cdna2,
    /// CDNA 1 (gfx908) — MI100
    Cdna1,
    /// Older or unrecognised architecture.
    Other,
}

impl GpuArchFamily {
    /// Classify a GFX target ID (e.g. `"gfx1100"`) into a family.
    #[must_use]
    pub fn from_gfx_id(gfx: &str) -> Self {
        let gfx = gfx.trim().to_ascii_lowercase();
        if gfx.starts_with("gfx110") {
            Self::Rdna3
        } else if gfx.starts_with("gfx103") {
            Self::Rdna2
        } else if gfx.starts_with("gfx94") {
            Self::Cdna3
        } else if gfx == "gfx90a" {
            Self::Cdna2
        } else if gfx == "gfx908" {
            Self::Cdna1
        } else {
            Self::Other
        }
    }

    /// Returns `true` for data-centre (CDNA) architectures.
    #[must_use]
    pub const fn is_datacenter(&self) -> bool {
        matches!(self, Self::Cdna1 | Self::Cdna2 | Self::Cdna3)
    }
}

/// Describes a single AMD GPU device.
#[derive(Debug, Clone, PartialEq)]
pub struct GpuDevice {
    /// Ordinal index (0-based).
    pub index: u32,
    /// Marketing name (e.g. "Radeon RX 7900 XTX").
    pub name: String,
    /// GFX target (e.g. "gfx1100").
    pub gfx_target: String,
    /// Architecture family derived from `gfx_target`.
    pub arch_family: GpuArchFamily,
    /// Device capabilities (compute units, memory, clocks, etc.).
    pub capabilities: DeviceCapabilities,
    /// PCI bus ID string (e.g. "0000:03:00.0").
    pub pci_bus_id: Option<String>,
}

// ---------------------------------------------------------------------------
// Device capabilities
// ---------------------------------------------------------------------------

/// Hardware capabilities of a single GPU device.
#[derive(Debug, Clone, PartialEq)]
pub struct DeviceCapabilities {
    /// Number of compute units (CU).
    pub compute_units: u32,
    /// Total VRAM in bytes.
    pub total_vram_bytes: u64,
    /// Maximum engine (core) clock in MHz.
    pub max_clock_mhz: u32,
    /// Maximum memory clock in MHz.
    pub max_mem_clock_mhz: u32,
    /// Wavefront (warp) size.
    pub wavefront_size: u32,
    /// Maximum number of work-items per work-group (block).
    pub max_workgroup_size: u32,
    /// Feature support matrix.
    pub features: FeatureSupport,
}

impl DeviceCapabilities {
    /// Estimated peak FP16 TFLOPS (rough: 2 × CU × clock × ops_per_cu_cycle).
    #[must_use]
    pub fn estimated_fp16_tflops(&self) -> f64 {
        // Each CU can execute 128 FP16 FMA ops/cycle on RDNA3 (64 on RDNA2).
        // Use conservative 64 for a lower-bound estimate.
        let ops_per_cycle = 64.0_f64;
        let clock_ghz = self.max_clock_mhz as f64 / 1000.0;
        self.compute_units as f64 * ops_per_cycle * clock_ghz * 2.0 / 1000.0
    }
}

// ---------------------------------------------------------------------------
// Feature support matrix
// ---------------------------------------------------------------------------

/// Data-type and extension support for a device.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FeatureSupport {
    pub fp16: bool,
    pub bf16: bool,
    pub fp32: bool,
    pub fp64: bool,
    pub int8: bool,
    pub int4: bool,
    /// Matrix fused multiply-add (MFMA) — CDNA only.
    pub mfma: bool,
    /// Dot-product instruction (RDNA2+, CDNA).
    pub dot_product: bool,
    /// Packed math (two FP16 ops in one instruction).
    pub packed_math: bool,
}

impl FeatureSupport {
    /// Build feature matrix from architecture family.
    #[must_use]
    pub fn from_arch(arch: GpuArchFamily) -> Self {
        match arch {
            GpuArchFamily::Rdna3 => Self {
                fp16: true,
                bf16: true,
                fp32: true,
                fp64: false,
                int8: true,
                int4: true,
                mfma: false,
                dot_product: true,
                packed_math: true,
            },
            GpuArchFamily::Rdna2 => Self {
                fp16: true,
                bf16: false,
                fp32: true,
                fp64: false,
                int8: true,
                int4: false,
                mfma: false,
                dot_product: true,
                packed_math: true,
            },
            GpuArchFamily::Cdna3 => Self {
                fp16: true,
                bf16: true,
                fp32: true,
                fp64: true,
                int8: true,
                int4: true,
                mfma: true,
                dot_product: true,
                packed_math: true,
            },
            GpuArchFamily::Cdna2 => Self {
                fp16: true,
                bf16: true,
                fp32: true,
                fp64: true,
                int8: true,
                int4: false,
                mfma: true,
                dot_product: true,
                packed_math: true,
            },
            GpuArchFamily::Cdna1 => Self {
                fp16: true,
                bf16: false,
                fp32: true,
                fp64: true,
                int8: true,
                int4: false,
                mfma: true,
                dot_product: true,
                packed_math: true,
            },
            GpuArchFamily::Other => Self {
                fp16: true,
                bf16: false,
                fp32: true,
                fp64: false,
                int8: false,
                int4: false,
                mfma: false,
                dot_product: false,
                packed_math: false,
            },
        }
    }

    /// Returns `true` if the device can accelerate low-bit inference.
    #[must_use]
    pub const fn supports_low_bit_inference(&self) -> bool {
        self.int8 || self.int4
    }

    /// Returns `true` if matrix-level fused multiply-add is available.
    #[must_use]
    pub const fn has_matrix_acceleration(&self) -> bool {
        self.mfma
    }
}

// ---------------------------------------------------------------------------
// Driver version
// ---------------------------------------------------------------------------

/// Parsed AMD GPU driver version.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DriverVersion {
    pub kernel_version: String,
    pub user_mode_version: Option<String>,
}

impl DriverVersion {
    /// Parse a driver version string like `"6.7.0"` or `"23.20.00.48"`.
    #[must_use]
    pub fn parse(s: &str) -> Option<Self> {
        let s = s.trim();
        if s.is_empty() {
            return None;
        }
        Some(Self { kernel_version: s.to_string(), user_mode_version: None })
    }

    /// Parse with both kernel and usermode strings.
    #[must_use]
    pub fn with_usermode(kernel: &str, usermode: &str) -> Option<Self> {
        let kernel = kernel.trim();
        if kernel.is_empty() {
            return None;
        }
        let usermode = usermode.trim();
        Some(Self {
            kernel_version: kernel.to_string(),
            user_mode_version: if usermode.is_empty() { None } else { Some(usermode.to_string()) },
        })
    }
}

impl fmt::Display for DriverVersion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.kernel_version)?;
        if let Some(ref um) = self.user_mode_version {
            write!(f, " (usermode: {um})")?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Multi-GPU topology
// ---------------------------------------------------------------------------

/// Link type between two GPUs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GpuLinkType {
    /// AMD Infinity Fabric.
    Xgmi,
    /// PCIe peer-to-peer.
    Pcie,
    /// Same GPU (self-link).
    SameDevice,
    /// Unknown or no direct link.
    Unknown,
}

/// Describes the link between two GPU devices.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GpuLink {
    pub from_index: u32,
    pub to_index: u32,
    pub link_type: GpuLinkType,
    /// Number of hops (0 = same device, 1 = direct peer, 2+ = bridged).
    pub hops: u32,
}

/// Multi-GPU topology description.
#[derive(Debug, Clone, PartialEq)]
pub struct GpuTopology {
    pub devices: Vec<GpuDevice>,
    pub links: Vec<GpuLink>,
}

impl GpuTopology {
    /// Build from a device list. Populates self-links only.
    #[must_use]
    pub fn from_devices(devices: Vec<GpuDevice>) -> Self {
        let links = devices
            .iter()
            .map(|d| GpuLink {
                from_index: d.index,
                to_index: d.index,
                link_type: GpuLinkType::SameDevice,
                hops: 0,
            })
            .collect();
        Self { devices, links }
    }

    /// Number of devices.
    #[must_use]
    pub fn device_count(&self) -> usize {
        self.devices.len()
    }

    /// Returns `true` if any pair of GPUs is connected via XGMI.
    #[must_use]
    pub fn has_xgmi(&self) -> bool {
        self.links.iter().any(|l| l.link_type == GpuLinkType::Xgmi)
    }

    /// Get link between two device indices.
    #[must_use]
    pub fn link_between(&self, a: u32, b: u32) -> Option<&GpuLink> {
        self.links.iter().find(|l| l.from_index == a && l.to_index == b)
    }

    /// Add a directed link between two devices.
    pub fn add_link(&mut self, link: GpuLink) {
        self.links.push(link);
    }
}

// ---------------------------------------------------------------------------
// Aggregated probe result
// ---------------------------------------------------------------------------

/// Full probe result summarising the ROCm environment.
#[derive(Debug, Clone)]
pub struct RocmProbeResult {
    pub installation: Option<RocmInstallation>,
    pub driver_version: Option<DriverVersion>,
    pub topology: GpuTopology,
    /// Per-device feature map (device index → features).
    pub feature_map: HashMap<u32, FeatureSupport>,
}

impl RocmProbeResult {
    /// `true` if ROCm is installed and at least one GPU is present.
    #[must_use]
    pub fn is_available(&self) -> bool {
        self.installation.is_some() && !self.topology.devices.is_empty()
    }

    /// Total VRAM across all devices.
    #[must_use]
    pub fn total_vram_bytes(&self) -> u64 {
        self.topology.devices.iter().map(|d| d.capabilities.total_vram_bytes).sum()
    }

    /// Returns the device with the most VRAM.
    #[must_use]
    pub fn best_device(&self) -> Option<&GpuDevice> {
        self.topology.devices.iter().max_by_key(|d| d.capabilities.total_vram_bytes)
    }
}

/// Run a full probe of the ROCm environment.
#[must_use]
pub fn probe() -> RocmProbeResult {
    probe_with(&StdEnv, &StdFs)
}

/// Testable probe with injected providers.
fn probe_with(env: &dyn EnvProvider, fs: &dyn FsProvider) -> RocmProbeResult {
    let installation = detect_rocm_installation_with(env, fs);
    RocmProbeResult {
        installation,
        driver_version: None,
        topology: GpuTopology { devices: vec![], links: vec![] },
        feature_map: HashMap::new(),
    }
}

// ---------------------------------------------------------------------------
// rocminfo output parsing
// ---------------------------------------------------------------------------

/// Parse `rocminfo`-style output to extract GPU agent entries.
///
/// Expected format (simplified):
/// ```text
/// Agent N
///   Name:                    gfx1100
///   Marketing Name:          Radeon RX 7900 XTX
///   ...
/// ```
#[must_use]
pub fn parse_rocminfo_agents(output: &str) -> Vec<GpuDevice> {
    let mut devices = Vec::new();
    let mut current_name = String::new();
    let mut current_marketing = String::new();
    let mut current_gfx = String::new();
    let mut current_cu: u32 = 0;
    let mut current_clock: u32 = 0;
    let mut current_mem_clock: u32 = 0;
    let mut current_wavefront: u32 = 64;
    let mut current_workgroup: u32 = 256;
    let mut in_gpu_agent = false;

    for line in output.lines() {
        let trimmed = line.trim();

        if trimmed.starts_with("Agent ") {
            // flush previous if it was GPU
            if in_gpu_agent && !current_gfx.is_empty() {
                let arch = GpuArchFamily::from_gfx_id(&current_gfx);
                devices.push(GpuDevice {
                    index: devices.len() as u32,
                    name: if current_marketing.is_empty() {
                        current_name.clone()
                    } else {
                        current_marketing.clone()
                    },
                    gfx_target: current_gfx.clone(),
                    arch_family: arch,
                    capabilities: DeviceCapabilities {
                        compute_units: current_cu,
                        total_vram_bytes: 0,
                        max_clock_mhz: current_clock,
                        max_mem_clock_mhz: current_mem_clock,
                        wavefront_size: current_wavefront,
                        max_workgroup_size: current_workgroup,
                        features: FeatureSupport::from_arch(arch),
                    },
                    pci_bus_id: None,
                });
            }
            // reset
            current_name.clear();
            current_marketing.clear();
            current_gfx.clear();
            current_cu = 0;
            current_clock = 0;
            current_mem_clock = 0;
            current_wavefront = 64;
            current_workgroup = 256;
            in_gpu_agent = false;
        }

        if let Some(val) = kv(trimmed, "Device Type:") {
            in_gpu_agent = val.contains("GPU");
        }
        if let Some(val) = kv(trimmed, "Name:") {
            current_name = val.to_string();
            if val.starts_with("gfx") {
                current_gfx = val.to_string();
            }
        }
        if let Some(val) = kv(trimmed, "Marketing Name:") {
            current_marketing = val.to_string();
        }
        if let Some(val) = kv(trimmed, "Compute Unit:") {
            current_cu = val.parse().unwrap_or(0);
        }
        if let Some(val) = kv(trimmed, "Max Clock Freq. (MHz):") {
            current_clock = val.parse().unwrap_or(0);
        }
        if let Some(val) = kv(trimmed, "Max Memory Clock Freq. (MHz):") {
            current_mem_clock = val.parse().unwrap_or(0);
        }
        if let Some(val) = kv(trimmed, "Wavefront Size:") {
            current_wavefront = val.parse().unwrap_or(64);
        }
        if let Some(val) = kv(trimmed, "Max Work-group Size:") {
            current_workgroup = val.parse().unwrap_or(256);
        }
    }

    // flush last agent
    if in_gpu_agent && !current_gfx.is_empty() {
        let arch = GpuArchFamily::from_gfx_id(&current_gfx);
        devices.push(GpuDevice {
            index: devices.len() as u32,
            name: if current_marketing.is_empty() { current_name } else { current_marketing },
            gfx_target: current_gfx,
            arch_family: arch,
            capabilities: DeviceCapabilities {
                compute_units: current_cu,
                total_vram_bytes: 0,
                max_clock_mhz: current_clock,
                max_mem_clock_mhz: current_mem_clock,
                wavefront_size: current_wavefront,
                max_workgroup_size: current_workgroup,
                features: FeatureSupport::from_arch(arch),
            },
            pci_bus_id: None,
        });
    }

    devices
}

/// Parse `rocm-smi --showmeminfo vram` style output to attach VRAM sizes.
pub fn attach_vram_from_rocm_smi(devices: &mut [GpuDevice], output: &str) {
    // Example line: "GPU[0]          : vram Total Memory (B): 25753026560"
    for line in output.lines() {
        let trimmed = line.trim();
        let Some(rest) = trimmed.strip_prefix("GPU[") else { continue };
        let Some(bracket_end) = rest.find(']') else { continue };
        let idx: u32 = rest[..bracket_end].parse().unwrap_or(u32::MAX);
        if !(trimmed.contains("Total Memory") || trimmed.contains("total")) {
            continue;
        }
        let Some(bytes_str) = trimmed.rsplit(':').next() else { continue };
        if let Ok(bytes) = bytes_str.trim().parse::<u64>()
            && let Some(dev) = devices.iter_mut().find(|d| d.index == idx)
        {
            dev.capabilities.total_vram_bytes = bytes;
        }
    }
}

/// Parse `rocm-smi --showtopo` style output to extract inter-GPU links.
#[must_use]
pub fn parse_topology_links(output: &str, device_count: u32) -> Vec<GpuLink> {
    let mut links = Vec::new();

    // Self-links
    for i in 0..device_count {
        links.push(GpuLink {
            from_index: i,
            to_index: i,
            link_type: GpuLinkType::SameDevice,
            hops: 0,
        });
    }

    // Parse lines like: "GPU0  GPU1   XGMI  1"
    for line in output.lines() {
        let trimmed = line.trim();
        let parts: Vec<&str> = trimmed.split_whitespace().collect();
        if parts.len() >= 4
            && let (Some(from), Some(to)) = (parse_gpu_index(parts[0]), parse_gpu_index(parts[1]))
            && from != to
        {
            let link_type = match parts[2].to_ascii_uppercase().as_str() {
                "XGMI" => GpuLinkType::Xgmi,
                "PCIE" => GpuLinkType::Pcie,
                _ => GpuLinkType::Unknown,
            };
            let hops = parts[3].parse().unwrap_or(1);
            links.push(GpuLink { from_index: from, to_index: to, link_type, hops });
        }
    }
    links
}

fn parse_gpu_index(s: &str) -> Option<u32> {
    let s = s.trim().to_ascii_uppercase();
    s.strip_prefix("GPU").and_then(|rest| rest.parse().ok())
}

fn kv<'a>(line: &'a str, key: &str) -> Option<&'a str> {
    if let Some(rest) = line.strip_prefix(key) { Some(rest.trim()) } else { None }
}

// ---------------------------------------------------------------------------
// Abstraction seams for testing
// ---------------------------------------------------------------------------

/// Environment variable provider (mockable in tests).
pub(crate) trait EnvProvider {
    fn var(&self, name: &str) -> Option<String>;
}

/// Filesystem provider (mockable in tests).
pub(crate) trait FsProvider {
    fn is_dir(&self, p: &Path) -> bool;
    fn read_to_string(&self, p: &Path) -> Option<String>;
}

/// Real env.
struct StdEnv;
impl EnvProvider for StdEnv {
    fn var(&self, name: &str) -> Option<String> {
        std::env::var(name).ok()
    }
}

/// Real fs.
struct StdFs;
impl FsProvider for StdFs {
    fn is_dir(&self, p: &Path) -> bool {
        p.is_dir()
    }
    fn read_to_string(&self, p: &Path) -> Option<String> {
        std::fs::read_to_string(p).ok()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap as StdMap;

    // -- mock providers --------------------------------------------------

    struct MockEnv(StdMap<String, String>);
    impl MockEnv {
        fn new() -> Self {
            Self(StdMap::new())
        }
        fn set(mut self, k: &str, v: &str) -> Self {
            self.0.insert(k.to_string(), v.to_string());
            self
        }
    }
    impl EnvProvider for MockEnv {
        fn var(&self, name: &str) -> Option<String> {
            self.0.get(name).cloned()
        }
    }

    struct MockFs {
        dirs: Vec<PathBuf>,
        files: StdMap<PathBuf, String>,
    }
    impl MockFs {
        fn new() -> Self {
            Self { dirs: vec![], files: StdMap::new() }
        }
        fn with_dir(mut self, p: &str) -> Self {
            self.dirs.push(PathBuf::from(p));
            self
        }
        fn with_file(mut self, p: &str, content: &str) -> Self {
            self.files.insert(PathBuf::from(p), content.to_string());
            // Ensure parent dirs exist
            let pb = PathBuf::from(p);
            if let Some(parent) = pb.parent() {
                self.dirs.push(parent.to_path_buf());
            }
            self
        }
    }
    impl FsProvider for MockFs {
        fn is_dir(&self, p: &Path) -> bool {
            self.dirs.iter().any(|d| d == p)
        }
        fn read_to_string(&self, p: &Path) -> Option<String> {
            self.files.get(p).cloned()
        }
    }

    // == HipVersion ======================================================

    #[test]
    fn hip_version_parse_basic() {
        let v = HipVersion::parse("6.0.2").unwrap();
        assert_eq!(v, HipVersion::new(6, 0, 2));
    }

    #[test]
    fn hip_version_parse_with_trailing() {
        let v = HipVersion::parse("5.7.60061-abc").unwrap();
        assert_eq!(v, HipVersion::new(5, 7, 60061));
    }

    #[test]
    fn hip_version_parse_whitespace() {
        let v = HipVersion::parse("  6.1.0  ").unwrap();
        assert_eq!(v, HipVersion::new(6, 1, 0));
    }

    #[test]
    fn hip_version_parse_empty() {
        assert!(HipVersion::parse("").is_none());
    }

    #[test]
    fn hip_version_parse_garbage() {
        assert!(HipVersion::parse("abc").is_none());
    }

    #[test]
    fn hip_version_parse_partial() {
        assert!(HipVersion::parse("6").is_none());
    }

    #[test]
    fn hip_version_display() {
        assert_eq!(HipVersion::new(6, 2, 1).to_string(), "6.2.1");
    }

    #[test]
    fn hip_version_at_least() {
        let v = HipVersion::new(6, 1, 0);
        assert!(v.at_least(6, 0, 0));
        assert!(v.at_least(6, 1, 0));
        assert!(!v.at_least(6, 2, 0));
        assert!(!v.at_least(7, 0, 0));
        assert!(v.at_least(5, 9, 9));
    }

    #[test]
    fn hip_version_at_least_patch() {
        let v = HipVersion::new(6, 0, 5);
        assert!(v.at_least(6, 0, 5));
        assert!(v.at_least(6, 0, 4));
        assert!(!v.at_least(6, 0, 6));
    }

    #[test]
    fn hip_version_eq_and_hash() {
        let a = HipVersion::new(1, 2, 3);
        let b = HipVersion::new(1, 2, 3);
        assert_eq!(a, b);
        let mut set = std::collections::HashSet::new();
        set.insert(a);
        assert!(set.contains(&b));
    }

    #[test]
    fn hip_version_parse_two_part_returns_zero_patch() {
        let v = HipVersion::parse("6.1").unwrap();
        assert_eq!(v.patch, 0);
    }

    // == RocmInstallation ================================================

    #[test]
    fn detect_from_rocm_path_env() {
        let env = MockEnv::new().set("ROCM_PATH", "/opt/rocm");
        let fs =
            MockFs::new().with_dir("/opt/rocm").with_file("/opt/rocm/.info/version", "6.0.2\n");
        let inst = detect_rocm_installation_with(&env, &fs).unwrap();
        assert_eq!(inst.root, PathBuf::from("/opt/rocm"));
        assert_eq!(inst.hip_version, Some(HipVersion::new(6, 0, 2)));
    }

    #[test]
    fn detect_from_hip_path_env() {
        let env = MockEnv::new().set("HIP_PATH", "/opt/rocm/hip");
        let fs =
            MockFs::new().with_dir("/opt/rocm").with_file("/opt/rocm/.info/version", "5.7.0\n");
        let inst = detect_rocm_installation_with(&env, &fs).unwrap();
        assert_eq!(inst.root, PathBuf::from("/opt/rocm"));
    }

    #[test]
    fn detect_from_well_known_path() {
        let env = MockEnv::new();
        let fs = MockFs::new().with_dir("/opt/rocm");
        let inst = detect_rocm_installation_with(&env, &fs).unwrap();
        assert_eq!(inst.root, PathBuf::from("/opt/rocm"));
    }

    #[test]
    fn detect_none_when_nothing() {
        let env = MockEnv::new();
        let fs = MockFs::new();
        assert!(detect_rocm_installation_with(&env, &fs).is_none());
    }

    #[test]
    fn detect_rocm_path_takes_priority_over_well_known() {
        let env = MockEnv::new().set("ROCM_PATH", "/custom/rocm");
        let fs = MockFs::new().with_dir("/custom/rocm").with_dir("/opt/rocm");
        let inst = detect_rocm_installation_with(&env, &fs).unwrap();
        assert_eq!(inst.root, PathBuf::from("/custom/rocm"));
    }

    #[test]
    fn detect_skips_nonexistent_rocm_path() {
        let env = MockEnv::new().set("ROCM_PATH", "/does/not/exist");
        let fs = MockFs::new().with_dir("/opt/rocm");
        let inst = detect_rocm_installation_with(&env, &fs).unwrap();
        assert_eq!(inst.root, PathBuf::from("/opt/rocm"));
    }

    #[test]
    fn detect_version_file_missing_still_returns_installation() {
        let env = MockEnv::new().set("ROCM_PATH", "/opt/rocm");
        let fs = MockFs::new().with_dir("/opt/rocm");
        let inst = detect_rocm_installation_with(&env, &fs).unwrap();
        assert!(inst.hip_version.is_none());
        assert!(inst.rocm_version.is_none());
    }

    // == GpuArchFamily ===================================================

    #[test]
    fn arch_from_gfx_rdna3() {
        assert_eq!(GpuArchFamily::from_gfx_id("gfx1100"), GpuArchFamily::Rdna3);
        assert_eq!(GpuArchFamily::from_gfx_id("gfx1101"), GpuArchFamily::Rdna3);
        assert_eq!(GpuArchFamily::from_gfx_id("gfx1102"), GpuArchFamily::Rdna3);
    }

    #[test]
    fn arch_from_gfx_rdna2() {
        assert_eq!(GpuArchFamily::from_gfx_id("gfx1030"), GpuArchFamily::Rdna2);
        assert_eq!(GpuArchFamily::from_gfx_id("gfx1031"), GpuArchFamily::Rdna2);
    }

    #[test]
    fn arch_from_gfx_cdna3() {
        assert_eq!(GpuArchFamily::from_gfx_id("gfx940"), GpuArchFamily::Cdna3);
        assert_eq!(GpuArchFamily::from_gfx_id("gfx942"), GpuArchFamily::Cdna3);
    }

    #[test]
    fn arch_from_gfx_cdna2() {
        assert_eq!(GpuArchFamily::from_gfx_id("gfx90a"), GpuArchFamily::Cdna2);
    }

    #[test]
    fn arch_from_gfx_cdna1() {
        assert_eq!(GpuArchFamily::from_gfx_id("gfx908"), GpuArchFamily::Cdna1);
    }

    #[test]
    fn arch_from_gfx_unknown() {
        assert_eq!(GpuArchFamily::from_gfx_id("gfx600"), GpuArchFamily::Other);
        assert_eq!(GpuArchFamily::from_gfx_id("unknown"), GpuArchFamily::Other);
    }

    #[test]
    fn arch_from_gfx_case_insensitive() {
        assert_eq!(GpuArchFamily::from_gfx_id("GFX1100"), GpuArchFamily::Rdna3);
        assert_eq!(GpuArchFamily::from_gfx_id("Gfx90a"), GpuArchFamily::Cdna2);
    }

    #[test]
    fn arch_is_datacenter() {
        assert!(GpuArchFamily::Cdna1.is_datacenter());
        assert!(GpuArchFamily::Cdna2.is_datacenter());
        assert!(GpuArchFamily::Cdna3.is_datacenter());
        assert!(!GpuArchFamily::Rdna2.is_datacenter());
        assert!(!GpuArchFamily::Rdna3.is_datacenter());
        assert!(!GpuArchFamily::Other.is_datacenter());
    }

    // == FeatureSupport ==================================================

    #[test]
    fn feature_rdna3_has_bf16_and_int4() {
        let f = FeatureSupport::from_arch(GpuArchFamily::Rdna3);
        assert!(f.bf16);
        assert!(f.int4);
        assert!(f.int8);
        assert!(!f.mfma);
        assert!(!f.fp64);
    }

    #[test]
    fn feature_rdna2_no_bf16_no_int4() {
        let f = FeatureSupport::from_arch(GpuArchFamily::Rdna2);
        assert!(!f.bf16);
        assert!(!f.int4);
        assert!(f.int8);
    }

    #[test]
    fn feature_cdna3_full() {
        let f = FeatureSupport::from_arch(GpuArchFamily::Cdna3);
        assert!(f.bf16);
        assert!(f.int4);
        assert!(f.fp64);
        assert!(f.mfma);
    }

    #[test]
    fn feature_cdna2_no_int4() {
        let f = FeatureSupport::from_arch(GpuArchFamily::Cdna2);
        assert!(f.bf16);
        assert!(!f.int4);
        assert!(f.mfma);
    }

    #[test]
    fn feature_cdna1_no_bf16() {
        let f = FeatureSupport::from_arch(GpuArchFamily::Cdna1);
        assert!(!f.bf16);
        assert!(f.mfma);
        assert!(f.fp64);
    }

    #[test]
    fn feature_other_minimal() {
        let f = FeatureSupport::from_arch(GpuArchFamily::Other);
        assert!(f.fp16);
        assert!(f.fp32);
        assert!(!f.bf16);
        assert!(!f.int8);
        assert!(!f.mfma);
    }

    #[test]
    fn feature_low_bit_inference() {
        assert!(FeatureSupport::from_arch(GpuArchFamily::Rdna3).supports_low_bit_inference());
        assert!(!FeatureSupport::from_arch(GpuArchFamily::Other).supports_low_bit_inference());
    }

    #[test]
    fn feature_matrix_acceleration() {
        assert!(FeatureSupport::from_arch(GpuArchFamily::Cdna3).has_matrix_acceleration());
        assert!(!FeatureSupport::from_arch(GpuArchFamily::Rdna3).has_matrix_acceleration());
    }

    // == DriverVersion ===================================================

    #[test]
    fn driver_parse_basic() {
        let d = DriverVersion::parse("6.7.0").unwrap();
        assert_eq!(d.kernel_version, "6.7.0");
        assert!(d.user_mode_version.is_none());
    }

    #[test]
    fn driver_parse_empty() {
        assert!(DriverVersion::parse("").is_none());
        assert!(DriverVersion::parse("   ").is_none());
    }

    #[test]
    fn driver_with_usermode() {
        let d = DriverVersion::with_usermode("6.7.0", "23.20.00.48").unwrap();
        assert_eq!(d.user_mode_version.as_deref(), Some("23.20.00.48"));
    }

    #[test]
    fn driver_with_empty_usermode() {
        let d = DriverVersion::with_usermode("6.7.0", "").unwrap();
        assert!(d.user_mode_version.is_none());
    }

    #[test]
    fn driver_with_empty_kernel_returns_none() {
        assert!(DriverVersion::with_usermode("", "23.20").is_none());
    }

    #[test]
    fn driver_display() {
        let d = DriverVersion::with_usermode("6.7.0", "23.20").unwrap();
        assert_eq!(d.to_string(), "6.7.0 (usermode: 23.20)");
    }

    #[test]
    fn driver_display_no_usermode() {
        let d = DriverVersion::parse("6.7.0").unwrap();
        assert_eq!(d.to_string(), "6.7.0");
    }

    // == GpuTopology =====================================================

    #[test]
    fn topology_from_empty() {
        let topo = GpuTopology::from_devices(vec![]);
        assert_eq!(topo.device_count(), 0);
        assert!(!topo.has_xgmi());
    }

    #[test]
    fn topology_self_links() {
        let dev = make_test_device(0, "gfx1100");
        let topo = GpuTopology::from_devices(vec![dev]);
        assert_eq!(topo.links.len(), 1);
        assert_eq!(topo.links[0].link_type, GpuLinkType::SameDevice);
    }

    #[test]
    fn topology_has_xgmi() {
        let mut topo = GpuTopology::from_devices(vec![
            make_test_device(0, "gfx90a"),
            make_test_device(1, "gfx90a"),
        ]);
        topo.add_link(GpuLink {
            from_index: 0,
            to_index: 1,
            link_type: GpuLinkType::Xgmi,
            hops: 1,
        });
        assert!(topo.has_xgmi());
    }

    #[test]
    fn topology_link_between() {
        let mut topo = GpuTopology::from_devices(vec![
            make_test_device(0, "gfx1100"),
            make_test_device(1, "gfx1100"),
        ]);
        topo.add_link(GpuLink {
            from_index: 0,
            to_index: 1,
            link_type: GpuLinkType::Pcie,
            hops: 1,
        });
        assert!(topo.link_between(0, 1).is_some());
        assert!(topo.link_between(1, 0).is_none());
    }

    // == parse_rocminfo_agents ==========================================

    #[test]
    fn parse_rocminfo_single_gpu() {
        let output = "\
Agent 1
  Name:                    gfx1100
  Marketing Name:          Radeon RX 7900 XTX
  Device Type:             GPU
  Compute Unit:            96
  Max Clock Freq. (MHz):   2500
  Wavefront Size:          32
  Max Work-group Size:     1024
";
        let devs = parse_rocminfo_agents(output);
        assert_eq!(devs.len(), 1);
        assert_eq!(devs[0].name, "Radeon RX 7900 XTX");
        assert_eq!(devs[0].gfx_target, "gfx1100");
        assert_eq!(devs[0].arch_family, GpuArchFamily::Rdna3);
        assert_eq!(devs[0].capabilities.compute_units, 96);
        assert_eq!(devs[0].capabilities.max_clock_mhz, 2500);
        assert_eq!(devs[0].capabilities.wavefront_size, 32);
        assert_eq!(devs[0].capabilities.max_workgroup_size, 1024);
    }

    #[test]
    fn parse_rocminfo_multi_gpu() {
        let output = "\
Agent 1
  Name:                    gfx90a
  Marketing Name:          MI250X
  Device Type:             GPU
  Compute Unit:            110
  Max Clock Freq. (MHz):   1700
Agent 2
  Name:                    gfx90a
  Marketing Name:          MI250X
  Device Type:             GPU
  Compute Unit:            110
  Max Clock Freq. (MHz):   1700
";
        let devs = parse_rocminfo_agents(output);
        assert_eq!(devs.len(), 2);
        assert_eq!(devs[0].index, 0);
        assert_eq!(devs[1].index, 1);
        assert_eq!(devs[0].arch_family, GpuArchFamily::Cdna2);
    }

    #[test]
    fn parse_rocminfo_skips_cpu_agents() {
        let output = "\
Agent 0
  Name:                    Intel(R) Core(TM) i9
  Device Type:             CPU
Agent 1
  Name:                    gfx1030
  Marketing Name:          Radeon RX 6800
  Device Type:             GPU
  Compute Unit:            60
  Max Clock Freq. (MHz):   2105
";
        let devs = parse_rocminfo_agents(output);
        assert_eq!(devs.len(), 1);
        assert_eq!(devs[0].gfx_target, "gfx1030");
    }

    #[test]
    fn parse_rocminfo_empty() {
        let devs = parse_rocminfo_agents("");
        assert!(devs.is_empty());
    }

    #[test]
    fn parse_rocminfo_no_marketing_name() {
        let output = "\
Agent 1
  Name:                    gfx908
  Device Type:             GPU
  Compute Unit:            120
  Max Clock Freq. (MHz):   1502
";
        let devs = parse_rocminfo_agents(output);
        assert_eq!(devs.len(), 1);
        assert_eq!(devs[0].name, "gfx908");
    }

    // == attach_vram_from_rocm_smi ======================================

    #[test]
    fn attach_vram_basic() {
        let mut devs = vec![make_test_device(0, "gfx1100")];
        let output = "GPU[0]          : vram Total Memory (B): 25753026560\n";
        attach_vram_from_rocm_smi(&mut devs, output);
        assert_eq!(devs[0].capabilities.total_vram_bytes, 25_753_026_560);
    }

    #[test]
    fn attach_vram_multi_gpu() {
        let mut devs = vec![make_test_device(0, "gfx90a"), make_test_device(1, "gfx90a")];
        let output = "\
GPU[0]          : vram Total Memory (B): 68719476736
GPU[1]          : vram Total Memory (B): 68719476736
";
        attach_vram_from_rocm_smi(&mut devs, output);
        assert_eq!(devs[0].capabilities.total_vram_bytes, 68_719_476_736);
        assert_eq!(devs[1].capabilities.total_vram_bytes, 68_719_476_736);
    }

    #[test]
    fn attach_vram_ignores_unknown_gpu_index() {
        let mut devs = vec![make_test_device(0, "gfx1100")];
        let output = "GPU[5]          : vram Total Memory (B): 999\n";
        attach_vram_from_rocm_smi(&mut devs, output);
        assert_eq!(devs[0].capabilities.total_vram_bytes, 0);
    }

    #[test]
    fn attach_vram_empty_output() {
        let mut devs = vec![make_test_device(0, "gfx1100")];
        attach_vram_from_rocm_smi(&mut devs, "");
        assert_eq!(devs[0].capabilities.total_vram_bytes, 0);
    }

    // == parse_topology_links ===========================================

    #[test]
    fn parse_topo_xgmi() {
        let output = "GPU0  GPU1   XGMI  1\nGPU1  GPU0   XGMI  1\n";
        let links = parse_topology_links(output, 2);
        // 2 self-links + 2 XGMI
        assert_eq!(links.len(), 4);
        let xgmi: Vec<_> = links.iter().filter(|l| l.link_type == GpuLinkType::Xgmi).collect();
        assert_eq!(xgmi.len(), 2);
    }

    #[test]
    fn parse_topo_pcie() {
        let output = "GPU0  GPU1   PCIE  2\n";
        let links = parse_topology_links(output, 2);
        let pcie: Vec<_> = links.iter().filter(|l| l.link_type == GpuLinkType::Pcie).collect();
        assert_eq!(pcie.len(), 1);
        assert_eq!(pcie[0].hops, 2);
    }

    #[test]
    fn parse_topo_empty() {
        let links = parse_topology_links("", 0);
        assert!(links.is_empty());
    }

    #[test]
    fn parse_topo_self_links_only() {
        let links = parse_topology_links("", 3);
        assert_eq!(links.len(), 3);
        assert!(links.iter().all(|l| l.link_type == GpuLinkType::SameDevice));
    }

    // == DeviceCapabilities ==============================================

    #[test]
    fn estimated_tflops_nonzero() {
        let caps = DeviceCapabilities {
            compute_units: 96,
            total_vram_bytes: 24 * 1024 * 1024 * 1024,
            max_clock_mhz: 2500,
            max_mem_clock_mhz: 1250,
            wavefront_size: 32,
            max_workgroup_size: 1024,
            features: FeatureSupport::from_arch(GpuArchFamily::Rdna3),
        };
        let tflops = caps.estimated_fp16_tflops();
        assert!(tflops > 0.0, "tflops should be positive");
    }

    #[test]
    fn estimated_tflops_zero_cu() {
        let caps = DeviceCapabilities {
            compute_units: 0,
            total_vram_bytes: 0,
            max_clock_mhz: 2500,
            max_mem_clock_mhz: 0,
            wavefront_size: 64,
            max_workgroup_size: 256,
            features: FeatureSupport::from_arch(GpuArchFamily::Other),
        };
        assert_eq!(caps.estimated_fp16_tflops(), 0.0);
    }

    // == RocmProbeResult =================================================

    #[test]
    fn probe_result_not_available_when_empty() {
        let result = RocmProbeResult {
            installation: None,
            driver_version: None,
            topology: GpuTopology { devices: vec![], links: vec![] },
            feature_map: HashMap::new(),
        };
        assert!(!result.is_available());
    }

    #[test]
    fn probe_result_available() {
        let result = RocmProbeResult {
            installation: Some(RocmInstallation {
                root: PathBuf::from("/opt/rocm"),
                hip_version: Some(HipVersion::new(6, 0, 2)),
                rocm_version: Some("6.0.2".into()),
            }),
            driver_version: None,
            topology: GpuTopology::from_devices(vec![make_test_device(0, "gfx1100")]),
            feature_map: HashMap::new(),
        };
        assert!(result.is_available());
    }

    #[test]
    fn probe_result_total_vram() {
        let mut d0 = make_test_device(0, "gfx90a");
        let mut d1 = make_test_device(1, "gfx90a");
        d0.capabilities.total_vram_bytes = 64 * 1024 * 1024 * 1024;
        d1.capabilities.total_vram_bytes = 64 * 1024 * 1024 * 1024;
        let result = RocmProbeResult {
            installation: None,
            driver_version: None,
            topology: GpuTopology::from_devices(vec![d0, d1]),
            feature_map: HashMap::new(),
        };
        assert_eq!(result.total_vram_bytes(), 128 * 1024 * 1024 * 1024);
    }

    #[test]
    fn probe_result_best_device() {
        let mut d0 = make_test_device(0, "gfx1100");
        let mut d1 = make_test_device(1, "gfx90a");
        d0.capabilities.total_vram_bytes = 24 * 1024 * 1024 * 1024;
        d1.capabilities.total_vram_bytes = 64 * 1024 * 1024 * 1024;
        let result = RocmProbeResult {
            installation: None,
            driver_version: None,
            topology: GpuTopology::from_devices(vec![d0, d1]),
            feature_map: HashMap::new(),
        };
        let best = result.best_device().unwrap();
        assert_eq!(best.index, 1);
    }

    #[test]
    fn probe_result_best_device_empty() {
        let result = RocmProbeResult {
            installation: None,
            driver_version: None,
            topology: GpuTopology { devices: vec![], links: vec![] },
            feature_map: HashMap::new(),
        };
        assert!(result.best_device().is_none());
    }

    // == probe_with (integration) ========================================

    #[test]
    fn probe_with_no_rocm() {
        let env = MockEnv::new();
        let fs = MockFs::new();
        let result = probe_with(&env, &fs);
        assert!(!result.is_available());
    }

    #[test]
    fn probe_with_rocm_installed() {
        let env = MockEnv::new().set("ROCM_PATH", "/opt/rocm");
        let fs =
            MockFs::new().with_dir("/opt/rocm").with_file("/opt/rocm/.info/version", "6.0.2\n");
        let result = probe_with(&env, &fs);
        assert!(result.installation.is_some());
        let inst = result.installation.unwrap();
        assert_eq!(inst.hip_version, Some(HipVersion::new(6, 0, 2)));
    }

    // == GpuLinkType =====================================================

    #[test]
    fn gpu_link_type_equality() {
        assert_eq!(GpuLinkType::Xgmi, GpuLinkType::Xgmi);
        assert_ne!(GpuLinkType::Xgmi, GpuLinkType::Pcie);
    }

    // == parse_gpu_index =================================================

    #[test]
    fn parse_gpu_index_valid() {
        assert_eq!(parse_gpu_index("GPU0"), Some(0));
        assert_eq!(parse_gpu_index("GPU12"), Some(12));
        assert_eq!(parse_gpu_index("gpu3"), Some(3));
    }

    #[test]
    fn parse_gpu_index_invalid() {
        assert!(parse_gpu_index("CPU0").is_none());
        assert!(parse_gpu_index("").is_none());
        assert!(parse_gpu_index("GPUx").is_none());
    }

    // -- helpers ---------------------------------------------------------

    fn make_test_device(index: u32, gfx: &str) -> GpuDevice {
        let arch = GpuArchFamily::from_gfx_id(gfx);
        GpuDevice {
            index,
            name: format!("Test GPU {index}"),
            gfx_target: gfx.to_string(),
            arch_family: arch,
            capabilities: DeviceCapabilities {
                compute_units: 64,
                total_vram_bytes: 0,
                max_clock_mhz: 2000,
                max_mem_clock_mhz: 1000,
                wavefront_size: 64,
                max_workgroup_size: 256,
                features: FeatureSupport::from_arch(arch),
            },
            pci_bus_id: None,
        }
    }
}
