//! Cross-compilation support matrix for multi-platform GPU backend builds.
//!
//! Provides platform detection, backend availability mapping, feature flag
//! resolution, and build configuration generation for all supported
//! target/backend combinations.
//!
//! # Platform-specific quirks
//!
//! - **Metal** is only available on macOS (x86_64 and aarch64).
//! - **CUDA** requires `nvcc` on the host and is unavailable on wasm32 and
//!   macOS aarch64 (no NVIDIA driver support).
//! - **Vulkan** is optional everywhere except wasm32 (no native Vulkan).
//! - **wasm32** only supports the CPU (scalar) backend via wasm-bindgen.
//! - **aarch64-linux** supports CUDA (e.g. Jetson) and Vulkan but not Metal.
//! - **x86_64-macos** can use CUDA via eGPU, but this is rare and optional.
//! - Linker flags vary: Linux uses `-lpthread -ldl`, Windows uses
//!   `user32.lib`, macOS uses `-framework Metal -framework CoreGraphics`.

use std::fmt;

// ── Target platforms ─────────────────────────────────────────────────────

/// Supported cross-compilation target platforms.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TargetPlatform {
    X86_64Linux,
    X86_64Windows,
    X86_64MacOS,
    Aarch64Linux,
    Aarch64MacOS,
    Wasm32,
}

impl TargetPlatform {
    /// All known platforms.
    pub const ALL: &[Self] = &[
        Self::X86_64Linux,
        Self::X86_64Windows,
        Self::X86_64MacOS,
        Self::Aarch64Linux,
        Self::Aarch64MacOS,
        Self::Wasm32,
    ];

    /// Rust target triple for this platform.
    pub const fn target_triple(self) -> &'static str {
        match self {
            Self::X86_64Linux => "x86_64-unknown-linux-gnu",
            Self::X86_64Windows => "x86_64-pc-windows-msvc",
            Self::X86_64MacOS => "x86_64-apple-darwin",
            Self::Aarch64Linux => "aarch64-unknown-linux-gnu",
            Self::Aarch64MacOS => "aarch64-apple-darwin",
            Self::Wasm32 => "wasm32-unknown-unknown",
        }
    }

    /// Whether this platform uses a POSIX-style linker.
    pub const fn is_posix(self) -> bool {
        matches!(
            self,
            Self::X86_64Linux
                | Self::X86_64MacOS
                | Self::Aarch64Linux
                | Self::Aarch64MacOS
        )
    }

    /// Whether this is a macOS target (Metal capable).
    pub const fn is_macos(self) -> bool {
        matches!(self, Self::X86_64MacOS | Self::Aarch64MacOS)
    }

    /// Whether this is an ARM target.
    pub const fn is_aarch64(self) -> bool {
        matches!(self, Self::Aarch64Linux | Self::Aarch64MacOS)
    }
}

impl fmt::Display for TargetPlatform {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.target_triple())
    }
}

// ── GPU backends ─────────────────────────────────────────────────────────

/// GPU compute backends that the HAL can target.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GpuBackend {
    /// CPU-only scalar/SIMD path (always available).
    Cpu,
    /// NVIDIA CUDA (requires nvcc and driver).
    Cuda,
    /// Apple Metal (macOS only).
    Metal,
    /// Vulkan (cross-platform, optional driver).
    Vulkan,
}

impl GpuBackend {
    pub const ALL: &[Self] = &[
        Self::Cpu,
        Self::Cuda,
        Self::Metal,
        Self::Vulkan,
    ];

    /// Cargo feature name that enables this backend.
    pub const fn feature_name(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Cuda => "cuda",
            Self::Metal => "metal",
            Self::Vulkan => "vulkan",
        }
    }
}

impl fmt::Display for GpuBackend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.feature_name())
    }
}

// ── Backend availability ─────────────────────────────────────────────────

/// Whether a backend is usable on a given platform.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Availability {
    /// Backend is always available (e.g. CPU everywhere).
    Available,
    /// Backend can be used if the right toolchain/driver is present.
    Optional,
    /// Backend cannot work on this platform at all.
    Unavailable,
}

/// Query the availability of a `(platform, backend)` pair.
pub const fn backend_availability(
    platform: TargetPlatform,
    backend: GpuBackend,
) -> Availability {
    match (platform, backend) {
        // CPU is universally available.
        (_, GpuBackend::Cpu) => Availability::Available,

        // CUDA: unavailable on wasm32 and aarch64-macos.
        (TargetPlatform::Wasm32, GpuBackend::Cuda) => Availability::Unavailable,
        (TargetPlatform::Aarch64MacOS, GpuBackend::Cuda) => {
            Availability::Unavailable
        }
        (_, GpuBackend::Cuda) => Availability::Optional,

        // Metal: only on macOS.
        (TargetPlatform::X86_64MacOS, GpuBackend::Metal) => {
            Availability::Available
        }
        (TargetPlatform::Aarch64MacOS, GpuBackend::Metal) => {
            Availability::Available
        }
        (_, GpuBackend::Metal) => Availability::Unavailable,

        // Vulkan: unavailable on wasm32, optional elsewhere.
        (TargetPlatform::Wasm32, GpuBackend::Vulkan) => {
            Availability::Unavailable
        }
        (_, GpuBackend::Vulkan) => Availability::Optional,
    }
}

// ── Platform capabilities ────────────────────────────────────────────────

/// Summary of which backends are available on a platform.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlatformCapabilities {
    pub platform: TargetPlatform,
    pub available: Vec<GpuBackend>,
    pub optional: Vec<GpuBackend>,
}

impl PlatformCapabilities {
    /// Compute capabilities for `platform`.
    pub fn for_platform(platform: TargetPlatform) -> Self {
        let mut available = Vec::new();
        let mut optional = Vec::new();
        for &backend in GpuBackend::ALL {
            match backend_availability(platform, backend) {
                Availability::Available => available.push(backend),
                Availability::Optional => optional.push(backend),
                Availability::Unavailable => {}
            }
        }
        Self { platform, available, optional }
    }

    /// All backends that *could* be enabled (available + optional).
    pub fn all_possible(&self) -> Vec<GpuBackend> {
        let mut v = self.available.clone();
        v.extend_from_slice(&self.optional);
        v
    }
}

// ── Build configuration ──────────────────────────────────────────────────

/// A concrete build configuration for one platform/backend combination.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BuildConfig {
    pub platform: TargetPlatform,
    pub backend: GpuBackend,
    pub target_triple: String,
    pub features: Vec<String>,
    pub linker_flags: Vec<String>,
    pub env_vars: Vec<(String, String)>,
}

impl BuildConfig {
    /// Generate the build config for the given combination.
    ///
    /// Returns `None` if the backend is [`Availability::Unavailable`].
    pub fn generate(
        platform: TargetPlatform,
        backend: GpuBackend,
    ) -> Option<Self> {
        if backend_availability(platform, backend) == Availability::Unavailable
        {
            return None;
        }

        let mut features = vec![backend.feature_name().to_owned()];
        // GPU umbrella feature when using a GPU backend.
        if !matches!(backend, GpuBackend::Cpu) {
            features.insert(0, "gpu".to_owned());
        }

        let linker_flags = linker_flags_for(platform, backend);
        let env_vars = env_vars_for(platform, backend);

        Some(Self {
            platform,
            backend,
            target_triple: platform.target_triple().to_owned(),
            features,
            linker_flags,
            env_vars,
        })
    }
}

/// Platform/backend-specific linker flags.
fn linker_flags_for(
    platform: TargetPlatform,
    backend: GpuBackend,
) -> Vec<String> {
    let mut flags = Vec::new();

    // POSIX baseline.
    if platform.is_posix() {
        flags.push("-lpthread".into());
        flags.push("-ldl".into());
    }

    match (platform, backend) {
        (p, GpuBackend::Metal) if p.is_macos() => {
            flags.push("-framework Metal".into());
            flags.push("-framework CoreGraphics".into());
        }
        (_, GpuBackend::Cuda) => {
            flags.push("-lcuda".into());
            flags.push("-lcudart".into());
        }
        (_, GpuBackend::Vulkan) => {
            flags.push("-lvulkan".into());
        }
        _ => {}
    }

    flags
}

/// Platform/backend-specific environment variables.
fn env_vars_for(
    platform: TargetPlatform,
    backend: GpuBackend,
) -> Vec<(String, String)> {
    let mut vars = Vec::new();
    if matches!(backend, GpuBackend::Cuda) {
        vars.push(("CUDA_HOME".into(), "/usr/local/cuda".into()));
    }
    if platform.is_posix() {
        vars.push(("CC".into(), "cc".into()));
    }
    if matches!(platform, TargetPlatform::X86_64Windows) {
        vars.push(("CC".into(), "cl.exe".into()));
    }
    vars
}

// ── Cross-compile matrix ─────────────────────────────────────────────────

/// The full matrix of valid platform×backend build configurations.
#[derive(Debug, Clone)]
pub struct CrossCompileMatrix {
    pub configs: Vec<BuildConfig>,
}

impl CrossCompileMatrix {
    /// Build the full matrix, skipping unavailable combinations.
    pub fn generate() -> Self {
        let mut configs = Vec::new();
        for &platform in TargetPlatform::ALL {
            for &backend in GpuBackend::ALL {
                if let Some(cfg) = BuildConfig::generate(platform, backend) {
                    configs.push(cfg);
                }
            }
        }
        Self { configs }
    }

    /// Filter the matrix to a single platform.
    pub fn for_platform(&self, p: TargetPlatform) -> Vec<&BuildConfig> {
        self.configs.iter().filter(|c| c.platform == p).collect()
    }

    /// Filter the matrix to a single backend.
    pub fn for_backend(&self, b: GpuBackend) -> Vec<&BuildConfig> {
        self.configs.iter().filter(|c| c.backend == b).collect()
    }
}

// ── Compile check ────────────────────────────────────────────────────────

/// Validation errors for a [`BuildConfig`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CompileCheckError {
    /// Backend is unavailable on the specified platform.
    UnavailableBackend {
        platform: TargetPlatform,
        backend: GpuBackend,
    },
    /// Target triple does not match the platform enum.
    TripleMismatch {
        expected: String,
        actual: String,
    },
    /// A required feature is missing from the feature list.
    MissingFeature(String),
}

impl fmt::Display for CompileCheckError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnavailableBackend { platform, backend } => {
                write!(f, "{backend} unavailable on {platform}")
            }
            Self::TripleMismatch { expected, actual } => {
                write!(
                    f,
                    "triple mismatch: expected {expected}, got {actual}"
                )
            }
            Self::MissingFeature(feat) => {
                write!(f, "missing required feature: {feat}")
            }
        }
    }
}

/// Validate that a [`BuildConfig`] is internally consistent.
pub fn compile_check(
    config: &BuildConfig,
) -> Result<(), Vec<CompileCheckError>> {
    let mut errors = Vec::new();

    // Backend must not be unavailable.
    if backend_availability(config.platform, config.backend)
        == Availability::Unavailable
    {
        errors.push(CompileCheckError::UnavailableBackend {
            platform: config.platform,
            backend: config.backend,
        });
    }

    // Triple must match.
    let expected = config.platform.target_triple();
    if config.target_triple != expected {
        errors.push(CompileCheckError::TripleMismatch {
            expected: expected.to_owned(),
            actual: config.target_triple.clone(),
        });
    }

    // Backend feature must be present.
    let feat = config.backend.feature_name();
    if !config.features.iter().any(|f| f == feat) {
        errors.push(CompileCheckError::MissingFeature(feat.to_owned()));
    }

    // GPU backends need the "gpu" umbrella feature.
    if !matches!(config.backend, GpuBackend::Cpu)
        && !config.features.iter().any(|f| f == "gpu")
    {
        errors.push(CompileCheckError::MissingFeature("gpu".to_owned()));
    }

    if errors.is_empty() { Ok(()) } else { Err(errors) }
}

// ── Platform detector ────────────────────────────────────────────────────

/// Detect the current platform at compile time.
pub const fn detect_compile_time() -> TargetPlatform {
    #[cfg(all(target_arch = "x86_64", target_os = "linux"))]
    {
        TargetPlatform::X86_64Linux
    }
    #[cfg(all(target_arch = "x86_64", target_os = "windows"))]
    {
        TargetPlatform::X86_64Windows
    }
    #[cfg(all(target_arch = "x86_64", target_os = "macos"))]
    {
        TargetPlatform::X86_64MacOS
    }
    #[cfg(all(target_arch = "aarch64", target_os = "linux"))]
    {
        TargetPlatform::Aarch64Linux
    }
    #[cfg(all(target_arch = "aarch64", target_os = "macos"))]
    {
        TargetPlatform::Aarch64MacOS
    }
    #[cfg(target_arch = "wasm32")]
    {
        TargetPlatform::Wasm32
    }
}

/// Detect the current platform at runtime (same result, but usable in
/// non-const contexts where the caller wants a value, not a compile-time
/// constant).
pub fn detect_runtime() -> TargetPlatform {
    detect_compile_time()
}

// ── Feature resolver ─────────────────────────────────────────────────────

/// Resolved set of Cargo features for a given target/backend pair.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedFeatures {
    pub features: Vec<String>,
}

/// Resolve which Cargo features should be enabled for a platform/backend.
///
/// Returns `None` if the combination is invalid (unavailable).
pub fn resolve_features(
    platform: TargetPlatform,
    backend: GpuBackend,
) -> Option<ResolvedFeatures> {
    if backend_availability(platform, backend) == Availability::Unavailable {
        return None;
    }

    let mut features = Vec::new();

    // Always include the backend's own feature.
    features.push(backend.feature_name().to_owned());

    // GPU umbrella.
    if !matches!(backend, GpuBackend::Cpu) {
        features.push("gpu".to_owned());
    }

    // Platform-specific extras.
    if platform.is_aarch64() && matches!(backend, GpuBackend::Cpu) {
        features.push("neon".to_owned());
    }

    Some(ResolvedFeatures { features })
}

// ── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── TargetPlatform ───────────────────────────────────────────────

    #[test]
    fn platform_all_has_six_entries() {
        assert_eq!(TargetPlatform::ALL.len(), 6);
    }

    #[test]
    fn platform_triples_are_unique() {
        let triples: Vec<_> =
            TargetPlatform::ALL.iter().map(|p| p.target_triple()).collect();
        let mut deduped = triples.clone();
        deduped.sort();
        deduped.dedup();
        assert_eq!(triples.len(), deduped.len());
    }

    #[test]
    fn x86_64_linux_triple() {
        assert_eq!(
            TargetPlatform::X86_64Linux.target_triple(),
            "x86_64-unknown-linux-gnu"
        );
    }

    #[test]
    fn x86_64_windows_triple() {
        assert_eq!(
            TargetPlatform::X86_64Windows.target_triple(),
            "x86_64-pc-windows-msvc"
        );
    }

    #[test]
    fn x86_64_macos_triple() {
        assert_eq!(
            TargetPlatform::X86_64MacOS.target_triple(),
            "x86_64-apple-darwin"
        );
    }

    #[test]
    fn aarch64_linux_triple() {
        assert_eq!(
            TargetPlatform::Aarch64Linux.target_triple(),
            "aarch64-unknown-linux-gnu"
        );
    }

    #[test]
    fn aarch64_macos_triple() {
        assert_eq!(
            TargetPlatform::Aarch64MacOS.target_triple(),
            "aarch64-apple-darwin"
        );
    }

    #[test]
    fn wasm32_triple() {
        assert_eq!(
            TargetPlatform::Wasm32.target_triple(),
            "wasm32-unknown-unknown"
        );
    }

    #[test]
    fn posix_platforms() {
        assert!(TargetPlatform::X86_64Linux.is_posix());
        assert!(TargetPlatform::X86_64MacOS.is_posix());
        assert!(TargetPlatform::Aarch64Linux.is_posix());
        assert!(TargetPlatform::Aarch64MacOS.is_posix());
        assert!(!TargetPlatform::X86_64Windows.is_posix());
        assert!(!TargetPlatform::Wasm32.is_posix());
    }

    #[test]
    fn macos_detection() {
        assert!(TargetPlatform::X86_64MacOS.is_macos());
        assert!(TargetPlatform::Aarch64MacOS.is_macos());
        assert!(!TargetPlatform::X86_64Linux.is_macos());
    }

    #[test]
    fn aarch64_detection() {
        assert!(TargetPlatform::Aarch64Linux.is_aarch64());
        assert!(TargetPlatform::Aarch64MacOS.is_aarch64());
        assert!(!TargetPlatform::X86_64Linux.is_aarch64());
    }

    #[test]
    fn platform_display_matches_triple() {
        for &p in TargetPlatform::ALL {
            assert_eq!(p.to_string(), p.target_triple());
        }
    }

    // ── GpuBackend ───────────────────────────────────────────────────

    #[test]
    fn backend_all_has_four_entries() {
        assert_eq!(GpuBackend::ALL.len(), 4);
    }

    #[test]
    fn backend_feature_names() {
        assert_eq!(GpuBackend::Cpu.feature_name(), "cpu");
        assert_eq!(GpuBackend::Cuda.feature_name(), "cuda");
        assert_eq!(GpuBackend::Metal.feature_name(), "metal");
        assert_eq!(GpuBackend::Vulkan.feature_name(), "vulkan");
    }

    #[test]
    fn backend_display_matches_feature() {
        for &b in GpuBackend::ALL {
            assert_eq!(b.to_string(), b.feature_name());
        }
    }

    // ── Availability ─────────────────────────────────────────────────

    #[test]
    fn cpu_available_everywhere() {
        for &p in TargetPlatform::ALL {
            assert_eq!(
                backend_availability(p, GpuBackend::Cpu),
                Availability::Available,
            );
        }
    }

    #[test]
    fn cuda_unavailable_on_wasm() {
        assert_eq!(
            backend_availability(
                TargetPlatform::Wasm32,
                GpuBackend::Cuda,
            ),
            Availability::Unavailable,
        );
    }

    #[test]
    fn cuda_unavailable_on_aarch64_macos() {
        assert_eq!(
            backend_availability(
                TargetPlatform::Aarch64MacOS,
                GpuBackend::Cuda,
            ),
            Availability::Unavailable,
        );
    }

    #[test]
    fn cuda_optional_on_x86_linux() {
        assert_eq!(
            backend_availability(
                TargetPlatform::X86_64Linux,
                GpuBackend::Cuda,
            ),
            Availability::Optional,
        );
    }

    #[test]
    fn cuda_optional_on_x86_windows() {
        assert_eq!(
            backend_availability(
                TargetPlatform::X86_64Windows,
                GpuBackend::Cuda,
            ),
            Availability::Optional,
        );
    }

    #[test]
    fn cuda_optional_on_aarch64_linux() {
        assert_eq!(
            backend_availability(
                TargetPlatform::Aarch64Linux,
                GpuBackend::Cuda,
            ),
            Availability::Optional,
        );
    }

    #[test]
    fn metal_available_on_macos() {
        assert_eq!(
            backend_availability(
                TargetPlatform::X86_64MacOS,
                GpuBackend::Metal,
            ),
            Availability::Available,
        );
        assert_eq!(
            backend_availability(
                TargetPlatform::Aarch64MacOS,
                GpuBackend::Metal,
            ),
            Availability::Available,
        );
    }

    #[test]
    fn metal_unavailable_off_macos() {
        assert_eq!(
            backend_availability(
                TargetPlatform::X86_64Linux,
                GpuBackend::Metal,
            ),
            Availability::Unavailable,
        );
        assert_eq!(
            backend_availability(
                TargetPlatform::X86_64Windows,
                GpuBackend::Metal,
            ),
            Availability::Unavailable,
        );
        assert_eq!(
            backend_availability(
                TargetPlatform::Wasm32,
                GpuBackend::Metal,
            ),
            Availability::Unavailable,
        );
    }

    #[test]
    fn vulkan_unavailable_on_wasm() {
        assert_eq!(
            backend_availability(
                TargetPlatform::Wasm32,
                GpuBackend::Vulkan,
            ),
            Availability::Unavailable,
        );
    }

    #[test]
    fn vulkan_optional_on_linux() {
        assert_eq!(
            backend_availability(
                TargetPlatform::X86_64Linux,
                GpuBackend::Vulkan,
            ),
            Availability::Optional,
        );
    }

    // ── PlatformCapabilities ─────────────────────────────────────────

    #[test]
    fn wasm_only_has_cpu() {
        let caps =
            PlatformCapabilities::for_platform(TargetPlatform::Wasm32);
        assert_eq!(caps.available, vec![GpuBackend::Cpu]);
        assert!(caps.optional.is_empty());
    }

    #[test]
    fn aarch64_macos_has_cpu_and_metal() {
        let caps = PlatformCapabilities::for_platform(
            TargetPlatform::Aarch64MacOS,
        );
        assert!(caps.available.contains(&GpuBackend::Cpu));
        assert!(caps.available.contains(&GpuBackend::Metal));
        assert!(!caps.available.contains(&GpuBackend::Cuda));
    }

    #[test]
    fn x86_linux_optional_includes_cuda_and_vulkan() {
        let caps = PlatformCapabilities::for_platform(
            TargetPlatform::X86_64Linux,
        );
        assert!(caps.optional.contains(&GpuBackend::Cuda));
        assert!(caps.optional.contains(&GpuBackend::Vulkan));
    }

    #[test]
    fn all_possible_is_superset_of_available() {
        for &p in TargetPlatform::ALL {
            let caps = PlatformCapabilities::for_platform(p);
            let all = caps.all_possible();
            for b in &caps.available {
                assert!(all.contains(b));
            }
        }
    }

    // ── BuildConfig ──────────────────────────────────────────────────

    #[test]
    fn generate_returns_none_for_unavailable() {
        assert!(BuildConfig::generate(
            TargetPlatform::Wasm32,
            GpuBackend::Cuda,
        )
        .is_none());
    }

    #[test]
    fn generate_cpu_on_linux() {
        let cfg = BuildConfig::generate(
            TargetPlatform::X86_64Linux,
            GpuBackend::Cpu,
        )
        .unwrap();
        assert_eq!(cfg.target_triple, "x86_64-unknown-linux-gnu");
        assert!(cfg.features.contains(&"cpu".to_owned()));
        assert!(!cfg.features.contains(&"gpu".to_owned()));
    }

    #[test]
    fn generate_cuda_includes_gpu_umbrella() {
        let cfg = BuildConfig::generate(
            TargetPlatform::X86_64Linux,
            GpuBackend::Cuda,
        )
        .unwrap();
        assert!(cfg.features.contains(&"gpu".to_owned()));
        assert!(cfg.features.contains(&"cuda".to_owned()));
    }

    #[test]
    fn cuda_config_has_cuda_env() {
        let cfg = BuildConfig::generate(
            TargetPlatform::X86_64Linux,
            GpuBackend::Cuda,
        )
        .unwrap();
        assert!(cfg.env_vars.iter().any(|(k, _)| k == "CUDA_HOME"));
    }

    #[test]
    fn metal_config_has_framework_flags() {
        let cfg = BuildConfig::generate(
            TargetPlatform::Aarch64MacOS,
            GpuBackend::Metal,
        )
        .unwrap();
        assert!(
            cfg.linker_flags.iter().any(|f| f == "-framework Metal")
        );
    }

    #[test]
    fn posix_configs_have_pthread() {
        let cfg = BuildConfig::generate(
            TargetPlatform::X86_64Linux,
            GpuBackend::Cpu,
        )
        .unwrap();
        assert!(cfg.linker_flags.contains(&"-lpthread".to_owned()));
    }

    #[test]
    fn windows_config_no_posix_flags() {
        let cfg = BuildConfig::generate(
            TargetPlatform::X86_64Windows,
            GpuBackend::Cpu,
        )
        .unwrap();
        assert!(!cfg.linker_flags.contains(&"-lpthread".to_owned()));
    }

    #[test]
    fn windows_config_has_cl_compiler() {
        let cfg = BuildConfig::generate(
            TargetPlatform::X86_64Windows,
            GpuBackend::Cpu,
        )
        .unwrap();
        assert!(
            cfg.env_vars.iter().any(|(k, v)| k == "CC" && v == "cl.exe")
        );
    }

    // ── CrossCompileMatrix ───────────────────────────────────────────

    #[test]
    fn matrix_has_no_unavailable_combos() {
        let matrix = CrossCompileMatrix::generate();
        for cfg in &matrix.configs {
            assert_ne!(
                backend_availability(cfg.platform, cfg.backend),
                Availability::Unavailable,
            );
        }
    }

    #[test]
    fn matrix_config_count() {
        let matrix = CrossCompileMatrix::generate();
        // 6 platforms × 4 backends minus unavailable combos.
        // Unavailable: wasm×cuda, wasm×metal, wasm×vulkan,
        //   aarch64-macos×cuda,
        //   x86-linux×metal, x86-win×metal, aarch64-linux×metal
        // = 7 unavailable → 24 - 7 = 17
        assert_eq!(matrix.configs.len(), 17);
    }

    #[test]
    fn matrix_for_platform_filters() {
        let matrix = CrossCompileMatrix::generate();
        let wasm = matrix.for_platform(TargetPlatform::Wasm32);
        assert_eq!(wasm.len(), 1); // only CPU
        assert_eq!(wasm[0].backend, GpuBackend::Cpu);
    }

    #[test]
    fn matrix_for_backend_filters() {
        let matrix = CrossCompileMatrix::generate();
        let metal = matrix.for_backend(GpuBackend::Metal);
        assert_eq!(metal.len(), 2); // x86_64-macos + aarch64-macos
    }

    #[test]
    fn matrix_cpu_on_all_platforms() {
        let matrix = CrossCompileMatrix::generate();
        let cpu = matrix.for_backend(GpuBackend::Cpu);
        assert_eq!(cpu.len(), TargetPlatform::ALL.len());
    }

    // ── CompileCheck ─────────────────────────────────────────────────

    #[test]
    fn valid_config_passes_check() {
        let cfg = BuildConfig::generate(
            TargetPlatform::X86_64Linux,
            GpuBackend::Cpu,
        )
        .unwrap();
        assert!(compile_check(&cfg).is_ok());
    }

    #[test]
    fn all_matrix_configs_pass_check() {
        let matrix = CrossCompileMatrix::generate();
        for cfg in &matrix.configs {
            assert!(
                compile_check(cfg).is_ok(),
                "check failed for {:?}/{:?}",
                cfg.platform,
                cfg.backend,
            );
        }
    }

    #[test]
    fn wrong_triple_fails_check() {
        let mut cfg = BuildConfig::generate(
            TargetPlatform::X86_64Linux,
            GpuBackend::Cpu,
        )
        .unwrap();
        cfg.target_triple = "wrong-triple".into();
        let errs = compile_check(&cfg).unwrap_err();
        assert!(errs.iter().any(|e| matches!(
            e,
            CompileCheckError::TripleMismatch { .. }
        )));
    }

    #[test]
    fn missing_feature_fails_check() {
        let mut cfg = BuildConfig::generate(
            TargetPlatform::X86_64Linux,
            GpuBackend::Cpu,
        )
        .unwrap();
        cfg.features.clear();
        let errs = compile_check(&cfg).unwrap_err();
        assert!(errs.iter().any(|e| matches!(
            e,
            CompileCheckError::MissingFeature(_)
        )));
    }

    #[test]
    fn gpu_missing_umbrella_fails() {
        let mut cfg = BuildConfig::generate(
            TargetPlatform::X86_64Linux,
            GpuBackend::Cuda,
        )
        .unwrap();
        cfg.features.retain(|f| f != "gpu");
        let errs = compile_check(&cfg).unwrap_err();
        assert!(errs.iter().any(|e| matches!(
            e,
            CompileCheckError::MissingFeature(f) if f == "gpu"
        )));
    }

    #[test]
    fn unavailable_backend_fails_check() {
        let cfg = BuildConfig {
            platform: TargetPlatform::Wasm32,
            backend: GpuBackend::Cuda,
            target_triple: "wasm32-unknown-unknown".into(),
            features: vec!["gpu".into(), "cuda".into()],
            linker_flags: vec![],
            env_vars: vec![],
        };
        let errs = compile_check(&cfg).unwrap_err();
        assert!(errs.iter().any(|e| matches!(
            e,
            CompileCheckError::UnavailableBackend { .. }
        )));
    }

    #[test]
    fn compile_check_error_display() {
        let e = CompileCheckError::UnavailableBackend {
            platform: TargetPlatform::Wasm32,
            backend: GpuBackend::Cuda,
        };
        let s = e.to_string();
        assert!(s.contains("cuda"));
        assert!(s.contains("wasm32"));
    }

    // ── PlatformDetector ─────────────────────────────────────────────

    #[test]
    fn compile_time_detection_returns_valid_platform() {
        let p = detect_compile_time();
        assert!(TargetPlatform::ALL.contains(&p));
    }

    #[test]
    fn runtime_detection_matches_compile_time() {
        assert_eq!(detect_compile_time(), detect_runtime());
    }

    // ── FeatureResolver ──────────────────────────────────────────────

    #[test]
    fn resolve_cpu_features() {
        let r = resolve_features(
            TargetPlatform::X86_64Linux,
            GpuBackend::Cpu,
        )
        .unwrap();
        assert!(r.features.contains(&"cpu".to_owned()));
        assert!(!r.features.contains(&"gpu".to_owned()));
    }

    #[test]
    fn resolve_cuda_features() {
        let r = resolve_features(
            TargetPlatform::X86_64Linux,
            GpuBackend::Cuda,
        )
        .unwrap();
        assert!(r.features.contains(&"cuda".to_owned()));
        assert!(r.features.contains(&"gpu".to_owned()));
    }

    #[test]
    fn resolve_unavailable_returns_none() {
        assert!(resolve_features(
            TargetPlatform::Wasm32,
            GpuBackend::Cuda,
        )
        .is_none());
    }

    #[test]
    fn resolve_metal_on_macos() {
        let r = resolve_features(
            TargetPlatform::Aarch64MacOS,
            GpuBackend::Metal,
        )
        .unwrap();
        assert!(r.features.contains(&"metal".to_owned()));
        assert!(r.features.contains(&"gpu".to_owned()));
    }

    #[test]
    fn resolve_aarch64_cpu_includes_neon() {
        let r = resolve_features(
            TargetPlatform::Aarch64Linux,
            GpuBackend::Cpu,
        )
        .unwrap();
        assert!(r.features.contains(&"neon".to_owned()));
    }

    #[test]
    fn resolve_x86_cpu_no_neon() {
        let r = resolve_features(
            TargetPlatform::X86_64Linux,
            GpuBackend::Cpu,
        )
        .unwrap();
        assert!(!r.features.contains(&"neon".to_owned()));
    }

    #[test]
    fn resolve_all_valid_combos_succeed() {
        for &p in TargetPlatform::ALL {
            for &b in GpuBackend::ALL {
                let result = resolve_features(p, b);
                if backend_availability(p, b) == Availability::Unavailable
                {
                    assert!(result.is_none());
                } else {
                    assert!(result.is_some());
                }
            }
        }
    }

    // ── Edge cases & integration ─────────────────────────────────────

    #[test]
    fn cuda_optional_on_x86_macos_egpu() {
        // x86_64-macos can use CUDA via eGPU.
        assert_eq!(
            backend_availability(
                TargetPlatform::X86_64MacOS,
                GpuBackend::Cuda,
            ),
            Availability::Optional,
        );
    }

    #[test]
    fn vulkan_optional_on_all_non_wasm() {
        for &p in TargetPlatform::ALL {
            if matches!(p, TargetPlatform::Wasm32) {
                continue;
            }
            assert_eq!(
                backend_availability(p, GpuBackend::Vulkan),
                Availability::Optional,
            );
        }
    }

    #[test]
    fn cuda_linker_flags_include_libraries() {
        let cfg = BuildConfig::generate(
            TargetPlatform::X86_64Linux,
            GpuBackend::Cuda,
        )
        .unwrap();
        assert!(cfg.linker_flags.contains(&"-lcuda".to_owned()));
        assert!(cfg.linker_flags.contains(&"-lcudart".to_owned()));
    }

    #[test]
    fn vulkan_linker_flags() {
        let cfg = BuildConfig::generate(
            TargetPlatform::X86_64Linux,
            GpuBackend::Vulkan,
        )
        .unwrap();
        assert!(cfg.linker_flags.contains(&"-lvulkan".to_owned()));
    }
}
