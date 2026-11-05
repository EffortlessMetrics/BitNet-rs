//! SPIR-V compilation pipeline and utilities.
//!
//! Provides offline compilation of `OpenCL` kernel sources to SPIR-V binary,
//! runtime compiler detection (`clang`, `ocloc`), lightweight validation,
//! and a thread-safe in-memory cache.

use std::collections::HashMap;
use std::io::Read;
use std::process::Command;
use std::sync::Mutex;

/// Errors that can occur during SPIR-V compilation or validation.
#[derive(Debug, thiserror::Error)]
pub enum SpirVError {
    /// The SPIR-V binary failed validation.
    #[error("SPIR-V validation failed: {0}")]
    ValidationFailed(String),

    /// Compilation of source to SPIR-V failed.
    #[error("SPIR-V compilation failed: {0}")]
    CompilationFailed(String),

    /// An I/O error occurred (e.g. reading/writing cache files).
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// No supported compiler was found on the system.
    #[error("no SPIR-V compiler available")]
    NoCompilerAvailable,
}

/// Optimization level for SPIR-V compilation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OptimizationLevel {
    /// No optimization (`-O0`).
    None,
    /// Basic optimization (`-O1`).
    Basic,
    /// Full optimization (`-O2`).
    Full,
}

impl OptimizationLevel {
    const fn as_flag(self) -> &'static str {
        match self {
            Self::None => "-O0",
            Self::Basic => "-O1",
            Self::Full => "-O2",
        }
    }
}

/// Options controlling SPIR-V compilation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompileOptions {
    /// Target device hint (informational; not all compilers use this).
    pub target_device: Option<String>,
    /// Optimisation level for the SPIR-V compiler.
    pub optimization_level: OptimizationLevel,
    /// Preprocessor defines passed to the compiler (`-D`).
    pub defines: Vec<(String, String)>,
}

impl Default for CompileOptions {
    fn default() -> Self {
        Self {
            target_device: None,
            optimization_level: OptimizationLevel::Full,
            defines: Vec::new(),
        }
    }
}

/// Which compiler backend is available on this system.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompilerBackend {
    /// LLVM/Clang with SPIR-V target support.
    Clang,
    /// Intel Graphics Offline Compiler.
    Ocloc,
}

/// Probe the system for a usable SPIR-V compiler.
///
/// Checks `clang` first, then `ocloc`. Returns `None` when neither is found.
#[must_use]
pub fn detect_compiler() -> Option<CompilerBackend> {
    if probe_clang() {
        Some(CompilerBackend::Clang)
    } else if probe_ocloc() {
        Some(CompilerBackend::Ocloc)
    } else {
        None
    }
}

fn probe_clang() -> bool {
    Command::new("clang")
        .arg("--version")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .is_ok_and(|s| s.success())
}

fn probe_ocloc() -> bool {
    Command::new("ocloc")
        .arg("--version")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .is_ok_and(|s| s.success())
}

/// A compiled SPIR-V module together with its metadata.
#[derive(Debug, Clone)]
pub struct SpirVModule {
    /// Raw SPIR-V binary (little-endian words).
    pub bytecode: Vec<u8>,
    /// Hex-encoded hash of the source and options that produced this module.
    pub source_hash: String,
    /// The compiler backend that produced this module (if any).
    pub compiler: Option<CompilerBackend>,
}

/// Offline compiler that turns source into SPIR-V binary.
pub struct SpirVCompiler {
    backend: Option<CompilerBackend>,
}

impl SpirVCompiler {
    /// Create a new compiler, auto-detecting available backends.
    #[must_use]
    pub fn new() -> Self {
        Self { backend: detect_compiler() }
    }

    /// Create a compiler with an explicit backend (useful for testing).
    #[must_use]
    pub const fn with_backend(backend: Option<CompilerBackend>) -> Self {
        Self { backend }
    }

    /// Returns the detected backend, if any.
    #[must_use]
    pub const fn backend(&self) -> Option<CompilerBackend> {
        self.backend
    }

    /// Compile source to SPIR-V.
    pub fn compile_to_spirv(
        &self,
        source: &str,
        options: &CompileOptions,
    ) -> Result<SpirVModule, SpirVError> {
        let backend = self.backend.ok_or(SpirVError::NoCompilerAvailable)?;

        let hash = source_hash(source, options);
        let bytecode = invoke_compiler(backend, source, options)?;

        SpirVValidator::validate_bytes(&bytecode)?;

        Ok(SpirVModule { bytecode, source_hash: hash, compiler: Some(backend) })
    }
}

impl Default for SpirVCompiler {
    fn default() -> Self {
        Self::new()
    }
}

fn invoke_compiler(
    backend: CompilerBackend,
    source: &str,
    options: &CompileOptions,
) -> Result<Vec<u8>, SpirVError> {
    let dir = tempfile::tempdir().map_err(SpirVError::Io)?;

    let src_path = dir.path().join("kernel.cl");
    let out_path = dir.path().join("kernel.spv");
    std::fs::write(&src_path, source)?;

    let status = match backend {
        CompilerBackend::Clang => build_clang_command(&src_path, &out_path, options)
            .status()
            .map_err(|e| SpirVError::CompilationFailed(format!("failed to launch clang: {e}")))?,
        CompilerBackend::Ocloc => build_ocloc_command(&src_path, &out_path, options)
            .status()
            .map_err(|e| SpirVError::CompilationFailed(format!("failed to launch ocloc: {e}")))?,
    };

    if !status.success() {
        return Err(SpirVError::CompilationFailed(format!("{backend:?} exited with {status}")));
    }

    let mut buf = Vec::new();
    std::fs::File::open(&out_path)?.read_to_end(&mut buf)?;
    Ok(buf)
}

fn build_clang_command(
    src: &std::path::Path,
    out: &std::path::Path,
    opts: &CompileOptions,
) -> Command {
    let mut cmd = Command::new("clang");
    cmd.args([
        "-cc1",
        "-triple",
        "spir64-unknown-unknown",
        "-emit-spirv",
        opts.optimization_level.as_flag(),
        "-o",
    ]);
    cmd.arg(out);
    for (k, v) in &opts.defines {
        cmd.arg(format!("-D{k}={v}"));
    }
    cmd.arg(src);
    cmd
}

fn build_ocloc_command(
    src: &std::path::Path,
    out: &std::path::Path,
    opts: &CompileOptions,
) -> Command {
    let mut cmd = Command::new("ocloc");
    cmd.args(["compile", "-file"]);
    cmd.arg(src);
    cmd.args(["-spirv_input", "-output"]);
    cmd.arg(out);
    if let Some(dev) = &opts.target_device {
        cmd.args(["-device", dev]);
    }
    for (k, v) in &opts.defines {
        cmd.arg(format!("-D{k}={v}"));
    }
    cmd
}

/// SPIR-V magic number (first four bytes, little-endian).
pub const SPIRV_MAGIC: u32 = 0x0723_0203;

/// Lightweight SPIR-V binary validator.
pub struct SpirVValidator;

impl SpirVValidator {
    /// Validate raw SPIR-V bytes.
    pub fn validate_bytes(bytes: &[u8]) -> Result<(), SpirVError> {
        Self::check_length(bytes)?;
        Self::check_magic(bytes)?;
        Self::check_version(bytes)?;
        Ok(())
    }

    /// Validate the magic number only.
    pub fn check_magic(bytes: &[u8]) -> Result<(), SpirVError> {
        if bytes.len() < 4 {
            return Err(SpirVError::ValidationFailed("too short for magic number".into()));
        }
        let magic = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        if magic != SPIRV_MAGIC {
            return Err(SpirVError::ValidationFailed(format!(
                "bad magic: expected 0x{SPIRV_MAGIC:08X}, got 0x{magic:08X}"
            )));
        }
        Ok(())
    }

    /// Validate minimum length (5 header words = 20 bytes).
    pub fn check_length(bytes: &[u8]) -> Result<(), SpirVError> {
        if bytes.len() < 20 {
            return Err(SpirVError::ValidationFailed(format!(
                "SPIR-V binary too short: {} bytes (minimum 20)",
                bytes.len()
            )));
        }
        Ok(())
    }

    /// Validate the version word (second 32-bit word).
    pub fn check_version(bytes: &[u8]) -> Result<(), SpirVError> {
        if bytes.len() < 8 {
            return Err(SpirVError::ValidationFailed("too short for version word".into()));
        }
        let version = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
        let major = (version >> 16) & 0xFF;
        let minor = (version >> 8) & 0xFF;
        if major != 1 || minor > 6 {
            return Err(SpirVError::ValidationFailed(format!(
                "unsupported SPIR-V version {major}.{minor}"
            )));
        }
        Ok(())
    }

    /// Check whether the binary might contain a given SPIR-V capability.
    #[must_use]
    pub fn has_capability(bytes: &[u8], capability_id: u32) -> bool {
        let op_capability: u32 = (2 << 16) | 0x0011;
        if bytes.len() < 24 {
            return false;
        }
        let mut offset = 20;
        while offset + 8 <= bytes.len() {
            let word = u32::from_le_bytes([
                bytes[offset],
                bytes[offset + 1],
                bytes[offset + 2],
                bytes[offset + 3],
            ]);
            if word == op_capability {
                let operand = u32::from_le_bytes([
                    bytes[offset + 4],
                    bytes[offset + 5],
                    bytes[offset + 6],
                    bytes[offset + 7],
                ]);
                if operand == capability_id {
                    return true;
                }
            }
            let wc = (word >> 16) as usize;
            if wc == 0 {
                break;
            }
            offset += wc * 4;
        }
        false
    }
}

/// Thread-safe in-memory cache for compiled SPIR-V modules.
pub struct SpirVCache {
    entries: Mutex<HashMap<String, SpirVModule>>,
}

impl SpirVCache {
    /// Create a new empty cache.
    #[must_use]
    pub fn new() -> Self {
        Self { entries: Mutex::new(HashMap::new()) }
    }

    /// Look up a cached module by its source hash.
    pub fn get(&self, hash: &str) -> Option<SpirVModule> {
        self.entries.lock().expect("spirv cache lock poisoned").get(hash).cloned()
    }

    /// Store a compiled module in the cache.
    pub fn insert(&self, module: SpirVModule) {
        self.entries
            .lock()
            .expect("spirv cache lock poisoned")
            .insert(module.source_hash.clone(), module);
    }

    /// Number of entries currently in the cache.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.lock().expect("spirv cache lock poisoned").len()
    }

    /// Returns `true` if the cache is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Remove all entries from the cache.
    pub fn clear(&self) {
        self.entries.lock().expect("spirv cache lock poisoned").clear();
    }
}

impl Default for SpirVCache {
    fn default() -> Self {
        Self::new()
    }
}

/// Compute a deterministic hash of the source + options, suitable for cache keys.
#[must_use]
pub fn source_hash(source: &str, options: &CompileOptions) -> String {
    use std::hash::{Hash, Hasher};

    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    source.hash(&mut hasher);
    options.optimization_level.hash(&mut hasher);
    options.target_device.hash(&mut hasher);
    for (k, v) in &options.defines {
        k.hash(&mut hasher);
        v.hash(&mut hasher);
    }
    format!("{:016x}", hasher.finish())
}

/// Build a minimal valid SPIR-V binary (header only) for testing.
#[must_use]
pub fn build_test_spirv(version_major: u8, version_minor: u8) -> Vec<u8> {
    let mut buf = Vec::with_capacity(20);
    buf.extend_from_slice(&SPIRV_MAGIC.to_le_bytes());
    let version_word = (u32::from(version_major) << 16) | (u32::from(version_minor) << 8);
    buf.extend_from_slice(&version_word.to_le_bytes());
    buf.extend_from_slice(&0u32.to_le_bytes());
    buf.extend_from_slice(&1u32.to_le_bytes());
    buf.extend_from_slice(&0u32.to_le_bytes());
    buf
}

#[cfg(test)]
mod tests {
    use super::*;

    fn module(hash: &str) -> SpirVModule {
        SpirVModule {
            bytecode: build_test_spirv(1, 0),
            source_hash: hash.to_string(),
            compiler: Some(CompilerBackend::Clang),
        }
    }

    fn validation_message(result: Result<(), SpirVError>) -> Option<String> {
        match result {
            Err(SpirVError::ValidationFailed(msg)) => Some(msg),
            _ => None,
        }
    }

    #[test]
    fn optimization_level_flag_strings() {
        assert_eq!(OptimizationLevel::None.as_flag(), "-O0");
        assert_eq!(OptimizationLevel::Basic.as_flag(), "-O1");
        assert_eq!(OptimizationLevel::Full.as_flag(), "-O2");
    }

    #[test]
    fn compile_options_default_values() {
        let opts = CompileOptions::default();
        assert!(opts.target_device.is_none());
        assert_eq!(opts.optimization_level, OptimizationLevel::Full);
        assert!(opts.defines.is_empty());
    }

    #[test]
    fn spirv_magic_constant_value() {
        assert_eq!(SPIRV_MAGIC, 0x0723_0203);
    }

    #[test]
    fn check_magic_accepts_valid_header() {
        let bin = build_test_spirv(1, 0);
        assert!(SpirVValidator::check_magic(&bin).is_ok());
    }

    #[test]
    fn check_magic_rejects_wrong_magic() {
        let mut bin = build_test_spirv(1, 0);
        bin[0] ^= 0xFF;
        let msg = validation_message(SpirVValidator::check_magic(&bin));
        assert!(msg.as_deref().is_some_and(|msg| msg.contains("bad magic")), "got: {msg:?}");
    }

    #[test]
    fn check_magic_rejects_too_short() {
        let msg = validation_message(SpirVValidator::check_magic(&[0x03, 0x02, 0x23]));
        assert!(msg.as_deref().is_some_and(|msg| msg.contains("too short")), "got: {msg:?}");
    }

    #[test]
    fn check_length_accepts_20_bytes() {
        let bin = build_test_spirv(1, 0);
        assert_eq!(bin.len(), 20);
        assert!(SpirVValidator::check_length(&bin).is_ok());
    }

    #[test]
    fn check_length_rejects_short_input() {
        let msg = validation_message(SpirVValidator::check_length(&[0u8; 19]));
        assert!(msg.as_deref().is_some_and(|msg| msg.contains("19 bytes")), "got: {msg:?}");
    }

    #[test]
    fn check_version_accepts_versions_1_0_through_1_6() {
        for minor in 0..=6 {
            let bin = build_test_spirv(1, minor);
            assert!(
                SpirVValidator::check_version(&bin).is_ok(),
                "version 1.{minor} should be accepted",
            );
        }
    }

    #[test]
    fn check_version_rejects_unsupported_versions() {
        // 1.7 too new
        let bin = build_test_spirv(1, 7);
        assert!(SpirVValidator::check_version(&bin).is_err());
        // 2.0 wrong major
        let bin = build_test_spirv(2, 0);
        let msg = validation_message(SpirVValidator::check_version(&bin));
        assert!(msg.as_deref().is_some_and(|msg| msg.contains("unsupported")), "got: {msg:?}");
    }

    #[test]
    fn check_version_rejects_too_short_for_version_word() {
        let msg = validation_message(SpirVValidator::check_version(&[0u8; 7]));
        assert!(msg.as_deref().is_some_and(|msg| msg.contains("version word")), "got: {msg:?}");
    }

    #[test]
    fn validate_bytes_passes_for_synthetic_header() {
        let bin = build_test_spirv(1, 3);
        assert!(SpirVValidator::validate_bytes(&bin).is_ok());
    }

    #[test]
    fn validate_bytes_fails_on_bad_magic_after_length_passes() {
        let mut bin = build_test_spirv(1, 0);
        // Corrupt magic but keep full length.
        bin[3] = 0;
        assert!(SpirVValidator::validate_bytes(&bin).is_err());
    }

    #[test]
    fn build_test_spirv_round_trip_header_layout() {
        let bin = build_test_spirv(1, 5);
        assert_eq!(bin.len(), 20);

        let magic = u32::from_le_bytes([bin[0], bin[1], bin[2], bin[3]]);
        assert_eq!(magic, SPIRV_MAGIC);

        let version = u32::from_le_bytes([bin[4], bin[5], bin[6], bin[7]]);
        assert_eq!((version >> 16) & 0xFF, 1);
        assert_eq!((version >> 8) & 0xFF, 5);
    }

    #[test]
    fn has_capability_finds_present_capability() {
        // Build a 28-byte binary: 20-byte header + 1 OpCapability instruction (2 words = 8 bytes).
        let mut bin = build_test_spirv(1, 0);
        let op_capability: u32 = (2 << 16) | 0x0011;
        bin.extend_from_slice(&op_capability.to_le_bytes());
        bin.extend_from_slice(&42u32.to_le_bytes());
        assert!(SpirVValidator::has_capability(&bin, 42));
    }

    #[test]
    fn has_capability_returns_false_when_absent() {
        let mut bin = build_test_spirv(1, 0);
        let op_capability: u32 = (2 << 16) | 0x0011;
        bin.extend_from_slice(&op_capability.to_le_bytes());
        bin.extend_from_slice(&7u32.to_le_bytes());
        assert!(!SpirVValidator::has_capability(&bin, 42));
    }

    #[test]
    fn has_capability_returns_false_on_short_binary() {
        let bin = build_test_spirv(1, 0); // 20 bytes, < 24
        assert!(!SpirVValidator::has_capability(&bin, 1));
    }

    #[test]
    fn source_hash_is_deterministic() {
        let opts = CompileOptions::default();
        let a = source_hash("kernel void k() {}", &opts);
        let b = source_hash("kernel void k() {}", &opts);
        assert_eq!(a, b);
        assert_eq!(a.len(), 16);
    }

    #[test]
    fn source_hash_changes_with_source() {
        let opts = CompileOptions::default();
        assert_ne!(source_hash("a", &opts), source_hash("b", &opts));
    }

    #[test]
    fn source_hash_changes_with_optimization_level() {
        let mut opts = CompileOptions::default();
        let h1 = source_hash("src", &opts);
        opts.optimization_level = OptimizationLevel::None;
        let h2 = source_hash("src", &opts);
        assert_ne!(h1, h2);
    }

    #[test]
    fn source_hash_changes_with_defines_and_target_device() {
        let mut opts = CompileOptions::default();
        let base = source_hash("src", &opts);
        opts.defines.push(("FOO".into(), "1".into()));
        let with_define = source_hash("src", &opts);
        assert_ne!(base, with_define);

        opts.target_device = Some("intel".into());
        let with_device = source_hash("src", &opts);
        assert_ne!(with_define, with_device);
    }

    #[test]
    fn cache_new_is_empty() {
        let c = SpirVCache::new();
        assert!(c.is_empty());
        assert_eq!(c.len(), 0);
    }

    #[test]
    fn cache_default_matches_new() {
        let c = SpirVCache::default();
        assert!(c.is_empty());
    }

    #[test]
    fn cache_insert_get_and_len() {
        let c = SpirVCache::new();
        c.insert(module("h1"));
        assert_eq!(c.len(), 1);
        assert!(!c.is_empty());

        let got = c.get("h1");
        assert!(got.as_ref().is_some_and(|module| module.source_hash == "h1"), "got: {got:?}");

        assert!(c.get("missing").is_none());
    }

    #[test]
    fn cache_insert_replaces_same_hash() {
        let c = SpirVCache::new();
        c.insert(module("h"));
        c.insert(module("h"));
        assert_eq!(c.len(), 1);
    }

    #[test]
    fn cache_clear_empties_entries() {
        let c = SpirVCache::new();
        c.insert(module("a"));
        c.insert(module("b"));
        assert_eq!(c.len(), 2);
        c.clear();
        assert!(c.is_empty());
    }

    #[test]
    fn compiler_with_backend_overrides_detection() {
        let compiler = SpirVCompiler::with_backend(Some(CompilerBackend::Ocloc));
        assert_eq!(compiler.backend(), Some(CompilerBackend::Ocloc));

        let none_compiler = SpirVCompiler::with_backend(None);
        assert!(none_compiler.backend().is_none());
    }

    #[test]
    fn compile_to_spirv_fails_without_backend() {
        let compiler = SpirVCompiler::with_backend(None);
        let result = compiler.compile_to_spirv("kernel void k() {}", &CompileOptions::default());
        assert!(matches!(result, Err(SpirVError::NoCompilerAvailable)));
    }
}
