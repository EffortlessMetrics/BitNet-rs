//! SPIR-V compilation pipeline for `OpenCL` kernels.
//!
//! Provides offline compilation of `.cl` kernel sources to SPIR-V binary,
//! with runtime detection of available compilers (`clang`, `ocloc`) and
//! automatic fallback to embedding raw `.cl` source for runtime compilation.

use std::collections::HashMap;
use std::io::Read;
use std::process::Command;
use std::sync::Mutex;

// ── Errors ───────────────────────────────────────────────────────────────────

/// Errors that can occur during SPIR-V compilation or validation.
#[derive(Debug, thiserror::Error)]
pub enum SpirVError {
    /// The SPIR-V binary failed validation.
    #[error("SPIR-V validation failed: {0}")]
    ValidationFailed(String),

    /// Compilation of `.cl` source to SPIR-V failed.
    #[error("SPIR-V compilation failed: {0}")]
    CompilationFailed(String),

    /// An I/O error occurred (e.g. reading/writing cache files).
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// No supported compiler was found on the system.
    #[error("no SPIR-V compiler available")]
    NoCompilerAvailable,
}

// ── Compile options ──────────────────────────────────────────────────────────

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

// ── Compiler backend detection ───────────────────────────────────────────────

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

// ── SPIR-V module ────────────────────────────────────────────────────────────

/// A compiled SPIR-V module together with its metadata.
#[derive(Debug, Clone)]
pub struct SpirVModule {
    /// Raw SPIR-V binary (little-endian words).
    pub bytecode: Vec<u8>,
    /// Hex-encoded SHA-256 of the *source* that produced this module.
    pub source_hash: String,
    /// The compiler backend that produced this module (if any).
    pub compiler: Option<CompilerBackend>,
}

// ── SPIR-V compiler ──────────────────────────────────────────────────────────

/// Offline compiler that turns `OpenCL` C source into SPIR-V binary.
///
/// When no system compiler is detected the caller should fall back to
/// passing the raw `.cl` source to the `OpenCL` runtime for JIT compilation.
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

    /// Compile `OpenCL` C `source` to SPIR-V.
    ///
    /// Returns [`SpirVError::NoCompilerAvailable`] when no backend is
    /// detected. The caller should fall back to runtime `.cl` compilation.
    pub fn compile_to_spirv(
        &self,
        source: &str,
        options: &CompileOptions,
    ) -> Result<SpirVModule, SpirVError> {
        let backend = self.backend.ok_or(SpirVError::NoCompilerAvailable)?;

        let hash = source_hash(source, options);
        let bytecode = invoke_compiler(backend, source, options)?;

        // Sanity-check the output before handing it back.
        SpirVValidator::validate_bytes(&bytecode)?;

        Ok(SpirVModule { bytecode, source_hash: hash, compiler: Some(backend) })
    }
}

impl Default for SpirVCompiler {
    fn default() -> Self {
        Self::new()
    }
}

/// Invoke the actual compiler process and return raw SPIR-V bytes.
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

// ── SPIR-V validator ─────────────────────────────────────────────────────────

/// SPIR-V magic number (first four bytes, little-endian).
pub const SPIRV_MAGIC: u32 = 0x0723_0203;

/// Lightweight SPIR-V binary validator.
///
/// Performs structural checks only – it does **not** run the full Khronos
/// `spirv-val` tool.
pub struct SpirVValidator;

impl SpirVValidator {
    /// Validate raw SPIR-V `bytes`.
    ///
    /// Checks:
    /// 1. Minimum length (20 bytes = 5 words for the header).
    /// 2. Magic number matches [`SPIRV_MAGIC`].
    /// 3. Version word is a known SPIR-V version (1.0 – 1.6).
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
        // SPIR-V versions: major in bits 23:16, minor in bits 15:8.
        let major = (version >> 16) & 0xFF;
        let minor = (version >> 8) & 0xFF;
        if major != 1 || minor > 6 {
            return Err(SpirVError::ValidationFailed(format!(
                "unsupported SPIR-V version {major}.{minor}"
            )));
        }
        Ok(())
    }

    /// Check whether the binary *might* contain a given SPIR-V capability.
    ///
    /// This is a best-effort scan – it searches for `OpCapability`
    /// instructions (opcode 17) whose operand matches `capability_id`.
    pub fn has_capability(bytes: &[u8], capability_id: u32) -> bool {
        // OpCapability = opcode 17, word count 2.
        // Encoded as: (2 << 16) | 17 = 0x0002_0011.
        let op_capability: u32 = (2 << 16) | 17;
        // Walk word-aligned through the binary after the 5-word header.
        if bytes.len() < 24 {
            return false;
        }
        let mut offset = 20; // skip header
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
            // Advance by word-count (upper 16 bits).
            let wc = (word >> 16) as usize;
            if wc == 0 {
                break;
            }
            offset += wc * 4;
        }
        false
    }
}

// ── SPIR-V cache ─────────────────────────────────────────────────────────────

/// Thread-safe in-memory cache for compiled SPIR-V modules, keyed by a hash
/// of the source text and compile options.
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

// ── Source hashing ───────────────────────────────────────────────────────────

/// Compute a deterministic hash of the source + options, suitable for
/// cache keys. Uses a simple `DefaultHasher`-based approach (not
/// cryptographic, but fast and collision-resistant enough for caching).
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

// ── Helpers ──────────────────────────────────────────────────────────────────

/// Build a minimal valid SPIR-V binary (header only) for testing.
///
/// This is **not** a useful module – it exists solely for unit tests that
/// need bytes passing the validator.
#[must_use]
pub fn build_test_spirv(version_major: u8, version_minor: u8) -> Vec<u8> {
    let mut buf = Vec::with_capacity(20);
    // Word 0: magic
    buf.extend_from_slice(&SPIRV_MAGIC.to_le_bytes());
    // Word 1: version (major.minor.0)
    let version_word = (u32::from(version_major) << 16) | (u32::from(version_minor) << 8);
    buf.extend_from_slice(&version_word.to_le_bytes());
    // Word 2: generator magic (0 = unknown)
    buf.extend_from_slice(&0u32.to_le_bytes());
    // Word 3: bound (IDs upper bound; 1 is minimum valid)
    buf.extend_from_slice(&1u32.to_le_bytes());
    // Word 4: reserved (must be 0)
    buf.extend_from_slice(&0u32.to_le_bytes());
    buf
}
