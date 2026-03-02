//! OpenCL kernel compilation, caching, and specialization.
//!
//! Provides a CPU-reference compilation pipeline that mirrors the real OpenCL
//! `clBuildProgram` workflow: preprocess → specialise → compile → cache.
//! When no OpenCL runtime is available the "compilation" is simulated by hashing
//! the preprocessed source and producing a deterministic binary blob, allowing
//! all upper layers (cache, warmup, stats) to be exercised without hardware.

use std::collections::HashMap;
use std::fmt;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

// ---------------------------------------------------------------------------
// KernelSource
// ---------------------------------------------------------------------------

/// OpenCL C source code with preprocessing directives.
#[derive(Debug, Clone)]
pub struct KernelSource {
    /// Raw OpenCL C source text.
    source: String,
    /// User-supplied `#define` directives prepended before compilation.
    defines: Vec<(String, String)>,
    /// Paths searched when resolving `#include` directives.
    include_paths: Vec<PathBuf>,
    /// Optional human-readable name (used in error messages).
    name: Option<String>,
}

impl KernelSource {
    /// Create a new kernel source from raw OpenCL C text.
    pub fn new(source: impl Into<String>) -> Self {
        Self { source: source.into(), defines: Vec::new(), include_paths: Vec::new(), name: None }
    }

    /// Set a human-readable name for this source.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Add a preprocessor `#define NAME VALUE`.
    pub fn define(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.defines.push((name.into(), value.into()));
        self
    }

    /// Add an include search path.
    pub fn include_path(mut self, path: impl Into<PathBuf>) -> Self {
        self.include_paths.push(path.into());
        self
    }

    /// Raw source text.
    pub fn source(&self) -> &str {
        &self.source
    }

    /// User-defined preprocessor defines.
    pub fn defines(&self) -> &[(String, String)] {
        &self.defines
    }

    /// Include search paths.
    pub fn include_paths(&self) -> &[PathBuf] {
        &self.include_paths
    }

    /// Human-readable name (if set).
    pub fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    /// Build the full preprocessor preamble (`#define` lines).
    pub fn preamble(&self) -> String {
        let mut out = String::new();
        for (k, v) in &self.defines {
            out.push_str(&format!("#define {k} {v}\n"));
        }
        out
    }

    /// Return the source with the define preamble prepended.
    pub fn full_source(&self) -> String {
        let preamble = self.preamble();
        if preamble.is_empty() {
            self.source.clone()
        } else {
            format!("{preamble}{}", self.source)
        }
    }

    /// Deterministic 64-bit hash of the full (preamble + source) text.
    pub fn content_hash(&self) -> u64 {
        let mut hasher = DefaultHasher::new();
        self.full_source().hash(&mut hasher);
        hasher.finish()
    }
}

// ---------------------------------------------------------------------------
// CompilationOptions
// ---------------------------------------------------------------------------

/// Flags passed to the OpenCL compiler (`clBuildProgram` build-options string).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CompilationOptions {
    /// Raw flag strings (e.g. `"-cl-fast-relaxed-math"`).
    flags: Vec<String>,
    /// Target OpenCL version (e.g. `"2.0"`).
    target_version: String,
    /// Optimization level: 0 = none, 1 = default, 2 = aggressive.
    opt_level: u8,
}

impl CompilationOptions {
    pub fn new() -> Self {
        Self { flags: Vec::new(), target_version: "1.2".into(), opt_level: 1 }
    }

    /// Enable `-cl-fast-relaxed-math`.
    pub fn fast_math(mut self) -> Self {
        self.flags.push("-cl-fast-relaxed-math".into());
        self
    }

    /// Enable `-cl-mad-enable`.
    pub fn mad_enable(mut self) -> Self {
        self.flags.push("-cl-mad-enable".into());
        self
    }

    /// Enable `-cl-unsafe-math-optimizations`.
    pub fn unsafe_math(mut self) -> Self {
        self.flags.push("-cl-unsafe-math-optimizations".into());
        self
    }

    /// Enable `-cl-no-signed-zeros`.
    pub fn no_signed_zeros(mut self) -> Self {
        self.flags.push("-cl-no-signed-zeros".into());
        self
    }

    /// Add an arbitrary flag.
    pub fn flag(mut self, f: impl Into<String>) -> Self {
        self.flags.push(f.into());
        self
    }

    /// Set the target OpenCL version string.
    pub fn target_version(mut self, v: impl Into<String>) -> Self {
        self.target_version = v.into();
        self
    }

    /// Set optimization level (0–2).
    pub fn opt_level(mut self, level: u8) -> Self {
        self.opt_level = level.min(2);
        self
    }

    /// Produce the single build-options string.
    pub fn to_build_string(&self) -> String {
        let mut parts = self.flags.clone();
        if self.opt_level == 0 {
            parts.push("-cl-opt-disable".into());
        }
        parts.join(" ")
    }

    /// Individual flags.
    pub fn flags(&self) -> &[String] {
        &self.flags
    }

    /// Target OpenCL version.
    pub fn version(&self) -> &str {
        &self.target_version
    }

    /// Deterministic 64-bit hash of all options.
    pub fn options_hash(&self) -> u64 {
        let mut hasher = DefaultHasher::new();
        self.hash(&mut hasher);
        hasher.finish()
    }
}

impl Default for CompilationOptions {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for CompilationOptions {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_build_string())
    }
}

// ---------------------------------------------------------------------------
// CompilationError
// ---------------------------------------------------------------------------

/// A single error/warning location parsed from an OpenCL compiler log.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DiagnosticLocation {
    /// Source file (or `"<kernel>"` for inline source).
    pub file: String,
    /// 1-based line number.
    pub line: u32,
    /// 1-based column number (0 if unknown).
    pub column: u32,
}

/// Severity of a single diagnostic.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiagnosticSeverity {
    Error,
    Warning,
    Note,
}

impl fmt::Display for DiagnosticSeverity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Error => write!(f, "error"),
            Self::Warning => write!(f, "warning"),
            Self::Note => write!(f, "note"),
        }
    }
}

/// One parsed diagnostic from the compiler log.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Diagnostic {
    pub severity: DiagnosticSeverity,
    pub location: Option<DiagnosticLocation>,
    pub message: String,
}

impl fmt::Display for Diagnostic {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(loc) = &self.location {
            write!(f, "{}:{}:{}: {}: {}", loc.file, loc.line, loc.column, self.severity, self.message)
        } else {
            write!(f, "{}: {}", self.severity, self.message)
        }
    }
}

/// Error returned when compilation fails.
#[derive(Debug, Clone)]
pub struct CompilationError {
    /// Raw build log text.
    pub raw_log: String,
    /// Parsed diagnostics.
    pub diagnostics: Vec<Diagnostic>,
    /// Source that failed to compile (if available).
    pub source_name: Option<String>,
}

impl CompilationError {
    /// Parse a raw OpenCL compiler log into structured diagnostics.
    ///
    /// Recognises the common format:
    /// `<file>:<line>:<col>: error: <message>`
    pub fn from_log(raw_log: impl Into<String>, source_name: Option<String>) -> Self {
        let raw_log = raw_log.into();
        let diagnostics = Self::parse_diagnostics(&raw_log);
        Self { raw_log, diagnostics, source_name }
    }

    fn parse_diagnostics(log: &str) -> Vec<Diagnostic> {
        let mut diags = Vec::new();
        for line in log.lines() {
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            if let Some(d) = Self::parse_line(trimmed) {
                diags.push(d);
            } else {
                // Treat as a note without location.
                diags.push(Diagnostic {
                    severity: DiagnosticSeverity::Note,
                    location: None,
                    message: trimmed.to_string(),
                });
            }
        }
        diags
    }

    /// Try to parse `file:line:col: severity: msg`.
    fn parse_line(line: &str) -> Option<Diagnostic> {
        // Split on ": " to find the severity marker.
        let severity_markers = ["error:", "warning:", "note:"];
        for marker in severity_markers {
            if let Some(idx) = line.find(&format!(": {marker}")) {
                let prefix = &line[..idx];
                let message = line[idx + 2 + marker.len()..].trim().to_string();
                let severity = match marker {
                    "error:" => DiagnosticSeverity::Error,
                    "warning:" => DiagnosticSeverity::Warning,
                    _ => DiagnosticSeverity::Note,
                };
                let location = Self::parse_location(prefix);
                return Some(Diagnostic { severity, location, message });
            }
        }
        None
    }

    /// Parse `file:line:col` or `file:line`.
    fn parse_location(s: &str) -> Option<DiagnosticLocation> {
        let parts: Vec<&str> = s.rsplitn(3, ':').collect();
        match parts.len() {
            // col:line:file (reversed)
            3 => {
                let col = parts[0].trim().parse::<u32>().ok()?;
                let line = parts[1].trim().parse::<u32>().ok()?;
                let file = parts[2].trim().to_string();
                Some(DiagnosticLocation { file, line, column: col })
            }
            // line:file (reversed)
            2 => {
                let line = parts[0].trim().parse::<u32>().ok()?;
                let file = parts[1].trim().to_string();
                Some(DiagnosticLocation { file, line, column: 0 })
            }
            _ => None,
        }
    }

    /// Number of errors.
    pub fn error_count(&self) -> usize {
        self.diagnostics.iter().filter(|d| d.severity == DiagnosticSeverity::Error).count()
    }

    /// Number of warnings.
    pub fn warning_count(&self) -> usize {
        self.diagnostics.iter().filter(|d| d.severity == DiagnosticSeverity::Warning).count()
    }

    /// `true` if any diagnostic is an error.
    pub fn has_errors(&self) -> bool {
        self.diagnostics.iter().any(|d| d.severity == DiagnosticSeverity::Error)
    }
}

impl fmt::Display for CompilationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(name) = &self.source_name {
            writeln!(f, "compilation of '{name}' failed:")?;
        } else {
            writeln!(f, "compilation failed:")?;
        }
        for d in &self.diagnostics {
            writeln!(f, "  {d}")?;
        }
        Ok(())
    }
}

impl std::error::Error for CompilationError {}

// ---------------------------------------------------------------------------
// CompiledBinary
// ---------------------------------------------------------------------------

/// Result of a successful (simulated) compilation.
#[derive(Debug, Clone)]
pub struct CompiledBinary {
    /// Opaque binary bytes (deterministic hash-based in CPU mode).
    pub binary: Vec<u8>,
    /// Build log (warnings, notes).
    pub build_log: String,
    /// How long compilation took.
    pub compile_time: Duration,
    /// Hash of the source that produced this binary.
    pub source_hash: u64,
    /// Hash of the options used.
    pub options_hash: u64,
}

impl CompiledBinary {
    /// Size of the binary in bytes.
    pub fn binary_size(&self) -> usize {
        self.binary.len()
    }
}

// ---------------------------------------------------------------------------
// PreprocessorEngine
// ---------------------------------------------------------------------------

/// Simple `#define` / `#ifdef` / `#ifndef` / `#else` / `#endif` expander.
///
/// Does **not** handle `#include` (which would require filesystem access);
/// those are left as-is for the real OpenCL compiler.
#[derive(Debug, Clone)]
pub struct PreprocessorEngine {
    defines: HashMap<String, String>,
}

impl PreprocessorEngine {
    pub fn new() -> Self {
        Self { defines: HashMap::new() }
    }

    /// Pre-seed a define.
    pub fn define(&mut self, name: impl Into<String>, value: impl Into<String>) {
        self.defines.insert(name.into(), value.into());
    }

    /// Bulk-load defines from a slice.
    pub fn define_all(&mut self, defs: &[(String, String)]) {
        for (k, v) in defs {
            self.defines.insert(k.clone(), v.clone());
        }
    }

    /// Return a snapshot of active defines.
    pub fn defines(&self) -> &HashMap<String, String> {
        &self.defines
    }

    /// Expand `#define`, `#ifdef`, `#ifndef`, `#else`, `#endif`, and
    /// simple `#define`-value substitution in the source text.
    pub fn process(&mut self, source: &str) -> Result<String, PreprocessorError> {
        let mut output = String::with_capacity(source.len());
        let mut condition_stack: Vec<CondFrame> = Vec::new();

        for (line_no, line) in source.lines().enumerate() {
            let trimmed = line.trim();

            // #define NAME VALUE
            if let Some(rest) = trimmed.strip_prefix("#define ") {
                if Self::emitting(&condition_stack) {
                    let (name, value) = Self::parse_define(rest)?;
                    self.defines.insert(name, value);
                }
                continue;
            }

            // #undef NAME
            if let Some(rest) = trimmed.strip_prefix("#undef ") {
                if Self::emitting(&condition_stack) {
                    self.defines.remove(rest.trim());
                }
                continue;
            }

            // #ifdef NAME
            if let Some(rest) = trimmed.strip_prefix("#ifdef ") {
                let name = rest.trim();
                let active = Self::emitting(&condition_stack) && self.defines.contains_key(name);
                condition_stack.push(CondFrame { active, seen_else: false });
                continue;
            }

            // #ifndef NAME
            if let Some(rest) = trimmed.strip_prefix("#ifndef ") {
                let name = rest.trim();
                let active = Self::emitting(&condition_stack) && !self.defines.contains_key(name);
                condition_stack.push(CondFrame { active, seen_else: false });
                continue;
            }

            // #else
            if trimmed == "#else" {
                let stack_len = condition_stack.len();
                if stack_len == 0 {
                    return Err(PreprocessorError {
                        line: line_no as u32 + 1,
                        message: "#else without matching #if/#ifdef".into(),
                    });
                }
                let parent_emitting = if stack_len > 1 {
                    condition_stack[..stack_len - 1].iter().all(|f| f.active)
                } else {
                    true
                };
                let frame = condition_stack.last_mut().unwrap();
                if frame.seen_else {
                    return Err(PreprocessorError {
                        line: line_no as u32 + 1,
                        message: "duplicate #else".into(),
                    });
                }
                frame.active = parent_emitting && !frame.active;
                frame.seen_else = true;
                continue;
            }

            // #endif
            if trimmed == "#endif" {
                if condition_stack.pop().is_none() {
                    return Err(PreprocessorError {
                        line: line_no as u32 + 1,
                        message: "#endif without matching #if/#ifdef".into(),
                    });
                }
                continue;
            }

            // Regular line — emit if all conditions active.
            if Self::emitting(&condition_stack) {
                // Substitute known defines in the line.
                let expanded = self.substitute(line);
                output.push_str(&expanded);
                output.push('\n');
            }
        }

        if !condition_stack.is_empty() {
            return Err(PreprocessorError {
                line: 0,
                message: format!(
                    "unterminated #ifdef/#ifndef ({} unclosed)",
                    condition_stack.len()
                ),
            });
        }

        Ok(output)
    }

    fn emitting(stack: &[CondFrame]) -> bool {
        stack.iter().all(|f| f.active)
    }

    fn parse_define(rest: &str) -> Result<(String, String), PreprocessorError> {
        let mut parts = rest.splitn(2, char::is_whitespace);
        let name = parts.next().unwrap_or("").trim().to_string();
        if name.is_empty() {
            return Err(PreprocessorError { line: 0, message: "empty #define name".into() });
        }
        let value = parts.next().unwrap_or("").trim().to_string();
        Ok((name, value))
    }

    /// Replace occurrences of defined names with their values.
    fn substitute(&self, line: &str) -> String {
        let mut result = line.to_string();
        for (name, value) in &self.defines {
            if !value.is_empty() {
                result = result.replace(name.as_str(), value);
            }
        }
        result
    }
}

impl Default for PreprocessorEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone)]
struct CondFrame {
    active: bool,
    seen_else: bool,
}

/// Error during preprocessing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PreprocessorError {
    /// 1-based line number (0 if not line-specific).
    pub line: u32,
    pub message: String,
}

impl fmt::Display for PreprocessorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.line > 0 {
            write!(f, "preprocessor error at line {}: {}", self.line, self.message)
        } else {
            write!(f, "preprocessor error: {}", self.message)
        }
    }
}

impl std::error::Error for PreprocessorError {}

// ---------------------------------------------------------------------------
// KernelSpecializer
// ---------------------------------------------------------------------------

/// Produces specialised kernel source by injecting compile-time constants.
///
/// Template placeholders are `#define`-style tokens that get replaced before
/// compilation — e.g. `TILE_SIZE`, `WORK_GROUP_X`, `LOCAL_MEM_BYTES`.
#[derive(Debug, Clone)]
pub struct KernelSpecializer {
    constants: HashMap<String, String>,
}

impl KernelSpecializer {
    pub fn new() -> Self {
        Self { constants: HashMap::new() }
    }

    /// Set a compile-time constant.
    pub fn set(mut self, name: impl Into<String>, value: impl fmt::Display) -> Self {
        self.constants.insert(name.into(), value.to_string());
        self
    }

    /// Set an integer constant.
    pub fn set_int(self, name: impl Into<String>, value: i64) -> Self {
        self.set(name, value)
    }

    /// Set a float constant.
    pub fn set_float(self, name: impl Into<String>, value: f64) -> Self {
        self.set(name, value)
    }

    /// Current constants.
    pub fn constants(&self) -> &HashMap<String, String> {
        &self.constants
    }

    /// Specialise the given source by prepending `#define` lines for every
    /// constant, then return a new [`KernelSource`].
    pub fn specialize(&self, source: &KernelSource) -> KernelSource {
        let mut specialised = source.clone();
        for (k, v) in &self.constants {
            specialised.defines.push((k.clone(), v.clone()));
        }
        specialised
    }

    /// Apply constants to raw source text (for quick inspection).
    pub fn apply_to_text(&self, text: &str) -> String {
        let mut out = String::new();
        for (k, v) in &self.constants {
            out.push_str(&format!("#define {k} {v}\n"));
        }
        out.push_str(text);
        out
    }
}

impl Default for KernelSpecializer {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// CompilationStats
// ---------------------------------------------------------------------------

/// Aggregate statistics for the compilation subsystem.
#[derive(Debug, Clone, Default)]
pub struct CompilationStats {
    /// Total compilations attempted.
    pub compilations: u64,
    /// Successful compilations.
    pub successes: u64,
    /// Failed compilations.
    pub failures: u64,
    /// Cache hits (compilation skipped).
    pub cache_hits: u64,
    /// Cache misses (had to compile).
    pub cache_misses: u64,
    /// Total time spent compiling.
    pub total_compile_time: Duration,
    /// Total binary bytes produced.
    pub total_binary_bytes: u64,
}

impl CompilationStats {
    pub fn cache_hit_rate(&self) -> f64 {
        let total = self.cache_hits + self.cache_misses;
        if total == 0 { 0.0 } else { self.cache_hits as f64 / total as f64 }
    }

    pub fn average_compile_time(&self) -> Duration {
        if self.successes == 0 {
            Duration::ZERO
        } else {
            self.total_compile_time / self.successes as u32
        }
    }

    pub fn success_rate(&self) -> f64 {
        if self.compilations == 0 { 0.0 } else { self.successes as f64 / self.compilations as f64 }
    }
}

impl fmt::Display for CompilationStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "CompilationStats(compiled={}, ok={}, fail={}, cache_hit={:.1}%, \
             avg_time={:?}, total_bytes={})",
            self.compilations,
            self.successes,
            self.failures,
            self.cache_hit_rate() * 100.0,
            self.average_compile_time(),
            self.total_binary_bytes,
        )
    }
}

// ---------------------------------------------------------------------------
// CompilationCache
// ---------------------------------------------------------------------------

/// Composite cache key: (source_hash, device_id, options_hash).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CacheKey {
    pub source_hash: u64,
    pub device_id: String,
    pub options_hash: u64,
}

impl CacheKey {
    pub fn new(source_hash: u64, device_id: impl Into<String>, options_hash: u64) -> Self {
        Self { source_hash, device_id: device_id.into(), options_hash }
    }
}

impl fmt::Display for CacheKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "CacheKey(src={:016x}, dev={}, opts={:016x})",
            self.source_hash, self.device_id, self.options_hash,
        )
    }
}

/// In-memory compilation cache with optional disk persistence.
pub struct CompilationCache {
    entries: RwLock<HashMap<CacheKey, CompiledBinary>>,
    disk_dir: Option<PathBuf>,
    max_entries: usize,
    stats: RwLock<CacheInternalStats>,
}

#[derive(Debug, Clone, Default)]
struct CacheInternalStats {
    hits: u64,
    misses: u64,
    stores: u64,
    evictions: u64,
}

impl CompilationCache {
    /// Create a memory-only cache.
    pub fn memory_only(max_entries: usize) -> Self {
        Self {
            entries: RwLock::new(HashMap::new()),
            disk_dir: None,
            max_entries,
            stats: RwLock::new(CacheInternalStats::default()),
        }
    }

    /// Create a cache backed by a disk directory.
    pub fn with_disk(max_entries: usize, disk_dir: impl Into<PathBuf>) -> Self {
        Self {
            entries: RwLock::new(HashMap::new()),
            disk_dir: Some(disk_dir.into()),
            max_entries,
            stats: RwLock::new(CacheInternalStats::default()),
        }
    }

    /// Look up a cached binary.
    pub fn get(&self, key: &CacheKey) -> Option<CompiledBinary> {
        // Try memory first.
        {
            let map = self.entries.read().unwrap();
            if let Some(entry) = map.get(key) {
                self.stats.write().unwrap().hits += 1;
                return Some(entry.clone());
            }
        }

        // Try disk.
        if let Some(bin) = self.load_from_disk(key) {
            // Promote to memory.
            let mut map = self.entries.write().unwrap();
            map.insert(key.clone(), bin.clone());
            self.stats.write().unwrap().hits += 1;
            return Some(bin);
        }

        self.stats.write().unwrap().misses += 1;
        None
    }

    /// Store a compiled binary.
    pub fn put(&self, key: CacheKey, binary: CompiledBinary) {
        let mut map = self.entries.write().unwrap();
        // Simple eviction: if at capacity, remove the first entry.
        if map.len() >= self.max_entries
            && !map.contains_key(&key)
            && let Some(evict_key) = map.keys().next().cloned()
        {
            map.remove(&evict_key);
            self.stats.write().unwrap().evictions += 1;
        }
        self.save_to_disk(&key, &binary);
        map.insert(key, binary);
        self.stats.write().unwrap().stores += 1;
    }

    /// Remove a specific entry.
    pub fn invalidate(&self, key: &CacheKey) -> bool {
        let mut map = self.entries.write().unwrap();
        let removed = map.remove(key).is_some();
        if removed {
            self.remove_from_disk(key);
        }
        removed
    }

    /// Remove all entries.
    pub fn clear(&self) {
        let mut map = self.entries.write().unwrap();
        map.clear();
    }

    /// Number of entries currently in memory.
    pub fn len(&self) -> usize {
        self.entries.read().unwrap().len()
    }

    /// Whether the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Snapshot of cache statistics.
    pub fn stats(&self) -> (u64, u64, u64, u64) {
        let s = self.stats.read().unwrap();
        (s.hits, s.misses, s.stores, s.evictions)
    }

    /// Cache hit rate.
    pub fn hit_rate(&self) -> f64 {
        let s = self.stats.read().unwrap();
        let total = s.hits + s.misses;
        if total == 0 { 0.0 } else { s.hits as f64 / total as f64 }
    }

    /// Disk directory, if configured.
    pub fn disk_dir(&self) -> Option<&Path> {
        self.disk_dir.as_deref()
    }

    // -- disk helpers -------------------------------------------------------

    fn disk_path(&self, key: &CacheKey) -> Option<PathBuf> {
        self.disk_dir.as_ref().map(|dir| {
            let filename = format!("{:016x}_{:016x}.bin", key.source_hash, key.options_hash);
            dir.join(filename)
        })
    }

    fn save_to_disk(&self, key: &CacheKey, binary: &CompiledBinary) {
        if let Some(path) = self.disk_path(key) {
            if let Some(parent) = path.parent() {
                let _ = std::fs::create_dir_all(parent);
            }
            let _ = std::fs::write(&path, &binary.binary);
        }
    }

    fn load_from_disk(&self, key: &CacheKey) -> Option<CompiledBinary> {
        let path = self.disk_path(key)?;
        let binary = std::fs::read(&path).ok()?;
        Some(CompiledBinary {
            binary,
            build_log: String::new(),
            compile_time: Duration::ZERO,
            source_hash: key.source_hash,
            options_hash: key.options_hash,
        })
    }

    fn remove_from_disk(&self, key: &CacheKey) {
        if let Some(path) = self.disk_path(key) {
            let _ = std::fs::remove_file(path);
        }
    }
}

impl fmt::Debug for CompilationCache {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CompilationCache")
            .field("entries", &self.len())
            .field("disk_dir", &self.disk_dir)
            .field("max_entries", &self.max_entries)
            .finish()
    }
}

// ---------------------------------------------------------------------------
// KernelCompiler (CPU reference implementation)
// ---------------------------------------------------------------------------

/// Simulated OpenCL kernel compiler.
///
/// In CPU-reference mode, "compilation" deterministically hashes the
/// preprocessed source and options to produce a fake binary whose content
/// is the hash bytes repeated to match the estimated binary size.
pub struct KernelCompiler {
    preprocessor: PreprocessorEngine,
    cache: Arc<CompilationCache>,
    device_id: String,
    stats: RwLock<CompilationStats>,
}

impl KernelCompiler {
    /// Create a compiler targeting the given (simulated) device.
    pub fn new(device_id: impl Into<String>, cache: Arc<CompilationCache>) -> Self {
        Self {
            preprocessor: PreprocessorEngine::new(),
            cache,
            device_id: device_id.into(),
            stats: RwLock::new(CompilationStats::default()),
        }
    }

    /// Create a compiler with a default memory-only cache.
    pub fn with_default_cache(device_id: impl Into<String>) -> Self {
        Self::new(device_id, Arc::new(CompilationCache::memory_only(256)))
    }

    /// Access the preprocessor to add built-in defines.
    pub fn preprocessor_mut(&mut self) -> &mut PreprocessorEngine {
        &mut self.preprocessor
    }

    /// Current device ID.
    pub fn device_id(&self) -> &str {
        &self.device_id
    }

    /// Reference to the underlying cache.
    pub fn cache(&self) -> &Arc<CompilationCache> {
        &self.cache
    }

    /// Compile a kernel source with the given options.
    ///
    /// 1. Preprocess (expand `#define`/`#ifdef`).
    /// 2. Check cache.
    /// 3. Simulate compilation (hash-based binary).
    /// 4. Store in cache.
    pub fn compile(
        &mut self,
        source: &KernelSource,
        options: &CompilationOptions,
    ) -> Result<CompiledBinary, CompilationError> {
        let start = Instant::now();
        self.stats.write().unwrap().compilations += 1;

        // 1. Preprocess.
        let full_source = source.full_source();
        self.preprocessor.define_all(source.defines());
        let preprocessed = self.preprocessor.process(&full_source).map_err(|e| {
            self.stats.write().unwrap().failures += 1;
            CompilationError::from_log(
                format!("<kernel>:{}:0: error: {}", e.line, e.message),
                source.name().map(String::from),
            )
        })?;

        // 2. Compute keys.
        let src_hash = hash_str(&preprocessed);
        let opts_hash = options.options_hash();
        let cache_key = CacheKey::new(src_hash, &self.device_id, opts_hash);

        // 3. Check cache.
        if let Some(cached) = self.cache.get(&cache_key) {
            let mut s = self.stats.write().unwrap();
            s.cache_hits += 1;
            s.successes += 1;
            s.total_compile_time += start.elapsed();
            return Ok(cached);
        }
        self.stats.write().unwrap().cache_misses += 1;

        // 4. Validate source (simulate syntax checking).
        if let Some(err) = Self::validate_source(&preprocessed, source.name()) {
            self.stats.write().unwrap().failures += 1;
            return Err(err);
        }

        // 5. Produce binary.
        let binary = Self::simulate_binary(src_hash, opts_hash, &preprocessed);
        let compile_time = start.elapsed();

        let compiled = CompiledBinary {
            binary,
            build_log: String::new(),
            compile_time,
            source_hash: src_hash,
            options_hash: opts_hash,
        };

        // 6. Cache.
        self.cache.put(cache_key, compiled.clone());

        let mut s = self.stats.write().unwrap();
        s.successes += 1;
        s.total_compile_time += compile_time;
        s.total_binary_bytes += compiled.binary_size() as u64;
        Ok(compiled)
    }

    /// Return a snapshot of compilation statistics.
    pub fn stats(&self) -> CompilationStats {
        self.stats.read().unwrap().clone()
    }

    /// Reset statistics.
    pub fn reset_stats(&self) {
        *self.stats.write().unwrap() = CompilationStats::default();
    }

    // -- internal -----------------------------------------------------------

    /// Minimal syntax validation (CPU reference).
    fn validate_source(
        preprocessed: &str,
        name: Option<&str>,
    ) -> Option<CompilationError> {
        let trimmed = preprocessed.trim();
        if trimmed.is_empty() {
            return Some(CompilationError::from_log(
                "<kernel>:1:0: error: empty kernel source",
                name.map(String::from),
            ));
        }

        // Check for balanced braces.
        let mut depth: i32 = 0;
        for (line_no, line) in trimmed.lines().enumerate() {
            for ch in line.chars() {
                match ch {
                    '{' => depth += 1,
                    '}' => depth -= 1,
                    _ => {}
                }
                if depth < 0 {
                    return Some(CompilationError::from_log(
                        format!(
                            "<kernel>:{}:0: error: unmatched closing brace",
                            line_no + 1
                        ),
                        name.map(String::from),
                    ));
                }
            }
        }
        if depth != 0 {
            return Some(CompilationError::from_log(
                format!(
                    "<kernel>:{}:0: error: {} unclosed brace(s)",
                    trimmed.lines().count(),
                    depth
                ),
                name.map(String::from),
            ));
        }

        None
    }

    /// Produce a deterministic binary from the preprocessed hash.
    ///
    /// Binary size is estimated as roughly 4× the source size (a crude
    /// approximation of what a real OpenCL JIT would produce).
    fn simulate_binary(src_hash: u64, opts_hash: u64, source: &str) -> Vec<u8> {
        let estimated_size = estimate_binary_size(source.len());
        let hash_bytes = {
            let mut v = Vec::with_capacity(16);
            v.extend_from_slice(&src_hash.to_le_bytes());
            v.extend_from_slice(&opts_hash.to_le_bytes());
            v
        };
        hash_bytes.iter().cycle().take(estimated_size).copied().collect()
    }
}

impl fmt::Debug for KernelCompiler {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("KernelCompiler")
            .field("device_id", &self.device_id)
            .field("cache_entries", &self.cache.len())
            .finish()
    }
}

// ---------------------------------------------------------------------------
// WarmupCompiler
// ---------------------------------------------------------------------------

/// Pre-compiles a set of kernels at startup to avoid first-use latency.
pub struct WarmupCompiler {
    kernels: Vec<(KernelSource, CompilationOptions)>,
}

/// Result of a single warmup compilation.
#[derive(Debug, Clone)]
pub struct WarmupResult {
    pub name: Option<String>,
    pub success: bool,
    pub compile_time: Duration,
    pub binary_size: usize,
    pub error: Option<String>,
}

/// Aggregate result of a warmup pass.
#[derive(Debug, Clone)]
pub struct WarmupSummary {
    pub results: Vec<WarmupResult>,
    pub total_time: Duration,
}

impl WarmupSummary {
    pub fn successes(&self) -> usize {
        self.results.iter().filter(|r| r.success).count()
    }

    pub fn failures(&self) -> usize {
        self.results.iter().filter(|r| !r.success).count()
    }

    pub fn total_binary_bytes(&self) -> usize {
        self.results.iter().map(|r| r.binary_size).sum()
    }

    pub fn all_succeeded(&self) -> bool {
        self.results.iter().all(|r| r.success)
    }
}

impl fmt::Display for WarmupSummary {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Warmup: {}/{} ok, total {:?}, {} bytes",
            self.successes(),
            self.results.len(),
            self.total_time,
            self.total_binary_bytes(),
        )
    }
}

impl WarmupCompiler {
    pub fn new() -> Self {
        Self { kernels: Vec::new() }
    }

    /// Register a kernel to be warmed up.
    pub fn add(&mut self, source: KernelSource, options: CompilationOptions) {
        self.kernels.push((source, options));
    }

    /// Number of registered kernels.
    pub fn kernel_count(&self) -> usize {
        self.kernels.len()
    }

    /// Run all registered compilations.
    pub fn warmup(&self, compiler: &mut KernelCompiler) -> WarmupSummary {
        let start = Instant::now();
        let mut results = Vec::with_capacity(self.kernels.len());

        for (source, options) in &self.kernels {
            let comp_start = Instant::now();
            match compiler.compile(source, options) {
                Ok(bin) => {
                    results.push(WarmupResult {
                        name: source.name().map(String::from),
                        success: true,
                        compile_time: comp_start.elapsed(),
                        binary_size: bin.binary_size(),
                        error: None,
                    });
                }
                Err(e) => {
                    results.push(WarmupResult {
                        name: source.name().map(String::from),
                        success: false,
                        compile_time: comp_start.elapsed(),
                        binary_size: 0,
                        error: Some(e.to_string()),
                    });
                }
            }
        }

        WarmupSummary { results, total_time: start.elapsed() }
    }
}

impl Default for WarmupCompiler {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Deterministic 64-bit hash of a string.
fn hash_str(s: &str) -> u64 {
    let mut hasher = DefaultHasher::new();
    s.hash(&mut hasher);
    hasher.finish()
}

/// Estimate compiled binary size from source length.
///
/// Real OpenCL binaries are typically 2–8× the source size; we use 4×
/// with a minimum of 64 bytes.
pub fn estimate_binary_size(source_len: usize) -> usize {
    (source_len * 4).max(64)
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    // -----------------------------------------------------------------------
    // KernelSource tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_kernel_source_new() {
        let src = KernelSource::new("__kernel void foo() {}");
        assert_eq!(src.source(), "__kernel void foo() {}");
        assert!(src.defines().is_empty());
        assert!(src.include_paths().is_empty());
        assert!(src.name().is_none());
    }

    #[test]
    fn test_kernel_source_with_name() {
        let src = KernelSource::new("code").with_name("matmul");
        assert_eq!(src.name(), Some("matmul"));
    }

    #[test]
    fn test_kernel_source_defines() {
        let src = KernelSource::new("x").define("TILE", "16").define("N", "1024");
        assert_eq!(src.defines().len(), 2);
    }

    #[test]
    fn test_kernel_source_preamble() {
        let src = KernelSource::new("body").define("A", "1").define("B", "2");
        let preamble = src.preamble();
        assert!(preamble.contains("#define A 1"));
        assert!(preamble.contains("#define B 2"));
    }

    #[test]
    fn test_kernel_source_full_source() {
        let src = KernelSource::new("body").define("X", "42");
        let full = src.full_source();
        assert!(full.starts_with("#define X 42\n"));
        assert!(full.ends_with("body"));
    }

    #[test]
    fn test_kernel_source_full_source_no_defines() {
        let src = KernelSource::new("just body");
        assert_eq!(src.full_source(), "just body");
    }

    #[test]
    fn test_kernel_source_content_hash_deterministic() {
        let a = KernelSource::new("foo").define("X", "1");
        let b = KernelSource::new("foo").define("X", "1");
        assert_eq!(a.content_hash(), b.content_hash());
    }

    #[test]
    fn test_kernel_source_content_hash_differs_on_change() {
        let a = KernelSource::new("foo").define("X", "1");
        let b = KernelSource::new("foo").define("X", "2");
        assert_ne!(a.content_hash(), b.content_hash());
    }

    #[test]
    fn test_kernel_source_include_path() {
        let src = KernelSource::new("x").include_path("/usr/include");
        assert_eq!(src.include_paths().len(), 1);
    }

    // -----------------------------------------------------------------------
    // CompilationOptions tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_options_default() {
        let opts = CompilationOptions::new();
        assert!(opts.flags().is_empty());
        assert_eq!(opts.version(), "1.2");
        assert_eq!(opts.to_build_string(), "");
    }

    #[test]
    fn test_options_fast_math() {
        let opts = CompilationOptions::new().fast_math();
        assert_eq!(opts.to_build_string(), "-cl-fast-relaxed-math");
    }

    #[test]
    fn test_options_multiple_flags() {
        let opts = CompilationOptions::new().fast_math().mad_enable();
        let s = opts.to_build_string();
        assert!(s.contains("-cl-fast-relaxed-math"));
        assert!(s.contains("-cl-mad-enable"));
    }

    #[test]
    fn test_options_opt_disable() {
        let opts = CompilationOptions::new().opt_level(0);
        assert!(opts.to_build_string().contains("-cl-opt-disable"));
    }

    #[test]
    fn test_options_hash_deterministic() {
        let a = CompilationOptions::new().fast_math();
        let b = CompilationOptions::new().fast_math();
        assert_eq!(a.options_hash(), b.options_hash());
    }

    #[test]
    fn test_options_hash_differs() {
        let a = CompilationOptions::new().fast_math();
        let b = CompilationOptions::new().mad_enable();
        assert_ne!(a.options_hash(), b.options_hash());
    }

    #[test]
    fn test_options_display() {
        let opts = CompilationOptions::new().fast_math();
        assert_eq!(format!("{opts}"), "-cl-fast-relaxed-math");
    }

    #[test]
    fn test_options_target_version() {
        let opts = CompilationOptions::new().target_version("3.0");
        assert_eq!(opts.version(), "3.0");
    }

    #[test]
    fn test_options_unsafe_math() {
        let opts = CompilationOptions::new().unsafe_math();
        assert!(opts.to_build_string().contains("-cl-unsafe-math-optimizations"));
    }

    #[test]
    fn test_options_no_signed_zeros() {
        let opts = CompilationOptions::new().no_signed_zeros();
        assert!(opts.to_build_string().contains("-cl-no-signed-zeros"));
    }

    #[test]
    fn test_options_custom_flag() {
        let opts = CompilationOptions::new().flag("-DFOO=1");
        assert_eq!(opts.to_build_string(), "-DFOO=1");
    }

    #[test]
    fn test_options_opt_level_clamped() {
        let opts = CompilationOptions::new().opt_level(99);
        // Should clamp to 2.
        assert_eq!(opts.to_build_string(), ""); // level 2 doesn't add special flag
    }

    // -----------------------------------------------------------------------
    // CompilationError / Diagnostic tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_parse_error_line() {
        let log = "<kernel>:10:5: error: undeclared identifier 'x'";
        let err = CompilationError::from_log(log, None);
        assert_eq!(err.error_count(), 1);
        let d = &err.diagnostics[0];
        assert_eq!(d.severity, DiagnosticSeverity::Error);
        let loc = d.location.as_ref().unwrap();
        assert_eq!(loc.file, "<kernel>");
        assert_eq!(loc.line, 10);
        assert_eq!(loc.column, 5);
        assert!(d.message.contains("undeclared"));
    }

    #[test]
    fn test_parse_warning_line() {
        let log = "foo.cl:3:1: warning: implicit conversion";
        let err = CompilationError::from_log(log, None);
        assert_eq!(err.warning_count(), 1);
        assert_eq!(err.error_count(), 0);
    }

    #[test]
    fn test_parse_note_line() {
        let log = "foo.cl:1:0: note: see previous definition";
        let err = CompilationError::from_log(log, None);
        let d = &err.diagnostics[0];
        assert_eq!(d.severity, DiagnosticSeverity::Note);
    }

    #[test]
    fn test_parse_multiple_diagnostics() {
        let log = "<k>:1:0: error: syntax error\n<k>:2:0: warning: unused variable";
        let err = CompilationError::from_log(log, None);
        assert_eq!(err.diagnostics.len(), 2);
        assert_eq!(err.error_count(), 1);
        assert_eq!(err.warning_count(), 1);
    }

    #[test]
    fn test_parse_unstructured_line() {
        let log = "some random compiler output";
        let err = CompilationError::from_log(log, None);
        assert_eq!(err.diagnostics.len(), 1);
        assert_eq!(err.diagnostics[0].severity, DiagnosticSeverity::Note);
    }

    #[test]
    fn test_error_has_errors() {
        let err = CompilationError::from_log("<k>:1:0: error: boom", None);
        assert!(err.has_errors());
    }

    #[test]
    fn test_error_no_errors_on_warning_only() {
        let err = CompilationError::from_log("<k>:1:0: warning: hmm", None);
        assert!(!err.has_errors());
    }

    #[test]
    fn test_error_display_with_name() {
        let err = CompilationError::from_log("<k>:1:0: error: x", Some("matmul".into()));
        let s = format!("{err}");
        assert!(s.contains("matmul"));
    }

    #[test]
    fn test_diagnostic_display() {
        let d = Diagnostic {
            severity: DiagnosticSeverity::Error,
            location: Some(DiagnosticLocation {
                file: "test.cl".into(),
                line: 5,
                column: 3,
            }),
            message: "bad".into(),
        };
        assert_eq!(format!("{d}"), "test.cl:5:3: error: bad");
    }

    #[test]
    fn test_diagnostic_display_no_location() {
        let d = Diagnostic {
            severity: DiagnosticSeverity::Warning,
            location: None,
            message: "hmm".into(),
        };
        assert_eq!(format!("{d}"), "warning: hmm");
    }

    #[test]
    fn test_empty_log() {
        let err = CompilationError::from_log("", None);
        assert!(err.diagnostics.is_empty());
        assert!(!err.has_errors());
    }

    // -----------------------------------------------------------------------
    // PreprocessorEngine tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_pp_passthrough() {
        let mut pp = PreprocessorEngine::new();
        let out = pp.process("hello\nworld\n").unwrap();
        assert_eq!(out, "hello\nworld\n");
    }

    #[test]
    fn test_pp_define_and_substitute() {
        let mut pp = PreprocessorEngine::new();
        let src = "#define TILE 16\nint x = TILE;\n";
        let out = pp.process(src).unwrap();
        assert!(out.contains("int x = 16;"));
    }

    #[test]
    fn test_pp_ifdef_true() {
        let mut pp = PreprocessorEngine::new();
        pp.define("FOO", "1");
        let src = "#ifdef FOO\nyes\n#endif\n";
        let out = pp.process(src).unwrap();
        assert!(out.contains("yes"));
    }

    #[test]
    fn test_pp_ifdef_false() {
        let mut pp = PreprocessorEngine::new();
        let src = "#ifdef FOO\nyes\n#endif\n";
        let out = pp.process(src).unwrap();
        assert!(!out.contains("yes"));
    }

    #[test]
    fn test_pp_ifndef_true() {
        let mut pp = PreprocessorEngine::new();
        let src = "#ifndef FOO\nyes\n#endif\n";
        let out = pp.process(src).unwrap();
        assert!(out.contains("yes"));
    }

    #[test]
    fn test_pp_ifndef_false() {
        let mut pp = PreprocessorEngine::new();
        pp.define("FOO", "1");
        let src = "#ifndef FOO\nyes\n#endif\n";
        let out = pp.process(src).unwrap();
        assert!(!out.contains("yes"));
    }

    #[test]
    fn test_pp_ifdef_else() {
        let mut pp = PreprocessorEngine::new();
        pp.define("FOO", "1");
        let src = "#ifdef FOO\nA\n#else\nB\n#endif\n";
        let out = pp.process(src).unwrap();
        assert!(out.contains("A"));
        assert!(!out.contains("B"));
    }

    #[test]
    fn test_pp_ifdef_else_false() {
        let mut pp = PreprocessorEngine::new();
        let src = "#ifdef FOO\nA\n#else\nB\n#endif\n";
        let out = pp.process(src).unwrap();
        assert!(!out.contains("A"));
        assert!(out.contains("B"));
    }

    #[test]
    fn test_pp_nested_ifdef() {
        let mut pp = PreprocessorEngine::new();
        pp.define("A", "1");
        pp.define("B", "1");
        let src = "#ifdef A\n#ifdef B\ndeep\n#endif\n#endif\n";
        let out = pp.process(src).unwrap();
        assert!(out.contains("deep"));
    }

    #[test]
    fn test_pp_nested_ifdef_outer_false() {
        let mut pp = PreprocessorEngine::new();
        pp.define("B", "1");
        let src = "#ifdef A\n#ifdef B\ndeep\n#endif\n#endif\n";
        let out = pp.process(src).unwrap();
        assert!(!out.contains("deep"));
    }

    #[test]
    fn test_pp_undef() {
        let mut pp = PreprocessorEngine::new();
        pp.define("X", "1");
        let src = "#undef X\n#ifdef X\nyes\n#else\nno\n#endif\n";
        let out = pp.process(src).unwrap();
        assert!(out.contains("no"));
        assert!(!out.contains("yes"));
    }

    #[test]
    fn test_pp_inline_define() {
        let mut pp = PreprocessorEngine::new();
        let src = "#define Y 99\nval = Y;\n";
        let out = pp.process(src).unwrap();
        assert!(out.contains("val = 99;"));
    }

    #[test]
    fn test_pp_error_unmatched_endif() {
        let mut pp = PreprocessorEngine::new();
        let err = pp.process("#endif\n").unwrap_err();
        assert!(err.message.contains("#endif"));
    }

    #[test]
    fn test_pp_error_unterminated_ifdef() {
        let mut pp = PreprocessorEngine::new();
        let err = pp.process("#ifdef X\nstuff\n").unwrap_err();
        assert!(err.message.contains("unterminated"));
    }

    #[test]
    fn test_pp_error_duplicate_else() {
        let mut pp = PreprocessorEngine::new();
        let err = pp.process("#ifdef X\n#else\n#else\n#endif\n").unwrap_err();
        assert!(err.message.contains("duplicate"));
    }

    #[test]
    fn test_pp_error_else_without_if() {
        let mut pp = PreprocessorEngine::new();
        let err = pp.process("#else\n").unwrap_err();
        assert!(err.message.contains("#else without"));
    }

    #[test]
    fn test_pp_define_all() {
        let mut pp = PreprocessorEngine::new();
        pp.define_all(&[("A".into(), "1".into()), ("B".into(), "2".into())]);
        assert_eq!(pp.defines().len(), 2);
    }

    #[test]
    fn test_pp_empty_source() {
        let mut pp = PreprocessorEngine::new();
        let out = pp.process("").unwrap();
        assert!(out.is_empty());
    }

    #[test]
    fn test_pp_define_empty_value() {
        let mut pp = PreprocessorEngine::new();
        let src = "#define GUARD\n#ifdef GUARD\nok\n#endif\n";
        let out = pp.process(src).unwrap();
        assert!(out.contains("ok"));
    }

    // -----------------------------------------------------------------------
    // KernelSpecializer tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_specializer_set() {
        let spec = KernelSpecializer::new().set("TILE", 16).set("WG", 256);
        assert_eq!(spec.constants().len(), 2);
    }

    #[test]
    fn test_specializer_set_int() {
        let spec = KernelSpecializer::new().set_int("N", 1024);
        assert_eq!(spec.constants()["N"], "1024");
    }

    #[test]
    fn test_specializer_set_float() {
        let spec = KernelSpecializer::new().set_float("EPSILON", 1e-6);
        assert!(spec.constants()["EPSILON"].contains("0.000001"));
    }

    #[test]
    fn test_specializer_specialize() {
        let src = KernelSource::new("__kernel void foo() {}");
        let spec = KernelSpecializer::new().set("TILE", 32);
        let specialised = spec.specialize(&src);
        assert!(specialised.full_source().contains("#define TILE 32"));
    }

    #[test]
    fn test_specializer_apply_to_text() {
        let spec = KernelSpecializer::new().set("X", 10);
        let out = spec.apply_to_text("use X");
        assert!(out.contains("#define X 10"));
        assert!(out.contains("use X"));
    }

    #[test]
    fn test_specializer_different_constants_different_hash() {
        let src = KernelSource::new("body");
        let s1 = KernelSpecializer::new().set("TILE", 16).specialize(&src);
        let s2 = KernelSpecializer::new().set("TILE", 32).specialize(&src);
        assert_ne!(s1.content_hash(), s2.content_hash());
    }

    #[test]
    fn test_specializer_preserves_original_defines() {
        let src = KernelSource::new("body").define("ORIG", "1");
        let spec = KernelSpecializer::new().set("NEW", "2");
        let specialised = spec.specialize(&src);
        let full = specialised.full_source();
        assert!(full.contains("#define ORIG 1"));
        assert!(full.contains("#define NEW 2"));
    }

    // -----------------------------------------------------------------------
    // CompilationCache tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_cache_miss() {
        let cache = CompilationCache::memory_only(16);
        let key = CacheKey::new(1, "dev", 2);
        assert!(cache.get(&key).is_none());
    }

    #[test]
    fn test_cache_hit() {
        let cache = CompilationCache::memory_only(16);
        let key = CacheKey::new(1, "dev", 2);
        let bin = CompiledBinary {
            binary: vec![0xDE, 0xAD],
            build_log: String::new(),
            compile_time: Duration::from_millis(10),
            source_hash: 1,
            options_hash: 2,
        };
        cache.put(key.clone(), bin);
        let got = cache.get(&key).unwrap();
        assert_eq!(got.binary, vec![0xDE, 0xAD]);
    }

    #[test]
    fn test_cache_stats() {
        let cache = CompilationCache::memory_only(16);
        let key = CacheKey::new(1, "d", 2);
        let bin = CompiledBinary {
            binary: vec![1],
            build_log: String::new(),
            compile_time: Duration::ZERO,
            source_hash: 1,
            options_hash: 2,
        };
        cache.put(key.clone(), bin);
        let _ = cache.get(&key);                               // hit
        let _ = cache.get(&CacheKey::new(99, "d", 99)); // miss
        let (hits, misses, stores, _) = cache.stats();
        assert_eq!(hits, 1);
        assert_eq!(misses, 1);
        assert_eq!(stores, 1);
    }

    #[test]
    fn test_cache_hit_rate() {
        let cache = CompilationCache::memory_only(16);
        assert_eq!(cache.hit_rate(), 0.0); // no lookups yet
        let key = CacheKey::new(1, "d", 2);
        let bin = CompiledBinary {
            binary: vec![1],
            build_log: String::new(),
            compile_time: Duration::ZERO,
            source_hash: 1,
            options_hash: 2,
        };
        cache.put(key.clone(), bin);
        let _ = cache.get(&key); // hit
        let _ = cache.get(&key); // hit
        let _ = cache.get(&CacheKey::new(0, "d", 0)); // miss
        // 2 hits, 1 miss = 2/3
        assert!((cache.hit_rate() - 2.0 / 3.0).abs() < 1e-9);
    }

    #[test]
    fn test_cache_invalidate() {
        let cache = CompilationCache::memory_only(16);
        let key = CacheKey::new(1, "d", 2);
        let bin = CompiledBinary {
            binary: vec![1],
            build_log: String::new(),
            compile_time: Duration::ZERO,
            source_hash: 1,
            options_hash: 2,
        };
        cache.put(key.clone(), bin);
        assert!(cache.invalidate(&key));
        assert!(cache.get(&key).is_none());
    }

    #[test]
    fn test_cache_invalidate_nonexistent() {
        let cache = CompilationCache::memory_only(16);
        assert!(!cache.invalidate(&CacheKey::new(0, "", 0)));
    }

    #[test]
    fn test_cache_clear() {
        let cache = CompilationCache::memory_only(16);
        for i in 0..5 {
            cache.put(
                CacheKey::new(i, "d", i),
                CompiledBinary {
                    binary: vec![i as u8],
                    build_log: String::new(),
                    compile_time: Duration::ZERO,
                    source_hash: i,
                    options_hash: i,
                },
            );
        }
        assert_eq!(cache.len(), 5);
        cache.clear();
        assert!(cache.is_empty());
    }

    #[test]
    fn test_cache_eviction() {
        let cache = CompilationCache::memory_only(2);
        for i in 0..3u64 {
            cache.put(
                CacheKey::new(i, "d", i),
                CompiledBinary {
                    binary: vec![i as u8],
                    build_log: String::new(),
                    compile_time: Duration::ZERO,
                    source_hash: i,
                    options_hash: i,
                },
            );
        }
        // Should have evicted down to max_entries.
        assert!(cache.len() <= 2);
    }

    #[test]
    fn test_cache_disk_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let cache = CompilationCache::with_disk(16, dir.path());
        let key = CacheKey::new(42, "gpu", 99);
        let bin = CompiledBinary {
            binary: vec![1, 2, 3, 4],
            build_log: String::new(),
            compile_time: Duration::ZERO,
            source_hash: 42,
            options_hash: 99,
        };
        cache.put(key.clone(), bin);

        // Evict from memory, then re-fetch (should come from disk).
        cache.clear();
        let got = cache.get(&key).unwrap();
        assert_eq!(got.binary, vec![1, 2, 3, 4]);
    }

    #[test]
    fn test_cache_disk_miss() {
        let dir = tempfile::tempdir().unwrap();
        let cache = CompilationCache::with_disk(16, dir.path());
        let key = CacheKey::new(1, "d", 2);
        assert!(cache.get(&key).is_none());
    }

    #[test]
    fn test_cache_key_display() {
        let key = CacheKey::new(0xABCD, "dev0", 0x1234);
        let s = format!("{key}");
        assert!(s.contains("dev0"));
    }

    // -----------------------------------------------------------------------
    // KernelCompiler tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_compile_simple() {
        let mut compiler = KernelCompiler::with_default_cache("cpu_ref");
        let src = KernelSource::new("__kernel void foo() {}");
        let opts = CompilationOptions::new();
        let bin = compiler.compile(&src, &opts).unwrap();
        assert!(!bin.binary.is_empty());
    }

    #[test]
    fn test_compile_empty_source_error() {
        let mut compiler = KernelCompiler::with_default_cache("cpu_ref");
        let src = KernelSource::new("");
        let opts = CompilationOptions::new();
        let err = compiler.compile(&src, &opts).unwrap_err();
        assert!(err.has_errors());
    }

    #[test]
    fn test_compile_whitespace_only_error() {
        let mut compiler = KernelCompiler::with_default_cache("cpu_ref");
        let src = KernelSource::new("   \n\n  ");
        let opts = CompilationOptions::new();
        let err = compiler.compile(&src, &opts).unwrap_err();
        assert!(err.has_errors());
    }

    #[test]
    fn test_compile_unbalanced_braces_error() {
        let mut compiler = KernelCompiler::with_default_cache("cpu_ref");
        let src = KernelSource::new("__kernel void foo() {");
        let opts = CompilationOptions::new();
        let err = compiler.compile(&src, &opts).unwrap_err();
        assert!(err.has_errors());
    }

    #[test]
    fn test_compile_extra_close_brace_error() {
        let mut compiler = KernelCompiler::with_default_cache("cpu_ref");
        let src = KernelSource::new("}");
        let opts = CompilationOptions::new();
        let err = compiler.compile(&src, &opts).unwrap_err();
        assert!(err.has_errors());
    }

    #[test]
    fn test_compile_cache_hit() {
        let cache = Arc::new(CompilationCache::memory_only(64));
        let mut compiler = KernelCompiler::new("dev", cache.clone());
        let src = KernelSource::new("__kernel void foo() {}");
        let opts = CompilationOptions::new();

        let b1 = compiler.compile(&src, &opts).unwrap();
        let b2 = compiler.compile(&src, &opts).unwrap();
        // Same binary from cache.
        assert_eq!(b1.binary, b2.binary);
        assert_eq!(compiler.stats().cache_hits, 1);
    }

    #[test]
    fn test_compile_different_options_different_binary() {
        let mut compiler = KernelCompiler::with_default_cache("dev");
        let src = KernelSource::new("__kernel void foo() {}");
        let opts_a = CompilationOptions::new();
        let opts_b = CompilationOptions::new().fast_math();
        let a = compiler.compile(&src, &opts_a).unwrap();
        let b = compiler.compile(&src, &opts_b).unwrap();
        // Different options hash → different binary.
        assert_ne!(a.options_hash, b.options_hash);
    }

    #[test]
    fn test_compile_with_defines() {
        let mut compiler = KernelCompiler::with_default_cache("dev");
        let src = KernelSource::new("int x = TILE;").define("TILE", "32");
        let opts = CompilationOptions::new();
        let bin = compiler.compile(&src, &opts).unwrap();
        assert!(!bin.binary.is_empty());
    }

    #[test]
    fn test_compile_stats() {
        let mut compiler = KernelCompiler::with_default_cache("dev");
        let src = KernelSource::new("__kernel void f() {}");
        let opts = CompilationOptions::new();
        compiler.compile(&src, &opts).unwrap();
        compiler.compile(&src, &opts).unwrap(); // cache hit
        let stats = compiler.stats();
        assert_eq!(stats.compilations, 2);
        assert_eq!(stats.successes, 2);
        assert_eq!(stats.cache_hits, 1);
        assert_eq!(stats.cache_misses, 1);
    }

    #[test]
    fn test_compile_stats_failure() {
        let mut compiler = KernelCompiler::with_default_cache("dev");
        let _ = compiler.compile(&KernelSource::new(""), &CompilationOptions::new());
        assert_eq!(compiler.stats().failures, 1);
    }

    #[test]
    fn test_compile_reset_stats() {
        let mut compiler = KernelCompiler::with_default_cache("dev");
        let _ = compiler.compile(
            &KernelSource::new("__kernel void f() {}"),
            &CompilationOptions::new(),
        );
        compiler.reset_stats();
        assert_eq!(compiler.stats().compilations, 0);
    }

    #[test]
    fn test_compiler_device_id() {
        let compiler = KernelCompiler::with_default_cache("Intel Arc A770");
        assert_eq!(compiler.device_id(), "Intel Arc A770");
    }

    #[test]
    fn test_compile_preprocessor_error_becomes_compilation_error() {
        let mut compiler = KernelCompiler::with_default_cache("dev");
        // Unterminated #ifdef → preprocessor error → CompilationError.
        let src = KernelSource::new("#ifdef X\nfoo\n");
        let err = compiler.compile(&src, &CompilationOptions::new()).unwrap_err();
        assert!(err.has_errors());
    }

    #[test]
    fn test_compile_binary_size_estimation() {
        let source_text = "__kernel void foo() { int x = 1; }";
        let mut compiler = KernelCompiler::with_default_cache("dev");
        let bin = compiler
            .compile(&KernelSource::new(source_text), &CompilationOptions::new())
            .unwrap();
        // Binary should be at least 64 bytes (minimum).
        assert!(bin.binary_size() >= 64);
    }

    // -----------------------------------------------------------------------
    // WarmupCompiler tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_warmup_empty() {
        let warmup = WarmupCompiler::new();
        let mut compiler = KernelCompiler::with_default_cache("dev");
        let summary = warmup.warmup(&mut compiler);
        assert!(summary.all_succeeded());
        assert_eq!(summary.results.len(), 0);
    }

    #[test]
    fn test_warmup_single_kernel() {
        let mut warmup = WarmupCompiler::new();
        warmup.add(
            KernelSource::new("__kernel void k() {}").with_name("test_kernel"),
            CompilationOptions::new(),
        );
        let mut compiler = KernelCompiler::with_default_cache("dev");
        let summary = warmup.warmup(&mut compiler);
        assert!(summary.all_succeeded());
        assert_eq!(summary.successes(), 1);
    }

    #[test]
    fn test_warmup_multiple_kernels() {
        let mut warmup = WarmupCompiler::new();
        for i in 0..5 {
            warmup.add(
                KernelSource::new(&format!("__kernel void k{i}() {{}}"))
                    .with_name(&format!("kernel_{i}")),
                CompilationOptions::new(),
            );
        }
        let mut compiler = KernelCompiler::with_default_cache("dev");
        let summary = warmup.warmup(&mut compiler);
        assert_eq!(summary.successes(), 5);
        assert!(summary.total_binary_bytes() > 0);
    }

    #[test]
    fn test_warmup_with_failure() {
        let mut warmup = WarmupCompiler::new();
        warmup.add(
            KernelSource::new("__kernel void ok() {}").with_name("good"),
            CompilationOptions::new(),
        );
        warmup.add(
            KernelSource::new("").with_name("bad"),
            CompilationOptions::new(),
        );
        let mut compiler = KernelCompiler::with_default_cache("dev");
        let summary = warmup.warmup(&mut compiler);
        assert_eq!(summary.successes(), 1);
        assert_eq!(summary.failures(), 1);
        assert!(!summary.all_succeeded());
    }

    #[test]
    fn test_warmup_populates_cache() {
        let cache = Arc::new(CompilationCache::memory_only(64));
        let mut compiler = KernelCompiler::new("dev", cache.clone());
        let mut warmup = WarmupCompiler::new();
        warmup.add(
            KernelSource::new("__kernel void w() {}"),
            CompilationOptions::new(),
        );
        warmup.warmup(&mut compiler);
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn test_warmup_kernel_count() {
        let mut warmup = WarmupCompiler::new();
        warmup.add(KernelSource::new("a"), CompilationOptions::new());
        warmup.add(KernelSource::new("b"), CompilationOptions::new());
        assert_eq!(warmup.kernel_count(), 2);
    }

    #[test]
    fn test_warmup_summary_display() {
        let summary = WarmupSummary {
            results: vec![WarmupResult {
                name: Some("k".into()),
                success: true,
                compile_time: Duration::from_millis(5),
                binary_size: 256,
                error: None,
            }],
            total_time: Duration::from_millis(5),
        };
        let s = format!("{summary}");
        assert!(s.contains("1/1 ok"));
    }

    // -----------------------------------------------------------------------
    // estimate_binary_size
    // -----------------------------------------------------------------------

    #[test]
    fn test_estimate_binary_size_minimum() {
        assert_eq!(estimate_binary_size(0), 64);
        assert_eq!(estimate_binary_size(1), 64);
        assert_eq!(estimate_binary_size(15), 64);
    }

    #[test]
    fn test_estimate_binary_size_scales() {
        assert_eq!(estimate_binary_size(100), 400);
        assert_eq!(estimate_binary_size(1000), 4000);
    }

    // -----------------------------------------------------------------------
    // CompilationStats
    // -----------------------------------------------------------------------

    #[test]
    fn test_compilation_stats_default() {
        let s = CompilationStats::default();
        assert_eq!(s.compilations, 0);
        assert_eq!(s.cache_hit_rate(), 0.0);
        assert_eq!(s.average_compile_time(), Duration::ZERO);
        assert_eq!(s.success_rate(), 0.0);
    }

    #[test]
    fn test_compilation_stats_display() {
        let s = CompilationStats {
            compilations: 10,
            successes: 8,
            failures: 2,
            cache_hits: 5,
            cache_misses: 5,
            total_compile_time: Duration::from_millis(100),
            total_binary_bytes: 4096,
        };
        let d = format!("{s}");
        assert!(d.contains("compiled=10"));
        assert!(d.contains("50.0%"));
    }

    // -----------------------------------------------------------------------
    // Property-like tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_same_source_same_hash() {
        for _ in 0..20 {
            let src = "__kernel void f() { int x = 1; }";
            let h1 = hash_str(src);
            let h2 = hash_str(src);
            assert_eq!(h1, h2);
        }
    }

    #[test]
    fn test_same_source_options_same_binary() {
        let src = KernelSource::new("__kernel void f() {}");
        let opts = CompilationOptions::new().fast_math();

        let mut c1 = KernelCompiler::with_default_cache("dev");
        let mut c2 = KernelCompiler::with_default_cache("dev");

        let b1 = c1.compile(&src, &opts).unwrap();
        let b2 = c2.compile(&src, &opts).unwrap();
        assert_eq!(b1.binary, b2.binary);
        assert_eq!(b1.source_hash, b2.source_hash);
        assert_eq!(b1.options_hash, b2.options_hash);
    }

    #[test]
    fn test_different_source_different_hash() {
        let hashes: HashSet<u64> = (0..20)
            .map(|i| hash_str(&format!("__kernel void f{i}() {{}}")))
            .collect();
        // All 20 should be distinct.
        assert_eq!(hashes.len(), 20);
    }

    #[test]
    fn test_cache_invalidation_on_source_change() {
        let cache = Arc::new(CompilationCache::memory_only(64));
        let mut compiler = KernelCompiler::new("dev", cache.clone());
        let opts = CompilationOptions::new();

        let src_v1 = KernelSource::new("__kernel void f() { int a = 1; }");
        let b1 = compiler.compile(&src_v1, &opts).unwrap();

        let src_v2 = KernelSource::new("__kernel void f() { int a = 2; }");
        let b2 = compiler.compile(&src_v2, &opts).unwrap();

        // Different source → different binary (no stale cache).
        assert_ne!(b1.source_hash, b2.source_hash);
        assert_ne!(b1.binary, b2.binary);
    }

    #[test]
    fn test_specializer_roundtrip_through_compiler() {
        let mut compiler = KernelCompiler::with_default_cache("dev");
        let base = KernelSource::new("__kernel void f() { int s = TILE_SIZE; }");
        let spec = KernelSpecializer::new().set_int("TILE_SIZE", 16);
        let specialised = spec.specialize(&base);
        let bin = compiler.compile(&specialised, &CompilationOptions::new()).unwrap();
        assert!(!bin.binary.is_empty());
    }

    #[test]
    fn test_compiled_binary_size() {
        let bin = CompiledBinary {
            binary: vec![0; 128],
            build_log: String::new(),
            compile_time: Duration::ZERO,
            source_hash: 0,
            options_hash: 0,
        };
        assert_eq!(bin.binary_size(), 128);
    }

    #[test]
    fn test_preprocessor_error_display() {
        let e = PreprocessorError { line: 5, message: "oops".into() };
        let s = format!("{e}");
        assert!(s.contains("line 5"));
        assert!(s.contains("oops"));
    }

    #[test]
    fn test_preprocessor_error_display_no_line() {
        let e = PreprocessorError { line: 0, message: "oops".into() };
        let s = format!("{e}");
        assert!(!s.contains("line 0"));
        assert!(s.contains("oops"));
    }

    #[test]
    fn test_diagnostic_severity_display() {
        assert_eq!(format!("{}", DiagnosticSeverity::Error), "error");
        assert_eq!(format!("{}", DiagnosticSeverity::Warning), "warning");
        assert_eq!(format!("{}", DiagnosticSeverity::Note), "note");
    }
}
