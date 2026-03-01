use std::time::SystemTime;

use crate::kernel_cache::{CacheEntry, CacheKey, KernelCache, hash_source};

/// Kernel compilation options.
#[derive(Debug, Clone)]
pub struct CompilationOptions {
    pub optimization_level: OptimizationLevel,
    pub target_device: String,
    pub defines: Vec<(String, String)>,
}

/// Optimisation levels mirroring `OpenCL` `-cl-optN`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptimizationLevel {
    None,
    O1,
    O2,
    O3,
}

impl std::fmt::Display for OptimizationLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::None => write!(f, "-cl-opt-disable"),
            Self::O1 => write!(f, "-O1"),
            Self::O2 => write!(f, "-O2"),
            Self::O3 => write!(f, "-O3"),
        }
    }
}

impl Default for CompilationOptions {
    fn default() -> Self {
        Self {
            optimization_level: OptimizationLevel::O2,
            target_device: String::new(),
            defines: Vec::new(),
        }
    }
}

impl CompilationOptions {
    /// Flatten options into a single compiler-flags string used as part of
    /// the cache key.
    #[must_use]
    pub fn to_flags_string(&self) -> String {
        use std::fmt::Write;
        let mut flags = self.optimization_level.to_string();
        for (k, v) in &self.defines {
            let _ = write!(flags, " -D{k}={v}");
        }
        flags
    }
}

/// Result of a kernel compilation.
#[derive(Debug, Clone)]
pub struct CompilationResult {
    pub binary: Vec<u8>,
    pub build_log: String,
    pub warnings: Vec<String>,
}

/// Error during kernel compilation.
#[derive(Debug, thiserror::Error)]
pub enum CompileError {
    #[error("kernel source not found: {0}")]
    SourceNotFound(String),

    #[error("compilation failed: {0}")]
    CompilationFailed(String),

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
}

/// Manages the kernel compilation workflow: load source → check cache →
/// compile if needed → store result → return handle.
pub struct KernelCompiler {
    cache: KernelCache,
    sources: std::collections::HashMap<String, String>,
}

impl KernelCompiler {
    /// Create a new compiler backed by the given cache.
    #[must_use]
    pub fn new(cache: KernelCache) -> Self {
        Self { cache, sources: std::collections::HashMap::new() }
    }

    /// Register an embedded kernel source (e.g. loaded from a `.cl` file at
    /// build time).
    pub fn register_source(&mut self, name: impl Into<String>, source: impl Into<String>) {
        self.sources.insert(name.into(), source.into());
    }

    /// Compile (or retrieve from cache) a kernel.
    ///
    /// In CPU-only mode the actual compilation is mocked: the source bytes are
    /// returned directly so that callers can still exercise the cache path.
    pub fn compile(
        &self,
        kernel_name: &str,
        options: &CompilationOptions,
    ) -> Result<CompilationResult, CompileError> {
        let source = self
            .sources
            .get(kernel_name)
            .ok_or_else(|| CompileError::SourceNotFound(kernel_name.to_string()))?
            .clone();

        let src_hash = hash_source(&source);

        let key = CacheKey {
            kernel_name: kernel_name.to_string(),
            device_id: options.target_device.clone(),
            compiler_options: options.to_flags_string(),
        };

        let device = options.target_device.clone();
        let entry =
            self.cache.get_or_compile(&key, src_hash, || -> Result<CacheEntry, CompileError> {
                let binary = mock_compile(&source, options)?;
                let ts = SystemTime::now()
                    .duration_since(SystemTime::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs();

                Ok(CacheEntry {
                    binary_data: binary,
                    source_hash: src_hash,
                    timestamp: ts,
                    device_name: device.clone(),
                })
            })?;

        Ok(CompilationResult {
            binary: entry.binary_data,
            build_log: String::new(),
            warnings: Vec::new(),
        })
    }

    /// Access the underlying cache (e.g. to query stats).
    #[must_use]
    pub const fn cache(&self) -> &KernelCache {
        &self.cache
    }
}

/// Mock compilation used in CPU-only mode.
fn mock_compile(source: &str, _options: &CompilationOptions) -> Result<Vec<u8>, CompileError> {
    if source.is_empty() {
        return Err(CompileError::CompilationFailed("empty kernel source".to_string()));
    }
    // Return the UTF-8 bytes of the source as a fake binary.
    Ok(source.as_bytes().to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_compiler(enabled: bool) -> KernelCompiler {
        let cache = KernelCache::with_config(None, enabled);
        let mut compiler = KernelCompiler::new(cache);
        compiler.register_source("matmul", "__kernel void matmul() {}");
        compiler
    }

    #[test]
    fn compile_cache_miss_then_hit() {
        let compiler = make_compiler(true);
        let opts = CompilationOptions::default();

        let r1 = compiler.compile("matmul", &opts).unwrap();
        assert!(!r1.binary.is_empty());
        assert_eq!(compiler.cache().stats().misses, 1);

        let _r2 = compiler.compile("matmul", &opts).unwrap();
        assert_eq!(compiler.cache().stats().hits, 1);
    }

    #[test]
    fn compile_unknown_kernel_errors() {
        let compiler = make_compiler(true);
        let opts = CompilationOptions::default();
        let err = compiler.compile("nonexistent", &opts);
        assert!(err.is_err());
    }

    #[test]
    fn compile_with_cache_disabled_always_compiles() {
        let compiler = make_compiler(false);
        let opts = CompilationOptions::default();

        let _ = compiler.compile("matmul", &opts).unwrap();
        let _ = compiler.compile("matmul", &opts).unwrap();
        // Two misses because caching is off.
        assert_eq!(compiler.cache().stats().misses, 2);
    }

    #[test]
    fn compilation_options_flags_string() {
        let opts = CompilationOptions {
            optimization_level: OptimizationLevel::O3,
            target_device: "gpu0".into(),
            defines: vec![("TILE".into(), "16".into()), ("USE_FP16".into(), "1".into())],
        };
        let flags = opts.to_flags_string();
        assert!(flags.contains("-O3"));
        assert!(flags.contains("-DTILE=16"));
        assert!(flags.contains("-DUSE_FP16=1"));
    }

    #[test]
    fn compilation_result_contains_binary() {
        let compiler = make_compiler(true);
        let opts = CompilationOptions::default();
        let result = compiler.compile("matmul", &opts).unwrap();
        assert_eq!(result.binary, b"__kernel void matmul() {}");
    }

    #[test]
    fn mock_compile_empty_source_fails() {
        let cache = KernelCache::with_config(None, true);
        let mut compiler = KernelCompiler::new(cache);
        compiler.register_source("empty", "");
        let opts = CompilationOptions::default();
        assert!(compiler.compile("empty", &opts).is_err());
    }
}
