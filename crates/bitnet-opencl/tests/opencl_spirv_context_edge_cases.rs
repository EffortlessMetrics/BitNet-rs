//! Edge-case tests for OpenCL SPIR-V kernel registry, SPIR-V types,
//! and context pool with mock factory.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use bitnet_opencl::context_pool::{
    ContextFactory, ContextPool, ContextPoolConfig, ContextPoolError, MemoryBytes,
};
use bitnet_opencl::{
    CompileOptions, CompilerBackend, KernelSource, OptimizationLevel, SPIRV_MAGIC, SpirVCompiler,
    SpirVError, SpirVModule, SpirVValidator, SpirvKernelRegistry,
};

// ── SpirvKernelRegistry tests ────────────────────────────────────────────────

#[test]
fn empty_registry() {
    let reg = SpirvKernelRegistry::new();
    assert!(reg.is_empty());
    assert_eq!(reg.len(), 0);
    assert!(reg.get("nonexistent").is_none());
}

#[test]
fn default_registry_is_empty() {
    let reg = SpirvKernelRegistry::default();
    assert!(reg.is_empty());
}

#[test]
fn register_cl_source() {
    let mut reg = SpirvKernelRegistry::new();
    reg.register("my_kernel", KernelSource::ClSource("__kernel void f() {}".into()));
    assert_eq!(reg.len(), 1);
    assert!(!reg.is_empty());
    assert!(reg.get("my_kernel").is_some());
}

#[test]
fn register_spirv_binary() {
    let mut reg = SpirvKernelRegistry::new();
    reg.register("binary_kernel", KernelSource::SpirV(vec![0x07, 0x23, 0x02, 0x03]));
    assert!(reg.get("binary_kernel").is_some());
    match reg.get("binary_kernel").unwrap() {
        KernelSource::SpirV(bytes) => assert_eq!(bytes.len(), 4),
        _ => panic!("expected SpirV variant"),
    }
}

#[test]
fn register_overwrites_same_name() {
    let mut reg = SpirvKernelRegistry::new();
    reg.register("k", KernelSource::ClSource("v1".into()));
    reg.register("k", KernelSource::ClSource("v2".into()));
    assert_eq!(reg.len(), 1);
    match reg.get("k").unwrap() {
        KernelSource::ClSource(s) => assert_eq!(s, "v2"),
        _ => panic!("expected ClSource"),
    }
}

#[test]
fn register_multiple_kernels() {
    let mut reg = SpirvKernelRegistry::new();
    reg.register("a", KernelSource::ClSource("a".into()));
    reg.register("b", KernelSource::SpirV(vec![1, 2, 3]));
    reg.register("c", KernelSource::ClSource("c".into()));
    assert_eq!(reg.len(), 3);
}

#[test]
fn names_iterator() {
    let mut reg = SpirvKernelRegistry::new();
    reg.register("alpha", KernelSource::ClSource("a".into()));
    reg.register("beta", KernelSource::SpirV(vec![]));
    let mut names: Vec<&str> = reg.names().collect();
    names.sort();
    assert_eq!(names, vec!["alpha", "beta"]);
}

#[test]
fn builtin_kernels_populated() {
    let reg = SpirvKernelRegistry::with_builtin_kernels();
    assert!(!reg.is_empty());
    assert!(reg.get("matmul_i2").is_some());
    assert!(reg.get("dequant_i2s").is_some());
}

#[test]
fn builtin_kernels_are_cl_source() {
    let reg = SpirvKernelRegistry::with_builtin_kernels();
    for name in reg.names() {
        match reg.get(name).unwrap() {
            KernelSource::ClSource(src) => assert!(!src.is_empty()),
            KernelSource::SpirV(_) => {} // also acceptable
        }
    }
}

#[test]
fn kernel_source_clone() {
    let src = KernelSource::ClSource("test".into());
    let cloned = src.clone();
    match (&src, &cloned) {
        (KernelSource::ClSource(a), KernelSource::ClSource(b)) => assert_eq!(a, b),
        _ => panic!("clone mismatch"),
    }
}

#[test]
fn kernel_source_debug() {
    let src = KernelSource::ClSource("hello".into());
    let dbg = format!("{src:?}");
    assert!(dbg.contains("ClSource"));
}

// ── SPIR-V types tests ──────────────────────────────────────────────────────

#[test]
fn spirv_magic_constant() {
    assert_eq!(SPIRV_MAGIC, 0x0723_0203);
}

#[test]
fn compile_options_default() {
    let opts = CompileOptions::default();
    assert!(opts.target_device.is_none());
    assert_eq!(opts.optimization_level, OptimizationLevel::Full);
    assert!(opts.defines.is_empty());
}

#[test]
fn compile_options_custom() {
    let opts = CompileOptions {
        target_device: Some("arc".into()),
        optimization_level: OptimizationLevel::None,
        defines: vec![("KEY".into(), "VAL".into())],
    };
    assert_eq!(opts.target_device.as_deref(), Some("arc"));
    assert_eq!(opts.optimization_level, OptimizationLevel::None);
    assert_eq!(opts.defines.len(), 1);
}

#[test]
fn optimization_level_equality() {
    assert_eq!(OptimizationLevel::None, OptimizationLevel::None);
    assert_ne!(OptimizationLevel::None, OptimizationLevel::Basic);
    assert_ne!(OptimizationLevel::Basic, OptimizationLevel::Full);
}

#[test]
fn compiler_backend_equality() {
    assert_eq!(CompilerBackend::Clang, CompilerBackend::Clang);
    assert_ne!(CompilerBackend::Clang, CompilerBackend::Ocloc);
}

#[test]
fn spirv_module_debug() {
    let m = SpirVModule {
        bytecode: vec![1, 2, 3, 4],
        source_hash: "abc123".into(),
        compiler: Some(CompilerBackend::Clang),
    };
    let dbg = format!("{m:?}");
    assert!(dbg.contains("abc123"));
    assert!(dbg.contains("Clang"));
}

#[test]
fn spirv_module_clone() {
    let m = SpirVModule { bytecode: vec![0x07, 0x23], source_hash: "hash".into(), compiler: None };
    let cloned = m.clone();
    assert_eq!(cloned.bytecode, m.bytecode);
    assert_eq!(cloned.source_hash, m.source_hash);
}

#[test]
fn spirv_compiler_no_backend() {
    let compiler = SpirVCompiler::with_backend(None);
    assert!(compiler.backend().is_none());
}

#[test]
fn spirv_compiler_explicit_backend() {
    let compiler = SpirVCompiler::with_backend(Some(CompilerBackend::Clang));
    assert_eq!(compiler.backend(), Some(CompilerBackend::Clang));
}

#[test]
fn spirv_compiler_no_backend_compile_fails() {
    let compiler = SpirVCompiler::with_backend(None);
    let result = compiler.compile_to_spirv("__kernel void f() {}", &CompileOptions::default());
    assert!(result.is_err());
    match result.unwrap_err() {
        SpirVError::NoCompilerAvailable => {}
        other => panic!("expected NoCompilerAvailable, got {other:?}"),
    }
}

// ── SpirVError tests ─────────────────────────────────────────────────────────

#[test]
fn spirv_error_display_validation() {
    let err = SpirVError::ValidationFailed("bad magic".into());
    assert!(format!("{err}").contains("bad magic"));
}

#[test]
fn spirv_error_display_compilation() {
    let err = SpirVError::CompilationFailed("syntax error".into());
    assert!(format!("{err}").contains("syntax error"));
}

#[test]
fn spirv_error_display_no_compiler() {
    let err = SpirVError::NoCompilerAvailable;
    assert!(format!("{err}").contains("no SPIR-V compiler"));
}

// ── ContextPool with mock factory ────────────────────────────────────────────

struct MockFactory {
    memory_per_context: MemoryBytes,
    total_used: AtomicU64,
    fail_on_create: bool,
}

impl MockFactory {
    fn new(memory_per_context: MemoryBytes) -> Self {
        Self { memory_per_context, total_used: AtomicU64::new(0), fail_on_create: false }
    }

    fn failing() -> Self {
        Self { memory_per_context: 0, total_used: AtomicU64::new(0), fail_on_create: true }
    }
}

impl ContextFactory for MockFactory {
    fn create_context(&self, id: &str) -> Result<MemoryBytes, ContextPoolError> {
        if self.fail_on_create {
            return Err(ContextPoolError::CreationFailed {
                id: id.to_string(),
                reason: "mock failure".into(),
            });
        }
        self.total_used.fetch_add(self.memory_per_context, Ordering::Relaxed);
        Ok(self.memory_per_context)
    }

    fn compile_programs(&self, _id: &str) -> Result<(), ContextPoolError> {
        Ok(())
    }

    fn release_context(&self, _id: &str) -> Result<(), ContextPoolError> {
        self.total_used.fetch_sub(self.memory_per_context, Ordering::Relaxed);
        Ok(())
    }

    fn total_gpu_memory_used(&self) -> MemoryBytes {
        self.total_used.load(Ordering::Relaxed)
    }
}

fn mock_pool(max_contexts: usize) -> ContextPool {
    let factory = Arc::new(MockFactory::new(1024 * 1024)); // 1 MiB each
    let config = ContextPoolConfig {
        max_contexts,
        memory_limit: 100 * 1024 * 1024, // 100 MiB
        idle_timeout: Duration::from_secs(60),
        lazy_creation: true,
    };
    ContextPool::new(config, factory)
}

#[test]
fn context_pool_starts_empty() {
    let pool = mock_pool(4);
    assert!(pool.is_empty());
    assert_eq!(pool.len(), 0);
    assert_eq!(pool.total_memory_usage(), 0);
}

#[test]
fn context_pool_acquire_creates() {
    let pool = mock_pool(4);
    let entry = pool.acquire("model-a").unwrap();
    assert_eq!(entry.id, "model-a");
    assert_eq!(pool.len(), 1);
    assert!(entry.compiled);
    assert!(entry.in_use);
}

#[test]
fn context_pool_acquire_existing_reuses() {
    let pool = mock_pool(4);
    let e1 = pool.acquire("model-a").unwrap();
    pool.release("model-a").unwrap();
    let e2 = pool.acquire("model-a").unwrap();
    assert_eq!(pool.len(), 1); // still just one context
    assert!(e2.use_count > e1.use_count);
}

#[test]
fn context_pool_release() {
    let pool = mock_pool(4);
    pool.acquire("model-a").unwrap();
    pool.release("model-a").unwrap();
    // Still in pool but not in use
    assert_eq!(pool.len(), 1);
}

#[test]
fn context_pool_release_nonexistent_fails() {
    let pool = mock_pool(4);
    let err = pool.release("nonexistent").unwrap_err();
    assert!(matches!(err, ContextPoolError::NotFound { .. }));
}

#[test]
fn context_pool_evict() {
    let pool = mock_pool(4);
    pool.acquire("model-a").unwrap();
    pool.release("model-a").unwrap();
    pool.evict("model-a").unwrap();
    assert!(pool.is_empty());
}

#[test]
fn context_pool_evict_nonexistent_fails() {
    let pool = mock_pool(4);
    let err = pool.evict("ghost").unwrap_err();
    assert!(matches!(err, ContextPoolError::NotFound { .. }));
}

#[test]
fn context_pool_memory_tracking() {
    let pool = mock_pool(4);
    pool.acquire("model-a").unwrap();
    assert!(pool.total_memory_usage() > 0);
}

#[test]
fn context_pool_entries_snapshot() {
    let pool = mock_pool(4);
    pool.acquire("model-a").unwrap();
    pool.acquire("model-b").unwrap();
    let entries = pool.entries();
    assert_eq!(entries.len(), 2);
}

#[test]
fn context_pool_capacity_exhausted() {
    let pool = mock_pool(1);
    pool.acquire("model-a").unwrap();
    // Context is in use, can't evict — capacity should be exhausted
    let err = pool.acquire("model-b").unwrap_err();
    assert!(matches!(err, ContextPoolError::CapacityExhausted { .. }));
}

#[test]
fn context_pool_creation_failure() {
    let factory = Arc::new(MockFactory::failing());
    let config = ContextPoolConfig {
        max_contexts: 4,
        memory_limit: 1_000_000_000,
        idle_timeout: Duration::from_secs(60),
        lazy_creation: true,
    };
    let pool = ContextPool::new(config, factory);
    let err = pool.acquire("model-a").unwrap_err();
    assert!(matches!(err, ContextPoolError::CreationFailed { .. }));
}

// ── ContextPoolConfig tests ──────────────────────────────────────────────────

#[test]
fn context_pool_config_default() {
    let cfg = ContextPoolConfig::default();
    assert_eq!(cfg.max_contexts, 4);
    assert_eq!(cfg.memory_limit, 2 * 1024 * 1024 * 1024);
    assert_eq!(cfg.idle_timeout, Duration::from_secs(300));
    assert!(cfg.lazy_creation);
}

// ── ContextPoolError display tests ───────────────────────────────────────────

#[test]
fn context_pool_error_capacity_display() {
    let err = ContextPoolError::CapacityExhausted { max: 4 };
    assert!(format!("{err}").contains("4"));
}

#[test]
fn context_pool_error_creation_display() {
    let err = ContextPoolError::CreationFailed { id: "test".into(), reason: "boom".into() };
    let msg = format!("{err}");
    assert!(msg.contains("test"));
    assert!(msg.contains("boom"));
}

#[test]
fn context_pool_error_not_found_display() {
    let err = ContextPoolError::NotFound { id: "missing".into() };
    assert!(format!("{err}").contains("missing"));
}

#[test]
fn context_pool_error_memory_pressure_display() {
    let err = ContextPoolError::MemoryPressure { used: 100, limit: 50 };
    let msg = format!("{err}");
    assert!(msg.contains("100"));
    assert!(msg.contains("50"));
}
