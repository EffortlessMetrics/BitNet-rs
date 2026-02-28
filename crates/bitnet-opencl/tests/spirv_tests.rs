//! Tests for the SPIR-V compilation pipeline.

use bitnet_opencl::spirv::{
    CompileOptions, OptimizationLevel, SPIRV_MAGIC, SpirVCache, SpirVCompiler, SpirVError,
    SpirVModule, SpirVValidator, build_test_spirv, source_hash,
};
use bitnet_opencl::spirv_kernels::{KernelSource, SpirvKernelRegistry};

// ── Magic number validation ──────────────────────────────────────────────────

#[test]
fn magic_number_constant_matches_spec() {
    assert_eq!(SPIRV_MAGIC, 0x0723_0203);
}

#[test]
fn valid_spirv_passes_magic_check() {
    let spv = build_test_spirv(1, 0);
    assert!(SpirVValidator::check_magic(&spv).is_ok());
}

#[test]
fn invalid_magic_rejected() {
    let mut spv = build_test_spirv(1, 0);
    // Corrupt the magic number.
    spv[0] = 0xFF;
    assert!(SpirVValidator::check_magic(&spv).is_err());
}

#[test]
fn empty_bytes_rejected() {
    assert!(SpirVValidator::validate_bytes(&[]).is_err());
}

#[test]
fn too_short_bytes_rejected() {
    // 19 bytes is one short of the minimum 20-byte header.
    let bytes = vec![0u8; 19];
    assert!(SpirVValidator::validate_bytes(&bytes).is_err());
}

// ── Version validation ───────────────────────────────────────────────────────

#[test]
fn version_1_0_accepted() {
    let spv = build_test_spirv(1, 0);
    assert!(SpirVValidator::validate_bytes(&spv).is_ok());
}

#[test]
fn version_1_5_accepted() {
    let spv = build_test_spirv(1, 5);
    assert!(SpirVValidator::validate_bytes(&spv).is_ok());
}

#[test]
fn version_1_6_accepted() {
    let spv = build_test_spirv(1, 6);
    assert!(SpirVValidator::validate_bytes(&spv).is_ok());
}

#[test]
fn version_2_0_rejected() {
    let spv = build_test_spirv(2, 0);
    let err = SpirVValidator::validate_bytes(&spv).unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("unsupported"), "unexpected error: {msg}");
}

#[test]
fn version_1_7_rejected() {
    let spv = build_test_spirv(1, 7);
    let err = SpirVValidator::validate_bytes(&spv).unwrap_err();
    assert!(err.to_string().contains("unsupported"));
}

// ── Full validation on valid binary ──────────────────────────────────────────

#[test]
fn full_validation_passes_for_valid_binary() {
    let spv = build_test_spirv(1, 3);
    assert!(SpirVValidator::validate_bytes(&spv).is_ok());
}

#[test]
fn all_zeroes_rejected() {
    let bytes = vec![0u8; 20];
    assert!(SpirVValidator::validate_bytes(&bytes).is_err());
}

// ── Capability checking ──────────────────────────────────────────────────────

#[test]
fn capability_check_on_minimal_binary_returns_false() {
    let spv = build_test_spirv(1, 5);
    // Minimal header has no capability instructions.
    assert!(!SpirVValidator::has_capability(&spv, 1));
}

#[test]
fn capability_check_finds_embedded_capability() {
    let mut spv = build_test_spirv(1, 5);
    // Append an OpCapability instruction: word-count 2, opcode 17 →
    // header word = (2 << 16) | 17 = 0x0002_0011
    let op_cap: u32 = (2 << 16) | 17;
    spv.extend_from_slice(&op_cap.to_le_bytes());
    // Operand: capability ID 22 (Image1D).
    spv.extend_from_slice(&22u32.to_le_bytes());
    assert!(SpirVValidator::has_capability(&spv, 22));
    assert!(!SpirVValidator::has_capability(&spv, 99));
}

// ── Compiler detection ───────────────────────────────────────────────────────

#[test]
fn compiler_with_no_backend_returns_error() {
    let compiler = SpirVCompiler::with_backend(None);
    assert!(compiler.backend().is_none());
    let err =
        compiler.compile_to_spirv("__kernel void f(){}", &CompileOptions::default()).unwrap_err();
    assert!(
        matches!(err, SpirVError::NoCompilerAvailable),
        "expected NoCompilerAvailable, got {err:?}"
    );
}

#[test]
#[ignore = "requires clang with SPIR-V target - run manually"]
fn clang_backend_compiles_trivial_kernel() {
    use bitnet_opencl::CompilerBackend;
    let compiler = SpirVCompiler::with_backend(Some(CompilerBackend::Clang));
    let module = compiler
        .compile_to_spirv("__kernel void noop() {}", &CompileOptions::default())
        .expect("clang compilation failed");
    assert!(!module.bytecode.is_empty());
    assert!(SpirVValidator::validate_bytes(&module.bytecode).is_ok());
}

#[test]
#[ignore = "requires Intel ocloc - run manually"]
fn ocloc_backend_compiles_trivial_kernel() {
    use bitnet_opencl::CompilerBackend;
    let compiler = SpirVCompiler::with_backend(Some(CompilerBackend::Ocloc));
    let module = compiler
        .compile_to_spirv("__kernel void noop() {}", &CompileOptions::default())
        .expect("ocloc compilation failed");
    assert!(!module.bytecode.is_empty());
}

// ── Cache ────────────────────────────────────────────────────────────────────

#[test]
fn cache_stores_and_retrieves() {
    let cache = SpirVCache::new();
    assert!(cache.is_empty());

    let module = SpirVModule {
        bytecode: build_test_spirv(1, 5),
        source_hash: "abc123".into(),
        compiler: None,
    };
    cache.insert(module.clone());
    assert_eq!(cache.len(), 1);

    let hit = cache.get("abc123").expect("cache miss");
    assert_eq!(hit.bytecode, module.bytecode);
}

#[test]
fn cache_miss_returns_none() {
    let cache = SpirVCache::new();
    assert!(cache.get("nonexistent").is_none());
}

#[test]
fn source_hash_changes_invalidate_cache() {
    let opts = CompileOptions::default();
    let hash_a = source_hash("kernel void a(){}", &opts);
    let hash_b = source_hash("kernel void b(){}", &opts);
    assert_ne!(hash_a, hash_b, "different sources must hash differently");

    // Simulate: cache only contains hash_a.
    let cache = SpirVCache::new();
    cache.insert(SpirVModule {
        bytecode: build_test_spirv(1, 5),
        source_hash: hash_a.clone(),
        compiler: None,
    });
    assert!(cache.get(&hash_a).is_some());
    assert!(cache.get(&hash_b).is_none());
}

#[test]
fn compile_options_affect_hash() {
    let src = "__kernel void f(){}";
    let opts_o0 =
        CompileOptions { optimization_level: OptimizationLevel::None, ..CompileOptions::default() };
    let opts_o2 =
        CompileOptions { optimization_level: OptimizationLevel::Full, ..CompileOptions::default() };
    assert_ne!(
        source_hash(src, &opts_o0),
        source_hash(src, &opts_o2),
        "different options must produce different hashes"
    );
}

#[test]
fn cache_clear_removes_all_entries() {
    let cache = SpirVCache::new();
    cache.insert(SpirVModule {
        bytecode: build_test_spirv(1, 5),
        source_hash: "h1".into(),
        compiler: None,
    });
    cache.insert(SpirVModule {
        bytecode: build_test_spirv(1, 3),
        source_hash: "h2".into(),
        compiler: None,
    });
    assert_eq!(cache.len(), 2);
    cache.clear();
    assert!(cache.is_empty());
}

// ── Kernel registry ──────────────────────────────────────────────────────────

#[test]
fn registry_starts_empty() {
    let reg = SpirvKernelRegistry::new();
    assert!(reg.is_empty());
    assert_eq!(reg.len(), 0);
}

#[test]
fn registry_registers_and_retrieves_cl_source() {
    let mut reg = SpirvKernelRegistry::new();
    reg.register("test_kernel", KernelSource::ClSource("src".into()));
    assert_eq!(reg.len(), 1);
    let src = reg.get("test_kernel").expect("registry miss");
    assert!(matches!(src, KernelSource::ClSource(s) if s == "src"));
}

#[test]
fn registry_registers_spirv_binary() {
    let mut reg = SpirvKernelRegistry::new();
    let spv = build_test_spirv(1, 5);
    reg.register("kern", KernelSource::SpirV(spv.clone()));
    let entry = reg.get("kern").expect("registry miss");
    assert!(matches!(entry, KernelSource::SpirV(b) if b == &spv));
}

#[test]
fn builtin_registry_has_expected_kernels() {
    let reg = SpirvKernelRegistry::with_builtin_kernels();
    assert!(reg.get("matmul_i2").is_some());
    assert!(reg.get("dequant_i2s").is_some());
    assert!(reg.get("nonexistent").is_none());
}

#[test]
fn registry_names_iterator() {
    let reg = SpirvKernelRegistry::with_builtin_kernels();
    let names: Vec<&str> = reg.names().collect();
    assert!(names.contains(&"matmul_i2"));
    assert!(names.contains(&"dequant_i2s"));
}

// ── Fallback behaviour ───────────────────────────────────────────────────────

#[test]
fn fallback_to_cl_source_when_no_compiler() {
    // When no compiler is available, callers should use ClSource directly.
    let compiler = SpirVCompiler::with_backend(None);
    let result = compiler.compile_to_spirv("__kernel void f(){}", &CompileOptions::default());
    assert!(matches!(result, Err(SpirVError::NoCompilerAvailable)));

    // The registry still provides the .cl source for runtime JIT.
    let reg = SpirvKernelRegistry::with_builtin_kernels();
    let src = reg.get("matmul_i2").unwrap();
    assert!(matches!(src, KernelSource::ClSource(_)), "should fall back to .cl source");
}

// ── Build test helper ────────────────────────────────────────────────────────

#[test]
fn build_test_spirv_produces_valid_binary() {
    for minor in 0..=6 {
        let spv = build_test_spirv(1, minor);
        assert!(SpirVValidator::validate_bytes(&spv).is_ok(), "v1.{minor} should be valid");
    }
}
