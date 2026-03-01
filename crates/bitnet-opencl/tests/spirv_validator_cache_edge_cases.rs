//! Edge-case tests for SpirVValidator, SpirVCache, SpirVModule, SpirVCompiler,
//! CompileOptions, OptimizationLevel, CompilerBackend, and build_test_spirv helper.
//!
//! All tests are pure-CPU — no GPU device or SPIR-V compiler needed.

use bitnet_opencl::spirv::{
    CompileOptions, CompilerBackend, OptimizationLevel, SPIRV_MAGIC, SpirVCache, SpirVCompiler,
    SpirVError, SpirVModule, SpirVValidator, build_test_spirv, source_hash,
};

// ---------------------------------------------------------------------------
// OptimizationLevel
// ---------------------------------------------------------------------------

#[test]
fn optimization_level_debug_all() {
    for level in [OptimizationLevel::None, OptimizationLevel::Basic, OptimizationLevel::Full] {
        let dbg = format!("{level:?}");
        assert!(!dbg.is_empty());
    }
}

#[test]
fn optimization_level_eq() {
    assert_eq!(OptimizationLevel::None, OptimizationLevel::None);
    assert_ne!(OptimizationLevel::None, OptimizationLevel::Full);
}

#[test]
fn optimization_level_copy_clone() {
    let level = OptimizationLevel::Basic;
    let level2 = level;
    let level3 = level.clone();
    assert_eq!(level2, level3);
}

// ---------------------------------------------------------------------------
// CompileOptions
// ---------------------------------------------------------------------------

#[test]
fn compile_options_default() {
    let opts = CompileOptions::default();
    assert!(opts.target_device.is_none());
    assert_eq!(opts.optimization_level, OptimizationLevel::Full);
    assert!(opts.defines.is_empty());
}

#[test]
fn compile_options_debug_clone_eq() {
    let opts = CompileOptions {
        target_device: Some("arc_a770".into()),
        optimization_level: OptimizationLevel::Basic,
        defines: vec![("KEY".into(), "VAL".into())],
    };
    let opts2 = opts.clone();
    assert_eq!(opts, opts2);
    let dbg = format!("{opts:?}");
    assert!(dbg.contains("CompileOptions"));
    assert!(dbg.contains("arc_a770"));
}

// ---------------------------------------------------------------------------
// CompilerBackend
// ---------------------------------------------------------------------------

#[test]
fn compiler_backend_debug_eq() {
    assert_eq!(CompilerBackend::Clang, CompilerBackend::Clang);
    assert_ne!(CompilerBackend::Clang, CompilerBackend::Ocloc);
    let dbg = format!("{:?}", CompilerBackend::Ocloc);
    assert!(dbg.contains("Ocloc"));
}

#[test]
fn compiler_backend_copy() {
    let b = CompilerBackend::Clang;
    let b2 = b;
    assert_eq!(b, b2);
}

// ---------------------------------------------------------------------------
// SpirVModule
// ---------------------------------------------------------------------------

#[test]
fn spirv_module_debug_clone() {
    let module = SpirVModule {
        bytecode: vec![1, 2, 3, 4],
        source_hash: "abc123".into(),
        compiler: Some(CompilerBackend::Clang),
    };
    let m2 = module.clone();
    assert_eq!(m2.bytecode, vec![1, 2, 3, 4]);
    assert_eq!(m2.source_hash, "abc123");
    assert_eq!(m2.compiler, Some(CompilerBackend::Clang));
    let dbg = format!("{module:?}");
    assert!(dbg.contains("SpirVModule"));
}

#[test]
fn spirv_module_no_compiler() {
    let module = SpirVModule { bytecode: vec![], source_hash: "none".into(), compiler: None };
    assert!(module.compiler.is_none());
}

// ---------------------------------------------------------------------------
// SpirVValidator
// ---------------------------------------------------------------------------

#[test]
fn validator_valid_spirv_10() {
    let bytes = build_test_spirv(1, 0);
    assert!(SpirVValidator::validate_bytes(&bytes).is_ok());
}

#[test]
fn validator_valid_spirv_16() {
    let bytes = build_test_spirv(1, 6);
    assert!(SpirVValidator::validate_bytes(&bytes).is_ok());
}

#[test]
fn validator_too_short() {
    let bytes = vec![0u8; 4];
    assert!(SpirVValidator::validate_bytes(&bytes).is_err());
}

#[test]
fn validator_empty() {
    assert!(SpirVValidator::validate_bytes(&[]).is_err());
}

#[test]
fn validator_bad_magic() {
    let mut bytes = build_test_spirv(1, 0);
    bytes[0] = 0xFF; // corrupt magic
    assert!(SpirVValidator::check_magic(&bytes).is_err());
}

#[test]
fn validator_bad_version() {
    let mut bytes = build_test_spirv(1, 0);
    // Set version to 2.0 (unsupported)
    let bad_version: u32 = (2 << 16) | (0 << 8);
    bytes[4..8].copy_from_slice(&bad_version.to_le_bytes());
    assert!(SpirVValidator::check_version(&bytes).is_err());
}

#[test]
fn validator_version_too_short() {
    let bytes = vec![0u8; 6]; // less than 8 bytes
    assert!(SpirVValidator::check_version(&bytes).is_err());
}

#[test]
fn validator_check_length_exactly_20() {
    let bytes = build_test_spirv(1, 0);
    assert_eq!(bytes.len(), 20);
    assert!(SpirVValidator::check_length(&bytes).is_ok());
}

#[test]
fn validator_check_length_19() {
    let bytes = vec![0u8; 19];
    assert!(SpirVValidator::check_length(&bytes).is_err());
}

#[test]
fn validator_has_capability_no_data() {
    let bytes = build_test_spirv(1, 0);
    // No capability instructions in minimal header
    assert!(!SpirVValidator::has_capability(&bytes, 0));
}

#[test]
fn validator_has_capability_too_short() {
    let bytes = vec![0u8; 10];
    assert!(!SpirVValidator::has_capability(&bytes, 0));
}

#[test]
fn validator_has_capability_with_data() {
    let mut bytes = build_test_spirv(1, 0);
    // Append OpCapability (opcode 17, wordcount 2): (2 << 16) | 17 = 0x00020011
    let op_cap: u32 = (2 << 16) | 17;
    bytes.extend_from_slice(&op_cap.to_le_bytes());
    // Capability operand: e.g., Shader = 1
    bytes.extend_from_slice(&1u32.to_le_bytes());

    assert!(SpirVValidator::has_capability(&bytes, 1));
    assert!(!SpirVValidator::has_capability(&bytes, 99));
}

// ---------------------------------------------------------------------------
// SPIRV_MAGIC constant
// ---------------------------------------------------------------------------

#[test]
fn spirv_magic_value() {
    assert_eq!(SPIRV_MAGIC, 0x0723_0203);
}

// ---------------------------------------------------------------------------
// build_test_spirv
// ---------------------------------------------------------------------------

#[test]
fn build_test_spirv_correct_length() {
    let bytes = build_test_spirv(1, 3);
    assert_eq!(bytes.len(), 20);
}

#[test]
fn build_test_spirv_has_magic() {
    let bytes = build_test_spirv(1, 0);
    let magic = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
    assert_eq!(magic, SPIRV_MAGIC);
}

#[test]
fn build_test_spirv_version_encoded() {
    let bytes = build_test_spirv(1, 5);
    let version = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
    let major = (version >> 16) & 0xFF;
    let minor = (version >> 8) & 0xFF;
    assert_eq!(major, 1);
    assert_eq!(minor, 5);
}

// ---------------------------------------------------------------------------
// SpirVCompiler
// ---------------------------------------------------------------------------

#[test]
fn compiler_with_no_backend() {
    let compiler = SpirVCompiler::with_backend(None);
    assert!(compiler.backend().is_none());
}

#[test]
fn compiler_with_explicit_backend() {
    let compiler = SpirVCompiler::with_backend(Some(CompilerBackend::Clang));
    assert_eq!(compiler.backend(), Some(CompilerBackend::Clang));
}

#[test]
fn compiler_no_backend_compile_fails() {
    let compiler = SpirVCompiler::with_backend(None);
    let result = compiler.compile_to_spirv("__kernel void k() {}", &CompileOptions::default());
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(format!("{err}").contains("no SPIR-V compiler"));
}

// ---------------------------------------------------------------------------
// SpirVCache
// ---------------------------------------------------------------------------

#[test]
fn cache_starts_empty() {
    let cache = SpirVCache::new();
    assert!(cache.is_empty());
    assert_eq!(cache.len(), 0);
}

#[test]
fn cache_default() {
    let cache = SpirVCache::default();
    assert!(cache.is_empty());
}

#[test]
fn cache_insert_and_get() {
    let cache = SpirVCache::new();
    let module =
        SpirVModule { bytecode: vec![1, 2, 3], source_hash: "hash1".into(), compiler: None };
    cache.insert(module);
    assert_eq!(cache.len(), 1);

    let found = cache.get("hash1").unwrap();
    assert_eq!(found.bytecode, vec![1, 2, 3]);
}

#[test]
fn cache_get_nonexistent() {
    let cache = SpirVCache::new();
    assert!(cache.get("nonexistent").is_none());
}

#[test]
fn cache_insert_replaces() {
    let cache = SpirVCache::new();
    let m1 = SpirVModule { bytecode: vec![1], source_hash: "same_hash".into(), compiler: None };
    let m2 = SpirVModule {
        bytecode: vec![2],
        source_hash: "same_hash".into(),
        compiler: Some(CompilerBackend::Ocloc),
    };
    cache.insert(m1);
    cache.insert(m2);
    assert_eq!(cache.len(), 1);
    let found = cache.get("same_hash").unwrap();
    assert_eq!(found.bytecode, vec![2]);
}

#[test]
fn cache_clear() {
    let cache = SpirVCache::new();
    cache.insert(SpirVModule { bytecode: vec![], source_hash: "a".into(), compiler: None });
    cache.insert(SpirVModule { bytecode: vec![], source_hash: "b".into(), compiler: None });
    assert_eq!(cache.len(), 2);
    cache.clear();
    assert!(cache.is_empty());
}

#[test]
fn cache_multiple_entries() {
    let cache = SpirVCache::new();
    for i in 0..5 {
        cache.insert(SpirVModule {
            bytecode: vec![i as u8],
            source_hash: format!("hash_{i}"),
            compiler: None,
        });
    }
    assert_eq!(cache.len(), 5);
    for i in 0..5 {
        assert!(cache.get(&format!("hash_{i}")).is_some());
    }
}

// ---------------------------------------------------------------------------
// source_hash
// ---------------------------------------------------------------------------

#[test]
fn source_hash_deterministic() {
    let opts = CompileOptions::default();
    let h1 = source_hash("kernel void k() {}", &opts);
    let h2 = source_hash("kernel void k() {}", &opts);
    assert_eq!(h1, h2);
}

#[test]
fn source_hash_different_source() {
    let opts = CompileOptions::default();
    let h1 = source_hash("kernel void a() {}", &opts);
    let h2 = source_hash("kernel void b() {}", &opts);
    assert_ne!(h1, h2);
}

#[test]
fn source_hash_different_options() {
    let opts1 =
        CompileOptions { optimization_level: OptimizationLevel::None, ..Default::default() };
    let opts2 =
        CompileOptions { optimization_level: OptimizationLevel::Full, ..Default::default() };
    let h1 = source_hash("same source", &opts1);
    let h2 = source_hash("same source", &opts2);
    assert_ne!(h1, h2);
}

#[test]
fn source_hash_is_hex_16_chars() {
    let h = source_hash("test", &CompileOptions::default());
    assert_eq!(h.len(), 16);
    assert!(h.chars().all(|c| c.is_ascii_hexdigit()));
}

// ---------------------------------------------------------------------------
// SpirVError
// ---------------------------------------------------------------------------

#[test]
fn spirv_error_display() {
    let e1 = SpirVError::ValidationFailed("bad magic".into());
    assert!(format!("{e1}").contains("bad magic"));

    let e2 = SpirVError::CompilationFailed("timeout".into());
    assert!(format!("{e2}").contains("timeout"));

    let e3 = SpirVError::NoCompilerAvailable;
    assert!(format!("{e3}").contains("no SPIR-V compiler"));
}

#[test]
fn spirv_error_debug() {
    let e = SpirVError::NoCompilerAvailable;
    let dbg = format!("{e:?}");
    assert!(dbg.contains("NoCompilerAvailable"));
}
