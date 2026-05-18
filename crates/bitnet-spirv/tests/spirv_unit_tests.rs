//! Unit tests for SPIR-V validation, hashing, compiler façade, and cache behavior.

use bitnet_spirv::{
    CompileOptions, CompilerBackend, OptimizationLevel, SPIRV_MAGIC, SpirVCache, SpirVCompiler,
    SpirVError, SpirVModule, SpirVValidator, build_test_spirv, source_hash,
};
use std::error::Error;

fn words_to_bytes(words: &[u32]) -> Vec<u8> {
    words.iter().flat_map(|word| word.to_le_bytes()).collect()
}

fn valid_header_words() -> [u32; 5] {
    [SPIRV_MAGIC, 0x0001_0000, 0, 1, 0]
}

#[test]
fn default_compile_options_use_full_optimization_without_target_or_defines() {
    let options = CompileOptions::default();

    assert_eq!(options.target_device, None);
    assert_eq!(options.optimization_level, OptimizationLevel::Full);
    assert!(options.defines.is_empty());
}

#[test]
fn compiler_with_no_backend_reports_no_compiler_available() {
    let compiler = SpirVCompiler::with_backend(None);

    let err = compiler
        .compile_to_spirv("__kernel void noop() {}", &CompileOptions::default())
        .expect_err("compilation should fail before invoking any external compiler");

    assert!(matches!(err, SpirVError::NoCompilerAvailable));
    assert_eq!(compiler.backend(), None);
}

#[test]
fn compiler_with_explicit_backend_exposes_backend_for_testability() {
    let compiler = SpirVCompiler::with_backend(Some(CompilerBackend::Clang));

    assert_eq!(compiler.backend(), Some(CompilerBackend::Clang));
}

#[test]
fn source_hash_is_deterministic_for_identical_source_and_options() {
    let options = CompileOptions {
        target_device: Some("pvc".to_string()),
        optimization_level: OptimizationLevel::Basic,
        defines: vec![("TILE".to_string(), "16".to_string())],
    };

    assert_eq!(source_hash("kernel", &options), source_hash("kernel", &options));
}

#[test]
fn source_hash_changes_when_compile_inputs_change() {
    let base = CompileOptions::default();
    let mut with_define = base.clone();
    with_define.defines.push(("WIDTH".to_string(), "32".to_string()));

    let mut with_target = base.clone();
    with_target.target_device = Some("dg2".to_string());

    let mut with_optimization = base.clone();
    with_optimization.optimization_level = OptimizationLevel::None;

    let baseline = source_hash("kernel", &base);
    assert_ne!(baseline, source_hash("kernel2", &base));
    assert_ne!(baseline, source_hash("kernel", &with_define));
    assert_ne!(baseline, source_hash("kernel", &with_target));
    assert_ne!(baseline, source_hash("kernel", &with_optimization));
}

#[test]
fn build_test_spirv_creates_minimal_valid_header_for_supported_versions()
-> Result<(), Box<dyn Error>> {
    let bytes = build_test_spirv(1, 6);

    assert_eq!(bytes.len(), 20);
    SpirVValidator::validate_bytes(&bytes)?;
    Ok(())
}

#[test]
fn validator_rejects_short_bad_magic_and_unsupported_version_inputs() {
    let short = vec![0; 19];
    assert!(matches!(
        SpirVValidator::validate_bytes(&short),
        Err(SpirVError::ValidationFailed(message)) if message.contains("too short")
    ));

    let bad_magic = words_to_bytes(&[0xDEAD_BEEF, 0x0001_0000, 0, 1, 0]);
    assert!(matches!(
        SpirVValidator::validate_bytes(&bad_magic),
        Err(SpirVError::ValidationFailed(message)) if message.contains("bad magic")
    ));

    let unsupported = build_test_spirv(1, 7);
    assert!(matches!(
        SpirVValidator::validate_bytes(&unsupported),
        Err(SpirVError::ValidationFailed(message)) if message.contains("unsupported SPIR-V version 1.7")
    ));
}

#[test]
fn has_capability_detects_matching_op_capability_operand() {
    let mut words = valid_header_words().to_vec();
    words.push((2 << 16) | 0x0011); // OpCapability, word count 2.
    words.push(1); // Shader capability.
    words.push((1 << 16) | 0x000e); // A non-capability instruction to prove iteration advances.
    let bytes = words_to_bytes(&words);

    assert!(SpirVValidator::has_capability(&bytes, 1));
    assert!(!SpirVValidator::has_capability(&bytes, 2));
}

#[test]
fn has_capability_returns_false_for_truncated_or_malformed_instruction_streams() {
    assert!(!SpirVValidator::has_capability(&build_test_spirv(1, 0), 1));

    let mut words = valid_header_words().to_vec();
    words.push(0); // Malformed zero word-count instruction should terminate scanning.
    words.push(1);
    assert!(!SpirVValidator::has_capability(&words_to_bytes(&words), 1));
}

#[test]
fn cache_starts_empty_round_trips_modules_and_can_be_cleared() -> Result<(), Box<dyn Error>> {
    let cache = SpirVCache::new();
    assert!(cache.is_empty());
    assert_eq!(cache.len(), 0);
    assert!(cache.get("missing").is_none());

    let module = SpirVModule {
        bytecode: build_test_spirv(1, 0),
        source_hash: "abc123".to_string(),
        compiler: Some(CompilerBackend::Ocloc),
    };
    cache.insert(module.clone());

    assert!(!cache.is_empty());
    assert_eq!(cache.len(), 1);
    let cached = cache.get("abc123").ok_or("cached module missing")?;
    assert_eq!(cached.bytecode, module.bytecode);
    assert_eq!(cached.source_hash, module.source_hash);
    assert_eq!(cached.compiler, module.compiler);

    cache.clear();
    assert!(cache.is_empty());
    assert!(cache.get("abc123").is_none());
    Ok(())
}

#[test]
fn cache_replaces_existing_module_with_same_source_hash() -> Result<(), Box<dyn Error>> {
    let cache = SpirVCache::default();
    let first = SpirVModule {
        bytecode: build_test_spirv(1, 0),
        source_hash: "same".to_string(),
        compiler: Some(CompilerBackend::Clang),
    };
    let second = SpirVModule {
        bytecode: build_test_spirv(1, 1),
        source_hash: "same".to_string(),
        compiler: Some(CompilerBackend::Ocloc),
    };

    cache.insert(first);
    cache.insert(second.clone());

    assert_eq!(cache.len(), 1);
    let cached = cache.get("same").ok_or("replacement module missing")?;
    assert_eq!(cached.bytecode, second.bytecode);
    assert_eq!(cached.source_hash, second.source_hash);
    assert_eq!(cached.compiler, second.compiler);
    Ok(())
}
