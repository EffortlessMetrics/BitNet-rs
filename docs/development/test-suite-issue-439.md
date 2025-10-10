# Test Suite for Issue #439: GPU Feature-Gate Hardening

## Overview

This document maps the test scaffolding created for Issue #439 GPU feature-gate hardening to the acceptance criteria and provides guidance for running tests across different feature configurations.

**Specification**: `docs/explanation/issue-439-spec.md`

## Test Coverage Matrix

| AC  | Test File | Test Count | Features Required | Status |
|-----|-----------|------------|-------------------|--------|
| AC1 | `crates/bitnet-kernels/tests/feature_gate_consistency.rs` | 5 | any | Scaffolded |
| AC2 | `crates/bitnet-kernels/tests/build_script_validation.rs` | 5 | any | Scaffolded |
| AC3 | `crates/bitnet-kernels/tests/device_features.rs` | 10 | cpu, gpu | Scaffolded |
| AC5 | `xtask/tests/preflight.rs` | 6 | any | Scaffolded |
| AC6 | `xtask/tests/verify_receipt.rs` | 12 | any | Scaffolded |
| AC7 | `xtask/tests/documentation_audit.rs` | 8 | any | Scaffolded |
| AC8 | `tests/gitignore_validation.rs` | 7 | any | Scaffolded |

**Total**: 53 tests across 7 test files

## Test File Descriptions

### AC1: Feature Gate Consistency Tests

**File**: `crates/bitnet-kernels/tests/feature_gate_consistency.rs`

**Purpose**: Validates unified GPU predicate usage across workspace crates

**Tests**:
- `ac1_no_standalone_cuda_gates_in_kernels` - Searches for standalone `#[cfg(feature="cuda")]` without unified predicate
- `ac1_gpu_validation_module_uses_unified_predicate` - Verifies gpu/validation.rs uses unified gates
- `ac1_workspace_wide_cuda_gate_consistency` - Comprehensive workspace search for gate consistency
- `ac1_build_scripts_check_both_gpu_features` - Validates build.rs uses unified detection
- `ac1_cfg_macro_uses_unified_predicate` - Checks runtime `cfg!()` macro usage

**Specification Reference**: `docs/explanation/issue-439-spec.md#implementation-approach-1`

### AC2: Build Script Validation Tests

**File**: `crates/bitnet-kernels/tests/build_script_validation.rs`

**Purpose**: Ensures build scripts check both CARGO_FEATURE_GPU and CARGO_FEATURE_CUDA

**Tests**:
- `ac2_build_script_checks_both_features` - Validates unified GPU detection pattern
- `ac2_build_script_emits_gpu_cfg` - Verifies bitnet_build_gpu cfg emission
- `ac2_workspace_build_scripts_consistency` - Checks workspace-wide build script consistency
- `ac2_build_script_handles_no_features` - Tests graceful handling of missing features
- `ac2_build_script_single_unified_cfg` - Validates minimal cfg flag emission

**Specification Reference**: `docs/explanation/issue-439-spec.md#build-script-parity`

### AC3: Device Features Module Tests

**File**: `crates/bitnet-kernels/tests/device_features.rs`

**Purpose**: Validates device_features module provides unified GPU capability detection

**Tests**:
- `ac3_gpu_compiled_true_with_features` - Tests compile-time detection with GPU features
- `ac3_gpu_compiled_false_without_features` - Tests compile-time detection without GPU features
- `ac3_gpu_fake_cuda_overrides_detection` - Validates BITNET_GPU_FAKE=cuda precedence
- `ac3_gpu_fake_none_disables_detection` - Validates BITNET_GPU_FAKE=none disables GPU
- `ac3_gpu_runtime_false_without_compile` - Tests runtime stub when GPU not compiled
- `ac3_gpu_fake_case_insensitive` - Validates case-insensitive fake GPU matching
- `ac3_device_capability_summary_format` - Tests diagnostic summary output
- `ac3_capability_summary_respects_fake` - Validates summary reflects fake GPU state
- `ac3_quantization_uses_device_features` - Integration test with bitnet-quantization
- `ac3_inference_uses_device_features` - Integration test with bitnet-inference

**Specification Reference**: `docs/explanation/issue-439-spec.md#device-feature-detection-api`

**Implementation Status**: Device features module stub created at `crates/bitnet-kernels/src/device_features.rs` with `unimplemented!()` placeholders (TDD Red phase)

### AC5: xtask Preflight Validation Tests

**File**: `xtask/tests/preflight.rs`

**Purpose**: Validates xtask preflight command reports GPU status correctly

**Tests**:
- `ac5_preflight_detects_no_gpu_with_fake_none` - Tests BITNET_GPU_FAKE=none reporting
- `ac5_preflight_detects_gpu_with_fake_cuda` - Tests BITNET_GPU_FAKE=cuda reporting
- `ac5_preflight_real_gpu_detection` - Tests real hardware detection
- `ac5_preflight_invalid_fake_value_fallback` - Tests graceful handling of invalid values
- `ac5_preflight_reports_compile_status` - Validates compile vs runtime distinction
- `ac5_preflight_exit_code_success` - Tests successful exit regardless of GPU status

**Specification Reference**: `docs/explanation/issue-439-spec.md#xtask-preflight`

### AC6: Receipt Validation Tests

**File**: `xtask/tests/verify_receipt.rs`

**Purpose**: Validates GPU backend receipts contain evidence of GPU kernel execution

**Tests**:
- `ac6_gpu_backend_requires_gpu_kernel` - GPU backend with CPU kernels fails
- `ac6_gpu_backend_with_valid_kernel_passes` - GPU backend with GPU kernel passes
- `ac6_cpu_backend_no_validation_required` - CPU backend requires no validation
- `ac6_gpu_backend_empty_kernels_fails` - Empty kernels array fails for GPU backend
- `ac6_all_gpu_kernel_prefixes_recognized` - All GPU kernel naming conventions recognized
- `ac6_cpu_kernel_prefixes_rejected` - CPU kernel naming patterns rejected
- `ac6_fixture_valid_gpu_receipt` - Valid GPU receipt fixture passes
- `ac6_fixture_invalid_gpu_receipt` - Invalid GPU receipt fixture fails
- `ac6_fixture_valid_cpu_receipt` - Valid CPU receipt fixture passes
- `ac6_fixture_all_kernel_types` - GPU receipt with all kernel types passes
- `ac6_detect_suspicious_gpu_performance` - Performance-based fallback detection
- `ac6_gpu_performance_baselines` - GPU performance baseline validation

**Specification Reference**: `docs/explanation/issue-439-spec.md#receipt-validation-architecture`

**Fixtures**: Located in `tests/fixtures/receipts/`
- `valid-gpu-receipt.json` - GPU backend with gemm_fp16, i2s_gpu_quantize
- `invalid-gpu-receipt.json` - GPU backend with CPU kernels (should fail)
- `valid-cpu-receipt.json` - CPU backend with CPU kernels
- `gpu-receipt-all-kernel-types.json` - All GPU kernel categories

### AC7: Documentation Audit Tests

**File**: `xtask/tests/documentation_audit.rs`

**Purpose**: Validates documentation uses standardized feature flag examples

**Tests**:
- `ac7_docs_use_no_default_features_pattern` - Documentation uses --no-default-features
- `ac7_no_standalone_cuda_examples` - No standalone --features cuda examples
- `ac7_claude_md_standardized_examples` - CLAUDE.md uses standardized flags
- `ac7_gpu_dev_guide_unified_flags` - GPU development guide mentions unified predicate
- `ac7_build_commands_standardized` - Build commands documentation standardized
- `ac7_readme_standardized_examples` - README uses standardized patterns
- `ac7_features_documentation_accurate` - FEATURES.md explains cuda/gpu relationship
- `ac7_cargo_toml_documents_empty_defaults` - Cargo.toml documents default = []
- `ac7_consistent_feature_terminology` - Cross-reference consistency check

**Specification Reference**: `docs/explanation/issue-439-spec.md#documentation-updates`

### AC8: Gitignore Validation Tests

**File**: `tests/gitignore_validation.rs`

**Purpose**: Validates ephemeral test artifacts are excluded from version control

**Tests**:
- `ac8_proptest_regressions_ignored` - .gitignore contains proptest pattern
- `ac8_cache_incremental_ignored` - .gitignore contains last_run.json pattern
- `ac8_common_test_artifacts_ignored` - Common test artifacts check
- `ac8_model_files_handling` - Model files appropriately handled
- `ac8_test_output_directories_ignored` - Test output directories excluded
- `ac8_no_committed_proptest_regressions` - No committed regression files
- `ac8_no_committed_last_run_json` - No committed cache files
- `ac8_gitignore_well_documented` - .gitignore has comments
- `ac8_gitignore_glob_patterns_correct` - Proper glob syntax used

**Specification Reference**: `docs/explanation/issue-439-spec.md#ac8

## Running Tests

### Full Test Suite Compilation Validation

```bash
# CPU features only (no execution)
cargo test --workspace --no-default-features --features cpu --no-run

# GPU features only (no execution)
cargo test --workspace --no-default-features --features gpu --no-run

# Both CPU and GPU features (no execution)
cargo test --workspace --no-default-features --features "cpu gpu" --no-run
```

**Expected Result**: All tests should compile successfully but fail when executed due to `unimplemented!()` placeholders (TDD Red phase)

### Running Specific AC Test Suites

```bash
# AC1: Feature gate consistency
cargo test --package bitnet-kernels --test feature_gate_consistency

# AC2: Build script validation
cargo test --package bitnet-kernels --test build_script_validation

# AC3: Device features module (CPU)
cargo test --package bitnet-kernels --test device_features --no-default-features --features cpu

# AC3: Device features module (GPU)
cargo test --package bitnet-kernels --test device_features --no-default-features --features gpu

# AC5: Preflight validation
cargo test --package xtask --test preflight

# AC6: Receipt validation
cargo test --package xtask --test verify_receipt

# AC7: Documentation audit
cargo test --package xtask --test documentation_audit

# AC8: Gitignore validation
cargo test --test gitignore_validation
```

### Feature Matrix Build Validation (AC4)

```bash
# No features (should build successfully)
cargo check --workspace --no-default-features

# CPU only
cargo check --workspace --no-default-features --features cpu

# GPU only
cargo check --workspace --no-default-features --features gpu

# CPU + GPU combined
cargo check --workspace --no-default-features --features "cpu gpu"
```

**Expected Result**: All feature combinations should compile successfully (AC4 validation)

## TDD Implementation Workflow

This test scaffolding follows Test-Driven Development (TDD) Red-Green-Refactor pattern:

### Red Phase (Current State)

All tests compile successfully but fail due to missing implementation:
- Device features module has `unimplemented!()` stubs
- Build scripts may not have unified detection yet
- Preflight command may not respect BITNET_GPU_FAKE yet
- Receipt validation logic not implemented yet

### Green Phase (Implementation)

Implement each AC to make tests pass:

1. **AC1**: Replace standalone `#[cfg(feature="cuda")]` with `#[cfg(any(feature="gpu", feature="cuda"))]`
2. **AC2**: Update build.rs to check both CARGO_FEATURE_GPU and CARGO_FEATURE_CUDA
3. **AC3**: Implement device_features module with gpu_compiled(), gpu_available_runtime(), device_capability_summary()
4. **AC5**: Update xtask preflight to respect BITNET_GPU_FAKE environment variable
5. **AC6**: Implement receipt validation with GPU kernel naming convention checks
6. **AC7**: Update documentation to use standardized feature flag examples
7. **AC8**: Update .gitignore with ephemeral artifact patterns

### Refactor Phase

After tests pass, refactor for:
- Code organization and clarity
- Performance optimization
- Documentation improvements
- Consistent error messages

## Test Execution Environment

### Environment Variables

- `BITNET_GPU_FAKE=cuda` - Override GPU detection to report GPU present (for testing without hardware)
- `BITNET_GPU_FAKE=none` - Override GPU detection to report no GPU (for testing CPU fallback)
- `BITNET_DETERMINISTIC=1` - Enable deterministic test mode
- `BITNET_SEED=42` - Set deterministic random seed

### Dependencies

Tests require:
- `ripgrep` (rg command) for grep-based validation tests
- Git repository context for gitignore tests
- Workspace structure intact for cross-crate integration tests

## Traceability

Each test includes doc comment references to specification sections:

```rust
/// Tests specification: docs/explanation/issue-439-spec.md#ac1-kernel-gate-unification
#[test]
fn ac1_no_standalone_cuda_gates_in_kernels() { ... }
```

This provides bidirectional traceability:
- **Spec → Tests**: Each AC maps to specific test functions
- **Tests → Spec**: Each test references its specification section

## Success Criteria

Test scaffolding is complete when:

1. All 53 tests compile successfully with CPU and GPU features
2. Tests fail predictably due to `unimplemented!()` (Red phase)
3. Each AC has comprehensive test coverage
4. Test fixtures are in place for AC6 receipt validation
5. Documentation provides clear guidance for running tests
6. Traceability links are established between tests and specs

## Next Steps

After test scaffolding approval:

1. **FINALIZE → fixture-builder**: Create additional test fixtures if needed for comprehensive validation
2. **FINALIZE → tests-finalizer**: Validate test scaffolding completeness and prepare for implementation phase
3. **Implementation**: Begin Green phase by implementing each AC to make tests pass

---

**Issue**: #439
**Specification**: `docs/explanation/issue-439-spec.md`
**Created**: 2025-10-10
**Test Scaffolding Status**: Complete - Ready for implementation (TDD Red phase)
