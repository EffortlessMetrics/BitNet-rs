# Inference Engine Type Visibility Specification

**Issue**: #447 (AC4-AC5)
**Status**: Ready for Implementation
**Priority**: P2 - Inference Full-Engine Tests
**Date**: 2025-10-11
**Affected Crate**: `bitnet-inference`

---

## Executive Summary

Export `ProductionInferenceEngine` and `ProductionInferenceConfig` types from `bitnet-inference` public API to enable full-engine feature tests. Provide minimal compile-only stubs for work-in-progress functionality to maintain clean compilation while preserving test structure for future implementation.

**Key Changes**:
- Export production engine types in `bitnet-inference/src/lib.rs`
- Add missing imports for test modules
- Implement compile-only test stubs with `#[ignore]` attribute
- Zero impact on existing inference functionality

---

## Acceptance Criteria

### AC4: Export production engine types for full-engine feature tests
**Test Tag**: `// AC4: Export engine types for tests`

**Requirements**:
- Export `ProductionInferenceEngine` in `bitnet-inference/src/lib.rs` public API
- Export `ProductionInferenceConfig` for test configuration
- Add missing imports to test modules:
  - `use std::env;` (for environment variable access)
  - `use anyhow::Context;` (for error context)

**Validation Command**:
```bash
cargo check -p bitnet-inference --all-features
```

**Expected Output**: All features compile with exported types accessible

**Evidence**: Current lib.rs exports `ProductionInferenceEngine` but tests may require additional configuration access

---

### AC5: Ensure full-engine feature compiles with stubs for WIP functionality
**Test Tag**: `// AC5: Full-engine compile stubs`

**Requirements**:
- Implement minimal compile-only stubs with `#[ignore = "WIP: full-engine in progress"]` attribute
- Preserve test structure for future implementation completion
- Ensure no runtime test failures (all WIP tests ignored)
- Maintain type safety and compilation validation

**Validation Command**:
```bash
cargo test -p bitnet-inference --no-default-features --features full-engine --no-run
```

**Expected Output**: Compilation succeeds, no tests executed (--no-run flag)

**Evidence**: Spec requires WIP functionality to compile without blocking development

---

## Technical Design

### Public API Exports

#### Current Exports (`crates/bitnet-inference/src/lib.rs`)

**Already Exported** (lines 42-45):
```rust
pub use production_engine::{
    GenerationResult, PerformanceMetricsCollector, ProductionInferenceEngine, ThroughputMetrics,
    TimingMetrics,
};
```

**Analysis**: `ProductionInferenceEngine` is already exported ✅

**Missing**: `ProductionInferenceConfig` (required for test configuration)

---

### Required API Additions

#### Export Configuration Types

**Add to** `crates/bitnet-inference/src/lib.rs`:

```rust
// Line 42 - Update existing export
pub use production_engine::{
    GenerationResult,
    PerformanceMetricsCollector,
    ProductionInferenceEngine,
    ProductionInferenceConfig,    // ← ADD THIS
    PrefillStrategy,              // ← ADD THIS
    ThroughputMetrics,
    TimingMetrics,
};
```

**Rationale**: Tests need access to configuration structures to create production engine instances with custom settings.

---

### Test Module Import Updates

#### Missing Imports in Test Modules

**Check**: `crates/bitnet-inference/tests/real_inference_engine.rs`

**Expected Pattern**:
```rust
use anyhow::{Context, Result};
use bitnet_inference::{
    ProductionInferenceEngine,
    ProductionInferenceConfig,
    PrefillStrategy,
};
use std::env;
use std::path::PathBuf;
```

**If Missing, Add**:
```rust
#[cfg(test)]
use anyhow::Context;  // For .context() error handling

#[cfg(test)]
use std::env;         // For env::var() and model path discovery
```

---

### Compile-Only Test Stubs

#### Pattern for WIP Tests

**Location**: `crates/bitnet-inference/tests/full_engine_tests.rs` (if exists)

```rust
//! Full-engine feature tests for production inference
//!
//! These tests validate the production inference engine with real models.
//! WIP tests are marked with #[ignore] until full implementation is complete.

use bitnet_inference::{ProductionInferenceEngine, ProductionInferenceConfig};
use std::sync::Arc;

#[test]
#[cfg(feature = "full-engine")]
#[ignore = "WIP: full-engine implementation in progress"] // AC:5
fn test_ac4_engine_config_visibility() {
    // Verify types are accessible from public API
    let config = ProductionInferenceConfig::default();
    assert!(config.enable_performance_monitoring);

    // This test just needs to compile, demonstrating type visibility
    let _phantom: Option<ProductionInferenceEngine> = None;
}

#[test]
#[cfg(feature = "full-engine")]
#[ignore = "WIP: full-engine implementation in progress"] // AC:5
fn test_ac5_minimal_compilation() {
    // Stub test for compilation validation
    // No runtime logic - just type checking
    use bitnet_inference::PrefillStrategy;

    let _strategy = PrefillStrategy::Adaptive { threshold_tokens: 10 };

    // Future implementation will:
    // 1. Load real model from BITNET_GGUF
    // 2. Create ProductionInferenceEngine
    // 3. Execute inference and validate results
}

#[test]
#[cfg(feature = "full-engine")]
#[ignore = "WIP: full-engine implementation in progress"] // AC:5
fn test_production_engine_initialization() {
    // Placeholder for full engine initialization test
    // Will be implemented when engine API is stabilized

    // Expected future implementation:
    // let model_path = env::var("BITNET_GGUF").context("BITNET_GGUF not set")?;
    // let model = load_model(&model_path)?;
    // let tokenizer = load_tokenizer(&model_path)?;
    // let engine = ProductionInferenceEngine::new(
    //     Arc::new(model),
    //     Arc::new(tokenizer),
    //     Device::Cpu,
    // )?;
    // assert!(engine.is_initialized());
}
```

**Key Attributes**:
- `#[cfg(feature = "full-engine")]` - Only compile with full-engine feature
- `#[ignore = "..."]` - Skip at runtime (compilation validation only)
- `// AC:N` tags - Traceability to acceptance criteria

---

### Feature Flag Configuration

#### Verify Feature Definition

**File**: `crates/bitnet-inference/Cargo.toml`

```toml
[features]
default = []
cpu = []
gpu = ["cuda"]
cuda = []
full-engine = ["inference"]  # ← Ensure this exists
inference = []               # ← Core inference feature
```

**Validation**:
```bash
cargo metadata --format-version=1 | jq '.packages[] | select(.name == "bitnet-inference") | .features'
```

**Expected Output**:
```json
{
  "default": [],
  "cpu": [],
  "gpu": ["cuda"],
  "cuda": [],
  "full-engine": ["inference"],
  "inference": []
}
```

---

## Migration Checklist

### Phase 1: API Exports (AC4)
- [ ] **AC4.1**: Add `ProductionInferenceConfig` to public exports in `lib.rs:42`
- [ ] **AC4.2**: Add `PrefillStrategy` to public exports in `lib.rs:42`
- [ ] **AC4.3**: Verify existing `ProductionInferenceEngine` export (already done)
- [ ] **AC4.4**: Add missing test module imports (`anyhow::Context`, `std::env`)

**Validation**:
```bash
cargo check -p bitnet-inference --all-features
# Expected: Compilation succeeds with all types visible
```

---

### Phase 2: Test Stubs (AC5)
- [ ] **AC5.1**: Create or update `tests/full_engine_tests.rs` with compile-only stubs
- [ ] **AC5.2**: Add `#[cfg(feature = "full-engine")]` guards to all test functions
- [ ] **AC5.3**: Add `#[ignore = "WIP: ..."]` attribute to incomplete tests
- [ ] **AC5.4**: Include `// AC:N` tags for traceability

**Validation**:
```bash
cargo test -p bitnet-inference --no-default-features --features full-engine --no-run
# Expected: Compilation succeeds, 0 tests executed (--no-run)
```

---

### Phase 3: Validation
- [ ] **AC4.5**: Test public API accessibility from external crate
- [ ] **AC5.5**: Verify ignored tests don't block CI
- [ ] **AC5.6**: Confirm compilation with all feature combinations

**Validation Matrix**:
```bash
# All features
cargo test -p bitnet-inference --all-features --no-run

# Full-engine only
cargo test -p bitnet-inference --no-default-features --features full-engine --no-run

# Inference only
cargo test -p bitnet-inference --no-default-features --features inference --no-run

# Expected: All succeed
```

---

## Validation Commands Matrix

| Phase | Command | Expected Output | AC |
|-------|---------|-----------------|-----|
| **Type Visibility** | `cargo check -p bitnet-inference --all-features` | Compilation succeeds | AC4 |
| **Stub Compilation** | `cargo test -p bitnet-inference --no-default-features --features full-engine --no-run` | Compilation succeeds, 0 tests run | AC5 |
| **Ignored Tests** | `cargo test -p bitnet-inference --no-default-features --features full-engine -- --ignored` | Lists ignored tests, none executed | AC5 |
| **Public API** | `cargo doc -p bitnet-inference --no-deps --open` | Documentation shows exported types | AC4 |

---

## Rollback Strategy

### Rollback Steps

1. **Revert API Exports**:
   ```bash
   git checkout crates/bitnet-inference/src/lib.rs
   ```

2. **Remove Test Stubs**:
   ```bash
   git checkout crates/bitnet-inference/tests/full_engine_tests.rs
   # Or remove file if newly created:
   rm crates/bitnet-inference/tests/full_engine_tests.rs
   ```

3. **Validate Rollback**:
   ```bash
   cargo check -p bitnet-inference --all-features
   cargo test -p bitnet-inference --no-default-features --features cpu
   ```

### Rollback Criteria
- API exports cause downstream compilation failures
- Test stubs introduce CI flakiness
- Feature flag combinations fail to compile

**Risk**: Low - changes are additive (exports) and non-functional (test stubs)

---

## API Contract Documentation

### New Public Exports

**Module**: `bitnet_inference`

```rust
/// Production inference engine configuration
pub struct ProductionInferenceConfig {
    pub enable_performance_monitoring: bool,
    pub enable_memory_tracking: bool,
    pub max_inference_time_seconds: u64,
    pub enable_quality_assessment: bool,
    pub prefill_strategy: PrefillStrategy,
}

/// Cache prefill strategy
pub enum PrefillStrategy {
    Always,
    Adaptive { threshold_tokens: usize },
    Never,
}
```

**Usage Example**:
```rust
use bitnet_inference::{
    ProductionInferenceEngine,
    ProductionInferenceConfig,
    PrefillStrategy,
};

let config = ProductionInferenceConfig {
    enable_performance_monitoring: true,
    prefill_strategy: PrefillStrategy::Adaptive { threshold_tokens: 20 },
    ..Default::default()
};

// Create engine with custom config (future implementation)
// let engine = ProductionInferenceEngine::with_config(model, tokenizer, device, config)?;
```

---

### Breaking Changes

**None** - All changes are additive exports. Existing API remains unchanged.

---

### Deprecation Warnings

**None** - No deprecated APIs in this change.

---

## Test Structure

### Unit Tests (Compile-Only)

```rust
// crates/bitnet-inference/tests/full_engine_tests.rs

#[test]
#[cfg(feature = "full-engine")]
fn test_ac4_config_types_exported() { // AC:4
    use bitnet_inference::{ProductionInferenceConfig, PrefillStrategy};

    let config = ProductionInferenceConfig::default();
    assert!(config.enable_performance_monitoring);

    matches!(
        config.prefill_strategy,
        PrefillStrategy::Adaptive { threshold_tokens: 10 }
    );
}

#[test]
#[cfg(feature = "full-engine")]
fn test_ac4_engine_type_exported() { // AC:4
    use bitnet_inference::ProductionInferenceEngine;

    // Type visibility check - compiles without runtime logic
    let _phantom: Option<ProductionInferenceEngine> = None;
}

#[test]
#[cfg(feature = "full-engine")]
#[ignore = "WIP: full-engine implementation in progress"] // AC:5
fn test_ac5_stub_compilation() {
    // Stub test for future implementation
    // TODO: Implement when engine API stabilizes
}
```

---

### Integration Tests (Future)

**Location**: `crates/bitnet-inference/tests/real_inference_engine.rs`

```rust
// Future implementation when full-engine is complete

#[test]
#[cfg(all(feature = "full-engine", not(feature = "ci")))]
fn test_production_engine_inference() {
    let model_path = env::var("BITNET_GGUF")
        .context("BITNET_GGUF not set")
        .unwrap();

    // Load model, create engine, run inference
    // (Implementation deferred until AC5 completed)
}
```

---

## BitNet.rs Standards Compliance

### Feature Flag Discipline
✅ All test functions use `#[cfg(feature = "full-engine")]`
✅ Commands specify features: `--no-default-features --features full-engine`
✅ Default features remain empty

### Workspace Structure Alignment
✅ Changes isolated to `bitnet-inference` crate
✅ Public API additions follow existing export patterns
✅ No breaking changes to downstream crates

### Neural Network Development Patterns
✅ Zero impact on existing inference algorithms
✅ Stub tests maintain type safety without runtime overhead
✅ Production engine config enables device-aware optimization

### TDD and Test Naming
✅ All tests tagged with `// AC:N` comments
✅ Test names follow `test_acN_*` convention
✅ WIP tests use `#[ignore]` with descriptive messages

### GGUF Compatibility
✅ No impact (type exports only, no model format changes)

---

## Performance Impact Analysis

### Compilation Time

**Expected Change**: +0.5-1 second for additional type exports
**Acceptable**: Yes (one-time compilation cost)

### Runtime Overhead

**Change**: None - stub tests are compilation-only
**Impact**: Zero (ignored tests never execute)

---

## Cross-Crate Impact Analysis

### Downstream Consumers

**Affected Crates**:
- `bitnet-server` (may use ProductionInferenceConfig)
- `bitnet-cli` (may use ProductionInferenceEngine)

**Impact**: Positive - additional types available for configuration
**Breaking**: No - existing imports remain valid

---

## References

### BitNet.rs Documentation
- `docs/development/test-suite.md` - Testing framework
- `docs/architecture-overview.md` - Inference engine design
- `docs/reference/real-model-api-contracts.md` - Production engine API

### Related Code
- `crates/bitnet-inference/src/production_engine.rs:266-322` - Engine implementation
- `crates/bitnet-inference/src/config.rs` - Inference configuration
- `crates/bitnet-inference/src/lib.rs:42-45` - Current exports

### Related Issues
- Issue #447 - Compilation fixes across workspace crates
- Issue #254 - Real neural network inference implementation
- PR #431 - Production engine introduction

---

## Implementation Notes

### Minimal Changes Philosophy

This specification follows BitNet.rs principle of **minimal, incremental changes**:

1. **Export Only What's Needed**: Only add `ProductionInferenceConfig` and `PrefillStrategy`
2. **Preserve Existing API**: No modifications to current exports
3. **Stub-Based Validation**: Use ignored tests for compilation validation only
4. **Future-Ready Structure**: Test scaffolding prepared for full implementation

---

### Future Implementation Path

**When full-engine is ready**:

1. Remove `#[ignore]` attributes from stub tests
2. Implement actual test logic with model loading
3. Add integration tests with BITNET_GGUF environment variable
4. Validate against C++ reference using crossval framework

**Estimated Timeline**: Post-Issue #447 (separate feature work)

---

## Approval Checklist

Before implementation:
- [x] Both acceptance criteria (AC4-AC5) clearly defined
- [x] Validation commands specified with expected outputs
- [x] API exports documented with usage examples
- [x] Test stub pattern defined with proper guards
- [x] Zero impact on existing inference confirmed
- [x] Feature flag discipline maintained
- [x] Rollback strategy documented
- [x] Cross-crate impact analyzed

**Status**: ✅ Ready for Implementation

**Next Steps**: NEXT → impl-creator (implementation of AC4-AC5)
