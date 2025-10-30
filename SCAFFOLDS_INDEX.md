# BitNet.rs TDD Scaffolding Index

## Quick Reference

**Report Location**: `/home/steven/code/Rust/BitNet-rs/TDD_SCAFFOLDS_COMPREHENSIVE_REPORT.md` (564 lines)

### Statistics at a Glance
- **98 ignored tests** across 20+ test files
- **~140+ individual test functions** scaffolded
- **4 active blocker issues**: #254, #260, #439, #469
- **Organized by**: Issue number and acceptance criteria

---

## Blocker Issues (4 Active)

### Issue #254: Layer-Norm Shape Mismatch in Real Inference
- **Files**: 3 test files
- **Tests**: 40+ scaffolds
- **Status**: In analysis phase
- **Key Blockers**:
  - Receipt generation (AC4) - 5 tests
  - Real inference validation - ~15 tests
  - Quantized linear layers (AC1) - 5 tests
  - Multi-head attention (AC2) - 5 tests
  - Autoregressive generation (AC3) - ~5 tests
  - Cross-validation accuracy (AC4) - 4 comprehensive tests
  - Performance targets (AC5) - 4 tests
  - Error handling (AC10) - 4 tests

### Issue #260: Mock Elimination & Strict Mode
- **Files**: 2-3 test files
- **Tests**: 15+ scaffolds
- **Status**: Awaiting refactoring
- **Key Blockers**:
  - Strict mode enforcement
  - Feature gate unification
  - Mock computation detection

### Issue #159: GGUF Weight Loading
- **Files**: 6 test files
- **Tests**: ~70+ scaffolds
- **Status**: TDD infrastructure
- **Key Blockers**:
  - Real weight loading (vs mock)
  - Quantization integration
  - Device-aware routing

### Issue #469: Tokenizer Parity & FFI
- **Files**: 1 test file
- **Tests**: 4+ scaffolds
- **Status**: Active development
- **Key Blockers**:
  - Auto-discovery from GGUF
  - JSON-BPE support
  - SentencePiece FFI

---

## Test File Organization

### Inference Tests (Critical Path)
```
crates/bitnet-inference/tests/
├── issue_254_ac4_receipt_generation.rs          # Receipt generation (5 tests)
├── ac1_quantized_linear_layers.rs               # Quantized layers (5 tests)
├── ac2_multi_head_attention.rs                  # Attention mechanism (5 tests)
├── ac3_autoregressive_generation.rs             # Token generation (~5 tests)
├── ac4_cross_validation_accuracy.rs             # Cross-validation (4 tests)
├── ac5_performance_targets.rs                   # Performance targets (4 tests)
├── ac10_error_handling_robustness.rs            # Error handling (4 tests)
├── issue_260_mock_elimination_inference_tests.rs # Mock detection (~10 tests)
├── test_real_inference.rs                       # Real inference tests (~15 tests)
└── test_real_vs_mock_comparison.rs              # Comparison tests
```

### Model Loading Tests (Infrastructure)
```
crates/bitnet-models/tests/
├── gguf_weight_loading_tests.rs                 # Real loading (~25 tests)
├── gguf_weight_loading_property_tests.rs        # Properties (~15 tests)
├── gguf_weight_loading_property_tests_enhanced.rs # Enhanced (~7 tests)
├── gguf_weight_loading_device_aware_tests.rs    # Device handling (~5 tests)
├── gguf_weight_loading_feature_matrix_tests.rs  # Features (~5 tests)
└── gguf_weight_loading_integration_tests.rs     # Integration (~5 tests)
```

### Kernel & Device Tests
```
crates/bitnet-kernels/tests/
├── issue_260_feature_gated_tests.rs             # Feature gates (3 tests)
├── gpu_real_compute.rs                          # GPU computation
├── cpu_simd_receipts.rs                         # SIMD validation
└── [various GPU tests]
```

### Tokenizer Tests
```
crates/bitnet-tokenizers/tests/
└── issue_254_ac8_tokenizer_auto_discovery.rs    # AC8 tests (4 tests)
```

### Server Tests
```
crates/bitnet-server/tests/
└── ac04_batch_processing.rs                     # Batch processing (~5 tests)
```

---

## Implementation Priority

### Phase 1: Critical Path (Resolves Issue #254)
1. Diagnose layer-norm shape mismatch
2. Implement AC1: Quantized linear layers (I2S, TL1, TL2)
3. Implement receipt generation (AC4)
4. Establish performance baselines

### Phase 2: Feature Completion (Resolves Issue #260)
5. Implement AC2: Multi-head attention
6. Implement AC3: Autoregressive generation
7. Implement AC4: Cross-validation accuracy
8. Eliminate mocks and enforce strict mode

### Phase 3: Quality & Scale (Resolves Issues #159, #469)
9. Real GGUF weight loading
10. Tokenizer auto-discovery and parity
11. Feature gate consistency validation
12. Server API batch processing

### Phase 4: Production Readiness
13. Performance target validation (AC5)
14. Device-aware quantization selection (AC6)
15. Comprehensive error handling (AC10)
16. Full documentation

---

## Key Infrastructure Gaps

### Quantization
- `I2SQuantizer::validate_accuracy()`
- `TL1Quantizer` with lookup metrics
- `TL2Quantizer` with 4096-entry tables
- IQ2_S GGML compatibility layer

### Tensors
- `BitNetTensor::validate_stability()` (NaN/Inf detection)
- `BitNetTensor::compare_consistency()` (cross-device parity)
- Slicing and manipulation APIs

### Performance
- Memory tracking (OS APIs)
- Latency measurement (with warm-up)
- Throughput calculation
- Regression detection

### Cross-Validation
- C++ FFI integration
- GGML reference execution
- Metric comparison (cosine similarity, exact match)
- Report parsing

### Tokenizers
- GGUF metadata parsing
- JSON-BPE implementation/FFI
- SentencePiece FFI wrapper
- Unicode round-trip handling

---

## Quick Lookup by Acceptance Criteria

| AC | Crate | File | Tests | Issue |
|----|-------|------|-------|-------|
| AC1 | inference | ac1_quantized_linear_layers.rs | 5 | #254 |
| AC2 | inference | ac2_multi_head_attention.rs | 5 | #254 |
| AC3 | inference | ac3_autoregressive_generation.rs | ~5 | #254 |
| AC4 | inference | ac4_cross_validation_accuracy.rs | 4 | #254 |
| AC4 | inference | issue_254_ac4_receipt_generation.rs | 5 | #254 |
| AC5 | inference | ac5_performance_targets.rs | 4 | #254 |
| AC8 | tokenizers | issue_254_ac8_tokenizer_auto_discovery.rs | 4 | #469 |
| AC10 | inference | ac10_error_handling_robustness.rs | 4 | #254 |

---

## Helper Modules

### Fixture Infrastructure
- Test data generators
- Mock model creation
- Mock quantizer setup
- Test configuration builders

### Support Modules
```
crates/bitnet-kernels/tests/support/
├── mod.rs           # Support infrastructure
├── env_guard.rs     # Environment variable management
└── receipt.rs       # Receipt validation
```

---

## How to Enable a Scaffold

When a blocker is resolved, enable tests by:

1. **Remove `#[ignore]`** attribute
2. **Implement stub functions** in the test module
3. **Run `cargo test`** to verify compilation
4. **Incrementally implement** test requirements
5. **Update CLAUDE.md** with new status

Example:
```rust
// Before: #[ignore] - blocked by Issue #254
// After:
#[tokio::test]
async fn test_ac4_receipt_generation_real_path() -> Result<()> {
    // Implementation here
}
```

---

## Current Test Status

- **Enabled**: ~500+ tests (passing)
- **Ignored**: ~98 tests (scaffolding)
- **Broken**: 0 (intentionally ignored, not broken)

All CI tests pass. Ignored tests represent intentional development placeholders.

---

## Related Documents

- `/home/steven/code/Rust/BitNet-rs/TDD_SCAFFOLDS_COMPREHENSIVE_REPORT.md` - Full details
- `/home/steven/code/Rust/BitNet-rs/CLAUDE.md` - Project status and roadmap
- `docs/development/test-suite.md` - Test infrastructure guide

