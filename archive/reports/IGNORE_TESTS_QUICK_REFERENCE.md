# #[ignore] Test Annotation Quick Reference

## Summary
- **Total**: 194 #[ignore] annotations across codebase
- **Annotated** (with reason): 59 (30.4%)
- **Bare** (no reason): 135 (69.6%)

## Priority: Top 5 Files to Annotate

1. **crates/bitnet-inference/tests/issue_254_ac3_deterministic_generation.rs** (10)
   - Category: Slow performance tests (~50-token generations)
   - Suggested reason: `slow: 50-token generation, see fast unit tests`

2. **crates/bitnet-models/tests/gguf_weight_loading_property_tests.rs** (9)
   - Category: Issue #159 TDD placeholders
   - Suggested reason: `Issue #159: TDD placeholder - I2S quantization integration needed`

3. **crates/bitnet-tokenizers/tests/test_ac4_smart_download_integration.rs** (9)
   - Category: Network-dependent tests
   - Suggested reason: `network: requires HuggingFace API access`

4. **crates/bitnet-inference/tests/neural_network_test_scaffolding.rs** (8)
   - Category: Issue #248 TDD scaffolding
   - Suggested reason: `Issue #248: TDD placeholder - <feature> unimplemented`

5. **crates/bitnet-models/tests/gguf_weight_loading_property_tests_enhanced.rs** (7)
   - Category: Issue #159 property-based tests
   - Suggested reason: `Issue #159: TDD placeholder - <quantization> test needed`

## Reason Template Guide

### For Issue-Blocked Tests (46 tests)
```rust
#[ignore = "Issue #254: shape mismatch in layer-norm - needs investigation"]
#[ignore = "Issue #260: TDD placeholder - quantized_matmul not yet implemented"]
#[ignore = "Issue #159: TDD placeholder - weight loading implementation needed"]
```

### For Slow/Performance Tests (17 tests)
```rust
#[ignore = "slow: 50+ token generations, use fast unit tests in ci mode"]
#[ignore = "slow: integration test, see faster equivalents in unit_tests.rs"]
#[ignore = "performance: timing-sensitive, causes non-deterministic CI failures"]
#[ignore = "benchmark: not a functional unit test"]
```

### For Model/File Requirements (29 tests)
```rust
#[ignore = "requires: real GGUF model with complete metadata"]
#[ignore = "requires: bitnet.cpp FFI implementation"]
#[ignore = "fixture: missing test data at tests/fixtures/tokenizer.gguf"]
```

### For Network/External Dependencies (10 tests)
```rust
#[ignore = "network: requires HuggingFace API access"]
#[ignore = "network: requires internet connection and HF_TOKEN"]
#[ignore = "fixture: missing file, run: cargo xtask download-model"]
```

### For GPU/Hardware-Specific (13 tests)
```rust
#[ignore = "gpu: requires CUDA toolkit installed"]
#[ignore = "gpu: run with --ignored flag when CUDA available"]
```

### For Feature Implementation/TODO (14 tests)
```rust
#[ignore = "TODO: update to use QuantizedLinear::new_tl1() with proper LookupTable"]
#[ignore = "TODO: implement GPU mixed-precision tests after #439 resolution"]
#[ignore = "TODO: replace mock inference with real path"]
```

## Categorization of Bare Ignores

| Category | Count | Rationale |
|----------|-------|-----------|
| Issue-blocked | 46 | Reference specific GitHub issues in reason |
| Slow performance | 17 | Note "slow:" prefix and reference fast alternatives |
| Requires model/GGUF | 29 | Use "requires:" prefix for clarity |
| GPU/CUDA | 13 | Note "gpu:" and conditional execution |
| Feature-gate/TODO | 14 | Keep "TODO:" or "Issue #" prefix |
| Network/async | 10 | Use "network:" or "fixture:" prefix |
| Quantization/kernels | 22 | Reference specific Issue #159 for weight loading |
| Parity/accuracy | 7 | Reference cross-validation or accuracy standards |
| Deterministic | 4 | Note seeding and reproducibility requirements |
| Strict mode | 2 | Reference validation configuration |
| Receipt | 1 | Reference receipt verification schema |
| Cross-validation | 1 | Reference C++ reference comparison |

## Existing Patterns (Reuse These)

Most common annotated reasons already in codebase:

1. "requires model file - run manually or in CI with BITNET_GGUF set" (19x)
2. "WIP: full-engine implementation in progress" (9x)
3. "TDD Red phase - AC5 accuracy thresholds not yet met (Issue #254)" (7x)
4. "requires model file and CLI binary - run manually or in CI" (4x)
5. "Fixture needed: log capture mechanism for tracing output" (3x)

## Quick Annotation Rules

1. **If comment references Issue #XXX**: Use `#[ignore = "Issue #XXX: brief description"]`
2. **If test is slow**: Use `#[ignore = "slow: description, see fast unit test"]`
3. **If requires external resource**: Use `#[ignore = "requires: description"]`
4. **If GPU/hardware dependent**: Use `#[ignore = "gpu: description"]`
5. **If network dependent**: Use `#[ignore = "network: description"]`
6. **If fixture missing**: Use `#[ignore = "fixture: filename or location"]`
7. **Always include actionable info**: What's blocking it or how to make it run?

## Testing Your Changes

```bash
# Check for any remaining bare ignores
cargo test --help  # Lists all ignored tests
grep -r '#\[ignore\]' --include='*.rs' crates/ tests/ | grep -v '=' | head -20

# Verify annotated reasons are clear
grep -r '#\[ignore = ' --include='*.rs' crates/ tests/ | sort | uniq -c | sort -rn
```
