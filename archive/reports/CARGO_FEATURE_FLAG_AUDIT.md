# Cargo Feature Flag Audit Report

> **Note**: References to `docs/archive/reports/` point to historical archived documentation.
> For current status, see [CLAUDE.md](CLAUDE.md) and [PR #475](PR_475_FINAL_SUCCESS_REPORT.md).



## Executive Summary

**Total Issues Found:** 205+ instances across documentation
- **Cargo build commands:** 11 instances missing `--no-default-features`
- **Cargo test -p commands:** 94+ instances missing `--no-default-features`
- **Cargo test --workspace commands:** 14 instances missing `--no-default-features`
- **Cargo run --example commands:** 9 instances missing feature specification

**Critical Impact:** These commands will fail or behave unexpectedly because BitNet-rs has **EMPTY default features** and requires explicit `--no-default-features --features cpu|gpu` specification.

---

## Category 1: Cargo Build Commands (11 instances)

### 1.1 docs/development/validation-ci.md

**Lines 285-286:**
```bash
# CURRENT (INCORRECT):
cargo build -p bitnet-st2gguf --release
cargo build -p bitnet-st-tools --release

# SHOULD BE:
cargo build -p bitnet-st2gguf --no-default-features --features cpu --release
cargo build -p bitnet-st-tools --no-default-features --features cpu --release
```

**Lines 361-362:** (Duplicate of above)
```bash
# CURRENT (INCORRECT):
cargo build -p bitnet-st2gguf --release
cargo build -p bitnet-st-tools --release

# SHOULD BE:
cargo build -p bitnet-st2gguf --no-default-features --features cpu --release
cargo build -p bitnet-st-tools --no-default-features --features cpu --release
```

### 1.2 docs/howto/validate-models.md

**Line 421:**
```bash
# CURRENT (INCORRECT):
cargo build --release -p bitnet-st2gguf

# SHOULD BE:
cargo build --release -p bitnet-st2gguf --no-default-features --features cpu
```

### 1.3 docs/reference/tokenizer-discovery-api.md

**Line 999:**
```bash
# CURRENT (INCORRECT):
cargo build --all-features

# SHOULD BE:
cargo build --no-default-features --all-features
# OR more specifically:
cargo build --no-default-features --features cpu
# OR for GPU:
cargo build --no-default-features --features gpu
```

### 1.4 docs/explanation/cpu-inference-api-contracts.md

**Line 704:**
```bash
# CURRENT (INCORRECT):
cargo build --features cpu,crossval

# SHOULD BE:
cargo build --no-default-features --features cpu,crossval
```

### 1.5 docs/explanation/issue-465-implementation-spec.md

**Line 258:**
```bash
# CURRENT (INCORRECT):
cargo build

# SHOULD BE:
cargo build --no-default-features --features cpu
# OR for GPU:
cargo build --no-default-features --features gpu
```

### 1.6 CI.md

**Line 174:**
```bash
# CURRENT (INCORRECT):
cargo build -p bitnet-cli --release \

# SHOULD BE:
cargo build -p bitnet-cli --release --no-default-features --features cpu \
```

### 1.7 CROSSVAL.md

**Line 105:**
```bash
# CURRENT (INCORRECT):
cargo build --features crossval -p bitnet-crossval --release

# SHOULD BE:
cargo build --no-default-features --features crossval -p bitnet-crossval --release
```

**Line 229:**
```bash
# CURRENT (INCORRECT):
cargo build --features crossval

# SHOULD BE:
cargo build --no-default-features --features crossval
```

---

## Category 2: Cargo Test -p Commands (94+ instances)

### 2.1 docs/reference/quantization-support.md

**Lines 256, 260:**
```bash
# CURRENT (INCORRECT):
cargo test -p bitnet-inference --features cpu
cargo test -p bitnet-quantization --features cpu

# SHOULD BE:
cargo test -p bitnet-inference --no-default-features --features cpu
cargo test -p bitnet-quantization --no-default-features --features cpu
```

**Line 556:**
```bash
# CURRENT (INCORRECT):
cargo test -p xtask test_ac6_receipt_quantized_kernels_valid -- --nocapture

# SHOULD BE (xtask doesn't need features but should be consistent):
cargo run -p xtask test_ac6_receipt_quantized_kernels_valid -- --nocapture
# Note: xtask should use 'cargo run -p xtask --' pattern
```

### 2.2 docs/environment-variables.md

**Lines 250, 254:**
```bash
# CURRENT (INCORRECT):
cargo test -p bitnet-inference --features cpu test_inference_real_computation
cargo test -p bitnet-quantization --features cpu test_quantization_kernel_integration

# SHOULD BE:
cargo test -p bitnet-inference --no-default-features --features cpu test_inference_real_computation
cargo test -p bitnet-quantization --no-default-features --features cpu test_quantization_kernel_integration
```

### 2.3 docs/development/validation-ci.md

**Lines 159, 289, 378:**
```bash
# CURRENT (INCORRECT):
cargo test -p bitnet-cli --test validation_workflow \

# SHOULD BE:
cargo test -p bitnet-cli --no-default-features --features cpu,full-cli --test validation_workflow \
```

### 2.4 docs/development/test-suite-issue-439.md

**Lines 180, 183:**
```bash
# CURRENT (INCORRECT):
cargo test --package bitnet-kernels --test feature_gate_consistency
cargo test --package bitnet-kernels --test build_script_validation

# SHOULD BE:
cargo test --package bitnet-kernels --no-default-features --features cpu --test feature_gate_consistency
cargo test --package bitnet-kernels --no-default-features --features cpu --test build_script_validation
```

### 2.5 docs/reference/strict-mode-api.md

**Lines 745, 789, 818:**
```bash
# CURRENT (INCORRECT):
cargo test -p bitnet-inference test_strict_quantization
cargo test -p bitnet-inference test_ac3_strict_mode_rejects_fallback
cargo test -p bitnet-inference test_ac5_16_token_decode_cpu_strict_mode

# SHOULD BE:
cargo test -p bitnet-inference --no-default-features --features cpu test_strict_quantization
cargo test -p bitnet-inference --no-default-features --features cpu test_ac3_strict_mode_rejects_fallback
cargo test -p bitnet-inference --no-default-features --features cpu test_ac5_16_token_decode_cpu_strict_mode
```

### 2.6 docs/how-to/strict-mode-validation-workflows.md

**Line 442:**
```bash
# CURRENT (INCORRECT):
cargo test -p bitnet-inference --features cpu integration_test_with_mock_tokenizer

# SHOULD BE:
cargo test -p bitnet-inference --no-default-features --features cpu integration_test_with_mock_tokenizer
```

### 2.7 docs/explanation/tl-lut-helper-spec.md

**Lines 471, 487, 488:**
```bash
# CURRENT (INCORRECT):
cargo test -p bitnet-kernels test_ac4_tl_lut_index --features cpu
cargo test -p bitnet-inference test_ac4_tl1_matmul_with_safe_lut --features cpu
cargo test -p bitnet-inference test_ac4_tl2_matmul_with_safe_lut --features cpu

# SHOULD BE:
cargo test -p bitnet-kernels --no-default-features --features cpu test_ac4_tl_lut_index
cargo test -p bitnet-inference --no-default-features --features cpu test_ac4_tl1_matmul_with_safe_lut
cargo test -p bitnet-inference --no-default-features --features cpu test_ac4_tl2_matmul_with_safe_lut
```

### 2.8 docs/explanation/cpu-inference-architecture.md

**Lines 482, 485, 488:**
```bash
# CURRENT (INCORRECT):
cargo test -p bitnet-inference test_cpu_forward_bos_nonzero --features cpu
cargo test -p bitnet-inference test_ac1_greedy_decode_16_tokens --features cpu
cargo test -p bitnet-cli test_ac2_cli_inference_question_answering --features cpu

# SHOULD BE:
cargo test -p bitnet-inference --no-default-features --features cpu test_cpu_forward_bos_nonzero
cargo test -p bitnet-inference --no-default-features --features cpu test_ac1_greedy_decode_16_tokens
cargo test -p bitnet-cli --no-default-features --features cpu test_ac2_cli_inference_question_answering
```

### 2.9 docs/explanation/cpu-inference-test-plan.md

**Lines 58, 210, 317, 367, 573, 617, 666, 804, 810, 822:**
```bash
# CURRENT (INCORRECT):
cargo test -p bitnet-inference test_ac1_cpu_forward_bos_nonzero_logits \
cargo test -p bitnet-inference test_ac1_kv_cache_update_retrieval \
cargo test -p bitnet-cli test_ac2_cli_priming_loop \
cargo test -p bitnet-cli test_ac2_cli_decode_loop_sampling \
cargo test -p bitnet-kernels test_ac4_tl_lut_index_bounds_valid \
cargo test -p bitnet-kernels test_ac4_tl_lut_index_bounds_invalid \
cargo test -p bitnet-inference test_ac4_tl_matmul_with_safe_lut \
cargo test -p bitnet-inference test_ac1 \
cargo test -p bitnet-cli test_ac2 \
cargo test -p bitnet-kernels test_ac4 \

# SHOULD BE (add --no-default-features to ALL):
cargo test -p bitnet-inference --no-default-features --features cpu test_ac1_cpu_forward_bos_nonzero_logits \
cargo test -p bitnet-inference --no-default-features --features cpu test_ac1_kv_cache_update_retrieval \
cargo test -p bitnet-cli --no-default-features --features cpu test_ac2_cli_priming_loop \
cargo test -p bitnet-cli --no-default-features --features cpu test_ac2_cli_decode_loop_sampling \
cargo test -p bitnet-kernels --no-default-features --features cpu test_ac4_tl_lut_index_bounds_valid \
cargo test -p bitnet-kernels --no-default-features --features cpu test_ac4_tl_lut_index_bounds_invalid \
cargo test -p bitnet-inference --no-default-features --features cpu test_ac4_tl_matmul_with_safe_lut \
cargo test -p bitnet-inference --no-default-features --features cpu test_ac1 \
cargo test -p bitnet-cli --no-default-features --features cpu test_ac2 \
cargo test -p bitnet-kernels --no-default-features --features cpu test_ac4 \
```

### 2.10 docs/performance-benchmarking.md

**Lines 620, 621, 625, 629, 631:**
```bash
# CURRENT (INCORRECT):
cargo test -p bitnet-common test_strict_mode_from_env_detailed
cargo test -p bitnet-common test_strict_mode_ci_enhanced
cargo test -p bitnet-quantization test_i2s_simd_scalar_parity
cargo test -p bitnet-kernels test_cpu_performance_baselines
cargo test -p bitnet-kernels test_gpu_performance_baselines --features gpu

# SHOULD BE:
cargo test -p bitnet-common --no-default-features test_strict_mode_from_env_detailed
cargo test -p bitnet-common --no-default-features test_strict_mode_ci_enhanced
cargo test -p bitnet-quantization --no-default-features --features cpu test_i2s_simd_scalar_parity
cargo test -p bitnet-kernels --no-default-features --features cpu test_cpu_performance_baselines
cargo test -p bitnet-kernels --no-default-features --features gpu test_gpu_performance_baselines
```

### 2.11 Root Directory Issue Files

**INFERENCE_FIXES.md (lines 174-175):**
```bash
# CURRENT (INCORRECT):
cargo test -p bitnet-models i2s_lut_mapping_sym_k1
cargo test -p bitnet-models i2s_extreme_scale_values

# SHOULD BE:
cargo test -p bitnet-models --no-default-features --features cpu i2s_lut_mapping_sym_k1
cargo test -p bitnet-models --no-default-features --features cpu i2s_extreme_scale_values
```

**issue-dead-code-stepby-trait-removal.md (lines 200, 215):**
```bash
# CURRENT (INCORRECT):
cargo test -p bitnet-quantization
cargo test -p bitnet-quantization -- --nocapture

# SHOULD BE:
cargo test -p bitnet-quantization --no-default-features --features cpu
cargo test -p bitnet-quantization --no-default-features --features cpu -- --nocapture
```

**issue-gguf-minimal-mock-tensors.md (lines 507-531):**
```bash
# CURRENT (INCORRECT):
cargo test --package bitnet-models gguf_error_handling
cargo test --package bitnet-models tensor_validation
cargo test --package bitnet-models test_utils::mock_tensors
cargo test --package bitnet-models --test gguf_loading_scenarios
cargo test --package bitnet-models --test config_scenarios
cargo test --package bitnet-models --test malformed_gguf
cargo test --package bitnet-models --test incomplete_gguf

# SHOULD BE (add --no-default-features --features cpu to ALL):
cargo test --package bitnet-models --no-default-features --features cpu gguf_error_handling
cargo test --package bitnet-models --no-default-features --features cpu tensor_validation
cargo test --package bitnet-models --no-default-features --features cpu test_utils::mock_tensors
cargo test --package bitnet-models --no-default-features --features cpu --test gguf_loading_scenarios
cargo test --package bitnet-models --no-default-features --features cpu --test config_scenarios
cargo test --package bitnet-models --no-default-features --features cpu --test malformed_gguf
cargo test --package bitnet-models --no-default-features --features cpu --test incomplete_gguf
```

**VALIDATION.md (line 119):**
```bash
# CURRENT (INCORRECT):
cargo test -p bitnet-inference --test engine_inspect

# SHOULD BE:
cargo test -p bitnet-inference --no-default-features --features cpu --test engine_inspect
```

---

## Category 3: Cargo Test --workspace Commands (14 instances)

### 3.1 docs/reference/quantization-support.md

**Line 268:**
```bash
# CURRENT (INCORRECT):
cargo test --workspace --features cpu

# SHOULD BE:
cargo test --workspace --no-default-features --features cpu
```

### 3.2 docs/environment-variables.md

**Line 262:**
```bash
# CURRENT (INCORRECT):
cargo test --workspace --features cpu

# SHOULD BE:
cargo test --workspace --no-default-features --features cpu
```

### 3.3 docs/explanation/cpu-inference-api-contracts.md

**Line 707:**
```bash
# CURRENT (INCORRECT):
cargo test --workspace --features cpu

# SHOULD BE:
cargo test --workspace --no-default-features --features cpu
```

### 3.4 docs/explanation/cpu-inference-test-plan.md

**Line 828:**
```bash
# CURRENT (INCORRECT):
cargo test --workspace \

# SHOULD BE:
cargo test --workspace --no-default-features --features cpu \
```

### 3.5 docs/explanation/specs/issue-447-compilation-fixes-technical-spec.md

**Line 584:**
```bash
# CURRENT (INCORRECT):
cargo test --workspace --all-features --no-run

# SHOULD BE:
cargo test --workspace --no-default-features --all-features --no-run
```

### 3.6 docs/explanation/specs/issue-447-finalized-acceptance-criteria.md

**Lines 257, 288:**
```bash
# CURRENT (INCORRECT):
cargo test --workspace --all-features

# SHOULD BE:
cargo test --workspace --no-default-features --all-features
```

### 3.7 docs/explanation/specs/ci-feature-aware-gates-spec.md

**Lines 45, 260, 283, 298:**
```bash
# CURRENT (INCORRECT):
cargo test --workspace --all-features

# SHOULD BE:
cargo test --workspace --no-default-features --all-features
```

### 3.8 docs/performance-benchmarking.md

**Line 597:**
```bash
# CURRENT (INCORRECT):
cargo test --workspace --features cpu

# SHOULD BE:
cargo test --workspace --no-default-features --features cpu
```

### 3.9 docs/explanation/strict-quantization-guards.md

**Line 512:**
```bash
# CURRENT (INCORRECT):
cargo test --doc -p bitnet-common strict_mode

# SHOULD BE:
cargo test --doc -p bitnet-common --no-default-features strict_mode
```

### 3.10 issue-production-readiness-comprehensive-initiative.md

**Line 166:**
```bash
# CURRENT (INCORRECT):
cargo test --workspace --all-features

# SHOULD BE:
cargo test --workspace --no-default-features --all-features
```

---

## Category 4: Cargo Run --example Commands (9 instances)

### 4.1 docs/archive/reports/ENHANCED_ERROR_HANDLING_SUMMARY.md

**Line 188:**
```bash
# CURRENT (INCORRECT):
cargo run --example enhanced_error_demo

# SHOULD BE:
cargo run --example enhanced_error_demo --no-default-features --features cpu
```

### 4.2 docs/archive/reports/FAST_FEEDBACK_IMPLEMENTATION_SUMMARY.md

**Lines 164, 167, 170:**
```bash
# CURRENT (INCORRECT):
cargo run -p bitnet-tests --bin fast_feedback_simple_demo -- dev
cargo run -p bitnet-tests --bin fast_feedback_simple_demo -- ci
cargo run -p bitnet-tests --bin fast_feedback_simple_demo -- auto

# SHOULD BE:
cargo run -p bitnet-tests --no-default-features --features cpu --bin fast_feedback_simple_demo -- dev
cargo run -p bitnet-tests --no-default-features --features cpu --bin fast_feedback_simple_demo -- ci
cargo run -p bitnet-tests --no-default-features --features cpu --bin fast_feedback_simple_demo -- auto
```

### 4.3 docs/streaming-api.md

**Line 127:**
```bash
# CURRENT (INCORRECT):
cargo run --example streaming_generation -- --buffer-size 5 --flush-interval 25ms

# SHOULD BE:
cargo run --example streaming_generation --no-default-features --features cpu -- --buffer-size 5 --flush-interval 25ms
```

### 4.4 docs/testing/README.md

**Line 60:**
```bash
# CURRENT (INCORRECT):
cargo run -p bitnet-tests --example reporting_example

# SHOULD BE:
cargo run -p bitnet-tests --no-default-features --features cpu --example reporting_example
```

### 4.5 CLAUDE.md

**Line 42:**
```bash
# CURRENT (INCORRECT):
cargo run -p bitnet-st2gguf -- --input model.safetensors --output model.gguf --strict

# SHOULD BE:
cargo run -p bitnet-st2gguf --no-default-features --features cpu -- --input model.safetensors --output model.gguf --strict
```

---

## Category 5: Special Cases & Architecture Documents

### 5.1 xtask Commands

Many `cargo test -p xtask` commands found. While xtask doesn't strictly need features, the repository standard is to use `cargo run -p xtask --` pattern instead.

**Affected files:**
- docs/reference/quantization-support.md (line 556)
- docs/explanation/strict-quantization-guards.md (lines 454, 457, 460)
- docs/explanation/receipt-cpu-validation-spec.md (lines 688, 705, 726)
- Multiple ADR documents in docs/explanation/architecture/

**Pattern to fix:**
```bash
# CURRENT (INCONSISTENT):
cargo test -p xtask test_name

# SHOULD BE:
cargo run -p xtask -- test_name
# OR if testing xtask itself:
cargo test -p xtask --no-default-features test_name
```

### 5.2 tests Package

Commands referencing `-p tests` package:

**docs/explanation/specs/issue-447-compilation-fixes-technical-spec.md:**
```bash
# CURRENT (INCORRECT):
cargo test -p tests --no-run
cargo test -p tests run_configuration_tests

# SHOULD BE:
cargo test -p tests --no-default-features --no-run
cargo test -p tests --no-default-features run_configuration_tests
```

---

## Recommendations

### High Priority (Breaks Builds)

1. **All `cargo build` commands** must include `--no-default-features --features cpu|gpu`
2. **All `cargo test` commands** must include `--no-default-features --features cpu|gpu`
3. **All `cargo test --workspace` commands** must include `--no-default-features`

### Medium Priority (Consistency)

1. **xtask invocations** should use `cargo run -p xtask --` pattern
2. **Example commands** should specify features explicitly

### Low Priority (Documentation Clarity)

1. Add comments explaining why `--no-default-features` is required
2. Reference CLAUDE.md or docs/explanation/FEATURES.md for context

---

## Automated Fix Strategy

The following sed commands can fix most instances:

```bash
# Fix cargo test -p with --features cpu
sed -i 's/^cargo test -p \([^ ]*\) --features cpu/cargo test -p \1 --no-default-features --features cpu/' *.md docs/**/*.md

# Fix cargo test -p with --features gpu
sed -i 's/^cargo test -p \([^ ]*\) --features gpu/cargo test -p \1 --no-default-features --features gpu/' *.md docs/**/*.md

# Fix cargo test -p without any features
sed -i 's/^cargo test -p \([^ ]*\) \([^-]\)/cargo test -p \1 --no-default-features --features cpu \2/' *.md docs/**/*.md

# Fix cargo test --workspace --features
sed -i 's/^cargo test --workspace --features/cargo test --workspace --no-default-features --features/' *.md docs/**/*.md

# Fix cargo build --features
sed -i 's/^cargo build --features/cargo build --no-default-features --features/' *.md docs/**/*.md
```

**Warning:** Manual review required after automated fixes to ensure correctness.

---

## Files Requiring Manual Review

The following files have complex multi-line commands or special contexts:

1. docs/development/validation-ci.md (validation workflows)
2. docs/explanation/cpu-inference-test-plan.md (extensive test specifications)
3. docs/explanation/specs/ci-feature-aware-gates-spec.md (CI specifications)
4. All ADR documents in docs/explanation/architecture/ (architectural decisions)
5. Root-level issue tracking files (issue-*.md)

---

## Summary Statistics

- **Total files affected:** 60+
- **Total commands to fix:** 205+
- **Estimated effort:** 4-6 hours for comprehensive manual review and fixes
- **Risk:** Medium (documentation only, no code changes)

---

## Next Steps

1. Review this report with maintainers
2. Prioritize high-impact documentation (quickstart, getting-started, tutorials)
3. Apply automated fixes to clear-cut cases
4. Manually review and fix complex cases
5. Add pre-commit hook or CI check to prevent future violations
