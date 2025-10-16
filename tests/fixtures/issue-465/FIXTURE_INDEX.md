# Issue #465 Test Fixtures Index

Comprehensive test fixtures for Issue #465: CPU Path Followup test infrastructure.

## Overview

This directory contains realistic test data for validating Issue #465 acceptance criteria. All fixtures follow BitNet.rs neural network patterns with proper I2_S quantization context, receipt schema v1.0.0 compliance, and feature-gated organization.

## Fixture Groups

### Group 1: Receipt Fixtures (AC3, AC4, AC6)

**Purpose:** Validate CPU baseline generation, receipt verification, and honest compute enforcement.

#### Valid Receipts

**File:** `cpu-baseline-valid.json`
- **Purpose:** Valid CPU baseline receipt with I2_S quantization
- **Kernels:** 11 realistic BitNet.rs kernel IDs (`i2s_cpu_quantized_matmul`, `tl1_lut_dequant_forward`, etc.)
- **Performance:** 15.3 tok/s (realistic for 2B I2_S model on AVX2 CPU)
- **Schema:** v1.0.0 compliant with all required fields
- **Usage:** AC3 (baseline generation), AC4 (baseline verification)
- **Test File:** `issue_465_baseline_tests.rs`

#### Invalid Receipts (Negative Tests)

**File:** `cpu-baseline-mocked.json`
- **Purpose:** Mocked receipt that should FAIL verification
- **Issue:** `compute_path: "mocked"` instead of `"real"`
- **Expected:** Receipt verification rejects with "invalid compute_path" error
- **Usage:** AC6 (smoke test validation)
- **Test File:** `issue_465_ci_gates_tests.rs`

**File:** `cpu-baseline-invalid-empty-kernels.json`
- **Purpose:** Receipt with empty kernels array
- **Issue:** `kernels: []` with `compute_path: "real"` (honest compute violation)
- **Expected:** Receipt verification rejects with "empty kernels array" error
- **Usage:** AC6 (smoke test validation)
- **Test File:** `issue_465_ci_gates_tests.rs`

**File:** `cpu-baseline-invalid-compute-path.json`
- **Purpose:** Receipt with wrong compute_path value
- **Issue:** `compute_path: "fake"` (not "real")
- **Expected:** Receipt verification rejects with "invalid compute_path" error
- **Usage:** AC6 (smoke test validation)
- **Test File:** `issue_465_ci_gates_tests.rs`

**File:** `cpu-baseline-invalid-kernel-hygiene.json`
- **Purpose:** Receipt with invalid kernel IDs
- **Issues:**
  - Empty kernel ID: `""`
  - Kernel ID exceeding 128 characters
- **Expected:** Receipt verification rejects with "kernel hygiene violation" error
- **Usage:** AC6 (smoke test validation)
- **Test File:** `issue_465_ci_gates_tests.rs`

### Group 2: README Documentation Templates (AC1, AC2)

**Purpose:** Provide template content for README.md documentation updates.

**Directory:** `readme-templates/`

**File:** `quickstart-section.md`
- **Purpose:** 10-line CPU quickstart template for AC1
- **Content:**
  - Build with explicit CPU features
  - Model download via xtask
  - Deterministic inference configuration
  - Receipt verification workflow
- **Features:**
  - Feature flag pattern: `--no-default-features --features cpu`
  - Environment variables: `BITNET_DETERMINISTIC=1`, `RAYON_NUM_THREADS=1`, `BITNET_SEED=42`
  - Performance context: "10-20 tok/s on CPU for 2B I2_S models"
  - Baseline reference: `docs/baselines/`
- **Usage:** AC1 (README quickstart)
- **Test File:** `issue_465_documentation_tests.rs::test_ac1_readme_quickstart_block_present`

**File:** `receipts-section.md`
- **Purpose:** Comprehensive receipts documentation for AC2
- **Content:**
  - Receipt schema v1.0.0 with example JSON
  - xtask commands (`benchmark`, `verify-receipt`)
  - Environment variables table
  - Kernel ID hygiene requirements
  - Baseline receipts reference
- **Features:**
  - Complete receipt schema example
  - Honest compute requirements
  - CI enforcement documentation
  - Baseline storage conventions
- **Usage:** AC2 (README receipts documentation)
- **Test File:** `issue_465_documentation_tests.rs::test_ac2_readme_receipts_block_present`

**File:** `environment-vars-table.md`
- **Purpose:** Environment variables reference table
- **Content:**
  - Inference configuration variables
  - Validation configuration variables
  - Testing configuration variables
  - Usage examples for each category
- **Features:**
  - Markdown table format
  - Default values documented
  - Example usage patterns
- **Usage:** AC2 (environment variables documentation)
- **Test File:** `issue_465_documentation_tests.rs::test_ac2_readme_receipts_block_present`

### Group 3: Documentation Audit Data (AC9, AC10)

**Purpose:** Support feature flag standardization and performance claims audit.

**Directory:** `doc-audit/`

**File:** `legacy-commands.txt`
- **Purpose:** Legacy cargo command patterns to audit (AC9)
- **Content:**
  - Legacy patterns: `cargo build`, `cargo test`, `cargo run` (without feature flags)
  - Replacement patterns with feature flags
  - Acceptable exceptions (fmt, clippy --all-features, doc)
- **Search Patterns:**
  - `cargo build` without `--no-default-features`
  - `cargo test` without `--no-default-features`
  - `cargo run` without `--no-default-features`
- **Usage:** AC9 (feature flag standardization)
- **Test File:** `issue_465_documentation_tests.rs::test_ac9_no_legacy_feature_commands`

**File:** `unsupported-claims.txt`
- **Purpose:** Unsupported performance claims patterns (AC10)
- **Content:**
  - Unsupported numbers: "200 tok/s", "500 tok/s", "1000 tok/s"
  - Vague claims: "blazing fast", "lightning speed", "ultra-fast"
  - Unsupported comparisons: "faster than X", "outperforms Y"
  - Acceptable patterns with evidence
- **Search Patterns:**
  - Performance numbers without receipt references
  - Vague claims without "measured", "baseline", or "receipt" context
- **Usage:** AC10 (remove legacy performance claims)
- **Test File:** `issue_465_documentation_tests.rs::test_ac10_no_unsupported_performance_claims`

**File:** `standardized-commands.txt`
- **Purpose:** Standardized feature-aware command patterns
- **Content:**
  - Build commands with CPU/GPU features
  - Test commands with feature flags
  - Run commands for CLI and xtask
  - Quality commands (fmt, clippy, doc)
  - Cross-validation and GPU development patterns
- **Features:**
  - All commands use `--no-default-features` (except xtask)
  - Explicit CPU/GPU feature selection
  - Unified GPU predicate pattern
- **Usage:** AC9 (standardized command reference)
- **Test File:** `issue_465_documentation_tests.rs`

### Group 4: GitHub API Mock Data (AC5, AC7, AC8, AC12)

**Purpose:** Mock GitHub API responses for CI gate and release QA validation.

**File:** `branch-protection-response.json`
- **Purpose:** GitHub branch protection settings for main branch (AC5)
- **Content:**
  - Required status checks: "Model Gates (CPU) / cpu-receipt-gate", "Model Gates (CPU) / gate-summary"
  - Required approving review count: 1
  - Enforce admins: true
  - Allow force pushes: false
- **API Endpoint:** `GET /repos/EffortlessMetrics/BitNet-rs/branches/main/protection`
- **Usage:** AC5 (branch protection configuration)
- **Test File:** `issue_465_ci_gates_tests.rs::test_ac5_branch_protection_configured`

**File:** `pr-435-merge-data.json`
- **Purpose:** PR #435 merge status data (AC7)
- **Content:**
  - PR number: 435
  - State: "closed"
  - Merged: true
  - Merged at: "2025-10-09T13:36:49Z"
  - Title: "Mock-elimination & baselines"
  - Milestone: "v0.1.0-mvp"
  - Commits: 12, additions: 450, deletions: 120
- **API Endpoint:** `GET /repos/EffortlessMetrics/BitNet-rs/pulls/435`
- **Usage:** AC7 (PR #435 merge status)
- **Test File:** `issue_465_release_qa_tests.rs::test_ac7_pr_435_merged`

**File:** `issue-closure-data.json`
- **Purpose:** Mock-inference issue closure data (AC8)
- **Content:**
  - Issue number: 420
  - Title: "Eliminate mocked inference from production code"
  - State: "closed"
  - Closed at: "2025-10-09T13:40:00Z"
  - Milestone: "v0.1.0-mvp"
  - Resolution: "Fixed in PR #435"
- **API Endpoint:** `GET /repos/EffortlessMetrics/BitNet-rs/issues/420`
- **Usage:** AC8 (issue closure validation)
- **Test File:** `issue_465_release_qa_tests.rs::test_ac8_mock_inference_issue_closed`

**File:** `tag-v0.1.0-mvp-data.json`
- **Purpose:** v0.1.0-mvp tag creation data (AC12)
- **Content:**
  - Tag: "v0.1.0-mvp"
  - Message: Comprehensive release notes with CPU baseline reference
  - Tagger: BitNet.rs Maintainer
  - Date: "2025-10-15T18:00:00Z"
  - Object: Commit SHA for main branch merge
  - Verification: PGP signature (verified: true)
- **API Endpoint:** `GET /repos/EffortlessMetrics/BitNet-rs/git/tags/{sha}`
- **Usage:** AC12 (tag creation validation)
- **Test File:** `issue_465_release_qa_tests.rs::test_ac12_v0_1_0_mvp_tag_created`

### Group 5: Pre-Tag Verification Script (AC11)

**Purpose:** Executable script for pre-tag quality gates.

**File:** `pre-tag-verification.sh`
- **Purpose:** Pre-tag verification workflow for v0.1.0-mvp (AC11)
- **Permissions:** Executable (`chmod +x`)
- **Steps:**
  1. Format check: `cargo fmt --all --check`
  2. Clippy: `cargo clippy --all-targets --all-features -- -D warnings`
  3. CPU tests: `cargo test --workspace --no-default-features --features cpu`
  4. Deterministic benchmark: `cargo run -p xtask -- benchmark --model <model> --tokens 128`
  5. Receipt verification: `cargo run -p xtask -- verify-receipt ci/inference.json`
  6. Baseline check: Verify `docs/baselines/*-cpu.json` exists and is valid
- **Configuration:**
  - Deterministic environment: `BITNET_DETERMINISTIC=1`, `RAYON_NUM_THREADS=1`, `BITNET_SEED=42`
  - Model auto-discovery: Finds first `*.gguf` in `models/` directory
  - Colored output: Red (errors), green (success), yellow (info)
- **Exit Codes:**
  - 0: All checks passed, ready to tag
  - 1: Quality gate failure (format, clippy, tests, benchmark, receipt, baseline)
- **Usage:** AC11 (pre-tag verification)
- **Test File:** `issue_465_release_qa_tests.rs::test_ac11_pre_tag_verification_passes`

## Fixture Loading Pattern

All test files use consistent fixture loading:

```rust
use std::fs;
use std::path::PathBuf;

fn load_fixture(name: &str) -> String {
    let fixture_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("fixtures/issue-465")
        .join(name);

    fs::read_to_string(&fixture_path)
        .expect(&format!("Failed to load fixture: {}", name))
}

// Usage
let valid_receipt = load_fixture("cpu-baseline-valid.json");
let receipt: Receipt = serde_json::from_str(&valid_receipt)?;
```

## Neural Network Context

All fixtures include realistic BitNet.rs neural network patterns:

### I2_S Quantization

- **Kernel IDs:** `i2s_cpu_quantized_matmul`, `i2s_cpu_quant_block_128`
- **Performance:** 10-20 tok/s for 2B models on CPU (AVX2)
- **Context:** Production 2-bit signed quantization (99%+ accuracy vs FP32)

### Table Lookup Quantization

- **Kernel IDs:** `tl1_lut_dequant_forward`, `tl2_lut_dequant_forward`
- **Context:** Device-aware TL1/TL2 with LUT-based dequantization

### Transformer Pipeline

- **Attention:** `attention_kv_cache_update`, `cpu_rope_embedding`, `cpu_softmax`
- **FFN:** `ffn_forward`, `cpu_vector_add`
- **Normalization:** `layernorm_forward`, `cpu_rmsnorm`

### Performance Baselines

- **CPU:** 10-20 tok/s (realistic for 2B I2_S model on AVX2)
- **Context:** Receipt-verified throughput with kernel IDs
- **Storage:** `docs/baselines/YYYYMMDD-cpu.json`

### Receipt Schema v1.0.0

All receipt fixtures conform to schema requirements:

- **Required Fields:** `schema_version`, `compute_path`, `backend`, `model`, `quantization`, `tokens_generated`, `throughput_tokens_per_sec`, `success`, `kernels`, `timestamp`
- **Kernel Hygiene:** Non-empty strings, ≤128 chars, ≤10,000 count
- **Honest Compute:** `compute_path: "real"`, non-empty kernels array

## Validation

All fixtures validated against BitNet.rs standards:

```bash
# Validate receipt fixtures
cargo test -p bitnet-tests --test issue_465_baseline_tests
cargo test -p bitnet-tests --test issue_465_ci_gates_tests

# Validate documentation fixtures
cargo test -p bitnet-tests --test issue_465_documentation_tests

# Validate release QA fixtures
cargo test -p bitnet-tests --test issue_465_release_qa_tests

# Run pre-tag verification script
./tests/fixtures/issue-465/pre-tag-verification.sh
```

## References

- **Issue Specification:** `docs/explanation/issue-465-implementation-spec.md`
- **Receipt Schema:** Schema v1.0.0 (see `receipts-section.md`)
- **Test Scaffolding:**
  - `tests/issue_465_documentation_tests.rs`
  - `tests/issue_465_baseline_tests.rs`
  - `tests/issue_465_ci_gates_tests.rs`
  - `tests/issue_465_release_qa_tests.rs`
- **Fixture Specification:** `tests/fixtures/issue_465_test_fixtures_spec.md`

## Maintenance

When updating fixtures:

1. **Preserve Neural Network Context:** Ensure kernel IDs match real BitNet.rs implementations
2. **Maintain Receipt Schema:** Keep receipts v1.0.0 compliant
3. **Update Documentation:** Reflect fixture changes in FIXTURE_INDEX.md
4. **Validate Tests:** Run test suite to ensure fixtures integrate correctly
5. **Commit Evidence:** Include fixture validation results in commit message

## Coverage

All 12 acceptance criteria covered:

- **AC1:** README quickstart block → `readme-templates/quickstart-section.md`
- **AC2:** README receipts documentation → `readme-templates/receipts-section.md`
- **AC3:** CPU baseline generation → `cpu-baseline-valid.json`
- **AC4:** Baseline verification → `cpu-baseline-valid.json`
- **AC5:** Branch protection → `branch-protection-response.json`
- **AC6:** Smoke test (mocked rejection) → `cpu-baseline-mocked.json`, `cpu-baseline-invalid-*.json`
- **AC7:** PR #435 merge status → `pr-435-merge-data.json`
- **AC8:** Issue closure → `issue-closure-data.json`
- **AC9:** Feature flag standardization → `doc-audit/legacy-commands.txt`, `doc-audit/standardized-commands.txt`
- **AC10:** Legacy performance claims → `doc-audit/unsupported-claims.txt`
- **AC11:** Pre-tag verification → `pre-tag-verification.sh`
- **AC12:** v0.1.0-mvp tag creation → `tag-v0.1.0-mvp-data.json`
