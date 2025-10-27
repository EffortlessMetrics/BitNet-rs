# Parity-Both Command: Preflight Integration and TokenizerAuthority Specification

**Status**: Implementation Ready
**Version**: 2.0 (Comprehensive Integration)
**Feature**: Dual-lane parity with auto-repair and tokenizer authority tracking
**Priority**: High (Cross-Validation Infrastructure)
**Scope**: `xtask/src/crossval/parity_both.rs`, `crossval/src/receipt.rs`, `crossval/src/preflight.rs`

---

## 1. Executive Summary

This specification defines the comprehensive integration of preflight auto-repair, tokenizer authority tracking, and dual-lane cross-validation for the `parity-both` command. It extends the baseline parity-both functionality with:

1. **Preflight integration** with automatic repair and retry semantics
2. **TokenizerAuthority metadata** for receipt reproducibility
3. **Exit code standardization** (0=both pass, 1=one fails, 2=usage error)
4. **Dual receipt generation** with shared tokenizer provenance
5. **Token parity validation** as fail-fast gate
6. **Per-position metrics** with configurable thresholds

### 1.1 Problem Statement

**Current gaps**:
- Auto-repair parameter accepted but unused in parity-both orchestration
- Exit code logic incomplete (no exit code 2 for usage errors)
- Tokenizer authority not tracked in receipts (reproducibility gap)
- Token parity validation happens per-lane (not fail-fast)
- Metrics parameter parsed but not applied to receipt generation

**Desired state**:
- Preflight with auto-repair integrated into dual-lane flow
- Exit codes standardized (0/1/2) with clear semantics
- Tokenizer authority captured and validated across lanes
- Token parity validated once during shared setup
- Metrics customization fully functional

### 1.2 Key Goals

1. **AC1**: Preflight both backends before dual-lane execution
2. **AC2**: Auto-repair integration (--repair flag, default enabled)
3. **AC3**: TokenizerAuthority in both receipts (source, file_hash, config_hash)
4. **AC4**: Exit code logic (0=both pass, 1=one fails, 2=both fail/usage error)
5. **AC5**: Dual receipt naming (receipt_bitnet.json, receipt_llama.json)
6. **AC6**: Shared Rust inference (reused for both lanes)
7. **AC7**: Per-position metrics (MSE, L2, KL divergence, TopK agreement)
8. **AC8**: Summary with first divergence position
9. **AC9**: CLI integration (main.rs wiring)
10. **AC10**: Test coverage (dual-lane, exit codes, tokenizer consistency)

---

## 2. Feature Overview

### 2.1 Dual-Lane Orchestration with Shared Cost Optimization

```
┌─────────────────────────────────────────────────────────────┐
│ STEP 1: Preflight Both Backends (AC1, AC2)                 │
├─────────────────────────────────────────────────────────────┤
│ • Check BitNet.cpp availability (libbitnet*)                │
│ • Check llama.cpp availability (libllama*, libggml*)        │
│ • RepairMode: Auto (default) | Never (--no-repair)          │
│ • If Auto + missing: setup-cpp-auto + rebuild + retry       │
│ • Exit code 2 if either backend unavailable after repair    │
└─────────────────────────────────────────────────────────────┘
    ↓ (both backends available)
┌─────────────────────────────────────────────────────────────┐
│ STEP 2: Shared Setup + TokenizerAuthority (AC3, AC5)       │
├─────────────────────────────────────────────────────────────┤
│ • Template processing (auto-detect or explicit)             │
│ • Load Rust tokenizer (bitnet_tokenizers::loader)           │
│ • Compute TokenizerAuthority:                               │
│   - source: GgufEmbedded | JsonFile | AutoDiscovered        │
│   - file_hash: SHA256 of tokenizer.json (if external)       │
│   - config_hash: SHA256 of canonical config (vocab sizes)   │
│ • Rust tokenization (once, reused for both lanes)           │
│ • C++ tokenization for both backends                        │
│ • Token parity validation (fail-fast if mismatch → exit 2)  │
└─────────────────────────────────────────────────────────────┘
    ↓ (token parity OK)
┌─────────────────────────────────────────────────────────────┐
│ STEP 3: Shared Rust Logits (AC6)                           │
├─────────────────────────────────────────────────────────────┤
│ • Load GGUF model (bitnet_models::loader)                   │
│ • Evaluate logits for all positions (once)                  │
│ • Return logits matrix: Vec<Vec<f32>>                       │
│ • Cost: ~10-30s depending on model size and token count     │
│ • Reused for both BitNet.cpp and llama.cpp comparisons      │
└─────────────────────────────────────────────────────────────┘
    ↓ (shared Rust logits ready)
┌──────────────────────┬──────────────────────────────────────┐
│ LANE A: BitNet.cpp   │ LANE B: llama.cpp                    │
├──────────────────────┼──────────────────────────────────────┤
│ STEP 4A: C++ Logits  │ STEP 4B: C++ Logits                  │
│ • BitnetSession ctx  │ • BitnetSession ctx (llama backend)  │
│ • C++ eval           │ • C++ eval                           │
│ (~10-30s)            │ (~10-30s)                            │
├──────────────────────┼──────────────────────────────────────┤
│ STEP 5A: Compare     │ STEP 5B: Compare                     │
│ • Per-position MSE   │ • Per-position MSE                   │
│ • L2 distance        │ • L2 distance                        │
│ • KL divergence (opt)│ • KL divergence (opt)                │
│ • TopK agreement (op)│ • TopK agreement (opt)               │
│ • Divergence detect  │ • Divergence detect                  │
│ (~100ms)             │ (~100ms)                             │
├──────────────────────┼──────────────────────────────────────┤
│ STEP 6A: Receipt     │ STEP 6B: Receipt                     │
│ • ParityReceipt      │ • ParityReceipt                      │
│ • TokenizerAuthority │ • TokenizerAuthority (same as A)     │
│ • PositionMetrics[]  │ • PositionMetrics[]                  │
│ • Summary stats      │ • Summary stats                      │
│ • Write JSON:        │ • Write JSON:                        │
│   receipt_bitnet.json│   receipt_llama.json                 │
│ (~50ms)              │ (~50ms)                              │
└──────────────────────┴──────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 7: Unified Summary + Exit Code (AC4, AC8)             │
├─────────────────────────────────────────────────────────────┤
│ • Load both receipts as LaneResult                          │
│ • Print unified summary (text or JSON)                      │
│ • Determine exit code:                                      │
│   - Exit 0: Both lanes passed                               │
│   - Exit 1: Either lane failed (divergence detected)        │
│   - Exit 2: Token parity violation or usage error           │
│ • Display first divergence position for each lane           │
│ • Show mean MSE, cosine similarity, and status              │
└─────────────────────────────────────────────────────────────┘
```

**Optimization: Shared Rust Inference**:
- Tokenization happens once (shared for both backends)
- Rust logits evaluated once with `eval_logits_all_positions`
- Reused for comparison with both BitNet.cpp and llama.cpp
- Reduces execution time by ~50% vs. running backends sequentially

---

## 3. Acceptance Criteria

### AC1: Preflight Both Backends Before Dual-Lane Execution

**Requirement**: Validate both BitNet.cpp and llama.cpp availability before proceeding.

**Implementation**:
```rust
// Location: xtask/src/crossval/parity_both.rs
// STEP 1: Preflight both backends
let backends = [CppBackend::BitNet, CppBackend::Llama];
for backend in backends {
    if verbose {
        eprintln!("⚙ Preflight: Checking {} backend...", backend.name());
    }

    preflight_backend_libs(backend, verbose)
        .with_context(|| format!("Preflight check failed for {}", backend.name()))?;

    if verbose {
        eprintln!("  ✓ {} backend available", backend.name());
    }
}
```

**Success Criteria**:
- Both backends preflighted before any inference work
- Clear diagnostic output for each backend
- Fail-fast if either backend unavailable (before auto-repair)
- Exit code 2 if preflight fails after auto-repair

**Test Coverage**:
```rust
#[test]
fn test_preflight_both_backends_success() {
    // Given: Both bitnet.cpp and llama.cpp available
    // When: Run parity-both with verbose
    // Then: Both preflights pass with checkmarks
}

#[test]
fn test_preflight_bitnet_missing_exit_2() {
    // Given: bitnet.cpp unavailable, llama.cpp available, --no-repair
    // When: Run parity-both
    // Then: Exit code 2 with "BitNet.cpp libraries NOT FOUND"
}

#[test]
fn test_preflight_llama_missing_exit_2() {
    // Given: llama.cpp unavailable, bitnet.cpp available, --no-repair
    // When: Run parity-both
    // Then: Exit code 2 with "llama.cpp libraries NOT FOUND"
}
```

---

### AC2: Auto-Repair Integration (--repair flag)

**Requirement**: Integrate `preflight_with_auto_repair()` with retry logic.

**CLI Flag**:
```bash
# Auto-repair enabled by default
cargo run -p xtask --features crossval-all -- parity-both \
  --model-gguf models/model.gguf \
  --tokenizer models/tokenizer.json

# Disable auto-repair (manual setup)
cargo run -p xtask --features crossval-all -- parity-both \
  --model-gguf models/model.gguf \
  --tokenizer models/tokenizer.json \
  --no-repair
```

**Implementation**:
```rust
// Location: xtask/src/crossval/parity_both.rs
// STEP 1: Preflight with auto-repair
let repair_mode = if auto_repair { RepairMode::Auto } else { RepairMode::Never };

for backend in backends {
    preflight_with_auto_repair(backend, repair_mode, verbose)
        .with_context(|| format!("Preflight with auto-repair failed for {}", backend.name()))?;
}
```

**Auto-Repair Flow**:
```
┌──────────────────────────────────────┐
│ preflight_with_auto_repair(backend)  │
├──────────────────────────────────────┤
│ 1. Check HAS_BITNET / HAS_LLAMA      │
│ 2. If available → return Ok(())      │
│ 3. If missing + RepairMode::Never    │
│    → return Err (exit code 2)        │
│ 4. If missing + RepairMode::Auto:    │
│    a. Run setup-cpp-auto --emit=sh   │
│    b. Rebuild xtask with new env     │
│    c. Re-exec self (max 2 retries)   │
│    d. Check HAS_BITNET / HAS_LLAMA   │
│    e. If still missing → Err         │
└──────────────────────────────────────┘
```

**Success Criteria**:
- `auto_repair` parameter (default true) wired into preflight step
- RepairMode::Auto triggers setup-cpp-auto + rebuild + re-exec
- RepairMode::Never fails immediately with clear error message
- Retry logic bounded (max 2 retries with exponential backoff)
- Exit code 2 if repair fails after retries

**Test Coverage**:
```rust
#[test]
fn test_auto_repair_bitnet_success() {
    // Given: bitnet.cpp missing, auto_repair=true
    // When: Run parity-both
    // Then: setup-cpp-auto executed, xtask rebuilt, preflight passes
}

#[test]
fn test_auto_repair_disabled_exit_2() {
    // Given: bitnet.cpp missing, auto_repair=false
    // When: Run parity-both --no-repair
    // Then: Exit code 2 with "BitNet.cpp libraries NOT FOUND"
}

#[test]
fn test_auto_repair_exhausts_retries() {
    // Given: bitnet.cpp missing, setup-cpp-auto fails
    // When: Run parity-both
    // Then: Exit code 2 after 2 retry attempts
}
```

---

### AC3: TokenizerAuthority in Both Receipts

**Requirement**: Capture tokenizer provenance in receipts for reproducibility.

**Schema** (`crossval/src/receipt.rs`):
```rust
/// Tokenizer authority metadata for receipt reproducibility
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenizerAuthority {
    /// Tokenizer source: GgufEmbedded, JsonFile, or AutoDiscovered
    pub source: TokenizerSource,

    /// SHA256 hash of tokenizer.json file (if external)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub file_hash: Option<String>,

    /// SHA256 hash of effective tokenizer config (canonical representation)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub config_hash: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum TokenizerSource {
    GgufEmbedded,
    JsonFile(PathBuf),
    AutoDiscovered(PathBuf),
}
```

**Integration** (STEP 2: Shared Setup):
```rust
// Compute TokenizerAuthority once during shared setup
let tokenizer_authority = {
    let source = detect_tokenizer_source(&tokenizer_path, &model_path)?;
    let file_hash = match &source {
        TokenizerSource::GgufEmbedded => None,
        TokenizerSource::JsonFile(path) | TokenizerSource::AutoDiscovered(path) => {
            Some(compute_tokenizer_file_hash(path)?)
        }
    };
    let config_hash = Some(compute_tokenizer_config_hash(&tokenizer_obj)?);

    TokenizerAuthority { source, file_hash, config_hash }
};

// Pass tokenizer_authority to both lanes
```

**Receipt Generation** (STEP 6A/6B):
```rust
// Lane A (BitNet.cpp)
let mut receipt = ParityReceipt::new(model_path, "bitnet", formatted_prompt);
receipt.set_tokenizer_authority(tokenizer_authority.clone());
receipt.set_prompt_template(prompt_template.to_string());
// ... add position metrics ...
receipt.finalize();
receipt.write_to_file(&out_dir.join("receipt_bitnet.json"))?;

// Lane B (llama.cpp) - identical tokenizer authority
let mut receipt = ParityReceipt::new(model_path, "llama", formatted_prompt);
receipt.set_tokenizer_authority(tokenizer_authority.clone());
receipt.set_prompt_template(prompt_template.to_string());
// ... add position metrics ...
receipt.finalize();
receipt.write_to_file(&out_dir.join("receipt_llama.json"))?;
```

**Success Criteria**:
- TokenizerAuthority computed once during shared setup
- Same authority object passed to both lane receipts
- Source detection: GgufEmbedded vs JsonFile vs AutoDiscovered
- File hash (SHA256) computed for external tokenizers
- Config hash (SHA256) computed from canonical vocab representation
- Both receipts contain identical tokenizer_authority field

**Test Coverage**:
```rust
#[test]
fn test_tokenizer_authority_gguf_embedded() {
    // Given: Model with embedded tokenizer
    // When: Run parity-both
    // Then: TokenizerAuthority.source = GgufEmbedded, file_hash = None, config_hash = Some(...)
}

#[test]
fn test_tokenizer_authority_external_json() {
    // Given: External tokenizer.json file
    // When: Run parity-both --tokenizer tokenizer.json
    // Then: TokenizerAuthority.source = JsonFile(path), file_hash = Some(...), config_hash = Some(...)
}

#[test]
fn test_tokenizer_authority_identical_across_lanes() {
    // Given: Parity-both with both lanes successful
    // When: Load receipt_bitnet.json and receipt_llama.json
    // Then: tokenizer_authority fields are identical (config_hash match)
}
```

---

### AC4: Exit Code Logic (0=both pass, 1=one fails, 2=usage error)

**Requirement**: Standardize exit codes with clear semantics.

**Exit Code Table**:

| Scenario | Exit Code | Source | Notes |
|----------|-----------|--------|-------|
| Both lanes pass | 0 | `determine_exit_code()` → `both_passed() == true` | Normal success |
| Lane A fails | 1 | `determine_exit_code()` → `!both_passed()` | BitNet.cpp divergence |
| Lane B fails | 1 | `determine_exit_code()` → `!both_passed()` | llama.cpp divergence |
| Both fail | 1 | `determine_exit_code()` → `!both_passed()` | Both backends diverged |
| Token parity violation | 2 | `validate_tokenizer_parity()` → `bail!()` | Rust vs C++ token mismatch |
| Preflight failure | 2 | `preflight_backend_libs()` → `bail!()` | Backend unavailable |
| Invalid arguments | 2 | Arg parsing → `clap::Error` | Missing required args |
| FFI not available | 2 | `run()` → `anyhow::bail!()` | Missing --features crossval-all |

**Implementation**:
```rust
// Exit code determination (STEP 7: Unified Summary)
pub fn determine_exit_code(lane_a: &LaneResult, lane_b: &LaneResult) -> i32 {
    if lane_a.passed && lane_b.passed {
        0  // Both passed
    } else {
        1  // Either or both failed
    }
}

// Usage error handling (STEP 2: Token parity)
pub fn validate_tokenizer_parity(
    rust_tokens: &[u32],
    cpp_tokens: &[u32],
    backend_name: &str,
) -> Result<()> {
    if rust_tokens.len() != cpp_tokens.len() {
        anyhow::bail!(
            "Token parity mismatch for {}: Rust {} tokens vs C++ {} tokens",
            backend_name,
            rust_tokens.len(),
            cpp_tokens.len()
        );
    }
    // ... token-by-token validation ...
    Ok(())
}
```

**Main Handler** (xtask/src/main.rs):
```rust
#[cfg(feature = "crossval-all")]
fn parity_both_cmd(...) -> Result<()> {
    // Convert CLI args to ParityBothArgs
    let args = ParityBothArgs { /* ... */ };

    // Run dual-lane orchestration
    parity_both::run(&args)?;  // Returns Err for exit code 2 (usage errors)

    // Load receipts and determine exit code
    let lane_a = load_receipt_as_lane_result(CppBackend::BitNet, &receipt_bitnet_path)?;
    let lane_b = load_receipt_as_lane_result(CppBackend::Llama, &receipt_llama_path)?;

    let exit_code = parity_both::determine_exit_code(&lane_a, &lane_b);

    if exit_code != 0 {
        std::process::exit(exit_code);
    }

    Ok(())
}
```

**Success Criteria**:
- Exit code 0: Both lanes passed (cosine similarity ≥ threshold)
- Exit code 1: Either lane failed (divergence detected)
- Exit code 2: Token parity violation, preflight failure, or invalid args
- Clear error messages for exit code 2 scenarios
- Process::exit() called explicitly for non-zero exits

**Test Coverage**:
```rust
#[test]
fn test_exit_code_both_pass() {
    // Given: Both lanes passed parity checks
    // When: Run parity-both
    // Then: Exit code 0
}

#[test]
fn test_exit_code_lane_a_fail() {
    // Given: BitNet.cpp diverges, llama.cpp passes
    // When: Run parity-both
    // Then: Exit code 1
}

#[test]
fn test_exit_code_token_parity_violation() {
    // Given: Rust and C++ produce different token counts
    // When: Run parity-both
    // Then: Exit code 2 with "Token parity mismatch"
}

#[test]
fn test_exit_code_preflight_failure() {
    // Given: BitNet.cpp unavailable, --no-repair
    // When: Run parity-both
    // Then: Exit code 2 with "Backend 'bitnet' libraries NOT FOUND"
}
```

---

### AC5: Dual Receipt Naming (receipt_bitnet.json, receipt_llama.json)

**Requirement**: Generate separate receipts for each backend with predictable naming.

**Naming Convention**:
```
{out_dir}/receipt_bitnet.json   # Lane A: BitNet.cpp backend
{out_dir}/receipt_llama.json    # Lane B: llama.cpp backend
```

**Implementation**:
```rust
// Lane A (BitNet.cpp) - STEP 6A
let receipt_bitnet_path = out_dir.join("receipt_bitnet.json");
receipt_a.write_to_file(&receipt_bitnet_path)?;

// Lane B (llama.cpp) - STEP 6B
let receipt_llama_path = out_dir.join("receipt_llama.json");
receipt_b.write_to_file(&receipt_llama_path)?;
```

**Success Criteria**:
- Receipt files named consistently (receipt_{backend}.json)
- Files written atomically to avoid partial receipts
- Output directory created if it doesn't exist
- Clear error message if write fails (e.g., permissions)

**Test Coverage**:
```rust
#[test]
fn test_dual_receipt_naming() {
    // Given: Successful parity-both run
    // When: Check output directory
    // Then: Files exist: receipt_bitnet.json, receipt_llama.json
}

#[test]
fn test_receipt_directory_creation() {
    // Given: Output directory does not exist
    // When: Run parity-both --out-dir /tmp/nonexistent
    // Then: Directory created, receipts written successfully
}
```

---

### AC6: Shared Rust Inference (Reused for Both Lanes)

**Requirement**: Evaluate Rust logits once and reuse for both backend comparisons.

**Implementation** (STEP 3):
```rust
// Shared Rust logits evaluation (once)
let rust_logits: Vec<Vec<f32>> = bitnet_inference::parity::eval_logits_all_positions(
    &model,
    &rust_tokens,
)?;

// Lane A: Compare Rust logits vs BitNet.cpp logits
let cpp_logits_a = cpp_session_bitnet.evaluate(&cpp_tokens_bitnet)?;
compare_per_position_logits(&rust_logits, &cpp_logits_a)?;

// Lane B: Compare Rust logits vs llama.cpp logits (reuse rust_logits)
let cpp_logits_b = cpp_session_llama.evaluate(&cpp_tokens_llama)?;
compare_per_position_logits(&rust_logits, &cpp_logits_b)?;
```

**Performance Impact**:
- Before: 2 × Rust inference (once per lane) ≈ 60s
- After: 1 × Rust inference (shared) ≈ 30s
- Savings: ~50% reduction in execution time

**Success Criteria**:
- Rust logits evaluated exactly once (not per-lane)
- Logits matrix reused for both BitNet.cpp and llama.cpp comparisons
- No duplication of GGUF model loading
- Verbose output shows "Shared Rust inference" step

**Test Coverage**:
```rust
#[test]
fn test_shared_rust_inference_called_once() {
    // Given: Parity-both run with --verbose
    // When: Capture stdout
    // Then: "Shared Rust inference" appears exactly once
}
```

---

### AC7: Per-Position Metrics (MSE, L2, KL Divergence, TopK Agreement)

**Requirement**: Compute and store detailed metrics for each token position.

**PositionMetrics Schema**:
```rust
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct PositionMetrics {
    /// Token position (0-indexed)
    pub pos: usize,

    /// Mean squared error between Rust and C++ logits
    pub mse: f32,

    /// Maximum absolute difference across all logits at this position
    pub max_abs: f32,

    /// Kullback-Leibler divergence (optional - requires softmax normalization)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub kl: Option<f32>,

    /// Top-K agreement (fraction of top-K tokens that match, optional)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub topk_agree: Option<f32>,

    /// Top-5 token IDs from Rust logits (highest to lowest)
    pub top5_rust: Vec<usize>,

    /// Top-5 token IDs from C++ logits (highest to lowest)
    pub top5_cpp: Vec<usize>,
}
```

**Metrics Computation**:
```rust
// Parse metrics parameter (AC7)
let metrics_set: HashSet<&str> = metrics.split(',').map(|s| s.trim()).collect();
let compute_mse = metrics_set.contains("mse");
let compute_kl = metrics_set.contains("kl");
let compute_topk = metrics_set.contains("topk");

// Per-position loop (STEP 5A/5B)
for pos in 0..rust_logits.len() {
    let rust_vec = &rust_logits[pos];
    let cpp_vec = &cpp_logits[pos];

    let mse = if compute_mse { mse(rust_vec, cpp_vec) } else { 0.0 };
    let kl = if compute_kl { Some(kl_divergence(rust_vec, cpp_vec)) } else { None };
    let topk_agree = if compute_topk { Some(topk_agreement(rust_vec, cpp_vec, 5)) } else { None };

    receipt.add_position(PositionMetrics {
        pos,
        mse,
        max_abs: max_abs_difference(rust_vec, cpp_vec),
        kl,
        topk_agree,
        top5_rust: top_k_indices(rust_vec, 5),
        top5_cpp: top_k_indices(cpp_vec, 5),
    });
}
```

**Success Criteria**:
- MSE always computed (core metric)
- KL divergence computed if `--metrics` includes "kl"
- TopK agreement computed if `--metrics` includes "topk"
- Top-5 token IDs always captured (for debugging)
- Metrics customization functional (not just parsed)

**Test Coverage**:
```rust
#[test]
fn test_metrics_mse_only() {
    // Given: --metrics mse
    // When: Load receipt
    // Then: PositionMetrics.mse populated, kl=None, topk_agree=None
}

#[test]
fn test_metrics_all() {
    // Given: --metrics mse,kl,topk
    // When: Load receipt
    // Then: All metrics populated (mse, kl, topk_agree)
}
```

---

### AC8: Summary with First Divergence Position

**Requirement**: Compute aggregate statistics including first divergence position.

**Summary Schema**:
```rust
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Summary {
    /// True if all positions passed quality thresholds
    pub all_passed: bool,

    /// First position where divergence was detected (None if all passed)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub first_divergence: Option<usize>,

    /// Mean MSE across all positions
    pub mean_mse: f32,

    /// Mean KL divergence across all positions (optional)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mean_kl: Option<f32>,
}
```

**Finalization Logic** (crossval/src/receipt.rs):
```rust
pub fn finalize(&mut self) {
    self.positions = self.rows.len();

    if self.rows.is_empty() {
        return;
    }

    // Compute mean MSE
    let total_mse: f32 = self.rows.iter().map(|r| r.mse).sum();
    self.summary.mean_mse = total_mse / self.rows.len() as f32;

    // Compute mean KL (if available for all positions)
    let kl_values: Vec<f32> = self.rows.iter().filter_map(|r| r.kl).collect();
    if kl_values.len() == self.rows.len() {
        let total_kl: f32 = kl_values.iter().sum();
        self.summary.mean_kl = Some(total_kl / kl_values.len() as f32);
    }

    // Determine first divergence based on MSE threshold
    self.summary.first_divergence = self.rows.iter()
        .position(|r| r.mse > self.thresholds.mse);

    // Update all_passed flag
    self.summary.all_passed = self.summary.first_divergence.is_none();
}
```

**Unified Summary Output** (STEP 7):
```
Parity-Both Cross-Validation Summary
════════════════════════════════════════════════════════════
Lane A: BitNet.cpp
────────────────────────────────────────────────────────────
Status:           ✓ PASSED
Mean MSE:         0.000013
Mean Cosine Sim:  0.99999
First Divergence: None
Receipt:          ci/parity/receipt_bitnet.json

Lane B: llama.cpp
────────────────────────────────────────────────────────────
Status:           ✗ FAILED
Mean MSE:         0.000324
Mean Cosine Sim:  0.99650
First Divergence: Position 2
Receipt:          ci/parity/receipt_llama.json

Overall Status
────────────────────────────────────────────────────────────
Result:           DIVERGENCE DETECTED (Lane B failed)
Exit Code:        1
```

**Success Criteria**:
- `all_passed` computed based on MSE threshold
- `first_divergence` set to first position exceeding threshold
- `mean_mse` computed across all positions
- `mean_kl` computed if KL available for all positions
- Unified summary displays both lanes side-by-side

**Test Coverage**:
```rust
#[test]
fn test_summary_all_passed() {
    // Given: All positions MSE < threshold
    // When: Finalize receipt
    // Then: all_passed=true, first_divergence=None
}

#[test]
fn test_summary_first_divergence() {
    // Given: Position 2 MSE > threshold
    // When: Finalize receipt
    // Then: all_passed=false, first_divergence=Some(2)
}
```

---

### AC9: CLI Integration (main.rs Wiring)

**Requirement**: Wire parity-both command into xtask main.rs dispatcher.

**Command Registration** (xtask/src/main.rs):
```rust
#[cfg(feature = "crossval-all")]
#[command(name = "parity-both")]
ParityBoth {
    // Input paths
    #[arg(long)]
    model_gguf: PathBuf,

    #[arg(long)]
    tokenizer: PathBuf,

    // Inference parameters
    #[arg(long, default_value = "What is 2+2?")]
    prompt: String,

    #[arg(long, default_value_t = 4)]
    max_tokens: usize,

    #[arg(long, default_value_t = 0.999)]
    cos_tol: f64,

    // Output configuration
    #[arg(long, default_value = ".parity")]
    out_dir: PathBuf,

    #[arg(long, default_value = "text")]
    format: String,

    // Template and prompting
    #[arg(long, default_value = "auto", value_enum)]
    prompt_template: PromptTemplateArg,

    #[arg(long)]
    system_prompt: Option<String>,

    // Auto-repair control
    #[arg(long)]
    no_repair: bool,

    // Debugging flags
    #[arg(long, short)]
    verbose: bool,

    #[arg(long)]
    dump_ids: bool,

    #[arg(long)]
    dump_cpp_ids: bool,

    // Metrics configuration
    #[arg(long, default_value = "mse")]
    metrics: String,
},
```

**Handler Function** (xtask/src/main.rs):
```rust
#[cfg(feature = "crossval-all")]
#[allow(clippy::too_many_arguments)]
fn parity_both_cmd(
    model_gguf: &Path,
    tokenizer: &Path,
    prompt: &str,
    max_tokens: usize,
    cos_tol: f64,
    out_dir: &Path,
    format: &str,
    prompt_template: PromptTemplateArg,
    system_prompt: Option<&str>,
    auto_repair: bool,
    verbose: bool,
    dump_ids: bool,
    dump_cpp_ids: bool,
    metrics: &str,
) -> Result<()> {
    let args = ParityBothArgs {
        model_gguf: model_gguf.to_path_buf(),
        tokenizer: tokenizer.to_path_buf(),
        prompt: prompt.to_string(),
        max_tokens,
        cos_tol,
        out_dir: out_dir.to_path_buf(),
        format: format.to_string(),
        prompt_template,
        system_prompt: system_prompt.map(|s| s.to_string()),
        auto_repair,
        verbose,
        dump_ids,
        dump_cpp_ids,
        metrics: metrics.to_string(),
    };

    parity_both::run(&args)
}
```

**Success Criteria**:
- Command registered with #[cfg(feature = "crossval-all")]
- Handler function delegates to parity_both::run()
- Exit codes propagated correctly (0, 1, 2)
- Feature gate prevents compilation errors without crossval-all

**Test Coverage**:
```rust
#[test]
fn test_cli_registration() {
    // Given: Build xtask with --features crossval-all
    // When: Run --help
    // Then: "parity-both" command listed
}

#[test]
fn test_cli_missing_model_exit_2() {
    // Given: Omit --model-gguf
    // When: Run parity-both
    // Then: Exit code 2 with clap usage error
}
```

---

### AC10: Test Coverage (Dual-Lane, Exit Codes, Tokenizer Consistency)

**Requirement**: Comprehensive test suite covering all acceptance criteria.

**Test Categories**:

1. **Preflight Integration Tests** (8 tests):
   - `test_preflight_both_backends_success`
   - `test_preflight_bitnet_missing_exit_2`
   - `test_preflight_llama_missing_exit_2`
   - `test_auto_repair_bitnet_success`
   - `test_auto_repair_disabled_exit_2`
   - `test_auto_repair_exhausts_retries`
   - `test_preflight_verbose_diagnostics`
   - `test_preflight_both_missing_exit_2`

2. **TokenizerAuthority Tests** (6 tests):
   - `test_tokenizer_authority_gguf_embedded`
   - `test_tokenizer_authority_external_json`
   - `test_tokenizer_authority_auto_discovered`
   - `test_tokenizer_authority_identical_across_lanes`
   - `test_tokenizer_file_hash_determinism`
   - `test_tokenizer_config_hash_consistency`

3. **Exit Code Tests** (6 tests):
   - `test_exit_code_both_pass`
   - `test_exit_code_lane_a_fail`
   - `test_exit_code_lane_b_fail`
   - `test_exit_code_both_fail`
   - `test_exit_code_token_parity_violation`
   - `test_exit_code_preflight_failure`

4. **Dual Receipt Tests** (5 tests):
   - `test_dual_receipt_naming`
   - `test_receipt_directory_creation`
   - `test_receipt_schema_v2_backward_compat`
   - `test_receipt_tokenizer_authority_serialization`
   - `test_receipt_summary_finalization`

5. **Metrics Tests** (4 tests):
   - `test_metrics_mse_only`
   - `test_metrics_all`
   - `test_metrics_customization_functional`
   - `test_per_position_metrics_accuracy`

6. **Shared Inference Tests** (3 tests):
   - `test_shared_rust_inference_called_once`
   - `test_rust_logits_reused_both_lanes`
   - `test_performance_optimization_timing`

7. **CLI Integration Tests** (3 tests):
   - `test_cli_registration`
   - `test_cli_missing_model_exit_2`
   - `test_cli_help_output`

8. **Token Parity Tests** (4 tests):
   - `test_token_parity_validation_success`
   - `test_token_parity_length_mismatch`
   - `test_token_parity_id_divergence`
   - `test_token_parity_fail_fast_before_inference`

9. **Summary Output Tests** (3 tests):
   - `test_unified_summary_text_format`
   - `test_unified_summary_json_format`
   - `test_summary_first_divergence_position`

10. **End-to-End Integration Tests** (4 tests):
    - `test_parity_both_full_workflow_both_pass`
    - `test_parity_both_full_workflow_one_fail`
    - `test_parity_both_with_auto_repair`
    - `test_parity_both_with_custom_metrics`

**Total Test Count**: 54 tests covering all acceptance criteria

**Test Infrastructure**:
```rust
// Location: xtask/tests/parity_both_integration_tests.rs

#[cfg(feature = "crossval-all")]
mod parity_both_integration_tests {
    use xtask::crossval::parity_both::{self, ParityBothArgs};
    use crossval::receipt::{ParityReceipt, TokenizerAuthority, TokenizerSource};
    use serial_test::serial;
    use tests::helpers::env_guard::EnvGuard;

    // AC1: Preflight integration tests
    // AC2: Auto-repair integration tests
    // AC3: TokenizerAuthority tests
    // AC4: Exit code tests
    // AC5: Dual receipt tests
    // AC6: Shared inference tests
    // AC7: Metrics tests
    // AC8: Summary tests
    // AC9: CLI integration tests
    // AC10: End-to-end tests
}
```

**Success Criteria**:
- All 54 tests pass with `cargo test -p xtask --features crossval-all`
- Tests use `#[serial(bitnet_env)]` for environment isolation
- EnvGuard pattern prevents test pollution
- Tests validate both success and failure paths
- Integration tests use temporary directories for receipts

---

## 4. TokenizerAuthority Schema

### 4.1 Structure Definition

```rust
/// Tokenizer authority metadata for receipt reproducibility (AC4-AC6)
///
/// Specification: docs/specs/parity-both-preflight-tokenizer.md#AC4
///
/// This structure captures complete tokenizer provenance to ensure
/// receipt reproducibility and enable tokenizer parity validation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenizerAuthority {
    /// Tokenizer source: GgufEmbedded, JsonFile, or AutoDiscovered
    pub source: TokenizerSource,

    /// SHA256 hash of tokenizer.json file (if external)
    ///
    /// This is None for GGUF-embedded tokenizers.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub file_hash: Option<String>,

    /// SHA256 hash of effective tokenizer config (canonical representation)
    ///
    /// This hash is computed from the tokenizer's configuration fingerprint,
    /// ensuring consistency across different instances.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub config_hash: Option<String>,
}
```

### 4.2 TokenizerSource Enum

```rust
/// Tokenizer source type (AC5)
///
/// Specification: docs/specs/parity-both-preflight-tokenizer.md#AC5
///
/// Represents where the tokenizer configuration originated from.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum TokenizerSource {
    /// Tokenizer embedded in GGUF file
    GgufEmbedded,

    /// External tokenizer.json file (explicitly provided)
    JsonFile(PathBuf),

    /// Auto-discovered tokenizer from model directory
    AutoDiscovered(PathBuf),
}
```

### 4.3 Field Requirements

| Field | Type | Required | Nullable | Purpose | Notes |
|-------|------|----------|----------|---------|-------|
| `source` | `TokenizerSource` | Yes | No | Where tokenizer came from | One of three variants |
| `file_hash` | `Option<String>` | No | Yes | SHA256 of tokenizer.json | None for GGUF-embedded |
| `config_hash` | `Option<String>` | No | Yes | SHA256 of canonical config | Computed from vocab |

### 4.4 Field Presence Rules

```rust
// For GGUF-embedded tokenizer:
TokenizerAuthority {
    source: TokenizerSource::GgufEmbedded,
    file_hash: None,           // ← Always None (no separate file)
    config_hash: Some("..."),  // ← Always present
}

// For external tokenizer.json:
TokenizerAuthority {
    source: TokenizerSource::JsonFile(path),
    file_hash: Some("..."),   // ← Always present (file exists)
    config_hash: Some("..."), // ← Always present
}

// For auto-discovered tokenizer:
TokenizerAuthority {
    source: TokenizerSource::AutoDiscovered(path),
    file_hash: Some("..."),   // ← Always present (file exists)
    config_hash: Some("..."), // ← Always present
}
```

### 4.5 JSON Serialization Examples

**GGUF-Embedded Tokenizer**:
```json
{
  "source": "gguf_embedded",
  "config_hash": "a1b2c3d4e5f6789012345678901234567890abcdefabcdefabcdefabcdefabc0"
}
```

**External Tokenizer.json**:
```json
{
  "source": "json_file",
  "path": "models/tokenizer.json",
  "file_hash": "6f3ef9d7a3c2b1e0d4f5a8c9b0e1d2f3a4b5c6d7e8f9a0b1c2d3e4f5a6b7c8d9",
  "config_hash": "a1b2c3d4e5f6789012345678901234567890abcdefabcdefabcdefabcdefabc0"
}
```

### 4.6 Hash Functions

**File Hash Computation**:
```rust
use sha2::{Digest, Sha256};

/// Compute SHA256 hash of tokenizer.json file (AC6)
pub fn compute_tokenizer_file_hash(tokenizer_path: &Path) -> anyhow::Result<String> {
    let contents = std::fs::read(tokenizer_path)
        .with_context(|| format!("Failed to read tokenizer file: {}", tokenizer_path.display()))?;

    let mut hasher = Sha256::new();
    hasher.update(&contents);
    Ok(format!("{:x}", hasher.finalize()))
}
```

**Config Hash Computation**:
```rust
/// Compute SHA256 hash of tokenizer config (AC6)
pub fn compute_tokenizer_config_hash(
    tokenizer: &dyn bitnet_tokenizers::Tokenizer,
) -> anyhow::Result<String> {
    use sha2::{Digest, Sha256};

    // Create canonical representation from vocab sizes
    let config_repr = serde_json::json!({
        "vocab_size": tokenizer.vocab_size(),
        "real_vocab_size": tokenizer.real_vocab_size(),
    });
    let canonical_json =
        serde_json::to_string(&config_repr).context("Failed to serialize tokenizer config")?;

    let mut hasher = Sha256::new();
    hasher.update(canonical_json.as_bytes());
    Ok(format!("{:x}", hasher.finalize()))
}
```

---

## 5. Receipt Schema v2.0.0 (Backward-Compatible Extension)

### 5.1 Extended Receipt Structure

```rust
/// Parity receipt - structured output for cross-validation comparison
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ParityReceipt {
    // V1 fields (always present)
    pub version: u32,
    pub timestamp: String,
    pub model: String,
    pub backend: String,
    pub prompt: String,
    pub positions: usize,
    pub thresholds: Thresholds,
    pub rows: Vec<PositionMetrics>,
    pub summary: Summary,

    // V2 fields (optional for backward compatibility)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tokenizer_authority: Option<TokenizerAuthority>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_template: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub determinism_seed: Option<u64>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_sha256: Option<String>,
}
```

### 5.2 Backward Compatibility Guarantees

**V1 → V2 Deserialization**:
```rust
// V1 receipts deserialize correctly into v2 struct
let json_v1 = r#"{ "version": 1, "model": "...", /* v1 fields */ }"#;
let receipt: ParityReceipt = serde_json::from_str(json_v1)?;
// v2 optional fields are None
assert_eq!(receipt.tokenizer_authority, None);
```

**V2 → V1 Serialization**:
```rust
// V2 fields omitted in JSON output if None (skip_serializing_if)
let receipt = ParityReceipt::new("model.gguf", "bitnet", "test");
let json_compact = receipt.to_json()?;
// No tokenizer_authority, prompt_template, etc. keys in JSON
```

### 5.3 Version Inference

Receipt version can be inferred from field presence:
- **v1.0.0**: No optional v2 fields present
- **v2.0.0**: At least one v2 optional field present

---

## 6. API Contracts

### 6.1 Public API Surface

**parity_both Module** (xtask/src/crossval/parity_both.rs):
```rust
/// Run dual-lane parity comparison (AC1-AC10)
pub fn run(args: &ParityBothArgs) -> Result<()>;

/// Compute exit code from lane results (AC4)
pub fn determine_exit_code(lane_a: &LaneResult, lane_b: &LaneResult) -> i32;

/// Check if both lanes passed (AC4)
pub fn both_passed(lane_a: &LaneResult, lane_b: &LaneResult) -> bool;

/// Get overall status string (AC8)
pub fn overall_status(lane_a: &LaneResult, lane_b: &LaneResult) -> &'static str;

/// Print unified summary (text or JSON) (AC8)
pub fn print_unified_summary(
    lane_a: &LaneResult,
    lane_b: &LaneResult,
    format: &str,
    verbose: bool,
) -> Result<()>;

/// Comparison helpers (AC7)
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f64;
pub fn l2_distance(a: &[f32], b: &[f32]) -> f64;
pub fn mse(a: &[f32], b: &[f32]) -> f64;
```

### 6.2 Function Signatures

**Preflight Integration**:
```rust
// Location: crossval/src/preflight.rs
pub fn preflight_backend_libs(backend: CppBackend, verbose: bool) -> Result<()>;
pub fn preflight_with_auto_repair(
    backend: CppBackend,
    repair_mode: RepairMode,
    verbose: bool,
) -> Result<()>;
```

**TokenizerAuthority Functions**:
```rust
// Location: crossval/src/receipt.rs
pub fn compute_tokenizer_file_hash(tokenizer_path: &Path) -> anyhow::Result<String>;
pub fn compute_tokenizer_config_hash(
    tokenizer: &dyn bitnet_tokenizers::Tokenizer,
) -> anyhow::Result<String>;
pub fn detect_tokenizer_source(
    tokenizer_path: &Path,
    model_path: &Path,
) -> Result<TokenizerSource>;
```

**Token Parity Validation**:
```rust
// Location: xtask/src/crossval/parity_both.rs
pub fn validate_tokenizer_parity(
    rust_tokens: &[u32],
    cpp_tokens: &[u32],
    backend_name: &str,
) -> Result<()>;
```

**Receipt Builder API**:
```rust
// Location: crossval/src/receipt.rs
impl ParityReceipt {
    pub fn new(model: &str, backend: &str, prompt: &str) -> Self;
    pub fn set_thresholds(&mut self, thresholds: Thresholds);
    pub fn set_tokenizer_authority(&mut self, authority: TokenizerAuthority);
    pub fn set_prompt_template(&mut self, template: String);
    pub fn add_position(&mut self, metrics: PositionMetrics);
    pub fn finalize(&mut self);
    pub fn to_json(&self) -> Result<String, serde_json::Error>;
    pub fn write_to_file(&self, path: &Path) -> anyhow::Result<()>;
}
```

---

## 7. Test Strategy

### 7.1 Unit Tests (crossval/src/)

**receipt.rs** (12 tests):
- TokenizerAuthority construction
- Hash determinism
- Receipt builder API
- Schema backward compatibility
- Serialization/deserialization

**parity_both.rs** (25 tests):
- Exit code logic
- Comparison helpers (cosine, L2, MSE)
- Lane result construction
- Summary formatting

**preflight.rs** (8 tests):
- Backend detection
- Auto-repair flow
- RepairMode semantics
- Retry logic

### 7.2 Integration Tests (xtask/tests/)

**parity_both_integration_tests.rs** (54 tests):
- AC1: Preflight integration (8 tests)
- AC2: Auto-repair (6 tests)
- AC3: TokenizerAuthority (6 tests)
- AC4: Exit codes (6 tests)
- AC5: Dual receipts (5 tests)
- AC6: Shared inference (3 tests)
- AC7: Metrics (4 tests)
- AC8: Summary (3 tests)
- AC9: CLI integration (3 tests)
- AC10: End-to-end (4 tests)

### 7.3 Test Execution

```bash
# Unit tests
cargo test -p crossval --lib --no-default-features --features cpu

# Integration tests
cargo test -p xtask --test parity_both_integration_tests --features crossval-all

# All tests
cargo test --workspace --features crossval-all

# With nextest (recommended)
cargo nextest run --workspace --features crossval-all
```

---

## 8. References

### 8.1 Analysis Artifacts

- **/tmp/parity_both_state_analysis.md**: Current implementation state (924 lines)
- **/tmp/receipt_tokenizer_analysis.md**: Receipt schema and TokenizerAuthority design (1002 lines)

### 8.2 Related Specifications

- **docs/specs/parity-both-command.md**: Baseline parity-both spec (v1.1)
- **docs/specs/preflight-auto-repair.md**: Auto-repair and retry logic
- **docs/explanation/dual-backend-crossval.md**: Dual-backend architecture

### 8.3 Code Locations

| Component | File | Lines | Purpose |
|-----------|------|-------|---------|
| CLI Registration | xtask/src/main.rs | 559-617 | ParityBoth command struct |
| Handler | xtask/src/main.rs | 3669-3710 | CLI dispatcher |
| Orchestration | xtask/src/crossval/parity_both.rs | 397-540 | Dual-lane flow |
| Receipt Schema | crossval/src/receipt.rs | 78-127 | ParityReceipt structure |
| TokenizerAuthority | crossval/src/receipt.rs | 38-76 | Tokenizer provenance |
| Hash Functions | crossval/src/receipt.rs | 313-345 | SHA256 computation |
| Preflight | crossval/src/preflight.rs | 366-512 | Backend availability |
| Auto-Repair | crossval/src/preflight.rs | 1143-1185 | Repair and retry |

---

## 9. Success Criteria Summary

| AC | Requirement | Implementation Status | Test Coverage |
|----|-------------|----------------------|---------------|
| AC1 | Preflight both backends | ✓ Implemented | 8 tests |
| AC2 | Auto-repair integration | ⚠ Partial (wiring needed) | 6 tests |
| AC3 | TokenizerAuthority in receipts | ✓ Implemented | 6 tests |
| AC4 | Exit code logic (0/1/2) | ⚠ Partial (exit 2 missing) | 6 tests |
| AC5 | Dual receipt naming | ✓ Implemented | 5 tests |
| AC6 | Shared Rust inference | ✓ Implemented | 3 tests |
| AC7 | Per-position metrics | ⚠ Partial (metrics unused) | 4 tests |
| AC8 | Summary with divergence | ✓ Implemented | 3 tests |
| AC9 | CLI integration | ✓ Implemented | 3 tests |
| AC10 | Test coverage | ⚠ Partial (54 tests defined) | 54 tests |

**Legend**:
- ✓ Implemented: Feature complete and functional
- ⚠ Partial: Infrastructure exists but integration incomplete
- Total Tests: 54 (covering all acceptance criteria)

---

## 10. Implementation Roadmap

### Phase 1: Preflight Integration (AC1, AC2)
1. Wire `auto_repair` parameter into preflight step
2. Replace `preflight_backend_libs()` with `preflight_with_auto_repair()`
3. Add RepairMode parameter handling
4. Test auto-repair flow with retry logic

### Phase 2: Exit Code Standardization (AC4)
1. Add exit code 2 handling for token parity violations
2. Add exit code 2 for preflight failures
3. Add exit code 2 for usage errors
4. Update main handler to call `std::process::exit()`

### Phase 3: Metrics Customization (AC7)
1. Parse metrics parameter into HashSet
2. Pass metrics flags to `compare_per_position_logits()`
3. Conditionally compute KL divergence and TopK agreement
4. Update receipt.add_position() calls

### Phase 4: Test Suite Implementation (AC10)
1. Implement 54 integration tests in xtask/tests/
2. Add fixtures for test models and tokenizers
3. Add EnvGuard for environment isolation
4. Validate all tests pass with nextest

### Phase 5: Documentation and Examples
1. Update CLAUDE.md with parity-both examples
2. Add quickstart guide for dual-lane validation
3. Document TokenizerAuthority schema
4. Add troubleshooting guide for common errors

---

**Specification Complete**: This document provides implementation-ready guidance for comprehensive parity-both integration with preflight auto-repair and tokenizer authority tracking.
