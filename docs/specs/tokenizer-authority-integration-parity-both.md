# Technical Specification: TokenizerAuthority SHA256 Integration for Parity-Both

**Status**: Draft
**Created**: 2025-10-27
**Author**: BitNet.rs Generative Adapter (Spec Gate)
**Related**: `docs/specs/parity-both-preflight-tokenizer-integration.md`

---

## 1. Overview

This specification defines the integration of TokenizerAuthority SHA256 computation into the `parity-both` dual-lane cross-validation workflow. The integration ensures that both Rust and C++ lanes use identical tokenizer configurations by computing SHA256 hashes of tokenizer files and validating consistency across lanes.

### 1.1 Current State Analysis

**Infrastructure Present (crossval/src/receipt.rs)**:
- ✅ `TokenizerAuthority` struct with `file_hash` and `config_hash` fields (lines 54-82)
- ✅ `compute_tokenizer_file_hash()` helper for file-based SHA256 computation (lines 336-350)
- ✅ `compute_tokenizer_config_hash_from_tokenizer()` for config-based SHA256 (lines 377-398)
- ✅ `detect_tokenizer_source()` for source detection (lines 400-412)
- ✅ `validate_tokenizer_consistency()` for cross-lane validation (lines 480-503)
- ✅ `ParityReceipt::set_tokenizer_authority()` setter method (lines 254-259)

**Gap Identified (xtask/src/crossval/parity_both.rs)**:
- ❌ `compute_tokenizer_file_hash()` **never called** during receipt generation (lines 636-689)
- ❌ `set_tokenizer_authority()` **never called** before `finalize()` (line 682)
- ❌ No cross-lane tokenizer consistency validation after dual-lane execution (lines 531-543)
- ❌ Tokenizer hash not included in summary output (lines 227-335)

### 1.2 Requirements Summary

| Requirement | Current Status | Gap |
|-------------|----------------|-----|
| AC1: Compute SHA256 hash once before dual-lane execution | ❌ Missing | Hash computation not called |
| AC2: Both receipts have matching `TokenizerAuthority.sha256_hash` | ❌ Missing | Authority not populated |
| AC3: Validate tokenizer consistency after both lanes complete | ❌ Missing | No validation call |
| AC4: Exit code 2 if tokenizer hashes differ | ❌ Missing | No consistency check |
| AC5: Summary output includes tokenizer hash | ❌ Missing | Not in summary display |
| AC6: Receipts serialize `TokenizerAuthority` properly | ⚠️ Partial | Schema exists, not populated |

---

## 2. Architecture Design

### 2.1 Integration Points

The integration occurs at four key phases in `run_dual_lanes_and_summarize()`:

```text
┌────────────────────────────────────────────────────────────────┐
│ PHASE 1: Preflight Checks (lines 428-446)                     │
│ • Backend availability validation                             │
└────────────────────────────────────────────────────────────────┘
                             ↓
┌────────────────────────────────────────────────────────────────┐
│ PHASE 2: Shared Setup (lines 448-490)                         │
│ • Template processing                                          │
│ • Rust tokenization                                            │
│ • ➕ NEW: TokenizerAuthority computation (AC1)                 │
│   - compute_tokenizer_file_hash()                              │
│   - compute_tokenizer_config_hash_from_tokenizer()             │
│   - detect_tokenizer_source()                                  │
└────────────────────────────────────────────────────────────────┘
                             ↓
┌──────────────────────┬─────────────────────────────────────────┐
│ PHASE 3: Dual Lanes  │ PHASE 4: Dual Lanes                     │
│ Lane A: BitNet.cpp   │ Lane B: llama.cpp                       │
│ • C++ tokenization   │ • C++ tokenization                      │
│ • Logits comparison  │ • Logits comparison                     │
│ • Receipt generation │ • Receipt generation                    │
│ • ➕ NEW: Populate   │ • ➕ NEW: Populate                       │
│   TokenizerAuthority │   TokenizerAuthority                    │
│   (AC2)              │   (AC2)                                 │
└──────────────────────┴─────────────────────────────────────────┘
                             ↓
┌────────────────────────────────────────────────────────────────┐
│ PHASE 5: Summary and Exit Code (lines 531-543)                │
│ • Load receipts                                                │
│ • ➕ NEW: Validate tokenizer consistency (AC3, AC4)            │
│ • Print summary with tokenizer hash (AC5)                     │
│ • Exit code determination                                      │
└────────────────────────────────────────────────────────────────┘
```

### 2.2 Data Flow

```text
tokenizer.json file
         ↓
    [compute_tokenizer_file_hash()]
         ↓
    file_hash: String (64-char hex SHA256)
         ↓
    ┌─────────────────────────┐
    │ TokenizerAuthority      │
    │ ├─ source: External     │
    │ ├─ path: tokenizer.json │
    │ ├─ file_hash: Some(...)  │
    │ ├─ config_hash: ...     │
    │ └─ token_count: 128000  │
    └─────────────────────────┘
         ↓
    [shared across lanes]
         ↓
    ┌────────────────┐      ┌────────────────┐
    │ Lane A Receipt │      │ Lane B Receipt │
    │ tokenizer_auth │      │ tokenizer_auth │
    └────────────────┘      └────────────────┘
         ↓                         ↓
    [validate_tokenizer_consistency()]
         ↓
    ✓ Exit code 0 (match)
    ✗ Exit code 2 (mismatch)
```

---

## 3. Implementation Specification

### 3.1 Phase 2: Shared TokenizerAuthority Computation (AC1)

**Location**: `xtask/src/crossval/parity_both.rs::run_dual_lanes_and_summarize()`
**Insert After**: Line 471 (after Rust tokenization)
**Before**: Line 473 (Rust logits evaluation)

```rust
// STEP 2.5: Compute shared TokenizerAuthority (AC1)
if verbose {
    eprintln!("\n⚙ Shared: Computing tokenizer authority...");
}

let tokenizer_authority = {
    use bitnet_crossval::receipt::{
        compute_tokenizer_file_hash,
        compute_tokenizer_config_hash_from_tokenizer,
        detect_tokenizer_source,
        TokenizerAuthority,
    };

    let source = detect_tokenizer_source(tokenizer);
    let file_hash = compute_tokenizer_file_hash(tokenizer)
        .context("Failed to compute tokenizer file hash")?;
    let config_hash = compute_tokenizer_config_hash_from_tokenizer(&*tokenizer_obj)
        .context("Failed to compute tokenizer config hash")?;
    let token_count = rust_tokens.len();

    if verbose {
        eprintln!("  Source: {:?}", source);
        eprintln!("  File hash: {}", &file_hash[..16]); // First 16 chars
        eprintln!("  Config hash: {}", &config_hash[..16]);
        eprintln!("  Token count: {}", token_count);
    }

    TokenizerAuthority {
        source,
        path: tokenizer.to_string_lossy().to_string(),
        file_hash: Some(file_hash),
        config_hash,
        token_count,
    }
};
```

**Rationale**:
- Compute **once** in shared setup to avoid redundant I/O
- Reuse same `TokenizerAuthority` instance for both lanes
- Fail-fast if hash computation fails (critical integrity check)
- Verbose output shows abbreviated hashes for debugging

### 3.2 Phase 3/4: Lane Receipt Population (AC2)

**Location**: `xtask/src/crossval/parity_both.rs::run_single_lane()`
**Modification**: Add `tokenizer_authority` parameter and populate receipt
**Insert Before**: Line 682 (`receipt.finalize()`)

#### 3.2.1 Function Signature Change

```rust
#[cfg(feature = "ffi")]
#[allow(clippy::too_many_arguments)]
fn run_single_lane(
    backend: CppBackend,
    model_path: &Path,
    formatted_prompt: &str,
    _add_bos: bool,
    _parse_special: bool,
    rust_logits: &[Vec<f32>],
    cos_tol: f32,
    metrics: &str,
    receipt_path: &Path,
    verbose: bool,
    dump_cpp_ids: bool,
    tokenizer_authority: &bitnet_crossval::receipt::TokenizerAuthority, // NEW PARAMETER
) -> Result<()> {
    // ... existing code ...
```

#### 3.2.2 Receipt Population

**Insert Before Line 682**:

```rust
// Populate tokenizer authority (AC2)
receipt.set_tokenizer_authority(tokenizer_authority.clone());

if verbose {
    eprintln!(
        "  ✓ TokenizerAuthority set: source={:?}, hash={}",
        tokenizer_authority.source,
        tokenizer_authority.file_hash.as_ref().map(|h| &h[..16]).unwrap_or("(none)")
    );
}

receipt.finalize();
```

#### 3.2.3 Call Site Updates

**Update Lane A Call** (line 500):

```rust
// Lane A: BitNet.cpp
run_single_lane(
    CppBackend::BitNet,
    model_gguf,
    &formatted_prompt,
    add_bos,
    parse_special,
    &rust_logits,
    cos_tol,
    metrics,
    &receipt_bitnet,
    verbose,
    dump_cpp_ids,
    &tokenizer_authority, // NEW ARGUMENT
)
.context("Lane A (BitNet.cpp) failed")?;
```

**Update Lane B Call** (line 516):

```rust
// Lane B: llama.cpp
run_single_lane(
    CppBackend::Llama,
    model_gguf,
    &formatted_prompt,
    add_bos,
    parse_special,
    &rust_logits,
    cos_tol,
    metrics,
    &receipt_llama,
    verbose,
    dump_cpp_ids,
    &tokenizer_authority, // NEW ARGUMENT
)
.context("Lane B (llama.cpp) failed")?;
```

### 3.3 Phase 5: Cross-Lane Validation (AC3, AC4)

**Location**: `xtask/src/crossval/parity_both.rs::run_dual_lanes_and_summarize()`
**Insert After**: Line 533 (after loading receipts)
**Before**: Line 535 (print summary)

```rust
// STEP 7.5: Validate tokenizer consistency across lanes (AC3, AC4)
if verbose {
    eprintln!("\n⚙ Validating tokenizer consistency across lanes...");
}

// Load receipts to extract tokenizer authority
let receipt_bitnet_content = std::fs::read_to_string(&receipt_bitnet)
    .context("Failed to read Lane A receipt")?;
let receipt_llama_content = std::fs::read_to_string(&receipt_llama)
    .context("Failed to read Lane B receipt")?;

let receipt_bitnet_obj: ParityReceipt = serde_json::from_str(&receipt_bitnet_content)
    .context("Failed to parse Lane A receipt")?;
let receipt_llama_obj: ParityReceipt = serde_json::from_str(&receipt_llama_content)
    .context("Failed to parse Lane B receipt")?;

// Extract tokenizer authorities
let auth_a = receipt_bitnet_obj
    .tokenizer_authority
    .as_ref()
    .context("Lane A receipt missing tokenizer authority")?;
let auth_b = receipt_llama_obj
    .tokenizer_authority
    .as_ref()
    .context("Lane B receipt missing tokenizer authority")?;

// Validate consistency (AC3)
use bitnet_crossval::receipt::validate_tokenizer_consistency;
if let Err(e) = validate_tokenizer_consistency(auth_a, auth_b) {
    eprintln!("\n✗ ERROR: Tokenizer consistency validation failed");
    eprintln!("  Lane A config hash: {}", auth_a.config_hash);
    eprintln!("  Lane B config hash: {}", auth_b.config_hash);
    eprintln!("  Details: {}", e);
    std::process::exit(2); // AC4: Exit code 2 for tokenizer mismatch
}

if verbose {
    eprintln!("  ✓ Tokenizer consistency validated");
    eprintln!("    Config hash: {}", &auth_a.config_hash[..16]);
    eprintln!("    Token count: {}", auth_a.token_count);
}
```

**Error Handling**:
- Missing `tokenizer_authority` → Context error propagation (should not happen if AC2 implemented)
- Hash mismatch → Explicit exit code 2 (AC4 requirement)
- Verbose output shows abbreviated hashes for debugging

### 3.4 Phase 5: Summary Output Enhancement (AC5)

**Location**: `xtask/src/crossval/parity_both.rs::print_unified_summary()`
**Modification**: Add tokenizer hash display in both text and JSON formats

#### 3.4.1 Function Signature Change

```rust
pub fn print_unified_summary(
    lane_a: &LaneResult,
    lane_b: &LaneResult,
    format: &str,
    _verbose: bool,
    tokenizer_hash: Option<&str>, // NEW PARAMETER
) -> Result<()> {
    // ... existing code ...
}
```

#### 3.4.2 Text Format Enhancement

**Insert After Line 272** (before "Overall Status" section):

```rust
// Tokenizer Information (AC5)
if let Some(hash) = tokenizer_hash {
    println!("Tokenizer Consistency");
    println!("{}", "─".repeat(60));
    println!("Config hash:      {}", &hash[..32]); // Show first 32 chars
    println!("Full hash:        {}", hash);
    println!();
}
```

#### 3.4.3 JSON Format Enhancement

**Modify `print_json_summary()` at line 306**:

```rust
fn print_json_summary(
    lane_a: &LaneResult,
    lane_b: &LaneResult,
    tokenizer_hash: Option<&str>,
) -> Result<()> {
    let both_passed = lane_a.passed && lane_b.passed;

    let mut output = serde_json::json!({
        "status": if both_passed { "ok" } else { "failed" },
        "lanes": {
            "bitnet": lane_metrics(lane_a),
            "llama": lane_metrics(lane_b),
        },
        "overall": {
            "both_passed": both_passed,
            "exit_code": if both_passed { 0 } else { 1 }
        }
    });

    // Add tokenizer hash if available (AC5)
    if let Some(hash) = tokenizer_hash {
        output["tokenizer"] = serde_json::json!({
            "config_hash": hash,
            "status": "consistent"
        });
    }

    println!("{}", serde_json::to_string_pretty(&output)?);
    Ok(())
}
```

#### 3.4.4 Call Site Update

**Update Call at Line 535**:

```rust
// Extract tokenizer hash for summary display (AC5)
let tokenizer_hash = receipt_bitnet_obj
    .tokenizer_authority
    .as_ref()
    .map(|auth| auth.config_hash.as_str());

print_unified_summary(&lane_a, &lane_b, format, verbose, tokenizer_hash)?;
```

---

## 4. Error Handling Specification

### 4.1 Hash Computation Failures

**Phase**: Shared Setup (AC1)
**Error Types**:
- File I/O error (tokenizer.json not readable)
- Hash computation error (unlikely - only if file corrupted mid-read)

**Handling**:
```rust
let file_hash = compute_tokenizer_file_hash(tokenizer)
    .context("Failed to compute tokenizer file hash")?;
```

**Exit Code**: 1 (propagated via `?` operator)
**User Message**: `Error: Failed to compute tokenizer file hash`

### 4.2 Tokenizer Consistency Validation Failures

**Phase**: Cross-Lane Validation (AC3, AC4)
**Error Types**:
- Config hash mismatch (different effective tokenizers)
- Token count mismatch (sanity check failure)

**Handling**:
```rust
if let Err(e) = validate_tokenizer_consistency(auth_a, auth_b) {
    eprintln!("\n✗ ERROR: Tokenizer consistency validation failed");
    eprintln!("  Lane A config hash: {}", auth_a.config_hash);
    eprintln!("  Lane B config hash: {}", auth_b.config_hash);
    eprintln!("  Details: {}", e);
    std::process::exit(2); // AC4: Exit code 2
}
```

**Exit Code**: **2** (explicit, distinct from parity failure exit code 1)
**User Message**: Clear diagnostic with both lane hashes displayed

### 4.3 Missing TokenizerAuthority in Receipts

**Phase**: Cross-Lane Validation (AC3)
**Error Type**: `tokenizer_authority` field is `None` in loaded receipt

**Handling**:
```rust
let auth_a = receipt_bitnet_obj
    .tokenizer_authority
    .as_ref()
    .context("Lane A receipt missing tokenizer authority")?;
```

**Exit Code**: 1 (propagated via `?` operator)
**User Message**: `Error: Lane A receipt missing tokenizer authority`

**Root Cause Prevention**: This should never occur if AC2 is properly implemented (set_tokenizer_authority called before finalize).

---

## 5. Receipt Schema Updates

### 5.1 Current Schema (v1.0.0)

**File**: `crossval/src/receipt.rs` (lines 88-137)

```rust
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ParityReceipt {
    pub version: u32,
    pub timestamp: String,
    pub model: String,
    pub backend: String,
    pub prompt: String,
    pub positions: usize,
    pub thresholds: Thresholds,
    pub rows: Vec<PositionMetrics>,
    pub summary: Summary,

    // v2 fields (optional for backward compatibility)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tokenizer_authority: Option<TokenizerAuthority>, // ← EXISTS, just not populated
    // ... other v2 fields ...
}
```

### 5.2 Schema Evolution

**No schema version bump required** because:
- `tokenizer_authority` field already exists as `Option<TokenizerAuthority>` (line 124)
- Uses `#[serde(skip_serializing_if = "Option::is_none")]` for backward compatibility
- Existing v1 receipts (with `None` value) remain valid
- New receipts (with `Some(TokenizerAuthority)`) are forward-compatible

**Version Detection** (existing logic at lines 268-278):

```rust
pub fn infer_version(&self) -> &str {
    match (&self.tokenizer_authority, &self.prompt_template) {
        (Some(_), _) | (_, Some(_)) => "2.0.0",
        _ => "1.0.0",
    }
}
```

**Result**: Receipts with populated `tokenizer_authority` automatically report as v2.0.0.

### 5.3 Example Receipt JSON

**Before Integration** (v1.0.0):
```json
{
  "version": 1,
  "timestamp": "2025-10-27T10:30:00Z",
  "model": "models/model.gguf",
  "backend": "bitnet",
  "prompt": "What is 2+2?",
  "positions": 4,
  "thresholds": { "mse": 0.0001, "kl": 0.1, "topk": 0.8 },
  "rows": [ /* ... */ ],
  "summary": { "all_passed": true, "mean_mse": 2.15e-05 }
}
```

**After Integration** (v2.0.0):
```json
{
  "version": 1,
  "timestamp": "2025-10-27T10:30:00Z",
  "model": "models/model.gguf",
  "backend": "bitnet",
  "prompt": "What is 2+2?",
  "positions": 4,
  "thresholds": { "mse": 0.0001, "kl": 0.1, "topk": 0.8 },
  "rows": [ /* ... */ ],
  "summary": { "all_passed": true, "mean_mse": 2.15e-05 },
  "tokenizer_authority": {
    "source": "external",
    "path": "models/tokenizer.json",
    "file_hash": "a3f7b8c9d1e2f3a4b5c6d7e8f9a0b1c2d3e4f5a6b7c8d9e0f1a2b3c4d5e6f7a8",
    "config_hash": "e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8",
    "token_count": 8
  }
}
```

---

## 6. Testing Strategy

### 6.1 Unit Tests (crossval/src/receipt.rs)

**Existing Coverage** (lines 702-742):
- ✅ `test_compute_tokenizer_config_hash_determinism()` (lines 703-718)
- ✅ `test_compute_tokenizer_file_hash_determinism()` (lines 721-741)

**Additional Tests Needed**: None (existing tests validate hash computation helpers).

### 6.2 Integration Tests (xtask/tests/)

**New Test Suite**: `xtask/tests/tokenizer_authority_integration_tests.rs`

#### Test Cases

```rust
#[test]
#[serial(bitnet_env)]
fn test_parity_both_populates_tokenizer_authority_in_both_receipts() {
    // AC2: Both receipts have TokenizerAuthority populated
    // 1. Run parity-both command
    // 2. Load both receipts
    // 3. Assert tokenizer_authority.is_some() for both
    // 4. Assert file_hash matches expected SHA256
}

#[test]
#[serial(bitnet_env)]
fn test_parity_both_validates_tokenizer_consistency() {
    // AC3: Consistency validation called
    // 1. Run parity-both with same tokenizer
    // 2. Verify exit code 0
    // 3. Verify summary includes tokenizer hash
}

#[test]
#[serial(bitnet_env)]
fn test_parity_both_exits_2_on_tokenizer_mismatch() {
    // AC4: Exit code 2 on hash mismatch
    // 1. Run lane A with tokenizer v1
    // 2. Run lane B with tokenizer v2 (different hash)
    // 3. Verify exit code 2 (not 0 or 1)
    // 4. Verify error message shows both hashes
}

#[test]
#[serial(bitnet_env)]
fn test_parity_both_summary_includes_tokenizer_hash() {
    // AC5: Summary output shows hash
    // 1. Run parity-both command
    // 2. Capture stdout
    // 3. Assert summary contains "Tokenizer Consistency" section
    // 4. Assert config_hash displayed (text format)
    // 5. Assert JSON output has "tokenizer.config_hash" field
}

#[test]
#[serial(bitnet_env)]
fn test_tokenizer_authority_computed_once() {
    // AC1: Hash computation happens once (shared setup)
    // 1. Mock file I/O to count hash computation calls
    // 2. Run parity-both
    // 3. Assert compute_tokenizer_file_hash called exactly once
}

#[test]
#[serial(bitnet_env)]
fn test_receipt_serialization_with_tokenizer_authority() {
    // AC6: Receipt schema serialization
    // 1. Create receipt with TokenizerAuthority
    // 2. Serialize to JSON
    // 3. Deserialize back
    // 4. Assert tokenizer_authority preserved
    // 5. Assert infer_version() returns "2.0.0"
}
```

### 6.3 End-to-End Validation

**Script**: `scripts/validate_tokenizer_authority_integration.sh`

```bash
#!/usr/bin/env bash
# Validate tokenizer authority integration in parity-both workflow

MODEL="models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf"
TOKENIZER="models/microsoft-bitnet-b1.58-2B-4T-gguf/tokenizer.json"
OUT_DIR="/tmp/parity-test-$$"

# Run parity-both
cargo run -p xtask --features crossval-all -- parity-both \
  --model-gguf "$MODEL" \
  --tokenizer "$TOKENIZER" \
  --prompt "What is 2+2?" \
  --max-tokens 4 \
  --out-dir "$OUT_DIR" \
  --format json \
  --verbose

EXIT_CODE=$?

# Validation checks
echo "=== Validation Checks ==="

# 1. Exit code should be 0 or 1 (not 2, since using same tokenizer)
if [ $EXIT_CODE -eq 2 ]; then
  echo "✗ FAIL: Exit code 2 (tokenizer mismatch should not happen)"
  exit 1
fi

# 2. Both receipts should exist
for receipt in receipt_bitnet.json receipt_llama.json; do
  if [ ! -f "$OUT_DIR/$receipt" ]; then
    echo "✗ FAIL: Receipt $receipt not found"
    exit 1
  fi
done

# 3. Both receipts should have tokenizer_authority field
for receipt in receipt_bitnet.json receipt_llama.json; do
  if ! jq -e '.tokenizer_authority' "$OUT_DIR/$receipt" > /dev/null; then
    echo "✗ FAIL: Receipt $receipt missing tokenizer_authority"
    exit 1
  fi
done

# 4. Both receipts should have matching config_hash
HASH_A=$(jq -r '.tokenizer_authority.config_hash' "$OUT_DIR/receipt_bitnet.json")
HASH_B=$(jq -r '.tokenizer_authority.config_hash' "$OUT_DIR/receipt_llama.json")

if [ "$HASH_A" != "$HASH_B" ]; then
  echo "✗ FAIL: Config hash mismatch: $HASH_A vs $HASH_B"
  exit 1
fi

# 5. File hash should be present and 64 chars (SHA256)
FILE_HASH=$(jq -r '.tokenizer_authority.file_hash' "$OUT_DIR/receipt_bitnet.json")
if [ ${#FILE_HASH} -ne 64 ]; then
  echo "✗ FAIL: File hash not 64 chars: $FILE_HASH"
  exit 1
fi

echo "✓ All validation checks passed"
echo "  Config hash: ${HASH_A:0:32}..."
echo "  File hash:   ${FILE_HASH:0:32}..."

# Cleanup
rm -rf "$OUT_DIR"
```

---

## 7. Performance Impact Analysis

### 7.1 Computational Overhead

**Hash Computation Cost** (AC1):
- SHA256 of tokenizer.json (typically ~500KB-2MB)
- Expected overhead: **5-20ms** (single-threaded I/O + hashing)
- Amortized: Computed once, reused for both lanes
- **Negligible** compared to total parity-both runtime (~2-5 seconds)

**Validation Cost** (AC3):
- String comparison of two 64-char hashes
- Expected overhead: **< 1µs** (negligible)

**Total Impact**: **< 1%** of total parity-both workflow time.

### 7.2 Memory Overhead

**TokenizerAuthority Structure**:
```rust
pub struct TokenizerAuthority {
    pub source: TokenizerSource,       // ~8 bytes (enum)
    pub path: String,                  // ~48 bytes + path length
    pub file_hash: Option<String>,     // ~72 bytes (64-char hex + overhead)
    pub config_hash: String,           // ~88 bytes (64-char hex + overhead)
    pub token_count: usize,            // 8 bytes
}
```

**Total per instance**: ~224 bytes + path length
**Instances**: 3 (shared + 2 receipts)
**Total overhead**: **~700 bytes** (negligible)

### 7.3 I/O Impact

**Additional I/O**:
1. Read tokenizer.json for file hash (AC1): **1 read** (~500KB-2MB)
2. Read both receipts for validation (AC3): **2 reads** (~5-10KB each)

**Optimization**: File hash uses `std::fs::read()` which is already buffered by OS page cache.

**Result**: **No significant I/O impact** (reads are small and cached).

---

## 8. Backward Compatibility

### 8.1 Receipt Schema Compatibility

**Existing v1.0.0 receipts**:
- `tokenizer_authority` field is `Option<TokenizerAuthority>` (line 124)
- Serialized as `None` → omitted from JSON (skip_serializing_if)
- Deserialization: missing field → `None` (serde default)

**New v2.0.0 receipts**:
- `tokenizer_authority` field is `Some(TokenizerAuthority)`
- Serialized with full TokenizerAuthority object
- `infer_version()` returns "2.0.0" (line 276)

**Result**: **Fully backward compatible** (no breaking changes).

### 8.2 Command-Line Interface

**No CLI changes required**:
- All integration is internal to `parity-both` logic
- No new flags or arguments
- Exit code 2 is a new semantic (previously unused)

**Result**: **No CLI breaking changes**.

---

## 9. Success Criteria (Acceptance Criteria)

### AC1: SHA256 Hash Computed Once Before Dual-Lane Execution

**Validation**:
- [ ] `compute_tokenizer_file_hash()` called in shared setup phase (after line 471)
- [ ] Hash computation happens **before** lane A and lane B execution
- [ ] Same `TokenizerAuthority` instance passed to both lanes
- [ ] Verbose output shows "Computing tokenizer authority" message

**Test**: `test_tokenizer_authority_computed_once()`

---

### AC2: Both Receipts Have Matching TokenizerAuthority.sha256_hash

**Validation**:
- [ ] `receipt.set_tokenizer_authority()` called in `run_single_lane()` before `finalize()`
- [ ] Lane A receipt has `tokenizer_authority.file_hash` populated
- [ ] Lane B receipt has `tokenizer_authority.file_hash` populated
- [ ] Both receipts have identical `file_hash` values

**Test**: `test_parity_both_populates_tokenizer_authority_in_both_receipts()`

---

### AC3: Validate Tokenizer Consistency After Both Lanes Complete

**Validation**:
- [ ] `validate_tokenizer_consistency()` called after loading both receipts (after line 533)
- [ ] Validation checks `config_hash` match between lanes
- [ ] Validation checks `token_count` match between lanes
- [ ] Verbose output shows "Validating tokenizer consistency" message

**Test**: `test_parity_both_validates_tokenizer_consistency()`

---

### AC4: Exit Code 2 if Tokenizer Hashes Differ

**Validation**:
- [ ] `validate_tokenizer_consistency()` failure triggers `std::process::exit(2)`
- [ ] Exit code is **2** (distinct from parity failure exit code 1)
- [ ] Error message displays both lane config hashes
- [ ] Error message clearly identifies tokenizer mismatch

**Test**: `test_parity_both_exits_2_on_tokenizer_mismatch()`

---

### AC5: Summary Output Includes Tokenizer Hash

**Validation**:
- [ ] Text format summary has "Tokenizer Consistency" section
- [ ] Text format shows first 32 chars of config hash
- [ ] Text format shows full hash (64 chars)
- [ ] JSON format has `tokenizer.config_hash` field
- [ ] JSON format has `tokenizer.status: "consistent"` field

**Test**: `test_parity_both_summary_includes_tokenizer_hash()`

---

### AC6: Receipts Serialize TokenizerAuthority Properly

**Validation**:
- [ ] Receipt serialization includes `tokenizer_authority` field
- [ ] Deserialized receipt preserves `tokenizer_authority` data
- [ ] `infer_version()` returns "2.0.0" when `tokenizer_authority` is `Some(...)`
- [ ] Schema evolution is backward-compatible (v1 receipts still valid)

**Test**: `test_receipt_serialization_with_tokenizer_authority()`

---

## 10. Implementation Checklist

### Phase 1: Core Integration (High Priority)

- [ ] **P1.1**: Add `tokenizer_authority` computation in shared setup (AC1)
  - File: `xtask/src/crossval/parity_both.rs`
  - Lines: Insert after line 471
  - Estimated LOC: +35

- [ ] **P1.2**: Add `tokenizer_authority` parameter to `run_single_lane()` (AC2)
  - File: `xtask/src/crossval/parity_both.rs`
  - Lines: Modify signature at line 555, add population before line 682
  - Estimated LOC: +10

- [ ] **P1.3**: Update `run_single_lane()` call sites (AC2)
  - File: `xtask/src/crossval/parity_both.rs`
  - Lines: Update calls at lines 500, 516
  - Estimated LOC: +2

- [ ] **P1.4**: Add cross-lane consistency validation (AC3, AC4)
  - File: `xtask/src/crossval/parity_both.rs`
  - Lines: Insert after line 533
  - Estimated LOC: +30

### Phase 2: Summary Enhancements (Medium Priority)

- [ ] **P2.1**: Enhance `print_unified_summary()` signature (AC5)
  - File: `xtask/src/crossval/parity_both.rs`
  - Lines: Modify function at line 241
  - Estimated LOC: +1

- [ ] **P2.2**: Add tokenizer hash to text summary (AC5)
  - File: `xtask/src/crossval/parity_both.rs`
  - Lines: Insert after line 272
  - Estimated LOC: +8

- [ ] **P2.3**: Add tokenizer hash to JSON summary (AC5)
  - File: `xtask/src/crossval/parity_both.rs`
  - Lines: Modify `print_json_summary()` at line 306
  - Estimated LOC: +8

- [ ] **P2.4**: Update summary call site (AC5)
  - File: `xtask/src/crossval/parity_both.rs`
  - Lines: Update call at line 535
  - Estimated LOC: +5

### Phase 3: Testing (High Priority)

- [ ] **P3.1**: Create integration test suite
  - File: `xtask/tests/tokenizer_authority_integration_tests.rs`
  - Tests: 6 test cases (see section 6.2)
  - Estimated LOC: +200

- [ ] **P3.2**: Create E2E validation script
  - File: `scripts/validate_tokenizer_authority_integration.sh`
  - Estimated LOC: +80

- [ ] **P3.3**: Run full test suite
  - Command: `cargo test -p xtask --features crossval-all --test tokenizer_authority_integration_tests`

### Phase 4: Documentation (Medium Priority)

- [ ] **P4.1**: Update CLAUDE.md with new exit code semantics
  - File: `CLAUDE.md`
  - Section: Exit code reference

- [ ] **P4.2**: Update parity-both command documentation
  - File: `docs/specs/parity-both-command.md`
  - Section: TokenizerAuthority integration

- [ ] **P4.3**: Add tokenizer authority examples to usage guide
  - File: `docs/howto/use-parity-both.md` (create if needed)

---

## 11. Risk Assessment and Mitigation

### Risk 1: Hash Computation Performance Impact

**Probability**: Low
**Impact**: Low
**Mitigation**: Hash computed once in shared setup (~5-20ms overhead).
**Fallback**: If performance critical, add `--skip-tokenizer-hash` flag (future enhancement).

---

### Risk 2: Tokenizer File I/O Errors

**Probability**: Medium
**Impact**: High (blocks parity-both execution)
**Mitigation**:
- Clear error context via `.context("Failed to compute tokenizer file hash")?`
- Fail-fast with exit code 1 (standard error propagation)
- Verbose output shows file path for debugging

---

### Risk 3: Receipt Schema Evolution Confusion

**Probability**: Low
**Impact**: Medium
**Mitigation**:
- `tokenizer_authority` is already `Option<T>` (backward compatible)
- `infer_version()` automatically detects v1 vs v2 based on field presence
- Documentation clearly explains schema evolution

---

### Risk 4: Exit Code 2 Confusion

**Probability**: Low
**Impact**: Medium
**Mitigation**:
- Clear error message: "Tokenizer consistency validation failed"
- Show both lane config hashes in error output
- Document exit code semantics in CLAUDE.md and command help text

---

## 12. Alternative Approaches Considered

### Alternative 1: Compute Hash Per-Lane Instead of Shared

**Rejected Reason**: Redundant I/O (reads tokenizer.json twice), violates DRY principle.

---

### Alternative 2: Use File Modification Time Instead of SHA256

**Rejected Reason**: Not cryptographically secure, doesn't detect file content changes with preserved mtime.

---

### Alternative 3: Store Only Config Hash (Skip File Hash)

**Rejected Reason**: File hash provides stronger provenance tracking (detects byte-level changes).

---

## 13. References

- **Exploration Document**: `/tmp/explore_parity_both.md`
- **Receipt Schema**: `crossval/src/receipt.rs` (lines 54-503)
- **Parity-Both Implementation**: `xtask/src/crossval/parity_both.rs` (lines 156-702)
- **Related Spec**: `docs/specs/parity-both-preflight-tokenizer-integration.md`
- **BitNet.rs Architecture**: `docs/architecture-overview.md`

---

## 14. Appendix: Complete File Modification Summary

### File 1: `xtask/src/crossval/parity_both.rs`

**Total Changes**: ~100 LOC added

| Section | Line Range | Change Type | LOC |
|---------|------------|-------------|-----|
| Shared setup (AC1) | After 471 | Insert | +35 |
| `run_single_lane()` signature (AC2) | 555 | Modify | +1 |
| `run_single_lane()` population (AC2) | Before 682 | Insert | +8 |
| Lane A call site (AC2) | 500 | Modify | +1 |
| Lane B call site (AC2) | 516 | Modify | +1 |
| Cross-lane validation (AC3, AC4) | After 533 | Insert | +30 |
| `print_unified_summary()` signature (AC5) | 241 | Modify | +1 |
| Text summary enhancement (AC5) | After 272 | Insert | +8 |
| JSON summary enhancement (AC5) | 306-322 | Modify | +8 |
| Summary call site (AC5) | 535 | Modify | +5 |

### File 2: `xtask/tests/tokenizer_authority_integration_tests.rs`

**Total Changes**: ~200 LOC (new file)

| Section | LOC |
|---------|-----|
| Test suite setup | +20 |
| AC2 test (receipt population) | +30 |
| AC3 test (consistency validation) | +25 |
| AC4 test (exit code 2) | +35 |
| AC5 test (summary output) | +40 |
| AC1 test (single hash computation) | +30 |
| AC6 test (schema serialization) | +20 |

### File 3: `scripts/validate_tokenizer_authority_integration.sh`

**Total Changes**: ~80 LOC (new file)

---

**End of Specification**
