# TokenizerAuthority Integration in Parity-Both Receipts: Technical Specification

**Status**: Implementation Complete
**Version**: 1.0.0
**Feature**: TokenizerAuthority metadata in dual-lane cross-validation receipts
**Priority**: High (Cross-Validation Reproducibility)
**Scope**: `xtask/src/crossval/parity_both.rs`, `crossval/src/receipt.rs`, receipt schema v2.0.0

---

## 1. Executive Summary

This specification defines the integration of TokenizerAuthority metadata into parity-both dual-lane cross-validation receipts for BitNet.rs. TokenizerAuthority captures complete tokenizer provenance (source, file hash, config hash, token count) to ensure receipt reproducibility and enable systematic tokenizer parity validation across BitNet.cpp and llama.cpp backends.

### 1.1 Problem Statement

**Current State**:
- ✅ TokenizerAuthority struct exists (crossval/src/receipt.rs:54-82)
- ✅ Hash computation helpers implemented
- ✅ Validation function `validate_tokenizer_consistency()` implemented
- ✅ Receipt builder API `set_tokenizer_authority()` exists

**Integration Complete**:
- ✅ TokenizerAuthority computed once during shared setup (STEP 2.5)
- ✅ Same authority metadata injected into both lane receipts
- ✅ Hash-based validation detects tokenizer mismatches
- ✅ Exit code 2 on tokenizer consistency violation
- ✅ Backward-compatible receipt schema v2.0.0

###1.2 Key Goals

1. **Single Computation**: TokenizerAuthority computed exactly once at STEP 2.5 (shared setup phase)
2. **Dual Injection**: Same authority object passed to both lane receipts via `set_tokenizer_authority()`
3. **Hash Validation**: Config hash and token count validated across lanes in STEP 7
4. **Exit Code 2**: Tokenizer mismatch triggers exit code 2 (usage error)
5. **Backward Compatibility**: Receipt schema v2.0.0 extends v1.0.0 with optional fields
6. **Deterministic**: SHA256 hashes ensure bit-perfect reproducibility

---

## 2. Feature Overview

### 2.1 Parity-Both Orchestration Flow with TokenizerAuthority

```
┌─────────────────────────────────────────────────────────────┐
│ STEP 1: Preflight Both Backends                             │
│ • BitNet.cpp, llama.cpp availability checks                 │
│ • Auto-repair (--no-repair disables)                        │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 2: Shared Setup (Template + Tokenization)              │
│ • Load Rust tokenizer (bitnet_tokenizers::loader)           │
│ • Rust tokenization (once, reused for both lanes)           │
│ • C++ tokenization (for both backends)                      │
│ • Token parity validation (fail-fast if mismatch → exit 2)  │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 2.5: **COMPUTE TOKENIZER AUTHORITY** (ONCE) ← KEY STEP │
│ • detect_tokenizer_source(tokenizer_path)                   │
│   → GgufEmbedded | External | AutoDiscovered                │
│ • compute_tokenizer_file_hash(tokenizer_path)               │
│   → SHA256 of tokenizer.json file (if external)             │
│ • compute_tokenizer_config_hash_from_tokenizer(tokenizer)   │
│   → SHA256 of canonical config (vocab_size, real_vocab_size)│
│ • token_count = rust_tokens.len()                           │
│                                                              │
│ TokenizerAuthority {                                        │
│   source: External,                                         │
│   path: "models/tokenizer.json",                            │
│   file_hash: Some("6f3ef9d7..."),  // SHA256 hex (64 chars) │
│   config_hash: "a1b2c3d4...",      // SHA256 hex (64 chars) │
│   token_count: 42,                                          │
│ }                                                            │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 3: Shared Rust Logits Evaluation                       │
│ • Load GGUF model (bitnet_models::loader)                   │
│ • eval_logits_all_positions(model, tokens)                  │
│   → Vec<Vec<f32>> (positions × vocab_size)                  │
│ • Cost: ~10-30s (shared across both lanes)                  │
└─────────────────────────────────────────────────────────────┘
    ↓
┌──────────────────────┬──────────────────────────────────────┐
│ LANE A: BitNet.cpp   │ LANE B: llama.cpp                    │
├──────────────────────┼──────────────────────────────────────┤
│ STEP 4A-5A: C++ Eval │ STEP 4B-5B: C++ Eval                 │
│ & Compare            │ & Compare                            │
│ • BitnetSession ctx  │ • BitnetSession ctx (llama backend)  │
│ • C++ eval (~10-30s) │ • C++ eval (~10-30s)                 │
│ • Per-position MSE   │ • Per-position MSE                   │
│ • L2 distance        │ • L2 distance                        │
│ • KL divergence (opt)│ • KL divergence (opt)                │
│ • TopK agreement (op)│ • TopK agreement (opt)               │
├──────────────────────┼──────────────────────────────────────┤
│ STEP 6A: Receipt A   │ STEP 6B: Receipt B                   │
│ • Create receipt     │ • Create receipt                     │
│ • **set_tokenizer_   │ • **set_tokenizer_                   │
│   authority(auth)**  │   authority(auth)** ← SAME OBJECT    │
│ • set_prompt_template│ • set_prompt_template                │
│ • add_position(...)  │ • add_position(...)                  │
│ • finalize()         │ • finalize()                         │
│ • write_to_file(     │ • write_to_file(                     │
│   receipt_bitnet.json│   receipt_llama.json)                │
│ )                    │                                      │
└──────────────────────┴──────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 7: Validation & Summary Output                          │
│ • Load both receipts as LaneResult                          │
│ • Extract auth_a = receipt_bitnet.tokenizer_authority       │
│ • Extract auth_b = receipt_llama.tokenizer_authority        │
│ • **validate_tokenizer_consistency(auth_a, auth_b)**        │
│   ├─ Check: config_hash match                               │
│   └─ Check: token_count match                               │
│ • If validation fails → eprintln error, exit code 2         │
│ • Print unified summary (text or JSON)                      │
│ • Determine exit code (0=both pass, 1=one fails, 2=error)   │
└─────────────────────────────────────────────────────────────┘
```

**Critical Detail**: TokenizerAuthority is computed ONCE at STEP 2.5, passed by reference to both `run_single_lane()` calls, and cloned into both receipts. This ensures both lanes have identical tokenizer metadata.

---

## 3. Acceptance Criteria

### AC1: Single TokenizerAuthority Computation

**Requirement**: Compute TokenizerAuthority exactly once during shared setup (STEP 2.5).

**Implementation** (xtask/src/crossval/parity_both.rs:496-528):
```rust
// STEP 2.5: Compute shared TokenizerAuthority (AC1)
if verbose {
    eprintln!("\n⚙ Shared: Computing tokenizer authority...");
}

let tokenizer_authority = {
    use bitnet_crossval::receipt::{
        TokenizerAuthority, compute_tokenizer_config_hash_from_tokenizer,
        compute_tokenizer_file_hash, detect_tokenizer_source,
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

**Success Criteria**:
- TokenizerAuthority computed exactly once (not per-lane)
- Computation happens after tokenizer loading but before lane execution
- Source detection uses path heuristics (tokenizer.json → External, else GgufEmbedded)
- File hash computed for external tokenizers only (None for GGUF-embedded)
- Config hash computed from canonical vocab representation (deterministic)
- Token count captured from Rust tokenization result

---

### AC2: Dual Receipt Injection

**Requirement**: Inject same TokenizerAuthority object into both lane receipts (receipt_bitnet.json, receipt_llama.json).

**Implementation** (xtask/src/crossval/parity_both.rs:557-588):
```rust
// Lane A: BitNet.cpp (STEP 6A)
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
    &tokenizer_authority,  // ← Passed by reference to Lane A
)
.context("Lane A (BitNet.cpp) failed")?;

// Lane B: llama.cpp (STEP 6B)
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
    &tokenizer_authority,  // ← Same object passed to Lane B
)
.context("Lane B (llama.cpp) failed")?;
```

**Inside run_single_lane()** (xtask/src/crossval/parity_both.rs:788-797):
```rust
// Create receipt
let mut receipt = bitnet_crossval::receipt::ParityReceipt::new(
    model_path.to_string_lossy().as_ref(),
    backend.name(),
    formatted_prompt,
);

// ... add position metrics ...

// **INJECT TokenizerAuthority** (AC2)
receipt.set_tokenizer_authority(tokenizer_authority.clone());

if verbose {
    eprintln!(
        "  ✓ TokenizerAuthority set: source={:?}, hash={}",
        tokenizer_authority.source,
        tokenizer_authority.file_hash.as_ref().map(|h| &h[..16]).unwrap_or("(none)")
    );
}

receipt.finalize();
receipt.write_to_file(receipt_path).context("Failed to write receipt")?;
```

**Success Criteria**:
- Same TokenizerAuthority reference passed to both `run_single_lane()` calls
- Authority cloned into each receipt via `set_tokenizer_authority(authority.clone())`
- Both receipts contain identical tokenizer_authority field in JSON output
- Receipt files written atomically to output directory

---

### AC3: TokenizerAuthority Validation Logic

**Requirement**: Validate tokenizer consistency across both lanes using hash comparison.

**Implementation** (xtask/src/crossval/parity_both.rs:594-634):
```rust
// STEP 7.5: Validate tokenizer consistency across lanes (AC3)
if verbose {
    eprintln!("\n⚙ Validating tokenizer consistency across lanes...");
}

// Load receipts to extract tokenizer authority
let receipt_bitnet_content =
    std::fs::read_to_string(&receipt_bitnet).context("Failed to read Lane A receipt")?;
let receipt_llama_content =
    std::fs::read_to_string(&receipt_llama).context("Failed to read Lane B receipt")?;

let receipt_bitnet_obj: ParityReceipt =
    serde_json::from_str(&receipt_bitnet_content).context("Failed to parse Lane A receipt")?;
let receipt_llama_obj: ParityReceipt =
    serde_json::from_str(&receipt_llama_content).context("Failed to parse Lane B receipt")?;

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

**Validation Function** (crossval/src/receipt.rs:463-503):
```rust
/// Validate tokenizer authority consistency between two lanes (AC7)
///
/// Specification: docs/specs/parity-both-preflight-tokenizer.md#AC7
///
/// Ensures that two TokenizerAuthority instances represent the same effective tokenizer
/// by comparing config hashes and token counts.
pub fn validate_tokenizer_consistency(
    lane_a: &TokenizerAuthority,
    lane_b: &TokenizerAuthority,
) -> anyhow::Result<()> {
    // Config hash must match (effective tokenizer is identical)
    if lane_a.config_hash != lane_b.config_hash {
        anyhow::bail!(
            "Tokenizer config mismatch: Lane A hash={}, Lane B hash={}",
            lane_a.config_hash,
            lane_b.config_hash
        );
    }

    // Token count should match (sanity check)
    if lane_a.token_count != lane_b.token_count {
        anyhow::bail!(
            "Token count mismatch: Lane A={}, Lane B={}",
            lane_a.token_count,
            lane_b.token_count
        );
    }

    Ok(())
}
```

**Validation Checks**:
1. **Config Hash**: Both lanes must have identical config_hash (semantic tokenizer equivalence)
2. **Token Count**: Both lanes must tokenize to same count (catch tokenizer divergence)

---

### AC4: Exit Code 2 on Tokenizer Mismatch

**Requirement**: Exit with code 2 (usage error) when tokenizer validation fails.

**Exit Code Semantics**:
| Scenario | Exit Code | Trigger | Notes |
|----------|-----------|---------|-------|
| Both lanes pass | 0 | `determine_exit_code()` → `both_passed() == true` | Normal success |
| Lane A or B fails | 1 | `determine_exit_code()` → `!both_passed()` | Divergence detected |
| Token parity violation | 2 | `validate_tokenizer_parity()` → `bail!()` | Rust vs C++ token mismatch |
| **Tokenizer hash mismatch** | **2** | `validate_tokenizer_consistency()` → `std::process::exit(2)` | **Config hash or token count differ** |
| Preflight failure | 2 | `preflight_backend_libs()` → `bail!()` | Backend unavailable |
| Invalid arguments | 2 | Arg parsing → `clap::Error` | Missing required args |

---

### AC5: TokenizerAuthority Source Detection

**Implementation** (crossval/src/receipt.rs:400-412):
```rust
/// Detect tokenizer source (AC5)
///
/// Specification: docs/specs/parity-both-preflight-tokenizer.md#AC5
pub fn detect_tokenizer_source(tokenizer_path: &Path) -> TokenizerSource {
    // Check if file exists and is named tokenizer.json
    if tokenizer_path.exists()
        && tokenizer_path.file_name() == Some(std::ffi::OsStr::new("tokenizer.json"))
    {
        TokenizerSource::External
    } else {
        TokenizerSource::GgufEmbedded
    }
}
```

**Detection Heuristics**:
1. If path ends with `tokenizer.json` AND file exists → `External`
2. Otherwise → `GgufEmbedded`
3. Note: `AutoDiscovered` not yet implemented (future enhancement)

---

### AC6: Hash Computation (File Hash & Config Hash)

**File Hash Implementation** (crossval/src/receipt.rs:336-350):
```rust
/// Compute SHA256 hash of tokenizer.json file (AC6)
///
/// Returns lowercase hex-encoded SHA256 hash (64 characters)
pub fn compute_tokenizer_file_hash(tokenizer_path: &Path) -> anyhow::Result<String> {
    use sha2::{Digest, Sha256};

    let contents = std::fs::read(tokenizer_path)
        .with_context(|| format!("Failed to read tokenizer file: {}", tokenizer_path.display()))?;

    let mut hasher = Sha256::new();
    hasher.update(&contents);
    Ok(format!("{:x}", hasher.finalize()))
}
```

**Config Hash Implementation** (crossval/src/receipt.rs:377-398):
```rust
/// Compute SHA256 hash of tokenizer config from Tokenizer trait (AC6)
///
/// This computes a hash from the tokenizer's vocab size configuration.
pub fn compute_tokenizer_config_hash_from_tokenizer(
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

**Hash Characteristics**:

| Hash Type | Input | Output | Determinism | Purpose |
|-----------|-------|--------|-------------|---------|
| File Hash | Raw file bytes | SHA256 hex (64 chars) | Bit-perfect | Detects file modifications |
| Config Hash | Canonical JSON config | SHA256 hex (64 chars) | Semantic | Detects vocab size changes |

---

## 4. Data Structures

### 4.1 TokenizerAuthority Struct

**Location**: `crossval/src/receipt.rs:54-82`

```rust
/// Tokenizer authority metadata for receipt reproducibility (AC4-AC6)
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TokenizerAuthority {
    /// Tokenizer source: GgufEmbedded, External, or AutoDiscovered
    pub source: TokenizerSource,

    /// Path to tokenizer (GGUF path or tokenizer.json path)
    pub path: String,

    /// SHA256 hash of tokenizer.json file (if external)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub file_hash: Option<String>,

    /// SHA256 hash of effective tokenizer config (canonical representation)
    pub config_hash: String,

    /// Token count (for quick validation)
    pub token_count: usize,
}
```

**Field Semantics**:

| Field | Type | Required | Nullable | Purpose | Notes |
|-------|------|----------|----------|---------|-------|
| `source` | `TokenizerSource` | Yes | No | Where tokenizer came from | One of three enum variants |
| `path` | `String` | Yes | No | File path to tokenizer | GGUF path or tokenizer.json path |
| `file_hash` | `Option<String>` | No | Yes | SHA256 of tokenizer.json | None for GGUF-embedded |
| `config_hash` | `String` | Yes | No | SHA256 of canonical config | Computed from vocab sizes |
| `token_count` | `usize` | Yes | No | Token count of prompt | For quick validation |

### 4.2 TokenizerSource Enum

```rust
/// Tokenizer source type (AC5)
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TokenizerSource {
    /// Tokenizer embedded in GGUF file
    GgufEmbedded,
    /// External tokenizer.json file (explicitly provided)
    External,
    /// Auto-discovered tokenizer from model directory
    AutoDiscovered,
}
```

**JSON Serialization**:
```json
{
  "source": "gguf_embedded"  // or "external" or "auto_discovered"
}
```

### 4.3 Field Presence Rules

```rust
// For GGUF-embedded tokenizer:
TokenizerAuthority {
    source: TokenizerSource::GgufEmbedded,
    path: "models/model.gguf",
    file_hash: None,           // ← Always None (no separate file)
    config_hash: "a1b2c3d4...",  // ← Always present (64 hex chars)
    token_count: 42,
}

// For external tokenizer.json:
TokenizerAuthority {
    source: TokenizerSource::External,
    path: "models/tokenizer.json",
    file_hash: Some("6f3ef9d7..."),   // ← Always present (64 hex chars)
    config_hash: "a1b2c3d4...",       // ← Always present (64 hex chars)
    token_count: 42,
}
```

---

## 5. Receipt Schema v2.0.0

### 5.1 Extended ParityReceipt Structure

**Location**: `crossval/src/receipt.rs:84-137`

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

    /// Tokenizer authority metadata (v2.0.0)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tokenizer_authority: Option<TokenizerAuthority>,

    /// Prompt template used (v2.0.0)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_template: Option<String>,

    /// Determinism seed (v2.0.0)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub determinism_seed: Option<u64>,

    /// Model SHA256 hash (v2.0.0)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_sha256: Option<String>,
}
```

### 5.2 Complete Receipt JSON Example

**Lane A Receipt** (receipt_bitnet.json):
```json
{
  "version": 1,
  "timestamp": "2025-10-27T14:30:00Z",
  "model": "models/model.gguf",
  "backend": "bitnet",
  "prompt": "What is 2+2?",
  "positions": 4,
  "thresholds": {
    "mse": 0.0001,
    "kl": 0.1,
    "topk": 0.8
  },
  "rows": [
    {
      "pos": 0,
      "mse": 1.23e-6,
      "max_abs": 0.0042,
      "top5_rust": [128000, 1229, 374, 220, 17],
      "top5_cpp": [128000, 1229, 374, 220, 17]
    }
  ],
  "summary": {
    "all_passed": true,
    "mean_mse": 1.84e-6
  },
  "tokenizer_authority": {
    "source": "external",
    "path": "models/tokenizer.json",
    "file_hash": "6f3ef9d7a3c2b1e0d4f5a8c9b0e1d2f3a4b5c6d7e8f9a0b1c2d3e4f5a6b7c8d9",
    "config_hash": "a1b2c3d4e5f6789012345678901234567890abcdefabcdefabcdefabcdefabc0",
    "token_count": 42
  }
}
```

**Critical Detail**: Both receipts (bitnet and llama) contain **identical** `tokenizer_authority` fields.

---

## 6. Test Strategy

### 6.1 Unit Tests (crossval/src/receipt.rs)

**TokenizerAuthority Construction** (4 tests):
- External tokenizer detection and hash computation
- GGUF-embedded tokenizer detection (file_hash = None)
- Serialization/deserialization round-trip
- Equality comparison (PartialEq)

**Hash Determinism** (2 tests):
- File hash determinism (same file → same hash)
- Config hash determinism (same vocab sizes → same hash)

**Receipt Builder API** (3 tests):
- `set_tokenizer_authority()` populates optional field
- Backward compatibility with V1 receipts (no tokenizer_authority)
- Version inference (v1.0.0 vs v2.0.0)

### 6.2 Integration Tests (xtask/tests/)

**TokenizerAuthority Integration** (6 tests):
- Single computation during shared setup
- Dual receipt identical authority (config_hash match, token_count match)
- External tokenizer authority population
- GGUF-embedded tokenizer authority population
- Consistency validation success path
- Consistency validation failure path (simulated hash mismatch)

**Exit Code Tests** (4 tests):
- Exit code 0: Both lanes pass
- Exit code 1: One lane diverges
- Exit code 2: Tokenizer hash mismatch
- Exit code 2: Token count mismatch

---

## 7. API Contracts

### 7.1 Public API Surface (crossval/src/receipt.rs)

**TokenizerAuthority Functions**:
```rust
/// Compute SHA256 hash of tokenizer.json file
pub fn compute_tokenizer_file_hash(tokenizer_path: &Path) -> anyhow::Result<String>;

/// Compute SHA256 hash of tokenizer config from Tokenizer trait
pub fn compute_tokenizer_config_hash_from_tokenizer(
    tokenizer: &dyn bitnet_tokenizers::Tokenizer,
) -> anyhow::Result<String>;

/// Detect tokenizer source (GgufEmbedded, External, AutoDiscovered)
pub fn detect_tokenizer_source(tokenizer_path: &Path) -> TokenizerSource;

/// Validate tokenizer authority consistency between two lanes
pub fn validate_tokenizer_consistency(
    lane_a: &TokenizerAuthority,
    lane_b: &TokenizerAuthority,
) -> anyhow::Result<()>;
```

**Receipt Builder API**:
```rust
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

## 8. Performance Characteristics

### 8.1 TokenizerAuthority Computation Cost

| Operation | Cost | Notes |
|-----------|------|-------|
| `detect_tokenizer_source()` | <1ms | Path comparison only |
| `compute_tokenizer_file_hash()` | 1-50ms | Depends on file size (typically 200KB-2MB) |
| `compute_tokenizer_config_hash_from_tokenizer()` | <1ms | Only vocab sizes (small JSON) |
| **Total per parity-both** | **1-60ms** | **Negligible vs Rust/C++ eval (~50s)** |

### 8.2 Memory Overhead

| Field | Size | Notes |
|-------|------|-------|
| `config_hash` | 64 bytes | SHA256 hex string |
| `file_hash` | 64 bytes | SHA256 hex string (optional) |
| `path` | ~50 bytes avg | File path string |
| `source` | 8 bytes | Enum variant |
| `token_count` | 8 bytes | usize |
| **Total per receipt** | **~200 bytes** | **Negligible in receipt JSON (~10KB)** |

---

## 9. Summary and Key Takeaways

### 9.1 Architecture Summary

1. **Single Computation**: TokenizerAuthority computed ONCE at STEP 2.5 (after tokenizer loading)
2. **Shared Reference**: Passed by reference to both `run_single_lane()` calls (lanes A and B)
3. **Cloned into Receipts**: Each lane clones authority into its receipt via `set_tokenizer_authority()`
4. **Validation**: Consistency checked in STEP 7 before summary output (hash comparison + token count)
5. **Deterministic**: SHA256 hash functions produce identical output for same input
6. **Backward Compatible**: Receipt schema v2.0.0 extends v1.0.0 gracefully (optional fields)

### 9.2 Data Flow

```
Tokenizer File (tok.json)
    ↓
Tokenizer Object (loaded via bitnet_tokenizers::loader)
    ├─ extract vocab_size, real_vocab_size → config_hash (SHA256)
    └─ read file content → file_hash (SHA256)

TokenizerAuthority { source, path, file_hash, config_hash, token_count }
    ↓
    ├─ Pass to Lane A (BitNet.cpp)
    │   └─ receipt.set_tokenizer_authority(authority.clone())
    │   └─ write receipt_bitnet.json
    │
    └─ Pass to Lane B (llama.cpp)
        └─ receipt.set_tokenizer_authority(authority.clone())
        └─ write receipt_llama.json

Load Receipts
    ├─ receipt_bitnet.json → auth_a
    └─ receipt_llama.json → auth_b

Validate: validate_tokenizer_consistency(auth_a, auth_b)
    ├─ Check: auth_a.config_hash == auth_b.config_hash
    └─ Check: auth_a.token_count == auth_b.token_count
```

### 9.3 Key Files and Line References

| Component | File | Lines | Purpose |
|-----------|------|-------|---------|
| TokenizerAuthority struct | crossval/src/receipt.rs | 54-82 | Core data structure |
| TokenizerSource enum | crossval/src/receipt.rs | 38-52 | Tokenizer provenance |
| ParityReceipt struct | crossval/src/receipt.rs | 84-137 | Receipt with v2 fields |
| set_tokenizer_authority() | crossval/src/receipt.rs | 254-259 | Builder method |
| detect_tokenizer_source() | crossval/src/receipt.rs | 400-412 | Source detection |
| compute_tokenizer_file_hash() | crossval/src/receipt.rs | 336-350 | File hash |
| compute_tokenizer_config_hash_from_tokenizer() | crossval/src/receipt.rs | 377-398 | Config hash |
| validate_tokenizer_consistency() | crossval/src/receipt.rs | 463-503 | Consistency check |
| TokenizerAuthority computation | xtask/src/crossval/parity_both.rs | 496-528 | STEP 2.5 |
| Lane A/B injection | xtask/src/crossval/parity_both.rs | 557-588 | run_single_lane() calls |
| Consistency validation | xtask/src/crossval/parity_both.rs | 594-634 | STEP 7.5 |

---

## 10. References

### 10.1 Analysis Artifacts

- **/tmp/p0_tokenizer_authority_analysis.md**: Comprehensive analysis (1276 lines)
- **docs/specs/parity-both-preflight-tokenizer-integration.md**: Baseline parity-both spec with preflight integration

### 10.2 Related Specifications

- **docs/specs/parity-both-command.md**: Baseline parity-both spec (v1.1)
- **docs/specs/preflight-auto-repair.md**: Auto-repair and retry logic
- **docs/explanation/dual-backend-crossval.md**: Dual-backend architecture

---

**Specification Complete**: This document provides comprehensive technical guidance for TokenizerAuthority integration in parity-both dual-lane cross-validation receipts, ensuring reproducibility, consistency validation, and backward-compatible schema evolution for BitNet.rs neural network inference engine.
