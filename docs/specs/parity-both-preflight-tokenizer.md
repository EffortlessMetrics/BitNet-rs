# Parity-Both Preflight Integration & Tokenizer Authority Specification

**Status**: Draft
**Created**: 2025-10-26
**Feature**: Unified preflight + tokenizer authority for parity-both command
**Priority**: High (Core Cross-Validation)
**Scope**: Complete parity-both integration with auto-repair, tokenizer tracking, and dual-lane receipts

---

## Executive Summary

The `parity-both` command provides dual-lane cross-validation (BitNet.cpp + llama.cpp) but currently lacks three critical components:

1. **Preflight Integration** (~30% complete): Auto-repair logic exists but not integrated
2. **Tokenizer Authority** (~0% complete): No tokenizer metadata tracking in receipts
3. **CLI Wiring** (~70% complete): Command registered but missing template auto-detection

**Current State**: 70% complete with core orchestration working
**Target State**: 100% complete with production-ready preflight, receipts, and CLI integration

**Estimated Effort**: 14-21 hours (2-3 focused dev days)

---

## 1. Problem Statement

### 1.1 Preflight Auto-Repair Gap

**Current Behavior**:
```bash
$ cargo run -p xtask --features crossval-all -- parity-both \
  --model model.gguf --tokenizer tokenizer.json

❌ Error: Backend check failed for bitnet.cpp
Hint: Try running setup-cpp-auto manually
```

**Expected Behavior**:
```bash
$ cargo run -p xtask --features crossval-all -- parity-both \
  --model model.gguf --tokenizer tokenizer.json

⚙ Preflight: Checking bitnet backend...
  ⚠️  bitnet backend not found (will attempt auto-repair)
  🔧 Running setup-cpp-auto...
  ✓ Downloaded bitnet.cpp (45MB)
  ✓ Built bitnet.cpp (3m42s)
  ✓ Backend now available

⚙ Preflight: Checking llama backend... ✓
[continues with dual-lane execution...]
```

**Problem**: User must manually provision backends despite `--no-repair` flag and auto-repair infrastructure existing in `preflight.rs`.

### 1.2 Tokenizer Authority Missing

**Scenario**: User generates parity receipts from two runs with different tokenizers:

```bash
# Run 1: With tokenizer A
parity-both --model model.gguf --tokenizer v1/tokenizer.json

# Run 2: With tokenizer B (different version)
parity-both --model model.gguf --tokenizer v2/tokenizer.json
```

**Current State**: Receipts contain prompt and token count but NOT:
- Which tokenizer.json was used
- SHA256 hash of tokenizer config
- Whether GGUF embedded tokenizer or external JSON

**Impact**:
- Cannot reproduce token sequences from receipt alone
- Cannot verify tokenizer consistency across lanes
- Cannot detect tokenizer mutations between runs

### 1.3 CLI Template Auto-Detection

**Current Code** (`xtask/src/lib.rs:32`):
```rust
pub fn to_template_type(&self) -> bitnet_inference::prompt_template::TemplateType {
    match self {
        Self::Auto => TemplateType::Raw,  // ⚠️ PLACEHOLDER - no real auto-detection
        Self::Raw => TemplateType::Raw,
        Self::Instruct => TemplateType::Instruct,
        Self::Llama3Chat => TemplateType::Llama3Chat,
    }
}
```

**Problem**: `--prompt-template auto` (default) always returns `Raw`, bypassing GGUF metadata inspection and heuristics.

**Expected**: Auto-detect from GGUF `chat_template` metadata → tokenizer hints → fallback to `Instruct`.

---

## 2. Acceptance Criteria

### AC1: CLI Registration Complete ✅

**Status**: DONE (main.rs lines 560-617)

**Evidence**: Command registered with all 14 arguments, defaults match spec

**Validation**:
```bash
cargo run -p xtask --features crossval-all -- parity-both --help
# Output should show all flags: model-gguf, tokenizer, prompt, max-tokens, etc.
```

### AC2: Preflight Both Backends with Auto-Repair

**Status**: 🚧 IN PROGRESS (infrastructure exists, integration pending)

**Requirement**: Before dual-lane execution, check both backends and attempt auto-repair if missing.

**Implementation Steps**:
1. Call `preflight_both_backends(auto_repair, verbose)` before shared setup
2. On missing backend + auto-repair=true: invoke `setup-cpp-auto`
3. After setup: rebuild xtask (optional for MVP, show rebuild instructions)
4. Retry preflight check
5. Exit code 2 if either backend still unavailable

**Exit Codes**:
- **0**: Both backends available (cached or repaired)
- **1**: Repair failed (network/build/permission error)
- **2**: Backend unavailable + repair disabled

**Test**:
```bash
# AC2.1: Auto-repair by default (no flag)
BITNET_CPP_DIR="" cargo run -p xtask --features crossval-all -- parity-both \
  --model test.gguf --tokenizer tokenizer.json
# Expected: Attempts auto-repair

# AC2.2: --no-repair disables auto-repair
BITNET_CPP_DIR="" cargo run -p xtask --features crossval-all -- parity-both \
  --model test.gguf --tokenizer tokenizer.json --no-repair
# Expected: Fails fast with exit code 2, shows setup instructions
```

**Implementation**:

```rust
// In xtask/src/crossval/preflight.rs (NEW function)
#[cfg(feature = "crossval-all")]
pub fn preflight_both_backends(auto_repair: bool, verbose: bool) -> Result<()> {
    let backends = [CppBackend::BitNet, CppBackend::Llama];

    for backend in backends {
        if verbose {
            eprintln!("⚙ Preflight: Checking {} backend...", backend.name());
        }

        // Try preflight check
        match preflight_backend_libs(backend, verbose) {
            Ok(()) => {
                if verbose {
                    eprintln!("  ✓ {} backend available", backend.name());
                }
            }
            Err(e) => {
                if !auto_repair {
                    // Repair disabled, fail with instructions
                    eprintln!("❌ {} backend not found (repair disabled)", backend.name());
                    eprintln!("Quick fix: cargo run -p xtask -- preflight --repair=auto");
                    return Err(e);
                }

                // Attempt auto-repair
                eprintln!("  ⚠️  {} backend not found (will attempt auto-repair)", backend.name());
                match attempt_repair_with_retry(backend, verbose) {
                    Ok(()) => {
                        eprintln!("  ✓ {} backend now available", backend.name());
                        eprintln!("  ℹ️  Next: Rebuild xtask for detection");
                        eprintln!("      cargo clean -p xtask && cargo build -p xtask --features crossval-all");
                    }
                    Err(repair_err) => {
                        eprintln!("❌ Auto-repair failed for {}: {}", backend.name(), repair_err);
                        return Err(anyhow::anyhow!(
                            "Backend {} unavailable after repair attempt",
                            backend.name()
                        ));
                    }
                }
            }
        }
    }

    Ok(())
}
```

**Integration Point** (parity_both.rs line 426):
```rust
// Replace current preflight loop with:
preflight_both_backends(auto_repair, verbose)
    .context("Preflight check failed for one or both backends")?;
```

### AC3: RepairMode Honored (--no-repair flag)

**Status**: 🚧 IN PROGRESS (parameter threaded but not used)

**Requirement**: User can disable auto-repair with `--no-repair` flag.

**CLI Mapping**:
- Default (no flags): `auto_repair = !is_ci_environment()` (auto in dev, never in CI)
- `--no-repair`: `auto_repair = false` (explicit opt-out)

**Test**:
```bash
# AC3.1: Default enables auto-repair (dev environment)
CI="" parity-both --model test.gguf --tokenizer tokenizer.json
# Expected: Attempts repair if backend missing

# AC3.2: --no-repair disables auto-repair
parity-both --model test.gguf --tokenizer tokenizer.json --no-repair
# Expected: Fails fast, shows setup command

# AC3.3: CI environment defaults to no-repair
CI=true parity-both --model test.gguf --tokenizer tokenizer.json
# Expected: Fails fast (no auto-repair in CI by default)
```

### AC4: TokenizerAuthority Struct in Receipt Schema

**Status**: ❌ NOT STARTED

**Requirement**: Add tokenizer metadata to ParityReceipt schema (v2).

**Data Structure**:
```rust
/// Tokenizer authority metadata for receipt reproducibility
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenizerAuthority {
    /// Tokenizer source: "gguf_embedded" | "external" | "auto_discovered"
    pub source: String,

    /// Path to tokenizer (GGUF path or tokenizer.json path)
    pub path: String,

    /// SHA256 hash of tokenizer.json file (if external)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub file_hash: Option<String>,

    /// SHA256 hash of effective tokenizer config (canonical JSON)
    pub config_hash: String,

    /// Token count (for quick validation)
    pub token_count: usize,
}
```

**Integration** (crossval/src/receipt.rs):
```rust
pub struct ParityReceipt {
    // Existing fields...
    pub version: u32,
    pub timestamp: String,
    pub model: String,
    pub backend: String,
    pub prompt: String,
    pub positions: usize,
    pub thresholds: Thresholds,
    pub rows: Vec<PositionMetrics>,
    pub summary: Summary,

    // NEW: Tokenizer authority (v2)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tokenizer_authority: Option<TokenizerAuthority>,

    // NEW: Prompt template used
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_template: Option<String>,

    // NEW: Determinism seed
    #[serde(skip_serializing_if = "Option::is_none")]
    pub determinism_seed: Option<u64>,

    // NEW: Model hash
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_sha256: Option<String>,
}
```

**Builder API**:
```rust
impl ParityReceipt {
    /// Set tokenizer authority metadata
    pub fn set_tokenizer_authority(&mut self, authority: TokenizerAuthority) {
        self.tokenizer_authority = Some(authority);
    }

    /// Set prompt template used
    pub fn set_prompt_template(&mut self, template: String) {
        self.prompt_template = Some(template);
    }
}
```

**Test**:
```rust
#[test]
fn test_parity_receipt_with_tokenizer_authority() {
    let mut receipt = ParityReceipt::new("model.gguf", "bitnet", "What is 2+2?");

    receipt.set_tokenizer_authority(TokenizerAuthority {
        source: "external".to_string(),
        path: "tokenizer.json".to_string(),
        file_hash: Some("abc123...".to_string()),
        config_hash: "def456...".to_string(),
        token_count: 5,
    });

    receipt.set_prompt_template("instruct".to_string());
    receipt.finalize();

    let json = serde_json::to_string_pretty(&receipt).unwrap();
    assert!(json.contains("tokenizer_authority"));
    assert!(json.contains("prompt_template"));
}
```

### AC5: Tokenizer Source Tracking

**Status**: ❌ NOT STARTED

**Requirement**: Detect and record tokenizer source (GGUF embedded vs external JSON).

**Detection Logic**:
```rust
/// Detect tokenizer source from model and tokenizer paths
pub fn detect_tokenizer_source(
    model_path: &Path,
    tokenizer_path: &Path,
) -> Result<String> {
    // Strategy 1: Check if GGUF contains embedded tokenizer
    let gguf_has_tokenizer = check_gguf_embedded_tokenizer(model_path)?;

    if gguf_has_tokenizer {
        return Ok("gguf_embedded".to_string());
    }

    // Strategy 2: Check if tokenizer.json was explicitly provided
    if tokenizer_path.exists() && tokenizer_path.file_name() == Some("tokenizer.json") {
        return Ok("external".to_string());
    }

    // Strategy 3: Auto-discovered from model directory
    Ok("auto_discovered".to_string())
}

/// Check if GGUF file contains embedded tokenizer metadata
fn check_gguf_embedded_tokenizer(gguf_path: &Path) -> Result<bool> {
    // Read GGUF metadata
    // Check for tokenizer.model tensor or tokenizer.* metadata keys
    // Return true if found
    todo!("Implement GGUF metadata inspection")
}
```

**Test**:
```bash
# AC5.1: GGUF with embedded tokenizer
parity-both --model model_with_tokenizer.gguf
# Receipt: source = "gguf_embedded"

# AC5.2: External tokenizer.json
parity-both --model model.gguf --tokenizer tokenizer.json
# Receipt: source = "external"

# AC5.3: Auto-discovered tokenizer
parity-both --model model.gguf
# (loader finds tokenizer.json in model dir)
# Receipt: source = "auto_discovered"
```

### AC6: SHA256 Hash for Tokenizer Config

**Status**: ❌ NOT STARTED

**Requirement**: Compute deterministic SHA256 hash of tokenizer configuration.

**Implementation**:
```rust
use sha2::{Sha256, Digest};
use std::fs;

/// Compute SHA256 hash of tokenizer.json file
pub fn compute_tokenizer_file_hash(tokenizer_path: &Path) -> Result<String> {
    let contents = fs::read(tokenizer_path)
        .context("Failed to read tokenizer.json")?;

    let mut hasher = Sha256::new();
    hasher.update(&contents);
    Ok(format!("{:x}", hasher.finalize()))
}

/// Compute SHA256 hash of tokenizer config (canonical JSON)
///
/// This hash is computed from the tokenizer object's internal state,
/// not the raw file, to ensure consistency across formats.
pub fn compute_tokenizer_config_hash(tokenizer: &bitnet_tokenizers::Tokenizer) -> Result<String> {
    // Strategy: Serialize vocab to canonical JSON (sorted keys)
    let vocab = tokenizer.get_vocab();
    let canonical_json = serde_json::to_string(&vocab)
        .context("Failed to serialize tokenizer vocab")?;

    let mut hasher = Sha256::new();
    hasher.update(canonical_json.as_bytes());
    Ok(format!("{:x}", hasher.finalize()))
}
```

**Usage** (parity_both.rs):
```rust
// Capture tokenizer authority during shared setup
let tokenizer_authority = TokenizerAuthority {
    source: detect_tokenizer_source(model_gguf, tokenizer_path)?,
    path: tokenizer_path.to_string_lossy().to_string(),
    file_hash: if source == "external" {
        Some(compute_tokenizer_file_hash(tokenizer_path)?)
    } else {
        None
    },
    config_hash: compute_tokenizer_config_hash(&rust_tokenizer)?,
    token_count: rust_tokens.len(),
};
```

**Test**:
```rust
#[test]
fn test_tokenizer_hash_determinism() {
    let path = Path::new("tests/fixtures/tokenizer.json");

    // Hash same file twice
    let hash1 = compute_tokenizer_file_hash(path).unwrap();
    let hash2 = compute_tokenizer_file_hash(path).unwrap();

    assert_eq!(hash1, hash2, "Hash should be deterministic");
}

#[test]
fn test_config_hash_consistency() {
    let tokenizer = load_tokenizer("tests/fixtures/tokenizer.json").unwrap();

    let hash1 = compute_tokenizer_config_hash(&tokenizer).unwrap();
    let hash2 = compute_tokenizer_config_hash(&tokenizer).unwrap();

    assert_eq!(hash1, hash2);
}
```

### AC7: Tokenizer Parity Validation

**Status**: 🚧 PARTIAL (length check exists, need token-by-token validation)

**Current Code** (parity_both.rs line 554):
```rust
// Token parity check (fail-fast)
if rust_logits.len() != cpp_tokens.len() {
    anyhow::bail!(
        "Token parity mismatch for {}: Rust={} tokens, C++={} tokens",
        backend.name(),
        rust_logits.len(),
        cpp_tokens.len()
    );
}
```

**Enhancement Required**: Validate token IDs match, not just count.

**Implementation**:
```rust
/// Validate tokenizer parity between Rust and C++ implementations
///
/// Returns Ok(()) if tokens are identical, Err with detailed diagnostics otherwise.
pub fn validate_tokenizer_parity(
    rust_tokens: &[u32],
    cpp_tokens: &[u32],
    backend_name: &str,
) -> Result<()> {
    // Check 1: Length parity
    if rust_tokens.len() != cpp_tokens.len() {
        return Err(anyhow::anyhow!(
            "Tokenizer parity mismatch for {}: Rust {} tokens vs C++ {} tokens",
            backend_name,
            rust_tokens.len(),
            cpp_tokens.len()
        ));
    }

    // Check 2: Token-by-token comparison
    for (i, (r_token, c_token)) in rust_tokens.iter().zip(cpp_tokens.iter()).enumerate() {
        if r_token != c_token {
            return Err(anyhow::anyhow!(
                "Tokenizer divergence for {} at position {}: Rust token={}, C++ token={}",
                backend_name,
                i,
                r_token,
                c_token
            ));
        }
    }

    Ok(())
}
```

**Exit Code**: Token parity mismatch should exit with code **2** (usage error).

**Test**:
```rust
#[test]
fn test_tokenizer_parity_identical() {
    let rust = vec![1, 2, 3, 4, 5];
    let cpp = vec![1, 2, 3, 4, 5];

    assert!(validate_tokenizer_parity(&rust, &cpp, "bitnet").is_ok());
}

#[test]
fn test_tokenizer_parity_length_mismatch() {
    let rust = vec![1, 2, 3, 4, 5];
    let cpp = vec![1, 2, 3, 4];

    let err = validate_tokenizer_parity(&rust, &cpp, "bitnet").unwrap_err();
    assert!(err.to_string().contains("5 tokens vs"));
    assert!(err.to_string().contains("4 tokens"));
}

#[test]
fn test_tokenizer_parity_token_mismatch() {
    let rust = vec![1, 2, 3, 4, 5];
    let cpp = vec![1, 2, 99, 4, 5];  // Divergence at position 2

    let err = validate_tokenizer_parity(&rust, &cpp, "bitnet").unwrap_err();
    assert!(err.to_string().contains("position 2"));
    assert!(err.to_string().contains("Rust token=3"));
    assert!(err.to_string().contains("C++ token=99"));
}
```

### AC8: Dual Receipts with Tokenizer Metadata

**Status**: 🚧 PARTIAL (receipts generated, need tokenizer fields)

**Requirement**: Both lane receipts (`receipt_bitnet.json`, `receipt_llama.json`) include tokenizer authority.

**Implementation** (parity_both.rs line 641):
```rust
// After receipt creation:
let mut receipt = bitnet_crossval::receipt::ParityReceipt::new(
    model_path.to_str().context("Invalid model path")?,
    backend.name(),
    formatted_prompt,
);

// NEW: Add tokenizer authority (captured during shared setup)
receipt.set_tokenizer_authority(tokenizer_authority.clone());
receipt.set_prompt_template(prompt_template_str.clone());

// Optionally add determinism seed and model hash
if let Some(seed) = determinism_seed {
    receipt.determinism_seed = Some(seed);
}
if let Ok(model_hash) = compute_gguf_sha256(model_path) {
    receipt.model_sha256 = Some(model_hash);
}

receipt.set_thresholds(/* ... */);
// ... rest of receipt building
```

**Test**:
```bash
# AC8.1: Receipts include tokenizer authority
parity-both --model test.gguf --tokenizer tokenizer.json --out-dir /tmp/test

jq '.tokenizer_authority' /tmp/test/receipt_bitnet.json
# Output: { "source": "external", "path": "tokenizer.json", ... }

jq '.tokenizer_authority' /tmp/test/receipt_llama.json
# Output: { "source": "external", "path": "tokenizer.json", ... }

# AC8.2: Tokenizer authority is identical across lanes
diff <(jq -S '.tokenizer_authority' /tmp/test/receipt_bitnet.json) \
     <(jq -S '.tokenizer_authority' /tmp/test/receipt_llama.json)
# Output: (no diff - identical)
```

### AC9: Prompt Template Auto-Detection

**Status**: ❌ NOT STARTED (placeholder returns Raw)

**Current Problem**: `--prompt-template auto` always falls back to `Raw`.

**Expected Flow**:
1. Load GGUF metadata
2. Check for `chat_template` key
3. Detect special tokens (e.g., `<|eot_id|>` = LLaMA-3)
4. Apply heuristics (model path contains "llama3", "instruct", etc.)
5. Fallback to `Instruct` (safer than `Raw` for most models)

**Implementation** (xtask/src/lib.rs):
```rust
impl PromptTemplateArg {
    /// Convert CLI arg to TemplateType with auto-detection
    pub fn to_template_type(&self, model_path: Option<&Path>) -> bitnet_inference::prompt_template::TemplateType {
        use bitnet_inference::prompt_template::TemplateType;

        match self {
            Self::Auto => {
                // Attempt auto-detection if model path provided
                if let Some(path) = model_path {
                    if let Ok(detected) = auto_detect_template(path) {
                        return detected;
                    }
                }
                // Fallback to Instruct (safer than Raw)
                TemplateType::Instruct
            }
            Self::Raw => TemplateType::Raw,
            Self::Instruct => TemplateType::Instruct,
            Self::Llama3Chat => TemplateType::Llama3Chat,
        }
    }
}

/// Auto-detect prompt template from GGUF metadata
fn auto_detect_template(model_path: &Path) -> Result<bitnet_inference::prompt_template::TemplateType> {
    use bitnet_inference::prompt_template::TemplateType;

    // Priority 1: Check GGUF metadata for chat_template
    if let Ok(gguf_meta) = load_gguf_metadata(model_path) {
        if let Some(template_str) = gguf_meta.get("chat_template") {
            // Detect LLaMA-3 format
            if template_str.contains("<|eot_id|>") || template_str.contains("<|start_header_id|>") {
                return Ok(TemplateType::Llama3Chat);
            }
            // Generic instruct pattern
            if template_str.contains("[INST]") || template_str.contains("### Instruction") {
                return Ok(TemplateType::Instruct);
            }
        }
    }

    // Priority 2: Heuristics from model path
    let path_str = model_path.to_string_lossy().to_lowercase();
    if path_str.contains("llama-3") || path_str.contains("llama3") {
        return Ok(TemplateType::Llama3Chat);
    }
    if path_str.contains("instruct") || path_str.contains("chat") {
        return Ok(TemplateType::Instruct);
    }

    // Priority 3: Fallback to Instruct
    Ok(TemplateType::Instruct)
}
```

**CLI Integration** (main.rs line 3700):
```rust
// Before: prompt_template.to_template_type()
// After:  prompt_template.to_template_type(Some(model_gguf))
```

**Test**:
```rust
#[test]
fn test_auto_detect_llama3_from_metadata() {
    let template = auto_detect_template(Path::new("tests/fixtures/llama3.gguf")).unwrap();
    assert_eq!(template, TemplateType::Llama3Chat);
}

#[test]
fn test_auto_detect_instruct_from_path() {
    let template = auto_detect_template(Path::new("models/bitnet-instruct.gguf")).unwrap();
    assert_eq!(template, TemplateType::Instruct);
}

#[test]
fn test_auto_detect_fallback_to_instruct() {
    let template = auto_detect_template(Path::new("models/unknown.gguf")).unwrap();
    assert_eq!(template, TemplateType::Instruct);
}
```

### AC10: (Optional) --parallel Lanes with FFI Thread-Safety

**Status**: ⏳ DEFERRED (sequential execution sufficient for MVP)

**Requirement**: Run BitNet.cpp and llama.cpp lanes in parallel for faster execution.

**Performance Gain**:
- Sequential: shared_setup (20s) + lane_a (15s) + lane_b (15s) = 50s
- Parallel: shared_setup (20s) + max(lane_a (15s), lane_b (15s)) = 35s
- **Speedup**: ~30% faster

**Implementation**:
```rust
use rayon::prelude::*;

// In run_dual_lanes_and_summarize (after shared setup):
if parallel {
    // Run lanes in parallel with rayon
    let (result_a, result_b) = rayon::join(
        || run_single_lane(CppBackend::BitNet, /* ... */),
        || run_single_lane(CppBackend::Llama, /* ... */),
    );

    // Collect results
    match (result_a, result_b) {
        (Ok(()), Ok(())) => { /* both passed */ }
        (Err(e), Ok(())) => { /* lane A failed */ }
        (Ok(()), Err(e)) => { /* lane B failed */ }
        (Err(e_a), Err(e_b)) => { /* both failed */ }
    }
} else {
    // Sequential execution (default)
    run_single_lane(CppBackend::BitNet, /* ... */)?;
    run_single_lane(CppBackend::Llama, /* ... */)?;
}
```

**FFI Safety Requirement**: Validate that `BitnetSession` is thread-safe (or use thread-local instances).

**Test**:
```bash
# AC10.1: Sequential (default)
parity-both --model test.gguf --tokenizer tokenizer.json
# Expected: Lanes run one after another

# AC10.2: Parallel (opt-in)
parity-both --model test.gguf --tokenizer tokenizer.json --parallel
# Expected: Lanes run concurrently, ~30% faster

# AC10.3: Receipts identical regardless of mode
diff <(parity-both --sequential) <(parity-both --parallel)
# Expected: No differences in receipt content
```

**Deferred Rationale**: Adds complexity (FFI thread-safety validation), minimal speedup for typical 4-token runs. Defer to post-MVP.

---

## 3. CLI Wiring

### 3.1 Command Registration ✅

**Status**: DONE (main.rs lines 560-617)

```rust
#[cfg(feature = "crossval-all")]
#[command(name = "parity-both")]
ParityBoth {
    /// Path to GGUF model file
    #[arg(long)]
    model_gguf: PathBuf,

    /// Path to tokenizer.json
    #[arg(long)]
    tokenizer: PathBuf,

    /// Input prompt
    #[arg(long, default_value = "What is 2+2?")]
    prompt: String,

    /// Maximum tokens to generate (excluding prompt)
    #[arg(long, default_value_t = 4)]
    max_tokens: usize,

    /// Cosine similarity threshold (0.0-1.0)
    #[arg(long, default_value_t = 0.999)]
    cos_tol: f64,

    /// Output directory for receipts
    #[arg(long, default_value = ".parity")]
    out_dir: PathBuf,

    /// Output format: "text" or "json"
    #[arg(long, default_value = "text")]
    format: String,

    /// Prompt template: "raw", "instruct", "llama3-chat", "auto"
    #[arg(long, default_value = "auto")]
    prompt_template: PromptTemplateArg,

    /// System prompt (for chat templates)
    #[arg(long)]
    system_prompt: Option<String>,

    /// Disable auto-repair (fail fast if backend missing)
    #[arg(long)]
    no_repair: bool,

    /// Verbose output
    #[arg(long)]
    verbose: bool,

    /// Dump Rust token IDs to stderr
    #[arg(long)]
    dump_ids: bool,

    /// Dump C++ token IDs to stderr
    #[arg(long)]
    dump_cpp_ids: bool,

    /// Comparison metric: "mse" (default)
    #[arg(long, default_value = "mse")]
    metrics: String,
}
```

### 3.2 Handler Dispatch ✅

**Status**: DONE (main.rs lines 1115-1148)

```rust
Cmd::ParityBoth {
    model_gguf, tokenizer, prompt, max_tokens, cos_tol,
    out_dir, format, prompt_template, system_prompt,
    no_repair, verbose, dump_ids, dump_cpp_ids, metrics,
} => {
    parity_both_cmd(
        &model_gguf, &tokenizer, &prompt, max_tokens, cos_tol,
        &out_dir, &format, prompt_template, system_prompt.as_deref(),
        !no_repair,  // auto_repair = !no_repair
        verbose, dump_ids, dump_cpp_ids, &metrics,
    )?;
    Ok(())
}
```

### 3.3 Command Wrapper Enhancement

**Status**: 🚧 NEEDS UPDATE (prompt template auto-detection)

**Current** (main.rs line 3700):
```rust
parity_both::run(&parity_both::ParityBothArgs {
    // ... other fields ...
    prompt_template: prompt_template.to_template_type(),  // ⚠️ Missing model_path
    // ... other fields ...
})?;
```

**Fixed**:
```rust
parity_both::run(&parity_both::ParityBothArgs {
    model_gguf: model_gguf.to_path_buf(),
    tokenizer: tokenizer.to_path_buf(),
    prompt: prompt.to_string(),
    max_tokens,
    cos_tol,
    out_dir: out_dir.to_path_buf(),
    format: format.to_string(),
    prompt_template: prompt_template.to_template_type(Some(model_gguf)),  // ✅ Pass model path
    system_prompt: system_prompt.map(|s| s.to_string()),
    auto_repair,
    verbose,
    dump_ids,
    dump_cpp_ids,
    metrics: metrics.to_string(),
})?;
```

---

## 4. Preflight Integration Architecture

### 4.1 RepairMode Decision Tree

```
User Invocation: parity-both [flags]
  │
  ├─ No flags (default)
  │  └─ is_ci_environment()?
  │     ├─ Yes (CI=true) → RepairMode::Never (safe default for CI)
  │     └─ No (local dev) → RepairMode::Auto (user-friendly)
  │
  ├─ --no-repair
  │  └─ RepairMode::Never (explicit opt-out)
  │
  └─ Backend check:
     ├─ RepairMode::Auto + backend missing → Attempt auto-repair
     ├─ RepairMode::Never + backend missing → Fail with exit code 2
     └─ Backend available → Continue
```

### 4.2 Preflight Flow with Auto-Repair

```
STEP 1: Preflight Both Backends
  ├─ Check BitNet.cpp (HAS_BITNET constant)
  │  ├─ Available → Continue
  │  ├─ Missing + RepairMode::Never → Exit code 2 (show setup instructions)
  │  └─ Missing + RepairMode::Auto:
  │     ├─ Invoke: setup-cpp-auto --emit=sh
  │     ├─ Parse exports: BITNET_CPP_DIR, LD_LIBRARY_PATH
  │     ├─ Apply to environment
  │     ├─ (Optional) Rebuild xtask: cargo build -p xtask --features crossval-all
  │     └─ Retry preflight check
  │        ├─ Success → Continue
  │        └─ Failure → Exit code 1 (repair failed)
  │
  ├─ Check llama.cpp (HAS_LLAMA constant)
  │  └─ Same logic as BitNet.cpp
  │
  └─ Both available → Continue to STEP 2

STEP 2: Shared Setup
  ├─ Load Rust tokenizer (tokenizer.json)
  ├─ Compute tokenizer authority:
  │  ├─ Detect source (GGUF embedded vs external)
  │  ├─ Compute file hash (SHA256 of tokenizer.json)
  │  └─ Compute config hash (SHA256 of vocab)
  ├─ Auto-detect prompt template (if --prompt-template=auto)
  ├─ Apply prompt template transformation
  ├─ Tokenize with Rust tokenizer
  └─ Evaluate Rust logits (shared baseline)

STEP 3: Dual Lanes (Sequential or Parallel)
  ├─ Lane A: BitNet.cpp
  │  ├─ Tokenize with C++ tokenizer
  │  ├─ VALIDATE: C++ tokens == Rust tokens (AC7)
  │  │  └─ If mismatch → Exit code 2 (tokenizer parity violation)
  │  ├─ Evaluate C++ logits
  │  ├─ Compare Rust vs C++ logits (MSE, cosine, L2)
  │  └─ Generate receipt_bitnet.json (with tokenizer authority)
  │
  └─ Lane B: llama.cpp
     └─ Same flow as Lane A → receipt_llama.json

STEP 4: Unified Summary
  ├─ Load both receipts
  ├─ Check tokenizer authority consistency (config_hash must match)
  ├─ Print summary (text or JSON format)
  └─ Exit code:
     ├─ 0: Both lanes pass
     ├─ 1: Either lane fails
     └─ 2: Token parity mismatch or usage error
```

### 4.3 Error Exit Codes

| Exit Code | Meaning | Scenario |
|-----------|---------|----------|
| 0 | Success | Both lanes pass parity checks |
| 1 | Lane failure | One or both lanes fail parity checks (MSE > threshold) |
| 2 | Usage error | Token parity mismatch, invalid args, backend unavailable + repair disabled |
| 3 | Network failure | Auto-repair failed due to network error (after retries) |
| 4 | Permission denied | Auto-repair failed due to permission error |
| 5 | Build failure | Auto-repair failed due to build error |
| 6 | Recursion detected | Auto-repair recursion guard triggered |

---

## 5. TokenizerAuthority Schema

### 5.1 Data Structure

```rust
/// Tokenizer authority metadata for receipt reproducibility
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TokenizerAuthority {
    /// Tokenizer source: "gguf_embedded" | "external" | "auto_discovered"
    pub source: String,

    /// Path to tokenizer (GGUF path or tokenizer.json path)
    pub path: String,

    /// SHA256 hash of tokenizer.json file (if external)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub file_hash: Option<String>,

    /// SHA256 hash of effective tokenizer config (canonical JSON)
    pub config_hash: String,

    /// Token count (for quick validation)
    pub token_count: usize,
}
```

### 5.2 Integration with ParityReceipt

```rust
// In crossval/src/receipt.rs
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

    // NEW: v2 fields (optional for backward compatibility)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tokenizer_authority: Option<TokenizerAuthority>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_template: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub determinism_seed: Option<u64>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_sha256: Option<String>,
}

impl ParityReceipt {
    /// Set tokenizer authority metadata
    pub fn set_tokenizer_authority(&mut self, authority: TokenizerAuthority) {
        self.tokenizer_authority = Some(authority);
    }

    /// Set prompt template used
    pub fn set_prompt_template(&mut self, template: String) {
        self.prompt_template = Some(template);
    }

    /// Infer schema version based on fields present
    pub fn infer_version(&self) -> &str {
        match (&self.tokenizer_authority, &self.prompt_template) {
            (Some(_), _) | (_, Some(_)) => "2.0.0",
            _ => "1.0.0",
        }
    }
}
```

### 5.3 JSON Example (v2 Receipt)

```json
{
  "version": 1,
  "timestamp": "2025-10-26T14:30:00Z",
  "model": "models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf",
  "backend": "bitnet",
  "prompt": "[INST] What is 2+2? [/INST]",
  "positions": 4,
  "tokenizer_authority": {
    "source": "external",
    "path": "models/microsoft-bitnet-b1.58-2B-4T-gguf/tokenizer.json",
    "file_hash": "6f3ef9d7a3c2b1e0d4f5a8c9b0e1d2f3a4b5c6d7e8f9a0b1c2d3e4f5a6b7c8d9",
    "config_hash": "a1b2c3d4e5f6789012345678901234567890abcdefabcdefabcdefabcdefabc0",
    "token_count": 4
  },
  "prompt_template": "instruct",
  "model_sha256": "fedcba9876543210fedcba9876543210fedcba9876543210fedcba9876543210",
  "thresholds": {
    "mse": 0.0001,
    "kl": 0.1,
    "topk": 0.8
  },
  "rows": [
    {
      "pos": 0,
      "mse": 0.000015,
      "max_abs": 0.00042,
      "kl": 0.0023,
      "topk_agree": 1.0,
      "top5_rust": [128000, 1229, 374, 220, 17],
      "top5_cpp": [128000, 1229, 374, 220, 17]
    }
  ],
  "summary": {
    "all_passed": true,
    "first_divergence": null,
    "mean_mse": 0.000013,
    "mean_kl": 0.0021
  }
}
```

---

## 6. Parity Validation Algorithm

### 6.1 Token Sequence Validation

```rust
/// Validate tokenizer parity between Rust and C++ implementations
///
/// Checks:
/// 1. Token count matches
/// 2. Each token ID matches at every position
///
/// Returns Ok(()) if tokens are identical, Err with detailed diagnostics otherwise.
pub fn validate_tokenizer_parity(
    rust_tokens: &[u32],
    cpp_tokens: &[u32],
    backend_name: &str,
) -> Result<()> {
    // Check 1: Length parity
    if rust_tokens.len() != cpp_tokens.len() {
        return Err(anyhow::anyhow!(
            "Tokenizer parity mismatch for {}: Rust {} tokens vs C++ {} tokens",
            backend_name,
            rust_tokens.len(),
            cpp_tokens.len()
        ));
    }

    // Check 2: Token-by-token comparison
    for (i, (r_token, c_token)) in rust_tokens.iter().zip(cpp_tokens.iter()).enumerate() {
        if r_token != c_token {
            return Err(anyhow::anyhow!(
                "Tokenizer divergence for {} at position {}: Rust token={}, C++ token={}",
                backend_name,
                i,
                r_token,
                c_token
            ));
        }
    }

    Ok(())
}
```

### 6.2 Tokenizer Authority Consistency

```rust
/// Validate tokenizer authority consistency between two lanes
///
/// Both lanes must use the same effective tokenizer config for valid parity comparison.
pub fn validate_tokenizer_consistency(
    lane_a: &TokenizerAuthority,
    lane_b: &TokenizerAuthority,
) -> Result<()> {
    // Config hash must match (effective tokenizer is identical)
    if lane_a.config_hash != lane_b.config_hash {
        return Err(anyhow::anyhow!(
            "Tokenizer config mismatch: Lane A hash={}, Lane B hash={}",
            lane_a.config_hash,
            lane_b.config_hash
        ));
    }

    // Token count should match (sanity check)
    if lane_a.token_count != lane_b.token_count {
        return Err(anyhow::anyhow!(
            "Token count mismatch: Lane A={}, Lane B={}",
            lane_a.token_count,
            lane_b.token_count
        ));
    }

    Ok(())
}
```

---

## 7. Receipt Generation

### 7.1 Per-Lane Receipt with Tokenizer Authority

```rust
// In parity_both.rs::run_single_lane() (line 641)

// Step 1: Create receipt
let mut receipt = bitnet_crossval::receipt::ParityReceipt::new(
    model_path.to_str().context("Invalid model path")?,
    backend.name(),
    formatted_prompt,
);

// Step 2: Set tokenizer authority (captured during shared setup)
receipt.set_tokenizer_authority(tokenizer_authority.clone());

// Step 3: Set prompt template
receipt.set_prompt_template(prompt_template.clone());

// Step 4: (Optional) Set determinism seed
if let Ok(seed_str) = std::env::var("BITNET_SEED") {
    if let Ok(seed) = seed_str.parse::<u64>() {
        receipt.determinism_seed = Some(seed);
    }
}

// Step 5: (Optional) Compute model hash
if let Ok(model_hash) = compute_gguf_sha256(model_path) {
    receipt.model_sha256 = Some(model_hash);
}

// Step 6: Set thresholds
receipt.set_thresholds(bitnet_crossval::receipt::Thresholds {
    mse: 0.0001,
    kl: 0.1,
    topk: 0.8,
});

// Step 7: Add per-position metrics
for (pos, metrics) in divergence_data.iter().enumerate() {
    receipt.add_position(bitnet_crossval::receipt::PositionMetrics {
        pos,
        mse: metrics.mse,
        max_abs: metrics.max_abs,
        kl: Some(metrics.kl),
        topk_agree: Some(metrics.topk_agree),
        top5_rust: metrics.top5_rust.clone(),
        top5_cpp: metrics.top5_cpp.clone(),
    });
}

// Step 8: Finalize and write
receipt.finalize();
receipt.write_to_file(receipt_path)?;
```

### 7.2 Dual Receipts Naming

```
Output directory: {out_dir}/ (default: .parity/)
  ├── receipt_bitnet.json    (Lane A: BitNet.cpp)
  ├── receipt_llama.json     (Lane B: llama.cpp)
  └── (future) receipt_merged.json (Merged dual-lane receipt)
```

### 7.3 Summary with Tokenizer Metadata

**Text Format**:
```
Parity-Both Cross-Validation Summary
═════════════════════════════════════════════════════

Tokenizer Authority
─────────────────────────────────────────────────────
Source:           external
Path:             models/tokenizer.json
File hash:        6f3ef9d7a3c2...
Config hash:      a1b2c3d4e5f6...
Token count:      5

Lane A: BitNet.cpp
─────────────────────────────────────────────────────
Backend:          bitnet.cpp
Status:           ✓ Parity OK
First divergence: None
Mean MSE:         2.15e-5
Receipt:          .parity/receipt_bitnet.json

Lane B: llama.cpp
─────────────────────────────────────────────────────
Backend:          llama.cpp
Status:           ✓ Parity OK
First divergence: None
Mean MSE:         1.98e-5
Receipt:          .parity/receipt_llama.json

Overall Status
─────────────────────────────────────────────────────
Both lanes:       ✓ PASSED
Tokenizer:        ✓ Consistent across lanes
Exit code:        0
```

**JSON Format**:
```json
{
  "tokenizer_authority": {
    "source": "external",
    "path": "models/tokenizer.json",
    "file_hash": "6f3ef9d7a3c2...",
    "config_hash": "a1b2c3d4e5f6...",
    "token_count": 5
  },
  "lanes": {
    "bitnet": {
      "backend": "bitnet.cpp",
      "passed": true,
      "first_divergence": null,
      "mean_mse": 0.0000215,
      "receipt": ".parity/receipt_bitnet.json"
    },
    "llama": {
      "backend": "llama.cpp",
      "passed": true,
      "first_divergence": null,
      "mean_mse": 0.0000198,
      "receipt": ".parity/receipt_llama.json"
    }
  },
  "overall": {
    "both_passed": true,
    "tokenizer_consistent": true,
    "exit_code": 0
  }
}
```

---

## 8. Testing Strategy

### 8.1 Unit Tests (Helpers & Schema)

```rust
// In crossval/src/receipt.rs

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenizer_authority_serialization() {
        let auth = TokenizerAuthority {
            source: "external".to_string(),
            path: "tokenizer.json".to_string(),
            file_hash: Some("abc123".to_string()),
            config_hash: "def456".to_string(),
            token_count: 5,
        };

        let json = serde_json::to_string(&auth).unwrap();
        let deserialized: TokenizerAuthority = serde_json::from_str(&json).unwrap();

        assert_eq!(auth, deserialized);
    }

    #[test]
    fn test_parity_receipt_with_tokenizer_authority() {
        let mut receipt = ParityReceipt::new("model.gguf", "bitnet", "What is 2+2?");

        receipt.set_tokenizer_authority(TokenizerAuthority {
            source: "external".to_string(),
            path: "tokenizer.json".to_string(),
            file_hash: Some("hash123".to_string()),
            config_hash: "hash456".to_string(),
            token_count: 5,
        });

        receipt.set_prompt_template("instruct".to_string());
        receipt.finalize();

        // Verify serialization
        let json = serde_json::to_string_pretty(&receipt).unwrap();
        assert!(json.contains("tokenizer_authority"));
        assert!(json.contains("hash123"));
        assert!(json.contains("prompt_template"));
        assert!(json.contains("instruct"));
    }

    #[test]
    fn test_receipt_backward_compatibility() {
        // Old v1 receipt without tokenizer authority
        let json_v1 = r#"{
            "version": 1,
            "timestamp": "2025-10-26T14:30:00Z",
            "model": "model.gguf",
            "backend": "bitnet",
            "prompt": "test",
            "positions": 1,
            "thresholds": { "mse": 0.0001, "kl": 0.1, "topk": 0.8 },
            "rows": [],
            "summary": { "all_passed": true, "first_divergence": null, "mean_mse": 0.0, "mean_kl": null }
        }"#;

        let receipt: ParityReceipt = serde_json::from_str(json_v1).unwrap();
        assert!(receipt.tokenizer_authority.is_none());
        assert_eq!(receipt.infer_version(), "1.0.0");
    }
}
```

### 8.2 Integration Tests (parity-both Command)

```rust
// In xtask/tests/parity_both_integration_tests.rs

#[test]
#[ignore] // Requires model and backend setup
fn test_parity_both_generates_receipts_with_tokenizer_authority() {
    let temp_dir = tempfile::tempdir().unwrap();
    let out_dir = temp_dir.path().join("receipts");

    // Run parity-both command
    let output = Command::new("cargo")
        .args(&[
            "run", "-p", "xtask", "--features", "crossval-all", "--",
            "parity-both",
            "--model-gguf", "tests/fixtures/test_model.gguf",
            "--tokenizer", "tests/fixtures/tokenizer.json",
            "--out-dir", out_dir.to_str().unwrap(),
            "--prompt", "Test",
            "--max-tokens", "1",
        ])
        .output()
        .expect("Failed to run parity-both");

    assert!(output.status.success(), "parity-both should succeed");

    // Verify receipt_bitnet.json exists
    let receipt_bitnet_path = out_dir.join("receipt_bitnet.json");
    assert!(receipt_bitnet_path.exists(), "receipt_bitnet.json should exist");

    // Verify receipt contains tokenizer authority
    let receipt_json = std::fs::read_to_string(&receipt_bitnet_path).unwrap();
    let receipt: serde_json::Value = serde_json::from_str(&receipt_json).unwrap();

    assert!(receipt["tokenizer_authority"].is_object());
    assert_eq!(receipt["tokenizer_authority"]["source"], "external");
    assert!(receipt["tokenizer_authority"]["config_hash"].is_string());

    // Verify receipt_llama.json also has tokenizer authority
    let receipt_llama_path = out_dir.join("receipt_llama.json");
    let receipt_llama_json = std::fs::read_to_string(&receipt_llama_path).unwrap();
    let receipt_llama: serde_json::Value = serde_json::from_str(&receipt_llama_json).unwrap();

    // Verify tokenizer authority is identical across lanes
    assert_eq!(
        receipt["tokenizer_authority"]["config_hash"],
        receipt_llama["tokenizer_authority"]["config_hash"]
    );
}

#[test]
#[serial(bitnet_env)]
fn test_parity_both_auto_repair_when_backend_missing() {
    // Mock missing backend by unsetting BITNET_CPP_DIR
    std::env::remove_var("BITNET_CPP_DIR");

    let output = Command::new("cargo")
        .args(&[
            "run", "-p", "xtask", "--features", "crossval-all", "--",
            "parity-both",
            "--model-gguf", "tests/fixtures/test_model.gguf",
            "--tokenizer", "tests/fixtures/tokenizer.json",
            "--prompt", "Test",
            "--max-tokens", "1",
        ])
        .output()
        .expect("Failed to run parity-both");

    // Should attempt auto-repair and succeed (or show repair instructions)
    assert!(
        output.status.success() || output.status.code() == Some(1),
        "Should attempt auto-repair or show instructions"
    );
}

#[test]
fn test_parity_both_no_repair_fails_fast() {
    // Mock missing backend
    std::env::remove_var("BITNET_CPP_DIR");

    let output = Command::new("cargo")
        .args(&[
            "run", "-p", "xtask", "--features", "crossval-all", "--",
            "parity-both",
            "--model-gguf", "tests/fixtures/test_model.gguf",
            "--tokenizer", "tests/fixtures/tokenizer.json",
            "--no-repair",
        ])
        .output()
        .expect("Failed to run parity-both");

    // Should fail fast with exit code 2 (backend unavailable)
    assert_eq!(output.status.code(), Some(2));

    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("backend not found"));
    assert!(stderr.contains("setup-cpp-auto"));
}
```

### 8.3 E2E Test Script

```bash
#!/bin/bash
# tests/e2e/test_parity_both_full_flow.sh

set -e

MODEL="models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf"
TOKENIZER="models/microsoft-bitnet-b1.58-2B-4T-gguf/tokenizer.json"
OUT_DIR="/tmp/parity-both-e2e-$(date +%s)"

echo "🧪 E2E Test: parity-both full flow"

# Test 1: Run parity-both with auto-repair
echo "Test 1: Running parity-both with auto-repair enabled..."
cargo run -p xtask --features crossval-all -- parity-both \
  --model-gguf "$MODEL" \
  --tokenizer "$TOKENIZER" \
  --out-dir "$OUT_DIR" \
  --prompt "What is 2+2?" \
  --max-tokens 4 \
  --verbose

# Verify exit code
if [ $? -ne 0 ]; then
  echo "❌ parity-both command failed"
  exit 1
fi

# Verify receipts exist
echo "Test 2: Verifying receipts exist..."
if [ ! -f "$OUT_DIR/receipt_bitnet.json" ]; then
  echo "❌ receipt_bitnet.json not found"
  exit 1
fi

if [ ! -f "$OUT_DIR/receipt_llama.json" ]; then
  echo "❌ receipt_llama.json not found"
  exit 1
fi

# Verify receipts are valid JSON
echo "Test 3: Verifying receipts are valid JSON..."
jq . "$OUT_DIR/receipt_bitnet.json" > /dev/null || {
  echo "❌ receipt_bitnet.json is invalid JSON"
  exit 1
}

jq . "$OUT_DIR/receipt_llama.json" > /dev/null || {
  echo "❌ receipt_llama.json is invalid JSON"
  exit 1
}

# Verify tokenizer authority present
echo "Test 4: Verifying tokenizer authority present..."
BITNET_AUTH=$(jq -r '.tokenizer_authority' "$OUT_DIR/receipt_bitnet.json")
if [ "$BITNET_AUTH" == "null" ]; then
  echo "❌ tokenizer_authority missing in receipt_bitnet.json"
  exit 1
fi

LLAMA_AUTH=$(jq -r '.tokenizer_authority' "$OUT_DIR/receipt_llama.json")
if [ "$LLAMA_AUTH" == "null" ]; then
  echo "❌ tokenizer_authority missing in receipt_llama.json"
  exit 1
fi

# Verify tokenizer config hash matches across lanes
echo "Test 5: Verifying tokenizer consistency..."
BITNET_HASH=$(jq -r '.tokenizer_authority.config_hash' "$OUT_DIR/receipt_bitnet.json")
LLAMA_HASH=$(jq -r '.tokenizer_authority.config_hash' "$OUT_DIR/receipt_llama.json")

if [ "$BITNET_HASH" != "$LLAMA_HASH" ]; then
  echo "❌ Tokenizer config hash mismatch:"
  echo "  BitNet: $BITNET_HASH"
  echo "  llama:  $LLAMA_HASH"
  exit 1
fi

# Verify prompt template present
echo "Test 6: Verifying prompt template..."
TEMPLATE=$(jq -r '.prompt_template' "$OUT_DIR/receipt_bitnet.json")
if [ "$TEMPLATE" == "null" ]; then
  echo "⚠️  prompt_template missing (acceptable for MVP)"
else
  echo "✓ prompt_template: $TEMPLATE"
fi

echo "✅ All E2E tests passed"
echo "Receipts written to: $OUT_DIR"
```

---

## 9. Implementation Roadmap

### Phase 1: Preflight Integration (6-8 hours)

**Goal**: Auto-repair functionality working end-to-end

**Tasks**:
1. ✅ Implement `preflight_both_backends()` in preflight.rs (AC2)
2. ✅ Integrate with parity_both.rs (replace current loop)
3. ✅ Honor `--no-repair` flag (AC3)
4. ✅ Test auto-repair flow with mocked missing backends
5. ⏳ (Optional) Implement xtask rebuild + re-exec

**Deliverables**:
- `preflight_both_backends()` function operational
- Auto-repair tests passing (AC2, AC3)
- Exit code semantics validated

**Estimated Effort**: 6-8 hours

### Phase 2: Tokenizer Authority Schema (4-5 hours)

**Goal**: TokenizerAuthority struct integrated with ParityReceipt

**Tasks**:
1. ✅ Define `TokenizerAuthority` struct in crossval/src/receipt.rs (AC4)
2. ✅ Add optional fields to ParityReceipt (backward compatible)
3. ✅ Implement builder API: `set_tokenizer_authority()`, `set_prompt_template()`
4. ✅ Implement `compute_tokenizer_file_hash()` (AC6)
5. ✅ Implement `compute_tokenizer_config_hash()` (AC6)
6. ✅ Unit tests for serialization and hash determinism

**Deliverables**:
- TokenizerAuthority struct with tests
- ParityReceipt v2 schema (backward compatible)
- Hash computation utilities

**Estimated Effort**: 4-5 hours

### Phase 3: Parity-Both Integration (4-6 hours)

**Goal**: Tokenizer authority captured and validated in dual-lane execution

**Tasks**:
1. ✅ Capture tokenizer authority during shared setup (AC8)
2. ✅ Implement `validate_tokenizer_parity()` (AC7)
3. ✅ Integrate tokenizer authority into receipt generation
4. ✅ Update summary output to show tokenizer metadata
5. ✅ Integration tests for parity-both with tokenizer authority

**Deliverables**:
- Tokenizer authority in both lane receipts
- Token-by-token parity validation
- Summary output includes tokenizer section

**Estimated Effort**: 4-6 hours

### Phase 4: Template Auto-Detection (2-3 hours)

**Goal**: `--prompt-template auto` correctly detects from GGUF metadata

**Tasks**:
1. ✅ Implement `auto_detect_template(model_path)` (AC9)
2. ✅ Update `PromptTemplateArg::to_template_type()` to accept model path
3. ✅ Fix CLI wiring in main.rs (pass model_path to conversion)
4. ✅ Test auto-detection with LLaMA-3, Instruct, and generic models

**Deliverables**:
- Auto-detection from GGUF `chat_template` metadata
- Fallback to Instruct (safer than Raw)
- Test coverage for detection heuristics

**Estimated Effort**: 2-3 hours

### Phase 5: Testing & Documentation (2-4 hours)

**Goal**: Comprehensive test coverage and user documentation

**Tasks**:
1. ✅ E2E test script for full parity-both flow
2. ✅ Integration tests for all AC (AC1-AC9)
3. ✅ Update CLAUDE.md with parity-both examples
4. ✅ Add troubleshooting guide for tokenizer mismatches

**Deliverables**:
- E2E test script passing
- Integration tests for AC1-AC9
- Documentation updates

**Estimated Effort**: 2-4 hours

---

## 10. Success Criteria Summary

| AC | Requirement | Status | Evidence |
|----|-----------|--------|----------|
| AC1 | CLI registration complete | ✅ DONE | main.rs lines 560-617 |
| AC2 | Preflight both backends with auto-repair | 🚧 PHASE 1 | preflight.rs implementation needed |
| AC3 | RepairMode honored (--no-repair) | 🚧 PHASE 1 | Flag threaded, logic needed |
| AC4 | TokenizerAuthority struct in schema | ❌ PHASE 2 | Define in receipt.rs |
| AC5 | Tokenizer source tracking | ❌ PHASE 2 | Detect GGUF vs JSON |
| AC6 | SHA256 hash for tokenizer config | ❌ PHASE 2 | Hash computation utilities |
| AC7 | Tokenizer parity validation | 🚧 PHASE 3 | Enhance with token-by-token check |
| AC8 | Dual receipts with tokenizer metadata | 🚧 PHASE 3 | Integrate in receipt generation |
| AC9 | Prompt template auto-detection | ❌ PHASE 4 | Implement GGUF metadata inspection |
| AC10 | (Optional) --parallel lanes | ⏳ DEFERRED | Post-MVP feature |

**Completion Estimate**: 14-21 hours (2-3 focused dev days)

**Risk Assessment**: LOW (all components have clear implementation paths, backward compatibility maintained)

---

## 11. Documentation & Examples

### 11.1 CLAUDE.md Integration

```markdown
## Parity-Both Cross-Validation

Run cross-validation against both BitNet.cpp and llama.cpp in a single command:

```bash
# Basic usage (auto-repair enabled by default)
cargo run -p xtask --features crossval-all -- parity-both \
  --model-gguf models/model.gguf \
  --tokenizer models/tokenizer.json

# With custom prompt and template
cargo run -p xtask --features crossval-all -- parity-both \
  --model-gguf models/model.gguf \
  --tokenizer models/tokenizer.json \
  --prompt "What is 2+2?" \
  --prompt-template instruct \
  --max-tokens 8

# Disable auto-repair (fail fast if backend missing)
cargo run -p xtask --features crossval-all -- parity-both \
  --model-gguf models/model.gguf \
  --tokenizer models/tokenizer.json \
  --no-repair

# JSON output for CI integration
cargo run -p xtask --features crossval-all -- parity-both \
  --model-gguf models/model.gguf \
  --tokenizer models/tokenizer.json \
  --format json \
  --out-dir ci/receipts/
```

### 11.2 Troubleshooting Guide

**Problem**: Tokenizer parity mismatch

```
❌ Tokenizer divergence for bitnet.cpp at position 2: Rust token=3, C++ token=99
```

**Solutions**:
1. Verify tokenizer.json version matches model
2. Check GGUF metadata for embedded tokenizer conflicts
3. Use `--dump-ids --dump-cpp-ids --verbose` for diagnostics
4. Ensure same tokenizer used for both backends

**Problem**: Auto-repair fails with network error

```
❌ Auto-repair failed for bitnet.cpp: Network error (after 3 retries)
```

**Solutions**:
1. Check internet connection: `ping github.com`
2. Verify firewall allows git clone
3. Manually provision: `cargo run -p xtask -- setup-cpp-auto`
4. Use `--no-repair` to skip auto-repair

---

## Appendix A: File Locations

**Core Implementation**:
- `xtask/src/crossval/parity_both.rs` (924 lines) - Dual-lane orchestration
- `xtask/src/crossval/preflight.rs` (1525 lines) - Auto-repair infrastructure
- `crossval/src/receipt.rs` (300+ lines) - ParityReceipt schema

**CLI Integration**:
- `xtask/src/main.rs` (lines 560-617, 1115-1148, 3669-3710)
- `xtask/src/lib.rs` (lines 23-45) - PromptTemplateArg conversion

**Tests**:
- `xtask/tests/parity_both_tests.rs` (838 lines, TDD scaffolding)
- `xtask/tests/preflight_auto_repair_tests.rs` (838 lines, AC1-AC7)

**Documentation**:
- `docs/specs/parity-both-command.md` (existing spec)
- `docs/specs/preflight-auto-repair.md` (existing spec)

---

## Appendix B: Exit Code Reference

| Exit Code | Name | Condition |
|-----------|------|-----------|
| 0 | Success | Both lanes pass parity checks |
| 1 | Lane failure | One or both lanes fail parity checks |
| 2 | Usage error | Token parity mismatch, invalid args, backend unavailable + repair disabled |
| 3 | Network failure | Auto-repair failed due to network error (after retries) |
| 4 | Permission denied | Auto-repair failed due to permission error |
| 5 | Build failure | Auto-repair failed due to build error |
| 6 | Recursion detected | Auto-repair recursion guard triggered |

---

## Appendix C: Glossary

| Term | Definition |
|------|-----------|
| **Parity-Both** | Dual-lane cross-validation command (BitNet.cpp + llama.cpp) |
| **Lane** | Single C++ backend evaluation (Lane A = BitNet.cpp, Lane B = llama.cpp) |
| **TokenizerAuthority** | Metadata tracking tokenizer source, hash, and configuration |
| **RepairMode** | Auto-repair control: Auto (repair if missing), Never (fail fast), Always (force refresh) |
| **Preflight** | Pre-execution backend availability check with auto-repair |
| **Receipt** | JSON document recording parity metrics and execution metadata |
| **Config Hash** | SHA256 hash of effective tokenizer configuration (canonical JSON) |
| **File Hash** | SHA256 hash of tokenizer.json file |
| **Tokenizer Parity** | Requirement that Rust and C++ tokenizers produce identical token sequences |

---

**Document Complete**: 850+ lines, comprehensive specification with AC1-AC10 coverage

**Next Steps**: Begin Phase 1 (Preflight Integration) implementation

