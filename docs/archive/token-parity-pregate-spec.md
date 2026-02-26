# Token Parity Pre-Gate Specification

**Feature:** Token Sequence Validation for Cross-Validation Commands
**Component:** `crossval-per-token` command enhancement
**Priority:** Critical (blocks reliable cross-validation)
**Status:** Specification → Implementation Ready
**Version:** 1.0.0

---

## Table of Contents

1. [Problem Statement](#problem-statement)
2. [User Stories](#user-stories)
3. [Scope](#scope)
4. [Technical Requirements](#technical-requirements)
5. [Acceptance Criteria](#acceptance-criteria)
6. [Architecture Design](#architecture-design)
7. [API Contracts](#api-contracts)
8. [Integration Points](#integration-points)
9. [Error Handling](#error-handling)
10. [Testing Strategy](#testing-strategy)
11. [Performance Requirements](#performance-requirements)
12. [Risks and Mitigations](#risks-and-mitigations)
13. [Future Enhancements](#future-enhancements)

---

## Problem Statement

### Current Behavior

The `crossval-per-token` command compares Rust and C++ logits without validating that both implementations operate on identical token sequences. This leads to:

1. **False divergence reports**: Different tokenization produces different logits, masking as inference bugs
2. **Wasted debugging time**: Users compare incomparable logits and chase phantom issues
3. **Unreliable baselines**: Cross-validation receipts show divergence when tokenization differs

### Root Cause Analysis

**Location**: `xtask/src/main.rs` lines 2918-2974

```rust
// Current implementation (PROBLEMATIC)
// Rust tokenization: raw, no BOS, no special tokens
let tokens = tokenizer.encode(prompt, false, false)?;

// C++ tokenization: llama.cpp with special handling
let cpp_tokens = cpp_session.tokenize(prompt)?;

// NO TOKEN VALIDATION - proceeds directly to logits comparison!
let rust_logits = eval_logits_all_positions(model_path_str, &token_ids)?;
let cpp_logits = cpp_session.context.get_all_logits(cpp_tokens.len())?;
```

**Tokenization Differences**:
- **Rust**: `encode(prompt, add_bos=false, add_special=false)` → raw tokens only
- **C++**: `Session::tokenize(prompt)` with `add_special=true` → includes special token handling
- **Result**: Different token sequences → incomparable logits → false divergence

### Business Impact

- **Developer productivity loss**: ~30-60 minutes per false divergence investigation
- **Cross-validation unreliability**: Cannot trust parity metrics when tokens differ
- **User confusion**: Error messages point to layer divergence, not tokenization root cause
- **Technical debt**: Workarounds in downstream tools to compensate for missing validation

---

## User Stories

### US1: Early Detection of Tokenization Mismatches

**As a** BitNet-rs developer running cross-validation
**I want** token parity validation before logits comparison
**So that** I immediately know when tokenization differs between Rust and C++

**Business Value**: Saves 30-60 minutes per debugging session by failing fast with actionable errors

**Acceptance Criteria**:
- AC1: Token mismatch detected before any logits evaluation
- AC2: Error message shows both token sequences
- AC3: Error includes first-diff position (index)
- AC4: Exit code 2 (usage error) on token mismatch

---

### US2: Actionable Guidance for Token Mismatches

**As a** BitNet-rs user encountering token parity errors
**I want** clear fix-it suggestions in error messages
**So that** I can resolve tokenization issues without consulting documentation

**Business Value**: Reduces support burden and accelerates debugging workflows

**Acceptance Criteria**:
- AC5: Error suggests `--prompt-template raw` for raw tokenization
- AC6: Error suggests `--no-bos` if BOS duplication detected
- AC7: Error suggests checking GGUF metadata for template conflicts
- AC8: Error provides example command with suggested fixes

---

### US3: Clean Output When Tokens Match

**As a** BitNet-rs developer with correctly matched tokens
**I want** silent token validation (no noise)
**So that** cross-validation output remains clean and focused on logits comparison

**Business Value**: Maintains clean CI/CD output and reduces log noise

**Acceptance Criteria**:
- AC9: No output when tokens match exactly
- AC10: Validation overhead <100ms for typical prompts (<1000 tokens)
- AC11: Proceeds directly to logits comparison on success

---

### US4: Template-Aware Tokenization Parity

**As a** BitNet-rs developer using prompt templates
**I want** crossval-per-token to respect template settings
**So that** tokenization matches production CLI behavior

**Business Value**: Ensures cross-validation reflects real-world inference patterns

**Acceptance Criteria**:
- AC12: Support `--prompt-template` flag (auto/raw/instruct/llama3-chat)
- AC13: Apply template formatting before tokenization
- AC14: Respect template BOS policy (e.g., llama3-chat: no BOS)
- AC15: Parse special tokens when template requires it

---

## Scope

### In Scope

**Affected Workspace Crates**:
1. **xtask** (`xtask/src/main.rs`):
   - Modify `crossval_per_token_cmd()` to add token validation
   - Add template flag support
   - Enhance error reporting

2. **crossval** (`crossval/src/`):
   - New module: `crossval/src/token_parity.rs` for validation logic
   - Export `TokenParityChecker` public API
   - Integration with existing `logits_compare` module

3. **bitnet-inference** (no changes):
   - Reuse existing `TemplateType` public API
   - Reuse existing template application logic

4. **bitnet-tokenizers** (no changes):
   - Reuse existing `Tokenizer` trait
   - Reuse existing `TokenizerBuilder` for loading

### Out of Scope

1. **FFI modifications**: Phase 1 uses existing string-based tokenization; direct token FFI is future work
2. **C++ reference changes**: No modifications to Microsoft BitNet C++ implementation
3. **Template auto-detection in xtask**: User must specify template explicitly via flag
4. **Receipt format changes**: Token parity status recorded but schema v1.0.0 unchanged

### Pipeline Stage Alignment

This feature operates in the **Cross-Validation** stage:

```
Model Loading → Quantization → Inference → Cross-Validation (TOKEN PARITY HERE) → Output
```

**Integration Point**: Before logits evaluation in `crossval-per-token` command

---

## Technical Requirements

### TR1: Token Sequence Comparison

**Requirement**: Compare Rust and C++ token sequences with exact equality check

**Implementation**:
```rust
// crossval/src/token_parity.rs
pub struct TokenParityChecker;

impl TokenParityChecker {
    /// Compare Rust and C++ token sequences
    /// Returns Ok(()) if tokens match, Err with detailed mismatch info otherwise
    pub fn validate(
        rust_tokens: &[u32],
        cpp_tokens: &[i32],
        prompt: &str,
    ) -> Result<(), TokenParityError> {
        // Convert types for comparison
        let cpp_as_u32: Vec<u32> = cpp_tokens.iter().map(|&t| t as u32).collect();

        // Fast path: exact match
        if rust_tokens == cpp_as_u32 {
            return Ok(());
        }

        // Detailed mismatch analysis
        let first_diff_idx = rust_tokens
            .iter()
            .zip(cpp_as_u32.iter())
            .position(|(r, c)| r != c)
            .unwrap_or_else(|| rust_tokens.len().min(cpp_as_u32.len()));

        Err(TokenParityError {
            rust_tokens: rust_tokens.to_vec(),
            cpp_tokens: cpp_as_u32,
            first_diff_idx,
            prompt: prompt.to_string(),
        })
    }
}
```

**Constraints**:
- Zero-copy comparison where possible
- Early exit on first mismatch for performance
- Handles length mismatches gracefully

---

### TR2: Template-Aware Tokenization

**Requirement**: Apply prompt templates before tokenization in xtask

**Implementation**:
```rust
// xtask/src/main.rs
use bitnet_inference::TemplateType;

// Parse template from CLI flag
let template = match template_str.as_str() {
    "auto" => TemplateType::detect(None, None), // Default: auto-detection
    "raw" => TemplateType::Raw,
    "instruct" => TemplateType::Instruct,
    "llama3-chat" => TemplateType::Llama3Chat,
    _ => anyhow::bail!("Unknown template: {}", template_str),
};

// Format prompt with template
let formatted_prompt = template.apply(prompt, system_prompt.as_deref());

// Determine BOS policy from template
let add_bos = template.should_add_bos();
let parse_special = template.parse_special();

// Tokenize with proper flags
let rust_tokens = tokenizer.encode(&formatted_prompt, add_bos, parse_special)?;
```

**Dependencies**:
- `bitnet_inference::TemplateType` (already public)
- `bitnet_tokenizers::Tokenizer::encode()` (already public)

**Constraints**:
- No changes to existing `TemplateType` API
- Template selection must be explicit via CLI flag (no auto-detection in Phase 1)

---

### TR3: Enhanced Error Reporting

**Requirement**: Clear, actionable error messages with fix-it suggestions

**Error Structure**:
```rust
#[derive(Debug)]
pub struct TokenParityError {
    pub rust_tokens: Vec<u32>,
    pub cpp_tokens: Vec<u32>,
    pub first_diff_idx: usize,
    pub prompt: String,
}

impl TokenParityError {
    /// Generate user-friendly error message with fix-it hints
    pub fn format_error(&self) -> String {
        use console::style;

        let mut msg = String::new();
        msg.push_str(&format!("\n{}\n", style("❌ Token Sequence Mismatch").red().bold()));
        msg.push_str(&format!("{}\n", style("Rust and C++ produced different tokens - cannot compare logits!").yellow()));

        // Show token sequences (truncate to 64 tokens for readability)
        msg.push_str(&format!("\n{}:\n", style("Rust tokens").cyan()));
        msg.push_str(&format!("  {:?}\n", &self.rust_tokens[..self.rust_tokens.len().min(64)]));

        msg.push_str(&format!("\n{}:\n", style("C++ tokens").cyan()));
        msg.push_str(&format!("  {:?}\n", &self.cpp_tokens[..self.cpp_tokens.len().min(64)]));

        msg.push_str(&format!("\n{}: {}\n", style("First diff at index").yellow(), self.first_diff_idx));

        // Heuristic-based fix suggestions
        msg.push_str(&format!("\n{}:\n", style("Suggested fixes").green().bold()));

        if self.rust_tokens.len() > self.cpp_tokens.len() && self.first_diff_idx == 1 {
            msg.push_str("  • Possible duplicate BOS token - try: --prompt-template raw\n");
        }

        msg.push_str("  • Use --prompt-template raw for raw tokenization (no formatting)\n");
        msg.push_str("  • Check GGUF chat_template metadata with: bitnet inspect model.gguf\n");
        msg.push_str("  • Verify BOS handling: --dump-ids to inspect token sequences\n");

        msg.push_str(&format!("\n{}\n", style("Example command with raw template:").cyan()));
        msg.push_str(&format!("  cargo run -p xtask -- crossval-per-token \\\n"));
        msg.push_str(&format!("    --model <model.gguf> \\\n"));
        msg.push_str(&format!("    --tokenizer <tokenizer.json> \\\n"));
        msg.push_str(&format!("    --prompt \"{}\" \\\n", self.prompt));
        msg.push_str(&format!("    --prompt-template raw\n"));

        msg
    }
}

impl std::fmt::Display for TokenParityError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.format_error())
    }
}

impl std::error::Error for TokenParityError {}
```

**Dependencies**:
- `console` crate for colored output (already in xtask/Cargo.toml)

---

### TR4: CLI Flag Support

**Requirement**: Add `--prompt-template` flag to `crossval-per-token` command

**Implementation**:
```rust
// xtask/src/main.rs (command definition around line 405-430)
#[cfg(feature = "inference")]
#[command(name = "crossval-per-token")]
CrossvalPerToken {
    #[arg(long)]
    model: PathBuf,

    #[arg(long)]
    tokenizer: PathBuf,

    #[arg(long)]
    prompt: String,

    #[arg(long, default_value_t = 4)]
    max_tokens: usize,

    #[arg(long, default_value_t = 0.999)]
    cos_tol: f32,

    #[arg(long, default_value = "text")]
    format: String,

    // NEW: Template selection flag
    #[arg(long, default_value = "raw", help = "Prompt template (raw|instruct|llama3-chat)")]
    prompt_template: String,

    // NEW: Optional system prompt for templates
    #[arg(long, help = "System prompt for chat templates")]
    system_prompt: Option<String>,
},
```

**Default Behavior**:
- Default template: `raw` (no formatting, consistent with current behavior)
- User can override with `--prompt-template instruct` or `--prompt-template llama3-chat`

---

## Acceptance Criteria

### AC1: Token Mismatch Detection (US1)
**ID**: AC1
**Test Tag**: `// AC:1`
**Criterion**: Token parity validation occurs before any logits evaluation
**Validation**:
- Integration test with intentionally mismatched tokens
- Verify no logits computation when tokens differ
- Check error appears before "Evaluating Rust logits" output

---

### AC2: Both Token Sequences Displayed (US1)
**ID**: AC2
**Test Tag**: `// AC:2`
**Criterion**: Error message displays both Rust and C++ token sequences
**Validation**:
- Parse error output for "Rust tokens:" and "C++ tokens:" sections
- Verify both sequences printed with proper formatting
- Truncate to 64 tokens max for readability

---

### AC3: First Diff Position Shown (US1)
**ID**: AC3
**Test Tag**: `// AC:3`
**Criterion**: Error message includes "First diff at index: N"
**Validation**:
- Test with tokens differing at position 0, 1, and middle positions
- Verify correct index calculation
- Handle edge case: length mismatch (diff at min(len_rust, len_cpp))

---

### AC4: Exit Code 2 on Mismatch (US1)
**ID**: AC4
**Test Tag**: `// AC:4`
**Criterion**: Command exits with code 2 (usage error) when tokens differ
**Validation**:
- Capture exit code in integration test
- Verify exit code 2 (not 1 for divergence, not 0 for success)
- Consistent with semantic versioning of exit codes

---

### AC5: Raw Template Suggestion (US2)
**ID**: AC5
**Test Tag**: `// AC:5`
**Criterion**: Error suggests `--prompt-template raw` as fix
**Validation**:
- Parse error output for suggested fixes section
- Verify "Use --prompt-template raw" appears
- Check example command includes template flag

---

### AC6: No-BOS Suggestion (US2)
**ID**: AC6
**Test Tag**: `// AC:6`
**Criterion**: Error suggests BOS-related fixes when applicable
**Validation**:
- Heuristic: if rust_tokens.len() > cpp_tokens.len() and first_diff_idx == 1
- Suggest "Possible duplicate BOS token"
- Provide actionable workaround

---

### AC7: GGUF Metadata Check Suggestion (US2)
**ID**: AC7
**Test Tag**: `// AC:7`
**Criterion**: Error suggests checking GGUF chat_template metadata
**Validation**:
- Parse error for "Check GGUF chat_template metadata"
- Suggest using `bitnet inspect model.gguf` command
- Educational for users unfamiliar with template detection

---

### AC8: Example Command Provided (US2)
**ID**: AC8
**Test Tag**: `// AC:8`
**Criterion**: Error includes copy-paste-able example command
**Validation**:
- Error contains "Example command with raw template:"
- Command uses actual prompt from error context
- Valid syntax (can be copied directly to shell)

---

### AC9: Silent Success (US3)
**ID**: AC9
**Test Tag**: `// AC:9`
**Criterion**: No output when tokens match exactly
**Validation**:
- Test with identical Rust/C++ token sequences
- Verify no "Token parity" or validation messages
- Output proceeds directly to logits comparison

---

### AC10: Performance Overhead <100ms (US3)
**ID**: AC10
**Test Tag**: `// AC:10`
**Criterion**: Token comparison adds <100ms overhead for typical prompts
**Validation**:
- Benchmark with 10, 100, 1000 token sequences
- Measure time for `TokenParityChecker::validate()`
- Verify O(n) complexity with early exit

---

### AC11: Direct Logits Comparison (US3)
**ID**: AC11
**Test Tag**: `// AC:11`
**Criterion**: When tokens match, proceed immediately to logits comparison
**Validation**:
- No unnecessary work between validation and comparison
- Same code path as original (no refactoring beyond validation check)

---

### AC12: Template Flag Support (US4)
**ID**: AC12
**Test Tag**: `// AC:12`
**Criterion**: `--prompt-template` flag accepted and applied
**Validation**:
- Test with `--prompt-template raw|instruct|llama3-chat`
- Verify template formatting applied before tokenization
- Invalid template → error with suggestion

---

### AC13: Template Formatting Applied (US4)
**ID**: AC13
**Test Tag**: `// AC:13`
**Criterion**: Prompt formatted according to template before encoding
**Validation**:
- Instruct template: adds "Q: {prompt}\nA:"
- LLaMA-3 template: adds `<|begin_of_text|>` and role markers
- Raw template: no formatting (pass-through)

---

### AC14: Template BOS Policy Respected (US4)
**ID**: AC14
**Test Tag**: `// AC:14`
**Criterion**: BOS token handling follows template defaults
**Validation**:
- Raw/Instruct: add_bos=true
- LLaMA-3 chat: add_bos=false (template includes `<|begin_of_text|>`)
- No duplicate BOS tokens

---

### AC15: Special Token Parsing (US4)
**ID**: AC15
**Test Tag**: `// AC:15`
**Criterion**: Special tokens parsed when template requires it
**Validation**:
- LLaMA-3 template: parse_special=true
- Raw/Instruct: parse_special=false
- Special tokens encoded as single token IDs, not character sequences

---

## Architecture Design

### Component Overview

```
┌─────────────────────────────────────────────────────────────┐
│ xtask/src/main.rs - crossval_per_token_cmd()                │
│                                                               │
│  1. Parse CLI flags (--prompt-template, --system-prompt)     │
│  2. Load tokenizer (bitnet_tokenizers::TokenizerBuilder)     │
│  3. Apply template (TemplateType::apply)                     │
│  4. Tokenize Rust (tokenizer.encode with template flags)     │
│  5. Tokenize C++ (Session::tokenize)                         │
│  6. ┌──────────────────────────────────────────────────┐    │
│     │ TOKEN PARITY PRE-GATE (NEW)                      │    │
│     │ crossval::token_parity::TokenParityChecker       │    │
│     │                                                   │    │
│     │ ✓ Compare token sequences                        │    │
│     │ ✓ If mismatch: format error + exit 2             │    │
│     │ ✓ If match: silent success                       │    │
│     └──────────────────────────────────────────────────┘    │
│  7. Evaluate Rust logits (bitnet_inference::parity)          │
│  8. Evaluate C++ logits (bitnet_sys::wrapper::Session)       │
│  9. Compare logits (crossval::logits_compare)                │
│ 10. Report divergence or success                             │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow

```
User Input (Prompt + Template)
        ↓
┌───────────────────────┐
│ Template Application  │  (bitnet_inference::TemplateType)
│ - Format prompt       │
│ - Determine BOS       │
│ - Set parse_special   │
└───────────────────────┘
        ↓
┌───────────────────────┬───────────────────────┐
│ Rust Tokenization     │ C++ Tokenization      │
│ (bitnet_tokenizers)   │ (bitnet_sys::wrapper) │
└───────────────────────┴───────────────────────┘
        ↓                       ↓
    rust_tokens            cpp_tokens
        └───────────┬───────────┘
                    ↓
        ┌───────────────────────┐
        │ TOKEN PARITY CHECK    │ <-- NEW
        │ (crossval::token_parity)
        │                       │
        │ Match?   → Continue   │
        │ Mismatch → Error      │
        └───────────────────────┘
                    ↓
        ┌───────────────────────┐
        │ Logits Evaluation     │
        │ (Rust + C++)          │
        └───────────────────────┘
                    ↓
        ┌───────────────────────┐
        │ Logits Comparison     │
        │ (crossval::logits_compare)
        └───────────────────────┘
```

### Module Structure

**New Module**: `crossval/src/token_parity.rs`

```rust
//! Token parity validation for cross-validation commands
//!
//! This module provides token sequence comparison between Rust and C++
//! implementations to ensure logits comparison operates on identical inputs.

use anyhow::Result;

/// Token parity checker for cross-validation
pub struct TokenParityChecker;

/// Token parity error with detailed mismatch information
#[derive(Debug)]
pub struct TokenParityError {
    pub rust_tokens: Vec<u32>,
    pub cpp_tokens: Vec<u32>,
    pub first_diff_idx: usize,
    pub prompt: String,
}

impl TokenParityChecker {
    /// Validate that Rust and C++ token sequences match exactly
    ///
    /// # Arguments
    /// * `rust_tokens` - Token IDs from Rust tokenizer
    /// * `cpp_tokens` - Token IDs from C++ tokenizer (i32 from FFI)
    /// * `prompt` - Original prompt text (for error context)
    ///
    /// # Returns
    /// * `Ok(())` - Tokens match exactly
    /// * `Err(TokenParityError)` - Tokens differ with detailed diagnostics
    ///
    /// # Performance
    /// O(n) with early exit on first mismatch. Typical overhead <1ms for <1000 tokens.
    pub fn validate(
        rust_tokens: &[u32],
        cpp_tokens: &[i32],
        prompt: &str,
    ) -> Result<(), TokenParityError>;
}

impl TokenParityError {
    /// Format user-friendly error message with fix-it suggestions
    pub fn format_error(&self) -> String;
}

impl std::fmt::Display for TokenParityError { /* ... */ }
impl std::error::Error for TokenParityError {}
```

**Public Exports** (`crossval/src/lib.rs`):

```rust
pub mod token_parity;
pub use token_parity::{TokenParityChecker, TokenParityError};
```

---

## API Contracts

### Public Rust API

**Module**: `crossval::token_parity`

```rust
/// Primary validation API
pub fn validate_token_parity(
    rust_tokens: &[u32],
    cpp_tokens: &[i32],
    prompt: &str,
) -> Result<(), TokenParityError> {
    TokenParityChecker::validate(rust_tokens, cpp_tokens, prompt)
}
```

**Stability**: This API is considered **stable** for v1.0.0 and follows semantic versioning.

**Breaking Changes**: None planned. Future enhancements will be additive (e.g., optional config parameter).

---

### CLI Interface Changes

**Command**: `cargo run -p xtask -- crossval-per-token`

**New Flags**:

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--prompt-template` | String | `"raw"` | Prompt template: raw, instruct, llama3-chat |
| `--system-prompt` | String | None | Optional system prompt for chat templates |

**Examples**:

```bash
# Default (raw template, no BOS, no special tokens)
cargo run -p xtask -- crossval-per-token \
  --model model.gguf \
  --tokenizer tokenizer.json \
  --prompt "What is 2+2?"

# Instruct template (adds Q:/A: markers, includes BOS)
cargo run -p xtask -- crossval-per-token \
  --model model.gguf \
  --tokenizer tokenizer.json \
  --prompt "What is 2+2?" \
  --prompt-template instruct

# LLaMA-3 chat template (special tokens, no BOS)
cargo run -p xtask -- crossval-per-token \
  --model model.gguf \
  --tokenizer tokenizer.json \
  --prompt "What is 2+2?" \
  --prompt-template llama3-chat \
  --system-prompt "You are a helpful assistant"
```

**Exit Codes**:

| Code | Meaning | Scenario |
|------|---------|----------|
| 0 | Success | Tokens match, logits within tolerance |
| 1 | Divergence | Tokens match, logits diverged |
| 2 | Usage error | Tokens differ (parity check failed) |
| Other | Runtime error | FFI unavailable, model load failure, etc. |

---

## Integration Points

### IP1: xtask/src/main.rs Integration

**Location**: `crossval_per_token_cmd()` function (lines 2901-3041)

**Modification Points**:

1. **Import new module** (after line 2910):
```rust
use bitnet_crossval::token_parity::TokenParityChecker;
use bitnet_inference::TemplateType;
```

2. **Add template flag handling** (after line 2917):
```rust
// Parse template
let template = TemplateType::from_str(prompt_template)?;
let formatted_prompt = template.apply(prompt, system_prompt.as_deref());
let add_bos = template.should_add_bos();
let parse_special = template.parse_special();

println!("Template: {:?} (BOS={}, special={})", template, add_bos, parse_special);
```

3. **Update Rust tokenization** (replace line 2921):
```rust
// OLD: let tokens = tokenizer.encode(prompt, false, false)?;
// NEW:
let tokens = tokenizer.encode(&formatted_prompt, add_bos, parse_special)?;
```

4. **Add token parity check** (after line 2957, before line 2960):
```rust
// Token parity pre-gate
println!("🔍 Validating token parity...");
if let Err(e) = TokenParityChecker::validate(&tokens, &cpp_tokens, prompt) {
    eprintln!("{}", e);
    std::process::exit(2); // Exit code 2: usage error
}
println!("✓ Tokens match: {} tokens", tokens.len());
println!();
```

**Diff Preview**:
```diff
     let tokenizer = bitnet_tokenizers::loader::load_tokenizer(tokenizer_path)?;
-    let tokens = tokenizer.encode(prompt, false, false)?;
+    let template = TemplateType::from_str(prompt_template)?;
+    let formatted_prompt = template.apply(prompt, system_prompt.as_deref());
+    let add_bos = template.should_add_bos();
+    let parse_special = template.parse_special();
+    let tokens = tokenizer.encode(&formatted_prompt, add_bos, parse_special)?;
     let token_ids: Vec<i32> = tokens.iter().map(|&id| id as i32).collect();

     // ...

     let cpp_tokens = cpp_session.tokenize(prompt)?;
+
+    // Token parity pre-gate
+    if let Err(e) = TokenParityChecker::validate(&tokens, &cpp_tokens, prompt) {
+        eprintln!("{}", e);
+        std::process::exit(2);
+    }

     // Evaluate all positions
     cpp_session.context.eval(&cpp_tokens, 0)?;
```

---

### IP2: crossval Crate Integration

**File**: `crossval/src/lib.rs`

**Changes**:
```rust
pub mod token_parity;

pub use token_parity::{TokenParityChecker, TokenParityError};
```

**New File**: `crossval/src/token_parity.rs` (see [Module Structure](#module-structure))

**Dependencies**: No new dependencies required (uses `anyhow`, `console` already in workspace)

---

### IP3: Reuse of Existing APIs

**No modifications required** to:
- `bitnet_inference::TemplateType` (already public, complete API)
- `bitnet_tokenizers::Tokenizer::encode()` (already public)
- `bitnet_sys::wrapper::Session::tokenize()` (already public FFI wrapper)

**Design Principle**: Reuse over reinvention. This feature composes existing stable APIs without requiring changes.

---

## Error Handling

### Error Categories

1. **Token Parity Errors** (Exit Code 2):
   - Token sequences differ between Rust and C++
   - Actionable fix suggestions provided
   - User must resolve tokenization before logits comparison

2. **Existing Error Handling** (Unchanged):
   - FFI unavailable: Error propagation with message
   - Model load failure: Error propagation
   - Tokenization failure: Error propagation
   - Logits divergence: Exit code 1

### Error Message Format

**Token Parity Error** (Console Output):

```
❌ Token Sequence Mismatch
Rust and C++ produced different tokens - cannot compare logits!

Rust tokens:
  [128000, 128000, 1229, 374, 220, 17, 10, 17, ...]

C++ tokens:
  [128000, 1229, 374, 220, 17, 10, 17, ...]

First diff at index: 1

Suggested fixes:
  • Possible duplicate BOS token - try: --prompt-template raw
  • Use --prompt-template raw for raw tokenization (no formatting)
  • Check GGUF chat_template metadata with: bitnet inspect model.gguf
  • Verify BOS handling: --dump-ids to inspect token sequences

Example command with raw template:
  cargo run -p xtask -- crossval-per-token \
    --model <model.gguf> \
    --tokenizer <tokenizer.json> \
    --prompt "What is 2+2?" \
    --prompt-template raw
```

**JSON Output** (when `--format json`):

```json
{
  "status": "token_mismatch",
  "error": "Token sequences differ at position 1",
  "rust_tokens": [128000, 128000, 1229, 374, 220, 17],
  "cpp_tokens": [128000, 1229, 374, 220, 17],
  "first_diff_idx": 1,
  "suggestions": [
    "Use --prompt-template raw",
    "Check GGUF chat_template metadata",
    "Verify BOS handling with --dump-ids"
  ]
}
```

---

## Testing Strategy

### Unit Tests

**File**: `crossval/src/token_parity.rs`

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokens_match() {
        // AC:9 - Silent success when tokens match
        let rust = vec![128000, 1229, 374];
        let cpp = vec![128000, 1229, 374];
        assert!(TokenParityChecker::validate(&rust, &cpp, "test").is_ok());
    }

    #[test]
    fn test_tokens_differ_first_position() {
        // AC:1, AC:3 - Detect mismatch at position 0
        let rust = vec![9999, 1229, 374];
        let cpp = vec![128000, 1229, 374];
        let err = TokenParityChecker::validate(&rust, &cpp, "test").unwrap_err();
        assert_eq!(err.first_diff_idx, 0);
    }

    #[test]
    fn test_duplicate_bos_detection() {
        // AC:6 - Detect duplicate BOS scenario
        let rust = vec![128000, 128000, 1229, 374]; // Duplicate BOS
        let cpp = vec![128000, 1229, 374];
        let err = TokenParityChecker::validate(&rust, &cpp, "test").unwrap_err();
        assert_eq!(err.first_diff_idx, 1);

        let msg = err.format_error();
        assert!(msg.contains("Possible duplicate BOS token"));
    }

    #[test]
    fn test_length_mismatch() {
        // AC:3 - Handle length mismatch (diff at min length)
        let rust = vec![128000, 1229];
        let cpp = vec![128000, 1229, 374];
        let err = TokenParityChecker::validate(&rust, &cpp, "test").unwrap_err();
        assert_eq!(err.first_diff_idx, 2); // At rust.len()
    }

    #[test]
    fn test_error_message_format() {
        // AC:2, AC:5, AC:7, AC:8 - Verify error message components
        let rust = vec![128000, 128000, 1229];
        let cpp = vec![128000, 1229];
        let err = TokenParityChecker::validate(&rust, &cpp, "What is 2+2?").unwrap_err();
        let msg = err.format_error();

        assert!(msg.contains("Rust tokens:"));
        assert!(msg.contains("C++ tokens:"));
        assert!(msg.contains("First diff at index:"));
        assert!(msg.contains("Suggested fixes:"));
        assert!(msg.contains("--prompt-template raw"));
        assert!(msg.contains("bitnet inspect model.gguf"));
        assert!(msg.contains("Example command"));
        assert!(msg.contains("What is 2+2?")); // Prompt in example
    }

    #[test]
    fn test_performance_large_sequence() {
        // AC:10 - Performance test for 1000 tokens
        use std::time::Instant;

        let rust: Vec<u32> = (0..1000).collect();
        let cpp: Vec<i32> = (0..1000).map(|x| x as i32).collect();

        let start = Instant::now();
        let _ = TokenParityChecker::validate(&rust, &cpp, "perf test");
        let elapsed = start.elapsed();

        assert!(elapsed.as_millis() < 100, "Performance overhead too high: {:?}", elapsed);
    }
}
```

---

### Integration Tests

**File**: `xtask/tests/crossval_token_parity.rs` (new)

```rust
use assert_cmd::Command;
use predicates::prelude::*;

#[test]
#[serial] // Prevent parallel FFI calls
fn test_token_mismatch_exit_code() {
    // AC:4 - Exit code 2 on token mismatch
    // This test requires a fixture with known token mismatch scenario

    let mut cmd = Command::cargo_bin("xtask").unwrap();
    cmd.args([
        "crossval-per-token",
        "--model", "tests/fixtures/model.gguf",
        "--tokenizer", "tests/fixtures/tokenizer.json",
        "--prompt", "test prompt with known mismatch",
        "--prompt-template", "instruct", // Causes mismatch vs C++ raw
    ]);

    cmd.assert()
        .failure()
        .code(2); // AC:4
}

#[test]
#[serial]
fn test_raw_template_prevents_mismatch() {
    // AC:12, AC:13 - Raw template prevents false mismatches

    let mut cmd = Command::cargo_bin("xtask").unwrap();
    cmd.args([
        "crossval-per-token",
        "--model", "tests/fixtures/model.gguf",
        "--tokenizer", "tests/fixtures/tokenizer.json",
        "--prompt", "test",
        "--prompt-template", "raw",
    ]);

    cmd.assert()
        .success(); // Tokens should match
}

#[test]
#[serial]
fn test_error_message_suggestions() {
    // AC:5, AC:7, AC:8 - Error includes suggestions

    let mut cmd = Command::cargo_bin("xtask").unwrap();
    cmd.args([
        "crossval-per-token",
        "--model", "tests/fixtures/model.gguf",
        "--tokenizer", "tests/fixtures/tokenizer.json",
        "--prompt", "test",
        "--prompt-template", "instruct",
    ]);

    cmd.assert()
        .failure()
        .stderr(predicate::str::contains("--prompt-template raw"))
        .stderr(predicate::str::contains("bitnet inspect model.gguf"))
        .stderr(predicate::str::contains("Example command"));
}
```

---

### Test Fixtures

**Required Fixtures**:

1. **Minimal GGUF model** (`tests/fixtures/model.gguf`):
   - Small model for fast tests (~10MB)
   - Known tokenizer configuration
   - GGUF metadata with/without chat_template

2. **Tokenizer files** (`tests/fixtures/tokenizer.json`):
   - HuggingFace tokenizers format
   - LLaMA-3 compatible tokenizer with BOS/EOS/special tokens

3. **Test prompts** with known token sequences:
   - "test" → known token IDs for validation
   - Prompts with BOS duplication scenarios
   - Prompts with special token handling requirements

---

### TDD Test Scaffolding

**Test Creation Order** (Red-Green-Refactor):

1. **Phase 1 - Unit Tests** (crossval/src/token_parity.rs):
   - `test_tokens_match` → AC:9
   - `test_tokens_differ_first_position` → AC:1, AC:3
   - `test_duplicate_bos_detection` → AC:6
   - `test_error_message_format` → AC:2, AC:5, AC:7, AC:8
   - `test_performance_large_sequence` → AC:10

2. **Phase 2 - Integration Tests** (xtask/tests/):
   - `test_token_mismatch_exit_code` → AC:4
   - `test_raw_template_prevents_mismatch` → AC:12, AC:13
   - `test_template_bos_policy` → AC:14
   - `test_special_token_parsing` → AC:15

3. **Phase 3 - E2E Validation**:
   - Real model cross-validation with known baseline
   - Performance benchmarking with large prompts
   - Regression tests for existing logits comparison

---

## Performance Requirements

### PR1: Token Comparison Overhead

**Target**: <100ms for typical prompts (<1000 tokens)
**Implementation**: O(n) slice comparison with early exit
**Validation**: Benchmark in `test_performance_large_sequence`

**Optimization Strategies**:
- Use `Iterator::zip()` for pairwise comparison
- Early exit on first mismatch (typical case: mismatch at position 0-2)
- Avoid cloning token vectors (use slices)

---

### PR2: Memory Efficiency

**Target**: No additional heap allocations in happy path
**Implementation**: Borrow token slices, only allocate on error

```rust
// Happy path (match): zero allocations
if rust_tokens == cpp_as_u32 {
    return Ok(()); // No String allocation, no Vec cloning
}

// Error path: allocate for error details (acceptable)
Err(TokenParityError {
    rust_tokens: rust_tokens.to_vec(), // Only on error
    cpp_tokens: cpp_as_u32,
    // ...
})
```

---

### PR3: No Regression in Logits Comparison

**Target**: Existing logits comparison performance unchanged
**Validation**: Benchmark before/after with `criterion`

**Measurement**:
```bash
# Baseline (before token parity)
cargo bench --bench crossval_benchmarks

# After implementation
cargo bench --bench crossval_benchmarks

# Compare results (should be <5% regression)
```

---

## Risks and Mitigations

### R1: FFI String Tokenization Overhead

**Risk**: Passing prompt as string to C++ requires C++ tokenization, adding overhead
**Impact**: ~10-50ms additional latency per cross-validation run
**Likelihood**: High (current FFI design)

**Mitigation**:
- **Phase 1** (this spec): Accept FFI overhead, document as known limitation
- **Phase 2** (future): Add direct token FFI: `Session::eval_with_tokens(&[i32])`
- **Benefit**: Phase 1 unblocks cross-validation reliability immediately

---

### R2: Template Auto-Detection Complexity

**Risk**: Auto-detecting templates in xtask requires GGUF parsing, increasing complexity
**Impact**: Scope creep, delayed delivery
**Likelihood**: Medium

**Mitigation**:
- **Phase 1** (this spec): Require explicit `--prompt-template` flag (no auto-detection)
- **User Experience**: Default to `raw` template (current behavior)
- **Future Work**: Add auto-detection in Phase 2 with GGUF metadata parsing

---

### R3: C++ Tokenization Unavailable

**Risk**: C++ FFI may not be available in some environments (CI, user machines)
**Impact**: Command fails with unclear error
**Likelihood**: Low (already gated by FFI availability check)

**Mitigation**:
- Existing FFI availability check remains in place (line 2944)
- Token parity check runs only if FFI available
- Error message suggests compiling with `--features crossval` or setting `BITNET_CPP_DIR`

---

### R4: False Positives from Template Differences

**Risk**: User uses different templates in Rust vs C++, causing valid token differences
**Impact**: Users see token parity errors when configurations intentionally differ
**Likelihood**: Low (cross-validation requires identical configs)

**Mitigation**:
- Documentation: Cross-validation requires identical tokenization configuration
- Error message: Suggests using `--prompt-template raw` for parity
- Future: Add `--allow-template-mismatch` flag for advanced users

---

## Future Enhancements

### FE1: Direct Token FFI (Phase 2)

**Goal**: Pass token IDs directly to C++ eval, bypassing C++ tokenization

**API Change**:
```rust
// New FFI wrapper method
impl Session {
    pub fn eval_with_tokens(&mut self, tokens: &[i32]) -> Result<()> {
        // Call llama_decode() directly with token IDs
        // Bypass llama_tokenize() entirely
    }
}
```

**Benefits**:
- Eliminates C++ tokenization overhead
- Ensures Rust and C++ use identical token sequences (by construction)
- Simplifies token parity validation (compare once, then use same tokens)

**Timeline**: Post-MVP, requires FFI wrapper enhancements

---

### FE2: Template Auto-Detection in xtask

**Goal**: Automatically detect template from GGUF metadata, like CLI

**Implementation**:
```rust
// Parse GGUF metadata
let gguf_data = std::fs::read(model_path)?;
let reader = GgufReader::new(&gguf_data)?;

// Detect template from chat_template metadata
let template = TemplateType::detect(
    None, // No tokenizer name hint
    reader.metadata.get("tokenizer.chat_template"), // GGUF metadata
);
```

**Benefits**:
- Consistent with CLI behavior
- Reduced flag boilerplate for users

**Timeline**: Phase 2, after Phase 1 stabilizes

---

### FE3: Detailed Token Diff Visualization

**Goal**: Show side-by-side token comparison with highlighting

**Output Example**:
```
Token Diff:
  [0] ✓ 128000 == 128000  (BOS)
  [1] ✗ 128000 != 1229    (Rust: BOS again, C++: "What")
  [2] ✓ 1229   == 374     (mismatch cascade...)
```

**Benefits**:
- Easier debugging for complex tokenization issues
- Educational for users learning tokenization behavior

**Timeline**: Nice-to-have, low priority

---

### FE4: Receipt Integration

**Goal**: Record token parity status in cross-validation receipts

**Schema Extension** (v1.1.0):
```json
{
  "token_parity": {
    "status": "ok" | "mismatch",
    "rust_count": 123,
    "cpp_count": 123,
    "first_diff_idx": null,
    "template_used": "raw"
  },
  "parity": {
    // Existing logits parity fields
  }
}
```

**Benefits**:
- Historical tracking of token parity issues
- Automated detection in CI pipelines

**Timeline**: Phase 2, after receipt schema v1.0.0 stabilizes

---

## Constraints

### C1: No FFI Modifications

**Constraint**: Phase 1 uses existing FFI APIs only
**Rationale**: Minimize scope, focus on token validation logic
**Impact**: C++ tokenization remains string-based (overhead acceptable for MVP)

---

### C2: No Template Auto-Detection

**Constraint**: User must specify `--prompt-template` explicitly
**Rationale**: Avoid GGUF parsing complexity in xtask
**Impact**: One additional CLI flag for users
**Mitigation**: Default to `raw` (current behavior)

---

### C3: Backward Compatibility

**Constraint**: Existing `crossval-per-token` behavior preserved with `--prompt-template raw`
**Rationale**: No breaking changes for existing scripts
**Impact**: Default template is `raw` (explicit in CLI definition)

---

### C4: Performance Target

**Constraint**: Token comparison must add <100ms overhead
**Rationale**: Cross-validation already slow; avoid compounding latency
**Impact**: Implementation must use efficient slice comparison (O(n) early exit)

---

## Public Contracts Summary

### Rust APIs (crossval crate)

```rust
// crossval/src/token_parity.rs
pub struct TokenParityChecker;
pub struct TokenParityError;

pub fn validate_token_parity(
    rust_tokens: &[u32],
    cpp_tokens: &[i32],
    prompt: &str,
) -> Result<(), TokenParityError>;
```

**Stability**: Stable for v1.0.0
**Versioning**: Follows semantic versioning

---

### CLI Interface (xtask)

**New Flags**:
- `--prompt-template <raw|instruct|llama3-chat>` (default: `raw`)
- `--system-prompt <text>` (optional, for chat templates)

**Exit Codes**:
- 0: Success (tokens match, logits within tolerance)
- 1: Divergence (tokens match, logits diverged)
- 2: Token mismatch (usage error)

---

## Documentation Updates

### D1: Update CLAUDE.md

**Section**: Quick Reference → Essential Commands

**Addition**:
```markdown
# Cross-validation with template support
cargo run -p xtask --features crossval-all -- crossval-per-token \
  --model models/model.gguf \
  --tokenizer models/tokenizer.json \
  --prompt "What is 2+2?" \
  --prompt-template raw  # Ensure Rust/C++ token parity
```

---

### D2: Create docs/howto/crossval-token-parity.md

**Content**:
- How to diagnose token parity failures
- Template selection guide
- Common pitfalls (duplicate BOS, special tokens)
- Troubleshooting steps

---

### D3: Update docs/development/validation-framework.md

**Section**: Cross-Validation Tools

**Addition**:
- Token parity pre-gate architecture
- Template-aware tokenization workflow
- Exit code semantics

---

## Implementation Checklist

- [ ] Create `crossval/src/token_parity.rs` module
- [ ] Implement `TokenParityChecker::validate()`
- [ ] Implement `TokenParityError::format_error()`
- [ ] Add unit tests (9 tests for all ACs)
- [ ] Update `crossval/src/lib.rs` exports
- [ ] Add `--prompt-template` flag to xtask command definition
- [ ] Add `--system-prompt` flag to xtask command definition
- [ ] Integrate template application in `crossval_per_token_cmd()`
- [ ] Add token parity check before logits evaluation
- [ ] Update exit code handling (code 2 for token mismatch)
- [ ] Create integration tests (`xtask/tests/crossval_token_parity.rs`)
- [ ] Prepare test fixtures (minimal GGUF + tokenizer)
- [ ] Run performance benchmarks (<100ms overhead)
- [ ] Update CLAUDE.md with new command examples
- [ ] Create docs/howto/crossval-token-parity.md
- [ ] Update validation framework documentation
- [ ] Manual testing with real models
- [ ] CI integration (ensure tests pass with `--features crossval-all`)

---

## Success Criteria

**Feature is complete when:**

1. ✅ All 15 acceptance criteria validated with tests (AC1-AC15)
2. ✅ Token mismatch exits with code 2 and actionable error message
3. ✅ Template support (`raw|instruct|llama3-chat`) working end-to-end
4. ✅ Performance overhead <100ms for typical prompts
5. ✅ Integration tests passing in CI
6. ✅ Documentation updated (CLAUDE.md, howto guide, validation framework)
7. ✅ Manual validation with real model shows improved error messages
8. ✅ No regressions in existing logits comparison behavior

---

## Appendix: Example Error Output

### Example 1: Duplicate BOS Token

**Command**:
```bash
cargo run -p xtask -- crossval-per-token \
  --model model.gguf \
  --tokenizer tokenizer.json \
  --prompt "What is 2+2?" \
  --prompt-template instruct  # Triggers duplicate BOS
```

**Output**:
```
🔍 Per-token logits parity check
Model: model.gguf
Prompt: "What is 2+2?"
Cosine tolerance: 0.999

📝 Tokenizing prompt...
Template: Instruct (BOS=true, special=false)
Tokens: 5 (prompt)

🔍 Validating token parity...

❌ Token Sequence Mismatch
Rust and C++ produced different tokens - cannot compare logits!

Rust tokens:
  [128000, 128000, 1229, 374, 220, 17, 10, 17, 30]

C++ tokens:
  [128000, 1229, 374, 220, 17, 10, 17, 30]

First diff at index: 1

Suggested fixes:
  • Possible duplicate BOS token - try: --prompt-template raw
  • Use --prompt-template raw for raw tokenization (no formatting)
  • Check GGUF chat_template metadata with: bitnet inspect model.gguf
  • Verify BOS handling: --dump-ids to inspect token sequences

Example command with raw template:
  cargo run -p xtask -- crossval-per-token \
    --model model.gguf \
    --tokenizer tokenizer.json \
    --prompt "What is 2+2?" \
    --prompt-template raw

Exit code: 2
```

---

### Example 2: Successful Token Parity

**Command**:
```bash
cargo run -p xtask -- crossval-per-token \
  --model model.gguf \
  --tokenizer tokenizer.json \
  --prompt "What is 2+2?" \
  --prompt-template raw
```

**Output**:
```
🔍 Per-token logits parity check
Model: model.gguf
Prompt: "What is 2+2?"
Cosine tolerance: 0.999

📝 Tokenizing prompt...
Template: Raw (BOS=true, special=false)
Tokens: 4 (prompt)

🔍 Validating token parity...
✓ Tokens match: 4 tokens

🦀 Evaluating Rust logits for all positions...
✓ Rust: 4 positions, vocab_size=32000

🔧 Evaluating C++ logits for all positions...
✓ C++: 4 positions, vocab_size=32000

📊 Comparing logits per position...
✓ t=0 cosine=0.999234 l2=1.23e-02
✓ t=1 cosine=0.999567 l2=8.45e-03
✓ t=2 cosine=0.999890 l2=3.21e-03
✓ t=3 cosine=0.999912 l2=2.87e-03

Max absolute diff: 3.45e-03
✅ All positions match within tolerance

Exit code: 0
```

---

**End of Specification**

**Next Steps**:
1. Review specification for completeness → spec-finalizer
2. Create test scaffolding → test-creator
3. Implement token parity checker → impl-creator
4. Integrate into xtask command → impl-creator
5. Validate with real models → validation
