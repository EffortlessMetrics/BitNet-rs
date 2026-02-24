# Token Parity Pre-Gate: Integration Design for crossval-per-token

## Executive Summary

The `crossval-per-token` command currently lacks token-parity validation, leading to false divergence detection when Rust and C++ use different tokenization approaches. This document outlines the token-parity pre-gate design for detecting and reporting tokenization mismatches **before** logits evaluation.

---

## Problem Statement

### Current Issue
The `crossval-per-token` command compares logits between Rust and C++ implementations but does NOT validate that they operate on the same token sequence:

```rust
// Current flow (main.rs lines 2920-2963)
let tokens = tokenizer.encode(prompt, false, false)?;      // Rust tokenizer: raw
let cpp_tokens = cpp_session.tokenize(prompt)?;             // C++ tokenizer: special handling

// Different token sequences → different logits → false divergence
let rust_logits = eval_logits_all_positions(model_path_str, &token_ids)?;
let cpp_logits = cpp_session.context.get_all_logits(cpp_tokens.len())?;
```

### Root Cause
1. **Rust tokenizer**: `encode(prompt, add_bos=false, add_special=false)` → raw tokens
2. **C++ tokenizer** (llama.cpp): `tokenize(prompt)` → includes special handling
3. **No validation**: Command proceeds despite potential token mismatch

### Impact
- False positive divergence reports at position 0 (tokens differ)
- Wasted debugging effort comparing incomparable logits
- Unreliable cross-validation baseline

---

## Pre-Gate Architecture

### Three-Level Pre-Gate System

```
┌─────────────────────────────────────────────────────────┐
│ crossval-per-token command execution                     │
└─────────────┬───────────────────────────────────────────┘
              │
              ├─► PRE-GATE 1: Token Sequence Validation
              │   ├─ Check: Rust tokens == C++ tokens
              │   ├─ Action: Report mismatch, suggest fixes
              │   └─ Exit if STRICT mode
              │
              ├─► Phase 1: Rust Logits Evaluation
              │   └─ eval_logits_all_positions()
              │
              ├─► PRE-GATE 2: Logits Baseline Validation
              │   ├─ Check: Logits non-zero, not NaN
              │   ├─ Check: L2 norm reasonable
              │   └─ Report issues but continue
              │
              ├─► Phase 2: C++ Logits Evaluation
              │   └─ Session::get_all_logits()
              │
              ├─► PRE-GATE 3: Per-Position Baseline Validation
              │   ├─ Check: Both have same logits shape
              │   ├─ Check: No catastrophic mismatches
              │   └─ Report but continue
              │
              └─► Phase 3: Logits Comparison
                  └─ compare_per_position_logits()
```

---

## Implementation Details

### Pre-Gate 1: Token Sequence Validation

**Location**: Between tokenization and evaluation (main.rs after line 2926)

**Module**: `bitnet_crossval::token_parity_check`

```rust
/// Token parity validation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenParityResult {
    /// Whether tokens match exactly
    pub tokens_match: bool,
    /// Rust token count
    pub rust_count: usize,
    /// C++ token count
    pub cpp_count: usize,
    /// Position of first mismatch (None if all match)
    pub first_mismatch_pos: Option<usize>,
    /// Detailed mismatch info for debugging
    pub mismatch_details: Vec<TokenMismatchDetail>,
    /// Diagnostics for root cause analysis
    pub diagnostics: TokenParityDiagnostics,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenMismatchDetail {
    /// Token position where mismatch occurs
    pub pos: usize,
    /// Rust token ID at this position
    pub rust_token: i32,
    /// C++ token ID at this position
    pub cpp_token: i32,
    /// Token text (if available from tokenizer)
    pub rust_text: Option<String>,
    pub cpp_text: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenParityDiagnostics {
    /// True if C++ has more tokens (likely BOS prepended)
    pub cpp_has_bos: bool,
    /// True if special tokens handled differently
    pub special_token_mismatch: bool,
    /// Suggested fix for user
    pub suggested_fix: String,
}

/// Validate token sequences match
pub fn check_token_parity(
    rust_tokens: &[i32],
    cpp_tokens: &[i32],
) -> TokenParityResult {
    let tokens_match = rust_tokens == cpp_tokens;
    
    if tokens_match {
        return TokenParityResult {
            tokens_match: true,
            rust_count: rust_tokens.len(),
            cpp_count: cpp_tokens.len(),
            first_mismatch_pos: None,
            mismatch_details: Vec::new(),
            diagnostics: TokenParityDiagnostics {
                cpp_has_bos: false,
                special_token_mismatch: false,
                suggested_fix: "Tokens match. Proceeding with logits comparison.".to_string(),
            },
        };
    }
    
    // Analyze mismatch
    let mut first_mismatch_pos = None;
    let mut mismatch_details = Vec::new();
    
    for (pos, (rust_tok, cpp_tok)) in rust_tokens.iter().zip(cpp_tokens.iter()).enumerate() {
        if rust_tok != cpp_tok {
            if first_mismatch_pos.is_none() {
                first_mismatch_pos = Some(pos);
            }
            mismatch_details.push(TokenMismatchDetail {
                pos,
                rust_token: *rust_tok,
                cpp_token: *cpp_tok,
                rust_text: None,
                cpp_text: None,
            });
        }
    }
    
    // Diagnose root cause
    let cpp_has_bos = cpp_tokens.len() > rust_tokens.len() && 
                      cpp_tokens[1..] == rust_tokens[..rust_tokens.len()];
    
    let suggested_fix = if cpp_has_bos {
        "C++ has BOS token prepended. Re-run with Rust tokenizer using add_bos=true.".to_string()
    } else {
        format!("Token mismatch at position {}. Check tokenizer configuration.", 
                first_mismatch_pos.unwrap_or(0))
    };
    
    TokenParityResult {
        tokens_match: false,
        rust_count: rust_tokens.len(),
        cpp_count: cpp_tokens.len(),
        first_mismatch_pos,
        mismatch_details,
        diagnostics: TokenParityDiagnostics {
            cpp_has_bos,
            special_token_mismatch: !cpp_has_bos, // If not BOS, assume special tokens
            suggested_fix,
        },
    }
}
```

**Integration Point** (main.rs line 2927):

```rust
// After tokenization, before evaluation
println!("📋 Validating token parity...");
let token_parity = bitnet_crossval::token_parity_check::check_token_parity(&token_ids, &cpp_tokens);

if !token_parity.tokens_match {
    println!("⚠️  Token sequence mismatch detected!");
    println!("   Rust tokens: {} | C++ tokens: {}", 
             token_parity.rust_count, token_parity.cpp_count);
    if let Some(pos) = token_parity.first_mismatch_pos {
        println!("   First mismatch at position {}", pos);
    }
    println!("   {}", token_parity.diagnostics.suggested_fix);
    println!();
    
    // In STRICT mode, exit here; otherwise continue with warning
    if std::env::var("BITNET_STRICT_TOKENIZATION").is_ok() {
        std::process::exit(2);
    }
} else {
    println!("✓ Token sequences match\n");
}
```

### Pre-Gate 2: Logits Baseline Validation

**Location**: After Rust evaluation (main.rs after line 2938)

**Module**: `bitnet_crossval::logits_validation`

```rust
/// Validate logits are reasonable (non-zero, finite, non-NaN)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogitsBaselineCheck {
    /// True if all logits pass baseline checks
    pub passed: bool,
    /// Total positions checked
    pub n_positions: usize,
    /// Number of positions with zero logits (all values ~0)
    pub zero_logits_positions: Vec<usize>,
    /// Number of positions with NaN/Inf logits
    pub invalid_logits_positions: Vec<usize>,
    /// L2 norm per position (useful for magnitude check)
    pub per_position_l2_norm: Vec<f32>,
    /// Warnings to report
    pub warnings: Vec<String>,
}

pub fn validate_logits_baseline(logits: &[Vec<f32>]) -> LogitsBaselineCheck {
    let n_positions = logits.len();
    let mut zero_logits_positions = Vec::new();
    let mut invalid_logits_positions = Vec::new();
    let mut per_position_l2_norm = Vec::new();
    let mut warnings = Vec::new();
    
    for (pos, logit_vec) in logits.iter().enumerate() {
        // Check for NaN/Inf
        let has_invalid = logit_vec.iter().any(|x| !x.is_finite());
        if has_invalid {
            invalid_logits_positions.push(pos);
            warnings.push(format!("Position {}: Contains NaN or Inf values", pos));
        }
        
        // Check for zero logits (all near zero)
        let l2_norm: f32 = logit_vec.iter().map(|x| x * x).sum::<f32>().sqrt();
        per_position_l2_norm.push(l2_norm);
        if l2_norm < 1e-6 {
            zero_logits_positions.push(pos);
            warnings.push(format!("Position {}: All logits near zero (L2 norm = {:.2e})", pos, l2_norm));
        }
    }
    
    let passed = zero_logits_positions.is_empty() && invalid_logits_positions.is_empty();
    
    LogitsBaselineCheck {
        passed,
        n_positions,
        zero_logits_positions,
        invalid_logits_positions,
        per_position_l2_norm,
        warnings,
    }
}
```

**Integration Point** (main.rs after line 2938):

```rust
println!("📊 Validating Rust logits baseline...");
let rust_baseline = bitnet_crossval::logits_validation::validate_logits_baseline(&rust_logits);

if !rust_baseline.passed {
    println!("⚠️  Rust logits baseline issues:");
    for warning in &rust_baseline.warnings {
        println!("   {}", warning);
    }
    println!();
} else {
    println!("✓ Rust logits baseline valid\n");
}

// Continue regardless (pre-gate is informational at this level)
let rust_logits = eval_logits_all_positions(model_path_str, &token_ids)?;
```

### Pre-Gate 3: Per-Position Shape & Range Validation

**Location**: Before comparison (main.rs after line 2970)

**Module**: `bitnet_crossval::comparison_validation`

```rust
/// Validate logits are comparable (same shape, magnitude bounds)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComparisonValidation {
    /// True if logits are comparable
    pub comparable: bool,
    /// Shape mismatch details
    pub shape_issues: Vec<String>,
    /// Magnitude range issues
    pub magnitude_issues: Vec<String>,
}

pub fn validate_before_comparison(
    rs_logits: &[Vec<f32>],
    cpp_logits: &[Vec<f32>],
) -> ComparisonValidation {
    let mut shape_issues = Vec::new();
    let mut magnitude_issues = Vec::new();
    
    // Check sequence length
    if rs_logits.len() != cpp_logits.len() {
        shape_issues.push(format!(
            "Sequence length mismatch: Rust {} vs C++ {}",
            rs_logits.len(),
            cpp_logits.len()
        ));
    }
    
    // Check vocab size and magnitude bounds
    let min_positions = rs_logits.len().min(cpp_logits.len());
    for pos in 0..min_positions {
        let rs_size = rs_logits[pos].len();
        let cpp_size = cpp_logits[pos].len();
        
        if rs_size != cpp_size {
            shape_issues.push(format!(
                "Vocab size mismatch at pos {}: Rust {} vs C++ {}",
                pos, rs_size, cpp_size
            ));
        }
        
        // Check magnitude bounds (logits should be roughly [-10, 10] range)
        let rs_max = rs_logits[pos].iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let cpp_max = cpp_logits[pos].iter().copied().fold(f32::NEG_INFINITY, f32::max);
        
        if rs_max > 100.0 || cpp_max > 100.0 {
            magnitude_issues.push(format!(
                "Position {}: Logits suspiciously large (Rust max={:.2e}, C++ max={:.2e})",
                pos, rs_max, cpp_max
            ));
        }
    }
    
    let comparable = shape_issues.is_empty() && magnitude_issues.is_empty();
    
    ComparisonValidation {
        comparable,
        shape_issues,
        magnitude_issues,
    }
}
```

**Integration Point** (main.rs before line 2972):

```rust
println!("📋 Validating comparison prerequisites...");
let comparison_valid = bitnet_crossval::comparison_validation::validate_before_comparison(
    &rust_logits,
    &cpp_logits,
);

if !comparison_valid.comparable {
    println!("❌ Logits not comparable:");
    for issue in &comparison_valid.shape_issues {
        println!("   {}", issue);
    }
    for issue in &comparison_valid.magnitude_issues {
        println!("   {}", issue);
    }
    std::process::exit(3);
}

println!("✓ Logits are comparable\n");
```

---

## Output Examples

### Token Mismatch Case

```
🔍 Per-token logits parity check
Model: models/model.gguf
Prompt: "Hello world"
Cosine tolerance: 0.999

📝 Tokenizing prompt...
Tokens: 3 (prompt)

📋 Validating token parity...
⚠️  Token sequence mismatch detected!
   Rust tokens: 3 | C++ tokens: 4
   First mismatch at position 0
   C++ has BOS token prepended. Re-run with Rust tokenizer using add_bos=true.

PRE-GATE: Token parity failed. Token sequences differ.

Suggested actions:
  1. Re-tokenize with add_bos=true to match C++
  2. Or use explicit tokenization: cargo run -p xtask -- \
       crossval-per-token --model ... --tokenizer ... \
       --prompt "Hello world" --use-bos-token

STRICT_TOKENIZATION mode: Run with BITNET_STRICT_TOKENIZATION=1 to fail here.
```

### Logits Baseline Issues

```
📊 Validating Rust logits baseline...
⚠️  Rust logits baseline issues:
   Position 0: All logits near zero (L2 norm = 1.23e-8)
   Position 1: Contains NaN or Inf values

These could indicate:
  - Model loading failure
  - Quantization format mismatch
  - Unsupported architecture

Run with BITNET_STRICT_VALIDATION=1 to fail on these issues.
```

### Success Case

```
📋 Validating token parity...
✓ Token sequences match

📊 Validating Rust logits baseline...
✓ Rust logits baseline valid

📋 Validating comparison prerequisites...
✓ Logits are comparable

📊 Comparing logits per position...
✓ t=0 cosine=0.999823 l2=1.23e-2
✓ t=1 cosine=0.999901 l2=8.45e-3
✓ t=2 cosine=0.999847 l2=1.09e-2

Max absolute diff: 1.23e-2
✅ All positions match within tolerance
```

---

## Environment Variable Control

```rust
// Use these to control pre-gate behavior:

// BITNET_STRICT_TOKENIZATION=1
//   Exit with code 2 on token mismatch
//   Default: Continue with warning

// BITNET_STRICT_VALIDATION=1
//   Exit with code 3 on logits baseline issues
//   Default: Continue with warning

// BITNET_PRE_GATE_JSON
//   Output pre-gate results in JSON format
//   Default: Human-readable text

// BITNET_PRE_GATE_VERBOSE=1
//   Show all pre-gate checks with detailed output
//   Default: Show only failures

// Example usage:
// BITNET_STRICT_TOKENIZATION=1 BITNET_PRE_GATE_JSON=1 \
//   cargo run -p xtask -- crossval-per-token --model m.gguf \
//     --tokenizer t.json --prompt "test" | jq .pre_gate
```

---

## Testing Strategy

### Unit Tests

```rust
#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_token_parity_exact_match() {
        let rust_tokens = vec![1, 2, 3, 4];
        let cpp_tokens = vec![1, 2, 3, 4];
        let result = check_token_parity(&rust_tokens, &cpp_tokens);
        assert!(result.tokens_match);
    }
    
    #[test]
    fn test_token_parity_bos_mismatch() {
        let rust_tokens = vec![1, 2, 3, 4];
        let cpp_tokens = vec![0, 1, 2, 3, 4];  // 0 is BOS
        let result = check_token_parity(&rust_tokens, &cpp_tokens);
        assert!(!result.tokens_match);
        assert!(result.diagnostics.cpp_has_bos);
    }
    
    #[test]
    fn test_logits_validation_zero_logits() {
        let logits = vec![
            vec![0.0, 0.0, 0.0],
            vec![1.0, 2.0, 3.0],
        ];
        let result = validate_logits_baseline(&logits);
        assert!(!result.passed);
        assert_eq!(result.zero_logits_positions, vec![0]);
    }
}
```

---

## Migration Path

### Phase 1: Add pre-gates (non-blocking)
- Implement all three pre-gates
- Print warnings but continue execution
- Collect pre-gate metrics in JSON output
- No impact on existing workflows

### Phase 2: Optional strict mode
- Add `BITNET_STRICT_TOKENIZATION` env var
- Enable users to fail early on token mismatches
- Improve debugging experience

### Phase 3: Make pre-gates configurable
- Add CLI flags: `--token-parity-strict`, `--logits-validation-strict`
- Allow users to choose behavior
- Document best practices

---

## Summary

The token-parity pre-gate system provides:

1. **Early Detection**: Catch tokenization mismatches before logits evaluation
2. **Root Cause Analysis**: Diagnose why tokens differ (BOS, special tokens, etc.)
3. **Actionable Feedback**: Suggest fixes for common issues
4. **Non-Blocking**: Proceed with warning by default
5. **Strict Mode**: Optional exit on pre-gate failures for CI/CD
6. **Detailed Diagnostics**: JSON output for automated analysis

This enables more reliable and faster cross-validation workflows.
