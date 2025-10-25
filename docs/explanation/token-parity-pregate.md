# Token Parity Pre-Gate Specification

**Feature:** Add fail-fast token sequence validation to `crossval-per-token` command
**Priority:** Critical (blocks reliable cross-validation)
**Status:** Specification → Ready for Implementation

---

## Problem Statement

Currently, `crossval-per-token` compares Rust and C++ logits without first validating that token sequences match. This means:

1. **Silent failures**: Duplicate BOS tokens cause logits divergence that's hard to debug
2. **Wasted time**: Users run expensive logits comparisons only to find tokenization was wrong
3. **Poor UX**: Error messages point to layer divergence, not token mismatch root cause

---

## Requirements

### Functional Requirements

**FR1:** Before comparing logits, validate Rust and C++ token sequences match exactly
**FR2:** On mismatch, print both sequences and first-diff index
**FR3:** Provide actionable fix-it guidance (template/BOS flags)
**FR4:** Exit with code 2 (usage error) on token mismatch
**FR5:** Only proceed to logits comparison if tokens match

### Non-Functional Requirements

**NFR1:** Add <50 lines of code to `xtask/src/main.rs`
**NFR2:** Zero overhead if tokens match (fast-path comparison)
**NFR3:** Clear, copy-paste-able error messages

---

## Design

### Architecture

```
User runs: cargo run -p xtask -- crossval-per-token --prompt "What is 2+2?" ...
                    ↓
         1. Tokenize in Rust (using CLI flags)
                    ↓
         2. Tokenize in C++ (via FFI wrapper)
                    ↓
         3. Compare sequences (NEW PRE-GATE)
                    ├─ Mismatch? → Print diagnostic + exit 2
                    └─ Match? → Continue to logits comparison
```

### Implementation Location

**File:** `xtask/src/main.rs`
**Function:** `crossval-per-token` command handler (around line 410-440)

**Add after tokenization, before logits comparison:**

```rust
// Token parity pre-gate
fn validate_token_parity(
    rust_tokens: &[u32],
    cpp_tokens: &[u32],
    prompt: &str,
) -> Result<()> {
    if rust_tokens != cpp_tokens {
        eprintln!("\n{}", style("❌ Token Sequence Mismatch").red().bold());
        eprintln!("{}", style("Fix BOS/template before comparing logits").yellow());

        eprintln!("\n{}:", style("Rust tokens").cyan());
        eprintln!("  {:?}", rust_tokens);

        eprintln!("\n{}:", style("C++ tokens").cyan());
        eprintln!("  {:?}", cpp_tokens);

        let first_diff = rust_tokens.iter()
            .zip(cpp_tokens.iter())
            .position(|(r, c)| r != c)
            .unwrap_or_else(|| rust_tokens.len().min(cpp_tokens.len()));

        eprintln!("\n{}: {}", style("First diff at index").yellow(), first_diff);

        eprintln!("\n{}:", style("Suggested fixes").green().bold());
        eprintln!("  • Use --prompt-template raw");
        eprintln!("  • Add --no-bos flag (if BOS is duplicate)");
        eprintln!("  • Check GGUF chat_template metadata");
        eprintln!("  • Use --dump-ids to inspect token sequences");

        std::process::exit(2);  // Exit 2 = usage error
    }
    Ok(())
}
```

---

## Acceptance Criteria

**AC1:** Token mismatch → clear error with both sequences printed
**AC2:** Error message shows first-diff index
**AC3:** Error includes 4+ actionable suggestions
**AC4:** Exit code 2 (usage error) on mismatch
**AC5:** Token match → silent success, proceeds to logits comparison
**AC6:** No performance regression (comparison is O(n) with early-exit)

---

## Test Scenarios

### Scenario 1: Duplicate BOS (Common Bug)

**Input:**
- Rust tokens: `[128000, 128000, 1229, 374]` (double BOS)
- C++ tokens: `[128000, 1229, 374]` (single BOS)

**Expected Output:**
```
❌ Token Sequence Mismatch
Fix BOS/template before comparing logits

Rust tokens:
  [128000, 128000, 1229, 374]

C++ tokens:
  [128000, 1229, 374]

First diff at index: 1

Suggested fixes:
  • Use --prompt-template raw
  • Add --no-bos flag (if BOS is duplicate)
  • Check GGUF chat_template metadata
  • Use --dump-ids to inspect token sequences
```

**Exit Code:** 2

### Scenario 2: Tokens Match (Happy Path)

**Input:**
- Rust tokens: `[128000, 1229, 374, 220, 17]`
- C++ tokens: `[128000, 1229, 374, 220, 17]`

**Expected Output:** (silent - no error message)

**Exit Code:** 0 (proceeds to logits comparison)

### Scenario 3: Length Mismatch

**Input:**
- Rust tokens: `[128000, 1229]` (shorter)
- C++ tokens: `[128000, 1229, 374]` (longer)

**Expected Output:**
```
❌ Token Sequence Mismatch
...
First diff at index: 2

Suggested fixes:
  ...
```

**Exit Code:** 2

---

## Integration Points

### Existing Code (xtask/src/main.rs)

**Before this change:**
```rust
// Tokenize
let rust_tokens = tokenizer.encode(&prompt, add_bos, parse_special)?;
let cpp_tokens = cpp_session.tokenize(&prompt)?;

// Compare logits (PROBLEM: skips token validation!)
compare_logits(rust_logits, cpp_logits)?;
```

**After this change:**
```rust
// Tokenize
let rust_tokens = tokenizer.encode(&prompt, add_bos, parse_special)?;
let cpp_tokens = cpp_session.tokenize(&prompt)?;

// NEW: Token parity pre-gate
validate_token_parity(&rust_tokens, &cpp_tokens, &prompt)?;

// Compare logits (only runs if tokens match)
compare_logits(rust_logits, cpp_logits)?;
```

---

## Dependencies

**Required:**
- `console` crate for colored output (`style()` function)
- `anyhow` for `Result<()>` type

**Already Available:** Both crates are already dependencies in xtask/Cargo.toml

---

## Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| C++ tokenization unavailable | Phase 3 will add direct token-based FFI; for now, gracefully handle missing C++ |
| Performance overhead | Slice comparison is O(n) with early-exit; negligible for typical prompts (<1000 tokens) |
| False positives | If legitimate cases exist where tokens differ, add `--skip-token-parity` flag |

---

## Future Enhancements

1. **Phase 3 Integration**: When `cpp_eval_with_tokens()` FFI is added, bypass C++ tokenization entirely
2. **Detailed Diff**: Show side-by-side comparison with highlighting (like `git diff`)
3. **Token Decoding**: Print decoded token strings alongside IDs for better debugging
4. **JSON Receipt**: Record token parity status in cross-validation receipts

---

**Status:** Ready for test creation and implementation
**Next Steps:**
1. Create test scaffolding (test-creator agent)
2. Implement `validate_token_parity()` function (impl-creator agent)
3. Integrate into `crossval-per-token` command
4. Validate with real cross-validation runs
