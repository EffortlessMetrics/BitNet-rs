# xtask crossval-per-token Command Analysis

## Overview

This directory contains comprehensive analysis of the `xtask crossval-per-token` command implementation in BitNet-rs, including identification of a critical execution order bug and complete documentation of the command's architecture.

## Documents in This Analysis

### 1. Executive Summary (Quick Reference)
**File**: `CROSSVAL_PER_TOKEN_EXECUTIVE_SUMMARY.md` (9 KB)

Quick-reference guide covering:
- Critical bug identification
- Command signature
- Execution flow (correct vs current)
- Integration point matrix
- Error handling & exit codes
- Missing flags
- Known issues
- Recommendations

**Best for**: Quick understanding of the issue and status

### 2. Complete Implementation Analysis (Comprehensive)
**File**: `xtask-crossval-per-token-implementation-analysis.md` (28 KB)

Thorough technical deep-dive containing:
- 15 major sections with subsections
- Complete line-by-line execution flow (phases 0-7)
- All CLI argument definitions
- Command dispatch mechanism
- Handler function signatures
- Integration points with exact locations
- Error handling patterns & exit codes
- Missing flags enumeration
- Diagnostic & logging integration
- Detailed component analysis (A-E)
- Feature dependencies & gates
- Test infrastructure status
- Known issues & limitations
- Documentation cross-references
- Recommendations by priority
- 4 appendices (call stack, data structures)

**Best for**: Deep understanding, implementation planning, bug fixing

## Key Finding

**CRITICAL BUG: Execution Order Violation**

```
Current (Buggy):
  Line 2933: eval_logits_all_positions() ← Rust logits (20-30 seconds)
  Line 2954-2957: Init C++, tokenize
  Line 2963: validate_token_parity() ← CHECKS TOKENS TOO LATE!
  Line 2972-2975: eval C++ logits ← WASTED if parity fails!

Correct (Per Spec):
  Initialize C++ (lines 2954-2957)
  Tokenize both implementations
  validate_token_parity() ← FIRST (before expensive ops!)
  IF PARITY OK: eval_logits_all_positions()
  IF PARITY OK: eval C++ logits
```

**Impact**: 20-30 seconds of wasted Rust inference computation per tokenization mismatch

**Specification Reference**: See `docs/explanation/token-parity-pregate.md`

## Command Overview

```bash
cargo run -p xtask -- crossval-per-token \
  --model <PATH>           # Required: GGUF model file
  --tokenizer <PATH>       # Required: tokenizer.json
  --prompt <STRING>        # Required: input prompt
  --max-tokens <N>         # Optional: default=4 (unused, reserved)
  --cos-tol <F>           # Optional: default=0.999 (display-only)
  --format <STR>          # Optional: "text" or "json" (default="text")
```

**Feature Requirements**:
- Compile: `--features crossval-all` or `--features inference,crossval,ffi`
- Runtime: C++ FFI availability via `bitnet_sys::is_available()`

## Main Implementation Location

**File**: `xtask/src/main.rs`
- **Lines 389-430**: CLI command definition (clap)
- **Lines 897-899**: Command dispatch
- **Lines 2901-3053**: Handler function `crossval_per_token_cmd()`

## Supporting Modules

| Module | Location | Purpose |
|--------|----------|---------|
| Token Parity | `crossval/src/token_parity.rs:79-110` | Pre-gate validation |
| Logits Comparison | `crossval/src/logits_compare.rs:49-102` | Per-position metrics |
| Rust Evaluation | `crates/bitnet-inference/src/parity.rs:157-223` | Logits calculation |
| C++ FFI Wrapper | `crates/bitnet-sys/src/wrapper.rs:285-293` | C++ integration |

## Exit Codes

| Code | Meaning | Location |
|------|---------|----------|
| 0 | Success (all positions match) | Line 3048 |
| 1 | Divergence found | Line 3045 |
| 2 | Token parity mismatch | Line 2966 |
| 1 | FFI unavailable | Line 2945 |
| 1 | Other errors (model load, tokenization) | Various |

## Known Issues

### Issue #254: Shape Mismatch in Layer-Norm
- Blocks real inference tests
- Status: In analysis phase

### Issue #260: Mock Elimination
- Test infrastructure contains mock paths
- Status: Awaiting refactoring

### Issue #469: Tokenizer Parity & FFI Build
- Blocks cross-validation test completion
- Status: Active development

### This Report's Bug: Execution Order Violation
- Lines: 2933 vs 2963
- Type: Performance bug (not correctness)
- Fix Effort: Low (~20 lines)
- Status: Identified, needs fixing

## Missing Flags (Feature Gaps)

High Priority:
- `--prompt-template` (control tokenization template)
- `--cpp-backend` (GPU vs CPU for C++)
- `--no-bos` / `--add-bos` (BOS control)

Medium Priority:
- `--cos-similarity-threshold` (override hardcoded 1e-4)
- `--max-positions` (limit eval to first N positions)
- `--auto-trace` (automatic trace capture)

## Recommendations

### CRITICAL (Fix Now)
1. **Reorder logits evaluation** to after parity check
   - Saves 20-30s per divergent case
   - Move lines 2933-2938 after line 2968

2. **Fix CLI threshold confusion**
   - Document that `--cos-tol` is display-only OR
   - Make it control actual divergence threshold OR
   - Add `--divergence-threshold` parameter

### HIGH (v0.2 Target)
3. Add `--prompt-template` support
4. Add `--no-bos` / `--add-bos` flags
5. Add `--cpp-backend` parameter

### MEDIUM (Future)
6. Add `--max-positions` (limit eval scope)
7. Add `--auto-trace` (auto-capture traces)
8. Add `--device` (Rust GPU/CPU selection)
9. Complete integration tests

## Testing Status

- **Unit Tests**: 12 token parity tests, 3 logits comparison tests
- **Ignored Tests**: 4 tests (require stderr capture)
- **Integration Tests**: Blocked by #254, #260, #469
- **Pass Rate**: 12/12 parity tests (when not blocked)

## How to Use This Analysis

1. **For Quick Overview**: Start with `CROSSVAL_PER_TOKEN_EXECUTIVE_SUMMARY.md`
2. **For Implementation Work**: Use `xtask-crossval-per-token-implementation-analysis.md`
3. **For Bug Investigation**: See Section 5 in the full report (Execution Order Bug)
4. **For Testing**: See Section 12 (Test Infrastructure)
5. **For Enhancement Planning**: See Section 15 (Recommendations)

## Related Documentation

- `docs/explanation/token-parity-pregate.md` - Token parity specification
- `docs/explanation/cpp-eval-with-tokens.md` - C++ evaluation details
- `docs/howto/cpp-setup.md` - C++ reference setup
- `CLAUDE.md` - Project status and crossval commands

## Analysis Metrics

- **Total Coverage**: Very Thorough
- **Sections**: 15 major + 4 appendices
- **Line Numbers**: All documented with exact coordinates
- **Files Analyzed**: 5+ source files, 2+ documentation files
- **Integration Points**: 6 documented with signatures
- **Error Patterns**: 5 documented with handlers
- **Missing Features**: 8 enumerated with severity
- **Recommendations**: 9 prioritized from critical to medium

---

**Report Generated**: October 25, 2025
**BitNet-rs Status**: v0.1.0-qna-mvp
**Branch**: feat/comprehensive-integration-qk256-envguard-receipts-strict-avx2

