EXPLORATION SUMMARY: crossval-per-token Implementation Analysis
================================================================

COMPLETED: Comprehensive exploration of the `crossval-per-token` command implementation
with focus on tokenization, FFI interface, and token-parity pre-gate integration points.

DOCUMENTS CREATED
=================

1. CROSSVAL_PER_TOKEN_IMPLEMENTATION.md (13 KB)
   Location: /home/steven/code/Rust/BitNet-rs/docs/
   
   Contains:
   - Command definition and overview
   - Detailed tokenization flow analysis
   - Rust logits evaluation mechanism (eval_logits_all_positions)
   - C++ FFI interface documentation with key function calls
   - Logits comparison logic (cosine similarity, L2 distance)
   - FFI system architecture and safety patterns
   - Tokenization differences between crossval-per-token and CLI
   - Token-parity pre-gate integration points (3 locations identified)
   - Error handling approach
   - Crate interdependencies table
   - Quick reference file paths and line numbers

2. CROSSVAL_TOKEN_PARITY_PRE_GATE_DESIGN.md (18 KB)
   Location: /home/steven/code/Rust/BitNet-rs/docs/
   
   Contains:
   - Detailed pre-gate architecture with 3-level system
   - Complete implementation specifications for each pre-gate:
     * Pre-Gate 1: Token Sequence Validation
     * Pre-Gate 2: Logits Baseline Validation
     * Pre-Gate 3: Per-Position Shape & Range Validation
   - Full Rust code templates with structs and functions
   - Integration points with line-by-line code examples
   - Output examples for different scenarios
   - Environment variable control system
   - Testing strategy with unit test examples
   - Migration path with 3 phases
   - Summary table of benefits

KEY FINDINGS
============

1. TOKENIZATION MISMATCH (Critical Issue)
   -------
   Rust side:  tokenizer.encode(prompt, false, false)
               - NO BOS token
               - NO special tokens
               - Raw tokenization
   
   C++ side:   cpp_session.tokenize(prompt)
               - Includes special handling
               - May add BOS token
               - Uses llama.cpp tokenizer
   
   Impact: Different token sequences → false divergence at position 0

2. FFI ARCHITECTURE
   ----------------
   Wrapper Layer: bitnet_sys::wrapper::Session
   - Safe wrapper around llama.cpp
   - Deterministic settings: 1 thread, 2048 context, logits_all=true
   - Memory-safe with explicit Drop impls
   
   Availability: bitnet_sys::is_available()
   - Catches panics from FFI calls
   - Returns false if C++ not available
   
   Function Calls (main.rs lines 2944-2963):
   1. bitnet_sys::is_available() - Check C++ available
   2. bitnet_sys::wrapper::init_backend() - Initialize
   3. Session::load_deterministic() - Load model
   4. cpp_session.tokenize(prompt) - Tokenize (add_special=true)
   5. context.eval(&cpp_tokens, 0) - Evaluate tokens
   6. context.get_all_logits(n_tokens) - Get all logits

3. LOGITS EVALUATION FLOW
   ----------------------
   Rust: eval_logits_all_positions()
   - File: crates/bitnet-inference/src/parity.rs (lines 157-223)
   - Loads GGUF with pure Rust loader
   - Handles QK256 tensor conversion
   - Returns Vec<Vec<f32>> [positions][vocab]
   
   C++: Session::get_all_logits()
   - File: crates/bitnet-sys/src/wrapper.rs (lines 285-293)
   - Calls llama_get_logits_ith() per position
   - Returns Vec<Vec<f32>> [positions][vocab]

4. COMPARISON METRICS
   -------------------
   Function: compare_per_position_logits()
   File: crossval/src/logits_compare.rs (lines 49-102)
   
   Metrics:
   - Cosine Similarity (direction match, 0-1 scale)
   - L2 Distance (Euclidean distance)
   - Max Absolute Difference (peak value)
   - First Divergence Position (where cos_sim < 1 - threshold)
   
   Threshold: cos_tol parameter (default 0.999)
   Uses: (1.0 - cosine_sim) > 1e-4 check

5. ERROR HANDLING
   ---------------
   Fail-closed approach:
   - FFI unavailable: Bail with message
   - Model load failure: Propagate error
   - Tokenization errors: Context and bail
   - Logits comparison: Always reports, exit 1 if divergence
   - Exit codes: 0 (success), 1 (divergence), 2+ (pre-gate failures)

INTEGRATION POINTS (3 Locations for Pre-Gate)
==============================================

Location 1: After tokenization (main.rs ~line 2927)
- BEFORE: Load Rust model
- PRE-GATE: Check token_ids == cpp_tokens
- ACTION: Report mismatch, diagnose (BOS, special tokens, etc.)
- CONTROL: BITNET_STRICT_TOKENIZATION env var

Location 2: After Rust evaluation (main.rs ~line 2938)
- BEFORE: Load C++ model
- PRE-GATE: Validate Rust logits baseline
- CHECKS: Non-zero, finite, reasonable L2 norm
- ACTION: Report issues but continue

Location 3: Before comparison (main.rs ~line 2970)
- BEFORE: compare_per_position_logits()
- PRE-GATE: Validate shape and magnitude bounds
- CHECKS: Same sequence length, vocab size, no NaN/Inf
- ACTION: Fail with code 3 on critical issues

CRATE DEPENDENCIES
==================

bitnet_tokenizers::loader
  └─ load_tokenizer() - Universal tokenizer loader

bitnet_inference::parity
  └─ eval_logits_all_positions() - Rust forward pass, all positions
     File: crates/bitnet-inference/src/parity.rs

bitnet_sys::wrapper
  └─ Session - Safe C++ wrapper (llama.cpp)
     File: crates/bitnet-sys/src/wrapper.rs

bitnet_crossval::logits_compare
  └─ compare_per_position_logits() - Per-position comparison
     File: crossval/src/logits_compare.rs

FILE PATHS & LINE NUMBERS
=========================

Command Parsing:
  xtask/src/main.rs:405-430          CrossvalPerToken struct definition
  xtask/src/main.rs:895-898          Command dispatch

Implementation:
  xtask/src/main.rs:2901-3041        crossval_per_token_cmd() function
  
Tokenization:
  xtask/src/main.rs:2920-2922        Rust tokenization (raw, no template)
  crates/bitnet-sys/src/wrapper.rs:144-186  C++ tokenization (llama.cpp)
  crates/bitnet-sys/src/wrapper.rs:351-352  Session::tokenize() wrapper

Rust Evaluation:
  crates/bitnet-inference/src/parity.rs:157-223  eval_logits_all_positions()

C++ Evaluation:
  crates/bitnet-sys/src/wrapper.rs:329-394  Session struct implementation
  crates/bitnet-sys/src/wrapper.rs:344-348  load_deterministic() settings
  crates/bitnet-sys/src/wrapper.rs:285-293  get_all_logits() function

Comparison:
  crossval/src/logits_compare.rs:49-102       compare_per_position_logits()
  
FFI System:
  crates/bitnet-sys/src/lib.rs:71-77         is_available() check
  crates/bitnet-sys/src/wrapper.rs:102-113   Model Drop impl
  crates/bitnet-sys/src/wrapper.rs:316-327   Context Drop impl

PRE-GATE DESIGN (From Design Document)
=======================================

Three-Level Pre-Gate System:
  PRE-GATE 1: Token Sequence Validation
    Module: bitnet_crossval::token_parity_check
    Function: check_token_parity(rust_tokens, cpp_tokens)
    Output: TokenParityResult struct with diagnostics
    Diagnoses: BOS token, special token handling, root cause
    
  PRE-GATE 2: Logits Baseline Validation
    Module: bitnet_crossval::logits_validation
    Function: validate_logits_baseline(logits)
    Output: LogitsBaselineCheck struct
    Checks: Zero logits, NaN/Inf, L2 norm
    
  PRE-GATE 3: Comparison Validation
    Module: bitnet_crossval::comparison_validation
    Function: validate_before_comparison(rs_logits, cpp_logits)
    Output: ComparisonValidation struct
    Checks: Shape match, vocab size, magnitude bounds

Environment Variables:
  BITNET_STRICT_TOKENIZATION=1    Exit on token mismatch (code 2)
  BITNET_STRICT_VALIDATION=1      Exit on logits issues (code 3)
  BITNET_PRE_GATE_JSON=1          Output JSON format
  BITNET_PRE_GATE_VERBOSE=1       Show detailed checks

NEXT STEPS (Recommended)
=========================

Phase 1: Implement Pre-Gate Infrastructure
  [ ] Create bitnet_crossval::token_parity_check module
  [ ] Create bitnet_crossval::logits_validation module
  [ ] Create bitnet_crossval::comparison_validation module
  [ ] Add structs and functions from design doc

Phase 2: Integrate Pre-Gates into crossval-per-token
  [ ] Add location 1: Token validation (after line 2926)
  [ ] Add location 2: Logits baseline (after line 2938)
  [ ] Add location 3: Comparison validation (before line 2972)
  [ ] Add console output with diagnostics

Phase 3: Testing & Validation
  [ ] Write unit tests for each pre-gate
  [ ] Test BOS token detection
  [ ] Test special token diagnostics
  [ ] Test NaN/Inf handling
  [ ] Test output formatting (text vs JSON)

Phase 4: Documentation & Examples
  [ ] Add command usage examples
  [ ] Document environment variables
  [ ] Add troubleshooting guide
  [ ] Update CLAUDE.md with pre-gate info

REFERENCES
==========

Main Analysis Document:
  docs/CROSSVAL_PER_TOKEN_IMPLEMENTATION.md
  
Pre-Gate Design Document:
  docs/CROSSVAL_TOKEN_PARITY_PRE_GATE_DESIGN.md

Related Existing Docs:
  docs/CROSSVAL.md
  docs/CROSSVAL_TESTING.md
  CLAUDE.md (project status and commands)

