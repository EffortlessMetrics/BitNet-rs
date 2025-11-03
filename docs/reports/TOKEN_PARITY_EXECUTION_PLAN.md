# Token Parity & Cross-Validation: Execution Plan

**Date:** 2025-10-24
**Status:** Exploration Complete → Ready for Implementation
**Goal:** Fix duplicate BOS tokens and implement robust token-parity validation

---

## Executive Summary

Based on comprehensive codebase exploration, BitNet.rs has **well-designed** BOS/template infrastructure with proper safeguards. However, three **critical gaps** block reliable cross-validation:

1. **CLI flag exposure**: `--print-input-tokens` exists in code but may not be wired to clap properly
2. **Token parity pre-gate**: crossval-per-token compares logits without first validating token sequences match
3. **FFI token interface**: C++ wrapper only accepts strings, not token IDs directly

---

## Exploration Findings

### ✅ What's Working Well

#### BOS/Template Safeguards (crates/bitnet-inference/src/prompt_template.rs)

```rust
pub fn should_add_bos(&self) -> bool {
    match self {
        Self::Raw | Self::Instruct => true,           // Add BOS via tokenizer
        Self::Llama3Chat => false,                     // Template includes <|begin_of_text|>
    }
}
```

**Duplicate Prevention** (crates/bitnet-tokenizers/src/hf_tokenizer.rs:128):
```rust
if add_bos
    && let Some(bos) = self.bos_id
    && (ids.is_empty() || ids[0] != bos)  // Only add if not already present
{
    ids.insert(0, bos);
}
```

#### Trace Infrastructure (bitnet-trace)

- **6 capture points**: Embeddings, Q/K/V projections, attention output, FFN output, logits
- **Environment-gated**: `BITNET_TRACE_DIR=/path/to/traces`
- **Blake3 hashing**: Fast comparison between Rust and C++ traces
- **Comparison tool**: `cargo run -p xtask -- trace-diff /tmp/rs /tmp/cpp`

#### Cross-Validation Infrastructure (crossval/)

- **Logits comparison**: Cosine similarity, L2 distance, max absolute difference
- **Per-position analysis**: Identifies first divergence token
- **Receipt generation**: JSON reports with parity metrics

### ⚠️ Critical Gaps

#### 1. CLI Flag Not Exposed

**Expected:** `--print-input-tokens` shows token IDs after tokenization
**Reality:** CLI rejects the flag (unknown argument error)
**Location:** Flag defined at `crates/bitnet-cli/src/commands/inference.rs:308`

```rust
/// Print input token IDs after tokenization (debug tokenizer/template/BOS behavior)
#[arg(long)]
pub print_input_tokens: bool,
```

**Root Cause:** Likely missing from `run` subcommand Args struct or feature-gated

#### 2. Token Parity Pre-Gate Missing

**Expected:** Validate token sequences match before comparing logits
**Reality:** `crossval-per-token` only compares logits, not tokens first
**Impact:** Tokenization mismatches (duplicate BOS) mask as logits divergence

**Current Flow:**
```
Rust tokenize → Rust logits
C++ tokenize  → C++ logits
                    ↓
              Compare logits (if tokens differ, this is meaningless!)
```

**Required Flow:**
```
Rust tokenize → tokens_rs
C++ tokenize  → tokens_cpp
                    ↓
              Compare tokens (FAIL FAST if mismatch)
                    ↓
         (if match) Compare logits
```

#### 3. FFI is String-Based Only

**Current API** (bitnet-ffi/src/c_api.rs):
```c
int bitnet_inference(const char* model_id, const char* prompt, ...)
```

**Problem:** C++ does its own tokenization (may differ from Rust)

**Required:**
```c
int cpp_eval_with_tokens(const uint32_t* token_ids, size_t n_ids, ...)
```

**Benefit:** Guarantees identical token sequences in both implementations

---

## Implementation Phases

### Phase 1: Fix CLI --print-input-tokens Flag

**Status:** IN PROGRESS
**Goal:** Enable token ID inspection for debugging
**Estimated Time:** 15-30 minutes

**Tasks:**
- [ ] Verify flag is in correct Args struct (InferenceCommand)
- [ ] Check if feature-gated or missing from clap derivation
- [ ] Test with `cargo run -p bitnet-cli -- run --help | grep print`
- [ ] Validate output shows provenance: `template=X add_bos=Y`

**Acceptance:**
```bash
cargo run -p bitnet-cli --features cpu,full-cli -- run \
  --model models/clean/clean-f16-fixed.gguf \
  --tokenizer models/.../tokenizer.json \
  --prompt "What is 2+2?" \
  --max-tokens 1 \
  --print-input-tokens

# Expected output:
# Tokenization provenance: template=instruct add_bos=true (BOS added by tokenizer)
# Input tokens (5): [128000, 1229, 374, 220, 17, ...]
```

### Phase 2: Add Token-Parity Pre-Gate

**Status:** PENDING
**Goal:** Fail fast if Rust/C++ token sequences mismatch
**Estimated Time:** 30-45 minutes

**Where:** `xtask/src/main.rs` → `crossval-per-token` command

**Add Before Logits Comparison:**
```rust
// 1. Tokenize in Rust (using same CLI flags)
let rust_tokens = tokenizer.encode(&formatted_prompt, add_bos, parse_special)?;

// 2. Get C++ tokens (via FFI or new token-based API)
let cpp_tokens = cpp_session.tokenize(&formatted_prompt)?;

// 3. Compare sequences
if rust_tokens != cpp_tokens {
    eprintln!("❌ Token mismatch (fix BOS/template before comparing logits)");
    eprintln!("Rust tokens: {:?}", rust_tokens);
    eprintln!("C++ tokens:  {:?}", cpp_tokens);
    eprintln!("First diff at index: {}", find_first_diff(&rust_tokens, &cpp_tokens));
    eprintln!("\nSuggested fixes:");
    eprintln!("  - Use --prompt-template raw");
    eprintln!("  - Add --no-bos flag");
    eprintln!("  - Check GGUF chat_template metadata");
    std::process::exit(2);  // Exit 2 = usage error
}

// 4. Only proceed if tokens match
compare_logits(rust_logits, cpp_logits)?;
```

**Acceptance:**
- Mismatched tokens → clear error with fix-it guidance, exit 2
- Matched tokens → proceeds to logits comparison

### Phase 3: Add FFI Token-Based Evaluation

**Status:** PENDING
**Goal:** Bypass C++ tokenization entirely
**Estimated Time:** 45-90 minutes

**Option A: Add FFI Function (Preferred)**

**C Header** (bitnet-sys/include/bitnet_cpp.h):
```c
// Evaluate with pre-tokenized inputs
int cpp_eval_with_tokens(
    const uint32_t* token_ids,     // Input token IDs
    size_t n_ids,                   // Number of tokens
    size_t max_new_tokens,          // Max decode steps
    float** out_logits_per_step,    // Output: [steps][vocab_size]
    size_t* out_steps,              // Output: number of steps
    size_t* out_vocab_size          // Output: vocabulary size
);

void cpp_free_logits(float* logits);
```

**Rust Wrapper** (bitnet-crossval/src/lib.rs):
```rust
pub fn cpp_eval_with_tokens(
    tokens: &[u32],
    max_new_tokens: usize,
) -> Result<Vec<Vec<f32>>> {
    // Call FFI, convert to Vec<Vec<f32>>
}
```

**Option B: CSV Shim (Fallback if FFI is Hard)**

1. Write `tokens.csv` from Rust
2. C++ tool reads CSV, runs forward pass, dumps `logits.json`
3. `xtask` reads JSON and compares

**Acceptance:**
- `crossval-per-token` can run with **Rust-computed tokens only**
- No C++ tokenization needed

### Phase 4: Add LM-Head Tie Validation

**Status:** PENDING
**Goal:** Catch head/vocab orientation bugs early
**Estimated Time:** 30-45 minutes

**Where:** `crates/bitnet-models/src/weight_mapper.rs` (debug builds only)

**Add After Model Build:**
```rust
#[cfg(debug_assertions)]
{
    let e_transposed = embeddings.to_dtype(F32)?.transpose(0,1)?; // [H,V]
    let w_head = lm_head.to_dtype(F32)?;                          // Expected [H,V]

    let diff_norm = (&w_head - &e_transposed)?.sqr()?.sum_all()?.to_scalar::<f32>()?.sqrt();
    let base_norm = e_transposed.sqr()?.sum_all()?.to_scalar::<f32>()?.sqrt().max(1e-12);
    let rel_error = diff_norm / base_norm;

    assert!(
        rel_error < 1e-5,
        "LM head not tied or wrong orientation: rel_error={:.2e}",
        rel_error
    );
}
```

**Acceptance:**
- Debug builds: assertion catches orientation bugs immediately
- Release builds: no overhead

### Phase 5: Add RoPE/Scale Constants Logging

**Status:** PENDING
**Goal:** Enable precise parity debugging
**Estimated Time:** 15-30 minutes

**Where:** `crates/bitnet-models/src/transformer.rs` (right before softmax)

**Add:**
```rust
debug!(
    "Attention constants: d_k={}, scale={:.6f}, rope_base={:.1f}, n_past={}",
    self.head_dim,
    1.0 / (self.head_dim as f32).sqrt(),
    self.rope_base,
    seq
);
```

**Acceptance:**
- Rust and C++ logs show identical constants
- Any mismatch → immediate fix

### Phase 6: Comprehensive Parity Validation

**Status:** PENDING
**Goal:** End-to-end validation with receipts
**Estimated Time:** 30-60 minutes (run + analysis)

**Run:**
```bash
# 1. Token parity check
cargo run -p xtask --features crossval-all -- crossval-per-token \
  --model models/clean/clean-f16-fixed.gguf \
  --tokenizer models/.../tokenizer.json \
  --prompt "What is 2+2?" \
  --max-tokens 4 \
  --cos-tol 1e-3

# 2. If t=0 diverges: run head-tie validation
# (already in debug builds from Phase 4)

# 3. If still diverges: capture traces
BITNET_TRACE_DIR=/tmp/rs RUST_LOG=warn \
  cargo run -p bitnet-cli --features cpu,trace -- run \
  --model models/clean/clean-f16-fixed.gguf \
  --tokenizer models/.../tokenizer.json \
  --prompt "What is 2+2?" \
  --max-tokens 1 --greedy

# C++ traces → /tmp/cpp (similar command)

# 4. Compare traces
cargo run -p xtask -- trace-diff /tmp/rs /tmp/cpp
```

**Acceptance:**
- Per-token parity: ✓ all positions
- Trace diff: "All tracepoints match"
- Tiny deterministic decode produces coherent text
- Receipt: `compute_path="real"`, no mock, no runtime corrections

---

## Definition of Done

### Token Sequences
- [ ] Single BOS token (or none with `--no-bos`)
- [ ] `--print-input-tokens` shows expected IDs
- [ ] Provenance logged: `template=X add_bos=Y`

### Token Parity
- [ ] Pre-gate validation in `crossval-per-token`
- [ ] Mismatch → clear error + fix-it guidance + exit 2
- [ ] Match → proceeds to logits comparison

### FFI Token Interface
- [ ] `cpp_eval_with_tokens()` accepts token IDs directly
- [ ] No C++ tokenization needed for parity
- [ ] CSV shim fallback (if FFI delayed)

### Validation Gates
- [ ] LM-head tie assertion (debug builds)
- [ ] RoPE/scale constants logged
- [ ] Per-token parity passes (F16 model, micro prompt)
- [ ] Trace-diff: "All tracepoints match"

### Output Quality
- [ ] Tiny deterministic decode: coherent text
- [ ] Receipts: `compute_path="real"`, no corrections

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| C++ wrapper tokenization differs | **Phase 3**: Direct token IDs bypass wrapper entirely |
| Double BOS returns | **Phase 2**: Token pre-gate exits early with guidance |
| Feature explosion | Use `crossval-all` feature flag; update docs |
| FFI complexity | **Fallback**: CSV shim (Phase 3 Option B) |

---

## File References

### CLI & Tokenization
- `crates/bitnet-cli/src/commands/inference.rs:308` - print_input_tokens flag
- `crates/bitnet-cli/src/commands/inference.rs:1432` - should_add_bos() logic
- `crates/bitnet-inference/src/prompt_template.rs:212` - Template BOS policy
- `crates/bitnet-tokenizers/src/hf_tokenizer.rs:128` - Duplicate BOS prevention

### Cross-Validation
- `xtask/src/main.rs` - crossval-per-token command (~line 410-440)
- `crossval/src/logits_compare.rs` - Logits comparison metrics
- `crossval/tests/parity.rs:141` - Working token validation test

### FFI & Trace
- `bitnet-ffi/src/c_api.rs` - Current string-based API
- `bitnet-sys/src/wrapper.rs:144` - Session::tokenize()
- `bitnet-trace/` - Trace capture infrastructure
- `xtask/src/trace_diff.rs` - Trace comparison tool

### Weight Mapping
- `crates/bitnet-models/src/weight_mapper.rs` - Weight orientation logic
- `crates/bitnet-models/src/transformer.rs` - RoPE/attention constants

---

## Next Steps

1. **Verify Phase 1** - Check `--print-input-tokens` flag status
2. **Implement Phase 2** - Add token pre-gate (highest priority)
3. **Design Phase 3** - FFI token interface or CSV shim
4. **Execute Phases 4-6** - Validation gates + comprehensive parity

**Status Tracking:** See todo list for current progress

---

**Generated:** 2025-10-24
**Explorer Agents:** BOS/template, token parity, FFI/trace
**Documentation:** 6 comprehensive reports in `docs/reports/`
