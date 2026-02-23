# Token Parity & Cross-Validation: Exploration Summary

**Date:** 2025-10-24
**Status:** ✅ Exploration Complete → Ready for Implementation
**Goal:** Fix duplicate BOS and implement robust cross-validation

---

## Executive Summary

Comprehensive codebase exploration reveals:

✅ **BOS/template infrastructure is well-designed** with proper safeguards
⚠️ **Token parity pre-gate is MISSING** (critical gap blocking cross-validation)
⚠️ **FFI is string-based only** (needs token-level interface)
✅ **Trace infrastructure is production-ready** (Blake3 hashing, 6 capture points)
✅ **`--dump-ids` flag exists** (formerly called `--print-input-tokens` in exploration docs)

**Priority Fix:** Add token-parity pre-gate to `crossval-per-token` command (Phase 2)

---

## Key Findings

### ✅ Phase 1 Complete: CLI Flag Discovery

**Finding:** The token inspection flag **already exists** as `--dump-ids`

**Two Command Structures Exist:**

1. **Active CLI** (main.rs:194-289) - `Run` variant with `--dump-ids`
   ```bash
   cargo run -p bitnet-cli -- run --model X --prompt Y --dump-ids
   ```

2. **Unused Struct** (commands/inference.rs:132) - `InferenceCommand` with `--print-input-tokens`

**Resolution:** Use `--dump-ids` for all token debugging

**Example Usage:**
```bash
RUST_LOG=warn cargo run -p bitnet-cli --no-default-features --features cpu,full-cli -- run \
  --model models/clean/clean-f16-fixed.gguf \
  --tokenizer models/.../tokenizer.json \
  --prompt "What is 2+2?" \
  --max-tokens 1 \
  --dump-ids  # ← Shows token IDs

# Output shows:
# Token IDs: [128000, 1229, 374, 220, 17]
```

---

### ⚠️ Phase 2 Ready: Token Parity Pre-Gate (CRITICAL)

**Problem:** `crossval-per-token` compares logits without first validating tokens match

**Current Flow (BROKEN):**
```
Rust tokenize → Rust logits  ┐
C++ tokenize  → C++ logits   ├─→ Compare logits (meaningless if tokens differ!)
                              ┘
```

**Required Flow:**
```
Rust tokenize → rust_tokens  ┐
C++ tokenize  → cpp_tokens   ├─→ Compare tokens (FAIL FAST if mismatch)
                              ┘                      ↓
                                           (if match) Compare logits
```

**Implementation Status:**
- ✅ Specification created: `docs/explanation/token-parity-pregate.md`
- ⏳ Test scaffolding: NEXT
- ⏳ Implementation: NEXT
- ⏳ Integration: NEXT

**Location:** `xtask/src/main.rs` → `crossval-per-token` command (~line 410-440)

**Estimated LOC:** +50 lines (validation function + integration)

---

### BOS/Template Safeguards (Already Working)

**Template BOS Policy** (`crates/bitnet-inference/src/prompt_template.rs:212`):
```rust
pub fn should_add_bos(&self) -> bool {
    match self {
        Self::Raw | Self::Instruct => true,           // Add BOS via tokenizer
        Self::Llama3Chat => false,                     // Template includes <|begin_of_text|>
    }
}
```

**Duplicate BOS Prevention** (`crates/bitnet-tokenizers/src/hf_tokenizer.rs:128`):
```rust
if add_bos
    && let Some(bos) = self.bos_id
    && (ids.is_empty() || ids[0] != bos)  // ← Only add if not already present
{
    ids.insert(0, bos);
}
```

**Result:** Template + tokenizer coordination prevents duplicate BOS in Rust code.

---

### Trace Infrastructure (Production-Ready)

**Capabilities:**
- **6 capture points**: Embeddings, Q/K/V projections, attention output, FFN output, logits
- **Blake3 hashing**: Fast comparison (O(1) per tracepoint)
- **JSON format**: Machine-readable with RMS, shape, dtype metadata
- **Environment-gated**: `BITNET_TRACE_DIR=/path/to/traces`

**Usage:**
```bash
# Capture Rust traces
BITNET_TRACE_DIR=/tmp/rs RUST_LOG=warn \
  cargo run -p bitnet-cli --features cpu,trace -- run \
  --model models/clean/clean-f16-fixed.gguf \
  --tokenizer models/.../tokenizer.json \
  --prompt "What is 2+2?" \
  --max-tokens 1 --greedy

# Capture C++ traces (similar command) → /tmp/cpp

# Compare traces
cargo run -p xtask -- trace-diff /tmp/rs /tmp/cpp
# Output: "First divergence at seq=0, layer=3, stage=attn_scores_softmax"
```

---

### Cross-Validation Infrastructure

**Components:**
- ✅ Logits comparison (`crossval/src/logits_compare.rs`) - Cosine, L2, MAD metrics
- ✅ Per-position analysis - Identifies first divergence token
- ✅ Receipt generation - JSON reports with parity status
- ❌ **Token parity pre-gate** - MISSING (Phase 2 fix)

**Current Limitations:**
- C++ wrapper does its own tokenization (may differ from Rust)
- No pre-validation of token sequences before logits comparison
- String-based FFI API only (needs token-level interface for Phase 3)

---

## Documentation Generated

### Comprehensive Reports (docs/reports/)

1. **BOS_TEMPLATE_HANDLING_EXPLORATION.md** (17KB)
   - Template resolution logic (3-tier priority)
   - BOS injection points (template vs tokenizer)
   - Duplicate prevention safeguards
   - Auto-detection heuristics

2. **CROSSVAL_TOKEN_PARITY_INFRASTRUCTURE.md** (17KB)
   - Logits comparison metrics
   - FFI wrapper architecture
   - Token validation patterns
   - Receipt schema

3. **CROSSVAL_TOKEN_PARITY_CODE_SNIPPETS.md** (18KB)
   - Copy-paste implementation examples
   - Token pre-gate validation code
   - FFI interface designs

4. **FFI_TRACE_INFRASTRUCTURE.md** (21KB)
   - 30+ C FFI functions
   - Trace capture workflow
   - Blake3 comparison details

5. **TOKEN_PARITY_EXECUTION_PLAN.md** (Current document)
   - 6-phase implementation roadmap
   - Definition of done criteria
   - Risk mitigation strategies

### Specifications (docs/explanation/)

6. **token-parity-pregate.md** (NEW - just created)
   - Functional requirements
   - Design architecture
   - Acceptance criteria
   - Test scenarios

---

## Implementation Roadmap

### ✅ Phase 1: CLI Flag (COMPLETE)

**Status:** Resolved - use `--dump-ids` flag
**Action:** Update user documentation to reference `--dump-ids`

### ⏳ Phase 2: Token Parity Pre-Gate (IN PROGRESS)

**Status:** Specification complete, ready for implementation
**Estimated Time:** 30-45 minutes
**Next Steps:**
1. Create test scaffolding (test-creator agent)
2. Implement `validate_token_parity()` function (impl-creator agent)
3. Integrate into `xtask/src/main.rs` crossval-per-token
4. Validate with test scenarios

**Acceptance:**
- ✓ Token mismatch → clear error + exit 2
- ✓ Token match → silent success, proceeds to logits

### ⏳ Phase 3: FFI Token Interface (PENDING)

**Status:** Design sketched, awaiting Phase 2 completion
**Estimated Time:** 45-90 minutes
**Options:**
- **A (Preferred):** Add `cpp_eval_with_tokens(token_ids[])` FFI function
- **B (Fallback):** CSV shim (write tokens.csv, C++ reads and evaluates)

### ⏳ Phase 4: LM-Head Tie Validation (PENDING)

**Status:** Design complete
**Estimated Time:** 30-45 minutes
**Location:** `crates/bitnet-models/src/weight_mapper.rs` (debug builds)

**Add:**
```rust
#[cfg(debug_assertions)]
{
    let rel_error = compute_weight_tie_error(embeddings, lm_head)?;
    assert!(
        rel_error < 1e-5,
        "LM head not tied or wrong orientation: rel_error={:.2e}",
        rel_error
    );
}
```

### ⏳ Phase 5: RoPE/Scale Logging (PENDING)

**Status:** Simple addition
**Estimated Time:** 15-30 minutes
**Location:** `crates/bitnet-models/src/transformer.rs` (before softmax)

### ⏳ Phase 6: Comprehensive Parity Validation (PENDING)

**Status:** Awaits Phases 2-5
**Estimated Time:** 30-60 minutes (run + analysis)

**Checklist:**
- Run per-token parity (F16 model, micro prompt)
- Validate head-tie assertion (debug build)
- Capture + compare traces (if needed)
- Generate receipt with `compute_path="real"`

---

## Files Modified/Created

### Created (Exploration Phase)
- `docs/reports/BOS_TEMPLATE_HANDLING_EXPLORATION.md`
- `docs/reports/CROSSVAL_TOKEN_PARITY_INFRASTRUCTURE.md`
- `docs/reports/CROSSVAL_TOKEN_PARITY_CODE_SNIPPETS.md`
- `docs/reports/CROSSVAL_EXPLORATION_INDEX.md`
- `docs/reports/FFI_TRACE_INFRASTRUCTURE.md`
- `docs/reports/QUICK_FFI_TRACE_REFERENCE.md`
- `docs/reports/FFI_TRACE_ARCHITECTURE_DIAGRAM.txt`
- `docs/reports/FFI_TRACE_EXPLORATION_INDEX.md`
- `docs/reports/TOKEN_PARITY_EXECUTION_PLAN.md`
- `docs/explanation/token-parity-pregate.md`
- `EXPLORATION_SUMMARY.md` (this file)

### To Modify (Phase 2)
- `xtask/src/main.rs` - Add `validate_token_parity()` and integrate into `crossval-per-token`
- `xtask/tests/` - Add token parity tests

### To Modify (Phase 3)
- `bitnet-sys/include/bitnet_cpp.h` - Add `cpp_eval_with_tokens()` C header
- `bitnet-crossval/src/lib.rs` - Add Rust wrapper for token-based evaluation

### To Modify (Phase 4)
- `crates/bitnet-models/src/weight_mapper.rs` - Add LM-head tie assertion

### To Modify (Phase 5)
- `crates/bitnet-models/src/transformer.rs` - Add RoPE/scale logging

---

## Definition of Done

### Token Sequences
- [x] `--dump-ids` flag documented and working
- [x] BOS/template safeguards validated
- [ ] Token parity pre-gate implemented

### Cross-Validation
- [ ] Token mismatch → clear error + exit 2
- [ ] Token match → proceeds to logits comparison
- [ ] Per-token parity passes (F16 model)
- [ ] Trace-diff: "All tracepoints match"

### Output Quality
- [ ] Tiny deterministic decode produces coherent text
- [ ] Receipt: `compute_path="real"`, no mock, no corrections

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| C++ tokenization unavailable | Medium | High | Phase 3: Direct token IDs via FFI |
| Double BOS returns | Low | Medium | Token pre-gate catches immediately |
| Feature explosion | Low | Low | Use `crossval-all` feature flag |
| FFI complexity | Medium | Medium | Fallback: CSV shim (Phase 3B) |

---

## Next Actions

**Immediate (< 1 hour):**
1. Create test scaffolding for token-parity pre-gate
2. Implement `validate_token_parity()` function
3. Integrate into `crossval-per-token` command
4. Run validation tests

**Short Term (1-2 hours):**
5. Add FFI token interface (Phase 3)
6. Add LM-head tie validation (Phase 4)
7. Add RoPE/scale logging (Phase 5)

**Medium Term (2-4 hours):**
8. Run comprehensive parity validation (Phase 6)
9. Generate final receipts and documentation
10. Update CLAUDE.md with new workflows

---

**Status:** Ready for Phase 2 implementation via test-creator → impl-creator agents

**Exploration Agents Used:**
- `Explore` (BOS/template, crossval, FFI/trace)
- Documentation generation (10 comprehensive files)

**Total Lines Explored:** ~15,000+ (across 25+ files)
**Documentation Generated:** ~100KB (10 markdown files)
**Time Spent:** ~2 hours (exploration + documentation)
