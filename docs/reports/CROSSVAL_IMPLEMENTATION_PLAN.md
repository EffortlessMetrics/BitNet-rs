# Cross-Validation Implementation Plan
**Date**: 2025-10-24
**Goal**: Pinpoint exact divergence between BitNet-rs and bitnet.cpp

---

## Exploration Summary ✅

Three comprehensive explorations completed:

1. **Crossval Infrastructure** ✅
   - Mature receipt system (v1.0.0)
   - KernelRecorder pattern (Arc<Mutex> design)
   - Parity harness with 1e-4 tolerance
   - **Reusable patterns identified**

2. **Inference Pipeline** ✅
   - **275 tracepoints mapped** (11 per layer × 24 + entry/exit)
   - Single file: `crates/bitnet-models/src/transformer.rs`
   - Environment-gated debug already exists
   - QK256 dispatch points identified

3. **bitnet.cpp Integration** ✅
   - FFI bridge robust and well-designed
   - **Critical limitation**: llama.cpp doesn't expose intermediate layer activations
   - Workaround: per-position logits comparison available
   - Global session pattern prevents memory issues

**Documentation created:**
- 9 comprehensive markdown files (~150 KB total)
- Line-by-line tracepoint maps
- API reference guides
- Implementation roadmaps

---

## Key Constraint: llama.cpp API Limitation

**Problem**: llama.cpp (which bitnet.cpp wraps) does NOT expose intermediate layer activations. Only final logits are accessible via FFI.

**This means**: We cannot directly compare layer-by-layer activations between Rust and C++.

**Workaround Strategy**: Multi-tier approach with increasing complexity

---

## Implementation Strategy: Tier 1 (Immediate, No C++ Changes)

### Phase 1: Rust-Side Tracing Infrastructure

**Goal**: Instrument BitNet-rs to capture and hash all intermediate activations

**Tasks**:
1. Add `trace` feature flag to workspace
2. Create `bitnet-trace` utility crate with blake3 hashing
3. Add tracepoints to first transformer block (11 points)
4. Implement trace file format (name/shape/dtype/blake3/rms)

**Deliverable**: Rust inference produces trace files with hashes of all activations

**Timeline**: 1-2 days

---

### Phase 2: Per-Position Logits Comparison

**Goal**: Detect which token position first diverges

**Tasks**:
1. Extend crossval to call `llama_get_logits_ith()` for each position
2. Compare Rust vs C++ logits per-token with cosine similarity
3. Generate divergence report: "First divergence at token T, layer inferred from context"

**Deliverable**: Know exact token/position where outputs diverge

**Why useful**: If token 0 matches but token 1 diverges → autoregressive state issue (KV cache, position encoding)

**Timeline**: 1 day

---

### Phase 3: Tokenizer Parity Gate

**Goal**: Ensure identical input tokens (hard requirement)

**Tasks**:
1. Create `scripts/check_tokenizer_parity.sh`
2. Tokenize same prompt with Rust and C++ tokenizers
3. Diff token IDs - must match exactly
4. Add to CI as pre-cross-validation gate

**Deliverable**: Automated tokenizer validation

**Timeline**: 0.5 days

---

### Phase 4: Deterministic Cross-Validation Sweep

**Goal**: Run both engines with identical settings and compare

**Environment**:
```bash
export BITNET_DETERMINISTIC=1
export BITNET_SEED=42
export RAYON_NUM_THREADS=4
export BITNET_TRACE_DIR=out/rs-trace
```

**Test cases**:
1. Single token generation (prefill only)
2. 2-token generation (prefill + 1 decode)
3. 4-token generation (multiple decode steps)

**Comparison points**:
- Tokenizer output (must match)
- Per-position logits (detect divergence point)
- Final output tokens (end-to-end validation)

**Deliverable**: Divergence report identifying:
- Token position of first divergence
- Magnitude of divergence (cosine sim, L2 norm)
- Layer hypothesis (based on trace data)

**Timeline**: 1 day

---

## Implementation Strategy: Tier 2 (If Tier 1 Insufficient)

### Phase 5: C++ Debug Tool (Separate Process)

**Goal**: If per-position logits aren't granular enough, extract layer activations from C++

**Approach**: Create standalone C++ utility that:
1. Loads same model
2. Runs same prompt
3. **Patches llama.cpp locally** to dump layer outputs
4. Writes traces to JSON/binary files
5. Rust tests load and compare

**Advantages**:
- No FFI changes needed
- No bitnet.cpp fork required
- Local llama.cpp patch (not upstreamed)

**Disadvantages**:
- Requires C++ compilation
- Manual activation (not CI-friendly)

**Timeline**: 2-3 days (if needed)

---

## Implementation Strategy: Tier 3 (Deep Debugging)

### Phase 6: Patch bitnet.cpp FFI

**Goal**: Expose layer activations through FFI

**Tasks**:
1. Fork bitnet.cpp locally
2. Add `bitnet_get_layer_output(ctx, layer_idx)` to C shim
3. Update bindgen for new API
4. Extend crossval to call layer-by-layer

**Advantages**:
- Most comprehensive comparison
- Can pinpoint exact layer/operation

**Disadvantages**:
- Invasive changes to C++ codebase
- Maintenance burden
- Requires bitnet.cpp coordination

**Timeline**: 5-7 days (if needed)

---

## Recommended Execution Order

**Start with Tier 1** (Phases 1-4) - **Total: ~3.5 days**

This will likely identify the issue because:
1. ✅ Tokenizer parity eliminates input differences
2. ✅ Per-position logits pinpoint divergence location
3. ✅ Rust traces show what OUR activations look like
4. ✅ Deterministic settings eliminate non-determinism

**If Tier 1 succeeds**: We know which token/position diverges → focus fixes there

**If Tier 1 insufficient**: Move to Tier 2 (C++ debug tool) for layer granularity

**Tier 3**: Only if Tier 1+2 fail (unlikely)

---

## Success Criteria

**Phase 1-4 Success**:
- ✅ Tokenizer produces identical token IDs
- ✅ Rust traces captured for all 11 points in block 0
- ✅ Per-position logits identify divergence at token T
- ✅ Divergence report generated with actionable data

**Phase 5-6 Success** (if needed):
- ✅ Layer-by-layer activation comparison
- ✅ First diverging layer identified
- ✅ First diverging operation identified (attn_norm, q_proj, etc.)

**Ultimate Success**:
- ✅ Fix applied to BitNet-rs
- ✅ Logits match bitnet.cpp within 1e-4 tolerance
- ✅ Output tokens identical for deterministic prompts

---

## Risk Assessment

| Phase | Risk | Mitigation |
|-------|------|------------|
| 1: Rust tracing | LOW | Pattern proven (KernelRecorder) |
| 2: Per-position logits | LOW | FFI already exposes `llama_get_logits_ith()` |
| 3: Tokenizer parity | MEDIUM | Issue #469 exists, but workaround proven |
| 4: Cross-validation | LOW | Existing parity harness, just extend |
| 5: C++ debug tool | MEDIUM | Requires C++ compilation setup |
| 6: FFI patching | HIGH | Invasive, maintenance burden |

**Overall Risk**: LOW (Tier 1), MEDIUM (Tier 2), HIGH (Tier 3)

---

## Implementation Artifacts

**Code**:
- `crates/bitnet-trace/` - New utility crate
- `crates/bitnet-models/src/transformer.rs` - Add tracepoints (feature-gated)
- `crossval/src/logits_compare.rs` - Per-position comparison
- `scripts/check_tokenizer_parity.sh` - Tokenizer validation
- `scripts/run_crossval_sweep.sh` - Deterministic test runner

**Documentation**:
- `docs/crossval/TRACING_GUIDE.md` - How to use tracing
- `docs/crossval/DIVERGENCE_ANALYSIS.md` - Interpreting results
- CI receipt schema extension (optional `trace` field)

**Tests**:
- `crossval/tests/per_position_logits.rs`
- `crossval/tests/tokenizer_parity.rs`
- `bitnet-models/tests/trace_capture.rs`

---

## Next Steps

1. **Immediate**: Implement Phase 1 (Rust tracing infrastructure)
2. **Day 1-2**: Add tracepoints to transformer.rs block 0
3. **Day 2**: Implement Phase 2 (per-position logits)
4. **Day 3**: Implement Phase 3 (tokenizer parity)
5. **Day 3-4**: Run Phase 4 (cross-validation sweep)
6. **Day 4**: Analyze results and create fix

**Expected outcome**: Phases 1-4 will identify the issue. Fix implemented by Day 5.

---

## Open Questions

1. **Q**: Do we need all 11 tracepoints or just key ones (attn_norm, q_proj, logits)?
   **A**: Start with 5 critical points, add more if needed

2. **Q**: Should traces be in-memory or written to disk?
   **A**: Disk (JSON) - easier debugging, CI artifacts

3. **Q**: Blake3 or simpler hash (xxhash)?
   **A**: Blake3 - cryptographic quality, already in Rust ecosystem

4. **Q**: Feature flag name?
   **A**: `trace` - simple, clear, follows conventions

5. **Q**: Should tracing be always-on in debug builds?
   **A**: No - feature-gated only, zero overhead in production

---

## Conclusion

**Tier 1 (Phases 1-4) is immediately actionable** and requires no C++ changes. This will likely identify the divergence point with high confidence. If successful, we'll know:

- ✅ Exact token position of divergence
- ✅ Rust activation patterns at that point
- ✅ Magnitude and direction of error

This is sufficient to target fixes (LayerNorm, attention, quantization, etc.).

**Start implementation**: Phase 1 (Rust tracing) using agents for scoped tasks.
