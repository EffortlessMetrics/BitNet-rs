# NEXT STEPS: Fixing BitNet-rs Garbling

**Date Updated**: 2025-10-24 (Session 2)
**Status**: C++ tokenization blocked, duplicate BOS tokens found
**TL;DR**: Systematic investigation completed. **BLOCKER**: C++ tokenization wrapper fails, preventing crossval. Found duplicate BOS tokens in Rust.

## Session 2 Updates (2025-10-24)

### 🔍 New Finding: Duplicate BOS Tokens

**Evidence**:
```bash
Input tokens (15): [128000, 128000, 128006, 882, 128007, 271, 17, 10, 17, 28]
                    ^^^^^^^  ^^^^^^^
                    BOS #1   BOS #2 (DUPLICATE!)
```

- Token 128000 (`<|begin_of_text|>`) appears **twice** at sequence start
- Occurs with llama3-chat template auto-detection
- Unknown if C++ has same pattern (blocked by tokenization wrapper)

### ❌ Fixes Attempted (All Produced Identical Gibberish)

1. **RoPE Layout** (transformer.rs:142-182) - Changed interleaved → split layout - **REVERTED**
2. **GQA KV Head Slicing** (weight_mapper.rs:649) - Tested sequential and sparse - **REVERTED**

Both fixes tested, neither resolved gibberish. All changes reverted to maintain clean baseline.

### 🚫 BLOCKER: C++ Tokenization Wrapper Fails

```
llm_load_vocab: missing pre-tokenizer type, using: 'default'
error: LLAMA error: Tokenization failed
```

**Impact**: Cannot run `crossval-per-token` to find exact divergence point.

**Root Cause**: C++ wrapper not correctly using external tokenizer.json

---

## What We Found (Original Investigation)

### ✅ Confirmed

1. **Cross-validation infrastructure is ready** (95% complete)
   - Per-token logits comparison
   - 92 tracepoints instrumented
   - trace_diff.py for hash/RMS comparison

2. **LayerNorm corruption in microsoft model** (56× scale error)
   - `attn_norm.weight`: RMS = 0.018 (should be ~1.0)
   - `ffn_norm.weight`: RMS = 1.2-1.5 (normal)
   - Pattern repeats across all 30 layers

3. **All tested models garble**
   - microsoft i2_s (1.2G, QK256): `'E-lived,SIGNALConvert`
   - clean-f16-fixed (6.2G, FP16): `independ independ developed...`

### ❓ Unknown

**Does bitnet.cpp produce coherent output?**

This is the **critical decision point**:
- **If C++ also garbles** → Model quality issue (expected behavior per CLAUDE.md)
- **If C++ outputs "4"** → Inference divergence (needs systematic fix)

## Your Next Command

```bash
# 1. Set up C++ reference (one-time, ~30-60 min)
git clone https://github.com/microsoft/BitNet.git ~/.cache/bitnet_cpp
cd ~/.cache/bitnet_cpp
mkdir build && cd build
cmake .. && make -j$(nproc)

# 2. Run parity check with C++ comparison
cd ~/code/Rust/BitNet-rs
export BITNET_CPP_DIR="$HOME/.cache/bitnet_cpp"
./scripts/parity_smoke.sh models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf
```

### Interpreting Results

**Scenario A: C++ also produces garbage**
```bash
# Example output:
✅ Parity validation PASSED
C++ output: "'E-lived,SIGNALConvert"  ← Same garbage as Rust
Rust output: "'E-lived,SIGNALConvert"
cosine_similarity: 0.9999
```

**Action**: Document as expected model behavior, no fix needed.

**Scenario B: C++ produces "4"**
```bash
# Example output:
✗ Parity validation FAILED
C++ output: "4"                         ← Coherent!
Rust output: "'E-lived,SIGNALConvert"  ← Garbage
cosine_similarity: 0.23
first_divergence_step: 0
```

**Action**: Follow the systematic workflow in `docs/reports/SYSTEMATIC_DEBUGGING_PLAN.md`:
1. Find first divergence token
2. Capture traces at that point
3. Diff traces to identify exact operation
4. Apply targeted fix

## If C++ Shows Divergence (Scenario B)

### Quick Triage Commands

```bash
# Step 1: Find first token divergence
cargo run -p xtask --features inference -- crossval-per-token \
  --model models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf \
  --tokenizer models/microsoft-bitnet-b1.58-2B-4T-gguf/tokenizer.json \
  --prompt "What is 2+2?" --max-tokens 4

# Step 2: Capture Rust traces at divergence point
BITNET_TRACE_DIR=/tmp/rs RUST_LOG=warn BITNET_DETERMINISTIC=1 BITNET_SEED=42 \
  cargo run -p bitnet-cli --features cpu,trace -- run \
  --model models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf \
  --tokenizer models/microsoft-bitnet-b1.58-2B-4T-gguf/tokenizer.json \
  --prompt "What is 2+2?" --max-tokens 1 --greedy --seed 42

# Step 3: Capture C++ traces (if instrumented)
# [Manual: Run equivalent C++ command to generate traces in /tmp/cpp]

# Step 4: Diff traces to find exact divergence
python3 scripts/trace_diff.py /tmp/rs /tmp/cpp
# Output: "✗ First divergence at seq=0, layer=6, stage=attn_scores_softmax"
```

### Common Divergence Points & Fixes

Based on `(layer, stage)` from trace diff:

| Divergence Point | Likely Cause | Check | Fix |
|-----------------|-------------|-------|-----|
| `embeddings` | LM head tie | Verify `W_out ≈ Eᵀ` | Embedding orientation |
| `attn_scores_softmax` | RoPE/scaling | Verify θ, `1/√d_k` | RoPE params |
| `q_proj` | Weight layout | Verify transpose | Projection orientation |
| `ffn_out` | Activation | Verify GELU/SiLU | Activation function |

## Documentation Created

1. **`docs/reports/GARBLING_ROOT_CAUSE_ANALYSIS.md`**
   - Detailed LayerNorm corruption analysis
   - Trace evidence (RMS progression)
   - Solution options

2. **`docs/reports/SYSTEMATIC_DEBUGGING_PLAN.md`**
   - Complete triage workflow
   - Tool usage guide
   - Expected outcomes

3. **`NEXT_STEPS.md`** (this file)
   - Executive summary
   - Next commands to run
   - Decision tree

## Key Artifacts

- **Trace files**: `/tmp/bitnet_traces/rust/t0_*.trace` (1317 lines)
- **LayerNorm inspection**: Run `cargo run -p bitnet-cli -- inspect --ln-stats ...`
- **Parity receipts**: `docs/baselines/2025-10-24/parity-bitnetcpp.json`

## Tools You Have

All these are **ready to use** right now:

```bash
# Per-token divergence detection
cargo run -p xtask --features inference -- crossval-per-token <args>

# LayerNorm validation
cargo run -p bitnet-cli --features cpu -- inspect --ln-stats <model.gguf>

# Trace capture
BITNET_TRACE_DIR=/tmp/traces cargo run -p bitnet-cli --features cpu,trace -- run <args>

# Trace comparison
python3 scripts/trace_diff.py /tmp/rs /tmp/cpp

# Quick parity check
./scripts/parity_smoke.sh <model.gguf>

# Multi-scenario sweep
./scripts/run_crossval_sweep.sh <model.gguf> <tokenizer.json> /tmp/output
```

## Timeline Estimate

| Task | Time | Priority |
|------|------|----------|
| Set up C++ reference | 30-60 min | HIGH |
| Run parity check | 5 min | HIGH |
| Interpret results | 5 min | HIGH |
| **If divergence found** |||
| Find first token divergence | 10 min | MEDIUM |
| Capture Rust traces | 15 min | MEDIUM |
| Capture C++ traces | 15 min | MEDIUM |
| Diff traces | 5 min | MEDIUM |
| Apply targeted fix | 30-120 min | MEDIUM |

**Total (best case)**: 40 min if model quality issue

**Total (worst case)**: 3-4 hours if inference divergence

## Questions?

- **"Can I skip C++ setup?"** No - it's required to determine if this is a bug or expected behavior.
- **"What if C++ build fails?"** Check BitNet.cpp README for dependencies (likely need CMake 3.15+, GCC/Clang).
- **"Can I use a different model?"** Try microsoft/bitnet-b1.58-1B if 2B is too large, but same pattern likely.

## Success Criteria

### Scenario A (Model Quality)
- [ ] C++ reference produces same garbage as Rust
- [ ] Documented in model baselines
- [ ] Warning added to README
- [ ] Issue closed as "expected behavior"

### Scenario B (Inference Bug)
- [ ] C++ produces coherent text
- [ ] First divergence point identified
- [ ] Targeted fix applied
- [ ] Rust now matches C++ output
- [ ] Regression test added
- [ ] Receipt validation passes

---

**Start here**: Run the C++ setup commands above, then report back what the parity check shows.
