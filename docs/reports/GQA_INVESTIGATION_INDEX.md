# GQA Investigation - Complete Documentation Index

## Quick Links

1. **[GQA_INVESTIGATION_SUMMARY.md](GQA_INVESTIGATION_SUMMARY.md)** - Executive summary (start here)
2. **[GQA_KV_SLICING_ANALYSIS.md](GQA_KV_SLICING_ANALYSIS.md)** - Comprehensive technical analysis
3. **[GQA_VISUAL_GUIDE.txt](GQA_VISUAL_GUIDE.txt)** - ASCII diagrams and visual explanations
4. **[GQA_CODE_COMPARISON.md](GQA_CODE_COMPARISON.md)** - Code examples and fixes

---

## Document Overview

### 1. GQA_INVESTIGATION_SUMMARY.md

**Read this first** - 5 minute executive summary

- Quick problem statement (30 seconds)
- Root cause chain
- Current vs correct behavior
- Recommended fixes
- Action items

**Key Finding**: Sparse row selection in weight_mapper.rs causes 75% parameter loss

---

### 2. GQA_KV_SLICING_ANALYSIS.md

**Deep dive** - Comprehensive technical analysis

Contains:
- Detailed code review (weight_mapper.rs:607-670)
- GQA expansion logic analysis (transformer.rs:412-426)
- Head dimension calculations verification
- Shape transformation through entire pipeline
- Comparison with llama.cpp reference
- Root cause analysis
- Proposed fixes (Option 1 & 2)
- Testing recommendations
- Summary table

**Use this when**: You need complete understanding of the issue

---

### 3. GQA_VISUAL_GUIDE.txt

**Visual explanations** - ASCII diagrams

Shows:
- Current (WRONG) implementation with detailed trace
- Correct implementation with detailed trace
- GQA expansion results
- Attention computation impact
- Comparison table
- Diagnostic flags

**Use this when**: You want visual understanding of the problem

**Key Diagram**:
```
Current: Selects rows [0, 512, 1024, 1536, 2048] (heads 0,4,8,12,16) - SPARSE
Correct: Selects rows [0-127, 128-255, 256-383, 384-511, 512-639] (heads 0,1,2,3,4) - SEQUENTIAL
```

---

### 4. GQA_CODE_COMPARISON.md

**Implementation details** - Code examples and fixes

Contains:
- Current implementation with traced execution
- Correct implementation (Option 1)
- Alternative rejection approach (Option 2)
- Unit test validation code
- Comparison table
- Risk assessment

**Use this when**: You're ready to implement the fix

---

## The Problem in 30 Seconds

**File**: `crates/bitnet-models/src/weight_mapper.rs:649`

**Current code**:
```rust
let head_idx = kv_idx * group_size;  // WRONG: sparse selection
```

**Effect**:
- Selects heads [0, 4, 8, 12, 16] instead of [0, 1, 2, 3, 4]
- Loses 75% of K/V parameters
- Causes degenerate attention (gibberish output)

**Fix**:
```rust
// Delete the line above and use sequential selection
let head_idx = kv_idx;  // Correct: sequential
```

---

## Investigation Results

### What's Working

- ✓ GQA expansion logic (transformer.rs)
- ✓ Head dimension calculations
- ✓ Shape transformations through pipeline
- ✓ Attention computation structure
- ✓ RoPE application
- ✓ KV cache handling

### What's Broken

- ✗ K/V head slicing (weight_mapper.rs)
  - Uses sparse selection instead of sequential
  - Causes 75% parameter loss
  - Produces degenerate attention
  - Leads to gibberish output

---

## Recommended Path Forward

1. **Read**: GQA_INVESTIGATION_SUMMARY.md (5 min)
2. **Understand**: GQA_VISUAL_GUIDE.txt (5 min)
3. **Review**: Relevant code sections in weight_mapper.rs
4. **Implement**: Option 1 (sequential slicing) from GQA_CODE_COMPARISON.md
5. **Test**: Cross-validate with C++ reference
6. **Validate**: Run diagnostic flags

---

## Code Locations

- **Weight Mapper Slicing**: `crates/bitnet-models/src/weight_mapper.rs:647-655`
- **GQA Expansion**: `crates/bitnet-models/src/transformer.rs:412-426`
- **Regression Test**: `crates/bitnet-models/src/weight_mapper.rs:1048-1153`

---

## Diagnostic Commands

### Enable Debug Logging

```bash
# Log GQA dimensions and norms
BITNET_DEBUG_GQA=1 cargo run -p bitnet-cli -- run \
  --model models/model.gguf \
  --prompt "test" \
  --max-tokens 4

# Log attention scale factors
BITNET_DEBUG_ATTN_SCALE=1 cargo run ...

# Log RoPE application
BITNET_DEBUG_ROPE=1 cargo run ...

# Combine all
BITNET_DEBUG_GQA=1 BITNET_DEBUG_ATTN_SCALE=1 BITNET_DEBUG_ROPE=1 cargo run ...
```

### Cross-Validate with C++

```bash
# Set up C++ reference (if available)
export BITNET_CPP_DIR=/path/to/bitnet.cpp

# Run per-token logits comparison
cargo run -p xtask --features crossval-all -- crossval-per-token \
  --model models/model.gguf \
  --tokenizer models/tokenizer.json \
  --prompt "test" \
  --max-tokens 4 \
  --cos-tol 0.99
```

---

## Files in This Analysis

| File | Size | Purpose | Read Time |
|------|------|---------|-----------|
| GQA_INVESTIGATION_SUMMARY.md | 8KB | Executive summary | 5 min |
| GQA_KV_SLICING_ANALYSIS.md | 13KB | Detailed analysis | 15 min |
| GQA_VISUAL_GUIDE.txt | 12KB | ASCII diagrams | 10 min |
| GQA_CODE_COMPARISON.md | 12KB | Code examples | 15 min |
| **Total** | **45KB** | Complete investigation | **45 min** |

---

## Confidence Assessment

**Confidence Level**: HIGH (85%+)

The sparse slicing logic is demonstrably incorrect:
1. Sparse row indices confirmed: [0, 512, 1024, 1536, 2048]
2. These correspond to heads [0, 4, 8, 12, 16] (not [0, 1, 2, 3, 4])
3. 75% parameter loss is mathematically certain
4. Degenerate attention pattern explanation matches observed gibberish output

**Evidence**:
- Static code analysis of mapper logic
- Traced execution with concrete numbers
- Mathematical proof of parameter loss
- Clear attention corruption mechanism

---

## Next Steps

1. [ ] Review GQA_INVESTIGATION_SUMMARY.md
2. [ ] Confirm root cause with team
3. [ ] Implement Option 1 fix (sequential slicing)
4. [ ] Add unit tests
5. [ ] Cross-validate with C++ reference
6. [ ] Test with diverse models
7. [ ] Merge and release

---

## Questions?

Refer to the appropriate document:
- **"What's the problem?"** → GQA_INVESTIGATION_SUMMARY.md (Quick Answer section)
- **"How does it fail?"** → GQA_VISUAL_GUIDE.txt (Current Implementation section)
- **"How to fix it?"** → GQA_CODE_COMPARISON.md (Correct Implementation section)
- **"Why is this the issue?"** → GQA_KV_SLICING_ANALYSIS.md (Root Cause Analysis)

---

Generated: 2025-10-24
Investigation Depth: Complete
Status: Ready for Implementation

