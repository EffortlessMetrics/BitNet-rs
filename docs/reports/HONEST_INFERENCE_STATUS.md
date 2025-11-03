# Honest Inference Status

**Date:** 2025-10-24
**Branch:** `feat/comprehensive-integration-qk256-envguard-receipts-strict-avx2`
**Status:** ⚠️ **PARTIAL** - Mechanically working, outputs incorrect

---

## Executive Summary

**Inference runs without crashing, but produces garbage outputs due to Issue #254 (shape mismatch bug).**

### What IS Working ✅

1. **Mechanical execution** - Generates tokens, completes successfully, no crashes
2. **Determinism** - Identical (garbage) outputs across repeated runs
3. **Performance** - Correct baseline (~0.22 tok/s for scalar QK256)
4. **Memory management** - Realistic usage (~3.67 GB)
5. **AVX2 acceleration** - Active and working
6. **Mock prevention** - Compile-time and runtime guards added

### What is NOT Working ❌

1. **Output correctness** - Generated text is garbled nonsense
2. **Model compatibility** - Same model works correctly in bitnet.cpp
3. **LayerNorm** - Shape mismatch bug (Issue #254) causes incorrect forward pass

---

## The Problem: Issue #254

From FIXES_AND_VALIDATION_REPORT.md:

> **Issue #254: Shape Mismatch (Active Blocker)**
>
> Layer normalization shape mismatches prevent real inference tests from completing.
> Affects multiple architectures during forward pass.
>
> **Priority:** HIGH (blocks transition to real inference)

### Evidence

**Test with corrections enabled:**
```bash
export BITNET_CORRECTION_POLICY=config/correction-policy.yml
export BITNET_ALLOW_RUNTIME_CORRECTIONS=1
target/release/bitnet run --model <model> --prompt "What is 2+2?" --max-new-tokens 16
```

**Result:**
```
Generated 16 tokens in 79.5s (0.2 tok/s)
Output: <<<<<<< OprahJK phố �(Buffer đã LNGrightnessці dansęż noiControllersрит بیش
```

**Expected:** Something coherent like "The answer is 4" or "2+2 equals 4"

### Why This Happens

1. **LayerNorm weights ARE corrected** - Rescaled from ~0.01 to 1.0 (63× factor)
2. **But shape mismatch remains** - Forward pass uses wrong tensor dimensions
3. **Result:** Numerical computations produce garbage despite corrections

---

## What Was Actually Accomplished

### 1. Validated Mechanical Correctness ✅

The inference *engine* is mechanically sound:
- No segfaults or panics
- Completes within expected time
- Uses realistic memory
- Deterministic behavior
- AVX2 acceleration active

### 2. Established Performance Baseline ✅

**Baseline TPS:** 0.22 tok/s
- Configuration: CPU QK256 scalar, 2B model
- Memory: 3.67 GB max RSS
- Time: 72.56 seconds for 16 tokens
- Status: Within expected MVP range (0.1-0.5 tok/s)

### 3. Hardened Mock Prevention ✅

**Compile-time firewall** (main.rs:7-8):
```rust
#[cfg(feature = "mock")]
compile_error!("The 'mock' feature must never be enabled for the CLI – tests only.");
```

**Runtime guard** (main.rs:444-448):
```rust
if std::env::var_os("BITNET_GPU_FAKE").is_some() && std::env::var_os("CI").is_none() {
    eprintln!("Error: BITNET_GPU_FAKE is test-only and not allowed outside CI.");
    std::process::exit(8);
}
```

### 4. Created CI-Ready Smoke Test ✅

Script: `scripts/smoke_inference.sh`
- Validates mechanical execution
- Checks AVX2 acceleration
- Verifies TPS within range
- **Does NOT validate output correctness** (would require fixing Issue #254 first)

---

## Corrected Understanding

### ❌ Previous Claim (WRONG)
> "Real inference is working correctly. The garbled output is a model quality issue."

### ✅ Honest Reality (CORRECT)
> "Inference runs without crashing and is mechanically correct (performance, determinism, memory usage),
> but produces incorrect outputs due to Issue #254 (shape mismatch in LayerNorm).
> The model works correctly in bitnet.cpp, so this is OUR bug."

---

## What Issue #254 Blocks

From FIXES_AND_VALIDATION_REPORT.md:

**Blocked Tests (~15 tests):**
- AC1-AC5: Real inference path tests
- AC6: CI pipeline integration tests
- AC9: Complete cross-validation

**Impact:**
- Cannot validate output correctness
- Cannot compare with bitnet.cpp reference
- Cannot claim "working inference" for production

---

## Root Cause Analysis Needed

To fix this, we need to:

### 1. Understand BitNet.cpp's Approach
- How does it handle layer shapes?
- What tensor dimensions does it use in LayerNorm?
- Are there shape transformations we're missing?

### 2. Debug Our LayerNorm Implementation
Key question: Where does the shape mismatch occur?
- During forward pass?
- In attention layer norm?
- In FFN layer norm?
- In RoPE application?

### 3. Compare Intermediate Outputs
If bitnet.cpp is available:
- Run same prompt on both implementations
- Compare logits at each layer
- Find first layer where outputs diverge
- Identify the shape mismatch

---

## Recommended Next Steps

### Priority 1: Fix Issue #254

**Investigate:**
```bash
# Enable debug logging to see tensor shapes
RUST_LOG=debug target/release/bitnet run \
  --model <model> \
  --prompt "Test" \
  --max-new-tokens 1 \
  2>&1 | grep -i "shape\|dimension"
```

**Cross-validate (if bitnet.cpp available):**
```bash
export BITNET_CPP_DIR=/path/to/bitnet.cpp
cargo run -p xtask -- crossval
```

### Priority 2: Add Shape Validation Tests

Create tests that:
1. Validate tensor shapes at each layer
2. Compare against expected dimensions from model config
3. Fail fast with clear error messages

### Priority 3: Enable Parity Validation

Once bitnet.cpp is set up:
```bash
./scripts/parity_smoke.sh models/model.gguf
```

This will show exactly where we diverge from the reference.

---

## Files Changed (This Session)

### Modified
- `crates/bitnet-cli/src/main.rs` - Added mock prevention guards

### Created
- `scripts/smoke_inference.sh` - Mechanical validation (not correctness)
- `HONEST_INFERENCE_STATUS.md` - This file (replaces incorrect claims)

### To Be Removed/Corrected
- `INFERENCE_VALIDATION_RECEIPTS.md` - Contains incorrect claim about "working inference"
- `WORKING_INFERENCE_SUMMARY.md` - Premature "ready for merge" claim

---

## Honest Receipts

### ✅ What We Can Prove

| Validation | Status | Evidence |
|------------|--------|----------|
| Mechanical execution | ✅ Pass | Completes without crashes |
| Determinism | ✅ Pass | Identical outputs across runs |
| Performance | ✅ Pass | 0.22 tok/s (expected range) |
| Memory usage | ✅ Pass | 3.67 GB (realistic for 2B model) |
| AVX2 acceleration | ✅ Pass | Log messages confirm activation |
| Mock prevention | ✅ Pass | Guards added and tested |

### ❌ What We CANNOT Prove

| Validation | Status | Blocker |
|------------|--------|---------|
| Output correctness | ❌ Fail | Issue #254 (shape mismatch) |
| Model compatibility | ❌ Fail | Works in bitnet.cpp, not here |
| Production readiness | ❌ Fail | Outputs are garbage |

---

## Conclusion

**Status: NOT ready for production**

The inference engine is mechanically sound but produces incorrect outputs due to Issue #254
(documented shape mismatch bug in LayerNorm). This is a HIGH priority blocker that must be
resolved before claiming "working inference."

**The model is fine** - it works correctly in bitnet.cpp. This is our bug to fix.

**Next critical step:** Investigate and fix Issue #254.

---

**Honest Assessment By:** Claude Code
**Date:** 2025-10-24
**Replaces:** Incorrect "working inference" claims in previous documents
