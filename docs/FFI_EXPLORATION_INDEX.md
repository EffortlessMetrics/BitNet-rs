# C++ FFI Integration Exploration: Complete Documentation Index

This directory contains comprehensive documentation on the C++ FFI integration for cross-validation testing in BitNet.rs.

## Quick Navigation

### For Sprint 1.3 Developers
Start here: **[C_FFI_QUICK_REFERENCE.md](./C_FFI_QUICK_REFERENCE.md)** (5 minutes)
- Common code patterns
- Essential APIs
- Build commands
- Test execution
- Debugging tips

### For Detailed Understanding
Read: **[C_FFI_INTEGRATION_ANALYSIS.md](./C_FFI_INTEGRATION_ANALYSIS.md)** (30 minutes)
- Complete architecture overview
- Current capabilities (what works)
- Optional optimizations (what could be improved)
- Implementation roadmaps (how to proceed)
- Fallback options (backup plans)

### For API Details
Reference: **[../crossval/README_PER_POSITION_LOGITS.md](../crossval/README_PER_POSITION_LOGITS.md)** (detailed)
- Per-position logits comparison module
- API function signatures
- Code examples
- Metrics explanation
- Test patterns

## Document Overview

### C_FFI_QUICK_REFERENCE.md
- **Length**: ~250 lines
- **Audience**: Developers working on Sprint 1.3
- **Content**:
  - Current capabilities summary
  - Key file locations
  - Essential APIs
  - 4 common code patterns
  - Build & test commands
  - Performance notes
  - Debugging checklist
  - Known issues & workarounds
  - Week-by-week roadmap

### C_FFI_INTEGRATION_ANALYSIS.md
- **Length**: ~650 lines
- **Audience**: Project leads and architects
- **Content**:
  - Executive summary
  - 4-tier architecture breakdown
  - Current FFI capabilities table
  - What already works (infrastructure present)
  - What would improve performance (optional)
  - Concrete implementation roadmap
  - Code examples for common scenarios
  - Existing test coverage (152+ tests)
  - Blockers & known issues
  - Conclusion & recommendations

### FFI_EXPLORATION_INDEX.md (this file)
- **Length**: ~200 lines
- **Audience**: All developers
- **Content**:
  - Navigation guide
  - Document overview
  - Key findings summary
  - File structure
  - How to use this documentation

## Key Findings Summary

### What's Ready Now ✅
1. **Per-Position Comparison Module** (`crossval/src/logits_compare.rs`)
   - Cosine similarity calculation
   - L2 Euclidean distance
   - Max absolute difference
   - First divergence detection

2. **C++ FFI Layer** (`crates/bitnet-sys/`)
   - Safe Rust wrappers for C++ API
   - Context with `logits_all=true` enabled
   - `get_logits_ith()` for per-position extraction
   - `get_all_logits()` for bulk extraction (ALREADY IMPLEMENTED)

3. **Session Management** (`crates/bitnet-inference/src/ffi_session.rs`)
   - Global reusable FFI session
   - Prevents memory corruption
   - Thread-safe via Mutex
   - Deterministic inference

4. **Test Infrastructure** (`crossval/tests/`)
   - 152+ passing parity tests
   - Full per_position_logits.rs test suite
   - Multi-token generation tracking
   - Prefill vs decode comparison

### What's Optional 🔧
1. **C++ Batch Helper Function** (`bitnet_get_all_logits()`)
   - Reduces N FFI calls to 1
   - 10-30% performance improvement
   - 30 lines of C++ code
   - Not blocking for Sprint 1.3

### What's Needed for Sprint 1.3 ⏱️
1. **Week 1 (Ready Now)**:
   - Use existing parity test infrastructure
   - Validate single-token logits match
   - No new code required

2. **Week 2 (Choose Path)**:
   - Path A: Add C++ batch function (30 lines, recommended)
   - Path B: Pure Rust per-position extraction (safer alternative)

3. **Week 3 (Testing)**:
   - Run full per_position_logits.rs test suite
   - Verify cosine similarity metrics
   - Document divergence findings

## File Structure

```
BitNet-rs/
├── docs/
│   ├── FFI_EXPLORATION_INDEX.md (this file)
│   ├── C_FFI_INTEGRATION_ANALYSIS.md (comprehensive analysis)
│   ├── C_FFI_QUICK_REFERENCE.md (quick start guide)
│   └── ... (other documentation)
│
├── crates/
│   ├── bitnet-sys/
│   │   ├── csrc/
│   │   │   └── bitnet_c_shim.cc (C++ wrapper)
│   │   ├── include/
│   │   │   └── bitnet_c.h (C API header)
│   │   └── src/
│   │       ├── wrapper.rs (Rust FFI wrappers) ⭐ KEY FILE
│   │       └── lib.rs (module exports)
│   │
│   └── bitnet-inference/
│       └── src/
│           └── ffi_session.rs (session management) ⭐ KEY FILE
│
├── crossval/
│   ├── src/
│   │   ├── logits_compare.rs (comparison module) ⭐ KEY FILE
│   │   └── lib.rs
│   ├── tests/
│   │   ├── per_position_logits.rs (integration tests) ⭐ KEY FILE
│   │   └── parity.rs (parity validation tests)
│   └── README_PER_POSITION_LOGITS.md (API docs) ⭐ KEY FILE
```

⭐ = Critical files for Sprint 1.3

## How to Use This Documentation

### Scenario 1: "I need to run cross-validation tests right now"
1. Read: C_FFI_QUICK_REFERENCE.md (5 min)
2. Copy: Code patterns from "Common Patterns" section
3. Run: Commands from "Test Execution" section
4. Done!

### Scenario 2: "I need to understand the full architecture"
1. Read: FFI_EXPLORATION_INDEX.md (5 min - overview)
2. Read: C_FFI_INTEGRATION_ANALYSIS.md (30 min - details)
3. Skim: Key files in codebase (30 min - implementation)
4. Reference: C_FFI_QUICK_REFERENCE.md (for API lookup)

### Scenario 3: "I need to add per-position logits extraction"
1. Read: C_FFI_QUICK_REFERENCE.md Pattern #2 (1 min)
2. Check: C_FFI_INTEGRATION_ANALYSIS.md "Immediate Implementation Path" (5 min)
3. Decide: Path A (C++) vs Path B (Rust)
4. Implement: ~30-60 lines of code
5. Test: Run per_position_logits.rs tests

### Scenario 4: "Something's broken with C++ FFI"
1. Read: C_FFI_QUICK_REFERENCE.md "Debugging" section (5 min)
2. Check: "Known Issues & Workarounds" section
3. Reference: C_FFI_INTEGRATION_ANALYSIS.md "Blockers & Known Issues" (5 min)
4. Try: Fallback option or workaround

## Key Capabilities at a Glance

| Capability | Status | Location | Effort |
|-----------|--------|----------|--------|
| Last-token parity testing | ✅ Ready | parity.rs | 0 |
| Per-position logits extraction | ✅ Ready | wrapper.rs:285-293 | 0 |
| Logits comparison (cosine sim, L2) | ✅ Ready | logits_compare.rs | 0 |
| Memory-safe FFI sessions | ✅ Ready | ffi_session.rs | 0 |
| Deterministic inference | ✅ Ready | bitnet_c_shim.cc:87 | 0 |
| Per-position tests | ✅ Ready | per_position_logits.rs | 0 |
| C++ batch helper (optional) | ❌ Missing | - | 1-2h |
| Pure Rust per-position (fallback) | ⚠️ Designed | - | 2-3h |

## Sprint 1.3 Timeline

### Week 1: Validation Phase
**Status**: Ready to start immediately
```
Day 1-2: Understand infrastructure (read C_FFI_QUICK_REFERENCE.md)
Day 3-4: Run existing parity tests with --features crossval
Day 5: Verify cosine similarity > 0.9999, document results
```

### Week 2: Implementation Phase
**Status**: Two paths ready to go
```
PATH A (Recommended - C++ optimization):
  - Add bitnet_get_all_logits() to csrc/ (30 lines, ~4h)
  - Update Rust bindings (automatic via build.rs)
  - Add wrapper in wrapper.rs (20 lines, ~2h)
  - Test (2h)

PATH B (Safer alternative - Pure Rust):
  - Extend parity.rs with per-position function (~2h)
  - Implement incremental evaluation (~3h)
  - Test & document (~2h)
```

### Week 3: Testing & Documentation
**Status**: Full test suite ready
```
Day 1-3: Run per_position_logits.rs full test suite
Day 4: Verify all metrics and document findings
Day 5: Update CLAUDE.md with per-position guidance
```

## Command Reference

```bash
# Initialize C++ backend
export BITNET_CPP_DIR=/path/to/bitnet.cpp

# Run parity tests
cargo test --features crossval --test parity -- --nocapture

# Run per-position tests
export CROSSVAL_GGUF=/path/to/model.gguf
cargo test --features crossval --test per_position_logits -- --nocapture

# Check C++ availability
cargo test --features crossval cpp_availability

# Verify determinism
export RAYON_NUM_THREADS=1
export BITNET_DETERMINISTIC=1
cargo test --features crossval
```

## Common Questions Answered

**Q: Do I need C++ to run cross-validation?**
A: No. C++ backend is optional. Tests skip gracefully if not available.

**Q: Can I extract per-position logits without C++ changes?**
A: Yes. Use existing `Context::get_all_logits()` function (wrapper.rs:285-293).

**Q: How many logits do I get per position?**
A: `vocab_size` logits per position. For 32K vocab, that's 32K f32 values.

**Q: What's the memory overhead for per-position extraction?**
A: `num_positions * vocab_size * 4 bytes`. For 10-token sequence with 32K vocab: ~1.3MB.

**Q: How long does per-position testing take?**
A: Typically 5-10 seconds per sequence. No FFI optimization needed for that.

**Q: What if tokenization differs between Rust and C++?**
A: Tests skip with a warning. This is Issue #469, not a blocking issue.

## References

- **Main analysis**: C_FFI_INTEGRATION_ANALYSIS.md
- **Quick start**: C_FFI_QUICK_REFERENCE.md
- **API docs**: ../crossval/README_PER_POSITION_LOGITS.md
- **C++ shim**: ../crates/bitnet-sys/csrc/bitnet_c_shim.cc
- **Rust wrappers**: ../crates/bitnet-sys/src/wrapper.rs
- **Session mgmt**: ../crates/bitnet-inference/src/ffi_session.rs
- **Comparison**: ../crossval/src/logits_compare.rs
- **Tests**: ../crossval/tests/per_position_logits.rs

## Next Steps

1. **For immediate use**: Open C_FFI_QUICK_REFERENCE.md
2. **For understanding**: Read C_FFI_INTEGRATION_ANALYSIS.md
3. **For implementation**: Check "Immediate Implementation Path" in analysis
4. **For debugging**: Consult "Debugging" section in quick reference

---

**Status**: All documentation complete. Infrastructure ready for Sprint 1.3.
**Created**: October 2024
**Last Updated**: October 24, 2024
