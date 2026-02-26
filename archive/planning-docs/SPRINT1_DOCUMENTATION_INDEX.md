# Sprint 1: Per-Position Logits Extraction - Documentation Index

## Overview

This directory contains comprehensive documentation for implementing all-positions logits extraction in BitNet-rs. Four documents are provided at different levels of detail for different audiences.

---

## Documents (Read in This Order)

### 1. EXPLORATION_SUMMARY.txt
**Size**: 9.9 KB  
**Audience**: Everyone - executive summary  
**Reading Time**: 10-15 minutes  
**Content**:
- Key discoveries and findings
- Current architecture overview
- Critical findings about latent infrastructure
- Exact file locations and line numbers
- Data flow transformations
- Risk assessment
- Q&A section

**When to Read**: First - gives you the big picture without technical details

---

### 2. SPRINT1_QUICK_REFERENCE.md
**Size**: 8.8 KB  
**Audience**: Implementers - quick start guide  
**Reading Time**: 15-20 minutes  
**Content**:
- Problem statement
- Solution overview (2 new functions needed)
- Key findings and flows
- What's already in place
- Implementation checklist
- Code templates (not final, for reference)
- Integration with crossval
- Testing strategy
- Success criteria

**When to Read**: Second - before starting implementation to understand scope

---

### 3. IMPLEMENTATION_GUIDE.md
**Size**: 14 KB  
**Audience**: Developers - step-by-step implementation  
**Reading Time**: 30-45 minutes  
**Content**:
- Exact file locations (absolute paths)
- Code sections to study first
- Complete code templates (copy-paste ready)
- Verification checklist
- Implementation steps in order
- Key differences from current code
- Common issues & solutions
- Performance notes

**When to Read**: Third - while actively implementing

---

### 4. LOGITS_EXTRACTION_ANALYSIS.md
**Size**: 20 KB  
**Audience**: Deep-dive technical analysis  
**Reading Time**: 60+ minutes  
**Content**:
- Part 1: Current logits extraction flow (step-by-step)
- Part 2: Data structures involved
- Part 3: The central problem
- Part 4: Exact line numbers
- Part 5: Data flow diagrams
- Part 6: What needs to change
- Part 7: Detailed implementation template
- Part 8: Integration with crossval
- Part 9: Dependencies & prerequisites
- Part 10: Validation checklist
- Part 11: Example test case
- Part 12: Performance considerations
- Part 13: Summary table

**When to Read**: Reference while implementing or for complete understanding

---

## Quick Navigation

### If you have 10 minutes:
Read: EXPLORATION_SUMMARY.txt

### If you have 30 minutes:
1. EXPLORATION_SUMMARY.txt
2. SPRINT1_QUICK_REFERENCE.md

### If you have 1 hour:
1. EXPLORATION_SUMMARY.txt
2. SPRINT1_QUICK_REFERENCE.md
3. IMPLEMENTATION_GUIDE.md (sections A-B)

### If you have 2+ hours:
1. All above in order
2. LOGITS_EXTRACTION_ANALYSIS.md (complete deep-dive)
3. Reference documents while implementing

### If you want to implement right now:
1. Skim SPRINT1_QUICK_REFERENCE.md (5 min)
2. Follow IMPLEMENTATION_GUIDE.md sections A-B-C (30 min)
3. Reference LOGITS_EXTRACTION_ANALYSIS.md if stuck (5-10 min)

---

## Key Facts (TL;DR)

**Problem**: Need function to return logits for ALL token positions (not just last)

**Solution**: Add 2 functions to `crates/bitnet-inference/src/parity.rs`:
- `eval_logits_all_positions()` (120 lines)
- `extract_all_position_logits()` (70 lines)

**Files to Modify**: 
- `/home/steven/code/Rust/BitNet-rs/crates/bitnet-inference/src/parity.rs` (add functions)
- `/home/steven/code/Rust/BitNet-rs/crates/bitnet-inference/src/lib.rs` (add 1 export)

**Files Already Ready**:
- `/home/steven/code/Rust/BitNet-rs/crates/bitnet-models/src/transformer.rs` (no changes)
- `/home/steven/code/Rust/BitNet-rs/crossval/src/logits_compare.rs` (no changes)

**Effort**: ~200 lines of code, 2-3 hours implementation

**Risk**: LOW - purely additive, no breaking changes

---

## Document Contents at a Glance

| Document | Key Tables/Sections | Best For |
|----------|---|---|
| EXPLORATION_SUMMARY.txt | Key Discoveries, Risk Assessment, Q&A | Understanding scope |
| SPRINT1_QUICK_REFERENCE.md | Implementation Checklist, Code Templates | Planning implementation |
| IMPLEMENTATION_GUIDE.md | Section A-B (templates), Verification Checklist | Writing code |
| LOGITS_EXTRACTION_ANALYSIS.md | Parts 1-7 (flow analysis), Part 13 (summary) | Deep understanding |

---

## Critical File Locations (Absolute Paths)

```
Primary Implementation:
/home/steven/code/Rust/BitNet-rs/crates/bitnet-inference/src/parity.rs

Supporting Files (no changes needed):
/home/steven/code/Rust/BitNet-rs/crates/bitnet-models/src/transformer.rs
/home/steven/code/Rust/BitNet-rs/crossval/src/logits_compare.rs

Exports:
/home/steven/code/Rust/BitNet-rs/crates/bitnet-inference/src/lib.rs

Documentation (this repo):
/home/steven/code/Rust/BitNet-rs/EXPLORATION_SUMMARY.txt
/home/steven/code/Rust/BitNet-rs/SPRINT1_QUICK_REFERENCE.md
/home/steven/code/Rust/BitNet-rs/IMPLEMENTATION_GUIDE.md
/home/steven/code/Rust/BitNet-rs/LOGITS_EXTRACTION_ANALYSIS.md
```

---

## Testing Commands

After implementation:

```bash
# Compile check
cargo build --no-default-features --features cpu

# Run unit tests
cargo test parity::tests --no-default-features --features cpu

# Run all inference tests
cargo test -p bitnet-inference --no-default-features --features cpu

# Test with C++ reference (if available)
cargo test -p bitnet-inference --features crossval
```

---

## Common Questions Answered

**Q: Do I need to understand the whole codebase?**  
A: No. Read SPRINT1_QUICK_REFERENCE.md and IMPLEMENTATION_GUIDE.md sections A-B only.

**Q: Will this break existing code?**  
A: No. These are purely additive functions. All existing code continues unchanged.

**Q: How long will implementation take?**  
A: 2-3 hours for experienced Rust developer (more with testing/debugging).

**Q: What if something breaks?**  
A: See "Common Issues & Solutions" in IMPLEMENTATION_GUIDE.md.

**Q: Where are the complete code templates?**  
A: In IMPLEMENTATION_GUIDE.md Section A (eval_logits_all_positions) and Section B (extract_all_position_logits).

**Q: What's the difference from eval_logits_once()?**  
A: See "Differences from eval_logits_once()" table in SPRINT1_QUICK_REFERENCE.md.

---

## Success Criteria

After implementation, you should be able to:

1. Call: `eval_logits_all_positions(model_path, tokens)`
2. Get: `Vec<Vec<f32>>` with shape `[seq_len][vocab_size]`
3. Use: `compare_per_position_logits()` for cross-validation
4. Measure: Cosine similarity at each position
5. Identify: First divergence point between Rust and C++

---

## Architecture Overview

```
Current (Single Token):
tokens → embed() → forward() → logits() → extract_last_token → Vec<f32>

New (All Positions):
tokens → embed() → forward_full() → logits() → extract_all_positions → Vec<Vec<f32>>
```

The key insight: both paths exist in the codebase. We just need to wire them together in `parity.rs`.

---

## Getting Help

If you get stuck:

1. **Build errors**: See "Common Issues & Solutions" in IMPLEMENTATION_GUIDE.md
2. **Logic questions**: Check LOGITS_EXTRACTION_ANALYSIS.md Part 1-3
3. **Template reference**: Use IMPLEMENTATION_GUIDE.md Section A-B
4. **Understanding flow**: See data flow diagram in SPRINT1_QUICK_REFERENCE.md

---

## Next Steps

1. Start with EXPLORATION_SUMMARY.txt (10 min)
2. Read SPRINT1_QUICK_REFERENCE.md (15 min)
3. Follow IMPLEMENTATION_GUIDE.md while coding (30-45 min)
4. Test and validate (30 min)
5. Reference LOGITS_EXTRACTION_ANALYSIS.md as needed

---

**Status**: Complete exploration, ready for implementation  
**Last Updated**: 2025-10-24  
**Version**: Sprint 1 Planning

---

## Document Statistics

- Total Documentation: ~52 KB
- Total Pages (if printed): ~65 pages
- Total Code Examples: 15+
- Total Line Numbers Referenced: 50+
- Diagrams: 5+
- Tables: 12+

All materials are self-contained and can be read offline.

