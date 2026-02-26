# BitNet-rs Tracing Infrastructure - Documentation Index

This directory contains comprehensive documentation of the tensor activation tracing system used for cross-validation debugging.

## Quick Start

Start here for a fast overview:
- **[TRACING_QUICK_REFERENCE.md](TRACING_QUICK_REFERENCE.md)** - One-page cheat sheet with key facts

## Complete Documentation

### Findings & Analysis

1. **[TRACING_FINDINGS_SUMMARY.md](TRACING_FINDINGS_SUMMARY.md)** - Executive summary
   - Answers to all 11 exploration questions
   - Current struct definition (TraceRecord)
   - Current JSON format
   - API surface (dump_trace function)
   - Blake3 hashing implementation
   - File naming conventions
   - Environment variable control
   - Performance characteristics
   - Recommendations

2. **[TRACING_INFRASTRUCTURE_ANALYSIS.md](TRACING_INFRASTRUCTURE_ANALYSIS.md)** - Deep technical dive
   - 22 KB comprehensive reference
   - Code examples and implementation details
   - Performance benchmarks
   - Extension strategies (Phase 1-3)
   - Backward compatibility patterns
   - Testing infrastructure
   - Cross-validation workflows
   - Real-world usage examples

### Quick Reference

3. **[TRACING_QUICK_REFERENCE.md](TRACING_QUICK_REFERENCE.md)** - Fast lookup guide
   - Current struct definition
   - JSON output example
   - API overview
   - Environment control
   - Feature gating
   - Blake3 details
   - Performance table
   - Extension guide
   - Design principles

## Current State Summary

### Struct Definition
```rust
pub struct TraceRecord {
    pub name: String,           // Tensor identifier
    pub shape: Vec<usize>,      // Tensor dimensions
    pub dtype: String,          // Data type
    pub blake3: String,         // 64-char hex hash
    pub rms: f64,               // Root mean square
    pub num_elements: usize,    // Element count
}
```

### API
```rust
pub fn dump_trace(name: &str, tensor: &Tensor) -> candle_core::Result<()>
```

### Key Characteristics
- **Zero-cost when disabled**: ~1-10 µs per call
- **Feature-gated**: Only compiled with `--features trace`
- **Stateless**: No global mutable state
- **Sequential I/O**: No synchronization needed
- **Simple**: Single public function for common case

## What's in Each Document

### TRACING_FINDINGS_SUMMARY.md
Use this for:
- Quick answers to specific questions
- Understanding current implementation
- Learning what fields exist
- Understanding how to extend safely
- Performance expectations
- Current test coverage

**Length:** ~16 KB (5 minutes to read)

### TRACING_INFRASTRUCTURE_ANALYSIS.md
Use this for:
- Detailed technical understanding
- Code examples and line numbers
- Phase-by-phase extension strategy
- Performance analysis
- Cross-validation workflows
- Integration patterns
- Testing strategies

**Length:** ~22 KB (15 minutes to read)

### TRACING_QUICK_REFERENCE.md
Use this for:
- Quick lookup of specific facts
- Example JSON output
- Current tracepoints
- Extension phases
- Backward compatibility strategy
- One-page reference during implementation

**Length:** ~6 KB (2 minutes to read)

## Key Files in Codebase

**Core implementation:**
- `crates/bitnet-trace/src/lib.rs` (194 lines)
- `crates/bitnet-trace/README.md` (120 lines)
- `crates/bitnet-trace/tests/integration_test.rs` (177 lines)
- `crates/bitnet-trace/Cargo.toml` (16 lines)

**Usage sites:**
- `crates/bitnet-models/src/transformer.rs` (~7 call sites)
- `crates/bitnet-models/Cargo.toml` (trace feature)

**Integration:**
- `scripts/run_crossval_sweep.sh` (comprehensive workflow)

## Current Tracepoints

### Full Sequence (`forward_full()`)
1. `t0/embeddings` - After embedding layer
2. `t0/blk{}/attn_norm` - Attention layer norm (per layer)
3. `t0/blk{}/q_proj` - Query projection (per layer)
4. `t0/blk{}/attn_scores_softmax` - Attention softmax (per layer)
5. `t0/logits` - Final output logits

### Token-by-Token (`forward_incremental()`)
1. `t0/embeddings` - Current token embedding
2. `t0/logits` - Current token logits

## Extension Roadmap

### Phase 1: Add Optional Fields
- Add `seq_index`, `layer_idx`, `stage` as `Option<T>`
- Use `#[serde(skip_serializing_if = "Option::is_none")]`
- Maintain full backward compatibility

### Phase 2: New API
- Create `dump_trace_with_context()` function
- Keep existing `dump_trace()` for compatibility

### Phase 3: Gradual Migration
- Update call sites at leisure
- Mix old and new styles
- Maintain compatibility

## Using These Documents

**I'm new, give me an overview:**
→ Read [TRACING_QUICK_REFERENCE.md](TRACING_QUICK_REFERENCE.md) (2 min)

**I need specific facts:**
→ Check [TRACING_FINDINGS_SUMMARY.md](TRACING_FINDINGS_SUMMARY.md) (5 min)

**I'm implementing changes:**
→ Use [TRACING_QUICK_REFERENCE.md](TRACING_QUICK_REFERENCE.md) as reference + [TRACING_INFRASTRUCTURE_ANALYSIS.md](TRACING_INFRASTRUCTURE_ANALYSIS.md) for patterns

**I need deep technical details:**
→ Read [TRACING_INFRASTRUCTURE_ANALYSIS.md](TRACING_INFRASTRUCTURE_ANALYSIS.md) (15 min)

**I'm reviewing backward compatibility:**
→ Search [TRACING_INFRASTRUCTURE_ANALYSIS.md](TRACING_INFRASTRUCTURE_ANALYSIS.md) for "skip_serializing_if" section

## Key Insights

### Zero-Cost Design
- When disabled: Single env var check (~1-10 µs)
- When unused feature: Code completely compiled out
- No memory overhead, no allocations

### Backward Compatibility
- New optional fields use `skip_serializing_if`
- Old JSON without fields still parses (defaults to None)
- New JSON with fields parses in old code (fields ignored)
- Progressive adoption possible without breaking changes

### Performance
- Per-tensor: ~1-20 ms (dominated by file I/O)
- Recommendation: Use ramdisk for BITNET_TRACE_DIR
- For 2B model: ~500 ms overhead for prefill, ~50s for 128 tokens

### Current Limitations
- seq/layer/stage encoded in name string, not explicit fields
- No timestamp or generation step metadata
- No schema versioning in JSON

## Related Documentation

- `crates/bitnet-trace/README.md` - User guide
- `docs/development/validation-framework.md` - Validation system
- `scripts/README_run_crossval_sweep.md` - Cross-validation script
- `CLAUDE.md` - Project overview (see "Inference Validation" section)

---

**Last updated:** 2025-10-24
**Status:** Complete analysis of current implementation
**Next steps:** Implement Phase 1 extension when ready

See individual documents for details.
