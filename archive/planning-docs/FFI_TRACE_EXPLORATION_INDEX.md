# FFI & Tracing Infrastructure - Exploration Index

**Exploration Date**: October 22, 2025  
**Status**: Complete  
**Thoroughness**: Medium (core components and data flow analyzed)

---

## Documentation Created

This exploration generated comprehensive documentation about BitNet-rs's FFI and tracing capabilities:

### 1. **Comprehensive Reference** (Primary Document)
📄 **File**: `/docs/reports/FFI_TRACE_INFRASTRUCTURE.md` (21 KB, 400+ lines)

**Contents**:
- Executive summary
- FFI architecture (3 crates: bitnet-ffi, bitnet-ggml-ffi, crossval)
- 30+ exposed C functions (models, inference, streaming, metrics)
- Token passing analysis (current string-only, workarounds for token access)
- Complete trace infrastructure (capture, format, stages)
- Logits divergence detection (cosine similarity, L2, max diff)
- Weight mapping rules (tied weights, transposition, head variants)
- Data flow diagrams
- Strengths & limitations table
- Quick reference commands
- File location map
- Extension recommendations

### 2. **Quick Reference** (Quick Lookup)
📄 **File**: `/docs/QUICK_FFI_TRACE_REFERENCE.md` (5.4 KB, 150 lines)

**Contents**:
- 1-minute overview table
- 4 key capabilities with code examples
- Essential commands (trace capture, comparison, debugging)
- Weight mapping rules table
- Trace capture points
- Troubleshooting section
- File reference with line numbers
- Next steps for extension

### 3. **Architecture Diagrams** (Visual Reference)
📄 **File**: `/docs/FFI_TRACE_ARCHITECTURE_DIAGRAM.txt` (14 KB, 300+ lines)

**Contents**:
- FFI layer stack (external consumers → C API → Rust library)
- Trace capture flow (during inference, record format, output)
- Cross-validation workflow (Rust vs C++ traces)
- Weight mapping scenarios (tied weights, dedicated head)
- Logits divergence detection (algorithm, example)
- Token passing comparison (current vs needed)

---

## What This Exploration Covered

### ✅ Covered in Depth

1. **FFI Interface Architecture**
   - bitnet-ffi crate structure (1178 lines c_api.rs)
   - 30+ C functions across model, inference, streaming, GPU
   - Thread-safe design, error handling, configuration

2. **Token Passing**
   - String-based public C API (bitnet_inference takes prompt string)
   - Token-level access via bitnet_sys::wrapper::Session
   - Missing C FFI functions identified: bitnet_tokenize(), bitnet_inference_tokens()

3. **Trace Infrastructure**
   - bitnet-trace crate (281 lines)
   - 6 capture points: embeddings, q_proj, attn_out, ffn_out, logits, layer_norm
   - Blake3 hashing, RMS computation
   - Metadata: shape, dtype, layer, stage, token position
   - JSON serialization format

4. **Logits Comparison**
   - Per-position divergence detection
   - Metrics: cosine similarity (threshold 1e-4), L2 distance, max absolute diff
   - Test infrastructure in per_position_logits.rs

5. **Weight Mapping**
   - Tied weight handling (embedding ↔ LM head)
   - Pre-transpose optimization (load-time, not per-token)
   - Transposition detection (GGUF storage variants)
   - Debug validation (BITNET_DEBUG_LOGITS)

### ⚠️ Partially Covered

- C++ reference setup (documented in docs/howto/cpp-setup.md)
- GPU-specific FFI features (available but not deeply analyzed)
- Performance tuning for trace capture (JSON overhead)

### ❌ Out of Scope

- Build system details (Cargo.toml parsing)
- Python trace_diff.py implementation details
- CUDA kernel integration with FFI
- Production deployment patterns

---

## Key Findings Summary

### Strengths
| Component | Capability |
|-----------|-----------|
| C FFI API | 30+ functions, complete error handling, thread-safe |
| Trace Capture | Environment-gated, Blake3 hashing, rich metadata |
| Divergence Detection | Multi-metric (cosine, L2, max diff) with thresholds |
| Weight Mapping | Handles tied weights, transposition, quantization variants |
| Performance | Pre-transpose optimization eliminates per-token overhead |

### Gaps
| Component | Issue |
|-----------|-------|
| Token FFI | No C functions for direct token inference |
| Streaming Logits | Text tokens only, no per-token logits |
| Trace Overhead | JSON serialization per tensor (env-gated) |
| C++ Dependency | Requires manual setup (BITNET_CPP_DIR) |
| Multi-GPU | No explicit multi-device FFI support |

---

## How to Use This Documentation

### For Understanding the System
1. Start with **QUICK_FFI_TRACE_REFERENCE.md** (5 min read)
2. Review **FFI_TRACE_ARCHITECTURE_DIAGRAM.txt** (10 min read)
3. Deep dive: **FFI_TRACE_INFRASTRUCTURE.md** (20+ min read)

### For Debugging
1. Check **Troubleshooting** section in QUICK_FFI_TRACE_REFERENCE.md
2. Use **Essential Commands** for trace capture/comparison
3. Reference **File Reference** for code locations

### For Extending the System
1. Review **Next Steps for Integration** in comprehensive document
2. Check **Weight Mapping Rules** for model handling
3. Reference critical file locations for implementation points

---

## Quick Navigation to Code

### FFI Implementation
- **Entry Point**: `crates/bitnet-ffi/src/c_api.rs` (1178 lines)
  - Model functions: lines ~143-261
  - Inference functions: lines ~284-445
  - Streaming functions: lines ~795-944
  - Metrics functions: lines ~962-1023

- **Inference Manager**: `crates/bitnet-ffi/src/inference.rs` (495 lines)
  - generate_with_config: lines ~50-87
  - get_or_create_engine: lines ~237-282

### Trace Capture
- **API**: `crates/bitnet-trace/src/lib.rs` (281 lines)
  - dump_trace function: lines ~106-167
  - TraceRecord struct: lines ~49-72

- **Tool**: `xtask/src/trace_diff.rs` (172 lines)
  - run function: lines ~78-155

### Weight Mapping
- **Logits Computation**: `crates/bitnet-models/src/transformer.rs:1609-1740`
  - Tied weights path: lines ~1632-1640
  - Dedicated LM head: lines ~1617-1620

- **Head-Tie Loading**: `crates/bitnet-models/src/transformer.rs:1355-1375`
  - Pre-transpose optimization: lines ~1366-1371

### Cross-Validation
- **C++ Bindings**: `crossval/src/cpp_bindings.rs` (200 lines)
- **Logits Compare**: `crossval/src/logits_compare.rs` (160 lines)
- **Tests**: `crossval/tests/per_position_logits.rs`

---

## Related Documentation

Other BitNet-rs documentation that complements this exploration:

- `docs/howto/cpp-setup.md` — C++ reference setup
- `docs/howto/validate-models.md` — Model validation workflow
- `docs/GPU_SETUP.md` — GPU configuration
- `docs/performance-benchmarking.md` — Benchmark infrastructure
- `CLAUDE.md` — Project status and essential commands

---

## Next Actions

### If You Need To...

**Add token-level FFI** → See FFI_TRACE_INFRASTRUCTURE.md Section 10.1
**Debug weight mapping** → Use BITNET_DEBUG_LOGITS and check transformer.rs:1609-1740
**Compare Rust vs C++ traces** → Run commands in QUICK_FFI_TRACE_REFERENCE.md "Essential Commands"
**Understand divergence detection** → Review FFI_TRACE_ARCHITECTURE_DIAGRAM.txt "Logits Divergence"
**Extend tracing** → Check recommendations in Section 10 of comprehensive document

---

**Document Version**: 1.0  
**Last Updated**: 2025-10-24  
**Status**: Complete and Verified
