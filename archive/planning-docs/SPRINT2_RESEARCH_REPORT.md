# BitNet-rs MVP Sprint 2 - Comprehensive Research & Implementation Plan

**Date**: 2025-11-11
**Status**: Research Complete, Plan Posted to GitHub Issue #472
**Target**: Transform BitNet-rs from 0.1 tok/s (unusable) to 10-100 tok/s (production-ready)

---

## Executive Summary

### Research Scope
Analyzed GitHub issue #472 ("MVP Sprint 2") - a comprehensive 8,000+ word roadmap for transforming BitNet-rs from a functional but slow MVP into a production-ready 1-bit neural network inference engine.

### Key Findings

**Current State**:
- Core I2_S quantization working end-to-end with 99%+ parity vs C++ reference
- QK256 (GGML) support implemented via scalar kernels
- Current throughput: **0.1 tokens/second** on 2B models (scalar CPU kernels only)
- Critical bottleneck: QK256 dequantization loop lacks SIMD optimization

**Critical Issues** (prioritized):
1. **Performance (P0 - Blocking)**: Scalar kernels 100x too slow; SIMD optimization is prerequisite for usability
2. **Correctness (P1)**: Prompt template mismatch on base models; generation loop needs verification
3. **Testing (P1)**: Multi-token parity and E2E Q&A tests needed for confidence
4. **GPU (P2)**: CUDA infrastructure exists but needs full implementation post-SIMD

### Deliverables

**Two Comprehensive GitHub Comments Posted**:

1. **Comment #1** - Executive Roadmap (5,500+ words)
   - 5 phased milestones from MVP to production
   - Detailed phase descriptions with sub-components
   - Risk mitigation and timeline
   - Acceptance criteria for each phase
   - URL: https://github.com/EffortlessMetrics/BitNet-rs/issues/472#issuecomment-3515704769

2. **Comment #2** - Technical Deep Dive (4,000+ words)
   - AVX2 optimization strategy with pseudocode
   - 4 key optimization techniques explained
   - Minimal Rust code skeleton for implementation
   - Week-by-week task breakdown with checkboxes
   - Risk & contingency planning
   - Performance metrics tracking table
   - URL: https://github.com/EffortlessMetrics/BitNet-rs/issues/472#issuecomment-3515711227

---

## Implementation Plan Summary

### Phase 1: SIMD Optimization (Weeks 1-6) - CRITICAL PATH

**Objective**: Achieve 10-20 tok/s on CPU (100x improvement)

**Sub-components**:

1. **AVX2 Kernel Implementation** (2-3 weeks)
   - LUT-based 2-bit dequantization via `pshufb` instruction (4-8x speedup)
   - 8-accumulator GEMV parallelization with FMA (6-8x speedup)
   - Memory optimization: aligned 32-byte loads, cache prefetching
   - File: `/home/steven/code/Rust/BitNet-rs/crates/bitnet-kernels/src/cpu/x86_qk256_avx2.rs`
   - Target: >6x speedup over scalar with <1e-5 numerical error

2. **Multi-threaded Row Parallelization** (1 week)
   - Rayon-based row distribution across CPU cores
   - Independent accumulator per thread (zero synchronization)
   - Expected: 6-7x scaling on 8-core system
   - File: `/home/steven/code/Rust/BitNet-rs/crates/bitnet-kernels/src/cpu/mod.rs`
   - Combined with AVX2: 50-70x total speedup realistic

3. **AVX-512 & NEON Architecture Preparation** (0.5 weeks)
   - Design modular kernel dispatcher
   - Runtime feature detection with graceful fallback
   - Documentation for future SIMD variants
   - No implementation yet; design & planning only

4. **Performance Validation & Benchmarking** (1 week)
   - Criterion.rs benchmarks: scalar vs AVX2 vs multi-threaded
   - End-to-end model benchmarks on real microsoft-bitnet-b1.58
   - CI regression job (fail if >5% slowdown)
   - File: `/home/steven/code/Rust/BitNet-rs/crates/bitnet-kernels/benches/kernel_benchmarks.rs`

**Acceptance Criteria**:
- AVX2 achieves >6x speedup over scalar
- Numerical parity: cosine similarity >= 0.9999 to scalar reference
- All QK256 tensor shapes handled without errors
- Property-based tests pass (100+ random inputs)
- Graceful fallback when AVX2 unavailable
- End-to-end: >= 10 tok/s on CPU

---

### Phase 2: Correctness & Generation Loop (Weeks 4-8)

**Overlaps with Phase 1 - can start after 2-3 weeks**

**2.1 Prompt Template & Model Detection**
- Auto-detect base vs instruction-tuned models
- BitNet-specific: use instruct template by default (not llama3-chat)
- LLaMA models: use llama3-chat template
- File: `/home/steven/code/Rust/BitNet-rs/crates/bitnet-inference/src/prompt_template.rs`

**2.2 Generation Loop Correctness** (1.5 weeks)
- Test EOS token behavior, stop sequences, max_tokens limit
- Verify greedy vs temperature sampling
- Tokenization round-trip validation
- File: `/home/steven/code/Rust/BitNet-rs/crates/bitnet-inference/tests/generation_loop.rs`
- Key test: Math sanity check (2+2=4) must pass reliably

**2.3 Multi-Token Parity Testing** (1 week)
- Deterministic sequence generation (fixed random seed)
- Compare Rust vs C++ token sequences
- Validate logits match at each step (cos sim >= 0.999)
- File: `/home/steven/code/Rust/BitNet-rs/crates/bitnet-models/tests/parity_multitoken.rs`

---

### Phase 3: Cross-Validation & Testing (Weeks 6-10)

**Overlaps with Phases 1-2**

**3.1 Comprehensive Parity Harness**
- 4 test scenarios: single-token logits, multi-token greedy, template variations, edge cases
- File: `/home/steven/code/Rust/BitNet-rs/crossval/src/`

**3.2 End-to-End Q&A Test Suite**
- Deterministic Q&A tests with BITNET_DETERMINISTIC=1
- Math: "2+2=" → "4"
- Geography: "Capital of France?" → "Paris"
- File: `/home/steven/code/Rust/BitNet-rs/crates/bitnet-cli/tests/qa_e2e.rs`

**3.3 CI Regression Testing**
- Run parity tests on every PR
- Fail if cosine similarity < 0.99 or throughput regresses >10%
- File: `.github/workflows/crossval.yml`

---

### Phase 4: GPU Acceleration (Weeks 8-14)

**Starts AFTER Phase 1 (CPU optimization completes)**

**4.1 CUDA Kernel Development** (2.5 weeks)
- Block-wise parallelism: one GPU block per output neuron
- LUT-based dequantization (mirrors CPU approach)
- Shared memory optimization for input vector caching
- File: `/home/steven/code/Rust/BitNet-rs/crates/bitnet-kernels/csrc/qk256_gemv.cu`
- Target: >50 tok/s on modern GPU (A100 reference)

**4.2 GPU Integration & Fallback** (1.5 weeks)
- Device selection at model load time
- Transparent CPU fallback on GPU error
- CLI flag: `--device auto|cpu|gpu`
- File: `/home/steven/code/Rust/BitNet-rs/crates/bitnet-inference/src/backend.rs`

**4.3 GPU Performance Validation** (1 week)
- Criterion.rs benchmarks: GPU vs CPU
- Parity tests with GPU backend (cos sim >= 0.99)
- Memory usage validation

---

### Phase 5: Polish & Release (Weeks 12-15)

**5.1 Codebase Cleanup**
- Remove legacy code paths, unify error handling

**5.2 Documentation**
- README with real performance numbers (10-20 tok/s CPU, 50-100 tok/s GPU)
- Architecture docs for CPU SIMD and GPU CUDA
- Performance tuning guide

**5.3 Release v0.2.0**
- Git tag, publish to crates.io, announce to community

---

## Technical Deep Dive: AVX2 Optimization

### Current Bottleneck

QK256 dequantization loop processes one 2-bit value at a time:
```rust
for i in 0..output_size {
    let mut sum = 0.0;
    for j in 0..input_size {
        let packed = quantized[i][j];           // 4 2-bit values in 1 byte
        let val = extract_2bits(packed);         // Sequential extraction
        sum += val * input[j];                   // One multiply-add per iteration
    }
    output[i] = sum;
}
```

Performance: ~0.1 tok/s (4 cycles per 2-bit value)

### AVX2 Optimization Strategy

**Key Techniques**:

1. **Lookup Table (LUT) Dequantization**
   - Convert 4 x 2-bit values to 4 x i8 values in parallel
   - Use `pshufb` (shuffle) instruction for 32 parallel LUT lookups
   - Expected: 4-6x speedup over scalar extraction

2. **8-Accumulator Loop Unrolling**
   - Process 8 rows simultaneously using independent accumulators
   - CPU executes out-of-order without data dependencies
   - Expected: 6-8x speedup from instruction parallelism

3. **Memory Optimization**
   - Align weights to 32-byte boundaries
   - Prefetch next cache line during computation
   - Expected: 2x efficiency from better cache behavior

4. **Combined Result**
   - AVX2 alone: 6-8x speedup
   - With 8 cores: 50-56x speedup (realistic limit ~70-80x with bandwidth)
   - Total: 0.1 tok/s → 5-8 tok/s with AVX2, 10-20 tok/s with threading

### Code Structure

Minimal implementation skeleton provided in technical comment:
- `Avx2QK256Kernel` struct with `dequantize_qk256_avx2()` and `gemv_qk256_avx2()`
- Runtime feature detection: `is_x86_feature_detected!("avx2")`
- Graceful fallback to scalar when AVX2 unavailable
- Property-based tests for correctness validation

---

## Timeline & Milestones

| Phase | Duration | Start | End | Priority | Target |
|-------|----------|-------|-----|----------|--------|
| **1: SIMD** | 4-6 wks | Wk 1 | Wk 6 | **P0** | 10-20 tok/s CPU |
| **2: Correctness** | 2-3 wks | Wk 4 | Wk 8 | **P1** | Math sanity check pass |
| **3: Testing** | 2-3 wks | Wk 6 | Wk 10 | **P1** | E2E Q&A confidence |
| **4: GPU** | 4-6 wks | Wk 8 | Wk 14 | **P2** | 50-100 tok/s GPU |
| **5: Polish** | 2-3 wks | Wk 12 | Wk 15 | **P3** | v0.2.0 release |

**Note**: Phases overlap; Phase 2 starts 2-3 weeks after Phase 1 begins

---

## Success Metrics

### Performance
- CPU: >= 10 tok/s (100x from 0.1 tok/s)
- GPU: >= 50 tok/s (post-Phase 1)
- Memory: No leaks, reasonable for 2B models

### Correctness
- Math sanity check: 100% pass rate (2+2=4)
- Parity vs C++: cosine similarity >= 0.999 at all steps
- Generation: EOS, stop sequences, max_tokens all working

### Testing
- Parity harness: 4 scenarios pass
- E2E Q&A: Deterministic tests pass
- CI: All regressions caught and reported

### Code Quality
- Clippy: 0 warnings
- Tests: All enabled tests pass on CI
- Docs: Complete architecture and usage guides

---

## Risk Mitigation

### Risk 1: AVX2 Speedup < 6x Target
- Profile with perf/Instruments to identify bottleneck
- Try alternative optimizations: prefetching, software pipelining
- Fallback to scalar if necessary (always available)

### Risk 2: GPU Kernel Diverges from C++
- Rigorous unit testing of GPU kernel
- Compare logits at every step vs C++
- Use Nsight profiler to debug

### Risk 3: Cross-Platform SIMD Issues
- Test on multiple CPU types (older without AVX2, newest Zen 4)
- Runtime feature detection is prerequisite
- Graceful fallback is mandatory

### Risk 4: Test Suite Becomes Too Slow
- Use smaller models for CI tests
- Separate fast unit tests from slow integration tests
- Run full suite only on main branch

---

## Critical Success Factor

**Why Phase 1 (SIMD) is Blocking**:

The current 0.1 tok/s throughput makes BitNet-rs unsuitable for any practical use:
- 100 token prompt takes 1,000 seconds (16+ minutes)
- Even a 16-token response takes 160 seconds
- This is a proof-of-concept, not a usable system

Without SIMD optimization, **all subsequent phases are blocked** because:
- Phase 2 (correctness testing) needs practical throughput to test
- Phase 3 (CI regression) needs baseline performance to detect regressions
- Phase 4 (GPU) can't achieve 100x improvement without CPU baseline

**SIMD is the critical path** - complete this first before other work.

---

## Actionable Next Steps

1. **Assign Phase 1 to primary developer** - highest priority, critical path
2. **Create tracking issue** with sub-tasks for each week
3. **Set up performance CI job** early (prevent regressions)
4. **Start Phase 2 work** in parallel (2-3 weeks after Phase 1 begins)
5. **Establish weekly sync** to track progress and unblock issues

---

## Files & Resources

### BitNet-rs Crate Structure Referenced

- `/home/steven/code/Rust/BitNet-rs/crates/bitnet-kernels/src/cpu/x86_qk256_avx2.rs` (new)
- `/home/steven/code/Rust/BitNet-rs/crates/bitnet-kernels/src/cpu/mod.rs` (update)
- `/home/steven/code/Rust/BitNet-rs/crates/bitnet-kernels/benches/kernel_benchmarks.rs` (update)
- `/home/steven/code/Rust/BitNet-rs/crates/bitnet-inference/src/prompt_template.rs` (update)
- `/home/steven/code/Rust/BitNet-rs/crates/bitnet-inference/tests/generation_loop.rs` (new)
- `/home/steven/code/Rust/BitNet-rs/crates/bitnet-models/tests/parity_multitoken.rs` (new)
- `/home/steven/code/Rust/BitNet-rs/crates/bitnet-cli/tests/qa_e2e.rs` (new)
- `/home/steven/code/Rust/BitNet-rs/crates/bitnet-kernels/csrc/qk256_gemv.cu` (new, Phase 4)
- `.github/workflows/performance.yml` (new)
- `.github/workflows/crossval.yml` (update)

### Documentation Resources

- Intel AVX2 Intrinsics: https://www.intel.com/content/dam/develop/external/us/en/documents/intrinsics-guide/
- Criterion.rs Benchmarking: https://bheisler.github.io/criterion.rs/book/
- Rayon Parallelization: https://docs.rs/rayon/
- CUDA Programming: https://docs.nvidia.com/cuda/cuda-c-programming-guide/

---

## Conclusion

This research and implementation plan successfully transforms the narrative roadmap in issue #472 into:

- **Clear prioritization**: P0 SIMD is critical path; GPU is P2 (post-CPU)
- **Realistic effort estimates**: 4-6 weeks for Phase 1, 12-15 weeks total
- **Concrete technical approaches**: AVX2 LUT + 8-accumulator parallelism with code skeleton
- **Measurable success criteria**: 10-20 tok/s target with acceptance tests
- **Comprehensive risk mitigation**: Contingency plans for 4 major risks
- **Phased delivery**: CPU → GPU → Polish → Release

**BitNet-rs can achieve production readiness (10-100 tok/s) in 3-4 months** with focused execution on this plan.

Two comprehensive GitHub comments have been posted to issue #472 with actionable week-by-week checklists, code examples, and detailed technical guidance.

**Ready to ship!**

---

**Research Completed By**: BitNet-rs GitHub Research Specialist
**Date**: 2025-11-11
**GitHub Issue**: https://github.com/EffortlessMetrics/BitNet-rs/issues/472
**Comment 1**: https://github.com/EffortlessMetrics/BitNet-rs/issues/472#issuecomment-3515704769
**Comment 2**: https://github.com/EffortlessMetrics/BitNet-rs/issues/472#issuecomment-3515711227
