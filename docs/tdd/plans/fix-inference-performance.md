# Fix Plan: Inference Performance Issues (Finding 3)

**Status:** 📋 Planning
**Priority:** HIGH
**Blocking:** Practical testing and user experience

**Related:**
- [Investigation Report](../receipts/inference_quality_investigation.md#finding-3)
- [CLAUDE.md Performance Notes](../../../CLAUDE.md#current-limitations-mvp-phase)

---

## Problem Statement

**Symptom:** Simple 1-token inference (`2+2=`) took >60 seconds before being killed

**Impact:**
- Blocks practical testing of intelligibility
- Prevents users from running inference
- Makes development iteration impossible
- Hides potential correctness bugs behind performance issues

**Current explanation (CLAUDE.md):**
> QK256 Performance: Scalar-only kernels. For quick validation, limit to --max-new-tokens 4-16.

**⚠️ Critical question:** Is this actually a QK256 issue, or are there deeper bugs?

---

## Investigation Plan

### Phase 1: Confirm Quantization Format ✅ PRIORITY

**Goal:** Verify which quantization path is actually being used

**Tasks:**

1. **Check model metadata:**
   ```bash
   cargo run -p bitnet-cli --release --features cpu -- inspect \
     models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf \
     --ln-stats --gate auto
   ```

   **Questions:**
   - What is the actual quantization type? (I2_S, QK256, TL1, TL2, IQ2_S)
   - Are we using the right dequantization path?
   - Are tensors correctly aligned?

2. **Instrument quantization dispatch:**

   Add logging to quantization selection:
   ```rust
   // In bitnet-quantization or bitnet-kernels
   tracing::info!("Selected quantization path: {:?} for tensor {}", format, name);
   ```

   **Expected:** Should see which format is being used for each tensor

3. **Compare against C++ reference:**
   ```bash
   # If BITNET_CPP_DIR is set
   RUST_LOG=debug cargo run -p xtask -- crossval 2>&1 | grep -i "quantization\|format\|kernel"
   ```

**Acceptance criteria:**
- ✅ We know exactly which quantization format is being used
- ✅ We confirm whether it's actually QK256 or something else
- ✅ We verify the dispatch logic is correct

---

### Phase 2: Profile Hot Paths 🔥 CRITICAL

**Goal:** Find actual bottleneck with data, not assumptions

**Tasks:**

1. **CPU profiling with perf/flamegraph:**
   ```bash
   # Install flamegraph
   cargo install flamegraph

   # Profile inference run
   cargo flamegraph -p bitnet-cli --release --features cpu -- run \
     --model models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf \
     --tokenizer models/microsoft-bitnet-b1.58-2B-4T-gguf/tokenizer.json \
     --prompt "2+2=" \
     --max-tokens 1

   # Output: flamegraph.svg showing hot paths
   ```

2. **Trace-level logging:**
   ```bash
   RUST_LOG=trace cargo run -p bitnet-cli --release --features cpu -- run \
     --model models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf \
     --tokenizer models/microsoft-bitnet-b1.58-2B-4T-gguf/tokenizer.json \
     --prompt "2+2=" \
     --max-tokens 1 2>&1 | tee trace.log

   # Analyze timing gaps between log entries
   ```

3. **Benchmark individual operations:**

   Add timing instrumentation:
   ```rust
   use std::time::Instant;

   let start = Instant::now();
   let result = dequantize_i2s(&tensor, &output);
   tracing::info!("Dequantize took: {:?}", start.elapsed());

   let start = Instant::now();
   let logits = forward_pass(&model, &input);
   tracing::info!("Forward pass took: {:?}", start.elapsed());
   ```

**Acceptance criteria:**
- ✅ We have flamegraph showing exact hot paths
- ✅ We know if bottleneck is in: dequantization, matmul, layer norm, attention, or something else
- ✅ We have concrete timing numbers for each operation

---

### Phase 3: Check for Algorithmic Issues 🐛 DEEP DIVE

**Goal:** Find potential O(n²) or worse bugs masquerading as "slow quantization"

**Potential bugs to check:**

1. **Memory allocation in hot loop:**
   ```rust
   // BAD: Allocating every token
   for _ in 0..max_tokens {
       let mut output = vec![0.0; vocab_size]; // ❌ Repeated allocation
       forward_pass(&model, &input, &mut output);
   }

   // GOOD: Reuse buffer
   let mut output = vec![0.0; vocab_size];
   for _ in 0..max_tokens {
       forward_pass(&model, &input, &mut output); // ✅ Reuse
   }
   ```

2. **Unnecessary copies:**
   ```rust
   // Check for .clone() or .to_vec() in generation loop
   rg "\.clone\(\)|\.to_vec\(\)" crates/bitnet-inference/src/generation/
   ```

3. **Repeated model loading:**
   ```rust
   // Are we re-parsing GGUF metadata every token?
   // Are we re-loading tensors from mmap?
   ```

4. **Inefficient tensor indexing:**
   ```rust
   // Check for bounds checking in tight loops
   // Check for slice creation instead of pointer arithmetic
   ```

5. **Lock contention:**
   ```rust
   // Are we holding locks during compute?
   // Are we using Mutex where we could use RefCell?
   ```

6. **Debug assertions:**
   ```rust
   // Even in release builds, some debug asserts can be expensive
   rg "debug_assert|assert!" crates/bitnet-kernels/src/
   ```

**Acceptance criteria:**
- ✅ We've audited all hot paths for algorithmic issues
- ✅ We've confirmed no O(n²) or worse behavior
- ✅ We've verified memory allocations are minimal

---

### Phase 4: Verify SIMD Codegen ⚡ OPTIMIZATION

**Goal:** Confirm SIMD paths are actually being used

**Tasks:**

1. **Check feature compilation:**
   ```bash
   # Verify CPU features detected
   cargo run -p xtask -- preflight

   # Check what gets compiled
   cargo build --release --features cpu --verbose 2>&1 | grep -i "simd\|avx\|neon"
   ```

2. **Inspect generated assembly:**
   ```bash
   # For a hot function (e.g., I2S dequantize)
   cargo rustc --release -p bitnet-kernels --features cpu -- --emit asm

   # Look for SIMD instructions (vmovaps, vpaddd, etc.)
   grep -i "vmov\|vpadd\|vpermute" target/release/deps/*.s
   ```

3. **Runtime feature detection:**
   ```rust
   // Add logging to device_features
   use bitnet_kernels::device_features::{cpu_features, avx2_available};

   tracing::info!("CPU features: {:?}", cpu_features());
   tracing::info!("AVX2 available: {}", avx2_available());
   ```

4. **Compare SIMD vs scalar:**
   ```bash
   # Force scalar path
   RUSTFLAGS="-C target-cpu=generic" cargo build --release --features cpu

   # Force native SIMD
   RUSTFLAGS="-C target-cpu=native" cargo build --release --features cpu

   # Benchmark both
   hyperfine \
     "./target-scalar/release/bitnet run ..." \
     "./target-native/release/bitnet run ..."
   ```

**Acceptance criteria:**
- ✅ We confirm SIMD instructions are being emitted
- ✅ We verify runtime detection works
- ✅ We measure SIMD speedup vs scalar

---

### Phase 5: Check Model Loading Path 📁 CORRECTNESS

**Goal:** Ensure we're not doing something silly in model loading that blocks inference

**Potential issues:**

1. **Lazy vs eager loading:**
   ```rust
   // Are we deferring tensor loading to first use?
   // Does first inference trigger expensive I/O?
   ```

2. **Mmap configuration:**
   ```rust
   // Is mmap configured correctly?
   // Are we using MAP_POPULATE?
   // Is page faulting causing slowdown?
   ```

3. **Tensor validation:**
   ```rust
   // Are we running expensive validation on every forward pass?
   // Is LayerNorm validation running in the hot path?
   ```

4. **Tokenizer initialization:**
   ```rust
   // Time tokenizer loading separately
   let start = Instant::now();
   let tokenizer = Tokenizer::from_file(...)?;
   tracing::info!("Tokenizer load: {:?}", start.elapsed());
   ```

**Acceptance criteria:**
- ✅ Model loading completes in <1 second
- ✅ Tokenizer loading completes in <100ms
- ✅ First token latency ≈ subsequent token latency (no lazy loading penalty)

---

## Fix Strategies (Based on Investigation Results)

### Strategy A: Optimize Quantization Hot Path

**If profiling shows dequantization is the bottleneck:**

1. **Enable AVX2 fast path** (already implemented in PR #473):
   ```bash
   RUSTFLAGS="-C target-cpu=native -C target-feature=+avx2" \
     cargo build --release --features cpu
   ```

2. **Implement planned QK256 optimizations:**
   - Nibble LUT unpack via `pshufb` (2-bit → signed i8 mapping)
   - FMA tiling (8-16 rows, unroll dot-products)
   - Load combine (reduce AVX crossings)
   - Prefetch (next code block & input)

3. **Target:** ≥3× speedup vs current scalar QK256

**Estimated effort:** 2-3 days (already planned for post-MVP)

---

### Strategy B: Fix Algorithmic Issues

**If profiling shows O(n²) behavior or allocation storms:**

1. **Pre-allocate buffers:**
   ```rust
   struct GenerationState {
       logits_buffer: Vec<f32>,
       attention_cache: AttentionCache,
       output_buffer: Vec<f32>,
   }
   ```

2. **Eliminate clones in hot path:**
   ```rust
   // Use borrows and slices instead of clones
   fn forward_pass(&self, input: &[u32], output: &mut [f32]) {
       // Work in-place
   }
   ```

3. **Cache reusable computations:**
   ```rust
   // Don't recompute positional encodings every token
   lazy_static! {
       static ref ROPE_CACHE: Vec<f32> = compute_rope_embeddings();
   }
   ```

**Estimated effort:** 1-2 days

---

### Strategy C: Optimize I/O and Loading

**If profiling shows mmap or I/O bottlenecks:**

1. **Prefault memory maps:**
   ```rust
   use memmap2::MmapOptions;

   let mmap = MmapOptions::new()
       .populate() // Prefault pages
       .map(&file)?;
   ```

2. **Eager load critical tensors:**
   ```rust
   // Load embedding and output weights into memory
   // Keep quantized weights in mmap
   ```

3. **Parallel tensor loading:**
   ```rust
   use rayon::prelude::*;

   tensors.par_iter_mut().for_each(|t| {
       t.load_from_mmap()?;
   });
   ```

**Estimated effort:** 1 day

---

### Strategy D: Debug Build Optimization

**If profiling confirms debug builds are the issue:**

1. **Add release-with-debug profile:**
   ```toml
   [profile.release-with-debug]
   inherits = "release"
   debug = true
   ```

2. **Document required build flags:**
   ```bash
   # Update CLAUDE.md and docs
   RUSTFLAGS="-C target-cpu=native -C opt-level=3 -C lto=thin" \
     cargo build --release --features cpu,full-cli
   ```

3. **Add performance smoke test to CI:**
   ```yaml
   - name: Performance smoke test
     run: |
       timeout 30s cargo run --release -p bitnet-cli -- run \
         --model test-model.gguf --prompt "test" --max-tokens 4
   ```

**Estimated effort:** Half day (documentation)

---

## Acceptance Criteria for "Fixed"

**Minimum viable performance (MVP complete):**
- ✅ Simple 1-token inference (`2+2=`) completes in <10 seconds (release build)
- ✅ 128-token generation completes in <2 minutes (release build)
- ✅ Performance is within 2× of bitnet.cpp C++ reference

**Good performance (post-MVP):**
- ✅ 1-token inference: <1 second
- ✅ 128-token generation: <30 seconds
- ✅ Performance within 1.5× of C++ reference

**Excellent performance (optimization target):**
- ✅ 1-token inference: <100ms
- ✅ 128-token generation: <10 seconds
- ✅ Performance matches or exceeds C++ reference

---

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| **Actual root cause is QK256 scalar** | High | Implement planned SIMD optimizations (Strategy A) |
| **Root cause is algorithmic bug** | Critical | Thorough profiling before jumping to solutions (Phase 2) |
| **Multiple compounding issues** | Medium | Investigate systematically, fix one at a time |
| **C++ reference is also slow** | Low | Verify C++ performance first as baseline |
| **Model is genuinely huge** | Medium | Document expected performance per model size |

---

## Execution Timeline

**Week 1: Investigation (Phases 1-3)**
- Day 1: Confirm quantization format (Phase 1)
- Day 2-3: Profile hot paths (Phase 2)
- Day 4-5: Audit for algorithmic issues (Phase 3)

**Week 2: Verification and Quick Wins (Phases 4-5)**
- Day 1-2: Verify SIMD codegen (Phase 4)
- Day 3: Check model loading (Phase 5)
- Day 4-5: Implement quick wins (e.g., fix allocations, enable optimizations)

**Week 3: Targeted Optimization**
- Execute appropriate strategy (A, B, C, or D) based on investigation results
- Measure improvements
- Iterate if needed

**Week 4: Validation and Documentation**
- Run full test suite with real models
- Update performance baselines
- Document findings and fixes

---

## Success Metrics

**Technical metrics:**
- Inference latency (tokens/second)
- First token latency (ms)
- Memory allocation rate (allocations/token)
- CPU utilization (% of available compute)

**User-facing metrics:**
- Time to first response (<10s for simple queries)
- Interactive chat responsiveness (<2s per exchange)
- Developer iteration speed (can test changes quickly)

---

## Open Questions

1. **What is the actual quantization format of microsoft-bitnet model?**
   - Answer via: Phase 1 investigation

2. **Is the C++ reference actually fast with this model?**
   - Answer via: Benchmark bitnet.cpp with same model

3. **Are we hitting any unintended debug/validation paths?**
   - Answer via: Trace-level logging (Phase 2)

4. **Is this reproducible across different machines/CPUs?**
   - Answer via: Test on CI and different hardware

5. **Does this affect all models or just microsoft-bitnet?**
   - Answer via: Test with alternative models

---

**Next Action:** Execute Phase 1 (Confirm Quantization Format) and report findings

**Owner:** TBD
**Due Date:** TBD
**Tracking Issue:** Create GitHub issue for this investigation
