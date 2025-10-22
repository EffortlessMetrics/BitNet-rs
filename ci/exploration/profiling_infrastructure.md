# BitNet.rs Profiling Infrastructure

**Status**: Exploration document for phase2_flamegraph integration
**Date**: 2025-10-22
**Purpose**: Document existing profiling/benchmarking patterns and recommend approach for flamegraph generation

## Executive Summary

BitNet.rs has a well-established benchmarking infrastructure with multiple layers:

1. **Criterion.rs benchmarks** (crates/*/benches/) - Regression detection via statistical analysis
2. **Receipt-based performance tracking** (xtask benchmark, ci/inference.json) - Production inference measurement
3. **Timing tracing** (BITNET_TRACE_TIMING env var) - Phase 1/2 timing collection into markdown receipts
4. **Manual profiling tools** (flamegraph CLI, perf-record) - Documented but not integrated

**Recommendation**: Create `scripts/phase2_flamegraph.sh` that integrates with existing `perf_phase2_timing.sh` pattern, outputting SVG flamegraphs to `docs/baselines/perf/flamegraphs/` directory.

---

## 1. Current Profiling/Benchmarking Patterns

### 1.1 Criterion.rs Benchmarks (Kernel/Quantization Level)

**Location**: `crates/*/benches/*.rs`

**Examples**:
- `/crates/bitnet-kernels/benches/kernel_benchmarks.rs` - Matmul, quantization regression detection
- `/crates/bitnet-quantization/benches/i2s_dequant.rs` - Quantization dequantization throughput
- `/crates/bitnet-quantization/benches/simd_comparison.rs` - AVX2 vs fallback comparison
- `/benches/avx2_vs_fallback.rs` - Edge case and performance regression tests

**Characteristics**:
```bash
# Run with statistical analysis and HTML reports
cargo bench --bench kernel_benchmarks --features cpu

# Output: target/criterion/report/index.html
# - Statistical significance testing
# - Regression detection (vs baseline)
# - Throughput metrics (GFLOPS, elements/sec)
# - Multiple matrix sizes and configurations
```

**Integration Pattern**:
- Uses `Throughput::Elements()` for automatic GFLOPS/throughput calculation
- `black_box()` to prevent compiler optimization
- `BenchmarkId` for parameterized tests
- Regression detection: automatic baseline comparison

**Limitations**:
- Microbenchmark-level only (individual kernels)
- Cannot see full call stacks or time attribution beyond measured functions
- No wall-clock CPU cycle-level detail

### 1.2 Receipt-Based Performance Receipts (End-to-End)

**Location**: `ci/inference.json` (generated), `docs/baselines/perf/` (documented)

**Command**:
```bash
# Generate production receipt with measured metrics and kernel IDs
cargo run -p xtask -- benchmark --model <model.gguf> --tokens 128

# Verify receipt honesty
cargo run -p xtask -- verify-receipt [--require-gpu-kernels]
```

**Schema (v1.0.0)**:
```json
{
  "schema_version": "1.0.0",
  "timestamp": "<RFC3339>",
  "compute_path": "real",
  "backend": "cpu|cuda",
  "deterministic": true,
  "tokens_requested": 128,
  "tokens_generated": <actual>,
  "tokens_per_second": <measured>,
  "kernels": ["kernel_id_1", "kernel_id_2", ...],
  "environment": {
    "BITNET_VERSION": "<version>",
    "OS": "<os-arch>",
    "RUST_VERSION": "<rustc-version>"
  },
  "model": {"path": "<model-path>"}
}
```

**Key Features**:
- Measures **actual** TPS during real inference
- Records **kernel IDs** for audit trail (i2s_matmul, avx2_quantize, etc.)
- Validates "honest compute" (no false GPU claims)
- Determinism tracking for reproducible baselines

**Documentation**:
- `/docs/how-to/receipt-verification.md` - How to verify receipts
- `/docs/explanation/receipt-validation.md` - Architecture rationale
- `/docs/explanation/receipt-cpu-validation-spec.md` - Technical specification

### 1.3 Timing Trace Collection (Phase 1/2 Measurements)

**Location**: `scripts/perf_phase*.sh`, `docs/baselines/perf/*.md`

**Phase 1 Example** (`scripts/perf_phase1_quant_probe.sh`):
```bash
# Collect quantization dispatch telemetry
BITNET_TRACE_QUANT=1 RUST_LOG=warn \
  target/release/bitnet run \
  --model model.gguf \
  --prompt "test" \
  --max-tokens 1 \
  2>&1 | grep "quant_dispatch"

# Output: docs/tdd/receipts/phase1_quant_probe.txt
```

**Phase 2 Example** (`scripts/perf_phase2_timing.sh`):
```bash
# Run 3 iterations, collect timing breakdown
BITNET_TRACE_TIMING=1 RUST_LOG=warn \
  target/release/bitnet run \
  --model model.gguf \
  --prompt "2+2=" \
  --max-tokens 1 \
  2>&1 | grep "timing:"

# Output in: docs/baselines/perf/phase2_timing_i2s.md
```

**Timing Breakdown** (from phase2_timing_i2s.md):
```
timing: embed_us=26
timing: forward_us=1865375    <- ~95% of total time
timing: logits_us=72092       <- ~4% overhead
timing: sample_us=155         <- negligible
```

**Current Output Format**:
- Plain markdown with timing breakdown
- Human-readable performance analysis
- System configuration snapshot (CPU features, Rust version, build flags)
- Model/tokenizer metadata

**Limitations**:
- No per-layer breakdowns
- No call stack visibility
- Can't attribute forward_us to specific operations (GEMM vs AllReduce, etc.)

### 1.4 Manual Profiling Tools (Documented but Not Integrated)

**From `/docs/troubleshooting/troubleshooting.md` and `/docs/performance-tuning.md`**:

```bash
# CPU profiling with flamegraph (requires sudo or perf permission)
cargo install --locked flamegraph

# Interactive profiling (needs to wrap binary in perf wrapper)
sudo flamegraph -- target/release/bitnet run \
  --model model.gguf \
  --prompt "Hello"
```

**Not Yet Integrated**:
- No automated scripts for flamegraph generation
- No CI/CD pipeline for profiling
- No historical flamegraph storage
- Manual perf-record invocation only

---

## 2. Where to Place Flamegraph SVGs and Documentation

### 2.1 Directory Structure Recommendation

```
docs/baselines/perf/
├── phase2_timing_i2s.md              [existing]
├── BUILD_SUMMARY.md                   [existing]
├── flamegraphs/                        [NEW]
│   ├── README.md
│   ├── phase2_flamegraph_baseline.svg
│   ├── phase2_flamegraph_baseline.md
│   ├── phase2_flamegraph_qk256.svg
│   ├── phase2_flamegraph_qk256.md
│   └── [per-config variants...]
└── [future: phase3_profile/, phase4_simd_analysis/]
```

### 2.2 Flamegraph Metadata Structure

Each SVG should be paired with a markdown file documenting:

```markdown
# Phase 2 Flamegraph - I2S Baseline

**Generated**: 2025-10-22T12:34:56Z
**Model**: microsoft-bitnet-b1.58-2B-4T-gguf (I2S)
**Workload**: 128 tokens, greedy decoding
**Hardware**: AMD Ryzen 9 9950X3D, 16 cores
**Build**: RUSTFLAGS="-C target-cpu=native -C opt-level=3"

## Top Hotspots (by flame height/width)

1. **Forward Pass** (~95% of time)
   - `bitnet_inference::forward` (1,865 ms)
   - Top sub-calls:
     - `matmul_i2s` (85% of forward)
     - `RMSNorm` (10% of forward)
     - `RoPE` (3% of forward)
     - `Attention` (2% of forward)

2. **Logits Computation** (~4% of time)
   - `compute_logits` (72 ms)
   - Top sub-calls:
     - `quantized_projection` (50%)
     - `softmax` (50%)

3. **Miscellaneous** (<1% of time)
   - Embedding lookup
   - Sampling
   - Memory allocation

## Performance Insights

- **Kernel**: `matmul_i2s` is 85% of forward pass
  - Consider SIMD optimization opportunities
  - Blocked by AVX-512 full utilization?

- **Memory**: Forward pass likely memory-bound
  - Recommend bandwidth analysis (perf counters)
  - KV cache prefetching opportunity

- **Latency**: Single-token decode is 1.95 seconds
  - Expected for scalar I2S kernels (~0.5 tok/s)
  - Target: >1 tok/s with SIMD optimization

## Comparison to Baseline

- **vs Phase 1 Timing Receipt**: 
  - forward_us matches (1,865 ms)
  - Flamegraph confirms forward pass dominance
  
- **vs QK256 Variant** (see phase2_flamegraph_qk256.md):
  - QK256 shows different hotspots (scalar dequant overhead)

## How to Generate This Flamegraph

See: `scripts/phase2_flamegraph.sh`

```bash
./scripts/phase2_flamegraph.sh \
  --output docs/baselines/perf/flamegraphs/phase2_flamegraph_baseline.svg
```

## Flamegraph Guide

1. **Click** any stack frame to zoom in
2. **Search** with Ctrl+F to find specific functions
3. **Width** = time spent in that function
4. **Height** = call stack depth
5. **Color** = function type (green=userspace, yellow=kernel, etc.)
```

### 2.3 Git Integration

```bash
# Flamegraph SVGs should be tracked (but be mindful of size)
# Typical SVG size: 2-5 MB per flamegraph

# .gitignore patterns to add (for development):
*.profraw
*.profdata
perf.data
perf.data.old
```

---

## 3. How to Wire Flamegraph Generation Into Existing Scripts

### 3.1 Integration Architecture

**Current Phase 2 Flow**:
```
scripts/perf_phase2_timing.sh
├── Build release with native ISA
├── Run 3 iterations with BITNET_TRACE_TIMING=1
├── Parse "timing: *_us" lines
└── Generate docs/baselines/perf/phase2_timing_i2s.md
```

**Proposed Enhanced Flow**:
```
scripts/phase2_flamegraph.sh (NEW)
├── Pre-flight check (sudo/perf capability)
├── Build release binary with debug symbols (if needed)
├── Run perf record for single inference iteration
│   ├── Model: microsoft-bitnet-b1.58-2B-4T-gguf
│   ├── Prompt: "2+2="
│   ├── Max tokens: 1
│   └── Capture: CPU cycles, memory bandwidth (optional)
├── Process: perf.data → Flamegraph SVG (via cargo-flamegraph)
├── Colorize SVG:
│   ├── Quantization kernels (red)
│   ├── Matmul operations (green)
│   ├── Memory operations (blue)
│   └── System calls (orange)
├── Generate summary markdown (metadata + insights)
└── Output: docs/baselines/perf/flamegraphs/
```

### 3.2 Implementation Sketch

**File**: `scripts/phase2_flamegraph.sh` (proposed)

```bash
#!/usr/bin/env bash
set -euo pipefail

# Configuration
MODEL="${1:-models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf}"
TOKENIZER="${2:-models/microsoft-bitnet-b1.58-2B-4T-gguf/tokenizer.json}"
OUTPUT_DIR="${3:-docs/baselines/perf/flamegraphs}"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Step 1: Pre-flight check
echo "=== Flamegraph Generation Pre-flight Check ==="
if ! command -v flamegraph &> /dev/null; then
    echo "Installing cargo-flamegraph..."
    cargo install flamegraph
fi

# Verify perf permission (need this for perf record)
if ! perf record -o /tmp/test.perf true 2>/dev/null; then
    echo "WARNING: perf record requires elevated privileges"
    echo "Run with: sudo ./scripts/phase2_flamegraph.sh"
    exit 1
fi

# Step 2: Build release (ensures clean build with debug info)
echo "=== Building Release Binary ==="
RUSTFLAGS="-C target-cpu=native -C opt-level=3 -C debug-assertions=off" \
  cargo build --release --no-default-features --features cpu,full-cli

# Step 3: Profile with cargo-flamegraph (wraps perf record + flamegraph)
echo "=== Running Profiling with perf (single iteration) ==="
BITNET_DETERMINISTIC=1 \
BITNET_SEED=42 \
RUST_LOG=warn \
cargo flamegraph \
  --bin bitnet \
  --output "$OUTPUT_DIR/phase2_flamegraph_baseline.svg" \
  -- run \
    --model "$MODEL" \
    --tokenizer "$TOKENIZER" \
    --prompt "2+2=" \
    --max-tokens 1 \
    --greedy \
    --temperature 0.0

echo "SVG generated: $OUTPUT_DIR/phase2_flamegraph_baseline.svg"

# Step 4: Generate metadata markdown
TIMESTAMP=$(date -u +%Y-%m-%dT%H:%M:%SZ)
cat > "$OUTPUT_DIR/phase2_flamegraph_baseline.md" << METADATA
# Phase 2 Flamegraph - I2S Baseline

**Generated**: $TIMESTAMP
**Model**: microsoft-bitnet-b1.58-2B-4T-gguf (I2S)
**Workload**: 1 token, greedy decoding
**Hardware**: \$(lscpu | grep "Model name" | cut -d: -f2 | xargs)

## Flamegraph Location

SVG: \`phase2_flamegraph_baseline.svg\`

## Key Observations

To be filled in by manual analysis of generated flamegraph.

### Top Hotspots
- [Click SVG to analyze]

### Kernel Identification
- Quantization kernels: [to identify]
- MatMul operations: [to identify]
- Memory operations: [to identify]

## Build Configuration

\`\`\`bash
RUSTFLAGS="-C target-cpu=native -C opt-level=3"
cargo build --release --no-default-features --features cpu,full-cli
\`\`\`

## How to Interpret Flamegraph

1. **Width** = time spent (wider = longer)
2. **Height** = call stack depth
3. **Color** = function type (varies by implementation)
4. **Click to zoom** into specific call stacks
5. **Search** (Ctrl+F) for specific functions

## Related Documentation

- Phase 2 Timing: \`phase2_timing_i2s.md\`
- Receipt Validation: \`/docs/how-to/receipt-verification.md\`
- Profiling Guide: \`/docs/performance-tuning.md\`

---
Generated by: \`scripts/phase2_flamegraph.sh\`
METADATA

echo "Metadata generated: $OUTPUT_DIR/phase2_flamegraph_baseline.md"
echo ""
echo "=== Complete ==="
echo "Flamegraph: $OUTPUT_DIR/phase2_flamegraph_baseline.svg"
echo "Metadata:   $OUTPUT_DIR/phase2_flamegraph_baseline.md"
```

### 3.3 Alternative: Lightweight Integration (No sudo required)

For CI environments without elevated privileges:

```bash
#!/usr/bin/env bash
# Lightweight approach: Use sampling profiler instead of perf
# Trade-off: Lower precision, but no privilege escalation needed

cargo install samply  # Or use built-in dhat with --profile bench

# Profile with built-in instrumentation (no perf required)
RUSTFLAGS="-Z sanitizer=thread" \
  cargo run --release \
    --no-default-features --features cpu \
    -- run --model model.gguf --prompt "test" --max-tokens 1 \
    2>&1 | tee /tmp/profile.log

# Post-process log into flamegraph format (custom implementation)
python3 tools/convert_to_flamegraph.py /tmp/profile.log \
  > "$OUTPUT_DIR/phase2_flamegraph_lightweight.svg"
```

---

## 4. Recommended Approach for Phase2_Flamegraph Generation

### 4.1 Phased Rollout

**Phase A (Immediate, Week 1)**:
1. Create `scripts/phase2_flamegraph.sh` template
2. Document setup requirements (cargo-flamegraph, perf, sudo)
3. Manual generation on developer machine
4. Store SVGs + metadata in `docs/baselines/perf/flamegraphs/`

**Phase B (Integration, Week 2-3)**:
1. Wire into CI/CD pipeline (optional job, only on nightly/scheduled)
2. Add flamegraph diff detection (compare against baseline)
3. Automate metadata extraction from SVG
4. Set up flamegraph regression alerts

**Phase C (Advanced, Post-MVP)**:
1. Multi-configuration flamegraphs (I2S, QK256, GPU variants)
2. Per-layer flamegraph breakdown
3. Kernel-specific color coding
4. Historical flamegraph timeline

### 4.2 File Structure

```bash
# Create directory
mkdir -p docs/baselines/perf/flamegraphs/
mkdir -p docs/baselines/perf/flamegraphs/archives/  # Historical

# Create README for flamegraph directory
cat > docs/baselines/perf/flamegraphs/README.md << 'EOF'
# BitNet.rs Flamegraph Baselines

This directory contains CPU flamegraphs for performance profiling.

## Current Baselines

- **phase2_flamegraph_baseline.svg** - I2S quantization, single-token decode
  - Metadata: `phase2_flamegraph_baseline.md`
  - Generated: [timestamp]
  - Hardware: [CPU model]

- **phase2_flamegraph_qk256.svg** - QK256 quantization (if available)
  - Metadata: `phase2_flamegraph_qk256.md`

## How to Regenerate

```bash
./scripts/phase2_flamegraph.sh
```

## Historical Archives

See `archives/` for previous baselines (for regression tracking).

## Interpreting Flamegraphs

1. **Open SVG in browser** - Click to zoom, search with Ctrl+F
2. **Read metadata** - Check `.md` file for insights
3. **Compare stacks** - diff SVGs to spot performance changes

See: `/docs/performance-tuning.md#profiling-and-monitoring`
