# Flamegraph Performance Baselines

This directory contains flamegraph-based performance baselines for BitNet-rs inference operations. Flamegraphs provide detailed CPU time visualization to identify performance bottlenecks and validate optimization efforts.

## 📊 Overview

**What are flamegraphs?**

Flamegraphs are interactive visualizations of CPU profiling data that show:
- **Call stack depth** (Y-axis): How functions call each other
- **CPU time** (X-axis width): Percentage of total CPU time spent in each function
- **Hot paths**: Wide bars indicate performance bottlenecks

**Why we generate them:**

1. **Identify bottlenecks**: Quickly spot functions consuming the most CPU time
2. **Validate optimizations**: Compare before/after flamegraphs to measure impact
3. **Regression detection**: Catch unexpected changes in CPU time distribution
4. **Architecture insights**: Understand real-world execution patterns

**Key characteristic**: X-axis is **alphabetical** (not chronological). Width indicates CPU time percentage, not execution time.

## 🔧 Generation

### Quick Start

Generate a flamegraph for I2_S inference:

```bash
# Install flamegraph tooling (one-time setup)
cargo install flamegraph

# Generate flamegraph with native CPU optimizations
RUSTFLAGS="-C target-cpu=native -C opt-level=3 -C force-frame-pointers=yes" \
cargo flamegraph \
  --bin bitnet-cli \
  --no-default-features \
  --features cpu,full-cli \
  --output flamegraph.svg \
  -- run \
  --model models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf \
  --tokenizer models/microsoft-bitnet-b1.58-2B-4T-gguf/tokenizer.json \
  --prompt "2+2=" \
  --max-tokens 1 \
  --greedy

# Open flamegraph in browser
firefox flamegraph.svg
# Or: chromium flamegraph.svg, open flamegraph.svg (macOS)
```

### Deterministic Profiling

For reproducible flamegraphs suitable for baseline comparison:

```bash
# Enable determinism and single-threaded execution
export BITNET_DETERMINISTIC=1
export BITNET_SEED=42
export RAYON_NUM_THREADS=1

# Generate deterministic flamegraph
RUSTFLAGS="-C target-cpu=native -C opt-level=3 -C force-frame-pointers=yes" \
cargo flamegraph \
  --bin bitnet-cli \
  --release \
  --no-default-features \
  --features cpu,full-cli \
  --freq 999 \
  --output docs/baselines/perf/flamegraph-i2s-$(date +%Y%m%d).svg \
  -- run \
  --model models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf \
  --tokenizer models/microsoft-bitnet-b1.58-2B-4T-gguf/tokenizer.json \
  --prompt "2+2=" \
  --max-tokens 4 \
  --greedy

# Unset environment variables
unset BITNET_DETERMINISTIC BITNET_SEED RAYON_NUM_THREADS
```

### Alternative: Using `perf` Directly

For manual control or CI environments without `cargo-flamegraph`:

```bash
# Build optimized binary with frame pointers
RUSTFLAGS="-C target-cpu=native -C opt-level=3 -C force-frame-pointers=yes" \
  cargo build --release --no-default-features --features cpu,full-cli

# Record performance data
perf record --call-graph=dwarf --freq=999 \
  target/release/bitnet run \
  --model models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf \
  --tokenizer models/microsoft-bitnet-b1.58-2B-4T-gguf/tokenizer.json \
  --prompt "2+2=" \
  --max-tokens 4 \
  --greedy

# Convert to flamegraph (requires flamegraph.pl)
perf script | stackcollapse-perf.pl | flamegraph.pl > flamegraph.svg
```

**Note**: Requires `linux-perf` tools and `flamegraph` scripts from https://github.com/brendangregg/FlameGraph

## 🖥️ Host Fingerprint Template

When recording a new flamegraph baseline, document the system configuration to enable reproducibility and cross-host comparison.

### Template

Create a fingerprint file `flamegraph-<model>-<date>-fingerprint.md`:

```markdown
# Flamegraph Baseline Fingerprint

**Date**: YYYY-MM-DD
**Model**: microsoft-bitnet-b1.58-2B-4T-gguf (I2_S format)
**Prompt**: "2+2="
**Max Tokens**: 4
**Sampling**: Greedy (temperature=0.0)

---

## System Configuration

### CPU

**Model**: [CPU model from /proc/cpuinfo]
```bash
cat /proc/cpuinfo | grep "model name" | head -1
# Example: AMD Ryzen 9 9950X3D 16-Core Processor
```

**Cores**: [Physical cores / Logical threads]
```bash
lscpu | grep -E "^CPU\(s\):|^Core\(s\) per socket:"
# Example: 16 physical / 32 logical
```

**CPU Features** (SIMD):
```bash
cat /proc/cpuinfo | grep flags | head -1
# Key features: avx, avx2, avx512f, avx512dq, fma, bmi1, bmi2
```

**Example**:
- AVX: Yes
- AVX2: Yes
- AVX-512: Yes (F, DQ, IFMA, CD, BW, VL, VBMI, VBMI2, VNNI, BITALG, VP2INTERSECT, BF16)
- FMA: Yes
- BMI1/BMI2: Yes

### Cache

```bash
lscpu | grep cache
```

**Example**:
- L1d cache: 768 KiB
- L1i cache: 512 KiB
- L2 cache: 16 MiB
- L3 cache: 96 MiB

### Software

**Rust Version**:
```bash
rustc --version
# Example: rustc 1.92.0-nightly (4082d6a3f 2025-09-27)
```

**Cargo Version**:
```bash
cargo --version
# Example: cargo 1.92.0-nightly (f2932725b 2025-09-24)
```

**OS Version**:
```bash
uname -a
# Example: Linux SZ-Corsair-RS 6.6.87.2-microsoft-standard-WSL2 #1 SMP PREEMPT_DYNAMIC
```

**Kernel**:
```bash
uname -r
# Example: 6.6.87.2-microsoft-standard-WSL2
```

### Build Configuration

**RUSTFLAGS**:
```
-C target-cpu=native -C opt-level=3 -C force-frame-pointers=yes
```

**Features**:
```
--no-default-features --features cpu,full-cli
```

**Profile**: `release`

**Binary Size**:
```bash
ls -lh target/release/bitnet
# Example: 8.6M
```

### Profiling Configuration

**Tool**: cargo-flamegraph 0.6.x
**Sampling Frequency**: 999 Hz
**Call Graph**: DWARF

**Environment Variables**:
- `BITNET_DETERMINISTIC=1`
- `BITNET_SEED=42`
- `RAYON_NUM_THREADS=1`
- `RUST_LOG=warn`

---

## Git Context

**Commit**: [git commit SHA]
```bash
git rev-parse HEAD
# Example: c150db3d...
```

**Branch**: [git branch name]
```bash
git branch --show-current
# Example: main
```

**Status**: [clean/dirty]
```bash
git status --short
# Example: clean (no uncommitted changes)
```

---

## Flamegraph Artifact

**Filename**: `flamegraph-i2s-YYYYMMDD.svg`
**Size**: [file size]
**Location**: `docs/baselines/perf/flamegraph-i2s-YYYYMMDD.svg`

---

## Notes

- Environment: [WSL2 | Native Linux | macOS | Windows]
- Performance characteristics: [Any notable observations]
- Known issues: [Link to GitHub issues if applicable]
```

### Automated Fingerprint Generation

Use this script to generate a fingerprint automatically:

```bash
#!/usr/bin/env bash
# scripts/generate_flamegraph_fingerprint.sh

DATE=$(date +%Y-%m-%d)
FINGERPRINT="docs/baselines/perf/flamegraph-i2s-$(date +%Y%m%d)-fingerprint.md"

cat > "$FINGERPRINT" <<EOF
# Flamegraph Baseline Fingerprint

**Date**: $DATE
**Model**: microsoft-bitnet-b1.58-2B-4T-gguf (I2_S format)
**Prompt**: "2+2="
**Max Tokens**: 4
**Sampling**: Greedy (temperature=0.0)

---

## System Configuration

### CPU

**Model**: $(cat /proc/cpuinfo | grep "model name" | head -1 | cut -d: -f2 | xargs)
**Cores**: $(lscpu | grep "^CPU(s):" | awk '{print $2}') logical
**CPU Features**: $(cat /proc/cpuinfo | grep flags | head -1 | cut -d: -f2 | xargs)

### Software

**Rust Version**: $(rustc --version)
**Cargo Version**: $(cargo --version)
**OS Version**: $(uname -a)

### Git Context

**Commit**: $(git rev-parse HEAD)
**Branch**: $(git branch --show-current)
**Status**: $(git status --short | wc -l) uncommitted changes

---

Generated: $DATE
EOF

echo "Fingerprint written to: $FINGERPRINT"
```

## 📖 Interpretation Guide

### Understanding Flamegraph Structure

**X-axis (Horizontal)**:
- **Width = CPU time percentage** (NOT execution order)
- Functions are **alphabetically sorted** within each stack level
- Wider bars = more CPU time consumed
- Search for specific functions using interactive search (click on SVG)

**Y-axis (Vertical)**:
- **Height = call stack depth**
- Bottom = entry points (e.g., `main`)
- Top = leaf functions (actual work)
- Hover to see full call path

**Colors**:
- Generally random to differentiate functions
- Some tools use hue to indicate package/crate

### Reading Patterns

**Wide, flat towers**: Hot path with deep call stacks (common in recursive functions)

**Wide, shallow bars**: Expensive leaf functions (e.g., SIMD kernels, memory operations)

**Many thin bars**: Fragmented execution (potential optimization target)

**Off-CPU time**: Not shown in CPU flamegraphs (use off-CPU profiling for I/O, locks)

### Interactive Features

Modern flamegraph SVGs support:
- **Click**: Zoom into a specific function and its callees
- **Search**: Type function name to highlight matches
- **Reset**: Click on bottom bar to zoom out
- **Tooltip**: Hover for full function name and CPU time percentage

## 🔥 Key Functions to Watch

These are the expected performance hotspots in BitNet-rs inference. Monitor their CPU time percentages to detect regressions or validate optimizations.

### I2_S Quantization Hotspots

| Function | Expected CPU % | Description | Optimization Opportunities |
|----------|---------------|-------------|----------------------------|
| `i2s_qk256_dequant` | 15-25% | QK256 dequantization (2-bit → F32) | AVX2/AVX-512 SIMD, nibble LUT, prefetch |
| `i2s_gemv` | 10-20% | Matrix-vector multiply (quantized weights) | FMA tiling, loop unrolling, cache blocking |
| `bitnet_matmul` | 20-35% | General matrix multiply | SIMD parallelization, tile optimization |
| `lm_head` | 5-15% | Final projection layer (vocab logits) | Sparse computation, quantization |
| `rope_apply` | 3-8% | Rotary position embeddings | Precomputation, SIMD sin/cos |
| `attention_softmax` | 5-10% | Softmax normalization | SIMD exp, fast approximations |
| `attention_matmul` | 8-15% | Attention score computation | Flash attention, sparse attention |
| `layernorm_forward` | 2-5% | Layer normalization | Fused kernels, SIMD |

### Forward Pass Components

The forward pass typically dominates CPU time (95%+ as shown in timing baselines):

```
forward_pass (95-98%)
├── transformer_block (per-layer, 80-90%)
│   ├── attention (40-50%)
│   │   ├── qkv_projection (15-20%)
│   │   │   └── i2s_gemv (10-15%)
│   │   ├── attention_matmul (10-15%)
│   │   └── attention_softmax (5-10%)
│   └── ffn (30-40%)
│       ├── gate_up_projection (15-20%)
│       │   └── i2s_gemv (10-15%)
│       └── down_projection (10-15%)
│           └── i2s_gemv (8-12%)
└── residual_add (2-5%)
```

### Supporting Operations

| Function | Expected CPU % | Description |
|----------|---------------|-------------|
| `embedding_lookup` | <0.1% | Token embedding (negligible per timing data) |
| `sample_token` | <0.1% | Logits sampling (negligible per timing data) |
| `memcpy` / `memmove` | 1-3% | Memory operations (should be minimal) |
| `allocator` functions | <1% | Heap allocation (watch for unexpected spikes) |

### Flags for Concern

**Red flags** (indicate potential issues):

- `i2s_qk256_dequant` <5%: Suggests mock computation or missing real kernels
- `memcpy` >10%: Excessive memory copying (optimize buffer management)
- `allocator` >5%: Memory allocation churn (use arena allocators)
- `forward_pass` <80%: Unexpected overhead in scaffolding code
- Any function with `mock` in the name: **Must be 0% in production**

## 📁 Baseline Versions

### Directory Structure

Organize flamegraph baselines by date and configuration:

```
docs/baselines/perf/
├── FLAMEGRAPH_README.md           # This file
├── flamegraph-i2s-20251022.svg    # I2_S baseline (Oct 22, 2025)
├── flamegraph-i2s-20251022-fingerprint.md
├── flamegraph-qk256-20251022.svg  # QK256 baseline
├── flamegraph-qk256-20251022-fingerprint.md
├── archives/
│   ├── 2025-10/
│   │   ├── flamegraph-i2s-20251015.svg
│   │   ├── flamegraph-i2s-20251015-fingerprint.md
│   │   └── hotspots-20251015.md
│   └── 2025-09/
│       └── ...
└── hotspots/
    ├── hotspots-i2s-20251022.md   # Top-3 analysis
    └── hotspots-qk256-20251022.md
```

### Retention Policy

- **Current baselines**: Keep in main `perf/` directory (latest per configuration)
- **Historical baselines**: Archive by month in `archives/YYYY-MM/`
- **Retention**: Keep monthly snapshots for 1 year, quarterly thereafter

### Naming Convention

```
flamegraph-<format>-<YYYYMMDD>.svg
flamegraph-<format>-<YYYYMMDD>-fingerprint.md
hotspots-<format>-<YYYYMMDD>.md
```

**Formats**:
- `i2s`: I2_S (BitNet32-F16) quantization
- `qk256`: QK256 (GGML-style) quantization
- `tl1`: TL1 table lookup (ARM NEON)
- `tl2`: TL2 table lookup (x86 AVX)
- `gpu-cuda`: GPU inference with CUDA

## 📊 Top-3 Hotspots Template

For each flamegraph baseline, document the top 3 performance hotspots using this template.

### Template: `hotspots-<format>-<YYYYMMDD>.md`

```markdown
# Flamegraph Top-3 Hotspots Analysis

**Date**: YYYY-MM-DD
**Model**: microsoft-bitnet-b1.58-2B-4T-gguf
**Format**: I2_S
**Flamegraph**: `flamegraph-i2s-YYYYMMDD.svg`
**Fingerprint**: `flamegraph-i2s-YYYYMMDD-fingerprint.md`

---

## Top-3 CPU Time Consumers

### 1. [Function Name] — X.X% CPU Time

**Call Path**:
```
main
└── inference_engine::run
    └── transformer::forward
        └── attention::qkv_projection
            └── i2s_gemv  ← HOTSPOT
```

**Metrics**:
- **CPU Time**: X.X% of total
- **Self Time**: X.X% (time spent in function itself, excluding callees)
- **Call Count**: ~X,XXX calls (estimated from profiling)

**Description**:
Brief explanation of what this function does and why it consumes CPU time.

**Optimization Opportunities**:
1. **[Optimization 1]**: [Description, expected impact]
2. **[Optimization 2]**: [Description, expected impact]
3. **[Optimization 3]**: [Description, expected impact]

**References**:
- GitHub Issue: [#XXX - Brief title]
- Related PR: [#YYY - Brief title]
- Documentation: [Link to relevant docs]

---

### 2. [Function Name] — X.X% CPU Time

[Same structure as above]

---

### 3. [Function Name] — X.X% CPU Time

[Same structure as above]

---

## Aggregate Analysis

**Total Hotspot Coverage**: XX.X% (sum of top-3 CPU time)

**Remaining CPU Distribution**:
- Supporting operations: X.X%
- Memory operations: X.X%
- Scaffolding/overhead: X.X%
- Other: X.X%

**Optimization Potential**:
If all top-3 hotspots are optimized by 50%, expected overall speedup: ~X.Xx

**Priority**: [High | Medium | Low]
- High: Top-3 cover >60% of CPU time, clear optimization paths exist
- Medium: Top-3 cover 40-60%, some optimization opportunities
- Low: Top-3 cover <40%, fragmented execution

---

## Comparison to Previous Baseline

**Previous Baseline**: `flamegraph-i2s-YYYYMMDD.svg` (YYYY-MM-DD)

**Changes**:
| Function | Previous | Current | Delta | Notes |
|----------|----------|---------|-------|-------|
| [Function 1] | X.X% | X.X% | ±X.X% | [Explanation] |
| [Function 2] | X.X% | X.X% | ±X.X% | [Explanation] |
| [Function 3] | X.X% | X.X% | ±X.X% | [Explanation] |

**Regressions**: [None | List of concerning increases]

**Improvements**: [None | List of successful optimizations]

---

## Action Items

- [ ] [Action 1 - Brief description] (Owner: [name/team], Priority: [H/M/L])
- [ ] [Action 2 - Brief description] (Owner: [name/team], Priority: [H/M/L])
- [ ] [Action 3 - Brief description] (Owner: [name/team], Priority: [H/M/L])

---

**Next Review**: YYYY-MM-DD (monthly cadence)
```

### Example: I2_S Baseline (October 2025)

```markdown
# Flamegraph Top-3 Hotspots Analysis

**Date**: 2025-10-22
**Model**: microsoft-bitnet-b1.58-2B-4T-gguf
**Format**: I2_S
**Flamegraph**: `flamegraph-i2s-20251022.svg`
**Fingerprint**: `flamegraph-i2s-20251022-fingerprint.md`

---

## Top-3 CPU Time Consumers

### 1. `bitnet_matmul` — 28.4% CPU Time

**Call Path**:
```
main
└── inference_engine::run
    └── transformer::forward (95.6% from timing data)
        └── attention::forward
            └── bitnet_matmul  ← HOTSPOT
```

**Metrics**:
- **CPU Time**: 28.4% of total
- **Self Time**: 18.2% (excluding i2s_gemv callees)
- **Call Count**: ~96 calls per token (32 layers × 3 projections)

**Description**:
General-purpose matrix multiplication for quantized weights. Called for Q/K/V projections, gate/up projections, and down projections in each transformer layer.

**Optimization Opportunities**:
1. **SIMD Tiling**: Apply AVX-512 tiling with FMA for 2-3× speedup
2. **Cache Blocking**: Improve L2 cache hit rate with tiled access patterns
3. **Loop Unrolling**: Unroll inner loops for better instruction-level parallelism

**References**:
- GitHub Issue: #XXX - Optimize bitnet_matmul with AVX-512 tiling
- Documentation: docs/gpu-kernel-architecture.md (CPU kernel patterns)

---

### 2. `i2s_qk256_dequant` — 19.7% CPU Time

**Call Path**:
```
main
└── inference_engine::run
    └── transformer::forward
        └── i2s_gemv
            └── i2s_qk256_dequant  ← HOTSPOT
```

**Metrics**:
- **CPU Time**: 19.7% of total
- **Self Time**: 19.7% (leaf function)
- **Call Count**: ~96 calls per token (same as bitnet_matmul)

**Description**:
Dequantizes QK256 2-bit weights to FP32 for matrix operations. Currently uses scalar implementation (~1.2× AVX2 uplift observed).

**Optimization Opportunities**:
1. **Nibble LUT via pshufb**: 2-bit → signed i8 mapping with AVX2 shuffle (~2× speedup)
2. **FMA Tiling**: Process 8-16 rows simultaneously with fused multiply-add (~1.5× speedup)
3. **Prefetch**: Prefetch next code block and input data to hide memory latency (~1.2× speedup)

**Expected Combined Uplift**: ≥3× with all optimizations applied

**References**:
- GitHub Issue: #XXX - QK256 AVX2 fast path optimization
- Documentation: docs/howto/use-qk256-models.md

---

### 3. `lm_head` — 12.8% CPU Time

**Call Path**:
```
main
└── inference_engine::run
    └── transformer::forward
        └── lm_head  ← HOTSPOT
```

**Metrics**:
- **CPU Time**: 12.8% of total (matches 3.7% logits timing + overhead)
- **Self Time**: 12.8% (leaf function)
- **Call Count**: 1 per token (final projection to vocabulary)

**Description**:
Final projection from hidden states to vocabulary logits. Large matrix (hidden_dim × vocab_size, e.g., 2048 × 128000).

**Optimization Opportunities**:
1. **Sparse Computation**: Only compute top-k vocabulary logits for sampling (~5-10× speedup)
2. **Quantized LM Head**: Apply I2_S quantization to lm_head weights (~2× speedup)
3. **Batch Projection**: Reuse computation across multiple tokens in batch mode

**References**:
- GitHub Issue: #XXX - Sparse lm_head computation
- Documentation: docs/performance-benchmarking.md

---

## Aggregate Analysis

**Total Hotspot Coverage**: 60.9% (sum of top-3 CPU time)

**Remaining CPU Distribution**:
- Attention operations: 15.2%
- FFN operations: 10.3%
- Memory operations: 2.1%
- Scaffolding/overhead: 4.5%
- Other: 7.0%

**Optimization Potential**:
If all top-3 hotspots are optimized by 50%, expected overall speedup: ~1.43× (conservative estimate)

**Priority**: **High**
- Top-3 cover 60.9% of CPU time
- Clear optimization paths exist (SIMD, tiling, sparsity)
- Aligned with forward pass bottleneck (95.6% from timing data)

---

## Comparison to Previous Baseline

**Previous Baseline**: None (initial baseline)

**Changes**: N/A (first measurement)

**Regressions**: None

**Improvements**: Baseline established

---

## Action Items

- [ ] Implement AVX-512 tiling for bitnet_matmul (Owner: kernels-team, Priority: High)
- [ ] Apply nibble LUT + FMA tiling to i2s_qk256_dequant (Owner: kernels-team, Priority: High)
- [ ] Prototype sparse lm_head computation (Owner: inference-team, Priority: Medium)
- [ ] Establish monthly flamegraph review cadence (Owner: perf-team, Priority: Medium)

---

**Next Review**: 2025-11-22 (monthly cadence)
```

## 🔄 Workflow Integration

### Baseline Update Cadence

- **Monthly**: Generate new flamegraph baselines on release commits
- **Pre-optimization**: Capture baseline before major optimization work
- **Post-optimization**: Validate optimization impact with comparison
- **Post-regression**: Document hotspot shift if performance regresses

### CI Integration (Future Work)

```yaml
# .github/workflows/performance-tracking.yml
- name: Generate flamegraph baseline
  if: github.event_name == 'schedule' || inputs.update_baselines
  run: |
    cargo install flamegraph
    ./scripts/generate_flamegraph.sh
    ./scripts/generate_flamegraph_fingerprint.sh
    ./scripts/analyze_flamegraph_hotspots.sh
```

### Local Development Workflow

```bash
# 1. Identify hotspot before optimization
cargo flamegraph --bin bitnet-cli --features cpu,full-cli -- run \
  --model model.gguf --prompt "Test" --max-tokens 4 --greedy
mv flamegraph.svg before-optimization.svg

# 2. Implement optimization

# 3. Measure impact
cargo flamegraph --bin bitnet-cli --features cpu,full-cli -- run \
  --model model.gguf --prompt "Test" --max-tokens 4 --greedy
mv flamegraph.svg after-optimization.svg

# 4. Compare visually
firefox before-optimization.svg after-optimization.svg

# 5. Document findings in hotspots analysis
```

## 🔗 Related Documentation

- [Performance Benchmarking](../../performance-benchmarking.md): Comprehensive performance infrastructure
- [Performance Guide](../../performance-guide.md): Performance tuning and optimization
- [Build Summary](BUILD_SUMMARY.md): Current build configuration and performance baseline
- [Phase 2 Timing](phase2_timing_i2s.md): Detailed timing breakdown for I2_S inference
- [QK256 Usage Guide](../../howto/use-qk256-models.md): QK256-specific performance considerations

## 📚 External Resources

- [Flamegraph.pl](https://github.com/brendangregg/FlameGraph): Original flamegraph implementation
- [cargo-flamegraph](https://github.com/flamegraph-rs/flamegraph): Rust-specific tooling
- [Brendan Gregg's Blog](https://www.brendangregg.com/flamegraphs.html): Flamegraph interpretation guide
- [perf Examples](https://www.brendangregg.com/perf.html): Linux perf profiling guide

---

**Last Updated**: 2025-10-22
**Maintainer**: BitNet-rs Performance Team
**Questions**: Open an issue with label `performance` or `documentation`
