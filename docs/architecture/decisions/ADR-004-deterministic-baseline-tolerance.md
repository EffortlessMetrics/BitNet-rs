# ADR-004: Deterministic Baseline with ±5% Performance Tolerance

**Status**: ACCEPTED
**Date**: 2025-10-15
**Context**: Issue #465 (v0.1.0-mvp Release Polish)
**Related**: AC3/AC4 (CPU Baseline Receipt Generation and Verification)

---

## Context

Issue #465 requires reproducible CPU baseline receipt for v0.1.0-mvp release. The baseline serves as a known-good reference for regression detection and reproducibility verification.

Two reproducibility approaches exist:

### Option 1: Exact Reproducibility (Bit-Identical Outputs)
- **Method**: Require identical outputs across all runs
- **Tolerance**: 0% variance (bit-identical results)
- **Pros**: Perfect determinism, no tolerance needed
- **Cons**: Impossible to achieve with timing measurements

### Option 2: Kernel-Level Determinism (Computational Path + Performance Tolerance)
- **Method**: Require identical kernel IDs, allow ±5% performance variance
- **Tolerance**: ±5% for timing measurements, 0% for kernel IDs
- **Pros**: Practical determinism, accounts for timing variance
- **Cons**: Requires documentation of tolerance expectations

---

## Decision

**Kernel-level determinism with ±5% performance tolerance (Option 2).**

---

## Rationale

### 1. Practical Determinism
- **Kernel IDs Exactness**: Prove identical computational path across runs
- **Timing Variance**: Performance metrics (tok/s, ms_per_token) affected by environmental factors
- **Reproducibility Focus**: Honest compute proven by kernel IDs, not performance exactness
- **Realistic Expectations**: System load, CPU throttling, background tasks cause timing variance

**Deterministic Environment Configuration**:
```bash
export BITNET_DETERMINISTIC=1  # Enable deterministic inference
export BITNET_SEED=42           # Fixed random seed
export RAYON_NUM_THREADS=1      # Single-threaded execution
```

**Expected Reproducibility**:
- ✅ **Kernel IDs**: Must match baseline exactly (0% variance allowed)
- ✅ **Kernel Order**: Execution order must be identical
- ⚠️ **Performance**: tok/s may vary by ±5% across runs (acceptable)

### 2. Environmental Factors Affecting Timing
- **CPU Throttling**: Dynamic frequency scaling affects wall-clock time
- **System Load**: Background processes consume CPU cycles
- **Memory Bandwidth**: DRAM contention from other processes
- **Thermal Throttling**: CPU temperature affects clock speed
- **OS Scheduling**: Thread scheduling variance (even with RAYON_NUM_THREADS=1)

**Example Timing Variance** (same hardware, same code):
```
Run 1: 15.3 tok/s (baseline)
Run 2: 15.1 tok/s (-1.3% variance, acceptable)
Run 3: 15.5 tok/s (+1.3% variance, acceptable)
Run 4: 14.8 tok/s (-3.3% variance, acceptable)
Run 5: 16.0 tok/s (+4.6% variance, acceptable)
Run 6: 13.0 tok/s (-15.0% variance, investigate!)
```

### 3. Conservative Tolerance
- **±5% Tolerance**: Conservative enough to catch real performance regressions
- **±10% Tolerance**: Too permissive, may mask real regressions
- **Zero Tolerance**: Too strict, causes false positives from system load

**Rationale for ±5%**:
- CPU frequency scaling: ±2% typical variance
- OS scheduling variance: ±1% typical variance
- Memory bandwidth contention: ±2% typical variance
- **Total**: ±5% accounts for combined environmental factors

### 4. Validation Focus
- **Primary Validation**: Kernel IDs (computational path)
- **Secondary Validation**: Performance (throughput)
- **Regression Detection**: Performance variance >20% triggers investigation

**Validation Hierarchy**:
1. **Kernel IDs**: Must match exactly (0% variance)
2. **Performance**: ±5% acceptable (environmental factors)
3. **Performance Regression**: >20% triggers pre-tag verification failure

### 5. CI/CD Tolerance
- **Baseline Comparison**: ±5% tolerance for baseline regeneration
- **Pre-Tag Verification**: ±20% tolerance for regression detection
- **Smoke Test**: Zero tolerance for kernel IDs (must match exactly)

**Tolerance Rationale**:
- **±5%**: Normal environmental variance (baseline regeneration)
- **±20%**: Significant regression threshold (pre-tag verification)
- **>20%**: Performance regression requiring investigation (blocks release)

---

## Consequences

### Positive
- ✅ **Practical Determinism**: Kernel IDs provide exact computational path reproducibility
- ✅ **Timing Flexibility**: ±5% tolerance accounts for environmental factors
- ✅ **Conservative Threshold**: Catches real regressions without false positives
- ✅ **Documented Expectations**: Baseline README clarifies tolerance policy
- ✅ **CI/CD Integration**: Pre-tag verification enforces ±20% regression threshold

### Negative
- ⚠️ **Timing Variance**: Performance metrics not bit-identical across runs
- ⚠️ **Tolerance Documentation**: Users must understand ±5% is acceptable
- ⚠️ **Regression Detection**: Performance improvements >5% may be missed (acceptable trade-off)

### Mitigation Strategies
1. **Kernel ID Exactness**: Document that kernel IDs must match exactly (0% variance)
2. **Baseline README**: Clarify ±5% performance tolerance for timing measurements
3. **Reproducibility Guide**: Document deterministic environment configuration
4. **Pre-Tag Verification**: Enforce ±20% regression threshold (blocks release if exceeded)

---

## Alternatives Considered

### Alternative 1: Exact Reproducibility (0% Variance)
**Rejected**: Impossible to achieve with timing measurements due to environmental factors.

**Why Not Feasible**:
- CPU frequency scaling cannot be fully disabled on modern processors
- OS scheduling variance exists even with single-threaded execution
- Memory bandwidth contention from background processes (kernel tasks, daemons)
- Thermal throttling affects clock speed dynamically

### Alternative 2: ±10% Tolerance
**Rejected**: Too permissive, may mask real performance regressions.

**Risk**:
- 10% regression could indicate kernel implementation bug
- Allows significant performance degradation without investigation
- Not conservative enough for production baseline

### Alternative 3: Zero Tolerance (Controlled Environment)
**Rejected**: Requires bare-metal hardware with OS-level isolation (not practical for MVP).

**Requirements** (not feasible for MVP):
- Bare-metal hardware (no virtualization)
- Dedicated CPU cores (no background processes)
- Fixed CPU frequency (no throttling)
- Isolated memory bandwidth (no DRAM contention)
- Temperature-controlled environment (no thermal throttling)

---

## Implementation Details

### Deterministic Configuration

**Environment Variables**:
```bash
export BITNET_DETERMINISTIC=1  # Enable deterministic inference
export BITNET_SEED=42           # Fixed random seed
export RAYON_NUM_THREADS=1      # Single-threaded execution
export BITNET_STRICT_MODE=1     # Fail on mock fallbacks
```

**Benchmark Command**:
```bash
cargo run -p xtask -- benchmark \
  --model models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf \
  --tokens 128 \
  --prompt "The capital of France is"
```

### Reproducibility Verification

**Kernel ID Comparison** (must be identical):
```bash
# Generate two receipts
cargo run -p xtask -- benchmark --model <model.gguf> --tokens 128
cp ci/inference.json run1.json

cargo run -p xtask -- benchmark --model <model.gguf> --tokens 128
cp ci/inference.json run2.json

# Compare kernel lists (should match exactly)
diff <(jq -S '.kernels' run1.json) <(jq -S '.kernels' run2.json)

# Expected: No differences (kernel IDs are deterministic)
```

**Performance Comparison** (±5% tolerance):
```bash
# Calculate variance
TPS1=$(jq -r '.performance.tokens_per_sec' run1.json)
TPS2=$(jq -r '.performance.tokens_per_sec' run2.json)

DIFF=$(echo "scale=2; ($TPS2 - $TPS1) / $TPS1 * 100" | bc)
echo "Performance variance: ${DIFF}%"

# Expected: Within ±5% (acceptable)
```

### Pre-Tag Verification

**Regression Threshold**: ±20%
```bash
# Compare current receipt against baseline
BASELINE_TPS=$(jq -r '.performance.tokens_per_sec' docs/baselines/YYYYMMDD-cpu.json)
CURRENT_TPS=$(jq -r '.performance.tokens_per_sec' ci/inference.json)

DIFF=$(echo "scale=2; ($CURRENT_TPS - $BASELINE_TPS) / $BASELINE_TPS * 100" | bc)

if (( $(echo "$DIFF < -20" | bc -l) )); then
  echo "❌ Performance regression: ${DIFF}% (exceeds -20% threshold)"
  exit 1
elif (( $(echo "$DIFF > 20" | bc -l) )); then
  echo "✅ Performance improvement: ${DIFF}%"
else
  echo "✅ Performance stable: ${DIFF}% (within ±20%)"
fi
```

---

## Baseline Documentation

### Reproducibility Guide (Baseline README)

**File**: `docs/baselines/YYYYMMDD-cpu-README.md`

```markdown
## Reproducibility

To regenerate this baseline:

\`\`\`bash
# Configure deterministic environment
export BITNET_DETERMINISTIC=1 BITNET_SEED=42 RAYON_NUM_THREADS=1 BITNET_STRICT_MODE=1

# Run benchmark
cargo run -p xtask -- benchmark \
  --model models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf \
  --tokens 128 \
  --prompt "The capital of France is"

# Compare kernel lists (should be identical)
diff <(jq -S '.kernels' ci/inference.json) <(jq -S '.kernels' docs/baselines/YYYYMMDD-cpu.json)

# Performance variance: ±5% acceptable (timing-dependent)
# Kernel IDs: Must match exactly (computational path)
\`\`\`

**Tolerance Policy**:
- **Kernel IDs**: 0% variance (must match exactly)
- **Performance**: ±5% variance acceptable (environmental factors)
- **Regression Threshold**: >20% variance triggers investigation
```

---

## CI/CD Integration

### Baseline Regeneration (±5% Tolerance)
```yaml
# .github/workflows/baseline-regeneration.yml (future work)
- name: Regenerate baseline
  run: |
    cargo run -p xtask -- benchmark --model <model.gguf> --tokens 128

    # Compare kernel lists
    diff <(jq -S '.kernels' ci/inference.json) <(jq -S '.kernels' docs/baselines/YYYYMMDD-cpu.json)

    # Check performance variance
    BASELINE_TPS=$(jq -r '.performance.tokens_per_sec' docs/baselines/YYYYMMDD-cpu.json)
    CURRENT_TPS=$(jq -r '.performance.tokens_per_sec' ci/inference.json)
    DIFF=$(echo "scale=2; ($CURRENT_TPS - $BASELINE_TPS) / $BASELINE_TPS * 100" | bc)

    if (( $(echo "abs($DIFF) > 5" | bc -l) )); then
      echo "⚠️ Performance variance: ${DIFF}% (exceeds ±5%)"
    fi
```

### Pre-Tag Verification (±20% Regression Threshold)
```bash
# scripts/pre-tag-verification.sh
if (( $(echo "$DIFF < -20" | bc -l) )); then
  echo "❌ Performance regression: ${DIFF}% (exceeds -20% threshold)"
  exit 1
elif (( $(echo "$DIFF > 20" | bc -l) )); then
  echo "✅ Performance improvement: ${DIFF}%"
else
  echo "✅ Performance stable: ${DIFF}% (within ±20%)"
fi
```

---

## References

- **Issue #465**: CPU Path Followup (v0.1.0-mvp Release Polish)
- **AC3**: Generate Pinned CPU Baseline Receipt
- **AC4**: Verify Baseline Receipt
- **AC11**: Pre-Tag Verification
- **BitNet Paper**: Deterministic inference with fixed seed

---

## Changelog

- **2025-10-15**: Initial decision for v0.1.0-mvp baseline tolerance
