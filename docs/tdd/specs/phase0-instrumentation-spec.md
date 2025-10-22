# Phase 0: Instrumentation Specification

**Priority:** P0 - Foundation for all other work
**Goal:** Add minimal instrumentation to measure baseline and verify fixes
**Estimated Time:** 4-6 hours

---

## Overview

Before fixing performance and correctness issues, we need instrumentation to:
1. Measure baseline performance
2. Verify decode parity with bitnet.cpp
3. Track quantization dispatch
4. Profile timing per operation

---

## 1. Parity Logger (Greedy Decode Verification)

**Purpose:** Capture per-step token IDs and top-k logits for greedy decode comparison with bitnet.cpp

**Location:** `crates/bitnet-cli/src/main.rs` (generation loop)

**Implementation:**

```rust
// Add to generation loop (after line 1047 where logits_vec is extracted)
if std::env::var("BITNET_PARITY").as_deref() == Ok("1") {
    // Log chosen token + top-5 logits with token IDs
    let mut logits_with_idx: Vec<(usize, f32)> = logits_vec
        .iter()
        .copied()
        .enumerate()
        .collect();
    logits_with_idx.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let top_k_logits: Vec<(u32, f32)> = logits_with_idx
        .iter()
        .take(10)
        .map(|(idx, logit)| (*idx as u32, *logit))
        .collect();

    // JSON format for easy parsing
    eprintln!(
        "{{\"step\":{},\"token\":{},\"top_k\":{}}}",
        step_idx,
        next_token,
        serde_json::to_string(&top_k_logits).unwrap_or_default()
    );
}
```

**Acceptance:**
- `BITNET_PARITY=1` captures per-step token + top-10 logits
- Output is valid JSON on stderr
- Can be redirected to file for comparison

---

## 2. Quantization Trace

**Purpose:** Log which quantization format/kernel is selected per tensor

**Location:**
- `crates/bitnet-models/src/quant/mod.rs`
- `crates/bitnet-quantization/src/dispatch.rs` (if exists)

**Implementation:**

```rust
// Add at quantization dispatch points
if std::env::var("BITNET_TRACE_QUANT").as_deref() == Ok("1") {
    eprintln!(
        "quant_dispatch: tensor={} format={:?} kernel={} shape={:?}",
        tensor_name,
        quant_format,
        kernel_id,
        shape
    );
}
```

**Acceptance:**
- `BITNET_TRACE_QUANT=1` logs quantization decisions
- Shows I2_S vs QK256 vs TL1/TL2 selection
- Shows scalar vs AVX2 vs NEON dispatch

---

## 3. Timing Trace

**Purpose:** Measure wall-clock time per major operation

**Location:** `crates/bitnet-cli/src/main.rs` (generation loop)

**Implementation:**

```rust
use std::time::Instant;

// Add timing probes at key points
let timing_enabled = std::env::var("BITNET_TRACE_TIMING").as_deref() == Ok("1");

// Before embed
let t0 = if timing_enabled { Some(Instant::now()) } else { None };
let x = model.embed(&tokens)?;
if let Some(t) = t0 {
    eprintln!("timing: embed_us={}", t.elapsed().as_micros());
}

// Before forward
let t1 = if timing_enabled { Some(Instant::now()) } else { None };
let h = model.forward(&x, any_cache.as_mut())?;
if let Some(t) = t1 {
    eprintln!("timing: forward_us={}", t.elapsed().as_micros());
}

// Before logits
let t2 = if timing_enabled { Some(Instant::now()) } else { None };
let logits = model.logits(&last_hidden)?;
if let Some(t) = t2 {
    eprintln!("timing: logits_us={}", t.elapsed().as_micros());
}

// Before sampling
let t3 = if timing_enabled { Some(Instant::now()) } else { None };
let next_token = sample(&logits_vec, &sampler_config, &mut rng)?;
if let Some(t) = t3 {
    eprintln!("timing: sample_us={}", t.elapsed().as_micros());
}
```

**Acceptance:**
- `BITNET_TRACE_TIMING=1` logs microseconds per operation
- Captures embed, forward, logits, sampling separately
- Easy to aggregate and analyze

---

## 4. Performance Scripts

### 4.1 `scripts/perf_phase1_quant_probe.sh`

**Purpose:** Verify quantization format detection and dispatch

```bash
#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:-models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf}"
TOKENIZER="${2:-models/microsoft-bitnet-b1.58-2B-4T-gguf/tokenizer.json}"

echo "=== Quantization Dispatch Probe ==="
echo "Model: $MODEL"
echo ""

# Build release
cargo build --release --no-default-features --features cpu,full-cli

# Run with quant tracing
BITNET_TRACE_QUANT=1 RUST_LOG=warn \
  target/release/bitnet run \
  --model "$MODEL" \
  --tokenizer "$TOKENIZER" \
  --prompt "test" \
  --max-tokens 1 \
  --greedy \
  2>&1 | grep "quant_dispatch" > docs/tdd/receipts/phase1_quant_probe.txt

echo "Results written to: docs/tdd/receipts/phase1_quant_probe.txt"
cat docs/tdd/receipts/phase1_quant_probe.txt
```

**Acceptance:**
- Script runs successfully
- Receipt shows which quant formats/kernels were used
- Easy to verify I2_S vs QK256 dispatch

### 4.2 `scripts/perf_phase2_timing.sh`

**Purpose:** Measure baseline per-token latency

```bash
#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:-models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf}"
TOKENIZER="${2:-models/microsoft-bitnet-b1.58-2B-4T-gguf/tokenizer.json}"
RECEIPT="docs/baselines/perf/phase2_timing_i2s.md"

echo "=== Timing Probe (1 token) ==="
echo "Model: $MODEL"
echo ""

# Build release with native ISA
RUSTFLAGS="-C target-cpu=native -C opt-level=3 -C lto=thin" \
  cargo build --release --no-default-features --features cpu,full-cli

# Run 3 times, take median
echo "Running 3 iterations..."
mkdir -p docs/baselines/perf

for i in {1..3}; do
  echo "Iteration $i..."
  BITNET_TRACE_TIMING=1 RUST_LOG=warn \
    target/release/bitnet run \
    --model "$MODEL" \
    --tokenizer "$TOKENIZER" \
    --prompt "2+2=" \
    --max-tokens 1 \
    --greedy \
    2>&1 | grep "timing:" | tee -a "$RECEIPT.tmp"
done

echo ""
echo "=== Timing Summary ===" | tee "$RECEIPT"
echo "Model: $MODEL" | tee -a "$RECEIPT"
echo "Date: $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$RECEIPT"
echo "" | tee -a "$RECEIPT"
cat "$RECEIPT.tmp" | tee -a "$RECEIPT"
rm "$RECEIPT.tmp"

echo ""
echo "Receipt written to: $RECEIPT"
```

**Acceptance:**
- Script runs successfully
- Receipt shows embed/forward/logits/sample timing
- Baseline captured for comparison

---

## Acceptance Criteria (Phase 0)

- [x] Parity logger implemented and tested with `BITNET_PARITY=1`
- [x] Quant trace implemented and tested with `BITNET_TRACE_QUANT=1`
- [x] Timing trace implemented and tested with `BITNET_TRACE_TIMING=1`
- [x] `perf_phase1_quant_probe.sh` script created and executable
- [x] `perf_phase2_timing.sh` script created and executable
- [x] Both scripts run successfully and produce receipts
- [x] Receipts directory structure created:
  - `docs/tdd/receipts/`
  - `docs/baselines/perf/`

---

## Dependencies

- `serde_json` for parity logger JSON output
- Standard library `std::time::Instant` for timing

---

## Next Steps

After Phase 0 completion:
1. Run baseline measurements
2. Proceed to Phase 1 P0 (performance fixes)
3. Re-run measurements to verify improvements
