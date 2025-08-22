# Validation Framework

## Overview

BitNet.rs implements comprehensive validation to ensure correctness, performance, and compatibility with llama.cpp. This document outlines our validation approach and requirements.

## JSON Schemas (Frozen)

All CLI JSON output includes `"schema_version": "1"` to track format changes.

### Run Schema

```json
{
  "type": "run",
  "schema_version": "1",
  "model": "path/to/model.gguf",
  "throughput": {
    "tokens_per_second": 42.5,
    "tokens_total": 128,
    "time_total_ms": 3012
  },
  "decoded_tokens": 64,
  "gen_policy": {
    "bos": true,
    "temperature": 0.0,
    "seed": "42"
  },
  "tokenizer": {
    "type": "sentencepiece",
    "origin": "embedded"
  },
  "counts": {
    "n_meta_keys": 42,
    "n_tensors": 128,
    "unmapped": 0
  },
  "memory": {
    "rss_mb": 512,
    "peak_mb": 768
  }
}
```

### Score Schema

```json
{
  "type": "score",
  "schema_version": "1",
  "model": "path/to/model.gguf",
  "dataset": "path/to/dataset.txt",
  "lines": 100,
  "tokens": 2048,
  "mean_nll": 2.831,
  "ppl": 16.96,
  "latency": {
    "total_ms": 5432
  },
  "tokenizer": {
    "type": "sentencepiece",
    "origin": "external"
  },
  "gen_policy": {
    "bos": true,
    "temperature": 0.0,
    "seed": "42"
  },
  "counts": {
    "n_meta_keys": 42,
    "n_tensors": 128,
    "unmapped": 0
  }
}
```

## Validation Gates

### 1. Model Compatibility (Required)
- Zero unmapped tensors (`counts.unmapped == 0`)
- Tokenizer origin validation (embedded vs external)
- Strict shape assertions for all critical tensors

### 2. CPU Accuracy Parity (Required)
- NLL parity with llama.cpp: |Δ mean_nll| ≤ 0.01
- Token ID exact match: ≥95% across test suite
- Deterministic output with fixed seed

### 3. CPU Performance (Required)
- Absolute floor: ≥1.0 tokens/second
- Ratio vs baseline: ≥95% throughput
- Memory usage: ≤103% of baseline RSS

### 4. GPU Validation (When Available)
- Determinism: identical outputs across runs
- Performance ratios: same gates as CPU
- Memory tracking: device + host allocations

## Test Datasets

### Quick Smoke Test (`crossval/data/ppl_smoke.txt`)
20 diverse prompts covering:
- Multiple languages (English, Chinese, Japanese, German, French, Spanish)
- Code snippets (Rust, Python, SQL, C++)
- Mathematical expressions
- Unicode and emoji
- Edge cases (leap dates, NaN comparisons)

### A/B Token Parity (`crossval/prompts.yaml`)
Structured prompts with:
- BOS policy specification
- Max new tokens control
- Long context synthesis (~1200 tokens)
- Mixed content types

## CI Integration

### PR Gates (Fast Path)
1. Model compatibility check
2. Unit tests with CPU features
3. Quick cross-validation smoke test
4. CPU performance ratio check

### Nightly Gates (Full Coverage)
1. All PR gates
2. Full dataset NLL parity
3. Token ID A/B suite (≥95% match)
4. GPU determinism (if available)
5. Baseline updates

## Running Validation

### Quick Validation
```bash
# CPU accuracy parity
scripts/ci-acceptance-gate.sh

# Token ID A/B testing
scripts/ab-suite.sh

# Performance gates
ci/cpu-perf-gate.sh out/run.json
```

### Full Validation
```bash
# Comprehensive validation suite
scripts/comprehensive-validation.sh

# Update baselines
scripts/update-baseline.sh
```

## Baselines

Performance baselines are tracked in `ci/baseline.json`:

```json
{
  "cpu": {
    "tinyllama_q2k_cpu": {
      "tok_s": 42.5,
      "rss_mb": 512
    }
  },
  "gpu": {
    "tinyllama_q2k_gpu": {
      "tok_s": 256.0,
      "rss_mb": 1024
    }
  }
}
```

## Determinism Requirements

For reproducible validation:
- `BITNET_DETERMINISTIC=1`
- `BITNET_SEED=42`
- `RAYON_NUM_THREADS=1`
- `OMP_NUM_THREADS=1`
- `MKL_NUM_THREADS=1`

## Failure Handling

### Model Compatibility Failures
- Check weight mapper synonyms
- Verify tensor shapes match expectations
- Review GGUF metadata completeness

### Accuracy Failures
- Compare tokenization policies (BOS, special tokens)
- Check numerical precision settings
- Verify KV cache management

### Performance Regressions
- Profile with `cargo flamegraph`
- Check SIMD utilization
- Review memory access patterns

## CI Acceptance (PR & Nightly)

**Schema:** All CLI subcommands (`run`, `score`, `tokenize`) MUST include `"schema_version": "1"` in their JSON outputs. For `run` outputs, `.gen_policy.bos` MUST be present.

**PR lane (embedded SPM; TinyLlama Q2\_K):**
- **Mapper gate:** zero unmapped tensors (`xtask gate mapper` JSON `.unmapped_count==0`).
- **Strict run:** `--strict-mapping --strict-tokenizer`; JSON must show `counts.unmapped==0`, tokenizer type `sentencepiece`.
- **Tokenization smoke:** ≥2/3 prompts produce non‑empty IDs.
- **Cross‑validation:**  
  - NLL parity (teacher‑forcing): `|Δ mean_nll| ≤ 1e-2` vs `llama.cpp`.  
  - Token‑ID A/B parity: ≥95% exact match across 20+ prompts (long‑ctx synthesized).
- **CPU Determinism:** T=0 two runs → identical IDs (BITNET_SEED=0, OMP_NUM_THREADS=1).
- **Performance:**  
  - Floor ≥ **1.0 tok/s**.  
  - Ratio ≥ **95%** of `ci/baseline.json.cpu[MODEL_KEY].tok_s`.  
  - RSS ≤ **103%** of `ci/baseline.json.cpu[MODEL_KEY].rss_mb`.

**Nightly lane (external SPM; MS BitNet):**  
Repeat all of the above with `TOKENIZER_PATH` provided. Optionally add GPU gates (IDs identical; perf ratios in `gpu` section of baseline).

**Exit codes:**  
3/4: resource missing; 6: tokenization smoke failed; 9: perf floor/ratio; 10: RSS ratio; 11: determinism fail; 1: general gate failure.

**Determinism env:**  
`BITNET_SEED=0`, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `RAYON_NUM_THREADS=1`, `--temperature 0`. For GPU nightlies, also disable TF32 for deterministic results.

## Examples

Example validation outputs are stored in `crossval/examples/`:
- `run_cpu.json` - CPU inference output
- `score_cpu.json` - CPU perplexity calculation
- `run_gpu.json` - GPU inference output
- `parity_report.json` - Token ID comparison results