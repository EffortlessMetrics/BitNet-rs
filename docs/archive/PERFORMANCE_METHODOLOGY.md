# Performance Measurement Methodology

This document describes how BitNet.rs measures and reports performance metrics.

## Methods & Environment Box

All performance documentation includes a standardized "Methods & Environment" box at the top:

```
Platform: Linux 6.6.87 WSL2, i7-1185G7 @ 3.00GHz, threads=1
BitNet CLI: v0.1.0 | Rust: 1.89.0 | Python: 3.10 | Transformers: 4.35 | Torch: 2.1.0 (cpu)
Determinism: BITNET_DETERMINISTIC=1 RAYON/OMP/MKL/BLAS=1
Prompts: 3 fixed, max_new_tokens=128, warmup=1, medians of 5 runs
```

This box provides:
- **Platform**: Operating system, CPU, and thread configuration
- **Toolchain**: Versions of all relevant software
- **Determinism**: Environment variables ensuring reproducible results
- **Methodology**: Test parameters and statistical approach

## Measurement Protocol

### 1. Environment Setup

Before any measurement:
```bash
# Source common utilities
source scripts/common.sh

# Set up deterministic environment
setup_deterministic_env

# Print platform information
print_platform_banner
```

This ensures:
- Single-threaded execution (RAYON_NUM_THREADS=1, etc.)
- Fixed random seed (BITNET_SEED=42)
- Deterministic mode (BITNET_DETERMINISTIC=1)

### 2. Test Corpus

We use standardized test inputs:
- **Tokenizer Battery**: `scripts/tokenizer_battery.txt` with diverse text samples
- **Perplexity Corpus**: `crossval/data/ppl_smoke.txt` for NLL evaluation
- **Generation Prompts**: Fixed set of 3 prompts for consistency

### 3. Statistical Rigor

- **Warmup Runs**: 1 run to prime caches
- **Sample Size**: Minimum 5 runs per measurement
- **Aggregation**: Median values (robust to outliers)
- **Variance**: Report standard deviation when >5%

### 4. Dual Format Testing

Both SafeTensors and GGUF formats are tested:
```bash
# Measure both formats
scripts/measure_perf_json.sh --model bitnet_b1_58-3B

# Outputs:
# bench/results/2024-01-15-linux-x64-safetensors.json
# bench/results/2024-01-15-linux-x64-gguf.json
```

### 5. Validation Gates

Before reporting performance, we validate:
1. **Format Parity**: SafeTensors vs GGUF produce equivalent results
2. **Determinism**: Multiple runs produce identical outputs
3. **Correctness**: Cross-validation with reference implementation

## Platform-Specific Notes

### WSL2 (Windows Subsystem for Linux)
- Performance reflects virtualized environment
- I/O operations may have additional overhead
- Memory measurements include WSL2 overhead

### macOS
- Metal backend performance varies by chip generation
- Unified memory affects CPU/GPU transfer metrics

### Linux Native
- Most accurate performance measurements
- Governor should be set to "performance" for consistency

## JSON Output Schema

All measurements produce structured JSON:
```json
{
  "meta": {
    "platform": { ... },
    "model": { ... },
    "timestamp": "2024-01-15T10:30:00Z"
  },
  "performance": {
    "throughput_tps": 42.3,
    "latency_ms": 23.1,
    "memory_mb": 1823
  },
  "validation": {
    "deterministic": true,
    "parity_passed": true,
    "tau_b_median": 0.95
  }
}
```

## Reproducibility Checklist

To reproduce our measurements:

1. ✅ Clone exact commit (git hash in docs)
2. ✅ Use specified Rust version (1.89.0)
3. ✅ Set deterministic environment variables
4. ✅ Use provided test corpus files
5. ✅ Run with single thread
6. ✅ Take median of 5+ runs

## Continuous Validation

Our CI ensures:
- No performance claims without measurement data
- All measurements use deterministic settings
- Format parity tests pass before merging
- Nightly runs with stricter tolerances

## Questions?

For questions about our methodology or to report inconsistencies:
- Open an issue with `[perf]` tag
- Include your platform info: `scripts/common.sh && print_platform_banner`
- Attach JSON output from your measurements
