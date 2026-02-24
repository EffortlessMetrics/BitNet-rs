# Crossval Crate Exploration - Complete Documentation Index

## Overview

This comprehensive exploration documents the architecture, APIs, and integration patterns of the `bitnet-crossval` crate (~6,500 LOC), which validates Rust BitNet implementation against C++ reference.

## Generated Reports

### 1. **CROSSVAL_EXPLORATION_REPORT.md** (1,177 lines)
Comprehensive technical documentation covering:

- **Module Hierarchy** - Complete dependency graph of all 8 core modules
- **Public API Entry Points** - Signatures for 30+ public functions
- **FFI Integration** - Boundary locations, call chains, build configuration
- **Test Structure** - 13 test modules with coverage analysis
- **Data Flow** - Token sequences, logits pipelines, receipt generation
- **Backend-Specific Patterns** - Feature gates, device detection, determinism
- **Configuration Patterns** - Environment variables, config structs
- **Known Limitations** - C++ wrapper status, blocked tests, coverage gaps
- **Integration with xtask** - CLI entry points and command registration

### 2. **CROSSVAL_QUICK_REFERENCE.md** (300 lines)
Quick lookup guide with:

- **File Locations** - 11 core files with line counts
- **Key Entry Points** - API functions, FFI boundaries, test modules
- **Feature Flags** - Complete feature matrix with dependencies
- **Configuration Patterns** - Environment variables, data structures
- **Command Integration** - xtask commands with examples
- **Known Limitations** - Quick reference to gaps and blockers
- **Key Design Decisions** - Rationale for major architectural choices

## Key Files in Crossval Crate

### Core Modules
```
crossval/src/
├── lib.rs (100 LOC)                    - Root exports, error types
├── token_parity.rs (475 LOC)           - Token validation pre-gate
├── logits_compare.rs (200+ LOC)        - Per-position divergence detection
├── cpp_bindings.rs (300+ LOC)          - Safe C++ FFI wrappers
├── comparison.rs (150+ LOC)            - Validation orchestration
├── validation.rs (200+ LOC)            - Comprehensive validation suite
├── utils.rs (150+ LOC)                 - Comparison & performance utilities
├── fixtures.rs (94 LOC)                - Test fixture management
├── score.rs (200+ LOC)                 - NLL/Perplexity evaluation
├── bitnet_cpp_wrapper.c (64 LOC)       - C wrapper for extern "C" functions
└── build.rs (119 LOC)                  - FFI compilation & library search
```

### Test Modules
```
crossval/tests/
├── smoke.rs (2.9 KB)                   - Environment checks
├── parity_bitnetcpp.rs (34.7 KB)       - Real parity tests (async)
├── parity_receipts.rs (18.7 KB)        - Receipt validation
├── parity.rs (12.8 KB)                 - Deterministic logits comparison
├── per_position_logits.rs (10.5 KB)    - Token-position divergence
├── qk256_crossval.rs (15.8 KB)         - QK256 vs FP32 validation
├── ffi_integration.rs (8.1 KB)         - FFI lifecycle tests
├── framework_validation.rs (17.8 KB)   - Validation suite tests
├── performance_validation.rs (19.5 KB) - Throughput/memory benchmarks
├── iq2s_validation.rs (10.4 KB)        - IQ2_S quantization parity
├── token_equivalence.rs (5.0 KB)       - Token A/B testing
├── cpp_probe.rs (0.4 KB)               - C++ availability detection
└── ms_bitnet_mapping.rs (1.0 KB)       - Model name mapping
```

### Build & Configuration
```
crossval/
├── Cargo.toml                          - Package manifest with 7 feature flags
├── build.rs                            - FFI compilation & library search
├── README.md                           - Basic usage guide
└── docs/
    └── PARITY_IMPLEMENTATION.md        - PR #468 implementation summary
```

## Architecture Highlights

### Module Dependency Graph
```
lib.rs (root)
├── token_parity.rs [always available]
├── logits_compare.rs [always available]
├── utils.rs [always available]
├── validation.rs [always available]
├── score.rs [always available]
├── fixtures.rs [always available]
├── comparison.rs [feature: crossval]
└── cpp_bindings.rs [feature: ffi, crossval]
    └── bitnet_cpp_wrapper.c [feature: ffi]
```

### FFI Boundary
```
Rust (cpp_bindings.rs)
  ↔ C (bitnet_cpp_wrapper.c)
    ↔ C++ (llama.cpp, ggml)
```

### Data Flow
```
Prompt
  ├─→ [Rust Tokenizer] → Vec<u32>
  ├─→ [C++ Tokenizer via FFI] → Vec<i32>
  └─→ validate_token_parity() [pre-gate]
       ├─→ Success: continue
       └─→ Mismatch: print diagnostic + exit(2)

Tokens
  ├─→ [Rust Engine] → Vec<Vec<f32>> (logits)
  ├─→ [C++ Engine via FFI] → Vec<Vec<f32>>
  └─→ compare_per_position_logits()
       ├─→ Cosine similarity: [0.9999, ...]
       ├─→ L2 distance: [0.0001, ...]
       └─→ First divergence: token position (if any)

JSON Receipt
  ├─→ Timestamp, paths
  ├─→ Parity metrics (similarity, match rate)
  ├─→ Kernel IDs (performance attribution)
  └─→ Validation gate results
```

## Public API Highlights

### Token Parity Pre-Gate
Entry point: `token_parity::validate_token_parity()`
- Input: rust_tokens (u32), cpp_tokens (i32), prompt
- Output: Ok() or Err with colored diagnostic
- Purpose: Fail-fast before expensive logits comparison
- Exit code: Caller should exit(2) on mismatch

### Logits Comparison
Entry point: `logits_compare::compare_per_position_logits()`
- Input: per-token logits from Rust & C++
- Output: LogitsDivergence with cosine similarity, L2 distance, max diff
- Purpose: Identify exact position of divergence
- Threshold: 1e-4 (cosine similarity), 1e-4 (absolute difference)

### Validation Suite
Entry point: `validation::ValidationSuite`
- Methods: validate_model_compatibility, validate_token_parity, validate_nll_parity, validate_performance
- Output: ValidationResult with pass/fail + metrics
- Purpose: Comprehensive 4-gate validation

### FFI Bindings
Entry point: `cpp_bindings::CppModel`
- Methods: load(), generate(), model_info(), is_ready()
- Purpose: Safe wrapper around C++ functions
- Fallback: Stub implementation when C++ unavailable

## Feature Flags

| Flag | Enables | Default |
|------|---------|---------|
| `crossval` | Full validation with C++ FFI | No |
| `ffi` | FFI compilation & linking | No |
| `iq2s-ffi` | IQ2_S quantization support | No |
| `cpp-probe` | C++ environment detection | No |
| `integration-tests` | Expensive integration tests | No |
| `cpu` | CPU-specific inference | No |
| `gpu` | GPU/CUDA support | No |

Note: Default is EMPTY - always specify features explicitly.

## Key Entry Points

### CLI Commands (via xtask)
```bash
cargo run -p xtask -- crossval-per-token
cargo run -p xtask -- crossval
cargo run -p xtask -- setup-crossval
```

### Direct API Usage
```rust
use bitnet_crossval::{
    token_parity::validate_token_parity,
    logits_compare::compare_per_position_logits,
    validation::ValidationSuite,
};

// Token validation
validate_token_parity(&rust_tokens, &cpp_tokens, prompt)?;

// Logits comparison
let divergence = compare_per_position_logits(&rs_logits, &cpp_logits);

// Full validation
let suite = ValidationSuite::new(model_path);
let results = suite.run_all()?;
```

## Test Coverage Summary

| Category | Tests | Coverage |
|----------|-------|----------|
| **Token Parity** | 9 | AC1-AC10 validated, 3 edge cases |
| **Logits** | 12 | Single-token, multi-token, divergence points |
| **QK256** | 8 | Kernel validation, FP32 reference, production models |
| **FFI** | 6 | Availability, loading, lifecycle, errors |
| **Performance** | 5 | Throughput, memory, baseline comparison |
| **Validation** | 4 | Model compatibility, tensor mapping, NLL/PPL |
| **Integration** | 5 | Smoke checks, environment preflight |
| **Total** | **49+** | ~152+ test functions across 13 modules |

## Known Limitations & Blockers

### C++ Wrapper Status
- **Current**: Mock implementation (file existence check only)
- **Actual behavior**: Hardcoded dummy results
- **Fix**: Set `BITNET_CPP_DIR` and build with real C++ libraries

### Blocked Tests (Issue Dependencies)
- Issue #254: Shape mismatch in layer norm
- Issue #260: Mock elimination
- Issue #469: Tokenizer parity + FFI hygiene

### Test Coverage Gaps
- No subprocess exit code validation
- No stderr capture test
- No negative token handling test
- No large model stress test

## Build Configuration

### Library Search Priority (build.rs)
1. `$BITNET_CROSSVAL_LIBDIR` (explicit from setup-cpp-auto)
2. `$BITNET_CPP_DIR/build/3rdparty/llama.cpp/src`
3. `$BITNET_CPP_DIR/build/3rdparty/llama.cpp/ggml/src`
4. `$BITNET_CPP_DIR/build/bin`
5. `$BITNET_CPP_DIR/build/lib`
6. `$BITNET_CPP_DIR/lib`
7. `$BITNET_CPP_DIR/build`

### C++ Wrapper FFI Functions
```c
void* bitnet_cpp_create_model(const char* model_path);
int bitnet_cpp_generate(void* model, const char* prompt, 
                        int max_tokens, unsigned int* tokens_out, 
                        int* tokens_count);
void bitnet_cpp_destroy_model(void* model);
```

## External Dependencies

| Category | Libraries |
|----------|-----------|
| **Serialization** | serde, serde_json |
| **Error Handling** | anyhow, thiserror |
| **FFI** | cc, bindgen, bitnet-sys |
| **Inference** | bitnet-inference, bitnet-models |
| **Tokenization** | bitnet-tokenizers |
| **Terminal** | console (colored output) |
| **Utilities** | scopeguard, dirs, humantime |
| **Async** | tokio |
| **Hashing** | sha2, blake3 |
| **Benchmarking** | criterion |

## Quick Start

### Run Token Parity Test
```bash
cargo run -p xtask -- crossval-per-token \
  --model models/model.gguf \
  --tokenizer models/tokenizer.json \
  --prompt "What is 2+2?" \
  --max-tokens 4
```

### Run Full Validation Suite
```bash
BITNET_CPP_DIR=~/.cache/bitnet_cpp \
cargo test -p bitnet-crossval --features crossval
```

### Execute Specific Test
```bash
cargo test -p bitnet-crossval --test parity -- --nocapture
```

## Documentation References

| Document | Lines | Purpose |
|----------|-------|---------|
| CROSSVAL_EXPLORATION_REPORT.md | 1,177 | Comprehensive technical documentation |
| CROSSVAL_QUICK_REFERENCE.md | 300 | Quick lookup guide |
| CROSSVAL_INDEX.md | This file | Navigation & overview |
| crossval/README.md | 200 | Basic usage guide |
| docs/PARITY_IMPLEMENTATION.md | 100 | PR #468 implementation details |

## Related Documentation (Repo)

- `docs/explanation/token-parity-pregate.md` - Token validation specification
- `docs/explanation/token-parity-pregate-spec.md` - Detailed spec with ACs
- `docs/explanation/dual-backend-crossval-spec.md` - Architecture specification
- `docs/explanation/architecture/adr-016-dual-backend-crossval-architecture.md` - ADR
- `docs/explanation/dual-backend-crossval-impact-analysis.md` - Impact analysis
- `docs/tdd/receipts/` - Example parity receipts

## Absolute File Paths

All paths in this exploration are absolute:

- `/home/steven/code/Rust/BitNet-rs/crossval/src/` - Source files
- `/home/steven/code/Rust/BitNet-rs/crossval/tests/` - Test files
- `/home/steven/code/Rust/BitNet-rs/crossval/Cargo.toml` - Configuration
- `/home/steven/code/Rust/BitNet-rs/crossval/build.rs` - Build script
- `/home/steven/code/Rust/BitNet-rs/CROSSVAL_EXPLORATION_REPORT.md` - Main report
- `/home/steven/code/Rust/BitNet-rs/CROSSVAL_QUICK_REFERENCE.md` - Quick ref
- `/home/steven/code/Rust/BitNet-rs/CROSSVAL_INDEX.md` - This index

---

**Exploration Date**: 2025-10-25
**Crate Version**: 0.1.0
**Total Documentation**: 1,500+ lines across 3 reports
**Coverage**: Very thorough (architecture, APIs, FFI, tests, configuration)
