# Test Fixtures for Issue #462: CPU Forward Pass with Real Inference

Comprehensive test data and integration fixtures for validating CPU forward pass implementation with real quantized inference.

## Directory Structure

```
tests/fixtures/
├── models/                  # GGUF model files
│   ├── tiny-bitnet.gguf    # Symlink to microsoft-bitnet-b1.58-2B-4T I2S model (1.2GB)
│   └── tiny-bitnet.gguf.sha256  # Model checksum
├── tokenizers/              # Tokenizer files (optional, auto-discovery)
│   └── (auto-discovered from models/ directory)
├── receipts/                # Receipt validation test data (AC3)
│   ├── cpu_valid.json      # ✓ Valid CPU receipt with quantized kernels
│   ├── cpu_no_kernels.json # ✗ CPU backend with empty kernels array
│   ├── cpu_fp32_fallback.json # ✗ CPU with FP32 fallback (no quantization)
│   ├── gpu_cpu_mismatch.json  # ✗ CUDA backend with CPU kernels
│   └── mock_compute.json   # ✗ Mock compute path (not production)
├── tl_lut/                  # TL quantization lookup tables (AC4)
│   ├── tl1_lut.bin         # TL1 lookup table (256 entries, 1KB)
│   ├── tl2_lut.bin         # TL2 lookup table (65536 entries, 256KB)
│   ├── tl_test_config.json # Test configuration metadata
│   └── generate_luts.rs    # LUT generator source code
├── prompts/                 # Test prompts and expected outputs (AC2)
│   └── test_cases.json     # 5 test prompts with validation rules
├── baselines/               # Baseline performance receipts (AC5)
│   └── cpu-baseline-2025.json # Expected CPU baseline receipt structure
└── README.md                # This file
```

## Test Data Coverage

### AC1: Autoregressive Forward Pass (bitnet-inference)
- **Model:** `tests/fixtures/models/tiny-bitnet.gguf` (1.2GB I2S GGUF)
- **Tokenizer:** Auto-discovered from `models/` directory or via `BITNET_GGUF`
- **Tests:** 4 tests validating forward pass, logits, token generation, determinism

### AC2: CLI Inference Command (bitnet-cli)
- **Model:** Same as AC1
- **Prompts:** `tests/fixtures/prompts/test_cases.json` (5 test cases)
- **Tests:** 4 tests validating CLI invocation, output format, determinism, error handling

### AC3: Receipt Validation (xtask)
- **Fixtures:** 5 receipt JSON files in `tests/fixtures/receipts/`
- **Tests:** 6 tests validating receipt schema, compute path, kernel hygiene, backend matching

### AC4: TL LUT Helper (bitnet-kernels)
- **Fixtures:** TL1/TL2 binary LUTs + config in `tests/fixtures/tl_lut/`
- **Tests:** 6 tests validating LUT loading, indexing, dequantization, error handling

### AC5: Baseline Receipt (ci/inference.json)
- **Fixture:** `tests/fixtures/baselines/cpu-baseline-2025.json` (expected structure)
- **Tests:** 2 tests validating baseline receipt generation and persistence

## Model Provisioning

The test fixtures use the existing BitNet 1.58-2B I2S GGUF model from the repository.

### Option 1: Use Existing Model (Recommended)

The fixtures automatically use the existing model via symlink:

```bash
# Model is already symlinked at:
tests/fixtures/models/tiny-bitnet.gguf -> ../../models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf
```

**Model Details:**
- **Size:** 1.2GB (1,187,801,280 bytes)
- **SHA256:** `4221b252fdd5fd25e15847adfeb5ee88886506ba50b8a34548374492884c2162`
- **Quantization:** I2S (2-bit signed)
- **Architecture:** BitNet 1.58-2B
- **Location:** `models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf`

### Option 2: Use Custom Model

Set `BITNET_GGUF` environment variable to override:

```bash
export BITNET_GGUF=/path/to/your/model.gguf
cargo test --no-default-features --features cpu
```

### Option 3: Download Fresh Model

If the existing model is unavailable:

```bash
cargo run -p xtask -- download-model \
  --id microsoft/bitnet-b1.58-2B-4T-gguf \
  --file ggml-model-i2_s.gguf \
  --output models/
```

## Receipt Fixtures

### 1. `cpu_valid.json` - Valid CPU Receipt ✓

**Purpose:** Positive test case for valid CPU inference receipt
**Characteristics:**
- `compute_path: "real"` (production inference)
- `backend: "cpu"` (CPU execution)
- 7 CPU-specific quantized kernels (I2S, TL1, layer norm, attention)
- Throughput metrics included

**Usage:** AC3 tests for valid receipt structure

### 2. `cpu_no_kernels.json` - Empty Kernels ✗

**Purpose:** Negative test case for missing kernel execution
**Characteristics:**
- `kernels: []` (no kernels executed)
- Should fail validation

**Usage:** AC3 tests for kernel hygiene enforcement

### 3. `cpu_fp32_fallback.json` - FP32 Fallback ✗

**Purpose:** Negative test case for FP32 fallback detection
**Characteristics:**
- Contains `matmul_f32`, `dequantize_f32`, `fallback_linear` kernels
- No quantized kernels present
- Should fail "no FP32 staging" requirement

**Usage:** AC3 tests for quantized hot-path enforcement

### 4. `gpu_cpu_mismatch.json` - Backend Mismatch ✗

**Purpose:** Negative test case for backend/kernel mismatch
**Characteristics:**
- `backend: "cuda"` but kernels are CPU (AVX2, vectorized)
- Should fail backend consistency check

**Usage:** AC3 tests for backend/kernel matching

### 5. `mock_compute.json` - Mock Compute Path ✗

**Purpose:** Negative test case for mock inference detection
**Characteristics:**
- `compute_path: "mock"` (not production)
- Should fail "honest compute" validation

**Usage:** AC3 tests for production inference requirement

## TL LUT Fixtures

### Binary Files

**`tl1_lut.bin` (1KB)**
- 256 f32 entries (8-bit lookup)
- Value range: [-1.008, 1.0]
- Symmetric quantization around 0

**`tl2_lut.bin` (256KB)**
- 65536 f32 entries (16-bit lookup)
- Value range: [-1.108, 1.1]
- Asymmetric weighting (high byte 100%, low byte 10%)

### Configuration

**`tl_test_config.json`**
- TL1/TL2 block sizes, element counts, LUT entry counts
- Test cases with input bytes and expected dequantized values
- Validation ranges and descriptions

### Regenerating LUTs

If you need to regenerate the binary LUT files:

```bash
cd tests/fixtures/tl_lut
rustc generate_luts.rs
./generate_luts
```

## Test Prompts

### `test_cases.json`

**5 Test Prompts:**

1. **simple_arithmetic**: "Q: What is 2+2? A:" → expects "4"
2. **simple_greeting**: "Hello, my name is" → expects capitalized name
3. **deterministic_test**: "The capital of France is" → expects "Paris"
4. **short_completion**: "Once upon a time" → expects ≥8 tokens
5. **code_completion**: "def fibonacci(n):" → expects Python keywords

**Deterministic Configuration:**
- `BITNET_DETERMINISTIC=1`
- `BITNET_SEED=42`
- `RAYON_NUM_THREADS=1`
- Expected: Exact token match across runs

## Baseline Receipt

### `cpu-baseline-2025.json`

**Expected Structure:**
- Schema v1.0.0
- `compute_path: "real"`
- 8 CPU quantized kernels (I2S, TL1, attention, layer norm)
- Throughput: ~11 TPS (1.2GB model, CPU inference)
- Deterministic config (seed=42, threads=1)

**Generation (Post-Implementation):**

Once Issue #462 is implemented, regenerate the real baseline:

```bash
BITNET_DETERMINISTIC=1 BITNET_SEED=42 RAYON_NUM_THREADS=1 \
  cargo run -p xtask -- benchmark \
    --model tests/fixtures/models/tiny-bitnet.gguf \
    --tokens 128

# Copy generated receipt to baseline
cp ci/inference.json tests/fixtures/baselines/cpu-baseline-2025.json
```

## Usage in Tests

### Loading Model Path

```rust
use std::path::PathBuf;

fn get_test_model_path() -> PathBuf {
    std::env::var("BITNET_GGUF")
        .ok()
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            let workspace_root = env!("CARGO_MANIFEST_DIR");
            PathBuf::from(workspace_root)
                .join("tests/fixtures/models/tiny-bitnet.gguf")
        })
}
```

### Loading Receipt Fixtures

```rust
use std::fs;
use serde_json::Value;

fn load_receipt_fixture(name: &str) -> Value {
    let path = format!("tests/fixtures/receipts/{}.json", name);
    let content = fs::read_to_string(path).expect("Failed to read receipt fixture");
    serde_json::from_str(&content).expect("Invalid JSON")
}

#[test]
fn test_valid_cpu_receipt() {
    let receipt = load_receipt_fixture("cpu_valid");
    assert_eq!(receipt["compute_path"], "real");
    assert_eq!(receipt["backend"], "cpu");
}
```

### Loading TL LUT Data

```rust
use std::fs;

fn load_tl1_lut() -> Vec<f32> {
    let bytes = fs::read("tests/fixtures/tl_lut/tl1_lut.bin")
        .expect("Failed to read TL1 LUT");

    // Cast bytes to f32 slice
    let (_, data, _) = unsafe { bytes.align_to::<f32>() };
    data.to_vec()
}

#[test]
fn test_tl1_lut_size() {
    let lut = load_tl1_lut();
    assert_eq!(lut.len(), 256);
}
```

### Loading Test Prompts

```rust
use serde_json::Value;
use std::fs;

fn load_test_prompts() -> Value {
    let content = fs::read_to_string("tests/fixtures/prompts/test_cases.json")
        .expect("Failed to read test prompts");
    serde_json::from_str(&content).expect("Invalid JSON")
}

#[test]
fn test_deterministic_inference() {
    let prompts = load_test_prompts();
    let test_case = &prompts["test_cases"][2]; // deterministic_test

    assert_eq!(test_case["name"], "deterministic_test");
    assert_eq!(test_case["temperature"], 0.0);
}
```

## Fixture Validation

### Validation Checklist

- ✅ All JSON files are valid (`jq . < file.json`)
- ✅ Binary files have correct sizes:
  - `tl1_lut.bin`: 1024 bytes (256 × 4)
  - `tl2_lut.bin`: 262144 bytes (65536 × 4)
- ✅ Model symlink resolves to existing GGUF file
- ✅ Model checksum matches SHA256
- ✅ Receipt fixtures cover all test scenarios (positive + negative)
- ✅ Test prompts include deterministic configuration
- ✅ Baseline receipt has expected structure

### Running Validation

```bash
# Validate all JSON fixtures
find tests/fixtures -name "*.json" -exec jq . {} \; > /dev/null && echo "✓ All JSON valid"

# Check binary file sizes
ls -lh tests/fixtures/tl_lut/*.bin

# Verify model symlink
readlink -f tests/fixtures/models/tiny-bitnet.gguf

# Verify model checksum
sha256sum tests/fixtures/models/tiny-bitnet.gguf
```

## Environment Variables

### Test Configuration

- `BITNET_GGUF`: Override model path (default: auto-discover from `tests/fixtures/models/`)
- `BITNET_TOKENIZER`: Override tokenizer path (default: auto-discover from model directory)
- `BITNET_DETERMINISTIC=1`: Enable deterministic inference
- `BITNET_SEED=42`: Set random seed for reproducibility
- `RAYON_NUM_THREADS=1`: Single-threaded execution for determinism

### Example Test Run

```bash
BITNET_DETERMINISTIC=1 BITNET_SEED=42 RAYON_NUM_THREADS=1 \
  cargo test --no-default-features --features cpu \
  --package bitnet-inference \
  --test issue_462_cpu_forward_tests
```

## Fixture Sizes

**Total Fixture Size: ~1.2GB**

| Directory | Size | Files | Description |
|-----------|------|-------|-------------|
| `models/` | 1.2GB | 1 symlink | GGUF model (symlinked) |
| `receipts/` | 5KB | 15 files | Receipt validation JSON |
| `tl_lut/` | 257KB | 4 files | TL LUT binaries + config |
| `prompts/` | 1KB | 1 file | Test prompts JSON |
| `baselines/` | 1KB | 1 file | Baseline receipt JSON |

**Note:** The large size is due to the GGUF model. The model is symlinked, so it doesn't duplicate storage.

## CI/CD Considerations

### Fast CI Execution

For CI environments where downloading the 1.2GB model is impractical:

1. **Cache the model:** Use GitHub Actions cache to persist `models/` directory
2. **Skip model tests:** Use `#[ignore]` attribute for tests requiring the full model
3. **Synthetic fixtures:** Use smaller synthetic GGUF models (future enhancement)

### Current CI Strategy

All tests requiring the model should:
- Check for model availability (`BITNET_GGUF` or fixture symlink)
- Skip gracefully with informative message if model unavailable
- Run successfully in local development with cached model

## Future Enhancements

1. **Minimal Synthetic GGUF**: Create tiny synthetic GGUF (<10MB) for CI
2. **Tokenizer Fixtures**: Add explicit tokenizer files to `tests/fixtures/tokenizers/`
3. **Cross-Validation Data**: Add C++ reference outputs for cross-validation
4. **GPU Receipts**: Add GPU baseline receipts for GPU feature validation
5. **Mixed Precision Data**: Add FP16/BF16 test fixtures for GPU acceleration

## Related Documentation

- **Issue #462:** https://github.com/EffortlessMetrics/BitNet-rs/issues/462
- **Test Specifications:** `specs/issue_462_*_spec.md` (5 files, 3,821 lines)
- **Test Scaffolding:** `tests/issue_462_*_tests.rs` (4 files, 1,296 lines)
- **CLAUDE.md:** `/home/steven/code/Rust/BitNet-rs/CLAUDE.md` (BitNet.rs development guide)

## Questions?

For questions about test fixtures or fixture generation:
1. Review test specifications in `specs/` directory
2. Check existing fixture usage in test files
3. Consult `CLAUDE.md` for BitNet.rs conventions
4. Review validation framework docs: `docs/development/validation-framework.md`
