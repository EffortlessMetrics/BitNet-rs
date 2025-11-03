# BitNet.rs Receipt JSON Files & Infrastructure Analysis

## Executive Summary

Receipt files in BitNet.rs serve as **honest compute evidence** for inference operations. They provide schema-validated records of:
- Actual kernel execution (CPU/GPU)
- Model metadata
- Performance metrics (tokens/sec, latency)
- Determinism configuration
- Environment details

The system includes:
- **2 production receipt directories**: `docs/tdd/receipts/` and `ci/`
- **8 validation gates** in `xtask` with kernel ID hygiene checks
- **Stop sequence matching** with tail-window optimization (O(window_size) decode cost)
- **15 test fixtures** for validation scenarios

---

## 1. Receipt Files Overview

### A. Production Receipt Locations

#### `docs/tdd/receipts/` - TDD & Baseline Receipts
| File | Purpose | Schema | Size |
|------|---------|--------|------|
| `baseline_parity_cpu.json` | Establish CPU baseline (greedy, det) | 1.0.0 | 5.7KB |
| `cpu_positive.json` | Valid CPU inference record | 1.0.0 | 1.5KB |
| `cpu_negative.json` | Invalid example (mock compute) | 1.0.0 | 764B |
| `cpu_positive_example.json` | Example valid receipt | 1.0.0 | 1.4KB |
| `cpu_negative_example.json` | Example invalid receipt | 1.0.0 | 776B |
| `decode_parity.json` | Decode correctness parity test | 1.0.0 | 2.9KB |

#### `ci/` - CI/CD Receipts
| File | Purpose | Schema | Size |
|------|---------|--------|------|
| `inference.json` | Current benchmark receipt | 1.0.0 | 644B |
| `baseline.json` | Performance baseline reference | Custom | 629B |
| `inference_gpu.json` | GPU benchmark receipt | 1.0.0 | 668B |

### B. Test Fixtures - `tests/fixtures/receipts/`
15 fixture files for validation testing:
- `valid-cpu-receipt.json` - Minimal valid CPU
- `valid-gpu-receipt.json` - Minimal valid GPU
- `empty-kernels-receipt.json` - Invalid (empty kernels)
- `cpu_valid.json` - Full CPU example
- `cpu_fp32_fallback.json` - Invalid (fallback detected)
- `cpu_no_kernels.json` - Invalid (no kernels)
- `gpu_cpu_mismatch.json` - Invalid (GPU claimed, CPU kernels)
- `mock_compute.json` - Invalid (compute_path="mock")
- And 7 more validation scenarios

---

## 2. Receipt Schema (v1.0.0)

### Core Structure
```json
{
  "schema_version": "1.0.0",              // Required: "1.0.0" or "1.0"
  "timestamp": "2025-10-23T00:39:09Z",    // ISO 8601
  "compute_path": "real",                  // Required: "real" (not "mock")
  "backend": "cpu",                        // "cpu", "cuda", or other
  "deterministic": true,                   // Whether run was deterministic
  "kernels": [                             // Required: non-empty array
    "i2s_gemv",                            // Kernel IDs (string)
    "rope_apply",
    "attention_real"
  ],
  "tokens_generated": 1,                   // Actual tokens produced
  "tokens_requested": 1,                   // Max tokens requested
  "tokens_per_second": 0.0,                // Measured throughput
  "environment": {
    "BITNET_VERSION": "0.1.0",
    "OS": "linux-x86_64",
    "RUST_VERSION": "rustc 1.92.0-nightly"
  },
  "model": {
    "path": "models/model.gguf"
  },
  "deterministic_configuration": {         // Optional
    "BITNET_DETERMINISTIC": "1",
    "RAYON_NUM_THREADS": "1",
    "temperature": 0.0,
    "seed": 42
  },
  "performance_baseline": {                // Optional
    "warmup_ms": 152803,
    "prefill_ms": 120441,
    "decode_ms": 0,
    "tokens_per_sec": 0.016
  },
  "model_metadata": {                      // Optional (full metadata)
    "architecture": "BitNet",
    "hidden_size": 2560,
    "num_layers": 30,
    "qk256_tensors": 210,
    "quantized_tensors_pct": 63.25
  },
  "corrections": []                        // Array of runtime corrections
}
```

### Optional Extensions
Some receipts include enhanced metadata:
- `parity_metadata` - C++ cross-validation status
- `quantization_validation` - Quantization flavor detection
- `simd_backend` - SIMD capabilities info
- `cpu_benchmarks` - Detailed CPU timing breakdown
- `validation_criteria` - Pass/fail checklist

---

## 3. Schema Validation (8 Gates)

### Implementation: `xtask/src/main.rs` (lines 4363-4505)
**Command**: `cargo run -p xtask -- verify-receipt [--require-gpu-kernels]`

#### Gate 1: Schema Version
- **Requirement**: `schema_version` must be "1.0.0" or "1.0"
- **Enforcement**: Rejects unsupported versions
- **Code**: Lines 4391-4398

#### Gate 2: Compute Path ("Real" Only)
- **Requirement**: `compute_path` must equal "real" (not "mock")
- **Purpose**: Prevent receipts from fake inference
- **Code**: Lines 4401-4408

#### Gate 3: Kernels Array Non-Empty
- **Requirement**: `kernels` array must exist and be non-empty
- **Purpose**: Prove actual compute was performed
- **Code**: Lines 4415-4422

#### Gate 4: Kernel ID String Format
- **Requirement**: All kernel array entries must be strings
- **Code**: Lines 4424-4426

#### Gate 5: Kernel ID Hygiene (3 sub-checks)

**5a. No Empty Strings**
```rust
if kernel_ids.iter().any(|s| s.trim().is_empty()) {
    bail!("kernels[] contains empty kernel ID")
}
```
- **Code**: Lines 4431-4433

**5b. Length ≤ 128 characters**
```rust
if kernel_ids.iter().any(|s| s.len() > 128) {
    bail!("kernel ID longer than 128 characters")
}
```
- **Code**: Lines 4436-4438

**5c. Count ≤ 10,000 entries**
```rust
if kernel_ids.len() > 10_000 {
    bail!("kernels[] contains too many entries (> 10,000)")
}
```
- **Code**: Lines 4441-4443

#### Gate 6: GPU Backend Auto-Enforcement
- **Requirement**: `backend="cuda"` automatically requires GPU kernels
- **Override**: `--require-gpu-kernels` flag forces GPU check regardless
- **Code**: Lines 4411-4413, 4454-4478

#### Gate 7: GPU Kernel Validation (when required)
**GPU Kernel Pattern Matching**:
```regex
^gemm_     // General GEMM
^wmma_     // Warp MMA (Tensor Core)
^cublas_   // cuBLAS wrappers
^cutlass_  // CUTLASS wrappers
^cuda_     // Generic CUDA kernels
^tl1_gpu_  // TL1 GPU quantization
^tl2_gpu_  // TL2 GPU quantization
^i2s_(quantize|dequantize)$  // I2S GPU
```
**Explicit CPU Kernel Exclusion**:
```rust
if id.starts_with("i2s_cpu_") {
    return false;  // Not a GPU kernel
}
```
- **Code**: Lines 4065-4104

#### Gate 8: CPU Backend Quantization Validation
**CPU Quantized Kernel Prefixes**:
```
i2s_*  // I2S 2-bit signed
tl1_*  // TL1 table lookup (4-bit)
tl2_*  // TL2 table lookup (8-bit)
```
**Fallback Kernel Detection** (failures):
```
fp32_*
fallback_*
dequant_*
*_dequant
matmul_f32
```
- **Code**: Lines 4310-4361, 4123-4186

---

## 4. Example Receipts & Validation

### Valid CPU Receipt (Minimal)
```json
{
  "backend": "cpu",
  "kernels": ["i2s_gemv", "rope_apply"],
  "tokens_per_second": 15.2
}
```
**Validation**: ✅ PASS
- Has quantized kernels (i2s_*)
- No fallback patterns
- String kernel IDs

### Invalid CPU Receipt (FP32 Fallback)
```json
{
  "backend": "cpu",
  "kernels": ["dequant_fp32", "matmul_f32"]
}
```
**Validation**: ❌ FAIL
- No quantized kernels (i2s_*, tl1_*, tl2_*)
- Fallback patterns detected
- Error: "CPU backend verification failed: no quantized kernels found"

### Valid GPU Receipt
```json
{
  "backend": "cuda",
  "kernels": ["gemm_fp16", "wmma_matmul"]
}
```
**Validation**: ✅ PASS
- Backend is CUDA
- Has GPU kernels (gemm_*, wmma_*)

### Invalid GPU Receipt (Silent CPU Fallback)
```json
{
  "backend": "cuda",
  "kernels": ["i2s_cpu_quantize"]  // CPU kernel!
}
```
**Validation**: ❌ FAIL
- GPU backend claimed
- Only CPU kernels found
- Error: "GPU kernel verification required (backend is 'cuda') but no GPU kernels found"

---

## 5. Stop Sequence Matching in Inference

### Implementation: `bitnet-inference/src/streaming.rs` (lines 457-496)

**Function**: `GenerationStream::should_stop()`
```rust
fn should_stop(
    token: u32,
    current_tokens: &[u32],
    config: &GenerationConfig,
    tokenizer: &Arc<dyn Tokenizer>,
) -> bool
```

### Evaluation Order (O(1) → O(window_size))

#### 1. Token ID Checks (Fast Path - O(1))
```rust
// Check stop_token_ids (mimics engine.rs:1322)
if !config.stop_token_ids.is_empty() && 
    config.stop_token_ids.contains(&token) {
    return true;
}
```
**Use Cases**:
- LLaMA-3: `<|eot_id|>` token ID 128009
- Custom stop tokens (numeric IDs)
**Performance**: O(1) binary search on sorted array

#### 2. EOS Token Fallback (O(1))
```rust
// Check for EOS token from config, fallback to tokenizer default
let eos_token = config.eos_token_id.or_else(|| tokenizer.eos_token_id());
if let Some(eos) = eos_token && token == eos {
    return true;
}
```
**Fallback Chain**:
1. Config's explicit `eos_token_id`
2. Tokenizer's default EOS token
3. None (skip check)

#### 3. String Stop Sequences (Tail Window Optimization)
```rust
// Tail window optimization: only decode the last N tokens
let window_size = config.stop_string_window.min(current_tokens.len());
let tail_start = current_tokens.len().saturating_sub(window_size);
let tail_tokens = &current_tokens[tail_start..];

if let Ok(current_text) = tokenizer.decode(tail_tokens) {
    for stop_seq in &config.stop_sequences {
        if current_text.ends_with(stop_seq) {
            return true;
        }
    }
}
```

### Configuration in `GenerationConfig`

**Struct Fields** (lines 40-75 of `config.rs`):
```rust
pub struct GenerationConfig {
    pub stop_sequences: Vec<String>,           // Manual stop strings
    pub stop_token_ids: Vec<u32>,              // Manual stop token IDs
    pub stop_string_window: usize,             // Tail window (default: 64)
    pub eos_token_id: Option<u32>,             // Explicit EOS override
    // ... other fields
}
```

**Default Values** (lines 99-119):
```rust
impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            stop_sequences: vec![],
            stop_token_ids: vec![],
            stop_string_window: 64,  // Default: decode only last 64 tokens
            eos_token_id: None,
            // ... other defaults
        }
    }
}
```

**Builder Methods** (lines 176-210):
```rust
pub fn with_stop_sequence(mut self, stop_seq: String) -> Self {
    self.stop_sequences.push(stop_seq);
    self
}

pub fn with_stop_string_window(mut self, window: usize) -> Self {
    self.stop_string_window = window;
    self
}
```

### Stop Semantics (Unified Across All Paths)

The stop matching logic is **identical** for:
- `InferenceEngine::generate()`
- `GenerationStream::generate_stream_internal()`
- Chat & run CLI modes

**Evaluation Order**:
1. **Token IDs** (`stop_token_ids`) — O(1) check, fastest
2. **EOS** (from config or tokenizer) — O(1) fallback
3. **String sequences** (`stop_sequences`) — O(window_size) decode

### Tail Window Optimization

**Problem**: Decoding full history per token = O(n²) for n-token generation
**Solution**: Decode only last N tokens where N = `stop_string_window`
**Default**: 64 bytes (typical stop sequences fit: `</s>`, `\n\nQ:`, etc.)

**Example**:
```
Generated: "Hello world\n\nQ:"  (100 tokens)
With window_size=64:
  - Only decode last 64 tokens
  - Compare suffix against stop_sequences
  - Found: "\n\nQ:" matches stop sequence
  - Cost: O(64) instead of O(100)
```

**Configuration**:
```bash
# Increase window for longer stop sequences
--stop-string-window 256  # For 256-byte stop sequences
```

---

## 6. Receipt Writing & Kernel Recording

### Command: `cargo run -p xtask -- benchmark`

**Function**: `write_inference_receipt()` (lines 4249-4295)

**Inputs**:
- Model path
- Tokens generated (actual count)
- Tokens per second (measured)
- Backend ("cpu" or "cuda")
- Kernels array (from `KernelRecorder`)

**Output**: `ci/inference.json` (atomic write via temp file)

**Receipt Contents**:
```rust
let receipt = serde_json::json!({
    "schema_version": "1.0.0",
    "timestamp": chrono::Utc::now().to_rfc3339(),
    "compute_path": "real",
    "backend": backend,
    "deterministic": true,  // Benchmark always uses greedy
    "tokens_requested": tokens_generated,
    "tokens_generated": tokens_generated,
    "tokens_per_second": tokens_per_second,
    "kernels": kernels,
    "environment": {
        "BITNET_VERSION": env!("CARGO_PKG_VERSION"),
        "OS": format!("{}-{}", std::env::consts::OS, std::env::consts::ARCH),
        "RUST_VERSION": <runtime rustc --version>
    },
    "model": { "path": model.display().to_string() }
});
```

### Kernel Recording Flow

**During Inference**:
1. `KernelRecorder::record_kernel(id)` is called per kernel
2. Kernels collected in `kernels_captured: Vec<String>`
3. After inference completes: pass to `write_inference_receipt()`

**Exit Codes**:
- `0`: Success (receipt written)
- `EXIT_BENCHMARK_FAILED` (17): Benchmark computation failed

---

## 7. File Structure Summary

```
BitNet-rs/
├── docs/tdd/receipts/          # TDD baseline receipts (6 files)
│   ├── baseline_parity_cpu.json
│   ├── cpu_positive.json
│   ├── cpu_negative.json
│   └── ... (3 more)
├── ci/                         # CI receipt artifacts (3 files)
│   ├── inference.json          # Current benchmark receipt
│   ├── baseline.json           # Performance reference
│   └── inference_gpu.json
├── tests/fixtures/receipts/    # Test fixtures (15 files)
│   ├── valid-cpu-receipt.json
│   ├── valid-gpu-receipt.json
│   ├── empty-kernels-receipt.json
│   └── ... (12 more)
├── xtask/tests/                # Receipt validation tests
│   ├── verify_receipt.rs       # Main validation
│   ├── verify_receipt_hardened.rs
│   ├── issue_462_receipt_validation_tests.rs
│   └── fixtures/receipts/      # Integration test fixtures
└── xtask/src/main.rs           # Receipt implementation (lines 4065-4505)
```

---

## 8. Key Takeaways

### Receipt Validation (8 Gates)
1. **Schema**: "1.0.0" or "1.0"
2. **Compute Path**: "real" only
3. **Kernels Array**: Non-empty
4. **String Format**: All strings
5. **Hygiene**: Empty check + length limit (128 chars) + count limit (10K)
6. **GPU Auto-Enforce**: `backend="cuda"` requires GPU kernels
7. **GPU Kernels**: Regex patterns (gemm_*, wmma_*, etc.)
8. **CPU Quantization**: Prefixes (i2s_*, tl1_*, tl2_*) without fallback

### Stop Matching Performance
- **Evaluation Order**: Token IDs (O(1)) → EOS (O(1)) → Strings (O(window_size))
- **Tail Window**: Default 64 bytes, configurable via `--stop-string-window`
- **Semantic Unification**: Same logic across all generation paths

### Stop Token Examples
```bash
# LLaMA-3 (automatic detection via tokenizer)
--prompt-template llama3-chat          # Auto-detects <|eot_id|> = 128009
--stop-id 128009                       # Manual override

# Generic stop sequences
--stop "</s>" --stop "\n\nQ:"          # String-based stops

# Combined (token IDs checked first)
--stop-id 128009 --stop "</s>"         # Both checked in order
```

---

## 9. Testing Infrastructure

### Receipt Validation Tests
Location: `/home/steven/code/Rust/BitNet-rs/xtask/tests/`

**Test Files**:
- `verify_receipt.rs` - Main validation tests (Issue #439)
- `verify_receipt_hardened.rs` - Hardened validation
- `issue_462_receipt_validation_tests.rs` - AC6 gate tests

**Test Status**: ✅ **25/25 passing**
- Schema validation tests
- GPU kernel verification
- CPU quantization checks
- Kernel ID hygiene tests
- Fallback detection tests

### Receipt Fixtures
Location: `/home/steven/code/Rust/BitNet-rs/tests/fixtures/receipts/`

**Coverage**: 15 scenarios
- Valid CPU/GPU minimal receipts
- Invalid examples (empty kernels, mock compute, etc.)
- Edge cases (null backend, unknown backend)
- Fallback detection scenarios

