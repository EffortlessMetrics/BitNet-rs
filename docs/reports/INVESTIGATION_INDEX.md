# LayerNorm Investigation - Documentation Index

## Quick Navigation

Three comprehensive investigation documents have been created:

### 1. **LAYERNORM_INVESTIGATION.md** (25 KB)
**Complete deep-dive technical report**

- **Section 1**: GGUF Tensor Loading & Classification (1.1-1.6)
  - LayerNorm weight classification patterns
  - Quantization handling and forbidding
  - F32/F16 loading paths
  - Statistics validation and RMS calculation
  - Policy-driven rescaling logic

- **Section 2**: LayerNorm Forward Implementation (2.1-2.3)
  - LayerNorm creation with/without bias
  - TransformerBlock forward pass
  - Final LayerNorm in stack

- **Section 3**: Shape Handling (3.1-3.2)
  - Embedding output shapes
  - Step processing and dimension flow

- **Section 4**: Candle RMSNorm Semantics
  - Mathematical formulas
  - Weight application

- **Section 5**: Root Cause Analysis
  - Problem statement
  - Three hypotheses (ruled out vs confirmed)
  - Model file pre-rescaling evidence

- **Section 6-10**: Tests, validation rules, recommendations, file paths

**Best for**: Comprehensive understanding of the entire system

---

### 2. **INVESTIGATION_SUMMARY.md** (6 KB)
**Executive summary with key findings**

- Key Finding #1: RMS ≈ 0.018 is EXPECTED (1/√2560)
- Key Finding #2: All code components are CORRECT
- Key Finding #3: Actual Issue #254 is different

- Code architecture diagrams
- File map with line numbers
- Key code snippets (4 critical sections)
- Why tests are blocked
- Recommendations (immediate/medium/long-term)

**Best for**: Quick overview and understanding the conclusion

---

### 3. **LAYERNORM_CODE_ANALYSIS.md** (14 KB)
**Detailed code-by-code walkthrough**

### Path 1: GGUF Loading & Validation
- 1.1: RMS Computation (lines 31-42)
  - Mathematical verification
- 1.2: Validation Gate (lines 295-318)
  - Analysis of acceptance thresholds
- 1.3: F32 LayerNorm Loading (lines 1530-1607)
  - Standard load path without modification
- 1.4: Rescaling Logic (lines 366-452)
  - Triggers and conditions

### Path 2: RMSNorm Forward Pass
- 2.1: LayerNorm Creation (lines 65-86)
  - RMSNorm instantiation
- 2.2: Attention Block Forward (lines 948-1010)
  - Execution flow with diagnostics
- 2.3: Final RMSNorm (line 1467)
  - Post-processing normalization

### Path 3: Shape Flow Through Inference
- 3.1: Embedding to First Layer (lines 1392-1430)
  - Complete shape evolution

- RMSNorm Mathematics section
- Validation Rules deep-dive
- Summary table of findings

**Best for**: Understanding exact code behavior and mathematical foundations

---

## Quick Facts

| Fact | Location | Evidence |
|------|----------|----------|
| **Gamma RMS = 1/√H is expected** | INVESTIGATION_SUMMARY.md | Validation gate allows 0.01-2.0 |
| **RMS calculation is correct** | LAYERNORM_CODE_ANALYSIS.md Path 1.1 | sqrt(mean(x²)) formula verified |
| **No GGUF modification happens** | LAYERNORM_CODE_ANALYSIS.md Path 1.3 | Raw bytes → tensor with no rescaling (default) |
| **LayerNorm uses RMSNorm (no bias)** | LAYERNORM_CODE_ANALYSIS.md Path 2.1 | Candle `LayerNorm::rms_norm()` |
| **Shape handling is correct** | LAYERNORM_CODE_ANALYSIS.md Path 3 | [B,T,H] flows consistently |
| **Validation gate passes** | LAYERNORM_CODE_ANALYSIS.md Validation Rules | attn_norm min=0.01, RMS≈0.018 passes |
| **Issue #254 is NOT about gamma RMS** | INVESTIGATION_SUMMARY.md | Real blocker is shape/dimension related |

---

## Reading Guide by Goal

### Goal: "Understand the issue in 5 minutes"
1. Read: INVESTIGATION_SUMMARY.md
2. Skip to: "Absolute Conclusion" section

### Goal: "Find the exact code causing problem"
1. Read: INVESTIGATION_INDEX.md (this file)
2. Go to: LAYERNORM_CODE_ANALYSIS.md
3. Jump to: Relevant path (1/2/3)
4. Check: Line numbers for exact locations

### Goal: "Understand the full context"
1. Start: INVESTIGATION_SUMMARY.md
2. Then: LAYERNORM_INVESTIGATION.md (sections 1-4)
3. Reference: LAYERNORM_CODE_ANALYSIS.md for specific implementations

### Goal: "Fix a LayerNorm-related bug"
1. Check: LAYERNORM_CODE_ANALYSIS.md Path 1 (loading)
2. Check: LAYERNORM_CODE_ANALYSIS.md Path 2 (forward)
3. Check: LAYERNORM_CODE_ANALYSIS.md Path 3 (shapes)
4. Verify: Validation Rules section

---

## Key Code Locations

### RMS Calculation
- **File**: `crates/bitnet-models/src/formats/gguf/loader.rs`
- **Lines**: 31-42
- **Function**: `rms_f32()`
- **Formula**: `sqrt(mean(x²))`

### Validation Gate
- **File**: `crates/bitnet-models/src/formats/gguf/loader.rs`
- **Lines**: 295-318
- **Function**: `check_ln_gamma_stats()`
- **Threshold**: [0.5, 2.0] (or [0.01, 2.0] for I2_S)

### GGUF Loading Path
- **File**: `crates/bitnet-models/src/formats/gguf/loader.rs`
- **Lines**: 1530-1607
- **Path**: F32 dtype case in `create_candle_tensor_with_policy()`
- **Key**: Raw bytes → tensor without modification

### RMSNorm Application
- **File**: `crates/bitnet-models/src/transformer.rs`
- **Lines**: 65-86 (creation), 948-1010 (block forward), 1467 (final)
- **Implementation**: Candle `LayerNorm::rms_norm()`
- **Formula**: `y = (x / rms(x)) * gamma[H]`

### Shape Flow
- **File**: `crates/bitnet-models/src/transformer.rs`
- **Lines**: 1392-1430 (forward_full)
- **Flow**: [B,T,H] → [B,1,H] per-step → [B,V] logits

### Validation Rules
- **File**: `crates/bitnet-cli/src/ln_rules.rs`
- **Lines**: 99-110 (I2_S), 76-89 (F16)
- **Key**: attn_norm allows RMS 0.01-2.0

### Tensor Classification
- **File**: `crates/bitnet-models/src/names.rs`
- **Lines**: 29-44
- **Function**: `is_layernorm_weight()`

---

## Mathematical Summary

### RMS Formula
```
RMS(x) = sqrt(mean(x²))

For uniform x[i] = k for all i:
  RMS = sqrt(mean(k²)) = sqrt(k²) = |k|

For BitNet LayerNorm with k ≈ 0.0198:
  RMS ≈ 0.0198 ✓
```

### RMSNorm Forward
```
y = (x / RMS(x)) * gamma[H]

Where:
  RMS(x) = sqrt(mean(x²) + eps)
  gamma is shape [H]
  eps is typically 1e-6
```

### Expected Values
```
Standard LayerNorm: gamma RMS ≈ 1.0
BitNet LayerNorm:   gamma RMS ≈ 1/sqrt(2560) ≈ 0.0198

Ratio: 0.0198 ≈ 1/50.5 ✓
```

---

## Status Summary

| Component | Implementation | Validation | Status |
|-----------|-----------------|------------|--------|
| **Tensor Loading** | `Tensor::from_slice()` | Correct | ✅ PASS |
| **RMS Calculation** | `sqrt(mean(x²))` | Verified | ✅ PASS |
| **Validation Gate** | [0.5,2.0] default | Allows RMS≈0.018 for I2_S | ✅ PASS |
| **RMSNorm Forward** | Candle `LayerNorm::rms_norm()` | Standard implementation | ✅ PASS |
| **Shape Handling** | [B,T,H] flow | Consistent dimensions | ✅ PASS |
| **Rescaling** | Policy-driven (default: none) | Correct triggers | ✅ PASS |

---

## Next Steps

1. **For Implementation Details**: Read LAYERNORM_CODE_ANALYSIS.md
2. **For Understanding the Conclusion**: Read INVESTIGATION_SUMMARY.md
3. **For Full Context**: Read LAYERNORM_INVESTIGATION.md
4. **For Specific Code**: Use file/line references from any document

---

**Generated**: 2025-10-24  
**Investigation**: BitNet-rs LayerNorm Implementation (Issue #254 context)  
**Scope**: GGUF tensor loading, RMSNorm forward pass, validation gates, shape handling

