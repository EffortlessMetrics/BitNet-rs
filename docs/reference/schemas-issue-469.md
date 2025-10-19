# Issue #469 Schema Definitions - MVP Sprint Polish

**Document Status:** Schema Reference
**Created:** 2025-10-18
**Issue:** #469
**Targets:** v0.1.0-mvp release
**Schema Version:** 1.0.0

---

## Overview

This document defines the data schemas for Issue #469's acceptance criteria. All schemas are JSON-serializable and follow BitNet.rs receipt conventions.

---

## Schema 1: GGUFLoaderConfig (AC1)

### Purpose
Configuration for GGUF model loader with strict mode support for QK256 tensor validation.

### Schema Definition

```rust
/// GGUF loader configuration
///
/// # JSON Schema
/// {
///   "type": "object",
///   "properties": {
///     "strict_mode": { "type": "boolean" },
///     "tolerance_bytes": { "type": "integer", "minimum": 0 }
///   },
///   "required": ["strict_mode", "tolerance_bytes"]
/// }
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GGUFLoaderConfig {
    /// Strict mode: reject ANY size deviation
    pub strict_mode: bool,

    /// Tolerance bytes for permissive mode (ignored in strict)
    pub tolerance_bytes: usize,
}
```

### JSON Example

```json
{
  "strict_mode": false,
  "tolerance_bytes": 128
}
```

### Validation Rules

1. **strict_mode**
   - Type: Boolean
   - Required: Yes
   - Default: `false`

2. **tolerance_bytes**
   - Type: Unsigned integer
   - Required: Yes
   - Minimum: 0
   - Default: `qk256_tolerance_bytes(131_072)` (≈131 bytes)
   - Ignored when: `strict_mode == true`

### Version History

- **v1.0.0** (AC1): Initial schema with strict mode support

---

## Schema 2: ParityMetadata (AC4)

### Purpose
Cross-validation parity metadata for inference receipts.

### Schema Definition

```rust
/// Parity validation metadata
///
/// # JSON Schema
/// {
///   "type": "object",
///   "properties": {
///     "cpp_available": { "type": "boolean" },
///     "cosine_similarity": { "type": "number", "minimum": 0.0, "maximum": 1.0 },
///     "exact_match_rate": { "type": "number", "minimum": 0.0, "maximum": 1.0 },
///     "status": {
///       "type": "string",
///       "enum": ["ok", "warn", "error", "rust_only"]
///     }
///   },
///   "required": ["cpp_available", "cosine_similarity", "exact_match_rate", "status"]
/// }
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParityMetadata {
    pub cpp_available: bool,
    pub cosine_similarity: f64,
    pub exact_match_rate: f64,
    pub status: String,
}
```

### JSON Example

```json
{
  "cpp_available": true,
  "cosine_similarity": 0.9923,
  "exact_match_rate": 1.0,
  "status": "ok"
}
```

### Validation Rules

1. **cpp_available**
   - Type: Boolean
   - Required: Yes
   - Meaning: Whether C++ reference was available for comparison
   - `true`: Rust vs C++ comparison performed
   - `false`: Rust-only execution (status must be "rust_only")

2. **cosine_similarity**
   - Type: Float64
   - Required: Yes
   - Range: [0.0, 1.0]
   - Meaning: Cosine similarity between Rust and C++ logits
   - 1.0: Perfect match
   - ≥0.99: Acceptable ("ok" status)
   - ≥0.95: Marginal ("warn" status)
   - <0.95: Unacceptable ("error" status)

3. **exact_match_rate**
   - Type: Float64
   - Required: Yes
   - Range: [0.0, 1.0]
   - Meaning: Token-level exact match rate
   - 1.0: All tokens match
   - ≥0.95: Acceptable for production

4. **status**
   - Type: String (enum)
   - Required: Yes
   - Values: `["ok", "warn", "error", "rust_only"]`
   - Validation logic:
     - `"ok"`: cosine_similarity ≥ 0.99 AND exact_match_rate ≥ 0.95
     - `"warn"`: cosine_similarity ≥ 0.95
     - `"error"`: cosine_similarity < 0.95
     - `"rust_only"`: cpp_available == false

### Status Invariants

```rust
// AC4: Status consistency validation
match status {
    "ok" => {
        assert!(cosine_similarity >= 0.99);
        assert!(exact_match_rate >= 0.95);
    },
    "warn" => {
        assert!(cosine_similarity >= 0.95);
        assert!(cosine_similarity < 0.99 || exact_match_rate < 0.95);
    },
    "error" => {
        assert!(cosine_similarity < 0.95);
    },
    "rust_only" => {
        assert!(!cpp_available);
    },
}
```

### Version History

- **v1.0.0** (AC4): Initial schema with parity metadata

---

## Schema 3: InferenceReceipt v1.0.0 Extension (AC4)

### Purpose
Comprehensive inference receipt with optional parity validation metadata.

### Schema Extension

```rust
/// Inference receipt v1.0.0
///
/// # JSON Schema (Extension)
/// {
///   ...existing fields...,
///   "parity": {
///     "type": "object",
///     "$ref": "#/definitions/ParityMetadata"
///   }
/// }
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceReceipt {
    // Existing fields (from base schema)
    pub schema_version: String,
    pub timestamp: String,
    pub compute_path: String,
    pub backend: String,
    pub kernels: Vec<String>,
    pub deterministic: bool,
    pub environment: HashMap<String, String>,
    pub model_info: ModelInfo,
    pub test_results: TestResults,
    pub performance_baseline: PerformanceBaseline,
    pub cross_validation: Option<CrossValidation>,
    pub corrections: Vec<CorrectionRecord>,

    // AC4: New parity field
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parity: Option<ParityMetadata>,
}
```

### JSON Example (with parity)

```json
{
  "schema_version": "1.0.0",
  "timestamp": "2025-10-18T12:34:56Z",
  "compute_path": "real",
  "backend": "cpu",
  "kernels": ["i2s_gemv", "rope_apply", "attention_real"],
  "deterministic": true,
  "environment": {
    "BITNET_DETERMINISTIC": "1",
    "BITNET_SEED": "42",
    "RAYON_NUM_THREADS": "1"
  },
  "model_info": {
    "model_path": "models/model.gguf",
    "quantization_type": "I2_S"
  },
  "test_results": {
    "total_tests": 10,
    "passed": 10,
    "failed": 0
  },
  "performance_baseline": {
    "tokens_generated": 128,
    "total_time_ms": 2500,
    "tokens_per_second": 51.2
  },
  "corrections": [],
  "parity": {
    "cpp_available": true,
    "cosine_similarity": 0.9923,
    "exact_match_rate": 1.0,
    "status": "ok"
  }
}
```

### Validation Rules

1. **parity** field
   - Type: Object (ParityMetadata)
   - Required: No (optional field)
   - Serialization: Omitted if None (`#[serde(skip_serializing_if = "Option::is_none")]`)
   - Validation: If present, MUST pass `ParityMetadata` schema validation

2. **Backward Compatibility**
   - Existing receipts without `parity` field remain valid
   - New receipts MAY include `parity` field
   - Deserializers MUST handle missing `parity` field

### Version History

- **v1.0.0** (AC4): Added optional `parity` field

---

## Schema 4: Tokenizer Interface (AC5)

### Purpose
Tokenizer trait interface with real vocabulary size exposure.

### Rust Trait Definition

```rust
/// Tokenizer trait
///
/// # Interface Schema
/// Methods:
/// - encode(text: &str, add_bos: bool, add_special: bool) -> Result<Vec<u32>>
/// - decode(tokens: &[u32]) -> Result<String>
/// - vocab_size() -> usize  // May include GGUF padding
/// - real_vocab_size() -> usize  // AC5: Real vocab (no padding)
pub trait Tokenizer: Send + Sync {
    fn encode(&self, text: &str, add_bos: bool, add_special: bool) -> Result<Vec<u32>>;
    fn decode(&self, tokens: &[u32]) -> Result<String>;
    fn vocab_size(&self) -> usize;
    fn real_vocab_size(&self) -> usize;
}
```

### Vocabulary Size Semantics

| Implementation | vocab_size() | real_vocab_size() | Difference |
|---------------|--------------|-------------------|------------|
| GgufTokenizer | 32064 (padded) | 32000 (real) | 64 (alignment padding) |
| HfTokenizer | 32000 | 32000 | 0 (no padding) |
| BasicTokenizer | 50257 | 50257 | 0 (no padding) |

### Validation Schema

```json
{
  "tokenizer": {
    "type": "object",
    "properties": {
      "vocab_size": {
        "type": "integer",
        "minimum": 1,
        "description": "Vocabulary size (may include GGUF padding)"
      },
      "real_vocab_size": {
        "type": "integer",
        "minimum": 1,
        "description": "Real vocabulary size (no padding)"
      },
      "padding": {
        "type": "integer",
        "minimum": 0,
        "description": "vocab_size - real_vocab_size"
      }
    },
    "required": ["vocab_size", "real_vocab_size"],
    "invariants": [
      "real_vocab_size <= vocab_size",
      "padding == vocab_size - real_vocab_size"
    ]
  }
}
```

### Parity Assertion Schema

```json
{
  "tokenizer_parity": {
    "rust_real_vocab_size": 32000,
    "rust_padded_vocab_size": 32064,
    "cpp_vocab_size": 32000,
    "parity_check": "rust_real_vocab_size == cpp_vocab_size",
    "status": "ok"
  }
}
```

### Version History

- **v1.0.0** (AC5): Added `real_vocab_size()` method

---

## Schema 5: FFI Build Configuration (AC6)

### Purpose
Unified FFI build configuration for C++ shim compilation.

### Rust Function Signature

```rust
/// Compile C++ shim
///
/// # Parameters Schema
/// {
///   "shim_path": { "type": "path", "must_exist": true },
///   "output_name": { "type": "string", "pattern": "^[a-zA-Z0-9_]+$" },
///   "include_dirs": { "type": "array", "items": { "type": "path" } },
///   "system_include_dirs": { "type": "array", "items": { "type": "path" } }
/// }
pub fn compile_cpp_shim(
    shim_path: &Path,
    output_name: &str,
    include_dirs: &[PathBuf],
    system_include_dirs: &[PathBuf],
) -> Result<(), Box<dyn std::error::Error>>;
```

### Build Configuration Example

```rust
compile_cpp_shim(
    Path::new("csrc/kernels_shim.cc"),
    "bitnet_kernels_shim",
    &[
        PathBuf::from("csrc/"),
        PathBuf::from("../bitnet-common/include/"),
    ],
    &[
        PathBuf::from("/usr/local/cuda/include"),
        PathBuf::from("/usr/local/cuda/targets/x86_64-linux/include"),
    ],
)?;
```

### Compiler Flags Schema

```json
{
  "compiler_flags": {
    "standard": "-std=c++17",
    "optimization": "-O2",
    "position_independent": "-fPIC",
    "include_dirs": ["-Icsrc/", "-I../bitnet-common/include/"],
    "system_include_dirs": ["-isystem/usr/local/cuda/include"],
    "warning_suppressions": ["-Wno-unknown-pragmas", "-Wno-deprecated-declarations"]
  }
}
```

### Validation Rules

1. **shim_path**
   - Type: Path
   - MUST exist
   - MUST have .cc or .cpp extension

2. **output_name**
   - Type: String
   - Pattern: `^[a-zA-Z0-9_]+$` (alphanumeric + underscore)

3. **include_dirs**
   - Type: Vec<PathBuf>
   - Compiled with: `-I{dir}` (show warnings)

4. **system_include_dirs**
   - Type: Vec<PathBuf>
   - Compiled with: `-isystem{dir}` (suppress warnings)

### Version History

- **v1.0.0** (AC6): Unified FFI build configuration

---

## Schema 6: CI Parity Environment (AC7)

### Purpose
Environment configuration for CI parity smoke tests.

### Environment Schema

```yaml
# CI Parity Environment Schema
env:
  BITNET_DISABLE_MINIMAL_LOADER:
    type: string
    required: true
    value: "1"
    description: "Enforce enhanced loader (no minimal fallback)"

  BITNET_DETERMINISTIC:
    type: string
    required: true
    value: "1"
    description: "Enable deterministic inference"

  BITNET_SEED:
    type: string
    required: true
    value: "42"
    description: "Seed for deterministic RNG"

  RAYON_NUM_THREADS:
    type: string
    required: true
    value: "1"
    description: "Single-threaded execution"

  BITNET_STRICT_MODE:
    type: string
    required: false
    value: "1"
    description: "Optional strict loader mode"

  BITNET_CPP_DIR:
    type: string
    required: false
    description: "Path to C++ reference (optional)"
```

### Receipt Validation Schema

```json
{
  "receipt_validation": {
    "parity_status": {
      "type": "string",
      "enum": ["ok", "rust_only"],
      "description": "CI fails if status is 'warn' or 'error'"
    },
    "cosine_similarity": {
      "type": "number",
      "minimum": 0.99,
      "description": "CI gate for C++ available case"
    },
    "i2s_flavor_detected": {
      "type": "string",
      "enum": ["BitNet32F16", "GgmlQk256NoScale", "mixed"],
      "description": "Detected I2_S flavor from receipt"
    }
  }
}
```

### CI Workflow Jobs Schema

```yaml
jobs:
  parity-bitnet32:
    env_required: [BITNET_DISABLE_MINIMAL_LOADER, BITNET_DETERMINISTIC, BITNET_SEED, RAYON_NUM_THREADS]
    model_format: "BitNet32-F16"
    receipt_validation:
      - jq -e '.parity.status == "ok" or .parity.status == "rust_only"'
      - jq -e '.parity.cosine_similarity >= 0.99 or .parity.cpp_available == false'

  parity-qk256:
    env_required: [BITNET_DISABLE_MINIMAL_LOADER, BITNET_DETERMINISTIC, BITNET_SEED, RAYON_NUM_THREADS, BITNET_STRICT_MODE]
    model_format: "QK256"
    receipt_validation:
      - jq -e '.parity.status == "ok" or .parity.status == "rust_only"'
      - jq -e '.quant.i2s_flavor_detected == "GgmlQk256NoScale"'
```

### Version History

- **v1.0.0** (AC7): CI parity environment schema

---

## Schema Compatibility Matrix

| Schema | Version | Backward Compatible | Breaking Changes |
|--------|---------|-------------------|------------------|
| GGUFLoaderConfig | 1.0.0 | N/A (new) | None |
| ParityMetadata | 1.0.0 | N/A (new) | None |
| InferenceReceipt (parity field) | 1.0.0 | Yes (optional field) | None |
| Tokenizer (real_vocab_size) | 1.0.0 | Yes (default impl) | None |
| FFI Build Config | 1.0.0 | N/A (build-time) | None |
| CI Parity Env | 1.0.0 | N/A (CI-only) | None |

---

## Validation Tools

### JSON Schema Validation (Python)

```python
import jsonschema

# ParityMetadata schema
parity_schema = {
    "type": "object",
    "properties": {
        "cpp_available": {"type": "boolean"},
        "cosine_similarity": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "exact_match_rate": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "status": {"type": "string", "enum": ["ok", "warn", "error", "rust_only"]}
    },
    "required": ["cpp_available", "cosine_similarity", "exact_match_rate", "status"]
}

# Validate receipt
jsonschema.validate(receipt_json["parity"], parity_schema)
```

### Rust Schema Validation

```rust
// AC4: Receipt schema validation
let receipt = InferenceReceipt::generate("cpu", kernels)?;
receipt.validate_schema_v1()?;  // Validates all fields including parity

// AC5: Tokenizer parity assertion
validate_tokenizer_parity(&rust_tokenizer, cpp_vocab_size)?;
```

---

## Schema Evolution Guidelines

1. **Additive Changes** (Non-Breaking)
   - Add optional fields with `#[serde(skip_serializing_if = "Option::is_none")]`
   - Add new enum variants (only if exhaustive matching not required)
   - Add default trait methods

2. **Breaking Changes** (Require Major Version Bump)
   - Remove required fields
   - Change field types
   - Change enum variant names
   - Remove enum variants
   - Change validation rules (stricter)

3. **Version Numbering**
   - Schema version follows SemVer
   - `schema_version` field in receipts tracks format version
   - Breaking changes increment major version

---

**Document Control:**
- Review Status: Schema Reference (Ready for Implementation)
- Owner: BitNet.rs Architecture Team
- Issue: #469
- Target: v0.1.0-mvp
