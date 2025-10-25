# BitNet.rs Serialization and Deserialization Patterns Report

**Generated**: October 25, 2025
**Scope**: Quick exploration of serialization frameworks, patterns, and best practices
**Focus Areas**: Serde usage, JSON output, YAML configs, custom serialization, schema versioning

---

## 1. Serde Derive Patterns

### 1.1 Core Derive Strategy

BitNet.rs uses **Serde 1.0** for all serialization needs with consistent derive patterns:

```rust
// Standard pattern - most common
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TypeName {
    pub field: String,
}

// With explicit serde namespace
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TypeName {
    pub field: String,
}

// With Display trait
#[derive(Clone, Serialize, Deserialize)]
pub struct TypeName {
    field: String,
}

impl std::fmt::Debug for TypeName {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Custom debug implementation
    }
}
```

**Key Finding**: All 91 serializable types found across codebase use `#[derive]` approach.

### 1.2 Optional Field Handling

BitNet.rs uses `skip_serializing_if` to omit optional fields from JSON output:

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_path: Option<String>,
    
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sha256: Option<String>,
}
```

**Pattern**: Applied to 40+ optional fields throughout codebase for clean output.

### 1.3 Field-Level Control

```rust
#[derive(Clone, Serialize, Deserialize)]
pub struct GenerationConfig {
    // Regular serializable fields
    pub max_new_tokens: u32,
    
    // Skip callback functions (not serializable)
    #[serde(skip)]
    pub logits_cb: Option<LogitsCallback>,
    
    // Default values for missing fields
    #[serde(default = "default_k")]
    pub k: f32,
}

fn default_k() -> f32 {
    1.0
}
```

**Use Cases**:
- `#[serde(skip)]` - Omit function pointers, non-serializable types
- `#[serde(default)]` - Provide fallback values
- `#[serde(default = "function_name")]` - Custom default factories

---

## 2. JSON Serialization Patterns

### 2.1 Basic JSON Operations

**Standard pattern** throughout codebase (40+ occurrences):

```rust
use serde_json;

// Serialize to string (compact)
let json = serde_json::to_string(&receipt)?;

// Serialize to pretty JSON (formatted)
let json = serde_json::to_string_pretty(&receipt)?;

// Deserialize from string
let obj: TypeName = serde_json::from_str(&json_string)?;

// Write to file
std::fs::write(path, json)?;
```

### 2.2 JSON Output Examples

#### Receipt Receipts (Production Quality Assurance)

**File**: `/crates/bitnet-inference/src/receipts.rs`

```json
{
  "schema_version": "1.0.0",
  "timestamp": "2025-10-25T12:34:56Z",
  "compute_path": "real",
  "backend": "cpu",
  "kernels": ["i2s_gemv", "rope_apply"],
  "deterministic": false,
  "environment": {
    "BITNET_VERSION": "0.1.0",
    "RUST_VERSION": "1.70.0",
    "OS": "linux-x86_64"
  },
  "model_info": {
    "model_path": "model.gguf",
    "quantization_type": "i2_s",
    "layers": 24
  },
  "test_results": {
    "total_tests": 10,
    "passed": 10,
    "failed": 0
  },
  "performance_baseline": {
    "tokens_per_second": 2.5,
    "first_token_latency_ms": 150
  },
  "parity": {
    "cpp_available": true,
    "cosine_similarity": 0.9923,
    "exact_match_rate": 1.0,
    "status": "ok"
  },
  "corrections": []
}
```

**Schema Version**: `1.0.0` - Enforced in validation:

```rust
pub const RECEIPT_SCHEMA_VERSION: &str = "1.0.0";

pub fn validate_schema(&self) -> Result<()> {
    if self.schema_version != "1.0.0" {
        return Err(anyhow!("Invalid schema version: {}", self.schema_version));
    }
    Ok(())
}
```

#### Validation Results (Structured Reporting)

**File**: `/crossval/src/validation.rs`

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationResult {
    pub gate: String,
    pub passed: bool,
    pub metrics: HashMap<String, serde_json::Value>,
    pub message: String,
}
```

### 2.3 JSON Output Patterns

**Pretty Printing** (human readable):
```rust
let json = serde_json::to_string_pretty(&receipt)?;
std::fs::write("ci/inference.json", json)?;
```

**Compact Output** (for APIs):
```rust
let json = serde_json::to_string(&chunk)?;
println!("{}", json);
```

**Dynamic Structures**:
```rust
pub metrics: HashMap<String, serde_json::Value>

// Insert typed values as JSON
metrics.insert("unmapped_count".to_string(), serde_json::json!(count));
metrics.insert("unmapped_tensors".to_string(), serde_json::json!(unmapped));
```

---

## 3. YAML Configuration Patterns

### 3.1 YAML Parsing

**Dependency**: `serde_yaml_ng` (in Cargo.toml)

**File**: `/crates/bitnet-models/src/correction_policy.rs`

```rust
pub fn from_yaml(yaml: &str) -> Result<Self> {
    serde_yaml_ng::from_str(yaml).map_err(|e| {
        BitNetError::Validation(format!("Failed to parse correction policy YAML: {}", e))
    })
}

pub fn from_json(json: &str) -> Result<Self> {
    serde_json::from_str(json).map_err(|e| {
        BitNetError::Validation(format!("Failed to parse correction policy JSON: {}", e))
    })
}
```

### 3.2 YAML Example: Correction Policy

```yaml
version: 1
models:
  - fingerprint: "sha256-0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
    notes: "Test model with quantized LN weights"
    corrections:
      - type: LN_GAMMA_RESCALE_RMS
        target_rms: 1.0
        clamp: [0.01, 100.0]
      - type: I2S_DEQUANT_OVERRIDE
        tensors: ["q_proj.weight", "k_proj.weight"]
        inv: false
        k: 1.0
```

### 3.3 YAML Loading Pattern

```rust
impl CorrectionPolicy {
    pub fn load_from_file(path: &Path) -> Result<Self> {
        let contents = std::fs::read_to_string(path)?;
        
        // File type detection
        if path.extension().and_then(|s| s.to_str()) == Some("json") {
            Self::from_json(&contents)
        } else {
            Self::from_yaml(&contents)
        }
    }
}
```

---

## 4. Custom Serialization Logic

### 4.1 Enum Tagged Union Pattern

**File**: `/crates/bitnet-models/src/correction_policy.rs`

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]  // Use "type" field as discriminator
pub enum CorrectionAction {
    #[serde(rename = "LN_GAMMA_RESCALE_RMS")]
    LnGammaRescaleRms {
        target_rms: f32,
        clamp: [f32; 2],
    },
    #[serde(rename = "I2S_DEQUANT_OVERRIDE")]
    I2SDequantOverride {
        tensors: Vec<String>,
        #[serde(default)]
        inv: bool,
        #[serde(default = "default_k")]
        k: f32,
    },
}
```

**Serialized Form**:
```json
{
  "type": "LN_GAMMA_RESCALE_RMS",
  "target_rms": 1.0,
  "clamp": [0.01, 100.0]
}
```

### 4.2 Complex Debug Trait (Non-Serializable Fields)

**File**: `/crates/bitnet-inference/src/config.rs`

```rust
impl std::fmt::Debug for GenerationConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GenerationConfig")
            .field("max_new_tokens", &self.max_new_tokens)
            .field("temperature", &self.temperature)
            // Skip logits_cb - it's a function pointer
            .field("logits_cb", &self.logits_cb.is_some())
            .finish()
    }
}
```

### 4.3 Serde Value HashMap Pattern

**Dynamic field-to-JSON mapping**:

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationResult {
    pub gate: String,
    pub passed: bool,
    pub metrics: HashMap<String, serde_json::Value>,  // Flexible structure
}

// Usage:
let mut result = ValidationResult { /* ... */ };
result.metrics.insert("count".to_string(), json!(42));
result.metrics.insert("items".to_string(), json!(vec!["a", "b"]));
```

---

## 5. Schema Versioning Patterns

### 5.1 Version Constants

**File**: `/crates/bitnet-inference/src/receipts.rs`

```rust
/// Schema version for receipt format
pub const RECEIPT_SCHEMA_VERSION: &str = "1.0.0";

/// Alias for schema version (for consistency)
pub const RECEIPT_SCHEMA: &str = RECEIPT_SCHEMA_VERSION;
```

**File**: `/crates/bitnet-models/src/correction_policy.rs`

```rust
/// Version of the correction policy schema
pub const POLICY_VERSION: u32 = 1;

impl CorrectionPolicy {
    pub fn validate(&self) -> Result<()> {
        if self.version != POLICY_VERSION {
            return Err(BitNetError::Validation(format!(
                "Unsupported policy version: expected {}, got {}",
                POLICY_VERSION, self.version
            )));
        }
        Ok(())
    }
}
```

### 5.2 Version in Struct

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorrectionPolicy {
    /// Schema version
    pub version: u32,
    /// Model-specific corrections
    pub models: Vec<ModelCorrection>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceReceipt {
    /// Schema version (always "1.0.0")
    pub schema_version: String,
    // ... rest of fields
}
```

### 5.3 Version Validation

```rust
pub fn validate_schema(&self) -> Result<()> {
    if self.schema_version != "1.0.0" {
        return Err(anyhow!(
            "Invalid schema version: {} (expected '1.0.0')",
            self.schema_version
        ));
    }
    Ok(())
}
```

---

## 6. Post-Deserialization Validation

### 6.1 Validation After Load

**File**: `/crates/bitnet-models/src/correction_policy.rs`

```rust
pub fn load_from_file(path: &Path) -> Result<Self> {
    let contents = std::fs::read_to_string(path)?;
    let policy = if is_json {
        Self::from_json(&contents)?
    } else {
        Self::from_yaml(&contents)?
    };
    
    // Validate structure after deserialization
    policy.validate()?;
    
    Ok(policy)
}

pub fn validate(&self) -> Result<()> {
    // Version check
    if self.version != POLICY_VERSION {
        return Err(/* ... */);
    }
    
    // Duplicate fingerprint check
    let mut seen = HashSet::new();
    for model in &self.models {
        if !seen.insert(&model.fingerprint) {
            return Err(BitNetError::Validation(
                format!("Duplicate fingerprint: {}", model.fingerprint)
            ));
        }
        
        // Fingerprint format validation
        if !model.fingerprint.starts_with("sha256-") {
            return Err(/* ... */);
        }
        
        let hash_part = &model.fingerprint[7..];
        if hash_part.len() != 64 || !hash_part.chars().all(|c| c.is_ascii_hexdigit()) {
            return Err(/* ... */);
        }
        
        // Action validation
        for action in &model.corrections {
            match action {
                CorrectionAction::LnGammaRescaleRms { target_rms, clamp } => {
                    if !target_rms.is_finite() || *target_rms <= 0.0 {
                        return Err(/* ... */);
                    }
                    if clamp[0] <= 0.0 || clamp[1] <= clamp[0] {
                        return Err(/* ... */);
                    }
                }
                CorrectionAction::I2SDequantOverride { tensors, k, .. } => {
                    if tensors.is_empty() {
                        return Err(/* ... */);
                    }
                    if !k.is_finite() || *k <= 0.0 {
                        return Err(/* ... */);
                    }
                }
            }
        }
    }
    
    Ok(())
}
```

### 6.2 Receipt Validation

**File**: `/crates/bitnet-inference/src/receipts.rs`

```rust
pub fn validate(&self) -> Result<()> {
    // Schema validation
    self.validate_schema()?;
    
    // Compute path validation (AC9 contract)
    self.validate_compute_path()?;
    
    // Kernel ID hygiene (8 validation gates)
    self.validate_kernel_ids()?;
    
    // Test result validation
    if self.test_results.failed > 0 {
        return Err(anyhow!("Failed tests detected: {}", self.test_results.failed));
    }
    
    // Accuracy tests
    if let Some(ref accuracy) = self.test_results.accuracy_tests {
        if let Some(ref i2s) = accuracy.i2s_accuracy {
            if !i2s.passed {
                return Err(anyhow!(
                    "I2S accuracy test failed: MSE {} > tolerance {}",
                    i2s.mse, i2s.tolerance
                ));
            }
        }
    }
    
    // Determinism validation
    if self.deterministic {
        if let Some(ref det) = self.test_results.determinism_tests {
            if !det.identical_sequences {
                return Err(anyhow!("Determinism test failed"));
            }
        }
    }
    
    Ok(())
}
```

**Kernel ID Validation Gates**:
- Kernel array must be non-empty
- No empty strings
- No whitespace-only strings
- Length <= 128 characters per ID
- Total count <= 10,000
- No "mock" kernels (case-insensitive)

---

## 7. Output Formatting Patterns

### 7.1 Pretty Printing Strategy

**Console Output**:
```rust
// For testing/debugging
let json = serde_json::to_string_pretty(&receipt)?;
println!("{}", json);

// To file (human readable)
receipt.save(Path::new("ci/inference.json"))?;  // Uses to_string_pretty internally
```

**Implementation**:
```rust
pub fn save(&self, path: &Path) -> Result<()> {
    let json = serde_json::to_string_pretty(self)?;
    std::fs::write(path, json)?;
    Ok(())
}
```

### 7.2 Report Format Abstraction

**File**: `/tests/common/reporting/formats/json.rs`

```rust
pub struct JsonReporter {
    pretty_print: bool,
}

impl JsonReporter {
    pub fn new() -> Self {
        Self { pretty_print: true }
    }
    
    pub fn new_compact() -> Self {
        Self { pretty_print: false }
    }
}

#[async_trait]
impl TestReporter for JsonReporter {
    async fn generate_report(
        &self,
        results: &[TestSuiteResult],
        output_path: &Path,
    ) -> Result<ReportResult, ReportError> {
        let json_content = if self.pretty_print {
            serde_json::to_string_pretty(&report)?
        } else {
            serde_json::to_string(&report)?
        };
        
        fs::write(output_path, &json_content).await?;
        Ok(ReportResult { /* ... */ })
    }
}
```

### 7.3 Multiple Output Formats

**File**: `/tests/common/reporting/reporter.rs`

```rust
pub enum ReporterType {
    Html(HtmlReporter),
    Json(JsonReporter),
    Junit(JunitReporter),
    Markdown(MarkdownReporter),
}

impl TestReporter for ReporterType {
    async fn generate_report(
        &self,
        results: &[TestSuiteResult],
        output_path: &Path,
    ) -> Result<ReportResult, ReportError> {
        match self {
            ReporterType::Html(r) => r.generate_report(results, output_path).await,
            ReporterType::Json(r) => r.generate_report(results, output_path).await,
            ReporterType::Junit(r) => r.generate_report(results, output_path).await,
            ReporterType::Markdown(r) => r.generate_report(results, output_path).await,
        }
    }
}
```

---

## 8. How to Add New Serializable Types

### 8.1 Step-by-Step Guide

**Step 1**: Define the struct with derive

```rust
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NewType {
    pub required_field: String,
    
    #[serde(skip_serializing_if = "Option::is_none")]
    pub optional_field: Option<usize>,
    
    #[serde(skip)]
    pub function_field: Option<Box<dyn Fn()>>,
}
```

**Step 2**: Add validation (if needed)

```rust
impl NewType {
    pub fn validate(&self) -> anyhow::Result<()> {
        if self.required_field.is_empty() {
            return Err(anyhow!("required_field cannot be empty"));
        }
        if let Some(opt) = self.optional_field {
            if opt > 1000 {
                return Err(anyhow!("optional_field must be <= 1000"));
            }
        }
        Ok(())
    }
}
```

**Step 3**: Implement serialization methods

```rust
impl NewType {
    pub fn to_json(&self) -> anyhow::Result<String> {
        self.validate()?;
        Ok(serde_json::to_string_pretty(self)?)
    }
    
    pub fn from_json(json: &str) -> anyhow::Result<Self> {
        let obj: Self = serde_json::from_str(json)?;
        obj.validate()?;
        Ok(obj)
    }
    
    pub fn from_yaml(yaml: &str) -> anyhow::Result<Self> {
        let obj: Self = serde_yaml_ng::from_str(yaml)?;
        obj.validate()?;
        Ok(obj)
    }
}
```

**Step 4**: Add tests

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_serialization() {
        let obj = NewType {
            required_field: "test".to_string(),
            optional_field: Some(42),
            function_field: None,
        };
        
        let json = obj.to_json().unwrap();
        assert!(json.contains("\"required_field\""));
        assert!(json.contains("\"optional_field\""));
        assert!(!json.contains("\"function_field\"")); // Skipped
    }
    
    #[test]
    fn test_deserialization() {
        let json = r#"{"required_field": "test", "optional_field": 42}"#;
        let obj = NewType::from_json(json).unwrap();
        assert_eq!(obj.required_field, "test");
        assert_eq!(obj.optional_field, Some(42));
    }
}
```

### 8.2 Pattern Checklist

- [ ] Add `#[derive(Debug, Clone, Serialize, Deserialize)]`
- [ ] Mark optional fields with `#[serde(skip_serializing_if = "Option::is_none")]`
- [ ] Skip non-serializable fields with `#[serde(skip)]`
- [ ] Use `#[serde(default)]` or `#[serde(default = "fn")]` for defaults
- [ ] Implement `validate()` method
- [ ] Add `to_json()` and `from_json()` helper methods
- [ ] Test serialization roundtrip
- [ ] Document schema version (if versioned)

---

## 9. Validation Strategies

### 9.1 Pre-Serialization Validation

```rust
pub fn save(&self, path: &Path) -> Result<()> {
    // Validate before serialization
    self.validate()?;
    
    let json = serde_json::to_string_pretty(self)?;
    std::fs::write(path, json)?;
    Ok(())
}
```

### 9.2 Post-Deserialization Validation

```rust
pub fn load_from_file(path: &Path) -> Result<Self> {
    let contents = std::fs::read_to_string(path)?;
    
    // Deserialize first
    let obj: Self = serde_json::from_str(&contents)?;
    
    // Validate after deserialization
    obj.validate()?;
    
    Ok(obj)
}
```

### 9.3 Comprehensive Validation Gates

**Receipt Validation Example** (8 gates):

```rust
// Gate 1: Schema version
if self.schema_version != "1.0.0" { /* fail */ }

// Gate 2: Compute path (real vs mock)
if self.compute_path != "real" { /* fail */ }

// Gate 3-7: Kernel ID hygiene (5 sub-gates)
if self.kernels.is_empty() { /* fail */ }
if self.kernels.len() > 10_000 { /* fail */ }
for kernel in &self.kernels {
    if kernel.is_empty() { /* fail */ }
    if kernel.contains("mock") { /* fail */ }
    if kernel.len() > 128 { /* fail */ }
}

// Gate 8: Test results
if self.test_results.failed > 0 { /* fail */ }
```

---

## 10. Best Practices Summary

| Pattern | Use Case | Example |
|---------|----------|---------|
| `#[derive(Serialize, Deserialize)]` | Standard types | All data structures |
| `#[serde(skip_serializing_if = "...")]` | Optional JSON fields | ModelInfo with many Optional<T> |
| `#[serde(skip)]` | Non-serializable | Function pointers, Arc<Mutex<>> |
| `#[serde(tag = "type")]` | Enum variants as types | CorrectionAction enum |
| `HashMap<String, Value>` | Dynamic fields | ValidationResult::metrics |
| `to_string_pretty()` | Human-readable output | Receipt files, logs |
| `from_str()` | Parse from strings | Config loading |
| `.validate()` | Post-deserialization | Policy and Receipt validation |
| Version constants | Schema tracking | RECEIPT_SCHEMA_VERSION |
| Builder pattern | Complex construction | ConfigBuilder |

---

## 11. Key Files Reference

| File | Purpose | Pattern |
|------|---------|---------|
| `/crates/bitnet-inference/src/receipts.rs` | AC4 receipt generation, schema v1.0.0 | JSON output, validation gates |
| `/crates/bitnet-models/src/correction_policy.rs` | YAML/JSON policy loading | Tagged unions, file format detection |
| `/crates/bitnet-trace/src/lib.rs` | Tensor tracing | Trace records, JSON serialization |
| `/crates/bitnet-inference/src/config.rs` | Configuration structures | Serializable configs with callbacks |
| `/tests/common/reporting/formats/json.rs` | Test report generation | Pretty printing, multiple formats |
| `/crossval/src/validation.rs` | Validation results | Structured metrics, HashMap<String, Value> |
| `/crates/bitnet-server/src/config.rs` | Server configuration | Environment variable integration |

---

## 12. Dependencies in Cargo.toml

```toml
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
serde_yaml_ng = "0.10"  # Used for policy files
toml = "0.8"            # For TOML configs (if needed)
```

---

## Conclusion

BitNet.rs demonstrates **production-grade serialization patterns** with:

1. **Consistent Serde usage** - All 91+ types use derive macros
2. **Strong validation** - Pre and post-serialization validation with clear error messages
3. **Schema versioning** - Explicit version constants with validation
4. **Flexible output** - Pretty-printing for humans, compact for APIs
5. **Multiple formats** - JSON, YAML, TOML support with format detection
6. **Type safety** - Enum tags, HashMap<String, Value> for dynamic fields
7. **Non-serializable types** - Clear patterns for skipping functions, Arc<T>
8. **Comprehensive tests** - Roundtrip serialization tests throughout

The patterns are ready to use as templates for new serializable types.

