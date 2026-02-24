# Serialization Quick Reference Guide

Fast lookup for common serialization patterns in BitNet.rs

## Quick Patterns

### Basic Serializable Type

```rust
use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MyType {
    pub field1: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub optional_field: Option<i32>,
    #[serde(skip)]
    pub non_serializable: Box<dyn Fn()>,
}
```

### Serialize to JSON

```rust
// Pretty (human-readable)
let json = serde_json::to_string_pretty(&obj)?;

// Compact (API responses)
let json = serde_json::to_string(&obj)?;

// Save to file
std::fs::write("output.json", json)?;
```

### Deserialize from JSON

```rust
let json_str = r#"{"field1": "value"}"#;
let obj: MyType = serde_json::from_str(json_str)?;
```

### Validate After Deserialization

```rust
impl MyType {
    pub fn validate(&self) -> anyhow::Result<()> {
        if self.field1.is_empty() {
            return Err(anyhow!("field1 required"));
        }
        Ok(())
    }
    
    pub fn load_from_file(path: &Path) -> anyhow::Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let obj: Self = serde_json::from_str(&content)?;
        obj.validate()?;
        Ok(obj)
    }
}
```

## Common Serde Attributes

| Attribute | Use | Example |
|-----------|-----|---------|
| `#[serde(skip)]` | Don't serialize/deserialize | Function pointers |
| `#[serde(skip_serializing_if = "Option::is_none")]` | Omit if None | Optional fields |
| `#[serde(default)]` | Use Default::default() if missing | Optional config |
| `#[serde(default = "func")]` | Use custom default | Custom values |
| `#[serde(rename = "...")]` | Different field name | JSON compatibility |
| `#[serde(tag = "type")]` | Enum discriminator field | Enum variants |

## Schema Versioning

### Version Constant

```rust
pub const SCHEMA_VERSION: &str = "1.0.0";

#[derive(Serialize, Deserialize)]
pub struct Versioned {
    pub schema_version: String,
    // ... rest of fields
}
```

### Version Validation

```rust
pub fn validate(&self) -> Result<()> {
    if self.schema_version != SCHEMA_VERSION {
        return Err(anyhow!("Invalid version"));
    }
    Ok(())
}
```

## YAML Support

### Load from YAML

```rust
pub fn from_yaml(yaml: &str) -> Result<Self> {
    serde_yaml_ng::from_str(yaml).map_err(|e| /* ... */)
}

pub fn from_file(path: &Path) -> Result<Self> {
    let content = std::fs::read_to_string(path)?;
    let obj = if is_json_file {
        serde_json::from_str(&content)?
    } else {
        Self::from_yaml(&content)?
    };
    obj.validate()?;
    Ok(obj)
}
```

## Validation Patterns

### Pre-Serialization

```rust
pub fn save(&self, path: &Path) -> Result<()> {
    self.validate()?;  // Check before saving
    let json = serde_json::to_string_pretty(self)?;
    std::fs::write(path, json)?;
    Ok(())
}
```

### Post-Deserialization

```rust
pub fn load(path: &Path) -> Result<Self> {
    let content = std::fs::read_to_string(path)?;
    let obj: Self = serde_json::from_str(&content)?;
    obj.validate()?;  // Check after loading
    Ok(obj)
}
```

## Multiple Output Formats

```rust
pub enum Format {
    Json { pretty: bool },
    Yaml,
    Toml,
}

impl MyType {
    pub fn to_format(&self, fmt: Format) -> Result<String> {
        match fmt {
            Format::Json { pretty } => {
                if pretty {
                    Ok(serde_json::to_string_pretty(self)?)
                } else {
                    Ok(serde_json::to_string(self)?)
                }
            }
            Format::Yaml => Ok(serde_yaml_ng::to_string(self)?),
            Format::Toml => Ok(toml::to_string_pretty(self)?),
        }
    }
}
```

## Enum Serialization (Tagged Union)

```rust
#[derive(Serialize, Deserialize)]
#[serde(tag = "type")]  // Use "type" field as discriminator
pub enum Action {
    #[serde(rename = "ACTION_ONE")]
    One { value: i32 },
    
    #[serde(rename = "ACTION_TWO")]
    Two { text: String },
}

// Serializes to:
// {"type": "ACTION_ONE", "value": 42}
// {"type": "ACTION_TWO", "text": "hello"}
```

## Non-Serializable Fields

```rust
pub struct Config {
    pub data: String,
    
    // Skip function pointers
    #[serde(skip)]
    pub callback: Option<Arc<dyn Fn()>>,
    
    // Skip Arc<Mutex<T>> and other complex types
    #[serde(skip)]
    pub internal_state: Arc<Mutex<State>>,
}

// Implement custom Debug if needed
impl std::fmt::Debug for Config {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.debug_struct("Config")
            .field("data", &self.data)
            .field("callback", &self.callback.is_some())
            .finish()
    }
}
```

## Dynamic Fields (HashMap<String, Value>)

```rust
#[derive(Serialize, Deserialize)]
pub struct Result {
    pub gate: String,
    pub passed: bool,
    pub metrics: HashMap<String, serde_json::Value>,
}

// Usage:
let mut result = Result { /* ... */ };
result.metrics.insert("count".to_string(), json!(42));
result.metrics.insert("items".to_string(), json!(vec!["a", "b"]));
```

## File Format Auto-Detection

```rust
pub fn load_from_file(path: &Path) -> Result<Self> {
    let content = std::fs::read_to_string(path)?;
    
    let obj = match path.extension().and_then(|s| s.to_str()) {
        Some("json") => serde_json::from_str(&content)?,
        Some("yaml") | Some("yml") => serde_yaml_ng::from_str(&content)?,
        Some("toml") => toml::from_str(&content)?,
        _ => serde_yaml_ng::from_str(&content)?,  // Default to YAML
    };
    
    obj.validate()?;
    Ok(obj)
}
```

## Testing Serialization

```rust
#[test]
fn test_roundtrip() {
    let original = MyType { field1: "test".into(), optional_field: Some(42) };
    
    // Serialize
    let json = serde_json::to_string(&original).unwrap();
    
    // Deserialize
    let restored: MyType = serde_json::from_str(&json).unwrap();
    
    // Verify
    assert_eq!(original.field1, restored.field1);
    assert_eq!(original.optional_field, restored.optional_field);
}

#[test]
fn test_optional_fields_omitted() {
    let obj = MyType { field1: "test".into(), optional_field: None };
    let json = serde_json::to_string_pretty(&obj).unwrap();
    
    // Verify optional field is not in JSON
    assert!(!json.contains("optional_field"));
}

#[test]
fn test_skip_fields_not_serialized() {
    let obj = MyType { /* ... */ };
    let json = serde_json::to_string(&obj).unwrap();
    
    // Verify non-serializable field is not in JSON
    assert!(!json.contains("callback"));
}
```

## Real-World Examples

### InferenceReceipt (AC4 Receipts)

```rust
#[derive(Serialize, Deserialize)]
pub struct InferenceReceipt {
    pub schema_version: String,      // "1.0.0"
    pub timestamp: String,            // ISO 8601
    pub compute_path: String,         // "real" or "mock"
    pub backend: String,              // "cpu", "cuda"
    pub kernels: Vec<String>,         // Kernel IDs
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parity: Option<ParityMetadata>,  // Optional parity data
}

// Save with validation
receipt.validate()?;  // 8 validation gates
receipt.save(Path::new("ci/inference.json"))?;
```

### CorrectionPolicy (YAML/JSON)

```rust
#[derive(Serialize, Deserialize)]
pub struct CorrectionPolicy {
    pub version: u32,
    pub models: Vec<ModelCorrection>,
}

#[derive(Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum CorrectionAction {
    #[serde(rename = "LN_GAMMA_RESCALE_RMS")]
    LnGammaRescaleRms { target_rms: f32, clamp: [f32; 2] },
}

// Load from file (auto-detects format)
let policy = CorrectionPolicy::load_from_file(Path::new("policy.yaml"))?;
policy.validate()?;
```

## Dependencies

```toml
[dependencies]
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
serde_yaml_ng = "0.10"
toml = "0.8"
anyhow = "1.0"
```

## Key Files

- `/crates/bitnet-inference/src/receipts.rs` - AC4 receipts with validation
- `/crates/bitnet-models/src/correction_policy.rs` - YAML/JSON policies
- `/crates/bitnet-trace/src/lib.rs` - Trace records
- `/tests/common/reporting/formats/json.rs` - Report generation

---

See `/docs/reference/SERIALIZATION_PATTERNS.md` for comprehensive documentation.
