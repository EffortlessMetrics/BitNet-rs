# [ENHANCEMENT] Hardcoded vendor key normalization patterns limit model compatibility

## Problem Description

The `normalize_vendor_key` function in `crates/bitnet-models/src/weight_mapper.rs` uses hardcoded regex patterns to map vendor-specific weight names to canonical names. This rigid approach prevents support for new model architectures, custom fine-tuned models, and evolving vendor naming conventions without code modifications.

## Environment

**Affected Component:** `crates/bitnet-models/src/weight_mapper.rs`
**Function:** `normalize_vendor_key`
**Impact:** Model compatibility, extensibility, maintenance overhead
**Related Features:** Model loading, GGUF compatibility, cross-vendor support

## Root Cause Analysis

### Current Implementation Limitations

1. **Hardcoded regex patterns**: All vendor mappings compiled into code
2. **Limited extensibility**: Adding new vendors requires code changes
3. **Maintenance overhead**: Regex patterns scattered throughout implementation
4. **Inflexible mapping**: Cannot handle vendor-specific variations without modification

### Code Analysis

```rust
pub fn normalize_vendor_key(k: &str) -> Option<String> {
    macro_rules! cap {
        ($re:expr, $k:expr, $fmt:expr) => {{ if let Some(c) = $re.captures($k) { Some(format!($fmt, &c[1])) } else { None } }};
    }

    // Hardcoded patterns for specific vendors
    cap!(RE_BLK_ATTN_Q, k, "layers.{}.attention.q_proj.weight")
        .or_else(|| cap!(RE_BLK_ATTN_K, k, "layers.{}.attention.k_proj.weight"))
        .or_else(|| cap!(RE_LLAMA_WQ, k, "layers.{}.attention.q_proj.weight"))
        // ... many more hardcoded patterns
}
```

Issues:
- Cannot support new model architectures without code changes
- No runtime configurability for custom models
- Difficult to maintain as vendor patterns evolve
- Lacks validation and error reporting for mapping failures

## Impact Assessment

### Model Compatibility Impact
- **Blocked architectures**: New models cannot be loaded without code changes
- **Custom models**: Fine-tuned models with non-standard naming fail to load
- **Vendor evolution**: Updates to vendor naming conventions break compatibility
- **Development velocity**: Adding new models requires development cycles

### Maintenance Impact
- **Code complexity**: Regex patterns scattered throughout codebase
- **Testing overhead**: Each new pattern requires comprehensive testing
- **Documentation burden**: Mapping rules not easily discoverable

## Proposed Solution

### Configurable Mapping System

Replace hardcoded patterns with flexible, runtime-configurable mapping system:

```rust
#[derive(Debug, Clone, Deserialize)]
pub struct VendorMappingRule {
    pub pattern: String,
    pub canonical_format: String,
    pub description: String,
    pub vendor: String,
    pub architecture: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct VendorMappingConfig {
    pub rules: Vec<VendorMappingRule>,
    pub fallback_patterns: Vec<VendorMappingRule>,
    pub strict_mode: bool,
}

pub struct VendorKeyNormalizer {
    rules: Vec<CompiledMappingRule>,
    fallback_rules: Vec<CompiledMappingRule>,
    strict_mode: bool,
}

impl VendorKeyNormalizer {
    pub fn from_config(config: VendorMappingConfig) -> Result<Self> {
        let rules = config.rules.into_iter()
            .map(|rule| CompiledMappingRule::compile(rule))
            .collect::<Result<Vec<_>>>()?;

        let fallback_rules = config.fallback_patterns.into_iter()
            .map(|rule| CompiledMappingRule::compile(rule))
            .collect::<Result<Vec<_>>>()?;

        Ok(Self {
            rules,
            fallback_rules,
            strict_mode: config.strict_mode,
        })
    }

    pub fn normalize_key(&self, vendor_key: &str) -> Result<Option<String>> {
        // Try primary rules first
        for rule in &self.rules {
            if let Some(normalized) = rule.apply(vendor_key)? {
                return Ok(Some(normalized));
            }
        }

        // Try fallback rules
        for rule in &self.fallback_rules {
            if let Some(normalized) = rule.apply(vendor_key)? {
                log::debug!("Using fallback rule for key: {}", vendor_key);
                return Ok(Some(normalized));
            }
        }

        if self.strict_mode {
            Err(BitNetError::ModelLoading(format!(
                "No mapping rule found for vendor key: {}", vendor_key
            )))
        } else {
            log::warn!("No mapping found for vendor key: {}", vendor_key);
            Ok(None)
        }
    }
}

struct CompiledMappingRule {
    regex: Regex,
    format_template: String,
    metadata: RuleMetadata,
}

impl CompiledMappingRule {
    fn compile(rule: VendorMappingRule) -> Result<Self> {
        let regex = Regex::new(&rule.pattern)
            .map_err(|e| BitNetError::Configuration(format!("Invalid regex pattern '{}': {}", rule.pattern, e)))?;

        Ok(Self {
            regex,
            format_template: rule.canonical_format,
            metadata: RuleMetadata {
                description: rule.description,
                vendor: rule.vendor,
                architecture: rule.architecture,
            },
        })
    }

    fn apply(&self, input: &str) -> Result<Option<String>> {
        if let Some(captures) = self.regex.captures(input) {
            let mut result = self.format_template.clone();

            // Replace capture group placeholders
            for (i, capture) in captures.iter().enumerate().skip(1) {
                if let Some(capture_str) = capture {
                    result = result.replace(&format!("{{{}}}", i - 1), capture_str.as_str());
                }
            }

            Ok(Some(result))
        } else {
            Ok(None)
        }
    }
}
```

### Configuration File Format

```yaml
# vendor_mappings.yaml
rules:
  # LLaMA-style attention patterns
  - pattern: "layers\\.([0-9]+)\\.attention\\.wq\\.weight"
    canonical_format: "layers.{0}.attention.q_proj.weight"
    description: "LLaMA query projection weight"
    vendor: "meta"
    architecture: "llama"

  - pattern: "layers\\.([0-9]+)\\.attention\\.wk\\.weight"
    canonical_format: "layers.{0}.attention.k_proj.weight"
    description: "LLaMA key projection weight"
    vendor: "meta"
    architecture: "llama"

  # BitNet-specific patterns
  - pattern: "blk\\.([0-9]+)\\.attn_q\\.weight"
    canonical_format: "layers.{0}.attention.q_proj.weight"
    description: "BitNet attention query weight"
    vendor: "microsoft"
    architecture: "bitnet"

  # Generic fallback patterns
  - pattern: ".*\\.([0-9]+)\\.(.*q.*)\\.weight"
    canonical_format: "layers.{0}.attention.q_proj.weight"
    description: "Generic query weight pattern"
    vendor: "generic"

fallback_patterns:
  # Fallback for unknown patterns
  - pattern: "layers\\.([0-9]+)\\.(.*)"
    canonical_format: "layers.{0}.{1}"
    description: "Pass-through for standard layer structure"
    vendor: "generic"

strict_mode: false
```

### Dynamic Configuration Loading

```rust
impl VendorKeyNormalizer {
    pub fn default() -> Result<Self> {
        // Load from embedded default configuration
        let default_config = include_str!("../configs/default_vendor_mappings.yaml");
        let config: VendorMappingConfig = serde_yaml::from_str(default_config)?;
        Self::from_config(config)
    }

    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let config: VendorMappingConfig = serde_yaml::from_str(&content)?;
        Self::from_config(config)
    }

    pub fn with_custom_rules(mut self, additional_rules: Vec<VendorMappingRule>) -> Result<Self> {
        let compiled_rules = additional_rules.into_iter()
            .map(|rule| CompiledMappingRule::compile(rule))
            .collect::<Result<Vec<_>>>()?;

        // Insert custom rules at the beginning for priority
        self.rules.splice(0..0, compiled_rules);
        Ok(self)
    }
}
```

## Implementation Plan

### Phase 1: Core Infrastructure (2-3 days)
- [ ] Design and implement `VendorMappingConfig` and related types
- [ ] Create `VendorKeyNormalizer` with compilation and caching
- [ ] Implement `CompiledMappingRule` with regex compilation
- [ ] Add configuration loading from YAML/JSON files

### Phase 2: Configuration Migration (1-2 days)
- [ ] Extract existing hardcoded patterns to configuration files
- [ ] Create comprehensive default mapping configuration
- [ ] Implement backward compatibility layer
- [ ] Add validation for mapping rule correctness

### Phase 3: Enhanced Features (1-2 days)
- [ ] Add runtime rule addition and modification
- [ ] Implement pattern testing and validation utilities
- [ ] Add metrics and logging for mapping performance
- [ ] Create debugging tools for pattern matching

### Phase 4: Integration & Testing (1-2 days)
- [ ] Update model loading pipeline to use new normalizer
- [ ] Add comprehensive test suite for mapping scenarios
- [ ] Validate compatibility with existing models
- [ ] Performance benchmarking and optimization

## Testing Strategy

### Configuration Testing
```rust
#[test]
fn test_mapping_configuration_loading() {
    let config_yaml = r#"
rules:
  - pattern: "layers\\.([0-9]+)\\.attention\\.wq\\.weight"
    canonical_format: "layers.{0}.attention.q_proj.weight"
    description: "LLaMA query projection"
    vendor: "meta"
strict_mode: true
"#;

    let config: VendorMappingConfig = serde_yaml::from_str(config_yaml).unwrap();
    let normalizer = VendorKeyNormalizer::from_config(config).unwrap();

    let result = normalizer.normalize_key("layers.5.attention.wq.weight").unwrap();
    assert_eq!(result, Some("layers.5.attention.q_proj.weight".to_string()));
}

#[test]
fn test_fallback_patterns() {
    let normalizer = VendorKeyNormalizer::default().unwrap();

    // Test unknown pattern with fallback
    let result = normalizer.normalize_key("unknown.pattern.weight").unwrap();
    assert!(result.is_some() || !normalizer.strict_mode);
}
```

### Model Compatibility Testing
```rust
#[test]
fn test_vendor_compatibility() {
    let test_cases = vec![
        // LLaMA patterns
        ("layers.0.attention.wq.weight", "layers.0.attention.q_proj.weight"),
        ("layers.0.attention.wk.weight", "layers.0.attention.k_proj.weight"),

        // BitNet patterns
        ("blk.0.attn_q.weight", "layers.0.attention.q_proj.weight"),
        ("blk.0.attn_k.weight", "layers.0.attention.k_proj.weight"),

        // GPT patterns
        ("transformer.h.0.attn.c_attn.weight", "layers.0.attention.qkv_proj.weight"),
    ];

    let normalizer = VendorKeyNormalizer::default().unwrap();

    for (input, expected) in test_cases {
        let result = normalizer.normalize_key(input).unwrap();
        assert_eq!(result, Some(expected.to_string()), "Failed for input: {}", input);
    }
}
```

### Performance Testing
```rust
#[test]
fn benchmark_normalization_performance() {
    let normalizer = VendorKeyNormalizer::default().unwrap();
    let test_keys: Vec<_> = (0..1000)
        .map(|i| format!("layers.{}.attention.wq.weight", i))
        .collect();

    let start = Instant::now();
    for key in &test_keys {
        let _result = normalizer.normalize_key(key).unwrap();
    }
    let elapsed = start.elapsed();

    // Should normalize 1000 keys in under 10ms
    assert!(elapsed < Duration::from_millis(10));
}
```

## Risk Assessment

### Implementation Risks
- **Performance impact**: Regex compilation and matching overhead
- **Configuration complexity**: Complex YAML configuration may be error-prone
- **Backward compatibility**: Existing hardcoded patterns must remain functional

### Mitigation Strategies
- Implement regex compilation caching and optimization
- Add comprehensive configuration validation and error reporting
- Provide migration tools and backward compatibility layer
- Include extensive test coverage for existing model compatibility

## Success Criteria

### Extensibility Improvements
- [ ] New vendor patterns can be added via configuration files
- [ ] Runtime rule modification without code changes
- [ ] Custom model support through external configuration
- [ ] Zero-downtime pattern updates for supported architectures

### Performance Targets
- [ ] Normalization performance < 10μs per key on average
- [ ] Memory usage increase < 5MB for comprehensive rule sets
- [ ] Startup time increase < 100ms for configuration loading
- [ ] Backward compatibility with all existing supported models

## Related Issues

- **Model Loading**: Integration with GGUF and SafeTensors loaders
- **Cross-validation**: Compatibility with Microsoft BitNet C++ reference
- **Configuration Management**: Runtime configuration system

## References

- Vendor-specific model architectures and naming conventions
- Regex performance optimization techniques
- Configuration management best practices
- Model compatibility testing frameworks

---

**Priority**: Medium-High
**Estimated Effort**: 4-6 developer days
**Components**: bitnet-models, configuration system
**Feature Flags**: None (core functionality)
