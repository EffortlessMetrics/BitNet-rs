# Hardcoded values: Performance thresholds in `validation.rs` are hardcoded

The `PerformanceThresholds` struct in `crates/bitnet-inference/src/validation.rs` has hardcoded default values. These values may not be appropriate for all models and hardware configurations.

**File:** `crates/bitnet-inference/src/validation.rs`

**Struct:** `PerformanceThresholds`

**Code:**
```rust
pub struct PerformanceThresholds {
    pub min_tokens_per_second: f64,
    pub max_latency_ms: f64,
    pub max_memory_usage_mb: f64,
    pub min_speedup_factor: f64,
}

impl Default for PerformanceThresholds {
    fn default() -> Self {
        Self {
            min_tokens_per_second: 10.0,
            max_latency_ms: 5000.0,
            max_memory_usage_mb: 8192.0,
            min_speedup_factor: 1.5, // Expect at least 1.5x speedup over Python
        }
    }
}
```

## Proposed Fix

The `PerformanceThresholds` should be configurable. This can be done by loading the thresholds from a configuration file or by allowing the user to specify them as command-line arguments.

### Example Implementation

```rust
// In crates/bitnet-inference/src/validation.rs

impl PerformanceThresholds {
    pub fn from_config(config: &ValidationConfig) -> Self {
        Self {
            min_tokens_per_second: config.min_tokens_per_second.unwrap_or(10.0),
            max_latency_ms: config.max_latency_ms.unwrap_or(5000.0),
            max_memory_usage_mb: config.max_memory_usage_mb.unwrap_or(8192.0),
            min_speedup_factor: config.min_speedup_factor.unwrap_or(1.5),
        }
    }
}

// In ValidationConfig, add optional fields for thresholds
pub struct ValidationConfig {
    // ...
    pub min_tokens_per_second: Option<f64>,
    pub max_latency_ms: Option<f64>,
    pub max_memory_usage_mb: Option<f64>,
    pub min_speedup_factor: Option<f64>,
}
```
