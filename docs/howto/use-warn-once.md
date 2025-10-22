# How to Use the Warn-Once Utility

The `warn_once` utility provides thread-safe, rate-limited logging to avoid log spam when the same warning condition occurs repeatedly. This is particularly useful in hot paths, deprecation warnings, and non-fatal error conditions.

## Overview

The warn-once infrastructure uses `OnceLock<Mutex<HashSet<String>>>` to track seen warnings without requiring `static mut` or other unsafe patterns. It follows Rust safety best practices and is fully thread-safe.

## Basic Usage

### Simple Warnings

```rust
use bitnet_common::warn_once;

fn deprecated_function() {
    warn_once!("deprecated_api_v1", "Using deprecated API v1, please migrate to v2");
    // Function logic...
}
```

### Formatted Warnings

```rust
use bitnet_common::warn_once;

fn check_threshold(value: i32, threshold: i32) {
    if value > threshold {
        warn_once!(
            "threshold_exceeded",
            "Value {} exceeds threshold of {}",
            value,
            threshold
        );
    }
}
```

## Behavior

### First Occurrence

The first time a warning with a given key is logged, it appears at **WARN** level:

```text
WARN bitnet_common::warn_once: Using deprecated API v1 key=deprecated_api_v1
```

### Subsequent Occurrences

Subsequent warnings with the same key are logged at **DEBUG** level with a rate-limited prefix:

```text
DEBUG bitnet_common::warn_once: (rate-limited) Using deprecated API v1 key=deprecated_api_v1
```

## Key Selection

The warning key should be a stable identifier that represents the warning condition:

- **Good**: `"deprecated_api_v1"`, `"gpu_unavailable"`, `"model_fallback"`
- **Bad**: Dynamic keys with timestamps or unique IDs that change per call

### Example: Stable vs Dynamic Keys

```rust
// Good: Stable key for consistent rate-limiting
warn_once!("gpu_unavailable", "GPU not available, falling back to CPU");

// Bad: Dynamic key defeats rate-limiting
let timestamp = std::time::SystemTime::now();
warn_once!(
    &format!("gpu_unavailable_{:?}", timestamp),  // DON'T DO THIS
    "GPU not available"
);
```

## Thread Safety

The warn-once utility is fully thread-safe and can be called concurrently from multiple threads. The first thread to encounter a new warning key will log at WARN level; other threads will see the key as already-warned and log at DEBUG level.

### Example: Concurrent Usage

```rust
use std::thread;
use bitnet_common::warn_once;

fn process_data(id: usize) {
    // All threads share the same warning key
    warn_once!("processing_warning", "Data processing may be slow");
    // Process data...
}

fn main() {
    let handles: Vec<_> = (0..10)
        .map(|i| thread::spawn(move || process_data(i)))
        .collect();

    for handle in handles {
        handle.join().unwrap();
    }
}
```

## Common Use Cases

### 1. Deprecation Warnings

```rust
use bitnet_common::warn_once;

#[deprecated(since = "0.2.0", note = "Use new_api instead")]
fn old_api() {
    warn_once!("old_api_deprecated", "old_api is deprecated, use new_api instead");
    // Implementation...
}
```

### 2. GPU/CPU Fallback

```rust
use bitnet_common::warn_once;

fn run_inference(use_gpu: bool) {
    if !use_gpu {
        warn_once!(
            "cpu_fallback",
            "GPU not available, falling back to CPU inference"
        );
    }
    // Inference logic...
}
```

### 3. Model Quality Warnings

```rust
use bitnet_common::warn_once;

fn validate_model_quality(model_name: &str, quality_score: f32) {
    if quality_score < 0.5 {
        warn_once!(
            "low_quality_model",
            "Model {} has low quality score: {:.2}",
            model_name,
            quality_score
        );
    }
}
```

### 4. Non-Fatal Errors in Hot Paths

```rust
use bitnet_common::warn_once;

fn process_token(token_id: u32) -> Result<(), String> {
    if token_id > MAX_VOCAB_SIZE {
        warn_once!(
            "token_out_of_bounds",
            "Token ID {} exceeds vocabulary size, clamping to max",
            token_id
        );
        // Clamp and continue...
    }
    Ok(())
}
```

## Testing

When testing code that uses warn_once, you may want to clear the registry between test cases to verify warning behavior:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use bitnet_common::warn_once;
    use serial_test::serial;  // Ensure sequential execution

    #[test]
    #[serial]
    fn test_warning_behavior() {
        #[cfg(test)]
        bitnet_common::warn_once::clear_registry_for_test();

        // First call logs at WARN
        my_function_with_warning();

        // Second call logs at DEBUG
        my_function_with_warning();
    }
}
```

**Note**: The `clear_registry_for_test()` function is only available in test builds (`#[cfg(test)]`) and should be used with `#[serial]` from `serial_test` to avoid race conditions.

## Implementation Details

### Architecture

```rust
use std::collections::HashSet;
use std::sync::{Mutex, OnceLock};

static WARN_REGISTRY: OnceLock<Mutex<HashSet<String>>> = OnceLock::new();

pub fn warn_once_fn(key: &str, message: &str) {
    let registry = get_registry();
    let mut seen = match registry.lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),  // Recover from poisoned lock
    };

    if seen.insert(key.to_string()) {
        tracing::warn!(key = %key, "{}", message);
    } else {
        tracing::debug!(key = %key, "(rate-limited) {}", message);
    }
}
```

### Key Design Decisions

1. **`OnceLock<Mutex<HashSet<String>>>`**: Provides thread-safe lazy initialization without `static mut`
2. **Poison Recovery**: Recovers from poisoned locks (if a thread panics while holding the lock)
3. **Structured Logging**: Uses `tracing` with key-value pairs for better observability
4. **Zero Unsafe**: No unsafe code, purely safe Rust patterns

### Performance Characteristics

- **Lock Contention**: Uses a single global mutex, acceptable for warning scenarios (not hot paths)
- **Memory**: O(n) where n is the number of unique warning keys
- **Thread Safety**: Fully thread-safe with lock acquisition per warning call

## Migration from Other Patterns

### From Manual `static mut`

```rust
// Old (unsafe):
static mut WARNED: bool = false;
unsafe {
    if !WARNED {
        warn!("Some warning");
        WARNED = true;
    }
}

// New (safe):
warn_once!("some_warning", "Some warning");
```

### From `once_cell` or `lazy_static`

```rust
// Old:
use once_cell::sync::Lazy;
use std::sync::Mutex;

static WARNED: Lazy<Mutex<bool>> = Lazy::new(|| Mutex::new(false));

let mut warned = WARNED.lock().unwrap();
if !*warned {
    warn!("Some warning");
    *warned = true;
}

// New:
warn_once!("some_warning", "Some warning");
```

## See Also

- [`tracing` documentation](https://docs.rs/tracing) for structured logging
- [Rust stdlib `OnceLock`](https://doc.rust-lang.org/std/sync/struct.OnceLock.html)
- BitNet.rs logging conventions in `docs/development/logging.md` (if exists)
