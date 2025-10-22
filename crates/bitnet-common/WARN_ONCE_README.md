# Warn-Once Utility Infrastructure

Thread-safe, rate-limited logging utility for BitNet.rs that eliminates log spam from repeated warning conditions.

## Quick Start

```rust
use bitnet_common::warn_once;

fn deprecated_api() {
    warn_once!("deprecated_v1", "This API is deprecated, please use v2");
    // Your code...
}
```

## Features

- **Thread-Safe**: Uses `OnceLock<Mutex<HashSet<String>>>` for safe concurrent access
- **No `static mut`**: Purely safe Rust patterns, no unsafe code
- **Rate-Limited**: First occurrence at WARN level, subsequent at DEBUG level
- **Zero Dependencies**: Uses only std and `tracing` (already in workspace)
- **Tested**: Comprehensive test suite with thread-safety validation

## Implementation

Located in `crates/bitnet-common/src/warn_once.rs`:

### Public API

```rust
// Macro interface (recommended)
warn_once!("key", "Message");
warn_once!("key", "Formatted: {}", value);

// Function interface
pub fn warn_once_fn(key: &str, message: &str);
```

### Test Utilities

```rust
#[cfg(test)]
pub fn clear_registry_for_test();
```

## Design Decisions

1. **`OnceLock<Mutex<HashSet<String>>>`**: Thread-safe lazy initialization without `static mut`
2. **Poison Recovery**: Gracefully handles poisoned locks from panicking threads
3. **Structured Logging**: Uses `tracing` with key-value pairs for observability
4. **Serial Tests**: Uses `#[serial]` to avoid test flakiness with shared global state

## Testing

Run the test suite:

```bash
cargo test -p bitnet-common --lib warn_once --no-default-features
```

Run the demo example:

```bash
cargo run -p bitnet-common --example warn_once_demo --no-default-features
```

## Test Coverage

- ✅ `test_warn_once_is_rate_limited` - Verifies first warn, subsequent debug
- ✅ `test_warn_once_macro_simple` - Tests macro with simple strings
- ✅ `test_warn_once_macro_formatted` - Tests macro with format strings
- ✅ `test_warn_once_thread_safety` - Validates concurrent access (10 threads)
- ✅ `test_clear_registry` - Tests registry clearing for test isolation
- ✅ `test_multiple_unique_keys` - Tests independent tracking per key

## Documentation

- **Module docs**: `crates/bitnet-common/src/warn_once.rs` (inline documentation)
- **Usage guide**: `docs/howto/use-warn-once.md` (comprehensive examples)
- **Demo example**: `crates/bitnet-common/examples/warn_once_demo.rs` (runnable)

## Integration

The module is automatically re-exported from `bitnet-common`:

```rust
use bitnet_common::warn_once;         // Macro
use bitnet_common::warn_once_fn;      // Function
```

## Performance

- **Lock Contention**: Single global mutex (acceptable for warnings, not hot paths)
- **Memory**: O(n) where n = number of unique warning keys
- **Overhead**: Minimal - one HashSet lookup per warning call

## Safety Guarantees

- ✅ No `unsafe` code
- ✅ No `static mut` patterns
- ✅ Thread-safe with lock-based synchronization
- ✅ Poison recovery for robustness
- ✅ MSRV 1.90.0 compatible (uses stable `OnceLock`)

## Migration Path

This infrastructure is ready for integration with existing warning callsites. To migrate:

```rust
// Before:
warn!("Repeated warning message");

// After:
warn_once!("unique_key", "Repeated warning message");
```

Choose stable keys that represent the warning condition, not per-invocation data.

## Future Enhancements

Potential improvements (not implemented yet):

- Time-based rate limiting (e.g., warn once per hour)
- Configurable verbosity levels
- Warning statistics/metrics export
- Per-key TTL for cache eviction

## References

- [Rust `OnceLock` docs](https://doc.rust-lang.org/std/sync/struct.OnceLock.html)
- [Tracing docs](https://docs.rs/tracing)
- [Serial Test docs](https://docs.rs/serial_test)
