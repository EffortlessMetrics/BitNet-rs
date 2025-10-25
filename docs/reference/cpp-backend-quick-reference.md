# C++ Backend Detection - Quick Reference Guide

## Overview
This quick reference shows how to implement C++ backend detection following BitNet.rs patterns.

## Quick Pattern Reference

### 1. Feature Gate
```rust
// In Cargo.toml
[features]
cpp = ["bitnet-ffi"]  # or custom C++ binding crate

// In code
#[cfg(feature = "cpp")]
pub fn cpp_compiled() -> bool { true }

#[cfg(not(feature = "cpp"))]
pub fn cpp_compiled() -> bool { false }
```

### 2. Runtime Detection
```rust
use std::env;
use std::path::PathBuf;

pub fn cpp_available_runtime() -> bool {
    // Priority 1: Check BITNET_CPP_FAKE for testing
    if let Ok(fake) = env::var("BITNET_CPP_FAKE") {
        return fake == "yes" || fake == "1" || fake.eq_ignore_ascii_case("reference");
    }
    
    // Priority 2: Strict mode override (real detection only)
    if env::var("BITNET_STRICT_MODE").is_ok() {
        return check_cpp_installation();
    }
    
    // Priority 3: Real detection
    check_cpp_installation()
}

fn check_cpp_installation() -> bool {
    // Check BITNET_CPP_DIR
    if let Ok(cpp_dir) = env::var("BITNET_CPP_DIR") {
        let lib_path = PathBuf::from(cpp_dir).join("lib/libbitnet.so");
        if lib_path.exists() {
            return true;
        }
    }
    
    // Check standard paths
    for path in &["/usr/local/lib", "/usr/lib", "~/.local/lib"] {
        if PathBuf::from(path).join("libbitnet.so").exists() {
            return true;
        }
    }
    
    // Check LD_LIBRARY_PATH
    if let Ok(ld_path) = env::var("LD_LIBRARY_PATH") {
        for path in ld_path.split(':') {
            if PathBuf::from(path).join("libbitnet.so").exists() {
                return true;
            }
        }
    }
    
    false
}
```

### 3. Backend Selection
```rust
pub fn select_cpp_backend() -> Result<Box<dyn Backend>> {
    // Compile-time check
    if !cpp_compiled() {
        return Err("C++ backend not compiled (missing --features cpp)".into());
    }
    
    // Runtime check
    if !cpp_available_runtime() {
        return Err("C++ backend not available at runtime".into());
    }
    
    // Create and return backend
    Ok(Box::new(CppBackendImpl::new()?))
}
```

### 4. Diagnostics
```rust
pub fn cpp_diagnostic_info() -> String {
    let mut info = String::new();
    
    // Compile-time status
    let compiled = if cpp_compiled() { "✓" } else { "✗" };
    info.push_str(&format!("C++ Backend Compiled: {}\n", compiled));
    
    // Runtime status
    let available = if cpp_available_runtime() { "✓" } else { "✗" };
    info.push_str(&format!("C++ Backend Available: {}\n", available));
    
    // Installation info
    if let Ok(cpp_dir) = std::env::var("BITNET_CPP_DIR") {
        info.push_str(&format!("C++ Installation: {}\n", cpp_dir));
    }
    
    info
}
```

## Testing Checklist

```rust
#[cfg(test)]
mod tests {
    use super::*;
    
    // Test 1: Compile-time detection
    #[test]
    fn test_cpp_compiled() {
        #[cfg(feature = "cpp")]
        assert!(cpp_compiled());
        
        #[cfg(not(feature = "cpp"))]
        assert!(!cpp_compiled());
    }
    
    // Test 2: Fake override (testing)
    #[test]
    fn test_cpp_fake_override() {
        std::env::set_var("BITNET_CPP_FAKE", "yes");
        assert!(cpp_available_runtime());
        
        std::env::set_var("BITNET_CPP_FAKE", "no");
        assert!(!cpp_available_runtime());
    }
    
    // Test 3: Strict mode (safety)
    #[test]
    fn test_cpp_strict_mode() {
        std::env::set_var("BITNET_STRICT_MODE", "1");
        std::env::set_var("BITNET_CPP_FAKE", "yes");
        
        // Strict mode should ignore BITNET_CPP_FAKE
        let result = cpp_available_runtime();
        // (result depends on real C++ installation)
        
        std::env::remove_var("BITNET_STRICT_MODE");
    }
    
    // Test 4: Backend selection
    #[test]
    fn test_cpp_backend_selection() {
        if cpp_compiled() && cpp_available_runtime() {
            assert!(select_cpp_backend().is_ok());
        } else {
            assert!(select_cpp_backend().is_err());
        }
    }
}
```

## Environment Variables

| Variable | Purpose | Values | Example |
|----------|---------|--------|---------|
| `BITNET_CPP_FAKE` | Fake C++ for testing | `yes`, `1`, `reference`, `no` | `BITNET_CPP_FAKE=yes` |
| `BITNET_CPP_DIR` | C++ installation path | Absolute path | `BITNET_CPP_DIR=/opt/bitnet.cpp` |
| `BITNET_STRICT_MODE` | Force real detection | `1`, `true` | `BITNET_STRICT_MODE=1` |
| `LD_LIBRARY_PATH` | Library search path | Colon-separated paths | `LD_LIBRARY_PATH=/opt/bitnet/lib` |

## Integration Points

### 1. Add to Feature Flags
```toml
[features]
default = []
cpu = []
gpu = ["cuda"]
cuda = []
cpp = ["bitnet-ffi"]  # C++ FFI bridge
crossval-all = ["inference", "crossval", "ffi"]
```

### 2. Add to Kernel Manager
```rust
pub struct KernelManager {
    providers: Vec<Box<dyn KernelProvider>>,
}

// In new() constructor:
#[cfg(feature = "cpp")]
if let Ok(cpp_kernel) = cpp::CppKernel::new() {
    if cpp_kernel.is_available() {
        providers.insert(0, Box::new(cpp_kernel));
    }
}
```

### 3. Add to Backend Selection
```rust
pub fn select_best_backend(config: &InferenceConfig) -> Result<Box<dyn Backend>> {
    match config.backend_preference {
        BackendPreference::Auto => {
            // Try: GPU → C++ → CPU
            if let Ok(backend) = select_gpu_backend() { return Ok(backend); }
            if let Ok(backend) = select_cpp_backend() { return Ok(backend); }
            Ok(Box::new(CpuBackend::new()?))
        }
        BackendPreference::Cpp => select_cpp_backend(),
        // ... other cases
    }
}
```

### 4. Add to Diagnostics
```rust
pub fn device_capability_summary() -> String {
    let mut summary = String::from("Device Capabilities:\n");
    
    // ... existing GPU/CPU code ...
    
    // Add C++ backend info
    #[cfg(feature = "cpp")]
    {
        summary.push_str("  C++ Backend: ");
        if cpp_available_runtime() {
            summary.push_str("✓\n");
        } else {
            summary.push_str("✗\n");
        }
    }
    
    summary
}
```

## Files to Modify

1. **`Cargo.toml`** - Add `cpp` feature
2. **`src/cpp_features.rs`** (new) - Compile-time checks
3. **`src/cpp_detection.rs`** (new) - Runtime detection
4. **`src/backends.rs`** - Add C++ backend selection
5. **`crates/bitnet-kernels/src/lib.rs`** - Add to kernel manager
6. **`crates/bitnet-kernels/src/device_features.rs`** - Extend capability summary
7. **`xtask/tests/preflight.rs`** - Add C++ preflight tests

## Command Examples

```bash
# Build with C++ support
cargo build --features cpp

# Test with fake C++ override
BITNET_CPP_FAKE=yes cargo test --features cpp

# Test with real C++ detection
BITNET_CPP_DIR=/opt/bitnet.cpp cargo test --features cpp

# Run preflight check
cargo run -p xtask -- preflight

# Inspect C++ capabilities
RUST_LOG=debug cargo run -p bitnet-cli -- inspect --cpp-info
```

## Related Documentation

- **Full Report**: `docs/explanation/backend-detection-and-device-selection-patterns.md`
- **GPU Patterns**: `crates/bitnet-kernels/src/device_features.rs`
- **Test Examples**: `xtask/tests/preflight.rs`
- **Kernel Selection**: `crates/bitnet-kernels/src/lib.rs`

## Key Principles

1. **No defaults** - Always require explicit `--features cpp`
2. **Two-tier checking** - Compile-time + runtime
3. **Test overrides** - `BITNET_CPP_FAKE` for reproducibility
4. **Safety gates** - `BITNET_STRICT_MODE` for production
5. **Graceful fallback** - Always have CPU as final fallback
6. **Clear diagnostics** - Human-readable capability summary
7. **Consistent naming** - Follow BITNET_* naming convention

---

**Quick Reference**: Copy-paste ready code patterns for C++ backend integration.
