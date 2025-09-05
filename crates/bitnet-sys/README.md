# bitnet-sys

Low-level FFI bindings to the BitNet C++ implementation for cross-validation purposes.

## Overview

This crate provides unsafe FFI bindings to the original BitNet C++ implementation, enabling cross-validation between the Rust and C++ versions. It is designed to be used internally by the `bitnet-crossval` crate and should not be used directly in most cases.

## Features

- `ffi`: Enables FFI bindings (requires C++ dependencies and clang)
- `crossval`: Backwards compatibility alias for `ffi`

## Requirements

### System Dependencies

- **clang**: Required for generating bindings from C++ headers
  - Ubuntu/Debian: `sudo apt install clang libclang-dev`
  - macOS: `xcode-select --install`
  - Windows: Install LLVM from https://llvm.org/

### BitNet C++ Implementation

The C++ implementation must be available and built:

```bash
# Download and build C++ implementation
./ci/fetch_bitnet_cpp.sh

# Or set custom path
export BITNET_CPP_DIR=/path/to/bitnet.cpp  # BITNET_CPP_PATH is also accepted
```

## Usage

This crate is primarily intended for internal use by `bitnet-crossval`. For cross-validation, use the higher-level API:

```rust
// Don't use bitnet-sys directly
use bitnet_crossval::comparison::CrossValidator;

// Instead of bitnet-sys raw bindings
let validator = CrossValidator::new(config);
```

### Direct Usage (Advanced)

If you need direct access to the FFI bindings:

```rust
#[cfg(feature = "ffi")]
use bitnet_sys::{cleanup, generate, initialize, load_model};

#[cfg(feature = "ffi")]
fn example() -> Result<(), Box<dyn std::error::Error>> {
    // Check if C++ implementation is available
    if !bitnet_sys::is_available() {
        return Err("C++ implementation not available".into());
    }

    // Initialize backend and load a model
    initialize()?;
    let mut model = load_model("path/to/model.gguf")?;

    // Generate tokens (prompt excluded)
    let tokens = generate(&mut model, "Hello, world!", 100)?;

    // Clean up global resources
    cleanup()?;
    
    println!("Generated {} tokens", tokens.len());
    Ok(())
}
```

## Safety

All functions in this crate are unsafe or wrap unsafe FFI calls. The crate provides some safety guarantees:

- **Memory management**: Model handles are automatically cleaned up
- **Error handling**: C++ errors are converted to Rust errors
- **String handling**: Proper conversion between Rust and C strings

However, you must still ensure:

- Models are not used after being dropped
- Thread safety (C++ implementation may not be thread-safe)
- Proper initialization and cleanup

## Build Process

When the `ffi` feature is enabled, the build script:

1. **Locates C++ implementation**: Uses `BITNET_CPP_DIR` (or legacy `BITNET_CPP_PATH`) or default cache location
2. **Checks dependencies**: Verifies clang is available
3. **Sets up paths**: Configures include and library paths
4. **Generates bindings**: Uses bindgen to create Rust bindings from C++ headers
5. **Links libraries**: Links against the built C++ libraries

If the C++ implementation cannot be located, the build now fails with a clear
error instead of generating stub bindings.

### Build Errors

Common build errors and solutions:

#### "BitNet C++ implementation not found"

```bash
# Download the C++ implementation
./ci/fetch_bitnet_cpp.sh

# Or set custom path
export BITNET_CPP_DIR=/path/to/bitnet.cpp
cargo build -p bitnet-sys --features bitnet-sys/ffi
```

#### "clang not found"

Install clang development tools:

```bash
# Ubuntu/Debian
sudo apt install clang libclang-dev

# macOS
xcode-select --install

# Windows
# Install LLVM from https://llvm.org/
```

#### "No BitNet header file found"

Ensure the C++ implementation is properly built:

```bash
cd ~/.cache/bitnet_cpp
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make
```

## Environment Variables

- `BITNET_CPP_DIR`: Path to BitNet C++ implementation (default: `~/.cache/bitnet_cpp`). `BITNET_CPP_PATH` is also recognized.
- `BITNET_CPP_INCLUDE_PATH`: Path to C++ headers (auto-detected)
- `BITNET_CPP_LIB_PATH`: Path to C++ libraries (auto-detected)

## Integration with Cross-Validation

This crate is designed to work seamlessly with the cross-validation framework:

```rust
// High-level cross-validation (recommended)
use bitnet_crossval::comparison::validate_all_fixtures;

let results = validate_all_fixtures(config)?;

// The crossval crate uses bitnet-sys internally
// You don't need to interact with bitnet-sys directly
```

## Troubleshooting

### Feature Not Enabled

If you get "BitNet C++ bindings not available" errors:

```bash
# Make sure to enable the ffi feature
cargo test -p bitnet-sys --features bitnet-sys/ffi
cargo bench -p bitnet-sys --features bitnet-sys/ffi
```

### Library Linking Issues

If you get linking errors:

1. Ensure the C++ implementation is built
2. Check that libraries are in the expected location
3. Verify library names match what the build script expects

### Binding Generation Issues

If bindgen fails:

1. Ensure clang is installed and in PATH
2. Check that C++ headers are present
3. Verify header file format is compatible

## License

This crate follows the same license as the main BitNet project.

## Contributing

When contributing to this crate:

1. Maintain the feature gate for all C++ dependencies
2. Provide helpful error messages for missing dependencies
3. Ensure safety guarantees are maintained
4. Test with and without the ffi feature
5. Update documentation for any API changes
