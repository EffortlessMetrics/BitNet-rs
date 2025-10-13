# Python Binding Environment Requirements

## Issue Description

The `bitnet-py` crate tests fail with linking errors when Python development libraries are not available in the build environment. This is a common issue with PyO3-based projects that require system Python development packages.

## Linking Errors Observed

```
undefined reference to `PyExc_TypeError'
undefined reference to `Py_IncRef'
undefined reference to `Py_DecRef'
undefined reference to `PyDict_New'
undefined reference to `PyGILState_Ensure'
```

## Environment Requirements

To successfully build and test `bitnet-py`, the following system packages must be installed:

### Ubuntu/Debian
```bash
sudo apt-get install python3-dev python3-pip
```

### CentOS/RHEL/Fedora
```bash
# CentOS/RHEL
sudo yum install python3-devel python3-pip

# Fedora
sudo dnf install python3-devel python3-pip
```

### macOS
```bash
# Usually included with Python from Homebrew or python.org
brew install python3
```

## Workspace Configuration

The `bitnet-py` crate is intentionally **NOT** included in the workspace's `default-members` list in `Cargo.toml`:

```toml
default-members = [
    "crates/bitnet-common",
    "crates/bitnet-models",
    "crates/bitnet-tokenizers",
    "crates/bitnet-quantization",
    "crates/bitnet-kernels",
    "crates/bitnet-inference",
    "crates/bitnet-cli",
    "crates/bitnet-server",
]
```

This means:
- `cargo test --workspace` will NOT build or test `bitnet-py` by default
- `bitnet-py` tests are environment-dependent and optional
- CI/CD can run without Python development libraries

## Testing Python Bindings

To explicitly test Python bindings in environments with proper setup:

```bash
# Test only Python bindings (requires Python dev libraries)
cargo test --package bitnet-py --no-default-features --features cpu

# Test everything including Python bindings
cargo test --workspace --no-default-features --features cpu --include bitnet-py
```

## CI/CD Recommendations

1. **Default CI**: Run tests without `bitnet-py` to avoid environment dependencies
2. **Optional CI Job**: Add a separate CI job with Python development libraries installed to test Python bindings
3. **Documentation**: Clearly document Python environment requirements for contributors

## Fix Applied

- Fixed unused variable warning in `crates/bitnet-py/tests/test_streaming_comprehensive.rs` (line 333)
- Documented environment-specific nature of the linking issue
- Confirmed workspace configuration properly isolates Python binding tests
