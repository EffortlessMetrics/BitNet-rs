# bitnet-ggml-ffi

FFI bindings for GGML used by BitNet.

## Compiler support

The C components of this crate compile with both GCC and Clang. Continuous integration checks build the library with both toolchains to ensure compatibility.

### Supported Compilers

- **GCC**: Tested with `gcc` and `g++`
- **Clang**: Tested with `clang` and `clang++` 

### Compiler Selection

To specify which compiler to use, set the `CC` and `CXX` environment variables:

```bash
# Use GCC (default on most Linux systems)
export CC=gcc CXX=g++
cargo build --no-default-features --features ffi

# Use Clang
export CC=clang CXX=clang++  
cargo build --no-default-features --features ffi
```

### GGML_DEPRECATED Macro

The GGML header includes compiler-specific deprecation warnings that work with both GCC and Clang:

- **Clang**: Uses `__attribute__((deprecated(hint)))` with `__clang__` detection
- **GCC**: Uses `__attribute__((deprecated(hint)))` with `__GNUC__` detection
- **MSVC**: Uses `__declspec(deprecated(hint))` with `_MSC_VER` detection

This ensures deprecation warnings are properly displayed across all supported compiler toolchains.
