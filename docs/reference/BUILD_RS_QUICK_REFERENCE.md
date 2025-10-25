# Build.rs Quick Reference

**For developers who need to understand build.rs patterns without reading the full 29KB analysis**

## One-Sentence Summary

BitNet.rs uses a **priority-ordered directory scanner** with environment variable overrides to locate C++ libraries (libllama, libggml, libbitnet) across multiple possible CMake build output locations.

## Library Search Chain

```
1. BITNET_CROSSVAL_LIBDIR (explicit override)
   ↓
2. $BITNET_CPP_DIR/build/3rdparty/llama.cpp/src
   ↓
3. $BITNET_CPP_DIR/build/3rdparty/llama.cpp/ggml/src
   ↓
4. $BITNET_CPP_DIR/build/bin
   ↓
5. $BITNET_CPP_DIR/build/lib
   ↓
6. $BITNET_CPP_DIR/build
   ↓
7. $BITNET_CPP_DIR/lib
   ↓
8. DEFAULT: ~/.cache/bitnet_cpp
```

## Key Files

| File | What It Does | Error Strategy |
|------|--------------|-----------------|
| `crossval/build.rs` | C++ linking for cross-validation | Soft warning (graceful degradation) |
| `crates/bitnet-sys/build.rs` | FFI bindings + shim compilation | Hard panic (FFI is required) |
| `crates/bitnet-kernels/build.rs` | GPU + optional FFI | Conditional fallback |
| `xtask-build-helper/src/lib.rs` | Shared FFI compilation logic | DRY helper |

## Environment Variables

| Name | Set By | Used In | Effect |
|------|--------|---------|--------|
| `BITNET_CPP_DIR` | User/CI | All FFI build.rs | Root directory for C++ sources |
| `BITNET_CPP_PATH` | User/CI | Legacy fallback | Same as CPP_DIR (deprecated) |
| `BITNET_CROSSVAL_LIBDIR` | `xtask setup-cpp-auto` | crossval/build.rs | Direct library directory override |
| `BITNET_GPU_FAKE` | Test runner | Runtime only | Override GPU detection (not build-time) |

## Linking Pattern

All build.rs follow this sequence:

```rust
// 1. Search for libraries
println!("cargo:rustc-link-search=native=/path/to/libs");

// 2. Link them
println!("cargo:rustc-link-lib=dylib=llama");  // or static=llama

// 3. Add RPATH (eliminates LD_LIBRARY_PATH need)
println!("cargo:rustc-link-arg=-Wl,-rpath,/path/to/libs");

// 4. Platform-specific dependencies
#[cfg(target_os = "linux")]
println!("cargo:rustc-link-lib=dylib=stdc++");

// 5. Mark as found (if needed)
println!("cargo:rustc-cfg=have_cpp");
```

## THE GAP: No Backend-Aware Discovery

**Current**: Hardcoded CUDA paths when GPU feature enabled

**Missing**: Auto-detection of GPU backend (ROCm, Metal, oneAPI)

**Impact**: Works on CUDA systems, fails on ROCm/Metal without manual override

**Solution**: Query `BITNET_GPU_BACKEND` env var or auto-detect via `rocm-smi`, `nvidia-smi`, etc.

## Error Handling Strategies

| File | Strategy | Example |
|------|----------|---------|
| bitnet-sys/build.rs | **Hard Panic** | `panic!("FFI requires C++. Run ./ci/fetch_bitnet_cpp.sh")` |
| crossval/build.rs | **Soft Warning** | `println!("cargo:warning=Using mock C wrapper")` |
| bitnet-kernels/build.rs | **Conditional** | Only links FFI if headers+libs found |

**Rule**: Use hard panic only for required dependencies. Use soft warning for optional.

## Device Detection (Runtime, NOT build-time)

```rust
// Compile-time
pub fn gpu_compiled() -> bool {
    cfg!(any(feature = "gpu", feature = "cuda"))
}

// Runtime
pub fn gpu_available_runtime() -> bool {
    // Check BITNET_GPU_FAKE first
    if let Ok(fake) = env::var("BITNET_GPU_FAKE") {
        return fake.eq_ignore_ascii_case("cuda");
    }
    // Real detection via nvidia-smi
    crate::gpu_utils::get_gpu_info().cuda
}
```

## Troubleshooting

| Problem | Check |
|---------|-------|
| "undefined reference to `cudaMalloc`" | Is BITNET_CPP_DIR set? Does it point to built C++? |
| "libllama.so not found" | Run `cargo run -p xtask -- setup-cpp-auto` |
| Wrong GPU backend | Set `BITNET_GPU_BACKEND` or ensure only one backend installed |
| Linker can't find lib at runtime | Missing RPATH - check `-Wl,-rpath,` directives |

## How to Add Backend Detection

**Pseudocode template**:

```rust
enum GpuBackend { Cuda, Rocm, Metal }

fn detect_backend() -> Option<GpuBackend> {
    // Priority: explicit env var → system detection → default
    if let Ok(backend) = env::var("BITNET_GPU_BACKEND") {
        return Some(GpuBackend::from_str(&backend)?);
    }
    
    // Auto-detect
    if Command::new("nvidia-smi").output().is_ok() {
        return Some(GpuBackend::Cuda);
    }
    if Command::new("rocm-smi").output().is_ok() {
        return Some(GpuBackend::Rocm);
    }
    
    None  // No GPU detected
}

fn link_backend(backend: GpuBackend) {
    match backend {
        GpuBackend::Cuda => {
            println!("cargo:rustc-link-search=/usr/local/cuda/lib64");
            println!("cargo:rustc-link-lib=cuda");
            // ... other CUDA libs
        }
        GpuBackend::Rocm => {
            println!("cargo:rustc-link-search=/opt/rocm/lib");
            println!("cargo:rustc-link-lib=amdhip64");
            // ... other ROCm libs
        }
        GpuBackend::Metal => {
            println!("cargo:rustc-link-lib=framework=Metal");
        }
    }
}
```

## See Also

- Full analysis: `/docs/reference/BUILD_RS_LIBRARY_DISCOVERY_AND_LINKING.md`
- Environment variables: `/docs/environment-variables.md`
- Device features: `crates/bitnet-kernels/src/device_features.rs`
- C++ setup: `xtask/src/cpp_setup_auto.rs`
