# GPU Backend Safety Audit

## Audit Scope

All GPU-related code in the bitnet-rs workspace. Unlike many GPU projects that
use separate per-backend crates, bitnet-rs centralizes GPU code in these
locations:

| Location | Description |
|----------|-------------|
| `bitnet-kernels/src/gpu/cuda.rs` | CUDA kernel launches via cudarc |
| `bitnet-kernels/src/gpu/mixed_precision.rs` | FP16/BF16 mixed-precision CUDA kernels |
| `bitnet-kernels/src/gpu/validation.rs` | GPU memory leak detection (raw CUDA driver API) |
| `bitnet-kernels/src/gpu/memory_optimization.rs` | GPU memory pool (safe abstractions) |
| `bitnet-kernels/src/gpu/benchmark.rs` | GPU vs CPU benchmarking (safe) |
| `bitnet-kernels/src/ffi/bridge.rs` | C++ FFI bridge for legacy kernel calls |
| `bitnet-kernels/src/cpu/x86.rs` | AVX2/AVX-512 SIMD intrinsics |
| `bitnet-kernels/src/cpu/arm.rs` | NEON SIMD intrinsics |
| `bitnet-device-probe/src/lib.rs` | Vulkan device enumeration |
| `bitnet-sys/src/wrapper.rs` | llama.cpp/bitnet.cpp FFI wrappers |
| `bitnet-ffi/src/c_api.rs` | C-callable API surface |
| `bitnet-ffi/src/llama_compat.rs` | llama.cpp compatibility layer |
| `bitnet-ffi/src/memory.rs` | Tracking allocator, memory pool |
| `bitnet-ffi/src/streaming.rs` | Streaming callback wrapper |
| `bitnet-ggml-ffi/src/lib.rs` | GGML quantization FFI bindings |

## Audit Date

2025-07-16

## Feature Gate Convention

GPU code must always use the unified predicate:
```rust
#[cfg(any(feature = "gpu", feature = "cuda"))]
```
Never use `#[cfg(feature = "cuda")]` alone. Runtime checks use
`bitnet_kernels::device_features::{gpu_compiled, gpu_available_runtime}`.

---

## Safety Categories

### 1. Memory Safety

- [x] All CUDA buffer allocations use cudarc safe wrappers (`alloc_zeros`,
      `memcpy_stod`, `memcpy_dtov`) — no raw `cuMalloc` in kernel launch paths
- [x] `validation.rs` uses raw `cuMemAlloc_v2`/`cuMemFree_v2` but cleans up on
      every error path (lines 386-389, 406-408)
- [x] Host-device copies validated by cudarc type system (`CudaSlice<T>`)
- [x] `bitnet-sys` wrapper validates all pointer returns from C++ for null
- [x] `bitnet-ffi/memory.rs` `TrackingAllocator` delegates to `System` allocator
- [ ] **Finding F-1**: `validation.rs` raw CUDA pointer arithmetic has no
      overflow check on `ALLOC_SIZE * ITERATIONS` (currently safe at 50 MB, but
      would be fragile if constants changed)

### 2. Unsafe Block Inventory

#### `bitnet-kernels/src/gpu/cuda.rs`

| Line | Operation | Justification |
|------|-----------|---------------|
| 144 | `cu_device::total_mem()` | CUDA driver FFI; device handle from cudarc context is valid |
| 246 | `builder.launch(cfg)` | CUDA kernel launch; arguments set by cudarc typed builder |
| 322 | `builder.launch(cfg)` | CUDA kernel launch for matmul; same pattern as line 246 |

#### `bitnet-kernels/src/gpu/mixed_precision.rs`

| Line | Operation | Justification |
|------|-----------|---------------|
| 471 | `builder_a.launch(cfg_convert)` | FP32→FP16 conversion kernel launch |
| 488 | `builder_b.launch(cfg_convert_b)` | FP32→FP16 conversion kernel launch (matrix B) |
| 515 | `builder_matmul.launch(cfg_matmul)` | FP16 matmul kernel launch |
| 533 | `builder_c.launch(cfg_convert_c)` | FP16→FP32 conversion kernel launch |
| 669 | `builder_a.launch(cfg_convert)` | FP32→BF16 conversion kernel launch |
| 686 | `builder_b.launch(cfg_convert_b)` | FP32→BF16 conversion kernel launch (matrix B) |
| 713 | `builder_matmul.launch(cfg_matmul)` | BF16 matmul kernel launch |
| 731 | `builder_c.launch(cfg_convert_c)` | BF16→FP32 conversion kernel launch |

#### `bitnet-kernels/src/gpu/validation.rs`

| Line | Operation | Justification |
|------|-----------|---------------|
| 354-470 | Large `unsafe` block | Raw CUDA driver API (`cuMemAlloc_v2`, `cuMemFree_v2`, `cuMemGetInfo_v2`) for memory leak detection; all allocations tracked and freed |

#### `bitnet-kernels/src/ffi/bridge.rs`

| Line | Operation | Justification |
|------|-----------|---------------|
| 42 | `bitnet_cpp_init()` | C++ library initialization; no preconditions |
| 45 | `bitnet_cpp_cleanup()` | C++ library cleanup; idempotent |
| 48 | `bitnet_cpp_is_available() != 0` | Pure query, no side effects |
| 59-68 | `bitnet_cpp_matmul_i2s(...)` | FFI matmul; slices guarantee valid pointers and lengths |
| 78-88 | `bitnet_cpp_quantize(...)` | FFI quantize; slice lengths passed as bounds |
| 93-99 | `bitnet_cpp_get_last_error()` | Returns C string pointer; null-checked before use |

#### `bitnet-kernels/src/cpu/x86.rs` (SIMD)

| Line | Operation | Justification |
|------|-----------|---------------|
| 41 | `self.matmul_i2s_avx2(...)` | AVX2 availability checked by `is_available()` guard |
| 67 | `self.quantize_tl2_avx2(...)` | AVX2 availability checked |
| 107 | `self.dequantize_qk256_avx2(...)` | AVX2 availability checked |
| 205 | `self.matmul_i2s_avx512(...)` | AVX-512 availability checked |
| 220 | `self.quantize_tl2_avx512(...)` | AVX-512 availability checked |
| 247-710+ | SIMD intrinsic functions | `#[target_feature]` gated; callers verify availability |

#### `bitnet-kernels/src/cpu/arm.rs` (NEON)

| Line | Operation | Justification |
|------|-----------|---------------|
| 73 | `self.matmul_i2s_neon(...)` | NEON availability checked |
| 88-90 | `self.quantize_tl{1,2}_neon(...)` | NEON availability checked |
| 99-616 | NEON intrinsic functions | `#[target_feature]` gated |

#### `bitnet-device-probe/src/lib.rs`

| Line | Operation | Justification |
|------|-----------|---------------|
| 260 | `ash::Entry::load()` | Vulkan loader dynamic linking — already has `// SAFETY:` ✓ |
| 270 | `entry.create_instance(...)` | Stack-local create_info — already has `// SAFETY:` ✓ |
| 276 | `instance.enumerate_physical_devices()` | Valid instance — already has `// SAFETY:` ✓ |
| 281 | `instance.destroy_instance(None)` | Valid instance, no further use — already has `// SAFETY:` ✓ |

#### `bitnet-sys/src/wrapper.rs`

| Line | Operation | Justification |
|------|-----------|---------------|
| 34-48 | `llama_backend_{init,free}`, `bitnet_backend_init` | Library lifecycle; called once |
| 69-72 | `llama_model_default_params`, `llama_load_model_from_file` | C API; path validated via `CString` |
| 83-93 | `llama_n_vocab`, `llama_n_ctx_train`, `llama_n_embd` | Pure queries on valid model ptr |
| 108 | `llama_free_model` | Drop impl; ptr guaranteed non-null by constructor |
| 116-117 | `unsafe impl Send/Sync for Model` | Model is ref-counted in C++; thread-safe |
| 127-168 | Context creation, tokenization | FFI with null-checked returns |
| 215-264 | Batch init, decode, logits access | Standard llama.cpp usage pattern |
| 409-438 | `bitnet_model_new_from_file`, Send/Sync | Same pattern as Model |
| 450-521 | BitNet context, tokenization | Mirrors llama.cpp pattern |

#### `bitnet-ffi/src/c_api.rs`

| Line | Operation | Justification |
|------|-----------|---------------|
| 38-1154 | `#[unsafe(no_mangle)]` extern fns | Public C API; all pointer args null-checked |
| 143-208 | `CStr::from_ptr`, model loading | Null-checked, `CStr` validates UTF-8 |
| 326, 425 | Output buffer writes | Buffer length validated before write |

#### `bitnet-ffi/src/memory.rs`

| Line | Operation | Justification |
|------|-----------|---------------|
| 95-107 | `GlobalAlloc` impl | Delegates to `System` allocator; tracks sizes atomically |
| 159-169 | `MemoryPool::deallocate` | Caller must pass matching `(ptr, size)` |

#### `bitnet-ffi/src/streaming.rs`

| Line | Operation | Justification |
|------|-----------|---------------|
| 171-172 | `unsafe impl Send/Sync for CallbackWrapper` | Callback stored behind `Arc<Mutex>` |

#### `bitnet-ggml-ffi/src/lib.rs`

| Line | Operation | Justification |
|------|-----------|---------------|
| 8 | `unsafe extern "C"` block | GGML C function declarations |
| 39-67 | Safe wrappers around `bitnet_iq2s_*` | Pure queries, no pointers |
| 94-114 | `dequantize_row_iq2_s`, `quantize_iq2_s` | Raw pointer FFI; caller must ensure valid buffers |

### 3. Kernel Source Safety

- [x] CUDA kernels compiled from static string (`include_str!`) — no
      user-controlled input in kernel source
- [x] Kernel names are compile-time constants loaded via
      `module.load_function("bitnet_matmul_i2s")` etc.
- [x] No OpenCL build option injection (no OpenCL backend exists)
- [x] PTX compilation happens once at kernel initialization, not per-request

### 4. Error Handling

- [x] All cudarc kernel launches return `Result` and map to `KernelError::GpuError`
- [x] `validation.rs` checks CUDA driver return codes for every raw API call
- [x] FFI bridge checks return codes and calls `get_last_error()` for diagnostics
- [x] Null pointer returns from C++ are detected and converted to `Err`
- [ ] **Finding F-2**: `bitnet-sys/wrapper.rs` line 264 creates
      `slice::from_raw_parts` from `llama_get_logits` return — if the C library
      returns null (e.g., after context invalidation), this would be UB. The null
      check at line 263 correctly guards this.
- [x] OOM from `alloc_zeros` returns `Err`, not panic

### 5. Concurrency Safety

- [x] `CudaKernel` holds `Arc<CudaContext>` and `Arc<CudaStream>` — single
      stream per kernel instance, no concurrent command queue access
- [x] `Model` and `BitnetModel` in `bitnet-sys` have `unsafe impl Send + Sync`
      justified by C++ thread-safety guarantees
- [x] `CallbackWrapper` in `bitnet-ffi/streaming.rs` uses `Arc<Mutex>` before
      the `unsafe impl Send + Sync`
- [x] GPU validation tests are `#[serial]` or `#[ignore]` to prevent concurrent
      GPU context creation
- [ ] **Finding F-3**: No documented invariant that `CudaKernel` must not be
      shared across threads without external synchronization — `CudaStream`
      serializes commands, but two threads calling `launch_matmul` on the same
      `CudaKernel` could interleave `memcpy_stod` and `launch` calls

---

## Findings Summary

| ID | Severity | Location | Description |
|----|----------|----------|-------------|
| F-1 | Low | `validation.rs:349-350` | `ALLOC_SIZE * ITERATIONS` not overflow-checked; safe at current values |
| F-2 | Info | `bitnet-sys/wrapper.rs:263-264` | `slice::from_raw_parts` guarded by null check — correct, but fragile if guard removed |
| F-3 | Medium | `bitnet-kernels/src/gpu/cuda.rs` | `CudaKernel` is `Send` (via `Arc`) but no `Sync` impl documented; concurrent use from multiple threads could interleave stream operations |

## Recommendations

1. **Add `// SAFETY:` comments to all unsafe blocks** — completed in this audit
   for GPU-specific crates; FFI crates (`bitnet-ffi`, `bitnet-sys`) should be
   addressed in a follow-up
2. **Enable `clippy::undocumented_unsafe_blocks`** lint workspace-wide to enforce
   SAFETY comments going forward
3. **Add fuzzing for kernel argument validation** — ensure dimension mismatches,
   zero-size inputs, and integer overflow in grid/block calculations are caught
4. **Consider `gpu-alloc` crate** for more robust GPU memory management if
   moving beyond cudarc's built-in allocator
5. **Document `CudaKernel` thread-safety invariants** — either add `Sync`
   with internal locking, or document that instances must not be shared
6. **Pin CUDA driver API usage** — the raw `cuMemAlloc_v2`/`cuMemFree_v2` in
   `validation.rs` bypasses cudarc's safety layer; consider migrating to
   cudarc's `CudaSlice` allocator for consistency
7. **Add safety regression tests** — boundary conditions, zero-size allocations,
   and concurrent access patterns (see `gpu_safety_tests.rs`)
