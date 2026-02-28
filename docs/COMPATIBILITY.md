# Compatibility Matrix

Hardware, OS, and driver requirements for each BitNet-rs backend.

---

## GPU Backend Compatibility

### CUDA (Production)

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| NVIDIA GPU | Compute Capability 6.0 (Pascal) | CC 8.0+ (Ampere/Ada) |
| CUDA Toolkit | 12.0 | 12.x (latest) |
| NVIDIA Driver | 525+ | 535+ |
| cudarc | 0.17.8 | — |
| OS | Linux (glibc 2.17+), Windows 10+ | Ubuntu 22.04/24.04 |
| VRAM | 4 GB (2B model) | 8 GB+ |

**Precision support by compute capability:**

| CC | FP32 | FP16 | Tensor Cores | BF16 |
|----|------|------|-------------|------|
| 6.0–6.2 (Pascal) | ✅ | ✅ | ❌ | ❌ |
| 7.0–7.5 (Volta/Turing) | ✅ | ✅ | ✅ | ❌ |
| 8.0–8.9 (Ampere/Ada) | ✅ | ✅ | ✅ | ✅ |
| 9.0+ (Hopper) | ✅ | ✅ | ✅ | ✅ |

### ROCm / HIP (Weekly CI)

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| AMD GPU | RDNA 2 / CDNA 2 | RDNA 3 / MI250+ |
| ROCm | 5.6 | 6.x |
| OS | Linux only | Ubuntu 22.04 |
| Feature flag | `--features rocm` | — |

### OpenCL / Intel oneAPI (Alpha)

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| Intel GPU | Arc A-series (A580, A750, A770) | A770 16 GB |
| Intel Compute Runtime | OpenCL 3.0 + Level Zero 1.3 | Latest |
| Linux kernel | 6.2 (i915 with Arc support) | 6.5+ |
| OS | Linux only (Ubuntu 22.04/24.04, Fedora 37+) | Ubuntu 24.04 |
| VRAM | 8 GB (2B model) | 16 GB |
| Feature flag | `--features oneapi` | — |
| User groups | `render`, `video` | — |

### Metal / NPU (Experimental)

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| Hardware | Apple Silicon (M1+) | M2 Pro / M3+ |
| macOS | 13 Ventura | 14 Sonoma+ |
| Feature flag | `--features metal` | — |

### Vulkan (Feature-gated)

| Requirement | Notes |
|-------------|-------|
| Feature flag | `--features vulkan` (currently maps to `gpu`) |
| Status | Feature gate exists; no separate Vulkan kernels yet |

---

## CPU Backend Compatibility

### x86_64

| SIMD Level | Feature Flag | Auto-detected | Min CPU |
|------------|-------------|---------------|---------|
| AVX-512 | `--features avx512` | ✅ | Skylake-X / Zen 4 |
| AVX2 | `--features avx2` | ✅ | Haswell / Zen 1 |
| SSE2 | (always) | ✅ | Any x86_64 |

### ARM64

| SIMD Level | Feature Flag | Auto-detected | Min CPU |
|------------|-------------|---------------|---------|
| NEON | `--features neon` | ✅ | Any ARMv8 |

### Generic Fallback

Always available. No SIMD — pure scalar Rust. Works on any platform with a
Rust target.

---

## OS Support Matrix

| OS | CUDA | ROCm | OpenCL (Intel) | Metal | CPU |
|----|------|------|----------------|-------|-----|
| Ubuntu 22.04/24.04 | ✅ | ✅ | ✅ | — | ✅ |
| Fedora 37+ | ✅ | ✅ | ✅ | — | ✅ |
| Other Linux (glibc 2.17+) | ✅ | ⚠️ | ⚠️ | — | ✅ |
| Windows 10/11 | ✅ | ❌ | ❌ | — | ✅ |
| macOS 13+ (Apple Silicon) | — | — | — | ✅ | ✅ |
| WebAssembly | — | — | — | — | ✅¹ |

¹ Via `bitnet-wasm` crate (limited functionality).

---

## Model Format Compatibility

| Format | Status | Notes |
|--------|--------|-------|
| GGUF v2 | ✅ | Full support |
| GGUF v3 | ✅ | Full support including early variants |
| SafeTensors | ✅ | Via `bitnet-models` loader |
| I2_S (BitNet32-F16) | ✅ | 32-element blocks, inline FP16 scales |
| I2_S (QK256) | ✅ | 256-element blocks, ~0.1 tok/s (scalar MVP) |

---

## Tokenizer Compatibility

| Tokenizer | Status |
|-----------|--------|
| GPT-2 (BPE) | ✅ |
| Llama 3 | ✅ |
| SentencePiece | ✅ |
| Tiktoken | ✅ |
| Falcon | ✅ |

---

## Rust Toolchain

| Requirement | Value |
|-------------|-------|
| MSRV | 1.92.0 |
| Edition | Rust 2024 |
| Recommended | `rust-toolchain.toml` pinned version |

---

## Driver Installation Quick Links

- **NVIDIA CUDA:** <https://developer.nvidia.com/cuda-downloads>
- **AMD ROCm:** <https://rocm.docs.amd.com/en/latest/deploy/linux/quick_start.html>
- **Intel Compute Runtime:** <https://github.com/intel/compute-runtime/releases>
- **Intel Arc setup:** See [INTEL_GPU_SETUP.md](INTEL_GPU_SETUP.md)
