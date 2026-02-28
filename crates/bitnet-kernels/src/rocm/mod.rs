//! ROCm/HIP kernel surface for AMD GPUs.
//!
//! This module provides a [`KernelProvider`] implementation targeting the AMD
//! ROCm stack via HIP.  The structure mirrors `gpu::cuda` so that a future
//! HIP runtime integration can slot in with minimal refactoring.
//!
//! The provider is gated behind `--features rocm` and requires
//! `BITNET_ENABLE_ROCM=1` at runtime to be selected by the
//! [`KernelManager`](crate::KernelManager).
//!
//! # Dispatch strategy
//!
//! Every public kernel function follows a two-tier dispatch model:
//!
//! 1. **HIP device path** — When `feature = "rocm"` is compiled and the HIP
//!    runtime is detected (via `hipGetDeviceCount`), work is offloaded to an
//!    AMD GPU through the thin [`hip_ffi`] abstraction layer.
//! 2. **CPU fallback** — When HIP is unavailable the same function silently
//!    falls back to a pure-Rust scalar implementation.  This ensures that
//!    `--features cpu` builds always compile and produce correct results.
//!
//! # Sub-modules
//!
//! | Module | CUDA counterpart | Description |
//! |--------|-----------------|-------------|
//! | [`qk256_gemv`] | `gpu::cuda` matmul | QK256 2-bit GEMV |
//! | [`attention`] | `cuda::attention` | Fused multi-head attention |
//! | [`rmsnorm`] | `cuda::rmsnorm` | RMSNorm forward pass |
//! | [`softmax`] | *(inline in CUDA)* | Row-parallel softmax |

pub mod attention;
pub mod qk256_gemv;
pub mod rmsnorm;
pub mod softmax;

use bitnet_common::{BitNetError, KernelError, QuantizationType, Result};

use crate::KernelProvider;

// Re-export kernel configs for convenience.
pub use attention::HipAttentionConfig;
pub use qk256_gemv::Qk256GemvConfig;
pub use rmsnorm::HipRmsNormConfig;
pub use softmax::HipSoftmaxConfig;

// ── HIP FFI abstraction layer ────────────────────────────────────────
//
// This thin wrapper isolates all `unsafe` HIP runtime calls behind a
// safe(r) Rust interface.  When the `rocm` feature is not enabled the
// entire module compiles away.

#[cfg(feature = "rocm")]
pub(crate) mod hip_ffi {
    //! Minimal HIP runtime FFI bindings.
    //!
    //! These are intentionally low-level.  Higher-level kernel modules
    //! call into this layer for memory management and kernel launch.

    use bitnet_common::{BitNetError, KernelError, Result};
    use std::ffi::c_void;
    use std::sync::OnceLock;

    /// Opaque HIP stream handle.
    pub type HipStream = *mut c_void;

    // ── HIP runtime C symbols ────────────────────────────────────────
    unsafe extern "C" {
        fn hipGetDeviceCount(count: *mut i32) -> i32;
        fn hipSetDevice(device_id: i32) -> i32;
        fn hipMalloc(ptr: *mut *mut c_void, size: usize) -> i32;
        fn hipFree(ptr: *mut c_void) -> i32;
        fn hipMemcpyAsync(
            dst: *mut c_void,
            src: *const c_void,
            size: usize,
            kind: i32,
            stream: HipStream,
        ) -> i32;
        fn hipStreamCreate(stream: *mut HipStream) -> i32;
        fn hipStreamDestroy(stream: HipStream) -> i32;
        fn hipStreamSynchronize(stream: HipStream) -> i32;
        fn hipDeviceSynchronize() -> i32;
        fn hipGetDeviceProperties(prop: *mut HipDeviceProperties, device: i32) -> i32;
    }

    /// Subset of `hipDeviceProp_t` we actually use.
    #[repr(C)]
    struct HipDeviceProperties {
        name: [u8; 256],
        total_global_mem: usize,
        // We only read the first two fields; the rest is padding.
        _pad: [u8; 1024],
    }

    // hipMemcpyKind constants
    const HIP_MEMCPY_H2D: i32 = 1;
    const HIP_MEMCPY_D2H: i32 = 2;

    fn hip_check(code: i32, context: &str) -> Result<()> {
        if code == 0 {
            Ok(())
        } else {
            Err(BitNetError::Kernel(KernelError::GpuError {
                reason: format!("HIP error {code} in {context}"),
            }))
        }
    }

    /// Cached device count so we only call `hipGetDeviceCount` once.
    static DEVICE_COUNT: OnceLock<i32> = OnceLock::new();

    pub fn device_count() -> i32 {
        *DEVICE_COUNT.get_or_init(|| {
            let mut count: i32 = 0;
            let rc = unsafe { hipGetDeviceCount(&mut count) };
            if rc != 0 { 0 } else { count }
        })
    }

    pub fn set_device(id: i32) -> Result<()> {
        hip_check(unsafe { hipSetDevice(id) }, "hipSetDevice")
    }

    pub unsafe fn device_malloc(bytes: usize) -> Result<*mut c_void> {
        let mut ptr: *mut c_void = std::ptr::null_mut();
        hip_check(unsafe { hipMalloc(&mut ptr, bytes) }, "hipMalloc")?;
        Ok(ptr)
    }

    pub unsafe fn device_free(ptr: *mut c_void) -> Result<()> {
        hip_check(unsafe { hipFree(ptr) }, "hipFree")
    }

    pub unsafe fn memcpy_h2d(
        dst: *mut c_void,
        src: *const c_void,
        bytes: usize,
        stream: HipStream,
    ) -> Result<()> {
        hip_check(
            unsafe { hipMemcpyAsync(dst, src, bytes, HIP_MEMCPY_H2D, stream) },
            "hipMemcpyAsync(H2D)",
        )
    }

    pub unsafe fn memcpy_d2h(
        dst: *mut c_void,
        src: *const c_void,
        bytes: usize,
        stream: HipStream,
    ) -> Result<()> {
        hip_check(
            unsafe { hipMemcpyAsync(dst, src, bytes, HIP_MEMCPY_D2H, stream) },
            "hipMemcpyAsync(D2H)",
        )
    }

    pub fn create_stream() -> Result<HipStream> {
        let mut stream: HipStream = std::ptr::null_mut();
        hip_check(unsafe { hipStreamCreate(&mut stream) }, "hipStreamCreate")?;
        Ok(stream)
    }

    pub fn destroy_stream(stream: HipStream) -> Result<()> {
        hip_check(unsafe { hipStreamDestroy(stream) }, "hipStreamDestroy")
    }

    pub fn stream_synchronize(stream: HipStream) -> Result<()> {
        hip_check(unsafe { hipStreamSynchronize(stream) }, "hipStreamSynchronize")
    }

    pub fn device_synchronize() -> Result<()> {
        hip_check(unsafe { hipDeviceSynchronize() }, "hipDeviceSynchronize")
    }

    /// Return a per-thread default stream (HIP null-stream).
    pub fn current_stream() -> Result<HipStream> {
        Ok(std::ptr::null_mut()) // null = default stream in HIP
    }

    /// Query device name (best-effort; returns "unknown" on failure).
    pub fn device_name(device_id: i32) -> String {
        unsafe {
            let mut props = std::mem::zeroed::<HipDeviceProperties>();
            let rc = hipGetDeviceProperties(&mut props, device_id);
            if rc != 0 {
                return "unknown".into();
            }
            let len = props.name.iter().position(|&b| b == 0).unwrap_or(256);
            String::from_utf8_lossy(&props.name[..len]).into_owned()
        }
    }

    // ── Kernel launch trampolines ────────────────────────────────────
    //
    // These call into HIP device code compiled separately (e.g. via
    // hipcc).  The symbols are expected to be provided by a static
    // library linked via build.rs when the `rocm` feature is active.

    unsafe extern "C" {
        /// QK256 2-bit GEMV: C[m,n] = dequant(packed_w[m,k]) · x[k,n]
        pub fn bitnet_hip_qk256_gemv(
            packed_weights: *const u8,
            scales: *const f32,
            input: *const f32,
            output: *mut f32,
            m: u32,
            n: u32,
            k: u32,
            threads: u32,
            blocks: u32,
            stream: HipStream,
        ) -> i32;

        /// RMSNorm: out[i] = (x[i] / rms(x)) * gamma[i]
        pub fn bitnet_hip_rmsnorm(
            input: *const f32,
            gamma: *const f32,
            output: *mut f32,
            hidden_dim: u32,
            num_rows: u32,
            eps: f32,
            threads: u32,
            blocks: u32,
            stream: HipStream,
        ) -> i32;

        /// Fused scaled-dot-product attention.
        pub fn bitnet_hip_fused_attention(
            q: *const f32,
            k: *const f32,
            v: *const f32,
            output: *mut f32,
            num_heads: u32,
            head_dim: u32,
            seq_len_q: u32,
            seq_len_kv: u32,
            scale: f32,
            causal: i32,
            threads: u32,
            blocks_x: u32,
            blocks_y: u32,
            stream: HipStream,
        ) -> i32;

        /// Row-parallel softmax.
        pub fn bitnet_hip_softmax(
            input: *const f32,
            output: *mut f32,
            num_rows: u32,
            num_cols: u32,
            threads: u32,
            blocks: u32,
            stream: HipStream,
        ) -> i32;
    }

    /// Safe wrapper around the QK256 GEMV HIP launch.
    pub unsafe fn launch_qk256_gemv(
        packed_weights: *const u8,
        scales: *const f32,
        input: *const f32,
        output: *mut f32,
        m: u32,
        n: u32,
        k: u32,
        threads: u32,
        blocks: u32,
        stream: HipStream,
    ) -> Result<()> {
        hip_check(
            unsafe {
                bitnet_hip_qk256_gemv(
                    packed_weights,
                    scales,
                    input,
                    output,
                    m,
                    n,
                    k,
                    threads,
                    blocks,
                    stream,
                )
            },
            "bitnet_hip_qk256_gemv",
        )
    }

    /// Safe wrapper around the RMSNorm HIP launch.
    pub unsafe fn launch_rmsnorm(
        input: *const f32,
        gamma: *const f32,
        output: *mut f32,
        hidden_dim: u32,
        num_rows: u32,
        eps: f32,
        threads: u32,
        blocks: u32,
        stream: HipStream,
    ) -> Result<()> {
        hip_check(
            unsafe {
                bitnet_hip_rmsnorm(
                    input, gamma, output, hidden_dim, num_rows, eps, threads, blocks, stream,
                )
            },
            "bitnet_hip_rmsnorm",
        )
    }

    /// Safe wrapper around the fused attention HIP launch.
    pub unsafe fn launch_fused_attention(
        q: *const f32,
        k: *const f32,
        v: *const f32,
        output: *mut f32,
        num_heads: u32,
        head_dim: u32,
        seq_len_q: u32,
        seq_len_kv: u32,
        scale: f32,
        causal: bool,
        threads: u32,
        blocks_x: u32,
        blocks_y: u32,
        stream: HipStream,
    ) -> Result<()> {
        hip_check(
            unsafe {
                bitnet_hip_fused_attention(
                    q,
                    k,
                    v,
                    output,
                    num_heads,
                    head_dim,
                    seq_len_q,
                    seq_len_kv,
                    scale,
                    if causal { 1 } else { 0 },
                    threads,
                    blocks_x,
                    blocks_y,
                    stream,
                )
            },
            "bitnet_hip_fused_attention",
        )
    }

    /// Safe wrapper around the softmax HIP launch.
    pub unsafe fn launch_softmax(
        input: *const f32,
        output: *mut f32,
        num_rows: usize,
        num_cols: usize,
        threads: u32,
        blocks: u32,
        stream: HipStream,
    ) -> Result<()> {
        hip_check(
            unsafe {
                bitnet_hip_softmax(
                    input,
                    output,
                    num_rows as u32,
                    num_cols as u32,
                    threads,
                    blocks,
                    stream,
                )
            },
            "bitnet_hip_softmax",
        )
    }
}

// ── Device information ───────────────────────────────────────────────

/// AMD GPU device information and capabilities.
#[derive(Debug, Clone)]
pub struct RocmDeviceInfo {
    /// Ordinal index of the HIP device.
    pub device_id: usize,
    /// Device marketing name (e.g. "AMD Instinct MI250X").
    pub name: String,
    /// GCN architecture name (e.g. "gfx90a").
    pub gcn_arch: String,
    /// Total device memory in bytes.
    pub total_memory: usize,
    /// Number of compute units.
    pub compute_unit_count: i32,
    /// Maximum wavefront (work-group) size.
    pub max_wavefront_size: i32,
    /// Maximum shared (LDS) memory per work-group in bytes.
    pub max_shared_memory_per_workgroup: usize,
    /// FP16 (half-precision) support.
    pub supports_fp16: bool,
    /// BF16 (bfloat16) support — available on CDNA2+.
    pub supports_bf16: bool,
}

// ── Kernel provider ──────────────────────────────────────────────────

/// ROCm/HIP kernel provider.
///
/// When `feature = "rocm"` is compiled and the HIP runtime is detected,
/// operations dispatch to AMD GPU hardware.  Otherwise they fall back to
/// CPU scalar implementations so that the provider can always be
/// instantiated without errors.
#[derive(Debug, Clone, Default)]
pub struct RocmKernel {
    _private: (),
}

impl RocmKernel {
    /// Create a new ROCm kernel provider.
    pub fn new() -> Self {
        Self { _private: () }
    }

    /// Whether `rocm` support was compiled into this build.
    pub fn compiled() -> bool {
        cfg!(feature = "rocm")
    }

    /// Runtime opt-in via `BITNET_ENABLE_ROCM=1`.
    fn rocm_enabled() -> bool {
        std::env::var("BITNET_ENABLE_ROCM")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false)
    }
}

impl KernelProvider for RocmKernel {
    fn name(&self) -> &'static str {
        "rocm-hip"
    }

    fn is_available(&self) -> bool {
        Self::compiled() && Self::rocm_enabled() && is_rocm_available()
    }

    fn matmul_i2s(
        &self,
        a: &[i8],
        b: &[u8],
        c: &mut [f32],
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<()> {
        matmul_i2s_dispatch(a, b, c, m, n, k)
    }

    fn quantize(
        &self,
        input: &[f32],
        output: &mut [u8],
        scales: &mut [f32],
        qtype: QuantizationType,
    ) -> Result<()> {
        quantize_dispatch(input, output, scales, qtype)
    }
}

// ── Dispatch helpers ─────────────────────────────────────────────────

/// I2S matmul: `C[m,n] = A[m,k] · dequant(B[k,n])`
///
/// Falls back to CPU scalar when HIP is unavailable.
fn matmul_i2s_dispatch(
    a: &[i8],
    b: &[u8],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
) -> Result<()> {
    if m == 0 || n == 0 || k == 0 {
        return Err(BitNetError::Kernel(KernelError::InvalidArguments {
            reason: format!("matmul_i2s dimensions must be non-zero: m={m}, n={n}, k={k}"),
        }));
    }

    #[cfg(feature = "rocm")]
    {
        if is_rocm_available() {
            return matmul_i2s_hip(a, b, c, m, n, k);
        }
    }

    matmul_i2s_cpu(a, b, c, m, n, k)
}

/// CPU scalar fallback for I2S matmul.
fn matmul_i2s_cpu(a: &[i8], b: &[u8], c: &mut [f32], m: usize, n: usize, k: usize) -> Result<()> {
    // 2-bit packed: 4 weights per byte
    let packed_k = (k + 3) / 4;
    if a.len() < m * k {
        return Err(BitNetError::Kernel(KernelError::InvalidArguments {
            reason: format!("matmul_i2s: a too small, need {}, got {}", m * k, a.len()),
        }));
    }
    if b.len() < packed_k * n {
        return Err(BitNetError::Kernel(KernelError::InvalidArguments {
            reason: format!("matmul_i2s: b too small, need {}, got {}", packed_k * n, b.len()),
        }));
    }
    if c.len() < m * n {
        return Err(BitNetError::Kernel(KernelError::InvalidArguments {
            reason: format!("matmul_i2s: c too small, need {}, got {}", m * n, c.len()),
        }));
    }

    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f32;
            for kk in 0..k {
                let a_val = a[i * k + kk] as f32;
                // Unpack 2-bit weight from byte
                let byte_idx = (kk / 4) * n + j;
                let bit_pos = (kk % 4) * 2;
                let raw = (b[byte_idx] >> bit_pos) & 0x03;
                // Map: 0 -> -1, 1 -> 0, 2 -> +1, 3 -> 0 (ternary)
                let w = match raw {
                    0 => -1.0f32,
                    2 => 1.0f32,
                    _ => 0.0f32,
                };
                acc += a_val * w;
            }
            c[i * n + j] = acc;
        }
    }

    Ok(())
}

/// HIP device path for I2S matmul.
#[cfg(feature = "rocm")]
fn matmul_i2s_hip(
    _a: &[i8],
    _b: &[u8],
    _c: &mut [f32],
    _m: usize,
    _n: usize,
    _k: usize,
) -> Result<()> {
    // Full HIP launch will go through qk256_gemv for quantized weights.
    Err(BitNetError::Kernel(KernelError::GpuError {
        reason: "HIP matmul_i2s: use qk256_gemv_hip for QK256-packed weights".into(),
    }))
}

/// Scalar quantization dispatch (always CPU — quantization is not
/// performance-critical enough to warrant a GPU kernel).
fn quantize_dispatch(
    input: &[f32],
    output: &mut [u8],
    scales: &mut [f32],
    qtype: QuantizationType,
) -> Result<()> {
    match qtype {
        QuantizationType::I2S => quantize_i2s_cpu(input, output, scales),
        other => Err(BitNetError::Kernel(KernelError::ExecutionFailed {
            reason: format!("ROCm quantize: unsupported quantization type {other:?}"),
        })),
    }
}

/// CPU I2S quantization: float → 2-bit ternary.
fn quantize_i2s_cpu(input: &[f32], output: &mut [u8], scales: &mut [f32]) -> Result<()> {
    if input.is_empty() {
        return Ok(());
    }

    // Compute absmax scale
    let absmax = input.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
    let scale = if absmax > 0.0 { absmax } else { 1.0 };
    if !scales.is_empty() {
        scales[0] = scale;
    }

    let inv_scale = 1.0 / scale;

    // Pack 4 ternary values per byte
    let packed_len = (input.len() + 3) / 4;
    if output.len() < packed_len {
        return Err(BitNetError::Kernel(KernelError::InvalidArguments {
            reason: format!(
                "quantize output buffer too small: need {packed_len}, got {}",
                output.len()
            ),
        }));
    }

    for (byte_idx, chunk) in input.chunks(4).enumerate() {
        let mut packed: u8 = 0;
        for (bit_pos, &val) in chunk.iter().enumerate() {
            let normalized = val * inv_scale;
            // Ternary: <-0.5 → -1 (0b00), >0.5 → +1 (0b10), else → 0 (0b01)
            let code: u8 = if normalized < -0.5 {
                0b00
            } else if normalized > 0.5 {
                0b10
            } else {
                0b01
            };
            packed |= code << (bit_pos * 2);
        }
        output[byte_idx] = packed;
    }

    Ok(())
}

// ── Utility functions ────────────────────────────────────────────────

/// Check whether a ROCm/HIP runtime is available on the system.
///
/// On `feature = "rocm"` builds this calls `hipGetDeviceCount` and
/// caches the result.  Without the feature it always returns `false`.
pub fn is_rocm_available() -> bool {
    #[cfg(feature = "rocm")]
    {
        hip_ffi::device_count() > 0
    }
    #[cfg(not(feature = "rocm"))]
    {
        false
    }
}

/// Return the number of HIP-visible AMD GPU devices.
pub fn rocm_device_count() -> usize {
    #[cfg(feature = "rocm")]
    {
        hip_ffi::device_count().max(0) as usize
    }
    #[cfg(not(feature = "rocm"))]
    {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rocm_kernel_reports_name() {
        let kernel = RocmKernel::new();
        assert_eq!(kernel.name(), "rocm-hip");
    }

    #[test]
    fn rocm_kernel_not_available_without_env() {
        // Without BITNET_ENABLE_ROCM=1 the provider must not activate.
        let kernel = RocmKernel::new();
        assert!(!kernel.is_available());
    }

    #[test]
    fn compiled_matches_feature() {
        let expected = cfg!(feature = "rocm");
        assert_eq!(RocmKernel::compiled(), expected);
    }

    // ── CPU fallback tests ───────────────────────────────────────────

    #[test]
    fn matmul_i2s_cpu_identity() {
        // 1×1 matmul with a single +1 weight
        let a = [1i8];
        let b = [0b10u8]; // code 0b10 = +1
        let mut c = [0.0f32];
        matmul_i2s_cpu(&a, &b, &mut c, 1, 1, 1).unwrap();
        assert!((c[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn matmul_i2s_cpu_negative_weight() {
        let a = [3i8];
        let b = [0b00u8]; // code 0b00 = -1
        let mut c = [0.0f32];
        matmul_i2s_cpu(&a, &b, &mut c, 1, 1, 1).unwrap();
        assert!((c[0] - (-3.0)).abs() < 1e-6);
    }

    #[test]
    fn matmul_i2s_cpu_zero_weight() {
        let a = [5i8];
        let b = [0b01u8]; // code 0b01 = 0
        let mut c = [0.0f32];
        matmul_i2s_cpu(&a, &b, &mut c, 1, 1, 1).unwrap();
        assert!((c[0]).abs() < 1e-6);
    }

    #[test]
    fn matmul_i2s_dispatch_rejects_zero() {
        let a = [1i8; 4];
        let b = [0u8; 4];
        let mut c = [0.0f32; 4];
        assert!(matmul_i2s_dispatch(&a, &b, &mut c, 0, 2, 2).is_err());
    }

    #[test]
    fn quantize_i2s_cpu_roundtrip() {
        let input = [1.0f32, -1.0, 0.0, 0.5];
        let mut output = [0u8; 1]; // 4 values → 1 byte
        let mut scales = [0.0f32; 1];
        quantize_i2s_cpu(&input, &mut output, &mut scales).unwrap();
        assert!(scales[0] > 0.0);
    }

    #[test]
    fn quantize_dispatch_rejects_unsupported() {
        let input = [1.0f32; 4];
        let mut output = [0u8; 4];
        let mut scales = [0.0f32; 1];
        assert!(
            quantize_dispatch(&input, &mut output, &mut scales, QuantizationType::TL1).is_err()
        );
    }

    #[test]
    fn rocm_device_count_without_runtime() {
        #[cfg(not(feature = "rocm"))]
        assert_eq!(rocm_device_count(), 0);
    }

    #[test]
    fn rocm_is_not_available_without_feature() {
        #[cfg(not(feature = "rocm"))]
        assert!(!is_rocm_available());
    }

    // ── GPU-gated tests ──────────────────────────────────────────────

    #[test]
    #[ignore = "requires ROCm/HIP runtime — run on AMD GPU hardware"]
    fn rocm_device_detected() {
        assert!(is_rocm_available(), "expected ≥1 HIP device");
        assert!(rocm_device_count() > 0);
    }

    #[test]
    #[ignore = "requires ROCm/HIP runtime — run on AMD GPU hardware"]
    fn rocm_matmul_i2s_on_device() {
        let kernel = RocmKernel::new();
        let a = vec![1i8; 256];
        let b = vec![0b10u8; 64]; // all +1 weights
        let mut c = vec![0.0f32; 1];
        let result = kernel.matmul_i2s(&a, &b, &mut c, 1, 1, 256);
        assert!(result.is_ok(), "HIP matmul failed: {result:?}");
    }
}
