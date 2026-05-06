//! QK256 dequantization + GEMV fused CUDA kernel.
//!
//! # Kernel strategy
//!
//! Microsoft BitNet GGUF models in this repo use the GGML I2_S QK256 no-scale
//! layout: each row is block-aligned, each block stores 256 two-bit codes in
//! 64 bytes, and codes map directly through `[-2, -1, 1, 2]`. The fused
//! dequant+GEMV kernel avoids materialising the full FP32 weight matrix by:
//!
//! 1. Reading one packed row without expanding it to an intermediate tensor.
//! 2. Unpacking two-bit codes with bit-shift/mask at the point of use.
//! 3. Mapping each code to its no-scale GGML I2_S FP32 value.
//! 4. Accumulating the dot product in FP32 and writing `[seq_len, n_out]`.
//!
//! This is a correctness-first packed kernel path for CUDA-BITNET-003, not a
//! full inference integration or benchmark claim.

use bitnet_common::{KernelError, Result};
#[cfg(feature = "cuda")]
use std::any::Any;
#[cfg(feature = "cuda")]
use std::sync::Mutex;

#[cfg(feature = "cuda")]
use cudarc::driver::{CudaContext, CudaSlice, LaunchConfig, PushKernelArg};
#[cfg(feature = "cuda")]
use cudarc::nvrtc::{Ptx, compile_ptx};

/// Number of matrix columns encoded by one QK256 block.
pub const QK256_BLOCK_COLS: usize = 256;

/// Number of packed bytes in one QK256 block.
pub const QK256_PACKED_BYTES_PER_BLOCK: usize = 64;

/// Kernel ID recorded by QK256 CUDA proof receipts.
pub const CUDA_QK256_GEMV_KERNEL_ID: &str = "qk256_gemv_cuda";

#[cfg(feature = "cuda")]
const CUDA_QK256_GEMV_KERNEL_SRC: &str = r#"
extern "C" __global__
void qk256_gemv_cuda(
    const unsigned char* packed_weights,
    const float* input,
    float* output,
    int seq_len,
    int n_out,
    int k,
    int row_stride_bytes
) {
    int out_col = blockIdx.x * blockDim.x + threadIdx.x;
    int token = blockIdx.y;

    if (out_col >= n_out || token >= seq_len) {
        return;
    }

    const unsigned char* row = packed_weights + ((long long)out_col * row_stride_bytes);
    const float* x = input + ((long long)token * k);
    float acc = 0.0f;

    for (int col = 0; col < k; ++col) {
        unsigned char packed = row[col >> 2];
        unsigned int code = (packed >> ((col & 3) << 1)) & 0x3u;
        float weight;
        if (code == 0u) {
            weight = -2.0f;
        } else if (code == 1u) {
            weight = -1.0f;
        } else if (code == 2u) {
            weight = 1.0f;
        } else {
            weight = 2.0f;
        }
        acc += weight * x[col];
    }

    output[((long long)token * n_out) + out_col] = acc;
}
"#;

#[cfg(feature = "cuda")]
static NVRTC_COMPILE_LOCK: Mutex<()> = Mutex::new(());

/// Launch configuration for the QK256 dequant+GEMV kernel.
///
/// The grid is 2-D: `(ceil(n_out / tile_n), ceil(seq_len / tile_m))`.
/// Each thread-block processes one `tile_m × tile_n` output tile.
#[derive(Debug, Clone)]
pub struct Qk256GemvConfig {
    /// CUDA block size in the M (sequence) dimension.
    pub block_m: u32,
    /// CUDA block size in the N (output-channel) dimension.
    pub block_n: u32,
    /// Number of threads per block (typically `block_m * block_n` capped at 256).
    pub threads_per_block: u32,
    /// Bytes of dynamic shared memory per block for packed weight tiles.
    pub shared_mem_bytes: u32,
    /// Number of output rows (sequence length).
    pub seq_len: usize,
    /// Number of output columns (hidden dimension).
    pub n_out: usize,
    /// Inner dimension (input hidden dimension).
    pub k: usize,
    /// Packed bytes per output row.
    pub row_stride_bytes: usize,
}

impl Default for Qk256GemvConfig {
    fn default() -> Self {
        Self {
            block_m: 1,
            block_n: 256,
            threads_per_block: 256,
            shared_mem_bytes: 0,
            seq_len: 1,
            n_out: 1,
            k: 256,
            row_stride_bytes: QK256_PACKED_BYTES_PER_BLOCK,
        }
    }
}

impl Qk256GemvConfig {
    /// Create a config tuned for the given matrix dimensions.
    ///
    /// QK256 rows are block-aligned, so `row_stride_bytes` is computed as
    /// `ceil(k / 256) * 64` and tail columns inside the final block are ignored.
    pub fn for_shape(seq_len: usize, n_out: usize, k: usize) -> Result<Self> {
        if k == 0 {
            return Err(KernelError::InvalidArguments {
                reason: "QK256 GEMV inner dimension k must be non-zero".to_string(),
            }
            .into());
        }
        if seq_len == 0 || n_out == 0 {
            return Err(KernelError::InvalidArguments {
                reason: format!(
                    "QK256 GEMV dimensions must be non-zero: seq_len={seq_len}, n_out={n_out}"
                ),
            }
            .into());
        }

        let row_stride_bytes = qk256_row_stride_bytes(k)?;

        Ok(Self {
            block_m: 1,
            block_n: 256,
            threads_per_block: 256,
            shared_mem_bytes: 0,
            seq_len,
            n_out,
            k,
            row_stride_bytes,
        })
    }

    /// Compute the CUDA grid dimensions `(grid_x, grid_y, 1)`.
    pub fn grid_dim(&self) -> (u32, u32, u32) {
        let grid_x = (self.n_out as u32).div_ceil(self.block_n);
        let grid_y = (self.seq_len as u32).div_ceil(self.block_m);
        (grid_x, grid_y, 1)
    }

    /// Compute the CUDA block dimensions.
    pub fn block_dim(&self) -> (u32, u32, u32) {
        (self.threads_per_block, 1, 1)
    }
}

/// Launch stub for the QK256 dequant+GEMV kernel.
///
/// # Arguments
///
/// * `packed_weights` — QK256-packed weight matrix (2-bit ternary, 64 B per 256-elem block)
/// * `scales`         — Must be empty for the MS BitNet no-scale QK256 layout
/// * `input`          — FP32 input activations `[seq_len, k]`
/// * `output`         — FP32 output buffer `[seq_len, n_out]` (written by kernel)
/// * `config`         — Launch configuration
///
/// # Errors
///
/// Returns an error if dimensions do not match, if scale bytes are supplied for
/// the no-scale layout, or if CUDA/NVRTC is unavailable.
pub fn launch_qk256_gemv(
    packed_weights: &[u8],
    scales: &[u8],
    input: &[f32],
    output: &mut [f32],
    config: &Qk256GemvConfig,
) -> Result<()> {
    validate_qk256_launch_inputs(packed_weights, scales, input, output, config)?;

    log::debug!(
        "QK256 GEMV CUDA launch: kernel={}, seq_len={}, n_out={}, k={}, row_stride_bytes={}, grid={:?}",
        CUDA_QK256_GEMV_KERNEL_ID,
        config.seq_len,
        config.n_out,
        config.k,
        config.row_stride_bytes,
        config.grid_dim(),
    );

    #[cfg(feature = "cuda")]
    {
        return launch_qk256_gemv_cuda(packed_weights, input, output, config);
    }

    #[cfg(not(feature = "cuda"))]
    Err(KernelError::DeviceUnavailable {
        reason: "QK256 GEMV CUDA kernel requires the cuda feature".to_string(),
    }
    .into())
}

fn validate_qk256_launch_inputs(
    packed_weights: &[u8],
    scales: &[u8],
    input: &[f32],
    output: &[f32],
    config: &Qk256GemvConfig,
) -> Result<()> {
    if !scales.is_empty() {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "QK256 GEMV CUDA currently supports the MS BitNet no-scale layout; got {} scale bytes",
                scales.len()
            ),
        }
        .into());
    }

    let expected_stride = qk256_row_stride_bytes(config.k)?;
    if config.row_stride_bytes != expected_stride {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "QK256 GEMV row_stride_bytes {} != expected {} for k={}",
                config.row_stride_bytes, expected_stride, config.k
            ),
        }
        .into());
    }

    let expected_packed = checked_mul(config.n_out, config.row_stride_bytes, "packed weights")?;
    if packed_weights.len() < expected_packed {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "QK256 GEMV packed weights too short: got {}, expected at least {}",
                packed_weights.len(),
                expected_packed
            ),
        }
        .into());
    }

    let expected_input = checked_mul(config.seq_len, config.k, "input")?;
    if input.len() < expected_input {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "QK256 GEMV input too short: got {}, expected at least {}",
                input.len(),
                expected_input
            ),
        }
        .into());
    }

    let expected_output = checked_mul(config.seq_len, config.n_out, "output")?;
    if output.len() < expected_output {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "QK256 GEMV output too short: got {}, expected at least {}",
                output.len(),
                expected_output
            ),
        }
        .into());
    }

    validate_cuda_i32_arg(config.seq_len, "seq_len")?;
    validate_cuda_i32_arg(config.n_out, "n_out")?;
    validate_cuda_i32_arg(config.k, "k")?;
    validate_cuda_i32_arg(config.row_stride_bytes, "row_stride_bytes")?;
    Ok(())
}

#[cfg(feature = "cuda")]
fn launch_qk256_gemv_cuda(
    packed_weights: &[u8],
    input: &[f32],
    output: &mut [f32],
    config: &Qk256GemvConfig,
) -> Result<()> {
    let ctx = CudaContext::new(0).map_err(|err| KernelError::GpuError {
        reason: format!("failed to create CUDA context for QK256 GEMV: {err:?}"),
    })?;
    let stream = ctx.default_stream();
    let ptx = compile_qk256_ptx()?;
    let module = ctx.load_module(ptx).map_err(|err| KernelError::GpuError {
        reason: format!("failed to load QK256 GEMV CUDA module: {err:?}"),
    })?;
    let function = module.load_function(CUDA_QK256_GEMV_KERNEL_ID).map_err(|err| {
        KernelError::GpuError { reason: format!("failed to load QK256 GEMV CUDA kernel: {err:?}") }
    })?;

    let packed_len = checked_mul(config.n_out, config.row_stride_bytes, "packed weights")?;
    let input_len = checked_mul(config.seq_len, config.k, "input")?;
    let output_len = checked_mul(config.seq_len, config.n_out, "output")?;

    let packed_dev =
        stream.memcpy_stod(&packed_weights[..packed_len]).map_err(|err| KernelError::GpuError {
            reason: format!("failed to copy QK256 packed weights to device: {err:?}"),
        })?;
    let input_dev = stream.memcpy_stod(&input[..input_len]).map_err(|err| {
        KernelError::GpuError { reason: format!("failed to copy QK256 input to device: {err:?}") }
    })?;
    let mut output_dev: CudaSlice<f32> =
        stream.alloc_zeros(output_len).map_err(|err| KernelError::GpuError {
            reason: format!("failed to allocate QK256 output on device: {err:?}"),
        })?;

    let launch_config = LaunchConfig {
        grid_dim: config.grid_dim(),
        block_dim: config.block_dim(),
        shared_mem_bytes: config.shared_mem_bytes,
    };
    let mut builder = stream.launch_builder(&function);
    builder.arg(&packed_dev);
    builder.arg(&input_dev);
    builder.arg(&mut output_dev);
    let seq_len_arg = i32::try_from(config.seq_len).map_err(|_| KernelError::InvalidArguments {
        reason: format!("QK256 GEMV seq_len exceeds i32: {}", config.seq_len),
    })?;
    let n_out_arg = i32::try_from(config.n_out).map_err(|_| KernelError::InvalidArguments {
        reason: format!("QK256 GEMV n_out exceeds i32: {}", config.n_out),
    })?;
    let k_arg = i32::try_from(config.k).map_err(|_| KernelError::InvalidArguments {
        reason: format!("QK256 GEMV k exceeds i32: {}", config.k),
    })?;
    let row_stride_arg =
        i32::try_from(config.row_stride_bytes).map_err(|_| KernelError::InvalidArguments {
            reason: format!("QK256 GEMV row_stride_bytes exceeds i32: {}", config.row_stride_bytes),
        })?;
    builder.arg(&seq_len_arg);
    builder.arg(&n_out_arg);
    builder.arg(&k_arg);
    builder.arg(&row_stride_arg);

    unsafe { builder.launch(launch_config) }.map_err(|err| KernelError::GpuError {
        reason: format!("failed to launch QK256 GEMV CUDA kernel: {err:?}"),
    })?;
    stream.synchronize().map_err(|err| KernelError::GpuError {
        reason: format!("failed to synchronize QK256 GEMV CUDA kernel: {err:?}"),
    })?;

    let output_host: Vec<f32> =
        stream.memcpy_dtov(&output_dev).map_err(|err| KernelError::GpuError {
            reason: format!("failed to copy QK256 output from device: {err:?}"),
        })?;
    output[..output_len].copy_from_slice(&output_host[..output_len]);
    Ok(())
}

#[cfg(feature = "cuda")]
fn compile_qk256_ptx() -> Result<Ptx> {
    let _hook_guard = NVRTC_COMPILE_LOCK.lock().ok();
    let previous_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let compile_result = std::panic::catch_unwind(|| compile_ptx(CUDA_QK256_GEMV_KERNEL_SRC));
    std::panic::set_hook(previous_hook);

    match compile_result {
        Ok(Ok(ptx)) => Ok(ptx),
        Ok(Err(err)) => Err(KernelError::GpuError {
            reason: format!("failed to compile QK256 GEMV CUDA PTX: {err:?}"),
        }
        .into()),
        Err(payload) => Err(KernelError::GpuError {
            reason: format!(
                "failed to compile QK256 GEMV CUDA PTX because NVRTC was unavailable: {}",
                panic_payload_message(&*payload)
            ),
        }
        .into()),
    }
}

#[cfg(feature = "cuda")]
fn panic_payload_message(payload: &(dyn Any + Send)) -> String {
    if let Some(message) = payload.downcast_ref::<&str>() {
        (*message).to_string()
    } else if let Some(message) = payload.downcast_ref::<String>() {
        message.clone()
    } else {
        "unknown panic payload".to_string()
    }
}

fn qk256_row_stride_bytes(cols: usize) -> Result<usize> {
    let blocks_per_row = cols.div_ceil(QK256_BLOCK_COLS);
    checked_mul(blocks_per_row, QK256_PACKED_BYTES_PER_BLOCK, "row stride")
}

fn checked_mul(lhs: usize, rhs: usize, label: &str) -> Result<usize> {
    lhs.checked_mul(rhs).ok_or_else(|| {
        KernelError::InvalidArguments {
            reason: format!("QK256 GEMV {label} length overflow: {lhs} * {rhs}"),
        }
        .into()
    })
}

fn validate_cuda_i32_arg(value: usize, label: &str) -> Result<()> {
    i32::try_from(value).map(|_| ()).map_err(|_| {
        KernelError::InvalidArguments { reason: format!("QK256 GEMV {label} exceeds i32: {value}") }
            .into()
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn code_to_f32(code: u8) -> f32 {
        match code & 0x03 {
            0 => -2.0,
            1 => -1.0,
            2 => 1.0,
            _ => 2.0,
        }
    }

    fn pack_codes_for_cols(codes: &[u8], cols: usize) -> Vec<u8> {
        let row_stride_bytes = qk256_row_stride_bytes(cols).expect("row stride");
        let mut packed = vec![0u8; row_stride_bytes];
        for (index, code) in codes.iter().copied().enumerate().take(cols) {
            assert!(code < 4, "test QK256 code must be 0..=3");
            packed[index / 4] |= code << ((index % 4) * 2);
        }
        packed
    }

    fn reference_qk256_gemv(
        packed_weights: &[u8],
        input: &[f32],
        seq_len: usize,
        n_out: usize,
        k: usize,
    ) -> Vec<f32> {
        let row_stride_bytes = qk256_row_stride_bytes(k).expect("row stride");
        let mut output = vec![0.0f32; seq_len * n_out];

        for token in 0..seq_len {
            let x = &input[token * k..(token + 1) * k];
            for row in 0..n_out {
                let row_start = row * row_stride_bytes;
                let row_bytes = &packed_weights[row_start..row_start + row_stride_bytes];
                let mut acc = 0.0f32;
                for col in 0..k {
                    let packed = row_bytes[col / 4];
                    let code = (packed >> ((col % 4) * 2)) & 0x03;
                    acc += code_to_f32(code) * x[col];
                }
                output[token * n_out + row] = acc;
            }
        }

        output
    }

    #[test]
    fn test_qk256_gemv_config_defaults() {
        let cfg = Qk256GemvConfig::default();
        assert_eq!(cfg.threads_per_block, 256);
        assert_eq!(cfg.k, 256);
        assert_eq!(cfg.row_stride_bytes, QK256_PACKED_BYTES_PER_BLOCK);
        assert_eq!(cfg.shared_mem_bytes, 0);
    }

    #[test]
    fn test_qk256_gemv_kernel_id_is_receipt_safe() {
        assert_eq!(CUDA_QK256_GEMV_KERNEL_ID, "qk256_gemv_cuda");
        assert!(!CUDA_QK256_GEMV_KERNEL_ID.contains("mock"));
        assert!(CUDA_QK256_GEMV_KERNEL_ID.len() <= 128);
    }

    #[test]
    fn test_qk256_gemv_config_for_shape() {
        let cfg = Qk256GemvConfig::for_shape(1, 2048, 2048).unwrap();
        assert_eq!(cfg.seq_len, 1);
        assert_eq!(cfg.n_out, 2048);
        assert_eq!(cfg.k, 2048);
        assert_eq!(cfg.row_stride_bytes, 512);
        let (gx, gy, gz) = cfg.grid_dim();
        assert_eq!(gx, 8); // 2048 / 256
        assert_eq!(gy, 1);
        assert_eq!(gz, 1);
    }

    #[test]
    fn test_qk256_gemv_config_supports_tail_k() {
        let cfg = Qk256GemvConfig::for_shape(1, 7, 300).unwrap();
        assert_eq!(cfg.k, 300);
        assert_eq!(cfg.row_stride_bytes, 128);
        let (gx, gy, gz) = cfg.grid_dim();
        assert_eq!((gx, gy, gz), (1, 1, 1));
    }

    #[test]
    fn test_qk256_gemv_config_rejects_zero_dims() {
        assert!(Qk256GemvConfig::for_shape(0, 2048, 256).is_err());
        assert!(Qk256GemvConfig::for_shape(1, 0, 256).is_err());
        assert!(Qk256GemvConfig::for_shape(1, 2048, 0).is_err());
    }

    #[test]
    fn test_qk256_gemv_grid_dim_rounding() {
        let cfg = Qk256GemvConfig::for_shape(3, 500, 512).unwrap();
        let (gx, gy, _) = cfg.grid_dim();
        assert_eq!(gx, 2); // ceil(500/256)
        assert_eq!(gy, 3); // ceil(3/1)
    }

    #[test]
    fn test_qk256_reference_fixture_uses_tail_columns_only() {
        let seq_len = 2usize;
        let n_out = 2usize;
        let k = 300usize;
        let cfg = Qk256GemvConfig::for_shape(seq_len, n_out, k).unwrap();
        let row0_codes: Vec<u8> = (0..k).map(|i| (i % 4) as u8).collect();
        let row1_codes: Vec<u8> = (0..k).map(|i| ((i + 1) % 4) as u8).collect();
        let mut packed = Vec::new();
        packed.extend_from_slice(&pack_codes_for_cols(&row0_codes, k));
        packed.extend_from_slice(&pack_codes_for_cols(&row1_codes, k));
        let input: Vec<f32> = (0..seq_len * k).map(|i| ((i % 13) as f32 - 6.0) * 0.125).collect();

        let expected = reference_qk256_gemv(&packed, &input, seq_len, n_out, k);

        assert_eq!(packed.len(), n_out * cfg.row_stride_bytes);
        assert_eq!(expected.len(), seq_len * n_out);
        assert_ne!(expected[0], expected[1]);
        assert_ne!(expected[0], expected[2]);
    }

    #[test]
    fn test_qk256_reference_fixture_matches_canonical_scalar_oracle() {
        let seq_len = 2usize;
        let n_out = 3usize;
        let k = 300usize;
        let mut packed = Vec::new();
        for row in 0..n_out {
            let codes: Vec<u8> = (0..k).map(|col| ((row + col) % 4) as u8).collect();
            packed.extend_from_slice(&pack_codes_for_cols(&codes, k));
        }
        let input: Vec<f32> = (0..seq_len * k).map(|i| ((i % 17) as f32 - 8.0) * 0.0625).collect();
        let expected = reference_qk256_gemv(&packed, &input, seq_len, n_out, k);
        let mut canonical = vec![0.0f32; seq_len * n_out];

        bitnet_quantization::i2s_qk256::qk256_gemm_scalar(
            &packed,
            &input,
            &mut canonical,
            seq_len,
            n_out,
            k,
        )
        .expect("canonical QK256 scalar oracle should accept fixture");

        for (index, (got, expected)) in canonical.iter().zip(expected).enumerate() {
            let diff = (got - expected).abs();
            assert!(
                diff <= 1e-5,
                "canonical scalar mismatch at {index}: got {got}, expected {expected}, diff {diff}"
            );
        }
    }

    #[test]
    fn test_qk256_launch_rejects_scale_blocks_for_no_scale_layout() {
        let cfg = Qk256GemvConfig::for_shape(1, 1, 256).unwrap();
        let packed = vec![0u8; cfg.row_stride_bytes];
        let scales = vec![0u8; 2];
        let input = vec![1.0f32; 256];
        let mut output = vec![0.0f32; 1];

        let err = launch_qk256_gemv(&packed, &scales, &input, &mut output, &cfg)
            .expect_err("scale blocks should be rejected for no-scale QK256");
        assert!(err.to_string().contains("no-scale"), "unexpected error: {err}");
    }

    #[test]
    fn test_qk256_launch_rejects_short_packed_weights() {
        let cfg = Qk256GemvConfig::for_shape(1, 2, 256).unwrap();
        let packed = vec![0u8; cfg.row_stride_bytes];
        let input = vec![1.0f32; 256];
        let mut output = vec![0.0f32; 2];

        let err = launch_qk256_gemv(&packed, &[], &input, &mut output, &cfg)
            .expect_err("packed buffer should be too short");
        assert!(err.to_string().contains("packed weights too short"), "unexpected error: {err}");
    }

    #[test]
    fn test_qk256_launch_rejects_wrong_row_stride() {
        let mut cfg = Qk256GemvConfig::for_shape(1, 1, 300).unwrap();
        cfg.row_stride_bytes = QK256_PACKED_BYTES_PER_BLOCK;
        let packed = vec![0u8; 128];
        let input = vec![1.0f32; 300];
        let mut output = vec![0.0f32; 1];

        let err = launch_qk256_gemv(&packed, &[], &input, &mut output, &cfg)
            .expect_err("wrong row stride should fail before CUDA launch");
        assert!(err.to_string().contains("row_stride_bytes"), "unexpected error: {err}");
    }

    #[cfg(not(feature = "cuda"))]
    #[test]
    fn test_qk256_launch_cpu_build_reports_cuda_unavailable() {
        let cfg = Qk256GemvConfig::for_shape(1, 1, 256).unwrap();
        let packed = vec![0xAAu8; cfg.row_stride_bytes];
        let input = vec![1.0f32; 256];
        let mut output = vec![0.0f32; 1];

        let err = launch_qk256_gemv(&packed, &[], &input, &mut output, &cfg)
            .expect_err("CPU-only build cannot launch CUDA QK256");
        assert!(err.to_string().contains("cuda feature"), "unexpected error: {err}");
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_cuda_qk256_gemv_live_parity_opt_in() -> Result<()> {
        if std::env::var("BITNET_RUN_CUDA_QK256_GEMV").as_deref() != Ok("1") {
            return Ok(());
        }

        let seq_len = 2usize;
        let n_out = 3usize;
        let k = 300usize;
        let cfg = Qk256GemvConfig::for_shape(seq_len, n_out, k)?;
        let mut packed = Vec::new();
        for row in 0..n_out {
            let codes: Vec<u8> = (0..k).map(|col| ((row + col) % 4) as u8).collect();
            packed.extend_from_slice(&pack_codes_for_cols(&codes, k));
        }
        let input: Vec<f32> = (0..seq_len * k).map(|i| ((i % 17) as f32 - 8.0) * 0.0625).collect();
        let expected = reference_qk256_gemv(&packed, &input, seq_len, n_out, k);
        let mut output = vec![0.0f32; seq_len * n_out];

        launch_qk256_gemv(&packed, &[], &input, &mut output, &cfg)?;

        for (index, (got, expected)) in output.iter().zip(expected).enumerate() {
            let diff = (got - expected).abs();
            assert!(
                diff <= 1e-4,
                "QK256 CUDA parity mismatch at {index}: got {got}, expected {expected}, diff {diff}"
            );
        }

        Ok(())
    }
}
