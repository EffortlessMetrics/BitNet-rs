//! Extended CUDA Tensor Core operations for BitNet inference.
//!
//! Provides high-level configuration structs and CPU fallback
//! implementations for Tensor Core GEMM variants:
//!
//! - [`TcBatchedGemm`]: Batched GEMM with configurable accumulation
//! - [`TcConvolution`]: Implicit GEMM convolution via Tensor Cores
//! - [`TcQuantizedGemm`]: INT8/INT4 GEMM targeting SM80+ (Ampere+)
//! - [`TcGroupedGemm`]: Grouped GEMM for multi-head attention
//! - [`TcSplitK`]: Split-K GEMM for improved parallelism on small M
//! - [`TcStreamK`]: Stream-K decomposition for load balancing
//! - [`TcPrecisionPolicy`]: Automatic precision selection
//!
//! All GPU launch stubs and CUDA kernel sources are behind
//! `#[cfg(any(feature = "gpu", feature = "cuda"))]`.

use bitnet_common::{BitNetError, KernelError, Result};

// ── Precision / accumulation enums ────────────────────────────────────

/// Tensor Core compute precision.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TcPrecision {
    /// IEEE FP16 (half precision).
    Fp16,
    /// BFloat16.
    Bf16,
    /// TF32 (Ampere+, 19-bit).
    Tf32,
    /// INT8.
    Int8,
    /// INT4 (SM80+ Ampere).
    Int4,
}

impl TcPrecision {
    /// Bytes per element for this precision.
    pub fn element_bytes(self) -> usize {
        match self {
            Self::Fp16 | Self::Bf16 => 2,
            Self::Tf32 => 4,
            Self::Int8 => 1,
            Self::Int4 => 1, // sub-byte; treated as 1 for packing
        }
    }
}

/// Accumulation type for Tensor Core GEMM.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TcAccumulation {
    /// 32-bit float accumulator.
    F32,
    /// 16-bit float accumulator.
    F16,
    /// 32-bit integer accumulator (for INT8/INT4 paths).
    I32,
}

// ── TcPrecisionPolicy ─────────────────────────────────────────────────

/// Automatic precision selection policy.
///
/// Given a compute capability and problem characteristics, selects the
/// best Tensor Core precision and accumulation type.
#[derive(Debug, Clone)]
pub struct TcPrecisionPolicy {
    /// SM architecture version (e.g. 70 = Volta, 80 = Ampere).
    pub sm_version: u32,
    /// Whether to prefer FP16 over BF16 when both are available.
    pub prefer_fp16: bool,
    /// Force a specific precision (overrides auto-selection).
    pub force_precision: Option<TcPrecision>,
}

impl Default for TcPrecisionPolicy {
    fn default() -> Self {
        Self { sm_version: 80, prefer_fp16: false, force_precision: None }
    }
}

impl TcPrecisionPolicy {
    /// Create a policy for a specific SM version.
    pub fn for_sm(sm_version: u32) -> Self {
        Self { sm_version, ..Self::default() }
    }

    /// Force a specific precision.
    pub fn with_forced_precision(mut self, precision: TcPrecision) -> Self {
        self.force_precision = Some(precision);
        self
    }

    /// Set FP16 preference.
    pub fn with_prefer_fp16(mut self, prefer: bool) -> Self {
        self.prefer_fp16 = prefer;
        self
    }

    /// Select the best precision for floating-point GEMM.
    pub fn select_float_precision(&self) -> TcPrecision {
        if let Some(forced) = self.force_precision {
            return forced;
        }
        match self.sm_version {
            sm if sm >= 80 => {
                if self.prefer_fp16 {
                    TcPrecision::Fp16
                } else {
                    TcPrecision::Tf32
                }
            }
            sm if sm >= 70 => TcPrecision::Fp16,
            _ => TcPrecision::Fp16,
        }
    }

    /// Select the best precision for integer/quantized GEMM.
    pub fn select_int_precision(&self) -> TcPrecision {
        if let Some(forced) = self.force_precision {
            return forced;
        }
        if self.sm_version >= 80 { TcPrecision::Int8 } else { TcPrecision::Fp16 }
    }

    /// Select the accumulation type for a given precision.
    pub fn select_accumulation(&self, precision: TcPrecision) -> TcAccumulation {
        match precision {
            TcPrecision::Int8 | TcPrecision::Int4 => TcAccumulation::I32,
            _ => TcAccumulation::F32,
        }
    }

    /// Whether Tensor Cores are available at this SM level.
    pub fn tensor_cores_available(&self) -> bool {
        self.sm_version >= 70
    }

    /// Whether INT8 Tensor Cores are available.
    pub fn int8_tc_available(&self) -> bool {
        self.sm_version >= 75
    }

    /// Whether INT4 Tensor Cores are available.
    pub fn int4_tc_available(&self) -> bool {
        self.sm_version >= 80
    }

    /// Recommended WMMA tile size for the given precision.
    pub fn wmma_tile_size(&self, precision: TcPrecision) -> (u32, u32, u32) {
        match precision {
            TcPrecision::Fp16 | TcPrecision::Bf16 => (16, 16, 16),
            TcPrecision::Tf32 => (16, 16, 8),
            TcPrecision::Int8 => (16, 16, 32),
            TcPrecision::Int4 => (8, 8, 32),
        }
    }
}

// ── TcBatchedGemm ─────────────────────────────────────────────────────

/// Batched GEMM using Tensor Cores with configurable accumulation.
///
/// Computes `C[b] = α · op(A[b]) · op(B[b]) + β · C[b]` for each
/// batch element `b`.
#[derive(Debug, Clone)]
pub struct TcBatchedGemm {
    /// Output rows.
    pub m: usize,
    /// Output columns.
    pub n: usize,
    /// Reduction dimension.
    pub k: usize,
    /// Batch count.
    pub batch_size: usize,
    /// Input precision.
    pub precision: TcPrecision,
    /// Accumulation type.
    pub accumulation: TcAccumulation,
    /// Scalar α.
    pub alpha: f32,
    /// Scalar β.
    pub beta: f32,
    /// Transpose A.
    pub transpose_a: bool,
    /// Transpose B.
    pub transpose_b: bool,
    /// WMMA tile M.
    pub tile_m: u32,
    /// WMMA tile N.
    pub tile_n: u32,
    /// WMMA tile K.
    pub tile_k: u32,
}

impl TcBatchedGemm {
    /// Create a new config for the given shape.
    ///
    /// # Errors
    ///
    /// Returns an error if any dimension is zero.
    pub fn new(m: usize, n: usize, k: usize, batch_size: usize) -> Result<Self> {
        if m == 0 || n == 0 || k == 0 {
            return Err(KernelError::InvalidArguments {
                reason: format!("TcBatchedGemm dimensions must be non-zero: m={m}, n={n}, k={k}"),
            }
            .into());
        }
        if batch_size == 0 {
            return Err(
                KernelError::InvalidArguments { reason: "batch_size must be > 0".into() }.into()
            );
        }
        Ok(Self {
            m,
            n,
            k,
            batch_size,
            precision: TcPrecision::Fp16,
            accumulation: TcAccumulation::F32,
            alpha: 1.0,
            beta: 0.0,
            transpose_a: false,
            transpose_b: false,
            tile_m: 16,
            tile_n: 16,
            tile_k: 16,
        })
    }

    /// Set precision and accumulation.
    pub fn with_precision(mut self, p: TcPrecision, acc: TcAccumulation) -> Self {
        self.precision = p;
        self.accumulation = acc;
        self
    }

    /// Set α, β scalars.
    pub fn with_alpha_beta(mut self, alpha: f32, beta: f32) -> Self {
        self.alpha = alpha;
        self.beta = beta;
        self
    }

    /// Set transpose flags.
    pub fn with_transpose(mut self, trans_a: bool, trans_b: bool) -> Self {
        self.transpose_a = trans_a;
        self.transpose_b = trans_b;
        self
    }

    /// Set WMMA tile dimensions.
    ///
    /// # Errors
    ///
    /// Returns an error if any tile dimension is zero.
    pub fn with_tiles(mut self, tile_m: u32, tile_n: u32, tile_k: u32) -> Result<Self> {
        if tile_m == 0 || tile_n == 0 || tile_k == 0 {
            return Err(KernelError::InvalidArguments {
                reason: format!("tile dimensions must be non-zero: {tile_m}×{tile_n}×{tile_k}"),
            }
            .into());
        }
        self.tile_m = tile_m;
        self.tile_n = tile_n;
        self.tile_k = tile_k;
        Ok(self)
    }

    /// Grid dimensions for CUDA launch.
    pub fn grid_dim(&self) -> (u32, u32, u32) {
        let gx = (self.n as u32).div_ceil(self.tile_n);
        let gy = (self.m as u32).div_ceil(self.tile_m);
        (gx, gy, self.batch_size as u32)
    }

    /// Block dimensions for CUDA launch.
    pub fn block_dim(&self) -> (u32, u32, u32) {
        // One warp per WMMA tile.
        (32, 1, 1)
    }

    /// Estimated shared memory (bytes).
    pub fn shared_mem_bytes(&self) -> u32 {
        let elem = match self.precision {
            TcPrecision::Int4 | TcPrecision::Int8 => 1u32,
            TcPrecision::Fp16 | TcPrecision::Bf16 => 2,
            TcPrecision::Tf32 => 4,
        };
        let a_bytes = self.tile_m * self.tile_k * elem;
        let b_bytes = self.tile_k * self.tile_n * elem;
        (a_bytes + b_bytes).max(1024)
    }

    /// Estimated GFLOPS at a given duration.
    pub fn gflops(&self, duration_secs: f64) -> f64 {
        let ops = 2.0 * self.m as f64 * self.n as f64 * self.k as f64 * self.batch_size as f64;
        ops / (duration_secs * 1e9)
    }
}

/// CPU fallback for batched Tensor Core GEMM.
///
/// Computes `C[b] = α · op(A[b]) · op(B[b]) + β · C[b]` per batch.
///
/// # Errors
///
/// Returns an error if buffer sizes are inconsistent.
pub fn tc_batched_gemm_cpu(a: &[f32], b: &[f32], c: &mut [f32], cfg: &TcBatchedGemm) -> Result<()> {
    let (m, n, k, batch) = (cfg.m, cfg.n, cfg.k, cfg.batch_size);
    let a_stride = m * k;
    let b_stride = k * n;
    let c_stride = m * n;

    if a.len() < batch * a_stride {
        return Err(buffer_error("A", batch * a_stride, a.len()));
    }
    if b.len() < batch * b_stride {
        return Err(buffer_error("B", batch * b_stride, b.len()));
    }
    if c.len() < batch * c_stride {
        return Err(buffer_error("C", batch * c_stride, c.len()));
    }

    for bi in 0..batch {
        let a_off = bi * a_stride;
        let b_off = bi * b_stride;
        let c_off = bi * c_stride;

        for i in 0..m {
            for j in 0..n {
                let mut acc = 0.0f32;
                for l in 0..k {
                    let a_idx = if cfg.transpose_a { l * m + i } else { i * k + l };
                    let b_idx = if cfg.transpose_b { j * k + l } else { l * n + j };
                    acc += a[a_off + a_idx] * b[b_off + b_idx];
                }
                let idx = c_off + i * n + j;
                c[idx] = cfg.alpha * acc + cfg.beta * c[idx];
            }
        }
    }
    Ok(())
}

// ── TcConvolution ─────────────────────────────────────────────────────

/// Implicit GEMM convolution via Tensor Cores.
///
/// Maps a convolution to a GEMM by im2col-style indexing without
/// materialising the full im2col matrix.
#[derive(Debug, Clone)]
pub struct TcConvolution {
    /// Batch size.
    pub batch_size: usize,
    /// Input channels.
    pub in_channels: usize,
    /// Output channels.
    pub out_channels: usize,
    /// Input height.
    pub input_h: usize,
    /// Input width.
    pub input_w: usize,
    /// Kernel height.
    pub kernel_h: usize,
    /// Kernel width.
    pub kernel_w: usize,
    /// Stride.
    pub stride: usize,
    /// Padding.
    pub padding: usize,
    /// Dilation.
    pub dilation: usize,
    /// Compute precision.
    pub precision: TcPrecision,
}

impl TcConvolution {
    /// Create a new convolution config.
    ///
    /// # Errors
    ///
    /// Returns an error if any spatial or channel dimension is zero.
    pub fn new(
        batch_size: usize,
        in_channels: usize,
        out_channels: usize,
        input_h: usize,
        input_w: usize,
        kernel_h: usize,
        kernel_w: usize,
    ) -> Result<Self> {
        if batch_size == 0
            || in_channels == 0
            || out_channels == 0
            || input_h == 0
            || input_w == 0
            || kernel_h == 0
            || kernel_w == 0
        {
            return Err(KernelError::InvalidArguments {
                reason: "TcConvolution: all dimensions must be non-zero".into(),
            }
            .into());
        }
        Ok(Self {
            batch_size,
            in_channels,
            out_channels,
            input_h,
            input_w,
            kernel_h,
            kernel_w,
            stride: 1,
            padding: 0,
            dilation: 1,
            precision: TcPrecision::Fp16,
        })
    }

    /// Set stride, padding, dilation.
    ///
    /// # Errors
    ///
    /// Returns an error if stride or dilation is zero.
    pub fn with_params(mut self, stride: usize, padding: usize, dilation: usize) -> Result<Self> {
        if stride == 0 || dilation == 0 {
            return Err(KernelError::InvalidArguments {
                reason: "stride and dilation must be non-zero".into(),
            }
            .into());
        }
        self.stride = stride;
        self.padding = padding;
        self.dilation = dilation;
        Ok(self)
    }

    /// Set compute precision.
    pub fn with_precision(mut self, p: TcPrecision) -> Self {
        self.precision = p;
        self
    }

    /// Output height.
    pub fn output_h(&self) -> usize {
        let effective_kh = (self.kernel_h - 1) * self.dilation + 1;
        (self.input_h + 2 * self.padding).saturating_sub(effective_kh) / self.stride + 1
    }

    /// Output width.
    pub fn output_w(&self) -> usize {
        let effective_kw = (self.kernel_w - 1) * self.dilation + 1;
        (self.input_w + 2 * self.padding).saturating_sub(effective_kw) / self.stride + 1
    }

    /// GEMM M dimension (batch * output_h * output_w).
    pub fn gemm_m(&self) -> usize {
        self.batch_size * self.output_h() * self.output_w()
    }

    /// GEMM K dimension (in_channels * kernel_h * kernel_w).
    pub fn gemm_k(&self) -> usize {
        self.in_channels * self.kernel_h * self.kernel_w
    }

    /// GEMM N dimension (out_channels).
    pub fn gemm_n(&self) -> usize {
        self.out_channels
    }
}

/// CPU fallback for implicit GEMM convolution.
///
/// # Layout
/// - `input`: NCHW `[batch, in_channels, input_h, input_w]`
/// - `weight`: OIHW `[out_channels, in_channels, kernel_h, kernel_w]`
/// - `output`: NCHW `[batch, out_channels, output_h, output_w]`
///
/// # Errors
///
/// Returns an error if buffer sizes are inconsistent.
pub fn tc_convolution_cpu(
    input: &[f32],
    weight: &[f32],
    output: &mut [f32],
    cfg: &TcConvolution,
) -> Result<()> {
    let oh = cfg.output_h();
    let ow = cfg.output_w();
    let in_per_sample = cfg.in_channels * cfg.input_h * cfg.input_w;
    let out_per_sample = cfg.out_channels * oh * ow;
    let w_per_filter = cfg.in_channels * cfg.kernel_h * cfg.kernel_w;

    if input.len() < cfg.batch_size * in_per_sample {
        return Err(buffer_error("input", cfg.batch_size * in_per_sample, input.len()));
    }
    if weight.len() < cfg.out_channels * w_per_filter {
        return Err(buffer_error("weight", cfg.out_channels * w_per_filter, weight.len()));
    }
    if output.len() < cfg.batch_size * out_per_sample {
        return Err(buffer_error("output", cfg.batch_size * out_per_sample, output.len()));
    }

    output[..cfg.batch_size * out_per_sample].fill(0.0);

    for bi in 0..cfg.batch_size {
        for oc in 0..cfg.out_channels {
            for ohi in 0..oh {
                for owi in 0..ow {
                    let mut acc = 0.0f32;
                    for ic in 0..cfg.in_channels {
                        for kh in 0..cfg.kernel_h {
                            for kw in 0..cfg.kernel_w {
                                let ih = ohi * cfg.stride + kh * cfg.dilation;
                                let iw = owi * cfg.stride + kw * cfg.dilation;
                                let ih_actual = ih as isize - cfg.padding as isize;
                                let iw_actual = iw as isize - cfg.padding as isize;

                                if ih_actual >= 0
                                    && (ih_actual as usize) < cfg.input_h
                                    && iw_actual >= 0
                                    && (iw_actual as usize) < cfg.input_w
                                {
                                    let in_idx = bi * in_per_sample
                                        + ic * cfg.input_h * cfg.input_w
                                        + ih_actual as usize * cfg.input_w
                                        + iw_actual as usize;
                                    let w_idx = oc * w_per_filter
                                        + ic * cfg.kernel_h * cfg.kernel_w
                                        + kh * cfg.kernel_w
                                        + kw;
                                    acc += input[in_idx] * weight[w_idx];
                                }
                            }
                        }
                    }
                    let out_idx = bi * out_per_sample + oc * oh * ow + ohi * ow + owi;
                    output[out_idx] = acc;
                }
            }
        }
    }
    Ok(())
}

// ── TcQuantizedGemm ───────────────────────────────────────────────────

/// Quantized integer precision for Tensor Core GEMM.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TcQuantBits {
    /// INT8 (SM75+).
    Int8,
    /// INT4 (SM80+).
    Int4,
}

impl TcQuantBits {
    /// Bits per element.
    pub fn bits(self) -> u32 {
        match self {
            Self::Int8 => 8,
            Self::Int4 => 4,
        }
    }

    /// Elements per byte.
    pub fn elems_per_byte(self) -> usize {
        (8 / self.bits()) as usize
    }

    /// Minimum SM version required.
    pub fn min_sm(self) -> u32 {
        match self {
            Self::Int8 => 75,
            Self::Int4 => 80,
        }
    }
}

/// INT8/INT4 GEMM using Tensor Cores (SM80+).
///
/// Computes `C = α · A_int · B_int + β · C` with integer inputs and
/// FP32 or INT32 accumulation.
#[derive(Debug, Clone)]
pub struct TcQuantizedGemm {
    /// Output rows.
    pub m: usize,
    /// Output columns.
    pub n: usize,
    /// Reduction dimension.
    pub k: usize,
    /// Batch count.
    pub batch_size: usize,
    /// Quantization bit-width.
    pub quant_bits: TcQuantBits,
    /// Accumulation type.
    pub accumulation: TcAccumulation,
    /// Per-column scale factors.
    pub use_scales: bool,
    /// Scalar α.
    pub alpha: f32,
    /// Scalar β.
    pub beta: f32,
    /// WMMA tile M.
    pub tile_m: u32,
    /// WMMA tile N.
    pub tile_n: u32,
    /// WMMA tile K.
    pub tile_k: u32,
}

impl TcQuantizedGemm {
    /// Create a new config.
    ///
    /// # Errors
    ///
    /// Returns an error if any dimension is zero.
    pub fn new(m: usize, n: usize, k: usize, quant_bits: TcQuantBits) -> Result<Self> {
        if m == 0 || n == 0 || k == 0 {
            return Err(KernelError::InvalidArguments {
                reason: format!("TcQuantizedGemm dimensions must be non-zero: m={m}, n={n}, k={k}"),
            }
            .into());
        }
        let (tm, tn, tk) = match quant_bits {
            TcQuantBits::Int8 => (16u32, 16, 32),
            TcQuantBits::Int4 => (8, 8, 32),
        };
        Ok(Self {
            m,
            n,
            k,
            batch_size: 1,
            quant_bits,
            accumulation: TcAccumulation::I32,
            use_scales: true,
            alpha: 1.0,
            beta: 0.0,
            tile_m: tm,
            tile_n: tn,
            tile_k: tk,
        })
    }

    /// Set batch size.
    ///
    /// # Errors
    ///
    /// Returns an error if `batch_size` is zero.
    pub fn with_batch_size(mut self, batch_size: usize) -> Result<Self> {
        if batch_size == 0 {
            return Err(
                KernelError::InvalidArguments { reason: "batch_size must be > 0".into() }.into()
            );
        }
        self.batch_size = batch_size;
        Ok(self)
    }

    /// Set accumulation type.
    pub fn with_accumulation(mut self, acc: TcAccumulation) -> Self {
        self.accumulation = acc;
        self
    }

    /// Set α, β scalars.
    pub fn with_alpha_beta(mut self, alpha: f32, beta: f32) -> Self {
        self.alpha = alpha;
        self.beta = beta;
        self
    }

    /// Toggle per-column scales.
    pub fn with_scales(mut self, use_scales: bool) -> Self {
        self.use_scales = use_scales;
        self
    }

    /// Packed bytes needed for k elements per column.
    pub fn packed_k_bytes(&self) -> usize {
        self.k.div_ceil(self.quant_bits.elems_per_byte())
    }

    /// Grid dimensions for CUDA launch.
    pub fn grid_dim(&self) -> (u32, u32, u32) {
        let gx = (self.n as u32).div_ceil(self.tile_n);
        let gy = (self.m as u32).div_ceil(self.tile_m);
        (gx, gy, self.batch_size as u32)
    }

    /// Block dimensions.
    pub fn block_dim(&self) -> (u32, u32, u32) {
        (32, 1, 1)
    }
}

/// Unpack a single INT8 or INT4 element from a packed buffer.
fn tc_unpack_int(packed: &[u8], index: usize, bits: TcQuantBits) -> f32 {
    match bits {
        TcQuantBits::Int8 => packed[index] as i8 as f32,
        TcQuantBits::Int4 => {
            let byte = packed[index / 2];
            let shift = (index % 2) * 4;
            let nibble = (byte >> shift) & 0x0F;
            let signed = if nibble & 0x08 != 0 { nibble as i8 | !0x0Fi8 } else { nibble as i8 };
            signed as f32
        }
    }
}

/// CPU fallback for quantized Tensor Core GEMM.
///
/// # Errors
///
/// Returns an error if buffer sizes are inconsistent.
pub fn tc_quantized_gemm_cpu(
    activations: &[f32],
    weights_packed: &[u8],
    scales: &[f32],
    output: &mut [f32],
    cfg: &TcQuantizedGemm,
) -> Result<()> {
    let (m, n, k, batch) = (cfg.m, cfg.n, cfg.k, cfg.batch_size);
    let packed_k = cfg.packed_k_bytes();

    if activations.len() < batch * m * k {
        return Err(buffer_error("activations", batch * m * k, activations.len()));
    }
    if weights_packed.len() < batch * packed_k * n {
        return Err(buffer_error("weights_packed", batch * packed_k * n, weights_packed.len()));
    }
    if cfg.use_scales && scales.len() < batch * n {
        return Err(buffer_error("scales", batch * n, scales.len()));
    }
    if output.len() < batch * m * n {
        return Err(buffer_error("output", batch * m * n, output.len()));
    }

    let a_stride = m * k;
    let w_stride = packed_k * n;
    let s_stride = n;
    let o_stride = m * n;

    for bi in 0..batch {
        for row in 0..m {
            for col in 0..n {
                let scale = if cfg.use_scales { scales[bi * s_stride + col] } else { 1.0 };
                let mut acc = 0.0f32;
                for l in 0..k {
                    let a_val = activations[bi * a_stride + row * k + l];
                    let w_base = bi * w_stride + col * packed_k;
                    let w_val = tc_unpack_int(&weights_packed[w_base..], l, cfg.quant_bits);
                    acc += a_val * w_val;
                }
                let idx = bi * o_stride + row * n + col;
                let val = cfg.alpha * acc * scale;
                output[idx] = if cfg.beta == 0.0 { val } else { val + cfg.beta * output[idx] };
            }
        }
    }
    Ok(())
}

// ── TcGroupedGemm ─────────────────────────────────────────────────────

/// Grouped GEMM for multi-head attention.
///
/// Each "group" is an independent GEMM with its own A/B/C slices but
/// shared M, N, K dimensions.
#[derive(Debug, Clone)]
pub struct TcGroupedGemm {
    /// Output rows per group.
    pub m: usize,
    /// Output columns per group.
    pub n: usize,
    /// Reduction dimension per group.
    pub k: usize,
    /// Number of groups (heads).
    pub num_groups: usize,
    /// Compute precision.
    pub precision: TcPrecision,
    /// Accumulation type.
    pub accumulation: TcAccumulation,
    /// Scalar α.
    pub alpha: f32,
    /// Scalar β.
    pub beta: f32,
}

impl TcGroupedGemm {
    /// Create a new grouped GEMM config.
    ///
    /// # Errors
    ///
    /// Returns an error if any dimension is zero.
    pub fn new(m: usize, n: usize, k: usize, num_groups: usize) -> Result<Self> {
        if m == 0 || n == 0 || k == 0 {
            return Err(KernelError::InvalidArguments {
                reason: format!("TcGroupedGemm dimensions must be non-zero: m={m}, n={n}, k={k}"),
            }
            .into());
        }
        if num_groups == 0 {
            return Err(
                KernelError::InvalidArguments { reason: "num_groups must be > 0".into() }.into()
            );
        }
        Ok(Self {
            m,
            n,
            k,
            num_groups,
            precision: TcPrecision::Fp16,
            accumulation: TcAccumulation::F32,
            alpha: 1.0,
            beta: 0.0,
        })
    }

    /// Set precision and accumulation.
    pub fn with_precision(mut self, p: TcPrecision, acc: TcAccumulation) -> Self {
        self.precision = p;
        self.accumulation = acc;
        self
    }

    /// Set α, β scalars.
    pub fn with_alpha_beta(mut self, alpha: f32, beta: f32) -> Self {
        self.alpha = alpha;
        self.beta = beta;
        self
    }

    /// Total FLOP count.
    pub fn total_flops(&self) -> u64 {
        2 * self.m as u64 * self.n as u64 * self.k as u64 * self.num_groups as u64
    }
}

/// CPU fallback for grouped GEMM.
///
/// # Errors
///
/// Returns an error if buffer sizes are inconsistent.
pub fn tc_grouped_gemm_cpu(a: &[f32], b: &[f32], c: &mut [f32], cfg: &TcGroupedGemm) -> Result<()> {
    let (m, n, k, g) = (cfg.m, cfg.n, cfg.k, cfg.num_groups);
    let a_stride = m * k;
    let b_stride = k * n;
    let c_stride = m * n;

    if a.len() < g * a_stride {
        return Err(buffer_error("A", g * a_stride, a.len()));
    }
    if b.len() < g * b_stride {
        return Err(buffer_error("B", g * b_stride, b.len()));
    }
    if c.len() < g * c_stride {
        return Err(buffer_error("C", g * c_stride, c.len()));
    }

    for gi in 0..g {
        let a_off = gi * a_stride;
        let b_off = gi * b_stride;
        let c_off = gi * c_stride;

        for i in 0..m {
            for j in 0..n {
                let mut acc = 0.0f32;
                for l in 0..k {
                    acc += a[a_off + i * k + l] * b[b_off + l * n + j];
                }
                let idx = c_off + i * n + j;
                c[idx] = cfg.alpha * acc + cfg.beta * c[idx];
            }
        }
    }
    Ok(())
}

// ── TcSplitK ──────────────────────────────────────────────────────────

/// Split-K GEMM for improved parallelism on small M.
///
/// Partitions the K dimension into `split_k` slices, each producing
/// partial sums that are then reduced.
#[derive(Debug, Clone)]
pub struct TcSplitK {
    /// Output rows.
    pub m: usize,
    /// Output columns.
    pub n: usize,
    /// Reduction dimension.
    pub k: usize,
    /// Number of K-partitions.
    pub split_k: u32,
    /// Compute precision.
    pub precision: TcPrecision,
    /// Accumulation type.
    pub accumulation: TcAccumulation,
    /// Scalar α.
    pub alpha: f32,
    /// Scalar β.
    pub beta: f32,
    /// Reduction mode: `true` = deterministic serial, `false` = atomic.
    pub deterministic_reduce: bool,
}

impl TcSplitK {
    /// Create a new split-K config.
    ///
    /// # Errors
    ///
    /// Returns an error if any dimension or split factor is zero.
    pub fn new(m: usize, n: usize, k: usize, split_k: u32) -> Result<Self> {
        if m == 0 || n == 0 || k == 0 {
            return Err(KernelError::InvalidArguments {
                reason: format!("TcSplitK dimensions must be non-zero: m={m}, n={n}, k={k}"),
            }
            .into());
        }
        if split_k == 0 {
            return Err(
                KernelError::InvalidArguments { reason: "split_k must be > 0".into() }.into()
            );
        }
        Ok(Self {
            m,
            n,
            k,
            split_k,
            precision: TcPrecision::Fp16,
            accumulation: TcAccumulation::F32,
            alpha: 1.0,
            beta: 0.0,
            deterministic_reduce: true,
        })
    }

    /// Set precision and accumulation.
    pub fn with_precision(mut self, p: TcPrecision, acc: TcAccumulation) -> Self {
        self.precision = p;
        self.accumulation = acc;
        self
    }

    /// Set α, β.
    pub fn with_alpha_beta(mut self, alpha: f32, beta: f32) -> Self {
        self.alpha = alpha;
        self.beta = beta;
        self
    }

    /// Set deterministic reduction mode.
    pub fn with_deterministic(mut self, det: bool) -> Self {
        self.deterministic_reduce = det;
        self
    }

    /// Size of each K-partition.
    pub fn k_per_split(&self) -> usize {
        self.k.div_ceil(self.split_k as usize)
    }

    /// Workspace elements needed for partial sums.
    pub fn workspace_elements(&self) -> usize {
        self.split_k as usize * self.m * self.n
    }
}

/// CPU fallback for split-K GEMM.
///
/// # Errors
///
/// Returns an error if buffer sizes are inconsistent.
pub fn tc_split_k_gemm_cpu(a: &[f32], b: &[f32], c: &mut [f32], cfg: &TcSplitK) -> Result<()> {
    let (m, n, k) = (cfg.m, cfg.n, cfg.k);

    if a.len() < m * k {
        return Err(buffer_error("A", m * k, a.len()));
    }
    if b.len() < k * n {
        return Err(buffer_error("B", k * n, b.len()));
    }
    if c.len() < m * n {
        return Err(buffer_error("C", m * n, c.len()));
    }

    let splits = cfg.split_k.max(1) as usize;
    let k_per = cfg.k_per_split();

    // Accumulate partial sums per split.
    let mut partials = vec![0.0f32; splits * m * n];

    for s in 0..splits {
        let k_start = s * k_per;
        let k_end = ((s + 1) * k_per).min(k);
        if k_start >= k {
            break;
        }
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0.0f32;
                for l in k_start..k_end {
                    acc += a[i * k + l] * b[l * n + j];
                }
                partials[s * m * n + i * n + j] = acc;
            }
        }
    }

    // Reduce.
    for i in 0..m {
        for j in 0..n {
            let mut total = 0.0f32;
            for s in 0..splits {
                total += partials[s * m * n + i * n + j];
            }
            let idx = i * n + j;
            let val = cfg.alpha * total;
            c[idx] = if cfg.beta == 0.0 { val } else { val + cfg.beta * c[idx] };
        }
    }
    Ok(())
}

// ── TcStreamK ─────────────────────────────────────────────────────────

/// Stream-K decomposition for load-balanced GEMM.
///
/// Distributes output tiles across a fixed number of virtual
/// processors (CTAs) for even work distribution.
#[derive(Debug, Clone)]
pub struct TcStreamK {
    /// Output rows.
    pub m: usize,
    /// Output columns.
    pub n: usize,
    /// Reduction dimension.
    pub k: usize,
    /// Number of CTAs (virtual processors).
    pub num_ctas: u32,
    /// Tile size M.
    pub tile_m: u32,
    /// Tile size N.
    pub tile_n: u32,
    /// Compute precision.
    pub precision: TcPrecision,
    /// Scalar α.
    pub alpha: f32,
    /// Scalar β.
    pub beta: f32,
}

impl TcStreamK {
    /// Create a new stream-K config.
    ///
    /// # Errors
    ///
    /// Returns an error if any dimension is zero.
    pub fn new(m: usize, n: usize, k: usize, num_ctas: u32) -> Result<Self> {
        if m == 0 || n == 0 || k == 0 {
            return Err(KernelError::InvalidArguments {
                reason: format!("TcStreamK dimensions must be non-zero: m={m}, n={n}, k={k}"),
            }
            .into());
        }
        if num_ctas == 0 {
            return Err(
                KernelError::InvalidArguments { reason: "num_ctas must be > 0".into() }.into()
            );
        }
        Ok(Self {
            m,
            n,
            k,
            num_ctas,
            tile_m: 16,
            tile_n: 16,
            precision: TcPrecision::Fp16,
            alpha: 1.0,
            beta: 0.0,
        })
    }

    /// Set tile dimensions.
    ///
    /// # Errors
    ///
    /// Returns an error if any tile is zero.
    pub fn with_tiles(mut self, tile_m: u32, tile_n: u32) -> Result<Self> {
        if tile_m == 0 || tile_n == 0 {
            return Err(KernelError::InvalidArguments {
                reason: format!("tile dimensions must be non-zero: {tile_m}×{tile_n}"),
            }
            .into());
        }
        self.tile_m = tile_m;
        self.tile_n = tile_n;
        Ok(self)
    }

    /// Set α, β.
    pub fn with_alpha_beta(mut self, alpha: f32, beta: f32) -> Self {
        self.alpha = alpha;
        self.beta = beta;
        self
    }

    /// Set compute precision.
    pub fn with_precision(mut self, p: TcPrecision) -> Self {
        self.precision = p;
        self
    }

    /// Total output tiles.
    pub fn total_tiles(&self) -> usize {
        let tm = self.m.div_ceil(self.tile_m as usize);
        let tn = self.n.div_ceil(self.tile_n as usize);
        tm * tn
    }

    /// Tiles per CTA (approximate).
    pub fn tiles_per_cta(&self) -> usize {
        self.total_tiles().div_ceil(self.num_ctas as usize)
    }
}

/// CPU fallback for stream-K GEMM.
///
/// # Errors
///
/// Returns an error if buffer sizes are inconsistent.
pub fn tc_stream_k_gemm_cpu(a: &[f32], b: &[f32], c: &mut [f32], cfg: &TcStreamK) -> Result<()> {
    let (m, n, k) = (cfg.m, cfg.n, cfg.k);

    if a.len() < m * k {
        return Err(buffer_error("A", m * k, a.len()));
    }
    if b.len() < k * n {
        return Err(buffer_error("B", k * n, b.len()));
    }
    if c.len() < m * n {
        return Err(buffer_error("C", m * n, c.len()));
    }

    let tile_m = cfg.tile_m as usize;
    let tile_n = cfg.tile_n as usize;
    let num_tiles_m = m.div_ceil(tile_m);
    let num_tiles_n = n.div_ceil(tile_n);
    let total = num_tiles_m * num_tiles_n;

    // Apply beta.
    if cfg.beta == 0.0 {
        c[..m * n].fill(0.0);
    } else if (cfg.beta - 1.0).abs() > f32::EPSILON {
        for v in c[..m * n].iter_mut() {
            *v *= cfg.beta;
        }
    }

    // Iterate tiles linearly (models stream-K scheduling).
    for tile_idx in 0..total {
        let tr = tile_idx / num_tiles_n;
        let tc_col = tile_idx % num_tiles_n;
        let i0 = tr * tile_m;
        let j0 = tc_col * tile_n;
        let i_end = (i0 + tile_m).min(m);
        let j_end = (j0 + tile_n).min(n);

        for i in i0..i_end {
            for j in j0..j_end {
                let mut acc = 0.0f32;
                for l in 0..k {
                    acc += a[i * k + l] * b[l * n + j];
                }
                c[i * n + j] += cfg.alpha * acc;
            }
        }
    }
    Ok(())
}

// ── CUDA kernel sources ───────────────────────────────────────────────

/// CUDA kernel source for Tensor Core batched GEMM.
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub const TC_BATCHED_GEMM_KERNEL_SRC: &str = r#"
extern "C" __global__ void tc_batched_gemm_f32(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K, int batch,
    float alpha, float beta)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int b   = blockIdx.z;
    if (row >= M || col >= N || b >= batch) return;

    int a_off = b * M * K;
    int b_off = b * K * N;
    int c_off = b * M * N;

    float acc = 0.0f;
    for (int l = 0; l < K; l++) {
        acc += A[a_off + row * K + l] * B[b_off + l * N + col];
    }
    int idx = c_off + row * N + col;
    C[idx] = alpha * acc + beta * C[idx];
}
"#;

/// CUDA kernel source for Tensor Core quantized GEMM (INT8/INT4).
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub const TC_QUANTIZED_GEMM_KERNEL_SRC: &str = r#"
extern "C" __global__ void tc_quantized_gemm_i8(
    const float* __restrict__ activations,
    const unsigned char* __restrict__ weights_packed,
    const float* __restrict__ scales,
    float* __restrict__ output,
    int M, int N, int K, int quant_bits,
    float alpha, float beta, int use_scales)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) return;

    int elems_per_byte = 8 / quant_bits;
    int packed_k = (K + elems_per_byte - 1) / elems_per_byte;
    float scale = use_scales ? scales[col] : 1.0f;
    float acc = 0.0f;

    for (int l = 0; l < K; l++) {
        int byte_idx = col * packed_k + l / elems_per_byte;
        int bit_off  = (l % elems_per_byte) * quant_bits;
        int mask     = (1 << quant_bits) - 1;
        unsigned char bits = (weights_packed[byte_idx] >> bit_off) & mask;
        float w;
        if (quant_bits == 4) {
            int s = (int)bits;
            if (s & 0x08) s |= ~0x0F;
            w = (float)s;
        } else {
            w = (float)((char)bits);
        }
        acc += activations[row * K + l] * w;
    }
    int idx = row * N + col;
    output[idx] = alpha * acc * scale + beta * output[idx];
}
"#;

/// CUDA kernel source for stream-K GEMM.
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub const TC_STREAM_K_KERNEL_SRC: &str = r#"
extern "C" __global__ void tc_stream_k_gemm_f32(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K,
    int tile_m, int tile_n,
    int num_tiles_n, int total_tiles,
    float alpha)
{
    int cta_id = blockIdx.x;
    int tiles_per_cta = (total_tiles + gridDim.x - 1) / gridDim.x;
    int start = cta_id * tiles_per_cta;
    int end   = min(start + tiles_per_cta, total_tiles);

    for (int t = start; t < end; t++) {
        int tr = t / num_tiles_n;
        int tc = t % num_tiles_n;
        int i0 = tr * tile_m;
        int j0 = tc * tile_n;

        int i = i0 + threadIdx.y;
        int j = j0 + threadIdx.x;
        if (i >= M || j >= N) continue;

        float acc = 0.0f;
        for (int l = 0; l < K; l++) {
            acc += A[i * K + l] * B[l * N + j];
        }
        atomicAdd(&C[i * N + j], alpha * acc);
    }
}
"#;

// ── CUDA launch stubs ─────────────────────────────────────────────────

/// Launch stub for batched Tensor Core GEMM.
///
/// # Errors
///
/// Returns `KernelError::GpuError` — scaffold only.
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub fn launch_tc_batched_gemm(
    _a: &[f32],
    _b: &[f32],
    _c: &mut [f32],
    config: &TcBatchedGemm,
) -> Result<()> {
    log::debug!(
        "TC batched GEMM CUDA stub: {}×{}×{} batch={} grid={:?}",
        config.m,
        config.n,
        config.k,
        config.batch_size,
        config.grid_dim(),
    );
    Err(KernelError::GpuError {
        reason: "TC batched GEMM CUDA kernel not yet compiled — scaffold only".into(),
    }
    .into())
}

/// Launch stub for quantized Tensor Core GEMM.
///
/// # Errors
///
/// Returns `KernelError::GpuError` — scaffold only.
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub fn launch_tc_quantized_gemm(
    _activations: &[f32],
    _weights_packed: &[u8],
    _scales: &[f32],
    _output: &mut [f32],
    config: &TcQuantizedGemm,
) -> Result<()> {
    log::debug!(
        "TC quantized GEMM CUDA stub: {}×{}×{} bits={:?} grid={:?}",
        config.m,
        config.n,
        config.k,
        config.quant_bits,
        config.grid_dim(),
    );
    Err(KernelError::GpuError {
        reason: "TC quantized GEMM CUDA kernel not yet compiled — scaffold only".into(),
    }
    .into())
}

/// Launch stub for stream-K GEMM.
///
/// # Errors
///
/// Returns `KernelError::GpuError` — scaffold only.
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub fn launch_tc_stream_k_gemm(
    _a: &[f32],
    _b: &[f32],
    _c: &mut [f32],
    config: &TcStreamK,
) -> Result<()> {
    log::debug!(
        "TC stream-K GEMM CUDA stub: {}×{}×{} ctas={} tiles={}",
        config.m,
        config.n,
        config.k,
        config.num_ctas,
        config.total_tiles(),
    );
    Err(KernelError::GpuError {
        reason: "TC stream-K GEMM CUDA kernel not yet compiled — scaffold only".into(),
    }
    .into())
}

// ── Unified dispatch ──────────────────────────────────────────────────

/// Batched GEMM dispatch: GPU-first with CPU fallback.
pub fn tc_batched_gemm(a: &[f32], b: &[f32], c: &mut [f32], cfg: &TcBatchedGemm) -> Result<()> {
    #[cfg(any(feature = "gpu", feature = "cuda"))]
    {
        if crate::device_features::gpu_available_runtime()
            && launch_tc_batched_gemm(a, b, c, cfg).is_ok()
        {
            return Ok(());
        }
    }
    tc_batched_gemm_cpu(a, b, c, cfg)
}

/// Quantized GEMM dispatch: GPU-first with CPU fallback.
pub fn tc_quantized_gemm(
    activations: &[f32],
    weights_packed: &[u8],
    scales: &[f32],
    output: &mut [f32],
    cfg: &TcQuantizedGemm,
) -> Result<()> {
    #[cfg(any(feature = "gpu", feature = "cuda"))]
    {
        if crate::device_features::gpu_available_runtime()
            && launch_tc_quantized_gemm(activations, weights_packed, scales, output, cfg).is_ok()
        {
            return Ok(());
        }
    }
    tc_quantized_gemm_cpu(activations, weights_packed, scales, output, cfg)
}

/// Stream-K GEMM dispatch: GPU-first with CPU fallback.
pub fn tc_stream_k_gemm(a: &[f32], b: &[f32], c: &mut [f32], cfg: &TcStreamK) -> Result<()> {
    #[cfg(any(feature = "gpu", feature = "cuda"))]
    {
        if crate::device_features::gpu_available_runtime()
            && launch_tc_stream_k_gemm(a, b, c, cfg).is_ok()
        {
            return Ok(());
        }
    }
    tc_stream_k_gemm_cpu(a, b, c, cfg)
}

/// Split-K GEMM dispatch: CPU fallback (no GPU launch stub yet).
pub fn tc_split_k_gemm(a: &[f32], b: &[f32], c: &mut [f32], cfg: &TcSplitK) -> Result<()> {
    tc_split_k_gemm_cpu(a, b, c, cfg)
}

/// Grouped GEMM dispatch: CPU fallback (no GPU launch stub yet).
pub fn tc_grouped_gemm(a: &[f32], b: &[f32], c: &mut [f32], cfg: &TcGroupedGemm) -> Result<()> {
    tc_grouped_gemm_cpu(a, b, c, cfg)
}

/// Convolution dispatch: CPU fallback (no GPU launch stub yet).
pub fn tc_convolution(
    input: &[f32],
    weight: &[f32],
    output: &mut [f32],
    cfg: &TcConvolution,
) -> Result<()> {
    tc_convolution_cpu(input, weight, output, cfg)
}

// ── Helpers ───────────────────────────────────────────────────────────

fn buffer_error(name: &str, expected: usize, actual: usize) -> BitNetError {
    BitNetError::Kernel(KernelError::ExecutionFailed {
        reason: format!("{name} buffer too small: expected {expected}, got {actual}"),
    })
}

// ── Tests ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(a: &[f32], b: &[f32], tol: f32) {
        assert_eq!(a.len(), b.len(), "length mismatch");
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            assert!((x - y).abs() <= tol, "mismatch at {i}: {x} vs {y} (tol {tol})");
        }
    }

    fn naive_matmul(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
        let mut c = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut s = 0.0f32;
                for l in 0..k {
                    s += a[i * k + l] * b[l * n + j];
                }
                c[i * n + j] = s;
            }
        }
        c
    }

    // ── TcPrecision tests ─────────────────────────────────────────

    #[test]
    fn test_precision_element_bytes() {
        assert_eq!(TcPrecision::Fp16.element_bytes(), 2);
        assert_eq!(TcPrecision::Bf16.element_bytes(), 2);
        assert_eq!(TcPrecision::Tf32.element_bytes(), 4);
        assert_eq!(TcPrecision::Int8.element_bytes(), 1);
        assert_eq!(TcPrecision::Int4.element_bytes(), 1);
    }

    // ── TcPrecisionPolicy tests ───────────────────────────────────

    #[test]
    fn test_policy_default_sm80() {
        let p = TcPrecisionPolicy::default();
        assert_eq!(p.sm_version, 80);
        assert!(p.tensor_cores_available());
        assert!(p.int8_tc_available());
        assert!(p.int4_tc_available());
    }

    #[test]
    fn test_policy_sm70_float_precision() {
        let p = TcPrecisionPolicy::for_sm(70);
        assert_eq!(p.select_float_precision(), TcPrecision::Fp16);
        assert!(p.tensor_cores_available());
        assert!(!p.int8_tc_available());
        assert!(!p.int4_tc_available());
    }

    #[test]
    fn test_policy_sm80_float_precision_tf32() {
        let p = TcPrecisionPolicy::for_sm(80);
        assert_eq!(p.select_float_precision(), TcPrecision::Tf32);
    }

    #[test]
    fn test_policy_sm80_prefer_fp16() {
        let p = TcPrecisionPolicy::for_sm(80).with_prefer_fp16(true);
        assert_eq!(p.select_float_precision(), TcPrecision::Fp16);
    }

    #[test]
    fn test_policy_forced_precision() {
        let p = TcPrecisionPolicy::for_sm(70).with_forced_precision(TcPrecision::Bf16);
        assert_eq!(p.select_float_precision(), TcPrecision::Bf16);
        assert_eq!(p.select_int_precision(), TcPrecision::Bf16);
    }

    #[test]
    fn test_policy_int_precision_sm80() {
        let p = TcPrecisionPolicy::for_sm(80);
        assert_eq!(p.select_int_precision(), TcPrecision::Int8);
    }

    #[test]
    fn test_policy_int_precision_sm70() {
        let p = TcPrecisionPolicy::for_sm(70);
        assert_eq!(p.select_int_precision(), TcPrecision::Fp16);
    }

    #[test]
    fn test_policy_accumulation_float() {
        let p = TcPrecisionPolicy::default();
        assert_eq!(p.select_accumulation(TcPrecision::Fp16), TcAccumulation::F32);
        assert_eq!(p.select_accumulation(TcPrecision::Tf32), TcAccumulation::F32);
    }

    #[test]
    fn test_policy_accumulation_int() {
        let p = TcPrecisionPolicy::default();
        assert_eq!(p.select_accumulation(TcPrecision::Int8), TcAccumulation::I32);
        assert_eq!(p.select_accumulation(TcPrecision::Int4), TcAccumulation::I32);
    }

    #[test]
    fn test_policy_wmma_tiles_fp16() {
        let p = TcPrecisionPolicy::default();
        assert_eq!(p.wmma_tile_size(TcPrecision::Fp16), (16, 16, 16));
    }

    #[test]
    fn test_policy_wmma_tiles_tf32() {
        let p = TcPrecisionPolicy::default();
        assert_eq!(p.wmma_tile_size(TcPrecision::Tf32), (16, 16, 8));
    }

    #[test]
    fn test_policy_wmma_tiles_int8() {
        let p = TcPrecisionPolicy::default();
        assert_eq!(p.wmma_tile_size(TcPrecision::Int8), (16, 16, 32));
    }

    #[test]
    fn test_policy_wmma_tiles_int4() {
        let p = TcPrecisionPolicy::default();
        assert_eq!(p.wmma_tile_size(TcPrecision::Int4), (8, 8, 32));
    }

    #[test]
    fn test_policy_sm60_no_tensor_cores() {
        let p = TcPrecisionPolicy::for_sm(60);
        assert!(!p.tensor_cores_available());
        assert!(!p.int8_tc_available());
        assert!(!p.int4_tc_available());
    }

    #[test]
    fn test_policy_sm75_int8_only() {
        let p = TcPrecisionPolicy::for_sm(75);
        assert!(p.tensor_cores_available());
        assert!(p.int8_tc_available());
        assert!(!p.int4_tc_available());
    }

    // ── TcBatchedGemm tests ───────────────────────────────────────

    #[test]
    fn test_batched_gemm_identity() {
        // A = I, B = I → C = I
        let cfg = TcBatchedGemm::new(2, 2, 2, 1).unwrap();
        let a = vec![1.0, 0.0, 0.0, 1.0];
        let b = vec![1.0, 0.0, 0.0, 1.0];
        let mut c = [0.0f32; 4];
        tc_batched_gemm_cpu(&a, &b, &mut c, &cfg).unwrap();
        assert_close(&c, &[1.0, 0.0, 0.0, 1.0], 1e-6);
    }

    #[test]
    fn test_batched_gemm_simple() {
        let cfg = TcBatchedGemm::new(2, 3, 2, 1).unwrap();
        let a = vec![1.0, 2.0, 3.0, 4.0]; // 2×2
        let b = vec![5.0, 6.0, 7.0, 8.0, 9.0, 10.0]; // 2×3
        let mut c = [0.0f32; 6];
        tc_batched_gemm_cpu(&a, &b, &mut c, &cfg).unwrap();
        let expected = naive_matmul(&a, &b, 2, 3, 2);
        assert_close(&c, &expected, 1e-5);
    }

    #[test]
    fn test_batched_gemm_multiple_batches() {
        let cfg = TcBatchedGemm::new(2, 2, 2, 3).unwrap();
        let a = vec![1.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 2.0, 3.0, 0.0, 0.0, 3.0];
        let b = [1.0; 12]; // all ones
        let mut c = [0.0f32; 12];
        tc_batched_gemm_cpu(&a, &b, &mut c, &cfg).unwrap();
        // batch 0: I * ones = [1,1; 1,1]
        assert_close(&c[0..4], &[1.0, 1.0, 1.0, 1.0], 1e-6);
        // batch 1: 2I * ones = [2,2; 2,2]
        assert_close(&c[4..8], &[2.0, 2.0, 2.0, 2.0], 1e-6);
    }

    #[test]
    fn test_batched_gemm_alpha_beta() {
        let cfg = TcBatchedGemm::new(2, 2, 2, 1).unwrap().with_alpha_beta(2.0, 1.0);
        let a = vec![1.0, 0.0, 0.0, 1.0];
        let b = vec![1.0, 0.0, 0.0, 1.0];
        let mut c = vec![1.0, 1.0, 1.0, 1.0];
        tc_batched_gemm_cpu(&a, &b, &mut c, &cfg).unwrap();
        // 2*I + 1*ones = [3,1; 1,3]
        assert_close(&c, &[3.0, 1.0, 1.0, 3.0], 1e-6);
    }

    #[test]
    fn test_batched_gemm_transpose_a() {
        let cfg = TcBatchedGemm::new(2, 2, 2, 1).unwrap().with_transpose(true, false);
        // A stored column-major for transpose
        let a = vec![1.0, 3.0, 2.0, 4.0]; // A^T = [[1,2],[3,4]]
        let b = vec![1.0, 0.0, 0.0, 1.0];
        let mut c = [0.0f32; 4];
        tc_batched_gemm_cpu(&a, &b, &mut c, &cfg).unwrap();
        assert_close(&c, &[1.0, 2.0, 3.0, 4.0], 1e-6);
    }

    #[test]
    fn test_batched_gemm_zero_dim_errors() {
        assert!(TcBatchedGemm::new(0, 2, 2, 1).is_err());
        assert!(TcBatchedGemm::new(2, 0, 2, 1).is_err());
        assert!(TcBatchedGemm::new(2, 2, 0, 1).is_err());
        assert!(TcBatchedGemm::new(2, 2, 2, 0).is_err());
    }

    #[test]
    fn test_batched_gemm_zero_tile_errors() {
        let cfg = TcBatchedGemm::new(4, 4, 4, 1).unwrap();
        assert!(cfg.with_tiles(0, 16, 16).is_err());
    }

    #[test]
    fn test_batched_gemm_grid_dim() {
        let cfg = TcBatchedGemm::new(32, 64, 16, 4).unwrap();
        let (gx, gy, gz) = cfg.grid_dim();
        assert_eq!(gx, 4); // 64/16
        assert_eq!(gy, 2); // 32/16
        assert_eq!(gz, 4);
    }

    #[test]
    fn test_batched_gemm_block_dim() {
        let cfg = TcBatchedGemm::new(4, 4, 4, 1).unwrap();
        assert_eq!(cfg.block_dim(), (32, 1, 1));
    }

    #[test]
    fn test_batched_gemm_shared_mem() {
        let cfg = TcBatchedGemm::new(4, 4, 4, 1).unwrap();
        assert!(cfg.shared_mem_bytes() >= 1024);
    }

    #[test]
    fn test_batched_gemm_gflops() {
        let cfg = TcBatchedGemm::new(64, 64, 64, 1).unwrap();
        let gflops = cfg.gflops(1.0);
        let expected = 2.0 * 64.0 * 64.0 * 64.0 / 1e9;
        assert!((gflops - expected).abs() < 1e-6);
    }

    #[test]
    fn test_batched_gemm_buffer_too_small_a() {
        let cfg = TcBatchedGemm::new(4, 4, 4, 1).unwrap();
        let a = [0.0f32; 8]; // need 16
        let b = [0.0f32; 16];
        let mut c = [0.0f32; 16];
        assert!(tc_batched_gemm_cpu(&a, &b, &mut c, &cfg).is_err());
    }

    #[test]
    fn test_batched_gemm_buffer_too_small_b() {
        let cfg = TcBatchedGemm::new(4, 4, 4, 1).unwrap();
        let a = [0.0f32; 16];
        let b = [0.0f32; 8]; // need 16
        let mut c = [0.0f32; 16];
        assert!(tc_batched_gemm_cpu(&a, &b, &mut c, &cfg).is_err());
    }

    #[test]
    fn test_batched_gemm_buffer_too_small_c() {
        let cfg = TcBatchedGemm::new(4, 4, 4, 1).unwrap();
        let a = [0.0f32; 16];
        let b = [0.0f32; 16];
        let mut c = [0.0f32; 8]; // need 16
        assert!(tc_batched_gemm_cpu(&a, &b, &mut c, &cfg).is_err());
    }

    #[test]
    fn test_batched_gemm_unified_dispatch() {
        let cfg = TcBatchedGemm::new(2, 2, 2, 1).unwrap();
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        let mut c = [0.0f32; 4];
        tc_batched_gemm(&a, &b, &mut c, &cfg).unwrap();
        let expected = naive_matmul(&a, &b, 2, 2, 2);
        assert_close(&c, &expected, 1e-5);
    }

    #[test]
    fn test_batched_gemm_large_batch() {
        let cfg = TcBatchedGemm::new(4, 4, 4, 8).unwrap();
        let a = vec![1.0f32; 8 * 16];
        let b = vec![0.25f32; 8 * 16];
        let mut c = vec![0.0f32; 8 * 16];
        tc_batched_gemm_cpu(&a, &b, &mut c, &cfg).unwrap();
        // Each batch: 4×4 ones * 4×4 0.25 = 4×4 of 1.0
        for bi in 0..8 {
            for v in &c[bi * 16..(bi + 1) * 16] {
                assert!((v - 1.0).abs() < 1e-5);
            }
        }
    }

    // ── TcConvolution tests ───────────────────────────────────────

    #[test]
    fn test_conv_output_dims_no_padding() {
        let cfg = TcConvolution::new(1, 1, 1, 4, 4, 3, 3).unwrap();
        assert_eq!(cfg.output_h(), 2);
        assert_eq!(cfg.output_w(), 2);
    }

    #[test]
    fn test_conv_output_dims_with_padding() {
        let cfg = TcConvolution::new(1, 1, 1, 4, 4, 3, 3).unwrap().with_params(1, 1, 1).unwrap();
        assert_eq!(cfg.output_h(), 4);
        assert_eq!(cfg.output_w(), 4);
    }

    #[test]
    fn test_conv_output_dims_stride2() {
        let cfg = TcConvolution::new(1, 1, 1, 8, 8, 3, 3).unwrap().with_params(2, 1, 1).unwrap();
        assert_eq!(cfg.output_h(), 4);
        assert_eq!(cfg.output_w(), 4);
    }

    #[test]
    fn test_conv_output_dims_dilation() {
        let cfg = TcConvolution::new(1, 1, 1, 7, 7, 3, 3).unwrap().with_params(1, 0, 2).unwrap();
        // effective kernel = (3-1)*2+1 = 5, output = 7-5+1 = 3
        assert_eq!(cfg.output_h(), 3);
        assert_eq!(cfg.output_w(), 3);
    }

    #[test]
    fn test_conv_gemm_dims() {
        let cfg = TcConvolution::new(2, 3, 8, 16, 16, 3, 3).unwrap();
        assert_eq!(cfg.gemm_m(), 2 * 14 * 14); // batch * oh * ow
        assert_eq!(cfg.gemm_k(), 3 * 3 * 3); // ic * kh * kw
        assert_eq!(cfg.gemm_n(), 8);
    }

    #[test]
    fn test_conv_1x1_is_matmul() {
        // 1×1 conv is equivalent to a per-pixel matmul
        let cfg = TcConvolution::new(1, 2, 3, 4, 4, 1, 1).unwrap();
        let input = vec![1.0f32; 2 * 4 * 4]; // [1, 2, 4, 4]
        let weight = vec![1.0f32; 3 * 2]; // [3, 2, 1, 1]
        let mut output = vec![0.0f32; 3 * 4 * 4];
        tc_convolution_cpu(&input, &weight, &mut output, &cfg).unwrap();
        // Each output pixel = sum of 2 input channels = 2.0
        for v in &output {
            assert!((v - 2.0).abs() < 1e-5);
        }
    }

    #[test]
    fn test_conv_zero_dim_errors() {
        assert!(TcConvolution::new(0, 1, 1, 4, 4, 3, 3).is_err());
        assert!(TcConvolution::new(1, 0, 1, 4, 4, 3, 3).is_err());
        assert!(TcConvolution::new(1, 1, 0, 4, 4, 3, 3).is_err());
        assert!(TcConvolution::new(1, 1, 1, 0, 4, 3, 3).is_err());
    }

    #[test]
    fn test_conv_zero_stride_error() {
        let cfg = TcConvolution::new(1, 1, 1, 4, 4, 3, 3).unwrap();
        assert!(cfg.with_params(0, 0, 1).is_err());
    }

    #[test]
    fn test_conv_zero_dilation_error() {
        let cfg = TcConvolution::new(1, 1, 1, 4, 4, 3, 3).unwrap();
        assert!(cfg.with_params(1, 0, 0).is_err());
    }

    #[test]
    fn test_conv_simple_3x3() {
        // 1 batch, 1 in, 1 out, 3×3 input, 3×3 kernel → 1×1 output
        let cfg = TcConvolution::new(1, 1, 1, 3, 3, 3, 3).unwrap();
        let input: Vec<f32> = (1..=9).map(|x| x as f32).collect();
        let weight = [1.0f32; 9];
        let mut output = [0.0f32; 1];
        tc_convolution_cpu(&input, &weight, &mut output, &cfg).unwrap();
        assert!((output[0] - 45.0).abs() < 1e-5);
    }

    #[test]
    fn test_conv_with_padding_3x3() {
        let cfg = TcConvolution::new(1, 1, 1, 3, 3, 3, 3).unwrap().with_params(1, 1, 1).unwrap();
        let input = [1.0f32; 9];
        let weight = [1.0f32; 9];
        let mut output = [0.0f32; 9]; // 3×3 output with padding=1
        tc_convolution_cpu(&input, &weight, &mut output, &cfg).unwrap();
        // center pixel sees all 9 inputs → 9.0
        assert!((output[4] - 9.0).abs() < 1e-5);
        // corner pixel sees 4 inputs → 4.0
        assert!((output[0] - 4.0).abs() < 1e-5);
    }

    #[test]
    fn test_conv_buffer_too_small() {
        let cfg = TcConvolution::new(1, 1, 1, 4, 4, 3, 3).unwrap();
        let input = [0.0f32; 4]; // too small
        let weight = [0.0f32; 9];
        let mut output = [0.0f32; 4];
        assert!(tc_convolution_cpu(&input, &weight, &mut output, &cfg).is_err());
    }

    #[test]
    fn test_conv_multi_channel() {
        let cfg = TcConvolution::new(1, 2, 1, 2, 2, 1, 1).unwrap();
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]; // [1,2,2,2]
        let weight = vec![1.0, 1.0]; // [1, 2, 1, 1]
        let mut output = [0.0f32; 4];
        tc_convolution_cpu(&input, &weight, &mut output, &cfg).unwrap();
        // pixel (0,0) = 1*1 + 5*1 = 6
        assert!((output[0] - 6.0).abs() < 1e-5);
    }

    #[test]
    fn test_conv_precision_setter() {
        let cfg =
            TcConvolution::new(1, 1, 1, 4, 4, 3, 3).unwrap().with_precision(TcPrecision::Tf32);
        assert_eq!(cfg.precision, TcPrecision::Tf32);
    }

    // ── TcQuantizedGemm tests ─────────────────────────────────────

    #[test]
    fn test_tc_quant_bits() {
        assert_eq!(TcQuantBits::Int8.bits(), 8);
        assert_eq!(TcQuantBits::Int4.bits(), 4);
        assert_eq!(TcQuantBits::Int8.elems_per_byte(), 1);
        assert_eq!(TcQuantBits::Int4.elems_per_byte(), 2);
    }

    #[test]
    fn test_tc_quant_min_sm() {
        assert_eq!(TcQuantBits::Int8.min_sm(), 75);
        assert_eq!(TcQuantBits::Int4.min_sm(), 80);
    }

    #[test]
    fn test_tc_quant_gemm_int8_identity() {
        let cfg = TcQuantizedGemm::new(2, 2, 2, TcQuantBits::Int8).unwrap();
        // Weight matrix = identity as INT8
        let weights = vec![1u8, 0u8, 0u8, 1u8]; // row-major per col
        let scales = [1.0f32; 2];
        let a = vec![1.0, 0.0, 0.0, 1.0];
        let mut out = [0.0f32; 4];
        tc_quantized_gemm_cpu(&a, &weights, &scales, &mut out, &cfg).unwrap();
        assert_close(&out, &[1.0, 0.0, 0.0, 1.0], 1e-5);
    }

    #[test]
    fn test_tc_quant_gemm_int8_simple() {
        let cfg = TcQuantizedGemm::new(1, 2, 4, TcQuantBits::Int8).unwrap();
        let a = vec![1.0, 2.0, 3.0, 4.0]; // 1×4
        // col 0: [1,1,1,1], col 1: [2,2,2,2]
        let weights = vec![1u8, 1, 1, 1, 2, 2, 2, 2];
        let scales = [1.0f32; 2];
        let mut out = [0.0f32; 2];
        tc_quantized_gemm_cpu(&a, &weights, &scales, &mut out, &cfg).unwrap();
        // col 0: 1+2+3+4=10, col 1: 2+4+6+8=20
        assert_close(&out, &[10.0, 20.0], 1e-5);
    }

    #[test]
    fn test_tc_quant_gemm_int4_simple() {
        let cfg = TcQuantizedGemm::new(1, 1, 4, TcQuantBits::Int4).unwrap();
        let a = [1.0f32; 4];
        // Pack 4 nibbles: [1, 2, 3, 4] → 2 bytes
        let weights = vec![0x21u8, 0x43]; // nibble 0=1, 1=2, 2=3, 3=4
        let scales = vec![1.0f32];
        let mut out = [0.0f32; 1];
        tc_quantized_gemm_cpu(&a, &weights, &scales, &mut out, &cfg).unwrap();
        assert!((out[0] - 10.0).abs() < 1e-5);
    }

    #[test]
    fn test_tc_quant_gemm_with_scales() {
        let cfg = TcQuantizedGemm::new(1, 2, 2, TcQuantBits::Int8).unwrap();
        let a = vec![1.0, 1.0];
        let weights = vec![1u8, 1, 1, 1]; // col0=[1,1], col1=[1,1]
        let scales = vec![2.0f32, 3.0]; // scale col 0 by 2, col 1 by 3
        let mut out = [0.0f32; 2];
        tc_quantized_gemm_cpu(&a, &weights, &scales, &mut out, &cfg).unwrap();
        assert_close(&out, &[4.0, 6.0], 1e-5);
    }

    #[test]
    fn test_tc_quant_gemm_no_scales() {
        let cfg = TcQuantizedGemm::new(1, 2, 2, TcQuantBits::Int8).unwrap().with_scales(false);
        let a = vec![1.0, 1.0];
        let weights = vec![1u8, 1, 1, 1];
        let scales = vec![];
        let mut out = [0.0f32; 2];
        tc_quantized_gemm_cpu(&a, &weights, &scales, &mut out, &cfg).unwrap();
        assert_close(&out, &[2.0, 2.0], 1e-5);
    }

    #[test]
    fn test_tc_quant_gemm_alpha_beta() {
        let cfg =
            TcQuantizedGemm::new(1, 1, 2, TcQuantBits::Int8).unwrap().with_alpha_beta(2.0, 1.0);
        let a = vec![1.0, 1.0];
        let weights = vec![1u8, 1];
        let scales = vec![1.0f32];
        let mut out = vec![10.0f32];
        tc_quantized_gemm_cpu(&a, &weights, &scales, &mut out, &cfg).unwrap();
        // 2*2*1 + 1*10 = 14
        assert!((out[0] - 14.0).abs() < 1e-5);
    }

    #[test]
    fn test_tc_quant_gemm_batch() {
        let cfg =
            TcQuantizedGemm::new(1, 1, 2, TcQuantBits::Int8).unwrap().with_batch_size(2).unwrap();
        let a = vec![1.0, 1.0, 2.0, 2.0]; // 2 batches
        let weights = vec![1u8, 1, 1, 1];
        let scales = vec![1.0f32, 1.0];
        let mut out = [0.0f32; 2];
        tc_quantized_gemm_cpu(&a, &weights, &scales, &mut out, &cfg).unwrap();
        assert_close(&out, &[2.0, 4.0], 1e-5);
    }

    #[test]
    fn test_tc_quant_gemm_zero_dim_error() {
        assert!(TcQuantizedGemm::new(0, 4, 4, TcQuantBits::Int8).is_err());
    }

    #[test]
    fn test_tc_quant_gemm_zero_batch_error() {
        let cfg = TcQuantizedGemm::new(4, 4, 4, TcQuantBits::Int8).unwrap();
        assert!(cfg.with_batch_size(0).is_err());
    }

    #[test]
    fn test_tc_quant_gemm_grid_dim() {
        let cfg = TcQuantizedGemm::new(32, 64, 128, TcQuantBits::Int8).unwrap();
        let (gx, gy, _gz) = cfg.grid_dim();
        assert_eq!(gx, 4); // 64/16
        assert_eq!(gy, 2); // 32/16
    }

    #[test]
    fn test_tc_quant_gemm_packed_k() {
        let cfg8 = TcQuantizedGemm::new(4, 4, 10, TcQuantBits::Int8).unwrap();
        assert_eq!(cfg8.packed_k_bytes(), 10);
        let cfg4 = TcQuantizedGemm::new(4, 4, 10, TcQuantBits::Int4).unwrap();
        assert_eq!(cfg4.packed_k_bytes(), 5);
    }

    #[test]
    fn test_tc_quant_gemm_unified_dispatch() {
        let cfg = TcQuantizedGemm::new(1, 1, 2, TcQuantBits::Int8).unwrap();
        let a = vec![1.0, 1.0];
        let weights = vec![3u8, 3];
        let scales = vec![1.0f32];
        let mut out = [0.0f32; 1];
        tc_quantized_gemm(&a, &weights, &scales, &mut out, &cfg).unwrap();
        assert!((out[0] - 6.0).abs() < 1e-5);
    }

    #[test]
    fn test_tc_quant_gemm_buffer_too_small() {
        let cfg = TcQuantizedGemm::new(4, 4, 4, TcQuantBits::Int8).unwrap();
        let a = [0.0f32; 4]; // need 16
        let w = [0u8; 16];
        let s = [1.0f32; 4];
        let mut out = [0.0f32; 16];
        assert!(tc_quantized_gemm_cpu(&a, &w, &s, &mut out, &cfg).is_err());
    }

    // ── TcGroupedGemm tests ──────────────────────────────────────

    #[test]
    fn test_grouped_gemm_single_group() {
        let cfg = TcGroupedGemm::new(2, 2, 2, 1).unwrap();
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        let mut c = [0.0f32; 4];
        tc_grouped_gemm_cpu(&a, &b, &mut c, &cfg).unwrap();
        let expected = naive_matmul(&a, &b, 2, 2, 2);
        assert_close(&c, &expected, 1e-5);
    }

    #[test]
    fn test_grouped_gemm_multi_head() {
        let cfg = TcGroupedGemm::new(2, 2, 2, 4).unwrap();
        let a = vec![1.0f32; 4 * 4];
        let b = vec![0.5f32; 4 * 4];
        let mut c = vec![0.0f32; 4 * 4];
        tc_grouped_gemm_cpu(&a, &b, &mut c, &cfg).unwrap();
        // Each group: 2×2 ones * 2×2 0.5s = 2×2 of 1.0
        for v in &c {
            assert!((v - 1.0).abs() < 1e-5);
        }
    }

    #[test]
    fn test_grouped_gemm_alpha_beta() {
        let cfg = TcGroupedGemm::new(2, 2, 2, 1).unwrap().with_alpha_beta(0.5, 2.0);
        let a = vec![1.0, 0.0, 0.0, 1.0];
        let b = vec![1.0, 0.0, 0.0, 1.0];
        let mut c = vec![1.0, 1.0, 1.0, 1.0];
        tc_grouped_gemm_cpu(&a, &b, &mut c, &cfg).unwrap();
        // 0.5*I + 2*ones = [2.5, 2.0, 2.0, 2.5]
        assert_close(&c, &[2.5, 2.0, 2.0, 2.5], 1e-6);
    }

    #[test]
    fn test_grouped_gemm_zero_dim_errors() {
        assert!(TcGroupedGemm::new(0, 2, 2, 1).is_err());
        assert!(TcGroupedGemm::new(2, 2, 2, 0).is_err());
    }

    #[test]
    fn test_grouped_gemm_total_flops() {
        let cfg = TcGroupedGemm::new(64, 64, 64, 8).unwrap();
        assert_eq!(cfg.total_flops(), 2 * 64 * 64 * 64 * 8);
    }

    #[test]
    fn test_grouped_gemm_precision_setter() {
        let cfg = TcGroupedGemm::new(4, 4, 4, 1)
            .unwrap()
            .with_precision(TcPrecision::Tf32, TcAccumulation::F32);
        assert_eq!(cfg.precision, TcPrecision::Tf32);
        assert_eq!(cfg.accumulation, TcAccumulation::F32);
    }

    #[test]
    fn test_grouped_gemm_buffer_too_small() {
        let cfg = TcGroupedGemm::new(4, 4, 4, 2).unwrap();
        let a = [0.0f32; 16]; // need 32
        let b = [0.0f32; 32];
        let mut c = [0.0f32; 32];
        assert!(tc_grouped_gemm_cpu(&a, &b, &mut c, &cfg).is_err());
    }

    // ── TcSplitK tests ───────────────────────────────────────────

    #[test]
    fn test_split_k_matches_naive() {
        let cfg = TcSplitK::new(4, 4, 16, 4).unwrap();
        let a: Vec<f32> = (0..64).map(|i| (i % 5) as f32).collect();
        let b: Vec<f32> = (0..64).map(|i| (i % 3) as f32).collect();
        let mut c = [0.0f32; 16];
        tc_split_k_gemm_cpu(&a, &b, &mut c, &cfg).unwrap();
        let expected = naive_matmul(&a, &b, 4, 4, 16);
        assert_close(&c, &expected, 1e-4);
    }

    #[test]
    fn test_split_k_single_split() {
        let cfg = TcSplitK::new(2, 2, 4, 1).unwrap();
        let a = [1.0f32; 8];
        let b = [0.5f32; 8];
        let mut c = [0.0f32; 4];
        tc_split_k_gemm_cpu(&a, &b, &mut c, &cfg).unwrap();
        let expected = naive_matmul(&a, &b, 2, 2, 4);
        assert_close(&c, &expected, 1e-5);
    }

    #[test]
    fn test_split_k_alpha_beta() {
        let cfg = TcSplitK::new(2, 2, 4, 2).unwrap().with_alpha_beta(2.0, 1.0);
        let a = [1.0f32; 8];
        let b = [1.0f32; 8];
        let mut c = [1.0f32; 4];
        tc_split_k_gemm_cpu(&a, &b, &mut c, &cfg).unwrap();
        // naive result = 4.0 per element, alpha=2 → 8.0, + beta*1 = 9.0
        for v in &c {
            assert!((v - 9.0).abs() < 1e-4);
        }
    }

    #[test]
    fn test_split_k_zero_dim_errors() {
        assert!(TcSplitK::new(0, 2, 4, 2).is_err());
        assert!(TcSplitK::new(2, 0, 4, 2).is_err());
        assert!(TcSplitK::new(2, 2, 0, 2).is_err());
        assert!(TcSplitK::new(2, 2, 4, 0).is_err());
    }

    #[test]
    fn test_split_k_workspace_elements() {
        let cfg = TcSplitK::new(4, 4, 32, 8).unwrap();
        assert_eq!(cfg.workspace_elements(), 8 * 4 * 4);
    }

    #[test]
    fn test_split_k_per_split() {
        let cfg = TcSplitK::new(4, 4, 32, 8).unwrap();
        assert_eq!(cfg.k_per_split(), 4);
    }

    #[test]
    fn test_split_k_deterministic_flag() {
        let cfg = TcSplitK::new(4, 4, 4, 2).unwrap().with_deterministic(false);
        assert!(!cfg.deterministic_reduce);
    }

    #[test]
    fn test_split_k_buffer_too_small() {
        let cfg = TcSplitK::new(4, 4, 4, 2).unwrap();
        let a = [0.0f32; 8]; // need 16
        let b = [0.0f32; 16];
        let mut c = [0.0f32; 16];
        assert!(tc_split_k_gemm_cpu(&a, &b, &mut c, &cfg).is_err());
    }

    #[test]
    fn test_split_k_uneven_partition() {
        // k=7, split_k=3 → partitions of ~3,3,1
        let cfg = TcSplitK::new(2, 2, 7, 3).unwrap();
        let a: Vec<f32> = (0..14).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..14).map(|i| (i % 3) as f32).collect();
        let mut c = [0.0f32; 4];
        tc_split_k_gemm_cpu(&a, &b, &mut c, &cfg).unwrap();
        let expected = naive_matmul(&a, &b, 2, 2, 7);
        assert_close(&c, &expected, 1e-4);
    }

    // ── TcStreamK tests ──────────────────────────────────────────

    #[test]
    fn test_stream_k_matches_naive() {
        let cfg = TcStreamK::new(4, 4, 8, 4).unwrap();
        let a: Vec<f32> = (0..32).map(|i| (i % 4) as f32).collect();
        let b: Vec<f32> = (0..32).map(|i| (i % 3) as f32).collect();
        let mut c = [0.0f32; 16];
        tc_stream_k_gemm_cpu(&a, &b, &mut c, &cfg).unwrap();
        let expected = naive_matmul(&a, &b, 4, 4, 8);
        assert_close(&c, &expected, 1e-4);
    }

    #[test]
    fn test_stream_k_single_cta() {
        let cfg = TcStreamK::new(2, 2, 2, 1).unwrap();
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        let mut c = [0.0f32; 4];
        tc_stream_k_gemm_cpu(&a, &b, &mut c, &cfg).unwrap();
        let expected = naive_matmul(&a, &b, 2, 2, 2);
        assert_close(&c, &expected, 1e-5);
    }

    #[test]
    fn test_stream_k_alpha_beta() {
        let cfg = TcStreamK::new(2, 2, 2, 2).unwrap().with_alpha_beta(0.5, 2.0);
        let a = vec![1.0, 0.0, 0.0, 1.0];
        let b = vec![1.0, 0.0, 0.0, 1.0];
        let mut c = [1.0f32; 4];
        tc_stream_k_gemm_cpu(&a, &b, &mut c, &cfg).unwrap();
        // 0.5*I + 2*ones = [2.5, 2.0, 2.0, 2.5]
        assert_close(&c, &[2.5, 2.0, 2.0, 2.5], 1e-5);
    }

    #[test]
    fn test_stream_k_zero_dim_errors() {
        assert!(TcStreamK::new(0, 2, 4, 2).is_err());
        assert!(TcStreamK::new(2, 0, 4, 2).is_err());
        assert!(TcStreamK::new(2, 2, 0, 2).is_err());
        assert!(TcStreamK::new(2, 2, 4, 0).is_err());
    }

    #[test]
    fn test_stream_k_total_tiles() {
        let cfg = TcStreamK::new(32, 64, 16, 8).unwrap();
        let tm = 32usize.div_ceil(16);
        let tn = 64usize.div_ceil(16);
        assert_eq!(cfg.total_tiles(), tm * tn);
    }

    #[test]
    fn test_stream_k_tiles_per_cta() {
        let cfg = TcStreamK::new(32, 32, 16, 4).unwrap();
        let total = cfg.total_tiles(); // 2*2 = 4
        assert_eq!(cfg.tiles_per_cta(), total.div_ceil(4));
    }

    #[test]
    fn test_stream_k_custom_tiles() {
        let cfg = TcStreamK::new(64, 64, 32, 8).unwrap().with_tiles(32, 32).unwrap();
        assert_eq!(cfg.tile_m, 32);
        assert_eq!(cfg.tile_n, 32);
        assert_eq!(cfg.total_tiles(), 4);
    }

    #[test]
    fn test_stream_k_zero_tiles_error() {
        let cfg = TcStreamK::new(4, 4, 4, 2).unwrap();
        assert!(cfg.with_tiles(0, 16).is_err());
    }

    #[test]
    fn test_stream_k_buffer_too_small() {
        let cfg = TcStreamK::new(4, 4, 4, 2).unwrap();
        let a = [0.0f32; 8]; // need 16
        let b = [0.0f32; 16];
        let mut c = [0.0f32; 16];
        assert!(tc_stream_k_gemm_cpu(&a, &b, &mut c, &cfg).is_err());
    }

    #[test]
    fn test_stream_k_unified_dispatch() {
        let cfg = TcStreamK::new(2, 2, 2, 2).unwrap();
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        let mut c = [0.0f32; 4];
        tc_stream_k_gemm(&a, &b, &mut c, &cfg).unwrap();
        let expected = naive_matmul(&a, &b, 2, 2, 2);
        assert_close(&c, &expected, 1e-5);
    }

    #[test]
    fn test_stream_k_precision_setter() {
        let cfg = TcStreamK::new(4, 4, 4, 2).unwrap().with_precision(TcPrecision::Bf16);
        assert_eq!(cfg.precision, TcPrecision::Bf16);
    }

    // ── Cross-operation consistency tests ─────────────────────────

    #[test]
    fn test_batched_vs_grouped_single() {
        // Single batch/group should give identical results.
        let m = 4;
        let n = 4;
        let k = 8;
        let a: Vec<f32> = (0..32).map(|i| (i % 5) as f32).collect();
        let b: Vec<f32> = (0..32).map(|i| (i % 3) as f32).collect();

        let bcfg = TcBatchedGemm::new(m, n, k, 1).unwrap();
        let gcfg = TcGroupedGemm::new(m, n, k, 1).unwrap();

        let mut c_batch = [0.0f32; 16];
        let mut c_group = [0.0f32; 16];
        tc_batched_gemm_cpu(&a, &b, &mut c_batch, &bcfg).unwrap();
        tc_grouped_gemm_cpu(&a, &b, &mut c_group, &gcfg).unwrap();
        assert_close(&c_batch, &c_group, 1e-5);
    }

    #[test]
    fn test_split_k_vs_stream_k() {
        let m = 4;
        let n = 4;
        let k = 16;
        let a: Vec<f32> = (0..64).map(|i| (i % 7) as f32).collect();
        let b: Vec<f32> = (0..64).map(|i| (i % 5) as f32).collect();

        let sk_cfg = TcSplitK::new(m, n, k, 4).unwrap();
        let stk_cfg = TcStreamK::new(m, n, k, 4).unwrap();

        let mut c_sk = [0.0f32; 16];
        let mut c_stk = [0.0f32; 16];
        tc_split_k_gemm_cpu(&a, &b, &mut c_sk, &sk_cfg).unwrap();
        tc_stream_k_gemm_cpu(&a, &b, &mut c_stk, &stk_cfg).unwrap();
        assert_close(&c_sk, &c_stk, 1e-4);
    }

    #[test]
    fn test_all_paths_agree() {
        let m = 3;
        let n = 5;
        let k = 7;
        let a: Vec<f32> = (0..21).map(|i| (i as f32) * 0.1).collect();
        let b: Vec<f32> = (0..35).map(|i| (i as f32) * 0.2).collect();
        let expected = naive_matmul(&a, &b, m, n, k);

        // Batched
        let bcfg = TcBatchedGemm::new(m, n, k, 1).unwrap();
        let mut c1 = [0.0f32; 15];
        tc_batched_gemm_cpu(&a, &b, &mut c1, &bcfg).unwrap();
        assert_close(&c1, &expected, 1e-3);

        // Grouped
        let gcfg = TcGroupedGemm::new(m, n, k, 1).unwrap();
        let mut c2 = [0.0f32; 15];
        tc_grouped_gemm_cpu(&a, &b, &mut c2, &gcfg).unwrap();
        assert_close(&c2, &expected, 1e-3);

        // Split-K
        let sk_cfg = TcSplitK::new(m, n, k, 3).unwrap();
        let mut c3 = [0.0f32; 15];
        tc_split_k_gemm_cpu(&a, &b, &mut c3, &sk_cfg).unwrap();
        assert_close(&c3, &expected, 1e-3);

        // Stream-K
        let stk_cfg = TcStreamK::new(m, n, k, 4).unwrap();
        let mut c4 = [0.0f32; 15];
        tc_stream_k_gemm_cpu(&a, &b, &mut c4, &stk_cfg).unwrap();
        assert_close(&c4, &expected, 1e-3);
    }

    #[test]
    fn test_precision_policy_with_batched_gemm() {
        let policy = TcPrecisionPolicy::for_sm(80);
        let precision = policy.select_float_precision();
        let acc = policy.select_accumulation(precision);
        let cfg = TcBatchedGemm::new(4, 4, 4, 1).unwrap().with_precision(precision, acc);
        assert_eq!(cfg.precision, TcPrecision::Tf32);
        assert_eq!(cfg.accumulation, TcAccumulation::F32);
    }

    #[test]
    fn test_precision_policy_int_with_quant_gemm() {
        let policy = TcPrecisionPolicy::for_sm(80);
        assert!(policy.int8_tc_available());
        let _cfg = TcQuantizedGemm::new(8, 8, 32, TcQuantBits::Int8).unwrap();
    }

    // ── Edge-case tests ──────────────────────────────────────────

    #[test]
    fn test_batched_gemm_1x1x1() {
        let cfg = TcBatchedGemm::new(1, 1, 1, 1).unwrap();
        let a = vec![3.0f32];
        let b = vec![4.0f32];
        let mut c = vec![0.0f32];
        tc_batched_gemm_cpu(&a, &b, &mut c, &cfg).unwrap();
        assert!((c[0] - 12.0).abs() < 1e-6);
    }

    #[test]
    fn test_split_k_1x1() {
        let cfg = TcSplitK::new(1, 1, 4, 4).unwrap();
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![1.0, 1.0, 1.0, 1.0];
        let mut c = vec![0.0f32];
        tc_split_k_gemm_cpu(&a, &b, &mut c, &cfg).unwrap();
        assert!((c[0] - 10.0).abs() < 1e-5);
    }

    #[test]
    fn test_conv_batch_size() {
        let cfg = TcConvolution::new(2, 1, 1, 3, 3, 1, 1).unwrap();
        let input = vec![1.0f32; 2 * 9]; // 2 batches of 1×3×3
        let weight = vec![2.0f32]; // 1×1×1×1
        let mut output = vec![0.0f32; 2 * 9];
        tc_convolution_cpu(&input, &weight, &mut output, &cfg).unwrap();
        for v in &output {
            assert!((v - 2.0).abs() < 1e-5);
        }
    }

    #[test]
    fn test_unpack_int8() {
        let packed = vec![0xFEu8]; // -2 as i8
        let val = tc_unpack_int(&packed, 0, TcQuantBits::Int8);
        assert!((val - (-2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_unpack_int4_positive() {
        let packed = vec![0x53u8]; // nibble 0 = 3, nibble 1 = 5
        let v0 = tc_unpack_int(&packed, 0, TcQuantBits::Int4);
        let v1 = tc_unpack_int(&packed, 1, TcQuantBits::Int4);
        assert!((v0 - 3.0).abs() < 1e-6);
        assert!((v1 - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_unpack_int4_negative() {
        // nibble = 0xF → -1
        let packed = vec![0x0Fu8];
        let val = tc_unpack_int(&packed, 0, TcQuantBits::Int4);
        assert!((val - (-1.0)).abs() < 1e-6);
    }
}
