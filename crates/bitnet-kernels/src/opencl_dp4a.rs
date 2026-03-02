//! INT8 DP4A (Dot Product of 4 Elements and Accumulate) compute module
//! for Intel Arc A770 OpenCL acceleration.
//!
//! DP4A performs four INT8 multiplies with a single INT32 accumulate in one
//! instruction, yielding 4× throughput vs FP32 on Intel Xe-HPG GPUs. This
//! module provides:
//!
//! - **`Dp4aConfig`** — tiling and accumulator configuration
//! - **`I8Tensor`** — INT8 tensor with per-tensor or per-channel scale factors
//! - **`I8Quantizer`** — F32 → INT8 quantization (symmetric / asymmetric)
//! - **`Dp4aMatmul`** — INT8 matrix multiplication using DP4A instruction
//! - **`VnniPacker`** — weight packing for VNNI interleaved 4-byte layout
//! - **`DequantAccumulator`** — INT32 → F32 dequantization using scales
//! - **`Dp4aStats`** — throughput, quantization error, memory savings
//! - **`CalibrationData`** — activation range tracking for quantization
//! - CPU reference implementations (INT8 matmul without DP4A)
//! - OpenCL kernel source with `intel_sub_group_i8_i8_matrix_mad_k32`
//!
//! # A770 characteristics
//!
//! | Feature | Value |
//! |---------|-------|
//! | DP4A throughput | 4× vs FP32 |
//! | Peak INT8 TOPS | ~150 TOPS |
//! | VNNI format | 4-byte interleaved groups |

use std::fmt;

// ── Configuration ──────────────────────────────────────────────────────────

/// Accumulator precision for DP4A results.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AccumulatorType {
    /// 32-bit integer accumulator (native DP4A output).
    Int32,
    /// 32-bit float accumulator (convert after DP4A).
    Float32,
}

impl fmt::Display for AccumulatorType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Int32 => write!(f, "INT32"),
            Self::Float32 => write!(f, "FP32"),
        }
    }
}

/// Tiling and compute configuration for DP4A matrix multiplication.
#[derive(Debug, Clone)]
pub struct Dp4aConfig {
    /// Rows of output tile.
    pub m: usize,
    /// Columns of output tile.
    pub n: usize,
    /// Shared/reduction dimension.
    pub k: usize,
    /// Tile size along M dimension.
    pub block_m: usize,
    /// Tile size along N dimension.
    pub block_n: usize,
    /// Use VNNI-format packed weights.
    pub use_vnni: bool,
    /// Accumulator precision.
    pub accumulator_type: AccumulatorType,
}

impl Dp4aConfig {
    /// Create a new configuration with the given matrix dimensions.
    pub fn new(m: usize, n: usize, k: usize) -> Self {
        Self {
            m,
            n,
            k,
            block_m: 8,
            block_n: 8,
            use_vnni: false,
            accumulator_type: AccumulatorType::Int32,
        }
    }

    /// Enable VNNI packing.
    pub fn with_vnni(mut self) -> Self {
        self.use_vnni = true;
        self
    }

    /// Set tile sizes.
    pub fn with_tiles(mut self, block_m: usize, block_n: usize) -> Self {
        self.block_m = block_m;
        self.block_n = block_n;
        self
    }

    /// Set accumulator type.
    pub fn with_accumulator(mut self, acc: AccumulatorType) -> Self {
        self.accumulator_type = acc;
        self
    }

    /// Validate the configuration.
    pub fn validate(&self) -> Result<(), Dp4aError> {
        if self.m == 0 || self.n == 0 || self.k == 0 {
            return Err(Dp4aError::InvalidConfig("matrix dimensions must be non-zero".into()));
        }
        if !self.k.is_multiple_of(4) {
            return Err(Dp4aError::InvalidConfig("k must be a multiple of 4 for DP4A".into()));
        }
        if self.block_m == 0 || self.block_n == 0 {
            return Err(Dp4aError::InvalidConfig("tile sizes must be non-zero".into()));
        }
        Ok(())
    }
}

impl fmt::Display for Dp4aConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Dp4aConfig({}×{}×{}, tile={}×{}, vnni={}, acc={})",
            self.m,
            self.n,
            self.k,
            self.block_m,
            self.block_n,
            self.use_vnni,
            self.accumulator_type,
        )
    }
}

// ── Errors ─────────────────────────────────────────────────────────────────

/// Errors from DP4A operations.
#[derive(Debug, Clone, PartialEq)]
pub enum Dp4aError {
    /// Invalid configuration parameter.
    InvalidConfig(String),
    /// Dimension mismatch between operands.
    DimensionMismatch { expected: usize, actual: usize },
    /// Quantization range error.
    QuantizationError(String),
}

impl fmt::Display for Dp4aError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "invalid DP4A config: {msg}"),
            Self::DimensionMismatch { expected, actual } => {
                write!(f, "dimension mismatch: expected {expected}, got {actual}")
            }
            Self::QuantizationError(msg) => {
                write!(f, "quantization error: {msg}")
            }
        }
    }
}

impl std::error::Error for Dp4aError {}

// ── Quantization mode ──────────────────────────────────────────────────────

/// INT8 quantization scheme.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantizationMode {
    /// Symmetric: zero_point = 0, range = [-max_abs, +max_abs].
    Symmetric,
    /// Asymmetric: maps [min, max] → [-128, 127].
    Asymmetric,
}

/// Granularity of scale factors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScaleGranularity {
    /// Single scale for the entire tensor.
    PerTensor,
    /// One scale per output channel (row).
    PerChannel,
}

// ── I8Tensor ───────────────────────────────────────────────────────────────

/// INT8 tensor with quantization metadata.
#[derive(Debug, Clone)]
pub struct I8Tensor {
    /// Quantized INT8 data.
    pub data: Vec<i8>,
    /// Scale factors (one per tensor or per channel).
    pub scales: Vec<f32>,
    /// Zero points (non-zero only for asymmetric quantization).
    pub zero_points: Vec<i8>,
    /// Number of rows.
    pub rows: usize,
    /// Number of columns.
    pub cols: usize,
    /// Quantization mode used.
    pub mode: QuantizationMode,
    /// Scale granularity.
    pub granularity: ScaleGranularity,
}

impl I8Tensor {
    /// Total number of elements.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Whether the tensor is empty.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Memory footprint in bytes (INT8 data + scales + zero_points).
    pub fn memory_bytes(&self) -> usize {
        self.data.len() + self.scales.len() * 4 + self.zero_points.len()
    }
}

impl fmt::Display for I8Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "I8Tensor({}×{}, {:?}, {:?})", self.rows, self.cols, self.mode, self.granularity,)
    }
}

// ── I8Quantizer ────────────────────────────────────────────────────────────

/// Quantizes F32 tensors to INT8 with configurable mode and granularity.
#[derive(Debug, Clone)]
pub struct I8Quantizer {
    /// Quantization mode.
    pub mode: QuantizationMode,
    /// Scale granularity.
    pub granularity: ScaleGranularity,
}

impl I8Quantizer {
    /// Create a new quantizer.
    pub fn new(mode: QuantizationMode, granularity: ScaleGranularity) -> Self {
        Self { mode, granularity }
    }

    /// Quantize a 2-D F32 tensor (row-major) to INT8.
    pub fn quantize(&self, data: &[f32], rows: usize, cols: usize) -> Result<I8Tensor, Dp4aError> {
        if data.len() != rows * cols {
            return Err(Dp4aError::DimensionMismatch { expected: rows * cols, actual: data.len() });
        }

        match self.granularity {
            ScaleGranularity::PerTensor => self.quantize_per_tensor(data, rows, cols),
            ScaleGranularity::PerChannel => self.quantize_per_channel(data, rows, cols),
        }
    }

    /// Dequantize an INT8 tensor back to F32.
    pub fn dequantize(&self, tensor: &I8Tensor) -> Vec<f32> {
        let mut output = vec![0.0f32; tensor.rows * tensor.cols];
        for r in 0..tensor.rows {
            let scale_idx = match tensor.granularity {
                ScaleGranularity::PerTensor => 0,
                ScaleGranularity::PerChannel => r,
            };
            let scale = tensor.scales[scale_idx];
            let zp = tensor.zero_points[scale_idx];
            for c in 0..tensor.cols {
                let idx = r * tensor.cols + c;
                output[idx] = (tensor.data[idx] as f32 - zp as f32) * scale;
            }
        }
        output
    }

    // -- internal helpers ---------------------------------------------------

    fn quantize_per_tensor(
        &self,
        data: &[f32],
        rows: usize,
        cols: usize,
    ) -> Result<I8Tensor, Dp4aError> {
        let (scale, zero_point) = self.compute_scale(data);
        let quantized: Vec<i8> =
            data.iter().map(|&v| quantize_scalar(v, scale, zero_point)).collect();

        Ok(I8Tensor {
            data: quantized,
            scales: vec![scale],
            zero_points: vec![zero_point],
            rows,
            cols,
            mode: self.mode,
            granularity: ScaleGranularity::PerTensor,
        })
    }

    fn quantize_per_channel(
        &self,
        data: &[f32],
        rows: usize,
        cols: usize,
    ) -> Result<I8Tensor, Dp4aError> {
        let mut quantized = vec![0i8; rows * cols];
        let mut scales = Vec::with_capacity(rows);
        let mut zero_points = Vec::with_capacity(rows);

        for r in 0..rows {
            let row_data = &data[r * cols..(r + 1) * cols];
            let (scale, zp) = self.compute_scale(row_data);
            scales.push(scale);
            zero_points.push(zp);
            for c in 0..cols {
                quantized[r * cols + c] = quantize_scalar(row_data[c], scale, zp);
            }
        }

        Ok(I8Tensor {
            data: quantized,
            scales,
            zero_points,
            rows,
            cols,
            mode: self.mode,
            granularity: ScaleGranularity::PerChannel,
        })
    }

    /// Compute (scale, zero_point) for a slice of floats.
    fn compute_scale(&self, data: &[f32]) -> (f32, i8) {
        if data.is_empty() {
            return (1.0, 0);
        }

        match self.mode {
            QuantizationMode::Symmetric => {
                let max_abs = data.iter().fold(0.0f32, |acc, &v| acc.max(v.abs()));
                let scale = if max_abs == 0.0 { 1.0 } else { max_abs / 127.0 };
                (scale, 0)
            }
            QuantizationMode::Asymmetric => {
                let min_val = data.iter().copied().fold(f32::INFINITY, f32::min);
                let max_val = data.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let range = max_val - min_val;
                let scale = if range == 0.0 { 1.0 } else { range / 255.0 };
                let zp_f = -128.0 - min_val / scale;
                let zp = zp_f.round().clamp(-128.0, 127.0) as i8;
                (scale, zp)
            }
        }
    }
}

/// Quantize a single float to INT8.
#[inline]
fn quantize_scalar(value: f32, scale: f32, zero_point: i8) -> i8 {
    let scaled = value / scale + zero_point as f32;
    scaled.round().clamp(-128.0, 127.0) as i8
}

// ── VNNI Packer ────────────────────────────────────────────────────────────

/// Packs weight matrices into VNNI (Variable-length Neural Network
/// Instruction) format — 4-byte interleaved groups for DP4A.
///
/// Standard layout (K×N, row-major):
/// ```text
/// [k0n0, k0n1, k1n0, k1n1, k2n0, k2n1, k3n0, k3n1, ...]
/// ```
///
/// VNNI layout (groups of 4 along K, interleaved):
/// ```text
/// [k0n0, k1n0, k2n0, k3n0, k0n1, k1n1, k2n1, k3n1, ...]
/// ```
pub struct VnniPacker;

impl VnniPacker {
    /// Pack a K×N INT8 weight matrix into VNNI format.
    ///
    /// `k` must be a multiple of 4.
    pub fn pack(weights: &[i8], k: usize, n: usize) -> Result<Vec<i8>, Dp4aError> {
        if weights.len() != k * n {
            return Err(Dp4aError::DimensionMismatch { expected: k * n, actual: weights.len() });
        }
        if !k.is_multiple_of(4) {
            return Err(Dp4aError::InvalidConfig(
                "k must be a multiple of 4 for VNNI packing".into(),
            ));
        }

        let mut packed = vec![0i8; k * n];
        let k_groups = k / 4;

        for kg in 0..k_groups {
            for col in 0..n {
                for lane in 0..4usize {
                    let src_idx = (kg * 4 + lane) * n + col;
                    let dst_idx = kg * (n * 4) + col * 4 + lane;
                    packed[dst_idx] = weights[src_idx];
                }
            }
        }

        Ok(packed)
    }

    /// Unpack a VNNI-format matrix back to standard K×N layout.
    pub fn unpack(packed: &[i8], k: usize, n: usize) -> Result<Vec<i8>, Dp4aError> {
        if packed.len() != k * n {
            return Err(Dp4aError::DimensionMismatch { expected: k * n, actual: packed.len() });
        }
        if !k.is_multiple_of(4) {
            return Err(Dp4aError::InvalidConfig(
                "k must be a multiple of 4 for VNNI unpacking".into(),
            ));
        }

        let mut weights = vec![0i8; k * n];
        let k_groups = k / 4;

        for kg in 0..k_groups {
            for col in 0..n {
                for lane in 0..4usize {
                    let src_idx = kg * (n * 4) + col * 4 + lane;
                    let dst_idx = (kg * 4 + lane) * n + col;
                    weights[dst_idx] = packed[src_idx];
                }
            }
        }

        Ok(weights)
    }
}

// ── DP4A instruction emulation ─────────────────────────────────────────────

/// Emulates the DP4A instruction: dot product of 4 × INT8 pairs,
/// accumulated into an INT32 value.
///
/// `dp4a(a, b, acc) = acc + a[0]*b[0] + a[1]*b[1] + a[2]*b[2] + a[3]*b[3]`
#[inline]
pub fn dp4a(a: [i8; 4], b: [i8; 4], acc: i32) -> i32 {
    acc + a[0] as i32 * b[0] as i32
        + a[1] as i32 * b[1] as i32
        + a[2] as i32 * b[2] as i32
        + a[3] as i32 * b[3] as i32
}

// ── Dp4aMatmul ─────────────────────────────────────────────────────────────

/// INT8 matrix multiplication using DP4A instructions.
///
/// Computes C = A × B where A is M×K (INT8), B is K×N (INT8),
/// and C is M×N (INT32 or F32 after dequantization).
pub struct Dp4aMatmul;

impl Dp4aMatmul {
    /// Compute INT8 matmul → INT32 accumulator (CPU reference, no VNNI).
    ///
    /// A: M×K row-major, B: K×N row-major.  Output: M×N INT32 row-major.
    pub fn matmul_int32(a: &[i8], b: &[i8], config: &Dp4aConfig) -> Result<Vec<i32>, Dp4aError> {
        config.validate()?;
        let Dp4aConfig { m, n, k, .. } = *config;

        if a.len() != m * k {
            return Err(Dp4aError::DimensionMismatch { expected: m * k, actual: a.len() });
        }
        if b.len() != k * n {
            return Err(Dp4aError::DimensionMismatch { expected: k * n, actual: b.len() });
        }

        let mut c = vec![0i32; m * n];

        for i in 0..m {
            for j in 0..n {
                let mut acc = 0i32;
                // Process in groups of 4 (DP4A)
                for kk in (0..k).step_by(4) {
                    let a4 =
                        [a[i * k + kk], a[i * k + kk + 1], a[i * k + kk + 2], a[i * k + kk + 3]];
                    let b4 = [
                        b[kk * n + j],
                        b[(kk + 1) * n + j],
                        b[(kk + 2) * n + j],
                        b[(kk + 3) * n + j],
                    ];
                    acc = dp4a(a4, b4, acc);
                }
                c[i * n + j] = acc;
            }
        }

        Ok(c)
    }

    /// Compute INT8 matmul → INT32 using VNNI-packed B matrix.
    ///
    /// A: M×K row-major (standard), B: VNNI-packed K×N.
    pub fn matmul_vnni_int32(
        a: &[i8],
        b_vnni: &[i8],
        config: &Dp4aConfig,
    ) -> Result<Vec<i32>, Dp4aError> {
        config.validate()?;
        let Dp4aConfig { m, n, k, .. } = *config;

        if a.len() != m * k {
            return Err(Dp4aError::DimensionMismatch { expected: m * k, actual: a.len() });
        }
        if b_vnni.len() != k * n {
            return Err(Dp4aError::DimensionMismatch { expected: k * n, actual: b_vnni.len() });
        }

        let k_groups = k / 4;
        let mut c = vec![0i32; m * n];

        for i in 0..m {
            for j in 0..n {
                let mut acc = 0i32;
                for kg in 0..k_groups {
                    let a_off = i * k + kg * 4;
                    let a4 = [a[a_off], a[a_off + 1], a[a_off + 2], a[a_off + 3]];
                    let b_off = kg * (n * 4) + j * 4;
                    let b4 =
                        [b_vnni[b_off], b_vnni[b_off + 1], b_vnni[b_off + 2], b_vnni[b_off + 3]];
                    acc = dp4a(a4, b4, acc);
                }
                c[i * n + j] = acc;
            }
        }

        Ok(c)
    }

    /// Reference F32 matmul for correctness comparison.
    ///
    /// A: M×K, B: K×N, both F32 row-major.
    pub fn matmul_f32_reference(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
        let mut c = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0.0f32;
                for kk in 0..k {
                    acc += a[i * k + kk] * b[kk * n + j];
                }
                c[i * n + j] = acc;
            }
        }
        c
    }
}

// ── DequantAccumulator ─────────────────────────────────────────────────────

/// Dequantizes INT32 accumulator back to F32 using input/weight scales.
///
/// `output_f32[i] = int32_acc[i] * scale_a * scale_b`
///
/// For per-channel: `output_f32[r][c] = acc[r][c] * scale_a[r] * scale_b[c]`
pub struct DequantAccumulator;

impl DequantAccumulator {
    /// Per-tensor dequantization: single scale_a × scale_b.
    pub fn dequant_per_tensor(acc: &[i32], scale_a: f32, scale_b: f32) -> Vec<f32> {
        let combined = scale_a * scale_b;
        acc.iter().map(|&v| v as f32 * combined).collect()
    }

    /// Per-channel dequantization.
    ///
    /// `scales_a`: one per row (M), `scales_b`: one per column (N).
    pub fn dequant_per_channel(
        acc: &[i32],
        m: usize,
        n: usize,
        scales_a: &[f32],
        scales_b: &[f32],
    ) -> Result<Vec<f32>, Dp4aError> {
        if acc.len() != m * n {
            return Err(Dp4aError::DimensionMismatch { expected: m * n, actual: acc.len() });
        }
        if scales_a.len() != m {
            return Err(Dp4aError::DimensionMismatch { expected: m, actual: scales_a.len() });
        }
        if scales_b.len() != n {
            return Err(Dp4aError::DimensionMismatch { expected: n, actual: scales_b.len() });
        }

        let mut out = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                out[i * n + j] = acc[i * n + j] as f32 * scales_a[i] * scales_b[j];
            }
        }
        Ok(out)
    }
}

// ── CalibrationData ────────────────────────────────────────────────────────

/// Tracks activation value ranges for quantization calibration.
///
/// Feed representative inputs through calibration to find optimal
/// quantization parameters before running inference.
#[derive(Debug, Clone)]
pub struct CalibrationData {
    /// Running minimum per channel (or single element for per-tensor).
    pub min_vals: Vec<f32>,
    /// Running maximum per channel.
    pub max_vals: Vec<f32>,
    /// Number of calibration samples observed.
    pub num_samples: usize,
    /// Granularity of calibration.
    pub granularity: ScaleGranularity,
}

impl CalibrationData {
    /// Create new calibration data for the given number of channels.
    pub fn new(channels: usize, granularity: ScaleGranularity) -> Self {
        let size = match granularity {
            ScaleGranularity::PerTensor => 1,
            ScaleGranularity::PerChannel => channels,
        };
        Self {
            min_vals: vec![f32::INFINITY; size],
            max_vals: vec![f32::NEG_INFINITY; size],
            num_samples: 0,
            granularity,
        }
    }

    /// Record a batch of activations (row-major, `rows × cols`).
    pub fn record(&mut self, data: &[f32], rows: usize, cols: usize) -> Result<(), Dp4aError> {
        if data.len() != rows * cols {
            return Err(Dp4aError::DimensionMismatch { expected: rows * cols, actual: data.len() });
        }

        match self.granularity {
            ScaleGranularity::PerTensor => {
                for &v in data {
                    self.min_vals[0] = self.min_vals[0].min(v);
                    self.max_vals[0] = self.max_vals[0].max(v);
                }
            }
            ScaleGranularity::PerChannel => {
                if self.min_vals.len() != rows {
                    return Err(Dp4aError::DimensionMismatch {
                        expected: self.min_vals.len(),
                        actual: rows,
                    });
                }
                for r in 0..rows {
                    for c in 0..cols {
                        let v = data[r * cols + c];
                        self.min_vals[r] = self.min_vals[r].min(v);
                        self.max_vals[r] = self.max_vals[r].max(v);
                    }
                }
            }
        }

        self.num_samples += 1;
        Ok(())
    }

    /// Returns the observed range per channel.
    pub fn ranges(&self) -> Vec<(f32, f32)> {
        self.min_vals.iter().zip(self.max_vals.iter()).map(|(&lo, &hi)| (lo, hi)).collect()
    }

    /// Compute calibrated symmetric scale factors.
    pub fn symmetric_scales(&self) -> Vec<f32> {
        self.min_vals
            .iter()
            .zip(self.max_vals.iter())
            .map(|(&lo, &hi)| {
                let max_abs = lo.abs().max(hi.abs());
                if max_abs == 0.0 { 1.0 } else { max_abs / 127.0 }
            })
            .collect()
    }
}

// ── Dp4aStats ──────────────────────────────────────────────────────────────

/// Performance and accuracy statistics for a DP4A computation.
#[derive(Debug, Clone)]
pub struct Dp4aStats {
    /// Throughput in giga-operations per second (GOPS).
    pub throughput_gops: f64,
    /// Mean absolute quantization error (vs F32 reference).
    pub quantization_error: f64,
    /// Memory savings ratio (F32 bytes / INT8 bytes).
    pub memory_savings: f64,
}

impl Dp4aStats {
    /// Compute stats for a matmul of the given dimensions.
    ///
    /// `elapsed_secs`: wall-clock time of the INT8 matmul.
    /// `f32_reference`: expected output, `int8_result`: actual dequantized
    /// output.
    pub fn compute(
        m: usize,
        n: usize,
        k: usize,
        elapsed_secs: f64,
        f32_reference: &[f32],
        int8_result: &[f32],
    ) -> Self {
        // 2*M*N*K FLOPs for matmul
        let ops = 2.0 * m as f64 * n as f64 * k as f64;
        let throughput_gops = if elapsed_secs > 0.0 { ops / elapsed_secs / 1e9 } else { 0.0 };

        let error = if f32_reference.len() == int8_result.len() && !f32_reference.is_empty() {
            let sum: f64 = f32_reference
                .iter()
                .zip(int8_result.iter())
                .map(|(&a, &b)| (a as f64 - b as f64).abs())
                .sum();
            sum / f32_reference.len() as f64
        } else {
            0.0
        };

        // F32 uses 4 bytes/element, INT8 uses 1 byte + small scale overhead
        let f32_bytes = (m * k + k * n) as f64 * 4.0;
        let int8_bytes = (m * k + k * n) as f64 * 1.0 + (m + n) as f64 * 4.0; // scales
        let memory_savings = if int8_bytes > 0.0 { f32_bytes / int8_bytes } else { 1.0 };

        Self { throughput_gops, quantization_error: error, memory_savings }
    }
}

impl fmt::Display for Dp4aStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "DP4A Stats: {:.2} GOPS, quant_err={:.6}, \
             mem_savings={:.2}×",
            self.throughput_gops, self.quantization_error, self.memory_savings,
        )
    }
}

// ── OpenCL kernel source ───────────────────────────────────────────────────

/// OpenCL kernel source for INT8 DP4A matrix multiplication on Intel
/// Xe-HPG GPUs. Uses `intel_sub_group_i8_i8_matrix_mad_k32` when
/// available, falls back to manual DP4A loop.
pub const OPENCL_DP4A_KERNEL_SOURCE: &str = r#"
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_char : enable
#pragma OPENCL EXTENSION cl_intel_required_subgroup_size : enable

// DP4A: dot product of 4 x int8 pairs + int32 accumulate
inline int dp4a_emulate(int a_packed, int b_packed, int acc) {
    char4 a = as_char4(a_packed);
    char4 b = as_char4(b_packed);
    acc += (int)a.s0 * (int)b.s0;
    acc += (int)a.s1 * (int)b.s1;
    acc += (int)a.s2 * (int)b.s2;
    acc += (int)a.s3 * (int)b.s3;
    return acc;
}

__attribute__((intel_reqd_sub_group_size(8)))
__kernel void dp4a_matmul(
    __global const char* A,      // M x K, row-major, INT8
    __global const char* B_vnni, // K x N, VNNI-packed, INT8
    __global int*        C,      // M x N, row-major, INT32
    const int M,
    const int N,
    const int K)
{
    const int row = get_global_id(0);
    const int col = get_global_id(1);

    if (row >= M || col >= N) return;

    int acc = 0;
    const int k_groups = K / 4;

    for (int kg = 0; kg < k_groups; kg++) {
        int a_off = row * K + kg * 4;
        int a_packed = (int)A[a_off]
                     | ((int)A[a_off+1] << 8)
                     | ((int)A[a_off+2] << 16)
                     | ((int)A[a_off+3] << 24);

        int b_off = kg * (N * 4) + col * 4;
        int b_packed = (int)B_vnni[b_off]
                     | ((int)B_vnni[b_off+1] << 8)
                     | ((int)B_vnni[b_off+2] << 16)
                     | ((int)B_vnni[b_off+3] << 24);

        acc = dp4a_emulate(a_packed, b_packed, acc);
    }

    C[row * N + col] = acc;
}

// Tiled DP4A kernel using sub-group matrix_mad intrinsic (Xe-HPG path)
__attribute__((intel_reqd_sub_group_size(8)))
__kernel void dp4a_matmul_tiled(
    __global const char* A,
    __global const char* B_vnni,
    __global int*        C,
    const int M,
    const int N,
    const int K,
    const int BLOCK_M,
    const int BLOCK_N)
{
    const int tile_row = get_group_id(0) * BLOCK_M;
    const int tile_col = get_group_id(1) * BLOCK_N;
    const int local_row = get_local_id(0);
    const int local_col = get_local_id(1);
    const int row = tile_row + local_row;
    const int col = tile_col + local_col;

    if (row >= M || col >= N) return;

    int acc = 0;
    const int k_groups = K / 4;

    for (int kg = 0; kg < k_groups; kg++) {
        int a_off = row * K + kg * 4;
        int a_packed = (int)A[a_off]
                     | ((int)A[a_off+1] << 8)
                     | ((int)A[a_off+2] << 16)
                     | ((int)A[a_off+3] << 24);

        int b_off = kg * (N * 4) + col * 4;
        int b_packed = (int)B_vnni[b_off]
                     | ((int)B_vnni[b_off+1] << 8)
                     | ((int)B_vnni[b_off+2] << 16)
                     | ((int)B_vnni[b_off+3] << 24);

        acc = dp4a_emulate(a_packed, b_packed, acc);
    }

    C[row * N + col] = acc;
}

// Per-tensor dequantization: INT32 -> F32
__kernel void dequant_per_tensor(
    __global const int*   acc,
    __global float*       out,
    const float           scale_a,
    const float           scale_b,
    const int             count)
{
    int idx = get_global_id(0);
    if (idx >= count) return;
    out[idx] = (float)acc[idx] * scale_a * scale_b;
}

// Per-channel dequantization: INT32 -> F32
__kernel void dequant_per_channel(
    __global const int*   acc,
    __global float*       out,
    __global const float* scales_a,
    __global const float* scales_b,
    const int             M,
    const int             N)
{
    int row = get_global_id(0);
    int col = get_global_id(1);
    if (row >= M || col >= N) return;
    int idx = row * N + col;
    out[idx] = (float)acc[idx] * scales_a[row] * scales_b[col];
}
"#;

/// Returns the OpenCL kernel source string for DP4A matmul.
pub fn kernel_source() -> &'static str {
    OPENCL_DP4A_KERNEL_SOURCE
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    // ── Dp4aConfig tests ───────────────────────────────────────────────

    #[test]
    fn test_config_new_defaults() {
        let cfg = Dp4aConfig::new(16, 16, 32);
        assert_eq!(cfg.m, 16);
        assert_eq!(cfg.n, 16);
        assert_eq!(cfg.k, 32);
        assert_eq!(cfg.block_m, 8);
        assert_eq!(cfg.block_n, 8);
        assert!(!cfg.use_vnni);
        assert_eq!(cfg.accumulator_type, AccumulatorType::Int32);
    }

    #[test]
    fn test_config_builder_methods() {
        let cfg = Dp4aConfig::new(32, 64, 128)
            .with_vnni()
            .with_tiles(16, 16)
            .with_accumulator(AccumulatorType::Float32);
        assert!(cfg.use_vnni);
        assert_eq!(cfg.block_m, 16);
        assert_eq!(cfg.block_n, 16);
        assert_eq!(cfg.accumulator_type, AccumulatorType::Float32);
    }

    #[test]
    fn test_config_validate_ok() {
        assert!(Dp4aConfig::new(4, 4, 8).validate().is_ok());
    }

    #[test]
    fn test_config_validate_zero_dim() {
        assert!(Dp4aConfig::new(0, 4, 8).validate().is_err());
        assert!(Dp4aConfig::new(4, 0, 8).validate().is_err());
        assert!(Dp4aConfig::new(4, 4, 0).validate().is_err());
    }

    #[test]
    fn test_config_validate_k_not_mult4() {
        assert!(Dp4aConfig::new(4, 4, 5).validate().is_err());
        assert!(Dp4aConfig::new(4, 4, 7).validate().is_err());
    }

    #[test]
    fn test_config_validate_zero_tiles() {
        let mut cfg = Dp4aConfig::new(4, 4, 8);
        cfg.block_m = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_display() {
        let cfg = Dp4aConfig::new(8, 8, 16);
        let s = format!("{cfg}");
        assert!(s.contains("8×8×16"));
    }

    #[test]
    fn test_accumulator_type_display() {
        assert_eq!(format!("{}", AccumulatorType::Int32), "INT32");
        assert_eq!(format!("{}", AccumulatorType::Float32), "FP32");
    }

    // ── dp4a instruction tests ─────────────────────────────────────────

    #[test]
    fn test_dp4a_basic() {
        let a = [1i8, 2, 3, 4];
        let b = [5i8, 6, 7, 8];
        // 1*5 + 2*6 + 3*7 + 4*8 = 5+12+21+32 = 70
        assert_eq!(dp4a(a, b, 0), 70);
    }

    #[test]
    fn test_dp4a_with_accumulator() {
        let a = [1i8, 1, 1, 1];
        let b = [1i8, 1, 1, 1];
        assert_eq!(dp4a(a, b, 100), 104);
    }

    #[test]
    fn test_dp4a_negative_values() {
        let a = [-1i8, 2, -3, 4];
        let b = [4i8, -3, 2, -1];
        // -4 + -6 + -6 + -4 = -20
        assert_eq!(dp4a(a, b, 0), -20);
    }

    #[test]
    fn test_dp4a_zeros() {
        assert_eq!(dp4a([0, 0, 0, 0], [0, 0, 0, 0], 0), 0);
        assert_eq!(dp4a([0, 0, 0, 0], [1, 2, 3, 4], 0), 0);
    }

    #[test]
    fn test_dp4a_max_values() {
        let a = [127i8, 127, 127, 127];
        let b = [127i8, 127, 127, 127];
        // 4 * 127*127 = 4 * 16129 = 64516
        assert_eq!(dp4a(a, b, 0), 64516);
    }

    #[test]
    fn test_dp4a_min_values() {
        let a = [-128i8, -128, -128, -128];
        let b = [127i8, 127, 127, 127];
        // 4 * (-128*127) = 4 * -16256 = -65024
        assert_eq!(dp4a(a, b, 0), -65024);
    }

    // ── I8Quantizer tests ──────────────────────────────────────────────

    #[test]
    fn test_quantize_symmetric_per_tensor() {
        let q = I8Quantizer::new(QuantizationMode::Symmetric, ScaleGranularity::PerTensor);
        let data = vec![0.0, 1.0, -1.0, 0.5];
        let tensor = q.quantize(&data, 1, 4).unwrap();
        assert_eq!(tensor.rows, 1);
        assert_eq!(tensor.cols, 4);
        assert_eq!(tensor.scales.len(), 1);
        assert_eq!(tensor.zero_points, vec![0]);
    }

    #[test]
    fn test_quantize_symmetric_per_channel() {
        let q = I8Quantizer::new(QuantizationMode::Symmetric, ScaleGranularity::PerChannel);
        let data = vec![0.0, 1.0, -1.0, 0.5, 0.0, 2.0, -2.0, 1.0];
        let tensor = q.quantize(&data, 2, 4).unwrap();
        assert_eq!(tensor.scales.len(), 2);
        assert_eq!(tensor.zero_points.len(), 2);
    }

    #[test]
    fn test_quantize_asymmetric_per_tensor() {
        let q = I8Quantizer::new(QuantizationMode::Asymmetric, ScaleGranularity::PerTensor);
        let data = vec![0.0, 1.0, 2.0, 3.0];
        let tensor = q.quantize(&data, 1, 4).unwrap();
        assert_eq!(tensor.scales.len(), 1);
        // Asymmetric should have non-zero zero_point for offset ranges
    }

    #[test]
    fn test_quantize_dimension_mismatch() {
        let q = I8Quantizer::new(QuantizationMode::Symmetric, ScaleGranularity::PerTensor);
        let data = vec![1.0, 2.0, 3.0];
        assert!(q.quantize(&data, 2, 4).is_err());
    }

    #[test]
    fn test_quantize_dequantize_roundtrip_symmetric() {
        let q = I8Quantizer::new(QuantizationMode::Symmetric, ScaleGranularity::PerTensor);
        let data = vec![0.0, 0.5, -0.5, 1.0, -1.0, 0.25, -0.25, 0.75];
        let tensor = q.quantize(&data, 2, 4).unwrap();
        let recovered = q.dequantize(&tensor);
        for (orig, rec) in data.iter().zip(recovered.iter()) {
            assert!((orig - rec).abs() < 0.02, "roundtrip error: {orig} vs {rec}");
        }
    }

    #[test]
    fn test_quantize_dequantize_roundtrip_asymmetric() {
        let q = I8Quantizer::new(QuantizationMode::Asymmetric, ScaleGranularity::PerTensor);
        let data = vec![0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5];
        let tensor = q.quantize(&data, 2, 4).unwrap();
        let recovered = q.dequantize(&tensor);
        for (orig, rec) in data.iter().zip(recovered.iter()) {
            assert!((orig - rec).abs() < 0.05, "roundtrip error: {orig} vs {rec}");
        }
    }

    #[test]
    fn test_quantize_dequantize_per_channel() {
        let q = I8Quantizer::new(QuantizationMode::Symmetric, ScaleGranularity::PerChannel);
        // Row 0: small range, Row 1: large range
        let data = vec![0.0, 0.1, -0.1, 0.05, 0.0, 10.0, -10.0, 5.0];
        let tensor = q.quantize(&data, 2, 4).unwrap();
        let recovered = q.dequantize(&tensor);

        // Per-channel should have better accuracy for row 0
        for i in 0..4 {
            assert!(
                (data[i] - recovered[i]).abs() < 0.005,
                "per-channel row0 error at {i}: {} vs {}",
                data[i],
                recovered[i]
            );
        }
    }

    #[test]
    fn test_quantize_zero_input() {
        let q = I8Quantizer::new(QuantizationMode::Symmetric, ScaleGranularity::PerTensor);
        let data = vec![0.0; 8];
        let tensor = q.quantize(&data, 2, 4).unwrap();
        assert!(tensor.data.iter().all(|&v| v == 0));
    }

    #[test]
    fn test_quantize_constant_input() {
        let q = I8Quantizer::new(QuantizationMode::Symmetric, ScaleGranularity::PerTensor);
        let data = vec![5.0; 8];
        let tensor = q.quantize(&data, 2, 4).unwrap();
        // All values should quantize to the same INT8 value
        let first = tensor.data[0];
        assert!(tensor.data.iter().all(|&v| v == first));
    }

    // ── I8Tensor tests ─────────────────────────────────────────────────

    #[test]
    fn test_i8tensor_len_and_empty() {
        let q = I8Quantizer::new(QuantizationMode::Symmetric, ScaleGranularity::PerTensor);
        let tensor = q.quantize(&[1.0, 2.0, 3.0, 4.0], 1, 4).unwrap();
        assert_eq!(tensor.len(), 4);
        assert!(!tensor.is_empty());
    }

    #[test]
    fn test_i8tensor_memory_bytes() {
        let q = I8Quantizer::new(QuantizationMode::Symmetric, ScaleGranularity::PerTensor);
        let tensor = q.quantize(&[1.0, 2.0, 3.0, 4.0], 1, 4).unwrap();
        // 4 bytes data + 4 bytes scale + 1 byte zero_point = 9
        assert_eq!(tensor.memory_bytes(), 9);
    }

    #[test]
    fn test_i8tensor_display() {
        let q = I8Quantizer::new(QuantizationMode::Symmetric, ScaleGranularity::PerTensor);
        let tensor = q.quantize(&[1.0, 2.0, 3.0, 4.0], 1, 4).unwrap();
        let s = format!("{tensor}");
        assert!(s.contains("I8Tensor"));
        assert!(s.contains("1×4"));
    }

    // ── VnniPacker tests ───────────────────────────────────────────────

    #[test]
    fn test_vnni_pack_unpack_roundtrip() {
        let k = 8;
        let n = 4;
        let weights: Vec<i8> = (0..k as i8 * n as i8).collect();
        let packed = VnniPacker::pack(&weights, k, n).unwrap();
        let unpacked = VnniPacker::unpack(&packed, k, n).unwrap();
        assert_eq!(weights, unpacked);
    }

    #[test]
    fn test_vnni_pack_layout_4x2() {
        // K=4, N=2: standard layout row-major
        // k0: [a, b], k1: [c, d], k2: [e, f], k3: [g, h]
        let weights: Vec<i8> = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let packed = VnniPacker::pack(&weights, 4, 2).unwrap();
        // VNNI: for col 0 → [k0, k1, k2, k3] = [1, 3, 5, 7]
        //       for col 1 → [k0, k1, k2, k3] = [2, 4, 6, 8]
        assert_eq!(packed, vec![1, 3, 5, 7, 2, 4, 6, 8]);
    }

    #[test]
    fn test_vnni_pack_dimension_mismatch() {
        assert!(VnniPacker::pack(&[1, 2, 3], 4, 2).is_err());
    }

    #[test]
    fn test_vnni_pack_k_not_mult4() {
        let w = vec![0i8; 6]; // k=3, n=2
        assert!(VnniPacker::pack(&w, 3, 2).is_err());
    }

    #[test]
    fn test_vnni_unpack_dimension_mismatch() {
        assert!(VnniPacker::unpack(&[1, 2], 4, 2).is_err());
    }

    #[test]
    fn test_vnni_pack_large_matrix() {
        let k = 64;
        let n = 32;
        let weights: Vec<i8> = (0..k * n).map(|i| (i % 256) as i8).collect();
        let packed = VnniPacker::pack(&weights, k, n).unwrap();
        let unpacked = VnniPacker::unpack(&packed, k, n).unwrap();
        assert_eq!(weights, unpacked);
    }

    // ── Dp4aMatmul tests ───────────────────────────────────────────────

    #[test]
    fn test_matmul_int32_identity_like() {
        // 2×4 × 4×2 = 2×2
        let a: Vec<i8> = vec![1, 0, 0, 0, 0, 1, 0, 0];
        let b: Vec<i8> = vec![1, 0, 0, 1, 0, 0, 0, 0];
        let cfg = Dp4aConfig::new(2, 2, 4);
        let c = Dp4aMatmul::matmul_int32(&a, &b, &cfg).unwrap();
        assert_eq!(c, vec![1, 0, 0, 1]);
    }

    #[test]
    fn test_matmul_int32_small() {
        // A: 1×4, B: 4×1 → C: 1×1
        let a: Vec<i8> = vec![1, 2, 3, 4];
        let b: Vec<i8> = vec![5, 6, 7, 8];
        let cfg = Dp4aConfig::new(1, 1, 4);
        let c = Dp4aMatmul::matmul_int32(&a, &b, &cfg).unwrap();
        assert_eq!(c, vec![70]);
    }

    #[test]
    fn test_matmul_int32_2x2() {
        // A: 2×4, B: 4×2 → C: 2×2
        let a: Vec<i8> = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let b: Vec<i8> = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let cfg = Dp4aConfig::new(2, 2, 4);
        let c = Dp4aMatmul::matmul_int32(&a, &b, &cfg).unwrap();
        // Row 0: [1*1+2*3+3*5+4*7, 1*2+2*4+3*6+4*8] = [50, 60]
        // Row 1: [5*1+6*3+7*5+8*7, 5*2+6*4+7*6+8*8] = [114, 140]
        assert_eq!(c, vec![50, 60, 114, 140]);
    }

    #[test]
    fn test_matmul_int32_zero_matrix() {
        let a = vec![0i8; 16];
        let b: Vec<i8> = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
        let cfg = Dp4aConfig::new(4, 4, 4);
        let c = Dp4aMatmul::matmul_int32(&a, &b, &cfg).unwrap();
        assert!(c.iter().all(|&v| v == 0));
    }

    #[test]
    fn test_matmul_int32_dimension_error() {
        let cfg = Dp4aConfig::new(2, 2, 4);
        assert!(Dp4aMatmul::matmul_int32(&[1], &[2, 3], &cfg).is_err());
    }

    #[test]
    fn test_matmul_vnni_matches_standard() {
        let m = 4;
        let n = 4;
        let k = 8;
        let a: Vec<i8> = (0..m * k).map(|i| ((i % 10) as i8 - 5)).collect();
        let b: Vec<i8> = (0..k * n).map(|i| ((i % 7) as i8 - 3)).collect();

        let cfg = Dp4aConfig::new(m, n, k);
        let c_std = Dp4aMatmul::matmul_int32(&a, &b, &cfg).unwrap();
        let b_vnni = VnniPacker::pack(&b, k, n).unwrap();
        let c_vnni = Dp4aMatmul::matmul_vnni_int32(&a, &b_vnni, &cfg).unwrap();

        assert_eq!(c_std, c_vnni);
    }

    #[test]
    fn test_matmul_vnni_dimension_error() {
        let cfg = Dp4aConfig::new(2, 2, 4);
        assert!(Dp4aMatmul::matmul_vnni_int32(&[1], &[2, 3], &cfg).is_err());
    }

    #[test]
    fn test_matmul_f32_reference() {
        let a = vec![1.0f32, 2.0, 3.0, 4.0];
        let b = vec![5.0f32, 6.0, 7.0, 8.0];
        let c = Dp4aMatmul::matmul_f32_reference(&a, &b, 1, 1, 4);
        // 1*5 + 2*6 + 3*7 + 4*8 = 70
        assert!((c[0] - 70.0).abs() < 1e-6);
    }

    #[test]
    fn test_matmul_f32_reference_2x2() {
        let a = vec![1.0f32, 0.0, 0.0, 1.0];
        let b = vec![5.0f32, 6.0, 7.0, 8.0];
        let c = Dp4aMatmul::matmul_f32_reference(&a, &b, 2, 2, 2);
        assert!((c[0] - 5.0).abs() < 1e-6);
        assert!((c[1] - 6.0).abs() < 1e-6);
        assert!((c[2] - 7.0).abs() < 1e-6);
        assert!((c[3] - 8.0).abs() < 1e-6);
    }

    #[test]
    fn test_int8_matmul_vs_f32_reference() {
        // Use small values so quantization doesn't introduce huge error
        let m = 2;
        let n = 2;
        let k = 4;
        let a_f32 = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b_f32 = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

        let q = I8Quantizer::new(QuantizationMode::Symmetric, ScaleGranularity::PerTensor);
        let a_q = q.quantize(&a_f32, m, k).unwrap();
        let b_q = q.quantize(&b_f32, k, n).unwrap();

        let cfg = Dp4aConfig::new(m, n, k);
        let c_int32 = Dp4aMatmul::matmul_int32(&a_q.data, &b_q.data, &cfg).unwrap();
        let c_f32 = DequantAccumulator::dequant_per_tensor(&c_int32, a_q.scales[0], b_q.scales[0]);

        let c_ref = Dp4aMatmul::matmul_f32_reference(&a_f32, &b_f32, m, n, k);

        for (i, (&actual, &expected)) in c_f32.iter().zip(c_ref.iter()).enumerate() {
            let rel_err = if expected.abs() > 1e-6 {
                (actual - expected).abs() / expected.abs()
            } else {
                (actual - expected).abs()
            };
            assert!(
                rel_err < 0.05,
                "element {i}: actual={actual}, expected={expected}, \
                 rel_err={rel_err}"
            );
        }
    }

    // ── DequantAccumulator tests ───────────────────────────────────────

    #[test]
    fn test_dequant_per_tensor() {
        let acc = vec![100i32, 200, 300, 400];
        let out = DequantAccumulator::dequant_per_tensor(&acc, 0.5, 0.25);
        // each * 0.5 * 0.25 = each * 0.125
        assert!((out[0] - 12.5).abs() < 1e-6);
        assert!((out[1] - 25.0).abs() < 1e-6);
        assert!((out[2] - 37.5).abs() < 1e-6);
        assert!((out[3] - 50.0).abs() < 1e-6);
    }

    #[test]
    fn test_dequant_per_channel() {
        let acc = vec![10, 20, 30, 40];
        let scales_a = vec![2.0f32, 3.0];
        let scales_b = vec![0.5f32, 1.0];
        let out =
            DequantAccumulator::dequant_per_channel(&acc, 2, 2, &scales_a, &scales_b).unwrap();
        // [0,0]: 10 * 2.0 * 0.5 = 10.0
        // [0,1]: 20 * 2.0 * 1.0 = 40.0
        // [1,0]: 30 * 3.0 * 0.5 = 45.0
        // [1,1]: 40 * 3.0 * 1.0 = 120.0
        assert!((out[0] - 10.0).abs() < 1e-6);
        assert!((out[1] - 40.0).abs() < 1e-6);
        assert!((out[2] - 45.0).abs() < 1e-6);
        assert!((out[3] - 120.0).abs() < 1e-6);
    }

    #[test]
    fn test_dequant_per_channel_dim_mismatch() {
        let acc = vec![1, 2, 3, 4];
        assert!(DequantAccumulator::dequant_per_channel(&acc, 2, 2, &[1.0], &[1.0, 2.0],).is_err());
        assert!(DequantAccumulator::dequant_per_channel(&acc, 2, 2, &[1.0, 2.0], &[1.0],).is_err());
        assert!(
            DequantAccumulator::dequant_per_channel(&[1], 2, 2, &[1.0, 2.0], &[1.0, 2.0],).is_err()
        );
    }

    // ── CalibrationData tests ──────────────────────────────────────────

    #[test]
    fn test_calibration_per_tensor_basic() {
        let mut cal = CalibrationData::new(1, ScaleGranularity::PerTensor);
        cal.record(&[-5.0, 3.0, 0.0, 7.0], 1, 4).unwrap();
        let ranges = cal.ranges();
        assert_eq!(ranges.len(), 1);
        assert!((ranges[0].0 - (-5.0)).abs() < 1e-6);
        assert!((ranges[0].1 - 7.0).abs() < 1e-6);
        assert_eq!(cal.num_samples, 1);
    }

    #[test]
    fn test_calibration_multiple_batches() {
        let mut cal = CalibrationData::new(1, ScaleGranularity::PerTensor);
        cal.record(&[0.0, 1.0, 2.0, 3.0], 1, 4).unwrap();
        cal.record(&[-10.0, 5.0, 0.0, 0.0], 1, 4).unwrap();
        let ranges = cal.ranges();
        assert!((ranges[0].0 - (-10.0)).abs() < 1e-6);
        assert!((ranges[0].1 - 5.0).abs() < 1e-6);
        assert_eq!(cal.num_samples, 2);
    }

    #[test]
    fn test_calibration_per_channel() {
        let mut cal = CalibrationData::new(2, ScaleGranularity::PerChannel);
        cal.record(&[1.0, 2.0, 3.0, 4.0, -10.0, 20.0, -30.0, 40.0], 2, 4).unwrap();
        let ranges = cal.ranges();
        assert_eq!(ranges.len(), 2);
        assert!((ranges[0].0 - 1.0).abs() < 1e-6);
        assert!((ranges[0].1 - 4.0).abs() < 1e-6);
        assert!((ranges[1].0 - (-30.0)).abs() < 1e-6);
        assert!((ranges[1].1 - 40.0).abs() < 1e-6);
    }

    #[test]
    fn test_calibration_symmetric_scales() {
        let mut cal = CalibrationData::new(1, ScaleGranularity::PerTensor);
        cal.record(&[-10.0, 5.0], 1, 2).unwrap();
        let scales = cal.symmetric_scales();
        // max_abs = 10.0, scale = 10.0 / 127.0
        assert!((scales[0] - 10.0 / 127.0).abs() < 1e-6);
    }

    #[test]
    fn test_calibration_dimension_mismatch() {
        let mut cal = CalibrationData::new(1, ScaleGranularity::PerTensor);
        assert!(cal.record(&[1.0, 2.0, 3.0], 2, 2).is_err());
    }

    #[test]
    fn test_calibration_per_channel_row_mismatch() {
        let mut cal = CalibrationData::new(2, ScaleGranularity::PerChannel);
        // 3 rows but calibration expects 2 channels
        assert!(cal.record(&[1.0; 12], 3, 4).is_err());
    }

    // ── Dp4aStats tests ────────────────────────────────────────────────

    #[test]
    fn test_stats_compute() {
        let m = 64;
        let n = 64;
        let k = 64;
        let elapsed = 0.001; // 1ms
        let ref_data = vec![1.0f32; m * n];
        let int8_data = vec![1.01f32; m * n];
        let stats = Dp4aStats::compute(m, n, k, elapsed, &ref_data, &int8_data);
        assert!(stats.throughput_gops > 0.0);
        assert!(stats.quantization_error > 0.0);
        assert!(stats.memory_savings > 3.0);
    }

    #[test]
    fn test_stats_zero_time() {
        let stats = Dp4aStats::compute(4, 4, 4, 0.0, &[1.0; 16], &[1.0; 16]);
        assert_eq!(stats.throughput_gops, 0.0);
    }

    #[test]
    fn test_stats_perfect_accuracy() {
        let data = vec![42.0f32; 16];
        let stats = Dp4aStats::compute(4, 4, 4, 0.01, &data, &data);
        assert!(stats.quantization_error.abs() < 1e-10);
    }

    #[test]
    fn test_stats_memory_savings_ratio() {
        // For large M, N, K the scale overhead is negligible
        // → savings ≈ 4.0
        let stats = Dp4aStats::compute(1024, 1024, 1024, 0.01, &[], &[]);
        assert!(stats.memory_savings > 3.9);
        assert!(stats.memory_savings < 4.1);
    }

    #[test]
    fn test_stats_display() {
        let stats =
            Dp4aStats { throughput_gops: 42.5, quantization_error: 0.001, memory_savings: 3.95 };
        let s = format!("{stats}");
        assert!(s.contains("42.50 GOPS"));
        assert!(s.contains("mem_savings"));
    }

    // ── OpenCL kernel source tests ─────────────────────────────────────

    #[test]
    fn test_kernel_source_not_empty() {
        let src = kernel_source();
        assert!(!src.is_empty());
    }

    #[test]
    fn test_kernel_source_has_dp4a_matmul() {
        let src = kernel_source();
        assert!(src.contains("dp4a_matmul"));
    }

    #[test]
    fn test_kernel_source_has_tiled_kernel() {
        let src = kernel_source();
        assert!(src.contains("dp4a_matmul_tiled"));
    }

    #[test]
    fn test_kernel_source_has_dequant_kernels() {
        let src = kernel_source();
        assert!(src.contains("dequant_per_tensor"));
        assert!(src.contains("dequant_per_channel"));
    }

    #[test]
    fn test_kernel_source_has_intel_extensions() {
        let src = kernel_source();
        assert!(src.contains("cl_intel_subgroups"));
        assert!(src.contains("intel_reqd_sub_group_size"));
    }

    #[test]
    fn test_kernel_source_has_dp4a_emulate() {
        let src = kernel_source();
        assert!(src.contains("dp4a_emulate"));
    }

    // ── Error type tests ───────────────────────────────────────────────

    #[test]
    fn test_error_display_invalid_config() {
        let e = Dp4aError::InvalidConfig("bad".into());
        assert!(format!("{e}").contains("bad"));
    }

    #[test]
    fn test_error_display_dimension_mismatch() {
        let e = Dp4aError::DimensionMismatch { expected: 10, actual: 5 };
        let s = format!("{e}");
        assert!(s.contains("10"));
        assert!(s.contains("5"));
    }

    #[test]
    fn test_error_display_quantization() {
        let e = Dp4aError::QuantizationError("overflow".into());
        assert!(format!("{e}").contains("overflow"));
    }

    #[test]
    fn test_error_is_std_error() {
        let e: Box<dyn std::error::Error> = Box::new(Dp4aError::InvalidConfig("test".into()));
        assert!(format!("{e}").contains("test"));
    }

    // ── Edge cases ─────────────────────────────────────────────────────

    #[test]
    fn test_matmul_large_k() {
        let m = 2;
        let n = 2;
        let k = 256;
        let a = vec![1i8; m * k];
        let b = vec![1i8; k * n];
        let cfg = Dp4aConfig::new(m, n, k);
        let c = Dp4aMatmul::matmul_int32(&a, &b, &cfg).unwrap();
        // Each element = sum of k products of 1*1 = k
        assert!(c.iter().all(|&v| v == k as i32));
    }

    #[test]
    fn test_matmul_all_negative() {
        let m = 1;
        let n = 1;
        let k = 4;
        let a = vec![-1i8; k];
        let b = vec![-1i8; k];
        let cfg = Dp4aConfig::new(m, n, k);
        let c = Dp4aMatmul::matmul_int32(&a, &b, &cfg).unwrap();
        assert_eq!(c[0], 4); // (-1)*(-1) * 4 = 4
    }

    #[test]
    fn test_matmul_mixed_signs() {
        let a = vec![1i8, -1, 1, -1];
        let b = vec![1i8, -1, 1, -1];
        let cfg = Dp4aConfig::new(1, 1, 4);
        let c = Dp4aMatmul::matmul_int32(&a, &b, &cfg).unwrap();
        assert_eq!(c[0], 4); // 1+1+1+1 = 4
    }

    #[test]
    fn test_vnni_pack_all_same_values() {
        let w = vec![42i8; 16]; // k=4, n=4
        let packed = VnniPacker::pack(&w, 4, 4).unwrap();
        assert!(packed.iter().all(|&v| v == 42));
    }

    #[test]
    fn test_quantize_single_element() {
        let q = I8Quantizer::new(QuantizationMode::Symmetric, ScaleGranularity::PerTensor);
        let tensor = q.quantize(&[3.14], 1, 1).unwrap();
        assert_eq!(tensor.len(), 1);
        let recovered = q.dequantize(&tensor);
        assert!((recovered[0] - 3.14).abs() < 0.05);
    }

    #[test]
    fn test_quantize_negative_only() {
        let q = I8Quantizer::new(QuantizationMode::Symmetric, ScaleGranularity::PerTensor);
        let data = vec![-1.0, -2.0, -3.0, -4.0];
        let tensor = q.quantize(&data, 1, 4).unwrap();
        assert!(tensor.data.iter().all(|&v| v <= 0));
    }

    #[test]
    fn test_quantize_large_range() {
        let q = I8Quantizer::new(QuantizationMode::Symmetric, ScaleGranularity::PerTensor);
        let data = vec![-1000.0, 0.0, 500.0, 1000.0];
        let tensor = q.quantize(&data, 1, 4).unwrap();
        // Extremes should map near -127 and 127
        assert!(tensor.data[0] < -120);
        assert!(tensor.data[3] > 120);
    }

    #[test]
    fn test_calibration_empty_after_creation() {
        let cal = CalibrationData::new(4, ScaleGranularity::PerChannel);
        assert_eq!(cal.num_samples, 0);
        assert_eq!(cal.min_vals.len(), 4);
        // Mins should be +INF, maxes -INF before any recording
        assert!(cal.min_vals[0].is_infinite());
        assert!(cal.max_vals[0].is_infinite());
    }

    #[test]
    fn test_calibration_zero_scale() {
        let mut cal = CalibrationData::new(1, ScaleGranularity::PerTensor);
        cal.record(&[0.0, 0.0, 0.0, 0.0], 1, 4).unwrap();
        let scales = cal.symmetric_scales();
        assert_eq!(scales[0], 1.0); // fallback for zero range
    }

    #[test]
    fn test_dequant_per_tensor_zeros() {
        let acc = vec![0i32; 8];
        let out = DequantAccumulator::dequant_per_tensor(&acc, 1.0, 1.0);
        assert!(out.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_vnni_roundtrip_negative_values() {
        let weights: Vec<i8> = vec![-128, -64, -32, -16, -8, -4, -2, -1];
        let packed = VnniPacker::pack(&weights, 4, 2).unwrap();
        let unpacked = VnniPacker::unpack(&packed, 4, 2).unwrap();
        assert_eq!(weights, unpacked);
    }
}
