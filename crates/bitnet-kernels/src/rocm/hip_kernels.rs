//! Expanded HIP kernel scaffolding for AMD ROCm GPUs.
//!
//! This module provides six kernel abstractions that mirror the CUDA / OpenCL
//! counterparts elsewhere in the crate:
//!
//! | Struct | Purpose |
//! |--------|---------|
//! | [`HipMatmulKernel`]    | General-purpose matrix multiplication |
//! | [`HipSoftmaxKernel`]   | Softmax with warp-level (wavefront) reduction |
//! | [`HipLayerNormKernel`] | Classic LayerNorm (mean + variance) |
//! | [`HipQuantKernel`]     | 2-bit quantization / dequantization |
//! | [`HipAttentionKernel`] | Fused scaled-dot-product attention |
//! | [`HipDeviceQuery`]     | Device capability detection and enumeration |
//!
//! **Status — stubs with mock fallback.**  Every kernel can operate in *mock
//! mode* (CPU reference computation) so that the full test surface executes on
//! machines without an AMD GPU.  When the real HIP runtime is wired in, mock
//! mode will be gated behind `cfg(test)` only.
//!
//! All public items are behind `#[cfg(feature = "rocm")]` at the module level
//! (see `mod.rs`).

use bitnet_common::{BitNetError, KernelError, Result};

// ── Helpers ──────────────────────────────────────────────────────────

/// AMD wavefront (warp) size — 64 threads on GCN / CDNA architectures.
pub const HIP_WAVEFRONT_SIZE: u32 = 64;

/// Maximum LDS (shared memory) per work-group on MI200-series (bytes).
pub const HIP_MAX_LDS_BYTES: usize = 65_536;

fn stub_err(op: &str) -> BitNetError {
    BitNetError::Kernel(KernelError::ExecutionFailed {
        reason: format!("ROCm/HIP '{op}' not yet wired to the AMD HIP runtime"),
    })
}

// ── 1. HipMatmulKernel ──────────────────────────────────────────────

/// Launch configuration for HIP matrix multiplication.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HipMatmulConfig {
    /// Work-group (block) size — must be a multiple of [`HIP_WAVEFRONT_SIZE`].
    pub workgroup_size: u32,
    /// Tile dimension for shared-memory blocking (M-tile = N-tile = tile_dim).
    pub tile_dim: u32,
    /// Whether to use mock (CPU) computation instead of the HIP runtime.
    pub mock: bool,
}

impl Default for HipMatmulConfig {
    fn default() -> Self {
        Self { workgroup_size: 256, tile_dim: 16, mock: false }
    }
}

impl HipMatmulConfig {
    /// Config that uses CPU mock computation (for testing without a GPU).
    pub fn mock() -> Self {
        Self { mock: true, ..Self::default() }
    }
}

/// HIP-optimized matrix multiplication kernel.
///
/// Computes `C = A × B` where A is `[M, K]`, B is `[K, N]`, C is `[M, N]`.
#[derive(Debug, Clone)]
pub struct HipMatmulKernel {
    config: HipMatmulConfig,
}

impl HipMatmulKernel {
    pub fn new(config: HipMatmulConfig) -> Self {
        Self { config }
    }

    /// Execute the matmul.  Returns [`Err`] in stub mode, or runs a CPU
    /// reference implementation in mock mode.
    pub fn execute(
        &self,
        a: &[f32],
        b: &[f32],
        c: &mut [f32],
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<()> {
        if a.len() < m * k {
            return Err(stub_err("matmul: a buffer too small"));
        }
        if b.len() < k * n {
            return Err(stub_err("matmul: b buffer too small"));
        }
        if c.len() < m * n {
            return Err(stub_err("matmul: c buffer too small"));
        }

        if self.config.mock {
            // CPU reference: naive triple-loop matmul.
            for i in 0..m {
                for j in 0..n {
                    let mut sum = 0.0f32;
                    for p in 0..k {
                        sum += a[i * k + p] * b[p * n + j];
                    }
                    c[i * n + j] = sum;
                }
            }
            Ok(())
        } else {
            Err(stub_err("matmul"))
        }
    }

    pub fn config(&self) -> &HipMatmulConfig {
        &self.config
    }
}

// ── 2. HipSoftmaxKernel ─────────────────────────────────────────────

/// Configuration for HIP softmax with wavefront-level reduction.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HipSoftmaxConfig {
    /// Whether to apply the softmax in-place.
    pub in_place: bool,
    /// Whether to use the numerically stable (online) variant.
    pub stable: bool,
    /// Mock mode for CPU fallback.
    pub mock: bool,
}

impl Default for HipSoftmaxConfig {
    fn default() -> Self {
        Self { in_place: false, stable: true, mock: false }
    }
}

impl HipSoftmaxConfig {
    pub fn mock() -> Self {
        Self { mock: true, ..Self::default() }
    }
}

/// HIP softmax kernel with warp-level (wavefront) reduction.
///
/// Operates on rows of a `[num_rows, row_len]` matrix.  Each wavefront
/// collaboratively computes the max, exp-sum, and normalization for one row.
#[derive(Debug, Clone)]
pub struct HipSoftmaxKernel {
    config: HipSoftmaxConfig,
}

impl HipSoftmaxKernel {
    pub fn new(config: HipSoftmaxConfig) -> Self {
        Self { config }
    }

    /// Execute row-wise softmax.
    pub fn execute(
        &self,
        input: &[f32],
        output: &mut [f32],
        num_rows: usize,
        row_len: usize,
    ) -> Result<()> {
        if input.len() < num_rows * row_len {
            return Err(stub_err("softmax: input too small"));
        }
        if output.len() < num_rows * row_len {
            return Err(stub_err("softmax: output too small"));
        }

        if self.config.mock {
            for r in 0..num_rows {
                let offset = r * row_len;
                let row = &input[offset..offset + row_len];

                let max_val = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let exp_sum: f32 = row.iter().map(|&x| (x - max_val).exp()).sum();

                for (j, &x) in row.iter().enumerate() {
                    output[offset + j] = (x - max_val).exp() / exp_sum;
                }
            }
            Ok(())
        } else {
            Err(stub_err("softmax"))
        }
    }

    /// In-place softmax variant.
    pub fn execute_inplace(&self, data: &mut [f32], num_rows: usize, row_len: usize) -> Result<()> {
        if data.len() < num_rows * row_len {
            return Err(stub_err("softmax_inplace: buffer too small"));
        }

        if self.config.mock {
            for r in 0..num_rows {
                let offset = r * row_len;
                let max_val = data[offset..offset + row_len]
                    .iter()
                    .copied()
                    .fold(f32::NEG_INFINITY, f32::max);
                let mut exp_sum = 0.0f32;
                for j in 0..row_len {
                    let e = (data[offset + j] - max_val).exp();
                    data[offset + j] = e;
                    exp_sum += e;
                }
                for j in 0..row_len {
                    data[offset + j] /= exp_sum;
                }
            }
            Ok(())
        } else {
            Err(stub_err("softmax_inplace"))
        }
    }

    pub fn config(&self) -> &HipSoftmaxConfig {
        &self.config
    }
}

// ── 3. HipLayerNormKernel ────────────────────────────────────────────

/// Configuration for HIP LayerNorm.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HipLayerNormConfig {
    /// Hidden dimension (elements per row).
    pub hidden_dim: usize,
    /// Epsilon for numerical stability.
    pub eps: f32,
    /// Mock mode.
    pub mock: bool,
}

impl HipLayerNormConfig {
    pub fn new(hidden_dim: usize) -> Self {
        Self { hidden_dim, eps: 1e-5, mock: false }
    }

    pub fn mock(hidden_dim: usize) -> Self {
        Self { hidden_dim, eps: 1e-5, mock: true }
    }
}

/// HIP LayerNorm kernel (classic mean + variance normalization).
///
/// For each row **x** of length `hidden_dim`:
/// ```text
/// mean = (1/d) * Σ x_i
/// var  = (1/d) * Σ (x_i - mean)²
/// out  = gamma * (x - mean) / sqrt(var + eps) + beta
/// ```
#[derive(Debug, Clone)]
pub struct HipLayerNormKernel {
    config: HipLayerNormConfig,
}

impl HipLayerNormKernel {
    pub fn new(config: HipLayerNormConfig) -> Self {
        Self { config }
    }

    /// Execute LayerNorm.
    pub fn execute(
        &self,
        input: &[f32],
        gamma: &[f32],
        beta: &[f32],
        output: &mut [f32],
        num_rows: usize,
    ) -> Result<()> {
        let d = self.config.hidden_dim;
        if gamma.len() < d || beta.len() < d {
            return Err(stub_err("layer_norm: gamma/beta size mismatch"));
        }
        if input.len() < num_rows * d || output.len() < num_rows * d {
            return Err(stub_err("layer_norm: buffer size mismatch"));
        }

        if self.config.mock {
            for r in 0..num_rows {
                let off = r * d;
                let row = &input[off..off + d];

                let mean: f32 = row.iter().sum::<f32>() / d as f32;
                let var: f32 = row.iter().map(|&x| (x - mean) * (x - mean)).sum::<f32>() / d as f32;
                let inv_std = 1.0 / (var + self.config.eps).sqrt();

                for j in 0..d {
                    output[off + j] = gamma[j] * (row[j] - mean) * inv_std + beta[j];
                }
            }
            Ok(())
        } else {
            Err(stub_err("layer_norm"))
        }
    }

    pub fn config(&self) -> &HipLayerNormConfig {
        &self.config
    }
}

// ── 4. HipQuantKernel ────────────────────────────────────────────────

/// Quantization bit-width selector.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HipQuantBits {
    /// 2-bit signed quantization (I2_S / QK256).
    Two,
    /// 4-bit quantization.
    Four,
    /// 8-bit quantization.
    Eight,
}

/// Configuration for HIP quantization / dequantization.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HipQuantConfig {
    /// Target bit-width.
    pub bits: HipQuantBits,
    /// Block size for block-wise quantization.
    pub block_size: usize,
    /// Mock mode.
    pub mock: bool,
}

impl Default for HipQuantConfig {
    fn default() -> Self {
        Self { bits: HipQuantBits::Two, block_size: 256, mock: false }
    }
}

impl HipQuantConfig {
    pub fn mock() -> Self {
        Self { mock: true, ..Self::default() }
    }

    pub fn mock_with_bits(bits: HipQuantBits) -> Self {
        Self { bits, block_size: 256, mock: true }
    }
}

/// HIP quantization / dequantization kernel.
///
/// Supports block-wise symmetric quantization with per-block scale factors.
#[derive(Debug, Clone)]
pub struct HipQuantKernel {
    config: HipQuantConfig,
}

impl HipQuantKernel {
    pub fn new(config: HipQuantConfig) -> Self {
        Self { config }
    }

    /// Quantize `input` floats → packed bytes + scales.
    ///
    /// In mock mode, uses a simple abs-max symmetric scheme.
    pub fn quantize(&self, input: &[f32], output: &mut [u8], scales: &mut [f32]) -> Result<()> {
        let bs = self.config.block_size;
        let num_blocks = input.len().div_ceil(bs);
        if scales.len() < num_blocks {
            return Err(stub_err("quantize: scales buffer too small"));
        }

        if self.config.mock {
            let max_int = match self.config.bits {
                HipQuantBits::Two => 1.0f32,  // {-1, 0, 1}
                HipQuantBits::Four => 7.0f32, // {-7..7}
                HipQuantBits::Eight => 127.0f32,
            };

            // Zero output buffer.
            for b in output.iter_mut() {
                *b = 0;
            }

            let mut out_idx = 0usize;
            for (blk, scale_slot) in scales.iter_mut().enumerate().take(num_blocks) {
                let start = blk * bs;
                let end = (start + bs).min(input.len());
                let block = &input[start..end];

                let abs_max = block.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
                let scale = if abs_max == 0.0 { 1.0 } else { abs_max / max_int };
                *scale_slot = scale;

                for &val in block {
                    let q = (val / scale).round().clamp(-max_int, max_int) as i8;
                    if out_idx < output.len() {
                        output[out_idx] = q as u8;
                        out_idx += 1;
                    }
                }
            }
            Ok(())
        } else {
            Err(stub_err("quantize"))
        }
    }

    /// Dequantize packed bytes + scales → floats.
    pub fn dequantize(&self, input: &[u8], scales: &[f32], output: &mut [f32]) -> Result<()> {
        let bs = self.config.block_size;
        let num_blocks = scales.len();
        let total = num_blocks * bs;
        if output.len() < total.min(input.len()) {
            return Err(stub_err("dequantize: output too small"));
        }

        if self.config.mock {
            for (blk, &scale) in scales.iter().enumerate().take(num_blocks) {
                let start = blk * bs;
                let end = (start + bs).min(input.len()).min(output.len());
                for i in start..end {
                    output[i] = (input[i] as i8) as f32 * scale;
                }
            }
            Ok(())
        } else {
            Err(stub_err("dequantize"))
        }
    }

    pub fn config(&self) -> &HipQuantConfig {
        &self.config
    }
}

// ── 5. HipAttentionKernel ────────────────────────────────────────────

/// Configuration for HIP fused attention.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HipAttentionKernelConfig {
    /// Number of attention heads.
    pub num_heads: usize,
    /// Per-head dimension.
    pub head_dim: usize,
    /// Apply causal mask.
    pub causal: bool,
    /// Scale factor (typically `1 / sqrt(head_dim)`).
    pub scale: f32,
    /// Mock mode.
    pub mock: bool,
}

impl HipAttentionKernelConfig {
    pub fn new(num_heads: usize, head_dim: usize) -> Self {
        Self {
            num_heads,
            head_dim,
            causal: true,
            scale: 1.0 / (head_dim as f32).sqrt(),
            mock: false,
        }
    }

    pub fn mock(num_heads: usize, head_dim: usize) -> Self {
        Self { mock: true, ..Self::new(num_heads, head_dim) }
    }
}

/// HIP fused scaled-dot-product attention kernel.
///
/// Computes `softmax(Q·K^T / scale) · V` with optional causal masking.
/// Input shapes (per batch): Q/K/V `[num_heads, seq_len, head_dim]`.
#[derive(Debug, Clone)]
pub struct HipAttentionKernel {
    config: HipAttentionKernelConfig,
}

impl HipAttentionKernel {
    pub fn new(config: HipAttentionKernelConfig) -> Self {
        Self { config }
    }

    /// Execute fused attention for a single batch.
    pub fn execute(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        output: &mut [f32],
        seq_len: usize,
    ) -> Result<()> {
        let h = self.config.num_heads;
        let d = self.config.head_dim;
        let expected = h * seq_len * d;
        if q.len() < expected || k.len() < expected || v.len() < expected {
            return Err(stub_err("attention: input size mismatch"));
        }
        if output.len() < expected {
            return Err(stub_err("attention: output size mismatch"));
        }

        if self.config.mock {
            for head in 0..h {
                let base = head * seq_len * d;

                // scores = Q · K^T  → [seq_len, seq_len]
                let mut scores = vec![0.0f32; seq_len * seq_len];
                for i in 0..seq_len {
                    for j in 0..seq_len {
                        let mut dot = 0.0f32;
                        for p in 0..d {
                            dot += q[base + i * d + p] * k[base + j * d + p];
                        }
                        scores[i * seq_len + j] = dot * self.config.scale;
                    }
                }

                // Causal mask: set future positions to -inf.
                if self.config.causal {
                    for i in 0..seq_len {
                        for j in (i + 1)..seq_len {
                            scores[i * seq_len + j] = f32::NEG_INFINITY;
                        }
                    }
                }

                // Row-wise softmax.
                for i in 0..seq_len {
                    let row = &mut scores[i * seq_len..(i + 1) * seq_len];
                    let max_val = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                    let exp_sum: f32 = row.iter().map(|&x| (x - max_val).exp()).sum();
                    for x in row.iter_mut() {
                        *x = (*x - max_val).exp() / exp_sum;
                    }
                }

                // output = scores · V
                for i in 0..seq_len {
                    for p in 0..d {
                        let mut acc = 0.0f32;
                        for j in 0..seq_len {
                            acc += scores[i * seq_len + j] * v[base + j * d + p];
                        }
                        output[base + i * d + p] = acc;
                    }
                }
            }
            Ok(())
        } else {
            Err(stub_err("attention"))
        }
    }

    pub fn config(&self) -> &HipAttentionKernelConfig {
        &self.config
    }
}

// ── 6. HipDeviceQuery ────────────────────────────────────────────────

/// GCN / CDNA architecture tier (determines available instructions).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum GcnArchTier {
    /// Unknown or unsupported architecture.
    Unknown,
    /// GCN 1–5 (e.g. Vega, Polaris) — basic wavefront ops.
    Gcn,
    /// CDNA 1 (MI100) — MFMA FP16.
    Cdna1,
    /// CDNA 2 (MI200 / MI250X) — MFMA FP16 + BF16.
    Cdna2,
    /// CDNA 3 (MI300 series) — FP8 matrix ops.
    Cdna3,
    /// RDNA 3 (RX 7000 series) — WMMA instructions.
    Rdna3,
}

/// Detected capabilities for a single HIP device.
#[derive(Debug, Clone, PartialEq)]
pub struct HipDeviceCaps {
    /// Device ordinal.
    pub device_id: usize,
    /// Marketing name.
    pub name: String,
    /// GCN architecture string (e.g. "gfx90a").
    pub gcn_arch: String,
    /// Architecture tier.
    pub arch_tier: GcnArchTier,
    /// Total device memory in bytes.
    pub total_memory_bytes: usize,
    /// Number of compute units (CUs).
    pub compute_units: u32,
    /// Wavefront size (64 on all current AMD GPUs).
    pub wavefront_size: u32,
    /// Maximum LDS per work-group in bytes.
    pub max_lds_per_workgroup: usize,
    /// FP16 support.
    pub fp16: bool,
    /// BF16 support.
    pub bf16: bool,
    /// Matrix core (MFMA / WMMA) support.
    pub matrix_ops: bool,
}

/// HIP device query and capability detection.
///
/// On systems without the HIP runtime, [`HipDeviceQuery::detect`] returns
/// an empty list.  Tests can use [`HipDeviceQuery::mock_device`] to
/// fabricate device capabilities.
#[derive(Debug, Clone)]
pub struct HipDeviceQuery;

impl HipDeviceQuery {
    /// Detect all HIP-visible devices.
    ///
    /// Stub — always returns an empty list until the HIP FFI is wired in.
    pub fn detect() -> Vec<HipDeviceCaps> {
        Vec::new()
    }

    /// Return the number of detected devices.
    pub fn device_count() -> usize {
        Self::detect().len()
    }

    /// Whether any HIP device is available.
    pub fn is_available() -> bool {
        Self::device_count() > 0
    }

    /// Classify a GCN architecture string into a tier.
    pub fn classify_arch(gcn_arch: &str) -> GcnArchTier {
        match gcn_arch {
            s if s.starts_with("gfx94") => GcnArchTier::Cdna3,
            s if s.starts_with("gfx90a") || s.starts_with("gfx90") => GcnArchTier::Cdna2,
            s if s.starts_with("gfx908") => GcnArchTier::Cdna1,
            s if s.starts_with("gfx11") => GcnArchTier::Rdna3,
            s if s.starts_with("gfx") => GcnArchTier::Gcn,
            _ => GcnArchTier::Unknown,
        }
    }

    /// Create a mock device for testing.
    pub fn mock_device(name: &str, gcn_arch: &str, memory_gb: usize) -> HipDeviceCaps {
        let tier = Self::classify_arch(gcn_arch);
        HipDeviceCaps {
            device_id: 0,
            name: name.to_string(),
            gcn_arch: gcn_arch.to_string(),
            arch_tier: tier,
            total_memory_bytes: memory_gb * 1024 * 1024 * 1024,
            compute_units: 120,
            wavefront_size: HIP_WAVEFRONT_SIZE,
            max_lds_per_workgroup: HIP_MAX_LDS_BYTES,
            fp16: tier >= GcnArchTier::Gcn,
            bf16: tier >= GcnArchTier::Cdna2,
            matrix_ops: tier >= GcnArchTier::Cdna1,
        }
    }

    /// Suggest optimal work-group size for a given device.
    pub fn suggest_workgroup_size(caps: &HipDeviceCaps) -> u32 {
        // 4 wavefronts per work-group is a good default for occupancy.
        let waves_per_wg: u32 = 4;
        caps.wavefront_size * waves_per_wg
    }

    /// Whether the device supports BF16 matrix operations.
    pub fn supports_bf16_matmul(caps: &HipDeviceCaps) -> bool {
        caps.bf16 && caps.matrix_ops
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Tests — 70+ covering all six kernel abstractions
// ═══════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    // ── HipMatmulKernel tests ────────────────────────────────────────

    #[test]
    fn matmul_mock_identity_2x2() {
        let k = HipMatmulKernel::new(HipMatmulConfig::mock());
        // A = I₂, B = [[1,2],[3,4]]  → C = B
        let a = vec![1.0, 0.0, 0.0, 1.0];
        let b = vec![1.0, 2.0, 3.0, 4.0];
        let mut c = vec![0.0; 4];
        k.execute(&a, &b, &mut c, 2, 2, 2).unwrap();
        assert_eq!(c, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn matmul_mock_3x2_times_2x3() {
        let k = HipMatmulKernel::new(HipMatmulConfig::mock());
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 3×2
        let b = vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]; // 2×3
        let mut c = vec![0.0; 9]; // 3×3
        k.execute(&a, &b, &mut c, 3, 3, 2).unwrap();
        assert_eq!(c, vec![27.0, 30.0, 33.0, 61.0, 68.0, 75.0, 95.0, 106.0, 117.0]);
    }

    #[test]
    fn matmul_mock_1x1() {
        let k = HipMatmulKernel::new(HipMatmulConfig::mock());
        let mut c = vec![0.0];
        k.execute(&[3.0], &[5.0], &mut c, 1, 1, 1).unwrap();
        assert!((c[0] - 15.0).abs() < 1e-6);
    }

    #[test]
    fn matmul_mock_zeros() {
        let k = HipMatmulKernel::new(HipMatmulConfig::mock());
        let a = vec![0.0; 16];
        let b = vec![1.0; 16];
        let mut c = vec![999.0; 16];
        k.execute(&a, &b, &mut c, 4, 4, 4).unwrap();
        assert!(c.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn matmul_stub_returns_err() {
        let k = HipMatmulKernel::new(HipMatmulConfig::default());
        let a = vec![1.0; 4];
        let b = vec![1.0; 4];
        let mut c = vec![0.0; 4];
        assert!(k.execute(&a, &b, &mut c, 2, 2, 2).is_err());
    }

    #[test]
    fn matmul_a_buffer_too_small() {
        let k = HipMatmulKernel::new(HipMatmulConfig::mock());
        let a = vec![1.0; 2]; // need 4
        let b = vec![1.0; 4];
        let mut c = vec![0.0; 4];
        assert!(k.execute(&a, &b, &mut c, 2, 2, 2).is_err());
    }

    #[test]
    fn matmul_b_buffer_too_small() {
        let k = HipMatmulKernel::new(HipMatmulConfig::mock());
        let a = vec![1.0; 4];
        let b = vec![1.0; 2]; // need 4
        let mut c = vec![0.0; 4];
        assert!(k.execute(&a, &b, &mut c, 2, 2, 2).is_err());
    }

    #[test]
    fn matmul_c_buffer_too_small() {
        let k = HipMatmulKernel::new(HipMatmulConfig::mock());
        let a = vec![1.0; 4];
        let b = vec![1.0; 4];
        let mut c = vec![0.0; 2]; // need 4
        assert!(k.execute(&a, &b, &mut c, 2, 2, 2).is_err());
    }

    #[test]
    fn matmul_config_default() {
        let cfg = HipMatmulConfig::default();
        assert_eq!(cfg.workgroup_size, 256);
        assert_eq!(cfg.tile_dim, 16);
        assert!(!cfg.mock);
    }

    #[test]
    fn matmul_config_accessor() {
        let k = HipMatmulKernel::new(HipMatmulConfig::mock());
        assert!(k.config().mock);
    }

    #[test]
    fn matmul_mock_rectangular_4x1_times_1x3() {
        let k = HipMatmulKernel::new(HipMatmulConfig::mock());
        let a = vec![1.0, 2.0, 3.0, 4.0]; // 4×1
        let b = vec![2.0, 3.0, 4.0]; // 1×3
        let mut c = vec![0.0; 12]; // 4×3
        k.execute(&a, &b, &mut c, 4, 3, 1).unwrap();
        assert_eq!(c, vec![2.0, 3.0, 4.0, 4.0, 6.0, 8.0, 6.0, 9.0, 12.0, 8.0, 12.0, 16.0]);
    }

    // ── HipSoftmaxKernel tests ───────────────────────────────────────

    #[test]
    fn softmax_mock_single_row() {
        let k = HipSoftmaxKernel::new(HipSoftmaxConfig::mock());
        let input = vec![1.0, 2.0, 3.0];
        let mut output = vec![0.0; 3];
        k.execute(&input, &mut output, 1, 3).unwrap();
        let sum: f32 = output.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn softmax_mock_two_rows() {
        let k = HipSoftmaxKernel::new(HipSoftmaxConfig::mock());
        let input = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let mut output = vec![0.0; 6];
        k.execute(&input, &mut output, 2, 3).unwrap();
        // Row of equal values → uniform distribution.
        for &v in &output[0..3] {
            assert!((v - 1.0 / 3.0).abs() < 1e-5);
        }
        for &v in &output[3..6] {
            assert!((v - 1.0 / 3.0).abs() < 1e-5);
        }
    }

    #[test]
    fn softmax_mock_large_values_stable() {
        let k = HipSoftmaxKernel::new(HipSoftmaxConfig::mock());
        let input = vec![1000.0, 1001.0, 1002.0];
        let mut output = vec![0.0; 3];
        k.execute(&input, &mut output, 1, 3).unwrap();
        let sum: f32 = output.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "should be stable with large values");
        assert!(output[2] > output[1] && output[1] > output[0]);
    }

    #[test]
    fn softmax_mock_negative_values() {
        let k = HipSoftmaxKernel::new(HipSoftmaxConfig::mock());
        let input = vec![-1.0, -2.0, -3.0];
        let mut output = vec![0.0; 3];
        k.execute(&input, &mut output, 1, 3).unwrap();
        let sum: f32 = output.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
        assert!(output[0] > output[1] && output[1] > output[2]);
    }

    #[test]
    fn softmax_inplace_mock() {
        let k = HipSoftmaxKernel::new(HipSoftmaxConfig::mock());
        let mut data = vec![1.0, 2.0, 3.0];
        k.execute_inplace(&mut data, 1, 3).unwrap();
        let sum: f32 = data.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn softmax_stub_returns_err() {
        let k = HipSoftmaxKernel::new(HipSoftmaxConfig::default());
        let input = vec![1.0; 3];
        let mut output = vec![0.0; 3];
        assert!(k.execute(&input, &mut output, 1, 3).is_err());
    }

    #[test]
    fn softmax_input_too_small() {
        let k = HipSoftmaxKernel::new(HipSoftmaxConfig::mock());
        let input = vec![1.0; 2]; // need 3
        let mut output = vec![0.0; 3];
        assert!(k.execute(&input, &mut output, 1, 3).is_err());
    }

    #[test]
    fn softmax_output_too_small() {
        let k = HipSoftmaxKernel::new(HipSoftmaxConfig::mock());
        let input = vec![1.0; 3];
        let mut output = vec![0.0; 2]; // need 3
        assert!(k.execute(&input, &mut output, 1, 3).is_err());
    }

    #[test]
    fn softmax_inplace_buffer_too_small() {
        let k = HipSoftmaxKernel::new(HipSoftmaxConfig::mock());
        let mut data = vec![1.0; 2]; // need 3
        assert!(k.execute_inplace(&mut data, 1, 3).is_err());
    }

    #[test]
    fn softmax_config_accessor() {
        let k = HipSoftmaxKernel::new(HipSoftmaxConfig::mock());
        assert!(k.config().mock);
        assert!(k.config().stable);
    }

    #[test]
    fn softmax_mock_single_element() {
        let k = HipSoftmaxKernel::new(HipSoftmaxConfig::mock());
        let input = vec![42.0];
        let mut output = vec![0.0];
        k.execute(&input, &mut output, 1, 1).unwrap();
        assert!((output[0] - 1.0).abs() < 1e-6);
    }

    // ── HipLayerNormKernel tests ─────────────────────────────────────

    #[test]
    fn layer_norm_mock_identity_gamma_zero_beta() {
        let cfg = HipLayerNormConfig::mock(4);
        let k = HipLayerNormKernel::new(cfg);
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let gamma = vec![1.0; 4];
        let beta = vec![0.0; 4];
        let mut output = vec![0.0; 4];
        k.execute(&input, &gamma, &beta, &mut output, 1).unwrap();
        // Output should be zero-mean with unit variance (approx).
        let mean: f32 = output.iter().sum::<f32>() / 4.0;
        assert!(mean.abs() < 1e-5, "mean should be ~0, got {mean}");
    }

    #[test]
    fn layer_norm_mock_constant_input() {
        let cfg = HipLayerNormConfig::mock(3);
        let k = HipLayerNormKernel::new(cfg);
        let input = vec![5.0, 5.0, 5.0];
        let gamma = vec![1.0; 3];
        let beta = vec![0.0; 3];
        let mut output = vec![0.0; 3];
        k.execute(&input, &gamma, &beta, &mut output, 1).unwrap();
        // Constant input → zero after normalization.
        for &v in &output {
            assert!(v.abs() < 1e-3);
        }
    }

    #[test]
    fn layer_norm_mock_with_beta() {
        let cfg = HipLayerNormConfig::mock(2);
        let k = HipLayerNormKernel::new(cfg);
        let input = vec![0.0, 0.0];
        let gamma = vec![1.0; 2];
        let beta = vec![3.0, 7.0];
        let mut output = vec![0.0; 2];
        k.execute(&input, &gamma, &beta, &mut output, 1).unwrap();
        // All-zero input after norm → 0; + beta → beta.
        assert!((output[0] - 3.0).abs() < 1e-3);
        assert!((output[1] - 7.0).abs() < 1e-3);
    }

    #[test]
    fn layer_norm_mock_multi_row() {
        let cfg = HipLayerNormConfig::mock(3);
        let k = HipLayerNormKernel::new(cfg);
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let gamma = vec![1.0; 3];
        let beta = vec![0.0; 3];
        let mut output = vec![0.0; 6];
        k.execute(&input, &gamma, &beta, &mut output, 2).unwrap();
        // Each row independently normalized.
        let mean0: f32 = output[0..3].iter().sum::<f32>() / 3.0;
        let mean1: f32 = output[3..6].iter().sum::<f32>() / 3.0;
        assert!(mean0.abs() < 1e-5);
        assert!(mean1.abs() < 1e-5);
    }

    #[test]
    fn layer_norm_stub_returns_err() {
        let cfg = HipLayerNormConfig::new(4);
        let k = HipLayerNormKernel::new(cfg);
        let input = vec![1.0; 4];
        let gamma = vec![1.0; 4];
        let beta = vec![0.0; 4];
        let mut output = vec![0.0; 4];
        assert!(k.execute(&input, &gamma, &beta, &mut output, 1).is_err());
    }

    #[test]
    fn layer_norm_gamma_too_small() {
        let cfg = HipLayerNormConfig::mock(4);
        let k = HipLayerNormKernel::new(cfg);
        let input = vec![1.0; 4];
        let gamma = vec![1.0; 2]; // need 4
        let beta = vec![0.0; 4];
        let mut output = vec![0.0; 4];
        assert!(k.execute(&input, &gamma, &beta, &mut output, 1).is_err());
    }

    #[test]
    fn layer_norm_input_too_small() {
        let cfg = HipLayerNormConfig::mock(4);
        let k = HipLayerNormKernel::new(cfg);
        let input = vec![1.0; 2]; // need 4
        let gamma = vec![1.0; 4];
        let beta = vec![0.0; 4];
        let mut output = vec![0.0; 4];
        assert!(k.execute(&input, &gamma, &beta, &mut output, 1).is_err());
    }

    #[test]
    fn layer_norm_config_default_eps() {
        let cfg = HipLayerNormConfig::new(128);
        assert!((cfg.eps - 1e-5).abs() < 1e-10);
        assert!(!cfg.mock);
    }

    #[test]
    fn layer_norm_config_accessor() {
        let cfg = HipLayerNormConfig::mock(64);
        let k = HipLayerNormKernel::new(cfg);
        assert_eq!(k.config().hidden_dim, 64);
        assert!(k.config().mock);
    }

    #[test]
    fn layer_norm_mock_gamma_scaling() {
        let cfg = HipLayerNormConfig::mock(2);
        let k = HipLayerNormKernel::new(cfg);
        let input = vec![-1.0, 1.0];
        let gamma = vec![2.0, 2.0];
        let beta = vec![0.0; 2];
        let mut output_g2 = vec![0.0; 2];
        k.execute(&input, &gamma, &beta, &mut output_g2, 1).unwrap();

        let gamma1 = vec![1.0, 1.0];
        let mut output_g1 = vec![0.0; 2];
        k.execute(&input, &gamma1, &beta, &mut output_g1, 1).unwrap();

        // gamma=2 should double the normalized output.
        for i in 0..2 {
            assert!((output_g2[i] - 2.0 * output_g1[i]).abs() < 1e-5);
        }
    }

    // ── HipQuantKernel tests ─────────────────────────────────────────

    #[test]
    fn quant_mock_roundtrip_2bit() {
        let k = HipQuantKernel::new(HipQuantConfig::mock());
        let input = vec![0.5, -0.5, 0.0, 1.0];
        let mut packed = vec![0u8; 4];
        let mut scales = vec![0.0f32; 1];
        k.quantize(&input, &mut packed, &mut scales).unwrap();
        assert!(scales[0] > 0.0);

        let mut output = vec![0.0f32; 4];
        k.dequantize(&packed, &scales, &mut output).unwrap();
        // Roundtrip should preserve sign.
        assert!(output[0] > 0.0);
        assert!(output[1] < 0.0);
    }

    #[test]
    fn quant_mock_zeros() {
        let k = HipQuantKernel::new(HipQuantConfig::mock());
        let input = vec![0.0; 8];
        let mut packed = vec![0u8; 8];
        let mut scales = vec![0.0f32; 1];
        k.quantize(&input, &mut packed, &mut scales).unwrap();
        let mut output = vec![999.0f32; 8];
        k.dequantize(&packed, &scales, &mut output).unwrap();
        for &v in &output[..8] {
            assert!(v.abs() < 1e-6);
        }
    }

    #[test]
    fn quant_mock_4bit() {
        let cfg = HipQuantConfig::mock_with_bits(HipQuantBits::Four);
        let k = HipQuantKernel::new(cfg);
        let input = vec![3.5, -3.5, 0.0, 7.0];
        let mut packed = vec![0u8; 4];
        let mut scales = vec![0.0f32; 1];
        k.quantize(&input, &mut packed, &mut scales).unwrap();
        assert!(scales[0] > 0.0);
    }

    #[test]
    fn quant_mock_8bit() {
        let cfg = HipQuantConfig::mock_with_bits(HipQuantBits::Eight);
        let k = HipQuantKernel::new(cfg);
        let input = vec![100.0, -50.0, 25.0, 0.0];
        let mut packed = vec![0u8; 4];
        let mut scales = vec![0.0f32; 1];
        k.quantize(&input, &mut packed, &mut scales).unwrap();
        assert!(scales[0] > 0.0);
    }

    #[test]
    fn quant_stub_returns_err() {
        let k = HipQuantKernel::new(HipQuantConfig::default());
        let input = vec![1.0; 4];
        let mut packed = vec![0u8; 4];
        let mut scales = vec![0.0f32; 1];
        assert!(k.quantize(&input, &mut packed, &mut scales).is_err());
    }

    #[test]
    fn dequant_stub_returns_err() {
        let k = HipQuantKernel::new(HipQuantConfig::default());
        let packed = vec![0u8; 4];
        let scales = vec![1.0f32; 1];
        let mut output = vec![0.0f32; 4];
        assert!(k.dequantize(&packed, &scales, &mut output).is_err());
    }

    #[test]
    fn quant_scales_too_small() {
        let k = HipQuantKernel::new(HipQuantConfig::mock());
        let input = vec![1.0; 512]; // 2 blocks of 256
        let mut packed = vec![0u8; 512];
        let mut scales = vec![0.0f32; 1]; // need 2
        assert!(k.quantize(&input, &mut packed, &mut scales).is_err());
    }

    #[test]
    fn quant_config_default() {
        let cfg = HipQuantConfig::default();
        assert_eq!(cfg.bits, HipQuantBits::Two);
        assert_eq!(cfg.block_size, 256);
        assert!(!cfg.mock);
    }

    #[test]
    fn quant_config_accessor() {
        let k = HipQuantKernel::new(HipQuantConfig::mock());
        assert!(k.config().mock);
        assert_eq!(k.config().bits, HipQuantBits::Two);
    }

    #[test]
    fn quant_mock_preserves_magnitude_order() {
        let k = HipQuantKernel::new(HipQuantConfig::mock_with_bits(HipQuantBits::Eight));
        let input = vec![10.0, 50.0, 100.0, 0.0];
        let mut packed = vec![0u8; 4];
        let mut scales = vec![0.0f32; 1];
        k.quantize(&input, &mut packed, &mut scales).unwrap();
        let mut output = vec![0.0f32; 4];
        k.dequantize(&packed, &scales, &mut output).unwrap();
        assert!(output[2] > output[1] && output[1] > output[0]);
    }

    #[test]
    fn quant_dequant_output_too_small() {
        let k = HipQuantKernel::new(HipQuantConfig::mock());
        let packed = vec![0u8; 256];
        let scales = vec![1.0f32; 1];
        let mut output = vec![0.0f32; 100]; // need 256
        assert!(k.dequantize(&packed, &scales, &mut output).is_err());
    }

    // ── HipAttentionKernel tests ─────────────────────────────────────

    #[test]
    fn attention_mock_single_head_seq1() {
        let cfg = HipAttentionKernelConfig::mock(1, 4);
        let k = HipAttentionKernel::new(cfg);
        let q = vec![1.0, 0.0, 0.0, 0.0];
        let kk = vec![1.0, 0.0, 0.0, 0.0];
        let v = vec![0.0, 1.0, 0.0, 0.0];
        let mut out = vec![0.0; 4];
        k.execute(&q, &kk, &v, &mut out, 1).unwrap();
        // With seq_len=1 the output is just V (softmax of a single score = 1).
        assert!((out[1] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn attention_mock_causal_mask() {
        let cfg = HipAttentionKernelConfig::mock(1, 2);
        let k = HipAttentionKernel::new(cfg);
        // seq_len=2, head_dim=2
        let q = vec![1.0, 0.0, 0.0, 1.0]; // 2 tokens
        let kk = vec![1.0, 0.0, 0.0, 1.0];
        let v = vec![1.0, 0.0, 0.0, 1.0];
        let mut out = vec![0.0; 4];
        k.execute(&q, &kk, &v, &mut out, 2).unwrap();
        // First token can only attend to itself (causal).
        assert!((out[0] - 1.0).abs() < 1e-4);
        assert!((out[1] - 0.0).abs() < 1e-4);
    }

    #[test]
    fn attention_mock_multi_head() {
        let cfg = HipAttentionKernelConfig::mock(2, 3);
        let k = HipAttentionKernel::new(cfg);
        let size = 2 * 1 * 3; // 2 heads, seq_len=1, head_dim=3
        let q = vec![1.0; size];
        let kk = vec![1.0; size];
        let v = vec![1.0; size];
        let mut out = vec![0.0; size];
        k.execute(&q, &kk, &v, &mut out, 1).unwrap();
        // With all-ones input, output should be all-ones (softmax(score)=1, V=1).
        for &o in &out {
            assert!((o - 1.0).abs() < 1e-4);
        }
    }

    #[test]
    fn attention_stub_returns_err() {
        let cfg = HipAttentionKernelConfig::new(1, 4);
        let k = HipAttentionKernel::new(cfg);
        let q = vec![1.0; 4];
        let kk = vec![1.0; 4];
        let v = vec![1.0; 4];
        let mut out = vec![0.0; 4];
        assert!(k.execute(&q, &kk, &v, &mut out, 1).is_err());
    }

    #[test]
    fn attention_input_size_mismatch() {
        let cfg = HipAttentionKernelConfig::mock(1, 4);
        let k = HipAttentionKernel::new(cfg);
        let q = vec![1.0; 2]; // too small
        let kk = vec![1.0; 4];
        let v = vec![1.0; 4];
        let mut out = vec![0.0; 4];
        assert!(k.execute(&q, &kk, &v, &mut out, 1).is_err());
    }

    #[test]
    fn attention_output_size_mismatch() {
        let cfg = HipAttentionKernelConfig::mock(1, 4);
        let k = HipAttentionKernel::new(cfg);
        let q = vec![1.0; 4];
        let kk = vec![1.0; 4];
        let v = vec![1.0; 4];
        let mut out = vec![0.0; 2]; // too small
        assert!(k.execute(&q, &kk, &v, &mut out, 1).is_err());
    }

    #[test]
    fn attention_config_scale() {
        let cfg = HipAttentionKernelConfig::new(8, 64);
        let expected = 1.0 / (64.0f32).sqrt();
        assert!((cfg.scale - expected).abs() < 1e-6);
    }

    #[test]
    fn attention_config_accessor() {
        let cfg = HipAttentionKernelConfig::mock(4, 32);
        let k = HipAttentionKernel::new(cfg);
        assert_eq!(k.config().num_heads, 4);
        assert_eq!(k.config().head_dim, 32);
        assert!(k.config().mock);
    }

    #[test]
    fn attention_mock_non_causal() {
        let mut cfg = HipAttentionKernelConfig::mock(1, 2);
        cfg.causal = false;
        let k = HipAttentionKernel::new(cfg);
        let q = vec![1.0, 0.0, 0.0, 1.0];
        let kk = vec![1.0, 0.0, 0.0, 1.0];
        let v = vec![1.0, 0.0, 0.0, 1.0];
        let mut out = vec![0.0; 4];
        // Non-causal: first token can attend to both tokens.
        k.execute(&q, &kk, &v, &mut out, 2).unwrap();
        // Score matrix without causal mask allows full attention.
        // Just verify it doesn't error and produces finite output.
        for &o in &out {
            assert!(o.is_finite());
        }
    }

    // ── HipDeviceQuery tests ─────────────────────────────────────────

    #[test]
    fn device_query_detect_empty() {
        // Without HIP runtime, should return empty.
        assert!(HipDeviceQuery::detect().is_empty());
    }

    #[test]
    fn device_query_count_zero() {
        assert_eq!(HipDeviceQuery::device_count(), 0);
    }

    #[test]
    fn device_query_not_available() {
        assert!(!HipDeviceQuery::is_available());
    }

    #[test]
    fn classify_arch_cdna3() {
        assert_eq!(HipDeviceQuery::classify_arch("gfx940"), GcnArchTier::Cdna3);
        assert_eq!(HipDeviceQuery::classify_arch("gfx942"), GcnArchTier::Cdna3);
    }

    #[test]
    fn classify_arch_cdna2() {
        assert_eq!(HipDeviceQuery::classify_arch("gfx90a"), GcnArchTier::Cdna2);
    }

    #[test]
    fn classify_arch_cdna1() {
        assert_eq!(HipDeviceQuery::classify_arch("gfx908"), GcnArchTier::Cdna1);
    }

    #[test]
    fn classify_arch_rdna3() {
        assert_eq!(HipDeviceQuery::classify_arch("gfx1100"), GcnArchTier::Rdna3);
        assert_eq!(HipDeviceQuery::classify_arch("gfx1103"), GcnArchTier::Rdna3);
    }

    #[test]
    fn classify_arch_gcn_generic() {
        assert_eq!(HipDeviceQuery::classify_arch("gfx803"), GcnArchTier::Gcn);
    }

    #[test]
    fn classify_arch_unknown() {
        assert_eq!(HipDeviceQuery::classify_arch(""), GcnArchTier::Unknown);
        assert_eq!(HipDeviceQuery::classify_arch("nvidia"), GcnArchTier::Unknown);
    }

    #[test]
    fn mock_device_mi250x() {
        let dev = HipDeviceQuery::mock_device("MI250X", "gfx90a", 128);
        assert_eq!(dev.name, "MI250X");
        assert_eq!(dev.arch_tier, GcnArchTier::Cdna2);
        assert_eq!(dev.total_memory_bytes, 128 * 1024 * 1024 * 1024);
        assert!(dev.fp16);
        assert!(dev.bf16);
        assert!(dev.matrix_ops);
        assert_eq!(dev.wavefront_size, HIP_WAVEFRONT_SIZE);
    }

    #[test]
    fn mock_device_mi300x() {
        let dev = HipDeviceQuery::mock_device("MI300X", "gfx942", 192);
        assert_eq!(dev.arch_tier, GcnArchTier::Cdna3);
        assert!(dev.bf16);
        assert!(dev.matrix_ops);
    }

    #[test]
    fn mock_device_rdna3() {
        let dev = HipDeviceQuery::mock_device("RX 7900 XTX", "gfx1100", 24);
        assert_eq!(dev.arch_tier, GcnArchTier::Rdna3);
        assert!(dev.fp16);
        // RDNA3 doesn't have BF16 in this classification.
        assert!(!dev.bf16);
    }

    #[test]
    fn mock_device_vega() {
        let dev = HipDeviceQuery::mock_device("Vega 56", "gfx900", 8);
        assert_eq!(dev.arch_tier, GcnArchTier::Cdna2);
        assert!(dev.fp16);
    }

    #[test]
    fn suggest_workgroup_size_default() {
        let dev = HipDeviceQuery::mock_device("MI250X", "gfx90a", 128);
        assert_eq!(HipDeviceQuery::suggest_workgroup_size(&dev), 256);
    }

    #[test]
    fn supports_bf16_matmul_cdna2() {
        let dev = HipDeviceQuery::mock_device("MI250X", "gfx90a", 128);
        assert!(HipDeviceQuery::supports_bf16_matmul(&dev));
    }

    #[test]
    fn supports_bf16_matmul_rdna3() {
        let dev = HipDeviceQuery::mock_device("RX 7900 XTX", "gfx1100", 24);
        // RDNA3 has WMMA but not BF16 per our tier classification.
        assert!(!HipDeviceQuery::supports_bf16_matmul(&dev));
    }

    #[test]
    fn supports_bf16_matmul_gcn() {
        let dev = HipDeviceQuery::mock_device("Polaris", "gfx803", 4);
        assert!(!HipDeviceQuery::supports_bf16_matmul(&dev));
    }

    // ── Constants tests ──────────────────────────────────────────────

    #[test]
    fn wavefront_size_is_64() {
        assert_eq!(HIP_WAVEFRONT_SIZE, 64);
    }

    #[test]
    fn max_lds_is_64k() {
        assert_eq!(HIP_MAX_LDS_BYTES, 65_536);
    }

    // ── GcnArchTier ordering ─────────────────────────────────────────

    #[test]
    fn arch_tier_ordering() {
        assert!(GcnArchTier::Unknown < GcnArchTier::Gcn);
        assert!(GcnArchTier::Gcn < GcnArchTier::Cdna1);
        assert!(GcnArchTier::Cdna1 < GcnArchTier::Cdna2);
        assert!(GcnArchTier::Cdna2 < GcnArchTier::Cdna3);
    }

    #[test]
    fn arch_tier_eq() {
        assert_eq!(GcnArchTier::Cdna2, GcnArchTier::Cdna2);
        assert_ne!(GcnArchTier::Cdna2, GcnArchTier::Cdna3);
    }

    // ── Cross-kernel integration ─────────────────────────────────────

    #[test]
    fn matmul_then_softmax() {
        let mm = HipMatmulKernel::new(HipMatmulConfig::mock());
        let sm = HipSoftmaxKernel::new(HipSoftmaxConfig::mock());

        let a = vec![1.0, 0.0, 0.0, 1.0]; // I₂
        let b = vec![2.0, 3.0, 4.0, 5.0];
        let mut c = vec![0.0; 4];
        mm.execute(&a, &b, &mut c, 2, 2, 2).unwrap();

        let mut softmax_out = vec![0.0; 4];
        sm.execute(&c, &mut softmax_out, 2, 2).unwrap();
        // Each row sums to 1.
        let sum0: f32 = softmax_out[0..2].iter().sum();
        let sum1: f32 = softmax_out[2..4].iter().sum();
        assert!((sum0 - 1.0).abs() < 1e-5);
        assert!((sum1 - 1.0).abs() < 1e-5);
    }

    #[test]
    fn quant_roundtrip_then_matmul() {
        let qk = HipQuantKernel::new(HipQuantConfig::mock_with_bits(HipQuantBits::Eight));
        let mk = HipMatmulKernel::new(HipMatmulConfig::mock());

        // Quantize → dequantize a small matrix.
        let orig = vec![1.0, 2.0, 3.0, 4.0];
        let mut packed = vec![0u8; 4];
        let mut scales = vec![0.0f32; 1];
        qk.quantize(&orig, &mut packed, &mut scales).unwrap();
        let mut restored = vec![0.0f32; 4];
        qk.dequantize(&packed, &scales, &mut restored).unwrap();

        // Use dequantized matrix in matmul (should not panic).
        let b = vec![1.0; 4];
        let mut c = vec![0.0; 4];
        mk.execute(&restored, &b, &mut c, 2, 2, 2).unwrap();
        for &v in &c {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn layer_norm_then_attention() {
        let ln = HipLayerNormKernel::new(HipLayerNormConfig::mock(4));
        let att = HipAttentionKernel::new(HipAttentionKernelConfig::mock(1, 4));

        let input = vec![1.0, 2.0, 3.0, 4.0];
        let gamma = vec![1.0; 4];
        let beta = vec![0.0; 4];
        let mut normed = vec![0.0; 4];
        ln.execute(&input, &gamma, &beta, &mut normed, 1).unwrap();

        let mut attn_out = vec![0.0; 4];
        att.execute(&normed, &normed, &normed, &mut attn_out, 1).unwrap();
        for &v in &attn_out {
            assert!(v.is_finite());
        }
    }
}
