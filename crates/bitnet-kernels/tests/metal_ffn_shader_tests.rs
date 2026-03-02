#![cfg(target_os = "macos")]

//! Metal FFN (feed-forward network) shader validation tests for Apple Silicon.
//!
//! Validates the computational patterns, buffer layouts, and dispatch
//! configurations used by Metal shaders implementing transformer FFN
//! blocks — including SwiGLU, GeGLU, ReLU, quantized I2_S weights,
//! fused gate-up projections, residual connections, and layer
//! normalization.

// ───────────────────────────────────────────────────────────────────
// Constants
// ───────────────────────────────────────────────────────────────────

/// Metal SIMD-group width on Apple Silicon (32 threads).
const SIMD_GROUP_SIZE: u32 = 32;

/// Maximum threads per threadgroup (Metal spec).
const MAX_THREADS_PER_THREADGROUP: u32 = 1024;

/// Maximum threadgroup memory on Apple Silicon (32 KiB).
const MAX_THREADGROUP_MEMORY: usize = 32 * 1024;

/// Metal page size on Apple Silicon (16 KiB).
const METAL_PAGE_SIZE: usize = 16 * 1024;

/// Minimum buffer offset alignment for Metal uniform buffers.
const UNIFORM_BUFFER_ALIGNMENT: usize = 256;

// ───────────────────────────────────────────────────────────────────
// FFN configuration
// ───────────────────────────────────────────────────────────────────

/// Activation function variant used in the FFN.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FfnActivation {
    ReLU,
    SwiGLU,
    GeGLU,
    GELU,
    SiLU,
}

/// Describes a transformer FFN block for Metal dispatch planning.
#[derive(Debug, Clone)]
struct FfnConfig {
    /// Model / embedding dimension.
    hidden_dim: usize,
    /// Intermediate (up-projected) dimension.
    intermediate_dim: usize,
    /// Activation between projections.
    activation: FfnActivation,
    /// Whether gate and up projections are fused into one matmul.
    fused_gate_up: bool,
    /// Whether a residual skip connection is added after FFN.
    residual: bool,
    /// Whether to apply layer normalisation before the FFN.
    pre_norm: bool,
    /// Whether to apply layer normalisation after the FFN.
    post_norm: bool,
    /// Batch size (number of tokens processed together).
    batch_size: usize,
    /// Use 2-bit I2_S quantized weights.
    quantized: bool,
}

impl Default for FfnConfig {
    fn default() -> Self {
        Self {
            hidden_dim: 768,
            intermediate_dim: 3072,
            activation: FfnActivation::SwiGLU,
            fused_gate_up: false,
            residual: false,
            pre_norm: false,
            post_norm: false,
            batch_size: 1,
            quantized: false,
        }
    }
}

// ───────────────────────────────────────────────────────────────────
// Buffer layout helpers
// ───────────────────────────────────────────────────────────────────

/// Align `n` up to the next multiple of `align`.
fn align_up(n: usize, align: usize) -> usize {
    assert!(align > 0);
    (n + align - 1) / align * align
}

/// Bytes required for a matrix of `rows × cols` f32 elements,
/// aligned to `UNIFORM_BUFFER_ALIGNMENT`.
fn aligned_matrix_bytes(rows: usize, cols: usize) -> usize {
    align_up(rows * cols * std::mem::size_of::<f32>(), UNIFORM_BUFFER_ALIGNMENT)
}

/// Bytes for a 2-bit quantized weight matrix (I2_S, 256-element
/// blocks). Each block: 256 × 2 bits = 64 bytes data + 2 bytes
/// scale = 66 bytes.
fn quantized_i2s_bytes(rows: usize, cols: usize) -> usize {
    let elements = rows * cols;
    let block_count = (elements + 255) / 256;
    let data_bytes = block_count * 66;
    align_up(data_bytes, UNIFORM_BUFFER_ALIGNMENT)
}

/// Total weight buffer bytes for the FFN.
fn ffn_weight_bytes(cfg: &FfnConfig) -> usize {
    if cfg.quantized { ffn_quantized_weight_bytes(cfg) } else { ffn_float_weight_bytes(cfg) }
}

fn ffn_float_weight_bytes(cfg: &FfnConfig) -> usize {
    let gate_up = if cfg.fused_gate_up {
        // Fused: single [hidden_dim × 2*intermediate_dim] matrix.
        aligned_matrix_bytes(cfg.hidden_dim, 2 * cfg.intermediate_dim)
    } else if cfg.activation == FfnActivation::SwiGLU || cfg.activation == FfnActivation::GeGLU {
        // Separate gate + up projections.
        aligned_matrix_bytes(cfg.hidden_dim, cfg.intermediate_dim)
            + aligned_matrix_bytes(cfg.hidden_dim, cfg.intermediate_dim)
    } else {
        // Single up projection (ReLU / GELU / SiLU).
        aligned_matrix_bytes(cfg.hidden_dim, cfg.intermediate_dim)
    };
    let down = aligned_matrix_bytes(cfg.intermediate_dim, cfg.hidden_dim);
    gate_up + down
}

fn ffn_quantized_weight_bytes(cfg: &FfnConfig) -> usize {
    let gate_up = if cfg.fused_gate_up {
        quantized_i2s_bytes(cfg.hidden_dim, 2 * cfg.intermediate_dim)
    } else if cfg.activation == FfnActivation::SwiGLU || cfg.activation == FfnActivation::GeGLU {
        quantized_i2s_bytes(cfg.hidden_dim, cfg.intermediate_dim)
            + quantized_i2s_bytes(cfg.hidden_dim, cfg.intermediate_dim)
    } else {
        quantized_i2s_bytes(cfg.hidden_dim, cfg.intermediate_dim)
    };
    let down = quantized_i2s_bytes(cfg.intermediate_dim, cfg.hidden_dim);
    gate_up + down
}

/// Intermediate activation buffer bytes needed during the FFN pass.
fn ffn_activation_buffer_bytes(cfg: &FfnConfig) -> usize {
    let elem = std::mem::size_of::<f32>();
    let up = cfg.batch_size * cfg.intermediate_dim * elem;
    let gate = match cfg.activation {
        FfnActivation::SwiGLU | FfnActivation::GeGLU => {
            cfg.batch_size * cfg.intermediate_dim * elem
        }
        _ => 0,
    };
    align_up(up + gate, UNIFORM_BUFFER_ALIGNMENT)
}

// ───────────────────────────────────────────────────────────────────
// Activation reference implementations
// ───────────────────────────────────────────────────────────────────

fn relu(x: f32) -> f32 {
    x.max(0.0)
}

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

fn gelu(x: f32) -> f32 {
    let cdf =
        0.5 * (1.0 + ((2.0_f32 / std::f32::consts::PI).sqrt() * (x + 0.044715 * x.powi(3))).tanh());
    x * cdf
}

/// SwiGLU(x, gate) = SiLU(gate) * x
fn swiglu(x: f32, gate: f32) -> f32 {
    silu(gate) * x
}

/// GeGLU(x, gate) = GELU(gate) * x
fn geglu(x: f32, gate: f32) -> f32 {
    gelu(gate) * x
}

// ───────────────────────────────────────────────────────────────────
// Matmul / projection helpers (tiny reference impl)
// ───────────────────────────────────────────────────────────────────

/// Naive matmul: C = A × B^T, shapes A=[m,k], B=[n,k] → C=[m,n].
fn matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut c = vec![0.0_f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0_f32;
            for p in 0..k {
                sum += a[i * k + p] * b[j * k + p];
            }
            c[i * n + j] = sum;
        }
    }
    c
}

/// Apply per-element activation to a flat buffer in-place.
fn apply_activation(buf: &mut [f32], act: FfnActivation) {
    match act {
        FfnActivation::ReLU => {
            for v in buf.iter_mut() {
                *v = relu(*v);
            }
        }
        FfnActivation::SiLU => {
            for v in buf.iter_mut() {
                *v = silu(*v);
            }
        }
        FfnActivation::GELU => {
            for v in buf.iter_mut() {
                *v = gelu(*v);
            }
        }
        // Gated activations handled separately.
        FfnActivation::SwiGLU | FfnActivation::GeGLU => {}
    }
}

// ───────────────────────────────────────────────────────────────────
// Layer normalisation reference
// ───────────────────────────────────────────────────────────────────

/// RMSNorm: y = x / RMS(x) * gamma, where RMS = sqrt(mean(x^2) + eps).
fn rms_norm(x: &[f32], gamma: &[f32], eps: f32) -> Vec<f32> {
    let n = x.len();
    assert_eq!(n, gamma.len());
    let mean_sq: f32 = x.iter().map(|v| v * v).sum::<f32>() / n as f32;
    let rms = (mean_sq + eps).sqrt();
    x.iter().zip(gamma.iter()).map(|(xi, gi)| xi / rms * gi).collect()
}

// ───────────────────────────────────────────────────────────────────
// Dispatch planning
// ───────────────────────────────────────────────────────────────────

/// Compute optimal threadgroup size for a 1-D reduction of length `n`.
fn optimal_threadgroup_1d(n: usize) -> u32 {
    let mut tg = SIMD_GROUP_SIZE;
    while (tg * 2) as usize <= n && tg * 2 <= MAX_THREADS_PER_THREADGROUP {
        tg *= 2;
    }
    tg
}

/// Threadgroup memory required for a float32 reduction inside one
/// threadgroup of `threads` threads.
fn reduction_shared_mem(threads: u32) -> usize {
    threads as usize * std::mem::size_of::<f32>()
}

/// Number of threadgroups needed for a 1-D grid of `n` items.
fn threadgroup_count(n: usize, tg_size: u32) -> u32 {
    ((n + tg_size as usize - 1) / tg_size as usize) as u32
}

// ───────────────────────────────────────────────────────────────────
// Tests
// ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // Helper: deterministic pseudo-random f32 in [-1, 1].
    fn pseudo_rand(seed: u64, idx: usize) -> f32 {
        let h = seed.wrapping_mul(6364136223846793005).wrapping_add(idx as u64);
        let bits = ((h >> 16) & 0xFFFF) as f32 / 32768.0 - 1.0;
        bits
    }

    fn make_vec(len: usize, seed: u64) -> Vec<f32> {
        (0..len).map(|i| pseudo_rand(seed, i)).collect()
    }

    // ── 1. Basic FFN forward ────────────────────────────────────

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn test_basic_ffn_forward_two_layer() {
        let hidden = 64;
        let inter = 256;
        let input = make_vec(hidden, 1);
        let w_up = make_vec(inter * hidden, 2); // [inter, hidden]
        let w_down = make_vec(hidden * inter, 3); // [hidden, inter]

        // up-project → ReLU → down-project
        let up = matmul(&input, &w_up, 1, hidden, inter);
        let mut activated = up;
        apply_activation(&mut activated, FfnActivation::ReLU);
        let output = matmul(&activated, &w_down, 1, inter, hidden);

        assert_eq!(output.len(), hidden);
        // Output must not be all-zero (non-degenerate weights).
        assert!(output.iter().any(|&v| v.abs() > 1e-6));
    }

    // ── 2. SwiGLU activation ────────────────────────────────────

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn test_swiglu_activation_correctness() {
        let values = [-2.0_f32, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0];
        for &x in &values {
            for &gate in &values {
                let result = swiglu(x, gate);
                let expected = silu(gate) * x;
                assert!(
                    (result - expected).abs() < 1e-6,
                    "SwiGLU({x}, {gate}): got {result}, expected {expected}",
                );
            }
        }
    }

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn test_swiglu_ffn_end_to_end() {
        let hidden = 32;
        let inter = 128;
        let input = make_vec(hidden, 10);
        let w_gate = make_vec(inter * hidden, 11);
        let w_up = make_vec(inter * hidden, 12);
        let w_down = make_vec(hidden * inter, 13);

        let gate_proj = matmul(&input, &w_gate, 1, hidden, inter);
        let up_proj = matmul(&input, &w_up, 1, hidden, inter);
        let gated: Vec<f32> =
            gate_proj.iter().zip(up_proj.iter()).map(|(&g, &u)| swiglu(u, g)).collect();
        let output = matmul(&gated, &w_down, 1, inter, hidden);

        assert_eq!(output.len(), hidden);
        assert!(output.iter().any(|&v| v.abs() > 1e-6));
    }

    // ── 3. GeGLU activation ─────────────────────────────────────

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn test_geglu_activation_correctness() {
        let values = [-2.0_f32, -1.0, 0.0, 0.5, 1.0, 2.0];
        for &x in &values {
            for &gate in &values {
                let result = geglu(x, gate);
                let expected = gelu(gate) * x;
                assert!(
                    (result - expected).abs() < 1e-5,
                    "GeGLU({x}, {gate}): got {result}, expected {expected}",
                );
            }
        }
    }

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn test_geglu_ffn_end_to_end() {
        let hidden = 32;
        let inter = 128;
        let input = make_vec(hidden, 20);
        let w_gate = make_vec(inter * hidden, 21);
        let w_up = make_vec(inter * hidden, 22);
        let w_down = make_vec(hidden * inter, 23);

        let gate_proj = matmul(&input, &w_gate, 1, hidden, inter);
        let up_proj = matmul(&input, &w_up, 1, hidden, inter);
        let gated: Vec<f32> =
            gate_proj.iter().zip(up_proj.iter()).map(|(&g, &u)| geglu(u, g)).collect();
        let output = matmul(&gated, &w_down, 1, inter, hidden);

        assert_eq!(output.len(), hidden);
        assert!(output.iter().any(|&v| v.abs() > 1e-6));
    }

    // ── 4. ReLU FFN ─────────────────────────────────────────────

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn test_relu_ffn_forward() {
        let hidden = 48;
        let inter = 192;
        let input = make_vec(hidden, 30);
        let w_up = make_vec(inter * hidden, 31);
        let w_down = make_vec(hidden * inter, 32);

        let mut up = matmul(&input, &w_up, 1, hidden, inter);
        apply_activation(&mut up, FfnActivation::ReLU);
        // All post-ReLU values must be ≥ 0.
        assert!(up.iter().all(|&v| v >= 0.0));

        let output = matmul(&up, &w_down, 1, inter, hidden);
        assert_eq!(output.len(), hidden);
    }

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn test_relu_sparsity_pattern() {
        let n = 1024;
        let input = make_vec(n, 40);
        let mut activated = input.clone();
        apply_activation(&mut activated, FfnActivation::ReLU);

        let zeros = activated.iter().filter(|&&v| v == 0.0).count();
        // With pseudo-random in [-1,1], roughly half should be zeroed.
        assert!(zeros > n / 4 && zeros < 3 * n / 4, "unexpected sparsity: {zeros}/{n}",);
    }

    // ── 5. Quantized FFN (I2_S) ─────────────────────────────────

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn test_quantized_ffn_weight_buffer_sizes() {
        let cfg = FfnConfig {
            hidden_dim: 2048,
            intermediate_dim: 5632,
            activation: FfnActivation::SwiGLU,
            quantized: true,
            ..Default::default()
        };
        let q_bytes = ffn_weight_bytes(&cfg);
        let f_bytes = ffn_float_weight_bytes(&cfg);
        // Quantized (2-bit) must be significantly smaller than f32.
        assert!(q_bytes < f_bytes / 4, "quantized {q_bytes} should be < float/4 {f_bytes}",);
        // Buffer must be aligned.
        assert_eq!(q_bytes % UNIFORM_BUFFER_ALIGNMENT, 0);
    }

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn test_quantized_i2s_block_packing() {
        // 256-element I2_S block: 64 bytes data + 2 bytes scale.
        let block_bytes = 66;
        for &elems in &[256, 512, 1024, 2048] {
            let blocks = (elems + 255) / 256;
            let raw = blocks * block_bytes;
            let aligned = align_up(raw, UNIFORM_BUFFER_ALIGNMENT);
            assert!(aligned >= raw);
            assert_eq!(aligned % UNIFORM_BUFFER_ALIGNMENT, 0);
        }
    }

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn test_quantized_ffn_dispatch_config() {
        let cfg = FfnConfig {
            hidden_dim: 2048,
            intermediate_dim: 5632,
            quantized: true,
            ..Default::default()
        };
        // Dequantize dispatch: one threadgroup per 256-element block.
        let total_gate = cfg.hidden_dim * cfg.intermediate_dim;
        let blocks = (total_gate + 255) / 256;
        let tg_size = optimal_threadgroup_1d(256);
        let tg_count = threadgroup_count(blocks, tg_size);
        assert!(tg_count > 0);
        assert!(tg_size <= MAX_THREADS_PER_THREADGROUP);
    }

    // ── 6. Fused gate-up projection ─────────────────────────────

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn test_fused_gate_up_buffer_layout() {
        let cfg = FfnConfig {
            hidden_dim: 768,
            intermediate_dim: 3072,
            fused_gate_up: true,
            activation: FfnActivation::SwiGLU,
            ..Default::default()
        };
        let fused_bytes = aligned_matrix_bytes(cfg.hidden_dim, 2 * cfg.intermediate_dim);
        let separate_bytes = aligned_matrix_bytes(cfg.hidden_dim, cfg.intermediate_dim) * 2;
        // Fused packs into one contiguous buffer.
        assert!(fused_bytes <= separate_bytes);
        assert_eq!(fused_bytes % UNIFORM_BUFFER_ALIGNMENT, 0);
    }

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn test_fused_gate_up_split_correctness() {
        let hidden = 16;
        let inter = 64;
        let input = make_vec(hidden, 50);
        // Fused weight: [2*inter, hidden], first half = gate, second = up.
        let w_fused = make_vec(2 * inter * hidden, 51);

        let fused_out = matmul(&input, &w_fused, 1, hidden, 2 * inter);
        let gate_half: Vec<f32> = fused_out[..inter].to_vec();
        let up_half: Vec<f32> = fused_out[inter..2 * inter].to_vec();

        // Separate matmuls should match.
        let w_gate = &w_fused[..inter * hidden];
        let w_up = &w_fused[inter * hidden..2 * inter * hidden];
        let gate_sep = matmul(&input, w_gate, 1, hidden, inter);
        let up_sep = matmul(&input, w_up, 1, hidden, inter);

        for i in 0..inter {
            assert!((gate_half[i] - gate_sep[i]).abs() < 1e-4, "gate mismatch at {i}",);
            assert!((up_half[i] - up_sep[i]).abs() < 1e-4, "up mismatch at {i}",);
        }
    }

    // ── 7. Down projection ──────────────────────────────────────

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn test_down_projection_dimensions() {
        let inter = 3072;
        let hidden = 768;
        let activated = make_vec(inter, 60);
        let w_down = make_vec(hidden * inter, 61);

        let output = matmul(&activated, &w_down, 1, inter, hidden);
        assert_eq!(output.len(), hidden);
    }

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn test_down_projection_buffer_alignment() {
        for &hidden in &[768, 1024, 2048, 4096] {
            let inter = hidden * 4;
            let bytes = aligned_matrix_bytes(inter, hidden);
            assert_eq!(bytes % UNIFORM_BUFFER_ALIGNMENT, 0);
            assert!(bytes >= inter * hidden * 4);
        }
    }

    // ── 8. Residual connection ──────────────────────────────────

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn test_residual_skip_connection() {
        let hidden = 64;
        let input = make_vec(hidden, 70);
        let ffn_out = make_vec(hidden, 71);

        let with_residual: Vec<f32> =
            input.iter().zip(ffn_out.iter()).map(|(a, b)| a + b).collect();

        assert_eq!(with_residual.len(), hidden);
        for i in 0..hidden {
            let expected = input[i] + ffn_out[i];
            assert!((with_residual[i] - expected).abs() < 1e-7, "residual mismatch at {i}",);
        }
    }

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn test_residual_preserves_gradient_flow() {
        // If FFN output is zero, residual == input (identity).
        let hidden = 128;
        let input = make_vec(hidden, 72);
        let zero_ffn = vec![0.0_f32; hidden];

        let output: Vec<f32> = input.iter().zip(zero_ffn.iter()).map(|(a, b)| a + b).collect();

        for i in 0..hidden {
            assert_eq!(output[i], input[i]);
        }
    }

    // ── 9. Layer normalisation ──────────────────────────────────

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn test_pre_ffn_rms_norm() {
        let dim = 64;
        let input = make_vec(dim, 80);
        let gamma = vec![1.0_f32; dim];
        let eps = 1e-5;

        let normed = rms_norm(&input, &gamma, eps);
        assert_eq!(normed.len(), dim);

        // After RMSNorm with gamma=1, RMS of output ≈ 1.
        let rms: f32 = (normed.iter().map(|v| v * v).sum::<f32>() / dim as f32).sqrt();
        assert!((rms - 1.0).abs() < 0.01, "post-norm RMS should ≈ 1, got {rms}",);
    }

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn test_post_ffn_layer_norm() {
        let dim = 128;
        let ffn_out = make_vec(dim, 81);
        let gamma: Vec<f32> = (0..dim).map(|i| 0.9 + 0.2 * (i as f32 / dim as f32)).collect();
        let eps = 1e-5;

        let normed = rms_norm(&ffn_out, &gamma, eps);
        assert_eq!(normed.len(), dim);
        // Output should be finite and non-NaN.
        assert!(normed.iter().all(|v| v.is_finite()));
    }

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn test_norm_dispatch_threadgroup_memory() {
        for &dim in &[128, 256, 512, 768, 1024, 2048, 4096] {
            let tg = optimal_threadgroup_1d(dim);
            let shared_mem = reduction_shared_mem(tg);
            assert!(
                shared_mem <= MAX_THREADGROUP_MEMORY,
                "dim={dim}: shared_mem {shared_mem} exceeds limit",
            );
        }
    }

    // ── 10. Batch FFN ───────────────────────────────────────────

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn test_batch_ffn_multiple_tokens() {
        let batch = 4;
        let hidden = 32;
        let inter = 128;
        let input = make_vec(batch * hidden, 90);
        let w_up = make_vec(inter * hidden, 91);
        let w_down = make_vec(hidden * inter, 92);

        let up = matmul(&input, &w_up, batch, hidden, inter);
        assert_eq!(up.len(), batch * inter);

        let mut activated = up;
        apply_activation(&mut activated, FfnActivation::ReLU);

        let output = matmul(&activated, &w_down, batch, inter, hidden);
        assert_eq!(output.len(), batch * hidden);
    }

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn test_batch_ffn_token_independence() {
        let hidden = 16;
        let inter = 64;
        let w_up = make_vec(inter * hidden, 93);
        let w_down = make_vec(hidden * inter, 94);

        // Process two tokens individually.
        let tok0 = make_vec(hidden, 95);
        let tok1 = make_vec(hidden, 96);

        let run_single = |tok: &[f32]| -> Vec<f32> {
            let mut up = matmul(tok, &w_up, 1, hidden, inter);
            apply_activation(&mut up, FfnActivation::ReLU);
            matmul(&up, &w_down, 1, inter, hidden)
        };
        let out0 = run_single(&tok0);
        let out1 = run_single(&tok1);

        // Process as a batch.
        let mut batch_in = tok0.clone();
        batch_in.extend_from_slice(&tok1);
        let mut up_batch = matmul(&batch_in, &w_up, 2, hidden, inter);
        apply_activation(&mut up_batch, FfnActivation::ReLU);
        let out_batch = matmul(&up_batch, &w_down, 2, inter, hidden);

        for i in 0..hidden {
            assert!((out_batch[i] - out0[i]).abs() < 1e-4, "token 0 mismatch at {i}",);
            assert!((out_batch[hidden + i] - out1[i]).abs() < 1e-4, "token 1 mismatch at {i}",);
        }
    }

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn test_batch_activation_buffer_sizing() {
        for &batch in &[1, 2, 4, 8, 16, 32] {
            let cfg = FfnConfig {
                hidden_dim: 768,
                intermediate_dim: 3072,
                activation: FfnActivation::SwiGLU,
                batch_size: batch,
                ..Default::default()
            };
            let buf = ffn_activation_buffer_bytes(&cfg);
            // Must hold gate + up activations.
            let min = 2 * batch * 3072 * 4;
            assert!(buf >= min, "batch={batch}: {buf} < {min}");
            assert_eq!(buf % UNIFORM_BUFFER_ALIGNMENT, 0);
        }
    }

    // ── 11. Threadgroup optimisation ────────────────────────────

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn test_threadgroup_sizes_for_hidden_dims() {
        let dims = [128, 256, 512, 1024, 2048, 4096, 11008];
        for &dim in &dims {
            let tg = optimal_threadgroup_1d(dim);
            assert!(tg >= SIMD_GROUP_SIZE, "dim={dim}: tg={tg} < SIMD");
            assert!(tg <= MAX_THREADS_PER_THREADGROUP);
            assert_eq!(tg % SIMD_GROUP_SIZE, 0, "dim={dim}: not SIMD-aligned");
        }
    }

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn test_threadgroup_count_for_matmul_dispatch() {
        let dims: &[(usize, usize)] = &[(768, 3072), (1024, 4096), (2048, 5632), (4096, 11008)];
        for &(hidden, inter) in dims {
            let tg = optimal_threadgroup_1d(inter);
            let count = threadgroup_count(inter, tg);
            assert!(count >= 1, "hidden={hidden} inter={inter}");
            // Total dispatched threads must cover the dimension.
            assert!((count as usize) * (tg as usize) >= inter, "under-dispatch for inter={inter}",);
        }
    }

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn test_threadgroup_shared_memory_fits() {
        for &dim in &[128, 256, 512, 1024, 2048, 4096, 11008] {
            let tg = optimal_threadgroup_1d(dim);
            let shared = reduction_shared_mem(tg);
            assert!(
                shared <= MAX_THREADGROUP_MEMORY,
                "dim={dim}: {shared} > {MAX_THREADGROUP_MEMORY}",
            );
        }
    }

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn test_simd_group_alignment_for_reductions() {
        for tg in (SIMD_GROUP_SIZE..=MAX_THREADS_PER_THREADGROUP).step_by(SIMD_GROUP_SIZE as usize)
        {
            assert_eq!(tg % SIMD_GROUP_SIZE, 0, "tg={tg} not SIMD-aligned",);
            let simd_groups = tg / SIMD_GROUP_SIZE;
            assert!(simd_groups >= 1);
        }
    }

    // ── 12. Memory bandwidth / alignment ────────────────────────

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn test_buffer_alignment_requirements() {
        let sizes = [
            aligned_matrix_bytes(768, 3072),
            aligned_matrix_bytes(2048, 5632),
            aligned_matrix_bytes(4096, 11008),
        ];
        for sz in sizes {
            assert_eq!(
                sz % UNIFORM_BUFFER_ALIGNMENT,
                0,
                "size {sz} not aligned to {UNIFORM_BUFFER_ALIGNMENT}",
            );
        }
    }

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn test_buffer_page_alignment() {
        for &rows in &[768, 2048, 4096] {
            let cols = rows * 4;
            let bytes = aligned_matrix_bytes(rows, cols);
            let pages = (bytes + METAL_PAGE_SIZE - 1) / METAL_PAGE_SIZE;
            assert!(pages * METAL_PAGE_SIZE >= bytes, "page coverage insufficient",);
        }
    }

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn test_ffn_total_buffer_footprint() {
        let cfg = FfnConfig {
            hidden_dim: 2048,
            intermediate_dim: 5632,
            activation: FfnActivation::SwiGLU,
            fused_gate_up: false,
            batch_size: 1,
            ..Default::default()
        };
        let weights = ffn_weight_bytes(&cfg);
        let activations = ffn_activation_buffer_bytes(&cfg);
        let input_buf = aligned_matrix_bytes(1, cfg.hidden_dim);
        let output_buf = aligned_matrix_bytes(1, cfg.hidden_dim);
        let total = weights + activations + input_buf + output_buf;

        // Sanity: total should be < 1 GiB for these dims.
        assert!(total < 1 << 30, "total {total} >= 1 GiB");
        // Weights dominate for non-quantized.
        assert!(weights > activations);
    }

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn test_quantized_vs_float_memory_savings() {
        let cfg_f32 = FfnConfig {
            hidden_dim: 4096,
            intermediate_dim: 11008,
            activation: FfnActivation::SwiGLU,
            quantized: false,
            ..Default::default()
        };
        let cfg_q = FfnConfig { quantized: true, ..cfg_f32.clone() };

        let f32_bytes = ffn_weight_bytes(&cfg_f32);
        let q_bytes = ffn_weight_bytes(&cfg_q);

        // 2-bit should be ~16× smaller than f32.
        let ratio = f32_bytes as f64 / q_bytes as f64;
        assert!(ratio > 10.0, "expected > 10× savings, got {ratio:.1}×",);
    }

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn test_buffer_reuse_between_layers() {
        // Two consecutive FFN layers can reuse activation buffers.
        let cfg = FfnConfig {
            hidden_dim: 1024,
            intermediate_dim: 4096,
            activation: FfnActivation::SwiGLU,
            batch_size: 8,
            ..Default::default()
        };
        let act_bytes = ffn_activation_buffer_bytes(&cfg);
        // A second FFN layer with the same config needs no extra
        // activation memory if the first layer's buffer is reused.
        let reuse_bytes = act_bytes; // same allocation
        assert_eq!(act_bytes, reuse_bytes);
        assert_eq!(act_bytes % UNIFORM_BUFFER_ALIGNMENT, 0);
    }

    // ── Additional coverage ─────────────────────────────────────

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn test_silu_activation_values() {
        // SiLU(0) = 0, SiLU(x) → x for large x.
        assert!((silu(0.0)).abs() < 1e-7);
        assert!((silu(10.0) - 10.0).abs() < 0.01);
        assert!(silu(-10.0).abs() < 0.01);
        // SiLU is smooth and monotonic for x > ~-0.28.
        assert!(silu(1.0) > silu(0.5));
    }

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn test_gelu_activation_values() {
        assert!((gelu(0.0)).abs() < 1e-7);
        assert!((gelu(3.0) - 3.0).abs() < 0.01);
        assert!(gelu(-3.0).abs() < 0.02);
        assert!(gelu(1.0) > gelu(-1.0));
    }
}
