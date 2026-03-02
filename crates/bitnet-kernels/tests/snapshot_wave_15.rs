//! Wave 15 snapshot tests for `bitnet-kernels` — quantization routines,
//! dequantization outputs, loss functions, gating kernels, FFN configs,
//! linear projection configs, softmax configs, attention masks, concat/
//! split operations, and small matmul/softmax kernel outputs.
//!
//! Covers areas not pinned by waves 5, 6, 9, or 11 so that unintentional
//! changes to kernel outputs and Display/Debug representations are caught
//! at review time.

// =========================================================================
// Helpers
// =========================================================================

/// Round f32 slice to 6 decimal places for deterministic snapshots.
fn fmt6(v: &[f32]) -> String {
    let parts: Vec<String> = v.iter().map(|x| format!("{x:.6}")).collect();
    format!("[{}]", parts.join(", "))
}

// =========================================================================
// Section 1 — Quantization and dequantization outputs
// =========================================================================

use bitnet_kernels::cpu::dequant::{
    dequant_i2s_block, dequant_i2s_row, dequant_ternary, pack_ternary,
};
use bitnet_kernels::cpu::quantize::{
    compute_quantization_error, dequantize_asymmetric_u8, dequantize_symmetric_i8,
    quantize_asymmetric_u8, quantize_binary, quantize_symmetric_i8, quantize_ternary,
};

#[test]
fn quantize_symmetric_i8_known_input() {
    let input = [1.0_f32, -0.5, 0.0, 0.75, -1.0];
    let (quantized, scale) = quantize_symmetric_i8(&input, 8);
    insta::assert_snapshot!(format!("quantized={quantized:?} scale={scale:.6}"));
}

#[test]
fn quantize_symmetric_i8_all_zero() {
    let input = [0.0_f32; 4];
    let (quantized, scale) = quantize_symmetric_i8(&input, 8);
    insta::assert_snapshot!(format!("quantized={quantized:?} scale={scale:.6}"));
}

#[test]
fn dequantize_symmetric_i8_round_trip() {
    let input = [1.0_f32, -0.5, 0.0, 0.75, -1.0];
    let (quantized, scale) = quantize_symmetric_i8(&input, 8);
    let restored = dequantize_symmetric_i8(&quantized, scale);
    insta::assert_snapshot!(fmt6(&restored));
}

#[test]
fn quantize_asymmetric_u8_known_input() {
    let input = [0.0_f32, 0.25, 0.5, 0.75, 1.0];
    let (quantized, scale, zero_point) = quantize_asymmetric_u8(&input);
    insta::assert_snapshot!(format!("quantized={quantized:?} scale={scale:.6} zp={zero_point}"));
}

#[test]
fn dequantize_asymmetric_u8_round_trip() {
    let input = [0.0_f32, 0.25, 0.5, 0.75, 1.0];
    let (quantized, scale, zp) = quantize_asymmetric_u8(&input);
    let restored = dequantize_asymmetric_u8(&quantized, scale, zp);
    insta::assert_snapshot!(fmt6(&restored));
}

#[test]
fn quantize_ternary_known_input() {
    let input = [0.9_f32, -0.8, 0.1, -0.05, 0.6, -1.2];
    let result = quantize_ternary(&input, 0.3);
    insta::assert_snapshot!(format!("{result:?}"));
}

#[test]
fn quantize_binary_known_input() {
    let input = [0.5_f32, -0.3, 0.0, 1.0, -2.0];
    let result = quantize_binary(&input);
    insta::assert_snapshot!(format!("{result:?}"));
}

#[test]
fn quantization_error_metrics() {
    let original = [1.0_f32, 2.0, 3.0, 4.0];
    let quantized = [1.1_f32, 1.9, 3.2, 3.8];
    let err = compute_quantization_error(&original, &quantized);
    insta::assert_debug_snapshot!(err);
}

#[test]
fn dequant_i2s_block_known() {
    // Pack: 4 values per byte. 0b01=+1, 0b11=-1, 0b00=0
    // byte = 0b11_00_01_01 → [+1, +1, 0, -1]
    let packed: &[u8] = &[0b11_00_01_01];
    let result = dequant_i2s_block(packed, 2.0, 4).unwrap();
    insta::assert_snapshot!(fmt6(&result));
}

#[test]
fn dequant_ternary_known() {
    let packed: &[u8] = &[0b11_00_01_01, 0b00_01_11_00];
    let result = dequant_ternary(packed, 1.5);
    insta::assert_snapshot!(fmt6(&result));
}

#[test]
fn pack_ternary_round_trip() {
    let values = [1.0_f32, -0.8, 0.05, 0.6, -1.2, 0.0, 0.9, -0.4];
    let (packed, scale) = pack_ternary(&values, 0.2);
    let restored = dequant_ternary(&packed, scale);
    insta::assert_snapshot!(format!("scale={scale:.6} restored={}", fmt6(&restored)));
}

#[test]
fn dequant_i2s_row_multi_block() {
    let packed: &[u8] = &[0b01_01_01_01, 0b11_11_11_11];
    let scales = &[1.0_f32, 2.0];
    let result = dequant_i2s_row(packed, scales, 4).unwrap();
    insta::assert_snapshot!(fmt6(&result));
}

// =========================================================================
// Section 2 — Loss function outputs
// =========================================================================

use bitnet_kernels::cpu::loss::{
    LossReduction, binary_cross_entropy, contrastive_loss, cosine_similarity_loss,
    cross_entropy_loss, kl_divergence, l1_loss, mse_loss, smooth_l1_loss,
};

#[test]
fn cross_entropy_loss_known() {
    let logits = [2.0_f32, 1.0, 0.1, 0.1, 1.0, 2.0];
    let targets = [0_usize, 2];
    let (loss, per_sample) = cross_entropy_loss(&logits, &targets, 3, LossReduction::Mean).unwrap();
    insta::assert_snapshot!(format!("loss={loss:.6} per_sample={}", fmt6(&per_sample)));
}

#[test]
fn mse_loss_known() {
    let preds = [1.0_f32, 2.0, 3.0];
    let targets = [1.1, 2.2, 2.8];
    let loss = mse_loss(&preds, &targets, LossReduction::Mean).unwrap();
    insta::assert_snapshot!(format!("{loss:.6}"));
}

#[test]
fn l1_loss_known() {
    let preds = [1.0_f32, 2.0, 3.0];
    let targets = [1.5, 1.5, 3.5];
    let loss = l1_loss(&preds, &targets, LossReduction::Sum).unwrap();
    insta::assert_snapshot!(format!("{loss:.6}"));
}

#[test]
fn smooth_l1_loss_known() {
    let preds = [1.0_f32, 2.0, 3.0];
    let targets = [1.5, 1.5, 3.5];
    let loss = smooth_l1_loss(&preds, &targets, 1.0, LossReduction::Mean).unwrap();
    insta::assert_snapshot!(format!("{loss:.6}"));
}

#[test]
fn cosine_similarity_loss_known() {
    let a = [1.0_f32, 0.0, 1.0];
    let b = [0.0, 1.0, 1.0];
    let loss = cosine_similarity_loss(&a, &b).unwrap();
    insta::assert_snapshot!(format!("{loss:.6}"));
}

#[test]
fn contrastive_loss_similar_pair() {
    let a = [1.0_f32, 2.0, 3.0];
    let b = [1.1, 2.1, 3.1];
    let loss = contrastive_loss(&a, &b, 1.0, 1.0).unwrap();
    insta::assert_snapshot!(format!("{loss:.6}"));
}

// =========================================================================
// Section 3 — Gating kernel outputs
// =========================================================================

use bitnet_kernels::cpu::gating::{GatingType, apply_gating, geglu, reglu, swiglu};

#[test]
fn gating_type_debug_all_variants() {
    let variants = [GatingType::SwiGLU, GatingType::GeGLU, GatingType::ReGLU];
    insta::assert_debug_snapshot!(variants);
}

#[test]
fn swiglu_known_output() {
    let gate = [1.0_f32, -1.0, 0.5, 2.0];
    let up = [1.0, 1.0, 1.0, 1.0];
    let mut output = vec![0.0; 4];
    swiglu(&gate, &up, &mut output).unwrap();
    insta::assert_snapshot!(fmt6(&output));
}

#[test]
fn geglu_known_output() {
    let gate = [1.0_f32, -1.0, 0.5, 2.0];
    let up = [2.0, 2.0, 2.0, 2.0];
    let mut output = vec![0.0; 4];
    geglu(&gate, &up, &mut output).unwrap();
    insta::assert_snapshot!(fmt6(&output));
}

#[test]
fn reglu_known_output() {
    let gate = [1.0_f32, -1.0, 0.5, 2.0];
    let up = [3.0, 3.0, 3.0, 3.0];
    let mut output = vec![0.0; 4];
    reglu(&gate, &up, &mut output).unwrap();
    insta::assert_snapshot!(fmt6(&output));
}

#[test]
fn apply_gating_swiglu() {
    let gate = [0.5_f32, -0.5];
    let up = [1.0, 1.0];
    let mut output = vec![0.0; 2];
    apply_gating(GatingType::SwiGLU, &gate, &up, &mut output).unwrap();
    insta::assert_snapshot!(fmt6(&output));
}

// =========================================================================
// Section 4 — FFN configuration Debug snapshots
// =========================================================================

use bitnet_kernels::cpu::ffn::{FfnActivation, FfnConfig};

#[test]
fn ffn_config_gelu_debug() {
    let cfg = FfnConfig::new(768, 3072, FfnActivation::GeLU).unwrap();
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn ffn_config_silu_debug() {
    let cfg = FfnConfig::new(512, 2048, FfnActivation::SiLU).unwrap();
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn ffn_activation_all_variants_debug() {
    let variants = [FfnActivation::GeLU, FfnActivation::SiLU, FfnActivation::ReLU];
    insta::assert_debug_snapshot!(variants);
}

// =========================================================================
// Section 5 — Linear config Debug + grid/block dims
// =========================================================================

use bitnet_kernels::cpu::linear::LinearConfig;

#[test]
fn linear_config_default_debug() {
    let cfg = LinearConfig::default();
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn linear_config_custom_debug() {
    let cfg = LinearConfig::new(4, 768, 3072).unwrap().with_bias(true);
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn linear_config_grid_block_dims() {
    let cfg = LinearConfig::new(8, 512, 256).unwrap();
    let grid = cfg.grid_dim();
    let block = cfg.block_dim();
    insta::assert_snapshot!(format!("grid={grid:?} block={block:?}"));
}

// =========================================================================
// Section 6 — SoftmaxConfig Debug + small output
// =========================================================================

use bitnet_kernels::cuda::softmax::{SoftmaxConfig, SoftmaxMode, softmax_cpu};

#[test]
fn softmax_config_basic_debug() {
    let cfg = SoftmaxConfig::for_shape(8, 2).unwrap();
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn softmax_config_with_temperature_debug() {
    let cfg = SoftmaxConfig::for_shape(4, 1).unwrap().with_temperature(0.5).unwrap();
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn softmax_config_with_causal_mask_debug() {
    let cfg = SoftmaxConfig::for_shape(4, 4).unwrap().with_causal_mask();
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn softmax_mode_all_variants_debug() {
    let variants = [SoftmaxMode::Standard, SoftmaxMode::LogSoftmax];
    insta::assert_debug_snapshot!(variants);
}

#[test]
fn softmax_cpu_small_vector() {
    let input = [1.0_f32, 2.0, 3.0, 4.0];
    let mut output = vec![0.0; 4];
    let cfg = SoftmaxConfig::for_shape(4, 1).unwrap();
    softmax_cpu(&input, &mut output, &cfg).unwrap();
    insta::assert_snapshot!(fmt6(&output));
}

// =========================================================================
// Section 7 — Attention mask outputs
// =========================================================================

use bitnet_kernels::cpu::attention_mask::{
    combine_masks, create_causal_mask, create_padding_mask, create_sliding_window_mask,
};

#[test]
fn causal_mask_3x3() {
    let mask = create_causal_mask(3);
    let display: Vec<String> = mask
        .iter()
        .map(|v| if v.is_infinite() { "-inf".into() } else { format!("{v:.0}") })
        .collect();
    insta::assert_snapshot!(format!("{display:?}"));
}

#[test]
fn padding_mask_known() {
    let mask = create_padding_mask(&[2, 4], 5);
    let display: Vec<String> = mask
        .iter()
        .map(|v| if v.is_infinite() { "-inf".into() } else { format!("{v:.0}") })
        .collect();
    insta::assert_snapshot!(format!("{display:?}"));
}

#[test]
fn sliding_window_mask_4x4_window2() {
    let mask = create_sliding_window_mask(4, 2);
    let display: Vec<String> = mask
        .iter()
        .map(|v| if v.is_infinite() { "-inf".into() } else { format!("{v:.0}") })
        .collect();
    insta::assert_snapshot!(format!("{display:?}"));
}

#[test]
fn combine_masks_known() {
    let a = create_causal_mask(3);
    let b = create_causal_mask(3);
    let combined = combine_masks(&a, &b, 3);
    let display: Vec<String> = combined
        .iter()
        .map(|v| if v.is_infinite() { "-inf".into() } else { format!("{v:.0}") })
        .collect();
    insta::assert_snapshot!(format!("{display:?}"));
}

// =========================================================================
// Section 8 — Concat/split kernel outputs
// =========================================================================

use bitnet_kernels::cpu::concat::ConcatKernel;

#[test]
fn concat_two_vectors() {
    let a = [1.0_f32, 2.0, 3.0];
    let b = [4.0, 5.0, 6.0];
    let result = ConcatKernel::concat(&[&a, &b], &[&[3], &[3]], 0).unwrap();
    insta::assert_snapshot!(fmt6(&result));
}

#[test]
fn concat_output_shape_axis0() {
    let shape = ConcatKernel::concat_output_shape(&[&[2, 3], &[4, 3]], 0).unwrap();
    insta::assert_debug_snapshot!(shape);
}

#[test]
fn split_even() {
    let input = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let parts = ConcatKernel::split(&input, &[6], 0, 3).unwrap();
    let formatted: Vec<String> = parts.iter().map(|p| fmt6(p)).collect();
    insta::assert_snapshot!(format!("{formatted:?}"));
}

#[test]
fn stack_two_vectors() {
    let a = [1.0_f32, 2.0];
    let b = [3.0, 4.0];
    let result = ConcatKernel::stack(&[&a, &b], &[2], 0).unwrap();
    insta::assert_snapshot!(fmt6(&result));
}

#[test]
fn stack_output_shape() {
    let shape = ConcatKernel::stack_output_shape(&[3, 4], 0, 2).unwrap();
    insta::assert_debug_snapshot!(shape);
}

// =========================================================================
// Section 9 — Small matmul kernel output
// =========================================================================

use bitnet_kernels::cpu::simd_matmul::{SimdMatmulConfig, TileConfig, simd_matmul_f32};

#[test]
fn simd_matmul_2x2() {
    // A = [[1,2],[3,4]], B = [[5,6],[7,8]]
    let a = [1.0_f32, 2.0, 3.0, 4.0];
    let b = [5.0, 6.0, 7.0, 8.0];
    let mut c = vec![0.0; 4];
    let cfg = SimdMatmulConfig::new(2, 2, 2);
    simd_matmul_f32(&a, &b, &mut c, &cfg).unwrap();
    insta::assert_snapshot!(fmt6(&c));
}

#[test]
fn simd_matmul_with_alpha_beta() {
    let a = [1.0_f32, 0.0, 0.0, 1.0];
    let b = [2.0, 3.0, 4.0, 5.0];
    let mut c = [10.0_f32, 10.0, 10.0, 10.0];
    let cfg = SimdMatmulConfig {
        m: 2,
        n: 2,
        k: 2,
        alpha: 2.0,
        beta: 0.5,
        transpose_a: false,
        transpose_b: false,
    };
    simd_matmul_f32(&a, &b, &mut c, &cfg).unwrap();
    insta::assert_snapshot!(fmt6(&c));
}

#[test]
fn tile_config_debug() {
    let cfg = TileConfig::new(8, 8, 4);
    insta::assert_debug_snapshot!(cfg);
}

// =========================================================================
// Section 10 — Reduction kernel outputs (axis variants)
// =========================================================================

use bitnet_kernels::cpu::reduction::{ReductionAxis, ReductionKernel};

#[test]
fn reduction_product_known() {
    let data = [2.0_f32, 3.0, 4.0];
    let result = ReductionKernel::product(&data).unwrap();
    insta::assert_snapshot!(format!("{result:.6}"));
}

#[test]
fn reduction_l1_norm_known() {
    let data = [1.0_f32, -2.0, 3.0, -4.0];
    let result = ReductionKernel::l1_norm(&data).unwrap();
    insta::assert_snapshot!(format!("{result:.6}"));
}

#[test]
fn reduction_product_axis_2x3() {
    let data = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let result = ReductionKernel::product_axis(&data, 2, 3, ReductionAxis::Row).unwrap();
    insta::assert_snapshot!(fmt6(&result));
}

#[test]
fn reduction_l1_norm_axis_2x3() {
    let data = [1.0_f32, -2.0, 3.0, -4.0, 5.0, -6.0];
    let result = ReductionKernel::l1_norm_axis(&data, 2, 3, ReductionAxis::Row).unwrap();
    insta::assert_snapshot!(fmt6(&result));
}

#[test]
fn reduction_l2_norm_axis_2x3() {
    let data = [3.0_f32, 4.0, 0.0, 5.0, 12.0, 0.0];
    let result = ReductionKernel::l2_norm_axis(&data, 2, 3, ReductionAxis::Row).unwrap();
    insta::assert_snapshot!(fmt6(&result));
}

// =========================================================================
// Section 11 — Batched operations
// =========================================================================

use bitnet_kernels::cpu::batch::{batched_add, batched_softmax};

#[test]
fn batched_softmax_2x3() {
    let input = [1.0_f32, 2.0, 3.0, 3.0, 2.0, 1.0];
    let result = batched_softmax(&input, 2, 3).unwrap();
    insta::assert_snapshot!(fmt6(&result));
}

#[test]
fn batched_add_known() {
    let a = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0];
    let result = batched_add(&a, &b, 2, 3).unwrap();
    insta::assert_snapshot!(fmt6(&result));
}

// =========================================================================
// Section 12 — ScatterGatherKernel config
// =========================================================================

use bitnet_kernels::scatter_gather::{GatherConfig, ScatterGatherKernel, ScatterMode};

#[test]
fn scatter_mode_all_variants_debug() {
    let modes = [ScatterMode::Assign, ScatterMode::Add, ScatterMode::Max, ScatterMode::Min];
    insta::assert_debug_snapshot!(modes);
}

#[test]
fn scatter_mode_identity_values() {
    let vals: Vec<(String, f32)> =
        [ScatterMode::Assign, ScatterMode::Add, ScatterMode::Max, ScatterMode::Min]
            .iter()
            .map(|m| (format!("{m:?}"), m.identity()))
            .collect();
    insta::assert_debug_snapshot!(vals);
}

#[test]
fn gather_config_debug() {
    let cfg = GatherConfig::new(0, (4, 2), true).unwrap();
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn scatter_gather_kernel_grid_block() {
    let kernel = ScatterGatherKernel::new(64, 128).unwrap();
    let grid = kernel.grid_dim(8192);
    let block = kernel.block_dim();
    insta::assert_snapshot!(format!("grid={grid:?} block={block:?}"));
}

// =========================================================================
// Section 13 — PoolType exhaustive debug
// =========================================================================

use bitnet_kernels::cpu::pooling::PoolType;

#[test]
fn pool_type_all_variants_debug() {
    let types = [
        PoolType::Max,
        PoolType::Average,
        PoolType::GlobalMax,
        PoolType::GlobalAverage,
        PoolType::AvgPoolCountIncludePad,
    ];
    insta::assert_debug_snapshot!(types);
}

// =========================================================================
// Section 14 — KL divergence and binary cross-entropy loss
// =========================================================================

#[test]
fn kl_divergence_known() {
    // log_probs (already log-space) and target distribution
    let log_probs = [-0.5_f32, -1.0, -2.0];
    let targets = [0.5, 0.3, 0.2];
    let loss = kl_divergence(&log_probs, &targets, LossReduction::Sum).unwrap();
    insta::assert_snapshot!(format!("{loss:.6}"));
}

#[test]
fn binary_cross_entropy_known() {
    let preds = [0.9_f32, 0.1, 0.8];
    let targets = [1.0, 0.0, 1.0];
    let loss = binary_cross_entropy(&preds, &targets, LossReduction::Mean).unwrap();
    insta::assert_snapshot!(format!("{loss:.6}"));
}

// =========================================================================
// Section 15 — CapabilityMatrix types
// =========================================================================

use bitnet_kernels::capability_matrix::{
    CapabilityEntry, CapabilityQuery, CompatibilityReport, DeviceCapabilityMatrix, DeviceClass,
    OperationCategory, PrecisionSupport, SupportLevel,
};

#[test]
fn capability_entry_full_support_debug() {
    let entry = CapabilityEntry {
        operation: OperationCategory::MatrixOps,
        precision: PrecisionSupport::FP32,
        support: SupportLevel::Full(1.0),
    };
    insta::assert_debug_snapshot!(entry);
}

#[test]
fn capability_query_supports_matrixops() {
    let matrix = DeviceCapabilityMatrix::with_builtin_profiles();
    if let Some(profile) = matrix.profile_for_class(DeviceClass::CpuSimd) {
        let query = CapabilityQuery::new(profile);
        let supports = query.supports(OperationCategory::MatrixOps, PrecisionSupport::FP32);
        insta::assert_snapshot!(format!("supports_matrixops_fp32={supports}"));
    } else {
        insta::assert_snapshot!("no_cpusimd_profile");
    }
}

#[test]
fn compatibility_report_summary_snapshot() {
    let matrix = DeviceCapabilityMatrix::with_builtin_profiles();
    if let Some(profile) = matrix.profile_for_class(DeviceClass::CpuSimd) {
        let required = vec![
            (OperationCategory::MatrixOps, PrecisionSupport::FP32),
            (OperationCategory::NormOps, PrecisionSupport::FP32),
        ];
        let report = CompatibilityReport::generate(profile, &required);
        insta::assert_snapshot!(report.summary());
    } else {
        insta::assert_snapshot!("no_cpusimd_profile");
    }
}
