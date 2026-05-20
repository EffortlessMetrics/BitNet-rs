//! SRP-focused tensor loading paths for the GGUF loader.
//!
//! `loader.rs` owns GGUF file orchestration.  This module owns tensor-materialization
//! concerns: dtype routing, quantized tensor adapters, dense tensor adapters, and
//! post-load correction policy application.

use super::*;
use crate::formats::gguf::TensorInfo;
use crate::qk256_utils::{detect_qk256_orientation_by_bytes, expected_qk256_shape};
use crate::quant::i2s::{self, I2SDequantCfg};
use bitnet_common::ModelError;

impl GgufLoader {
    /// Create a Candle tensor from GGUF tensor info, optionally applying policy-driven corrections.
    /// Returns (tensor, raw_qk256_tensor_opt, correction_record_opt).
    pub(super) fn create_candle_tensor_with_policy(
        &self,
        info: &TensorInfo,
        data: &[u8],
        device: &Device,
        model_config: &BitNetConfig,
        policy_plan: Option<&crate::correction_policy::CorrectionPlan>,
        norm_validation_policy: NormValidationPolicy,
    ) -> TensorLoadResult {
        let dtype = tensor_dtype(info.tensor_type);
        let candle_device = Self::device_to_candle(device)?;

        if info.tensor_type.is_quantized() {
            load_quantized_tensor(
                info,
                data,
                dtype,
                &candle_device,
                model_config,
                policy_plan,
                norm_validation_policy,
            )
        } else {
            load_dense_tensor(
                info,
                data,
                dtype,
                &candle_device,
                policy_plan,
                norm_validation_policy,
            )
        }
    }
}

fn tensor_dtype(tensor_type: GgufTensorType) -> DType {
    match tensor_type {
        GgufTensorType::F32 => DType::F32,
        GgufTensorType::F16 => DType::F16,
        GgufTensorType::F64 => DType::F64,
        GgufTensorType::Q4_0
        | GgufTensorType::Q4_1
        | GgufTensorType::Q5_0
        | GgufTensorType::Q5_1
        | GgufTensorType::Q8_0
        | GgufTensorType::Q8_1
        | GgufTensorType::Q2_K
        | GgufTensorType::Q3_K
        | GgufTensorType::Q4_K
        | GgufTensorType::Q5_K
        | GgufTensorType::Q6_K
        | GgufTensorType::Q8_K
        | GgufTensorType::IQ2_S
        | GgufTensorType::I2_S => DType::U8,
    }
}

fn load_quantized_tensor(
    info: &TensorInfo,
    data: &[u8],
    dtype: DType,
    candle_device: &candle_core::Device,
    model_config: &BitNetConfig,
    policy_plan: Option<&crate::correction_policy::CorrectionPlan>,
    norm_validation_policy: NormValidationPolicy,
) -> TensorLoadResult {
    if matches!(info.tensor_type, GgufTensorType::IQ2_S) {
        return load_iq2s_tensor(info, data, candle_device);
    }

    if matches!(info.tensor_type, GgufTensorType::I2_S) {
        return load_i2s_tensor(info, data, candle_device, model_config, policy_plan);
    }

    if is_supported_dense_standard_quant(info.tensor_type) {
        return load_dense_standard_quant_tensor(info, data, candle_device, norm_validation_policy);
    }

    // Other standard GGUF quantized types stay fail-closed in strict mode until a dedicated
    // adapter/dequantizer is added.
    let tensor = Tensor::from_raw_buffer(data, dtype, &info.shape, candle_device)
        .map_err(|e| BitNetError::Validation(e.to_string()))?;
    Ok((tensor, Vec::new(), None))
}

#[cfg(feature = "iq2s-ffi")]
fn load_iq2s_tensor(
    info: &TensorInfo,
    data: &[u8],
    candle_device: &candle_core::Device,
) -> TensorLoadResult {
    use crate::quant::iq2s;

    let f32_data = iq2s::dequantize_to_f32(data, &info.shape)
        .map_err(|e| BitNetError::Validation(e.to_string()))?;
    let tensor = Tensor::from_slice(&f32_data, info.shape.as_slice(), candle_device)
        .map_err(|e| BitNetError::Validation(e.to_string()))?;
    Ok((tensor, Vec::new(), None))
}

#[cfg(not(feature = "iq2s-ffi"))]
fn load_iq2s_tensor(
    info: &TensorInfo,
    _data: &[u8],
    _candle_device: &candle_core::Device,
) -> TensorLoadResult {
    Err(BitNetError::Model(ModelError::InvalidFormat {
        format: format!(
            "IQ2_S tensor '{}' found but support not compiled in. \
            Rebuild with `--features iq2s-ffi` to enable IQ2_S support.",
            info.name
        ),
    }))
}

fn load_i2s_tensor(
    info: &TensorInfo,
    data: &[u8],
    candle_device: &candle_core::Device,
    model_config: &BitNetConfig,
    policy_plan: Option<&crate::correction_policy::CorrectionPlan>,
) -> TensorLoadResult {
    use crate::formats::gguf::types::{I2SFlavor, detect_i2s_flavor};

    validate_i2s_can_be_quantized(info)?;

    let nelems = info.shape.iter().product::<usize>();
    let has_scale_sibling = false;
    let flavor = detect_i2s_flavor(info, has_scale_sibling, nelems)?;

    if matches!(flavor, I2SFlavor::GgmlQk256NoScale) {
        return load_qk256_tensor(info, data, candle_device, model_config);
    }

    let logical_size = flavor.logical_size_bytes(nelems);
    if data.len() < logical_size {
        return Err(BitNetError::Validation(format!(
            "I2_S '{}': available bytes {} shorter than logical {} for {:?}",
            info.name,
            data.len(),
            logical_size,
            flavor
        )));
    }
    let logical_data = &data[..logical_size];
    if data.len() > logical_size {
        tracing::debug!(
            "I2_S '{}': trimming {} GGUF alignment padding bytes before decode",
            info.name,
            data.len() - logical_size
        );
    }

    load_dequantized_i2s_tensor(info, logical_data, candle_device, policy_plan)
}

fn validate_i2s_can_be_quantized(info: &TensorInfo) -> Result<()> {
    if is_layernorm_weight(&info.name) {
        return Err(BitNetError::Validation(format!(
            "LayerNorm weight '{}' should not be quantized with I2_S. \
            This indicates a corrupted GGUF file. LayerNorm weights must be FP16/FP32.",
            info.name
        )));
    }

    Ok(())
}

fn load_qk256_tensor(
    info: &TensorInfo,
    data: &[u8],
    candle_device: &candle_core::Device,
    model_config: &BitNetConfig,
) -> TensorLoadResult {
    tracing::debug!(
        "Detected QK256 tensor '{}' ({}x{}, {} bytes) - preserving raw bytes",
        info.name,
        info.shape[0],
        info.shape[1],
        data.len()
    );

    let is_transposed_embedding = GgufLoader::is_embedding_tensor(&info.name)
        && GgufLoader::embedding_is_transposed(&info.shape);
    let (rows, cols) = qk256_logical_shape(info, data, model_config, is_transposed_embedding);
    let (_raw_tensor, raw_entries, scale, row_stride_bytes) =
        GgufLoader::qk256_raw_entries(&info.name, data, rows, cols, candle_device)?;

    if is_transposed_embedding {
        let f32_data = GgufLoader::dequantize_qk256_token_embedding_rows(
            data,
            rows,
            cols,
            scale,
            row_stride_bytes,
            &info.name,
        )?;
        let tensor = Tensor::from_slice(&f32_data, &[rows, cols], candle_device)
            .map_err(|e| BitNetError::Validation(e.to_string()))?;
        Ok((tensor, raw_entries, None))
    } else {
        // Return placeholder f32 tensor for main collection; transformer consumes raw_tensors.
        let placeholder = Tensor::zeros(&[rows, cols], DType::F32, candle_device)
            .map_err(|e| BitNetError::Validation(e.to_string()))?;
        Ok((placeholder, raw_entries, None))
    }
}

fn qk256_logical_shape(
    info: &TensorInfo,
    data: &[u8],
    model_config: &BitNetConfig,
    is_transposed_embedding: bool,
) -> (usize, usize) {
    if is_transposed_embedding {
        tracing::info!(
            "QK256 token embedding '{}' uses GGML [hidden, vocab] dims {:?} -> preserving token-major raw rows [vocab, hidden]",
            info.name,
            info.shape
        );
        return (info.shape[1], info.shape[0]);
    }

    let shape_as_is = (info.shape[0], info.shape[1]);
    let shape_transposed = (info.shape[1], info.shape[0]);
    match expected_qk256_shape(&info.name, model_config) {
        Some((expected_rows, expected_cols)) if shape_as_is == (expected_rows, expected_cols) => {
            tracing::debug!(
                "QK256 '{}': using as-is [{}, {}] (matches expected)",
                info.name,
                shape_as_is.0,
                shape_as_is.1
            );
            shape_as_is
        }
        Some((expected_rows, expected_cols))
            if shape_transposed == (expected_rows, expected_cols) =>
        {
            tracing::debug!(
                "QK256 '{}': using transposed [{}, {}] (matches expected)",
                info.name,
                shape_transposed.0,
                shape_transposed.1
            );
            shape_transposed
        }
        Some((expected_rows, expected_cols)) => {
            tracing::warn!(
                "QK256 '{}': shape mismatch - expected [{}, {}], got [{}, {}] or [{}, {}]",
                info.name,
                expected_rows,
                expected_cols,
                shape_as_is.0,
                shape_as_is.1,
                shape_transposed.0,
                shape_transposed.1
            );
            detect_qk256_orientation_by_bytes(shape_as_is, shape_transposed, data.len())
        }
        None => detect_qk256_orientation_by_bytes(shape_as_is, shape_transposed, data.len()),
    }
}

fn load_dequantized_i2s_tensor(
    info: &TensorInfo,
    data: &[u8],
    candle_device: &candle_core::Device,
    policy_plan: Option<&crate::correction_policy::CorrectionPlan>,
) -> TensorLoadResult {
    let (inv, k, correction_opt) =
        GgufLoader::select_i2s_config(&info.name, Some(data), policy_plan);
    let cfg = I2SDequantCfg { inv, k };

    if GgufLoader::is_embedding_tensor(&info.name)
        && GgufLoader::embedding_is_transposed(&info.shape)
    {
        return load_transposed_embedding_i2s(info, data, candle_device, cfg, correction_opt);
    }

    if GgufLoader::is_projection_tensor(&info.name) && info.shape.len() == 2 {
        return load_projection_i2s(info, data, candle_device, cfg, inv, k, correction_opt);
    }

    load_regular_i2s(info, data, candle_device, cfg, inv, k, correction_opt)
}

fn load_transposed_embedding_i2s(
    info: &TensorInfo,
    data: &[u8],
    candle_device: &candle_core::Device,
    cfg: I2SDequantCfg,
    correction_opt: Option<CorrectionRecord>,
) -> TensorLoadResult {
    info!("Embedding appears transposed ({:?}) -> decoding transposed", info.shape);
    let f32_data = i2s::dequantize_to_f32_transposed_with_cfg(data, &info.shape, cfg)
        .map_err(|e| BitNetError::Validation(e.to_string()))?;
    let (rows, cols) = (info.shape[1], info.shape[0]);
    let tensor = Tensor::from_slice(&f32_data, &[rows, cols], candle_device)
        .map_err(|e| BitNetError::Validation(e.to_string()))?;
    Ok((tensor, Vec::new(), correction_opt))
}

fn load_projection_i2s(
    info: &TensorInfo,
    data: &[u8],
    candle_device: &candle_core::Device,
    cfg: I2SDequantCfg,
    inv: bool,
    k: f32,
    correction_opt: Option<CorrectionRecord>,
) -> TensorLoadResult {
    debug!(
        "Transposing projection tensor '{}' from {:?} to {:?}",
        info.name,
        info.shape,
        [info.shape[1], info.shape[0]]
    );
    let tensor =
        GgufLoader::create_transposed_i2s_tensor_with_cfg(data, &info.shape, candle_device, cfg)?;
    log_projection_rms(&info.name, "I2_S->F32", &tensor, tracing::Level::DEBUG, Some((inv, k)));
    Ok((tensor, Vec::new(), correction_opt))
}

fn load_regular_i2s(
    info: &TensorInfo,
    data: &[u8],
    candle_device: &candle_core::Device,
    cfg: I2SDequantCfg,
    inv: bool,
    k: f32,
    correction_opt: Option<CorrectionRecord>,
) -> TensorLoadResult {
    let mut f32_data = i2s::dequantize_to_f32_with_cfg(data, &info.shape, cfg)
        .map_err(|e| BitNetError::Validation(e.to_string()))?;
    let mut want_shape = info.shape.clone();

    if GgufLoader::maybe_transpose_to_out_in(&info.shape, &info.name) {
        f32_data = transpose_flat_f32(std::mem::take(&mut f32_data), info.shape[0], info.shape[1]);
        want_shape = vec![info.shape[1], info.shape[0]];
        tracing::debug!("pre-transposed {} to [out,in]={:?}", info.name, want_shape);
    }

    let tensor = Tensor::from_slice(&f32_data, want_shape.as_slice(), candle_device)
        .map_err(|e| BitNetError::Validation(e.to_string()))?;
    log_projection_rms(&info.name, "I2_S->F32", &tensor, tracing::Level::DEBUG, Some((inv, k)));
    Ok((tensor, Vec::new(), correction_opt))
}

fn is_supported_dense_standard_quant(tensor_type: GgufTensorType) -> bool {
    matches!(
        tensor_type,
        GgufTensorType::Q8_0 | GgufTensorType::Q5_0 | GgufTensorType::Q4_K | GgufTensorType::Q6_K
    )
}

fn load_dense_standard_quant_tensor(
    info: &TensorInfo,
    data: &[u8],
    candle_device: &candle_core::Device,
    norm_validation_policy: NormValidationPolicy,
) -> TensorLoadResult {
    if matches!(
        info.tensor_type,
        GgufTensorType::Q5_0 | GgufTensorType::Q4_K | GgufTensorType::Q6_K
    ) && !matches!(norm_validation_policy, NormValidationPolicy::DenseQwen)
    {
        return Err(BitNetError::Validation(format!(
            "standard GGUF quantization {:?} in tensor '{}' is only enabled for \
             supported dense Qwen adapters; no compatibility fallback was used.",
            info.tensor_type, info.name
        )));
    }

    if is_layernorm_weight(&info.name) {
        return Err(BitNetError::Validation(format!(
            "LayerNorm weight '{}' should not be quantized with {:?}. \
            Dense GGUF adapters require normalization weights in FP16/FP32.",
            info.name, info.tensor_type
        )));
    }

    let boundary = dense_standard_quant_load_boundary(info);
    debug!(
        tensor = %boundary.tensor_name,
        tensor_type = %boundary.tensor_type,
        source_shape = ?boundary.source_shape,
        candle_shape = ?boundary.candle_shape,
        block_size = boundary.block_size,
        element_size = boundary.element_size,
        dequantizes_before_compute = boundary.dequantizes_before_compute,
        materializes_f32_tensor = boundary.materializes_f32_tensor,
        values_transposed = boundary.values_transposed,
        shape_reshaped_without_transpose = boundary.shape_reshaped_without_transpose,
        next_safe_change = boundary.next_safe_change,
        "dense standard quant load boundary"
    );

    let f32_data = GgufLoader::dequantize_supported_dense_standard_quant_to_f32(
        data,
        &info.shape,
        &info.name,
        info.tensor_type,
    )?;
    let want_shape = dense_standard_quant_shape(info);
    let tensor = Tensor::from_slice(&f32_data, want_shape.as_slice(), candle_device)
        .map_err(|e| BitNetError::Validation(e.to_string()))?;
    log_projection_rms(
        &info.name,
        &format!("{:?}->F32", info.tensor_type),
        &tensor,
        tracing::Level::DEBUG,
        None,
    );
    Ok((tensor, Vec::new(), None))
}

fn dense_standard_quant_shape(info: &TensorInfo) -> Vec<usize> {
    if (GgufLoader::is_embedding_tensor(&info.name)
        || GgufLoader::is_output_head_tensor(&info.name))
        && GgufLoader::embedding_is_transposed(&info.shape)
    {
        info!(
            "{} uses GGML [hidden, vocab] dims {:?} -> reshaping token-major data to [vocab, hidden]",
            info.name, info.shape
        );
        vec![info.shape[1], info.shape[0]]
    } else if GgufLoader::maybe_transpose_to_out_in(&info.shape, &info.name) {
        debug!(
            "{:?} projection '{}' uses GGML [in, out] dims {:?} -> reshaping token-major data to [out, in]",
            info.tensor_type, info.name, info.shape
        );
        vec![info.shape[1], info.shape[0]]
    } else {
        info.shape.clone()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct DenseStandardQuantLoadBoundary {
    tensor_name: String,
    tensor_type: &'static str,
    source_shape: Vec<usize>,
    candle_shape: Vec<usize>,
    block_size: usize,
    element_size: usize,
    dequantizes_before_compute: bool,
    materializes_f32_tensor: bool,
    values_transposed: bool,
    shape_reshaped_without_transpose: bool,
    locality_boundary: &'static str,
    next_safe_change: &'static str,
}

fn dense_standard_quant_load_boundary(info: &TensorInfo) -> DenseStandardQuantLoadBoundary {
    let candle_shape = dense_standard_quant_shape(info);
    DenseStandardQuantLoadBoundary {
        tensor_name: info.name.clone(),
        tensor_type: match info.tensor_type {
            GgufTensorType::Q8_0 => "Q8_0",
            GgufTensorType::Q5_0 => "Q5_0",
            GgufTensorType::Q4_K => "Q4_K",
            GgufTensorType::Q6_K => "Q6_K",
            _ => "unsupported_dense_standard_quant",
        },
        source_shape: info.shape.clone(),
        candle_shape: candle_shape.clone(),
        block_size: info.tensor_type.block_size(),
        element_size: info.tensor_type.element_size(),
        dequantizes_before_compute: true,
        materializes_f32_tensor: true,
        values_transposed: false,
        shape_reshaped_without_transpose: candle_shape != info.shape,
        locality_boundary: "eager_dense_standard_quant_dequant_to_f32_before_candle_tensor",
        next_safe_change: "replace the eager Vec<f32> plus Tensor::from_slice boundary only with a behavior-preserving Q8_0 dense linear locality path that keeps generated IDs and strict receipts unchanged",
    }
}

fn load_dense_tensor(
    info: &TensorInfo,
    data: &[u8],
    dtype: DType,
    candle_device: &candle_core::Device,
    policy_plan: Option<&crate::correction_policy::CorrectionPlan>,
    norm_validation_policy: NormValidationPolicy,
) -> TensorLoadResult {
    match dtype {
        DType::F32 => {
            load_f32_tensor(info, data, candle_device, policy_plan, norm_validation_policy)
        }
        DType::F16 => {
            load_f16_tensor(info, data, candle_device, policy_plan, norm_validation_policy)
        }
        _ => Err(BitNetError::Model(ModelError::InvalidFormat {
            format: format!("Unsupported data type: {:?}", dtype),
        })),
    }
}

fn load_f32_tensor(
    info: &TensorInfo,
    data: &[u8],
    candle_device: &candle_core::Device,
    policy_plan: Option<&crate::correction_policy::CorrectionPlan>,
    norm_validation_policy: NormValidationPolicy,
) -> TensorLoadResult {
    log_layer0_attention_norm_stats(info, data);

    let tensor = if GgufLoader::is_embedding_tensor(&info.name)
        && GgufLoader::embedding_is_transposed(&info.shape)
    {
        info!(
            "Embedding uses GGML [hidden, vocab] dims {:?} -> reshaping token-major payload to [vocab, hidden]",
            info.shape
        );
        let f32_data = bytemuck::cast_slice::<u8, f32>(data);
        Tensor::from_slice(f32_data, &[info.shape[1], info.shape[0]], candle_device)
            .map_err(|e| BitNetError::Validation(e.to_string()))?
    } else if GgufLoader::maybe_transpose_to_out_in(&info.shape, &info.name) {
        debug!("pre-transposing F32 projection '{}' from {:?} to [out,in]", info.name, info.shape);
        let f32_data = GgufLoader::transpose_f32_to_f32(data, &info.shape)?;
        Tensor::from_slice(&f32_data, &[info.shape[1], info.shape[0]], candle_device)
            .map_err(|e| BitNetError::Validation(e.to_string()))?
    } else {
        let float_data = bytemuck::cast_slice::<u8, f32>(data);
        Tensor::from_slice(float_data, info.shape.as_slice(), candle_device)
            .map_err(|e| BitNetError::Validation(e.to_string()))?
    };

    finalize_dense_tensor(info, tensor, "F32", policy_plan, norm_validation_policy)
}

fn load_f16_tensor(
    info: &TensorInfo,
    data: &[u8],
    candle_device: &candle_core::Device,
    policy_plan: Option<&crate::correction_policy::CorrectionPlan>,
    norm_validation_policy: NormValidationPolicy,
) -> TensorLoadResult {
    let tensor = if GgufLoader::is_embedding_tensor(&info.name)
        && GgufLoader::embedding_is_transposed(&info.shape)
    {
        info!(
            "Embedding uses GGML [hidden, vocab] dims {:?} -> reshaping token-major payload to [vocab, hidden]",
            info.shape
        );
        let f32_data = GgufLoader::f16_values_to_f32(data, &info.shape, &info.name)?;
        Tensor::from_slice(&f32_data, &[info.shape[1], info.shape[0]], candle_device)
            .map_err(|e| BitNetError::Validation(e.to_string()))?
    } else if GgufLoader::maybe_transpose_to_out_in(&info.shape, &info.name) {
        debug!("pre-transposing F16 projection '{}' from {:?} to [out,in]", info.name, info.shape);
        let f32_data = GgufLoader::transpose_f16_to_f32(data, &info.shape)?;
        Tensor::from_slice(&f32_data, &[info.shape[1], info.shape[0]], candle_device)
            .map_err(|e| BitNetError::Validation(e.to_string()))?
    } else {
        let float_data = GgufLoader::f16_values_to_f32(data, &info.shape, &info.name)?;
        Tensor::from_slice(&float_data, info.shape.as_slice(), candle_device)
            .map_err(|e| BitNetError::Validation(e.to_string()))?
    };

    finalize_dense_tensor(info, tensor, "F16->F32", policy_plan, norm_validation_policy)
}

fn finalize_dense_tensor(
    info: &TensorInfo,
    tensor: Tensor,
    dtype_label: &str,
    policy_plan: Option<&crate::correction_policy::CorrectionPlan>,
    norm_validation_policy: NormValidationPolicy,
) -> TensorLoadResult {
    if is_layernorm_weight(&info.name)
        && !matches!(norm_validation_policy, NormValidationPolicy::DenseQwen)
    {
        if let Some(record) = GgufLoader::check_ln_gamma_stats_with_policy(
            &info.name,
            &tensor,
            norm_validation_policy,
        )? {
            return Ok((tensor, Vec::new(), Some(record)));
        }
        let (rescaled, correction1) =
            GgufLoader::maybe_rescale_ln_gamma_with_policy(&info.name, tensor, policy_plan)?;
        let (final_tensor, correction2) =
            GgufLoader::maybe_rescale_gamma_by_sqrt_hidden(&info.name, rescaled)?;
        Ok((final_tensor, Vec::new(), correction2.or(correction1)))
    } else {
        log_projection_rms(&info.name, dtype_label, &tensor, tracing::Level::INFO, None);
        Ok((tensor, Vec::new(), None))
    }
}

fn log_layer0_attention_norm_stats(info: &TensorInfo, data: &[u8]) {
    if info.name != "layers.0.attention_norm.weight" && info.name != "blk.0.attn_norm.weight" {
        return;
    }

    let float_data = bytemuck::cast_slice::<u8, f32>(data);
    if float_data.is_empty() {
        return;
    }

    let sum: f64 = float_data.iter().map(|&x| x as f64).sum();
    let mean = sum / float_data.len() as f64;
    let variance = float_data
        .iter()
        .map(|&x| {
            let diff = x as f64 - mean;
            diff * diff
        })
        .sum::<f64>()
        / float_data.len() as f64;
    let std = variance.sqrt();
    info!(
        "LayerNorm layer-0 attention_norm.weight: mean={:.6}, std={:.6} (should be ~1.0, small std)",
        mean, std
    );
}

fn transpose_flat_f32(data: Vec<f32>, rows: usize, cols: usize) -> Vec<f32> {
    let mut transposed = Vec::with_capacity(rows * cols);
    for col in 0..cols {
        for row in 0..rows {
            transposed.push(data[row * cols + col]);
        }
    }
    transposed
}

fn log_projection_rms(
    name: &str,
    dtype_label: &str,
    tensor: &Tensor,
    level: tracing::Level,
    i2s_cfg: Option<(bool, f32)>,
) {
    if !is_projection_weight(name) {
        return;
    }

    let Ok(rms) = GgufLoader::rms_f32(tensor) else {
        return;
    };

    match (level, i2s_cfg) {
        (tracing::Level::DEBUG, Some((inv, k))) => debug!(
            "PROJ load: '{}' dtype={} shape={:?} rms={:.6} (inv={} k={})",
            name,
            dtype_label,
            tensor.dims(),
            rms,
            inv,
            k
        ),
        (tracing::Level::DEBUG, None) => debug!(
            "PROJ load: '{}' dtype={} shape={:?} rms={:.6}",
            name,
            dtype_label,
            tensor.dims(),
            rms
        ),
        _ => info!(
            "PROJ load: '{}' dtype={} shape={:?} rms={:.6}",
            name,
            dtype_label,
            tensor.dims(),
            rms
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn info(name: &str, shape: &[usize]) -> TensorInfo {
        TensorInfo {
            name: name.to_string(),
            shape: shape.to_vec(),
            tensor_type: GgufTensorType::Q8_0,
            offset: 0,
            size: 0,
        }
    }

    #[test]
    fn dense_standard_quant_reshapes_projection_values_without_transpose() {
        let values = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let shape = dense_standard_quant_shape(&info("blk.0.attn_q.weight", &[2, 3]));

        assert_eq!(shape, vec![3, 2]);
        assert_eq!(values, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn dense_standard_quant_reshapes_hidden_vocab_embedding_values_without_transpose() {
        let values: Vec<f32> = (0..(2 * 32_768)).map(|value| value as f32).collect();
        let shape = dense_standard_quant_shape(&info("token_embd.weight", &[2, 32_768]));

        assert_eq!(shape, vec![32_768, 2]);
        assert_eq!(&values[..6], &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn dense_standard_quant_load_boundary_records_q8_eager_dequant_locality_target() {
        let boundary = dense_standard_quant_load_boundary(&info("blk.0.ffn_down.weight", &[2, 3]));

        assert_eq!(boundary.tensor_type, "Q8_0");
        assert_eq!(boundary.source_shape, vec![2, 3]);
        assert_eq!(boundary.candle_shape, vec![3, 2]);
        assert_eq!(boundary.block_size, 32);
        assert_eq!(boundary.element_size, 34);
        assert!(boundary.dequantizes_before_compute);
        assert!(boundary.materializes_f32_tensor);
        assert!(!boundary.values_transposed);
        assert!(boundary.shape_reshaped_without_transpose);
        assert_eq!(
            boundary.locality_boundary,
            "eager_dense_standard_quant_dequant_to_f32_before_candle_tensor"
        );
        assert!(boundary.next_safe_change.contains("Q8_0 dense linear locality path"));
    }
}
