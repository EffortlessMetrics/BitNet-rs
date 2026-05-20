//! GGUF format loader implementation

mod tensor_loading;

use super::{GgufReader, GgufTensorType, GgufTensors};
use crate::architecture::{DenseQwenArchitecture, classify_dense_qwen_architecture};
use crate::dense_gguf_q8_sidecar::{
    DenseGgufQ8SidecarRegistry, dense_q8_payload_candidate_tensor_from_env,
};
use crate::loader::{FormatLoader, LoadConfig, MmapFile};
use crate::names::{is_layernorm_weight, is_projection_weight};
use crate::{BitNetModel, Model};
use bitnet_common::{BitNetConfig, BitNetError, CorrectionRecord, Device, ModelMetadata, Result};
use bitnet_layer_index_core::extract_structured_layer_index_segment;
use bitnet_quantization::i2s_qk256::{
    QK256_BLOCK, QK256_PACKED_BYTES, code_to_f32, unpack_qk256_block,
};
use candle_core::{DType, Tensor};
use std::path::Path;
use tracing::{debug, info};

/// Type alias for tensor load result with optional raw tensors and correction record.
type TensorLoadResult = Result<(Tensor, Vec<(String, Tensor)>, Option<CorrectionRecord>)>;
type Qk256RawEntries = (Tensor, Vec<(String, Tensor)>, Option<f32>, usize);
type Qk256RawEntriesResult = Result<Qk256RawEntries>;
type DenseQ8SidecarLoadResult =
    Result<(GgufTensors, std::collections::HashMap<String, Tensor>, DenseGgufQ8SidecarRegistry)>;

pub(crate) const SMOLLM2_360M_CONTRACT_ID: &str = "smollm2_360m_instruct_q8_0";
pub(crate) const SMOLLM2_360M_FINGERPRINT: &str =
    "sha256-48ab3034d0dd401fbc721eb1df3217902fee7dab9078992d66431f09b7750201";
pub(crate) const SMOLLM2_360M_HIDDEN_SIZE: usize = 960;
pub(crate) const SMOLLM2_360M_LAYER_COUNT: usize = 32;
pub(crate) const SMOLLM2_360M_VOCAB_SIZE: usize = 49_152;
pub(crate) const SMOLLM2_360M_INTERMEDIATE_SIZE: usize = 2_560;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum NormValidationPolicy {
    Generic,
    BitNetPreScaled,
    DenseQwen,
    SmolLm2_360MInstructQ8,
}

/// GGUF format loader
pub struct GgufLoader;

impl GgufLoader {
    /// Helper to parse environment variables as truthy boolean values.
    /// Accepts: "1", "true", "yes", "on" (case-insensitive).
    #[inline]
    fn env_truthy(key: &str) -> bool {
        std::env::var(key)
            .map(|v| matches!(v.to_ascii_lowercase().as_str(), "1" | "true" | "yes" | "on"))
            .unwrap_or(false)
    }

    /// Compute RMS (root mean square) of a tensor in F32.
    /// RMS = sqrt(mean(x^2))
    fn rms_f32(t: &Tensor) -> Result<f32> {
        let mean_sq = t
            .sqr()
            .map_err(|e| BitNetError::Validation(e.to_string()))?
            .mean_all()
            .map_err(|e| BitNetError::Validation(e.to_string()))?
            .to_scalar::<f32>()
            .map_err(|e| BitNetError::Validation(e.to_string()))?;
        Ok(mean_sq.sqrt())
    }

    #[inline]
    fn maybe_transpose_to_out_in(shape: &[usize], name: &str) -> bool {
        // All projection weights are stored/consumed as [out,in] in our kernels.
        // GGUF frequently provides them as [in,out]. Normalize here once.
        // Use name-only gating since model dims vary across architectures.
        is_projection_weight(name) && shape.len() == 2
    }

    fn qk256_inline_scale(
        data: &[u8],
        expected_raw_bytes: usize,
        name: &str,
    ) -> Result<Option<f32>> {
        let Some(trailing) = data.len().checked_sub(expected_raw_bytes) else {
            return Ok(None);
        };
        if trailing == 0 {
            return Ok(None);
        }
        if trailing < std::mem::size_of::<f32>() {
            tracing::debug!(
                "QK256 '{}': trailing bytes too short for inline scale: {}",
                name,
                trailing
            );
            return Ok(None);
        }

        let scale = f32::from_le_bytes(
            data[expected_raw_bytes..expected_raw_bytes + std::mem::size_of::<f32>()]
                .try_into()
                .expect("slice length checked"),
        );
        if !scale.is_finite() {
            return Err(BitNetError::Validation(format!(
                "QK256 '{}': inline scale is not finite: {}",
                name, scale
            )));
        }

        Ok(Some(scale))
    }

    fn qk256_raw_entries(
        name: &str,
        data: &[u8],
        rows: usize,
        cols: usize,
        device: &candle_core::Device,
    ) -> Qk256RawEntriesResult {
        let blocks_per_row = cols.div_ceil(QK256_BLOCK);
        let row_stride_bytes = blocks_per_row.checked_mul(QK256_PACKED_BYTES).ok_or_else(|| {
            BitNetError::Validation(format!(
                "QK256 '{}': row stride overflow for cols={}",
                name, cols
            ))
        })?;
        let expected_raw_bytes = rows.checked_mul(row_stride_bytes).ok_or_else(|| {
            BitNetError::Validation(format!(
                "QK256 '{}': raw byte shape overflow for rows={} stride={}",
                name, rows, row_stride_bytes
            ))
        })?;
        if data.len() < expected_raw_bytes {
            return Err(BitNetError::Validation(format!(
                "QK256 '{}': missing raw bytes: available={}, expected={}",
                name,
                data.len(),
                expected_raw_bytes
            )));
        }

        let scale = Self::qk256_inline_scale(data, expected_raw_bytes, name)?;
        let raw_data = if data.len() > expected_raw_bytes {
            tracing::debug!(
                "QK256 '{}': preserving {} trailing bytes as inline scale/padding",
                name,
                data.len() - expected_raw_bytes
            );
            &data[..expected_raw_bytes]
        } else {
            data
        };

        let raw_tensor =
            Tensor::from_raw_buffer(raw_data, DType::U8, &[rows, row_stride_bytes], device)
                .map_err(|e| BitNetError::Validation(e.to_string()))?;

        let qk256_key = format!("{}.qk256_qs", name);
        let mut raw_entries = vec![(qk256_key.clone(), raw_tensor.clone())];
        if let Some(scale) = scale {
            let scale_key = format!("{}.qk256_scale", name);
            let scale_tensor = Tensor::from_vec(vec![scale], &[1], device)
                .map_err(|e| BitNetError::Validation(e.to_string()))?;
            raw_entries.push((scale_key.clone(), scale_tensor));
            tracing::debug!(
                "QK256 inline scale stored with key '{}' value={:.8e}",
                scale_key,
                scale
            );
        }

        tracing::debug!(
            "QK256 raw tensor stored with key '{}' [shape: {:?}]",
            qk256_key,
            raw_tensor.dims()
        );

        Ok((raw_tensor, raw_entries, scale, row_stride_bytes))
    }

    fn dequantize_qk256_token_embedding_rows(
        data: &[u8],
        rows: usize,
        cols: usize,
        scale: Option<f32>,
        row_stride_bytes: usize,
        name: &str,
    ) -> Result<Vec<f32>> {
        let expected_raw_bytes = rows.checked_mul(row_stride_bytes).ok_or_else(|| {
            BitNetError::Validation(format!(
                "QK256 embedding '{}': raw byte shape overflow for rows={} stride={}",
                name, rows, row_stride_bytes
            ))
        })?;
        if data.len() < expected_raw_bytes {
            return Err(BitNetError::Validation(format!(
                "QK256 embedding '{}': missing raw bytes: available={}, expected={}",
                name,
                data.len(),
                expected_raw_bytes
            )));
        }

        let scale = scale.unwrap_or(1.0);
        let total = rows.checked_mul(cols).ok_or_else(|| {
            BitNetError::Validation(format!(
                "QK256 embedding '{}': output shape overflow for rows={} cols={}",
                name, rows, cols
            ))
        })?;
        let mut out = vec![0f32; total];
        let mut codes = [0u8; QK256_BLOCK];

        for row in 0..rows {
            let row_start = row * row_stride_bytes;
            let row_bytes = &data[row_start..row_start + row_stride_bytes];
            let mut col = 0usize;
            for block in row_bytes.chunks_exact(QK256_PACKED_BYTES) {
                let block: &[u8; QK256_PACKED_BYTES] =
                    block.try_into().expect("QK256 block must be 64 bytes");
                unpack_qk256_block(block, &mut codes);
                let take = QK256_BLOCK.min(cols - col);
                let out_base = row * cols + col;
                for idx in 0..take {
                    out[out_base + idx] = scale * code_to_f32(codes[idx]);
                }
                col += take;
                if col >= cols {
                    break;
                }
            }
        }

        Ok(out)
    }

    /// Helper to fetch an unsigned integer by trying a list of keys
    fn get_u32_any(reader: &GgufReader, keys: &[&str]) -> Option<u32> {
        for k in keys {
            if let Some(v) = reader.get_u32_metadata(k) {
                return Some(v);
            }
            if let Some(v) = reader.get_i32_metadata(k)
                && v >= 0
            {
                return Some(v as u32);
            }
        }
        None
    }

    /// Helper to fetch a float by trying a list of keys
    fn get_f32_any(reader: &GgufReader, keys: &[&str]) -> Option<f32> {
        for k in keys {
            if let Some(v) = reader.get_f32_metadata(k) {
                return Some(v);
            }
        }
        None
    }

    /// Helper to fetch a boolean by trying a list of keys
    fn get_bool_any(reader: &GgufReader, keys: &[&str]) -> Option<bool> {
        for k in keys {
            if let Some(v) = reader.get_bool_metadata(k) {
                return Some(v);
            }
        }
        None
    }

    /// Infer hidden_size from embedding tensor shapes when metadata is missing.
    fn infer_hidden_size_from_tensors(reader: &GgufReader) -> Option<usize> {
        let emb_names = [
            // common names across llama.cpp/HF exports
            "token_embd.weight",
            "tok_embeddings.weight",
            "embed_tokens.weight",
            "model.embed_tokens.weight",
            "transformer.wte.weight",
        ];
        for n in &emb_names {
            if let Some(info) = reader.get_tensor_info_by_name(n)
                && info.shape.len() == 2
            {
                let a = info.shape[0];
                let b = info.shape[1];
                // Heuristic: vocab is big (>= 32768). Hidden is the other dim.
                let hidden = if a >= 32768 && b < a {
                    b
                } else if b >= 32768 && a < b {
                    a
                } else {
                    a.min(b)
                }; // fallback: pick the smaller
                tracing::info!("inferred hidden_size={} from {}", hidden, n);
                return Some(hidden);
            }
        }
        None
    }

    /// Infer vocab size from embedding tensor shapes when metadata stores only
    /// tokenizer arrays or omits architecture-prefixed vocab size.
    fn infer_vocab_size_from_tensors(reader: &GgufReader) -> Option<usize> {
        let emb_names = [
            "token_embd.weight",
            "tok_embeddings.weight",
            "embed_tokens.weight",
            "model.embed_tokens.weight",
            "transformer.wte.weight",
        ];
        for n in &emb_names {
            if let Some(info) = reader.get_tensor_info_by_name(n)
                && info.shape.len() == 2
            {
                let vocab = info.shape[0].max(info.shape[1]);
                if vocab >= 32768 {
                    return Some(vocab);
                }
            }
        }
        None
    }

    /// Infer intermediate_size from feed-forward tensor shapes when metadata is missing.
    fn infer_intermediate_size_from_tensors(
        reader: &GgufReader,
        hidden_size: usize,
    ) -> Option<usize> {
        let ffn_names = [
            // Common feed-forward projection tensor names
            "blk.0.ffn_gate.weight", // Microsoft BitNet style
            "layers.0.feed_forward.gate_proj.weight", // LLaMA style
            "model.layers.0.mlp.gate_proj.weight",
            "transformer.h.0.mlp.c_fc.weight",
        ];
        for n in &ffn_names {
            if let Some(info) = reader.get_tensor_info_by_name(n)
                && info.shape.len() == 2
            {
                let w_in = info.shape[0];
                let w_out = info.shape[1];
                // gate_proj should be [hidden_size, intermediate_size]
                if w_in == hidden_size {
                    tracing::info!("inferred intermediate_size={} from {}", w_out, n);
                    return Some(w_out);
                }
                // Handle transposed case [intermediate_size, hidden_size]
                if w_out == hidden_size {
                    tracing::info!("inferred intermediate_size={} from {} (transposed)", w_in, n);
                    return Some(w_in);
                }
            }
        }
        None
    }

    /// Infer number of layers from tensor names when metadata is missing or incorrect.
    fn infer_num_layers_from_tensors(reader: &GgufReader) -> Option<usize> {
        let mut max_layer = 0;
        let tensor_names = reader.tensor_names();

        for name in tensor_names {
            // Look for patterns like "blk.N." or "layers.N."
            if let Some(layer_num) = Self::extract_layer_number(name) {
                max_layer = max_layer.max(layer_num);
            }
        }

        if max_layer > 0 {
            // Layer numbers are 0-indexed, so add 1 to get total count
            Some(max_layer + 1)
        } else {
            None
        }
    }

    /// Extract layer number from tensor name patterns like "blk.N." or "layers.N."
    fn extract_layer_number(name: &str) -> Option<usize> {
        extract_structured_layer_index_segment(name)
    }

    /// Infer number of KV heads from tensor shapes (for models without explicit metadata)
    fn infer_kv_heads_from_tensors(reader: &GgufReader, config: &BitNetConfig) -> Result<usize> {
        let hidden_size = config.model.hidden_size;
        let num_heads = config.model.num_heads;

        debug!("Shape inference: hidden_size={}, num_heads={}", hidden_size, num_heads);

        if num_heads == 0 || hidden_size == 0 {
            debug!("Cannot infer GQA: missing basic dimensions");
            return Ok(num_heads); // fallback to MHA
        }

        let head_dim = hidden_size / num_heads;
        debug!("Calculated head_dim: {}", head_dim);

        // Look for k_proj tensor in first layer to infer KV head count
        let k_proj_names = [
            "blk.0.attn_k.weight",              // Microsoft BitNet style
            "layers.0.attention.k_proj.weight", // LLaMA style
            "model.layers.0.self_attn.k_proj.weight",
            "transformer.h.0.attn.k_proj.weight",
        ];

        for tensor_name in &k_proj_names {
            debug!("Checking tensor: {}", tensor_name);
            if let Some(info) = reader.get_tensor_info_by_name(tensor_name) {
                debug!("Found tensor {} with shape {:?}", tensor_name, info.shape);
                if info.shape.len() == 2 {
                    let w_in = info.shape[0];
                    let w_out = info.shape[1];
                    // Microsoft 2B: [hidden=2560, kv_out=640]
                    if w_in == hidden_size && w_out % head_dim == 0 {
                        let inferred_kv_heads = w_out / head_dim;
                        debug!("inferred_kv_heads={}, num_heads={}", inferred_kv_heads, num_heads);
                        if inferred_kv_heads != 0
                            && inferred_kv_heads <= num_heads
                            && num_heads.is_multiple_of(inferred_kv_heads)
                        {
                            info!(
                                "Inferred GQA: {} KV heads from tensor {} shape {:?}",
                                inferred_kv_heads, tensor_name, info.shape
                            );
                            return Ok(inferred_kv_heads);
                        }
                    }
                }
            } else {
                debug!("Tensor {} not found", tensor_name);
            }
        }

        // No inference possible, default to MHA
        Ok(num_heads)
    }

    /// Convert our Device to candle Device
    fn device_to_candle(device: &Device) -> Result<candle_core::Device> {
        match device {
            Device::Cpu => Ok(candle_core::Device::Cpu),
            Device::Cuda(id) => {
                #[cfg(any(feature = "gpu", feature = "cuda"))]
                {
                    use candle_core::backend::BackendDevice;
                    let cuda = candle_core::CudaDevice::new(*id)
                        .map_err(|e| BitNetError::Validation(e.to_string()))?;
                    Ok(candle_core::Device::Cuda(cuda))
                }
                #[cfg(not(any(feature = "gpu", feature = "cuda")))]
                {
                    let _ = id; // Suppress unused variable warning
                    Err(BitNetError::Validation(
                        "CUDA support not enabled; rebuild with --features gpu".to_string(),
                    ))
                }
            }
            // Compile this arm only on macOS with the 'gpu' feature.
            #[cfg(all(target_os = "macos", any(feature = "gpu", feature = "metal")))]
            Device::Metal => {
                use candle_core::backend::BackendDevice; // provides `new`
                let metal = candle_core::MetalDevice::new(0)
                    .map_err(|e| BitNetError::Validation(e.to_string()))?;
                Ok(candle_core::Device::Metal(metal))
            }
            // Everywhere else, emit a clear error without referencing Metal symbols.
            #[cfg(not(all(target_os = "macos", any(feature = "gpu", feature = "metal"))))]
            Device::Metal => Err(BitNetError::Validation(
                "Metal support not enabled; rebuild with --features metal (or gpu) on macOS"
                    .to_string(),
            )),
            Device::Hip(_) | Device::Npu => Err(BitNetError::Validation(
                "HIP/NPU devices are not yet supported for model loading".to_string(),
            )),
            Device::OpenCL(_) => Ok(candle_core::Device::Cpu), // OpenCL uses its own buffer management
        }
    }

    /// Validate LayerNorm gamma statistics to catch quantization artifacts.
    ///
    /// LayerNorm gamma RMS should be near 1.0 (acceptable envelope: [0.5, 2.0]).
    /// Family-scoped policies may accept additional narrow envelopes through
    /// `check_ln_gamma_stats_with_policy`; this generic validator stays unit-scaled.
    /// If stats are suspicious, fail in strict mode or warn otherwise.
    ///
    /// Set BITNET_STRICT_MODE=1 to fail on invalid LN gamma.
    pub(crate) fn check_ln_gamma_stats(name: &str, w: &Tensor) -> Result<()> {
        use bitnet_common::SecurityError;

        // Convert to FP32 for reliable statistics
        let w32 = w.to_dtype(DType::F32).map_err(|e| BitNetError::Validation(e.to_string()))?;
        let rms = Self::rms_f32(&w32)?;

        // Acceptable envelopes for γ RMS.
        let unit_scaled_ok = (0.5..=2.0).contains(&rms);
        let ok = rms.is_finite() && unit_scaled_ok;

        if !ok {
            let msg = format!(
                "LayerNorm gamma '{}' suspicious: rms={:.5} (expected near 1.0)",
                name, rms
            );

            // In strict mode, fail immediately
            if Self::env_truthy("BITNET_STRICT_MODE") {
                return Err(BitNetError::Security(SecurityError::MalformedData { reason: msg }));
            } else {
                tracing::info!("{} (continuing: non-strict mode)", msg);
            }
        }

        Ok(())
    }

    pub(crate) fn check_ln_gamma_stats_with_policy(
        name: &str,
        w: &Tensor,
        policy: NormValidationPolicy,
    ) -> Result<Option<CorrectionRecord>> {
        match policy {
            NormValidationPolicy::DenseQwen => Ok(None),
            NormValidationPolicy::BitNetPreScaled => {
                if Self::hidden_scaled_ln_gamma_ok(w)? {
                    return Ok(None);
                }
                Self::check_ln_gamma_stats(name, w)?;
                Ok(None)
            }
            NormValidationPolicy::SmolLm2_360MInstructQ8 => {
                if let Some(record) = Self::smollm2_norm_acceptance_record(name, w)? {
                    return Ok(Some(record));
                }
                Self::check_ln_gamma_stats(name, w)?;
                Ok(None)
            }
            NormValidationPolicy::Generic => {
                Self::check_ln_gamma_stats(name, w)?;
                Ok(None)
            }
        }
    }

    fn hidden_scaled_ln_gamma_ok(w: &Tensor) -> Result<bool> {
        let w32 = w.to_dtype(DType::F32).map_err(|e| BitNetError::Validation(e.to_string()))?;
        let rms = Self::rms_f32(&w32)?;
        Ok(rms.is_finite()
            && w32
                .dims()
                .last()
                .copied()
                .filter(|hidden| *hidden >= 512)
                .map(|hidden| {
                    let target = 1.0 / (hidden as f32).sqrt();
                    ((target * 0.5)..=(target * 1.5)).contains(&rms)
                })
                .unwrap_or(false))
    }

    fn smollm2_norm_acceptance_record(name: &str, w: &Tensor) -> Result<Option<CorrectionRecord>> {
        if !is_layernorm_weight(name) {
            return Ok(None);
        }

        let dims = w.dims();
        if dims.last().copied() != Some(SMOLLM2_360M_HIDDEN_SIZE) {
            return Ok(None);
        }

        let w32 = w.to_dtype(DType::F32).map_err(|e| BitNetError::Validation(e.to_string()))?;
        let rms = Self::rms_f32(&w32)?;
        if !Self::smollm2_norm_name_supported(name) {
            return Ok(None);
        }
        let min_rms = 0.02;
        let max_rms = 2.0;
        if !(rms.is_finite() && (min_rms..=max_rms).contains(&rms)) {
            return Ok(None);
        }

        let metadata = serde_json::json!({
            "policy": "slm_cpu_020_smollm2_exact_metadata_norm_envelope",
            "contract_id": SMOLLM2_360M_CONTRACT_ID,
            "artifact_fingerprint": SMOLLM2_360M_FINGERPRINT,
            "architecture": "llama",
            "tokenizer_pre": "smollm",
            "hidden_size": SMOLLM2_360M_HIDDEN_SIZE,
            "block_count": SMOLLM2_360M_LAYER_COUNT,
            "vocab_size": SMOLLM2_360M_VOCAB_SIZE,
            "rms_envelope": [min_rms, max_rms],
            "rms_envelope_basis": "exact SmolLM2 360M Q8_0 artifact normalization weights observed below unit-scaled generic llama; generic llama remains fail-closed",
            "source": "docs/slm/SLM_CPU_SMOLLM2_NORMALIZATION_POLICY.md",
        });

        Ok(Some(CorrectionRecord {
            layer: name.to_string(),
            correction_type: "smollm2_norm_gamma_envelope_accept".to_string(),
            rms_before: Some(rms),
            rms_after: None,
            factor: None,
            policy_fingerprint: format!("slm-cpu-020:{SMOLLM2_360M_CONTRACT_ID}"),
            metadata: Some(metadata),
        }))
    }

    fn smollm2_norm_name_supported(name: &str) -> bool {
        name.ends_with("attn_norm.weight")
            || name.ends_with("attention_norm.weight")
            || name.ends_with("ffn_norm.weight")
            || name.ends_with("post_attention_layernorm.weight")
    }

    /// Select LayerNorm rescale configuration from policy
    ///
    /// Priority order:
    /// 1. Explicit policy override from BITNET_CORRECTION_POLICY
    /// 2. Environment-based fallback (BITNET_FIX_LN_SCALE=1)
    /// 3. None (no correction)
    ///
    /// Returns: Option<(target_rms, clamp)>
    #[inline]
    fn select_ln_rescale_cfg(
        policy_plan: Option<&crate::correction_policy::CorrectionPlan>,
    ) -> Option<(f32, [f32; 2])> {
        use crate::correction_policy::CorrectionAction;

        // Step 1: Check policy override
        if let Some(plan) = policy_plan {
            for action in &plan.actions {
                if let CorrectionAction::LnGammaRescaleRms { target_rms, clamp } = action {
                    tracing::info!(
                        "POLICY: LayerNorm rescale config: target_rms={}, clamp={:?} (fingerprint={})",
                        target_rms,
                        clamp,
                        plan.fingerprint
                    );
                    return Some((*target_rms, *clamp));
                }
            }
        }

        // Step 2: Environment-based fallback
        if Self::env_truthy("BITNET_FIX_LN_SCALE") {
            tracing::info!("ENV: LayerNorm rescale enabled via BITNET_FIX_LN_SCALE=1");
            return Some((1.0, [1e-2, 1e2]));
        }

        None
    }

    /// Policy-aware LayerNorm gamma rescaling
    ///
    /// This is a temporary workaround for GGUF files with quantized LayerNorm weights.
    /// Rescales LN gamma RMS to target value (typically ~1.0).
    ///
    /// **Remove this once GGUF is regenerated with proper float LayerNorm weights.**
    ///
    /// Returns: (rescaled_tensor, optional_correction_record)
    fn maybe_rescale_ln_gamma_with_policy(
        name: &str,
        w: Tensor,
        policy_plan: Option<&crate::correction_policy::CorrectionPlan>,
    ) -> Result<(Tensor, Option<CorrectionRecord>)> {
        if !is_layernorm_weight(name) {
            return Ok((w, None));
        }

        // Never apply corrections in strict mode
        if Self::env_truthy("BITNET_STRICT_MODE") {
            return Ok((w, None));
        }

        // Check if correction is configured (policy or env)
        let cfg = Self::select_ln_rescale_cfg(policy_plan);
        if cfg.is_none() {
            return Ok((w, None));
        }

        let (target_rms, clamp) = cfg.unwrap();

        // Convert to FP32 for statistics
        let w32 = w.to_dtype(DType::F32).map_err(|e| BitNetError::Validation(e.to_string()))?;
        let rms_before = Self::rms_f32(&w32)?;

        // If already close to target, skip rescaling
        if (rms_before - target_rms).abs() < 1e-3 {
            tracing::debug!(
                "LayerNorm '{}' already close to target RMS ({:.5} ≈ {:.5}), skipping rescale",
                name,
                rms_before,
                target_rms
            );
            return Ok((w, None));
        }

        // Calculate rescale factor with clamping for safety
        let mut factor = target_rms / (rms_before + 1e-12);
        factor = factor.clamp(clamp[0], clamp[1]);

        tracing::warn!(
            "CORRECTION: rescaling '{}' gamma RMS {:.5}→{:.5} (factor {:.3}). \
             Remove when GGUF is fixed.",
            name,
            rms_before,
            target_rms,
            factor
        );

        // Apply affine transformation: x' = factor * x
        let rescaled =
            w32.affine(factor as f64, 0.0).map_err(|e| BitNetError::Validation(e.to_string()))?;

        // Calculate RMS after rescaling
        let rms_after = Self::rms_f32(&rescaled)?;

        // Convert back to original dtype
        let result =
            rescaled.to_dtype(w.dtype()).map_err(|e| BitNetError::Validation(e.to_string()))?;

        // Determine policy fingerprint source
        let policy_fp = if let Some(plan) = policy_plan {
            format!("policy:{}", plan.fingerprint)
        } else {
            "BITNET_FIX_LN_SCALE=1".to_string()
        };

        // Create correction record
        let metadata = serde_json::json!({
            "target_rms": target_rms,
            "clamp": clamp,
            "source": if policy_plan.is_some() { "policy" } else { "env" },
        });

        let correction = CorrectionRecord {
            layer: name.to_string(),
            correction_type: "ln_gamma_rescale_rms".to_string(),
            rms_before: Some(rms_before),
            rms_after: Some(rms_after),
            factor: Some(factor),
            policy_fingerprint: policy_fp,
            metadata: Some(metadata),
        };

        Ok((result, Some(correction)))
    }

    /// Legacy environment-based LayerNorm rescaling (deprecated, kept for compatibility)
    ///
    /// **Prefer `maybe_rescale_ln_gamma_with_policy` for new code.**
    #[allow(dead_code)]
    fn maybe_rescale_ln_gamma(name: &str, w: Tensor) -> Result<(Tensor, Option<CorrectionRecord>)> {
        Self::maybe_rescale_ln_gamma_with_policy(name, w, None)
    }

    /// Experimental: Rescale LayerNorm gamma by √hidden_size during loading
    ///
    /// **Hypothesis:** bitnet.cpp rescales pre-scaled gamma weights on load.
    /// This function mimics that behavior when `BITNET_RESCALE_GAMMA_ON_LOAD=1`.
    ///
    /// **Algorithm:**
    /// - Detect LayerNorm tensors (using `is_layernorm_weight`)
    /// - Calculate: `hidden_size` = last dimension
    /// - Apply: `gamma' = gamma * sqrt(hidden_size)`
    ///
    /// **Use case:** If gamma RMS ≈ 0.018 = 1/√2560, this rescales to RMS ≈ 1.0
    ///
    /// Returns: (rescaled_tensor, optional_correction_record)
    fn maybe_rescale_gamma_by_sqrt_hidden(
        name: &str,
        w: Tensor,
    ) -> Result<(Tensor, Option<CorrectionRecord>)> {
        // Only apply if enabled via environment variable
        if !Self::env_truthy("BITNET_RESCALE_GAMMA_ON_LOAD") {
            return Ok((w, None));
        }

        // Only apply to LayerNorm weights
        if !is_layernorm_weight(name) {
            return Ok((w, None));
        }

        // Never apply in strict mode
        if Self::env_truthy("BITNET_STRICT_MODE") {
            return Ok((w, None));
        }

        // Get hidden_size from last dimension
        let dims = w.dims();
        if dims.is_empty() {
            return Ok((w, None));
        }
        let hidden_size = dims[dims.len() - 1];
        let scale_factor = (hidden_size as f32).sqrt();

        // Convert to F32 for statistics
        let w32 = w.to_dtype(DType::F32).map_err(|e| BitNetError::Validation(e.to_string()))?;
        let rms_before = Self::rms_f32(&w32)?;

        // Apply rescaling: gamma' = gamma * sqrt(H)
        tracing::warn!(
            "EXPERIMENTAL: Rescaling '{}' gamma by √{} = {:.2}× (RMS {:.6} → expected {:.6})",
            name,
            hidden_size,
            scale_factor,
            rms_before,
            rms_before * scale_factor
        );

        let rescaled = w32
            .affine(scale_factor as f64, 0.0)
            .map_err(|e| BitNetError::Validation(e.to_string()))?;

        // Calculate RMS after rescaling
        let rms_after = Self::rms_f32(&rescaled)?;

        tracing::info!(
            "EXPERIMENTAL: Rescaled '{}': RMS {:.6} → {:.6} (factor: {:.2}×)",
            name,
            rms_before,
            rms_after,
            scale_factor
        );

        // Convert back to original dtype
        let result =
            rescaled.to_dtype(w.dtype()).map_err(|e| BitNetError::Validation(e.to_string()))?;

        // Create correction record
        let metadata = serde_json::json!({
            "hidden_size": hidden_size,
            "scale_factor": scale_factor,
            "source": "BITNET_RESCALE_GAMMA_ON_LOAD=1",
            "experimental": true,
        });

        let correction = CorrectionRecord {
            layer: name.to_string(),
            correction_type: "ln_gamma_rescale_sqrt_hidden".to_string(),
            rms_before: Some(rms_before),
            rms_after: Some(rms_after),
            factor: Some(scale_factor),
            policy_fingerprint: "BITNET_RESCALE_GAMMA_ON_LOAD=1".to_string(),
            metadata: Some(metadata),
        };

        Ok((result, Some(correction)))
    }

    /// Collect I2_S block scales from raw tensor data (best-effort heuristic)
    ///
    /// I2_S blocks typically start with an f16 scale. This function samples those scales
    /// to build a histogram for heuristic inversion detection.
    ///
    /// Returns None if the data doesn't match expected I2_S block layout.
    fn i2s_collect_scales(raw: &[u8], block_bytes: usize) -> Option<Vec<f32>> {
        if block_bytes == 0 || raw.len() < 2 {
            return None;
        }

        let num_blocks = raw.len() / block_bytes;
        if num_blocks == 0 {
            return None;
        }

        let mut scales = Vec::with_capacity(num_blocks);
        for block_idx in 0..num_blocks {
            let offset = block_idx * block_bytes;
            if offset + 2 > raw.len() {
                break;
            }

            // Read f16 scale (little-endian) at start of block
            let scale_bits = u16::from_le_bytes([raw[offset], raw[offset + 1]]);
            let scale = half::f16::from_bits(scale_bits).to_f32();
            scales.push(scale);
        }

        if scales.is_empty() { None } else { Some(scales) }
    }

    /// Generate histogram summary string for scale distribution
    fn scale_histogram(scales: &[f32]) -> String {
        let mut counts = [0usize; 8];
        for &scale in scales {
            let abs_scale = scale.abs();
            let bucket = match abs_scale {
                s if s < 1e-6 => 0,
                s if s < 1e-4 => 1,
                s if s < 1e-3 => 2,
                s if s < 1e-2 => 3,
                s if s < 1e-1 => 4,
                s if s < 1e0 => 5,
                s if s < 1e1 => 6,
                _ => 7,
            };
            counts[bucket] += 1;
        }

        format!(
            "<1e-6:{} <1e-4:{} <1e-3:{} <1e-2:{} <1e-1:{} <1e0:{} <1e1:{} >=1e1:{}",
            counts[0], counts[1], counts[2], counts[3], counts[4], counts[5], counts[6], counts[7]
        )
    }

    /// Check if a tensor name matches any pattern in the list
    fn tensor_matches_patterns(tensor_name: &str, patterns: &[String]) -> bool {
        patterns.iter().any(|pattern| tensor_name.ends_with(pattern))
    }

    /// Select I2_S dequantization config (inv, k) for a specific tensor
    ///
    /// Priority order:
    /// 1. Explicit policy override from BITNET_CORRECTION_POLICY
    /// 2. Heuristic detection (if BITNET_ALLOW_RUNTIME_CORRECTIONS=1)
    /// 3. Default (inv=false, k=1.0)
    ///
    /// Returns: (inv, k, Option<CorrectionRecord>)
    fn select_i2s_config(
        tensor_name: &str,
        raw_data: Option<&[u8]>,
        policy_plan: Option<&crate::correction_policy::CorrectionPlan>,
    ) -> (bool, f32, Option<CorrectionRecord>) {
        use crate::correction_policy::CorrectionAction;

        // Step 1: Check policy override
        if let Some(plan) = policy_plan {
            for action in &plan.actions {
                if let CorrectionAction::I2SDequantOverride { tensors, inv, k } = action
                    && Self::tensor_matches_patterns(tensor_name, tensors)
                {
                    tracing::warn!(
                        "POLICY: I2_S override for '{}': inv={}, k={} (fingerprint={})",
                        tensor_name,
                        inv,
                        k,
                        plan.fingerprint
                    );

                    let metadata = serde_json::json!({
                        "i2s_inv_before": false,
                        "i2s_inv_after": *inv,
                        "i2s_k_before": 1.0,
                        "i2s_k_after": *k,
                        "source": "policy",
                        "policy_fingerprint": plan.fingerprint,
                    });

                    let record = CorrectionRecord {
                        layer: tensor_name.to_string(),
                        correction_type: "i2s_dequant_override".to_string(),
                        rms_before: None,
                        rms_after: None,
                        factor: Some(*k),
                        policy_fingerprint: format!("policy:{}", plan.fingerprint),
                        metadata: Some(metadata),
                    };

                    return (*inv, *k, Some(record));
                }
            }
        }

        // Step 2: Heuristic detection (if enabled)
        if Self::env_truthy("BITNET_ALLOW_RUNTIME_CORRECTIONS")
            && let Some(data) = raw_data
        {
            // Try common I2_S block sizes (66 bytes = 256 weights + scale is most common)
            for block_size in [66usize, 82, 64] {
                if let Some(scales) = Self::i2s_collect_scales(data, block_size) {
                    if scales.is_empty() {
                        continue;
                    }

                    // Calculate percentage of tiny scales (<1e-4)
                    let tiny_count = scales.iter().filter(|s| s.abs() < 1e-4).count();
                    let tiny_fraction = tiny_count as f32 / scales.len() as f32;

                    tracing::debug!(
                        "I2_S scale analysis for '{}': {} (tiny={:.1}%)",
                        tensor_name,
                        Self::scale_histogram(&scales),
                        tiny_fraction * 100.0
                    );

                    // Heuristic: if ≥75% of scales are tiny, assume inversion
                    if tiny_fraction >= 0.75 {
                        tracing::warn!(
                            "HEURISTIC: '{}' scales look inverted ({:.0}% tiny); using inv=true",
                            tensor_name,
                            tiny_fraction * 100.0
                        );

                        let metadata = serde_json::json!({
                            "i2s_inv_before": false,
                            "i2s_inv_after": true,
                            "i2s_k_before": 1.0,
                            "i2s_k_after": 1.0,
                            "source": "heuristic",
                            "tiny_fraction": tiny_fraction,
                            "scale_histogram": Self::scale_histogram(&scales),
                        });

                        let record = CorrectionRecord {
                            layer: tensor_name.to_string(),
                            correction_type: "i2s_dequant_heuristic".to_string(),
                            rms_before: None,
                            rms_after: None,
                            factor: Some(1.0),
                            policy_fingerprint: "heuristic".to_string(),
                            metadata: Some(metadata),
                        };

                        return (true, 1.0, Some(record));
                    }

                    // Successfully analyzed scales; no need to try other block sizes
                    break;
                }
            }
        }

        // Step 3: Default (no correction)
        (false, 1.0, None)
    }
}

impl GgufLoader {}

impl FormatLoader for GgufLoader {
    fn name(&self) -> &'static str {
        "GGUF"
    }

    fn can_load(&self, path: &Path) -> bool {
        path.extension()
            .and_then(|ext| ext.to_str())
            .map(|ext| ext.to_lowercase() == "gguf")
            .unwrap_or(false)
    }

    fn detect_format(&self, path: &Path) -> Result<bool> {
        if !path.exists() {
            return Ok(false);
        }

        // Check file extension first
        if self.can_load(path) {
            return Ok(true);
        }

        // Check magic bytes
        let mmap = MmapFile::open(path)?;
        if mmap.len() < 4 {
            return Ok(false);
        }

        let magic = &mmap.as_slice()[0..4];
        Ok(magic == b"GGUF")
    }

    fn extract_metadata(&self, path: &Path) -> Result<ModelMetadata> {
        debug!("Extracting GGUF metadata from: {}", path.display());

        let mmap = MmapFile::open(path)?;
        let reader = GgufReader::new(mmap.as_slice())?;

        // Validate the file structure
        reader.validate()?;

        // Compute GGUF fingerprint for policy matching
        let fingerprint = crate::fingerprint::compute_gguf_fingerprint(mmap.as_slice());
        debug!("Model fingerprint: {}", fingerprint);

        let architecture = reader
            .get_string_metadata("general.architecture")
            .unwrap_or_else(|| "bitnet".to_string());
        let arch_prefix = architecture.clone();

        let metadata = ModelMetadata {
            name: reader.get_string_metadata("general.name").unwrap_or_else(|| {
                path.file_stem().and_then(|s| s.to_str()).unwrap_or("unknown").to_string()
            }),
            version: reader
                .get_string_metadata("general.version")
                .unwrap_or_else(|| format!("gguf-v{}", reader.version())),
            architecture,
            vocab_size: reader
                .get_u32_metadata("llama.vocab_size")
                .or_else(|| reader.get_u32_metadata(&format!("{arch_prefix}.vocab_size")))
                .or_else(|| reader.get_u32_metadata("tokenizer.ggml.tokens"))
                .map(|v| v as usize)
                .or_else(|| Self::infer_vocab_size_from_tensors(&reader))
                .unwrap_or(32000),
            context_length: reader
                .get_u32_metadata("llama.context_length")
                .or_else(|| reader.get_u32_metadata(&format!("{arch_prefix}.context_length")))
                .or_else(|| reader.get_u32_metadata("llama.rope.dimension_count"))
                .or_else(|| reader.get_u32_metadata(&format!("{arch_prefix}.rope.dimension_count")))
                .unwrap_or(2048) as usize,
            quantization: reader.get_quantization_type(),
            fingerprint: Some(fingerprint),
            corrections_applied: None, // Not available during lightweight metadata extraction
        };

        debug!("Extracted GGUF metadata: {:?}", metadata);
        Ok(metadata)
    }

    fn load(&self, path: &Path, device: &Device, config: &LoadConfig) -> Result<Box<dyn Model>> {
        info!("Loading GGUF model from: {}", path.display());

        let mmap = if config.use_mmap { Some(MmapFile::open(path)?) } else { None };

        // Keep buffer alive if not using mmap
        let mut _owned: Option<Vec<u8>> = None;
        let data: &[u8] = if let Some(ref mmap) = mmap {
            mmap.as_slice()
        } else {
            // Read entire file into memory
            let buf = std::fs::read(path).map_err(BitNetError::Io)?;
            _owned = Some(buf);
            _owned.as_ref().unwrap().as_slice()
        };

        let reader = GgufReader::new(data)?;

        // Report progress
        if let Some(callback) = &config.progress_callback {
            callback(0.3, "Parsing GGUF header...");
        }

        // Validate file structure
        reader.validate()?;

        // Compute GGUF fingerprint for policy matching
        let fingerprint = crate::fingerprint::compute_gguf_fingerprint(data);
        tracing::info!("Model fingerprint: {}", fingerprint);

        // Extract model configuration
        let model_config = self.extract_config(&reader)?;
        let norm_validation_policy =
            self.select_norm_validation_policy(&reader, &model_config, &fingerprint);
        if Self::env_truthy("BITNET_STRICT_MODE") {
            if let Some((tensor_name, tensor_type)) =
                reader.first_unsupported_standard_quantized_tensor()
            {
                return Err(BitNetError::Validation(format!(
                    "strict GGUF load rejects unsupported standard quantization {tensor_type:?} \
                     in tensor '{tensor_name}'. Supported dense adapters currently cover Q8_0 \
                     and the Qwen Q4_K_M tensor mix (Q5_0/Q4_K/Q6_K); no compatibility fallback \
                     was used."
                )));
            }
            self.validate_strict_tensor_authority(&reader, &model_config)?;
        }

        if let Some(callback) = &config.progress_callback {
            callback(0.5, "Loading tensors...");
        }

        // Load tensors with fingerprint for policy matching (returns both regular and raw QK256 tensors)
        let (tensors, raw_tensors, dense_q8_sidecars) =
            self.load_tensors(&reader, device, config, &fingerprint, norm_validation_policy)?;

        if let Some(callback) = &config.progress_callback {
            callback(0.9, "Initializing model...");
        }

        // Create model instance (pass both tensors and raw_tensors for QK256 dispatch)
        let model = BitNetModel::from_gguf_with_dense_q8_sidecars(
            model_config,
            tensors,
            raw_tensors,
            dense_q8_sidecars,
            *device,
        )?;

        Ok(Box::new(model))
    }
}

impl GgufLoader {
    fn validate_strict_tensor_authority(
        &self,
        reader: &GgufReader,
        config: &BitNetConfig,
    ) -> Result<()> {
        let tensor_names = reader.tensor_names();
        Self::validate_strict_tensor_authority_names(&tensor_names, config)
    }

    fn validate_strict_tensor_authority_names(
        tensor_names: &[&str],
        config: &BitNetConfig,
    ) -> Result<()> {
        let has_any = |candidates: &[&str]| -> bool {
            candidates.iter().any(|candidate| tensor_names.iter().any(|name| name == candidate))
        };

        let mut missing = Vec::new();

        let has_embeddings = has_any(&[
            "token_embd.weight",
            "tok_embeddings.weight",
            "embed_tokens.weight",
            "model.embed_tokens.weight",
            "transformer.wte.weight",
        ]);
        if !has_embeddings {
            missing.push("token embedding weight".to_string());
        }

        let has_output = has_any(&["output.weight", "lm_head.weight", "model.lm_head.weight"]);
        if !has_output && !has_embeddings {
            missing.push("output/lm head weight".to_string());
        } else if !has_output {
            tracing::info!(
                "Strict real_gguf load: no output/lm head tensor; using tied token embeddings"
            );
        }

        let layer_prefixes = ["blk", "layers", "model.layers", "transformer.h"];
        let required_suffix_groups: &[&[&str]] = &[
            &["attn_q.weight", "attention.q_proj.weight", "self_attn.q_proj.weight"],
            &["attn_k.weight", "attention.k_proj.weight", "self_attn.k_proj.weight"],
            &["attn_v.weight", "attention.v_proj.weight", "self_attn.v_proj.weight"],
            &["attn_output.weight", "attention.o_proj.weight", "self_attn.o_proj.weight"],
            &["ffn_gate.weight", "feed_forward.gate_proj.weight", "mlp.gate_proj.weight"],
            &["ffn_up.weight", "feed_forward.up_proj.weight", "mlp.up_proj.weight"],
            &["ffn_down.weight", "feed_forward.down_proj.weight", "mlp.down_proj.weight"],
            &["attn_norm.weight", "attention_norm.weight", "input_layernorm.weight"],
            &["ffn_norm.weight", "post_attention_layernorm.weight"],
        ];

        for layer_idx in 0..config.model.num_layers {
            for suffix_group in required_suffix_groups {
                let candidates: Vec<String> = layer_prefixes
                    .iter()
                    .flat_map(|prefix| {
                        suffix_group
                            .iter()
                            .map(move |suffix| format!("{}.{}.{}", prefix, layer_idx, suffix))
                    })
                    .collect();
                let has_group = candidates
                    .iter()
                    .any(|candidate| tensor_names.iter().any(|name| name == candidate));
                if !has_group {
                    missing.push(format!("layer {} tensor group {:?}", layer_idx, suffix_group));
                }
            }
        }

        if missing.is_empty() {
            return Ok(());
        }

        Err(BitNetError::Validation(format!(
            "Strict real_gguf load rejected unsupported/incomplete tensor layout: missing {}",
            missing.join(", ")
        )))
    }

    pub(crate) fn select_norm_validation_policy(
        &self,
        reader: &GgufReader,
        config: &BitNetConfig,
        fingerprint: &str,
    ) -> NormValidationPolicy {
        if reader
            .get_string_metadata("general.architecture")
            .map(|architecture| {
                matches!(
                    classify_dense_qwen_architecture(&architecture),
                    DenseQwenArchitecture::Supported(_)
                )
            })
            .unwrap_or(false)
        {
            return NormValidationPolicy::DenseQwen;
        }

        if reader
            .get_string_metadata("general.architecture")
            .map(|architecture| architecture.to_ascii_lowercase().contains("bitnet"))
            .unwrap_or(false)
        {
            return NormValidationPolicy::BitNetPreScaled;
        }

        if self.smollm2_360m_metadata_matches(reader, config, fingerprint) {
            return NormValidationPolicy::SmolLm2_360MInstructQ8;
        }

        NormValidationPolicy::Generic
    }

    fn smollm2_360m_metadata_matches(
        &self,
        reader: &GgufReader,
        config: &BitNetConfig,
        fingerprint: &str,
    ) -> bool {
        let architecture = reader
            .get_string_metadata("general.architecture")
            .map(|value| value.eq_ignore_ascii_case("llama"))
            .unwrap_or(false);
        let tokenizer_pre = reader
            .get_string_metadata("tokenizer.ggml.pre")
            .map(|value| value.eq_ignore_ascii_case("smollm"))
            .unwrap_or(false);

        architecture
            && tokenizer_pre
            && fingerprint == SMOLLM2_360M_FINGERPRINT
            && config.model.hidden_size == SMOLLM2_360M_HIDDEN_SIZE
            && config.model.num_layers == SMOLLM2_360M_LAYER_COUNT
            && config.model.vocab_size == SMOLLM2_360M_VOCAB_SIZE
            && config.model.intermediate_size == SMOLLM2_360M_INTERMEDIATE_SIZE
    }

    /// Check if a tensor name indicates it's an embedding tensor
    fn is_embedding_tensor(name: &str) -> bool {
        matches!(
            name,
            "embed_tokens.weight"
                | "tok_embeddings.weight"
                | "token_embd.weight"
                | "model.embed_tokens.weight"
                | "transformer.wte.weight"
        )
    }

    /// Check if a tensor name indicates the final vocabulary projection.
    fn is_output_head_tensor(name: &str) -> bool {
        matches!(
            name,
            "output.weight"
                | "lm_head.weight"
                | "model.lm_head.weight"
                | "generator.weight"
                | "transformer.lm_head.weight"
                | "language_model_head.weight"
                | "head.weight"
                | "cls.weight"
        )
    }

    /// Check if a tensor name indicates it's a projection tensor that needs transposition
    /// This includes both attention and feed-forward projection tensors
    fn is_projection_tensor(name: &str) -> bool {
        // Attention projection tensors
        name.contains("attn_q.weight") ||
        name.contains("attn_k.weight") ||
        name.contains("attn_v.weight") ||
        name.contains("attn_output.weight") ||
        name.contains("q_proj.weight") ||
        name.contains("k_proj.weight") ||
        name.contains("v_proj.weight") ||
        name.contains("o_proj.weight") ||
        // Feed-forward projection tensors
        name.contains("ffn_gate.weight") ||
        name.contains("ffn_up.weight") ||
        name.contains("ffn_down.weight") ||
        name.contains("gate_proj.weight") ||
        name.contains("up_proj.weight") ||
        name.contains("down_proj.weight")
    }

    /// Heuristic: Microsoft 2B ships [hidden, vocab]; we want [vocab, hidden].
    fn embedding_is_transposed(dims: &[usize]) -> bool {
        dims.len() == 2 && dims[0] < dims[1] && dims[1] >= 32768
    }

    /// Helper to transpose F16 data to F32 transposed layout
    fn transpose_f16_to_f32(bytes: &[u8], dims: &[usize]) -> Result<Vec<f32>> {
        use std::io::Read;
        let (rows, cols) = (dims[0], dims[1]);
        let mut out = vec![0f32; rows * cols]; // transposed [cols, rows]
        let mut rdr = std::io::Cursor::new(bytes);
        for r in 0..rows {
            for c in 0..cols {
                let mut buf = [0u8; 2];
                rdr.read_exact(&mut buf).map_err(BitNetError::Io)?;
                let v = half::f16::from_bits(u16::from_le_bytes(buf)).to_f32();
                out[c * rows + r] = v;
            }
        }
        Ok(out)
    }

    /// Helper to transpose F32 data to F32 transposed layout
    fn transpose_f32_to_f32(bytes: &[u8], dims: &[usize]) -> Result<Vec<f32>> {
        let (rows, cols) = (dims[0], dims[1]);

        // Read F32 values from bytes using safe byte casting
        let f32_values = bytemuck::cast_slice::<u8, f32>(bytes);

        // Transpose from [rows, cols] to [cols, rows] using efficient indexing
        let mut transposed = Vec::with_capacity(rows * cols);
        for col in 0..cols {
            for row in 0..rows {
                transposed.push(f32_values[row * cols + col]);
            }
        }

        Ok(transposed)
    }

    /// Dequantize GGML Q8_0 blocks into F32 values.
    ///
    /// Each Q8_0 block stores one little-endian f16 scale followed by 32 signed
    /// 8-bit quantized values. GGUF tensor payloads can include trailing
    /// alignment bytes, so callers provide the full slice and this helper
    /// consumes exactly the blocks required by the logical tensor shape.
    pub(super) fn dequantize_q8_0_to_f32(
        bytes: &[u8],
        dims: &[usize],
        tensor_name: &str,
    ) -> Result<Vec<f32>> {
        let elements = dims.iter().try_fold(1usize, |acc, dim| {
            acc.checked_mul(*dim).ok_or_else(|| {
                BitNetError::Validation(format!(
                    "Q8_0 tensor '{tensor_name}' shape {:?} overflows element count",
                    dims
                ))
            })
        })?;
        let blocks = elements.div_ceil(32);
        let expected =
            blocks.checked_mul(GgufTensorType::Q8_0.element_size()).ok_or_else(|| {
                BitNetError::Validation(format!(
                    "Q8_0 tensor '{tensor_name}' byte size overflows for {blocks} blocks"
                ))
            })?;

        if bytes.len() < expected {
            return Err(BitNetError::Validation(format!(
                "Q8_0 tensor '{tensor_name}' has {} bytes, expected at least {} for {} elements",
                bytes.len(),
                expected,
                elements
            )));
        }

        let mut out = Vec::with_capacity(elements);
        for block_idx in 0..blocks {
            let offset = block_idx * GgufTensorType::Q8_0.element_size();
            let scale_bits = u16::from_le_bytes([bytes[offset], bytes[offset + 1]]);
            let scale = half::f16::from_bits(scale_bits).to_f32();
            for code_idx in 0..32 {
                if out.len() == elements {
                    break;
                }
                let q = bytes[offset + 2 + code_idx] as i8;
                out.push(scale * q as f32);
            }
        }

        Ok(out)
    }

    fn checked_dense_quant_blocks(
        bytes: &[u8],
        dims: &[usize],
        tensor_name: &str,
        tensor_type: GgufTensorType,
    ) -> Result<(usize, usize)> {
        let elements = dims.iter().try_fold(1usize, |acc, dim| {
            acc.checked_mul(*dim).ok_or_else(|| {
                BitNetError::Validation(format!(
                    "{tensor_type:?} tensor '{tensor_name}' shape {:?} overflows element count",
                    dims
                ))
            })
        })?;
        let block_size = tensor_type.block_size();
        let blocks = elements.div_ceil(block_size);
        let expected = blocks.checked_mul(tensor_type.element_size()).ok_or_else(|| {
            BitNetError::Validation(format!(
                "{tensor_type:?} tensor '{tensor_name}' byte size overflows for {blocks} blocks"
            ))
        })?;

        if bytes.len() < expected {
            return Err(BitNetError::Validation(format!(
                "{tensor_type:?} tensor '{tensor_name}' has {} bytes, expected at least {} for {} elements",
                bytes.len(),
                expected,
                elements
            )));
        }

        Ok((elements, blocks))
    }

    pub(super) fn dequantize_q5_0_to_f32(
        bytes: &[u8],
        dims: &[usize],
        tensor_name: &str,
    ) -> Result<Vec<f32>> {
        let (elements, blocks) =
            Self::checked_dense_quant_blocks(bytes, dims, tensor_name, GgufTensorType::Q5_0)?;

        let mut out = Vec::with_capacity(elements);
        for block_idx in 0..blocks {
            let offset = block_idx * GgufTensorType::Q5_0.element_size();
            let scale =
                half::f16::from_bits(u16::from_le_bytes([bytes[offset], bytes[offset + 1]]))
                    .to_f32();
            let qh = u32::from_le_bytes([
                bytes[offset + 2],
                bytes[offset + 3],
                bytes[offset + 4],
                bytes[offset + 5],
            ]);
            let qs = &bytes[offset + 6..offset + 22];

            let mut block = [0.0f32; 32];
            for j in 0..16 {
                let xh_0 = ((qh >> j) << 4) & 0x10;
                let xh_1 = (qh >> (j + 12)) & 0x10;
                let x0 = i32::from((qs[j] & 0x0f) | xh_0 as u8) - 16;
                let x1 = i32::from((qs[j] >> 4) | xh_1 as u8) - 16;
                block[j] = scale * x0 as f32;
                block[j + 16] = scale * x1 as f32;
            }

            for value in block {
                if out.len() == elements {
                    break;
                }
                out.push(value);
            }
        }

        Ok(out)
    }

    #[inline]
    fn q4_k_scale_min(index: usize, scales: &[u8]) -> (u8, u8) {
        if index < 4 {
            (scales[index] & 63, scales[index + 4] & 63)
        } else {
            (
                (scales[index + 4] & 0x0f) | ((scales[index - 4] >> 6) << 4),
                (scales[index + 4] >> 4) | ((scales[index] >> 6) << 4),
            )
        }
    }

    pub(super) fn dequantize_q4_k_to_f32(
        bytes: &[u8],
        dims: &[usize],
        tensor_name: &str,
    ) -> Result<Vec<f32>> {
        let (elements, blocks) =
            Self::checked_dense_quant_blocks(bytes, dims, tensor_name, GgufTensorType::Q4_K)?;

        let mut out = Vec::with_capacity(elements);
        for block_idx in 0..blocks {
            let offset = block_idx * GgufTensorType::Q4_K.element_size();
            let d = half::f16::from_bits(u16::from_le_bytes([bytes[offset], bytes[offset + 1]]))
                .to_f32();
            let dmin =
                half::f16::from_bits(u16::from_le_bytes([bytes[offset + 2], bytes[offset + 3]]))
                    .to_f32();
            let scales = &bytes[offset + 4..offset + 16];
            let qs = &bytes[offset + 16..offset + 144];

            let mut scale_index = 0usize;
            let mut q_offset = 0usize;
            for _ in (0..256).step_by(64) {
                let (sc1, m1) = Self::q4_k_scale_min(scale_index, scales);
                let d1 = d * f32::from(sc1);
                let m1 = dmin * f32::from(m1);
                let (sc2, m2) = Self::q4_k_scale_min(scale_index + 1, scales);
                let d2 = d * f32::from(sc2);
                let m2 = dmin * f32::from(m2);

                let q = &qs[q_offset..q_offset + 32];
                for code in q {
                    if out.len() == elements {
                        return Ok(out);
                    }
                    out.push(d1 * f32::from(code & 0x0f) - m1);
                }
                for code in q {
                    if out.len() == elements {
                        return Ok(out);
                    }
                    out.push(d2 * f32::from(code >> 4) - m2);
                }

                q_offset += 32;
                scale_index += 2;
            }
        }

        Ok(out)
    }

    pub(super) fn dequantize_q6_k_to_f32(
        bytes: &[u8],
        dims: &[usize],
        tensor_name: &str,
    ) -> Result<Vec<f32>> {
        let (elements, blocks) =
            Self::checked_dense_quant_blocks(bytes, dims, tensor_name, GgufTensorType::Q6_K)?;

        let mut out = Vec::with_capacity(elements);
        for block_idx in 0..blocks {
            let offset = block_idx * GgufTensorType::Q6_K.element_size();
            let ql = &bytes[offset..offset + 128];
            let qh = &bytes[offset + 128..offset + 192];
            let scales = &bytes[offset + 192..offset + 208];
            let d = half::f16::from_bits(u16::from_le_bytes([
                bytes[offset + 208],
                bytes[offset + 209],
            ]))
            .to_f32();

            let mut ql_offset = 0usize;
            let mut qh_offset = 0usize;
            let mut scale_offset = 0usize;
            for _ in (0..256).step_by(128) {
                let mut block = [0.0f32; 128];
                for l in 0..32 {
                    let is = l / 16;
                    let qh_l = qh[qh_offset + l];
                    let q1 = i32::from((ql[ql_offset + l] & 0x0f) | ((qh_l & 3) << 4)) - 32;
                    let q2 =
                        i32::from((ql[ql_offset + l + 32] & 0x0f) | (((qh_l >> 2) & 3) << 4)) - 32;
                    let q3 = i32::from((ql[ql_offset + l] >> 4) | (((qh_l >> 4) & 3) << 4)) - 32;
                    let q4 =
                        i32::from((ql[ql_offset + l + 32] >> 4) | (((qh_l >> 6) & 3) << 4)) - 32;

                    let sc1 = scales[scale_offset + is] as i8 as f32;
                    let sc2 = scales[scale_offset + is + 2] as i8 as f32;
                    let sc3 = scales[scale_offset + is + 4] as i8 as f32;
                    let sc4 = scales[scale_offset + is + 6] as i8 as f32;

                    block[l] = d * sc1 * q1 as f32;
                    block[l + 32] = d * sc2 * q2 as f32;
                    block[l + 64] = d * sc3 * q3 as f32;
                    block[l + 96] = d * sc4 * q4 as f32;
                }

                for value in block {
                    if out.len() == elements {
                        return Ok(out);
                    }
                    out.push(value);
                }

                ql_offset += 64;
                qh_offset += 32;
                scale_offset += 8;
            }
        }

        Ok(out)
    }

    fn dequantize_supported_dense_standard_quant_to_f32(
        bytes: &[u8],
        dims: &[usize],
        tensor_name: &str,
        tensor_type: GgufTensorType,
    ) -> Result<Vec<f32>> {
        match tensor_type {
            GgufTensorType::Q8_0 => Self::dequantize_q8_0_to_f32(bytes, dims, tensor_name),
            GgufTensorType::Q5_0 => Self::dequantize_q5_0_to_f32(bytes, dims, tensor_name),
            GgufTensorType::Q4_K => Self::dequantize_q4_k_to_f32(bytes, dims, tensor_name),
            GgufTensorType::Q6_K => Self::dequantize_q6_k_to_f32(bytes, dims, tensor_name),
            other => Err(BitNetError::Validation(format!(
                "unsupported dense standard GGUF quantization {other:?} in tensor '{tensor_name}'"
            ))),
        }
    }

    fn f16_values_to_f32(bytes: &[u8], dims: &[usize], name: &str) -> Result<Vec<f32>> {
        let elements = dims.iter().try_fold(1usize, |acc, dim| {
            acc.checked_mul(*dim).ok_or_else(|| {
                BitNetError::Validation(format!(
                    "F16 tensor '{name}' shape {:?} overflows element count",
                    dims
                ))
            })
        })?;
        let expected_bytes = elements.checked_mul(2).ok_or_else(|| {
            BitNetError::Validation(format!(
                "F16 tensor '{name}' byte count overflows for shape {:?}",
                dims
            ))
        })?;
        if bytes.len() < expected_bytes {
            return Err(BitNetError::Validation(format!(
                "F16 tensor '{name}' has {} bytes, expected at least {} for {:?}",
                bytes.len(),
                expected_bytes,
                dims
            )));
        }

        Ok(bytes[..expected_bytes]
            .chunks_exact(2)
            .map(|chunk| half::f16::from_bits(u16::from_le_bytes([chunk[0], chunk[1]])).to_f32())
            .collect())
    }

    #[allow(dead_code)]
    pub(super) fn transpose_f32_values(values: &[f32], dims: &[usize]) -> Vec<f32> {
        let (rows, cols) = (dims[0], dims[1]);
        let mut transposed = Vec::with_capacity(rows * cols);
        for col in 0..cols {
            for row in 0..rows {
                transposed.push(values[row * cols + col]);
            }
        }
        transposed
    }

    /// Helper to create a transposed I2_S tensor (for attention projections)
    #[allow(dead_code)]
    fn create_transposed_i2s_tensor(
        data: &[u8],
        dims: &[usize],
        device: &candle_core::Device,
    ) -> Result<Tensor> {
        use crate::quant::i2s::I2SDequantCfg;
        Self::create_transposed_i2s_tensor_with_cfg(data, dims, device, I2SDequantCfg::default())
    }

    fn create_transposed_i2s_tensor_with_cfg(
        data: &[u8],
        dims: &[usize],
        device: &candle_core::Device,
        cfg: crate::quant::i2s::I2SDequantCfg,
    ) -> Result<Tensor> {
        use crate::quant::i2s;

        // First dequantize to F32 with config
        let f32_data = i2s::dequantize_to_f32_with_cfg(data, dims, cfg).map_err(|e| {
            BitNetError::Validation(format!(
                "I2_S dequantization failed for tensor with shape {:?}: {}",
                dims, e
            ))
        })?;

        // Then transpose from [rows, cols] to [cols, rows] using efficient indexing
        let (rows, cols) = (dims[0], dims[1]);
        let mut transposed = Vec::with_capacity(rows * cols);
        for col in 0..cols {
            for row in 0..rows {
                transposed.push(f32_data[row * cols + col]);
            }
        }

        // Create tensor with transposed dimensions
        let tensor = Tensor::from_slice(&transposed, &[cols, rows], device)
            .map_err(|e| BitNetError::Validation(e.to_string()))?;
        Ok(tensor)
    }

    pub(super) fn extract_config(&self, reader: &GgufReader) -> Result<BitNetConfig> {
        let mut config = BitNetConfig::default();
        let architecture = reader
            .get_string_metadata("general.architecture")
            .unwrap_or_else(|| "bitnet".to_string());
        config.model.apply_architecture_defaults(&architecture);
        let arch_vocab = format!("{architecture}.vocab_size");
        let arch_block_count = format!("{architecture}.block_count");
        let arch_embedding_length = format!("{architecture}.embedding_length");
        let arch_head_count = format!("{architecture}.attention.head_count");
        let arch_head_count_kv = format!("{architecture}.attention.head_count_kv");
        let arch_attention_key_length = format!("{architecture}.attention.key_length");
        let arch_attention_value_length = format!("{architecture}.attention.value_length");
        let arch_feed_forward_length = format!("{architecture}.feed_forward_length");
        let arch_context_length = format!("{architecture}.context_length");
        let arch_rope_freq_base = format!("{architecture}.rope.freq_base");
        let arch_rms_eps = format!("{architecture}.attention.layer_norm_rms_epsilon");

        // Extract model configuration from GGUF metadata
        if let Some(vocab_size) = Self::get_u32_any(
            reader,
            &[
                "llama.vocab_size",
                "bitnet-b1.58.vocab_size",
                arch_vocab.as_str(),
                "tokenizer.ggml.tokens",
            ],
        ) {
            config.model.vocab_size = vocab_size as usize;
        }
        if (config.model.vocab_size == 0
            || config.model.vocab_size == BitNetConfig::default().model.vocab_size)
            && let Some(vocab) = Self::infer_vocab_size_from_tensors(reader)
        {
            tracing::info!("inferred vocab_size={} from token embedding tensor", vocab);
            config.model.vocab_size = vocab;
        }

        if let Some(num_layers) = Self::get_u32_any(
            reader,
            &[
                "llama.block_count",
                "bitnet-b1.58.block_count",
                arch_block_count.as_str(),
                "n_layer",
            ],
        ) {
            config.model.num_layers = num_layers as usize;
        }

        // If layer count wasn't in metadata or seems wrong, infer from tensors
        if (config.model.num_layers == 0
            || config.model.num_layers == BitNetConfig::default().model.num_layers)
            && let Some(layers) = Self::infer_num_layers_from_tensors(reader)
        {
            tracing::info!("Inferred num_layers={} from tensor analysis", layers);
            config.model.num_layers = layers;
        }

        // 1) hidden_size: try metadata, else infer from embeddings
        if let Some(h) = Self::get_u32_any(
            reader,
            &[
                "llama.embedding_length",
                "bitnet-b1.58.embedding_length",
                arch_embedding_length.as_str(),
                "n_embd",
                "hidden_size",
            ],
        ) {
            config.model.hidden_size = h as usize;
        }
        if (config.model.hidden_size == 0
            || config.model.hidden_size == BitNetConfig::default().model.hidden_size)
            && let Some(h) = Self::infer_hidden_size_from_tensors(reader)
        {
            config.model.hidden_size = h;
        }

        // 2) num_heads: broaden key set (MS 2B commonly has "n_head")
        // Include bitnet-b1.58 specific keys which are architecture-prefixed
        if let Some(h) = Self::get_u32_any(
            reader,
            &[
                "llama.attention.head_count",
                "bitnet-b1.58.attention.head_count", // BitNet 2B models
                arch_head_count.as_str(),
                "n_head",
                "attn.n_heads",
                "num_attention_heads",
            ],
        ) {
            config.model.num_heads = h as usize;
        }

        // 3) num_key_value_heads:
        //    a) metadata if present
        let kv_keys = [
            "llama.attention.head_count_kv",
            "bitnet-b1.58.attention.head_count_kv", // BitNet 2B models
            arch_head_count_kv.as_str(),
            "n_head_kv",
            "n_kv_heads",
            "attn.n_kv_heads",
            "attn_n_kv_heads",
            "num_key_value_heads",
        ];
        config.model.num_key_value_heads =
            Self::get_u32_any(reader, &kv_keys).map(|v| v as usize).unwrap_or(0);

        //    b) if not present, infer from tensor shapes (now that hidden_size & num_heads are set)
        if config.model.num_key_value_heads == 0
            && config.model.num_heads > 0
            && config.model.hidden_size > 0
        {
            debug!("No explicit GQA metadata found, attempting shape inference...");
            config.model.num_key_value_heads = Self::infer_kv_heads_from_tensors(reader, &config)?;
            debug!("Final num_key_value_heads: {}", config.model.num_key_value_heads);
        }

        //    c) final fallback: MHA
        if config.model.num_key_value_heads == 0 {
            config.model.num_key_value_heads = config.model.num_heads;
        }

        let attention_key_length = Self::get_u32_any(
            reader,
            &[arch_attention_key_length.as_str(), "llama.attention.key_length"],
        )
        .map(|value| value as usize);
        let attention_value_length = Self::get_u32_any(
            reader,
            &[arch_attention_value_length.as_str(), "llama.attention.value_length"],
        )
        .map(|value| value as usize);
        if let (Some(key_length), Some(value_length)) =
            (attention_key_length, attention_value_length)
            && key_length != value_length
        {
            return Err(BitNetError::Validation(format!(
                "attention key/value dimensions differ: key_length={key_length}, value_length={value_length}"
            )));
        }
        config.model.attention_head_dim = attention_key_length.or(attention_value_length);

        // Log one-liner so you can grep it during runs
        let hidden = config.model.hidden_size;
        let q = config.model.num_heads;
        let kv = config.model.num_key_value_heads;
        if q > 0 && kv > 0 && q % kv == 0 {
            let head_dim = config.model.attention_head_dim.unwrap_or(hidden / q);
            let group = q / kv;
            info!("heads: q={} kv={} (group={}) head_dim={}", q, kv, group, head_dim);
        }

        // 4) intermediate_size: try metadata, else infer from feed-forward tensors
        if let Some(intermediate_size) = Self::get_u32_any(
            reader,
            &[
                "llama.feed_forward_length",
                "bitnet-b1.58.feed_forward_length",
                arch_feed_forward_length.as_str(),
                "n_ff",
            ],
        ) {
            config.model.intermediate_size = intermediate_size as usize;
        }
        // If no metadata or if it seems wrong (based on tensor shapes), infer from tensors
        if (config.model.intermediate_size == 0
            || config.model.intermediate_size == BitNetConfig::default().model.intermediate_size)
            && let Some(inferred_size) =
                Self::infer_intermediate_size_from_tensors(reader, config.model.hidden_size)
        {
            config.model.intermediate_size = inferred_size;
        }

        if let Some(context_length) = Self::get_u32_any(
            reader,
            &["llama.context_length", "bitnet-b1.58.context_length", arch_context_length.as_str()],
        ) {
            config.model.max_position_embeddings = context_length as usize;
        }

        // Read ROPE parameters from header
        // Note: GGUF uses "rope.freq_base" while config uses "rope_theta" (same meaning)
        if let Some(rope_base) = reader
            .get_f32_metadata("bitnet-b1.58.rope.freq_base")
            .or_else(|| reader.get_f32_metadata("llama.rope.freq_base"))
            .or_else(|| reader.get_f32_metadata(arch_rope_freq_base.as_str()))
            .or_else(|| reader.get_f32_metadata("rope.freq_base"))
        {
            config.model.rope_theta = Some(rope_base);
            tracing::info!("ROPE freq_base from header: {}", rope_base);
        }

        // Read RMSNorm epsilon
        if let Some(eps) = Self::get_f32_any(
            reader,
            &[
                "bitnet-b1.58.attention.layer_norm_rms_epsilon",
                "llama.attention.layer_norm_rms_epsilon",
                arch_rms_eps.as_str(),
                "llama.attention.layer_norm_epsilon",
                "general.layer_norm_epsilon",
            ],
        ) {
            config.model.rms_norm_eps = Some(eps);
            tracing::info!("RMSNorm epsilon from header: {}", eps);
        }

        // Read tokenizer special token IDs
        if let Some(bos) = Self::get_u32_any(
            reader,
            &[
                "bitnet-b1.58.tokenizer.bos_token_id",
                "llama.tokenizer.bos_token_id",
                "tokenizer.ggml.bos_token_id",
                "general.bos_token_id",
            ],
        ) {
            config.model.tokenizer.bos_id = Some(bos as i32);
            tracing::info!("BOS token ID from header: {}", bos);
        }

        if let Some(eos) = Self::get_u32_any(
            reader,
            &[
                "bitnet-b1.58.tokenizer.eos_token_id",
                "llama.tokenizer.eos_token_id",
                "tokenizer.ggml.eos_token_id",
                "general.eos_token_id",
            ],
        ) {
            config.model.tokenizer.eos_id = Some(eos as i32);
            tracing::info!("EOS token ID from header: {}", eos);
        }

        if let Some(unk) = Self::get_u32_any(
            reader,
            &[
                "bitnet-b1.58.tokenizer.unknown_token_id",
                "llama.tokenizer.unknown_token_id",
                "tokenizer.ggml.unknown_token_id",
                "general.unknown_token_id",
            ],
        ) {
            config.model.tokenizer.unk_id = Some(unk as i32);
            tracing::info!("UNK token ID from header: {}", unk);
        }

        if let Some(pad) = Self::get_u32_any(
            reader,
            &[
                "bitnet-b1.58.tokenizer.padding_token_id",
                "llama.tokenizer.padding_token_id",
                "tokenizer.ggml.padding_token_id",
                "general.padding_token_id",
            ],
        ) {
            config.model.tokenizer.pad_id = Some(pad as i32);
            tracing::info!("PAD token ID from header: {}", pad);
        }

        // Read tokenizer behavior flags
        if let Some(add_bos) = Self::get_bool_any(
            reader,
            &[
                "bitnet-b1.58.tokenizer.add_bos",
                "tokenizer.ggml.add_bos_token",
                "tokenizer.ggml.add_bos",
                "general.add_bos",
            ],
        ) {
            config.inference.add_bos = add_bos;
            tracing::info!("add_bos from header: {}", add_bos);
        }

        if let Some(append_eos) = Self::get_bool_any(
            reader,
            &[
                "bitnet-b1.58.tokenizer.append_eos",
                "tokenizer.ggml.add_eos_token",
                "tokenizer.ggml.append_eos",
                "general.append_eos",
            ],
        ) {
            config.inference.append_eos = append_eos;
            tracing::info!("append_eos from header: {}", append_eos);
        }

        if let Some(mask_pad) = Self::get_bool_any(
            reader,
            &["bitnet-b1.58.tokenizer.mask_pad", "tokenizer.ggml.mask_pad", "general.mask_pad"],
        ) {
            config.inference.mask_pad = mask_pad;
            tracing::info!("mask_pad from header: {}", mask_pad);
        }

        // Log final model configuration
        info!(
            "model dimensions: hidden={}, intermediate={}, layers={}, vocab={}",
            config.model.hidden_size,
            config.model.intermediate_size,
            config.model.num_layers,
            config.model.vocab_size
        );

        // Set quantization type based on tensor types
        if let Some(qtype) = reader.get_quantization_type() {
            config.quantization.quantization_type = qtype;
        }

        // Extract additional BitNet-specific configuration
        if let Some(block_size) = reader.get_u32_metadata("bitnet.block_size") {
            config.quantization.block_size = block_size as usize;
        }

        if let Some(precision) = reader.get_f32_metadata("bitnet.precision") {
            config.quantization.precision = precision;
        }

        Ok(config)
    }

    fn load_tensors(
        &self,
        reader: &GgufReader,
        device: &Device,
        config: &LoadConfig,
        fingerprint: &str,
        norm_validation_policy: NormValidationPolicy,
    ) -> DenseQ8SidecarLoadResult {
        let tensor_count = reader.tensor_count() as usize;
        let mut tensors = GgufTensors::new();
        let mut raw_tensors: std::collections::HashMap<String, Tensor> =
            std::collections::HashMap::new();
        let mut dense_q8_sidecars = DenseGgufQ8SidecarRegistry::default();
        let dense_q8_payload_candidate_tensor = dense_q8_payload_candidate_tensor_from_env();

        info!("Loading {} tensors", tensor_count);

        // Extract model config for QK256 orientation detection
        let model_config = self.extract_config(reader)?;

        // Load correction policy if BITNET_CORRECTION_POLICY is set
        let policy = if let Ok(policy_path) = std::env::var("BITNET_CORRECTION_POLICY") {
            match crate::correction_policy::CorrectionPolicy::load_from_file(std::path::Path::new(
                &policy_path,
            )) {
                Ok(p) => {
                    p.validate()?;
                    info!("Loaded correction policy from: {}", policy_path);
                    Some(p)
                }
                Err(e) => {
                    tracing::warn!("Failed to load correction policy from {}: {}", policy_path, e);
                    None
                }
            }
        } else {
            None
        };

        // Find plan for this model (if policy exists and fingerprint matches)
        let policy_plan =
            if let Some(ref pol) = policy { pol.find_plan(fingerprint) } else { None };

        let mut corrections = Vec::new();

        for i in 0..tensor_count {
            if let Some(callback) = &config.progress_callback {
                let progress = 0.5 + (i as f32 / tensor_count as f32) * 0.4;
                callback(progress, &format!("Loading tensor {}/{}", i + 1, tensor_count));
            }

            let tensor_info = reader.get_tensor_info(i)?;
            let tensor_data = reader.get_tensor_data(i)?;

            debug!(
                "Loading tensor '{}' with shape {:?} and type {:?}",
                tensor_info.name, tensor_info.shape, tensor_info.tensor_type
            );

            // Convert to Candle tensor (now with policy plan and QK256 handling)
            let (candle_tensor, raw_qk256_entries, correction_opt) = self
                .create_candle_tensor_with_policy(
                    tensor_info,
                    tensor_data,
                    device,
                    &model_config,
                    policy_plan.as_ref(),
                    norm_validation_policy,
                )?;
            tensors.insert(tensor_info.name.clone(), candle_tensor);
            dense_q8_sidecars.try_push_tensor_with_payload_candidate(
                tensor_info,
                tensor_data,
                dense_q8_payload_candidate_tensor.as_deref(),
            )?;

            // Store raw QK256 tensors if present.
            for (key, raw_tensor) in raw_qk256_entries {
                raw_tensors.insert(key, raw_tensor);
            }

            // Collect correction records
            if let Some(corr) = correction_opt {
                corrections.push(corr);
            }
        }

        // Log correction summary and complete metadata
        if !corrections.is_empty() {
            info!("Applied {} corrections during model load", corrections.len());
            for corr in &corrections {
                info!(
                    "  CORRECTION: layer='{}' type='{}' fingerprint='{}'",
                    corr.layer, corr.correction_type, corr.policy_fingerprint
                );
            }

            // Log complete metadata summary for receipts
            info!(
                "Model corrections applied: fingerprint={}, corrections_count={}",
                fingerprint,
                corrections.len()
            );

            // Log individual correction details in debug
            if tracing::enabled!(tracing::Level::DEBUG) {
                for corr in &corrections {
                    if let Some(ref metadata) = corr.metadata {
                        debug!("  Correction metadata: {}", metadata);
                    }
                }
            }
        }

        info!(
            "Successfully loaded {} tensors (detected {} QK256 tensors) with fingerprint: {}",
            tensors.len(),
            raw_tensors.len(),
            fingerprint
        );
        if !dense_q8_sidecars.is_empty() {
            info!(
                "Carried {} inert dense Q8_0 sidecar descriptors; eager F32 Candle tensors remain the runtime path",
                dense_q8_sidecars.descriptor_count()
            );
        }
        Ok((tensors, raw_tensors, dense_q8_sidecars))
    }

    /// Validate tensor data integrity
    #[cfg(any(test, feature = "validation"))]
    #[allow(dead_code)]
    fn validate_tensor_data(
        &self,
        info: &crate::formats::gguf::TensorInfo,
        data: &[u8],
    ) -> Result<()> {
        // Check data size matches expected size
        let expected_size = info.size as usize;
        if data.len() != expected_size {
            return Err(BitNetError::Validation(format!(
                "Tensor '{}' data size mismatch: expected {}, got {}",
                info.name,
                expected_size,
                data.len()
            )));
        }

        // For quantized tensors, validate block alignment
        if info.tensor_type.is_quantized() {
            let block_size = info.tensor_type.block_size();
            let total_elements: usize = info.shape.iter().product();

            if !total_elements.is_multiple_of(block_size) {
                return Err(BitNetError::Validation(format!(
                    "Tensor '{}' elements ({}) not aligned to block size ({})",
                    info.name, total_elements, block_size
                )));
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn q8_vocab_projection_uses_token_major_shape() {
        let ggml_hidden_vocab = [896, 151_936];

        assert!(GgufLoader::embedding_is_transposed(&ggml_hidden_vocab));
        assert!(GgufLoader::is_embedding_tensor("token_embd.weight"));
        assert!(GgufLoader::is_output_head_tensor("output.weight"));
        assert!(GgufLoader::is_output_head_tensor("lm_head.weight"));
        assert!(GgufLoader::is_output_head_tensor("generator.weight"));
        assert!(GgufLoader::is_output_head_tensor("transformer.lm_head.weight"));
        assert!(GgufLoader::is_output_head_tensor("language_model_head.weight"));
        assert!(GgufLoader::is_output_head_tensor("cls.weight"));
        assert!(!GgufLoader::is_output_head_tensor("blk.0.attn_output.weight"));

        let output_shape = if (GgufLoader::is_embedding_tensor("output.weight")
            || GgufLoader::is_output_head_tensor("output.weight"))
            && GgufLoader::embedding_is_transposed(&ggml_hidden_vocab)
        {
            vec![ggml_hidden_vocab[1], ggml_hidden_vocab[0]]
        } else {
            ggml_hidden_vocab.to_vec()
        };

        assert_eq!(output_shape, vec![151_936, 896]);
    }

    #[test]
    fn qk256_inline_scale_reads_first_trailing_f32() {
        let expected_raw_bytes = 64;
        let mut data = vec![0u8; expected_raw_bytes];
        data.extend_from_slice(&0.125f32.to_le_bytes());
        data.extend_from_slice(&[0u8; 28]);

        let scale =
            GgufLoader::qk256_inline_scale(&data, expected_raw_bytes, "blk.0.attn_q.weight")
                .expect("scale parse");

        assert_eq!(scale, Some(0.125));
    }

    #[test]
    fn qk256_inline_scale_is_absent_without_trailing_bytes() {
        let data = vec![0u8; 64];

        let scale =
            GgufLoader::qk256_inline_scale(&data, 64, "blk.0.attn_q.weight").expect("scale parse");

        assert_eq!(scale, None);
    }

    #[test]
    fn qk256_token_embedding_rows_use_bitnetcpp_dequant_map_and_scale() {
        let mut data = vec![0x00u8; 64];
        data.extend(std::iter::repeat_n(0xAAu8, 64));
        data.extend_from_slice(&0.5f32.to_le_bytes());

        let values = GgufLoader::dequantize_qk256_token_embedding_rows(
            &data,
            2,
            256,
            Some(0.5),
            64,
            "token_embd.weight",
        )
        .expect("embedding dequant");

        assert_eq!(values.len(), 512);
        assert_eq!(values[0], -0.5);
        assert_eq!(values[255], -0.5);
        assert_eq!(values[256], 0.5);
        assert_eq!(values[511], 0.5);
    }

    #[test]
    fn f16_embedding_payload_is_token_major_for_ggml_hidden_vocab_shape() {
        let values = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut data = Vec::new();
        for value in values {
            data.extend_from_slice(&half::f16::from_f32(value).to_bits().to_le_bytes());
        }

        let f32_values =
            GgufLoader::f16_values_to_f32(&data, &[3, 2], "token_embd.weight").expect("f16 parse");

        assert_eq!(f32_values, values);
    }

    fn one_layer_config() -> BitNetConfig {
        let mut config = BitNetConfig::default();
        config.model.num_layers = 1;
        config
    }

    fn one_layer_names_with_embedding() -> Vec<&'static str> {
        vec![
            "token_embd.weight",
            "blk.0.attn_q.weight",
            "blk.0.attn_k.weight",
            "blk.0.attn_v.weight",
            "blk.0.attn_output.weight",
            "blk.0.ffn_gate.weight",
            "blk.0.ffn_up.weight",
            "blk.0.ffn_down.weight",
            "blk.0.attn_norm.weight",
            "blk.0.ffn_norm.weight",
        ]
    }

    #[test]
    fn strict_tensor_authority_allows_tied_lm_head_from_embeddings() -> Result<()> {
        let names = one_layer_names_with_embedding();

        GgufLoader::validate_strict_tensor_authority_names(&names, &one_layer_config())?;

        Ok(())
    }

    #[test]
    fn strict_tensor_authority_rejects_missing_logits_source() {
        let mut names = one_layer_names_with_embedding();
        names.retain(|name| *name != "token_embd.weight");

        let err = GgufLoader::validate_strict_tensor_authority_names(&names, &one_layer_config())
            .expect_err("missing embeddings and output head should be rejected");
        let msg = err.to_string();
        assert!(msg.contains("token embedding weight"), "got: {msg}");
        assert!(msg.contains("output/lm head weight"), "got: {msg}");
    }
}
