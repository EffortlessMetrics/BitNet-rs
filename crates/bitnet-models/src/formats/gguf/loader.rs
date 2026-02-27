//! GGUF format loader implementation

use super::{GgufReader, GgufTensorType, GgufTensors};
use crate::loader::{FormatLoader, LoadConfig, MmapFile};
use crate::names::{is_layernorm_weight, is_projection_weight};
use crate::qk256_utils::{detect_qk256_orientation_by_bytes, expected_qk256_shape};
use crate::{BitNetModel, Model};
use bitnet_common::{
    BitNetConfig, BitNetError, CorrectionRecord, Device, ModelError, ModelMetadata, Result,
};
use candle_core::{DType, Tensor};
use std::path::Path;
use tracing::{debug, info};

/// Type alias for tensor load result with optional raw tensor and correction record
type TensorLoadResult = Result<(Tensor, Option<(String, Tensor)>, Option<CorrectionRecord>)>;

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
        // Check for "blk.N." pattern
        if let Some(start) = name.find("blk.") {
            let after_blk = &name[start + 4..];
            if let Some(dot_pos) = after_blk.find('.') {
                let number_str = &after_blk[..dot_pos];
                if let Ok(layer_num) = number_str.parse::<usize>() {
                    return Some(layer_num);
                }
            }
        }

        // Check for "layers.N." pattern
        if let Some(start) = name.find("layers.") {
            let after_layers = &name[start + 7..];
            if let Some(dot_pos) = after_layers.find('.') {
                let number_str = &after_layers[..dot_pos];
                if let Ok(layer_num) = number_str.parse::<usize>() {
                    return Some(layer_num);
                }
            }
        }

        None
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
        }
    }

    /// Validate LayerNorm gamma statistics to catch quantization artifacts.
    ///
    /// LayerNorm gamma RMS should be near 1.0 (acceptable envelope: [0.5, 2.0]).
    /// If stats are suspicious, fail in strict mode or warn otherwise.
    ///
    /// Set BITNET_STRICT_MODE=1 to fail on invalid LN gamma.
    pub(crate) fn check_ln_gamma_stats(name: &str, w: &Tensor) -> Result<()> {
        use bitnet_common::SecurityError;

        // Convert to FP32 for reliable statistics
        let w32 = w.to_dtype(DType::F32).map_err(|e| BitNetError::Validation(e.to_string()))?;
        let rms = Self::rms_f32(&w32)?;

        // Acceptable envelope for γ RMS
        let ok = (0.5..=2.0).contains(&rms) && rms.is_finite();

        if !ok {
            let msg =
                format!("LayerNorm gamma '{}' suspicious: rms={:.5} (expected ≈1.0)", name, rms);

            // In strict mode, fail immediately
            if Self::env_truthy("BITNET_STRICT_MODE") {
                return Err(BitNetError::Security(SecurityError::MalformedData { reason: msg }));
            } else {
                tracing::info!("{} (continuing: non-strict mode)", msg);
            }
        }

        Ok(())
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

        let metadata = ModelMetadata {
            name: reader.get_string_metadata("general.name").unwrap_or_else(|| {
                path.file_stem().and_then(|s| s.to_str()).unwrap_or("unknown").to_string()
            }),
            version: reader
                .get_string_metadata("general.version")
                .unwrap_or_else(|| format!("gguf-v{}", reader.version())),
            architecture: reader
                .get_string_metadata("general.architecture")
                .unwrap_or_else(|| "bitnet".to_string()),
            vocab_size: reader
                .get_u32_metadata("llama.vocab_size")
                .or_else(|| reader.get_u32_metadata("tokenizer.ggml.tokens"))
                .unwrap_or(32000) as usize,
            context_length: reader
                .get_u32_metadata("llama.context_length")
                .or_else(|| reader.get_u32_metadata("llama.rope.dimension_count"))
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

        if let Some(callback) = &config.progress_callback {
            callback(0.5, "Loading tensors...");
        }

        // Load tensors with fingerprint for policy matching (returns both regular and raw QK256 tensors)
        let (tensors, raw_tensors) = self.load_tensors(&reader, device, config, &fingerprint)?;

        if let Some(callback) = &config.progress_callback {
            callback(0.9, "Initializing model...");
        }

        // Create model instance (pass both tensors and raw_tensors for QK256 dispatch)
        let model = BitNetModel::from_gguf(model_config, tensors, raw_tensors, *device)?;

        Ok(Box::new(model))
    }
}

impl GgufLoader {
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

    fn extract_config(&self, reader: &GgufReader) -> Result<BitNetConfig> {
        let mut config = BitNetConfig::default();

        // Extract model configuration from GGUF metadata
        if let Some(vocab_size) = Self::get_u32_any(
            reader,
            &["llama.vocab_size", "bitnet-b1.58.vocab_size", "tokenizer.ggml.tokens"],
        ) {
            config.model.vocab_size = vocab_size as usize;
        }

        if let Some(num_layers) =
            Self::get_u32_any(reader, &["llama.block_count", "bitnet-b1.58.block_count", "n_layer"])
        {
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
            &["llama.embedding_length", "bitnet-b1.58.embedding_length", "n_embd", "hidden_size"],
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

        // Log one-liner so you can grep it during runs
        let hidden = config.model.hidden_size;
        let q = config.model.num_heads;
        let kv = config.model.num_key_value_heads;
        if q > 0 && hidden % q == 0 && kv > 0 && q % kv == 0 {
            let head_dim = hidden / q;
            let group = q / kv;
            info!("heads: q={} kv={} (group={}) head_dim={}", q, kv, group, head_dim);
        }

        // 4) intermediate_size: try metadata, else infer from feed-forward tensors
        if let Some(intermediate_size) = Self::get_u32_any(
            reader,
            &["llama.feed_forward_length", "bitnet-b1.58.feed_forward_length", "n_ff"],
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

        if let Some(context_length) =
            Self::get_u32_any(reader, &["llama.context_length", "bitnet-b1.58.context_length"])
        {
            config.model.max_position_embeddings = context_length as usize;
        }

        // Read ROPE parameters from header
        // Note: GGUF uses "rope.freq_base" while config uses "rope_theta" (same meaning)
        if let Some(rope_base) = reader
            .get_f32_metadata("bitnet-b1.58.rope.freq_base")
            .or_else(|| reader.get_f32_metadata("llama.rope.freq_base"))
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
    ) -> Result<(GgufTensors, std::collections::HashMap<String, Tensor>)> {
        let tensor_count = reader.tensor_count() as usize;
        let mut tensors = GgufTensors::new();
        let mut raw_tensors: std::collections::HashMap<String, Tensor> =
            std::collections::HashMap::new();

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
            let (candle_tensor, raw_qk256_opt, correction_opt) = self
                .create_candle_tensor_with_policy(
                    tensor_info,
                    tensor_data,
                    device,
                    &model_config,
                    policy_plan.as_ref(),
                )?;
            tensors.insert(tensor_info.name.clone(), candle_tensor);

            // Store raw QK256 tensor if present
            if let Some((key, raw_tensor)) = raw_qk256_opt {
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
        Ok((tensors, raw_tensors))
    }

    /// Create a Candle tensor from GGUF tensor info, optionally applying policy-driven corrections
    /// Returns (tensor, raw_qk256_tensor_opt, correction_record_opt)
    fn create_candle_tensor_with_policy(
        &self,
        info: &crate::formats::gguf::TensorInfo,
        data: &[u8],
        device: &Device,
        model_config: &BitNetConfig,
        policy_plan: Option<&crate::correction_policy::CorrectionPlan>,
    ) -> TensorLoadResult {
        let dtype = match info.tensor_type {
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
            | GgufTensorType::I2_S => DType::U8, // Quantized types stored as bytes
        };

        let candle_device = Self::device_to_candle(device)?;

        // For quantized tensors, we need special handling
        if info.tensor_type.is_quantized() {
            // Handle IQ2_S quantization with FFI dequantization
            #[cfg(feature = "iq2s-ffi")]
            if matches!(info.tensor_type, GgufTensorType::IQ2_S) {
                use crate::quant::iq2s;
                let f32_data = iq2s::dequantize_to_f32(data, &info.shape)
                    .map_err(|e| BitNetError::Validation(e.to_string()))?;
                let tensor = Tensor::from_slice(&f32_data, info.shape.as_slice(), &candle_device)
                    .map_err(|e| BitNetError::Validation(e.to_string()))?;
                return Ok((tensor, None, None));
            }

            // For IQ2_S without FFI support, fail with clear message
            #[cfg(not(feature = "iq2s-ffi"))]
            if matches!(info.tensor_type, GgufTensorType::IQ2_S) {
                return Err(BitNetError::Model(ModelError::InvalidFormat {
                    format: format!(
                        "IQ2_S tensor '{}' found but support not compiled in. \
                        Rebuild with `--features iq2s-ffi` to enable IQ2_S support.",
                        info.name
                    ),
                }));
            }

            // Handle I2_S quantization with native Rust dequantization
            if matches!(info.tensor_type, GgufTensorType::I2_S) {
                use super::types::{I2SFlavor, detect_i2s_flavor};
                use crate::quant::i2s::{self, I2SDequantCfg};

                // PATCH 2: LayerNorm weights should NEVER be quantized - skip I2_S path
                if is_layernorm_weight(&info.name) {
                    return Err(BitNetError::Validation(format!(
                        "LayerNorm weight '{}' should not be quantized with I2_S. \
                        This indicates a corrupted GGUF file. LayerNorm weights must be FP16/FP32.",
                        info.name
                    )));
                }

                // PATCH: Detect I2_S flavor to determine if this is QK256
                let nelems = info.shape.iter().product::<usize>();
                let has_scale_sibling = false; // QK256 has no sibling scale tensor (scales are inline or absent)

                let flavor = detect_i2s_flavor(info, has_scale_sibling, nelems)?;

                // If QK256, preserve raw bytes instead of dequantizing
                if matches!(flavor, I2SFlavor::GgmlQk256NoScale) {
                    tracing::debug!(
                        "Detected QK256 tensor '{}' ({}x{}, {} bytes) - preserving raw bytes",
                        info.name,
                        info.shape[0],
                        info.shape[1],
                        data.len()
                    );

                    // Determine correct orientation for QK256 tensors
                    // QK256 format requires [output_dim, input_dim] layout (one row per output feature)
                    // GGUF may store as [input_dim, output_dim] which needs transposition
                    let (rows, cols) = {
                        let shape_as_is = (info.shape[0], info.shape[1]);
                        let shape_transposed = (info.shape[1], info.shape[0]);

                        // Use tensor name and config to determine expected shape
                        let expected_shape = expected_qk256_shape(&info.name, model_config);

                        if let Some((expected_rows, expected_cols)) = expected_shape {
                            // Check which orientation matches the expected shape
                            if shape_as_is.0 == expected_rows && shape_as_is.1 == expected_cols {
                                tracing::debug!(
                                    "QK256 '{}': using as-is [{}, {}] (matches expected)",
                                    info.name,
                                    shape_as_is.0,
                                    shape_as_is.1
                                );
                                shape_as_is
                            } else if shape_transposed.0 == expected_rows
                                && shape_transposed.1 == expected_cols
                            {
                                tracing::debug!(
                                    "QK256 '{}': using transposed [{}, {}] (matches expected)",
                                    info.name,
                                    shape_transposed.0,
                                    shape_transposed.1
                                );
                                shape_transposed
                            } else {
                                // Fall back to byte-based detection
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
                                detect_qk256_orientation_by_bytes(
                                    shape_as_is,
                                    shape_transposed,
                                    data.len(),
                                )
                            }
                        } else {
                            // No expected shape - use byte-based detection
                            detect_qk256_orientation_by_bytes(
                                shape_as_is,
                                shape_transposed,
                                data.len(),
                            )
                        }
                    };

                    let blocks_per_row = cols.div_ceil(256); // 256 elements per block
                    let row_stride_bytes = blocks_per_row * 64; // 64 bytes per 256-element block

                    // Store raw bytes as U8 tensor [rows, row_stride_bytes]
                    let raw_tensor = Tensor::from_raw_buffer(
                        data,
                        DType::U8,
                        &[rows, row_stride_bytes],
                        &candle_device,
                    )
                    .map_err(|e| BitNetError::Validation(e.to_string()))?;

                    // Generate key for raw_tensors collection
                    let qk256_key = format!("{}.qk256_qs", info.name);

                    // Return placeholder f32 tensor for main collection (will not be used)
                    // We need a valid tensor to satisfy the API, but transformer will use raw_tensors
                    let placeholder = Tensor::zeros(&[rows, cols], DType::F32, &candle_device)
                        .map_err(|e| BitNetError::Validation(e.to_string()))?;

                    tracing::debug!(
                        "QK256 raw tensor stored with key '{}' [shape: {:?}]",
                        qk256_key,
                        raw_tensor.dims()
                    );

                    return Ok((placeholder, Some((qk256_key, raw_tensor)), None));
                }

                // For other I2_S flavors (Split32, Inline), continue with dequantization
                // Select per-tensor I2_S config (policy → heuristic → default)
                let (inv, k, correction_opt) =
                    Self::select_i2s_config(&info.name, Some(data), policy_plan);
                let cfg = I2SDequantCfg { inv, k };

                // Log projection weight RMS after dequant for diagnosis
                let is_proj = is_projection_weight(&info.name);

                // Check for embedding transposition
                if Self::is_embedding_tensor(&info.name)
                    && Self::embedding_is_transposed(&info.shape)
                {
                    info!("Embedding appears transposed ({:?}) -> decoding transposed", info.shape);
                    let f32_data =
                        i2s::dequantize_to_f32_transposed_with_cfg(data, &info.shape, cfg)
                            .map_err(|e| BitNetError::Validation(e.to_string()))?;

                    // Now dims become [vocab, hidden]
                    let (rows, cols) = (info.shape[1], info.shape[0]);
                    let tensor = Tensor::from_slice(&f32_data, &[rows, cols], &candle_device)
                        .map_err(|e| BitNetError::Validation(e.to_string()))?;
                    return Ok((tensor, None, correction_opt));
                } else if Self::is_projection_tensor(&info.name) && info.shape.len() == 2 {
                    // Projection tensors need transposition for linear layer compatibility
                    debug!(
                        "Transposing projection tensor '{}' from {:?} to {:?}",
                        info.name,
                        info.shape,
                        [info.shape[1], info.shape[0]]
                    );
                    let tensor = Self::create_transposed_i2s_tensor_with_cfg(
                        data,
                        &info.shape,
                        &candle_device,
                        cfg,
                    )?;

                    // Log projection RMS for diagnosis
                    if is_proj && let Ok(rms) = Self::rms_f32(&tensor) {
                        debug!(
                            "PROJ load: '{}' dtype=I2_S->F32 shape={:?} rms={:.6} (inv={} k={})",
                            info.name,
                            tensor.dims(),
                            rms,
                            inv,
                            k
                        );
                    }

                    return Ok((tensor, None, correction_opt));
                } else {
                    // Normal I2_S dequantization with config
                    let mut f32_data = i2s::dequantize_to_f32_with_cfg(data, &info.shape, cfg)
                        .map_err(|e| BitNetError::Validation(e.to_string()))?;

                    // Transpose once to [out,in] if this is a projection weight
                    let (mut rows, mut cols) = (info.shape[0], info.shape[1]);
                    let mut want_shape = info.shape.clone();
                    if Self::maybe_transpose_to_out_in(&info.shape, &info.name) {
                        // In-place transpose to save memory allocation
                        // f32_data currently [rows, cols]=[in,out]; flip to [out,in]
                        let original_data = std::mem::take(&mut f32_data);
                        f32_data = Vec::with_capacity(rows * cols);
                        for col in 0..cols {
                            for row in 0..rows {
                                f32_data.push(original_data[row * cols + col]);
                            }
                        }
                        (rows, cols) = (cols, rows);
                        want_shape = vec![rows, cols];
                        tracing::debug!(
                            "pre-transposed {} to [out,in]={:?}",
                            info.name,
                            want_shape
                        );
                    }

                    let tensor =
                        Tensor::from_slice(&f32_data, want_shape.as_slice(), &candle_device)
                            .map_err(|e| BitNetError::Validation(e.to_string()))?;

                    // Log projection RMS for diagnosis
                    if is_proj && let Ok(rms) = Self::rms_f32(&tensor) {
                        debug!(
                            "PROJ load: '{}' dtype=I2_S->F32 shape={:?} rms={:.6} (inv={} k={})",
                            info.name,
                            tensor.dims(),
                            rms,
                            inv,
                            k
                        );
                    }

                    return Ok((tensor, None, correction_opt));
                }
            }

            // For other quantized types, keep as raw bytes for now
            // (would need specific dequantizers for Q4_0, Q8_0, etc.)
            let tensor = Tensor::from_raw_buffer(data, dtype, &info.shape, &candle_device)
                .map_err(|e| BitNetError::Validation(e.to_string()))?;
            Ok((tensor, None, None))
        } else {
            // For regular tensors, interpret the bytes according to the data type
            match dtype {
                DType::F32 => {
                    // PATCH 2: Log layer-0 attention_norm.weight stats for verification
                    if info.name == "layers.0.attention_norm.weight"
                        || info.name == "blk.0.attn_norm.weight"
                    {
                        let float_data = bytemuck::cast_slice::<u8, f32>(data);
                        if !float_data.is_empty() {
                            let sum: f64 = float_data.iter().map(|&x| x as f64).sum();
                            let mean = sum / float_data.len() as f64;
                            let variance: f64 = float_data
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
                    }

                    // Check for embedding transposition
                    let tensor = if Self::is_embedding_tensor(&info.name)
                        && Self::embedding_is_transposed(&info.shape)
                    {
                        info!(
                            "Embedding appears transposed ({:?}) -> decoding transposed",
                            info.shape
                        );
                        let f32_data = Self::transpose_f32_to_f32(data, &info.shape)?;
                        // Now dims become [vocab, hidden]
                        let (rows, cols) = (info.shape[1], info.shape[0]);
                        Tensor::from_slice(&f32_data, &[rows, cols], &candle_device)
                            .map_err(|e| BitNetError::Validation(e.to_string()))?
                    } else if Self::maybe_transpose_to_out_in(&info.shape, &info.name) {
                        // Apply unified transpose logic for F32 projection weights
                        debug!(
                            "pre-transposing F32 projection '{}' from {:?} to [out,in]",
                            info.name, info.shape
                        );
                        let f32_data = Self::transpose_f32_to_f32(data, &info.shape)?;
                        let (rows, cols) = (info.shape[1], info.shape[0]);
                        Tensor::from_slice(&f32_data, &[rows, cols], &candle_device)
                            .map_err(|e| BitNetError::Validation(e.to_string()))?
                    } else {
                        let float_data = bytemuck::cast_slice::<u8, f32>(data);
                        Tensor::from_slice(float_data, info.shape.as_slice(), &candle_device)
                            .map_err(|e| BitNetError::Validation(e.to_string()))?
                    };

                    // PATCH 3: Validate and optionally rescale LayerNorm gamma (policy-driven)
                    if is_layernorm_weight(&info.name) {
                        Self::check_ln_gamma_stats(&info.name, &tensor)?;

                        // Apply policy-driven rescaling first (if configured)
                        let (rescaled, correction1) = Self::maybe_rescale_ln_gamma_with_policy(
                            &info.name,
                            tensor,
                            policy_plan,
                        )?;

                        // Apply experimental sqrt(hidden_size) rescaling (if enabled)
                        let (final_tensor, correction2) =
                            Self::maybe_rescale_gamma_by_sqrt_hidden(&info.name, rescaled)?;

                        // Prefer correction2 if both are present, otherwise use correction1
                        let final_correction = correction2.or(correction1);
                        Ok((final_tensor, None, final_correction))
                    } else {
                        // Log projection RMS for F32 projections
                        if is_projection_weight(&info.name)
                            && let Ok(rms) = Self::rms_f32(&tensor)
                        {
                            info!(
                                "PROJ load: '{}' dtype=F32 shape={:?} rms={:.6}",
                                info.name,
                                tensor.dims(),
                                rms
                            );
                        }
                        Ok((tensor, None, None))
                    }
                }
                DType::F16 => {
                    // Check for embedding transposition
                    let tensor = if Self::is_embedding_tensor(&info.name)
                        && Self::embedding_is_transposed(&info.shape)
                    {
                        info!(
                            "Embedding appears transposed ({:?}) -> decoding transposed",
                            info.shape
                        );
                        let f32_data = Self::transpose_f16_to_f32(data, &info.shape)?;
                        // Now dims become [vocab, hidden]
                        let (rows, cols) = (info.shape[1], info.shape[0]);
                        Tensor::from_slice(&f32_data, &[rows, cols], &candle_device)
                            .map_err(|e| BitNetError::Validation(e.to_string()))?
                    } else if Self::maybe_transpose_to_out_in(&info.shape, &info.name) {
                        // Apply unified transpose logic for F16 projection weights
                        debug!(
                            "pre-transposing F16 projection '{}' from {:?} to [out,in]",
                            info.name, info.shape
                        );
                        let f32_data = Self::transpose_f16_to_f32(data, &info.shape)?;
                        let (rows, cols) = (info.shape[1], info.shape[0]);
                        Tensor::from_slice(&f32_data, &[rows, cols], &candle_device)
                            .map_err(|e| BitNetError::Validation(e.to_string()))?
                    } else {
                        // For now, convert F16 data to F32 for compatibility
                        let half_data = bytemuck::cast_slice::<u8, u16>(data);
                        let float_data: Vec<f32> =
                            half_data.iter().map(|&h| half::f16::from_bits(h).to_f32()).collect();
                        Tensor::from_slice(&float_data, info.shape.as_slice(), &candle_device)
                            .map_err(|e| BitNetError::Validation(e.to_string()))?
                    };

                    // PATCH 3: Validate and optionally rescale LayerNorm gamma (policy-driven)
                    if is_layernorm_weight(&info.name) {
                        Self::check_ln_gamma_stats(&info.name, &tensor)?;

                        // Apply policy-driven rescaling first (if configured)
                        let (rescaled, correction1) = Self::maybe_rescale_ln_gamma_with_policy(
                            &info.name,
                            tensor,
                            policy_plan,
                        )?;

                        // Apply experimental sqrt(hidden_size) rescaling (if enabled)
                        let (final_tensor, correction2) =
                            Self::maybe_rescale_gamma_by_sqrt_hidden(&info.name, rescaled)?;

                        // Prefer correction2 if both are present, otherwise use correction1
                        let final_correction = correction2.or(correction1);
                        Ok((final_tensor, None, final_correction))
                    } else {
                        // Log projection RMS for F16→F32 projections
                        if is_projection_weight(&info.name)
                            && let Ok(rms) = Self::rms_f32(&tensor)
                        {
                            info!(
                                "PROJ load: '{}' dtype=F16->F32 shape={:?} rms={:.6}",
                                info.name,
                                tensor.dims(),
                                rms
                            );
                        }
                        Ok((tensor, None, None))
                    }
                }
                _ => Err(BitNetError::Model(ModelError::InvalidFormat {
                    format: format!("Unsupported data type: {:?}", dtype),
                })),
            }
        }
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
