//! BitNet model implementation

use crate::dense_gguf_q8_dispatch::{DenseQ8DispatchSelection, select_dense_q8_runtime};
use crate::dense_gguf_q8_sidecar::{DenseGgufQ8SidecarDescriptor, DenseGgufQ8SidecarRegistry};
use crate::transformer::{KVCache, TransformerModel};
use bitnet_common::{
    BitNetConfig, BitNetError, BitNetTensor, ConcreteTensor, Device, Result, Tensor,
};
use bitnet_transformer::{
    DenseLinearPackedQ8Payload, DenseLinearRuntimeHookDescriptor, DenseLinearRuntimeHookRegistry,
};
use candle_core::{DType, Tensor as CandleTensor};
use std::collections::HashMap;
use std::sync::Arc;

/// Trait for BitNet models
pub trait Model: Send + Sync {
    fn config(&self) -> &BitNetConfig;
    fn forward(
        &self,
        input: &ConcreteTensor,
        cache: &mut dyn std::any::Any,
    ) -> Result<ConcreteTensor>;
    fn embed(&self, tokens: &[u32]) -> Result<ConcreteTensor>;
    fn logits(&self, hidden: &ConcreteTensor) -> Result<ConcreteTensor>;
    fn dense_q8_hook_selection_receipt(&self) -> serde_json::Value {
        serde_json::json!({
            "schema": 1,
            "artifact_kind": "dense_gguf_q8_hook_selection_receipt_gate",
            "selected_path": "eager_f32_candle",
            "selected_kernel": "dense-f32-candle-linear",
            "sidecar_descriptor_count": 0,
            "hook_boundary_count": 0,
            "sidecar_descriptors_reached_transformer": false,
            "example_boundary": serde_json::Value::Null,
            "all_boundaries_preserve_eager_f32": true,
            "runtime_compute_enabled": false,
            "dense_runtime_replaced": false,
            "speedup_claim": false,
            "fallback_used": false,
            "remaining_blockers": [
                "dense_q8_sidecar_registry_missing_or_not_exposed_for_model_trait_object"
            ],
            "next_safe_step": "use a strict dense GGUF model with carried Q8_0 sidecar descriptors before evaluating packed sidecar compute"
        })
    }
}

/// BitNet model implementation
pub struct BitNetModel {
    config: BitNetConfig,
    device: Device,
    tensors: HashMap<String, CandleTensor>,
    transformer: Option<Arc<TransformerModel>>,
    dense_q8_sidecars: DenseGgufQ8SidecarRegistry,
}

impl BitNetModel {
    pub fn new(config: BitNetConfig, device: Device) -> Self {
        Self {
            config,
            device,
            tensors: HashMap::new(),
            transformer: None,
            dense_q8_sidecars: DenseGgufQ8SidecarRegistry::default(),
        }
    }

    /// Create a BitNet model from GGUF tensors
    pub fn from_gguf(
        config: BitNetConfig,
        tensors: HashMap<String, CandleTensor>,
        raw_tensors: HashMap<String, CandleTensor>,
        device: Device,
    ) -> Result<Self> {
        Self::from_gguf_with_dense_q8_sidecars(
            config,
            tensors,
            raw_tensors,
            DenseGgufQ8SidecarRegistry::default(),
            device,
        )
    }

    /// Create a BitNet model from GGUF tensors and inert dense Q8 sidecar metadata.
    pub fn from_gguf_with_dense_q8_sidecars(
        config: BitNetConfig,
        tensors: HashMap<String, CandleTensor>,
        raw_tensors: HashMap<String, CandleTensor>,
        dense_q8_sidecars: DenseGgufQ8SidecarRegistry,
        device: Device,
    ) -> Result<Self> {
        tracing::debug!(
            "from_gguf: received config: hidden={}, n_heads={}, n_kv_heads={}",
            config.model.hidden_size,
            config.model.num_heads,
            config.model.num_key_value_heads
        );
        tracing::debug!(
            "from_gguf: received {} tensors, {} raw QK256 tensors",
            tensors.len(),
            raw_tensors.len()
        );
        if !dense_q8_sidecars.is_empty() {
            tracing::debug!(
                "from_gguf: carrying {} inert dense Q8_0 sidecar descriptors; eager F32 runtime remains selected",
                dense_q8_sidecars.descriptor_count()
            );
        }

        // Validate that required tensors are present
        // LM head can be tied to embeddings, so check for either output.weight or embeddings
        let has_output = tensors.contains_key("output.weight")
            || tensors.contains_key("lm_head.weight")
            || tensors.contains_key("head.weight");

        let has_embeddings = tensors.contains_key("token_embd.weight")
            || tensors.contains_key("tok_embeddings.weight")
            || tensors.contains_key("model.embed_tokens.weight");

        if !has_embeddings {
            return Err(BitNetError::Validation(
                "Missing required tensor: token embeddings (token_embd.weight or equivalent)"
                    .to_string(),
            ));
        }

        if !has_output && !has_embeddings {
            return Err(BitNetError::Validation(
                "Missing both output.weight and token_embd.weight - cannot compute logits"
                    .to_string(),
            ));
        }

        // Try to build transformer model; propagate errors so missing weights fail fast
        let transformer =
            Self::build_transformer(&config, &tensors, &raw_tensors, &dense_q8_sidecars, &device)?;

        Ok(Self { config, device, tensors, transformer: Some(transformer), dense_q8_sidecars })
    }

    /// Build transformer model from loaded tensors
    fn build_transformer(
        config: &BitNetConfig,
        tensors: &HashMap<String, CandleTensor>,
        raw_tensors: &HashMap<String, CandleTensor>,
        dense_q8_sidecars: &DenseGgufQ8SidecarRegistry,
        device: &Device,
    ) -> Result<Arc<TransformerModel>> {
        use crate::weight_mapper::{
            create_var_builder, normalize_model_tensors, remap_gguf_weights,
        };

        // Create a VarBuilder that uses our loaded tensors
        let device = device.to_candle().map_err(|e| BitNetError::Validation(e.to_string()))?;

        if tensors.is_empty() {
            return Err(BitNetError::Validation("No model tensors provided".to_string()));
        }

        // Remap tensor names to match our transformer module structure
        let mut mapped = remap_gguf_weights(tensors)?;

        // Normalize embeddings, lm_head, and all layer tensors, detect vocab size and hidden size
        let (detected_vocab, detected_hidden) = normalize_model_tensors(&mut mapped, config)?;

        // Update config with detected values
        let mut updated_config = config.clone();
        if updated_config.model.vocab_size != detected_vocab {
            tracing::info!(
                "Updating vocab_size from {} to {} based on tensor shapes",
                updated_config.model.vocab_size,
                detected_vocab
            );
            updated_config.model.vocab_size = detected_vocab;
        }
        if updated_config.model.hidden_size != detected_hidden {
            tracing::info!(
                "Updating hidden_size from {} to {} based on tensor shapes",
                updated_config.model.hidden_size,
                detected_hidden
            );
            updated_config.model.hidden_size = detected_hidden;
        }

        // Remap raw_tensors keys (QK256 tensors) to match transformer structure
        // Keys like "blk.0.attn_q.weight.qk256_qs" -> "layers.0.attention.q_proj.weight.qk256_qs"
        // The remapper now handles .qk256_qs suffix (strips, remaps, re-appends)
        let raw_mapped = remap_gguf_weights(raw_tensors)?;
        let dense_linear_hooks = dense_q8_runtime_hooks_from_sidecars(dense_q8_sidecars);

        let vb = create_var_builder(mapped.clone(), DType::F32, &device)?;
        let model = TransformerModel::new_with_tensors_and_dense_linear_hooks(
            updated_config,
            vb,
            raw_mapped,
            dense_linear_hooks,
        )?;
        Ok(Arc::new(model))
    }

    /// Get a tensor by name
    pub fn get_tensor(&self, name: &str) -> Option<&CandleTensor> {
        self.tensors.get(name)
    }

    /// List all tensor names
    pub fn tensor_names(&self) -> Vec<&String> {
        self.tensors.keys().collect()
    }

    /// Inert dense Q8_0 sidecar descriptors carried from GGUF loading.
    pub fn dense_q8_sidecars(&self) -> &DenseGgufQ8SidecarRegistry {
        &self.dense_q8_sidecars
    }

    /// Behavior-preserving dense Q8_0 dispatch selector.
    ///
    /// This currently always selects the eager F32 Candle path. Packed Q8_0
    /// sidecar descriptors are exposed only as unavailable candidates until a
    /// later slice proves generated-ID/text and strict-receipt equivalence.
    pub fn dense_q8_dispatch_selection(&self, tensor_name: &str) -> DenseQ8DispatchSelection {
        select_dense_q8_runtime(tensor_name, &self.dense_q8_sidecars)
    }

    /// Receipt-oriented dense-linear hook selection identity.
    ///
    /// This is intentionally observational. It records which Q8_0 sidecar
    /// descriptors reached the transformer hook boundary while preserving the
    /// eager F32 Candle runtime path until behavior-preserving after receipts
    /// and a packed compute kernel exist.
    pub fn dense_q8_hook_selection_receipt(&self) -> serde_json::Value {
        let hook_boundaries = self
            .transformer
            .as_ref()
            .map(|transformer| transformer.dense_linear_runtime_hook_boundaries())
            .unwrap_or_default();
        let example = hook_boundaries.first();
        let payload_boundary =
            hook_boundaries.iter().find(|boundary| boundary.sidecar_payload_bytes_available);
        let payload_bearing_boundary_count = hook_boundaries
            .iter()
            .filter(|boundary| boundary.sidecar_payload_bytes_available)
            .count();
        let sidecar_descriptor_count = self.dense_q8_sidecars.descriptor_count();
        let hook_boundary_count = hook_boundaries.len();
        let sidecar_descriptors_reached_transformer = hook_boundary_count > 0
            && hook_boundaries.iter().any(|boundary| {
                boundary.sidecar_descriptor_present && boundary.preserves_eager_f32()
            });
        let boundary_json = |boundary: &bitnet_transformer::DenseLinearRuntimeHookBoundary| {
            serde_json::json!({
                "tensor_name": boundary.tensor_name,
                "selected_path": boundary.selected_path,
                "selected_kernel": boundary.selected_kernel,
                "sidecar_descriptor_present": boundary.sidecar_descriptor_present,
                "sidecar_role": boundary.sidecar_role,
                "sidecar_payload_sha256": boundary.sidecar_payload_sha256,
                "sidecar_payload_bytes_available": boundary.sidecar_payload_bytes_available,
                "sidecar_payload_bytes": boundary.sidecar_payload_bytes,
                "sidecar_q8_block_count": boundary.sidecar_q8_block_count,
                "sidecar_matrix_rows": boundary.sidecar_matrix_rows,
                "sidecar_matrix_cols": boundary.sidecar_matrix_cols,
                "sidecar_payload_contract_valid": boundary.sidecar_payload_contract_valid,
                "runtime_compute_enabled": boundary.runtime_compute_enabled,
                "eager_f32_runtime_preserved": boundary.eager_f32_runtime_preserved,
                "dense_runtime_replaced": boundary.dense_runtime_replaced,
                "speedup_claim": boundary.speedup_claim,
                "generated_id_preservation_required_before_runtime_use": boundary.generated_id_preservation_required_before_runtime_use,
                "next_receipt_gate": boundary.next_receipt_gate,
            })
        };

        serde_json::json!({
            "schema": 1,
            "artifact_kind": "dense_gguf_q8_hook_selection_receipt_gate",
            "selected_path": "eager_f32_candle",
            "selected_kernel": "dense-f32-candle-linear",
            "sidecar_descriptor_count": sidecar_descriptor_count,
            "hook_boundary_count": hook_boundary_count,
            "sidecar_descriptors_reached_transformer": sidecar_descriptors_reached_transformer,
            "example_boundary": example.map(boundary_json),
            "payload_bearing_boundary_count": payload_bearing_boundary_count,
            "payload_bearing_boundary": payload_boundary.map(boundary_json),
            "all_boundaries_preserve_eager_f32": hook_boundaries.iter().all(|boundary| boundary.preserves_eager_f32()),
            "runtime_compute_enabled": false,
            "dense_runtime_replaced": false,
            "speedup_claim": false,
            "fallback_used": false,
            "remaining_blockers": [
                "packed_q8_sidecar_compute_kernel_not_enabled",
                "before_after_qwen3_q8_warm_session_receipts_with_hook_selection_identity_missing"
            ],
            "next_safe_step": "run before/after Qwen3 Q8_0 warm-session receipts that record identical model SHA, tokenizer source/strictness, prompt IDs, generated IDs, decoded text, selected CPU backend/kernel identity, dense hook selection identity, and fallback_used=false before enabling packed sidecar compute"
        })
    }

    /// Convert ConcreteTensor to Candle tensor
    fn to_candle_tensor(&self, tensor: &ConcreteTensor) -> Result<CandleTensor> {
        match tensor {
            ConcreteTensor::BitNet(t) => t.to_candle(),
            ConcreteTensor::Mock(mock) => {
                // Create a dummy tensor for mock
                let shape = mock.shape();
                let device =
                    self.device.to_candle().map_err(|e| BitNetError::Validation(e.to_string()))?;
                Ok(CandleTensor::zeros(shape, DType::F32, &device)?)
            }
        }
    }

    /// Convert Candle tensor to ConcreteTensor
    fn candle_to_concrete(&self, tensor: CandleTensor) -> ConcreteTensor {
        ConcreteTensor::BitNet(BitNetTensor::new(tensor))
    }
}

fn dense_q8_runtime_hooks_from_sidecars(
    registry: &DenseGgufQ8SidecarRegistry,
) -> DenseLinearRuntimeHookRegistry {
    let mut hooks = DenseLinearRuntimeHookRegistry::default();
    for descriptor in &registry.descriptors {
        let Some(canonical_name) = dense_q8_transformer_hook_name(descriptor) else {
            continue;
        };
        hooks.insert(
            canonical_name,
            DenseLinearRuntimeHookDescriptor {
                tensor_name: descriptor.tensor_name.clone(),
                role: format!("{:?}", descriptor.role),
                sidecar_payload_sha256: Some(descriptor.packed_q8_bytes_sha256.clone()),
                packed_q8_payload: dense_q8_packed_payload_from_sidecar(descriptor),
                runtime_compute_enabled: descriptor.runtime_compute_enabled,
            },
        );
    }
    hooks
}

fn dense_q8_packed_payload_from_sidecar(
    descriptor: &DenseGgufQ8SidecarDescriptor,
) -> Option<DenseLinearPackedQ8Payload> {
    let [matrix_rows, matrix_cols]: [usize; 2] =
        descriptor.runtime_candle_shape.as_slice().try_into().ok()?;
    Some(DenseLinearPackedQ8Payload {
        tensor_name: descriptor.tensor_name.clone(),
        packed_q8_bytes: descriptor.packed_q8_bytes.clone()?,
        q8_block_size: descriptor.q8_block_size,
        q8_block_count: descriptor.q8_block_count,
        matrix_rows,
        matrix_cols,
    })
}

fn dense_q8_transformer_hook_name(descriptor: &DenseGgufQ8SidecarDescriptor) -> Option<String> {
    bitnet_weight_name_core::normalize_vendor_key(&descriptor.tensor_name).or_else(|| {
        match descriptor.tensor_name.as_str() {
            "output.weight" | "lm_head.weight" => Some("lm_head.weight".to_string()),
            "token_embd.weight" | "model.embed_tokens.weight" => {
                Some("embed_tokens.weight".to_string())
            }
            _ => None,
        }
    })
}

#[cfg(test)]
mod dense_q8_runtime_hook_tests {
    use super::*;
    use crate::formats::gguf::{GgufTensorType, TensorInfo};

    fn q8_tensor_info(name: &str, shape: Vec<usize>) -> TensorInfo {
        let value_count = shape.iter().copied().product::<usize>();
        let block_count = value_count.div_ceil(GgufTensorType::Q8_0.block_size());
        let size = block_count * GgufTensorType::Q8_0.element_size();
        TensorInfo {
            name: name.to_string(),
            shape,
            tensor_type: GgufTensorType::Q8_0,
            offset: 0,
            size: size as u64,
        }
    }

    #[test]
    fn dense_gguf_q8_sidecar_hooks_map_vendor_tensor_names() -> Result<()> {
        let mut registry = DenseGgufQ8SidecarRegistry::default();
        let tensor_info = q8_tensor_info("blk.0.attn_q.weight", vec![64, 32]);
        let data = vec![0_u8; tensor_info.size as usize];
        registry.try_push_tensor(&tensor_info, &data)?;

        let hooks = dense_q8_runtime_hooks_from_sidecars(&registry);
        let Some(hook) = hooks.get("layers.0.attention.q_proj.weight") else {
            return Err(BitNetError::Validation(
                "expected canonical attention q_proj hook".to_string(),
            ));
        };

        assert_eq!(hook.tensor_name, "blk.0.attn_q.weight");
        assert_eq!(hook.role, "AttentionQ");
        assert_eq!(hook.runtime_compute_enabled, false);
        assert!(hook.sidecar_payload_sha256.is_some());
        Ok(())
    }

    #[test]
    fn dense_gguf_q8_sidecar_hook_can_carry_one_real_payload_candidate() -> Result<()> {
        let mut registry = DenseGgufQ8SidecarRegistry::default();
        let tensor_info = q8_tensor_info("blk.0.attn_q.weight", vec![64, 32]);
        let data = vec![7_u8; tensor_info.size as usize];
        registry.try_push_tensor_with_payload_candidate(
            &tensor_info,
            &data,
            Some("blk.0.attn_q.weight"),
        )?;

        let hooks = dense_q8_runtime_hooks_from_sidecars(&registry);
        let Some(hook) = hooks.get("layers.0.attention.q_proj.weight") else {
            return Err(BitNetError::Validation(
                "expected canonical attention q_proj hook".to_string(),
            ));
        };
        let Some(payload) = hook.packed_q8_payload.as_ref() else {
            return Err(BitNetError::Validation("expected packed payload".to_string()));
        };

        assert_eq!(payload.tensor_name, "blk.0.attn_q.weight");
        assert_eq!(payload.payload_len(), tensor_info.size as usize);
        assert_eq!(payload.q8_block_size, 32);
        assert_eq!(payload.q8_block_count, 64);
        assert_eq!(payload.matrix_rows, 32);
        assert_eq!(payload.matrix_cols, 64);
        assert!(payload.shape_matches_matvec_contract());
        assert!(payload.payload_len_matches_contract());
        assert!(!hook.runtime_compute_enabled);
        Ok(())
    }
}

impl Model for BitNetModel {
    fn config(&self) -> &BitNetConfig {
        &self.config
    }

    fn forward(
        &self,
        input: &ConcreteTensor,
        cache: &mut dyn std::any::Any,
    ) -> Result<ConcreteTensor> {
        // Fail fast if transformer not initialized - prevents silent zero-logit failures
        let transformer = self.transformer.as_ref().ok_or_else(|| {
            BitNetError::Model(bitnet_common::ModelError::LoadingFailed {
                reason: "BitNetModel::transformer not initialized (GGUF load failed or build_transformer returned error)".to_string()
            })
        })?;

        // Get or create KV cache
        let kv_cache = cache.downcast_mut::<KVCache>();

        // Convert input to Candle tensor
        let input_tensor = self.to_candle_tensor(input)?;

        // Run transformer forward pass (passes ownership to avoid clone on hot path)
        let output = transformer.forward(input_tensor, kv_cache)?;

        // Convert back to ConcreteTensor
        Ok(self.candle_to_concrete(output))
    }

    fn embed(&self, tokens: &[u32]) -> Result<ConcreteTensor> {
        // Fail fast if transformer not initialized
        let transformer = self.transformer.as_ref().ok_or_else(|| {
            BitNetError::Model(bitnet_common::ModelError::LoadingFailed {
                reason: "BitNetModel::transformer not initialized (cannot embed tokens)"
                    .to_string(),
            })
        })?;

        let embedded = transformer.embed(tokens)?;
        Ok(self.candle_to_concrete(embedded))
    }

    fn logits(&self, hidden: &ConcreteTensor) -> Result<ConcreteTensor> {
        // Fail fast if transformer not initialized
        let transformer = self.transformer.as_ref().ok_or_else(|| {
            BitNetError::Model(bitnet_common::ModelError::LoadingFailed {
                reason: "BitNetModel::transformer not initialized (cannot compute logits)"
                    .to_string(),
            })
        })?;

        let hidden_tensor = self.to_candle_tensor(hidden)?;
        let logits = transformer.logits(&hidden_tensor)?;
        Ok(self.candle_to_concrete(logits))
    }

    fn dense_q8_hook_selection_receipt(&self) -> serde_json::Value {
        BitNetModel::dense_q8_hook_selection_receipt(self)
    }
}
