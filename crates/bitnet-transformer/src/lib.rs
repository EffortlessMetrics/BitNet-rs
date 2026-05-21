#[cfg(test)]
use bitnet_common::config::NormType;
#[cfg(test)]
use bitnet_common::dtype_convert::f32_to_fp16;
use bitnet_common::dtype_convert::fp16_to_f32;
use bitnet_common::{BitNetConfig, BitNetError, Result, config::ActivationType};
use bitnet_qk256_dispatch::{
    forward_qk256_with_scale, record_bitnet_linear_cpu_fallback, record_bitnet_linear_unsupported,
    strict_cuda_bitnet_backend_requested,
};
use bitnet_rope::{build_tables as build_rope_tables, resolve_base as resolve_rope_base};
use candle_core::{DType, Device, Module, Tensor};
use candle_nn::{LayerNorm, Linear, VarBuilder};
mod attention_forward;

mod diagnostics;
mod layer_builders;
mod qk256;

use diagnostics::{
    dbg_finite, dbg_stats, debug_attn_enabled, debug_attn_scale_enabled, debug_gqa_enabled,
    debug_mlp_enabled, debug_rmsnorm_enabled, debug_rope_enabled, qwen_trace_event,
    qwen_trace_layer_enabled, qwen_trace_number, qwen_trace_tensor, trace_rms_enabled,
};
#[cfg(test)]
use layer_builders::layer_norm_with_optional_bias;
use layer_builders::{
    linear_with_optional_bias, norm_with_optional_bias, optional_layer_norm_with_optional_bias,
};
use qk256::{TIED_EMBED_QK256_KEY, qk256_inline_scale};
use std::collections::HashMap;

pub type DenseLinearRuntimeHookRegistry = HashMap<String, DenseLinearRuntimeHookDescriptor>;
const SLM_CPU_067_EXACT_Q8_RUNTIME_TENSOR: &str = "layers.0.attention.q_proj.weight";

/// Evidence-scoped packed Q8_0 payload for a dense-linear runtime hook.
///
/// This carries bytes only when a caller intentionally wires one tensor path
/// for before/after proof. Runtime compute stays disabled until receipts prove
/// generated-ID/text preservation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DenseLinearPackedQ8Payload {
    pub tensor_name: String,
    pub packed_q8_bytes: std::sync::Arc<[u8]>,
    pub q8_block_size: usize,
    pub q8_block_count: usize,
    pub matrix_rows: usize,
    pub matrix_cols: usize,
}

impl DenseLinearPackedQ8Payload {
    pub fn payload_len(&self) -> usize {
        self.packed_q8_bytes.len()
    }

    pub fn expected_q8_payload_len(&self) -> Option<usize> {
        self.q8_block_count.checked_mul(2 + self.q8_block_size)
    }

    pub fn shape_matches_matvec_contract(&self) -> bool {
        self.matrix_rows > 0
            && self.matrix_cols > 0
            && self.q8_block_size == 32
            && self
                .matrix_rows
                .checked_mul(self.matrix_cols)
                .is_some_and(|values| self.q8_block_count == values.div_ceil(self.q8_block_size))
    }

    pub fn payload_len_matches_contract(&self) -> bool {
        self.expected_q8_payload_len().is_some_and(|expected| expected == self.payload_len())
    }
}

/// Descriptor passed from model loading into transformer dense-linear calls.
///
/// Production loading may still pass metadata-only descriptors. A later
/// evidence-scoped slice can attach `packed_q8_payload` for exactly one tensor
/// path, but this descriptor still does not enable packed compute by itself.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DenseLinearRuntimeHookDescriptor {
    pub tensor_name: String,
    pub role: String,
    pub sidecar_payload_sha256: Option<String>,
    pub packed_q8_payload: Option<DenseLinearPackedQ8Payload>,
    pub runtime_compute_enabled: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DenseLinearRuntimeHookBoundary {
    pub tensor_name: String,
    pub selected_path: &'static str,
    pub selected_kernel: &'static str,
    pub sidecar_descriptor_present: bool,
    pub sidecar_role: Option<String>,
    pub sidecar_payload_sha256: Option<String>,
    pub sidecar_payload_bytes_available: bool,
    pub sidecar_payload_bytes: Option<usize>,
    pub sidecar_q8_block_count: Option<usize>,
    pub sidecar_matrix_rows: Option<usize>,
    pub sidecar_matrix_cols: Option<usize>,
    pub sidecar_payload_contract_valid: bool,
    pub runtime_compute_enabled: bool,
    pub eager_f32_runtime_preserved: bool,
    pub dense_runtime_replaced: bool,
    pub speedup_claim: bool,
    pub generated_id_preservation_required_before_runtime_use: bool,
    pub next_receipt_gate: &'static str,
}

impl DenseLinearRuntimeHookBoundary {
    pub fn eager_f32(tensor_name: impl Into<String>) -> Self {
        Self {
            tensor_name: tensor_name.into(),
            selected_path: "eager_f32_candle",
            selected_kernel: "dense-f32-candle-linear",
            sidecar_descriptor_present: false,
            sidecar_role: None,
            sidecar_payload_sha256: None,
            sidecar_payload_bytes_available: false,
            sidecar_payload_bytes: None,
            sidecar_q8_block_count: None,
            sidecar_matrix_rows: None,
            sidecar_matrix_cols: None,
            sidecar_payload_contract_valid: false,
            runtime_compute_enabled: false,
            eager_f32_runtime_preserved: true,
            dense_runtime_replaced: false,
            speedup_claim: false,
            generated_id_preservation_required_before_runtime_use: true,
            next_receipt_gate: "before_after_qwen3_q8_generated_id_text_receipts",
        }
    }

    pub fn from_sidecar_descriptor(
        tensor_name: impl Into<String>,
        descriptor: &DenseLinearRuntimeHookDescriptor,
    ) -> Self {
        let tensor_name = tensor_name.into();
        let payload = descriptor.packed_q8_payload.as_ref();
        let payload_contract_valid = payload.is_some_and(|payload| {
            payload.tensor_name == descriptor.tensor_name
                && payload.shape_matches_matvec_contract()
                && payload.payload_len_matches_contract()
        });
        let runtime_compute_enabled = descriptor.runtime_compute_enabled
            && tensor_name == SLM_CPU_067_EXACT_Q8_RUNTIME_TENSOR
            && payload_contract_valid;
        Self {
            tensor_name,
            selected_path: if runtime_compute_enabled {
                "packed_q8_sidecar"
            } else {
                "eager_f32_candle"
            },
            selected_kernel: if runtime_compute_enabled {
                "dense-q8-sidecar-linear"
            } else {
                "dense-f32-candle-linear"
            },
            sidecar_descriptor_present: true,
            sidecar_role: Some(descriptor.role.clone()),
            sidecar_payload_sha256: descriptor.sidecar_payload_sha256.clone(),
            sidecar_payload_bytes_available: payload.is_some(),
            sidecar_payload_bytes: payload.map(DenseLinearPackedQ8Payload::payload_len),
            sidecar_q8_block_count: payload.map(|payload| payload.q8_block_count),
            sidecar_matrix_rows: payload.map(|payload| payload.matrix_rows),
            sidecar_matrix_cols: payload.map(|payload| payload.matrix_cols),
            sidecar_payload_contract_valid: payload_contract_valid,
            runtime_compute_enabled,
            eager_f32_runtime_preserved: !runtime_compute_enabled,
            dense_runtime_replaced: runtime_compute_enabled,
            speedup_claim: false,
            generated_id_preservation_required_before_runtime_use: true,
            next_receipt_gate: "before_after_qwen3_q8_generated_id_text_receipts",
        }
    }

    pub fn preserves_eager_f32(&self) -> bool {
        self.selected_path == "eager_f32_candle"
            && self.selected_kernel == "dense-f32-candle-linear"
            && !self.runtime_compute_enabled
            && self.eager_f32_runtime_preserved
            && !self.dense_runtime_replaced
            && !self.speedup_claim
    }
}

fn dense_linear_runtime_hook_boundary(
    tensor_name: &str,
    hooks: &DenseLinearRuntimeHookRegistry,
) -> DenseLinearRuntimeHookBoundary {
    hooks
        .get(tensor_name)
        .map(|descriptor| {
            DenseLinearRuntimeHookBoundary::from_sidecar_descriptor(tensor_name, descriptor)
        })
        .unwrap_or_else(|| DenseLinearRuntimeHookBoundary::eager_f32(tensor_name))
}

fn maybe_forward_dense_q8_sidecar_linear(
    input: &Tensor,
    linear: &Linear,
    tensor_name: &str,
    hooks: &DenseLinearRuntimeHookRegistry,
) -> Result<Option<Tensor>> {
    let Some(descriptor) = hooks.get(tensor_name) else {
        return Ok(None);
    };
    let boundary = DenseLinearRuntimeHookBoundary::from_sidecar_descriptor(tensor_name, descriptor);
    if !boundary.runtime_compute_enabled {
        return Ok(None);
    }
    let Some(payload) = descriptor.packed_q8_payload.as_ref() else {
        return Err(BitNetError::Validation(format!(
            "packed Q8 runtime hook for {tensor_name} was enabled without payload bytes"
        )));
    };
    if tensor_name != SLM_CPU_067_EXACT_Q8_RUNTIME_TENSOR {
        return Err(BitNetError::Validation(format!(
            "packed Q8 runtime hook is scoped to {SLM_CPU_067_EXACT_Q8_RUNTIME_TENSOR}, got {tensor_name}"
        )));
    }
    if !payload.shape_matches_matvec_contract() || !payload.payload_len_matches_contract() {
        return Err(BitNetError::Validation(format!(
            "packed Q8 runtime hook payload contract is invalid for {tensor_name}"
        )));
    }
    if linear.weight().dims() != [payload.matrix_rows, payload.matrix_cols] {
        return Err(BitNetError::Validation(format!(
            "packed Q8 runtime hook shape {:?} does not match Candle linear weight {:?} for {tensor_name}",
            [payload.matrix_rows, payload.matrix_cols],
            linear.weight().dims()
        )));
    }

    dense_q8_sidecar_linear_forward(input, linear.bias(), payload)
        .map(Some)
        .map_err(BitNetError::from)
}

fn dense_q8_sidecar_linear_forward(
    input: &Tensor,
    bias: Option<&Tensor>,
    payload: &DenseLinearPackedQ8Payload,
) -> candle_core::Result<Tensor> {
    let dims = input.dims();
    let Some((&input_cols, prefix)) = dims.split_last() else {
        candle_core::bail!("packed Q8 runtime hook requires a tensor with at least one dimension");
    };
    if input_cols != payload.matrix_cols {
        candle_core::bail!(
            "packed Q8 runtime hook input cols {} do not match payload matrix cols {}",
            input_cols,
            payload.matrix_cols
        );
    }
    let input_values = input.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    if input_values.len() % payload.matrix_cols != 0 {
        candle_core::bail!(
            "packed Q8 runtime hook input value count {} is not divisible by cols {}",
            input_values.len(),
            payload.matrix_cols
        );
    }

    let bias_values = match bias {
        Some(bias) => Some(bias.to_dtype(DType::F32)?.to_vec1::<f32>()?),
        None => None,
    };
    if let Some(bias_values) = bias_values.as_ref()
        && bias_values.len() != payload.matrix_rows
    {
        candle_core::bail!(
            "packed Q8 runtime hook bias length {} does not match rows {}",
            bias_values.len(),
            payload.matrix_rows
        );
    }

    let mut output = Vec::with_capacity(
        input_values
            .len()
            .checked_div(payload.matrix_cols)
            .unwrap_or(0)
            .saturating_mul(payload.matrix_rows),
    );
    for input_row in input_values.chunks_exact(payload.matrix_cols) {
        for row in 0..payload.matrix_rows {
            let mut sum = bias_values.as_ref().map_or(0.0, |bias| bias[row]);
            let row_start = row * payload.matrix_cols;
            for (col, input_value) in input_row.iter().enumerate() {
                let weight_idx = row_start + col;
                let block_offset =
                    (weight_idx / payload.q8_block_size) * (2 + payload.q8_block_size);
                let scale_bits = u16::from_le_bytes([
                    payload.packed_q8_bytes[block_offset],
                    payload.packed_q8_bytes[block_offset + 1],
                ]);
                let scale = fp16_to_f32(scale_bits);
                let q = payload.packed_q8_bytes
                    [block_offset + 2 + (weight_idx % payload.q8_block_size)]
                    as i8;
                sum += scale * f32::from(q) * *input_value;
            }
            output.push(sum);
        }
    }

    let mut output_shape = prefix.to_vec();
    output_shape.push(payload.matrix_rows);
    Tensor::from_vec(output, output_shape, input.device())
}

fn attention_f16_dot_input(tensor: &Tensor) -> Result<Tensor> {
    Ok(tensor.to_dtype(DType::F16)?.to_dtype(DType::F32)?)
}

fn attention_score_key_input(tensor: &Tensor) -> Result<Tensor> {
    Ok(tensor.to_dtype(DType::F32)?)
}

/// Rotary Position Embedding
pub struct RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
}

impl RotaryEmbedding {
    pub fn new(
        dim: usize,
        max_seq_len: usize,
        rope_theta: Option<f32>,
        device: &Device,
    ) -> Result<Self> {
        let theta = resolve_rope_base(rope_theta);
        let tables = build_rope_tables(dim, max_seq_len, theta)
            .map_err(|err| BitNetError::Validation(format!("invalid RoPE configuration: {err}")))?;
        let bitnet_rope::RopeTables { half_dim, sin, cos } = tables;

        let sin = Tensor::from_vec(sin, &[max_seq_len, half_dim], device)?;
        let cos = Tensor::from_vec(cos, &[max_seq_len, half_dim], device)?;

        // Log ROPE initialization parameters
        tracing::info!(
            "ROPE initialized: base={}, rope_dims={}, max_seq_len={}",
            theta,
            dim,
            max_seq_len
        );

        Ok(Self { sin, cos })
    }

    pub fn apply(&self, x: &Tensor, position: usize) -> Result<Tensor> {
        // x shape: [B, H, T, D] for multi-head attention
        if x.dims().len() == 4 {
            let (batch, n_heads, seq_len, head_dim) = x.dims4()?;
            let half_dim = head_dim / 2;

            // LLaMA RoPE uses SPLIT layout: [r0,r1,...,r_{d/2-1}, i0,i1,...,i_{d/2-1}]
            // NOT interleaved [r0,i0,r1,i1,...]
            let x0 = x.narrow(3, 0, half_dim)?; // First half (real)
            let x1 = x.narrow(3, half_dim, half_dim)?; // Second half (imaginary)

            // Get cos/sin for the position
            let cos = self.cos.narrow(0, position, seq_len)?
                .unsqueeze(0)?  // Add batch dim
                .unsqueeze(1)?  // Add heads dim
                .broadcast_as(&[batch, n_heads, seq_len, half_dim])?;
            let sin = self
                .sin
                .narrow(0, position, seq_len)?
                .unsqueeze(0)?
                .unsqueeze(1)?
                .broadcast_as(&[batch, n_heads, seq_len, half_dim])?;

            let x0_rot = (x0.mul(&cos)? - x1.mul(&sin)?)?;
            let x1_rot = (x0.mul(&sin)? + x1.mul(&cos)?)?;

            // Concatenate back in split layout [real, imag]
            let rotated = Tensor::cat(&[x0_rot, x1_rot], 3)?;

            Ok(rotated)
        } else {
            // Original 3D implementation for other uses
            let (_batch, _seq, dim) = x.dims3()?;
            let half_dim = dim / 2;

            // LLaMA RoPE uses SPLIT layout: [r0,r1,...,i0,i1,...]
            let x0 = x.narrow(2, 0, half_dim)?; // First half (real)
            let x1 = x.narrow(2, half_dim, half_dim)?; // Second half (imaginary)

            let cos = self.cos.narrow(0, position, 1)?;
            let sin = self.sin.narrow(0, position, 1)?;

            let x0_rot = (x0.mul(&cos)? - x1.mul(&sin)?)?;
            let x1_rot = (x0.mul(&sin)? + x1.mul(&cos)?)?;

            // Concatenate back in split layout [real, imag]
            let rotated = Tensor::cat(&[x0_rot, x1_rot], 2)?;

            Ok(rotated)
        }
    }
}

/// Multi-Head Attention Layer
pub struct MultiHeadAttention {
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    group_size: usize, // n_heads / n_kv_heads
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    q_norm: Option<LayerNorm>,
    k_norm: Option<LayerNorm>,
    sub_layernorm: Option<LayerNorm>,
    rope: Option<RotaryEmbedding>,
    layer_idx: usize, // Layer index for QK256 weight name generation
}

impl MultiHeadAttention {
    pub fn new(config: &BitNetConfig, vb: VarBuilder, layer_idx: usize) -> Result<Self> {
        let hidden_size = config.model.hidden_size;
        let n_heads = config.model.num_heads;
        let head_dim = config.model.attention_head_dim.unwrap_or_else(|| hidden_size / n_heads);

        if config.model.attention_head_dim.is_none() && !hidden_size.is_multiple_of(n_heads) {
            return Err(BitNetError::Validation(format!(
                "hidden_size {} not divisible by num_heads {}",
                hidden_size, n_heads
            )));
        }

        let n_kv_heads = config.model.num_key_value_heads.max(1).min(n_heads);
        if !n_heads.is_multiple_of(n_kv_heads) {
            return Err(BitNetError::Validation(format!(
                "num_heads {} must be divisible by num_key_value_heads {}",
                n_heads, n_kv_heads
            )));
        }
        let group_size = n_heads / n_kv_heads;
        let q_out = n_heads * head_dim;
        let kv_out = n_kv_heads * head_dim;

        tracing::info!(
            "layer{}: MultiHeadAttention dims: hidden={}, n_heads={}, n_kv_heads={}, head_dim={}, q_out={}, kv_out={}, group_size={}",
            layer_idx,
            hidden_size,
            n_heads,
            n_kv_heads,
            head_dim,
            q_out,
            kv_out,
            group_size
        );

        tracing::info!(
            "layer{}: About to create linear layers with: q_proj([{}, {}]), k_proj([{}, {}]), v_proj([{}, {}]), o_proj([{}, {}])",
            layer_idx,
            q_out,
            hidden_size,
            kv_out,
            hidden_size,
            kv_out,
            hidden_size,
            hidden_size,
            q_out
        );

        let q_proj = linear_with_optional_bias(hidden_size, q_out, vb.pp("q_proj"))?;
        let k_proj = linear_with_optional_bias(hidden_size, kv_out, vb.pp("k_proj"))?;
        let v_proj = linear_with_optional_bias(hidden_size, kv_out, vb.pp("v_proj"))?;
        let o_proj = linear_with_optional_bias(q_out, hidden_size, vb.pp("o_proj"))?;
        let q_norm = optional_layer_norm_with_optional_bias(
            config.model.norm_type,
            head_dim,
            eps_from_config(config),
            vb.pp("q_norm"),
        )?;
        let k_norm = optional_layer_norm_with_optional_bias(
            config.model.norm_type,
            head_dim,
            eps_from_config(config),
            vb.pp("k_norm"),
        )?;
        let sub_layernorm = optional_layer_norm_with_optional_bias(
            config.model.norm_type,
            q_out,
            eps_from_config(config),
            vb.pp("sub_layernorm"),
        )?;

        let rope = RotaryEmbedding::new(
            head_dim,
            config.model.max_position_embeddings,
            config.model.rope_theta,
            vb.device(),
        )
        .ok();

        Ok(Self {
            n_heads,
            n_kv_heads,
            head_dim,
            group_size,
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            sub_layernorm,
            rope,
            layer_idx,
        })
    }

    // `forward` lives in the `attention_forward` module so the projection,
    // RoPE/cache, GQA, score, softmax, and output-projection responsibilities
    // stay isolated from QK256 linear dispatch helpers below.

    /// Apply linear transformation with QK256 dispatch
    /// Apply linear transformation with QK256 dispatch
    /// Apply linear transformation with QK256 dispatch
    fn apply_linear(
        &self,
        input: &Tensor,
        linear: &Linear,
        proj_name: &str,
        raw_tensors: &HashMap<String, Tensor>,
        dense_linear_hooks: &DenseLinearRuntimeHookRegistry,
    ) -> Result<Tensor> {
        // Generate weight name based on layer index and projection name
        // Format: "layers.{idx}.attention.{proj_name}.weight.qk256_qs"
        let qk256_key =
            format!("layers.{}.attention.{}.weight.qk256_qs", self.layer_idx, proj_name);

        // Check for QK256 data
        if let Some(qk256_tensor) = raw_tensors.get(&qk256_key) {
            tracing::debug!("Using QK256 kernel for {}", qk256_key);
            let inline_scale = qk256_inline_scale(raw_tensors, &qk256_key)?;
            return forward_qk256_with_scale(input, qk256_tensor, &qk256_key, inline_scale);
        }

        if strict_cuda_bitnet_backend_requested() {
            record_bitnet_linear_unsupported();
            return Err(BitNetError::Validation(format!(
                "strict CUDA BitNet linear dispatch requires QK256 raw tensor {}; refusing CPU fallback",
                qk256_key
            )));
        }

        // Probe: Why is QK256 not found? (layer 0 only, once)
        if trace_rms_enabled() && self.layer_idx == 0 {
            static FALLBACK_LOGGED: std::sync::Once = std::sync::Once::new();
            FALLBACK_LOGGED.call_once(|| {
                eprintln!(
                    "trace_fallback: QK256 key '{}' not found in raw_tensors ({}keys total)",
                    qk256_key,
                    raw_tensors.len()
                );
                // Show first few keys for debugging
                let sample_keys: Vec<_> = raw_tensors.keys().take(5).collect();
                eprintln!("trace_fallback: Sample keys: {:?}", sample_keys);
            });
        }

        // Fall back to standard linear
        tracing::trace!(
            "Using standard linear for layers.{}.attention.{}",
            self.layer_idx,
            proj_name
        );
        let dense_tensor_name = format!("layers.{}.attention.{}.weight", self.layer_idx, proj_name);
        let hook_boundary =
            dense_linear_runtime_hook_boundary(&dense_tensor_name, dense_linear_hooks);
        tracing::trace!(
            tensor_name = %hook_boundary.tensor_name,
            selected_path = hook_boundary.selected_path,
            selected_kernel = hook_boundary.selected_kernel,
            sidecar_descriptor_present = hook_boundary.sidecar_descriptor_present,
            runtime_compute_enabled = hook_boundary.runtime_compute_enabled,
            "dense linear production hook boundary"
        );
        if let Some(output) = maybe_forward_dense_q8_sidecar_linear(
            input,
            linear,
            &dense_tensor_name,
            dense_linear_hooks,
        )? {
            return Ok(output);
        }
        record_bitnet_linear_cpu_fallback();
        linear.forward(input).map_err(BitNetError::from)
    }

    /// PATCH 5: Create causal mask with [1, 1, Tq, Tk] shape
    fn create_causal_mask(&self, q_len: usize, k_len: usize, device: &Device) -> Result<Tensor> {
        // Past tokens are stored in the KV cache and increase k_len.
        // For each query position i, disallow attention to key positions
        // greater than past_len + i.
        let past_len = k_len.saturating_sub(q_len);
        let mut mask_vec = vec![0.0f32; q_len * k_len];
        for i in 0..q_len {
            let start = past_len + i + 1;
            for j in start..k_len {
                mask_vec[i * k_len + j] = f32::NEG_INFINITY;
            }
        }
        // Create [1, 1, q_len, k_len] shape directly for broadcast compatibility
        Tensor::from_vec(mask_vec, &[1, 1, q_len, k_len], device).map_err(BitNetError::from)
    }
}

/// Feed-Forward Network
pub struct FeedForward {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
    sub_layernorm: Option<LayerNorm>,
    activation_type: ActivationType,
    layer_idx: usize, // Layer index for QK256 weight name generation
}

impl FeedForward {
    pub fn new(config: &BitNetConfig, vb: VarBuilder, layer_idx: usize) -> Result<Self> {
        let hidden_size = config.model.hidden_size;
        let intermediate_size = config.model.intermediate_size;

        Ok(Self {
            gate_proj: linear_with_optional_bias(
                hidden_size,
                intermediate_size,
                vb.pp("gate_proj"),
            )?,
            up_proj: linear_with_optional_bias(hidden_size, intermediate_size, vb.pp("up_proj"))?,
            down_proj: linear_with_optional_bias(
                intermediate_size,
                hidden_size,
                vb.pp("down_proj"),
            )?,
            sub_layernorm: optional_layer_norm_with_optional_bias(
                config.model.norm_type,
                intermediate_size,
                eps_from_config(config),
                vb.pp("sub_layernorm"),
            )?,
            activation_type: config.model.activation_type,
            layer_idx,
        })
    }

    pub fn forward(
        &self,
        x: &Tensor,
        raw_tensors: &HashMap<String, Tensor>,
        dense_linear_hooks: &DenseLinearRuntimeHookRegistry,
    ) -> Result<Tensor> {
        self.forward_impl(x, raw_tensors, dense_linear_hooks, None)
    }

    fn forward_impl(
        &self,
        x: &Tensor,
        raw_tensors: &HashMap<String, Tensor>,
        dense_linear_hooks: &DenseLinearRuntimeHookRegistry,
        workspace: Option<&mut TransformerForwardWorkspace>,
    ) -> Result<Tensor> {
        let gate =
            self.apply_linear(x, &self.gate_proj, "gate_proj", raw_tensors, dense_linear_hooks)?;
        if qwen_trace_layer_enabled(self.layer_idx) {
            qwen_trace_tensor("mlp.gate_proj", Some(self.layer_idx), &gate)?;
        }

        // MLP gating diagnostics (point 3 of user's plan)
        if debug_mlp_enabled()
            && let Ok(u_norm) = gate.sqr()?.mean_all()?.sqrt()?.to_scalar::<f32>()
        {
            tracing::debug!("MLP ||u|| (gate_proj): {:.6e}", u_norm);
        }

        let gate = self.apply_activation(&gate)?;
        if qwen_trace_layer_enabled(self.layer_idx) {
            qwen_trace_tensor("mlp.gate_activation", Some(self.layer_idx), &gate)?;
        }

        if debug_mlp_enabled()
            && let Ok(activation_norm) = gate.sqr()?.mean_all()?.sqrt()?.to_scalar::<f32>()
        {
            tracing::debug!("MLP ||activation(u)||: {:.6e}", activation_norm);
        }

        let up = self.apply_linear(x, &self.up_proj, "up_proj", raw_tensors, dense_linear_hooks)?;
        if qwen_trace_layer_enabled(self.layer_idx) {
            qwen_trace_tensor("mlp.up_proj", Some(self.layer_idx), &up)?;
        }

        if debug_mlp_enabled()
            && let Ok(v_norm) = up.sqr()?.mean_all()?.sqrt()?.to_scalar::<f32>()
        {
            tracing::debug!("MLP ||v|| (up_proj): {:.6e}", v_norm);
        }

        let hidden = gate.mul(&up)?;
        if qwen_trace_layer_enabled(self.layer_idx) {
            qwen_trace_tensor("mlp.gated_product", Some(self.layer_idx), &hidden)?;
        }
        let hidden = if let Some(sub_layernorm) = &self.sub_layernorm {
            let normalized = sub_layernorm.forward(&hidden)?;
            if qwen_trace_layer_enabled(self.layer_idx) {
                qwen_trace_tensor("mlp.sub_layernorm", Some(self.layer_idx), &normalized)?;
            }
            normalized
        } else {
            hidden
        };

        if debug_mlp_enabled()
            && let Ok(prod_norm) = hidden.sqr()?.mean_all()?.sqrt()?.to_scalar::<f32>()
        {
            tracing::debug!("MLP ||silu(u) * v||: {:.6e}", prod_norm);
        }

        let output = self.apply_linear(
            &hidden,
            &self.down_proj,
            "down_proj",
            raw_tensors,
            dense_linear_hooks,
        )?;
        if let Some(workspace) = workspace {
            let boundary = self.down_proj_output_storage_boundary(&output);
            workspace.record_down_proj_output_storage_boundary(&output, boundary);
        }
        if qwen_trace_layer_enabled(self.layer_idx) {
            qwen_trace_tensor("mlp.down_proj", Some(self.layer_idx), &output)?;
        }

        if debug_mlp_enabled()
            && let Ok(out_norm) = output.sqr()?.mean_all()?.sqrt()?.to_scalar::<f32>()
        {
            tracing::debug!("MLP ||W2 * (...)||: {:.6e}", out_norm);
        }

        Ok(output)
    }

    pub fn forward_with_workspace(
        &self,
        x: &Tensor,
        raw_tensors: &HashMap<String, Tensor>,
        dense_linear_hooks: &DenseLinearRuntimeHookRegistry,
        workspace: &mut TransformerForwardWorkspace,
    ) -> Result<Tensor> {
        workspace.record_feed_forward_input(x);
        let output = self.forward_impl(x, raw_tensors, dense_linear_hooks, Some(workspace))?;
        workspace.record_feed_forward_output(&output);
        workspace.store_feed_forward_output(output);
        workspace.take_feed_forward_output()
    }

    fn apply_activation(&self, input: &Tensor) -> Result<Tensor> {
        apply_ffn_activation(input, self.activation_type)
    }

    /// Apply linear transformation with QK256 dispatch
    fn apply_linear(
        &self,
        input: &Tensor,
        linear: &Linear,
        proj_name: &str,
        raw_tensors: &HashMap<String, Tensor>,
        dense_linear_hooks: &DenseLinearRuntimeHookRegistry,
    ) -> Result<Tensor> {
        // Generate weight name based on layer index and projection name
        // Format: "layers.{idx}.feed_forward.{proj_name}.weight.qk256_qs"
        let qk256_key =
            format!("layers.{}.feed_forward.{}.weight.qk256_qs", self.layer_idx, proj_name);

        // Check for QK256 data
        if let Some(qk256_tensor) = raw_tensors.get(&qk256_key) {
            tracing::debug!("Using QK256 kernel for {}", qk256_key);
            let inline_scale = qk256_inline_scale(raw_tensors, &qk256_key)?;
            return forward_qk256_with_scale(input, qk256_tensor, &qk256_key, inline_scale);
        }

        if strict_cuda_bitnet_backend_requested() {
            record_bitnet_linear_unsupported();
            return Err(BitNetError::Validation(format!(
                "strict CUDA BitNet linear dispatch requires QK256 raw tensor {}; refusing CPU fallback",
                qk256_key
            )));
        }

        // Fall back to standard linear
        tracing::trace!(
            "Using standard linear for layers.{}.feed_forward.{}",
            self.layer_idx,
            proj_name
        );
        let dense_tensor_name =
            format!("layers.{}.feed_forward.{}.weight", self.layer_idx, proj_name);
        let hook_boundary =
            dense_linear_runtime_hook_boundary(&dense_tensor_name, dense_linear_hooks);
        tracing::trace!(
            tensor_name = %hook_boundary.tensor_name,
            selected_path = hook_boundary.selected_path,
            selected_kernel = hook_boundary.selected_kernel,
            sidecar_descriptor_present = hook_boundary.sidecar_descriptor_present,
            runtime_compute_enabled = hook_boundary.runtime_compute_enabled,
            "dense linear production hook boundary"
        );
        if let Some(output) = maybe_forward_dense_q8_sidecar_linear(
            input,
            linear,
            &dense_tensor_name,
            dense_linear_hooks,
        )? {
            return Ok(output);
        }
        record_bitnet_linear_cpu_fallback();
        linear.forward(input).map_err(BitNetError::from)
    }

    fn down_proj_output_storage_boundary(
        &self,
        output: &Tensor,
    ) -> TransformerWorkspaceOutputSurface {
        let boundary = DenseLinearOutputStorageApiBoundary::from_candle_linear(
            "feed_forward.down_proj.output",
            &self.down_proj,
        );
        TransformerWorkspaceOutputSurface {
            name: "feed_forward.down_proj.output",
            storage_owner: "TransformerForwardWorkspace",
            status: boundary.status,
            reason: boundary.reason,
            next_api_hook: boundary.next_api_hook,
            last_shape: output.dims().to_vec(),
            linear_weight_shape: boundary.weight_shape,
            linear_bias_shape: boundary.bias_shape,
            weight_accessible: boundary.weight_accessible,
            bias_accessible: boundary.bias_accessible,
            can_fill_caller_output_storage: boundary.can_fill_caller_output_storage,
        }
    }
}

fn apply_ffn_activation(input: &Tensor, activation_type: ActivationType) -> Result<Tensor> {
    match activation_type {
        ActivationType::Silu => input.silu().map_err(BitNetError::from),
        ActivationType::Relu2 => input.relu()?.sqr().map_err(BitNetError::from),
        ActivationType::Gelu => input.gelu_erf().map_err(BitNetError::from),
    }
}

/// Transformer Block
pub struct TransformerBlock {
    attention: MultiHeadAttention,
    feed_forward: FeedForward,
    attention_norm: LayerNorm,
    ffn_norm: LayerNorm,
}

/// Typed transformer forward workspace boundary.
///
/// This is intentionally conservative: it records the API surface that future
/// slices can use for reusable transformer-owned buffers, but it does not reuse
/// Candle tensor outputs yet. Keeping tensor math delegated to the existing
/// `forward` path preserves the current Qwen3 Q8_0 behavior oracle while making
/// the owned-output boundary explicit in code.
#[derive(Debug, Clone, Default)]
pub struct TransformerForwardWorkspace {
    model_forward_calls: usize,
    block_forward_calls: usize,
    feed_forward_calls: usize,
    last_input_shape: Vec<usize>,
    last_output_shape: Vec<usize>,
    feed_forward_output_slot: Option<Tensor>,
    feed_forward_output_surface: Option<TransformerWorkspaceOutputSurface>,
    workspace_owned_output_count: usize,
    down_proj_output_storage_attempts: usize,
    tensor_reuse_enabled: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TransformerWorkspaceOutputSurface {
    pub name: &'static str,
    pub storage_owner: &'static str,
    pub status: &'static str,
    pub reason: &'static str,
    pub next_api_hook: &'static str,
    pub last_shape: Vec<usize>,
    pub linear_weight_shape: Vec<usize>,
    pub linear_bias_shape: Option<Vec<usize>>,
    pub weight_accessible: bool,
    pub bias_accessible: bool,
    pub can_fill_caller_output_storage: bool,
}

/// Behavior-preserving dense linear output-storage API boundary.
///
/// Candle exposes the read-side pieces (`Linear::weight` and `Linear::bias`),
/// but the compute-side `Tensor::matmul` and optional bias add still allocate
/// and return owned tensors. This boundary records that narrower fact so Kaby
/// SLM allocation work does not mistake read access for a reusable output slot.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DenseLinearOutputStorageApiBoundary {
    pub role: &'static str,
    pub status: &'static str,
    pub reason: &'static str,
    pub next_api_hook: &'static str,
    pub weight_shape: Vec<usize>,
    pub bias_shape: Option<Vec<usize>>,
    pub weight_accessible: bool,
    pub bias_accessible: bool,
    pub can_fill_caller_output_storage: bool,
}

impl DenseLinearOutputStorageApiBoundary {
    pub fn from_candle_linear(role: &'static str, linear: &Linear) -> Self {
        let weight_shape = linear.weight().dims().to_vec();
        let bias_shape = linear.bias().map(|bias| bias.dims().to_vec());

        Self {
            role,
            status: "dense_linear_output_storage_blocked_by_candle_tensor_ops",
            reason: "candle_nn::Linear exposes weight and optional bias tensors, but its behavior-preserving compute path is Tensor::matmul plus optional broadcast_add, and those operations return owned Tensors without a caller-provided output-storage parameter",
            next_api_hook: "add or adopt a Candle Tensor matmul/bias-add output-storage API before replacing FeedForward::down_proj output construction with reusable workspace-backed storage",
            weight_shape,
            bias_shape,
            weight_accessible: true,
            bias_accessible: true,
            can_fill_caller_output_storage: false,
        }
    }
}

impl TransformerForwardWorkspace {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn clear(&mut self) {
        *self = Self::default();
    }

    pub fn model_forward_calls(&self) -> usize {
        self.model_forward_calls
    }

    pub fn block_forward_calls(&self) -> usize {
        self.block_forward_calls
    }

    pub fn feed_forward_calls(&self) -> usize {
        self.feed_forward_calls
    }

    pub fn last_input_shape(&self) -> &[usize] {
        &self.last_input_shape
    }

    pub fn last_output_shape(&self) -> &[usize] {
        &self.last_output_shape
    }

    pub fn tensor_reuse_enabled(&self) -> bool {
        self.tensor_reuse_enabled
    }

    pub fn workspace_owned_output_count(&self) -> usize {
        self.workspace_owned_output_count
    }

    pub fn down_proj_output_storage_attempts(&self) -> usize {
        self.down_proj_output_storage_attempts
    }

    pub fn first_output_surface(&self) -> Option<&TransformerWorkspaceOutputSurface> {
        self.feed_forward_output_surface.as_ref()
    }

    pub fn reuse_status(&self) -> &'static str {
        if self.tensor_reuse_enabled {
            "typed_transformer_forward_workspace_reuse_enabled"
        } else if self.feed_forward_output_surface.is_some() {
            "dense_linear_output_storage_blocked_by_candle_tensor_ops"
        } else {
            "api_boundary_present_owned_tensor_reuse_not_enabled"
        }
    }

    fn record_model_input(&mut self, tensor: &Tensor) {
        self.model_forward_calls += 1;
        self.last_input_shape = tensor.dims().to_vec();
    }

    fn record_model_output(&mut self, tensor: &Tensor) {
        self.last_output_shape = tensor.dims().to_vec();
    }

    fn record_block_input(&mut self, tensor: &Tensor) {
        self.block_forward_calls += 1;
        self.last_input_shape = tensor.dims().to_vec();
    }

    fn record_block_output(&mut self, tensor: &Tensor) {
        self.last_output_shape = tensor.dims().to_vec();
    }

    fn record_feed_forward_input(&mut self, tensor: &Tensor) {
        self.feed_forward_calls += 1;
        self.last_input_shape = tensor.dims().to_vec();
    }

    fn record_feed_forward_output(&mut self, tensor: &Tensor) {
        self.last_output_shape = tensor.dims().to_vec();
    }

    fn record_down_proj_output_storage_boundary(
        &mut self,
        tensor: &Tensor,
        boundary: TransformerWorkspaceOutputSurface,
    ) {
        self.down_proj_output_storage_attempts += 1;
        self.last_output_shape = tensor.dims().to_vec();
        self.feed_forward_output_surface = Some(boundary);
    }

    fn store_feed_forward_output(&mut self, tensor: Tensor) {
        let last_shape = tensor.dims().to_vec();
        self.last_output_shape = last_shape.clone();
        if let Some(surface) = self.feed_forward_output_surface.as_mut() {
            surface.last_shape = last_shape;
        }
        self.workspace_owned_output_count += 1;
        self.feed_forward_output_slot = Some(tensor);
    }

    fn take_feed_forward_output(&mut self) -> Result<Tensor> {
        self.feed_forward_output_slot.take().ok_or_else(|| {
            BitNetError::Validation(
                "TransformerForwardWorkspace feed-forward output slot must be populated before take"
                    .to_string(),
            )
        })
    }
}

impl TransformerBlock {
    pub fn new(config: &BitNetConfig, vb: VarBuilder, layer_idx: usize) -> Result<Self> {
        let hidden_size = config.model.hidden_size;
        // PATCH 1: Use RMSNorm epsilon from config header for ALL norms (per-layer + final)
        let eps = eps_from_config(config);

        tracing::debug!("TransformerBlock using RMSNorm eps={} (from header)", eps);

        Ok(Self {
            attention: MultiHeadAttention::new(config, vb.pp("attention"), layer_idx)?,
            feed_forward: FeedForward::new(config, vb.pp("feed_forward"), layer_idx)?,
            attention_norm: norm_with_optional_bias(
                config.model.norm_type,
                hidden_size,
                eps,
                vb.pp("attention_norm"),
            )?,
            ffn_norm: norm_with_optional_bias(
                config.model.norm_type,
                hidden_size,
                eps,
                vb.pp("post_attention_layernorm"),
            )?,
        })
    }

    pub fn forward(
        &self,
        x: &Tensor,
        kv_cache: Option<&mut LayerKVCache>,
        raw_tensors: &HashMap<String, Tensor>,
        dense_linear_hooks: &DenseLinearRuntimeHookRegistry,
    ) -> Result<Tensor> {
        self.forward_impl(x, kv_cache, raw_tensors, dense_linear_hooks, None)
    }

    pub fn forward_with_workspace(
        &self,
        x: &Tensor,
        kv_cache: Option<&mut LayerKVCache>,
        raw_tensors: &HashMap<String, Tensor>,
        dense_linear_hooks: &DenseLinearRuntimeHookRegistry,
        workspace: &mut TransformerForwardWorkspace,
    ) -> Result<Tensor> {
        workspace.record_block_input(x);
        let output =
            self.forward_impl(x, kv_cache, raw_tensors, dense_linear_hooks, Some(workspace))?;
        workspace.record_block_output(&output);
        Ok(output)
    }

    fn forward_impl(
        &self,
        x: &Tensor,
        kv_cache: Option<&mut LayerKVCache>,
        raw_tensors: &HashMap<String, Tensor>,
        dense_linear_hooks: &DenseLinearRuntimeHookRegistry,
        mut workspace: Option<&mut TransformerForwardWorkspace>,
    ) -> Result<Tensor> {
        // Debug input activation norms
        if debug_attn_enabled() {
            let norm = x.sqr()?.mean_all()?.sqrt()?.to_scalar::<f32>()?;
            eprintln!("[norm] input: {norm:.6e}");
        }

        // Pre-norm attention
        let residual = x;

        // RMSNorm diagnostics (Layer 0 only) - attention norm
        // User's diagnostic: log mean(x^2) and rms = sqrt(mean(x^2) + eps) before/after norm
        if debug_rmsnorm_enabled() {
            static ATTN_NORM_LOGGED: std::sync::Once = std::sync::Once::new();
            ATTN_NORM_LOGGED.call_once(|| {
                if let Ok(mean_sq) =
                    x.sqr().and_then(|s| s.mean_all()).and_then(|m| m.to_scalar::<f32>())
                {
                    // Note: RMSNorm formula is: rms = sqrt(mean(x^2) + eps), y = (x / rms) * weight
                    // The actual eps value is in the LayerNorm (handled by candle)
                    let rms_approx = mean_sq.sqrt(); // Approximate (actual includes eps inside sqrt)
                    tracing::info!(
                        "RMSNorm (attn, layer 0) - input mean(x^2): {:.6e}, approx_rms: {:.6e}",
                        mean_sq,
                        rms_approx
                    );
                    if !rms_approx.is_finite() {
                        tracing::warn!("⚠️  RMSNorm (attn) - input has non-finite values!");
                    }
                }
            });
        }

        let x = self.attention_norm.forward(x)?;
        if qwen_trace_layer_enabled(self.attention.layer_idx) {
            qwen_trace_tensor("block.attention_norm", Some(self.attention.layer_idx), &x)?;
        }

        // Probe A2: LayerNorm gamma RMS + LN output RMS (layer 0, step 0 only)
        if trace_rms_enabled() && self.attention.layer_idx == 0 {
            static LN0_LOGGED: std::sync::Once = std::sync::Once::new();
            LN0_LOGGED.call_once(|| {
                let _ = (|| -> candle_core::Result<()> {
                    // Get gamma (weight) from LayerNorm
                    let gamma_vec = self.attention_norm.weight().to_vec1::<f32>()?;
                    let g_rms = (gamma_vec.iter().map(|x| x * x).sum::<f32>()
                        / gamma_vec.len().max(1) as f32)
                        .sqrt();

                    // Get LN output RMS
                    let ln_vec = x.flatten_all()?.to_vec1::<f32>()?;
                    let ln_rms = (ln_vec.iter().map(|x| x * x).sum::<f32>()
                        / ln_vec.len().max(1) as f32)
                        .sqrt();
                    eprintln!("trace: ln0_gamma_rms={:.6} ln0_out_rms={:.6}", g_rms, ln_rms);
                    Ok(())
                })();
            });
        }

        // Tracepoint 2: Attention norm output (layer-specific)
        #[cfg(feature = "trace")]
        {
            let trace_name = format!("t0/blk{}/attn_norm", self.attention.layer_idx);
            bitnet_trace::dump_trace(
                &trace_name,
                &x,
                Some(0),
                Some(self.attention.layer_idx as isize),
                Some("attn_norm"),
            )
            .map_err(BitNetError::from)?;
        }

        // Check norm output
        if debug_rmsnorm_enabled() {
            static ATTN_NORM_OUT_LOGGED: std::sync::Once = std::sync::Once::new();
            ATTN_NORM_OUT_LOGGED.call_once(|| {
                if let Ok(norm_out) = x
                    .sqr()
                    .and_then(|s| s.mean_all())
                    .and_then(|m| m.sqrt())
                    .and_then(|r| r.to_scalar::<f32>())
                {
                    tracing::info!("RMSNorm (attn, layer 0) - output L2 norm: {:.6e}", norm_out);
                    if !norm_out.is_finite() {
                        tracing::warn!("⚠️  RMSNorm (attn) - output is non-finite!");
                    }
                }
            });
        }

        let x = self.attention.forward(&x, kv_cache, raw_tensors, dense_linear_hooks)?;
        let x = (x + residual)?;
        if qwen_trace_layer_enabled(self.attention.layer_idx) {
            qwen_trace_tensor("block.post_attention_residual", Some(self.attention.layer_idx), &x)?;
        }

        // Debug post-attention activation norms
        if debug_attn_enabled() {
            let norm = x.sqr()?.mean_all()?.sqrt()?.to_scalar::<f32>()?;
            eprintln!("[norm] post-attn: {norm:.6e}");
        }

        // Pre-norm FFN
        let residual = &x;

        // RMSNorm diagnostics (Layer 0 only) - FFN norm
        if debug_rmsnorm_enabled() {
            static FFN_NORM_LOGGED: std::sync::Once = std::sync::Once::new();
            FFN_NORM_LOGGED.call_once(|| {
                if let Ok(mean_sq) =
                    x.sqr().and_then(|s| s.mean_all()).and_then(|m| m.to_scalar::<f32>())
                {
                    let rms_approx = mean_sq.sqrt();
                    tracing::info!(
                        "RMSNorm (ffn, layer 0) - input mean(x^2): {:.6e}, approx_rms: {:.6e}",
                        mean_sq,
                        rms_approx
                    );
                    if !rms_approx.is_finite() {
                        tracing::warn!("⚠️  RMSNorm (ffn) - input has non-finite values!");
                    }
                }
            });
        }

        let x = self.ffn_norm.forward(&x)?;
        if qwen_trace_layer_enabled(self.attention.layer_idx) {
            qwen_trace_tensor("block.ffn_norm", Some(self.attention.layer_idx), &x)?;
        }

        // Check norm output
        if debug_rmsnorm_enabled() {
            static FFN_NORM_OUT_LOGGED: std::sync::Once = std::sync::Once::new();
            FFN_NORM_OUT_LOGGED.call_once(|| {
                if let Ok(norm_out) = x
                    .sqr()
                    .and_then(|s| s.mean_all())
                    .and_then(|m| m.sqrt())
                    .and_then(|r| r.to_scalar::<f32>())
                {
                    tracing::info!("RMSNorm (ffn, layer 0) - output L2 norm: {:.6e}", norm_out);
                    if !norm_out.is_finite() {
                        tracing::warn!("⚠️  RMSNorm (ffn) - output is non-finite!");
                    }
                }
            });
        }

        let x = if let Some(workspace) = workspace.as_mut() {
            self.feed_forward.forward_with_workspace(
                &x,
                raw_tensors,
                dense_linear_hooks,
                workspace,
            )?
        } else {
            self.feed_forward.forward(&x, raw_tensors, dense_linear_hooks)?
        };
        let x = (x + residual)?;
        if qwen_trace_layer_enabled(self.attention.layer_idx) {
            qwen_trace_tensor("block.output", Some(self.attention.layer_idx), &x)?;
        }

        // Debug post-FFN activation norms
        if debug_attn_enabled() {
            let norm = x.sqr()?.mean_all()?.sqrt()?.to_scalar::<f32>()?;
            eprintln!("[norm] post-ffn: {norm:.6e}");
        }

        Ok(x)
    }
}

fn eps_from_config(config: &BitNetConfig) -> f64 {
    config.model.rms_norm_eps.map(|e| e as f64).unwrap_or(1e-5)
}

/// KV Cache for a single layer
pub struct LayerKVCache {
    pub k: Tensor,
    pub v: Tensor,
    pub seq_len: usize,
    pub max_seq_len: usize,
    pub n_kv_heads: usize, // Store the number of KV heads for validation
}

impl LayerKVCache {
    pub fn new(
        batch_size: usize,
        n_kv_heads: usize, // Changed from n_heads to n_kv_heads
        max_seq_len: usize,
        head_dim: usize,
        device: &Device,
    ) -> Result<Self> {
        let k =
            Tensor::zeros(&[batch_size, n_kv_heads, max_seq_len, head_dim], DType::F32, device)?;
        let v =
            Tensor::zeros(&[batch_size, n_kv_heads, max_seq_len, head_dim], DType::F32, device)?;

        Ok(Self { k, v, seq_len: 0, max_seq_len, n_kv_heads })
    }

    /// Append new K/V tensors to the cache
    ///
    /// **Performance note**: The clones on first append (lines 1130-1131) are necessary
    /// because we accept `&Tensor` but need to store owned tensors. Candle's `Tensor::clone()`
    /// is cheap - it only increments the Arc reference count, not a deep data copy.
    /// Subsequent appends use `Tensor::cat` which allocates new tensors regardless.
    ///
    /// To eliminate these clones would require API changes to accept owned tensors,
    /// which would complicate calling code.
    pub fn append(&mut self, k_new: &Tensor, v_new: &Tensor) -> Result<()> {
        // Expect shapes: k: [B,HKV,T_new,Hd], v: [B,HKV,T_new,Hd] where HKV = n_kv_heads
        let new_seq_len = k_new.dims()[2];

        // Validate that the incoming tensors have the expected number of KV heads
        let k_heads = k_new.dims()[1];
        if k_heads != self.n_kv_heads {
            return Err(BitNetError::Validation(format!(
                "KV cache expects {} heads, but received K tensor with {} heads",
                self.n_kv_heads, k_heads
            )));
        }

        if self.seq_len == 0 {
            // First append: clone is necessary (Arc increment only, not deep copy)
            self.k = k_new.clone();
            self.v = v_new.clone();
        } else {
            // Concatenate along time dimension (dim=2)
            if self.seq_len + new_seq_len > self.max_seq_len {
                return Err(BitNetError::from(candle_core::Error::Msg(
                    "KV cache overflow".to_string(),
                )));
            }
            // Tensor::cat allocates new tensor - no optimization possible here
            self.k = Tensor::cat(&[&self.k, k_new], 2)?;
            self.v = Tensor::cat(&[&self.v, v_new], 2)?;
        }

        self.seq_len += new_seq_len;
        Ok(())
    }

    pub fn clear(&mut self) {
        self.seq_len = 0;
    }
}

/// Full KV Cache for all layers
pub struct KVCache {
    pub layers: Vec<LayerKVCache>,
}

impl KVCache {
    pub fn new(config: &BitNetConfig, batch_size: usize, device: &Device) -> Result<Self> {
        let n_layers = config.model.num_layers;
        let n_heads = config.model.num_heads;
        let hidden_size = config.model.hidden_size;

        // Validate shape assumptions before calculating dimensions
        if config.model.attention_head_dim.is_none() && !hidden_size.is_multiple_of(n_heads) {
            return Err(BitNetError::Validation(format!(
                "KVCache: hidden_size {} not divisible by num_heads {}",
                hidden_size, n_heads
            )));
        }

        let n_kv_heads = config.model.num_key_value_heads.max(1).min(n_heads);
        if !n_heads.is_multiple_of(n_kv_heads) {
            return Err(BitNetError::Validation(format!(
                "KVCache: num_heads {} not divisible by num_key_value_heads {}",
                n_heads, n_kv_heads
            )));
        }

        let head_dim = config.model.attention_head_dim.unwrap_or_else(|| hidden_size / n_heads);
        let max_seq_len = config.model.max_position_embeddings;

        let mut layers = Vec::with_capacity(n_layers);
        for _ in 0..n_layers {
            layers.push(LayerKVCache::new(batch_size, n_kv_heads, max_seq_len, head_dim, device)?);
        }

        Ok(Self { layers })
    }

    pub fn layer_mut(&mut self, idx: usize) -> Option<&mut LayerKVCache> {
        self.layers.get_mut(idx)
    }

    pub fn clear(&mut self) {
        for layer in &mut self.layers {
            layer.clear();
        }
    }
}

/// Complete Transformer Model
pub struct TransformerModel {
    pub config: BitNetConfig,
    pub embed_tokens: candle_nn::Embedding,
    pub embed_transposed: bool, // True if embeddings are stored as [hidden, vocab]
    pub embed_tied_weight: Option<Tensor>, // Cached transposed embedding weight for tied models [H, V]
    pub layers: Vec<TransformerBlock>,
    pub norm: LayerNorm,
    pub lm_head: Option<Linear>,        // Optional for tied weights
    pub lm_head_weight: Option<Tensor>, // Direct access to lm_head weight for transposed handling
    pub lm_head_transposed: bool,       // True if lm_head is stored as [hidden, vocab]
    device: Device,
    raw_tensors: HashMap<String, Tensor>, // Store raw tensors for QK256 dispatch
    dense_linear_hooks: DenseLinearRuntimeHookRegistry,
}

impl TransformerModel {
    pub fn new(config: BitNetConfig, vb: VarBuilder) -> Result<Self> {
        Self::new_with_tensors(config, vb, HashMap::new())
    }

    pub fn new_with_tensors(
        config: BitNetConfig,
        vb: VarBuilder,
        raw_tensors: HashMap<String, Tensor>,
    ) -> Result<Self> {
        Self::new_with_tensors_and_dense_linear_hooks(
            config,
            vb,
            raw_tensors,
            DenseLinearRuntimeHookRegistry::default(),
        )
    }

    pub fn new_with_tensors_and_dense_linear_hooks(
        config: BitNetConfig,
        vb: VarBuilder,
        raw_tensors: HashMap<String, Tensor>,
        dense_linear_hooks: DenseLinearRuntimeHookRegistry,
    ) -> Result<Self> {
        let device = vb.device().clone();
        let vocab_size = config.model.vocab_size;
        let hidden_size = config.model.hidden_size;
        let n_layers = config.model.num_layers;

        let embed_tokens = candle_nn::embedding(vocab_size, hidden_size, vb.pp("embed_tokens"))?;

        // Read transpose flag for embeddings (1-element tensor)
        let embed_transposed = match vb.get((1,), "embed_tokens.transposed") {
            Ok(t) => {
                let vals = t.to_vec1::<f32>()?;
                vals.first().copied().unwrap_or(0.0) > 0.5
            }
            Err(_) => false, // If flag doesn't exist, assume not transposed
        };

        if embed_transposed {
            tracing::info!(
                "Embeddings are transposed [hidden, vocab] - will handle efficiently at runtime"
            );
        }

        let mut layers = Vec::with_capacity(n_layers);
        for i in 0..n_layers {
            layers.push(TransformerBlock::new(&config, vb.pp(format!("layers.{}", i)), i)?);
        }

        // Use RMSNorm epsilon from config header (CRITICAL: must match per-layer norms)
        let eps = config.model.rms_norm_eps.map(|e| e as f64).unwrap_or(1e-5);
        tracing::info!("Final norm using RMSNorm eps={} (from header)", eps);

        let norm =
            norm_with_optional_bias(config.model.norm_type, hidden_size, eps, vb.pp("final_norm"))?;

        // Try to load lm_head, but it's optional (can be tied to embeddings)
        // Try to create the linear layer, catching errors if weights don't exist
        let (lm_head, lm_head_weight, lm_head_transposed) = match linear_with_optional_bias(
            hidden_size,
            vocab_size,
            vb.pp("lm_head"),
        ) {
            Ok(layer) => {
                // Also get the weight tensor directly for transposed handling
                // Note: weight dimensions might be transposed
                let weight = vb
                    .get((vocab_size, hidden_size), "lm_head.weight")
                    .or_else(|_| vb.get((hidden_size, vocab_size), "lm_head.weight"))
                    .ok();

                // Read transpose flag for lm_head
                let transposed = match vb.get((1,), "lm_head.transposed") {
                    Ok(t) => {
                        let vals = t.to_vec1::<f32>()?;
                        vals.first().copied().unwrap_or(0.0) > 0.5
                    }
                    Err(_) => false, // If flag doesn't exist, assume not transposed
                };

                if transposed {
                    tracing::info!(
                        "LM head is transposed [hidden, vocab] - will handle efficiently at runtime"
                    );
                }
                (Some(layer), weight, transposed)
            }
            Err(err) => match vb
                .get((vocab_size, hidden_size), "lm_head.weight")
                .or_else(|_| vb.get((vocab_size, hidden_size), "output.weight"))
            {
                Ok(weight) => {
                    tracing::warn!(
                        "lm_head linear construction failed ({err}); recovered canonical \
                         lm_head/output weight [{}, {}] through direct lookup",
                        vocab_size,
                        hidden_size
                    );
                    (Some(Linear::new(weight.clone(), None)), Some(weight), false)
                }
                Err(_) => match vb.get((hidden_size, vocab_size), "lm_head.weight") {
                    Ok(weight) => {
                        tracing::info!(
                            "LM head is stored transposed [hidden, vocab] - using direct matmul path"
                        );
                        (None, Some(weight), true)
                    }
                    Err(_) => match vb.get((hidden_size, vocab_size), "output.weight") {
                        Ok(weight) => {
                            tracing::warn!(
                                "lm_head linear construction failed ({err}); recovered GGUF \
                                 output.weight with hidden/vocab dims by reshaping to token-major \
                                 [{}, {}] without transposing values",
                                vocab_size,
                                hidden_size
                            );
                            let weight = weight.reshape((vocab_size, hidden_size))?;
                            (Some(Linear::new(weight.clone(), None)), Some(weight), false)
                        }
                        Err(_) => {
                            tracing::info!(
                                "lm_head/output weight not found after linear construction failed ({err}); \
                                 will use tied weights"
                            );
                            (None, None, false)
                        }
                    },
                },
            },
        };

        // PATCH 2: Optimize tied weights by pre-transposing embeddings once at load
        // NOTE: embed_tokens.embeddings() ALWAYS returns [V,H] (Candle's internal format)
        // regardless of how they were stored in GGUF. We need [H,V] for tied weights.
        let (embed_transposed, embed_tied_weight) = if lm_head.is_none() && lm_head_weight.is_none()
        {
            // No dedicated lm_head, we'll use tied weights - pre-transpose for efficiency
            let embed_weight = embed_tokens.embeddings();
            tracing::info!(
                "Embedding matrix from Candle: {:?} (always [V,H] internally)",
                embed_weight.dims()
            );

            // Always transpose [V,H] -> [H,V] for tied weights, regardless of embed_transposed flag
            // The embed_transposed flag tells us how GGUF stored it, but Candle normalizes to [V,H]
            tracing::info!("Pre-transposing tied embeddings [V,H] -> [H,V] for logits computation");
            let transposed_weight = embed_weight.transpose(0, 1)?; // [H, V]
            tracing::info!("Transposed weight shape: {:?}", transposed_weight.dims());
            (embed_transposed, Some(transposed_weight)) // Cache transposed weight
        } else {
            // Dedicated lm_head exists, no need to optimize embeddings
            (embed_transposed, None)
        };
        qwen_trace_event(
            "model_config",
            &format!(
                "\"vocab_size\":{},\"hidden_size\":{},\"layers\":{},\"heads\":{},\"kv_heads\":{},\"norm_type\":\"{:?}\",\"rms_norm_eps\":{},\"rope_theta\":{},\"embed_transposed\":{},\"lm_head_present\":{},\"lm_head_weight_present\":{},\"lm_head_transposed\":{}",
                vocab_size,
                hidden_size,
                n_layers,
                config.model.num_heads,
                config.model.num_key_value_heads,
                config.model.norm_type,
                config
                    .model
                    .rms_norm_eps
                    .map(|value| qwen_trace_number(value as f64))
                    .unwrap_or_else(|| "null".to_string()),
                config
                    .model
                    .rope_theta
                    .map(|value| qwen_trace_number(value as f64))
                    .unwrap_or_else(|| "null".to_string()),
                embed_transposed,
                lm_head.is_some(),
                lm_head_weight.is_some(),
                lm_head_transposed
            ),
        );

        Ok(Self {
            config,
            embed_tokens,
            embed_transposed,
            embed_tied_weight,
            layers,
            norm,
            lm_head,
            lm_head_weight,
            lm_head_transposed,
            device,
            raw_tensors,
            dense_linear_hooks,
        })
    }

    pub fn dense_linear_runtime_hook_boundary(
        &self,
        tensor_name: &str,
    ) -> DenseLinearRuntimeHookBoundary {
        dense_linear_runtime_hook_boundary(tensor_name, &self.dense_linear_hooks)
    }

    pub fn dense_linear_runtime_hook_boundaries(&self) -> Vec<DenseLinearRuntimeHookBoundary> {
        let mut tensor_names: Vec<_> = self.dense_linear_hooks.keys().cloned().collect();
        tensor_names.sort();
        tensor_names
            .into_iter()
            .map(|tensor_name| {
                dense_linear_runtime_hook_boundary(&tensor_name, &self.dense_linear_hooks)
            })
            .collect()
    }

    pub fn embed(&self, tokens: &[u32]) -> Result<Tensor> {
        let token_ids = Tensor::from_vec(tokens.to_vec(), &[1, tokens.len()], &self.device)?;

        // Get dimensions
        let batch_size = token_ids.dims()[0];
        let seq_len = token_ids.dims()[1];
        let hidden_size = self.config.model.hidden_size;

        // Flatten to [B*S] for index_select
        let flat_ids = token_ids.flatten_all()?;

        if self.embed_transposed {
            // Column-gather path for [hidden, vocab] storage
            // This avoids materializing the full transpose
            let weight = self.embed_tokens.embeddings();

            // index_select on dim=1 gathers columns from [H, V]
            // Result: [H, B*S]
            let cols = weight.index_select(&flat_ids, 1)?;

            // Transpose to [B*S, H] (small transpose, only B*S elements)
            let embeddings = cols.t()?;

            // Reshape to [B, S, H]
            Ok(embeddings.reshape(&[batch_size, seq_len, hidden_size])?)
        } else {
            // Row-gather path for standard [vocab, hidden] storage
            let weight = self.embed_tokens.embeddings();

            // index_select on dim=0 gathers rows from [V, H]
            // Result: [B*S, H]
            let rows = weight.index_select(&flat_ids, 0)?;

            // Reshape to [B, S, H]
            Ok(rows.reshape(&[batch_size, seq_len, hidden_size])?)
        }
    }

    /// Teacher-forcing forward: full sequence `[B,T] -> [B,T,V]` logits
    ///
    /// This implementation mirrors the incremental decoding path by
    /// processing tokens step-by-step with a KV cache. This ensures that
    /// rotary (or absolute) positional encodings are applied per layer with
    /// the correct positions and that a causal mask prevents attending to
    /// future tokens.
    pub fn forward_full(&self, token_ids: &Tensor) -> Result<Tensor> {
        // Token ids expected shape: [B,T]
        let (batch_size, seq_len) = token_ids.dims2()?;

        // Embed the entire sequence once.
        let flat_ids = token_ids.flatten_all()?;
        let ids_vec: Vec<u32> = flat_ids.to_vec1()?;
        let hidden = self.embed(&ids_vec)?;
        let hidden_size = self.config.model.hidden_size;
        let hidden = hidden.reshape(&[batch_size, seq_len, hidden_size])?;

        // Probe A1: Embedding RMS (step 0 only)
        if trace_rms_enabled() {
            static EMB_LOGGED: std::sync::Once = std::sync::Once::new();
            EMB_LOGGED.call_once(|| {
                let _ = (|| -> candle_core::Result<()> {
                    let emb_vec = hidden.narrow(1, 0, 1)?.flatten_all()?.to_vec1::<f32>()?;
                    let rms = (emb_vec.iter().map(|x| x * x).sum::<f32>()
                        / emb_vec.len().max(1) as f32)
                        .sqrt();
                    eprintln!("trace: emb_rms={:.6}", rms);
                    Ok(())
                })();
            });
        }

        // Tracepoint 1: Embeddings output (after embed, before layers)
        #[cfg(feature = "trace")]
        {
            use bitnet_trace::dump_trace;
            // Extract first token's embedding for tracing [B, 1, H]
            let first_token_emb = hidden.narrow(1, 0, 1)?;
            let _ = dump_trace(
                "embeddings",
                &first_token_emb,
                Some(0),            // seq=0 (prefill step)
                Some(-1),           // layer=-1 (pre-layer operation)
                Some("embeddings"), // stage name
            );
        }

        // Create per-layer KV cache so that rotary/absolute positional
        // encodings use the proper positions during iterative decoding.
        let mut kv_cache = KVCache::new(&self.config, batch_size, &self.device)?;

        // Collect logits for each position.
        let mut logits_steps = Vec::with_capacity(seq_len);
        for t in 0..seq_len {
            // Select the current token's embedding as [B, 1, H] (keep seq dim for attention)
            let step_hidden = hidden.narrow(1, t, 1)?;

            // Run through all layers using the incremental path which applies
            // positional encoding per layer and causal masking internally.
            let step_hidden = self.forward(step_hidden, Some(&mut kv_cache))?;

            // Tracepoint: All layers output for this position
            #[cfg(feature = "trace")]
            {
                use bitnet_trace::dump_trace;
                let _ = dump_trace(
                    &format!("t{}_all_layers_out", t),
                    &step_hidden,
                    Some(t),                // seq=t (current position)
                    Some(-2),               // layer=-2 (post-all-layers)
                    Some("all_layers_out"), // stage name
                );
            }

            // Project to vocabulary logits for this step.
            let step_logits = self.logits(&step_hidden)?;

            // Trace logits for this position
            #[cfg(feature = "trace")]
            {
                use bitnet_trace::dump_trace;
                let _ = dump_trace(
                    &format!("t{}_logits", t),
                    &step_logits,
                    Some(t),        // seq=t (current position)
                    Some(-1),       // layer=-1 (post-layers stage)
                    Some("logits"), // stage name
                );
            }

            logits_steps.push(step_logits);
        }

        // Stack logits: handle both [B,V] and [B,1,V] shapes
        let logits = if logits_steps[0].dims().len() == 2 {
            // logits are [B, V], stack them to [B, T, V]
            let logits_2d: Vec<_> = logits_steps
                .iter()
                .map(|t| t.unsqueeze(1))
                .collect::<std::result::Result<Vec<_>, _>>()?;
            Tensor::cat(&logits_2d, 1)?
        } else {
            // logits are [B, 1, V], concatenate along time dimension
            Tensor::cat(&logits_steps, 1)?
        };

        // Tracepoint 5: Final logits (first token only)
        #[cfg(feature = "trace")]
        {
            // Extract first token's logits for tracing [B, 1, V]
            let first_token_logits = logits.narrow(1, 0, 1)?;
            bitnet_trace::dump_trace(
                "t0/logits",
                &first_token_logits,
                Some(0),
                Some(-1),
                Some("logits"),
            )
            .map_err(BitNetError::from)?;
        }

        Ok(logits)
    }

    /// Forward pass through transformer layers
    ///
    /// **Performance note**: Accepts ownership of `hidden` to avoid cloning on hot path.
    /// Caller should pass owned tensor or use `.clone()` explicitly if needed.
    pub fn forward(&self, hidden: Tensor, kv_cache: Option<&mut KVCache>) -> Result<Tensor> {
        self.forward_impl(hidden, kv_cache, None)
    }

    pub fn forward_with_workspace(
        &self,
        hidden: Tensor,
        kv_cache: Option<&mut KVCache>,
        workspace: &mut TransformerForwardWorkspace,
    ) -> Result<Tensor> {
        workspace.record_model_input(&hidden);
        let output = self.forward_impl(hidden, kv_cache, Some(workspace))?;
        workspace.record_model_output(&output);
        Ok(output)
    }

    fn forward_impl(
        &self,
        hidden: Tensor,
        mut kv_cache: Option<&mut KVCache>,
        mut workspace: Option<&mut TransformerForwardWorkspace>,
    ) -> Result<Tensor> {
        let mut x = hidden; // Take ownership - no clone needed!

        // Tracepoint 1: Embeddings (incremental path - single token)
        // This captures the embedding for the current token being processed
        #[cfg(feature = "trace")]
        {
            // For incremental path, hidden is already [B, H] (single token)
            // Trace it directly without narrowing (unlike forward_full which has [B, T, H])
            bitnet_trace::dump_trace("t0/embeddings", &x, Some(0), Some(-1), Some("embeddings"))
                .map_err(BitNetError::from)?;
        }

        // Debug input activation norm
        if debug_attn_enabled()
            && let Ok(norm) = x.sqr()?.mean_all()?.sqrt()?.to_scalar::<f32>()
        {
            eprintln!("[norm] input: {:.6e}", norm);
        }

        for (i, layer) in self.layers.iter().enumerate() {
            let layer_cache = kv_cache.as_mut().and_then(|c| c.layer_mut(i));
            x = if let Some(workspace) = workspace.as_mut() {
                layer.forward_with_workspace(
                    &x,
                    layer_cache,
                    &self.raw_tensors,
                    &self.dense_linear_hooks,
                    workspace,
                )?
            } else {
                layer.forward(&x, layer_cache, &self.raw_tensors, &self.dense_linear_hooks)?
            };

            // Debug layer activation norms (show all layers when debugging)
            if debug_attn_enabled()
                && let Ok(norm) = x.sqr()?.mean_all()?.sqrt()?.to_scalar::<f32>()
            {
                eprintln!("[norm] layer {i}: {:.6e}", norm);
            }
        }

        let normalized = self.norm.forward(&x)?;
        qwen_trace_tensor("model.final_norm", None, &normalized)?;
        if debug_attn_enabled()
            && let Ok(norm) = normalized.sqr()?.mean_all()?.sqrt()?.to_scalar::<f32>()
        {
            eprintln!("[norm] final: {:.6e}", norm);
        }

        Ok(normalized)
    }

    pub fn logits(&self, hidden: &Tensor) -> Result<Tensor> {
        let vocab_size = self.config.model.vocab_size;
        let has_tied_qk256_output = self.raw_tensors.contains_key(TIED_EMBED_QK256_KEY);
        qwen_trace_tensor("lm_head.input_hidden", None, hidden)?;
        qwen_trace_event(
            "lm_head.metadata",
            &format!(
                "\"lm_head_present\":{},\"lm_head_transposed\":{},\"embed_transposed\":{},\"has_cached_tied_weight\":{},\"has_tied_qk256_output\":{}",
                self.lm_head.is_some(),
                self.lm_head_transposed,
                self.embed_transposed,
                self.embed_tied_weight.is_some(),
                has_tied_qk256_output
            ),
        );

        match hidden.rank() {
            2 => {
                // [B, H] - last token only
                let (b, _h) = (hidden.dims()[0], hidden.dims()[1]);

                let logits = if self.lm_head_transposed {
                    if let Some(ref weight) = self.lm_head_weight {
                        hidden.matmul(weight)?.reshape(&[b, vocab_size])?
                    } else if let Some(ref lm_head) = self.lm_head {
                        let logits = lm_head.forward(hidden)?; // [B, V]
                        logits.reshape(&[b, vocab_size])?
                    } else {
                        return Err(BitNetError::Validation(
                            "lm_head is marked transposed but lm_head.weight is unavailable".into(),
                        ));
                    }
                } else if let Some(ref lm_head) = self.lm_head {
                    // Use dedicated LM head if available
                    let logits = lm_head.forward(hidden)?; // [B, V]
                    logits.reshape(&[b, vocab_size])?
                } else if let Some(qk256_tensor) = self.raw_tensors.get(TIED_EMBED_QK256_KEY) {
                    static LOGGED_QK256_TIED: std::sync::Once = std::sync::Once::new();
                    LOGGED_QK256_TIED.call_once(|| {
                        tracing::info!(
                            "LM head tied to raw QK256 token embeddings for BitNet.cpp parity"
                        );
                    });
                    let inline_scale = qk256_inline_scale(&self.raw_tensors, TIED_EMBED_QK256_KEY)?;
                    forward_qk256_with_scale(
                        hidden,
                        qk256_tensor,
                        TIED_EMBED_QK256_KEY,
                        inline_scale,
                    )?
                } else {
                    // Tied weights: use embedding matrix
                    static LOGGED: std::sync::Once = std::sync::Once::new();
                    LOGGED.call_once(|| {
                        tracing::info!("LM head tied to input embeddings");
                    });

                    let result = if self.embed_transposed {
                        // Embeddings are [hidden, vocab]
                        let embeddings = self.embed_tokens.embeddings();
                        hidden.matmul(embeddings)? // [B, V]
                    } else if let Some(ref cached_weight) = self.embed_tied_weight {
                        // Use pre-transposed cached weight [H, V] - avoids per-step transpose!
                        hidden.matmul(cached_weight)? // [B, V]
                    } else {
                        // Fallback: transpose on-demand (should be rare after optimization)
                        let embeddings = self.embed_tokens.embeddings();
                        let w = embeddings.transpose(0, 1)?; // [H, V]
                        hidden.matmul(&w)? // [B, V]
                    };

                    // Debug: sanity check tied embeddings orientation (runs once)
                    if std::env::var("BITNET_DEBUG_LOGITS").is_ok() {
                        static SANITY_LOGGED: std::sync::Once = std::sync::Once::new();
                        SANITY_LOGGED.call_once(|| {
                            if let Ok(mean_val) = result.mean_all().and_then(|m| m.to_scalar::<f32>())
                                && let Ok(std_val) = result.broadcast_sub(&result.mean_all().unwrap())
                                    .and_then(|d| d.sqr())
                                    .and_then(|s| s.mean_all())
                                    .and_then(|v| v.sqrt())
                                    .and_then(|s| s.to_scalar::<f32>())
                            {
                                tracing::info!("tied logits sanity check - mean/std: {:.4}/{:.4}", mean_val, std_val);

                                // Float sanity check: compare with non-quantized path
                                if let Ok(emb) = self.embed_tokens.embeddings().transpose(0, 1)
                                    && let Ok(ref_logits) = hidden.matmul(&emb)
                                    && let Ok(ref_mean) = ref_logits.mean_all().and_then(|m| m.to_scalar::<f32>())
                                    && let Ok(ref_std) = ref_logits.broadcast_sub(&ref_logits.mean_all().unwrap())
                                        .and_then(|d| d.sqr())
                                        .and_then(|s| s.mean_all())
                                        .and_then(|v| v.sqrt())
                                        .and_then(|s| s.to_scalar::<f32>())
                                {
                                    tracing::info!("float ref logits - mean/std: {:.4}/{:.4}", ref_mean, ref_std);
                                    tracing::info!("correlation check: quantized vs float stats should be similar");
                                }
                            }
                        });
                    }

                    result
                };

                // Debug logits std
                if debug_attn_enabled()
                    && let Ok(mean) = logits.mean_all()
                    && let Ok(diff) = logits.broadcast_sub(&mean)
                    && let Ok(variance) = diff.sqr()?.mean_all()
                    && let Ok(std_val) = variance.sqrt()?.to_scalar::<f32>()
                {
                    eprintln!("[norm] logits std: {:.6e}", std_val);
                }
                qwen_trace_tensor("lm_head.logits", None, &logits)?;

                // Tracepoint 5: Logits (incremental path - single token)
                // This captures the final logits for the current token [B, V]
                #[cfg(feature = "trace")]
                {
                    // For incremental path, logits are [B, V] (single token)
                    // Trace directly without narrowing (unlike forward_full which has [B, T, V])
                    bitnet_trace::dump_trace(
                        "t0/logits",
                        &logits,
                        Some(0),
                        Some(-1),
                        Some("logits"),
                    )
                    .map_err(BitNetError::from)?;
                }

                Ok(logits)
            }
            3 => {
                // [B, T, H] - all timesteps
                let (b, t, h) = (hidden.dims()[0], hidden.dims()[1], hidden.dims()[2]);

                if self.lm_head_transposed {
                    if let Some(ref weight) = self.lm_head_weight {
                        // LM head weight is stored as [hidden, vocab].
                        let hidden_2d = hidden.reshape(&[b * t, h])?;
                        let logits_2d = hidden_2d.matmul(weight)?;
                        Ok(logits_2d.reshape(&[b, t, vocab_size])?)
                    } else if let Some(ref lm_head) = self.lm_head {
                        let hidden_2d = hidden.reshape(&[b * t, h])?;
                        let logits_2d = lm_head.forward(&hidden_2d)?;
                        Ok(logits_2d.reshape(&[b, t, vocab_size])?)
                    } else {
                        Err(BitNetError::Validation(
                            "lm_head is marked transposed but lm_head.weight is unavailable".into(),
                        ))
                    }
                } else if let Some(ref lm_head) = self.lm_head {
                    // Use dedicated LM head if available
                    // Standard path: LM head weight is [vocab, hidden]
                    // Flatten to 2D for proper matmul
                    let hidden_2d = hidden.reshape(&[b * t, h])?;
                    let logits_2d = lm_head.forward(&hidden_2d)?;
                    Ok(logits_2d.reshape(&[b, t, vocab_size])?)
                } else if let Some(qk256_tensor) = self.raw_tensors.get(TIED_EMBED_QK256_KEY) {
                    static LOGGED_QK256_TIED: std::sync::Once = std::sync::Once::new();
                    LOGGED_QK256_TIED.call_once(|| {
                        tracing::info!(
                            "LM head tied to raw QK256 token embeddings for BitNet.cpp parity"
                        );
                    });
                    let inline_scale = qk256_inline_scale(&self.raw_tensors, TIED_EMBED_QK256_KEY)?;
                    let logits = forward_qk256_with_scale(
                        hidden,
                        qk256_tensor,
                        TIED_EMBED_QK256_KEY,
                        inline_scale,
                    )?;
                    Ok(logits.reshape(&[b, t, vocab_size])?)
                } else {
                    // Tied weights: use embedding matrix
                    static LOGGED: std::sync::Once = std::sync::Once::new();
                    LOGGED.call_once(|| {
                        tracing::info!("LM head tied to input embeddings");
                    });

                    if self.embed_transposed {
                        // Embeddings are [hidden, vocab], flatten hidden for matmul
                        let embeddings = self.embed_tokens.embeddings();
                        let hidden_2d = hidden.reshape(&[b * t, h])?;
                        let logits_2d = hidden_2d.matmul(embeddings)?;
                        Ok(logits_2d.reshape(&[b, t, vocab_size])?)
                    } else if let Some(ref cached_weight) = self.embed_tied_weight {
                        // Use pre-transposed cached weight [H, V] - avoids per-step transpose!
                        let hidden_2d = hidden.reshape(&[b * t, h])?;
                        let logits_2d = hidden_2d.matmul(cached_weight)?;
                        Ok(logits_2d.reshape(&[b, t, vocab_size])?)
                    } else {
                        // Fallback: transpose on-demand (should be rare after optimization)
                        let embeddings = self.embed_tokens.embeddings();
                        let w = embeddings.transpose(0, 1)?; // [H, V]
                        let hidden_2d = hidden.reshape(&[b * t, h])?;
                        let logits_2d = hidden_2d.matmul(&w)?;
                        Ok(logits_2d.reshape(&[b, t, vocab_size])?)
                    }
                }
            }
            _ => Err(BitNetError::Validation("unexpected hidden rank".into())),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bitnet_common::config::ModelConfig;
    use candle_nn::RmsNorm;
    use serial_test::serial;
    use std::collections::HashMap;

    /// Helper to compute RMS (root mean square) of a tensor
    fn compute_rms(tensor: &Tensor) -> candle_core::Result<f64> {
        let squared = tensor.sqr()?;
        let mean = squared.mean_all()?;
        let rms = mean.sqrt()?.to_scalar::<f32>()? as f64;
        Ok(rms)
    }

    #[test]
    fn test_relu2_activation_squares_positive_values() -> Result<()> {
        let device = Device::Cpu;
        let input = Tensor::from_slice(&[-2.0f32, -0.5, 0.0, 2.0, 3.0], (5,), &device)?;
        let output = apply_ffn_activation(&input, ActivationType::Relu2)?;
        let values = output.to_vec1::<f32>()?;

        assert_eq!(values, vec![0.0, 0.0, 0.0, 4.0, 9.0]);
        Ok(())
    }

    #[test]
    fn test_layer_norm_with_standard_gamma() -> candle_core::Result<()> {
        // Test that RMSNorm behaves correctly with standard gamma (RMS ≈ 1.0)
        let device = Device::Cpu;
        let hidden_size = 2560;
        let eps = 1e-5;

        // Create input tensor [1, 1, 2560]
        let input_data: Vec<f32> = (0..hidden_size)
            .map(|i| {
                let x = i as f32 / hidden_size as f32;
                ((x * 10.0).sin() + (x * 20.0).cos()) * 0.5
            })
            .collect();

        let input = Tensor::from_slice(&input_data, (1, 1, hidden_size), &device)?;

        // Create standard gamma (all ones)
        let gamma = Tensor::ones(hidden_size, DType::F32, &device)?;

        // Apply RMSNorm
        let rms_norm = RmsNorm::new(gamma, eps);
        let output = rms_norm.forward(&input)?;

        // Verify output RMS is reasonable (should be close to 1.0)
        let output_rms = compute_rms(&output)?;

        assert!(
            output_rms > 0.5 && output_rms < 2.0,
            "Output RMS should be reasonable with standard gamma, got {:.6e}",
            output_rms
        );

        // Verify no NaN/Inf
        let vec_data: Vec<f32> = output.flatten_all()?.to_vec1()?;
        let has_nan = vec_data.iter().any(|x| x.is_nan());
        let has_inf = vec_data.iter().any(|x| x.is_infinite());
        assert!(!has_nan, "Output should not contain NaN");
        assert!(!has_inf, "Output should not contain Inf");

        Ok(())
    }

    #[test]
    fn attention_f16_dot_input_uses_f16_roundtrip_values() -> Result<()> {
        let device = Device::Cpu;
        let input =
            Tensor::from_slice(&[1.0003f32, -2.0007, 3.1259, -4.2509], (1, 1, 1, 4), &device)?;

        let output = attention_f16_dot_input(&input)?;
        let values = output.flatten_all()?.to_vec1::<f32>()?;

        assert_eq!(values, vec![1.0, -2.0, 3.125, -4.25]);
        Ok(())
    }

    #[test]
    fn attention_score_key_input_preserves_f32_values() -> Result<()> {
        let device = Device::Cpu;
        let input =
            Tensor::from_slice(&[1.0003f32, -2.0007, 3.1259, -4.2509], (1, 1, 1, 4), &device)?;

        let output = attention_score_key_input(&input)?;
        let values = output.flatten_all()?.to_vec1::<f32>()?;

        assert_eq!(values, vec![1.0003, -2.0007, 3.1259, -4.2509]);
        Ok(())
    }

    #[test]
    fn exact_q8_sidecar_runtime_hook_matches_dense_linear_reference() -> Result<()> {
        let device = Device::Cpu;
        let weight = Tensor::from_slice(&[0.5f32, 1.0, 1.5, 2.0], (2, 2), &device)?;
        let linear = Linear::new(weight, None);
        let input = Tensor::from_slice(&[2.0f32, 3.0], (1, 1, 2), &device)?;

        let mut packed = Vec::new();
        packed.extend_from_slice(&f32_to_fp16(0.5).to_le_bytes());
        for value in [1i8, 2, 3, 4] {
            packed.push(value as u8);
        }
        packed.resize(34, 0);

        let mut hooks = DenseLinearRuntimeHookRegistry::default();
        hooks.insert(
            SLM_CPU_067_EXACT_Q8_RUNTIME_TENSOR.to_string(),
            DenseLinearRuntimeHookDescriptor {
                tensor_name: "blk.0.attn_q.weight".to_string(),
                role: "AttentionQ".to_string(),
                sidecar_payload_sha256: Some("sha256:test".to_string()),
                packed_q8_payload: Some(DenseLinearPackedQ8Payload {
                    tensor_name: "blk.0.attn_q.weight".to_string(),
                    packed_q8_bytes: std::sync::Arc::from(packed.into_boxed_slice()),
                    q8_block_size: 32,
                    q8_block_count: 1,
                    matrix_rows: 2,
                    matrix_cols: 2,
                }),
                runtime_compute_enabled: true,
            },
        );

        let Some(output) = maybe_forward_dense_q8_sidecar_linear(
            &input,
            &linear,
            SLM_CPU_067_EXACT_Q8_RUNTIME_TENSOR,
            &hooks,
        )?
        else {
            return Err(BitNetError::Validation(
                "expected exact Q8 sidecar runtime hook to run".to_string(),
            ));
        };

        assert_eq!(output.dims(), &[1, 1, 2]);
        assert_eq!(output.flatten_all()?.to_vec1::<f32>()?, vec![4.0, 9.0]);
        assert_eq!(
            output.flatten_all()?.to_vec1::<f32>()?,
            linear.forward(&input)?.flatten_all()?.to_vec1::<f32>()?
        );
        Ok(())
    }

    #[test]
    fn attention_score_qk_inputs_match_reference_precision_contract() -> Result<()> {
        let device = Device::Cpu;
        let query =
            Tensor::from_slice(&[1.0003f32, -2.0007, 3.1259, -4.2509], (1, 1, 1, 4), &device)?;
        let key = Tensor::from_slice(&[5.0003f32, 6.0007, -7.1259, 8.2509], (1, 1, 1, 4), &device)?;

        let q_values = attention_f16_dot_input(&query)?.flatten_all()?.to_vec1::<f32>()?;
        let k_values = attention_score_key_input(&key)?.flatten_all()?.to_vec1::<f32>()?;
        let score = q_values.iter().zip(k_values.iter()).fold(0.0f32, |sum, (q, k)| sum + q * k);

        assert_eq!(q_values, vec![1.0, -2.0, 3.125, -4.25]);
        assert_eq!(k_values, vec![5.0003, 6.0007, -7.1259, 8.2509]);
        assert_eq!(score, -64.33586);

        Ok(())
    }

    #[test]
    fn attention_value_mix_uses_f16_roundtrip_values() -> Result<()> {
        let device = Device::Cpu;
        let weights = Tensor::from_slice(&[0.25f32, 0.25, 0.25, 0.25], (1, 1, 1, 4), &device)?;
        let values =
            Tensor::from_slice(&[1.0003f32, -2.0007, 3.1259, -4.2509], (1, 1, 4, 1), &device)?;

        let rounded_values = attention_f16_dot_input(&values)?;
        let mixed = weights.matmul(&rounded_values)?;
        let mixed = mixed.flatten_all()?.to_vec1::<f32>()?;

        assert_eq!(mixed, vec![-0.53125]);
        Ok(())
    }

    #[test]
    fn test_layer_norm_with_small_gamma() -> candle_core::Result<()> {
        // Test RMSNorm with gamma RMS ≈ 0.018 (our model's case)
        let device = Device::Cpu;
        let hidden_size = 2560;
        let eps = 1e-5;

        // Create input tensor [1, 1, 2560]
        let input_data: Vec<f32> = (0..hidden_size)
            .map(|i| {
                let x = i as f32 / hidden_size as f32;
                ((x * 10.0).sin() + (x * 20.0).cos()) * 0.5
            })
            .collect();

        let input = Tensor::from_slice(&input_data, (1, 1, hidden_size), &device)?;

        // Create gamma with RMS ≈ 1/√2560 ≈ 0.01976
        let target_rms = 1.0 / (hidden_size as f64).sqrt();
        let gamma_data: Vec<f32> = vec![target_rms as f32; hidden_size];
        let gamma = Tensor::from_slice(&gamma_data, hidden_size, &device)?;

        // Verify gamma RMS
        let gamma_rms = compute_rms(&gamma)?;
        assert!(
            (gamma_rms - target_rms).abs() < 0.001,
            "Gamma RMS should be close to {:.6e}, got {:.6e}",
            target_rms,
            gamma_rms
        );

        // Apply RMSNorm
        let rms_norm = RmsNorm::new(gamma, eps);
        let output = rms_norm.forward(&input)?;

        // Verify output RMS is smaller but reasonable
        let output_rms = compute_rms(&output)?;

        assert!(
            output_rms > 0.001 && output_rms < 0.1,
            "Output RMS should be reasonable with small gamma, got {:.6e}",
            output_rms
        );

        // Verify no NaN/Inf
        let vec_data: Vec<f32> = output.flatten_all()?.to_vec1()?;
        let has_nan = vec_data.iter().any(|x| x.is_nan());
        let has_inf = vec_data.iter().any(|x| x.is_infinite());
        assert!(!has_nan, "Output should not contain NaN");
        assert!(!has_inf, "Output should not contain Inf");

        Ok(())
    }

    #[test]
    #[serial]
    fn test_layer_norm_with_optional_bias() -> candle_core::Result<()> {
        // Test layer_norm_with_optional_bias helper with no-bias LayerNorm path
        let device = Device::Cpu;
        let hidden_size = 128;
        let eps = 1e-5;

        // Create VarBuilder with only weight (no bias)
        use std::collections::HashMap;

        let mut tensors = HashMap::new();
        let weight = Tensor::ones(hidden_size, DType::F32, &device)?;
        tensors.insert("weight".to_string(), weight);

        let vb = VarBuilder::from_tensors(tensors, DType::F32, &device);

        // Create LayerNorm (should use no-bias LayerNorm path due to missing bias)
        let layer_norm = layer_norm_with_optional_bias(hidden_size, eps, vb)?;

        // Test forward pass
        let input_data: Vec<f32> =
            (0..hidden_size).map(|i| (i as f32 / hidden_size as f32).sin()).collect();
        let input = Tensor::from_slice(&input_data, (1, hidden_size), &device)?;

        let output = layer_norm.forward(&input)?;

        // Verify output shape
        assert_eq!(output.shape(), input.shape());

        // Verify no NaN/Inf
        let vec_data: Vec<f32> = output.flatten_all()?.to_vec1()?;
        let has_nan = vec_data.iter().any(|x| x.is_nan());
        let has_inf = vec_data.iter().any(|x| x.is_infinite());
        assert!(!has_nan, "Output should not contain NaN");
        assert!(!has_inf, "Output should not contain Inf");

        Ok(())
    }

    #[test]
    fn test_linear_with_optional_bias_uses_no_bias_path() -> candle_core::Result<()> {
        let device = Device::Cpu;
        let weight = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], (2, 2), &device)?;
        let input = Tensor::from_slice(&[5.0f32, 6.0], (1, 2), &device)?;

        let mut tensors = HashMap::new();
        tensors.insert("weight".to_string(), weight.clone());
        let vb = VarBuilder::from_tensors(tensors, DType::F32, &device);

        let no_bias = linear_with_optional_bias(2, 2, vb)?;
        let no_bias_output = no_bias.forward(&input)?;

        let explicit_zero_bias = Linear::new(weight, Some(Tensor::zeros(2, DType::F32, &device)?));
        let zero_bias_output = explicit_zero_bias.forward(&input)?;

        assert_eq!(no_bias_output.to_vec2::<f32>()?, zero_bias_output.to_vec2::<f32>()?);
        assert_eq!(no_bias_output.to_vec2::<f32>()?, vec![vec![17.0, 39.0]]);

        Ok(())
    }

    #[test]
    fn test_rms_norm_type_does_not_remove_mean() -> candle_core::Result<()> {
        let device = Device::Cpu;
        let hidden_size = 3;
        let eps = 1e-5;

        use std::collections::HashMap;

        let mut tensors = HashMap::new();
        tensors.insert("weight".to_string(), Tensor::ones(hidden_size, DType::F32, &device)?);

        let vb = VarBuilder::from_tensors(tensors, DType::F32, &device);
        let norm = norm_with_optional_bias(NormType::RmsNorm, hidden_size, eps, vb)?;

        let input = Tensor::from_slice(&[1.0f32, 2.0, 3.0], (1, hidden_size), &device)?;
        let output = norm.forward(&input)?;
        let values = output.to_vec2::<f32>()?;

        assert!(
            values[0].iter().all(|value| *value > 0.0),
            "RMSNorm should preserve positive sign instead of mean-centering: {values:?}"
        );

        Ok(())
    }

    #[test]
    fn transposed_lm_head_uses_dedicated_output_weight() -> Result<()> {
        use std::collections::HashMap;

        let device = Device::Cpu;
        let vocab_size = 4;
        let hidden_size = 2;
        let mut config = BitNetConfig::default();
        config.model.vocab_size = vocab_size;
        config.model.hidden_size = hidden_size;
        config.model.num_layers = 0;

        let mut tensors = HashMap::new();
        tensors.insert(
            "embed_tokens.weight".to_string(),
            Tensor::zeros((vocab_size, hidden_size), DType::F32, &device)?,
        );
        tensors.insert(
            "final_norm.weight".to_string(),
            Tensor::ones(hidden_size, DType::F32, &device)?,
        );
        tensors.insert(
            "final_norm.bias".to_string(),
            Tensor::zeros(hidden_size, DType::F32, &device)?,
        );
        tensors.insert(
            "lm_head.weight".to_string(),
            Tensor::from_slice(
                &[
                    1.0f32, 0.0, 0.0, 0.0, // hidden dim 0 -> token 0
                    0.0, 1.0, 0.0, 0.0, // hidden dim 1 -> token 1
                ],
                (hidden_size, vocab_size),
                &device,
            )?,
        );
        tensors.insert(
            "lm_head.transposed".to_string(),
            Tensor::from_slice(&[1.0f32], (1,), &device)?,
        );

        let vb = VarBuilder::from_tensors(tensors, DType::F32, &device);
        let model = TransformerModel::new_with_tensors(config, vb, HashMap::new())?;

        assert!(
            model.lm_head_transposed,
            "transposed lm_head flag must survive model construction"
        );
        assert!(
            model.lm_head_weight.is_some(),
            "transposed lm_head must keep its dedicated output weight"
        );
        assert!(
            model.embed_tied_weight.is_none(),
            "dedicated transposed lm_head must not fall back to tied embeddings"
        );

        let hidden = Tensor::from_slice(&[2.0f32, 3.0], (1, hidden_size), &device)?;
        let logits = model.logits(&hidden)?;
        let values = logits.to_vec2::<f32>()?;
        assert_eq!(values, vec![vec![2.0, 3.0, 0.0, 0.0]]);

        Ok(())
    }

    #[test]
    fn tied_embedding_logits_use_cached_embedding_transpose() -> Result<()> {
        use std::collections::HashMap;

        let device = Device::Cpu;
        let vocab_size = 4;
        let hidden_size = 3;
        let mut config = BitNetConfig::default();
        config.model.vocab_size = vocab_size;
        config.model.hidden_size = hidden_size;
        config.model.num_layers = 0;

        let mut tensors = HashMap::new();
        tensors.insert(
            "embed_tokens.weight".to_string(),
            Tensor::from_slice(
                &[
                    1.0f32, 0.0, 0.0, // token 0
                    0.0, 1.0, 0.0, // token 1
                    0.0, 0.0, 1.0, // token 2
                    1.0, 1.0, 1.0, // token 3
                ],
                (vocab_size, hidden_size),
                &device,
            )?,
        );
        tensors.insert(
            "final_norm.weight".to_string(),
            Tensor::ones(hidden_size, DType::F32, &device)?,
        );
        tensors.insert(
            "final_norm.bias".to_string(),
            Tensor::zeros(hidden_size, DType::F32, &device)?,
        );

        let vb = VarBuilder::from_tensors(tensors, DType::F32, &device);
        let model = TransformerModel::new_with_tensors(config, vb, HashMap::new())?;

        assert!(
            model.lm_head.is_none() && model.lm_head_weight.is_none(),
            "missing lm_head must use tied embeddings"
        );
        assert!(
            model.embed_tied_weight.is_some(),
            "tied embedding logits should cache [hidden, vocab] weight"
        );

        let hidden_2d = Tensor::from_slice(&[2.0f32, 3.0, 5.0], (1, hidden_size), &device)?;
        let logits_2d = model.logits(&hidden_2d)?;
        assert_eq!(logits_2d.to_vec2::<f32>()?, vec![vec![2.0, 3.0, 5.0, 10.0]]);

        let hidden_3d = Tensor::from_slice(
            &[
                2.0f32, 3.0, 5.0, // step 0
                7.0, 11.0, 13.0, // step 1
            ],
            (1, 2, hidden_size),
            &device,
        )?;
        let logits_3d = model.logits(&hidden_3d)?;
        assert_eq!(
            logits_3d.to_vec3::<f32>()?,
            vec![vec![vec![2.0, 3.0, 5.0, 10.0], vec![7.0, 11.0, 13.0, 31.0]]]
        );

        Ok(())
    }

    #[test]
    #[serial(bitnet_env)]
    fn test_layer_norm_requires_bias_when_guard_enabled() -> candle_core::Result<()> {
        let device = Device::Cpu;
        let hidden_size = 64;
        let eps = 1e-5;

        use std::collections::HashMap;

        let mut tensors = HashMap::new();
        let weight = Tensor::ones(hidden_size, DType::F32, &device)?;
        tensors.insert("weight".to_string(), weight);

        let vb = VarBuilder::from_tensors(tensors, DType::F32, &device);

        let mut scope = bitnet_test_support::EnvScope::new();
        scope.set("BITNET_REQUIRE_LAYER_NORM_BIAS", "1");

        let err = layer_norm_with_optional_bias(hidden_size, eps, vb)
            .expect_err("missing bias should error when BITNET_REQUIRE_LAYER_NORM_BIAS=1");
        assert!(
            err.to_string().contains("LayerNorm bias tensor is required"),
            "unexpected error: {err}"
        );

        Ok(())
    }

    fn tiny_bitnet_config() -> BitNetConfig {
        BitNetConfig {
            model: ModelConfig {
                hidden_size: 2,
                vocab_size: 8,
                num_heads: 1,
                num_key_value_heads: 1,
                num_layers: 1,
                intermediate_size: 2,
                max_position_embeddings: 8,
                rms_norm_eps: Some(1e-5),
                norm_type: NormType::LayerNorm,
                ..Default::default()
            },
            ..Default::default()
        }
    }

    fn identity_2(device: &Device) -> candle_core::Result<Tensor> {
        Tensor::from_vec(vec![1.0f32, 0.0, 0.0, 1.0], &[2, 2], device)
    }

    #[test]
    #[serial]
    fn attention_applies_bitnet_sub_layernorm_before_output_projection() -> Result<()> {
        let device = Device::Cpu;
        let config = tiny_bitnet_config();
        let mut tensors = HashMap::new();
        for name in ["q_proj.weight", "k_proj.weight", "v_proj.weight", "o_proj.weight"] {
            tensors.insert(name.to_string(), identity_2(&device)?);
        }
        tensors.insert("sub_layernorm.weight".to_string(), Tensor::ones(2, DType::F32, &device)?);
        tensors.insert("sub_layernorm.bias".to_string(), Tensor::zeros(2, DType::F32, &device)?);

        let vb = VarBuilder::from_tensors(tensors, DType::F32, &device);
        let attention = MultiHeadAttention::new(&config, vb, 0)?;
        let x = Tensor::from_vec(vec![1.0f32, 3.0], &[1, 1, 2], &device)?;
        let output =
            attention.forward(&x, None, &HashMap::new(), &DenseLinearRuntimeHookRegistry::new())?;
        let values: Vec<f32> = output.flatten_all()?.to_vec1()?;

        assert!(
            values[0] < -0.99 && values[0] > -1.01,
            "attention sub-layernorm should center first value near -1, got {}",
            values[0]
        );
        assert!(
            values[1] > 0.99 && values[1] < 1.01,
            "attention sub-layernorm should center second value near 1, got {}",
            values[1]
        );

        Ok(())
    }

    #[test]
    #[serial]
    fn feed_forward_applies_bitnet_sub_layernorm_before_down_projection() -> Result<()> {
        let device = Device::Cpu;
        let config = tiny_bitnet_config();
        let mut tensors = HashMap::new();
        for name in ["gate_proj.weight", "up_proj.weight", "down_proj.weight"] {
            tensors.insert(name.to_string(), identity_2(&device)?);
        }
        tensors.insert("sub_layernorm.weight".to_string(), Tensor::ones(2, DType::F32, &device)?);
        tensors.insert("sub_layernorm.bias".to_string(), Tensor::zeros(2, DType::F32, &device)?);

        let vb = VarBuilder::from_tensors(tensors, DType::F32, &device);
        let feed_forward = FeedForward::new(&config, vb, 0)?;
        let x = Tensor::from_vec(vec![1.0f32, 2.0], &[1, 1, 2], &device)?;
        let output =
            feed_forward.forward(&x, &HashMap::new(), &DenseLinearRuntimeHookRegistry::new())?;
        let values: Vec<f32> = output.flatten_all()?.to_vec1()?;

        assert!(
            values[0] < -0.99 && values[0] > -1.01,
            "feed-forward sub-layernorm should center first value near -1, got {}",
            values[0]
        );
        assert!(
            values[1] > 0.99 && values[1] < 1.01,
            "feed-forward sub-layernorm should center second value near 1, got {}",
            values[1]
        );

        Ok(())
    }

    #[test]
    fn feed_forward_uses_relu2_activation_when_configured() -> Result<()> {
        let device = Device::Cpu;
        let mut config = tiny_bitnet_config();
        config.model.activation_type = ActivationType::Relu2;
        let mut tensors = HashMap::new();
        for name in ["gate_proj.weight", "up_proj.weight", "down_proj.weight"] {
            tensors.insert(name.to_string(), identity_2(&device)?);
        }

        let vb = VarBuilder::from_tensors(tensors, DType::F32, &device);
        let feed_forward = FeedForward::new(&config, vb, 0)?;
        let x = Tensor::from_vec(vec![1.0f32, 2.0], &[1, 1, 2], &device)?;
        let output =
            feed_forward.forward(&x, &HashMap::new(), &DenseLinearRuntimeHookRegistry::new())?;
        let values: Vec<f32> = output.flatten_all()?.to_vec1()?;

        assert!(
            (values[0] - 1.0).abs() < 1e-5,
            "relu2 gate should produce first output 1, got {}",
            values[0]
        );
        assert!(
            (values[1] - 8.0).abs() < 1e-5,
            "relu2 gate should square ReLU before multiplying by up projection, got {}",
            values[1]
        );

        Ok(())
    }

    #[test]
    fn qk256_inline_scale_reads_sibling_raw_tensor() -> Result<()> {
        let device = Device::Cpu;
        let mut raw_tensors = HashMap::new();
        raw_tensors.insert(
            "layers.0.attention.q_proj.weight.qk256_scale".to_string(),
            Tensor::from_vec(vec![0.25f32], &[1], &device)?,
        );

        let scale = qk256_inline_scale(&raw_tensors, "layers.0.attention.q_proj.weight.qk256_qs")?;

        assert_eq!(scale, Some(0.25));

        Ok(())
    }

    #[test]
    fn logits_prefer_raw_qk256_tied_embeddings_when_present() -> Result<()> {
        let device = Device::Cpu;
        let mut config = BitNetConfig::default();
        config.model.hidden_size = 256;
        config.model.vocab_size = 2;
        config.model.num_layers = 0;
        config.model.num_heads = 1;
        config.model.num_key_value_heads = 1;
        config.model.intermediate_size = 256;
        config.model.rms_norm_eps = Some(1e-5);
        config.model.norm_type = NormType::RmsNorm;

        let mut tensors = HashMap::new();
        tensors.insert(
            "embed_tokens.weight".to_string(),
            Tensor::zeros((2, 256), DType::F32, &device)?,
        );
        tensors.insert("final_norm.weight".to_string(), Tensor::ones(256, DType::F32, &device)?);

        let mut raw_tensors = HashMap::new();
        let mut packed = vec![0x00u8; 64];
        packed.extend(std::iter::repeat_n(0xAAu8, 64));
        raw_tensors.insert(
            TIED_EMBED_QK256_KEY.to_string(),
            Tensor::from_raw_buffer(&packed, DType::U8, &[2, 64], &device)?,
        );
        raw_tensors.insert(
            "embed_tokens.weight.qk256_scale".to_string(),
            Tensor::from_vec(vec![1.0f32], &[1], &device)?,
        );

        let vb = VarBuilder::from_tensors(tensors, DType::F32, &device);
        let model = TransformerModel::new_with_tensors(config, vb, raw_tensors)?;
        let hidden = Tensor::ones((1, 256), DType::F32, &device)?;
        let logits = model.logits(&hidden)?.to_vec2::<f32>()?;

        assert!(
            logits[0][0] < -200.0,
            "first raw QK256 tied logit should come from packed code-0 row, got {}",
            logits[0][0]
        );
        assert!(
            logits[0][1] > 200.0,
            "second raw QK256 tied logit should come from packed code-2 row, got {}",
            logits[0][1]
        );

        Ok(())
    }

    #[test]
    fn transformer_uses_dedicated_lm_head_when_present() -> Result<()> {
        let device = Device::Cpu;
        let mut config = BitNetConfig::default();
        config.model.hidden_size = 4;
        config.model.vocab_size = 3;
        config.model.num_layers = 0;
        config.model.num_heads = 1;
        config.model.num_key_value_heads = 1;
        config.model.intermediate_size = 4;
        config.model.rms_norm_eps = Some(1e-5);
        config.model.norm_type = NormType::RmsNorm;

        let mut tensors = HashMap::new();
        tensors.insert(
            "embed_tokens.weight".to_string(),
            Tensor::from_vec(
                vec![
                    10.0f32, 0.0, 0.0, 0.0, //
                    0.0, 10.0, 0.0, 0.0, //
                    0.0, 0.0, 10.0, 0.0,
                ],
                (3, 4),
                &device,
            )?,
        );
        tensors.insert("final_norm.weight".to_string(), Tensor::ones(4, DType::F32, &device)?);
        tensors.insert(
            "lm_head.weight".to_string(),
            Tensor::from_vec(
                vec![
                    0.0f32, 0.0, 0.0, 0.0, //
                    1.0, 0.0, 0.0, 0.0, //
                    0.0, 0.0, 0.0, 0.0,
                ],
                (3, 4),
                &device,
            )?,
        );

        let vb = VarBuilder::from_tensors(tensors, DType::F32, &device);
        let model = TransformerModel::new_with_tensors(config, vb, HashMap::new())?;

        assert!(model.lm_head.is_some(), "dedicated lm_head.weight must be loaded");
        assert!(
            model.embed_tied_weight.is_none(),
            "explicit lm_head.weight must not fall back to tied embeddings"
        );

        let hidden = Tensor::from_vec(vec![1.0f32, 0.0, 0.0, 0.0], (1, 4), &device)?;
        let logits = model.logits(&hidden)?.to_vec2::<f32>()?;
        assert!(
            logits[0][1] > logits[0][0],
            "logits should come from dedicated lm_head, got {:?}",
            logits[0]
        );

        Ok(())
    }

    #[test]
    fn transformer_uses_dedicated_output_weight_when_present() -> Result<()> {
        let device = Device::Cpu;
        let mut config = BitNetConfig::default();
        config.model.hidden_size = 4;
        config.model.vocab_size = 3;
        config.model.num_layers = 0;
        config.model.num_heads = 1;
        config.model.num_key_value_heads = 1;
        config.model.intermediate_size = 4;
        config.model.rms_norm_eps = Some(1e-5);
        config.model.norm_type = NormType::RmsNorm;

        let mut tensors = HashMap::new();
        tensors.insert(
            "embed_tokens.weight".to_string(),
            Tensor::from_vec(
                vec![
                    10.0f32, 0.0, 0.0, 0.0, //
                    0.0, 10.0, 0.0, 0.0, //
                    0.0, 0.0, 10.0, 0.0,
                ],
                (3, 4),
                &device,
            )?,
        );
        tensors.insert("final_norm.weight".to_string(), Tensor::ones(4, DType::F32, &device)?);
        tensors.insert(
            "output.weight".to_string(),
            Tensor::from_vec(
                vec![
                    0.0f32, 0.0, 0.0, 0.0, //
                    0.0, 0.0, 0.0, 0.0, //
                    1.0, 0.0, 0.0, 0.0,
                ],
                (3, 4),
                &device,
            )?,
        );

        let vb = VarBuilder::from_tensors(tensors, DType::F32, &device);
        let model = TransformerModel::new_with_tensors(config, vb, HashMap::new())?;

        assert!(
            model.lm_head.is_some(),
            "dedicated GGUF output.weight must be loaded as the lm head"
        );
        assert!(
            model.embed_tied_weight.is_none(),
            "explicit output.weight must not fall back to tied embeddings"
        );

        let hidden = Tensor::from_vec(vec![1.0f32, 0.0, 0.0, 0.0], (1, 4), &device)?;
        let logits = model.logits(&hidden)?.to_vec2::<f32>()?;
        assert!(
            logits[0][2] > logits[0][0],
            "logits should come from dedicated output.weight, got {:?}",
            logits[0]
        );

        Ok(())
    }

    #[test]
    fn transformer_reshapes_gguf_output_weight_hidden_vocab_without_transpose() -> Result<()> {
        let device = Device::Cpu;
        let mut config = BitNetConfig::default();
        config.model.hidden_size = 4;
        config.model.vocab_size = 3;
        config.model.num_layers = 0;
        config.model.num_heads = 1;
        config.model.num_key_value_heads = 1;
        config.model.intermediate_size = 4;
        config.model.rms_norm_eps = Some(1e-5);
        config.model.norm_type = NormType::RmsNorm;

        let mut tensors = HashMap::new();
        tensors.insert(
            "embed_tokens.weight".to_string(),
            Tensor::from_vec(
                vec![
                    10.0f32, 0.0, 0.0, 0.0, //
                    0.0, 10.0, 0.0, 0.0, //
                    0.0, 0.0, 10.0, 0.0,
                ],
                (3, 4),
                &device,
            )?,
        );
        tensors.insert("final_norm.weight".to_string(), Tensor::ones(4, DType::F32, &device)?);
        tensors.insert(
            "output.weight".to_string(),
            Tensor::from_vec(
                vec![
                    0.0f32, 0.0, 0.0, 0.0, //
                    0.0, 0.0, 0.0, 0.0, //
                    1.0, 0.0, 0.0, 0.0,
                ],
                (4, 3),
                &device,
            )?,
        );

        let vb = VarBuilder::from_tensors(tensors, DType::F32, &device);
        let model = TransformerModel::new_with_tensors(config, vb, HashMap::new())?;

        assert!(
            model.lm_head.is_some(),
            "GGUF output.weight with hidden/vocab dims must be reshaped into a dedicated lm head"
        );
        assert!(
            !model.lm_head_transposed,
            "GGUF output.weight hidden/vocab dims are token-major storage, not a transposed head"
        );
        assert!(
            model.embed_tied_weight.is_none(),
            "explicit output.weight must not fall back to tied embeddings"
        );
        assert_eq!(
            model.lm_head_weight.as_ref().map(|weight| weight.dims().to_vec()),
            Some(vec![3, 4]),
            "reshaped output.weight should be stored as [vocab, hidden]"
        );

        let hidden = Tensor::from_vec(vec![1.0f32, 0.0, 0.0, 0.0], (1, 4), &device)?;
        let logits = model.logits(&hidden)?.to_vec2::<f32>()?;
        assert_eq!(logits, vec![vec![0.0, 0.0, 1.0]]);
        assert!(
            logits[0][2] > logits[0][0],
            "hidden/vocab output.weight must be reshaped without transposing values, got {:?}",
            logits[0]
        );

        let hidden = Tensor::from_vec(
            vec![
                1.0f32, 0.0, 0.0, 0.0, //
                0.0, 0.0, 1.0, 0.0,
            ],
            (1, 2, 4),
            &device,
        )?;
        let logits = model.logits(&hidden)?.to_vec3::<f32>()?;
        assert_eq!(logits, vec![vec![vec![0.0, 0.0, 1.0], vec![0.0, 0.0, 0.0]]]);

        Ok(())
    }

    #[test]
    fn transformer_keeps_hidden_vocab_lm_head_transposed() -> Result<()> {
        let device = Device::Cpu;
        let mut config = BitNetConfig::default();
        config.model.hidden_size = 4;
        config.model.vocab_size = 3;
        config.model.num_layers = 0;
        config.model.num_heads = 1;
        config.model.num_key_value_heads = 1;
        config.model.intermediate_size = 4;
        config.model.rms_norm_eps = Some(1e-5);
        config.model.norm_type = NormType::RmsNorm;

        let mut tensors = HashMap::new();
        tensors.insert(
            "embed_tokens.weight".to_string(),
            Tensor::from_vec(
                vec![
                    10.0f32, 0.0, 0.0, 0.0, //
                    0.0, 10.0, 0.0, 0.0, //
                    0.0, 0.0, 10.0, 0.0,
                ],
                (3, 4),
                &device,
            )?,
        );
        tensors.insert("final_norm.weight".to_string(), Tensor::ones(4, DType::F32, &device)?);
        tensors.insert(
            "lm_head.weight".to_string(),
            Tensor::from_vec(
                vec![
                    0.0f32, 1.0, 0.0, //
                    0.0, 0.0, 0.0, //
                    0.0, 0.0, 0.0, //
                    0.0, 0.0, 0.0,
                ],
                (4, 3),
                &device,
            )?,
        );

        let vb = VarBuilder::from_tensors(tensors, DType::F32, &device);
        let model = TransformerModel::new_with_tensors(config, vb, HashMap::new())?;

        assert!(
            model.lm_head.is_none(),
            "hidden/vocab lm_head.weight should use direct matmul, not Linear"
        );
        assert!(
            model.lm_head_transposed,
            "hidden/vocab lm_head.weight must remain marked as a true transposed head"
        );
        assert_eq!(
            model.lm_head_weight.as_ref().map(|weight| weight.dims().to_vec()),
            Some(vec![4, 3])
        );

        let hidden = Tensor::from_vec(vec![1.0f32, 0.0, 0.0, 0.0], (1, 4), &device)?;
        let logits = model.logits(&hidden)?.to_vec2::<f32>()?;
        assert_eq!(logits, vec![vec![0.0, 1.0, 0.0]]);

        let hidden = Tensor::from_vec(
            vec![
                1.0f32, 0.0, 0.0, 0.0, //
                0.0, 1.0, 0.0, 0.0,
            ],
            (1, 2, 4),
            &device,
        )?;
        let logits = model.logits(&hidden)?.to_vec3::<f32>()?;
        assert_eq!(logits, vec![vec![vec![0.0, 1.0, 0.0], vec![0.0, 0.0, 0.0]]]);

        Ok(())
    }

    #[test]
    fn test_rmsnorm_formula_consistency() -> candle_core::Result<()> {
        // Verify RMSNorm formula: output = (x / sqrt(mean(x²) + eps)) * gamma
        let device = Device::Cpu;
        let hidden_size = 256;
        let eps = 1e-5;

        // Create input
        let input_data: Vec<f32> = (0..hidden_size).map(|i| (i as f32 / 100.0).sin()).collect();
        let input = Tensor::from_slice(&input_data, (1, hidden_size), &device)?;

        // Create gamma
        let gamma = Tensor::ones(hidden_size, DType::F32, &device)?;

        // Apply RMSNorm via Candle
        let rms_norm = RmsNorm::new(gamma.clone(), eps);
        let output_candle = rms_norm.forward(&input)?;

        // Manually compute RMSNorm
        let squared = input.sqr()?;
        let mean_squared = squared.mean_keepdim(1)?; // Mean over last dimension
        let rms_denominator = (mean_squared + eps)?.sqrt()?;
        let normalized = input.broadcast_div(&rms_denominator)?;
        let output_manual = normalized.broadcast_mul(&gamma)?;

        // Compare outputs
        let diff = (output_candle.sub(&output_manual))?.abs()?;
        let diff_vec: Vec<f32> = diff.flatten_all()?.to_vec1()?;
        let max_diff = diff_vec.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;

        assert!(
            max_diff < 1e-5,
            "Candle's RMSNorm should match manual computation: max_diff={:.6e}",
            max_diff
        );

        Ok(())
    }

    #[test]
    fn test_rmsnorm_output_scale_relationship() -> candle_core::Result<()> {
        // Test that output RMS scales proportionally with gamma RMS
        let device = Device::Cpu;
        let hidden_size = 256;
        let eps = 1e-5;

        // Create same input for both tests
        let input_data: Vec<f32> = (0..hidden_size)
            .map(|i| {
                let x = i as f32 / hidden_size as f32;
                ((x * 10.0).sin() + (x * 20.0).cos()) * 2.0
            })
            .collect();
        let input = Tensor::from_slice(&input_data, (1, hidden_size), &device)?;

        // Test 1: Standard gamma (RMS ≈ 1.0)
        let gamma_std = Tensor::ones(hidden_size, DType::F32, &device)?;
        let rms_norm_std = RmsNorm::new(gamma_std.clone(), eps);
        let output_std = rms_norm_std.forward(&input)?;
        let output_std_rms = compute_rms(&output_std)?;

        // Test 2: Small gamma (RMS ≈ 0.02)
        let target_rms = 0.02;
        let gamma_small =
            Tensor::from_slice(&vec![target_rms as f32; hidden_size], hidden_size, &device)?;
        let rms_norm_small = RmsNorm::new(gamma_small.clone(), eps);
        let output_small = rms_norm_small.forward(&input)?;
        let output_small_rms = compute_rms(&output_small)?;

        // Verify scaling relationship
        let gamma_std_rms = compute_rms(&gamma_std)?;
        let gamma_small_rms = compute_rms(&gamma_small)?;
        let expected_ratio = gamma_small_rms / gamma_std_rms;
        let actual_ratio = output_small_rms / output_std_rms;

        assert!(
            (actual_ratio - expected_ratio).abs() < 0.01,
            "Output RMS should scale with gamma RMS: expected ratio {:.6}, got {:.6}",
            expected_ratio,
            actual_ratio
        );

        Ok(())
    }
}
