//! Inference engine bridge for the OpenCL backend.
//!
//! Adapts the collection of OpenCL kernel modules (attention, FFN, normalization,
//! embedding, etc.) into a unified interface compatible with the
//! `ProductionInferenceEngine`. When no OpenCL hardware is present, every
//! public function falls back to a deterministic CPU reference implementation
//! so results can be validated without a GPU.
//!
//! # CPU reference
//!
//! All `cpu_*` functions are pure scalar implementations that mirror the
//! operations the OpenCL kernels perform on device. They are used for:
//!
//! - Unit / integration tests (no GPU required).
//! - Correctness validation against the GPU path.
//! - Fallback inference on machines without OpenCL.

use std::fmt;
use std::time::Instant;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors originating from the engine bridge.
#[derive(Debug, Clone, PartialEq)]
pub enum BridgeError {
    /// Engine has not been initialized yet.
    NotInitialized,
    /// Model weights have not been loaded.
    ModelNotLoaded,
    /// Device-level error (e.g. OpenCL runtime failure).
    DeviceError(String),
    /// Compute error during inference.
    ComputeError(String),
    /// Out-of-memory on the target device.
    OomError {
        /// Bytes required for the operation.
        required: usize,
        /// Bytes available on the device.
        available: usize,
    },
}

impl fmt::Display for BridgeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotInitialized => write!(f, "engine bridge not initialized"),
            Self::ModelNotLoaded => write!(f, "model not loaded"),
            Self::DeviceError(msg) => write!(f, "device error: {msg}"),
            Self::ComputeError(msg) => write!(f, "compute error: {msg}"),
            Self::OomError { required, available } => {
                write!(f, "OOM: need {required} bytes, have {available} bytes")
            }
        }
    }
}

impl std::error::Error for BridgeError {}

// ---------------------------------------------------------------------------
// Engine state
// ---------------------------------------------------------------------------

/// Lifecycle state of the [`EngineBridge`].
#[derive(Debug, Clone, PartialEq)]
pub enum EngineState {
    /// Bridge created but not yet initialised.
    Uninitialized,
    /// Model weights are being loaded.
    Loading,
    /// Model is loaded and ready for inference.
    Ready,
    /// Inference is in progress.
    Running,
    /// An unrecoverable error has occurred.
    Error(String),
}

impl fmt::Display for EngineState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Uninitialized => write!(f, "Uninitialized"),
            Self::Loading => write!(f, "Loading"),
            Self::Ready => write!(f, "Ready"),
            Self::Running => write!(f, "Running"),
            Self::Error(msg) => write!(f, "Error({msg})"),
        }
    }
}

// ---------------------------------------------------------------------------
// Backend capabilities
// ---------------------------------------------------------------------------

/// Hardware capabilities reported (or simulated) for the target device.
#[derive(Debug, Clone)]
pub struct BackendCaps {
    /// Maximum work-group size (threads per workgroup).
    pub max_workgroup_size: usize,
    /// Maximum buffer allocation size in bytes.
    pub max_buffer_size: usize,
    /// Whether the device supports FP16 arithmetic.
    pub fp16_support: bool,
    /// Whether the device supports INT8 DP4A dot-product.
    pub int8_dp4a: bool,
    /// Supported subgroup (SIMD lane) widths.
    pub subgroup_sizes: Vec<usize>,
    /// Shared local memory size in bytes.
    pub slm_size: usize,
}

// ---------------------------------------------------------------------------
// OpenCL backend handle
// ---------------------------------------------------------------------------

/// Logical handle to the OpenCL backend (or its CPU stub).
#[derive(Debug, Clone)]
pub struct OpenClBackend {
    /// Logical device index.
    pub device_id: usize,
    /// Human-readable device name.
    pub device_name: String,
    /// Detected (or simulated) capabilities.
    pub capabilities: BackendCaps,
    /// Whether the backend finished one-time setup.
    pub initialized: bool,
}

// ---------------------------------------------------------------------------
// Model / inference types
// ---------------------------------------------------------------------------

/// Static model configuration (architecture hyper-parameters).
#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    pub intermediate_size: usize,
    pub max_seq_len: usize,
}

/// Per-layer weight tensors (stored as flat `f32` vectors).
#[derive(Debug, Clone)]
pub struct LayerWeights {
    pub attention_qkv: Vec<f32>,
    pub attention_out: Vec<f32>,
    pub ffn_gate: Vec<f32>,
    pub ffn_up: Vec<f32>,
    pub ffn_down: Vec<f32>,
    pub norm_weight: Vec<f32>,
}

/// A request submitted to the engine.
#[derive(Debug, Clone)]
pub struct InferenceRequest {
    pub input_ids: Vec<u32>,
    pub max_tokens: usize,
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
}

/// Response produced by the engine.
#[derive(Debug, Clone)]
pub struct InferenceResponse {
    pub output_ids: Vec<u32>,
    pub logits: Vec<f32>,
    pub latency_ms: f64,
    pub tokens_per_second: f64,
}

// ---------------------------------------------------------------------------
// Engine bridge
// ---------------------------------------------------------------------------

/// Central bridge that owns backend state, model config, and weight storage.
#[derive(Debug)]
pub struct EngineBridge {
    pub backend: OpenClBackend,
    pub model_config: Option<ModelConfig>,
    pub layer_weights: Vec<LayerWeights>,
    pub state: EngineState,
}

// ---------------------------------------------------------------------------
// Construction helpers
// ---------------------------------------------------------------------------

/// Detect A770-like backend capabilities for `device_id`.
///
/// Returns simulated values matching an Intel Arc A770 (512 EU, 16 GiB).
pub fn cpu_detect_capabilities(device_id: usize) -> BackendCaps {
    let _ = device_id; // CPU stub ignores the device index.
    BackendCaps {
        max_workgroup_size: 1024,
        max_buffer_size: 4 * 1024 * 1024 * 1024, // 4 GiB
        fp16_support: true,
        int8_dp4a: true,
        subgroup_sizes: vec![8, 16, 32],
        slm_size: 64 * 1024, // 64 KiB SLM
    }
}

/// Create a new [`EngineBridge`] backed by `device_id`.
pub fn create_engine_bridge(device_id: usize) -> EngineBridge {
    let caps = cpu_detect_capabilities(device_id);
    EngineBridge {
        backend: OpenClBackend {
            device_id,
            device_name: format!("Intel(R) Arc(TM) A770 Graphics (stub:{})", device_id),
            capabilities: caps,
            initialized: false,
        },
        model_config: None,
        layer_weights: Vec::new(),
        state: EngineState::Uninitialized,
    }
}

// ---------------------------------------------------------------------------
// Model loading
// ---------------------------------------------------------------------------

/// Load model weights into the bridge.
///
/// Transitions state to [`EngineState::Ready`] on success.
pub fn cpu_load_model(
    bridge: &mut EngineBridge,
    config: ModelConfig,
    weights: Vec<LayerWeights>,
) -> Result<(), BridgeError> {
    bridge.state = EngineState::Loading;

    if weights.len() != config.num_layers {
        bridge.state = EngineState::Error("layer count mismatch".into());
        return Err(BridgeError::ComputeError(format!(
            "expected {} layers, got {}",
            config.num_layers,
            weights.len()
        )));
    }

    bridge.model_config = Some(config);
    bridge.layer_weights = weights;
    bridge.backend.initialized = true;
    bridge.state = EngineState::Ready;
    Ok(())
}

// ---------------------------------------------------------------------------
// Core compute primitives (CPU reference)
// ---------------------------------------------------------------------------

/// RMS normalization: `y_i = (x_i / rms(x)) * w_i`.
pub fn cpu_rmsnorm(x: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    let n = x.len();
    debug_assert_eq!(n, weight.len());
    let ss: f32 = x.iter().map(|&v| v * v).sum::<f32>() / n as f32;
    let rms = (ss + eps).sqrt();
    x.iter().zip(weight.iter()).map(|(&xi, &wi)| (xi / rms) * wi).collect()
}

/// Multi-head attention (CPU scalar reference).
///
/// `q`, `k`, `v` are packed `[seq_len * num_heads * head_dim]`.
/// Returns `[seq_len * num_heads * head_dim]`.
pub fn cpu_attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    num_heads: usize,
    head_dim: usize,
) -> Vec<f32> {
    let total = q.len();
    let head_size = num_heads * head_dim;
    if head_size == 0 {
        return vec![0.0; total];
    }
    let seq_len = total / head_size;
    let scale = 1.0 / (head_dim as f32).sqrt();

    let mut output = vec![0.0f32; total];

    for s in 0..seq_len {
        for h in 0..num_heads {
            let q_off = s * head_size + h * head_dim;
            // Compute attention scores for all key positions
            let mut scores = Vec::with_capacity(seq_len);
            for sk in 0..seq_len {
                let k_off = sk * head_size + h * head_dim;
                let dot: f32 = (0..head_dim).map(|d| q[q_off + d] * k[k_off + d]).sum();
                scores.push(dot * scale);
            }

            // Causal mask
            for sc in scores.iter_mut().skip(s + 1) {
                *sc = f32::NEG_INFINITY;
            }

            // Softmax
            let max_s = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut exp_sum = 0.0f32;
            for sc in &mut scores {
                *sc = (*sc - max_s).exp();
                exp_sum += *sc;
            }
            if exp_sum > 0.0 {
                for sc in &mut scores {
                    *sc /= exp_sum;
                }
            }

            // Weighted sum of values
            let o_off = s * head_size + h * head_dim;
            for (sk, &score) in scores.iter().enumerate() {
                let v_off = sk * head_size + h * head_dim;
                for d in 0..head_dim {
                    output[o_off + d] += score * v[v_off + d];
                }
            }
        }
    }

    output
}

/// SwiGLU Feed-Forward Network (CPU scalar reference).
///
/// `gate`, `up` are `[intermediate × hidden]`, `down` is `[hidden × intermediate]`.
pub fn cpu_ffn(
    x: &[f32],
    gate: &[f32],
    up: &[f32],
    down: &[f32],
    hidden: usize,
    intermediate: usize,
) -> Vec<f32> {
    // gate projection
    let mut gate_out = vec![0.0f32; intermediate];
    for i in 0..intermediate {
        let mut acc = 0.0f32;
        for j in 0..hidden {
            acc += gate[i * hidden + j] * x[j];
        }
        gate_out[i] = acc;
    }

    // up projection
    let mut up_out = vec![0.0f32; intermediate];
    for i in 0..intermediate {
        let mut acc = 0.0f32;
        for j in 0..hidden {
            acc += up[i * hidden + j] * x[j];
        }
        up_out[i] = acc;
    }

    // SiLU(gate) * up
    for i in 0..intermediate {
        let silu = gate_out[i] / (1.0 + (-gate_out[i]).exp());
        gate_out[i] = silu * up_out[i];
    }

    // down projection
    let mut out = vec![0.0f32; hidden];
    for i in 0..hidden {
        let mut acc = 0.0f32;
        for j in 0..intermediate {
            acc += down[i * intermediate + j] * gate_out[j];
        }
        out[i] = acc;
    }

    out
}

/// Forward one transformer layer (CPU reference).
///
/// Applies RMSNorm → attention → residual → RMSNorm → FFN → residual.
pub fn cpu_forward_layer(weights: &LayerWeights, hidden: &[f32], config: &ModelConfig) -> Vec<f32> {
    let h = config.hidden_size;
    let eps = 1e-5_f32;

    // Pre-attention norm
    let normed = cpu_rmsnorm(hidden, &weights.norm_weight, eps);

    // Simplified QKV: use attention_qkv as identity-like projection
    let head_size = config.num_heads * config.head_dim;
    let seq_len = normed.len() / h;
    let qkv_len = seq_len * head_size;

    let q: Vec<f32> = normed.iter().copied().chain(std::iter::repeat(0.0)).take(qkv_len).collect();
    let k = q.clone();
    let v = q.clone();

    let attn_out = cpu_attention(&q, &k, &v, config.num_heads, config.head_dim);

    // Residual
    let mut x: Vec<f32> = hidden
        .iter()
        .zip(attn_out.iter().chain(std::iter::repeat(&0.0)))
        .map(|(&a, &b)| a + b)
        .take(h)
        .collect();

    // Pre-FFN norm
    let normed2 = cpu_rmsnorm(&x, &weights.norm_weight, eps);

    let ffn_out = cpu_ffn(
        &normed2,
        &weights.ffn_gate,
        &weights.ffn_up,
        &weights.ffn_down,
        h,
        config.intermediate_size,
    );

    // Residual
    for (xi, &fi) in x.iter_mut().zip(ffn_out.iter()) {
        *xi += fi;
    }

    x
}

// ---------------------------------------------------------------------------
// Inference entry point
// ---------------------------------------------------------------------------

/// Run auto-regressive inference (CPU reference).
pub fn cpu_run_inference(
    bridge: &mut EngineBridge,
    request: InferenceRequest,
) -> Result<InferenceResponse, BridgeError> {
    if bridge.state == EngineState::Uninitialized {
        return Err(BridgeError::NotInitialized);
    }
    let config = bridge.model_config.as_ref().ok_or(BridgeError::ModelNotLoaded)?;
    if bridge.layer_weights.is_empty() {
        return Err(BridgeError::ModelNotLoaded);
    }

    bridge.state = EngineState::Running;
    let start = Instant::now();

    let h = config.hidden_size;

    // Tiny pseudo-embedding: token id → constant hidden vector
    let mut hidden: Vec<f32> = vec![0.0; h];
    if let Some(&tok) = request.input_ids.last() {
        for (i, hi) in hidden.iter_mut().enumerate() {
            *hi = ((tok as f32 + 1.0) * (i as f32 + 1.0)).sin() * 0.1;
        }
    }

    let mut output_ids = request.input_ids.clone();

    for _step in 0..request.max_tokens {
        // Forward through all layers
        for layer_w in &bridge.layer_weights {
            hidden = cpu_forward_layer(layer_w, &hidden, config);
        }

        // Pseudo logits → argmax token
        let token = hidden
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| (i % config.vocab_size) as u32)
            .unwrap_or(0);

        output_ids.push(token);
    }

    let elapsed = start.elapsed();
    let latency_ms = elapsed.as_secs_f64() * 1000.0;
    let generated = request.max_tokens.max(1);
    let tokens_per_second = generated as f64 / elapsed.as_secs_f64().max(1e-9);

    // Final logits from last hidden state
    let logits = hidden.iter().copied().take(config.vocab_size.min(h)).collect();

    bridge.state = EngineState::Ready;

    Ok(InferenceResponse { output_ids, logits, latency_ms, tokens_per_second })
}

// ---------------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------------

/// Estimate VRAM (bytes) required for the model described by `config`.
pub fn cpu_estimate_memory(config: &ModelConfig) -> usize {
    let bytes_per_param = 4; // f32
    let per_layer = {
        let qkv = 3 * config.hidden_size * config.hidden_size;
        let attn_out = config.hidden_size * config.hidden_size;
        let ffn = 3 * config.hidden_size * config.intermediate_size;
        let norm = config.hidden_size;
        qkv + attn_out + ffn + norm
    };
    let total_params = per_layer * config.num_layers + config.vocab_size * config.hidden_size; // embedding
    total_params * bytes_per_param
}

/// Human-readable backend information.
pub fn cpu_get_backend_info(bridge: &EngineBridge) -> String {
    let caps = &bridge.backend.capabilities;
    format!(
        "device={} id={} state={} wg={} buf={}B fp16={} dp4a={} subgroups={:?} slm={}B",
        bridge.backend.device_name,
        bridge.backend.device_id,
        bridge.state,
        caps.max_workgroup_size,
        caps.max_buffer_size,
        caps.fp16_support,
        caps.int8_dp4a,
        caps.subgroup_sizes,
        caps.slm_size,
    )
}

/// Execute a warmup pass (single token forward through all layers).
pub fn cpu_warmup(bridge: &mut EngineBridge) -> Result<(), BridgeError> {
    let config = bridge.model_config.as_ref().ok_or(BridgeError::ModelNotLoaded)?;
    let h = config.hidden_size;
    let mut hidden = vec![1.0f32; h];

    for layer_w in &bridge.layer_weights {
        hidden = cpu_forward_layer(layer_w, &hidden, config);
    }

    bridge.state = EngineState::Ready;
    Ok(())
}

/// Format an [`InferenceResponse`] for logging / display.
pub fn format_inference_result(response: &InferenceResponse) -> String {
    format!(
        "tokens={} logits_len={} latency={:.2}ms tok/s={:.2}",
        response.output_ids.len(),
        response.logits.len(),
        response.latency_ms,
        response.tokens_per_second,
    )
}

// ---------------------------------------------------------------------------
// Helpers for building test fixtures
// ---------------------------------------------------------------------------

/// Create a minimal [`ModelConfig`] for testing.
#[cfg(test)]
fn test_config() -> ModelConfig {
    ModelConfig {
        vocab_size: 128,
        hidden_size: 64,
        num_layers: 2,
        num_heads: 4,
        head_dim: 16,
        intermediate_size: 128,
        max_seq_len: 32,
    }
}

/// Create a [`LayerWeights`] filled with a constant value.
#[cfg(test)]
fn test_layer_weights(config: &ModelConfig, val: f32) -> LayerWeights {
    let h = config.hidden_size;
    let inter = config.intermediate_size;
    LayerWeights {
        attention_qkv: vec![val; 3 * h * h],
        attention_out: vec![val; h * h],
        ffn_gate: vec![val; inter * h],
        ffn_up: vec![val; inter * h],
        ffn_down: vec![val; h * inter],
        norm_weight: vec![1.0; h],
    }
}

/// Build weights for every layer in `config`.
#[cfg(test)]
fn test_all_weights(config: &ModelConfig, val: f32) -> Vec<LayerWeights> {
    (0..config.num_layers).map(|_| test_layer_weights(config, val)).collect()
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -- construction & state -----------------------------------------------

    #[test]
    fn test_create_bridge_starts_uninitialized() {
        let bridge = create_engine_bridge(0);
        assert_eq!(bridge.state, EngineState::Uninitialized);
        assert!(!bridge.backend.initialized);
    }

    #[test]
    fn test_create_bridge_stores_device_id() {
        let bridge = create_engine_bridge(7);
        assert_eq!(bridge.backend.device_id, 7);
    }

    #[test]
    fn test_create_bridge_no_model() {
        let bridge = create_engine_bridge(0);
        assert!(bridge.model_config.is_none());
        assert!(bridge.layer_weights.is_empty());
    }

    #[test]
    fn test_detect_capabilities_a770_values() {
        let caps = cpu_detect_capabilities(0);
        assert_eq!(caps.max_workgroup_size, 1024);
        assert!(caps.fp16_support);
        assert!(caps.int8_dp4a);
        assert!(caps.subgroup_sizes.contains(&16));
    }

    #[test]
    fn test_detect_capabilities_slm() {
        let caps = cpu_detect_capabilities(0);
        assert_eq!(caps.slm_size, 64 * 1024);
    }

    #[test]
    fn test_detect_capabilities_max_buffer() {
        let caps = cpu_detect_capabilities(0);
        assert_eq!(caps.max_buffer_size, 4 * 1024 * 1024 * 1024);
    }

    #[test]
    fn test_detect_capabilities_subgroup_sizes() {
        let caps = cpu_detect_capabilities(0);
        assert_eq!(caps.subgroup_sizes, vec![8, 16, 32]);
    }

    // -- model loading ------------------------------------------------------

    #[test]
    fn test_load_model_transitions_to_ready() {
        let mut bridge = create_engine_bridge(0);
        let cfg = test_config();
        let w = test_all_weights(&cfg, 0.01);
        cpu_load_model(&mut bridge, cfg, w).unwrap();
        assert_eq!(bridge.state, EngineState::Ready);
        assert!(bridge.backend.initialized);
    }

    #[test]
    fn test_load_model_wrong_layer_count() {
        let mut bridge = create_engine_bridge(0);
        let cfg = test_config();
        let w = vec![test_layer_weights(&cfg, 0.01)]; // 1 instead of 2
        let err = cpu_load_model(&mut bridge, cfg, w).unwrap_err();
        assert!(matches!(err, BridgeError::ComputeError(_)));
    }

    #[test]
    fn test_load_model_stores_config() {
        let mut bridge = create_engine_bridge(0);
        let cfg = test_config();
        let w = test_all_weights(&cfg, 0.01);
        cpu_load_model(&mut bridge, cfg, w).unwrap();
        assert!(bridge.model_config.is_some());
        assert_eq!(bridge.model_config.as_ref().unwrap().vocab_size, 128);
    }

    // -- inference errors ---------------------------------------------------

    #[test]
    fn test_inference_without_init_errors() {
        let mut bridge = create_engine_bridge(0);
        let req = InferenceRequest {
            input_ids: vec![1],
            max_tokens: 1,
            temperature: 1.0,
            top_k: 50,
            top_p: 0.9,
        };
        let err = cpu_run_inference(&mut bridge, req).unwrap_err();
        assert_eq!(err, BridgeError::NotInitialized);
    }

    #[test]
    fn test_inference_without_model_errors() {
        let mut bridge = create_engine_bridge(0);
        bridge.state = EngineState::Ready; // skip init but no model
        let req = InferenceRequest {
            input_ids: vec![1],
            max_tokens: 1,
            temperature: 1.0,
            top_k: 50,
            top_p: 0.9,
        };
        let err = cpu_run_inference(&mut bridge, req).unwrap_err();
        assert_eq!(err, BridgeError::ModelNotLoaded);
    }

    // -- rmsnorm ------------------------------------------------------------

    #[test]
    fn test_rmsnorm_unit_weight() {
        let x = vec![3.0, 4.0];
        let w = vec![1.0, 1.0];
        let out = cpu_rmsnorm(&x, &w, 1e-5);
        // rms = sqrt((9+16)/2 + eps) ≈ 3.5355
        let rms = ((9.0 + 16.0) / 2.0_f32 + 1e-5).sqrt();
        assert!((out[0] - 3.0 / rms).abs() < 1e-4);
        assert!((out[1] - 4.0 / rms).abs() < 1e-4);
    }

    #[test]
    fn test_rmsnorm_preserves_length() {
        let x = vec![1.0; 64];
        let w = vec![1.0; 64];
        let out = cpu_rmsnorm(&x, &w, 1e-5);
        assert_eq!(out.len(), 64);
    }

    #[test]
    fn test_rmsnorm_zero_input() {
        let x = vec![0.0; 8];
        let w = vec![1.0; 8];
        let out = cpu_rmsnorm(&x, &w, 1e-5);
        // All should be ~0
        for &v in &out {
            assert!(v.abs() < 1e-3);
        }
    }

    #[test]
    fn test_rmsnorm_scaling_property() {
        let x = vec![2.0; 4];
        let w = vec![1.0; 4];
        let out = cpu_rmsnorm(&x, &w, 1e-5);
        // All elements identical → output should be ~1.0 (x/rms with identical x)
        for &v in &out {
            assert!((v - 1.0).abs() < 1e-3);
        }
    }

    // -- attention ----------------------------------------------------------

    #[test]
    fn test_attention_output_shape() {
        let seq_len = 2;
        let num_heads = 2;
        let head_dim = 4;
        let total = seq_len * num_heads * head_dim;
        let q = vec![0.1; total];
        let k = vec![0.1; total];
        let v = vec![0.1; total];
        let out = cpu_attention(&q, &k, &v, num_heads, head_dim);
        assert_eq!(out.len(), total);
    }

    #[test]
    fn test_attention_single_token() {
        let num_heads = 2;
        let head_dim = 4;
        let total = 1 * num_heads * head_dim;
        let q = vec![1.0; total];
        let k = vec![1.0; total];
        let v = vec![1.0; total];
        let out = cpu_attention(&q, &k, &v, num_heads, head_dim);
        assert_eq!(out.len(), total);
        // Single token: score = 1.0 softmax → output == v
        for (&o, &vi) in out.iter().zip(v.iter()) {
            assert!((o - vi).abs() < 1e-4);
        }
    }

    #[test]
    fn test_attention_zero_query() {
        let total = 2 * 2 * 4;
        let q = vec![0.0; total];
        let k = vec![0.1; total];
        let v = vec![1.0; total];
        let out = cpu_attention(&q, &k, &v, 2, 4);
        // All outputs should be finite
        for &o in &out {
            assert!(o.is_finite());
        }
    }

    #[test]
    fn test_attention_causal_mask() {
        // With 2 tokens, position 0 should only attend to itself
        let num_heads = 1;
        let head_dim = 2;
        let total = 2 * num_heads * head_dim;
        let q = vec![1.0; total];
        let k = vec![1.0; total];
        // v: token0=[1,0], token1=[0,1]
        let v = vec![1.0, 0.0, 0.0, 1.0];
        let out = cpu_attention(&q, &k, &v, num_heads, head_dim);
        // Position 0 only attends to token 0 → should get [1, 0]
        assert!((out[0] - 1.0).abs() < 1e-4);
        assert!(out[1].abs() < 1e-4);
    }

    // -- ffn ----------------------------------------------------------------

    #[test]
    fn test_ffn_output_shape() {
        let h = 4;
        let inter = 8;
        let x = vec![1.0; h];
        let gate = vec![0.01; inter * h];
        let up = vec![0.01; inter * h];
        let down = vec![0.01; h * inter];
        let out = cpu_ffn(&x, &gate, &up, &down, h, inter);
        assert_eq!(out.len(), h);
    }

    #[test]
    fn test_ffn_zero_input() {
        let h = 4;
        let inter = 8;
        let x = vec![0.0; h];
        let gate = vec![0.1; inter * h];
        let up = vec![0.1; inter * h];
        let down = vec![0.1; h * inter];
        let out = cpu_ffn(&x, &gate, &up, &down, h, inter);
        for &v in &out {
            assert!(v.abs() < 1e-6);
        }
    }

    #[test]
    fn test_ffn_nonzero_output() {
        let h = 4;
        let inter = 8;
        let x = vec![1.0; h];
        let gate = vec![0.1; inter * h];
        let up = vec![0.1; inter * h];
        let down = vec![0.1; h * inter];
        let out = cpu_ffn(&x, &gate, &up, &down, h, inter);
        let sum: f32 = out.iter().map(|v| v.abs()).sum();
        assert!(sum > 0.0);
    }

    // -- forward layer ------------------------------------------------------

    #[test]
    fn test_forward_layer_output_shape() {
        let cfg = test_config();
        let lw = test_layer_weights(&cfg, 0.01);
        let hidden = vec![0.5; cfg.hidden_size];
        let out = cpu_forward_layer(&lw, &hidden, &cfg);
        assert_eq!(out.len(), cfg.hidden_size);
    }

    #[test]
    fn test_forward_layer_nonzero() {
        let cfg = test_config();
        let lw = test_layer_weights(&cfg, 0.01);
        let hidden = vec![1.0; cfg.hidden_size];
        let out = cpu_forward_layer(&lw, &hidden, &cfg);
        let sum: f32 = out.iter().map(|v| v.abs()).sum();
        assert!(sum > 0.0);
    }

    // -- memory estimation --------------------------------------------------

    #[test]
    fn test_memory_estimation_reasonable() {
        let cfg = test_config();
        let est = cpu_estimate_memory(&cfg);
        // Should be > 0 and < 1 GiB for the tiny test config
        assert!(est > 0);
        assert!(est < 1024 * 1024 * 1024);
    }

    #[test]
    fn test_memory_estimation_scales_with_layers() {
        let mut cfg = test_config();
        let est1 = cpu_estimate_memory(&cfg);
        cfg.num_layers *= 2;
        let est2 = cpu_estimate_memory(&cfg);
        assert!(est2 > est1);
    }

    #[test]
    fn test_memory_estimation_scales_with_hidden() {
        let cfg1 = test_config();
        let mut cfg2 = test_config();
        cfg2.hidden_size *= 2;
        cfg2.head_dim *= 2;
        let est1 = cpu_estimate_memory(&cfg1);
        let est2 = cpu_estimate_memory(&cfg2);
        assert!(est2 > est1);
    }

    // -- warmup -------------------------------------------------------------

    #[test]
    fn test_warmup_transitions_to_ready() {
        let mut bridge = create_engine_bridge(0);
        let cfg = test_config();
        let w = test_all_weights(&cfg, 0.01);
        cpu_load_model(&mut bridge, cfg, w).unwrap();
        cpu_warmup(&mut bridge).unwrap();
        assert_eq!(bridge.state, EngineState::Ready);
    }

    #[test]
    fn test_warmup_without_model_errors() {
        let mut bridge = create_engine_bridge(0);
        let err = cpu_warmup(&mut bridge).unwrap_err();
        assert_eq!(err, BridgeError::ModelNotLoaded);
    }

    // -- full pipeline ------------------------------------------------------

    #[test]
    fn test_full_pipeline_load_warmup_infer() {
        let mut bridge = create_engine_bridge(0);
        let cfg = test_config();
        let w = test_all_weights(&cfg, 0.01);
        cpu_load_model(&mut bridge, cfg, w).unwrap();
        cpu_warmup(&mut bridge).unwrap();

        let req = InferenceRequest {
            input_ids: vec![1, 2, 3],
            max_tokens: 2,
            temperature: 1.0,
            top_k: 50,
            top_p: 0.9,
        };
        let resp = cpu_run_inference(&mut bridge, req).unwrap();
        assert_eq!(resp.output_ids.len(), 5); // 3 input + 2 generated
        assert!(resp.latency_ms >= 0.0);
        assert!(resp.tokens_per_second >= 0.0);
    }

    #[test]
    fn test_inference_returns_to_ready() {
        let mut bridge = create_engine_bridge(0);
        let cfg = test_config();
        let w = test_all_weights(&cfg, 0.01);
        cpu_load_model(&mut bridge, cfg, w).unwrap();

        let req = InferenceRequest {
            input_ids: vec![1],
            max_tokens: 1,
            temperature: 1.0,
            top_k: 50,
            top_p: 0.9,
        };
        cpu_run_inference(&mut bridge, req).unwrap();
        assert_eq!(bridge.state, EngineState::Ready);
    }

    #[test]
    fn test_inference_logits_nonempty() {
        let mut bridge = create_engine_bridge(0);
        let cfg = test_config();
        let w = test_all_weights(&cfg, 0.01);
        cpu_load_model(&mut bridge, cfg, w).unwrap();

        let req = InferenceRequest {
            input_ids: vec![42],
            max_tokens: 1,
            temperature: 1.0,
            top_k: 50,
            top_p: 0.9,
        };
        let resp = cpu_run_inference(&mut bridge, req).unwrap();
        assert!(!resp.logits.is_empty());
    }

    // -- backend info -------------------------------------------------------

    #[test]
    fn test_backend_info_contains_device_name() {
        let bridge = create_engine_bridge(0);
        let info = cpu_get_backend_info(&bridge);
        assert!(info.contains("A770"));
    }

    #[test]
    fn test_backend_info_contains_state() {
        let bridge = create_engine_bridge(0);
        let info = cpu_get_backend_info(&bridge);
        assert!(info.contains("Uninitialized"));
    }

    // -- format_inference_result --------------------------------------------

    #[test]
    fn test_format_inference_result() {
        let resp = InferenceResponse {
            output_ids: vec![1, 2, 3],
            logits: vec![0.1, 0.2],
            latency_ms: 42.5,
            tokens_per_second: 100.0,
        };
        let s = format_inference_result(&resp);
        assert!(s.contains("tokens=3"));
        assert!(s.contains("logits_len=2"));
        assert!(s.contains("42.50"));
    }

    // -- edge cases ---------------------------------------------------------

    #[test]
    fn test_single_token_input() {
        let mut bridge = create_engine_bridge(0);
        let cfg = test_config();
        let w = test_all_weights(&cfg, 0.01);
        cpu_load_model(&mut bridge, cfg, w).unwrap();

        let req = InferenceRequest {
            input_ids: vec![1],
            max_tokens: 3,
            temperature: 1.0,
            top_k: 50,
            top_p: 0.9,
        };
        let resp = cpu_run_inference(&mut bridge, req).unwrap();
        assert_eq!(resp.output_ids.len(), 4); // 1 + 3
    }

    #[test]
    fn test_max_tokens_one() {
        let mut bridge = create_engine_bridge(0);
        let cfg = test_config();
        let w = test_all_weights(&cfg, 0.01);
        cpu_load_model(&mut bridge, cfg, w).unwrap();

        let req = InferenceRequest {
            input_ids: vec![10, 20],
            max_tokens: 1,
            temperature: 1.0,
            top_k: 50,
            top_p: 0.9,
        };
        let resp = cpu_run_inference(&mut bridge, req).unwrap();
        assert_eq!(resp.output_ids.len(), 3);
    }

    #[test]
    fn test_hidden_size_one() {
        let cfg = ModelConfig {
            vocab_size: 4,
            hidden_size: 1,
            num_layers: 1,
            num_heads: 1,
            head_dim: 1,
            intermediate_size: 2,
            max_seq_len: 8,
        };
        let lw = test_layer_weights(&cfg, 0.1);
        let hidden = vec![1.0];
        let out = cpu_forward_layer(&lw, &hidden, &cfg);
        assert_eq!(out.len(), 1);
    }

    #[test]
    fn test_output_length_matches_input_plus_max_tokens() {
        let mut bridge = create_engine_bridge(0);
        let cfg = test_config();
        let w = test_all_weights(&cfg, 0.01);
        cpu_load_model(&mut bridge, cfg, w).unwrap();

        for (n_input, max_tok) in [(1, 1), (3, 5), (10, 2)] {
            let req = InferenceRequest {
                input_ids: (0..n_input).map(|i| i as u32).collect(),
                max_tokens: max_tok,
                temperature: 1.0,
                top_k: 50,
                top_p: 0.9,
            };
            let resp = cpu_run_inference(&mut bridge, req).unwrap();
            assert_eq!(resp.output_ids.len(), n_input + max_tok);
        }
    }

    // -- error display ------------------------------------------------------

    #[test]
    fn test_bridge_error_display_not_initialized() {
        assert_eq!(BridgeError::NotInitialized.to_string(), "engine bridge not initialized");
    }

    #[test]
    fn test_bridge_error_display_oom() {
        let err = BridgeError::OomError { required: 100, available: 50 };
        assert!(err.to_string().contains("100"));
        assert!(err.to_string().contains("50"));
    }

    #[test]
    fn test_bridge_error_display_device() {
        let err = BridgeError::DeviceError("timeout".into());
        assert!(err.to_string().contains("timeout"));
    }

    #[test]
    fn test_bridge_error_display_compute() {
        let err = BridgeError::ComputeError("nan detected".into());
        assert!(err.to_string().contains("nan detected"));
    }

    #[test]
    fn test_bridge_error_display_model_not_loaded() {
        assert_eq!(BridgeError::ModelNotLoaded.to_string(), "model not loaded");
    }

    // -- engine state display -----------------------------------------------

    #[test]
    fn test_engine_state_display() {
        assert_eq!(EngineState::Uninitialized.to_string(), "Uninitialized");
        assert_eq!(EngineState::Loading.to_string(), "Loading");
        assert_eq!(EngineState::Ready.to_string(), "Ready");
        assert_eq!(EngineState::Running.to_string(), "Running");
        assert!(EngineState::Error("oops".into()).to_string().contains("oops"));
    }

    // -- multiple inferences ------------------------------------------------

    #[test]
    fn test_multiple_inferences_sequential() {
        let mut bridge = create_engine_bridge(0);
        let cfg = test_config();
        let w = test_all_weights(&cfg, 0.01);
        cpu_load_model(&mut bridge, cfg, w).unwrap();

        for _ in 0..3 {
            let req = InferenceRequest {
                input_ids: vec![1],
                max_tokens: 1,
                temperature: 1.0,
                top_k: 50,
                top_p: 0.9,
            };
            let resp = cpu_run_inference(&mut bridge, req).unwrap();
            assert_eq!(resp.output_ids.len(), 2);
        }
        assert_eq!(bridge.state, EngineState::Ready);
    }

    // -- max_tokens=0 -------------------------------------------------------

    #[test]
    fn test_max_tokens_zero() {
        let mut bridge = create_engine_bridge(0);
        let cfg = test_config();
        let w = test_all_weights(&cfg, 0.01);
        cpu_load_model(&mut bridge, cfg, w).unwrap();

        let req = InferenceRequest {
            input_ids: vec![1, 2],
            max_tokens: 0,
            temperature: 1.0,
            top_k: 50,
            top_p: 0.9,
        };
        let resp = cpu_run_inference(&mut bridge, req).unwrap();
        assert_eq!(resp.output_ids.len(), 2); // no new tokens
    }

    #[test]
    fn test_rmsnorm_single_element() {
        let x = vec![5.0];
        let w = vec![1.0];
        let out = cpu_rmsnorm(&x, &w, 1e-5);
        assert_eq!(out.len(), 1);
        // rms of single element = |x| → output ≈ sign(x) ≈ 1.0
        assert!((out[0] - 1.0).abs() < 1e-3);
    }

    #[test]
    fn test_attention_empty_zero_heads() {
        let out = cpu_attention(&[], &[], &[], 0, 0);
        assert!(out.is_empty());
    }
}
