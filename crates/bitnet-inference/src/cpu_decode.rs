//! CPU transformer decode-step helpers.
//!
//! This module keeps the first strict CPU decode boundary narrow: one token is
//! embedded, run through the transformer with the real transformer KV cache,
//! projected to logits, and returned with the selected QK256 kernel identity.

use anyhow::{Result, bail};
use bitnet_common::{ConcreteTensor, Tensor as _};
use bitnet_models::{Model, transformer::KVCache as TransformerKVCache};
use bitnet_quantization::i2s_qk256::{
    QK256_SCALAR_GEMV_KERNEL_ID, Qk256KernelSelection, select_qk256_gemv_kernel,
};

/// Authoritative CPU decode operations covered by one decode step.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CpuDecodeOps {
    pub embedding_gather: bool,
    pub transformer_layers: bool,
    pub kv_cache_append_read: bool,
    pub logits_output_head: bool,
    pub sampling_handoff: bool,
}

impl CpuDecodeOps {
    fn one_step_complete() -> Self {
        Self {
            embedding_gather: true,
            transformer_layers: true,
            kv_cache_append_read: true,
            logits_output_head: true,
            sampling_handoff: true,
        }
    }
}

/// Metadata describing the CPU decode step.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CpuDecodeStepReport {
    pub requested_kernel: Option<&'static str>,
    pub selected_kernel: &'static str,
    pub fallback_used: bool,
    pub fallback_reason: Option<String>,
    pub cpu_features: Vec<&'static str>,
    pub selected_kernel_family: &'static str,
    pub logits_shape: Vec<usize>,
    pub kv_cache_seq_lens: Vec<usize>,
    pub ops: CpuDecodeOps,
}

impl CpuDecodeStepReport {
    fn from_selection(
        selection: Qk256KernelSelection,
        logits_shape: Vec<usize>,
        kv_cache: &TransformerKVCache,
    ) -> Self {
        Self {
            requested_kernel: selection.requested_kernel,
            selected_kernel: selection.selected_kernel,
            fallback_used: selection.fallback_used,
            fallback_reason: selection.fallback_reason,
            cpu_features: selection.cpu_features,
            selected_kernel_family: selection.selected_kernel,
            logits_shape,
            kv_cache_seq_lens: kv_cache.layers.iter().map(|layer| layer.seq_len).collect(),
            ops: CpuDecodeOps::one_step_complete(),
        }
    }
}

/// Output from one CPU transformer decode step.
#[derive(Clone, Debug)]
pub struct CpuDecodeStepOutput {
    pub logits: ConcreteTensor,
    pub report: CpuDecodeStepReport,
}

/// Execute one CPU decode step with real model tensors and a transformer KV cache.
///
/// `requested_kernel = None` means auto-select the QK256 decode GEMV kernel.
/// Strict mode rejects an explicit AVX2 request when AVX2/FMA is unavailable;
/// it does not invent a fallback decode path.
pub fn decode_one_cpu_token(
    model: &dyn Model,
    kv_cache: &mut TransformerKVCache,
    token_id: u32,
    requested_kernel: Option<&'static str>,
    strict: bool,
) -> Result<CpuDecodeStepOutput> {
    let selection = select_qk256_gemv_kernel(requested_kernel, strict)?;

    let embedding = model.embed(&[token_id])?;
    ensure_rank(&embedding, 3, "CPU decode embedding")?;

    let hidden = model.forward(&embedding, kv_cache as &mut dyn std::any::Any)?;
    ensure_rank(&hidden, 3, "CPU decode hidden state")?;

    let logits = model.logits(&hidden)?;
    ensure_logits_shape(&logits)?;

    let report = CpuDecodeStepReport::from_selection(selection, logits.shape().to_vec(), kv_cache);
    Ok(CpuDecodeStepOutput { logits, report })
}

fn ensure_rank(tensor: &ConcreteTensor, expected_rank: usize, label: &str) -> Result<()> {
    let shape = tensor.shape();
    if shape.len() != expected_rank {
        bail!("{label}: expected rank {expected_rank}, got shape {shape:?}");
    }
    Ok(())
}

fn ensure_logits_shape(logits: &ConcreteTensor) -> Result<()> {
    let shape = logits.shape();
    match shape {
        [batch, vocab] if *batch > 0 && *vocab > 0 => Ok(()),
        [batch, seq, vocab] if *batch > 0 && *seq == 1 && *vocab > 0 => Ok(()),
        _ => bail!("CPU decode logits: expected [B,V] or [B,1,V], got shape {shape:?}"),
    }
}

/// Stable baseline kernel family used when CPU decode is forced to scalar.
pub const CPU_DECODE_SCALAR_KERNEL_FAMILY: &str = QK256_SCALAR_GEMV_KERNEL_ID;
