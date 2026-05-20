//! Allocation-audit support for CLI receipt generation.
//!
//! This module owns the global allocator shim, counter snapshots, and JSON
//! summarizers used by warm-session receipts. Keeping this concern isolated
//! prevents the CLI entrypoint from mixing command dispatch with allocator
//! telemetry internals.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

static ALLOCATION_AUDIT_ENABLED: AtomicBool = AtomicBool::new(false);
static ALLOCATION_AUDIT_ALLOC_COUNT: AtomicU64 = AtomicU64::new(0);
static ALLOCATION_AUDIT_ALLOC_BYTES: AtomicU64 = AtomicU64::new(0);
static ALLOCATION_AUDIT_DEALLOC_COUNT: AtomicU64 = AtomicU64::new(0);
static ALLOCATION_AUDIT_DEALLOC_BYTES: AtomicU64 = AtomicU64::new(0);

pub(crate) struct AllocationAuditAllocator;

unsafe impl GlobalAlloc for AllocationAuditAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc(layout) };
        if !ptr.is_null() && ALLOCATION_AUDIT_ENABLED.load(Ordering::Relaxed) {
            record_allocation_audit_alloc(layout.size());
        }
        ptr
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc_zeroed(layout) };
        if !ptr.is_null() && ALLOCATION_AUDIT_ENABLED.load(Ordering::Relaxed) {
            record_allocation_audit_alloc(layout.size());
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        if ALLOCATION_AUDIT_ENABLED.load(Ordering::Relaxed) {
            record_allocation_audit_dealloc(layout.size());
        }
        unsafe { System.dealloc(ptr, layout) };
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let new_ptr = unsafe { System.realloc(ptr, layout, new_size) };
        if !new_ptr.is_null() && ALLOCATION_AUDIT_ENABLED.load(Ordering::Relaxed) {
            record_allocation_audit_dealloc(layout.size());
            record_allocation_audit_alloc(new_size);
        }
        new_ptr
    }
}

fn record_allocation_audit_alloc(size: usize) {
    ALLOCATION_AUDIT_ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
    ALLOCATION_AUDIT_ALLOC_BYTES.fetch_add(size as u64, Ordering::Relaxed);
}

fn record_allocation_audit_dealloc(size: usize) {
    ALLOCATION_AUDIT_DEALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
    ALLOCATION_AUDIT_DEALLOC_BYTES.fetch_add(size as u64, Ordering::Relaxed);
}

fn rounded_ms(ms: f64) -> f64 {
    (ms * 1000.0).round() / 1000.0
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct AllocationAuditSnapshot {
    pub(crate) alloc_count: u64,
    pub(crate) alloc_bytes: u64,
    pub(crate) dealloc_count: u64,
    pub(crate) dealloc_bytes: u64,
}

impl AllocationAuditSnapshot {
    pub(crate) fn current() -> Self {
        Self {
            alloc_count: ALLOCATION_AUDIT_ALLOC_COUNT.load(Ordering::Relaxed),
            alloc_bytes: ALLOCATION_AUDIT_ALLOC_BYTES.load(Ordering::Relaxed),
            dealloc_count: ALLOCATION_AUDIT_DEALLOC_COUNT.load(Ordering::Relaxed),
            dealloc_bytes: ALLOCATION_AUDIT_DEALLOC_BYTES.load(Ordering::Relaxed),
        }
    }

    pub(crate) fn delta_since(start: Self) -> Self {
        let current = Self::current();
        Self {
            alloc_count: current.alloc_count.saturating_sub(start.alloc_count),
            alloc_bytes: current.alloc_bytes.saturating_sub(start.alloc_bytes),
            dealloc_count: current.dealloc_count.saturating_sub(start.dealloc_count),
            dealloc_bytes: current.dealloc_bytes.saturating_sub(start.dealloc_bytes),
        }
    }
}

pub(crate) struct AllocationAuditGuard {
    previous: bool,
}

impl AllocationAuditGuard {
    pub(crate) fn enable(enabled: bool) -> Self {
        let previous = ALLOCATION_AUDIT_ENABLED.swap(enabled, Ordering::Relaxed);
        Self { previous }
    }
}

impl Drop for AllocationAuditGuard {
    fn drop(&mut self) {
        ALLOCATION_AUDIT_ENABLED.store(self.previous, Ordering::Relaxed);
    }
}

pub(crate) fn allocation_samples_json(samples: &[AllocationAuditSnapshot]) -> serde_json::Value {
    if samples.is_empty() {
        return serde_json::json!({
            "count": 0,
            "alloc_count_total": 0,
            "alloc_bytes_total": 0,
            "dealloc_count_total": 0,
            "dealloc_bytes_total": 0,
            "net_bytes_total": 0,
            "mean_alloc_count_per_token": serde_json::Value::Null,
            "mean_alloc_bytes_per_token": serde_json::Value::Null,
            "max_alloc_count_per_token": serde_json::Value::Null,
            "max_alloc_bytes_per_token": serde_json::Value::Null,
        });
    }

    let total_alloc_count = samples.iter().map(|sample| sample.alloc_count).sum::<u64>();
    let total_alloc_bytes = samples.iter().map(|sample| sample.alloc_bytes).sum::<u64>();
    let total_dealloc_count = samples.iter().map(|sample| sample.dealloc_count).sum::<u64>();
    let total_dealloc_bytes = samples.iter().map(|sample| sample.dealloc_bytes).sum::<u64>();
    let max_alloc_count = samples.iter().map(|sample| sample.alloc_count).max().unwrap_or(0);
    let max_alloc_bytes = samples.iter().map(|sample| sample.alloc_bytes).max().unwrap_or(0);
    let count = samples.len() as f64;

    let net_bytes_total = total_alloc_bytes as i64 - total_dealloc_bytes as i64;

    serde_json::json!({
        "count": samples.len(),
        "alloc_count_total": total_alloc_count,
        "alloc_bytes_total": total_alloc_bytes,
        "dealloc_count_total": total_dealloc_count,
        "dealloc_bytes_total": total_dealloc_bytes,
        "net_bytes_total": net_bytes_total,
        "mean_alloc_count_per_token": rounded_ms(total_alloc_count as f64 / count),
        "mean_alloc_bytes_per_token": rounded_ms(total_alloc_bytes as f64 / count),
        "max_alloc_count_per_token": max_alloc_count,
        "max_alloc_bytes_per_token": max_alloc_bytes,
    })
}

#[cfg(feature = "full-cli")]
pub(crate) struct WarmSessionPromptAllocationAudit<'a> {
    pub(crate) enabled: bool,
    pub(crate) requested_backend: &'a str,
    pub(crate) prompt_tokenize: AllocationAuditSnapshot,
    pub(crate) prompt_setup: AllocationAuditSnapshot,
    pub(crate) prompt_setup_breakdown: WarmSessionPromptSetupAllocationAudit,
    pub(crate) prompt_prefill: &'a [AllocationAuditSnapshot],
    pub(crate) prompt_prefill_embed: &'a [AllocationAuditSnapshot],
    pub(crate) prompt_prefill_forward: &'a [AllocationAuditSnapshot],
    pub(crate) decode_total: &'a [AllocationAuditSnapshot],
    pub(crate) embed: &'a [AllocationAuditSnapshot],
    pub(crate) forward: &'a [AllocationAuditSnapshot],
    pub(crate) logits: &'a [AllocationAuditSnapshot],
    pub(crate) sample: &'a [AllocationAuditSnapshot],
    pub(crate) token_vector_update: &'a [AllocationAuditSnapshot],
    pub(crate) token_decode: &'a [AllocationAuditSnapshot],
    pub(crate) stop_tail_update: &'a [AllocationAuditSnapshot],
    pub(crate) receipt_construction: AllocationAuditSnapshot,
}

#[cfg(feature = "full-cli")]
pub(crate) struct WarmSessionPromptSetupAllocationAudit {
    pub(crate) buffer_reset: AllocationAuditSnapshot,
    pub(crate) token_seed: AllocationAuditSnapshot,
    pub(crate) kv_cache: AllocationAuditSnapshot,
    pub(crate) sampler_setup: AllocationAuditSnapshot,
}

#[cfg(feature = "full-cli")]
pub(crate) fn warm_session_prompt_allocation_audit_json(
    audit: WarmSessionPromptAllocationAudit<'_>,
) -> serde_json::Value {
    if !audit.enabled {
        return serde_json::json!({
            "enabled": false,
            "method": "not_requested",
            "scope": "not_requested",
        });
    }

    let prompt_tokenize = std::slice::from_ref(&audit.prompt_tokenize);
    let prompt_setup = std::slice::from_ref(&audit.prompt_setup);
    let prompt_setup_buffer_reset =
        std::slice::from_ref(&audit.prompt_setup_breakdown.buffer_reset);
    let prompt_setup_token_seed = std::slice::from_ref(&audit.prompt_setup_breakdown.token_seed);
    let prompt_setup_kv_cache = std::slice::from_ref(&audit.prompt_setup_breakdown.kv_cache);
    let prompt_setup_sampler_setup =
        std::slice::from_ref(&audit.prompt_setup_breakdown.sampler_setup);
    let receipt_construction = std::slice::from_ref(&audit.receipt_construction);
    let mut hotspots = vec![
        allocation_hotspot("prompt_tokenize", prompt_tokenize),
        allocation_hotspot("prompt_setup", prompt_setup),
        allocation_hotspot("prompt_setup.buffer_reset", prompt_setup_buffer_reset),
        allocation_hotspot("prompt_setup.token_seed", prompt_setup_token_seed),
        allocation_hotspot("prompt_setup.kv_cache", prompt_setup_kv_cache),
        allocation_hotspot("prompt_setup.sampler_setup", prompt_setup_sampler_setup),
        allocation_hotspot("prompt_prefill", audit.prompt_prefill),
        allocation_hotspot("prompt_prefill.embed", audit.prompt_prefill_embed),
        allocation_hotspot("prompt_prefill.forward", audit.prompt_prefill_forward),
        allocation_hotspot("decode_total", audit.decode_total),
        allocation_hotspot("model.embed", audit.embed),
        allocation_hotspot("model.forward", audit.forward),
        allocation_hotspot("model.logits_and_extract", audit.logits),
        allocation_hotspot("sampler.sample", audit.sample),
        allocation_hotspot("token_vector_updates", audit.token_vector_update),
        allocation_hotspot("tokenizer.decode", audit.token_decode),
        allocation_hotspot("stop_tail_updates", audit.stop_tail_update),
        allocation_hotspot("receipt_construction", receipt_construction),
    ];
    hotspots.retain(|hotspot| hotspot.alloc_count > 0 || hotspot.alloc_bytes > 0);
    hotspots.sort_by(|left, right| {
        right
            .alloc_bytes
            .cmp(&left.alloc_bytes)
            .then_with(|| right.alloc_count.cmp(&left.alloc_count))
            .then_with(|| left.component.cmp(right.component))
    });

    serde_json::json!({
        "enabled": true,
        "method": "process_global_allocator_counter_delta",
        "scope": warm_session_allocation_scope(audit.requested_backend),
        "claim_scope": "allocation counter deltas for this prompt/profile only; sampling scratch cleanup is scoped and no broad performance improvement is claimed",
        "optimization_deferred": false,
        "unavoidable_candidates_named_before_optimization": true,
        "ranked_hotspots": hotspots.iter().map(AllocationHotspot::to_json).collect::<Vec<_>>(),
        "unavoidable_candidates": [
            "model.embed/model.forward/model.logits tensor outputs from the current dense Qwen CPU execution path",
            "tokenizer.decode allocation for per-token text and stop-tail checks",
            "receipt construction outside the decode hot loop",
            "prompt token vector growth is controlled by reusable session buffers; model tensor outputs remain the dominant allocation source"
        ],
        "instrumentation_included": [
            "prompt_tokenize",
            "prompt_setup",
            "prompt_setup.buffer_reset",
            "prompt_setup.token_seed",
            "prompt_setup.kv_cache",
            "prompt_setup.sampler_setup",
            "prompt_prefill_step",
            "prompt_prefill.embed",
            "prompt_prefill.forward",
            "decode_step_total",
            "model.embed",
            "model.forward",
            "model.logits_and_extract",
            "sampler.sample",
            "token_vector_updates",
            "tokenizer.decode",
            "stop_tail_updates",
            "receipt_construction"
        ],
        "instrumentation_excluded": [
            "model_load",
            "tokenizer_load",
            "aggregate_receipt_serialization",
            "OS allocator reuse and fragmentation outside counter deltas"
        ],
        "prompt_tokenize": allocation_samples_json(prompt_tokenize),
        "prompt_setup": allocation_samples_json(prompt_setup),
        "prompt_setup_breakdown_scope": "subcomponent counter deltas nested inside prompt_setup; these are attribution evidence and do not change generation behavior",
        "prompt_setup_breakdown": {
            "buffer_reset": allocation_samples_json(prompt_setup_buffer_reset),
            "token_seed": allocation_samples_json(prompt_setup_token_seed),
            "kv_cache": allocation_samples_json(prompt_setup_kv_cache),
            "sampler_setup": allocation_samples_json(prompt_setup_sampler_setup),
        },
        "prompt_prefill": allocation_samples_json(audit.prompt_prefill),
        "prompt_prefill_breakdown_scope": "subcomponent counter deltas nested inside prompt_prefill; these are attribution evidence and do not change generation behavior",
        "prompt_prefill_breakdown": {
            "embed": allocation_samples_json(audit.prompt_prefill_embed),
            "forward": allocation_samples_json(audit.prompt_prefill_forward),
            "forward_boundary": {
                "first_reusable_allocation_surface": "feed_forward.down_proj.output",
                "classification": "FeedForward::forward_with_workspace reaches the exact FeedForward::down_proj output boundary and the Candle Linear weight/bias tensors are readable, but Tensor::matmul plus optional broadcast_add still return owned tensors without caller-provided output storage",
                "reuse_status": "dense_linear_output_storage_blocked_by_candle_tensor_ops",
                "workspace_storage_owner": "TransformerForwardWorkspace",
                "workspace_owned_output_surface": "feed_forward.down_proj.output",
                "no_reuse_reason": "candle_nn::Linear exposes weight and optional bias tensors, but its behavior-preserving compute path is Tensor::matmul plus optional broadcast_add, and those operations return owned Tensors without a caller-provided output-storage parameter",
                "required_api_boundary": "dense_linear_output_storage_api_boundary",
                "next_safe_change": "instrument and optimize the Q8_0 dense linear locality boundary after SLM-CPU-041 proved reusable output storage is blocked by Candle Tensor matmul/bias-add owned returns",
                "next_dense_math_boundary": {
                    "target": "q8_dense_linear_locality_boundary",
                    "source": "SLM-CPU-042",
                    "current_path": "eager_dense_standard_quant_dequant_to_f32_before_candle_tensor",
                    "dequantizes_before_compute": true,
                    "materializes_f32_tensor": true,
                    "must_preserve": "generated IDs, decoded text, strict GGUF tokenizer authority, selected CPU backend/kernel, model SHA, and fallback=false"
                },
                "weight_accessible": true,
                "bias_accessible": true,
                "can_fill_caller_output_storage": false,
                "behavior_gate": "generated IDs, decoded text, strict GGUF tokenizer authority, selected CPU backend/kernel, model SHA, and fallback=false must match the Qwen3 Q8_0 baseline",
                "claim_scope": "allocation-boundary classification only; no dense math, kernel, or sustained-throughput claim is made",
            },
        },
        "decode": {
            "total": allocation_samples_json(audit.decode_total),
            "steady_state": allocation_samples_json(audit.decode_total.get(1..).unwrap_or(&[])),
            "embed": allocation_samples_json(audit.embed),
            "forward": allocation_samples_json(audit.forward),
            "logits": allocation_samples_json(audit.logits),
            "sample": allocation_samples_json(audit.sample),
            "token_vector_update": allocation_samples_json(audit.token_vector_update),
            "token_decode": allocation_samples_json(audit.token_decode),
            "stop_tail_update": allocation_samples_json(audit.stop_tail_update),
        },
        "receipt_construction": allocation_samples_json(receipt_construction),
    })
}

#[cfg(feature = "full-cli")]
pub(crate) fn warm_session_aggregate_allocation_audit_json(
    enabled: bool,
    requested_backend: &str,
    prompt_summaries: &[serde_json::Value],
) -> serde_json::Value {
    if !enabled {
        return serde_json::json!({
            "enabled": false,
            "method": "not_requested",
            "scope": "not_requested",
        });
    }

    let mut totals = std::collections::BTreeMap::<String, (u64, u64)>::new();
    for prompt in prompt_summaries {
        let Some(hotspots) = prompt["allocation_audit"]["ranked_hotspots"].as_array() else {
            continue;
        };
        for hotspot in hotspots {
            let Some(component) = hotspot["component"].as_str() else {
                continue;
            };
            let entry = totals.entry(component.to_string()).or_default();
            entry.0 += hotspot["alloc_count"].as_u64().unwrap_or_default();
            entry.1 += hotspot["alloc_bytes"].as_u64().unwrap_or_default();
        }
    }
    let mut ranked = totals
        .into_iter()
        .map(|(component, (alloc_count, alloc_bytes))| {
            serde_json::json!({
                "component": component,
                "alloc_count": alloc_count,
                "alloc_bytes": alloc_bytes,
            })
        })
        .collect::<Vec<_>>();
    ranked.sort_by(|left, right| {
        right["alloc_bytes"]
            .as_u64()
            .unwrap_or_default()
            .cmp(&left["alloc_bytes"].as_u64().unwrap_or_default())
            .then_with(|| {
                right["alloc_count"]
                    .as_u64()
                    .unwrap_or_default()
                    .cmp(&left["alloc_count"].as_u64().unwrap_or_default())
            })
            .then_with(|| {
                left["component"]
                    .as_str()
                    .unwrap_or_default()
                    .cmp(right["component"].as_str().unwrap_or_default())
            })
    });
    let dominant_hotspot = ranked.first().cloned().unwrap_or(serde_json::Value::Null);
    let next_optimization_target = warm_session_next_optimization_target(&ranked);
    let optimization_deferred = matches!(
        next_optimization_target["status"].as_str(),
        Some(
            "workspace_api_present_reuse_deferred"
                | "workspace_owned_output_reuse_deferred"
                | "dense_linear_output_storage_blocked_by_candle_tensor_ops"
        )
    );

    serde_json::json!({
        "enabled": true,
        "method": "process_global_allocator_counter_delta",
        "scope": warm_session_allocation_scope(requested_backend),
        "claim_scope": "aggregate of prompt-level allocation counter deltas; sampling scratch cleanup is scoped and no broad performance improvement is claimed",
        "prompt_count": prompt_summaries.len(),
        "ranked_hotspots": ranked,
        "dominant_hotspot": dominant_hotspot,
        "next_optimization_target": next_optimization_target,
        "optimization_deferred": optimization_deferred,
    })
}

#[cfg(feature = "full-cli")]
fn warm_session_next_optimization_target(
    ranked_hotspots: &[serde_json::Value],
) -> serde_json::Value {
    let component =
        ranked_hotspots.first().and_then(|hotspot| hotspot["component"].as_str()).unwrap_or("none");
    let (target, rationale, status) = match component {
        "prompt_prefill" => (
            "q8_dense_linear_locality_boundary",
            "prompt prefill dominates aggregate allocation counters; prompt_prefill.forward is the measured subcomponent, reusable output storage is blocked by Candle Tensor matmul/bias-add owned returns, and the next safe slice is Q8_0 dense linear locality instrumentation",
            "dense_linear_output_storage_blocked_by_candle_tensor_ops",
        ),
        "prompt_prefill.forward" => (
            "q8_dense_linear_locality_boundary",
            "prompt_prefill.forward dominates aggregate allocation counters; the exact FeedForward::down_proj output boundary is reached, reusable output storage is blocked by Candle Tensor APIs, and the next safe slice is Q8_0 dense linear locality instrumentation",
            "dense_linear_output_storage_blocked_by_candle_tensor_ops",
        ),
        "prompt_prefill.embed" => (
            "prefill_embedding_allocation_attribution",
            "prompt embedding allocation dominates aggregate allocation counters; preserve prompt IDs before changing embedding layout",
            "needs_attribution",
        ),
        "decode_total" | "model.forward" => (
            "decode_model_forward_allocation_attribution",
            "decode/model.forward dominates aggregate allocation counters; attribute dense tensor outputs before changing kernels",
            "needs_attribution",
        ),
        "model.logits_and_extract" => (
            "logits_extraction_boundary",
            "logits extraction remains the dominant allocation counter source after sampler and logits scratch reuse",
            "needs_attribution",
        ),
        "prompt_tokenize" => (
            "prompt_token_cache_or_tokenizer_boundary",
            "prompt tokenization dominates aggregate allocation counters; keep prompt-cache behavior receipt-visible",
            "needs_attribution",
        ),
        "prompt_setup"
        | "prompt_setup.buffer_reset"
        | "prompt_setup.token_seed"
        | "prompt_setup.kv_cache"
        | "prompt_setup.sampler_setup" => (
            "prompt_setup_boundary",
            "prompt setup dominates aggregate allocation counters; preserve prompt isolation while narrowing setup work",
            "needs_attribution",
        ),
        "none" => {
            ("none", "no allocation hotspots were recorded by the aggregate audit", "not_available")
        }
        _ => (
            "measured_hotspot_followup",
            "the next target follows the dominant measured allocation hotspot and must preserve generated IDs",
            "needs_attribution",
        ),
    };

    serde_json::json!({
        "component": component,
        "target": target,
        "rationale": rationale,
        "status": status,
        "claim_scope": "diagnostic prioritization only; no runtime optimization or sustained-throughput claim is made",
    })
}

#[cfg(feature = "full-cli")]
pub(crate) fn warm_session_allocation_scope(requested_backend: &str) -> &'static str {
    match requested_backend.trim().to_ascii_lowercase().as_str() {
        "cpu" => "selected generic CPU SLM warm-session prompt hot path",
        "apple-m3-air-cpu-neon" => {
            "selected Apple M3 Air CPU/NEON SLM warm-session prompt hot path"
        }
        _ => "selected Apple M4 CPU/NEON SLM warm-session prompt hot path",
    }
}

#[cfg(feature = "full-cli")]
pub(crate) struct AllocationHotspot {
    pub(crate) component: &'static str,
    pub(crate) alloc_count: u64,
    pub(crate) alloc_bytes: u64,
}

#[cfg(feature = "full-cli")]
impl AllocationHotspot {
    pub(crate) fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "component": self.component,
            "alloc_count": self.alloc_count,
            "alloc_bytes": self.alloc_bytes,
        })
    }
}

#[cfg(feature = "full-cli")]
pub(crate) fn allocation_hotspot(
    component: &'static str,
    samples: &[AllocationAuditSnapshot],
) -> AllocationHotspot {
    AllocationHotspot {
        component,
        alloc_count: samples.iter().map(|sample| sample.alloc_count).sum(),
        alloc_bytes: samples.iter().map(|sample| sample.alloc_bytes).sum(),
    }
}

pub(crate) fn allocation_count_delta_json(
    samples: &[AllocationAuditSnapshot],
) -> serde_json::Value {
    if samples.is_empty() {
        return serde_json::json!({
            "count": 0,
            "total": 0,
            "mean_per_token": serde_json::Value::Null,
            "max_per_token": serde_json::Value::Null,
        });
    }

    let total = samples.iter().map(|sample| sample.alloc_count).sum::<u64>();
    let max = samples.iter().map(|sample| sample.alloc_count).max().unwrap_or(0);

    serde_json::json!({
        "count": samples.len(),
        "total": total,
        "mean_per_token": rounded_ms(total as f64 / samples.len() as f64),
        "max_per_token": max,
    })
}

pub(crate) fn allocation_bytes_delta_json(
    samples: &[AllocationAuditSnapshot],
) -> serde_json::Value {
    if samples.is_empty() {
        return serde_json::json!({
            "count": 0,
            "total": 0,
            "mean_per_token": serde_json::Value::Null,
            "max_per_token": serde_json::Value::Null,
        });
    }

    let total = samples.iter().map(|sample| sample.alloc_bytes).sum::<u64>();
    let max = samples.iter().map(|sample| sample.alloc_bytes).max().unwrap_or(0);

    serde_json::json!({
        "count": samples.len(),
        "total": total,
        "mean_per_token": rounded_ms(total as f64 / samples.len() as f64),
        "max_per_token": max,
    })
}

pub(crate) fn mean_alloc_count(samples: &[AllocationAuditSnapshot]) -> Option<f64> {
    if samples.is_empty() {
        return None;
    }
    Some(samples.iter().map(|sample| sample.alloc_count).sum::<u64>() as f64 / samples.len() as f64)
}

pub(crate) fn mean_alloc_bytes(samples: &[AllocationAuditSnapshot]) -> Option<f64> {
    if samples.is_empty() {
        return None;
    }
    Some(samples.iter().map(|sample| sample.alloc_bytes).sum::<u64>() as f64 / samples.len() as f64)
}
