//! NEON inference bridge — connects ARM NEON kernel implementations to the
//! inference pipeline.
//!
//! [`NeonInferenceProvider`] detects NEON capabilities at runtime and provides
//! dispatch methods for common inference operations (matmul, softmax,
//! layernorm, RoPE, attention, embedding, quantization, activation).
//!
//! All kernel calls are stubs that track dispatch/fallback counts; the actual
//! wiring to individual NEON kernels is done in follow-up PRs.

use std::sync::atomic::{AtomicU64, Ordering};

// ── Capability flags ────────────────────────────────────────────────────

/// Flags describing which NEON kernels are available on the current platform.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NeonKernelCapabilities {
    /// ARM NEON is present (always true on AArch64).
    pub neon: bool,
    /// NEON-accelerated matrix multiplication.
    pub matmul: bool,
    /// NEON-accelerated softmax.
    pub softmax: bool,
    /// NEON-accelerated layer normalization.
    pub layernorm: bool,
    /// NEON-accelerated rotary position embedding.
    pub rope: bool,
    /// NEON-accelerated attention.
    pub attention: bool,
    /// NEON-accelerated embedding lookup.
    pub embedding: bool,
    /// NEON-accelerated quantization/dequantization.
    pub quantization: bool,
    /// NEON-accelerated activation functions.
    pub activation: bool,
}

impl NeonKernelCapabilities {
    /// All capabilities disabled.
    pub fn none() -> Self {
        Self {
            neon: false,
            matmul: false,
            softmax: false,
            layernorm: false,
            rope: false,
            attention: false,
            embedding: false,
            quantization: false,
            activation: false,
        }
    }

    /// Returns the number of kernel categories that are available.
    pub fn available_count(&self) -> usize {
        [
            self.matmul,
            self.softmax,
            self.layernorm,
            self.rope,
            self.attention,
            self.embedding,
            self.quantization,
            self.activation,
        ]
        .iter()
        .filter(|&&v| v)
        .count()
    }
}

/// Detect NEON capabilities at runtime.
///
/// On AArch64 targets NEON is architecturally guaranteed, so all kernel
/// categories are reported as available.  On non-AArch64 this returns
/// [`NeonKernelCapabilities::none()`].
pub fn detect_capabilities() -> NeonKernelCapabilities {
    #[cfg(target_arch = "aarch64")]
    {
        NeonKernelCapabilities {
            neon: true,
            matmul: true,
            softmax: true,
            layernorm: true,
            rope: true,
            attention: true,
            embedding: true,
            quantization: true,
            activation: true,
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        NeonKernelCapabilities::none()
    }
}

// ── Dispatch configuration ──────────────────────────────────────────────

/// Configuration for a single inference dispatch call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NeonDispatchConfig {
    /// Batch size (must be ≥ 1).
    pub batch_size: usize,
    /// Sequence length (must be ≥ 1).
    pub sequence_length: usize,
    /// Hidden dimension (must be ≥ 1).
    pub hidden_dim: usize,
    /// Number of attention heads (must be ≥ 1).
    pub num_heads: usize,
}

impl NeonDispatchConfig {
    /// Create a new config, returning `None` if any dimension is zero.
    pub fn new(
        batch_size: usize,
        sequence_length: usize,
        hidden_dim: usize,
        num_heads: usize,
    ) -> Option<Self> {
        if batch_size == 0 || sequence_length == 0 || hidden_dim == 0 || num_heads == 0 {
            return None;
        }
        Some(Self { batch_size, sequence_length, hidden_dim, num_heads })
    }

    /// Validate that `hidden_dim` is evenly divisible by `num_heads`.
    pub fn head_dim_valid(&self) -> bool {
        self.hidden_dim % self.num_heads == 0
    }

    /// Per-head dimension (panics if not evenly divisible).
    pub fn head_dim(&self) -> usize {
        assert!(
            self.head_dim_valid(),
            "hidden_dim ({}) not divisible by num_heads ({})",
            self.hidden_dim,
            self.num_heads,
        );
        self.hidden_dim / self.num_heads
    }

    /// Total number of elements in [batch, seq, hidden] tensor.
    pub fn total_elements(&self) -> usize {
        self.batch_size.saturating_mul(self.sequence_length).saturating_mul(self.hidden_dim)
    }
}

// ── Dispatch result ─────────────────────────────────────────────────────

/// Outcome of a dispatch call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DispatchOutcome {
    /// Dispatched to NEON kernel (stub — actual call pending).
    Neon,
    /// Fell back to scalar / generic path.
    Fallback,
}

// ── Stats ───────────────────────────────────────────────────────────────

/// Accumulated statistics for dispatch operations.
///
/// Counters use `AtomicU64` so the provider can be shared across threads
/// (though current usage is single-threaded).
#[derive(Debug)]
pub struct NeonInferenceStats {
    pub neon_dispatches: AtomicU64,
    pub fallback_dispatches: AtomicU64,
    pub matmul_ops: AtomicU64,
    pub softmax_ops: AtomicU64,
    pub layernorm_ops: AtomicU64,
    pub rope_ops: AtomicU64,
    pub attention_ops: AtomicU64,
    pub embedding_ops: AtomicU64,
    pub quantization_ops: AtomicU64,
    pub activation_ops: AtomicU64,
}

impl NeonInferenceStats {
    fn new() -> Self {
        Self {
            neon_dispatches: AtomicU64::new(0),
            fallback_dispatches: AtomicU64::new(0),
            matmul_ops: AtomicU64::new(0),
            softmax_ops: AtomicU64::new(0),
            layernorm_ops: AtomicU64::new(0),
            rope_ops: AtomicU64::new(0),
            attention_ops: AtomicU64::new(0),
            embedding_ops: AtomicU64::new(0),
            quantization_ops: AtomicU64::new(0),
            activation_ops: AtomicU64::new(0),
        }
    }

    /// Total dispatch count (NEON + fallback).
    pub fn total_dispatches(&self) -> u64 {
        self.neon_dispatches.load(Ordering::Relaxed)
            + self.fallback_dispatches.load(Ordering::Relaxed)
    }

    /// Snapshot all counters into plain `u64` values.
    pub fn snapshot(&self) -> (u64, u64, u64, u64, u64, u64, u64, u64, u64, u64) {
        (
            self.neon_dispatches.load(Ordering::Relaxed),
            self.fallback_dispatches.load(Ordering::Relaxed),
            self.matmul_ops.load(Ordering::Relaxed),
            self.softmax_ops.load(Ordering::Relaxed),
            self.layernorm_ops.load(Ordering::Relaxed),
            self.rope_ops.load(Ordering::Relaxed),
            self.attention_ops.load(Ordering::Relaxed),
            self.embedding_ops.load(Ordering::Relaxed),
            self.quantization_ops.load(Ordering::Relaxed),
            self.activation_ops.load(Ordering::Relaxed),
        )
    }
}

// ── Provider ────────────────────────────────────────────────────────────

/// Main dispatch coordinator for NEON inference kernels.
///
/// Detects NEON capability once at construction, then every `dispatch_*`
/// call either increments the NEON counter (stub) or the fallback counter.
pub struct NeonInferenceProvider {
    capabilities: NeonKernelCapabilities,
    stats: NeonInferenceStats,
}

impl NeonInferenceProvider {
    /// Create a provider, probing NEON capability.
    pub fn new() -> Self {
        Self { capabilities: detect_capabilities(), stats: NeonInferenceStats::new() }
    }

    /// Create a provider with explicitly supplied capabilities (for testing).
    pub fn with_capabilities(caps: NeonKernelCapabilities) -> Self {
        Self { capabilities: caps, stats: NeonInferenceStats::new() }
    }

    /// Runtime capabilities.
    pub fn capabilities(&self) -> &NeonKernelCapabilities {
        &self.capabilities
    }

    /// Accumulated dispatch statistics.
    pub fn stats(&self) -> &NeonInferenceStats {
        &self.stats
    }

    // ── Dispatch helpers (private) ──────────────────────────────────

    fn record(&self, neon: bool, op: &AtomicU64) -> DispatchOutcome {
        op.fetch_add(1, Ordering::Relaxed);
        if neon {
            self.stats.neon_dispatches.fetch_add(1, Ordering::Relaxed);
            DispatchOutcome::Neon
        } else {
            self.stats.fallback_dispatches.fetch_add(1, Ordering::Relaxed);
            DispatchOutcome::Fallback
        }
    }

    // ── Public dispatch methods ─────────────────────────────────────

    /// Dispatch a matrix multiplication.
    ///
    /// Stub: records the dispatch and returns the outcome.
    pub fn dispatch_matmul(&self, _config: &NeonDispatchConfig) -> DispatchOutcome {
        self.record(self.capabilities.matmul, &self.stats.matmul_ops)
    }

    /// Dispatch a softmax operation.
    pub fn dispatch_softmax(&self, _config: &NeonDispatchConfig) -> DispatchOutcome {
        self.record(self.capabilities.softmax, &self.stats.softmax_ops)
    }

    /// Dispatch layer normalization.
    pub fn dispatch_layernorm(&self, _config: &NeonDispatchConfig) -> DispatchOutcome {
        self.record(self.capabilities.layernorm, &self.stats.layernorm_ops)
    }

    /// Dispatch rotary position embedding.
    pub fn dispatch_rope(&self, _config: &NeonDispatchConfig) -> DispatchOutcome {
        self.record(self.capabilities.rope, &self.stats.rope_ops)
    }

    /// Dispatch attention computation.
    pub fn dispatch_attention(&self, _config: &NeonDispatchConfig) -> DispatchOutcome {
        self.record(self.capabilities.attention, &self.stats.attention_ops)
    }

    /// Dispatch embedding lookup.
    pub fn dispatch_embedding(&self, _config: &NeonDispatchConfig) -> DispatchOutcome {
        self.record(self.capabilities.embedding, &self.stats.embedding_ops)
    }

    /// Dispatch quantization / dequantization.
    pub fn dispatch_quantization(&self, _config: &NeonDispatchConfig) -> DispatchOutcome {
        self.record(self.capabilities.quantization, &self.stats.quantization_ops)
    }

    /// Dispatch activation function.
    pub fn dispatch_activation(&self, _config: &NeonDispatchConfig) -> DispatchOutcome {
        self.record(self.capabilities.activation, &self.stats.activation_ops)
    }
}

impl Default for NeonInferenceProvider {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_config() -> NeonDispatchConfig {
        NeonDispatchConfig::new(1, 128, 768, 12).unwrap()
    }

    fn all_caps() -> NeonKernelCapabilities {
        NeonKernelCapabilities {
            neon: true,
            matmul: true,
            softmax: true,
            layernorm: true,
            rope: true,
            attention: true,
            embedding: true,
            quantization: true,
            activation: true,
        }
    }

    fn no_caps() -> NeonKernelCapabilities {
        NeonKernelCapabilities::none()
    }

    // ── Capability detection ────────────────────────────────────────

    #[test]
    fn detect_capabilities_returns_all_on_aarch64() {
        let caps = detect_capabilities();
        // On aarch64 everything is true; on other arches everything false.
        #[cfg(target_arch = "aarch64")]
        assert!(caps.neon);
        #[cfg(not(target_arch = "aarch64"))]
        assert!(!caps.neon);
    }

    #[test]
    fn capabilities_none_has_all_false() {
        let caps = NeonKernelCapabilities::none();
        assert!(!caps.neon);
        assert!(!caps.matmul);
        assert!(!caps.softmax);
        assert_eq!(caps.available_count(), 0);
    }

    #[test]
    fn capabilities_all_count_is_eight() {
        let caps = all_caps();
        assert_eq!(caps.available_count(), 8);
    }

    #[test]
    fn capabilities_partial_count() {
        let mut caps = no_caps();
        caps.matmul = true;
        caps.softmax = true;
        assert_eq!(caps.available_count(), 2);
    }

    #[test]
    fn capabilities_equality() {
        assert_eq!(no_caps(), no_caps());
        assert_ne!(all_caps(), no_caps());
    }

    // ── Config validation ───────────────────────────────────────────

    #[test]
    fn config_valid() {
        let cfg = NeonDispatchConfig::new(2, 64, 512, 8);
        assert!(cfg.is_some());
        let cfg = cfg.unwrap();
        assert_eq!(cfg.head_dim(), 64);
    }

    #[test]
    fn config_zero_batch_is_none() {
        assert!(NeonDispatchConfig::new(0, 64, 512, 8).is_none());
    }

    #[test]
    fn config_zero_seq_is_none() {
        assert!(NeonDispatchConfig::new(1, 0, 512, 8).is_none());
    }

    #[test]
    fn config_zero_hidden_is_none() {
        assert!(NeonDispatchConfig::new(1, 64, 0, 8).is_none());
    }

    #[test]
    fn config_zero_heads_is_none() {
        assert!(NeonDispatchConfig::new(1, 64, 512, 0).is_none());
    }

    #[test]
    fn config_head_dim_not_divisible() {
        let cfg = NeonDispatchConfig::new(1, 64, 100, 3).unwrap();
        assert!(!cfg.head_dim_valid());
    }

    #[test]
    #[should_panic(expected = "not divisible")]
    fn config_head_dim_panics_on_bad_div() {
        let cfg = NeonDispatchConfig::new(1, 64, 100, 3).unwrap();
        let _ = cfg.head_dim();
    }

    #[test]
    fn config_total_elements() {
        let cfg = NeonDispatchConfig::new(2, 128, 768, 12).unwrap();
        assert_eq!(cfg.total_elements(), 2 * 128 * 768);
    }

    #[test]
    fn config_huge_dims_saturate() {
        let cfg = NeonDispatchConfig::new(usize::MAX, 2, 2, 1).unwrap();
        // Should saturate rather than panic.
        assert!(cfg.total_elements() <= usize::MAX);
    }

    // ── Dispatch with NEON ──────────────────────────────────────────

    #[test]
    fn dispatch_matmul_neon() {
        let p = NeonInferenceProvider::with_capabilities(all_caps());
        assert_eq!(p.dispatch_matmul(&sample_config()), DispatchOutcome::Neon);
        assert_eq!(p.stats().matmul_ops.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn dispatch_softmax_neon() {
        let p = NeonInferenceProvider::with_capabilities(all_caps());
        assert_eq!(p.dispatch_softmax(&sample_config()), DispatchOutcome::Neon);
    }

    #[test]
    fn dispatch_layernorm_neon() {
        let p = NeonInferenceProvider::with_capabilities(all_caps());
        assert_eq!(p.dispatch_layernorm(&sample_config()), DispatchOutcome::Neon);
    }

    #[test]
    fn dispatch_rope_neon() {
        let p = NeonInferenceProvider::with_capabilities(all_caps());
        assert_eq!(p.dispatch_rope(&sample_config()), DispatchOutcome::Neon);
    }

    #[test]
    fn dispatch_attention_neon() {
        let p = NeonInferenceProvider::with_capabilities(all_caps());
        assert_eq!(p.dispatch_attention(&sample_config()), DispatchOutcome::Neon);
    }

    #[test]
    fn dispatch_embedding_neon() {
        let p = NeonInferenceProvider::with_capabilities(all_caps());
        assert_eq!(p.dispatch_embedding(&sample_config()), DispatchOutcome::Neon);
    }

    #[test]
    fn dispatch_quantization_neon() {
        let p = NeonInferenceProvider::with_capabilities(all_caps());
        assert_eq!(p.dispatch_quantization(&sample_config()), DispatchOutcome::Neon);
    }

    #[test]
    fn dispatch_activation_neon() {
        let p = NeonInferenceProvider::with_capabilities(all_caps());
        assert_eq!(p.dispatch_activation(&sample_config()), DispatchOutcome::Neon);
    }

    // ── Dispatch with fallback ──────────────────────────────────────

    #[test]
    fn dispatch_matmul_fallback() {
        let p = NeonInferenceProvider::with_capabilities(no_caps());
        assert_eq!(p.dispatch_matmul(&sample_config()), DispatchOutcome::Fallback);
    }

    #[test]
    fn dispatch_softmax_fallback() {
        let p = NeonInferenceProvider::with_capabilities(no_caps());
        assert_eq!(p.dispatch_softmax(&sample_config()), DispatchOutcome::Fallback);
    }

    #[test]
    fn dispatch_rope_fallback() {
        let p = NeonInferenceProvider::with_capabilities(no_caps());
        assert_eq!(p.dispatch_rope(&sample_config()), DispatchOutcome::Fallback);
    }

    // ── Stats accumulation ──────────────────────────────────────────

    #[test]
    fn stats_accumulate_across_operations() {
        let p = NeonInferenceProvider::with_capabilities(all_caps());
        let cfg = sample_config();
        p.dispatch_matmul(&cfg);
        p.dispatch_matmul(&cfg);
        p.dispatch_softmax(&cfg);
        p.dispatch_layernorm(&cfg);
        p.dispatch_rope(&cfg);
        p.dispatch_attention(&cfg);
        assert_eq!(p.stats().matmul_ops.load(Ordering::Relaxed), 2);
        assert_eq!(p.stats().softmax_ops.load(Ordering::Relaxed), 1);
        assert_eq!(p.stats().neon_dispatches.load(Ordering::Relaxed), 6);
        assert_eq!(p.stats().total_dispatches(), 6);
    }

    #[test]
    fn stats_mixed_neon_and_fallback() {
        let mut caps = no_caps();
        caps.matmul = true; // only matmul uses NEON
        let p = NeonInferenceProvider::with_capabilities(caps);
        let cfg = sample_config();
        p.dispatch_matmul(&cfg);
        p.dispatch_softmax(&cfg);
        assert_eq!(p.stats().neon_dispatches.load(Ordering::Relaxed), 1);
        assert_eq!(p.stats().fallback_dispatches.load(Ordering::Relaxed), 1);
        assert_eq!(p.stats().total_dispatches(), 2);
    }

    #[test]
    fn stats_snapshot_returns_all_counters() {
        let p = NeonInferenceProvider::with_capabilities(all_caps());
        let cfg = sample_config();
        p.dispatch_matmul(&cfg);
        p.dispatch_softmax(&cfg);
        let (neon, fb, mm, sm, ln, rp, att, emb, qt, act) = p.stats().snapshot();
        assert_eq!(neon, 2);
        assert_eq!(fb, 0);
        assert_eq!(mm, 1);
        assert_eq!(sm, 1);
        assert_eq!(ln, 0);
        assert_eq!(rp, 0);
        assert_eq!(att, 0);
        assert_eq!(emb, 0);
        assert_eq!(qt, 0);
        assert_eq!(act, 0);
    }

    #[test]
    fn stats_fresh_provider_is_zero() {
        let p = NeonInferenceProvider::new();
        assert_eq!(p.stats().total_dispatches(), 0);
    }

    // ── Provider construction ───────────────────────────────────────

    #[test]
    fn default_provider_matches_new() {
        let a = NeonInferenceProvider::new();
        let b = NeonInferenceProvider::default();
        assert_eq!(a.capabilities(), b.capabilities());
    }

    #[test]
    fn provider_with_custom_caps() {
        let mut caps = no_caps();
        caps.rope = true;
        let p = NeonInferenceProvider::with_capabilities(caps);
        assert!(p.capabilities().rope);
        assert!(!p.capabilities().matmul);
    }
}
