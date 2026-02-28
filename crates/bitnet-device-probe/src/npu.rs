//! Intel NPU (Neural Processing Unit) detection and workload routing.
//!
//! Provides capability detection for Intel NPU hardware (Meteor Lake and later)
//! and intelligent workload routing decisions between NPU, GPU, and CPU backends.
//!
//! # Environment Variables
//!
//! - `BITNET_NPU_FAKE=1` — simulate NPU presence for testing
//! - `BITNET_STRICT_MODE=1` — ignore fake env vars, probe real hardware only

use std::fmt;

// ── NPU operation types ──────────────────────────────────────────────────────

/// Workload operation types that can be routed to NPU.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NpuOp {
    /// Matrix multiplication (the primary NPU-accelerated operation).
    MatMul,
    /// Attention computation (Q·K^T · V).
    Attention,
    /// Layer normalization / RMS normalization.
    LayerNorm,
    /// Activation functions (SiLU, GELU, etc.).
    Activation,
    /// Embedding lookup.
    Embedding,
    /// Quantization / dequantization.
    Quantize,
}

impl fmt::Display for NpuOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MatMul => write!(f, "matmul"),
            Self::Attention => write!(f, "attention"),
            Self::LayerNorm => write!(f, "layer_norm"),
            Self::Activation => write!(f, "activation"),
            Self::Embedding => write!(f, "embedding"),
            Self::Quantize => write!(f, "quantize"),
        }
    }
}

// ── Performance hints ────────────────────────────────────────────────────────

/// Hint for workload latency/throughput preference.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PerfHint {
    /// Minimize latency (prefer NPU for small batches).
    LowLatency,
    /// Maximize throughput (prefer GPU for large batches).
    HighThroughput,
    /// Let the router decide based on workload characteristics.
    Auto,
}

// ── NPU capabilities ─────────────────────────────────────────────────────────

/// Detected NPU capabilities.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NpuCapabilities {
    /// NPU hardware is present and accessible.
    pub available: bool,
    /// NPU device name (e.g., "Intel AI Boost" on Meteor Lake).
    pub device_name: String,
    /// Maximum supported matrix dimension for efficient execution.
    pub max_matrix_dim: usize,
    /// Whether INT8 inference is supported.
    pub supports_int8: bool,
    /// Whether INT4 / ternary inference is supported.
    pub supports_int4: bool,
}

impl NpuCapabilities {
    /// Returns capabilities for when no NPU is detected.
    pub fn unavailable() -> Self {
        Self {
            available: false,
            device_name: String::new(),
            max_matrix_dim: 0,
            supports_int8: false,
            supports_int4: false,
        }
    }

    /// Returns simulated capabilities for testing.
    fn fake() -> Self {
        Self {
            available: true,
            device_name: "Intel AI Boost (simulated)".to_string(),
            max_matrix_dim: 4096,
            supports_int8: true,
            supports_int4: true,
        }
    }
}

// ── Route target & decision ──────────────────────────────────────────────────

/// Target backend for workload execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RouteTarget {
    /// Execute on the Neural Processing Unit.
    Npu,
    /// Execute on a discrete/integrated GPU.
    Gpu,
    /// Execute on CPU with SIMD.
    Cpu,
}

impl fmt::Display for RouteTarget {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Npu => write!(f, "npu"),
            Self::Gpu => write!(f, "gpu"),
            Self::Cpu => write!(f, "cpu"),
        }
    }
}

/// Result of a workload routing decision.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RouteDecision {
    /// Chosen execution target.
    pub target: RouteTarget,
    /// Human-readable reason for the routing decision.
    pub reason: String,
}

// ── Routing logic ────────────────────────────────────────────────────────────

/// Route a workload to the best available backend.
///
/// Decision factors:
/// - NPU is preferred for MatMul and Attention when matrix dims fit within
///   NPU limits and the perf hint is `LowLatency` or `Auto`.
/// - GPU is preferred for large batches with `HighThroughput` hint.
/// - CPU is the universal fallback.
///
/// # Arguments
///
/// * `op` — The operation type to route.
/// * `matrix_dim` — Largest matrix dimension in the workload (M, N, or K).
/// * `hint` — Performance preference hint.
/// * `npu` — Detected NPU capabilities.
/// * `gpu_available` — Whether a GPU backend is available.
pub fn route_workload(
    op: NpuOp,
    matrix_dim: usize,
    hint: PerfHint,
    npu: &NpuCapabilities,
    gpu_available: bool,
) -> RouteDecision {
    // NPU-eligible operations
    let npu_eligible = matches!(op, NpuOp::MatMul | NpuOp::Attention | NpuOp::Quantize);

    // Try NPU first if available, eligible, and fits
    if npu.available && npu_eligible && matrix_dim <= npu.max_matrix_dim {
        if hint != PerfHint::HighThroughput {
            return RouteDecision {
                target: RouteTarget::Npu,
                reason: format!(
                    "{op} (dim={matrix_dim}) fits NPU limit ({})",
                    npu.max_matrix_dim,
                ),
            };
        }
    }

    // GPU for throughput-oriented or large workloads
    if gpu_available {
        if hint == PerfHint::HighThroughput {
            return RouteDecision {
                target: RouteTarget::Gpu,
                reason: format!("{op} routed to GPU for high throughput"),
            };
        }
        if matrix_dim > npu.max_matrix_dim || !npu.available {
            return RouteDecision {
                target: RouteTarget::Gpu,
                reason: format!("{op} (dim={matrix_dim}) exceeds NPU capacity or NPU unavailable"),
            };
        }
    }

    // CPU fallback
    RouteDecision {
        target: RouteTarget::Cpu,
        reason: format!("{op} falling back to CPU"),
    }
}

/// Build a prioritized fallback chain for a given operation.
///
/// Returns targets in priority order: best first, CPU always last.
pub fn fallback_chain(
    npu: &NpuCapabilities,
    gpu_available: bool,
) -> Vec<RouteTarget> {
    let mut chain = Vec::with_capacity(3);

    if npu.available {
        chain.push(RouteTarget::Npu);
    }
    if gpu_available {
        chain.push(RouteTarget::Gpu);
    }
    chain.push(RouteTarget::Cpu);

    chain
}

// ── Detection ────────────────────────────────────────────────────────────────

/// Detect Intel NPU hardware.
///
/// Respects `BITNET_NPU_FAKE=1` for deterministic testing (unless
/// `BITNET_STRICT_MODE=1` is set).
pub fn detect_npu() -> NpuCapabilities {
    let strict = std::env::var("BITNET_STRICT_MODE")
        .map(|v| v == "1" || v.to_lowercase() == "true")
        .unwrap_or(false);

    if !strict {
        if let Ok(fake) = std::env::var("BITNET_NPU_FAKE") {
            if fake == "1" || fake.to_lowercase() == "true" {
                return NpuCapabilities::fake();
            }
        }
    }

    // Real hardware detection: try Intel NPU driver via accel subsystem.
    // On Linux this would check /dev/accel/accel* or /sys/class/accel/.
    // On Windows this would check for Intel NPU driver via SetupAPI.
    // For now, return unavailable — real detection is platform-specific.
    NpuCapabilities::unavailable()
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn npu_op_display() {
        assert_eq!(NpuOp::MatMul.to_string(), "matmul");
        assert_eq!(NpuOp::Attention.to_string(), "attention");
        assert_eq!(NpuOp::LayerNorm.to_string(), "layer_norm");
        assert_eq!(NpuOp::Activation.to_string(), "activation");
        assert_eq!(NpuOp::Embedding.to_string(), "embedding");
        assert_eq!(NpuOp::Quantize.to_string(), "quantize");
    }

    #[test]
    fn route_target_display() {
        assert_eq!(RouteTarget::Npu.to_string(), "npu");
        assert_eq!(RouteTarget::Gpu.to_string(), "gpu");
        assert_eq!(RouteTarget::Cpu.to_string(), "cpu");
    }

    #[test]
    fn unavailable_npu_caps() {
        let caps = NpuCapabilities::unavailable();
        assert!(!caps.available);
        assert!(caps.device_name.is_empty());
        assert_eq!(caps.max_matrix_dim, 0);
    }

    #[test]
    fn fake_npu_caps() {
        let caps = NpuCapabilities::fake();
        assert!(caps.available);
        assert!(caps.device_name.contains("simulated"));
        assert_eq!(caps.max_matrix_dim, 4096);
        assert!(caps.supports_int8);
        assert!(caps.supports_int4);
    }

    #[test]
    fn route_matmul_to_npu_when_available() {
        let npu = NpuCapabilities::fake();
        let decision = route_workload(NpuOp::MatMul, 2048, PerfHint::Auto, &npu, true);
        assert_eq!(decision.target, RouteTarget::Npu);
        assert!(decision.reason.contains("fits NPU"));
    }

    #[test]
    fn route_large_matmul_to_gpu() {
        let npu = NpuCapabilities::fake(); // max_matrix_dim = 4096
        let decision = route_workload(NpuOp::MatMul, 8192, PerfHint::Auto, &npu, true);
        assert_eq!(decision.target, RouteTarget::Gpu);
        assert!(decision.reason.contains("exceeds NPU"));
    }

    #[test]
    fn route_high_throughput_to_gpu() {
        let npu = NpuCapabilities::fake();
        let decision =
            route_workload(NpuOp::MatMul, 1024, PerfHint::HighThroughput, &npu, true);
        assert_eq!(decision.target, RouteTarget::Gpu);
        assert!(decision.reason.contains("high throughput"));
    }

    #[test]
    fn route_fallback_to_cpu() {
        let npu = NpuCapabilities::unavailable();
        let decision = route_workload(NpuOp::LayerNorm, 512, PerfHint::Auto, &npu, false);
        assert_eq!(decision.target, RouteTarget::Cpu);
        assert!(decision.reason.contains("CPU"));
    }

    #[test]
    fn fallback_chain_all_available() {
        let npu = NpuCapabilities::fake();
        let chain = fallback_chain(&npu, true);
        assert_eq!(chain, vec![RouteTarget::Npu, RouteTarget::Gpu, RouteTarget::Cpu]);
    }

    #[test]
    fn fallback_chain_no_npu() {
        let npu = NpuCapabilities::unavailable();
        let chain = fallback_chain(&npu, true);
        assert_eq!(chain, vec![RouteTarget::Gpu, RouteTarget::Cpu]);
    }

    #[test]
    fn fallback_chain_cpu_only() {
        let npu = NpuCapabilities::unavailable();
        let chain = fallback_chain(&npu, false);
        assert_eq!(chain, vec![RouteTarget::Cpu]);
    }

    #[test]
    #[serial_test::serial(bitnet_env)]
    fn detect_npu_fake_env() {
        temp_env::with_var("BITNET_STRICT_MODE", None::<&str>, || {
            temp_env::with_var("BITNET_NPU_FAKE", Some("1"), || {
                let caps = detect_npu();
                assert!(caps.available);
                assert!(caps.device_name.contains("simulated"));
            });
        });
    }

    #[test]
    #[serial_test::serial(bitnet_env)]
    fn detect_npu_strict_ignores_fake() {
        temp_env::with_var("BITNET_STRICT_MODE", Some("1"), || {
            temp_env::with_var("BITNET_NPU_FAKE", Some("1"), || {
                let caps = detect_npu();
                // Strict mode ignores fake — real detection returns unavailable on test machines
                assert!(!caps.available);
            });
        });
    }

    #[test]
    #[serial_test::serial(bitnet_env)]
    fn detect_npu_no_fake_env() {
        temp_env::with_var("BITNET_STRICT_MODE", None::<&str>, || {
            temp_env::with_var("BITNET_NPU_FAKE", None::<&str>, || {
                let caps = detect_npu();
                // No fake env → real detection → unavailable on test machines
                assert!(!caps.available);
            });
        });
    }
}
