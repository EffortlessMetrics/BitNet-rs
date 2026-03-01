//! SIMD-OpenCL compatibility and dispatch layer.
//!
//! Provides unified interfaces for operations that can run on either
//! CPU SIMD or OpenCL GPU, with automatic backend selection based on
//! problem size, hardware availability, and user configuration.

use std::fmt;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// Compute backend variants available for dispatch.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ComputeBackend {
    CpuScalar,
    CpuSimdAvx2,
    CpuSimdAvx512,
    CpuSimdNeon,
    OpenClGpu,
    AutoSelect,
}

impl fmt::Display for ComputeBackend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CpuScalar => write!(f, "CpuScalar"),
            Self::CpuSimdAvx2 => write!(f, "CpuSimdAvx2"),
            Self::CpuSimdAvx512 => write!(f, "CpuSimdAvx512"),
            Self::CpuSimdNeon => write!(f, "CpuSimdNeon"),
            Self::OpenClGpu => write!(f, "OpenClGpu"),
            Self::AutoSelect => write!(f, "AutoSelect"),
        }
    }
}

/// Describes the capabilities of a detected backend.
#[derive(Debug, Clone)]
pub struct BackendCapability {
    pub name: String,
    pub backend: ComputeBackend,
    pub fp32_supported: bool,
    pub fp16_supported: bool,
    pub int8_supported: bool,
    pub estimated_gflops: f32,
}

/// Result of the dispatch decision process.
#[derive(Debug, Clone)]
pub struct DispatchDecision {
    pub selected_backend: ComputeBackend,
    pub reason: String,
    pub estimated_speedup: f32,
}

/// Configuration for the bridge layer.
#[derive(Debug, Clone)]
pub struct BridgeConfig {
    pub prefer_gpu: bool,
    pub min_problem_size_for_gpu: usize,
    pub fallback_on_error: bool,
    pub log_decisions: bool,
}

impl Default for BridgeConfig {
    fn default() -> Self {
        Self {
            prefer_gpu: true,
            min_problem_size_for_gpu: 128,
            fallback_on_error: true,
            log_decisions: false,
        }
    }
}

/// Errors that can occur during bridge dispatch.
#[derive(Debug, Clone, PartialEq)]
pub enum BridgeError {
    NoBackendAvailable,
    BackendFailed(ComputeBackend, String),
    SizeBelowThreshold,
}

impl fmt::Display for BridgeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoBackendAvailable => {
                write!(f, "no compute backend available")
            }
            Self::BackendFailed(b, msg) => {
                write!(f, "backend {b} failed: {msg}")
            }
            Self::SizeBelowThreshold => {
                write!(f, "problem size below GPU threshold")
            }
        }
    }
}

impl std::error::Error for BridgeError {}

// ---------------------------------------------------------------------------
// Backend detection
// ---------------------------------------------------------------------------

/// Enumerate available backends.
///
/// In a real deployment this would probe CPUID and OpenCL drivers.
/// For testing purposes the function returns a mock set covering all
/// backend types so that every code-path is exercisable.
pub fn detect_available_backends() -> Vec<BackendCapability> {
    vec![
        BackendCapability {
            name: "Scalar (reference)".into(),
            backend: ComputeBackend::CpuScalar,
            fp32_supported: true,
            fp16_supported: false,
            int8_supported: true,
            estimated_gflops: 2.0,
        },
        BackendCapability {
            name: "AVX2".into(),
            backend: ComputeBackend::CpuSimdAvx2,
            fp32_supported: true,
            fp16_supported: false,
            int8_supported: true,
            estimated_gflops: 50.0,
        },
        BackendCapability {
            name: "AVX-512".into(),
            backend: ComputeBackend::CpuSimdAvx512,
            fp32_supported: true,
            fp16_supported: true,
            int8_supported: true,
            estimated_gflops: 100.0,
        },
        BackendCapability {
            name: "NEON".into(),
            backend: ComputeBackend::CpuSimdNeon,
            fp32_supported: true,
            fp16_supported: true,
            int8_supported: true,
            estimated_gflops: 40.0,
        },
        BackendCapability {
            name: "OpenCL GPU".into(),
            backend: ComputeBackend::OpenClGpu,
            fp32_supported: true,
            fp16_supported: true,
            int8_supported: true,
            estimated_gflops: 500.0,
        },
    ]
}

// ---------------------------------------------------------------------------
// Backend selection
// ---------------------------------------------------------------------------

/// Select the best backend given available capabilities and config.
pub fn select_backend(
    backends: &[BackendCapability],
    config: &BridgeConfig,
    problem_size: usize,
) -> Result<DispatchDecision, BridgeError> {
    if backends.is_empty() {
        return Err(BridgeError::NoBackendAvailable);
    }

    // When GPU is not preferred, filter to CPU-only backends.
    let candidates: Vec<&BackendCapability> = if config.prefer_gpu {
        backends.iter().collect()
    } else {
        backends
            .iter()
            .filter(|b| b.backend != ComputeBackend::OpenClGpu)
            .collect()
    };

    if candidates.is_empty() {
        return Err(BridgeError::NoBackendAvailable);
    }

    // If GPU is preferred and available, check threshold.
    if config.prefer_gpu
        && problem_size >= config.min_problem_size_for_gpu
        && let Some(gpu) = candidates
            .iter()
            .find(|b| b.backend == ComputeBackend::OpenClGpu)
    {
        let speedup = estimate_gpu_advantage("matmul", problem_size);
        return Ok(DispatchDecision {
            selected_backend: gpu.backend,
            reason: format!(
                "GPU preferred; problem_size={problem_size} \
                 >= threshold={}",
                config.min_problem_size_for_gpu
            ),
            estimated_speedup: speedup,
        });
    }

    // Pick the CPU backend with the highest estimated GFLOPS.
    let best = candidates
        .iter()
        .filter(|b| b.backend != ComputeBackend::OpenClGpu)
        .max_by(|a, b| {
            a.estimated_gflops
                .partial_cmp(&b.estimated_gflops)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

    match best {
        Some(b) => Ok(DispatchDecision {
            selected_backend: b.backend,
            reason: format!(
                "best CPU backend ({}) at {:.0} GFLOPS",
                b.name, b.estimated_gflops
            ),
            estimated_speedup: 1.0,
        }),
        None => Err(BridgeError::NoBackendAvailable),
    }
}

// ---------------------------------------------------------------------------
// CPU reference implementations (all backends share the same logic today)
// ---------------------------------------------------------------------------

/// Reference matrix-multiply: C[m×n] = A[m×k] · B[k×n] (row-major).
pub fn cpu_dispatch_matmul(
    a: &[f32],
    b: &[f32],
    m: usize,
    n: usize,
    k: usize,
    _backend: ComputeBackend,
) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for p in 0..k {
                sum += a[i * k + p] * b[p * n + j];
            }
            c[i * n + j] = sum;
        }
    }
    c
}

/// Reference softmax (numerically stable).
pub fn cpu_dispatch_softmax(
    input: &[f32],
    _backend: ComputeBackend,
) -> Vec<f32> {
    if input.is_empty() {
        return vec![];
    }
    let max = input.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = input.iter().map(|&x| (x - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    if sum == 0.0 {
        return exps;
    }
    exps.iter().map(|&e| e / sum).collect()
}

/// Reference RMSNorm: out_i = (x_i / rms) * w_i.
pub fn cpu_dispatch_rmsnorm(
    input: &[f32],
    weight: &[f32],
    eps: f32,
    _backend: ComputeBackend,
) -> Vec<f32> {
    if input.is_empty() {
        return vec![];
    }
    let n = input.len();
    let mean_sq: f32 = input.iter().map(|&x| x * x).sum::<f32>() / n as f32;
    let rms = (mean_sq + eps).sqrt();
    input
        .iter()
        .zip(weight.iter())
        .map(|(&x, &w)| (x / rms) * w)
        .collect()
}

// ---------------------------------------------------------------------------
// GPU advantage heuristics
// ---------------------------------------------------------------------------

/// Estimate the speed-up factor of GPU over CPU for a given operation and
/// problem size.  For `"matmul"` the crossover is around N=128; for
/// element-wise operations it is around N=4096.
pub fn estimate_gpu_advantage(operation: &str, problem_size: usize) -> f32 {
    let threshold: usize = match operation {
        "matmul" => 128,
        _ => 4096, // element-wise / softmax / rmsnorm
    };

    if problem_size <= threshold {
        // Below crossover: GPU is slower due to launch overhead.
        return 0.5_f32.max(problem_size as f32 / threshold as f32);
    }

    // Above crossover the advantage grows logarithmically.
    1.0 + (problem_size as f32 / threshold as f32).ln()
}

/// Decide whether to use the GPU for the given workload.
pub fn should_use_gpu(
    operation: &str,
    problem_size: usize,
    config: &BridgeConfig,
) -> bool {
    if !config.prefer_gpu {
        return false;
    }
    if problem_size < config.min_problem_size_for_gpu {
        return false;
    }
    estimate_gpu_advantage(operation, problem_size) > 1.0
}

// ---------------------------------------------------------------------------
// Fallback chain
// ---------------------------------------------------------------------------

/// Return an ordered fallback chain from the supplied list following
/// priority: OpenCL → AVX-512 → AVX2 → NEON → Scalar.
pub fn cpu_fallback_chain(
    backends: &[ComputeBackend],
) -> Vec<ComputeBackend> {
    let priority_order = [
        ComputeBackend::OpenClGpu,
        ComputeBackend::CpuSimdAvx512,
        ComputeBackend::CpuSimdAvx2,
        ComputeBackend::CpuSimdNeon,
        ComputeBackend::CpuScalar,
    ];
    priority_order
        .iter()
        .filter(|p| backends.contains(p))
        .copied()
        .collect()
}

// ---------------------------------------------------------------------------
// Logging helper
// ---------------------------------------------------------------------------

/// Format a human-readable dispatch log line.
pub fn format_dispatch_log(decision: &DispatchDecision) -> String {
    format!(
        "[dispatch] backend={} speedup={:.2}x reason=\"{}\"",
        decision.selected_backend,
        decision.estimated_speedup,
        decision.reason,
    )
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- Backend detection ---------------------------------------------------

    #[test]
    fn detect_backends_includes_cpu_scalar() {
        let backends = detect_available_backends();
        assert!(backends
            .iter()
            .any(|b| b.backend == ComputeBackend::CpuScalar));
    }

    #[test]
    fn detect_backends_non_empty() {
        assert!(!detect_available_backends().is_empty());
    }

    #[test]
    fn detect_backends_all_support_fp32() {
        for b in detect_available_backends() {
            assert!(b.fp32_supported, "{} missing fp32", b.name);
        }
    }

    #[test]
    fn detect_backends_gpu_has_highest_gflops() {
        let backends = detect_available_backends();
        let gpu = backends
            .iter()
            .find(|b| b.backend == ComputeBackend::OpenClGpu)
            .unwrap();
        for b in &backends {
            if b.backend != ComputeBackend::OpenClGpu {
                assert!(gpu.estimated_gflops >= b.estimated_gflops);
            }
        }
    }

    // -- Backend selection ---------------------------------------------------

    #[test]
    fn select_gpu_for_large_problem() {
        let backends = detect_available_backends();
        let config = BridgeConfig { prefer_gpu: true, ..Default::default() };
        let decision =
            select_backend(&backends, &config, 1024).unwrap();
        assert_eq!(
            decision.selected_backend,
            ComputeBackend::OpenClGpu
        );
    }

    #[test]
    fn select_cpu_for_small_problem() {
        let backends = detect_available_backends();
        let config = BridgeConfig {
            prefer_gpu: true,
            min_problem_size_for_gpu: 2048,
            ..Default::default()
        };
        let decision =
            select_backend(&backends, &config, 64).unwrap();
        assert_ne!(
            decision.selected_backend,
            ComputeBackend::OpenClGpu
        );
    }

    #[test]
    fn select_cpu_when_gpu_not_preferred() {
        let backends = detect_available_backends();
        let config =
            BridgeConfig { prefer_gpu: false, ..Default::default() };
        let decision =
            select_backend(&backends, &config, 100_000).unwrap();
        assert_ne!(
            decision.selected_backend,
            ComputeBackend::OpenClGpu
        );
    }

    #[test]
    fn select_returns_error_when_empty() {
        let config = BridgeConfig::default();
        let err = select_backend(&[], &config, 128).unwrap_err();
        assert_eq!(err, BridgeError::NoBackendAvailable);
    }

    #[test]
    fn select_best_cpu_picks_highest_gflops() {
        let backends = vec![
            BackendCapability {
                name: "slow".into(),
                backend: ComputeBackend::CpuScalar,
                fp32_supported: true,
                fp16_supported: false,
                int8_supported: false,
                estimated_gflops: 1.0,
            },
            BackendCapability {
                name: "fast".into(),
                backend: ComputeBackend::CpuSimdAvx512,
                fp32_supported: true,
                fp16_supported: true,
                int8_supported: true,
                estimated_gflops: 100.0,
            },
        ];
        let config =
            BridgeConfig { prefer_gpu: false, ..Default::default() };
        let d = select_backend(&backends, &config, 64).unwrap();
        assert_eq!(d.selected_backend, ComputeBackend::CpuSimdAvx512);
    }

    #[test]
    fn select_auto_picks_best_available() {
        let backends = detect_available_backends();
        let config = BridgeConfig {
            prefer_gpu: true,
            min_problem_size_for_gpu: 256,
            ..Default::default()
        };
        let decision =
            select_backend(&backends, &config, 512).unwrap();
        // With all mocked backends and large problem, GPU is best.
        assert_eq!(
            decision.selected_backend,
            ComputeBackend::OpenClGpu
        );
    }

    #[test]
    fn select_min_threshold_boundary() {
        let backends = detect_available_backends();
        let config = BridgeConfig {
            prefer_gpu: true,
            min_problem_size_for_gpu: 256,
            ..Default::default()
        };
        // Exactly at threshold → GPU should be selected.
        let at = select_backend(&backends, &config, 256).unwrap();
        assert_eq!(at.selected_backend, ComputeBackend::OpenClGpu);
        // Below threshold → CPU.
        let below = select_backend(&backends, &config, 255).unwrap();
        assert_ne!(below.selected_backend, ComputeBackend::OpenClGpu);
    }

    // -- Matmul dispatch -----------------------------------------------------

    #[test]
    fn matmul_scalar_correct() {
        let a = vec![1.0, 2.0, 3.0, 4.0]; // 2×2
        let b = vec![5.0, 6.0, 7.0, 8.0];
        let c =
            cpu_dispatch_matmul(&a, &b, 2, 2, 2, ComputeBackend::CpuScalar);
        assert_eq!(c, vec![19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn matmul_all_backends_same_result() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2×3
        let b = vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]; // 3×2
        let expected =
            cpu_dispatch_matmul(&a, &b, 2, 2, 3, ComputeBackend::CpuScalar);
        for backend in [
            ComputeBackend::CpuSimdAvx2,
            ComputeBackend::CpuSimdAvx512,
            ComputeBackend::CpuSimdNeon,
            ComputeBackend::OpenClGpu,
        ] {
            let result =
                cpu_dispatch_matmul(&a, &b, 2, 2, 3, backend);
            assert_eq!(result, expected, "mismatch for {backend:?}");
        }
    }

    #[test]
    fn matmul_identity() {
        // I × A == A
        let a = vec![1.0, 0.0, 0.0, 1.0];
        let b = vec![3.0, 7.0, 2.0, 5.0];
        let c =
            cpu_dispatch_matmul(&a, &b, 2, 2, 2, ComputeBackend::CpuScalar);
        assert_eq!(c, b);
    }

    #[test]
    fn matmul_zero_size() {
        let c = cpu_dispatch_matmul(
            &[],
            &[],
            0,
            0,
            0,
            ComputeBackend::CpuScalar,
        );
        assert!(c.is_empty());
    }

    // -- Softmax dispatch ----------------------------------------------------

    #[test]
    fn softmax_sums_to_one() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let out =
            cpu_dispatch_softmax(&input, ComputeBackend::CpuScalar);
        let sum: f32 = out.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn softmax_monotonic() {
        let input = vec![1.0, 2.0, 3.0];
        let out =
            cpu_dispatch_softmax(&input, ComputeBackend::CpuScalar);
        assert!(out[0] < out[1] && out[1] < out[2]);
    }

    #[test]
    fn softmax_all_backends_same() {
        let input = vec![0.5, 1.5, -0.5, 2.0];
        let expected =
            cpu_dispatch_softmax(&input, ComputeBackend::CpuScalar);
        for backend in [
            ComputeBackend::CpuSimdAvx2,
            ComputeBackend::OpenClGpu,
        ] {
            let result = cpu_dispatch_softmax(&input, backend);
            for (a, b) in result.iter().zip(expected.iter()) {
                assert!((a - b).abs() < 1e-6, "mismatch for {backend:?}");
            }
        }
    }

    #[test]
    fn softmax_empty_input() {
        let out =
            cpu_dispatch_softmax(&[], ComputeBackend::CpuScalar);
        assert!(out.is_empty());
    }

    #[test]
    fn softmax_single_element() {
        let out =
            cpu_dispatch_softmax(&[42.0], ComputeBackend::CpuScalar);
        assert_eq!(out.len(), 1);
        assert!((out[0] - 1.0).abs() < 1e-6);
    }

    // -- RMSNorm dispatch ----------------------------------------------------

    #[test]
    fn rmsnorm_basic() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let weight = vec![1.0; 4];
        let out = cpu_dispatch_rmsnorm(
            &input,
            &weight,
            1e-5,
            ComputeBackend::CpuScalar,
        );
        assert_eq!(out.len(), 4);
        // RMS of [1,2,3,4] ≈ 2.7386
        let rms = (7.5_f32 + 1e-5).sqrt();
        for (i, &v) in out.iter().enumerate() {
            let expected = input[i] / rms;
            assert!(
                (v - expected).abs() < 1e-4,
                "elem {i}: {v} vs {expected}"
            );
        }
    }

    #[test]
    fn rmsnorm_all_backends_same() {
        let input = vec![1.0, -1.0, 0.5];
        let weight = vec![2.0, 0.5, 1.0];
        let expected = cpu_dispatch_rmsnorm(
            &input,
            &weight,
            1e-6,
            ComputeBackend::CpuScalar,
        );
        for backend in [
            ComputeBackend::CpuSimdAvx2,
            ComputeBackend::CpuSimdAvx512,
            ComputeBackend::OpenClGpu,
        ] {
            let result = cpu_dispatch_rmsnorm(
                &input, &weight, 1e-6, backend,
            );
            for (a, b) in result.iter().zip(expected.iter()) {
                assert!(
                    (a - b).abs() < 1e-6,
                    "mismatch for {backend:?}"
                );
            }
        }
    }

    #[test]
    fn rmsnorm_empty_input() {
        let out = cpu_dispatch_rmsnorm(
            &[],
            &[],
            1e-5,
            ComputeBackend::CpuScalar,
        );
        assert!(out.is_empty());
    }

    #[test]
    fn rmsnorm_single_element() {
        let out = cpu_dispatch_rmsnorm(
            &[3.0],
            &[2.0],
            1e-5,
            ComputeBackend::CpuScalar,
        );
        assert_eq!(out.len(), 1);
        let rms = (9.0_f32 + 1e-5).sqrt();
        let expected = (3.0 / rms) * 2.0;
        assert!((out[0] - expected).abs() < 1e-4);
    }

    // -- GPU advantage estimation --------------------------------------------

    #[test]
    fn gpu_advantage_increases_with_size() {
        let small = estimate_gpu_advantage("matmul", 64);
        let large = estimate_gpu_advantage("matmul", 4096);
        assert!(large > small);
    }

    #[test]
    fn gpu_advantage_below_threshold_less_than_one() {
        let adv = estimate_gpu_advantage("matmul", 32);
        assert!(adv <= 1.0);
    }

    #[test]
    fn gpu_advantage_elementwise_higher_threshold() {
        // At N=256, matmul has advantage >1 but elementwise does not.
        let mat = estimate_gpu_advantage("matmul", 256);
        let elem = estimate_gpu_advantage("softmax", 256);
        assert!(mat > elem);
    }

    // -- should_use_gpu ------------------------------------------------------

    #[test]
    fn should_use_gpu_large_matmul() {
        let config = BridgeConfig::default();
        assert!(should_use_gpu("matmul", 1024, &config));
    }

    #[test]
    fn should_use_gpu_false_when_disabled() {
        let config =
            BridgeConfig { prefer_gpu: false, ..Default::default() };
        assert!(!should_use_gpu("matmul", 100_000, &config));
    }

    #[test]
    fn should_use_gpu_false_below_threshold() {
        let config = BridgeConfig {
            prefer_gpu: true,
            min_problem_size_for_gpu: 512,
            ..Default::default()
        };
        assert!(!should_use_gpu("matmul", 256, &config));
    }

    // -- Fallback chain ------------------------------------------------------

    #[test]
    fn fallback_chain_correct_order() {
        let all = vec![
            ComputeBackend::CpuScalar,
            ComputeBackend::CpuSimdNeon,
            ComputeBackend::CpuSimdAvx2,
            ComputeBackend::CpuSimdAvx512,
            ComputeBackend::OpenClGpu,
        ];
        let chain = cpu_fallback_chain(&all);
        assert_eq!(chain[0], ComputeBackend::OpenClGpu);
        assert_eq!(chain[1], ComputeBackend::CpuSimdAvx512);
        assert_eq!(chain[2], ComputeBackend::CpuSimdAvx2);
        assert_eq!(chain[3], ComputeBackend::CpuSimdNeon);
        assert_eq!(chain[4], ComputeBackend::CpuScalar);
    }

    #[test]
    fn fallback_chain_subset() {
        let subset = vec![
            ComputeBackend::CpuScalar,
            ComputeBackend::CpuSimdAvx2,
        ];
        let chain = cpu_fallback_chain(&subset);
        assert_eq!(chain, vec![
            ComputeBackend::CpuSimdAvx2,
            ComputeBackend::CpuScalar,
        ]);
    }

    #[test]
    fn fallback_chain_empty_input() {
        assert!(cpu_fallback_chain(&[]).is_empty());
    }

    // -- Dispatch log --------------------------------------------------------

    #[test]
    fn dispatch_log_contains_backend_name() {
        let d = DispatchDecision {
            selected_backend: ComputeBackend::CpuSimdAvx2,
            reason: "fastest CPU".into(),
            estimated_speedup: 1.5,
        };
        let log = format_dispatch_log(&d);
        assert!(log.contains("CpuSimdAvx2"));
    }

    #[test]
    fn dispatch_log_contains_reason() {
        let d = DispatchDecision {
            selected_backend: ComputeBackend::OpenClGpu,
            reason: "GPU preferred".into(),
            estimated_speedup: 3.0,
        };
        let log = format_dispatch_log(&d);
        assert!(log.contains("GPU preferred"));
    }

    #[test]
    fn dispatch_log_contains_speedup() {
        let d = DispatchDecision {
            selected_backend: ComputeBackend::CpuScalar,
            reason: "only option".into(),
            estimated_speedup: 1.0,
        };
        let log = format_dispatch_log(&d);
        assert!(log.contains("1.00x"));
    }

    // -- Determinism & properties -------------------------------------------

    #[test]
    fn dispatch_is_deterministic() {
        let backends = detect_available_backends();
        let config = BridgeConfig::default();
        let d1 = select_backend(&backends, &config, 512).unwrap();
        let d2 = select_backend(&backends, &config, 512).unwrap();
        assert_eq!(d1.selected_backend, d2.selected_backend);
        assert!((d1.estimated_speedup - d2.estimated_speedup).abs() < 1e-9);
    }

    #[test]
    fn all_dispatched_ops_produce_valid_output() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        for backend in [
            ComputeBackend::CpuScalar,
            ComputeBackend::CpuSimdAvx2,
            ComputeBackend::OpenClGpu,
        ] {
            let sm = cpu_dispatch_softmax(&input, backend);
            assert!(sm.iter().all(|v| v.is_finite()));

            let rms = cpu_dispatch_rmsnorm(
                &input,
                &[1.0; 4],
                1e-5,
                backend,
            );
            assert!(rms.iter().all(|v| v.is_finite()));
        }
    }

    // -- Edge cases ----------------------------------------------------------

    #[test]
    fn matmul_single_element() {
        let c = cpu_dispatch_matmul(
            &[3.0],
            &[4.0],
            1,
            1,
            1,
            ComputeBackend::CpuScalar,
        );
        assert_eq!(c, vec![12.0]);
    }

    // -- BridgeError display -------------------------------------------------

    #[test]
    fn bridge_error_display() {
        let e = BridgeError::NoBackendAvailable;
        assert!(e.to_string().contains("no compute backend"));

        let e2 = BridgeError::BackendFailed(
            ComputeBackend::OpenClGpu,
            "timeout".into(),
        );
        assert!(e2.to_string().contains("OpenClGpu"));
        assert!(e2.to_string().contains("timeout"));

        let e3 = BridgeError::SizeBelowThreshold;
        assert!(e3.to_string().contains("below"));
    }

    // -- ComputeBackend Display ---------------------------------------------

    #[test]
    fn compute_backend_display() {
        assert_eq!(ComputeBackend::CpuScalar.to_string(), "CpuScalar");
        assert_eq!(
            ComputeBackend::OpenClGpu.to_string(),
            "OpenClGpu"
        );
    }

    // -- BridgeConfig default -----------------------------------------------

    #[test]
    fn bridge_config_default_prefers_gpu() {
        let cfg = BridgeConfig::default();
        assert!(cfg.prefer_gpu);
        assert!(cfg.fallback_on_error);
    }
}
