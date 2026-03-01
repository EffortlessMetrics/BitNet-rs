//! Edge-case tests for GPU HAL backend selector.
//!
//! Tests BackendType, BackendPriority, BackendCapability, DType,
//! BackendScorer, WorkloadRequirements, BackendFallback, BackendConfig,
//! and BackendSelector — all without GPU hardware.

use bitnet_gpu_hal::backend_selector::*;
use std::time::Duration;

// ── BackendType ─────────────────────────────────────────────────────────────

#[test]
fn backend_type_all_has_8_variants() {
    assert_eq!(BackendType::all().len(), 8);
}

#[test]
fn backend_type_gpu_variants() {
    assert!(BackendType::CUDA.is_gpu());
    assert!(BackendType::OpenCL.is_gpu());
    assert!(BackendType::Vulkan.is_gpu());
    assert!(BackendType::Metal.is_gpu());
    assert!(BackendType::ROCm.is_gpu());
    assert!(BackendType::WebGPU.is_gpu());
    assert!(BackendType::LevelZero.is_gpu());
}

#[test]
fn backend_type_cpu_is_not_gpu() {
    assert!(!BackendType::CPU.is_gpu());
}

#[test]
fn backend_type_clone_eq() {
    let b = BackendType::CUDA;
    assert_eq!(b, b.clone());
}

#[test]
fn backend_type_debug() {
    let s = format!("{:?}", BackendType::Metal);
    assert!(s.contains("Metal"));
}

// ── DType ───────────────────────────────────────────────────────────────────

#[test]
fn dtype_all_variants_exist() {
    let _ = DType::F32;
    let _ = DType::F16;
    let _ = DType::BF16;
    let _ = DType::I8;
    let _ = DType::I2;
}

#[test]
fn dtype_clone_eq() {
    assert_eq!(DType::F32, DType::F32.clone());
    assert_ne!(DType::F32, DType::F16);
}

// ── BackendPriority ─────────────────────────────────────────────────────────

#[test]
fn backend_priority_order() {
    let p = BackendPriority::new(vec![BackendType::CUDA, BackendType::CPU]);
    assert_eq!(p.order().len(), 2);
    assert_eq!(p.order()[0], BackendType::CUDA);
}

#[test]
fn backend_priority_rank() {
    let p = BackendPriority::new(vec![BackendType::CUDA, BackendType::OpenCL, BackendType::CPU]);
    assert_eq!(p.rank(BackendType::CUDA), Some(0));
    assert_eq!(p.rank(BackendType::OpenCL), Some(1));
    assert_eq!(p.rank(BackendType::CPU), Some(2));
}

#[test]
fn backend_priority_rank_missing() {
    let p = BackendPriority::new(vec![BackendType::CPU]);
    assert_eq!(p.rank(BackendType::Metal), None);
}

#[test]
fn backend_priority_first() {
    let p = BackendPriority::new(vec![BackendType::Vulkan, BackendType::CPU]);
    assert_eq!(p.first(), Some(BackendType::Vulkan));
}

#[test]
fn backend_priority_empty() {
    let p = BackendPriority::new(vec![]);
    assert_eq!(p.first(), None);
    assert_eq!(p.rank(BackendType::CPU), None);
}

// ── BackendCapability ───────────────────────────────────────────────────────

#[test]
fn backend_capability_supports_dtype() {
    let cap = BackendCapability {
        backend: BackendType::CUDA,
        supported_dtypes: vec![DType::F32, DType::F16],
        max_buffer_bytes: 8_000_000_000,
        shared_memory_bytes: 49152,
        max_workgroup_size: 1024,
        supports_unified_memory: true,
        compute_units: 80,
        driver_version: "535.0".into(),
        device_name: "RTX 4090".into(),
    };
    assert!(cap.supports_dtype(DType::F32));
    assert!(cap.supports_dtype(DType::F16));
    assert!(!cap.supports_dtype(DType::I8));
}

#[test]
fn backend_capability_throughput_score() {
    let cap = BackendCapability {
        backend: BackendType::CUDA,
        supported_dtypes: vec![DType::F32],
        max_buffer_bytes: 8_000_000_000,
        shared_memory_bytes: 49152,
        max_workgroup_size: 1024,
        supports_unified_memory: false,
        compute_units: 80,
        driver_version: "535.0".into(),
        device_name: "GPU".into(),
    };
    let score = cap.throughput_score();
    assert!(score > 0.0);
}

#[test]
fn backend_capability_zero_compute_units() {
    let cap = BackendCapability {
        backend: BackendType::CPU,
        supported_dtypes: vec![DType::F32],
        max_buffer_bytes: 1024,
        shared_memory_bytes: 0,
        max_workgroup_size: 1,
        supports_unified_memory: false,
        compute_units: 0,
        driver_version: String::new(),
        device_name: "cpu".into(),
    };
    let score = cap.throughput_score();
    assert!(score >= 0.0);
}

// ── WorkloadRequirements ────────────────────────────────────────────────────

#[test]
fn workload_requirements_basic() {
    let reqs = WorkloadRequirements {
        required_dtypes: vec![DType::F32],
        min_buffer_bytes: 1024,
        min_shared_memory_bytes: 0,
        prefer_gpu: true,
    };
    assert!(reqs.prefer_gpu);
    assert_eq!(reqs.required_dtypes.len(), 1);
}

#[test]
fn workload_requirements_no_dtypes() {
    let reqs = WorkloadRequirements {
        required_dtypes: vec![],
        min_buffer_bytes: 0,
        min_shared_memory_bytes: 0,
        prefer_gpu: false,
    };
    assert!(!reqs.prefer_gpu);
}

// ── BackendFallback ─────────────────────────────────────────────────────────

#[test]
fn backend_fallback_chain() {
    let fb =
        BackendFallback::new(vec![BackendType::CUDA, BackendType::OpenCL, BackendType::CPU], 3);
    assert_eq!(fb.chain().len(), 3);
    assert_eq!(fb.max_retries(), 3);
    assert!(!fb.is_empty());
    assert_eq!(fb.len(), 3);
}

#[test]
fn backend_fallback_empty() {
    let fb = BackendFallback::new(vec![], 0);
    assert!(fb.is_empty());
    assert_eq!(fb.len(), 0);
}

#[test]
fn backend_fallback_try_fallback_success() {
    let fb =
        BackendFallback::new(vec![BackendType::CUDA, BackendType::OpenCL, BackendType::CPU], 3);
    let result = fb.try_fallback(|b| b == BackendType::OpenCL);
    assert_eq!(result, Some(BackendType::OpenCL));
}

#[test]
fn backend_fallback_try_fallback_all_fail() {
    let fb = BackendFallback::new(vec![BackendType::CUDA, BackendType::Metal], 2);
    let result = fb.try_fallback(|_| false);
    assert_eq!(result, None);
}

// ── BackendConfig ───────────────────────────────────────────────────────────

#[test]
fn backend_config_validate_default() {
    let config = BackendConfig {
        forced_backend: None,
        excluded_backends: vec![],
        probe_timeout: Duration::from_secs(5),
        fallback: BackendFallback::new(vec![BackendType::CPU], 1),
        allow_cpu_fallback: true,
        extra: Default::default(),
    };
    assert!(config.validate().is_ok());
}

#[test]
fn backend_config_forced_backend() {
    let config = BackendConfig {
        forced_backend: Some(BackendType::CUDA),
        excluded_backends: vec![],
        probe_timeout: Duration::from_secs(5),
        fallback: BackendFallback::new(vec![], 0),
        allow_cpu_fallback: false,
        extra: Default::default(),
    };
    assert_eq!(config.forced_backend, Some(BackendType::CUDA));
}

#[test]
fn backend_config_excluded_backends() {
    let config = BackendConfig {
        forced_backend: None,
        excluded_backends: vec![BackendType::Vulkan, BackendType::WebGPU],
        probe_timeout: Duration::from_secs(5),
        fallback: BackendFallback::new(vec![], 0),
        allow_cpu_fallback: true,
        extra: Default::default(),
    };
    assert_eq!(config.excluded_backends.len(), 2);
}

// ── BackendScore ────────────────────────────────────────────────────────────

#[test]
fn backend_score_fields() {
    let score = BackendScore {
        backend: BackendType::CUDA,
        total: 95.0,
        dtype_match: 1.0,
        memory_match: 1.0,
        throughput: 0.9,
        priority_bonus: 0.5,
        gpu_bonus: 0.3,
        meets_requirements: true,
    };
    assert!(score.meets_requirements);
    assert_eq!(score.backend, BackendType::CUDA);
}

#[test]
fn backend_score_zero() {
    let score = BackendScore {
        backend: BackendType::CPU,
        total: 0.0,
        dtype_match: 0.0,
        memory_match: 0.0,
        throughput: 0.0,
        priority_bonus: 0.0,
        gpu_bonus: 0.0,
        meets_requirements: false,
    };
    assert!(!score.meets_requirements);
    assert_eq!(score.total, 0.0);
}

// ── SelectionResult ─────────────────────────────────────────────────────────

#[test]
fn selection_result_fields() {
    let r = SelectionResult {
        selected: BackendType::CUDA,
        score: 90.0,
        reason: "best match".into(),
        alternatives: vec![(BackendType::CPU, 50.0)],
    };
    assert_eq!(r.selected, BackendType::CUDA);
    assert_eq!(r.alternatives.len(), 1);
}

// ── BackendScorer ───────────────────────────────────────────────────────────

#[test]
fn backend_scorer_score_matching_capability() {
    let scorer = BackendScorer::default();
    let cap = BackendCapability {
        backend: BackendType::CUDA,
        supported_dtypes: vec![DType::F32, DType::F16],
        max_buffer_bytes: 8_000_000_000,
        shared_memory_bytes: 49152,
        max_workgroup_size: 1024,
        supports_unified_memory: false,
        compute_units: 80,
        driver_version: "535".into(),
        device_name: "GPU".into(),
    };
    let reqs = WorkloadRequirements {
        required_dtypes: vec![DType::F32],
        min_buffer_bytes: 1024,
        min_shared_memory_bytes: 0,
        prefer_gpu: true,
    };
    let priority = BackendPriority::new(vec![BackendType::CUDA, BackendType::CPU]);
    let score = scorer.score(&cap, &reqs, &priority);
    assert!(score.meets_requirements);
    assert!(score.total > 0.0);
}

#[test]
fn backend_scorer_score_insufficient_memory() {
    let scorer = BackendScorer::default();
    let cap = BackendCapability {
        backend: BackendType::CPU,
        supported_dtypes: vec![DType::F32],
        max_buffer_bytes: 100,
        shared_memory_bytes: 0,
        max_workgroup_size: 1,
        supports_unified_memory: false,
        compute_units: 1,
        driver_version: String::new(),
        device_name: "cpu".into(),
    };
    let reqs = WorkloadRequirements {
        required_dtypes: vec![DType::F32],
        min_buffer_bytes: 1_000_000_000,
        min_shared_memory_bytes: 0,
        prefer_gpu: false,
    };
    let priority = BackendPriority::new(vec![BackendType::CPU]);
    let score = scorer.score(&cap, &reqs, &priority);
    assert!(!score.meets_requirements);
}

#[test]
fn backend_scorer_score_all() {
    let scorer = BackendScorer::default();
    let caps = vec![
        BackendCapability {
            backend: BackendType::CUDA,
            supported_dtypes: vec![DType::F32],
            max_buffer_bytes: 8_000_000_000,
            shared_memory_bytes: 49152,
            max_workgroup_size: 1024,
            supports_unified_memory: false,
            compute_units: 80,
            driver_version: "535".into(),
            device_name: "GPU".into(),
        },
        BackendCapability {
            backend: BackendType::CPU,
            supported_dtypes: vec![DType::F32],
            max_buffer_bytes: 16_000_000_000,
            shared_memory_bytes: 0,
            max_workgroup_size: 1,
            supports_unified_memory: false,
            compute_units: 8,
            driver_version: String::new(),
            device_name: "cpu".into(),
        },
    ];
    let reqs = WorkloadRequirements {
        required_dtypes: vec![DType::F32],
        min_buffer_bytes: 1024,
        min_shared_memory_bytes: 0,
        prefer_gpu: true,
    };
    let priority = BackendPriority::new(vec![BackendType::CUDA, BackendType::CPU]);
    let scores = scorer.score_all(&caps, &reqs, &priority);
    assert_eq!(scores.len(), 2);
}

// ── ManagerState / ManagerEventKind ─────────────────────────────────────────

#[test]
fn manager_state_all_variants() {
    let _ = ManagerState::Idle;
    let _ = ManagerState::Selecting;
    let _ = ManagerState::Initializing;
    let _ = ManagerState::Running;
    let _ = ManagerState::FallingBack;
    let _ = ManagerState::ShuttingDown;
    let _ = ManagerState::Stopped;
}

#[test]
fn manager_event_kind_all_variants() {
    let _ = ManagerEventKind::ProbeStarted;
    let _ = ManagerEventKind::ProbeCompleted;
    let _ = ManagerEventKind::SelectionMade;
    let _ = ManagerEventKind::InitStarted;
    let _ = ManagerEventKind::InitCompleted;
    let _ = ManagerEventKind::FallbackTriggered;
    let _ = ManagerEventKind::Shutdown;
    let _ = ManagerEventKind::Error;
}
