//! Tests for the multi-backend GPU dispatcher and backend registry.

use bitnet_opencl::{
    BackendDispatcher, BackendInfo, BackendProvider, BackendRegistry, BackendStatus, DispatchError,
    DispatchStrategy, Operation,
};

// ── Test helpers ─────────────────────────────────────────────────────────────

/// Configurable mock backend for testing dispatch logic.
struct MockBackend {
    name: String,
    status: BackendStatus,
    ops: Vec<Operation>,
    priority: u32,
}

impl MockBackend {
    fn available(name: &str, priority: u32, ops: Vec<Operation>) -> Box<Self> {
        Box::new(Self { name: name.to_owned(), status: BackendStatus::Available, ops, priority })
    }

    fn unavailable(name: &str, reason: &str) -> Box<Self> {
        Box::new(Self {
            name: name.to_owned(),
            status: BackendStatus::Unavailable(reason.to_owned()),
            ops: vec![],
            priority: 0,
        })
    }

    fn degraded(name: &str, reason: &str, priority: u32, ops: Vec<Operation>) -> Box<Self> {
        Box::new(Self {
            name: name.to_owned(),
            status: BackendStatus::Degraded(reason.to_owned()),
            ops,
            priority,
        })
    }
}

impl BackendProvider for MockBackend {
    fn name(&self) -> &str {
        &self.name
    }

    fn status(&self) -> BackendStatus {
        self.status.clone()
    }

    fn capabilities(&self) -> Vec<Operation> {
        self.ops.clone()
    }

    fn priority_score(&self) -> u32 {
        self.priority
    }
}

/// All common GPU operations.
fn all_ops() -> Vec<Operation> {
    vec![
        Operation::MatMul,
        Operation::Quantize,
        Operation::Dequantize,
        Operation::Softmax,
        Operation::LayerNorm,
        Operation::Attention,
        Operation::RoPE,
        Operation::Sampling,
    ]
}

/// Build a registry with the standard CUDA > OpenCL > Vulkan > CPU
/// hierarchy.
fn standard_registry() -> BackendRegistry {
    let mut reg = BackendRegistry::new();
    reg.register("cuda", MockBackend::available("cuda", 100, all_ops()));
    reg.register("opencl", MockBackend::available("opencl", 80, all_ops()));
    reg.register("vulkan", MockBackend::available("vulkan", 60, all_ops()));
    reg.register("cpu", MockBackend::available("cpu", 10, all_ops()));
    reg
}

// ── Priority dispatch tests ──────────────────────────────────────────────────

#[test]
fn priority_dispatch_selects_highest_priority() {
    let dispatcher = BackendDispatcher::new(standard_registry(), DispatchStrategy::Priority);
    let d = dispatcher.dispatch(Operation::MatMul).unwrap();
    assert_eq!(d.chosen_backend, "cuda");
    assert!(!d.alternatives_available.is_empty());
}

#[test]
fn priority_dispatch_skips_unavailable_backend() {
    let mut reg = BackendRegistry::new();
    reg.register("cuda", MockBackend::unavailable("cuda", "driver missing"));
    reg.register("opencl", MockBackend::available("opencl", 80, all_ops()));
    reg.register("cpu", MockBackend::available("cpu", 10, all_ops()));

    let dispatcher = BackendDispatcher::new(reg, DispatchStrategy::Priority);
    let d = dispatcher.dispatch(Operation::MatMul).unwrap();
    assert_eq!(d.chosen_backend, "opencl");
}

#[test]
fn priority_dispatch_with_degraded_backend() {
    let mut reg = BackendRegistry::new();
    reg.register("cuda", MockBackend::degraded("cuda", "thermal throttled", 100, all_ops()));
    reg.register("cpu", MockBackend::available("cpu", 10, all_ops()));

    let dispatcher = BackendDispatcher::new(reg, DispatchStrategy::Priority);
    // Degraded is still usable, so CUDA should be selected.
    let d = dispatcher.dispatch(Operation::MatMul).unwrap();
    assert_eq!(d.chosen_backend, "cuda");
}

#[test]
fn priority_lists_alternatives() {
    let dispatcher = BackendDispatcher::new(standard_registry(), DispatchStrategy::Priority);
    let d = dispatcher.dispatch(Operation::Softmax).unwrap();
    assert_eq!(d.chosen_backend, "cuda");
    assert!(d.alternatives_available.len() >= 3);
}

// ── Round-robin tests ────────────────────────────────────────────────────────

#[test]
fn round_robin_distributes_evenly() {
    let dispatcher = BackendDispatcher::new(standard_registry(), DispatchStrategy::RoundRobin);

    let mut seen = std::collections::HashMap::new();
    for _ in 0..40 {
        let d = dispatcher.dispatch(Operation::MatMul).unwrap();
        *seen.entry(d.chosen_backend).or_insert(0u32) += 1;
    }

    assert_eq!(seen.len(), 4, "all 4 backends should appear");
    for count in seen.values() {
        assert_eq!(*count, 10);
    }
}

#[test]
fn round_robin_skips_unavailable() {
    let mut reg = BackendRegistry::new();
    reg.register("cuda", MockBackend::unavailable("cuda", "no driver"));
    reg.register("cpu", MockBackend::available("cpu", 10, all_ops()));

    let dispatcher = BackendDispatcher::new(reg, DispatchStrategy::RoundRobin);
    for _ in 0..5 {
        let d = dispatcher.dispatch(Operation::MatMul).unwrap();
        assert_eq!(d.chosen_backend, "cpu");
    }
}

// ── Specific-backend tests ───────────────────────────────────────────────────

#[test]
fn specific_backend_override() {
    let dispatcher = BackendDispatcher::new(
        standard_registry(),
        DispatchStrategy::SpecificBackend("vulkan".to_owned()),
    );
    let d = dispatcher.dispatch(Operation::MatMul).unwrap();
    assert_eq!(d.chosen_backend, "vulkan");
}

#[test]
fn specific_backend_not_found_errors() {
    let dispatcher = BackendDispatcher::new(
        standard_registry(),
        DispatchStrategy::SpecificBackend("metal".to_owned()),
    );
    let err = dispatcher.dispatch(Operation::MatMul).unwrap_err();
    assert!(matches!(err, DispatchError::BackendNotFound { .. }));
}

#[test]
fn specific_backend_unavailable_errors() {
    let mut reg = BackendRegistry::new();
    reg.register("cuda", MockBackend::unavailable("cuda", "no driver"));

    let dispatcher =
        BackendDispatcher::new(reg, DispatchStrategy::SpecificBackend("cuda".to_owned()));
    let err = dispatcher.dispatch(Operation::MatMul).unwrap_err();
    assert!(matches!(err, DispatchError::BackendNotUsable { .. }));
}

#[test]
fn specific_backend_unsupported_op_errors() {
    let mut reg = BackendRegistry::new();
    reg.register("cpu", MockBackend::available("cpu", 10, vec![Operation::MatMul]));

    let dispatcher =
        BackendDispatcher::new(reg, DispatchStrategy::SpecificBackend("cpu".to_owned()));
    let err = dispatcher.dispatch(Operation::Attention).unwrap_err();
    assert!(matches!(err, DispatchError::OperationNotSupported { .. }));
}

// ── Fallback chain tests ─────────────────────────────────────────────────────

#[test]
fn fallback_chain_works() {
    let mut reg = BackendRegistry::new();
    reg.register("cuda", MockBackend::unavailable("cuda", "no driver"));
    reg.register("opencl", MockBackend::available("opencl", 80, all_ops()));
    reg.register("cpu", MockBackend::available("cpu", 10, all_ops()));

    let dispatcher = BackendDispatcher::new(reg, DispatchStrategy::Priority);
    let d = dispatcher.dispatch_with_fallback(Operation::MatMul).unwrap();
    assert_eq!(d.chosen_backend, "opencl");
}

#[test]
fn fallback_chain_reaches_cpu() {
    let mut reg = BackendRegistry::new();
    reg.register("cuda", MockBackend::unavailable("cuda", "no driver"));
    reg.register("opencl", MockBackend::unavailable("opencl", "no ICD"));
    reg.register("cpu", MockBackend::available("cpu", 10, all_ops()));

    let dispatcher = BackendDispatcher::new(reg, DispatchStrategy::Priority);
    let d = dispatcher.dispatch_with_fallback(Operation::MatMul).unwrap();
    assert_eq!(d.chosen_backend, "cpu");
}

// ── All-unavailable tests ────────────────────────────────────────────────────

#[test]
fn all_backends_unavailable_gives_error() {
    let mut reg = BackendRegistry::new();
    reg.register("cuda", MockBackend::unavailable("cuda", "no driver"));
    reg.register("opencl", MockBackend::unavailable("opencl", "no ICD"));

    let dispatcher = BackendDispatcher::new(reg, DispatchStrategy::Priority);
    let err = dispatcher.dispatch(Operation::MatMul).unwrap_err();
    assert!(matches!(err, DispatchError::NoBackendAvailable { .. }));
}

#[test]
fn fallback_all_fail_gives_error() {
    let reg = BackendRegistry::new();
    let dispatcher = BackendDispatcher::new(reg, DispatchStrategy::Priority);
    let err = dispatcher.dispatch_with_fallback(Operation::MatMul).unwrap_err();
    assert!(matches!(err, DispatchError::NoBackendAvailable { .. }));
}

// ── Dispatch log tests ───────────────────────────────────────────────────────

#[test]
fn dispatch_log_records_decisions() {
    let dispatcher = BackendDispatcher::new(standard_registry(), DispatchStrategy::Priority);
    assert!(dispatcher.log().is_empty());

    dispatcher.dispatch(Operation::MatMul).unwrap();
    dispatcher.dispatch(Operation::Softmax).unwrap();

    assert_eq!(dispatcher.log().len(), 2);
    let entries = dispatcher.log().entries();
    assert_eq!(entries[0].operation, Operation::MatMul);
    assert_eq!(entries[1].operation, Operation::Softmax);
}

#[test]
fn dispatch_log_clear() {
    let dispatcher = BackendDispatcher::new(standard_registry(), DispatchStrategy::Priority);
    dispatcher.dispatch(Operation::MatMul).unwrap();
    assert_eq!(dispatcher.log().len(), 1);

    dispatcher.log().clear();
    assert!(dispatcher.log().is_empty());
}

// ── Capability matrix tests ──────────────────────────────────────────────────

#[test]
fn capability_matrix_queried_correctly() {
    let dispatcher = BackendDispatcher::new(standard_registry(), DispatchStrategy::Priority);
    let matrix = dispatcher.capability_matrix();

    let matmul_backends = matrix.backends_for(Operation::MatMul);
    assert!(matmul_backends.contains(&"cuda".to_owned()));
    assert!(matmul_backends.contains(&"cpu".to_owned()));
    assert!(matrix.is_supported(Operation::MatMul));
}

#[test]
fn capability_matrix_excludes_unavailable() {
    let mut reg = BackendRegistry::new();
    reg.register("cuda", MockBackend::unavailable("cuda", "missing"));
    reg.register("cpu", MockBackend::available("cpu", 10, all_ops()));

    let dispatcher = BackendDispatcher::new(reg, DispatchStrategy::Priority);
    let matrix = dispatcher.capability_matrix();
    let backends = matrix.backends_for(Operation::MatMul);
    assert!(!backends.contains(&"cuda".to_owned()));
    assert!(backends.contains(&"cpu".to_owned()));
}

#[test]
fn capability_matrix_partial_ops() {
    let mut reg = BackendRegistry::new();
    reg.register(
        "cpu",
        MockBackend::available("cpu", 10, vec![Operation::MatMul, Operation::Softmax]),
    );

    let dispatcher = BackendDispatcher::new(reg, DispatchStrategy::Priority);
    let matrix = dispatcher.capability_matrix();
    assert!(matrix.is_supported(Operation::MatMul));
    assert!(matrix.is_supported(Operation::Softmax));
    assert!(!matrix.is_supported(Operation::Attention));
}

// ── Registry tests ───────────────────────────────────────────────────────────

#[test]
fn registration_adds_new_backend() {
    let mut reg = BackendRegistry::new();
    assert!(reg.is_empty());

    reg.register("cpu", MockBackend::available("cpu", 10, all_ops()));
    assert_eq!(reg.len(), 1);
    assert!(reg.get("cpu").is_some());
}

#[test]
fn discovery_finds_all_registered() {
    let reg = standard_registry();
    let infos = reg.discover_available();
    assert_eq!(infos.len(), 4);

    let names: Vec<String> = infos.iter().map(|i| i.name.clone()).collect();
    assert!(names.contains(&"cuda".to_owned()));
    assert!(names.contains(&"opencl".to_owned()));
    assert!(names.contains(&"vulkan".to_owned()));
    assert!(names.contains(&"cpu".to_owned()));
}

#[test]
fn unregister_removes_backend() {
    let mut reg = standard_registry();
    assert!(reg.unregister("vulkan"));
    assert_eq!(reg.len(), 3);
    assert!(reg.get("vulkan").is_none());
}

#[test]
fn unregister_missing_returns_false() {
    let mut reg = BackendRegistry::new();
    assert!(!reg.unregister("metal"));
}

#[test]
fn register_replaces_existing() {
    let mut reg = BackendRegistry::new();
    reg.register("cpu", MockBackend::available("cpu", 10, all_ops()));
    reg.register("cpu", MockBackend::available("cpu", 50, all_ops()));
    assert_eq!(reg.len(), 1);

    let info: Vec<BackendInfo> = reg.discover_available();
    assert_eq!(info[0].priority_score, 50);
}

// ── BackendStatus tests ──────────────────────────────────────────────────────

#[test]
fn backend_status_usability() {
    assert!(BackendStatus::Available.is_usable());
    assert!(BackendStatus::Degraded("hot".into()).is_usable());
    assert!(!BackendStatus::Unavailable("gone".into()).is_usable());
}

// ── Load-based strategy (delegates to priority for now) ──────────────────────

#[test]
fn load_based_delegates_to_priority() {
    let dispatcher = BackendDispatcher::new(standard_registry(), DispatchStrategy::LoadBased);
    let d = dispatcher.dispatch(Operation::MatMul).unwrap();
    assert_eq!(d.chosen_backend, "cuda");
}

// ── Strategy mutation ────────────────────────────────────────────────────────

#[test]
fn strategy_can_be_changed_at_runtime() {
    let mut dispatcher = BackendDispatcher::new(standard_registry(), DispatchStrategy::Priority);
    assert_eq!(*dispatcher.strategy(), DispatchStrategy::Priority);

    dispatcher.set_strategy(DispatchStrategy::RoundRobin);
    assert_eq!(*dispatcher.strategy(), DispatchStrategy::RoundRobin);
}
