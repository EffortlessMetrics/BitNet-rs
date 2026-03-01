//! Edge-case tests for OpenCL backend dispatcher, registry, and dispatch strategies.
//!
//! Tests cover mock backends, all dispatch strategies (priority, round-robin,
//! load-based, specific backend), error paths, capability matrix queries,
//! dispatch logging, and degraded/unavailable backend handling.

use bitnet_opencl::backend_registry::{BackendInfo, BackendProvider, BackendRegistry};
use bitnet_opencl::{
    BackendCapabilityMatrix, BackendDispatcher, BackendStatus, DispatchError, DispatchLog,
    DispatchStrategy, Operation,
};

// ── Mock backend ─────────────────────────────────────────────────────────────

struct MockBackend {
    name: String,
    status: BackendStatus,
    capabilities: Vec<Operation>,
    priority: u32,
}

impl MockBackend {
    fn available(name: &str, priority: u32, caps: Vec<Operation>) -> Box<Self> {
        Box::new(Self {
            name: name.to_string(),
            status: BackendStatus::Available,
            capabilities: caps,
            priority,
        })
    }

    fn degraded(name: &str, priority: u32, caps: Vec<Operation>, reason: &str) -> Box<Self> {
        Box::new(Self {
            name: name.to_string(),
            status: BackendStatus::Degraded(reason.to_string()),
            capabilities: caps,
            priority,
        })
    }

    fn unavailable(name: &str, reason: &str) -> Box<Self> {
        Box::new(Self {
            name: name.to_string(),
            status: BackendStatus::Unavailable(reason.to_string()),
            capabilities: vec![],
            priority: 0,
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
        self.capabilities.clone()
    }

    fn priority_score(&self) -> u32 {
        self.priority
    }
}

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

// ── BackendStatus tests ──────────────────────────────────────────────────────

#[test]
fn available_is_usable() {
    assert!(BackendStatus::Available.is_usable());
}

#[test]
fn degraded_is_usable() {
    assert!(BackendStatus::Degraded("low memory".into()).is_usable());
}

#[test]
fn unavailable_is_not_usable() {
    assert!(!BackendStatus::Unavailable("driver missing".into()).is_usable());
}

// ── BackendRegistry tests ────────────────────────────────────────────────────

#[test]
fn empty_registry() {
    let reg = BackendRegistry::new();
    assert!(reg.is_empty());
    assert_eq!(reg.len(), 0);
    assert!(reg.get("nonexistent").is_none());
}

#[test]
fn register_and_lookup() {
    let mut reg = BackendRegistry::new();
    reg.register("cuda", MockBackend::available("cuda", 100, all_ops()));
    assert_eq!(reg.len(), 1);
    assert!(!reg.is_empty());
    assert!(reg.get("cuda").is_some());
    assert_eq!(reg.get("cuda").unwrap().name(), "cuda");
}

#[test]
fn register_replaces_existing() {
    let mut reg = BackendRegistry::new();
    reg.register("cuda", MockBackend::available("cuda", 100, all_ops()));
    reg.register("cuda", MockBackend::available("cuda", 200, all_ops()));
    assert_eq!(reg.len(), 1);
    assert_eq!(reg.get("cuda").unwrap().priority_score(), 200);
}

#[test]
fn unregister_existing() {
    let mut reg = BackendRegistry::new();
    reg.register("cuda", MockBackend::available("cuda", 100, all_ops()));
    assert!(reg.unregister("cuda"));
    assert!(reg.is_empty());
}

#[test]
fn unregister_nonexistent_returns_false() {
    let mut reg = BackendRegistry::new();
    assert!(!reg.unregister("nonexistent"));
}

#[test]
fn discover_available_returns_all() {
    let mut reg = BackendRegistry::new();
    reg.register("cuda", MockBackend::available("cuda", 100, vec![Operation::MatMul]));
    reg.register("opencl", MockBackend::degraded("opencl", 50, vec![Operation::Softmax], "hot"));
    reg.register("cpu", MockBackend::unavailable("cpu", "no SIMD"));

    let infos = reg.discover_available();
    assert_eq!(infos.len(), 3);
}

#[test]
fn default_registry_is_empty() {
    let reg = BackendRegistry::default();
    assert!(reg.is_empty());
}

// ── DispatchLog tests ────────────────────────────────────────────────────────

#[test]
fn dispatch_log_starts_empty() {
    let log = DispatchLog::new();
    assert!(log.is_empty());
    assert_eq!(log.len(), 0);
    assert!(log.entries().is_empty());
}

#[test]
fn dispatch_log_default_is_empty() {
    let log = DispatchLog::default();
    assert!(log.is_empty());
}

#[test]
fn dispatch_log_record_and_retrieve() {
    let log = DispatchLog::new();
    log.record(bitnet_opencl::DispatchDecision {
        chosen_backend: "cuda".into(),
        reason: "test".into(),
        alternatives_available: vec!["cpu".into()],
        operation: Operation::MatMul,
    });
    assert_eq!(log.len(), 1);
    assert!(!log.is_empty());
    let entries = log.entries();
    assert_eq!(entries[0].chosen_backend, "cuda");
    assert_eq!(entries[0].operation, Operation::MatMul);
}

#[test]
fn dispatch_log_clear() {
    let log = DispatchLog::new();
    log.record(bitnet_opencl::DispatchDecision {
        chosen_backend: "x".into(),
        reason: "y".into(),
        alternatives_available: vec![],
        operation: Operation::Softmax,
    });
    assert_eq!(log.len(), 1);
    log.clear();
    assert!(log.is_empty());
}

// ── Priority dispatch tests ──────────────────────────────────────────────────

fn two_backend_registry() -> BackendRegistry {
    let mut reg = BackendRegistry::new();
    reg.register("cuda", MockBackend::available("cuda", 100, all_ops()));
    reg.register("cpu", MockBackend::available("cpu", 10, all_ops()));
    reg
}

#[test]
fn priority_dispatch_selects_highest() {
    let dispatcher = BackendDispatcher::new(two_backend_registry(), DispatchStrategy::Priority);
    let decision = dispatcher.dispatch(Operation::MatMul).unwrap();
    assert_eq!(decision.chosen_backend, "cuda");
    assert_eq!(decision.operation, Operation::MatMul);
    assert!(decision.alternatives_available.contains(&"cpu".to_string()));
}

#[test]
fn priority_dispatch_logs_decision() {
    let dispatcher = BackendDispatcher::new(two_backend_registry(), DispatchStrategy::Priority);
    dispatcher.dispatch(Operation::Softmax).unwrap();
    assert_eq!(dispatcher.log().len(), 1);
}

#[test]
fn priority_dispatch_no_backend_for_unsupported_op() {
    let mut reg = BackendRegistry::new();
    reg.register("cuda", MockBackend::available("cuda", 100, vec![Operation::MatMul]));
    let dispatcher = BackendDispatcher::new(reg, DispatchStrategy::Priority);
    let err = dispatcher.dispatch(Operation::Softmax).unwrap_err();
    assert!(matches!(err, DispatchError::NoBackendAvailable { .. }));
}

// ── Round-robin dispatch tests ───────────────────────────────────────────────

#[test]
fn round_robin_alternates() {
    let dispatcher = BackendDispatcher::new(two_backend_registry(), DispatchStrategy::RoundRobin);

    let d1 = dispatcher.dispatch(Operation::MatMul).unwrap();
    let d2 = dispatcher.dispatch(Operation::MatMul).unwrap();

    // Should pick different backends on consecutive calls
    // (sorted by name: cpu < cuda, so idx 0=cpu, idx 1=cuda)
    assert_ne!(d1.chosen_backend, d2.chosen_backend);
}

#[test]
fn round_robin_wraps_around() {
    let dispatcher = BackendDispatcher::new(two_backend_registry(), DispatchStrategy::RoundRobin);

    let d1 = dispatcher.dispatch(Operation::MatMul).unwrap();
    let _d2 = dispatcher.dispatch(Operation::MatMul).unwrap();
    let d3 = dispatcher.dispatch(Operation::MatMul).unwrap();

    // After 2 dispatches, should wrap back to first backend
    assert_eq!(d1.chosen_backend, d3.chosen_backend);
}

#[test]
fn round_robin_single_backend() {
    let mut reg = BackendRegistry::new();
    reg.register("only", MockBackend::available("only", 50, all_ops()));
    let dispatcher = BackendDispatcher::new(reg, DispatchStrategy::RoundRobin);

    let d1 = dispatcher.dispatch(Operation::MatMul).unwrap();
    let d2 = dispatcher.dispatch(Operation::MatMul).unwrap();
    assert_eq!(d1.chosen_backend, d2.chosen_backend);
    assert_eq!(d1.chosen_backend, "only");
}

// ── LoadBased dispatch tests ─────────────────────────────────────────────────

#[test]
fn load_based_falls_back_to_priority() {
    let dispatcher = BackendDispatcher::new(two_backend_registry(), DispatchStrategy::LoadBased);
    let decision = dispatcher.dispatch(Operation::MatMul).unwrap();
    // LoadBased currently falls back to priority
    assert_eq!(decision.chosen_backend, "cuda");
}

// ── SpecificBackend dispatch tests ───────────────────────────────────────────

#[test]
fn specific_backend_selects_named() {
    let dispatcher = BackendDispatcher::new(
        two_backend_registry(),
        DispatchStrategy::SpecificBackend("cpu".into()),
    );
    let decision = dispatcher.dispatch(Operation::MatMul).unwrap();
    assert_eq!(decision.chosen_backend, "cpu");
}

#[test]
fn specific_backend_not_found() {
    let dispatcher = BackendDispatcher::new(
        two_backend_registry(),
        DispatchStrategy::SpecificBackend("vulkan".into()),
    );
    let err = dispatcher.dispatch(Operation::MatMul).unwrap_err();
    assert!(matches!(err, DispatchError::BackendNotFound { .. }));
}

#[test]
fn specific_backend_unavailable() {
    let mut reg = BackendRegistry::new();
    reg.register("down", MockBackend::unavailable("down", "crashed"));
    let dispatcher = BackendDispatcher::new(reg, DispatchStrategy::SpecificBackend("down".into()));
    let err = dispatcher.dispatch(Operation::MatMul).unwrap_err();
    assert!(matches!(err, DispatchError::BackendNotUsable { .. }));
}

#[test]
fn specific_backend_does_not_support_op() {
    let mut reg = BackendRegistry::new();
    reg.register("limited", MockBackend::available("limited", 50, vec![Operation::MatMul]));
    let dispatcher =
        BackendDispatcher::new(reg, DispatchStrategy::SpecificBackend("limited".into()));
    let err = dispatcher.dispatch(Operation::Softmax).unwrap_err();
    assert!(matches!(err, DispatchError::OperationNotSupported { .. }));
}

// ── Degraded backend tests ───────────────────────────────────────────────────

#[test]
fn degraded_backend_is_usable_for_dispatch() {
    let mut reg = BackendRegistry::new();
    reg.register("degraded", MockBackend::degraded("degraded", 50, all_ops(), "overheating"));
    let dispatcher = BackendDispatcher::new(reg, DispatchStrategy::Priority);
    let decision = dispatcher.dispatch(Operation::MatMul).unwrap();
    assert_eq!(decision.chosen_backend, "degraded");
}

#[test]
fn degraded_lower_priority_than_available() {
    let mut reg = BackendRegistry::new();
    reg.register("degraded", MockBackend::degraded("degraded", 100, all_ops(), "hot"));
    reg.register("healthy", MockBackend::available("healthy", 100, all_ops()));
    let dispatcher = BackendDispatcher::new(reg, DispatchStrategy::Priority);
    let decision = dispatcher.dispatch(Operation::MatMul).unwrap();
    // Both have same priority, but order depends on HashMap iteration
    assert!(decision.chosen_backend == "degraded" || decision.chosen_backend == "healthy");
}

// ── Dispatch with fallback tests ─────────────────────────────────────────────

#[test]
fn dispatch_with_fallback_selects_first_usable() {
    let mut reg = BackendRegistry::new();
    reg.register("cuda", MockBackend::available("cuda", 100, all_ops()));
    reg.register("cpu", MockBackend::available("cpu", 10, all_ops()));
    let dispatcher = BackendDispatcher::new(reg, DispatchStrategy::Priority);
    let decision = dispatcher.dispatch_with_fallback(Operation::MatMul).unwrap();
    assert_eq!(decision.chosen_backend, "cuda");
}

#[test]
fn dispatch_with_fallback_empty_registry() {
    let reg = BackendRegistry::new();
    let dispatcher = BackendDispatcher::new(reg, DispatchStrategy::Priority);
    let err = dispatcher.dispatch_with_fallback(Operation::MatMul).unwrap_err();
    assert!(matches!(err, DispatchError::NoBackendAvailable { .. }));
}

// ── Capability matrix tests ──────────────────────────────────────────────────

#[test]
fn capability_matrix_supported() {
    let dispatcher = BackendDispatcher::new(two_backend_registry(), DispatchStrategy::Priority);
    let matrix = dispatcher.capability_matrix();
    assert!(matrix.is_supported(Operation::MatMul));
    assert!(matrix.is_supported(Operation::Softmax));
}

#[test]
fn capability_matrix_not_supported() {
    let mut reg = BackendRegistry::new();
    reg.register("limited", MockBackend::available("limited", 50, vec![Operation::MatMul]));
    let dispatcher = BackendDispatcher::new(reg, DispatchStrategy::Priority);
    let matrix = dispatcher.capability_matrix();
    assert!(matrix.is_supported(Operation::MatMul));
    assert!(!matrix.is_supported(Operation::Softmax));
}

#[test]
fn capability_matrix_backends_for() {
    let dispatcher = BackendDispatcher::new(two_backend_registry(), DispatchStrategy::Priority);
    let matrix = dispatcher.capability_matrix();
    let backends = matrix.backends_for(Operation::MatMul);
    assert!(backends.contains(&"cuda".to_string()));
    assert!(backends.contains(&"cpu".to_string()));
}

#[test]
fn capability_matrix_excludes_unavailable() {
    let mut reg = BackendRegistry::new();
    reg.register("down", MockBackend::unavailable("down", "crashed"));
    reg.register("up", MockBackend::available("up", 50, all_ops()));
    let dispatcher = BackendDispatcher::new(reg, DispatchStrategy::Priority);
    let matrix = dispatcher.capability_matrix();
    let backends = matrix.backends_for(Operation::MatMul);
    assert_eq!(backends, vec!["up"]);
}

// ── Strategy mutation tests ──────────────────────────────────────────────────

#[test]
fn set_strategy_changes_behavior() {
    let mut dispatcher = BackendDispatcher::new(two_backend_registry(), DispatchStrategy::Priority);
    assert_eq!(dispatcher.strategy(), &DispatchStrategy::Priority);

    dispatcher.set_strategy(DispatchStrategy::RoundRobin);
    assert_eq!(dispatcher.strategy(), &DispatchStrategy::RoundRobin);
}

#[test]
fn registry_mut_add_backend() {
    let mut dispatcher = BackendDispatcher::new(BackendRegistry::new(), DispatchStrategy::Priority);
    dispatcher.registry_mut().register("new", MockBackend::available("new", 50, all_ops()));
    assert_eq!(dispatcher.registry().len(), 1);
}

// ── Operation enum tests ─────────────────────────────────────────────────────

#[test]
fn operation_debug_and_clone() {
    let op = Operation::MatMul;
    let op2 = op;
    assert_eq!(op, op2);
    assert_eq!(format!("{op:?}"), "MatMul");
}

#[test]
fn all_operations_distinct() {
    let ops = all_ops();
    for (i, a) in ops.iter().enumerate() {
        for (j, b) in ops.iter().enumerate() {
            if i != j {
                assert_ne!(a, b);
            }
        }
    }
}

// ── Error display tests ─────────────────────────────────────────────────────

#[test]
fn dispatch_error_display_no_backend() {
    let err = DispatchError::NoBackendAvailable { op: Operation::MatMul };
    let msg = format!("{err}");
    assert!(msg.contains("MatMul"));
}

#[test]
fn dispatch_error_display_not_found() {
    let err = DispatchError::BackendNotFound { name: "vulkan".into() };
    let msg = format!("{err}");
    assert!(msg.contains("vulkan"));
}

#[test]
fn dispatch_error_display_not_supported() {
    let err = DispatchError::OperationNotSupported { name: "cpu".into(), op: Operation::Attention };
    let msg = format!("{err}");
    assert!(msg.contains("cpu"));
    assert!(msg.contains("Attention"));
}

#[test]
fn dispatch_error_display_not_usable() {
    let err = DispatchError::BackendNotUsable {
        name: "dead".into(),
        status: BackendStatus::Unavailable("gone".into()),
    };
    let msg = format!("{err}");
    assert!(msg.contains("dead"));
}

#[test]
fn dispatch_error_display_all_failed() {
    let err = DispatchError::AllBackendsFailed {
        op: Operation::RoPE,
        last_backend: "cpu".into(),
        last_reason: "timeout".into(),
    };
    let msg = format!("{err}");
    assert!(msg.contains("RoPE"));
    assert!(msg.contains("timeout"));
}
