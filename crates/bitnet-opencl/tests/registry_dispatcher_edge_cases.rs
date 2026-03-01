//! Edge-case tests for BackendRegistry, BackendDispatcher, DispatchLog,
//! BackendCapabilityMatrix, and associated enums.
//!
//! All tests use mock backends (no real OpenCL devices needed).

use bitnet_opencl::backend_dispatcher::{
    BackendCapabilityMatrix, BackendDispatcher, BackendStatus, DispatchError, DispatchLog,
    DispatchStrategy, Operation,
};
use bitnet_opencl::backend_registry::{BackendInfo, BackendProvider, BackendRegistry};

// ---------------------------------------------------------------------------
// Mock backend for testing
// ---------------------------------------------------------------------------

struct MockBackend {
    name: String,
    status: BackendStatus,
    capabilities: Vec<Operation>,
    priority: u32,
}

impl MockBackend {
    fn available(name: &str, caps: Vec<Operation>, priority: u32) -> Self {
        Self {
            name: name.to_string(),
            status: BackendStatus::Available,
            capabilities: caps,
            priority,
        }
    }

    fn degraded(name: &str, caps: Vec<Operation>, priority: u32, reason: &str) -> Self {
        Self {
            name: name.to_string(),
            status: BackendStatus::Degraded(reason.to_string()),
            capabilities: caps,
            priority,
        }
    }

    fn unavailable(name: &str, reason: &str) -> Self {
        Self {
            name: name.to_string(),
            status: BackendStatus::Unavailable(reason.to_string()),
            capabilities: vec![],
            priority: 0,
        }
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

// ---------------------------------------------------------------------------
// BackendStatus
// ---------------------------------------------------------------------------

#[test]
fn status_available_is_usable() {
    assert!(BackendStatus::Available.is_usable());
}

#[test]
fn status_degraded_is_usable() {
    assert!(BackendStatus::Degraded("low memory".into()).is_usable());
}

#[test]
fn status_unavailable_is_not_usable() {
    assert!(!BackendStatus::Unavailable("no device".into()).is_usable());
}

#[test]
fn status_debug() {
    let dbg = format!("{:?}", BackendStatus::Available);
    assert!(dbg.contains("Available"));
}

#[test]
fn status_clone_eq() {
    let s = BackendStatus::Degraded("test".into());
    let s2 = s.clone();
    assert_eq!(s, s2);
}

// ---------------------------------------------------------------------------
// Operation enum
// ---------------------------------------------------------------------------

#[test]
fn operation_debug_all_variants() {
    let ops = [
        Operation::MatMul,
        Operation::Quantize,
        Operation::Dequantize,
        Operation::Softmax,
        Operation::LayerNorm,
        Operation::Attention,
        Operation::RoPE,
        Operation::Sampling,
    ];
    for op in ops {
        let dbg = format!("{op:?}");
        assert!(!dbg.is_empty());
    }
}

#[test]
fn operation_eq() {
    assert_eq!(Operation::MatMul, Operation::MatMul);
    assert_ne!(Operation::MatMul, Operation::Softmax);
}

#[test]
fn operation_copy_clone() {
    let op = Operation::Attention;
    let op2 = op; // Copy
    let op3 = op.clone();
    assert_eq!(op2, op3);
}

// ---------------------------------------------------------------------------
// BackendRegistry
// ---------------------------------------------------------------------------

#[test]
fn registry_starts_empty() {
    let reg = BackendRegistry::new();
    assert!(reg.is_empty());
    assert_eq!(reg.len(), 0);
}

#[test]
fn registry_default_is_empty() {
    let reg = BackendRegistry::default();
    assert!(reg.is_empty());
}

#[test]
fn registry_register_and_get() {
    let mut reg = BackendRegistry::new();
    reg.register("cuda", Box::new(MockBackend::available("cuda", vec![Operation::MatMul], 100)));
    assert_eq!(reg.len(), 1);
    assert!(!reg.is_empty());
    let backend = reg.get("cuda").unwrap();
    assert_eq!(backend.name(), "cuda");
    assert_eq!(backend.priority_score(), 100);
}

#[test]
fn registry_get_nonexistent() {
    let reg = BackendRegistry::new();
    assert!(reg.get("cuda").is_none());
}

#[test]
fn registry_register_replaces() {
    let mut reg = BackendRegistry::new();
    reg.register("cuda", Box::new(MockBackend::available("cuda", vec![Operation::MatMul], 50)));
    reg.register("cuda", Box::new(MockBackend::available("cuda", vec![Operation::MatMul], 100)));
    assert_eq!(reg.len(), 1);
    assert_eq!(reg.get("cuda").unwrap().priority_score(), 100);
}

#[test]
fn registry_unregister_existing() {
    let mut reg = BackendRegistry::new();
    reg.register("cuda", Box::new(MockBackend::available("cuda", vec![], 0)));
    assert!(reg.unregister("cuda"));
    assert!(reg.is_empty());
}

#[test]
fn registry_unregister_nonexistent() {
    let mut reg = BackendRegistry::new();
    assert!(!reg.unregister("cuda"));
}

#[test]
fn registry_discover_available() {
    let mut reg = BackendRegistry::new();
    reg.register("cuda", Box::new(MockBackend::available("cuda", vec![Operation::MatMul], 100)));
    reg.register(
        "opencl",
        Box::new(MockBackend::available("opencl", vec![Operation::MatMul, Operation::Softmax], 50)),
    );
    let infos = reg.discover_available();
    assert_eq!(infos.len(), 2);
}

#[test]
fn registry_multiple_backends() {
    let mut reg = BackendRegistry::new();
    for i in 0..10 {
        let name = format!("backend_{i}");
        reg.register(&name, Box::new(MockBackend::available(&name, vec![Operation::MatMul], i)));
    }
    assert_eq!(reg.len(), 10);
}

// ---------------------------------------------------------------------------
// BackendInfo
// ---------------------------------------------------------------------------

#[test]
fn backend_info_debug() {
    let info = BackendInfo {
        name: "test".into(),
        status: BackendStatus::Available,
        capabilities: vec![Operation::MatMul],
        priority_score: 42,
    };
    let dbg = format!("{info:?}");
    assert!(dbg.contains("BackendInfo"));
    assert!(dbg.contains("test"));
}

#[test]
fn backend_info_clone() {
    let info = BackendInfo {
        name: "test".into(),
        status: BackendStatus::Available,
        capabilities: vec![Operation::Softmax],
        priority_score: 10,
    };
    let info2 = info.clone();
    assert_eq!(info2.name, "test");
    assert_eq!(info2.priority_score, 10);
}

// ---------------------------------------------------------------------------
// DispatchStrategy
// ---------------------------------------------------------------------------

#[test]
fn strategy_debug_all() {
    let strategies = [
        DispatchStrategy::Priority,
        DispatchStrategy::RoundRobin,
        DispatchStrategy::LoadBased,
        DispatchStrategy::SpecificBackend("cuda".into()),
    ];
    for s in &strategies {
        let dbg = format!("{s:?}");
        assert!(!dbg.is_empty());
    }
}

#[test]
fn strategy_eq() {
    assert_eq!(DispatchStrategy::Priority, DispatchStrategy::Priority);
    assert_ne!(DispatchStrategy::Priority, DispatchStrategy::RoundRobin);
}

// ---------------------------------------------------------------------------
// DispatchLog
// ---------------------------------------------------------------------------

#[test]
fn dispatch_log_starts_empty() {
    let log = DispatchLog::new();
    assert!(log.is_empty());
    assert_eq!(log.len(), 0);
}

#[test]
fn dispatch_log_default() {
    let log = DispatchLog::default();
    assert!(log.is_empty());
}

#[test]
fn dispatch_log_record_and_read() {
    let log = DispatchLog::new();
    log.record(bitnet_opencl::DispatchDecision {
        chosen_backend: "cuda".into(),
        reason: "test".into(),
        alternatives_available: vec![],
        operation: Operation::MatMul,
    });
    assert_eq!(log.len(), 1);
    let entries = log.entries();
    assert_eq!(entries[0].chosen_backend, "cuda");
}

#[test]
fn dispatch_log_clear() {
    let log = DispatchLog::new();
    log.record(bitnet_opencl::DispatchDecision {
        chosen_backend: "test".into(),
        reason: "r".into(),
        alternatives_available: vec![],
        operation: Operation::Softmax,
    });
    assert!(!log.is_empty());
    log.clear();
    assert!(log.is_empty());
}

// ---------------------------------------------------------------------------
// BackendDispatcher — Priority strategy
// ---------------------------------------------------------------------------

fn two_backend_registry() -> BackendRegistry {
    let mut reg = BackendRegistry::new();
    reg.register(
        "cuda",
        Box::new(MockBackend::available(
            "cuda",
            vec![Operation::MatMul, Operation::Softmax, Operation::Attention],
            100,
        )),
    );
    reg.register(
        "opencl",
        Box::new(MockBackend::available("opencl", vec![Operation::MatMul, Operation::Softmax], 50)),
    );
    reg
}

#[test]
fn dispatcher_priority_picks_highest() {
    let dispatcher = BackendDispatcher::new(two_backend_registry(), DispatchStrategy::Priority);
    let decision = dispatcher.dispatch(Operation::MatMul).unwrap();
    assert_eq!(decision.chosen_backend, "cuda");
    assert!(decision.alternatives_available.contains(&"opencl".to_string()));
}

#[test]
fn dispatcher_priority_no_backend_for_op() {
    let dispatcher = BackendDispatcher::new(two_backend_registry(), DispatchStrategy::Priority);
    let result = dispatcher.dispatch(Operation::RoPE);
    assert!(result.is_err());
    assert!(matches!(result.unwrap_err(), DispatchError::NoBackendAvailable { .. }));
}

// ---------------------------------------------------------------------------
// BackendDispatcher — RoundRobin strategy
// ---------------------------------------------------------------------------

#[test]
fn dispatcher_round_robin_alternates() {
    let dispatcher = BackendDispatcher::new(two_backend_registry(), DispatchStrategy::RoundRobin);
    let d1 = dispatcher.dispatch(Operation::MatMul).unwrap();
    let d2 = dispatcher.dispatch(Operation::MatMul).unwrap();
    // Should alternate between the two backends
    assert_ne!(d1.chosen_backend, d2.chosen_backend);
}

// ---------------------------------------------------------------------------
// BackendDispatcher — Specific backend
// ---------------------------------------------------------------------------

#[test]
fn dispatcher_specific_backend() {
    let dispatcher = BackendDispatcher::new(
        two_backend_registry(),
        DispatchStrategy::SpecificBackend("opencl".into()),
    );
    let decision = dispatcher.dispatch(Operation::MatMul).unwrap();
    assert_eq!(decision.chosen_backend, "opencl");
}

#[test]
fn dispatcher_specific_backend_not_found() {
    let dispatcher = BackendDispatcher::new(
        two_backend_registry(),
        DispatchStrategy::SpecificBackend("vulkan".into()),
    );
    let result = dispatcher.dispatch(Operation::MatMul);
    assert!(matches!(result.unwrap_err(), DispatchError::BackendNotFound { .. }));
}

#[test]
fn dispatcher_specific_backend_unsupported_op() {
    let dispatcher = BackendDispatcher::new(
        two_backend_registry(),
        DispatchStrategy::SpecificBackend("opencl".into()),
    );
    let result = dispatcher.dispatch(Operation::Attention);
    assert!(matches!(result.unwrap_err(), DispatchError::OperationNotSupported { .. }));
}

#[test]
fn dispatcher_specific_backend_unavailable() {
    let mut reg = BackendRegistry::new();
    reg.register("broken", Box::new(MockBackend::unavailable("broken", "no device")));
    let dispatcher =
        BackendDispatcher::new(reg, DispatchStrategy::SpecificBackend("broken".into()));
    let result = dispatcher.dispatch(Operation::MatMul);
    assert!(matches!(result.unwrap_err(), DispatchError::BackendNotUsable { .. }));
}

// ---------------------------------------------------------------------------
// BackendDispatcher — with_fallback
// ---------------------------------------------------------------------------

#[test]
fn dispatcher_with_fallback_succeeds() {
    let dispatcher = BackendDispatcher::new(two_backend_registry(), DispatchStrategy::Priority);
    let decision = dispatcher.dispatch_with_fallback(Operation::MatMul).unwrap();
    assert!(!decision.chosen_backend.is_empty());
}

#[test]
fn dispatcher_with_fallback_no_candidates() {
    let reg = BackendRegistry::new();
    let dispatcher = BackendDispatcher::new(reg, DispatchStrategy::Priority);
    let result = dispatcher.dispatch_with_fallback(Operation::MatMul);
    assert!(matches!(result.unwrap_err(), DispatchError::NoBackendAvailable { .. }));
}

// ---------------------------------------------------------------------------
// BackendDispatcher — degraded backend
// ---------------------------------------------------------------------------

#[test]
fn dispatcher_uses_degraded_backend() {
    let mut reg = BackendRegistry::new();
    reg.register(
        "gpu",
        Box::new(MockBackend::degraded("gpu", vec![Operation::MatMul], 80, "low memory")),
    );
    let dispatcher = BackendDispatcher::new(reg, DispatchStrategy::Priority);
    let decision = dispatcher.dispatch(Operation::MatMul).unwrap();
    assert_eq!(decision.chosen_backend, "gpu");
}

// ---------------------------------------------------------------------------
// BackendDispatcher — accessors and log
// ---------------------------------------------------------------------------

#[test]
fn dispatcher_accessors() {
    let mut dispatcher = BackendDispatcher::new(two_backend_registry(), DispatchStrategy::Priority);
    assert_eq!(*dispatcher.strategy(), DispatchStrategy::Priority);
    assert_eq!(dispatcher.registry().len(), 2);
    assert!(dispatcher.log().is_empty());

    dispatcher.set_strategy(DispatchStrategy::RoundRobin);
    assert_eq!(*dispatcher.strategy(), DispatchStrategy::RoundRobin);
}

#[test]
fn dispatcher_log_records_dispatches() {
    let dispatcher = BackendDispatcher::new(two_backend_registry(), DispatchStrategy::Priority);
    dispatcher.dispatch(Operation::MatMul).unwrap();
    dispatcher.dispatch(Operation::Softmax).unwrap();
    assert_eq!(dispatcher.log().len(), 2);
}

// ---------------------------------------------------------------------------
// BackendCapabilityMatrix
// ---------------------------------------------------------------------------

#[test]
fn capability_matrix_backends_for() {
    let dispatcher = BackendDispatcher::new(two_backend_registry(), DispatchStrategy::Priority);
    let matrix = dispatcher.capability_matrix();
    let matmul_backends = matrix.backends_for(Operation::MatMul);
    assert_eq!(matmul_backends.len(), 2);

    let attention_backends = matrix.backends_for(Operation::Attention);
    assert_eq!(attention_backends.len(), 1);
    assert!(attention_backends.contains(&"cuda".to_string()));
}

#[test]
fn capability_matrix_is_supported() {
    let dispatcher = BackendDispatcher::new(two_backend_registry(), DispatchStrategy::Priority);
    let matrix = dispatcher.capability_matrix();
    assert!(matrix.is_supported(Operation::MatMul));
    assert!(!matrix.is_supported(Operation::RoPE));
}

#[test]
fn capability_matrix_excludes_unavailable() {
    let mut reg = BackendRegistry::new();
    reg.register("broken", Box::new(MockBackend::unavailable("broken", "no device")));
    reg.register("ok", Box::new(MockBackend::available("ok", vec![Operation::MatMul], 10)));
    let matrix = BackendCapabilityMatrix::new(&reg);
    let backends = matrix.backends_for(Operation::MatMul);
    assert_eq!(backends.len(), 1);
    assert_eq!(backends[0], "ok");
}

// ---------------------------------------------------------------------------
// DispatchError Display
// ---------------------------------------------------------------------------

#[test]
fn dispatch_error_display() {
    let err = DispatchError::NoBackendAvailable { op: Operation::MatMul };
    assert!(format!("{err}").contains("MatMul"));

    let err = DispatchError::BackendNotFound { name: "cuda".into() };
    assert!(format!("{err}").contains("cuda"));

    let err = DispatchError::OperationNotSupported { name: "cpu".into(), op: Operation::Softmax };
    assert!(format!("{err}").contains("cpu"));
    assert!(format!("{err}").contains("Softmax"));
}

#[test]
fn dispatch_error_debug() {
    let err = DispatchError::AllBackendsFailed {
        op: Operation::MatMul,
        last_backend: "gpu".into(),
        last_reason: "broken".into(),
    };
    let dbg = format!("{err:?}");
    assert!(dbg.contains("AllBackendsFailed"));
}

// ---------------------------------------------------------------------------
// DispatchDecision
// ---------------------------------------------------------------------------

#[test]
fn dispatch_decision_debug_clone() {
    let d = bitnet_opencl::DispatchDecision {
        chosen_backend: "cuda".into(),
        reason: "test".into(),
        alternatives_available: vec!["opencl".into()],
        operation: Operation::MatMul,
    };
    let dbg = format!("{d:?}");
    assert!(dbg.contains("DispatchDecision"));
    let d2 = d.clone();
    assert_eq!(d2.chosen_backend, "cuda");
}

// ---------------------------------------------------------------------------
// BackendProvider default supports() method
// ---------------------------------------------------------------------------

#[test]
fn backend_provider_supports_default_impl() {
    let backend = MockBackend::available("test", vec![Operation::MatMul, Operation::Softmax], 10);
    assert!(backend.supports(Operation::MatMul));
    assert!(backend.supports(Operation::Softmax));
    assert!(!backend.supports(Operation::RoPE));
}

// ---------------------------------------------------------------------------
// Empty registry dispatch
// ---------------------------------------------------------------------------

#[test]
fn dispatch_empty_registry() {
    let reg = BackendRegistry::new();
    let dispatcher = BackendDispatcher::new(reg, DispatchStrategy::Priority);
    assert!(dispatcher.dispatch(Operation::MatMul).is_err());
}
