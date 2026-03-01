//! Edge-case tests for model_registry: ModelState, ModelEntry, LoadModelRequest,
//! UnloadModelRequest, ListModelsResponse, RegistryError, ModelRegistry lifecycle,
//! memory accounting, state transitions, and concurrent access patterns.

use bitnet_server::model_registry::{
    ListModelsResponse, LoadModelRequest, ModelEntry, ModelRegistry, ModelState, RegistryError,
    UnloadModelRequest,
};

// ─── Helper ─────────────────────────────────────────────────────────

fn load_req(model: &str, device: &str, mem: u64) -> LoadModelRequest {
    LoadModelRequest {
        model_id: model.to_string(),
        device_id: device.to_string(),
        memory_bytes: mem,
    }
}

fn unload_req(model: &str, device: &str) -> UnloadModelRequest {
    UnloadModelRequest { model_id: model.to_string(), device_id: device.to_string() }
}

// ─── ModelState ─────────────────────────────────────────────────────

#[test]
fn model_state_display() {
    assert_eq!(format!("{}", ModelState::Loading), "loading");
    assert_eq!(format!("{}", ModelState::Ready), "ready");
    assert_eq!(format!("{}", ModelState::Serving), "serving");
    assert_eq!(format!("{}", ModelState::Unloading), "unloading");
}

#[test]
fn model_state_debug() {
    let debug = format!("{:?}", ModelState::Loading);
    assert!(debug.contains("Loading"));
}

#[test]
fn model_state_clone_copy() {
    let state = ModelState::Ready;
    let cloned = state.clone();
    let copied = state;
    assert_eq!(cloned, copied);
    assert_eq!(cloned, ModelState::Ready);
}

#[test]
fn model_state_eq() {
    assert_eq!(ModelState::Loading, ModelState::Loading);
    assert_ne!(ModelState::Loading, ModelState::Ready);
    assert_ne!(ModelState::Ready, ModelState::Serving);
    assert_ne!(ModelState::Serving, ModelState::Unloading);
}

#[test]
fn model_state_serde_roundtrip() {
    for state in
        [ModelState::Loading, ModelState::Ready, ModelState::Serving, ModelState::Unloading]
    {
        let json = serde_json::to_string(&state).unwrap();
        let deser: ModelState = serde_json::from_str(&json).unwrap();
        assert_eq!(deser, state);
    }
}

#[test]
fn model_state_serde_snake_case() {
    let json = serde_json::to_string(&ModelState::Loading).unwrap();
    assert_eq!(json, "\"loading\"");
    let json = serde_json::to_string(&ModelState::Ready).unwrap();
    assert_eq!(json, "\"ready\"");
}

#[test]
fn model_state_hash() {
    use std::collections::HashSet;
    let mut set = HashSet::new();
    set.insert(ModelState::Loading);
    set.insert(ModelState::Ready);
    set.insert(ModelState::Loading); // duplicate
    assert_eq!(set.len(), 2);
}

// ─── ModelEntry ─────────────────────────────────────────────────────

#[test]
fn model_entry_construction() {
    let entry = ModelEntry {
        model_id: "test-model".to_string(),
        device_id: "gpu0".to_string(),
        state: ModelState::Ready,
        memory_bytes: 1_000_000,
        state_changed_at: None,
    };
    assert_eq!(entry.model_id, "test-model");
    assert_eq!(entry.device_id, "gpu0");
    assert_eq!(entry.state, ModelState::Ready);
    assert_eq!(entry.memory_bytes, 1_000_000);
    assert!(entry.state_changed_at.is_none());
}

#[test]
fn model_entry_serde_roundtrip() {
    let entry = ModelEntry {
        model_id: "m1".to_string(),
        device_id: "d1".to_string(),
        state: ModelState::Serving,
        memory_bytes: 5_000_000_000,
        state_changed_at: None,
    };
    let json = serde_json::to_string(&entry).unwrap();
    let deser: ModelEntry = serde_json::from_str(&json).unwrap();
    assert_eq!(deser.model_id, "m1");
    assert_eq!(deser.state, ModelState::Serving);
    assert_eq!(deser.memory_bytes, 5_000_000_000);
    // state_changed_at is #[serde(skip)] so it's None after deser
    assert!(deser.state_changed_at.is_none());
}

#[test]
fn model_entry_serde_skips_instant() {
    let entry = ModelEntry {
        model_id: "m".to_string(),
        device_id: "d".to_string(),
        state: ModelState::Loading,
        memory_bytes: 0,
        state_changed_at: Some(std::time::Instant::now()),
    };
    let json = serde_json::to_string(&entry).unwrap();
    assert!(!json.contains("state_changed_at"));
}

#[test]
fn model_entry_clone() {
    let entry = ModelEntry {
        model_id: "m1".to_string(),
        device_id: "gpu0".to_string(),
        state: ModelState::Ready,
        memory_bytes: 100,
        state_changed_at: None,
    };
    let cloned = entry.clone();
    assert_eq!(cloned.model_id, "m1");
    assert_eq!(cloned.state, ModelState::Ready);
}

#[test]
fn model_entry_debug() {
    let entry = ModelEntry {
        model_id: "m".to_string(),
        device_id: "d".to_string(),
        state: ModelState::Unloading,
        memory_bytes: 42,
        state_changed_at: None,
    };
    let debug = format!("{:?}", entry);
    assert!(debug.contains("ModelEntry"));
    assert!(debug.contains("Unloading"));
}

// ─── LoadModelRequest ───────────────────────────────────────────────

#[test]
fn load_model_request_serde() {
    let req = load_req("phi-4", "gpu:0", 29_000_000_000);
    let json = serde_json::to_string(&req).unwrap();
    let deser: LoadModelRequest = serde_json::from_str(&json).unwrap();
    assert_eq!(deser.model_id, "phi-4");
    assert_eq!(deser.device_id, "gpu:0");
    assert_eq!(deser.memory_bytes, 29_000_000_000);
}

#[test]
fn load_model_request_clone() {
    let req = load_req("model", "dev", 0);
    let cloned = req.clone();
    assert_eq!(cloned.model_id, "model");
}

// ─── UnloadModelRequest ─────────────────────────────────────────────

#[test]
fn unload_model_request_serde() {
    let req = unload_req("m1", "d1");
    let json = serde_json::to_string(&req).unwrap();
    let deser: UnloadModelRequest = serde_json::from_str(&json).unwrap();
    assert_eq!(deser.model_id, "m1");
    assert_eq!(deser.device_id, "d1");
}

// ─── ListModelsResponse ─────────────────────────────────────────────

#[test]
fn list_models_response_empty() {
    let resp = ListModelsResponse { models: vec![] };
    assert!(resp.models.is_empty());
    let json = serde_json::to_string(&resp).unwrap();
    assert!(json.contains("models"));
}

#[test]
fn list_models_response_serde() {
    let resp = ListModelsResponse {
        models: vec![ModelEntry {
            model_id: "m1".to_string(),
            device_id: "d1".to_string(),
            state: ModelState::Ready,
            memory_bytes: 100,
            state_changed_at: None,
        }],
    };
    let json = serde_json::to_string(&resp).unwrap();
    let deser: ListModelsResponse = serde_json::from_str(&json).unwrap();
    assert_eq!(deser.models.len(), 1);
    assert_eq!(deser.models[0].model_id, "m1");
}

// ─── RegistryError ──────────────────────────────────────────────────

#[test]
fn registry_error_display_already_loaded() {
    let e =
        RegistryError::AlreadyLoaded { model_id: "m1".to_string(), device_id: "gpu0".to_string() };
    let display = format!("{}", e);
    assert!(display.contains("m1"));
    assert!(display.contains("gpu0"));
    assert!(display.contains("already loaded"));
}

#[test]
fn registry_error_display_not_found() {
    let e = RegistryError::NotFound { model_id: "m1".to_string(), device_id: "gpu0".to_string() };
    let display = format!("{}", e);
    assert!(display.contains("not found"));
}

#[test]
fn registry_error_display_invalid_transition() {
    let e = RegistryError::InvalidTransition { from: ModelState::Loading, to: ModelState::Serving };
    let display = format!("{}", e);
    assert!(display.contains("invalid transition"));
    assert!(display.contains("loading"));
    assert!(display.contains("serving"));
}

#[test]
fn registry_error_display_insufficient_memory() {
    let e = RegistryError::InsufficientMemory {
        device_id: "gpu0".to_string(),
        required: 10_000,
        available: 5_000,
    };
    let display = format!("{}", e);
    assert!(display.contains("gpu0"));
    assert!(display.contains("10000"));
    assert!(display.contains("5000"));
}

#[test]
fn registry_error_eq() {
    let e1 = RegistryError::AlreadyLoaded { model_id: "m".to_string(), device_id: "d".to_string() };
    let e2 = RegistryError::AlreadyLoaded { model_id: "m".to_string(), device_id: "d".to_string() };
    assert_eq!(e1, e2);
}

#[test]
fn registry_error_ne() {
    let e1 = RegistryError::AlreadyLoaded { model_id: "m".to_string(), device_id: "d".to_string() };
    let e2 = RegistryError::NotFound { model_id: "m".to_string(), device_id: "d".to_string() };
    assert_ne!(e1, e2);
}

#[test]
fn registry_error_serde_roundtrip() {
    let errors = vec![
        RegistryError::AlreadyLoaded { model_id: "m".to_string(), device_id: "d".to_string() },
        RegistryError::NotFound { model_id: "m".to_string(), device_id: "d".to_string() },
        RegistryError::InvalidTransition { from: ModelState::Loading, to: ModelState::Serving },
        RegistryError::InsufficientMemory {
            device_id: "gpu0".to_string(),
            required: 100,
            available: 50,
        },
    ];
    for e in &errors {
        let json = serde_json::to_string(e).unwrap();
        let deser: RegistryError = serde_json::from_str(&json).unwrap();
        assert_eq!(&deser, e);
    }
}

#[test]
fn registry_error_is_std_error() {
    let e: Box<dyn std::error::Error> =
        Box::new(RegistryError::NotFound { model_id: "m".to_string(), device_id: "d".to_string() });
    assert!(e.to_string().contains("not found"));
}

// ─── ModelRegistry — Basic Operations ───────────────────────────────

#[test]
fn registry_new_is_empty() {
    let reg = ModelRegistry::new();
    assert!(reg.is_empty());
    assert_eq!(reg.len(), 0);
}

#[test]
fn registry_default_is_empty() {
    let reg = ModelRegistry::default();
    assert!(reg.is_empty());
}

#[test]
fn registry_load_model() {
    let reg = ModelRegistry::new();
    reg.load_model(&load_req("m1", "gpu0", 1000)).unwrap();
    assert_eq!(reg.len(), 1);
    assert!(!reg.is_empty());
}

#[test]
fn registry_load_duplicate() {
    let reg = ModelRegistry::new();
    reg.load_model(&load_req("m1", "gpu0", 1000)).unwrap();
    let err = reg.load_model(&load_req("m1", "gpu0", 1000)).unwrap_err();
    assert!(matches!(err, RegistryError::AlreadyLoaded { .. }));
}

#[test]
fn registry_same_model_different_devices() {
    let reg = ModelRegistry::new();
    reg.load_model(&load_req("m1", "gpu0", 1000)).unwrap();
    reg.load_model(&load_req("m1", "gpu1", 1000)).unwrap();
    assert_eq!(reg.len(), 2);
}

#[test]
fn registry_different_models_same_device() {
    let reg = ModelRegistry::new();
    reg.load_model(&load_req("m1", "gpu0", 1000)).unwrap();
    reg.load_model(&load_req("m2", "gpu0", 2000)).unwrap();
    assert_eq!(reg.len(), 2);
}

// ─── ModelRegistry — State Transitions ──────────────────────────────

#[test]
fn registry_valid_loading_to_ready() {
    let reg = ModelRegistry::new();
    reg.load_model(&load_req("m", "d", 100)).unwrap();
    reg.set_state("m", "d", ModelState::Ready).unwrap();
    assert_eq!(reg.get("m", "d").unwrap().state, ModelState::Ready);
}

#[test]
fn registry_valid_ready_to_serving() {
    let reg = ModelRegistry::new();
    reg.load_model(&load_req("m", "d", 100)).unwrap();
    reg.set_state("m", "d", ModelState::Ready).unwrap();
    reg.set_state("m", "d", ModelState::Serving).unwrap();
    assert_eq!(reg.get("m", "d").unwrap().state, ModelState::Serving);
}

#[test]
fn registry_valid_serving_to_ready() {
    let reg = ModelRegistry::new();
    reg.load_model(&load_req("m", "d", 100)).unwrap();
    reg.set_state("m", "d", ModelState::Ready).unwrap();
    reg.set_state("m", "d", ModelState::Serving).unwrap();
    reg.set_state("m", "d", ModelState::Ready).unwrap();
    assert_eq!(reg.get("m", "d").unwrap().state, ModelState::Ready);
}

#[test]
fn registry_valid_loading_to_unloading() {
    let reg = ModelRegistry::new();
    reg.load_model(&load_req("m", "d", 100)).unwrap();
    reg.set_state("m", "d", ModelState::Unloading).unwrap();
}

#[test]
fn registry_valid_ready_to_unloading() {
    let reg = ModelRegistry::new();
    reg.load_model(&load_req("m", "d", 100)).unwrap();
    reg.set_state("m", "d", ModelState::Ready).unwrap();
    reg.set_state("m", "d", ModelState::Unloading).unwrap();
}

#[test]
fn registry_valid_serving_to_unloading() {
    let reg = ModelRegistry::new();
    reg.load_model(&load_req("m", "d", 100)).unwrap();
    reg.set_state("m", "d", ModelState::Ready).unwrap();
    reg.set_state("m", "d", ModelState::Serving).unwrap();
    reg.set_state("m", "d", ModelState::Unloading).unwrap();
}

#[test]
fn registry_invalid_loading_to_serving() {
    let reg = ModelRegistry::new();
    reg.load_model(&load_req("m", "d", 100)).unwrap();
    let err = reg.set_state("m", "d", ModelState::Serving).unwrap_err();
    assert!(matches!(err, RegistryError::InvalidTransition { .. }));
}

#[test]
fn registry_invalid_unloading_to_ready() {
    let reg = ModelRegistry::new();
    reg.load_model(&load_req("m", "d", 100)).unwrap();
    reg.set_state("m", "d", ModelState::Unloading).unwrap();
    let err = reg.set_state("m", "d", ModelState::Ready).unwrap_err();
    assert!(matches!(err, RegistryError::InvalidTransition { .. }));
}

#[test]
fn registry_invalid_loading_to_loading() {
    let reg = ModelRegistry::new();
    reg.load_model(&load_req("m", "d", 100)).unwrap();
    let err = reg.set_state("m", "d", ModelState::Loading).unwrap_err();
    assert!(matches!(err, RegistryError::InvalidTransition { .. }));
}

#[test]
fn registry_set_state_not_found() {
    let reg = ModelRegistry::new();
    let err = reg.set_state("nonexistent", "gpu0", ModelState::Ready).unwrap_err();
    assert!(matches!(err, RegistryError::NotFound { .. }));
}

// ─── ModelRegistry — Remove & Unload ────────────────────────────────

#[test]
fn registry_remove() {
    let reg = ModelRegistry::new();
    reg.load_model(&load_req("m", "d", 100)).unwrap();
    let entry = reg.remove("m", "d").unwrap();
    assert_eq!(entry.model_id, "m");
    assert!(reg.is_empty());
}

#[test]
fn registry_remove_not_found() {
    let reg = ModelRegistry::new();
    let err = reg.remove("m", "d").unwrap_err();
    assert!(matches!(err, RegistryError::NotFound { .. }));
}

#[test]
fn registry_unload_from_ready() {
    let reg = ModelRegistry::new();
    reg.load_model(&load_req("m", "d", 100)).unwrap();
    reg.set_state("m", "d", ModelState::Ready).unwrap();
    let entry = reg.unload_model(&unload_req("m", "d")).unwrap();
    assert_eq!(entry.model_id, "m");
    assert!(reg.is_empty());
}

#[test]
fn registry_unload_from_serving() {
    let reg = ModelRegistry::new();
    reg.load_model(&load_req("m", "d", 100)).unwrap();
    reg.set_state("m", "d", ModelState::Ready).unwrap();
    reg.set_state("m", "d", ModelState::Serving).unwrap();
    let entry = reg.unload_model(&unload_req("m", "d")).unwrap();
    assert_eq!(entry.model_id, "m");
}

#[test]
fn registry_unload_not_found() {
    let reg = ModelRegistry::new();
    let err = reg.unload_model(&unload_req("m", "d")).unwrap_err();
    assert!(matches!(err, RegistryError::NotFound { .. }));
}

// ─── ModelRegistry — Query Operations ───────────────────────────────

#[test]
fn registry_get() {
    let reg = ModelRegistry::new();
    reg.load_model(&load_req("m1", "gpu0", 1000)).unwrap();
    let entry = reg.get("m1", "gpu0").unwrap();
    assert_eq!(entry.model_id, "m1");
    assert_eq!(entry.state, ModelState::Loading);
}

#[test]
fn registry_get_not_found() {
    let reg = ModelRegistry::new();
    assert!(reg.get("nonexistent", "gpu0").is_none());
}

#[test]
fn registry_list_models() {
    let reg = ModelRegistry::new();
    reg.load_model(&load_req("m1", "gpu0", 1000)).unwrap();
    reg.load_model(&load_req("m2", "gpu1", 2000)).unwrap();
    let list = reg.list_models();
    assert_eq!(list.models.len(), 2);
}

#[test]
fn registry_list_models_empty() {
    let reg = ModelRegistry::new();
    let list = reg.list_models();
    assert!(list.models.is_empty());
}

#[test]
fn registry_models_on_device() {
    let reg = ModelRegistry::new();
    reg.load_model(&load_req("m1", "gpu0", 1000)).unwrap();
    reg.load_model(&load_req("m2", "gpu0", 2000)).unwrap();
    reg.load_model(&load_req("m3", "gpu1", 3000)).unwrap();

    let gpu0_models = reg.models_on_device("gpu0");
    assert_eq!(gpu0_models.len(), 2);
    let gpu1_models = reg.models_on_device("gpu1");
    assert_eq!(gpu1_models.len(), 1);
    let gpu2_models = reg.models_on_device("gpu2");
    assert!(gpu2_models.is_empty());
}

// ─── ModelRegistry — Memory Accounting ──────────────────────────────

#[test]
fn registry_used_memory() {
    let reg = ModelRegistry::new();
    reg.load_model(&load_req("m1", "gpu0", 1000)).unwrap();
    reg.load_model(&load_req("m2", "gpu0", 2000)).unwrap();
    assert_eq!(reg.used_memory("gpu0"), 3000);
}

#[test]
fn registry_used_memory_empty() {
    let reg = ModelRegistry::new();
    assert_eq!(reg.used_memory("gpu0"), 0);
}

#[test]
fn registry_memory_check_rejects_overcommit() {
    let reg = ModelRegistry::new();
    reg.set_device_capacity("gpu0", 5000);
    reg.load_model(&load_req("m1", "gpu0", 3000)).unwrap();
    let err = reg.load_model(&load_req("m2", "gpu0", 3000)).unwrap_err();
    assert!(matches!(err, RegistryError::InsufficientMemory { .. }));
    if let RegistryError::InsufficientMemory { required, available, .. } = err {
        assert_eq!(required, 3000);
        assert_eq!(available, 2000);
    }
}

#[test]
fn registry_memory_check_allows_within_capacity() {
    let reg = ModelRegistry::new();
    reg.set_device_capacity("gpu0", 10000);
    reg.load_model(&load_req("m1", "gpu0", 5000)).unwrap();
    reg.load_model(&load_req("m2", "gpu0", 5000)).unwrap();
    assert_eq!(reg.used_memory("gpu0"), 10000);
}

#[test]
fn registry_memory_check_exact_fit() {
    let reg = ModelRegistry::new();
    reg.set_device_capacity("gpu0", 5000);
    reg.load_model(&load_req("m1", "gpu0", 5000)).unwrap();
    // Exactly fills capacity
    let err = reg.load_model(&load_req("m2", "gpu0", 1)).unwrap_err();
    assert!(matches!(err, RegistryError::InsufficientMemory { .. }));
}

#[test]
fn registry_no_capacity_set_allows_any() {
    let reg = ModelRegistry::new();
    // No capacity set — should allow any size
    reg.load_model(&load_req("m1", "gpu0", u64::MAX / 2)).unwrap();
    assert_eq!(reg.len(), 1);
}

#[test]
fn registry_memory_freed_after_remove() {
    let reg = ModelRegistry::new();
    reg.set_device_capacity("gpu0", 5000);
    reg.load_model(&load_req("m1", "gpu0", 3000)).unwrap();
    assert_eq!(reg.used_memory("gpu0"), 3000);
    reg.remove("m1", "gpu0").unwrap();
    assert_eq!(reg.used_memory("gpu0"), 0);
    // Can now load again
    reg.load_model(&load_req("m2", "gpu0", 5000)).unwrap();
}

// ─── ModelRegistry — Clone shares state ─────────────────────────────

#[test]
fn registry_clone_shares_state() {
    let reg = ModelRegistry::new();
    let cloned = reg.clone();
    reg.load_model(&load_req("m1", "gpu0", 100)).unwrap();
    // Cloned registry sees the same data (Arc<RwLock>)
    assert_eq!(cloned.len(), 1);
    assert!(!cloned.is_empty());
}

// ─── ModelRegistry — Zero memory model ──────────────────────────────

#[test]
fn registry_zero_memory_model() {
    let reg = ModelRegistry::new();
    reg.set_device_capacity("gpu0", 1000);
    reg.load_model(&load_req("tiny", "gpu0", 0)).unwrap();
    assert_eq!(reg.used_memory("gpu0"), 0);
}

// ─── ModelRegistry — Multiple devices independence ──────────────────

#[test]
fn registry_memory_per_device_independent() {
    let reg = ModelRegistry::new();
    reg.set_device_capacity("gpu0", 5000);
    reg.set_device_capacity("gpu1", 3000);
    reg.load_model(&load_req("m1", "gpu0", 4000)).unwrap();
    reg.load_model(&load_req("m2", "gpu1", 2000)).unwrap();
    assert_eq!(reg.used_memory("gpu0"), 4000);
    assert_eq!(reg.used_memory("gpu1"), 2000);
    // gpu0 has 1000 left, gpu1 has 1000 left
    let err = reg.load_model(&load_req("m3", "gpu0", 2000)).unwrap_err();
    assert!(matches!(err, RegistryError::InsufficientMemory { .. }));
    reg.load_model(&load_req("m3", "gpu1", 1000)).unwrap();
}
