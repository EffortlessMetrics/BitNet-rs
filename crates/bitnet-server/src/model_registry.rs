//! Server model registry with GPU affinity and lifecycle tracking.
//!
//! [`ModelRegistry`] tracks which models are loaded on which GPU devices,
//! manages model lifecycle transitions, and accounts for VRAM usage so
//! that routing and admission decisions can be made correctly.

use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::Instant;

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// Unique model identifier (e.g. "bitnet-2b-4t").
pub type ModelId = String;

/// Unique device identifier matching [`super::gpu_router::DeviceId`].
pub type DeviceId = String;

/// Lifecycle state of a loaded model instance.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelState {
    /// Model is being loaded into device memory.
    Loading,
    /// Model is loaded and ready to serve requests.
    Ready,
    /// Model is actively serving inference requests.
    Serving,
    /// Model is being unloaded from device memory.
    Unloading,
}

impl std::fmt::Display for ModelState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ModelState::Loading => write!(f, "loading"),
            ModelState::Ready => write!(f, "ready"),
            ModelState::Serving => write!(f, "serving"),
            ModelState::Unloading => write!(f, "unloading"),
        }
    }
}

/// Descriptor for a model loaded (or being loaded) on a device.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelEntry {
    pub model_id: ModelId,
    pub device_id: DeviceId,
    pub state: ModelState,
    /// Estimated VRAM consumed by this model in bytes.
    pub memory_bytes: u64,
    /// When the model entered its current state.
    #[serde(skip)]
    pub state_changed_at: Option<Instant>,
}

// ---------------------------------------------------------------------------
// REST-style API types
// ---------------------------------------------------------------------------

/// Response for listing loaded models.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ListModelsResponse {
    pub models: Vec<ModelEntry>,
}

/// Request to load a model onto a device.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadModelRequest {
    pub model_id: ModelId,
    pub device_id: DeviceId,
    /// Expected VRAM usage in bytes.
    pub memory_bytes: u64,
}

/// Request to unload a model from a device.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnloadModelRequest {
    pub model_id: ModelId,
    pub device_id: DeviceId,
}

/// Errors from registry operations.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RegistryError {
    /// Model is already loaded on the specified device.
    AlreadyLoaded { model_id: ModelId, device_id: DeviceId },
    /// Model not found on the specified device.
    NotFound { model_id: ModelId, device_id: DeviceId },
    /// Invalid state transition.
    InvalidTransition { from: ModelState, to: ModelState },
    /// Insufficient GPU memory on the target device.
    InsufficientMemory { device_id: DeviceId, required: u64, available: u64 },
}

impl std::fmt::Display for RegistryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RegistryError::AlreadyLoaded { model_id, device_id } => {
                write!(f, "model '{model_id}' already loaded on '{device_id}'")
            }
            RegistryError::NotFound { model_id, device_id } => {
                write!(f, "model '{model_id}' not found on '{device_id}'")
            }
            RegistryError::InvalidTransition { from, to } => {
                write!(f, "invalid transition {from} -> {to}")
            }
            RegistryError::InsufficientMemory { device_id, required, available } => write!(
                f,
                "device '{device_id}': need {required} B, \
                 have {available} B",
            ),
        }
    }
}

impl std::error::Error for RegistryError {}

// ---------------------------------------------------------------------------
// ModelRegistry
// ---------------------------------------------------------------------------

/// Composite key: (model_id, device_id).
type EntryKey = (ModelId, DeviceId);

/// Thread-safe registry of models loaded across GPU devices.
#[derive(Clone)]
pub struct ModelRegistry {
    inner: Arc<RwLock<RegistryInner>>,
}

struct RegistryInner {
    entries: HashMap<EntryKey, ModelEntry>,
    /// Per-device memory capacity (set via `set_device_capacity`).
    device_capacity: HashMap<DeviceId, u64>,
}

impl ModelRegistry {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(RwLock::new(RegistryInner {
                entries: HashMap::new(),
                device_capacity: HashMap::new(),
            })),
        }
    }

    /// Register a device's total VRAM capacity for memory accounting.
    pub fn set_device_capacity(&self, device_id: &str, capacity_bytes: u64) {
        let mut inner = self.inner.write().unwrap();
        inner.device_capacity.insert(device_id.to_string(), capacity_bytes);
    }

    /// Begin loading a model onto a device.
    pub fn load_model(&self, req: &LoadModelRequest) -> Result<(), RegistryError> {
        let mut inner = self.inner.write().unwrap();
        let key = (req.model_id.clone(), req.device_id.clone());

        if inner.entries.contains_key(&key) {
            return Err(RegistryError::AlreadyLoaded {
                model_id: req.model_id.clone(),
                device_id: req.device_id.clone(),
            });
        }

        // Memory check
        if let Some(&cap) = inner.device_capacity.get(&req.device_id) {
            let used: u64 = inner
                .entries
                .values()
                .filter(|e| e.device_id == req.device_id)
                .map(|e| e.memory_bytes)
                .sum();
            let avail = cap.saturating_sub(used);
            if req.memory_bytes > avail {
                return Err(RegistryError::InsufficientMemory {
                    device_id: req.device_id.clone(),
                    required: req.memory_bytes,
                    available: avail,
                });
            }
        }

        inner.entries.insert(
            key,
            ModelEntry {
                model_id: req.model_id.clone(),
                device_id: req.device_id.clone(),
                state: ModelState::Loading,
                memory_bytes: req.memory_bytes,
                state_changed_at: Some(Instant::now()),
            },
        );
        Ok(())
    }

    /// Transition a model to a new state.
    pub fn set_state(
        &self,
        model_id: &str,
        device_id: &str,
        new_state: ModelState,
    ) -> Result<(), RegistryError> {
        let mut inner = self.inner.write().unwrap();
        let key = (model_id.to_string(), device_id.to_string());
        let entry = inner.entries.get_mut(&key).ok_or_else(|| RegistryError::NotFound {
            model_id: model_id.to_string(),
            device_id: device_id.to_string(),
        })?;

        // Validate transitions
        let valid = matches!(
            (entry.state, new_state),
            (ModelState::Loading, ModelState::Ready)
                | (ModelState::Ready, ModelState::Serving)
                | (ModelState::Serving, ModelState::Ready)
                | (ModelState::Ready, ModelState::Unloading)
                | (ModelState::Serving, ModelState::Unloading)
                | (ModelState::Loading, ModelState::Unloading)
        );

        if !valid {
            return Err(RegistryError::InvalidTransition { from: entry.state, to: new_state });
        }

        entry.state = new_state;
        entry.state_changed_at = Some(Instant::now());
        Ok(())
    }

    /// Remove a model entry (after unloading completes).
    pub fn remove(&self, model_id: &str, device_id: &str) -> Result<ModelEntry, RegistryError> {
        let mut inner = self.inner.write().unwrap();
        let key = (model_id.to_string(), device_id.to_string());
        inner.entries.remove(&key).ok_or_else(|| RegistryError::NotFound {
            model_id: model_id.to_string(),
            device_id: device_id.to_string(),
        })
    }

    /// Unload a model: transitions to `Unloading` then removes.
    pub fn unload_model(&self, req: &UnloadModelRequest) -> Result<ModelEntry, RegistryError> {
        self.set_state(&req.model_id, &req.device_id, ModelState::Unloading)?;
        self.remove(&req.model_id, &req.device_id)
    }

    /// List all model entries.
    pub fn list_models(&self) -> ListModelsResponse {
        let inner = self.inner.read().unwrap();
        ListModelsResponse { models: inner.entries.values().cloned().collect() }
    }

    /// List models on a specific device.
    pub fn models_on_device(&self, device_id: &str) -> Vec<ModelEntry> {
        let inner = self.inner.read().unwrap();
        inner.entries.values().filter(|e| e.device_id == device_id).cloned().collect()
    }

    /// Get a specific model entry.
    pub fn get(&self, model_id: &str, device_id: &str) -> Option<ModelEntry> {
        let inner = self.inner.read().unwrap();
        inner.entries.get(&(model_id.to_string(), device_id.to_string())).cloned()
    }

    /// Total VRAM used by models on a device.
    pub fn used_memory(&self, device_id: &str) -> u64 {
        let inner = self.inner.read().unwrap();
        inner.entries.values().filter(|e| e.device_id == device_id).map(|e| e.memory_bytes).sum()
    }

    /// Number of entries in the registry.
    pub fn len(&self) -> usize {
        self.inner.read().unwrap().entries.len()
    }

    /// Whether the registry is empty.
    pub fn is_empty(&self) -> bool {
        self.inner.read().unwrap().entries.is_empty()
    }
}

impl Default for ModelRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// =========================================================================
// Tests
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn lr(model: &str, dev: &str, mem: u64) -> LoadModelRequest {
        LoadModelRequest { model_id: model.into(), device_id: dev.into(), memory_bytes: mem }
    }

    #[test]
    fn load_and_list() {
        let reg = ModelRegistry::new();
        reg.load_model(&lr("m1", "gpu0", 1_000)).unwrap();
        let list = reg.list_models();
        assert_eq!(list.models.len(), 1);
        assert_eq!(list.models[0].state, ModelState::Loading);
    }

    #[test]
    fn duplicate_load_rejected() {
        let reg = ModelRegistry::new();
        reg.load_model(&lr("m1", "gpu0", 1_000)).unwrap();
        let err = reg.load_model(&lr("m1", "gpu0", 1_000)).unwrap_err();
        assert!(matches!(err, RegistryError::AlreadyLoaded { .. }));
    }

    #[test]
    fn same_model_different_devices() {
        let reg = ModelRegistry::new();
        reg.load_model(&lr("m1", "gpu0", 1_000)).unwrap();
        reg.load_model(&lr("m1", "gpu1", 1_000)).unwrap();
        assert_eq!(reg.len(), 2);
    }

    #[test]
    fn state_lifecycle() {
        let reg = ModelRegistry::new();
        reg.load_model(&lr("m1", "gpu0", 1_000)).unwrap();
        reg.set_state("m1", "gpu0", ModelState::Ready).unwrap();
        assert_eq!(reg.get("m1", "gpu0").unwrap().state, ModelState::Ready,);
        reg.set_state("m1", "gpu0", ModelState::Serving).unwrap();
        assert_eq!(reg.get("m1", "gpu0").unwrap().state, ModelState::Serving,);
        reg.set_state("m1", "gpu0", ModelState::Ready).unwrap();
        assert_eq!(reg.get("m1", "gpu0").unwrap().state, ModelState::Ready,);
    }

    #[test]
    fn invalid_transition_rejected() {
        let reg = ModelRegistry::new();
        reg.load_model(&lr("m1", "gpu0", 1_000)).unwrap();
        let err = reg.set_state("m1", "gpu0", ModelState::Serving).unwrap_err();
        assert!(matches!(err, RegistryError::InvalidTransition { .. }));
    }

    #[test]
    fn unload_model() {
        let reg = ModelRegistry::new();
        reg.load_model(&lr("m1", "gpu0", 1_000)).unwrap();
        reg.set_state("m1", "gpu0", ModelState::Ready).unwrap();
        let req = UnloadModelRequest { model_id: "m1".into(), device_id: "gpu0".into() };
        let entry = reg.unload_model(&req).unwrap();
        assert_eq!(entry.model_id, "m1");
        assert!(reg.is_empty());
    }

    #[test]
    fn unload_not_found() {
        let reg = ModelRegistry::new();
        let req = UnloadModelRequest { model_id: "m1".into(), device_id: "gpu0".into() };
        let err = reg.unload_model(&req).unwrap_err();
        assert!(matches!(err, RegistryError::NotFound { .. }));
    }

    #[test]
    fn memory_accounting() {
        let reg = ModelRegistry::new();
        reg.load_model(&lr("m1", "gpu0", 4_000_000_000)).unwrap();
        reg.load_model(&lr("m2", "gpu0", 2_000_000_000)).unwrap();
        assert_eq!(reg.used_memory("gpu0"), 6_000_000_000);
        assert_eq!(reg.used_memory("gpu1"), 0);
    }

    #[test]
    fn memory_capacity_enforcement() {
        let reg = ModelRegistry::new();
        reg.set_device_capacity("gpu0", 8_000_000_000);
        reg.load_model(&lr("m1", "gpu0", 6_000_000_000)).unwrap();
        let err = reg.load_model(&lr("m2", "gpu0", 4_000_000_000)).unwrap_err();
        assert!(matches!(err, RegistryError::InsufficientMemory { .. }));
    }

    #[test]
    fn models_on_device() {
        let reg = ModelRegistry::new();
        reg.load_model(&lr("m1", "gpu0", 1_000)).unwrap();
        reg.load_model(&lr("m2", "gpu0", 2_000)).unwrap();
        reg.load_model(&lr("m3", "gpu1", 3_000)).unwrap();
        assert_eq!(reg.models_on_device("gpu0").len(), 2);
        assert_eq!(reg.models_on_device("gpu1").len(), 1);
    }

    #[test]
    fn remove_frees_memory() {
        let reg = ModelRegistry::new();
        reg.set_device_capacity("gpu0", 8_000_000_000);
        reg.load_model(&lr("m1", "gpu0", 6_000_000_000)).unwrap();
        reg.set_state("m1", "gpu0", ModelState::Ready).unwrap();
        reg.unload_model(&UnloadModelRequest { model_id: "m1".into(), device_id: "gpu0".into() })
            .unwrap();
        reg.load_model(&lr("m2", "gpu0", 7_000_000_000)).unwrap();
        assert_eq!(reg.len(), 1);
    }

    #[test]
    fn get_nonexistent_returns_none() {
        let reg = ModelRegistry::new();
        assert!(reg.get("nope", "gpu0").is_none());
    }
}
