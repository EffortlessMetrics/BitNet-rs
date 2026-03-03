//! Runtime model registry.
//!
//! Track loaded models, their status, and capabilities for serving.

use std::collections::HashMap;
use std::time::Instant;

/// Model loading status.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelStatus {
    Pending,
    Loading,
    Ready,
    Failed,
    Unloaded,
}

/// Information about a loaded model.
#[derive(Debug, Clone)]
pub struct ModelInfo {
    pub id: String,
    pub name: String,
    pub status: ModelStatus,
    pub param_count: Option<usize>,
    pub context_length: Option<usize>,
    pub loaded_at: Option<Instant>,
    pub request_count: u64,
    pub error_message: Option<String>,
}

impl ModelInfo {
    pub fn new(id: impl Into<String>, name: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            status: ModelStatus::Pending,
            param_count: None,
            context_length: None,
            loaded_at: None,
            request_count: 0,
            error_message: None,
        }
    }

    pub fn is_ready(&self) -> bool {
        self.status == ModelStatus::Ready
    }
    pub fn is_failed(&self) -> bool {
        self.status == ModelStatus::Failed
    }

    pub fn mark_loading(&mut self) {
        self.status = ModelStatus::Loading;
    }

    pub fn mark_ready(&mut self) {
        self.status = ModelStatus::Ready;
        self.loaded_at = Some(Instant::now());
        self.error_message = None;
    }

    pub fn mark_failed(&mut self, msg: impl Into<String>) {
        self.status = ModelStatus::Failed;
        self.error_message = Some(msg.into());
    }

    pub fn mark_unloaded(&mut self) {
        self.status = ModelStatus::Unloaded;
        self.loaded_at = None;
    }

    pub fn record_request(&mut self) {
        self.request_count += 1;
    }
}

/// Registry of models.
#[derive(Debug)]
pub struct RuntimeModelRegistry {
    models: HashMap<String, ModelInfo>,
    default_model: Option<String>,
}

impl RuntimeModelRegistry {
    pub fn new() -> Self {
        Self { models: HashMap::new(), default_model: None }
    }

    pub fn register(&mut self, info: ModelInfo) {
        let id = info.id.clone();
        if self.default_model.is_none() {
            self.default_model = Some(id.clone());
        }
        self.models.insert(id, info);
    }

    pub fn unregister(&mut self, id: &str) -> Option<ModelInfo> {
        let info = self.models.remove(id);
        if self.default_model.as_deref() == Some(id) {
            self.default_model = self.models.keys().next().cloned();
        }
        info
    }

    pub fn get(&self, id: &str) -> Option<&ModelInfo> {
        self.models.get(id)
    }
    pub fn get_mut(&mut self, id: &str) -> Option<&mut ModelInfo> {
        self.models.get_mut(id)
    }

    pub fn set_default(&mut self, id: &str) -> bool {
        if self.models.contains_key(id) {
            self.default_model = Some(id.to_string());
            true
        } else {
            false
        }
    }

    pub fn default_model(&self) -> Option<&ModelInfo> {
        self.default_model.as_ref().and_then(|id| self.models.get(id))
    }

    pub fn model_count(&self) -> usize {
        self.models.len()
    }

    pub fn ready_models(&self) -> Vec<&ModelInfo> {
        self.models.values().filter(|m| m.is_ready()).collect()
    }

    pub fn all_models(&self) -> Vec<&ModelInfo> {
        self.models.values().collect()
    }

    pub fn model_ids(&self) -> Vec<&str> {
        self.models.keys().map(|s| s.as_str()).collect()
    }

    pub fn total_requests(&self) -> u64 {
        self.models.values().map(|m| m.request_count).sum()
    }
}

impl Default for RuntimeModelRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_model(id: &str) -> ModelInfo {
        ModelInfo::new(id, format!("Model {id}"))
    }

    #[test]
    fn test_register_model() {
        let mut reg = RuntimeModelRegistry::new();
        reg.register(sample_model("m1"));
        assert_eq!(reg.model_count(), 1);
        assert!(reg.get("m1").is_some());
    }

    #[test]
    fn test_first_model_is_default() {
        let mut reg = RuntimeModelRegistry::new();
        reg.register(sample_model("m1"));
        assert_eq!(reg.default_model().unwrap().id, "m1");
    }

    #[test]
    fn test_set_default() {
        let mut reg = RuntimeModelRegistry::new();
        reg.register(sample_model("m1"));
        reg.register(sample_model("m2"));
        assert!(reg.set_default("m2"));
        assert_eq!(reg.default_model().unwrap().id, "m2");
    }

    #[test]
    fn test_set_default_invalid() {
        let mut reg = RuntimeModelRegistry::new();
        assert!(!reg.set_default("nonexistent"));
    }

    #[test]
    fn test_model_lifecycle() {
        let mut info = sample_model("m1");
        assert_eq!(info.status, ModelStatus::Pending);
        info.mark_loading();
        assert_eq!(info.status, ModelStatus::Loading);
        info.mark_ready();
        assert!(info.is_ready());
        info.mark_unloaded();
        assert_eq!(info.status, ModelStatus::Unloaded);
    }

    #[test]
    fn test_model_failed() {
        let mut info = sample_model("m1");
        info.mark_failed("out of memory");
        assert!(info.is_failed());
        assert_eq!(info.error_message.as_deref(), Some("out of memory"));
    }

    #[test]
    fn test_ready_models() {
        let mut reg = RuntimeModelRegistry::new();
        let mut m1 = sample_model("m1");
        m1.mark_ready();
        let m2 = sample_model("m2");
        reg.register(m1);
        reg.register(m2);
        assert_eq!(reg.ready_models().len(), 1);
    }

    #[test]
    fn test_unregister() {
        let mut reg = RuntimeModelRegistry::new();
        reg.register(sample_model("m1"));
        reg.register(sample_model("m2"));
        reg.unregister("m1");
        assert_eq!(reg.model_count(), 1);
        assert!(reg.default_model().is_some());
    }

    #[test]
    fn test_request_tracking() {
        let mut reg = RuntimeModelRegistry::new();
        reg.register(sample_model("m1"));
        reg.get_mut("m1").unwrap().record_request();
        reg.get_mut("m1").unwrap().record_request();
        assert_eq!(reg.total_requests(), 2);
    }

    #[test]
    fn test_model_ids() {
        let mut reg = RuntimeModelRegistry::new();
        reg.register(sample_model("a"));
        reg.register(sample_model("b"));
        let ids = reg.model_ids();
        assert!(ids.contains(&"a"));
        assert!(ids.contains(&"b"));
    }

    #[test]
    fn test_empty_registry() {
        let reg = RuntimeModelRegistry::new();
        assert_eq!(reg.model_count(), 0);
        assert!(reg.default_model().is_none());
        assert_eq!(reg.total_requests(), 0);
    }

    #[test]
    fn test_unregister_default_resets() {
        let mut reg = RuntimeModelRegistry::new();
        reg.register(sample_model("m1"));
        reg.register(sample_model("m2"));
        reg.unregister("m1");
        // Default should be reassigned
        assert!(reg.default_model().is_some());
    }
}
