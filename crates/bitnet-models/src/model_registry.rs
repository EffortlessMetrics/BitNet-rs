//! Model registry for tracking loaded and available models.
//!
//! Register, query, and manage model metadata: paths, sizes,
//! capabilities, and lifecycle states.

use std::collections::HashMap;

/// Model lifecycle state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelState {
    Available,
    Loading,
    Ready,
    Unloading,
    Failed,
}

impl ModelState {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Available => "available",
            Self::Loading => "loading",
            Self::Ready => "ready",
            Self::Unloading => "unloading",
            Self::Failed => "failed",
        }
    }

    pub fn is_usable(&self) -> bool {
        *self == Self::Ready
    }
}

/// Model metadata entry in the registry.
#[derive(Debug, Clone)]
pub struct ModelEntry {
    pub id: String,
    pub name: String,
    pub path: String,
    pub format: String,
    pub size_bytes: u64,
    pub param_count: Option<u64>,
    pub state: ModelState,
    pub tags: Vec<String>,
}

impl ModelEntry {
    pub fn new(id: impl Into<String>, name: impl Into<String>, path: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            path: path.into(),
            format: String::new(),
            size_bytes: 0,
            param_count: None,
            state: ModelState::Available,
            tags: vec![],
        }
    }

    pub fn with_format(mut self, format: impl Into<String>) -> Self {
        self.format = format.into();
        self
    }

    pub fn with_size(mut self, bytes: u64) -> Self {
        self.size_bytes = bytes;
        self
    }

    pub fn with_params(mut self, count: u64) -> Self {
        self.param_count = Some(count);
        self
    }

    pub fn with_tag(mut self, tag: impl Into<String>) -> Self {
        self.tags.push(tag.into());
        self
    }

    pub fn has_tag(&self, tag: &str) -> bool {
        self.tags.iter().any(|t| t == tag)
    }
}

/// Model registry: tracks available and loaded models.
#[derive(Debug, Default)]
pub struct ModelRegistry {
    models: HashMap<String, ModelEntry>,
}

impl ModelRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register(&mut self, entry: ModelEntry) {
        self.models.insert(entry.id.clone(), entry);
    }

    pub fn get(&self, id: &str) -> Option<&ModelEntry> {
        self.models.get(id)
    }

    pub fn get_mut(&mut self, id: &str) -> Option<&mut ModelEntry> {
        self.models.get_mut(id)
    }

    pub fn remove(&mut self, id: &str) -> Option<ModelEntry> {
        self.models.remove(id)
    }

    pub fn contains(&self, id: &str) -> bool {
        self.models.contains_key(id)
    }

    pub fn count(&self) -> usize {
        self.models.len()
    }

    pub fn list(&self) -> Vec<&ModelEntry> {
        self.models.values().collect()
    }

    /// Find models by tag.
    pub fn find_by_tag(&self, tag: &str) -> Vec<&ModelEntry> {
        self.models.values().filter(|m| m.has_tag(tag)).collect()
    }

    /// Find models in a specific state.
    pub fn find_by_state(&self, state: ModelState) -> Vec<&ModelEntry> {
        self.models.values().filter(|m| m.state == state).collect()
    }

    /// Get all ready (usable) models.
    pub fn ready_models(&self) -> Vec<&ModelEntry> {
        self.find_by_state(ModelState::Ready)
    }

    /// Set model state.
    pub fn set_state(&mut self, id: &str, state: ModelState) -> bool {
        if let Some(entry) = self.models.get_mut(id) {
            entry.state = state;
            true
        } else {
            false
        }
    }

    /// Total size of all registered models.
    pub fn total_size(&self) -> u64 {
        self.models.values().map(|m| m.size_bytes).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_register_and_get() {
        let mut reg = ModelRegistry::new();
        reg.register(ModelEntry::new("m1", "Model 1", "/path/m1.gguf"));
        assert!(reg.contains("m1"));
        assert_eq!(reg.get("m1").unwrap().name, "Model 1");
    }

    #[test]
    fn test_remove() {
        let mut reg = ModelRegistry::new();
        reg.register(ModelEntry::new("m1", "Model 1", "/path"));
        assert_eq!(reg.count(), 1);
        reg.remove("m1");
        assert_eq!(reg.count(), 0);
    }

    #[test]
    fn test_find_by_tag() {
        let mut reg = ModelRegistry::new();
        reg.register(ModelEntry::new("m1", "M1", "/p").with_tag("slm"));
        reg.register(ModelEntry::new("m2", "M2", "/p").with_tag("llm"));
        let slms = reg.find_by_tag("slm");
        assert_eq!(slms.len(), 1);
        assert_eq!(slms[0].id, "m1");
    }

    #[test]
    fn test_find_by_state() {
        let mut reg = ModelRegistry::new();
        reg.register(ModelEntry::new("m1", "M1", "/p"));
        reg.set_state("m1", ModelState::Ready);
        assert_eq!(reg.ready_models().len(), 1);
    }

    #[test]
    fn test_set_state() {
        let mut reg = ModelRegistry::new();
        reg.register(ModelEntry::new("m1", "M1", "/p"));
        assert!(reg.set_state("m1", ModelState::Loading));
        assert_eq!(reg.get("m1").unwrap().state, ModelState::Loading);
        assert!(!reg.set_state("nonexistent", ModelState::Ready));
    }

    #[test]
    fn test_total_size() {
        let mut reg = ModelRegistry::new();
        reg.register(ModelEntry::new("m1", "M1", "/p").with_size(1000));
        reg.register(ModelEntry::new("m2", "M2", "/p").with_size(2000));
        assert_eq!(reg.total_size(), 3000);
    }

    #[test]
    fn test_model_entry_builder() {
        let entry = ModelEntry::new("id", "name", "/path")
            .with_format("gguf")
            .with_size(1024)
            .with_params(2_000_000_000)
            .with_tag("slm");
        assert_eq!(entry.format, "gguf");
        assert_eq!(entry.size_bytes, 1024);
        assert_eq!(entry.param_count, Some(2_000_000_000));
        assert!(entry.has_tag("slm"));
    }

    #[test]
    fn test_model_state() {
        assert!(ModelState::Ready.is_usable());
        assert!(!ModelState::Loading.is_usable());
        assert!(!ModelState::Failed.is_usable());
    }

    #[test]
    fn test_empty_registry() {
        let reg = ModelRegistry::new();
        assert_eq!(reg.count(), 0);
        assert!(reg.list().is_empty());
        assert_eq!(reg.total_size(), 0);
    }

    #[test]
    fn test_list() {
        let mut reg = ModelRegistry::new();
        reg.register(ModelEntry::new("a", "A", "/a"));
        reg.register(ModelEntry::new("b", "B", "/b"));
        assert_eq!(reg.list().len(), 2);
    }

    #[test]
    fn test_state_as_str() {
        assert_eq!(ModelState::Ready.as_str(), "ready");
        assert_eq!(ModelState::Failed.as_str(), "failed");
    }

    #[test]
    fn test_has_tag_negative() {
        let entry = ModelEntry::new("id", "name", "/path");
        assert!(!entry.has_tag("nonexistent"));
    }
}
