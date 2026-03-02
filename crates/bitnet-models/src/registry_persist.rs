//! Model registry persistence.
//!
//! Serialize and deserialize model registry entries.

use std::collections::HashMap;

/// Serializable model entry.
#[derive(Debug, Clone, PartialEq)]
pub struct ModelEntry {
    pub id: String,
    pub name: String,
    pub family: String,
    pub version: String,
    pub path: String,
    pub format: String,
    pub size_bytes: u64,
    pub metadata: HashMap<String, String>,
}

impl ModelEntry {
    pub fn new(id: &str, name: &str, family: &str) -> Self {
        Self {
            id: id.to_string(),
            name: name.to_string(),
            family: family.to_string(),
            version: String::new(),
            path: String::new(),
            format: String::new(),
            size_bytes: 0,
            metadata: HashMap::new(),
        }
    }

    pub fn with_path(mut self, path: &str) -> Self {
        self.path = path.to_string();
        self
    }
    pub fn with_format(mut self, format: &str) -> Self {
        self.format = format.to_string();
        self
    }
    pub fn with_size(mut self, size: u64) -> Self {
        self.size_bytes = size;
        self
    }
    pub fn with_version(mut self, version: &str) -> Self {
        self.version = version.to_string();
        self
    }
}

/// In-memory registry store.
#[derive(Debug, Clone)]
pub struct RegistryStore {
    entries: HashMap<String, ModelEntry>,
}

impl Default for RegistryStore {
    fn default() -> Self {
        Self::new()
    }
}

impl RegistryStore {
    pub fn new() -> Self {
        Self { entries: HashMap::new() }
    }

    pub fn insert(&mut self, entry: ModelEntry) {
        self.entries.insert(entry.id.clone(), entry);
    }

    pub fn get(&self, id: &str) -> Option<&ModelEntry> {
        self.entries.get(id)
    }

    pub fn remove(&mut self, id: &str) -> bool {
        self.entries.remove(id).is_some()
    }

    pub fn count(&self) -> usize {
        self.entries.len()
    }

    pub fn by_family(&self, family: &str) -> Vec<&ModelEntry> {
        self.entries.values().filter(|e| e.family == family).collect()
    }

    pub fn all(&self) -> Vec<&ModelEntry> {
        self.entries.values().collect()
    }

    pub fn ids(&self) -> Vec<&str> {
        self.entries.keys().map(|k| k.as_str()).collect()
    }

    /// Serialize to simple key=value lines.
    pub fn serialize(&self) -> String {
        let mut lines = Vec::new();
        for entry in self.entries.values() {
            lines.push(format!("[{}]", entry.id));
            lines.push(format!("name={}", entry.name));
            lines.push(format!("family={}", entry.family));
            lines.push(format!("version={}", entry.version));
            lines.push(format!("path={}", entry.path));
            lines.push(format!("format={}", entry.format));
            lines.push(format!("size={}", entry.size_bytes));
            for (k, v) in &entry.metadata {
                lines.push(format!("meta.{k}={v}"));
            }
            lines.push(String::new());
        }
        lines.join("\n")
    }

    /// Deserialize from key=value lines.
    pub fn deserialize(data: &str) -> Self {
        let mut store = Self::new();
        let mut current: Option<ModelEntry> = None;

        for line in data.lines() {
            let line = line.trim();
            if line.is_empty() {
                if let Some(entry) = current.take() {
                    store.insert(entry);
                }
                continue;
            }
            if line.starts_with('[') && line.ends_with(']') {
                if let Some(entry) = current.take() {
                    store.insert(entry);
                }
                let id = &line[1..line.len() - 1];
                current = Some(ModelEntry::new(id, "", ""));
                continue;
            }
            if let Some(ref mut entry) = current {
                if let Some((key, value)) = line.split_once('=') {
                    match key {
                        "name" => entry.name = value.to_string(),
                        "family" => entry.family = value.to_string(),
                        "version" => entry.version = value.to_string(),
                        "path" => entry.path = value.to_string(),
                        "format" => entry.format = value.to_string(),
                        "size" => entry.size_bytes = value.parse().unwrap_or(0),
                        k if k.starts_with("meta.") => {
                            entry.metadata.insert(k[5..].to_string(), value.to_string());
                        }
                        _ => {}
                    }
                }
            }
        }
        if let Some(entry) = current {
            store.insert(entry);
        }
        store
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_store() {
        let s = RegistryStore::new();
        assert_eq!(s.count(), 0);
    }

    #[test]
    fn test_insert_get() {
        let mut s = RegistryStore::new();
        s.insert(ModelEntry::new("phi4", "Phi-4", "phi"));
        assert_eq!(s.get("phi4").unwrap().name, "Phi-4");
    }

    #[test]
    fn test_remove() {
        let mut s = RegistryStore::new();
        s.insert(ModelEntry::new("phi4", "Phi-4", "phi"));
        assert!(s.remove("phi4"));
        assert!(s.get("phi4").is_none());
    }

    #[test]
    fn test_by_family() {
        let mut s = RegistryStore::new();
        s.insert(ModelEntry::new("phi4", "Phi-4", "phi"));
        s.insert(ModelEntry::new("phi3", "Phi-3", "phi"));
        s.insert(ModelEntry::new("llama3", "LLaMA-3", "llama"));
        assert_eq!(s.by_family("phi").len(), 2);
    }

    #[test]
    fn test_builder_methods() {
        let entry = ModelEntry::new("test", "Test", "test")
            .with_path("/models/test.gguf")
            .with_format("gguf")
            .with_size(1000)
            .with_version("1.0");
        assert_eq!(entry.path, "/models/test.gguf");
        assert_eq!(entry.size_bytes, 1000);
    }

    #[test]
    fn test_serialize_deserialize() {
        let mut s = RegistryStore::new();
        let mut entry = ModelEntry::new("phi4", "Phi-4", "phi");
        entry.format = "gguf".into();
        entry.size_bytes = 29000000000;
        entry.metadata.insert("layers".into(), "40".into());
        s.insert(entry);

        let serialized = s.serialize();
        let restored = RegistryStore::deserialize(&serialized);
        assert_eq!(restored.count(), 1);
        let e = restored.get("phi4").unwrap();
        assert_eq!(e.name, "Phi-4");
        assert_eq!(e.size_bytes, 29000000000);
        assert_eq!(e.metadata.get("layers").unwrap(), "40");
    }

    #[test]
    fn test_deserialize_empty() {
        let s = RegistryStore::deserialize("");
        assert_eq!(s.count(), 0);
    }

    #[test]
    fn test_all() {
        let mut s = RegistryStore::new();
        s.insert(ModelEntry::new("a", "A", "x"));
        s.insert(ModelEntry::new("b", "B", "y"));
        assert_eq!(s.all().len(), 2);
    }

    #[test]
    fn test_ids() {
        let mut s = RegistryStore::new();
        s.insert(ModelEntry::new("alpha", "A", "x"));
        let ids = s.ids();
        assert!(ids.contains(&"alpha"));
    }

    #[test]
    fn test_default() {
        let s = RegistryStore::default();
        assert_eq!(s.count(), 0);
    }

    #[test]
    fn test_overwrite() {
        let mut s = RegistryStore::new();
        s.insert(ModelEntry::new("phi4", "Phi-4-old", "phi"));
        s.insert(ModelEntry::new("phi4", "Phi-4-new", "phi"));
        assert_eq!(s.count(), 1);
        assert_eq!(s.get("phi4").unwrap().name, "Phi-4-new");
    }
}
