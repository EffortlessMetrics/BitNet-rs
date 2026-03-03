//! GGUF metadata builder.
//!
//! Construct and manipulate GGUF metadata key-value pairs.

use std::collections::HashMap;

/// GGUF metadata value types.
#[derive(Debug, Clone, PartialEq)]
pub enum MetaValue {
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    String(String),
    U64(u64),
    F64(f64),
    Array(Vec<MetaValue>),
}

impl MetaValue {
    pub fn as_u32(&self) -> Option<u32> {
        if let Self::U32(v) = self { Some(*v) } else { None }
    }
    pub fn as_i32(&self) -> Option<i32> {
        if let Self::I32(v) = self { Some(*v) } else { None }
    }
    pub fn as_f32(&self) -> Option<f32> {
        if let Self::F32(v) = self { Some(*v) } else { None }
    }
    pub fn as_bool(&self) -> Option<bool> {
        if let Self::Bool(v) = self { Some(*v) } else { None }
    }
    pub fn as_str(&self) -> Option<&str> {
        if let Self::String(v) = self { Some(v) } else { None }
    }
    pub fn as_u64(&self) -> Option<u64> {
        if let Self::U64(v) = self { Some(*v) } else { None }
    }

    pub fn type_name(&self) -> &'static str {
        match self {
            Self::U32(_) => "u32",
            Self::I32(_) => "i32",
            Self::F32(_) => "f32",
            Self::Bool(_) => "bool",
            Self::String(_) => "string",
            Self::U64(_) => "u64",
            Self::F64(_) => "f64",
            Self::Array(_) => "array",
        }
    }
}

/// GGUF metadata builder.
#[derive(Debug, Clone)]
pub struct GgufMetadataBuilder {
    entries: HashMap<String, MetaValue>,
}

impl Default for GgufMetadataBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl GgufMetadataBuilder {
    pub fn new() -> Self {
        Self { entries: HashMap::new() }
    }

    pub fn set(&mut self, key: &str, value: MetaValue) -> &mut Self {
        self.entries.insert(key.to_string(), value);
        self
    }

    pub fn set_string(&mut self, key: &str, value: &str) -> &mut Self {
        self.set(key, MetaValue::String(value.to_string()))
    }

    pub fn set_u32(&mut self, key: &str, value: u32) -> &mut Self {
        self.set(key, MetaValue::U32(value))
    }

    pub fn set_f32(&mut self, key: &str, value: f32) -> &mut Self {
        self.set(key, MetaValue::F32(value))
    }

    pub fn set_bool(&mut self, key: &str, value: bool) -> &mut Self {
        self.set(key, MetaValue::Bool(value))
    }

    pub fn get(&self, key: &str) -> Option<&MetaValue> {
        self.entries.get(key)
    }
    pub fn contains(&self, key: &str) -> bool {
        self.entries.contains_key(key)
    }
    pub fn count(&self) -> usize {
        self.entries.len()
    }
    pub fn keys(&self) -> Vec<&str> {
        self.entries.keys().map(|k| k.as_str()).collect()
    }

    pub fn remove(&mut self, key: &str) -> Option<MetaValue> {
        self.entries.remove(key)
    }

    /// Set standard architecture metadata.
    pub fn set_architecture(&mut self, arch: &str) -> &mut Self {
        self.set_string("general.architecture", arch)
    }

    pub fn set_name(&mut self, name: &str) -> &mut Self {
        self.set_string("general.name", name)
    }

    /// Build standard Phi-4 metadata.
    pub fn phi4() -> Self {
        let mut b = Self::new();
        b.set_architecture("phi");
        b.set_name("phi-4");
        b.set_u32("phi.block_count", 40);
        b.set_u32("phi.embedding_length", 5120);
        b.set_u32("phi.attention.head_count", 40);
        b.set_u32("phi.attention.head_count_kv", 10);
        b.set_u32("phi.context_length", 16384);
        b.set_f32("phi.attention.layer_norm_rms_epsilon", 1e-5);
        b
    }

    /// Build standard BitNet metadata.
    pub fn bitnet_2b() -> Self {
        let mut b = Self::new();
        b.set_architecture("bitnet");
        b.set_name("bitnet-b1.58-2B-4T");
        b.set_u32("bitnet.block_count", 30);
        b.set_u32("bitnet.embedding_length", 2560);
        b.set_u32("bitnet.attention.head_count", 20);
        b.set_u32("bitnet.attention.head_count_kv", 5);
        b.set_u32("bitnet.context_length", 4096);
        b
    }

    pub fn build(self) -> HashMap<String, MetaValue> {
        self.entries
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_builder() {
        let b = GgufMetadataBuilder::new();
        assert_eq!(b.count(), 0);
    }

    #[test]
    fn test_set_get() {
        let mut b = GgufMetadataBuilder::new();
        b.set_string("key", "value");
        assert_eq!(b.get("key").unwrap().as_str(), Some("value"));
    }

    #[test]
    fn test_set_u32() {
        let mut b = GgufMetadataBuilder::new();
        b.set_u32("layers", 40);
        assert_eq!(b.get("layers").unwrap().as_u32(), Some(40));
    }

    #[test]
    fn test_set_f32() {
        let mut b = GgufMetadataBuilder::new();
        b.set_f32("eps", 1e-5);
        assert!(b.get("eps").unwrap().as_f32().is_some());
    }

    #[test]
    fn test_set_bool() {
        let mut b = GgufMetadataBuilder::new();
        b.set_bool("flag", true);
        assert_eq!(b.get("flag").unwrap().as_bool(), Some(true));
    }

    #[test]
    fn test_phi4() {
        let b = GgufMetadataBuilder::phi4();
        assert!(b.count() >= 7);
        assert_eq!(b.get("general.architecture").unwrap().as_str(), Some("phi"));
        assert_eq!(b.get("phi.block_count").unwrap().as_u32(), Some(40));
    }

    #[test]
    fn test_bitnet() {
        let b = GgufMetadataBuilder::bitnet_2b();
        assert_eq!(b.get("general.architecture").unwrap().as_str(), Some("bitnet"));
        assert_eq!(b.get("bitnet.block_count").unwrap().as_u32(), Some(30));
    }

    #[test]
    fn test_contains() {
        let mut b = GgufMetadataBuilder::new();
        b.set_string("test", "val");
        assert!(b.contains("test"));
        assert!(!b.contains("missing"));
    }

    #[test]
    fn test_remove() {
        let mut b = GgufMetadataBuilder::new();
        b.set_string("key", "val");
        b.remove("key");
        assert!(!b.contains("key"));
    }

    #[test]
    fn test_type_name() {
        assert_eq!(MetaValue::U32(0).type_name(), "u32");
        assert_eq!(MetaValue::String("".into()).type_name(), "string");
        assert_eq!(MetaValue::Array(vec![]).type_name(), "array");
    }

    #[test]
    fn test_build() {
        let mut b = GgufMetadataBuilder::new();
        b.set_string("a", "1");
        b.set_u32("b", 2);
        let map = b.build();
        assert_eq!(map.len(), 2);
    }

    #[test]
    fn test_default() {
        let b = GgufMetadataBuilder::default();
        assert_eq!(b.count(), 0);
    }
}
