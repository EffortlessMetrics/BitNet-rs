//! Model shard index for multi-file models.
//!
//! Tracks which tensors reside in which shard files.

use std::collections::HashMap;

/// A single shard file entry.
#[derive(Debug, Clone)]
pub struct ShardEntry {
    pub filename: String,
    pub byte_size: u64,
    pub tensor_count: usize,
}

/// Mapping of tensor name to shard file.
#[derive(Debug, Clone)]
pub struct TensorLocation {
    pub tensor_name: String,
    pub shard_filename: String,
    pub byte_offset: u64,
    pub byte_length: u64,
}

/// Index for a sharded model (like HuggingFace SafeTensors).
#[derive(Debug, Clone)]
pub struct ShardIndex {
    pub total_size: u64,
    pub shards: Vec<ShardEntry>,
    pub weight_map: HashMap<String, String>, // tensor_name → shard_filename
}

impl ShardIndex {
    pub fn new() -> Self {
        Self { total_size: 0, shards: Vec::new(), weight_map: HashMap::new() }
    }

    /// Add a shard file.
    pub fn add_shard(&mut self, filename: &str, byte_size: u64, tensor_count: usize) {
        self.shards.push(ShardEntry { filename: filename.to_string(), byte_size, tensor_count });
        self.total_size += byte_size;
    }

    /// Map a tensor to its shard.
    pub fn map_tensor(&mut self, tensor_name: &str, shard_filename: &str) {
        self.weight_map.insert(tensor_name.to_string(), shard_filename.to_string());
    }

    /// Find which shard contains a tensor.
    pub fn locate_tensor(&self, tensor_name: &str) -> Option<&str> {
        self.weight_map.get(tensor_name).map(|s| s.as_str())
    }

    /// All tensors in a specific shard.
    pub fn tensors_in_shard(&self, shard_filename: &str) -> Vec<&str> {
        self.weight_map
            .iter()
            .filter(|(_, v)| v.as_str() == shard_filename)
            .map(|(k, _)| k.as_str())
            .collect()
    }

    pub fn shard_count(&self) -> usize {
        self.shards.len()
    }

    pub fn tensor_count(&self) -> usize {
        self.weight_map.len()
    }

    /// Average shard size in bytes.
    pub fn avg_shard_size(&self) -> u64 {
        if self.shards.is_empty() {
            return 0;
        }
        self.total_size / self.shards.len() as u64
    }

    /// Largest shard by byte size.
    pub fn largest_shard(&self) -> Option<&ShardEntry> {
        self.shards.iter().max_by_key(|s| s.byte_size)
    }

    /// Build from a HuggingFace model.safetensors.index.json weight_map.
    pub fn from_weight_map(weight_map: HashMap<String, String>) -> Self {
        let mut index = Self::new();

        // Count tensors per shard
        let mut shard_tensors: HashMap<String, usize> = HashMap::new();
        for (tensor_name, shard_file) in &weight_map {
            *shard_tensors.entry(shard_file.clone()).or_insert(0) += 1;
            index.map_tensor(tensor_name, shard_file);
        }

        for (shard_file, count) in shard_tensors {
            index.add_shard(&shard_file, 0, count); // byte_size unknown without files
        }

        index
    }

    /// Validate index consistency.
    pub fn validate(&self) -> Vec<String> {
        let mut issues = Vec::new();

        // Check all mapped tensors reference known shards
        let shard_names: Vec<&str> = self.shards.iter().map(|s| s.filename.as_str()).collect();
        for (tensor, shard) in &self.weight_map {
            if !shard_names.contains(&shard.as_str()) {
                issues.push(format!("tensor '{tensor}' references unknown shard '{shard}'"));
            }
        }

        if self.shards.is_empty() && !self.weight_map.is_empty() {
            issues.push("weight map has entries but no shards registered".into());
        }

        issues
    }
}

impl Default for ShardIndex {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new() {
        let idx = ShardIndex::new();
        assert_eq!(idx.shard_count(), 0);
        assert_eq!(idx.tensor_count(), 0);
    }

    #[test]
    fn test_add_shard() {
        let mut idx = ShardIndex::new();
        idx.add_shard("model-00001.safetensors", 5_000_000_000, 50);
        assert_eq!(idx.shard_count(), 1);
        assert_eq!(idx.total_size, 5_000_000_000);
    }

    #[test]
    fn test_map_and_locate() {
        let mut idx = ShardIndex::new();
        idx.add_shard("shard-1.safetensors", 1000, 2);
        idx.map_tensor("model.embed.weight", "shard-1.safetensors");
        assert_eq!(idx.locate_tensor("model.embed.weight"), Some("shard-1.safetensors"));
        assert_eq!(idx.locate_tensor("nonexistent"), None);
    }

    #[test]
    fn test_tensors_in_shard() {
        let mut idx = ShardIndex::new();
        idx.add_shard("s1.safetensors", 100, 2);
        idx.map_tensor("a", "s1.safetensors");
        idx.map_tensor("b", "s1.safetensors");
        idx.map_tensor("c", "s2.safetensors");
        let tensors = idx.tensors_in_shard("s1.safetensors");
        assert_eq!(tensors.len(), 2);
    }

    #[test]
    fn test_avg_shard_size() {
        let mut idx = ShardIndex::new();
        idx.add_shard("a.safetensors", 100, 1);
        idx.add_shard("b.safetensors", 200, 1);
        assert_eq!(idx.avg_shard_size(), 150);
    }

    #[test]
    fn test_avg_shard_size_empty() {
        let idx = ShardIndex::new();
        assert_eq!(idx.avg_shard_size(), 0);
    }

    #[test]
    fn test_largest_shard() {
        let mut idx = ShardIndex::new();
        idx.add_shard("small.safetensors", 100, 1);
        idx.add_shard("large.safetensors", 5000, 10);
        let largest = idx.largest_shard().unwrap();
        assert_eq!(largest.filename, "large.safetensors");
    }

    #[test]
    fn test_from_weight_map() {
        let mut wm = HashMap::new();
        wm.insert("embed.weight".into(), "shard-1.safetensors".into());
        wm.insert("lm_head.weight".into(), "shard-2.safetensors".into());
        let idx = ShardIndex::from_weight_map(wm);
        assert_eq!(idx.tensor_count(), 2);
        assert_eq!(idx.shard_count(), 2);
    }

    #[test]
    fn test_validate_ok() {
        let mut idx = ShardIndex::new();
        idx.add_shard("s1.safetensors", 100, 1);
        idx.map_tensor("w", "s1.safetensors");
        assert!(idx.validate().is_empty());
    }

    #[test]
    fn test_validate_unknown_shard() {
        let mut idx = ShardIndex::new();
        idx.add_shard("s1.safetensors", 100, 1);
        idx.map_tensor("w", "s2.safetensors"); // s2 not registered
        assert!(!idx.validate().is_empty());
    }

    #[test]
    fn test_default() {
        let idx = ShardIndex::default();
        assert_eq!(idx.shard_count(), 0);
    }

    #[test]
    fn test_phi4_shard_simulation() {
        let mut idx = ShardIndex::new();
        for i in 1..=6 {
            idx.add_shard(&format!("model-{i:05}.safetensors"), 4_800_000_000, 40);
        }
        assert_eq!(idx.shard_count(), 6);
        assert!(idx.total_size > 28_000_000_000);
    }

    #[test]
    fn test_tensor_count() {
        let mut idx = ShardIndex::new();
        idx.map_tensor("a", "s1");
        idx.map_tensor("b", "s1");
        assert_eq!(idx.tensor_count(), 2);
    }
}
