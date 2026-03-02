//! Vocabulary index builder.
//!
//! Efficient lookup structures for token vocabularies.

use std::collections::HashMap;

/// Token entry in the vocabulary.
#[derive(Debug, Clone)]
pub struct VocabEntry {
    pub id: u32,
    pub text: String,
    pub score: f32,
    pub is_special: bool,
}

/// Vocabulary index for fast lookups.
#[derive(Debug, Clone)]
pub struct VocabIndex {
    entries: Vec<VocabEntry>,
    text_to_id: HashMap<String, u32>,
    special_tokens: Vec<u32>,
}

impl Default for VocabIndex {
    fn default() -> Self {
        Self::new()
    }
}

impl VocabIndex {
    pub fn new() -> Self {
        Self { entries: Vec::new(), text_to_id: HashMap::new(), special_tokens: Vec::new() }
    }

    pub fn add(&mut self, id: u32, text: String, score: f32, is_special: bool) {
        self.text_to_id.insert(text.clone(), id);
        if is_special {
            self.special_tokens.push(id);
        }
        // Extend vector if needed
        if id as usize >= self.entries.len() {
            self.entries.resize(
                id as usize + 1,
                VocabEntry { id: 0, text: String::new(), score: 0.0, is_special: false },
            );
        }
        self.entries[id as usize] = VocabEntry { id, text, score, is_special };
    }

    pub fn lookup_id(&self, text: &str) -> Option<u32> {
        self.text_to_id.get(text).copied()
    }

    pub fn lookup_text(&self, id: u32) -> Option<&str> {
        self.entries.get(id as usize).map(|e| e.text.as_str())
    }

    pub fn size(&self) -> usize {
        self.text_to_id.len()
    }

    pub fn special_count(&self) -> usize {
        self.special_tokens.len()
    }

    pub fn special_ids(&self) -> &[u32] {
        &self.special_tokens
    }

    pub fn contains_id(&self, id: u32) -> bool {
        (id as usize) < self.entries.len() && !self.entries[id as usize].text.is_empty()
    }

    pub fn contains_text(&self, text: &str) -> bool {
        self.text_to_id.contains_key(text)
    }

    /// Get entries with text matching a prefix.
    pub fn prefix_search(&self, prefix: &str) -> Vec<&VocabEntry> {
        self.entries.iter().filter(|e| !e.text.is_empty() && e.text.starts_with(prefix)).collect()
    }

    /// Build from parallel arrays.
    pub fn from_tokens(tokens: &[String], scores: &[f32], special_ids: &[u32]) -> Self {
        let mut index = Self::new();
        for (i, token) in tokens.iter().enumerate() {
            let score = scores.get(i).copied().unwrap_or(0.0);
            let is_special = special_ids.contains(&(i as u32));
            index.add(i as u32, token.clone(), score, is_special);
        }
        index
    }
}

/// Builder for constructing vocabulary indexes.
#[derive(Debug)]
pub struct VocabIndexBuilder {
    entries: Vec<(u32, String, f32, bool)>,
}

impl Default for VocabIndexBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl VocabIndexBuilder {
    pub fn new() -> Self {
        Self { entries: Vec::new() }
    }

    pub fn add_token(mut self, id: u32, text: String) -> Self {
        self.entries.push((id, text, 0.0, false));
        self
    }

    pub fn add_special(mut self, id: u32, text: String) -> Self {
        self.entries.push((id, text, 0.0, true));
        self
    }

    pub fn build(self) -> VocabIndex {
        let mut index = VocabIndex::new();
        for (id, text, score, special) in self.entries {
            index.add(id, text, score, special);
        }
        index
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_index() {
        let idx = VocabIndex::new();
        assert_eq!(idx.size(), 0);
    }

    #[test]
    fn test_add_lookup() {
        let mut idx = VocabIndex::new();
        idx.add(0, "hello".into(), 1.0, false);
        assert_eq!(idx.lookup_id("hello"), Some(0));
        assert_eq!(idx.lookup_text(0), Some("hello"));
    }

    #[test]
    fn test_special_tokens() {
        let mut idx = VocabIndex::new();
        idx.add(0, "<s>".into(), 0.0, true);
        idx.add(1, "</s>".into(), 0.0, true);
        idx.add(2, "hello".into(), 0.0, false);
        assert_eq!(idx.special_count(), 2);
    }

    #[test]
    fn test_contains() {
        let mut idx = VocabIndex::new();
        idx.add(5, "test".into(), 0.0, false);
        assert!(idx.contains_id(5));
        assert!(!idx.contains_id(99));
        assert!(idx.contains_text("test"));
    }

    #[test]
    fn test_prefix_search() {
        let mut idx = VocabIndex::new();
        idx.add(0, "hello".into(), 0.0, false);
        idx.add(1, "help".into(), 0.0, false);
        idx.add(2, "world".into(), 0.0, false);
        let results = idx.prefix_search("hel");
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_from_tokens() {
        let tokens = vec!["a".into(), "b".into(), "c".into()];
        let scores = vec![1.0, 2.0, 3.0];
        let specials = vec![0u32];
        let idx = VocabIndex::from_tokens(&tokens, &scores, &specials);
        assert_eq!(idx.size(), 3);
        assert_eq!(idx.special_count(), 1);
    }

    #[test]
    fn test_builder() {
        let idx = VocabIndexBuilder::new()
            .add_token(0, "hello".into())
            .add_special(1, "<eos>".into())
            .build();
        assert_eq!(idx.size(), 2);
        assert_eq!(idx.special_count(), 1);
    }

    #[test]
    fn test_builder_default() {
        let b = VocabIndexBuilder::default();
        let idx = b.build();
        assert_eq!(idx.size(), 0);
    }

    #[test]
    fn test_lookup_missing() {
        let idx = VocabIndex::new();
        assert!(idx.lookup_id("missing").is_none());
        assert!(idx.lookup_text(999).is_none());
    }

    #[test]
    fn test_large_vocab() {
        let mut idx = VocabIndex::new();
        for i in 0..1000 {
            idx.add(i, format!("token_{i}"), i as f32, i < 5);
        }
        assert_eq!(idx.size(), 1000);
        assert_eq!(idx.special_count(), 5);
        assert_eq!(idx.lookup_id("token_500"), Some(500));
    }

    #[test]
    fn test_default() {
        let idx = VocabIndex::default();
        assert_eq!(idx.size(), 0);
    }
}
