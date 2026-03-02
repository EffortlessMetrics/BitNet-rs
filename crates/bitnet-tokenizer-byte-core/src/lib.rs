use std::collections::HashMap;

/// Lookup tables derived from a token vocabulary for byte-level tokenization.
#[derive(Debug, Clone)]
pub struct ByteVocabulary {
    vocab: HashMap<String, u32>,
    reverse_vocab: HashMap<u32, String>,
    byte_to_id: [Option<u32>; 256],
    id_to_byte: HashMap<u32, u8>,
}

impl ByteVocabulary {
    /// Build byte lookup tables from an ordered token list.
    #[must_use]
    pub fn from_tokens(tokens: &[String]) -> Self {
        let mut vocab = HashMap::with_capacity(tokens.len());
        let mut reverse_vocab = HashMap::with_capacity(tokens.len());
        let mut byte_to_id = [None; 256];
        let mut id_to_byte = HashMap::new();

        for (i, token) in tokens.iter().enumerate() {
            let id = i as u32;
            vocab.insert(token.clone(), id);
            reverse_vocab.insert(id, token.clone());

            if token.len() == 6
                && token.starts_with("<0x")
                && token.ends_with('>')
                && let Ok(byte) = u8::from_str_radix(&token[3..5], 16)
            {
                byte_to_id[byte as usize] = Some(id);
                id_to_byte.insert(id, byte);
            }
        }

        Self { vocab, reverse_vocab, byte_to_id, id_to_byte }
    }

    #[must_use]
    pub fn vocab(&self) -> &HashMap<String, u32> {
        &self.vocab
    }

    #[must_use]
    pub fn reverse_vocab(&self) -> &HashMap<u32, String> {
        &self.reverse_vocab
    }

    #[must_use]
    pub fn byte_to_id(&self) -> &[Option<u32>; 256] {
        &self.byte_to_id
    }

    #[must_use]
    pub fn id_to_byte(&self) -> &HashMap<u32, u8> {
        &self.id_to_byte
    }
}

#[cfg(test)]
mod tests {
    use super::ByteVocabulary;

    #[test]
    fn builds_vocab_and_reverse_vocab_from_tokens() {
        let tokens = vec!["hello".to_string(), "world".to_string()];

        let vocab = ByteVocabulary::from_tokens(&tokens);

        assert_eq!(vocab.vocab().get("hello"), Some(&0));
        assert_eq!(vocab.vocab().get("world"), Some(&1));
        assert_eq!(vocab.reverse_vocab().get(&0), Some(&"hello".to_string()));
        assert_eq!(vocab.reverse_vocab().get(&1), Some(&"world".to_string()));
    }

    #[test]
    fn recognizes_hex_byte_tokens() {
        let tokens = vec!["<0x41>".to_string(), "<0x00>".to_string(), "plain".to_string()];

        let vocab = ByteVocabulary::from_tokens(&tokens);

        assert_eq!(vocab.byte_to_id()[0x41], Some(0));
        assert_eq!(vocab.byte_to_id()[0x00], Some(1));
        assert_eq!(vocab.id_to_byte().get(&0), Some(&0x41));
        assert_eq!(vocab.id_to_byte().get(&1), Some(&0x00));
        assert!(!vocab.id_to_byte().contains_key(&2));
    }

    #[test]
    fn ignores_malformed_hex_tokens() {
        let tokens = vec!["<0xG1>".to_string(), "<0x1>".to_string(), "<0x414>".to_string()];

        let vocab = ByteVocabulary::from_tokens(&tokens);

        assert!(vocab.id_to_byte().is_empty());
        assert!(vocab.byte_to_id().iter().all(Option::is_none));
    }
}
