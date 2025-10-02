//! Mock GGUF Reader for Unit Testing
//!
//! Provides in-memory GGUF data structures for testing without file I/O.

#![cfg(test)]
#![cfg(feature = "cpu")]

use std::collections::HashMap;

/// Mock GGUF file reader for unit tests
#[derive(Debug, Clone)]
pub struct MockGgufReader {
    pub model_type: String,
    pub vocab_size: u32,
    pub metadata: HashMap<String, MetadataValue>,
    pub tensors: Vec<MockTensor>,
}

/// Mock metadata value types
#[derive(Debug, Clone)]
pub enum MetadataValue {
    String(String),
    UInt32(u32),
    Float32(f32),
    Bool(bool),
}

/// Mock tensor information
#[derive(Debug, Clone)]
pub struct MockTensor {
    pub name: String,
    pub shape: Vec<usize>,
    pub data_type: String,
}

impl MockGgufReader {
    /// Create a mock LLaMA-2 GGUF reader
    pub fn mock_llama2() -> Self {
        let mut metadata = HashMap::new();
        metadata
            .insert("general.architecture".to_string(), MetadataValue::String("llama".to_string()));
        metadata.insert("llama.vocab_size".to_string(), MetadataValue::UInt32(32000));
        metadata.insert("llama.embedding_length".to_string(), MetadataValue::UInt32(4096));
        metadata.insert("llama.block_count".to_string(), MetadataValue::UInt32(32));
        metadata
            .insert("tokenizer.ggml.model".to_string(), MetadataValue::String("llama".to_string()));
        metadata.insert("tokenizer.ggml.bos_token_id".to_string(), MetadataValue::UInt32(1));
        metadata.insert("tokenizer.ggml.eos_token_id".to_string(), MetadataValue::UInt32(2));

        let tensors = vec![
            MockTensor {
                name: "token_embd.weight".to_string(),
                shape: vec![32000, 4096],
                data_type: "f16".to_string(),
            },
            MockTensor {
                name: "blk.0.attn_q.weight".to_string(),
                shape: vec![4096, 4096],
                data_type: "f16".to_string(),
            },
            MockTensor {
                name: "blk.0.attn_k.weight".to_string(),
                shape: vec![4096, 4096],
                data_type: "f16".to_string(),
            },
            MockTensor {
                name: "output.weight".to_string(),
                shape: vec![4096, 32000],
                data_type: "f16".to_string(),
            },
        ];

        Self { model_type: "llama".to_string(), vocab_size: 32000, metadata, tensors }
    }

    /// Create a mock LLaMA-3 GGUF reader
    pub fn mock_llama3() -> Self {
        let mut metadata = HashMap::new();
        metadata
            .insert("general.architecture".to_string(), MetadataValue::String("llama".to_string()));
        metadata.insert("llama.vocab_size".to_string(), MetadataValue::UInt32(128256));
        metadata.insert("llama.embedding_length".to_string(), MetadataValue::UInt32(4096));

        let tensors = vec![MockTensor {
            name: "token_embd.weight".to_string(),
            shape: vec![128256, 4096],
            data_type: "f16".to_string(),
        }];

        Self { model_type: "llama".to_string(), vocab_size: 128256, metadata, tensors }
    }

    /// Create a mock GPT-2 GGUF reader
    pub fn mock_gpt2() -> Self {
        let mut metadata = HashMap::new();
        metadata
            .insert("general.architecture".to_string(), MetadataValue::String("gpt2".to_string()));
        metadata.insert("gpt2.vocab_size".to_string(), MetadataValue::UInt32(50257));
        metadata.insert("gpt2.embedding_length".to_string(), MetadataValue::UInt32(1024));

        let tensors = vec![MockTensor {
            name: "transformer.wte.weight".to_string(),
            shape: vec![50257, 1024],
            data_type: "f32".to_string(),
        }];

        Self { model_type: "gpt2".to_string(), vocab_size: 50257, metadata, tensors }
    }

    /// Create a mock BitNet GGUF reader
    pub fn mock_bitnet() -> Self {
        let mut metadata = HashMap::new();
        metadata.insert(
            "general.architecture".to_string(),
            MetadataValue::String("bitnet".to_string()),
        );
        metadata.insert("bitnet.vocab_size".to_string(), MetadataValue::UInt32(32000));
        metadata.insert("bitnet.hidden_size".to_string(), MetadataValue::UInt32(2048));

        let tensors = vec![
            MockTensor {
                name: "token_embd.weight".to_string(),
                shape: vec![32000, 2048],
                data_type: "i2s".to_string(),
            },
            MockTensor {
                name: "blk.0.attn_q.weight".to_string(),
                shape: vec![2048, 2048],
                data_type: "i2s".to_string(),
            },
        ];

        Self { model_type: "bitnet".to_string(), vocab_size: 32000, metadata, tensors }
    }

    /// Get metadata value
    pub fn get_metadata(&self, key: &str) -> Option<&MetadataValue> {
        self.metadata.get(key)
    }

    /// Get tensor by name
    pub fn get_tensor(&self, name: &str) -> Option<&MockTensor> {
        self.tensors.iter().find(|t| t.name == name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mock_llama2_reader() {
        let reader = MockGgufReader::mock_llama2();
        assert_eq!(reader.model_type, "llama");
        assert_eq!(reader.vocab_size, 32000);

        let arch = reader.get_metadata("general.architecture");
        assert!(matches!(arch, Some(MetadataValue::String(s)) if s == "llama"));
    }

    #[test]
    fn test_mock_tensor_access() {
        let reader = MockGgufReader::mock_llama2();
        let tensor = reader.get_tensor("token_embd.weight");
        assert!(tensor.is_some());

        let tensor = tensor.unwrap();
        assert_eq!(tensor.shape, vec![32000, 4096]);
    }
}
