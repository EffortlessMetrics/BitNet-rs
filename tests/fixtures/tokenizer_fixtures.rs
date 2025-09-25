//! Tokenizer test fixtures for BitNet.rs neural network components
//!
//! Provides comprehensive test data for tokenizer discovery, strategy resolution,
//! and integration with GGUF model files for neural network inference validation.

use bitnet_common::{BitNetError, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::LazyLock;

/// Comprehensive tokenizer test fixture configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenizerTestFixture {
    pub tokenizer_type: TokenizerType,
    pub model_architecture: String,
    pub vocab_size: u32,
    pub special_tokens: SpecialTokens,
    pub test_prompts: Vec<TestPrompt>,
    pub expected_outputs: Vec<ExpectedTokenization>,
    pub gguf_metadata: GgufTokenizerMetadata,
    pub performance_baseline: PerformanceBaseline,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TokenizerType {
    LLaMA2,
    LLaMA3,
    GPT2,
    SentencePiece,
    Mock,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpecialTokens {
    pub bos_token: Option<u32>,
    pub eos_token: Option<u32>,
    pub pad_token: Option<u32>,
    pub unk_token: Option<u32>,
    pub sep_token: Option<u32>,
    pub cls_token: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestPrompt {
    pub text: String,
    pub description: String,
    pub test_scenario: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpectedTokenization {
    pub prompt_text: String,
    pub expected_tokens: Vec<u32>,
    pub expected_decoded: String,
    pub tolerance_percent: f32,
    pub quantization_compatible: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GgufTokenizerMetadata {
    pub tokenizer_model: String,
    pub tokenizer_list: Vec<String>,
    pub tokenizer_type: String,
    pub tokenizer_scores: Option<Vec<f32>>,
    pub tokenizer_token_type: Option<Vec<u32>>,
    pub add_bos_token: bool,
    pub add_eos_token: bool,
    pub chat_template: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceBaseline {
    pub tokens_per_second: f32,
    pub memory_usage_mb: f32,
    pub startup_time_ms: u32,
    pub max_regression_percent: f32,
}

/// Static fixture definitions for different model architectures
static LLAMA3_FIXTURE: LazyLock<TokenizerTestFixture> = LazyLock::new(|| {
    TokenizerTestFixture {
    tokenizer_type: TokenizerType::LLaMA3,
    model_architecture: "BitNet".to_string(),
    vocab_size: 128256,
    special_tokens: SpecialTokens {
        bos_token: Some(128000),
        eos_token: Some(128001),
        pad_token: None,
        unk_token: None,
        sep_token: None,
        cls_token: None,
    },
    test_prompts: vec![
        TestPrompt {
            text: "Hello world".to_string(),
            description: "Simple greeting".to_string(),
            test_scenario: "basic_tokenization".to_string(),
        },
        TestPrompt {
            text: "Neural network inference with BitNet quantization".to_string(),
            description: "Technical AI terminology".to_string(),
            test_scenario: "domain_specific_vocabulary".to_string(),
        },
        TestPrompt {
            text: "The quick brown fox jumps over the lazy dog. This sentence contains every letter of the alphabet at least once.".to_string(),
            description: "Comprehensive character coverage".to_string(),
            test_scenario: "character_coverage".to_string(),
        },
        TestPrompt {
            text: "🤖 AI systems process language using neural networks 🧠".to_string(),
            description: "Unicode and emoji handling".to_string(),
            test_scenario: "unicode_emoji_support".to_string(),
        },
        TestPrompt {
            text: "Code snippet: def quantize_weights(weights: torch.Tensor) -> torch.Tensor:".to_string(),
            description: "Programming code tokenization".to_string(),
            test_scenario: "code_tokenization".to_string(),
        },
    ],
    expected_outputs: vec![
        ExpectedTokenization {
            prompt_text: "Hello world".to_string(),
            expected_tokens: vec![9906, 1917],
            expected_decoded: "Hello world".to_string(),
            tolerance_percent: 5.0,
            quantization_compatible: vec!["I2S".to_string(), "TL1".to_string(), "TL2".to_string()],
        },
        ExpectedTokenization {
            prompt_text: "Neural network".to_string(),
            expected_tokens: vec![8989, 4632, 4009],
            expected_decoded: "Neural network".to_string(),
            tolerance_percent: 2.0,
            quantization_compatible: vec!["I2S".to_string()],
        },
    ],
    gguf_metadata: GgufTokenizerMetadata {
        tokenizer_model: "llama".to_string(),
        tokenizer_list: vec!["<|begin_of_text|>".to_string(), "<|end_of_text|>".to_string()],
        tokenizer_type: "BPE".to_string(),
        tokenizer_scores: None,
        tokenizer_token_type: None,
        add_bos_token: true,
        add_eos_token: false,
        chat_template: Some("{{% if messages[0]['role'] == 'system' %}}{{% set system_message = messages[0]['content'] %}}{{% endif %}}".to_string()),
    },
    performance_baseline: PerformanceBaseline {
        tokens_per_second: 50000.0,
        memory_usage_mb: 512.0,
        startup_time_ms: 200,
        max_regression_percent: 20.0,
    },
}
});

static LLAMA2_FIXTURE: LazyLock<TokenizerTestFixture> = LazyLock::new(|| TokenizerTestFixture {
    tokenizer_type: TokenizerType::LLaMA2,
    model_architecture: "BitNet".to_string(),
    vocab_size: 32000,
    special_tokens: SpecialTokens {
        bos_token: Some(1),
        eos_token: Some(2),
        pad_token: None,
        unk_token: Some(0),
        sep_token: None,
        cls_token: None,
    },
    test_prompts: vec![
        TestPrompt {
            text: "Hello world".to_string(),
            description: "Simple greeting for LLaMA-2".to_string(),
            test_scenario: "basic_tokenization".to_string(),
        },
        TestPrompt {
            text: "Quantized neural networks enable efficient inference".to_string(),
            description: "Technical domain vocabulary".to_string(),
            test_scenario: "quantization_vocabulary".to_string(),
        },
    ],
    expected_outputs: vec![ExpectedTokenization {
        prompt_text: "Hello world".to_string(),
        expected_tokens: vec![15043, 3186],
        expected_decoded: "Hello world".to_string(),
        tolerance_percent: 5.0,
        quantization_compatible: vec!["I2S".to_string(), "TL1".to_string(), "TL2".to_string()],
    }],
    gguf_metadata: GgufTokenizerMetadata {
        tokenizer_model: "llama".to_string(),
        tokenizer_list: vec!["<s>".to_string(), "</s>".to_string()],
        tokenizer_type: "BPE".to_string(),
        tokenizer_scores: None,
        tokenizer_token_type: None,
        add_bos_token: true,
        add_eos_token: false,
        chat_template: None,
    },
    performance_baseline: PerformanceBaseline {
        tokens_per_second: 60000.0,
        memory_usage_mb: 128.0,
        startup_time_ms: 150,
        max_regression_percent: 15.0,
    },
});

static GPT2_FIXTURE: LazyLock<TokenizerTestFixture> = LazyLock::new(|| TokenizerTestFixture {
    tokenizer_type: TokenizerType::GPT2,
    model_architecture: "GPT".to_string(),
    vocab_size: 50257,
    special_tokens: SpecialTokens {
        bos_token: None,
        eos_token: Some(50256),
        pad_token: Some(50256),
        unk_token: Some(50257),
        sep_token: None,
        cls_token: None,
    },
    test_prompts: vec![
        TestPrompt {
            text: "Hello world".to_string(),
            description: "Simple GPT-2 tokenization".to_string(),
            test_scenario: "basic_tokenization".to_string(),
        },
        TestPrompt {
            text: "Generate text with GPT-2 neural network".to_string(),
            description: "Model-specific terminology".to_string(),
            test_scenario: "model_specific_vocab".to_string(),
        },
    ],
    expected_outputs: vec![ExpectedTokenization {
        prompt_text: "Hello world".to_string(),
        expected_tokens: vec![15496, 995],
        expected_decoded: "Hello world".to_string(),
        tolerance_percent: 8.0,
        quantization_compatible: vec!["TL1".to_string(), "TL2".to_string()],
    }],
    gguf_metadata: GgufTokenizerMetadata {
        tokenizer_model: "gpt2".to_string(),
        tokenizer_list: vec!["<|endoftext|>".to_string()],
        tokenizer_type: "BPE".to_string(),
        tokenizer_scores: None,
        tokenizer_token_type: None,
        add_bos_token: false,
        add_eos_token: false,
        chat_template: None,
    },
    performance_baseline: PerformanceBaseline {
        tokens_per_second: 45000.0,
        memory_usage_mb: 200.0,
        startup_time_ms: 180,
        max_regression_percent: 25.0,
    },
});

/// Mock GGUF model data for testing
#[derive(Debug, Clone)]
pub struct MockGgufModel {
    pub file_path: PathBuf,
    pub architecture: String,
    pub vocab_size: u32,
    pub metadata_kvs: HashMap<String, MockGgufValue>,
    pub tensor_info: Vec<MockTensorInfo>,
    pub file_content: Vec<u8>,
}

#[derive(Debug, Clone)]
pub enum MockGgufValue {
    String(String),
    UInt32(u32),
    UInt64(u64),
    Float32(f32),
    Bool(bool),
    Array(Vec<String>),
}

#[derive(Debug, Clone)]
pub struct MockTensorInfo {
    pub name: String,
    pub shape: Vec<u64>,
    pub ggml_type: u32, // GGML data type
    pub offset: u64,
    pub size_bytes: u64,
}

impl MockGgufModel {
    /// Create LLaMA-3 model with 128k vocabulary
    pub fn create_llama3_128k() -> Self {
        let mut metadata_kvs = HashMap::new();
        metadata_kvs.insert(
            "general.architecture".to_string(),
            MockGgufValue::String("BitNet".to_string()),
        );
        metadata_kvs.insert(
            "general.name".to_string(),
            MockGgufValue::String("llama3-128k-test".to_string()),
        );
        metadata_kvs.insert("llama.vocab_size".to_string(), MockGgufValue::UInt32(128256));
        metadata_kvs.insert("llama.embedding_length".to_string(), MockGgufValue::UInt32(4096));
        metadata_kvs.insert("llama.block_count".to_string(), MockGgufValue::UInt32(32));
        metadata_kvs.insert("llama.attention.head_count".to_string(), MockGgufValue::UInt32(32));
        metadata_kvs.insert("llama.attention.head_count_kv".to_string(), MockGgufValue::UInt32(8));
        metadata_kvs.insert("llama.context_length".to_string(), MockGgufValue::UInt32(8192));
        metadata_kvs
            .insert("tokenizer.ggml.model".to_string(), MockGgufValue::String("llama".to_string()));
        metadata_kvs.insert(
            "tokenizer.ggml.tokens".to_string(),
            MockGgufValue::Array(vec![
                "<|begin_of_text|>".to_string(),
                "<|end_of_text|>".to_string(),
            ]),
        );
        metadata_kvs.insert(
            "tokenizer.ggml.token_type".to_string(),
            MockGgufValue::Array(
                vec!["3".to_string(), "3".to_string()], // SPECIAL tokens
            ),
        );
        metadata_kvs.insert("tokenizer.ggml.add_bos_token".to_string(), MockGgufValue::Bool(true));
        metadata_kvs.insert("tokenizer.ggml.add_eos_token".to_string(), MockGgufValue::Bool(false));
        metadata_kvs.insert(
            "tokenizer.chat_template".to_string(),
            MockGgufValue::String(
                "{{% if messages[0]['role'] == 'system' %}}{{messages[0]['content']}}".to_string(),
            ),
        );

        let tensor_info = vec![
            MockTensorInfo {
                name: "token_embd.weight".to_string(),
                shape: vec![4096, 128256],
                ggml_type: 24, // I2_S quantization
                offset: 4096,
                size_bytes: 4096 * 128256 * 2 / 8, // 2-bit quantized
            },
            MockTensorInfo {
                name: "blk.0.attn_q.weight".to_string(),
                shape: vec![4096, 4096],
                ggml_type: 24, // I2_S
                offset: 4096 + (4096 * 128256 * 2 / 8),
                size_bytes: 4096 * 4096 * 2 / 8,
            },
            MockTensorInfo {
                name: "output.weight".to_string(),
                shape: vec![128256, 4096],
                ggml_type: 24, // I2_S
                offset: 8192 + (4096 * 128256 * 2 / 8) + (4096 * 4096 * 2 / 8),
                size_bytes: 128256 * 4096 * 2 / 8,
            },
        ];

        Self {
            file_path: PathBuf::from(
                "/home/steven/code/Rust/BitNet-rs/tests/fixtures/gguf_models/llama3_128k.gguf",
            ),
            architecture: "BitNet".to_string(),
            vocab_size: 128256,
            metadata_kvs,
            tensor_info,
            file_content: Self::generate_gguf_binary(
                "BitNet",
                128256,
                &[
                    ("token_embd.weight", vec![4096, 128256]),
                    ("blk.0.attn_q.weight", vec![4096, 4096]),
                    ("output.weight", vec![128256, 4096]),
                ],
            ),
        }
    }

    /// Create LLaMA-2 model with 32k vocabulary
    pub fn create_llama2_32k() -> Self {
        let mut metadata_kvs = HashMap::new();
        metadata_kvs.insert(
            "general.architecture".to_string(),
            MockGgufValue::String("BitNet".to_string()),
        );
        metadata_kvs.insert(
            "general.name".to_string(),
            MockGgufValue::String("llama2-32k-test".to_string()),
        );
        metadata_kvs.insert("llama.vocab_size".to_string(), MockGgufValue::UInt32(32000));
        metadata_kvs.insert("llama.embedding_length".to_string(), MockGgufValue::UInt32(2048));
        metadata_kvs.insert("llama.block_count".to_string(), MockGgufValue::UInt32(16));
        metadata_kvs.insert("llama.attention.head_count".to_string(), MockGgufValue::UInt32(16));
        metadata_kvs.insert("llama.context_length".to_string(), MockGgufValue::UInt32(4096));
        metadata_kvs
            .insert("tokenizer.ggml.model".to_string(), MockGgufValue::String("llama".to_string()));
        metadata_kvs.insert(
            "tokenizer.ggml.tokens".to_string(),
            MockGgufValue::Array(vec!["<s>".to_string(), "</s>".to_string()]),
        );
        metadata_kvs.insert("tokenizer.ggml.add_bos_token".to_string(), MockGgufValue::Bool(true));

        let tensor_info = vec![
            MockTensorInfo {
                name: "token_embd.weight".to_string(),
                shape: vec![2048, 32000],
                ggml_type: 25, // TL1 quantization for smaller vocab
                offset: 4096,
                size_bytes: 2048 * 32000 * 4 / 8, // 4-bit quantized
            },
            MockTensorInfo {
                name: "output.weight".to_string(),
                shape: vec![32000, 2048],
                ggml_type: 25, // TL1
                offset: 4096 + (2048 * 32000 * 4 / 8),
                size_bytes: 32000 * 2048 * 4 / 8,
            },
        ];

        Self {
            file_path: PathBuf::from(
                "/home/steven/code/Rust/BitNet-rs/tests/fixtures/gguf_models/llama2_32k.gguf",
            ),
            architecture: "BitNet".to_string(),
            vocab_size: 32000,
            metadata_kvs,
            tensor_info,
            file_content: Self::generate_gguf_binary(
                "BitNet",
                32000,
                &[("token_embd.weight", vec![2048, 32000]), ("output.weight", vec![32000, 2048])],
            ),
        }
    }

    /// Create GPT-2 model with 50k vocabulary
    pub fn create_gpt2_50k() -> Self {
        let mut metadata_kvs = HashMap::new();
        metadata_kvs
            .insert("general.architecture".to_string(), MockGgufValue::String("GPT".to_string()));
        metadata_kvs
            .insert("general.name".to_string(), MockGgufValue::String("gpt2-50k-test".to_string()));
        metadata_kvs.insert("gpt2.vocab_size".to_string(), MockGgufValue::UInt32(50257));
        metadata_kvs.insert("gpt2.embedding_length".to_string(), MockGgufValue::UInt32(768));
        metadata_kvs.insert("gpt2.block_count".to_string(), MockGgufValue::UInt32(12));
        metadata_kvs.insert("gpt2.attention.head_count".to_string(), MockGgufValue::UInt32(12));
        metadata_kvs.insert("gpt2.context_length".to_string(), MockGgufValue::UInt32(1024));
        metadata_kvs
            .insert("tokenizer.ggml.model".to_string(), MockGgufValue::String("gpt2".to_string()));
        metadata_kvs.insert(
            "tokenizer.ggml.tokens".to_string(),
            MockGgufValue::Array(vec!["<|endoftext|>".to_string()]),
        );
        metadata_kvs.insert("tokenizer.ggml.add_bos_token".to_string(), MockGgufValue::Bool(false));

        let tensor_info = vec![
            MockTensorInfo {
                name: "wte.weight".to_string(), // GPT-2 token embedding name
                shape: vec![768, 50257],
                ggml_type: 26, // TL2 quantization for medium vocab
                offset: 4096,
                size_bytes: 768 * 50257 * 3 / 8, // 3-bit quantized
            },
            MockTensorInfo {
                name: "wpe.weight".to_string(), // GPT-2 position embedding
                shape: vec![768, 1024],
                ggml_type: 0, // FP32 for position embeddings
                offset: 4096 + (768 * 50257 * 3 / 8),
                size_bytes: 768 * 1024 * 4,
            },
        ];

        Self {
            file_path: PathBuf::from(
                "/home/steven/code/Rust/BitNet-rs/tests/fixtures/gguf_models/gpt2_50k.gguf",
            ),
            architecture: "GPT".to_string(),
            vocab_size: 50257,
            metadata_kvs,
            tensor_info,
            file_content: Self::generate_gguf_binary(
                "GPT",
                50257,
                &[("wte.weight", vec![768, 50257]), ("wpe.weight", vec![768, 1024])],
            ),
        }
    }

    /// Create corrupted GGUF for error handling tests
    pub fn create_corrupted_gguf() -> Self {
        Self {
            file_path: PathBuf::from(
                "/home/steven/code/Rust/BitNet-rs/tests/fixtures/gguf_models/corrupted.gguf",
            ),
            architecture: "Unknown".to_string(),
            vocab_size: 0,
            metadata_kvs: HashMap::new(),
            tensor_info: Vec::new(),
            file_content: vec![0x47, 0x47, 0x55, 0x46, 0xFF, 0xFF, 0xFF, 0xFF], // Invalid header
        }
    }

    /// Generate realistic GGUF binary content
    fn generate_gguf_binary(
        architecture: &str,
        vocab_size: u32,
        tensors: &[(&str, Vec<u64>)],
    ) -> Vec<u8> {
        let mut buffer = Vec::new();

        // GGUF magic number
        buffer.extend_from_slice(b"GGUF");

        // Version (3)
        buffer.extend_from_slice(&3u32.to_le_bytes());

        // Tensor count
        buffer.extend_from_slice(&(tensors.len() as u64).to_le_bytes());

        // KV metadata count
        buffer.extend_from_slice(&12u64.to_le_bytes());

        // Add essential metadata
        Self::add_kv_string(&mut buffer, "general.architecture", architecture);
        Self::add_kv_string(&mut buffer, "general.name", "test-model");
        Self::add_kv_uint32(&mut buffer, "llama.vocab_size", vocab_size);
        Self::add_kv_uint32(&mut buffer, "llama.embedding_length", 2048);
        Self::add_kv_uint32(&mut buffer, "llama.block_count", 16);
        Self::add_kv_uint32(&mut buffer, "llama.attention.head_count", 16);
        Self::add_kv_uint32(&mut buffer, "llama.context_length", 4096);
        Self::add_kv_string(&mut buffer, "tokenizer.ggml.model", "llama");
        Self::add_kv_bool(&mut buffer, "tokenizer.ggml.add_bos_token", true);
        Self::add_kv_bool(&mut buffer, "tokenizer.ggml.add_eos_token", false);
        Self::add_kv_string(&mut buffer, "tokenizer.ggml.bos_token", "<s>");
        Self::add_kv_string(&mut buffer, "tokenizer.ggml.eos_token", "</s>");

        // Add tensor info
        let mut offset = 0u64;
        for (name, shape) in tensors {
            // Tensor name
            let name_bytes = name.as_bytes();
            buffer.extend_from_slice(&(name_bytes.len() as u64).to_le_bytes());
            buffer.extend_from_slice(name_bytes);

            // Dimensions
            buffer.extend_from_slice(&(shape.len() as u32).to_le_bytes());
            for &dim in shape {
                buffer.extend_from_slice(&dim.to_le_bytes());
            }

            // Type (I2_S = 24)
            buffer.extend_from_slice(&24u32.to_le_bytes());

            // Offset
            buffer.extend_from_slice(&offset.to_le_bytes());

            let size = shape.iter().product::<u64>() * 2 / 8; // 2-bit quantized
            offset += size;
        }

        // Align to 32 bytes
        while buffer.len() % 32 != 0 {
            buffer.push(0);
        }

        // Add mock tensor data
        let total_tensor_size = offset;
        buffer.resize(buffer.len() + total_tensor_size as usize, 0x42); // Fill with test pattern

        buffer
    }

    fn add_kv_string(buffer: &mut Vec<u8>, key: &str, value: &str) {
        // Key
        let key_bytes = key.as_bytes();
        buffer.extend_from_slice(&(key_bytes.len() as u64).to_le_bytes());
        buffer.extend_from_slice(key_bytes);

        // Value type (8 = STRING)
        buffer.extend_from_slice(&8u32.to_le_bytes());

        // Value
        let value_bytes = value.as_bytes();
        buffer.extend_from_slice(&(value_bytes.len() as u64).to_le_bytes());
        buffer.extend_from_slice(value_bytes);
    }

    fn add_kv_uint32(buffer: &mut Vec<u8>, key: &str, value: u32) {
        // Key
        let key_bytes = key.as_bytes();
        buffer.extend_from_slice(&(key_bytes.len() as u64).to_le_bytes());
        buffer.extend_from_slice(key_bytes);

        // Value type (4 = UINT32)
        buffer.extend_from_slice(&4u32.to_le_bytes());

        // Value
        buffer.extend_from_slice(&value.to_le_bytes());
    }

    fn add_kv_bool(buffer: &mut Vec<u8>, key: &str, value: bool) {
        // Key
        let key_bytes = key.as_bytes();
        buffer.extend_from_slice(&(key_bytes.len() as u64).to_le_bytes());
        buffer.extend_from_slice(key_bytes);

        // Value type (7 = BOOL)
        buffer.extend_from_slice(&7u32.to_le_bytes());

        // Value
        buffer.push(if value { 1 } else { 0 });
    }
}

/// Fixture manager for tokenizer test data
pub struct TokenizerFixtures {
    pub fixtures: HashMap<TokenizerType, TokenizerTestFixture>,
    pub gguf_models: Vec<MockGgufModel>,
    pub fixtures_dir: PathBuf,
}

impl TokenizerFixtures {
    /// Initialize all tokenizer test fixtures
    pub fn new() -> Self {
        let mut fixtures = HashMap::new();
        fixtures.insert(TokenizerType::LLaMA3, LLAMA3_FIXTURE.clone());
        fixtures.insert(TokenizerType::LLaMA2, LLAMA2_FIXTURE.clone());
        fixtures.insert(TokenizerType::GPT2, GPT2_FIXTURE.clone());

        let gguf_models = vec![
            MockGgufModel::create_llama3_128k(),
            MockGgufModel::create_llama2_32k(),
            MockGgufModel::create_gpt2_50k(),
            MockGgufModel::create_corrupted_gguf(),
        ];

        Self {
            fixtures,
            gguf_models,
            fixtures_dir: PathBuf::from("/home/steven/code/Rust/BitNet-rs/tests/fixtures"),
        }
    }

    /// Get fixture by tokenizer type
    pub fn get_fixture(&self, tokenizer_type: &TokenizerType) -> Option<&TokenizerTestFixture> {
        self.fixtures.get(tokenizer_type)
    }

    /// Get GGUF model by vocabulary size
    pub fn get_gguf_model_by_vocab(&self, vocab_size: u32) -> Option<&MockGgufModel> {
        self.gguf_models.iter().find(|model| model.vocab_size == vocab_size)
    }

    /// Get all test prompts for cross-validation
    pub fn get_all_test_prompts(&self) -> Vec<(&TokenizerType, &TestPrompt)> {
        let mut prompts = Vec::new();
        for (tokenizer_type, fixture) in &self.fixtures {
            for prompt in &fixture.test_prompts {
                prompts.push((tokenizer_type, prompt));
            }
        }
        prompts
    }

    /// Get quantization-compatible test cases
    pub fn get_quantization_test_cases(
        &self,
        quantization_type: &str,
    ) -> Vec<&ExpectedTokenization> {
        let mut test_cases = Vec::new();
        for fixture in self.fixtures.values() {
            for expected in &fixture.expected_outputs {
                if expected.quantization_compatible.contains(&quantization_type.to_string()) {
                    test_cases.push(expected);
                }
            }
        }
        test_cases
    }

    /// Write all fixtures to disk for testing
    pub async fn write_all_fixtures(&self) -> Result<()> {
        use tokio::fs;

        // Create directory structure
        fs::create_dir_all(&self.fixtures_dir.join("gguf_models"))
            .await
            .map_err(BitNetError::Io)?;
        fs::create_dir_all(&self.fixtures_dir.join("tokenizers")).await.map_err(BitNetError::Io)?;

        // Write GGUF model files
        for model in &self.gguf_models {
            fs::write(&model.file_path, &model.file_content).await.map_err(BitNetError::Io)?;
        }

        // Write tokenizer JSON files
        for (tokenizer_type, fixture) in &self.fixtures {
            let tokenizer_json = self.generate_tokenizer_json(fixture)?;
            let filename = format!("{:?}_tokenizer.json", tokenizer_type).to_lowercase();
            let filepath = self.fixtures_dir.join("tokenizers").join(filename);
            fs::write(filepath, tokenizer_json).await.map_err(BitNetError::Io)?;
        }

        Ok(())
    }

    /// Generate HuggingFace tokenizer.json format
    fn generate_tokenizer_json(&self, fixture: &TokenizerTestFixture) -> Result<String> {
        let tokenizer_config = match fixture.tokenizer_type {
            TokenizerType::LLaMA3 => serde_json::json!({
                "version": "1.0",
                "truncation": null,
                "padding": null,
                "added_tokens": [
                    {
                        "id": 128000,
                        "content": "<|begin_of_text|>",
                        "single_word": false,
                        "lstrip": false,
                        "rstrip": false,
                        "normalized": false,
                        "special": true
                    },
                    {
                        "id": 128001,
                        "content": "<|end_of_text|>",
                        "single_word": false,
                        "lstrip": false,
                        "rstrip": false,
                        "normalized": false,
                        "special": true
                    }
                ],
                "normalizer": {
                    "type": "NFC"
                },
                "pre_tokenizer": {
                    "type": "Metaspace",
                    "replacement": "▁",
                    "add_prefix_space": true,
                    "prepend_scheme": "first"
                },
                "post_processor": {
                    "type": "TemplateProcessing",
                    "single": [
                        {
                            "SpecialToken": {
                                "id": "<|begin_of_text|>",
                                "type_id": 0
                            }
                        },
                        {
                            "Sequence": {
                                "id": "A",
                                "type_id": 0
                            }
                        }
                    ],
                    "pair": [
                        {
                            "SpecialToken": {
                                "id": "<|begin_of_text|>",
                                "type_id": 0
                            }
                        },
                        {
                            "Sequence": {
                                "id": "A",
                                "type_id": 0
                            }
                        },
                        {
                            "Sequence": {
                                "id": "B",
                                "type_id": 1
                            }
                        }
                    ],
                    "special_tokens": {
                        "<|begin_of_text|>": {
                            "id": "<|begin_of_text|>",
                            "ids": [128000],
                            "tokens": ["<|begin_of_text|>"]
                        }
                    }
                },
                "decoder": {
                    "type": "Metaspace",
                    "replacement": "▁",
                    "add_prefix_space": true,
                    "prepend_scheme": "first"
                },
                "model": {
                    "type": "BPE",
                    "dropout": null,
                    "unk_token": null,
                    "continuing_subword_prefix": null,
                    "end_of_word_suffix": null,
                    "fuse_unk": false,
                    "byte_fallback": true,
                    "vocab": {
                        "<|begin_of_text|>": 128000,
                        "<|end_of_text|>": 128001,
                        "Hello": 9906,
                        "world": 1917,
                        "▁Neural": 8989,
                        "▁network": 4632
                    },
                    "merges": [
                        "H e",
                        "l l",
                        "o o",
                        "w o",
                        "r l",
                        "d d"
                    ]
                }
            }),
            TokenizerType::LLaMA2 => serde_json::json!({
                "version": "1.0",
                "model": {
                    "type": "BPE",
                    "vocab": {
                        "<s>": 1,
                        "</s>": 2,
                        "<unk>": 0,
                        "Hello": 15043,
                        "▁world": 3186
                    },
                    "merges": ["H e", "l l", "o o"]
                },
                "normalizer": {"type": "NFC"},
                "pre_tokenizer": {
                    "type": "Metaspace",
                    "replacement": "▁",
                    "add_prefix_space": true
                }
            }),
            TokenizerType::GPT2 => serde_json::json!({
                "version": "1.0",
                "model": {
                    "type": "BPE",
                    "vocab": {
                        "<|endoftext|>": 50256,
                        "Hello": 15496,
                        " world": 995
                    },
                    "merges": ["H e", "l l", "o o"]
                },
                "normalizer": null,
                "pre_tokenizer": {
                    "type": "ByteLevel",
                    "add_prefix_space": false,
                    "trim_offsets": true
                }
            }),
            _ => return Err(BitNetError::Configuration("Unsupported tokenizer type".to_string())),
        };

        serde_json::to_string_pretty(&tokenizer_config).map_err(|e| {
            BitNetError::Configuration(format!("Failed to serialize tokenizer config: {}", e))
        })
    }
}

/// Load tokenizer test fixture by type with proper feature gates
#[cfg(test)]
pub fn load_tokenizer_fixture(tokenizer_type: TokenizerType) -> &'static TokenizerTestFixture {
    match tokenizer_type {
        TokenizerType::LLaMA3 => &LLAMA3_FIXTURE,
        TokenizerType::LLaMA2 => &LLAMA2_FIXTURE,
        TokenizerType::GPT2 => &GPT2_FIXTURE,
        _ => panic!("Fixture not available for tokenizer type: {:?}", tokenizer_type),
    }
}

/// CPU-specific fixture loading utilities
#[cfg(feature = "cpu")]
pub mod cpu_fixtures {
    use super::*;

    pub fn load_cpu_optimized_fixtures() -> Vec<&'static TokenizerTestFixture> {
        vec![&LLAMA3_FIXTURE, &LLAMA2_FIXTURE, &GPT2_FIXTURE]
    }

    pub fn get_cpu_quantization_test_data() -> Vec<(&'static str, u32, Vec<u32>)> {
        vec![
            ("I2S", 128256, vec![9906, 1917, 8989, 4632]), // LLaMA-3 compatible
            ("TL1", 32000, vec![15043, 3186]),             // LLaMA-2 compatible
            ("TL2", 50257, vec![15496, 995]),              // GPT-2 compatible
        ]
    }
}

/// GPU-specific fixture loading utilities
#[cfg(feature = "gpu")]
pub mod gpu_fixtures {
    use super::*;

    pub fn load_gpu_accelerated_fixtures() -> Vec<&'static TokenizerTestFixture> {
        // GPU optimized for large vocabularies with I2S quantization
        vec![&LLAMA3_FIXTURE]
    }

    pub fn get_mixed_precision_test_data() -> Vec<(&'static str, &'static str, Vec<u32>)> {
        vec![("FP16", "LLaMA3", vec![9906, 1917]), ("BF16", "LLaMA3", vec![8989, 4632])]
    }
}

/// SentencePiece-specific fixtures
#[cfg(feature = "spm")]
pub mod spm_fixtures {
    use super::*;

    pub fn create_spm_model_fixture() -> Vec<u8> {
        // Mock SentencePiece binary model file
        let mut buffer = Vec::new();

        // SentencePiece magic header
        buffer.extend_from_slice(b"\x08\x01\x12\x04test"); // Protocol buffer format

        // Mock vocabulary entries
        buffer.extend_from_slice(b"\x1a\x06\x08\x01\x12\x02<s>"); // <s> token
        buffer.extend_from_slice(b"\x1a\x07\x08\x02\x12\x03</s>"); // </s> token
        buffer.extend_from_slice(b"\x1a\x08\x08\x03\x12\x04<unk>"); // <unk> token

        buffer
    }

    pub fn get_spm_test_cases() -> Vec<(&'static str, Vec<u32>)> {
        vec![("Hello world", vec![1, 15043, 3186, 2]), ("Neural network", vec![1, 8989, 4632, 2])]
    }
}
