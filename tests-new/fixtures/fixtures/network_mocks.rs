//! Network mocking data for BitNet.rs tokenizer download and HuggingFace API testing
//!
//! Provides comprehensive mock data for testing tokenizer discovery, download scenarios,
//! network failures, and HuggingFace Hub integration without requiring real network access.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::LazyLock;

/// HuggingFace Hub API response mock data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HuggingFaceApiResponse {
    pub endpoint: String,
    pub status_code: u16,
    pub headers: HashMap<String, String>,
    pub response_body: String,
    pub delay_ms: Option<u64>,
    pub should_fail: bool,
    pub failure_reason: Option<String>,
}

/// Model repository information mock data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelRepositoryMock {
    pub repo_id: String,
    pub model_files: Vec<ModelFileMock>,
    pub tokenizer_files: Vec<TokenizerFileMock>,
    pub config_files: Vec<ConfigFileMock>,
    pub readme_content: String,
    pub model_card: ModelCardMock,
    pub download_count: u64,
    pub last_modified: String,
}

/// Mock model file metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelFileMock {
    pub filename: String,
    pub size: u64,
    pub download_url: String,
    pub sha256: String,
    pub lfs_pointer: Option<LfsPointerMock>,
}

/// Mock tokenizer file metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenizerFileMock {
    pub filename: String,
    pub tokenizer_type: String,
    pub size: u64,
    pub download_url: String,
    pub sha256: String,
    pub content: Option<String>, // For small files like tokenizer.json
}

/// Mock configuration file metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfigFileMock {
    pub filename: String,
    pub size: u64,
    pub content: String,
}

/// Git LFS pointer mock data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LfsPointerMock {
    pub oid: String,
    pub size: u64,
    pub download_url: String,
}

/// Model card mock data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelCardMock {
    pub model_type: String,
    pub license: String,
    pub language: Vec<String>,
    pub datasets: Vec<String>,
    pub metrics: HashMap<String, f64>,
    pub widget: Vec<WidgetExampleMock>,
}

/// Widget example for model card
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WidgetExampleMock {
    pub text: String,
    pub example_title: Option<String>,
}

/// Network error simulation scenarios
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkErrorScenario {
    pub scenario_name: String,
    pub error_type: String,
    pub trigger_condition: String,
    pub error_message: String,
    pub retry_after_ms: Option<u64>,
    pub max_retries: u32,
    pub should_recover: bool,
}

// Static mock data for different model repositories

/// Microsoft BitNet 2B model repository mock
static MICROSOFT_BITNET_2B_MOCK: LazyLock<ModelRepositoryMock> = LazyLock::new(|| {
    ModelRepositoryMock {
    repo_id: "microsoft/bitnet-b1.58-2B-4T-gguf".to_string(),
    model_files: vec![
        ModelFileMock {
            filename: "ggml-model-i2_s.gguf".to_string(),
            size: 1_800_000_000, // ~1.8GB
            download_url: "https://huggingface.co/microsoft/bitnet-b1.58-2B-4T-gguf/resolve/main/ggml-model-i2_s.gguf".to_string(),
            sha256: "abcd1234567890abcd1234567890abcd1234567890abcd1234567890abcd1234".to_string(),
            lfs_pointer: Some(LfsPointerMock {
                oid: "sha256:abcd1234567890abcd1234567890abcd1234567890abcd1234567890abcd1234".to_string(),
                size: 1_800_000_000,
                download_url: "https://cdn-lfs.huggingface.co/repos/96/27/...".to_string(),
            }),
        },
        ModelFileMock {
            filename: "ggml-model-f16.gguf".to_string(),
            size: 3_600_000_000, // ~3.6GB
            download_url: "https://huggingface.co/microsoft/bitnet-b1.58-2B-4T-gguf/resolve/main/ggml-model-f16.gguf".to_string(),
            sha256: "efgh5678901234efgh5678901234efgh5678901234efgh5678901234efgh5678".to_string(),
            lfs_pointer: Some(LfsPointerMock {
                oid: "sha256:efgh5678901234efgh5678901234efgh5678901234efgh5678901234efgh5678".to_string(),
                size: 3_600_000_000,
                download_url: "https://cdn-lfs.huggingface.co/repos/96/27/...".to_string(),
            }),
        },
    ],
    tokenizer_files: vec![
        TokenizerFileMock {
            filename: "tokenizer.json".to_string(),
            tokenizer_type: "LLaMA-3".to_string(),
            size: 17_000_000, // ~17MB
            download_url: "https://huggingface.co/microsoft/bitnet-b1.58-2B-4T-gguf/resolve/main/tokenizer.json".to_string(),
            sha256: "ijkl9012345678ijkl9012345678ijkl9012345678ijkl9012345678ijkl9012".to_string(),
            content: Some(include_str!("tokenizers/llama3_tokenizer.json").to_string()),
        },
        TokenizerFileMock {
            filename: "tokenizer_config.json".to_string(),
            tokenizer_type: "config".to_string(),
            size: 1024,
            download_url: "https://huggingface.co/microsoft/bitnet-b1.58-2B-4T-gguf/resolve/main/tokenizer_config.json".to_string(),
            sha256: "mnop3456789012mnop3456789012mnop3456789012mnop3456789012mnop3456".to_string(),
            content: Some(r#"{
                "add_bos_token": true,
                "add_eos_token": false,
                "bos_token": {
                    "__type": "AddedToken",
                    "content": "<|begin_of_text|>",
                    "lstrip": false,
                    "normalized": false,
                    "rstrip": false,
                    "single_word": false,
                    "special": true
                },
                "eos_token": {
                    "__type": "AddedToken",
                    "content": "<|end_of_text|>",
                    "lstrip": false,
                    "normalized": false,
                    "rstrip": false,
                    "single_word": false,
                    "special": true
                },
                "tokenizer_class": "LlamaTokenizer"
            }"#.to_string()),
        },
    ],
    config_files: vec![
        ConfigFileMock {
            filename: "config.json".to_string(),
            size: 2048,
            content: r#"{
                "architectures": ["BitNetForCausalLM"],
                "model_type": "bitnet",
                "vocab_size": 128256,
                "hidden_size": 2048,
                "intermediate_size": 5504,
                "num_hidden_layers": 24,
                "num_attention_heads": 16,
                "num_key_value_heads": 16,
                "max_position_embeddings": 8192,
                "torch_dtype": "bfloat16",
                "quantization_config": {
                    "quant_type": "I2_S",
                    "group_size": 128,
                    "bits": 2
                }
            }"#.to_string(),
        },
    ],
    readme_content: "# BitNet b1.58 2B Model\n\nA 2-billion parameter BitNet model with 1.58-bit quantization.".to_string(),
    model_card: ModelCardMock {
        model_type: "bitnet".to_string(),
        license: "mit".to_string(),
        language: vec!["en".to_string()],
        datasets: vec!["RedPajama-Data-1T".to_string(), "The Stack".to_string()],
        metrics: {
            let mut metrics = HashMap::new();
            metrics.insert("perplexity".to_string(), 12.5);
            metrics.insert("accuracy".to_string(), 0.85);
            metrics
        },
        widget: vec![
            WidgetExampleMock {
                text: "The future of AI is".to_string(),
                example_title: Some("Text Generation".to_string()),
            },
        ],
    },
    download_count: 15420,
    last_modified: "2024-03-15T10:30:00Z".to_string(),
}
});

/// LLaMA-2 7B model repository mock
static LLAMA2_7B_MOCK: LazyLock<ModelRepositoryMock> = LazyLock::new(|| ModelRepositoryMock {
    repo_id: "meta-llama/Llama-2-7b-hf".to_string(),
    model_files: vec![ModelFileMock {
        filename: "pytorch_model.bin".to_string(),
        size: 13_500_000_000, // ~13.5GB
        download_url:
            "https://huggingface.co/meta-llama/Llama-2-7b-hf/resolve/main/pytorch_model.bin"
                .to_string(),
        sha256: "qrst7890123456qrst7890123456qrst7890123456qrst7890123456qrst7890".to_string(),
        lfs_pointer: Some(LfsPointerMock {
            oid: "sha256:qrst7890123456qrst7890123456qrst7890123456qrst7890123456qrst7890"
                .to_string(),
            size: 13_500_000_000,
            download_url: "https://cdn-lfs.huggingface.co/repos/18/93/...".to_string(),
        }),
    }],
    tokenizer_files: vec![
        TokenizerFileMock {
            filename: "tokenizer.json".to_string(),
            tokenizer_type: "LLaMA-2".to_string(),
            size: 1_800_000, // ~1.8MB
            download_url:
                "https://huggingface.co/meta-llama/Llama-2-7b-hf/resolve/main/tokenizer.json"
                    .to_string(),
            sha256: "uvwx4567890123uvwx4567890123uvwx4567890123uvwx4567890123uvwx4567".to_string(),
            content: Some(include_str!("tokenizers/llama2_tokenizer.json").to_string()),
        },
        TokenizerFileMock {
            filename: "tokenizer.model".to_string(),
            tokenizer_type: "SentencePiece".to_string(),
            size: 500_000, // ~500KB
            download_url:
                "https://huggingface.co/meta-llama/Llama-2-7b-hf/resolve/main/tokenizer.model"
                    .to_string(),
            sha256: "yzab8901234567yzab8901234567yzab8901234567yzab8901234567yzab8901".to_string(),
            content: None, // Binary file
        },
    ],
    config_files: vec![ConfigFileMock {
        filename: "config.json".to_string(),
        size: 1024,
        content: r#"{
                "architectures": ["LlamaForCausalLM"],
                "model_type": "llama",
                "vocab_size": 32000,
                "hidden_size": 4096,
                "intermediate_size": 11008,
                "num_hidden_layers": 32,
                "num_attention_heads": 32,
                "max_position_embeddings": 4096,
                "torch_dtype": "float16"
            }"#
        .to_string(),
    }],
    readme_content: "# Llama 2 7B Model\n\nA 7-billion parameter language model from Meta."
        .to_string(),
    model_card: ModelCardMock {
        model_type: "llama".to_string(),
        license: "llama2".to_string(),
        language: vec!["en".to_string()],
        datasets: vec!["custom".to_string()],
        metrics: {
            let mut metrics = HashMap::new();
            metrics.insert("perplexity".to_string(), 9.2);
            metrics.insert("accuracy".to_string(), 0.91);
            metrics
        },
        widget: vec![WidgetExampleMock {
            text: "The capital of France is".to_string(),
            example_title: Some("Question Answering".to_string()),
        }],
    },
    download_count: 285_420,
    last_modified: "2023-07-18T15:45:00Z".to_string(),
});

/// HuggingFace API response mocks
static HUGGINGFACE_API_RESPONSES: LazyLock<Vec<HuggingFaceApiResponse>> = LazyLock::new(|| {
    vec![
    // Model info API response
    HuggingFaceApiResponse {
        endpoint: "https://huggingface.co/api/models/microsoft/bitnet-b1.58-2B-4T-gguf".to_string(),
        status_code: 200,
        headers: {
            let mut headers = HashMap::new();
            headers.insert("content-type".to_string(), "application/json".to_string());
            headers.insert("cache-control".to_string(), "max-age=300".to_string());
            headers
        },
        response_body: r#"{
            "modelId": "microsoft/bitnet-b1.58-2B-4T-gguf",
            "author": "microsoft",
            "sha": "main",
            "lastModified": "2024-03-15T10:30:00.000Z",
            "private": false,
            "disabled": false,
            "gated": false,
            "pipeline_tag": "text-generation",
            "tags": ["pytorch", "bitnet", "quantized", "gguf"],
            "downloads": 15420,
            "library_name": "transformers",
            "likes": 892,
            "model-index": null,
            "config": {
                "architectures": ["BitNetForCausalLM"],
                "model_type": "bitnet"
            },
            "siblings": [
                {
                    "rfilename": "ggml-model-i2_s.gguf"
                },
                {
                    "rfilename": "tokenizer.json"
                },
                {
                    "rfilename": "config.json"
                }
            ]
        }"#.to_string(),
        delay_ms: Some(50),
        should_fail: false,
        failure_reason: None,
    },
    // File download response - success
    HuggingFaceApiResponse {
        endpoint: "https://huggingface.co/microsoft/bitnet-b1.58-2B-4T-gguf/resolve/main/tokenizer.json".to_string(),
        status_code: 200,
        headers: {
            let mut headers = HashMap::new();
            headers.insert("content-type".to_string(), "application/json".to_string());
            headers.insert("content-length".to_string(), "17000000".to_string());
            headers.insert("etag".to_string(), "\"abcd1234567890\"".to_string());
            headers
        },
        response_body: include_str!("tokenizers/llama3_tokenizer.json").to_string(),
        delay_ms: Some(200),
        should_fail: false,
        failure_reason: None,
    },
    // Network error simulation - timeout
    HuggingFaceApiResponse {
        endpoint: "https://huggingface.co/slow-endpoint".to_string(),
        status_code: 408,
        headers: HashMap::new(),
        response_body: "Request timeout".to_string(),
        delay_ms: Some(30000), // 30 seconds
        should_fail: true,
        failure_reason: Some("Timeout".to_string()),
    },
    // Rate limiting response
    HuggingFaceApiResponse {
        endpoint: "https://huggingface.co/rate-limited".to_string(),
        status_code: 429,
        headers: {
            let mut headers = HashMap::new();
            headers.insert("retry-after".to_string(), "60".to_string());
            headers.insert("x-ratelimit-limit".to_string(), "1000".to_string());
            headers.insert("x-ratelimit-remaining".to_string(), "0".to_string());
            headers
        },
        response_body: r#"{"error": "Rate limit exceeded"}"#.to_string(),
        delay_ms: Some(100),
        should_fail: true,
        failure_reason: Some("RateLimited".to_string()),
    },
    // Repository not found
    HuggingFaceApiResponse {
        endpoint: "https://huggingface.co/api/models/nonexistent/model".to_string(),
        status_code: 404,
        headers: HashMap::new(),
        response_body: r#"{"error": "Repository not found"}"#.to_string(),
        delay_ms: Some(50),
        should_fail: true,
        failure_reason: Some("NotFound".to_string()),
    },
]
});

/// Network error scenarios for testing
static NETWORK_ERROR_SCENARIOS: LazyLock<Vec<NetworkErrorScenario>> = LazyLock::new(|| {
    vec![
        NetworkErrorScenario {
            scenario_name: "connection_timeout".to_string(),
            error_type: "Timeout".to_string(),
            trigger_condition: "request_duration > 5s".to_string(),
            error_message: "Connection timeout after 5 seconds".to_string(),
            retry_after_ms: Some(1000),
            max_retries: 3,
            should_recover: true,
        },
        NetworkErrorScenario {
            scenario_name: "dns_resolution_failure".to_string(),
            error_type: "DnsError".to_string(),
            trigger_condition: "hostname == 'invalid.domain'".to_string(),
            error_message: "DNS resolution failed for invalid.domain".to_string(),
            retry_after_ms: Some(5000),
            max_retries: 2,
            should_recover: false,
        },
        NetworkErrorScenario {
            scenario_name: "ssl_certificate_error".to_string(),
            error_type: "TlsError".to_string(),
            trigger_condition: "url.starts_with('https://expired-cert.')'".to_string(),
            error_message: "SSL certificate verification failed".to_string(),
            retry_after_ms: None,
            max_retries: 0,
            should_recover: false,
        },
        NetworkErrorScenario {
            scenario_name: "connection_refused".to_string(),
            error_type: "ConnectionRefused".to_string(),
            trigger_condition: "port == 9999".to_string(),
            error_message: "Connection refused to localhost:9999".to_string(),
            retry_after_ms: Some(2000),
            max_retries: 5,
            should_recover: true,
        },
        NetworkErrorScenario {
            scenario_name: "partial_download_corruption".to_string(),
            error_type: "CorruptedData".to_string(),
            trigger_condition: "download_progress >= 0.5 && random() < 0.1".to_string(),
            error_message: "Downloaded data corrupted - checksum mismatch".to_string(),
            retry_after_ms: Some(1000),
            max_retries: 3,
            should_recover: true,
        },
        NetworkErrorScenario {
            scenario_name: "server_internal_error".to_string(),
            error_type: "ServerError".to_string(),
            trigger_condition: "status_code == 500".to_string(),
            error_message: "Internal server error - please try again later".to_string(),
            retry_after_ms: Some(5000),
            max_retries: 2,
            should_recover: true,
        },
    ]
});

/// Network mock fixtures manager
pub struct NetworkMockFixtures {
    pub model_repositories: HashMap<String, ModelRepositoryMock>,
    pub api_responses: Vec<HuggingFaceApiResponse>,
    pub error_scenarios: Vec<NetworkErrorScenario>,
    pub download_patterns: HashMap<String, DownloadPattern>,
}

/// Download pattern for simulating realistic download behavior
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DownloadPattern {
    pub name: String,
    pub file_size: u64,
    pub chunk_size: usize,
    pub bandwidth_kbps: u32,
    pub error_probability: f64,
    pub resume_support: bool,
}

impl NetworkMockFixtures {
    /// Initialize network mock fixtures
    pub fn new() -> Self {
        let mut model_repositories = HashMap::new();
        model_repositories.insert(
            "microsoft/bitnet-b1.58-2B-4T-gguf".to_string(),
            MICROSOFT_BITNET_2B_MOCK.clone(),
        );
        model_repositories.insert("meta-llama/Llama-2-7b-hf".to_string(), LLAMA2_7B_MOCK.clone());

        let mut download_patterns = HashMap::new();
        download_patterns.insert(
            "fast_local".to_string(),
            DownloadPattern {
                name: "fast_local".to_string(),
                file_size: 1_000_000,   // 1MB
                chunk_size: 65536,      // 64KB chunks
                bandwidth_kbps: 10_000, // 10MB/s
                error_probability: 0.001,
                resume_support: true,
            },
        );
        download_patterns.insert(
            "slow_network".to_string(),
            DownloadPattern {
                name: "slow_network".to_string(),
                file_size: 1_800_000_000, // 1.8GB
                chunk_size: 8192,         // 8KB chunks
                bandwidth_kbps: 500,      // 500KB/s
                error_probability: 0.05,
                resume_support: true,
            },
        );
        download_patterns.insert(
            "unreliable_connection".to_string(),
            DownloadPattern {
                name: "unreliable_connection".to_string(),
                file_size: 17_000_000, // 17MB
                chunk_size: 4096,      // 4KB chunks
                bandwidth_kbps: 1000,  // 1MB/s
                error_probability: 0.1,
                resume_support: false,
            },
        );

        Self {
            model_repositories,
            api_responses: HUGGINGFACE_API_RESPONSES.clone(),
            error_scenarios: NETWORK_ERROR_SCENARIOS.clone(),
            download_patterns,
        }
    }

    /// Get mock data for specific model repository
    pub fn get_repository_mock(&self, repo_id: &str) -> Option<&ModelRepositoryMock> {
        self.model_repositories.get(repo_id)
    }

    /// Get API response mock for specific endpoint
    pub fn get_api_response(&self, endpoint: &str) -> Option<&HuggingFaceApiResponse> {
        self.api_responses.iter().find(|response| response.endpoint == endpoint)
    }

    /// Get network error scenario by name
    pub fn get_error_scenario(&self, scenario_name: &str) -> Option<&NetworkErrorScenario> {
        self.error_scenarios.iter().find(|scenario| scenario.scenario_name == scenario_name)
    }

    /// Get download pattern by name
    pub fn get_download_pattern(&self, pattern_name: &str) -> Option<&DownloadPattern> {
        self.download_patterns.get(pattern_name)
    }

    /// Generate mock tokenizer file content
    pub fn generate_mock_tokenizer_content(&self, tokenizer_type: &str, vocab_size: u32) -> String {
        match tokenizer_type {
            "LLaMA-3" => serde_json::json!({
                "version": "1.0",
                "model": {
                    "type": "BPE",
                    "vocab_size": vocab_size,
                    "vocab": {
                        "<|begin_of_text|>": 128000,
                        "<|end_of_text|>": 128001,
                        "Hello": 9906,
                        "world": 1917,
                        "▁Neural": 8989,
                        "▁network": 4632
                    },
                    "merges": ["H e", "l l", "o o", "w o", "r l", "d d"]
                }
            })
            .to_string(),
            "LLaMA-2" => serde_json::json!({
                "version": "1.0",
                "model": {
                    "type": "BPE",
                    "vocab_size": vocab_size,
                    "vocab": {
                        "<s>": 1,
                        "</s>": 2,
                        "<unk>": 0,
                        "Hello": 15043,
                        "▁world": 3186
                    },
                    "merges": ["H e", "l l", "o o"]
                }
            })
            .to_string(),
            "GPT-2" => serde_json::json!({
                "version": "1.0",
                "model": {
                    "type": "BPE",
                    "vocab_size": vocab_size,
                    "vocab": {
                        "<|endoftext|>": 50256,
                        "Hello": 15496,
                        " world": 995
                    },
                    "merges": ["H e", "l l", "o o"]
                }
            })
            .to_string(),
            _ => serde_json::json!({"error": "Unknown tokenizer type"}).to_string(),
        }
    }

    /// Create mock HTTP server responses for testing
    pub fn create_http_mock_responses(
        &self,
    ) -> HashMap<String, (u16, String, HashMap<String, String>)> {
        let mut responses = HashMap::new();

        for response in &self.api_responses {
            responses.insert(
                response.endpoint.clone(),
                (response.status_code, response.response_body.clone(), response.headers.clone()),
            );
        }

        responses
    }

    /// Write all network mock data to JSON files
    pub async fn write_network_mocks(
        &self,
        fixtures_dir: &std::path::Path,
    ) -> Result<(), Box<dyn std::error::Error>> {
        use tokio::fs;

        let network_dir = fixtures_dir.join("network_mocks");
        fs::create_dir_all(&network_dir).await?;

        // Write repository mocks
        let repositories_json = serde_json::to_string_pretty(&self.model_repositories)?;
        fs::write(network_dir.join("model_repositories.json"), repositories_json).await?;

        // Write API response mocks
        let api_responses_json = serde_json::to_string_pretty(&self.api_responses)?;
        fs::write(network_dir.join("api_responses.json"), api_responses_json).await?;

        // Write error scenarios
        let error_scenarios_json = serde_json::to_string_pretty(&self.error_scenarios)?;
        fs::write(network_dir.join("error_scenarios.json"), error_scenarios_json).await?;

        // Write download patterns
        let download_patterns_json = serde_json::to_string_pretty(&self.download_patterns)?;
        fs::write(network_dir.join("download_patterns.json"), download_patterns_json).await?;

        // Write individual tokenizer files
        let tokenizers_dir = network_dir.join("tokenizer_files");
        fs::create_dir_all(&tokenizers_dir).await?;

        for repo in self.model_repositories.values() {
            for tokenizer_file in &repo.tokenizer_files {
                if let Some(content) = &tokenizer_file.content {
                    let safe_filename = tokenizer_file.filename.replace('/', "_");
                    let file_path = tokenizers_dir.join(format!(
                        "{}_{}",
                        repo.repo_id.replace('/', "_"),
                        safe_filename
                    ));
                    fs::write(file_path, content).await?;
                }
            }
        }

        Ok(())
    }

    /// Create test scenario for specific network condition
    pub fn create_test_scenario(&self, scenario_type: &str) -> NetworkTestScenario {
        match scenario_type {
            "fast_download" => NetworkTestScenario {
                name: "Fast reliable download".to_string(),
                bandwidth_limit: Some(10_000), // 10MB/s
                latency_ms: 20,
                packet_loss_rate: 0.0,
                connection_errors: Vec::new(),
                timeout_seconds: 30,
            },
            "slow_download" => NetworkTestScenario {
                name: "Slow network download".to_string(),
                bandwidth_limit: Some(100), // 100KB/s
                latency_ms: 200,
                packet_loss_rate: 0.01,
                connection_errors: vec!["TimeoutError".to_string()],
                timeout_seconds: 300, // 5 minutes
            },
            "unreliable_network" => NetworkTestScenario {
                name: "Unreliable network with frequent errors".to_string(),
                bandwidth_limit: Some(500), // 500KB/s
                latency_ms: 500,
                packet_loss_rate: 0.05,
                connection_errors: vec![
                    "TimeoutError".to_string(),
                    "ConnectionReset".to_string(),
                    "PartialContent".to_string(),
                ],
                timeout_seconds: 60,
            },
            _ => NetworkTestScenario {
                name: "Default test scenario".to_string(),
                bandwidth_limit: None,
                latency_ms: 50,
                packet_loss_rate: 0.001,
                connection_errors: Vec::new(),
                timeout_seconds: 30,
            },
        }
    }
}

/// Network test scenario configuration
#[derive(Debug, Clone)]
pub struct NetworkTestScenario {
    pub name: String,
    pub bandwidth_limit: Option<u32>, // KB/s
    pub latency_ms: u32,
    pub packet_loss_rate: f64,
    pub connection_errors: Vec<String>,
    pub timeout_seconds: u32,
}

/// CPU-specific network testing utilities
#[cfg(feature = "cpu")]
pub mod cpu_network_mocks {
    use super::*;

    pub fn get_cpu_optimized_downloads() -> Vec<&'static str> {
        vec!["fast_local", "slow_network"]
    }

    pub fn create_cpu_mock_responses() -> Vec<HuggingFaceApiResponse> {
        vec![HuggingFaceApiResponse {
            endpoint: "https://huggingface.co/api/models/cpu-optimized-model".to_string(),
            status_code: 200,
            headers: HashMap::new(),
            response_body: r#"{"model_type": "cpu_optimized"}"#.to_string(),
            delay_ms: Some(50),
            should_fail: false,
            failure_reason: None,
        }]
    }
}

/// GPU-specific network testing utilities
#[cfg(feature = "gpu")]
pub mod gpu_network_mocks {
    use super::*;

    pub fn get_gpu_model_downloads() -> Vec<&'static str> {
        vec!["microsoft/bitnet-b1.58-2B-4T-gguf"]
    }

    pub fn create_gpu_mock_responses() -> Vec<HuggingFaceApiResponse> {
        vec![HuggingFaceApiResponse {
            endpoint: "https://huggingface.co/api/models/gpu-accelerated-model".to_string(),
            status_code: 200,
            headers: HashMap::new(),
            response_body: r#"{"model_type": "gpu_accelerated", "quantization": "I2_S"}"#
                .to_string(),
            delay_ms: Some(100),
            should_fail: false,
            failure_reason: None,
        }]
    }
}

/// Load network mock fixtures for testing
#[cfg(test)]
pub fn load_network_mock_fixtures() -> NetworkMockFixtures {
    NetworkMockFixtures::new()
}

/// Create minimal network test scenario
#[cfg(test)]
pub fn create_minimal_network_scenario() -> NetworkTestScenario {
    NetworkTestScenario {
        name: "Minimal test".to_string(),
        bandwidth_limit: Some(1000), // 1MB/s
        latency_ms: 10,
        packet_loss_rate: 0.0,
        connection_errors: Vec::new(),
        timeout_seconds: 5,
    }
}

/// Mock HuggingFace tokenizer download simulation
#[cfg(test)]
pub async fn simulate_tokenizer_download(
    repo_id: &str,
    filename: &str,
) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let fixtures = NetworkMockFixtures::new();

    if let Some(repo) = fixtures.get_repository_mock(repo_id) {
        if let Some(tokenizer) = repo.tokenizer_files.iter().find(|t| t.filename == filename) {
            if let Some(content) = &tokenizer.content {
                return Ok(content.as_bytes().to_vec());
            }
        }
    }

    Err("Mock file not found".into())
}

/// Validate download checksum simulation
#[cfg(test)]
pub fn validate_mock_checksum(data: &[u8], expected_sha256: &str) -> bool {
    // Mock checksum validation - in real implementation would use SHA-256
    let mock_hash = format!("{:x}", data.len()); // Simple mock based on length
    expected_sha256.starts_with(&mock_hash[0..8])
}
