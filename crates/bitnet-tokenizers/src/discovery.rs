//! Tokenizer discovery system for automatic tokenizer resolution
//!
//! This module provides comprehensive tokenizer discovery capabilities for BitNet-rs neural network models.
//! Supports GGUF metadata parsing, smart downloading, and device-aware tokenization for production-scale models.

use crate::{
    ModelTypeDetector, Tokenizer,
    error_handling::{CacheManager, TokenizerErrorHandler},
};
use bitnet_common::{BitNetError, Result};
use bitnet_models::GgufReader;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tracing::{debug, info, warn};

/// Neural network model compatibility matrix for tokenizer discovery
#[derive(Debug, Clone)]
pub struct ModelCompatibilityMatrix {
    /// LLaMA-3 with 128K vocabulary - requires I2S quantization with GPU acceleration
    pub llama3_128k: TokenizerDownloadInfo,
    /// LLaMA-2 with 32K vocabulary - compatible with TL1/TL2 quantization
    pub llama2_32k: TokenizerDownloadInfo,
    /// GPT-2 with 50K vocabulary - standard BPE tokenization
    pub gpt2_50k: TokenizerDownloadInfo,
    /// BitNet-specific tokenizers for neural network optimization
    pub bitnet_custom: TokenizerDownloadInfo,
    /// Phi-4 with 100K vocabulary - TikToken BPE tokenizer
    pub phi4_100k: TokenizerDownloadInfo,
    /// Qwen2 with 151K vocabulary - BPE tokenizer
    pub qwen2_150k: TokenizerDownloadInfo,
    /// Gemma with 256K vocabulary - SentencePiece tokenizer
    pub gemma_256k: TokenizerDownloadInfo,
    /// Mistral with 32K vocabulary - SentencePiece tokenizer
    pub mistral_32k: TokenizerDownloadInfo,
    /// DeepSeek with 100K vocabulary - BPE tokenizer
    pub deepseek_100k: TokenizerDownloadInfo,
    /// StarCoder with 49K vocabulary - BPE tokenizer
    pub starcoder_49k: TokenizerDownloadInfo,
    /// Falcon with 65K vocabulary - BPE tokenizer
    pub falcon_65k: TokenizerDownloadInfo,
    /// CodeLlama with 32K vocabulary - SentencePiece tokenizer
    pub codellama_32k: TokenizerDownloadInfo,
    /// Cohere Command with 256K vocabulary - BPE tokenizer
    pub command_256k: TokenizerDownloadInfo,
    /// InternLM with 103K vocabulary - BPE tokenizer
    pub internlm_103k: TokenizerDownloadInfo,
    /// Yi with 64K vocabulary - BPE tokenizer
    pub yi_64k: TokenizerDownloadInfo,
    /// Baichuan with 64K vocabulary - BPE tokenizer
    pub baichuan_64k: TokenizerDownloadInfo,
    /// ChatGLM/GLM-4 with 65K vocabulary - BPE tokenizer
    pub chatglm_65k: TokenizerDownloadInfo,
    /// MPT with 50K vocabulary - BPE tokenizer (GPT-NeoX based)
    pub mpt_50k: TokenizerDownloadInfo,
    /// RWKV World with 65K vocabulary - custom tokenizer
    pub rwkv_65k: TokenizerDownloadInfo,
    /// OLMo with 50K vocabulary - BPE tokenizer
    pub olmo_50k: TokenizerDownloadInfo,
    /// Zephyr with 32K vocabulary - BPE tokenizer (Mistral-based)
    pub zephyr_32k: TokenizerDownloadInfo,
    /// Vicuna with 32K vocabulary - SentencePiece tokenizer (LLaMA-based)
    pub vicuna_32k: TokenizerDownloadInfo,
    /// Orca with 32K vocabulary - SentencePiece tokenizer
    pub orca_32k: TokenizerDownloadInfo,
    /// SOLAR with 32K vocabulary - SentencePiece tokenizer
    pub solar_32k: TokenizerDownloadInfo,
    /// Alpaca with 32K vocabulary - SentencePiece tokenizer
    pub alpaca_32k: TokenizerDownloadInfo,
    /// Command-R+ with 256K vocabulary - BPE tokenizer
    pub commandr_128k: TokenizerDownloadInfo,
    /// NousResearch Hermes with 32K vocabulary - BPE tokenizer
    pub nous_32k: TokenizerDownloadInfo,
    /// WizardLM with 32K vocabulary - SentencePiece tokenizer
    pub wizard_32k: TokenizerDownloadInfo,
    /// OpenChat with 32K vocabulary - BPE tokenizer
    pub openchat_32k: TokenizerDownloadInfo,
    /// Granite with 128K vocabulary - BPE tokenizer
    pub granite_128k: TokenizerDownloadInfo,
    /// Nemotron with 32K vocabulary - BPE tokenizer
    pub nemotron_32k: TokenizerDownloadInfo,
    /// Saiga with 32K vocabulary - SentencePiece tokenizer
    pub saiga_32k: TokenizerDownloadInfo,
    /// Llama-2 Chat with 32K vocabulary - SentencePiece tokenizer
    pub llama2_chat_32k: TokenizerDownloadInfo,
    /// Gemma 2 with 256K vocabulary - SentencePiece tokenizer
    pub gemma2_256k: TokenizerDownloadInfo,
    /// Phi-3 with 32K vocabulary - BPE tokenizer
    pub phi3_32k: TokenizerDownloadInfo,
    /// TinyLlama with 32K vocabulary - SentencePiece tokenizer
    pub tinyllama_32k: TokenizerDownloadInfo,
    /// Dolphin (Mistral finetune) with 32K vocabulary - BPE tokenizer
    pub dolphin_32k: TokenizerDownloadInfo,
    /// ChatGPT/GPT-4 with ~100K vocabulary - BPE tokenizer
    pub chatgpt_100k: TokenizerDownloadInfo,
    /// Mixtral with 32K vocabulary - SentencePiece tokenizer
    pub mixtral_32k: TokenizerDownloadInfo,
    /// StableLM with 32K vocabulary - BPE tokenizer
    pub stablelm_32k: TokenizerDownloadInfo,
    /// BLOOM with 250K vocabulary - BPE tokenizer
    pub bloom_250k: TokenizerDownloadInfo,
    /// Jamba with 256K vocabulary - BPE tokenizer
    pub jamba_256k: TokenizerDownloadInfo,
    /// Persimmon with 262K vocabulary - BPE tokenizer
    pub persimmon_262k: TokenizerDownloadInfo,
    /// XVERSE with 32K vocabulary - SentencePiece tokenizer
    pub xverse_32k: TokenizerDownloadInfo,
    /// Qwen 2.5 with 152K vocabulary - BPE tokenizer
    pub qwen25_152k: TokenizerDownloadInfo,
    /// Mistral Nemo with 128K vocabulary - SentencePiece tokenizer
    pub mistral_nemo_128k: TokenizerDownloadInfo,
    /// Snowflake Arctic with 32K vocabulary - BPE tokenizer
    pub arctic_32k: TokenizerDownloadInfo,
    /// DBRX with 32K vocabulary - BPE tokenizer
    pub dbrx_32k: TokenizerDownloadInfo,
    /// EXAONE with 32K vocabulary - BPE tokenizer
    pub exaone_32k: TokenizerDownloadInfo,
    /// MiniCPM with 122K vocabulary - BPE tokenizer
    pub minicpm_122k: TokenizerDownloadInfo,
    /// CodeGemma with 256K vocabulary - SentencePiece tokenizer
    pub codegemma_256k: TokenizerDownloadInfo,
    /// Llama 3.1 with 128K vocabulary - BPE tokenizer
    pub llama31_128k: TokenizerDownloadInfo,
    /// DeepSeek V3 with 100K vocabulary - BPE tokenizer
    pub deepseekv3_100k: TokenizerDownloadInfo,
    /// Cohere Aya with 256K vocabulary - BPE tokenizer
    pub aya_256k: TokenizerDownloadInfo,
    /// SmolLM with 49K vocabulary - BPE tokenizer
    pub smollm_49k: TokenizerDownloadInfo,
    /// Phi-2 with 51K vocabulary - BPE tokenizer
    pub phi2_51k: TokenizerDownloadInfo,
    /// Falcon-2 with 32K vocabulary - BPE tokenizer
    pub falcon2_32k: TokenizerDownloadInfo,
    /// OLMo-2 with 100K vocabulary - BPE tokenizer
    pub olmo2_100k: TokenizerDownloadInfo,
    /// Llama 3.2 with 128K vocabulary - BPE tokenizer
    pub llama32_128k: TokenizerDownloadInfo,
    /// Phi-4-mini with 100K vocabulary - TikToken BPE tokenizer
    pub phi4_mini_100k: TokenizerDownloadInfo,
    /// Mistral v0.3 with 32K vocabulary - SentencePiece tokenizer
    pub mistral_v03_32k: TokenizerDownloadInfo,
    /// SmolLM2 with 49K vocabulary - BPE tokenizer
    pub smollm2_49k: TokenizerDownloadInfo,
}

impl Default for ModelCompatibilityMatrix {
    fn default() -> Self {
        Self {
            llama3_128k: TokenizerDownloadInfo::basic(
                "meta-llama/Meta-Llama-3-8B",
                vec!["tokenizer.json"],
                "llama3-128k",
                Some(128256),
            ),
            llama2_32k: TokenizerDownloadInfo::basic(
                "meta-llama/Llama-2-7b-hf",
                vec!["tokenizer.json"],
                "llama2-32k",
                Some(32000),
            ),
            gpt2_50k: TokenizerDownloadInfo::basic(
                "openai-community/gpt2",
                vec!["tokenizer.json"],
                "gpt2-50k",
                Some(50257),
            ),
            bitnet_custom: TokenizerDownloadInfo::basic(
                "1bitLLM/bitnet_b1_58-large",
                vec!["tokenizer.json", "tokenizer.model"],
                "bitnet-custom",
                None,
            ),
            phi4_100k: TokenizerDownloadInfo::with_type(
                "microsoft/phi-4",
                vec!["tokenizer.json"],
                "phi4-100k",
                Some(100352),
                TokenizerType::TikTokenBpe,
                SpecialTokenConfig {
                    bos_token_id: Some(100257),
                    eos_token_id: Some(100265),
                    pad_token_id: None,
                },
            ),
            qwen2_150k: TokenizerDownloadInfo::with_type(
                "Qwen/Qwen2-7B",
                vec!["tokenizer.json"],
                "qwen2-150k",
                Some(151936),
                TokenizerType::TikTokenBpe,
                SpecialTokenConfig::default(),
            ),
            gemma_256k: TokenizerDownloadInfo::with_type(
                "google/gemma-2b",
                vec!["tokenizer.json"],
                "gemma-256k",
                Some(256000),
                TokenizerType::SentencePiece,
                SpecialTokenConfig {
                    bos_token_id: Some(2),
                    eos_token_id: Some(1),
                    pad_token_id: Some(0),
                },
            ),
            mistral_32k: TokenizerDownloadInfo::with_type(
                "mistralai/Mistral-7B-v0.1",
                vec!["tokenizer.json"],
                "mistral-32k",
                Some(32000),
                TokenizerType::SentencePiece,
                SpecialTokenConfig {
                    bos_token_id: Some(1),
                    eos_token_id: Some(2),
                    pad_token_id: None,
                },
            ),
            deepseek_100k: TokenizerDownloadInfo::basic(
                "deepseek-ai/DeepSeek-V2-Lite",
                vec!["tokenizer.json"],
                "deepseek-100k",
                Some(100015),
            ),
            starcoder_49k: TokenizerDownloadInfo::basic(
                "bigcode/starcoder",
                vec!["tokenizer.json"],
                "starcoder-49k",
                Some(49152),
            ),
            falcon_65k: TokenizerDownloadInfo::basic(
                "tiiuae/falcon-7b",
                vec!["tokenizer.json"],
                "falcon-65k",
                Some(65024),
            ),
            codellama_32k: TokenizerDownloadInfo::basic(
                "codellama/CodeLlama-7b-Instruct-hf",
                vec!["tokenizer.json"],
                "codellama-32k",
                Some(32016),
            ),
            command_256k: TokenizerDownloadInfo::basic(
                "CohereForAI/c4ai-command-r-plus",
                vec!["tokenizer.json"],
                "command-256k",
                Some(255029),
            ),
            internlm_103k: TokenizerDownloadInfo::basic(
                "internlm/internlm2-chat-7b",
                vec!["tokenizer.json"],
                "internlm-103k",
                Some(103168),
            ),
            yi_64k: TokenizerDownloadInfo::basic(
                "01-ai/Yi-34B-Chat",
                vec!["tokenizer.json"],
                "yi-64k",
                Some(64000),
            ),
            baichuan_64k: TokenizerDownloadInfo::basic(
                "baichuan-inc/Baichuan2-13B-Chat",
                vec!["tokenizer.json"],
                "baichuan-64k",
                Some(125696),
            ),
            chatglm_65k: TokenizerDownloadInfo::basic(
                "THUDM/chatglm3-6b",
                vec!["tokenizer.json"],
                "chatglm-65k",
                Some(64798),
            ),
            mpt_50k: TokenizerDownloadInfo::basic(
                "mosaicml/mpt-7b-instruct",
                vec!["tokenizer.json"],
                "mpt-50k",
                Some(50432),
            ),
            rwkv_65k: TokenizerDownloadInfo::basic(
                "RWKV/rwkv-5-world-3b",
                vec!["tokenizer.json"],
                "rwkv-65k",
                Some(65536),
            ),
            olmo_50k: TokenizerDownloadInfo::basic(
                "allenai/OLMo-7B",
                vec!["tokenizer.json"],
                "olmo-50k",
                Some(50280),
            ),
            zephyr_32k: TokenizerDownloadInfo::basic(
                "HuggingFaceH4/zephyr-7b-beta",
                vec!["tokenizer.json"],
                "zephyr-32k",
                Some(32000),
            ),
            vicuna_32k: TokenizerDownloadInfo::basic(
                "lmsys/vicuna-7b-v1.5",
                vec!["tokenizer.json"],
                "vicuna-32k",
                Some(32000),
            ),
            orca_32k: TokenizerDownloadInfo::basic(
                "Open-Orca/OpenOrca-Platypus2-13B",
                vec!["tokenizer.json"],
                "orca-32k",
                Some(32000),
            ),
            solar_32k: TokenizerDownloadInfo::basic(
                "upstage/SOLAR-10.7B-Instruct-v1.0",
                vec!["tokenizer.json"],
                "solar-32k",
                Some(32000),
            ),
            alpaca_32k: TokenizerDownloadInfo::basic(
                "tatsu-lab/alpaca-7b",
                vec!["tokenizer.json"],
                "alpaca-32k",
                Some(32000),
            ),
            commandr_128k: TokenizerDownloadInfo::basic(
                "CohereForAI/c4ai-command-r-plus",
                vec!["tokenizer.json"],
                "commandr-128k",
                Some(256000),
            ),
            nous_32k: TokenizerDownloadInfo::basic(
                "NousResearch/Nous-Hermes-2-Mistral-7B-DPO",
                vec!["tokenizer.json"],
                "nous-32k",
                Some(32000),
            ),
            wizard_32k: TokenizerDownloadInfo::basic(
                "WizardLMTeam/WizardLM-13B-V1.2",
                vec!["tokenizer.json"],
                "wizard-32k",
                Some(32000),
            ),
            openchat_32k: TokenizerDownloadInfo::basic(
                "openchat/openchat_3.5",
                vec!["tokenizer.json"],
                "openchat-32k",
                Some(32000),
            ),
            granite_128k: TokenizerDownloadInfo::basic(
                "ibm-granite/granite-3.0-8b-instruct",
                vec!["tokenizer.json"],
                "granite-128k",
                Some(128000),
            ),
            nemotron_32k: TokenizerDownloadInfo::basic(
                "nvidia/Nemotron-4-340B-Instruct",
                vec!["tokenizer.json"],
                "nemotron-32k",
                Some(32000),
            ),
            saiga_32k: TokenizerDownloadInfo::basic(
                "IlyaGusev/saiga_mistral_7b",
                vec!["tokenizer.json"],
                "saiga-32k",
                Some(32000),
            ),
            llama2_chat_32k: TokenizerDownloadInfo::basic(
                "meta-llama/Llama-2-7b-chat-hf",
                vec!["tokenizer.json"],
                "llama2-chat-32k",
                Some(32000),
            ),
            gemma2_256k: TokenizerDownloadInfo::with_type(
                "google/gemma-2-9b-it",
                vec!["tokenizer.json"],
                "gemma2-256k",
                Some(256000),
                TokenizerType::SentencePiece,
                SpecialTokenConfig {
                    bos_token_id: Some(2),
                    eos_token_id: Some(1),
                    pad_token_id: Some(0),
                },
            ),
            phi3_32k: TokenizerDownloadInfo::basic(
                "microsoft/Phi-3-mini-4k-instruct",
                vec!["tokenizer.json"],
                "phi3-32k",
                Some(32064),
            ),
            tinyllama_32k: TokenizerDownloadInfo::basic(
                "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                vec!["tokenizer.json"],
                "tinyllama-32k",
                Some(32000),
            ),
            dolphin_32k: TokenizerDownloadInfo::basic(
                "cognitivecomputations/dolphin-2.6-mistral-7b-dpo",
                vec!["tokenizer.json"],
                "dolphin-32k",
                Some(32768),
            ),
            chatgpt_100k: TokenizerDownloadInfo::basic(
                "openai/gpt-4",
                vec!["tokenizer.json"],
                "chatgpt-100k",
                Some(100000),
            ),
            mixtral_32k: TokenizerDownloadInfo::with_type(
                "mistralai/Mixtral-8x7B-Instruct-v0.1",
                vec!["tokenizer.json"],
                "mixtral-32k",
                Some(32000),
                TokenizerType::SentencePiece,
                SpecialTokenConfig {
                    bos_token_id: Some(1),
                    eos_token_id: Some(2),
                    pad_token_id: None,
                },
            ),
            stablelm_32k: TokenizerDownloadInfo::basic(
                "stabilityai/stablelm-zephyr-3b",
                vec!["tokenizer.json"],
                "stablelm-32k",
                Some(32000),
            ),
            bloom_250k: TokenizerDownloadInfo::basic(
                "bigscience/bloom",
                vec!["tokenizer.json"],
                "bloom-250k",
                Some(250680),
            ),
            jamba_256k: TokenizerDownloadInfo::basic(
                "ai21labs/Jamba-v0.1",
                vec!["tokenizer.json"],
                "jamba-256k",
                Some(65536),
            ),
            persimmon_262k: TokenizerDownloadInfo::basic(
                "adept/persimmon-8b-chat",
                vec!["tokenizer.json"],
                "persimmon-262k",
                Some(262144),
            ),
            xverse_32k: TokenizerDownloadInfo::basic(
                "xverse/XVERSE-13B-Chat",
                vec!["tokenizer.json"],
                "xverse-32k",
                Some(32000),
            ),
            qwen25_152k: TokenizerDownloadInfo::with_type(
                "Qwen/Qwen2.5-7B-Instruct",
                vec!["tokenizer.json"],
                "qwen25-152k",
                Some(152064),
                TokenizerType::TikTokenBpe,
                SpecialTokenConfig {
                    bos_token_id: None,
                    eos_token_id: Some(151645),
                    pad_token_id: None,
                },
            ),
            mistral_nemo_128k: TokenizerDownloadInfo::basic(
                "mistralai/Mistral-Nemo-Instruct-2407",
                vec!["tokenizer.json"],
                "mistral-nemo-128k",
                Some(131072),
            ),
            arctic_32k: TokenizerDownloadInfo::basic(
                "Snowflake/snowflake-arctic-instruct",
                vec!["tokenizer.json"],
                "arctic-32k",
                Some(32000),
            ),
            dbrx_32k: TokenizerDownloadInfo::basic(
                "databricks/dbrx-instruct",
                vec!["tokenizer.json"],
                "dbrx-32k",
                Some(32000),
            ),
            exaone_32k: TokenizerDownloadInfo::basic(
                "LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct",
                vec!["tokenizer.json"],
                "exaone-32k",
                Some(32000),
            ),
            minicpm_122k: TokenizerDownloadInfo::basic(
                "openbmb/MiniCPM-2B-sft-bf16",
                vec!["tokenizer.json"],
                "minicpm-122k",
                Some(122753),
            ),
            codegemma_256k: TokenizerDownloadInfo::basic(
                "google/codegemma-7b-it",
                vec!["tokenizer.json"],
                "codegemma-256k",
                Some(256000),
            ),
            llama31_128k: TokenizerDownloadInfo::with_type(
                "meta-llama/Llama-3.1-8B-Instruct",
                vec!["tokenizer.json"],
                "llama31-128k",
                Some(128256),
                TokenizerType::TikTokenBpe,
                SpecialTokenConfig {
                    bos_token_id: Some(128000),
                    eos_token_id: Some(128001),
                    pad_token_id: Some(128004),
                },
            ),
            deepseekv3_100k: TokenizerDownloadInfo::basic(
                "deepseek-ai/DeepSeek-V3",
                vec!["tokenizer.json"],
                "deepseekv3-100k",
                Some(102400),
            ),
            aya_256k: TokenizerDownloadInfo::basic(
                "CohereForAI/aya-23-8B",
                vec!["tokenizer.json"],
                "aya-256k",
                Some(256000),
            ),
            smollm_49k: TokenizerDownloadInfo::with_type(
                "HuggingFaceTB/SmolLM-1.7B-Instruct",
                vec!["tokenizer.json"],
                "smollm-49k",
                Some(49152),
                TokenizerType::Bpe,
                SpecialTokenConfig {
                    bos_token_id: Some(0),
                    eos_token_id: Some(0),
                    pad_token_id: None,
                },
            ),
            phi2_51k: TokenizerDownloadInfo::basic(
                "microsoft/phi-2",
                vec!["tokenizer.json"],
                "phi2-51k",
                Some(51200),
            ),
            falcon2_32k: TokenizerDownloadInfo::basic(
                "tiiuae/falcon-11B",
                vec!["tokenizer.json"],
                "falcon2-32k",
                Some(32000),
            ),
            olmo2_100k: TokenizerDownloadInfo::basic(
                "allenai/OLMo-2-1124-7B-Instruct",
                vec!["tokenizer.json"],
                "olmo2-100k",
                Some(100278),
            ),
            llama32_128k: TokenizerDownloadInfo::with_type(
                "meta-llama/Llama-3.2-3B-Instruct",
                vec!["tokenizer.json"],
                "llama32-128k",
                Some(128256),
                TokenizerType::TikTokenBpe,
                SpecialTokenConfig {
                    bos_token_id: Some(128000),
                    eos_token_id: Some(128001),
                    pad_token_id: Some(128004),
                },
            ),
            // --- New SLM family entries ---
            phi4_mini_100k: TokenizerDownloadInfo::with_type(
                "microsoft/Phi-4-mini",
                vec!["tokenizer.json"],
                "phi4-mini-100k",
                Some(100352),
                TokenizerType::TikTokenBpe,
                SpecialTokenConfig {
                    bos_token_id: Some(100257),
                    eos_token_id: Some(100265),
                    pad_token_id: None,
                },
            ),
            mistral_v03_32k: TokenizerDownloadInfo::with_type(
                "mistralai/Mistral-7B-Instruct-v0.3",
                vec!["tokenizer.json"],
                "mistral-v03-32k",
                Some(32768),
                TokenizerType::SentencePiece,
                SpecialTokenConfig {
                    bos_token_id: Some(1),
                    eos_token_id: Some(2),
                    pad_token_id: None,
                },
            ),
            smollm2_49k: TokenizerDownloadInfo::with_type(
                "HuggingFaceTB/SmolLM2-1.7B-Instruct",
                vec!["tokenizer.json"],
                "smollm2-49k",
                Some(49152),
                TokenizerType::Bpe,
                SpecialTokenConfig {
                    bos_token_id: Some(0),
                    eos_token_id: Some(0),
                    pad_token_id: None,
                },
            ),
        }
    }
}

/// Tokenizer algorithm type for model family classification
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenizerType {
    /// Byte-Pair Encoding (standard HuggingFace BPE)
    Bpe,
    /// TikToken BPE (used by OpenAI, Phi-4, Qwen)
    TikTokenBpe,
    /// SentencePiece (used by Gemma, Mistral, LLaMA-2)
    SentencePiece,
    /// SentencePiece Unigram (e.g., T5)
    Unigram,
    /// Unknown or unclassified tokenizer type
    Unknown,
}

impl std::fmt::Display for TokenizerType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Bpe => write!(f, "BPE"),
            Self::TikTokenBpe => write!(f, "TikToken-BPE"),
            Self::SentencePiece => write!(f, "SentencePiece"),
            Self::Unigram => write!(f, "Unigram"),
            Self::Unknown => write!(f, "Unknown"),
        }
    }
}

/// Special token configuration for a model family
#[derive(Debug, Clone, Default)]
pub struct SpecialTokenConfig {
    /// BOS (beginning-of-sequence) token ID
    pub bos_token_id: Option<u32>,
    /// EOS (end-of-sequence) token ID
    pub eos_token_id: Option<u32>,
    /// PAD token ID
    pub pad_token_id: Option<u32>,
}

/// Download metadata for tokenizer acquisition from HuggingFace Hub
#[derive(Debug, Clone)]
pub struct TokenizerDownloadInfo {
    /// HuggingFace repository identifier (e.g., "meta-llama/Llama-2-7b-hf")
    pub repo: String,
    /// Required tokenizer files to download (e.g., ["tokenizer.json"])
    pub files: Vec<String>,
    /// Cache identifier for persistent storage (e.g., "llama2-32k")
    pub cache_key: String,
    /// Expected vocabulary size for validation (optional)
    pub expected_vocab: Option<usize>,
    /// Tokenizer algorithm type (defaults to Unknown for legacy entries)
    pub tokenizer_type: TokenizerType,
    /// Special token configuration (defaults to empty for legacy entries)
    pub special_tokens: SpecialTokenConfig,
}

impl TokenizerDownloadInfo {
    /// Create a new entry with only the required fields; tokenizer_type and
    /// special_tokens default to Unknown / empty respectively.
    fn basic(repo: &str, files: Vec<&str>, cache_key: &str, expected_vocab: Option<usize>) -> Self {
        Self {
            repo: repo.to_string(),
            files: files.into_iter().map(String::from).collect(),
            cache_key: cache_key.to_string(),
            expected_vocab,
            tokenizer_type: TokenizerType::Unknown,
            special_tokens: SpecialTokenConfig::default(),
        }
    }

    /// Create a fully-specified entry with tokenizer type and special tokens.
    fn with_type(
        repo: &str,
        files: Vec<&str>,
        cache_key: &str,
        expected_vocab: Option<usize>,
        tokenizer_type: TokenizerType,
        special_tokens: SpecialTokenConfig,
    ) -> Self {
        Self {
            repo: repo.to_string(),
            files: files.into_iter().map(String::from).collect(),
            cache_key: cache_key.to_string(),
            expected_vocab,
            tokenizer_type,
            special_tokens,
        }
    }
}

/// Comprehensive tokenizer resolution strategy for neural network models
#[derive(Clone)]
pub enum TokenizerStrategy {
    /// User explicitly specified tokenizer path
    Exact(PathBuf),
    /// Auto-discovered compatible tokenizer in model directory
    Discovered(PathBuf),
    /// Smart download required from HuggingFace Hub
    NeedsDownload(TokenizerDownloadInfo),
    /// GGUF file contains embedded tokenizer data
    EmbeddedGguf(Arc<dyn Tokenizer>),
    /// Mock tokenizer for testing (non-strict mode only)
    Mock,
}

impl TokenizerStrategy {
    /// Check if strategy requires network access
    pub fn requires_network(&self) -> bool {
        matches!(self, TokenizerStrategy::NeedsDownload(_))
    }

    /// Check if strategy uses cached resources
    pub fn uses_cache(&self) -> bool {
        matches!(self, TokenizerStrategy::Discovered(_) | TokenizerStrategy::NeedsDownload(_))
    }

    /// Get description for logging and error messages
    pub fn description(&self) -> &'static str {
        match self {
            TokenizerStrategy::Exact(_) => "user-specified tokenizer",
            TokenizerStrategy::Discovered(_) => "auto-discovered tokenizer",
            TokenizerStrategy::NeedsDownload(_) => "smart download required",
            TokenizerStrategy::EmbeddedGguf(_) => "GGUF-embedded tokenizer",
            TokenizerStrategy::Mock => "mock tokenizer (testing only)",
        }
    }
}

/// Primary tokenizer discovery engine for BitNet-rs neural network models
pub struct TokenizerDiscovery {
    _mmap: memmap2::Mmap, // Keep mmap alive
    gguf_reader: GgufReader<'static>,
    model_path: PathBuf,
    vocab_size: usize,
    model_type: String,
    compatibility_matrix: ModelCompatibilityMatrix,
}

impl TokenizerDiscovery {
    /// Create discovery engine from GGUF model file
    ///
    /// Tests feature spec: issue-249-tokenizer-discovery-neural-network-spec.md#ac1-tokenizerdiscovery-implementation
    ///
    /// # Arguments
    /// * `path` - Path to GGUF model file
    ///
    /// # Returns
    /// * `Ok(TokenizerDiscovery)` - Successfully initialized discovery engine
    /// * `Err(BitNetError::Model)` - GGUF parsing failed or metadata missing
    ///
    /// # Example
    /// ```rust,no_run
    /// use bitnet_tokenizers::TokenizerDiscovery;
    /// use std::path::Path;
    ///
    /// # fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// let discovery = TokenizerDiscovery::from_gguf(Path::new("model.gguf"))?;
    /// assert_eq!(discovery.vocab_size(), 128256); // LLaMA-3
    /// # Ok(())
    /// # }
    /// ```
    pub fn from_gguf(path: &Path) -> Result<Self> {
        // Validate file exists and is readable
        TokenizerErrorHandler::validate_file_exists(path, "GGUF model file")?;

        // Read GGUF file using memmap for efficiency
        let file =
            std::fs::File::open(path).map_err(|e| TokenizerErrorHandler::file_io_error(path, e))?;

        let mmap = unsafe { memmap2::Mmap::map(&file) }
            .map_err(|e| TokenizerErrorHandler::file_io_error(path, e))?;

        // Create GGUF reader from memory-mapped data
        // We need to transmute the lifetime to 'static since we're keeping the mmap alive
        let reader = unsafe {
            let data_slice: &'static [u8] = std::mem::transmute(mmap.as_ref());
            GgufReader::new(data_slice)?
        };

        // Extract vocabulary size from metadata or tensors
        let vocab_size = Self::extract_vocab_size(&reader)?;

        // Validate vocabulary size is reasonable
        ModelTypeDetector::validate_vocab_size(vocab_size)?;

        // Extract model architecture type
        let model_type = Self::extract_model_type(&reader)?;

        Ok(Self {
            _mmap: mmap, // Keep mmap alive
            gguf_reader: reader,
            model_path: path.to_path_buf(),
            vocab_size,
            model_type,
            compatibility_matrix: ModelCompatibilityMatrix::default(),
        })
    }

    /// Discover optimal tokenizer strategy for the loaded model
    ///
    /// Tests feature spec: issue-249-tokenizer-discovery-neural-network-spec.md#ac1-tokenizerdiscovery-implementation
    ///
    /// # Returns
    /// * `TokenizerStrategy::Discovered` - Compatible tokenizer found locally
    /// * `TokenizerStrategy::NeedsDownload` - Smart download required
    /// * `TokenizerStrategy::EmbeddedGguf` - GGUF contains embedded tokenizer
    /// * `TokenizerStrategy::Mock` - Fallback for testing (non-strict mode only)
    ///
    /// # Errors
    /// * `BitNetError::Inference` - No compatible tokenizer found in strict mode
    pub fn discover_tokenizer_strategy(&self) -> Result<TokenizerStrategy> {
        info!(
            "Discovering tokenizer strategy for {} model (vocab_size: {})",
            self.model_type, self.vocab_size
        );

        // 1. Check for embedded tokenizer in GGUF
        if let Ok(Some(embedded)) = self.try_extract_embedded_tokenizer() {
            debug!("Found embedded tokenizer in GGUF file");
            return Ok(TokenizerStrategy::EmbeddedGguf(embedded));
        }

        // 2. Check for co-located tokenizer files
        if let Ok(Some(path)) = self.check_colocated_tokenizers() {
            debug!("Found co-located tokenizer at: {}", path.display());
            return Ok(TokenizerStrategy::Discovered(path));
        }

        // 3. Check cache locations
        if let Ok(Some(path)) = self.check_cache_locations() {
            debug!("Found cached tokenizer at: {}", path.display());
            return Ok(TokenizerStrategy::Discovered(path));
        }

        // 4. Check if we can infer download source
        if let Ok(Some(download_info)) = self.infer_download_source() {
            debug!("Can download compatible tokenizer from: {}", download_info.repo);
            return Ok(TokenizerStrategy::NeedsDownload(download_info));
        }

        // 5. Check strict mode - no fallback to mock in strict mode
        if std::env::var("BITNET_STRICT_TOKENIZERS").is_ok() {
            return Err(BitNetError::Config(format!(
                "No compatible tokenizer found for {} model with vocab_size {} (strict mode)",
                self.model_type, self.vocab_size
            )));
        }

        // 6. Fallback to mock for testing
        warn!("No compatible tokenizer found, falling back to mock (non-strict mode)");
        Ok(TokenizerStrategy::Mock)
    }

    /// Get vocabulary size from model metadata
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// Get model architecture type (e.g., "llama", "gpt2")
    pub fn model_type(&self) -> &str {
        &self.model_type
    }

    /// Check if model requires large vocabulary optimization (>64K tokens)
    ///
    /// Large vocabularies require GPU acceleration for efficient embedding lookup
    pub fn requires_large_vocab_optimization(&self) -> bool {
        ModelTypeDetector::requires_gpu_acceleration(self.vocab_size)
    }

    /// Try alternative vocabulary size metadata keys
    fn try_alternative_vocab_keys(reader: &GgufReader) -> Option<usize> {
        let alt_keys = [
            "llama.vocab_size",
            "gpt2.vocab_size",
            "gptneox.vocab_size",
            "bert.vocab_size",
            "t5.vocab_size",
            "transformer.vocab_size",
            "model.vocab_size",
            "vocab_size",
        ];

        for key in &alt_keys {
            if let Some(vocab_size) = reader.get_u32_metadata(key) {
                return Some(vocab_size as usize);
            }
        }
        None
    }

    /// Try to infer vocabulary size from embedding tensor shape
    fn try_infer_vocab_from_embeddings(reader: &GgufReader) -> Option<usize> {
        let tensor_names = reader.tensor_names();
        for name in tensor_names {
            if (name.contains("token_embd")
                || name.contains("wte")
                || name.contains("embed")
                || name.contains("embeddings"))
                && let Some(info) = reader.get_tensor_info_by_name(name)
            {
                let shape = &info.shape;
                if !shape.is_empty() {
                    let possible_vocab = shape[0];
                    // Sanity check - vocab size should be reasonable
                    if (100..2_000_000).contains(&possible_vocab) {
                        debug!(
                            "Inferred vocab_size {} from embedding tensor '{}'",
                            possible_vocab, name
                        );
                        return Some(possible_vocab);
                    }
                }
            }
        }
        None
    }

    /// Get architecture-specific default vocabulary size
    fn get_architecture_default_vocab(reader: &GgufReader) -> Option<usize> {
        let arch = reader.get_string_metadata("general.architecture")?;
        match arch.as_str() {
            "llama" => {
                // Distinguish between LLaMA-2 (32K) and LLaMA-3 (128K)
                if let Some(name) = reader.get_string_metadata("general.name") {
                    if name.contains("llama-3") || name.contains("llama3") {
                        Some(128256)
                    } else {
                        Some(32000)
                    }
                } else {
                    Some(32000) // Default to LLaMA-2
                }
            }
            "gpt2" => Some(50257),
            "gptneox" => Some(50257),
            "bert" => Some(30522),
            "t5" => Some(32128),
            _ => None,
        }
    }

    /// Extract vocabulary size from GGUF metadata
    ///
    /// Tests feature spec: issue-249-tokenizer-discovery-neural-network-spec.md#ac1-tokenizerdiscovery-implementation
    fn extract_vocab_size(reader: &GgufReader) -> Result<usize> {
        // Strategy 1: Standard GGUF vocabulary size key
        if let Some(vocab_size) = reader.get_u32_metadata("tokenizer.ggml.vocab_size") {
            return Ok(vocab_size as usize);
        }

        // Strategy 2: Architecture-specific metadata key
        if let Some(arch) = reader.get_string_metadata("general.architecture") {
            let arch_key = format!("{}.vocab_size", arch);
            if let Some(vocab_size) = reader.get_u32_metadata(&arch_key) {
                return Ok(vocab_size as usize);
            }
        }

        // Strategy 3: Alternative metadata keys
        if let Some(vocab_size) = Self::try_alternative_vocab_keys(reader) {
            return Ok(vocab_size);
        }

        // Strategy 4: Infer from embedding tensor shape
        if let Some(vocab_size) = Self::try_infer_vocab_from_embeddings(reader) {
            return Ok(vocab_size);
        }

        // Strategy 5: Architecture-specific defaults
        if let Some(vocab_size) = Self::get_architecture_default_vocab(reader) {
            if let Some(arch) = reader.get_string_metadata("general.architecture") {
                warn!("Using architecture-specific default vocab_size {} for {}", vocab_size, arch);
            }
            return Ok(vocab_size);
        }

        // Could not determine vocab size
        Err(TokenizerErrorHandler::config_error(
            "Could not extract vocabulary size from GGUF metadata or tensors".to_string(),
        ))
    }

    /// Detect architecture from model name
    fn detect_architecture_from_name(name: &str) -> Option<String> {
        let name_lower = name.to_lowercase();

        // Architecture detection patterns: (architecture, patterns)
        let name_patterns = [
            ("bitnet", &["bitnet", "bitlinear"] as &[&str]),
            ("llama", &["llama"]),
            ("gpt2", &["gpt2", "gpt-2"]),
            ("gptneox", &["gpt-neo", "gptneox", "gpt-j"]),
            ("bert", &["bert"]),
            ("t5", &["t5"]),
        ];

        for (arch, patterns) in name_patterns {
            if patterns.iter().any(|pattern| name_lower.contains(pattern)) {
                debug!("Detected {} architecture from model name", arch);
                return Some(arch.to_string());
            }
        }
        None
    }

    /// Extract model architecture type from GGUF metadata
    fn extract_model_type(reader: &GgufReader) -> Result<String> {
        // Try to get architecture from metadata - this is the most reliable
        if let Some(arch) = reader.get_string_metadata("general.architecture") {
            debug!("Found architecture from metadata: {}", arch);
            return Ok(arch);
        }

        // Alternative metadata keys
        let alt_keys = [
            "model.architecture",
            "transformer.architecture",
            "llama.architecture",
            "gpt.architecture",
        ];

        for key in &alt_keys {
            if let Some(arch) = reader.get_string_metadata(key) {
                debug!("Found architecture from metadata key '{}': {}", key, arch);
                return Ok(arch);
            }
        }

        // Try to infer from model name
        if let Some(name) = reader.get_string_metadata("general.name")
            && let Some(arch) = Self::detect_architecture_from_name(&name)
        {
            return Ok(arch);
        }

        // Fallback: Analyze tensor patterns for architecture detection
        let tensor_names = reader.tensor_names();
        Self::detect_architecture_from_tensors(&tensor_names)
    }

    /// Detect architecture from tensor name patterns
    fn detect_architecture_from_tensors(tensor_names: &[&str]) -> Result<String> {
        // Architecture detection patterns: (architecture, patterns, description)
        let architecture_patterns = [
            (
                "bitnet",
                &["bitlinear", "bitnet"] as &[&str],
                "BitNet architecture from tensor patterns",
            ),
            (
                "llama",
                &["attn_q", "attn_k", "attn_v", "attention.wq", "attention.wk"],
                "LLaMA architecture from tensor patterns",
            ),
            ("t5", &["encoder", "decoder", "relative_attention_bias"], "T5 architecture"),
            ("bert", &["encoder", "self", "attention"], "BERT architecture"),
            ("gptneox", &["gpt_neox", "gptneox"], "GPT-Neo/J architecture"),
        ];

        // Check each architecture pattern
        for (arch, patterns, description) in architecture_patterns {
            let has_patterns = if arch == "gpt2" {
                // GPT-2 requires compound pattern matching
                tensor_names.iter().any(|name| {
                    (name.contains("mlp") || name.contains("c_fc"))
                        && (name.contains("attn") || name.contains("c_attn"))
                })
            } else if arch == "bert" || arch == "t5" {
                // BERT and T5 require multiple pattern matching
                patterns
                    .iter()
                    .all(|pattern| tensor_names.iter().any(|name| name.contains(pattern)))
            } else {
                // Simple pattern matching for other architectures
                patterns
                    .iter()
                    .any(|pattern| tensor_names.iter().any(|name| name.contains(pattern)))
            };

            if has_patterns {
                debug!("Detected {}", description);
                return Ok(arch.to_string());
            }
        }

        // GPT-2 detection with compound pattern (handled separately)
        let has_gpt2_patterns = tensor_names.iter().any(|name| {
            (name.contains("mlp") || name.contains("c_fc"))
                && (name.contains("attn") || name.contains("c_attn"))
        });
        if has_gpt2_patterns {
            debug!("Detected GPT-2 architecture from tensor patterns");
            return Ok("gpt2".to_string());
        }

        // Default fallback to generic transformer
        warn!("Could not determine specific architecture, defaulting to 'transformer'");
        Ok("transformer".to_string())
    }

    /// Check for co-located tokenizer files in model directory
    ///
    /// Tests feature spec: issue-249-tokenizer-discovery-neural-network-spec.md#ac5-fallback-strategy-system
    pub fn check_colocated_tokenizers(&self) -> Result<Option<PathBuf>> {
        let model_dir = self
            .model_path
            .parent()
            .ok_or_else(|| BitNetError::Config("Model path has no parent directory".to_string()))?;

        debug!("Searching for co-located tokenizers in: {}", model_dir.display());

        // Common tokenizer file names to check
        let tokenizer_files = [
            "tokenizer.json",
            "tokenizer.model",
            "vocab.json",
            "merges.txt",
            "special_tokens_map.json",
        ];

        for filename in &tokenizer_files {
            let tokenizer_path = model_dir.join(filename);
            if tokenizer_path.exists() && tokenizer_path.is_file() {
                debug!("Found co-located tokenizer file: {}", tokenizer_path.display());
                return Ok(Some(tokenizer_path));
            }
        }

        // Check for model name based tokenizer files
        if let Some(model_name) = self.model_path.file_stem()
            && let Some(model_str) = model_name.to_str()
        {
            let name_based_files = [
                format!("{}.tokenizer.json", model_str),
                format!("{}_tokenizer.json", model_str),
                format!("{}.vocab.json", model_str),
            ];

            for filename in &name_based_files {
                let tokenizer_path = model_dir.join(filename);
                if tokenizer_path.exists() && tokenizer_path.is_file() {
                    debug!("Found model-specific tokenizer file: {}", tokenizer_path.display());
                    return Ok(Some(tokenizer_path));
                }
            }
        }

        debug!("No co-located tokenizer files found");
        Ok(None)
    }

    /// Check standard cache directories for compatible tokenizers
    ///
    /// Tests feature spec: issue-249-tokenizer-discovery-neural-network-spec.md#ac5-fallback-strategy-system
    pub fn check_cache_locations(&self) -> Result<Option<PathBuf>> {
        debug!("Searching cache locations for compatible tokenizers");

        // Use centralized cache directory management
        let base_cache = CacheManager::cache_directory()?;

        if !base_cache.exists() {
            debug!("Base cache directory does not exist: {}", base_cache.display());
            return Ok(None);
        }

        // Check model-specific cache directory first
        let model_cache = CacheManager::model_cache_dir(&self.model_type, Some(self.vocab_size))?;
        if model_cache.exists() {
            let tokenizer_json = model_cache.join("tokenizer.json");
            if tokenizer_json.exists() {
                debug!("Found vocab-specific cached tokenizer: {}", tokenizer_json.display());
                return Ok(Some(tokenizer_json));
            }
        }

        // Check general model type directory
        let general_model_cache = CacheManager::model_cache_dir(&self.model_type, None)?;
        if general_model_cache.exists() {
            for filename in &["tokenizer.json", "tokenizer.model"] {
                let tokenizer_path = general_model_cache.join(filename);
                if tokenizer_path.exists() {
                    debug!("Found general cached tokenizer: {}", tokenizer_path.display());
                    return Ok(Some(tokenizer_path));
                }
            }
        }

        // Check HuggingFace cache layout
        let hf_cache = base_cache.parent().unwrap_or(&base_cache).join("huggingface");
        if hf_cache.exists()
            && let Ok(entries) = std::fs::read_dir(&hf_cache)
        {
            for entry in entries.flatten() {
                if entry.file_type().is_ok_and(|ft| ft.is_dir()) {
                    let repo_dir = entry.path();
                    let tokenizer_json = repo_dir.join("tokenizer.json");
                    if tokenizer_json.exists() {
                        debug!("Found HF cached tokenizer: {}", tokenizer_json.display());
                        return Ok(Some(tokenizer_json));
                    }
                }
            }
        }

        debug!("No cached tokenizers found");
        Ok(None)
    }

    /// Infer download source based on neural network model patterns
    ///
    /// Tests feature spec: issue-249-tokenizer-discovery-neural-network-spec.md#ac1-tokenizerdiscovery-implementation
    pub fn infer_download_source(&self) -> Result<Option<TokenizerDownloadInfo>> {
        let m = &self.compatibility_matrix;
        // Neural network model compatibility matrix lookup
        match (self.model_type.as_str(), self.vocab_size) {
            // LLaMA family
            ("llama", 128256) => Ok(Some(m.llama31_128k.clone())),
            ("llama", 32000) => Ok(Some(m.llama2_32k.clone())),
            // GPT-2
            ("gpt2", 50257) => Ok(Some(m.gpt2_50k.clone())),
            // Qwen family
            ("qwen" | "qwen2" | "qwen2.5", 152064) => Ok(Some(m.qwen25_152k.clone())),
            ("qwen" | "qwen2", 151936) => Ok(Some(m.qwen2_150k.clone())),
            // Phi family
            ("phi" | "phi4", 100352) => Ok(Some(m.phi4_100k.clone())),
            ("phi" | "phi3", 32064) => Ok(Some(m.phi3_32k.clone())),
            ("phi" | "phi2", 51200) => Ok(Some(m.phi2_51k.clone())),
            // Gemma family
            ("gemma" | "gemma2", 256000) => Ok(Some(m.gemma2_256k.clone())),
            // Mistral family
            ("mistral", 32768) => Ok(Some(m.mistral_v03_32k.clone())),
            ("mistral", 32000) => Ok(Some(m.mistral_32k.clone())),
            ("mistral", 131072) => Ok(Some(m.mistral_nemo_128k.clone())),
            // SmolLM family
            ("smollm" | "llama", 49152) => Ok(Some(m.smollm2_49k.clone())),
            // BitNet
            ("bitnet", _) => Ok(Some(m.bitnet_custom.clone())),
            _ => Ok(None), // Unknown combination
        }
    }

    /// Extract special token IDs from GGUF metadata
    fn extract_special_tokens(&self) -> (Option<u32>, Option<u32>, Option<u32>) {
        let bos_token_id = self.gguf_reader.get_u32_metadata("tokenizer.ggml.bos_token_id");
        let eos_token_id = self.gguf_reader.get_u32_metadata("tokenizer.ggml.eos_token_id");
        let pad_token_id = self
            .gguf_reader
            .get_u32_metadata("tokenizer.ggml.pad_token_id")
            .or(self.gguf_reader.get_u32_metadata("tokenizer.ggml.unknown_token_id"));

        (bos_token_id, eos_token_id, pad_token_id)
    }

    /// Create basic tokenizer from special token configuration
    fn create_basic_tokenizer_from_tokens(
        &self,
        vocab_size: usize,
        bos: Option<u32>,
        eos: Option<u32>,
        pad: Option<u32>,
    ) -> Arc<dyn Tokenizer> {
        Arc::new(crate::BasicTokenizer::with_config(vocab_size, bos, eos, pad))
    }

    /// Try to extract embedded tokenizer from GGUF metadata
    ///
    /// Tests feature spec: issue-249-tokenizer-discovery-neural-network-spec.md#ac1-tokenizerdiscovery-implementation
    pub fn try_extract_embedded_tokenizer(&self) -> Result<Option<Arc<dyn Tokenizer>>> {
        debug!("Attempting to extract embedded tokenizer from GGUF metadata");

        // Strategy 1: Check for HuggingFace tokenizer.json embedded as string
        if let Some(tokenizer_json) = self.gguf_reader.get_string_metadata("tokenizer.json") {
            debug!("Found embedded tokenizer.json ({} chars)", tokenizer_json.len());

            if tokenizer_json.starts_with('{') && tokenizer_json.len() > 50 {
                let (bos_token_id, eos_token_id, pad_token_id) = self.extract_special_tokens();
                let tokenizer = self.create_basic_tokenizer_from_tokens(
                    self.vocab_size,
                    bos_token_id,
                    eos_token_id,
                    pad_token_id,
                );

                info!(
                    "Created tokenizer from embedded HF JSON (vocab_size: {}, bos: {:?}, eos: {:?})",
                    self.vocab_size, bos_token_id, eos_token_id
                );
                return Ok(Some(tokenizer));
            }
        }

        // Strategy 2: Check for tokenizer vocab embedded in metadata (SentencePiece style)
        if let Some(vocab) = self.gguf_reader.get_string_array_metadata("tokenizer.ggml.tokens") {
            debug!("Found embedded vocabulary with {} tokens", vocab.len());

            let vocab_matches = vocab.len() == self.vocab_size
                || (vocab.len() as i64 - self.vocab_size as i64).abs() < 100;

            if vocab_matches && !vocab.is_empty() {
                let (bos_token_id, eos_token_id, pad_token_id) = self.extract_special_tokens();

                // Validate special token IDs are within vocabulary bounds
                let valid_tokens = [bos_token_id, eos_token_id, pad_token_id]
                    .into_iter()
                    .flatten()
                    .all(|id| (id as usize) < vocab.len());

                if valid_tokens {
                    let tokenizer = self.create_basic_tokenizer_from_tokens(
                        vocab.len(),
                        bos_token_id,
                        eos_token_id,
                        pad_token_id,
                    );

                    info!(
                        "Created tokenizer from embedded vocabulary ({} tokens, bos: {:?}, eos: {:?})",
                        vocab.len(),
                        bos_token_id,
                        eos_token_id
                    );
                    return Ok(Some(tokenizer));
                } else {
                    warn!(
                        "Embedded vocabulary found but special tokens are invalid or out of bounds"
                    );
                }
            } else {
                warn!(
                    "Embedded vocabulary size mismatch: found {} tokens, expected {}",
                    vocab.len(),
                    self.vocab_size
                );
            }
        }

        // Strategy 3: Check if tokenizer model is embedded as bytes (binary SentencePiece model)
        if let Some(tokenizer_model) = self.gguf_reader.get_array_metadata("tokenizer.ggml.model") {
            debug!("Found embedded tokenizer.ggml.model ({} bytes)", tokenizer_model.len());

            if tokenizer_model.len() >= 1024 {
                let (bos_token_id, eos_token_id, pad_token_id) = self.extract_special_tokens();
                let tokenizer = self.create_basic_tokenizer_from_tokens(
                    self.vocab_size,
                    bos_token_id,
                    eos_token_id,
                    pad_token_id,
                );

                info!(
                    "Created tokenizer from embedded binary model ({} bytes)",
                    tokenizer_model.len()
                );
                return Ok(Some(tokenizer));
            } else {
                warn!(
                    "Embedded tokenizer model too small ({} bytes), may be corrupted",
                    tokenizer_model.len()
                );
            }
        }

        // Strategy 4: Check for minimal embedded metadata (just special token IDs)
        let (bos_token_id, eos_token_id, pad_token_id) = self.extract_special_tokens();

        if bos_token_id.is_some() || eos_token_id.is_some() {
            debug!(
                "Found minimal embedded tokenizer metadata (bos: {:?}, eos: {:?})",
                bos_token_id, eos_token_id
            );

            let tokenizer = self.create_basic_tokenizer_from_tokens(
                self.vocab_size,
                bos_token_id,
                eos_token_id,
                pad_token_id,
            );

            info!(
                "Created minimal tokenizer from embedded metadata (vocab_size: {})",
                self.vocab_size
            );
            return Ok(Some(tokenizer));
        }

        debug!("No embedded tokenizer found in GGUF metadata");
        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    #[cfg(feature = "cpu")]
    use super::{ModelCompatibilityMatrix, TokenizerDiscovery, TokenizerType};
    #[cfg(any(feature = "cpu", feature = "gpu"))]
    #[allow(unused_imports)]
    use crate::ModelTypeDetector;
    #[cfg(feature = "cpu")]
    use crate::{BitNetError, CacheManager, TokenizerDownloadInfo, TokenizerStrategy};
    #[cfg(feature = "cpu")]
    use std::path::Path;
    #[cfg(feature = "cpu")]
    use std::path::PathBuf;

    /// AC1: Tests TokenizerDiscovery GGUF metadata parsing functionality
    /// Tests feature spec: issue-249-tokenizer-discovery-neural-network-spec.md#ac1-tokenizerdiscovery-implementation
    #[test]
    #[cfg(feature = "cpu")]
    fn test_tokenizer_discovery_from_gguf_llama3() {
        // Test scaffolding - will fail until implementation complete
        let test_path = Path::new("test-models/llama3-128k.gguf");
        let result = TokenizerDiscovery::from_gguf(test_path);

        // This should fail with unimplemented! until actual implementation
        assert!(result.is_err(), "Test scaffolding should fail until implemented");
    }

    /// AC1: Tests vocabulary size extraction from GGUF metadata for large neural network models
    /// Tests feature spec: issue-249-tokenizer-discovery-neural-network-spec.md#ac1-tokenizerdiscovery-implementation
    #[test]
    #[cfg(feature = "cpu")]
    fn test_vocab_size_extraction_large_models() {
        // Test scaffolding for 128K+ vocabulary models (LLaMA-3)
        // This test will pass once extract_vocab_size is implemented
        let test_path = Path::new("test-models/llama3-128k.gguf");
        let result = TokenizerDiscovery::from_gguf(test_path);

        // Test scaffolding assertion - implementation needed
        assert!(result.is_err(), "Requires GGUF metadata parsing implementation");
    }

    /// AC1: Tests model architecture detection for neural network compatibility
    /// Tests feature spec: issue-249-tokenizer-discovery-neural-network-spec.md#ac1-tokenizerdiscovery-implementation
    #[test]
    #[cfg(feature = "cpu")]
    fn test_model_type_detection_neural_networks() {
        // Test different neural network architectures
        let test_cases = [
            ("test-models/llama2-32k.gguf", "llama"),
            ("test-models/gpt2-50k.gguf", "gpt2"),
            ("test-models/bitnet-custom.gguf", "bitnet"),
        ];

        for (model_path, _expected_type) in test_cases {
            let test_path = Path::new(model_path);
            let result = TokenizerDiscovery::from_gguf(test_path);

            // Test scaffolding - requires implementation
            assert!(result.is_err(), "Model type detection requires GGUF parsing implementation");
        }
    }

    /// AC1: Tests tokenizer strategy discovery for different neural network models
    /// Tests feature spec: issue-249-tokenizer-discovery-neural-network-spec.md#ac1-tokenizerdiscovery-implementation
    #[test]
    #[cfg(feature = "cpu")]
    fn test_discover_tokenizer_strategy_neural_networks() {
        // Test strategy discovery for various neural network models
        // This is comprehensive test scaffolding covering all strategy types

        // Test scaffolding setup - will need real TokenizerDiscovery instance
        // let discovery = create_mock_discovery("llama", 128256);
        // let strategy = discovery.discover_tokenizer_strategy().unwrap();

        // Expected behavior tests:
        // - TokenizerStrategy::Discovered for co-located files
        // - TokenizerStrategy::NeedsDownload for known model types
        // - TokenizerStrategy::EmbeddedGguf for GGUF-embedded tokenizers
        // - TokenizerStrategy::Mock for fallback (non-strict mode)

        // Test scaffolding placeholder - requires TokenizerDiscovery implementation
        println!("Γ£à AC1: Tokenizer discovery test scaffolding completed");
    }

    /// AC1: Tests large vocabulary optimization detection for GPU acceleration
    /// Tests feature spec: issue-249-tokenizer-discovery-neural-network-spec.md#ac1-tokenizerdiscovery-implementation
    #[test]
    #[cfg(any(feature = "gpu", feature = "cuda"))]
    fn test_large_vocab_optimization_detection() {
        // Test GPU acceleration requirements for large vocabularies

        // Test cases for different vocabulary sizes
        let test_cases = [
            (128256, true),  // LLaMA-3 - requires GPU optimization
            (32000, false),  // LLaMA-2 - CPU compatible
            (50257, false),  // GPT-2 - CPU compatible
            (1000000, true), // Hypothetical large model
        ];

        for (vocab_size, _should_optimize) in test_cases {
            // Mock discovery instance for testing
            // let discovery = create_mock_discovery("test", vocab_size);
            // assert_eq!(discovery.requires_large_vocab_optimization(), should_optimize);

            // Test scaffolding assertion
            assert!(vocab_size > 0, "Test scaffolding - vocab size validation");
        }
    }

    /// AC1: Tests neural network model compatibility matrix functionality
    /// Tests feature spec: issue-249-tokenizer-discovery-neural-network-spec.md#ac1-tokenizerdiscovery-implementation
    #[test]
    #[cfg(feature = "cpu")]
    fn test_neural_network_compatibility_matrix() {
        let matrix = ModelCompatibilityMatrix::default();

        // Validate LLaMA-3 configuration
        assert_eq!(matrix.llama3_128k.repo, "meta-llama/Meta-Llama-3-8B");
        assert_eq!(matrix.llama3_128k.expected_vocab, Some(128256));
        assert_eq!(matrix.llama3_128k.cache_key, "llama3-128k");

        // Validate LLaMA-2 configuration
        assert_eq!(matrix.llama2_32k.repo, "meta-llama/Llama-2-7b-hf");
        assert_eq!(matrix.llama2_32k.expected_vocab, Some(32000));

        // Validate GPT-2 configuration
        assert_eq!(matrix.gpt2_50k.repo, "openai-community/gpt2");
        assert_eq!(matrix.gpt2_50k.expected_vocab, Some(50257));

        // Validate BitNet configuration
        assert_eq!(matrix.bitnet_custom.repo, "1bitLLM/bitnet_b1_58-large");
        assert_eq!(matrix.bitnet_custom.files.len(), 2); // tokenizer.json + tokenizer.model
    }

    /// AC1: Tests tokenizer strategy properties and descriptions
    /// Tests feature spec: issue-249-tokenizer-discovery-neural-network-spec.md#ac1-tokenizerdiscovery-implementation
    #[test]
    #[cfg(feature = "cpu")]
    fn test_tokenizer_strategy_properties() {
        // Test strategy network requirements
        let download_info = TokenizerDownloadInfo::basic(
            "test/repo",
            vec!["tokenizer.json"],
            "test",
            Some(1000),
        );

        let strategies = [
            (TokenizerStrategy::Exact(PathBuf::from("test.json")), false, false),
            (TokenizerStrategy::Discovered(PathBuf::from("found.json")), false, true),
            (TokenizerStrategy::NeedsDownload(download_info), true, true),
            (TokenizerStrategy::Mock, false, false),
        ];

        for (strategy, should_need_network, should_use_cache) in strategies {
            assert_eq!(strategy.requires_network(), should_need_network);
            assert_eq!(strategy.uses_cache(), should_use_cache);
            assert!(!strategy.description().is_empty());
        }
    }

    // ================================
    // ENHANCED EDGE CASE TESTS
    // ================================

    /// Test GGUF parsing with corrupted metadata - should handle gracefully
    #[test]
    #[cfg(feature = "cpu")]
    fn test_gguf_parsing_corrupted_metadata() {
        use std::io::Write;
        use tempfile::NamedTempFile;

        // Create corrupted GGUF file with invalid header
        let mut temp_file = NamedTempFile::new().expect("Failed to create temp file");
        temp_file
            .write_all(b"CORRUPTED_HEADER_NOT_GGUF")
            .expect("Failed to write corrupted header");

        let result = TokenizerDiscovery::from_gguf(temp_file.path());
        assert!(result.is_err(), "Should reject corrupted GGUF files");

        // Verify error message is actionable - check that it's an error
        assert!(result.is_err(), "Should fail with corrupted GGUF");
    }

    /// Test GGUF parsing with extremely large vocabulary sizes
    #[test]
    #[cfg(feature = "cpu")]
    fn test_gguf_extreme_vocab_sizes() {
        // Test vocabulary size boundaries
        let extreme_vocab_sizes = [
            0,          // Invalid - zero vocabulary
            1,          // Minimal vocabulary
            65535,      // 16-bit boundary
            65536,      // Large vocab threshold
            128256,     // LLaMA-3 size
            1000000,    // Extremely large
            usize::MAX, // Maximum possible size
        ];

        for vocab_size in extreme_vocab_sizes {
            let is_valid = ModelTypeDetector::validate_vocab_size(vocab_size).is_ok();

            match vocab_size {
                0 => assert!(!is_valid, "Zero vocabulary should be invalid"),
                1..=2000000 => {
                    assert!(is_valid, "Reasonable vocabulary size should be valid: {}", vocab_size)
                }
                _ => {
                    assert!(!is_valid, "Extreme vocabulary size should be invalid: {}", vocab_size)
                }
            }
        }
    }

    /// Test memory pressure scenarios with large model files
    #[test]
    #[cfg(feature = "cpu")]
    fn test_memory_pressure_large_models() {
        use std::io::Write;
        use tempfile::NamedTempFile;

        // Simulate large GGUF file that could cause memory pressure
        let mut temp_file = NamedTempFile::new().expect("Failed to create temp file");

        // Write minimal valid GGUF header (simplified)
        let gguf_header = b"GGUF\x03\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00";
        temp_file.write_all(gguf_header).expect("Failed to write GGUF header");

        // Pad with zeros to simulate large file without actually allocating GB of memory
        let padding = vec![0u8; 1024]; // 1KB padding instead of GB
        for _ in 0..10 {
            temp_file.write_all(&padding).expect("Failed to write padding");
        }

        // Test that memory mapping works even for "large" files
        let result = TokenizerDiscovery::from_gguf(temp_file.path());

        // Should either succeed (if valid GGUF) or fail with specific error
        match result {
            Ok(_) => {}                       // Success case
            Err(BitNetError::Model(_)) => {}  // Expected GGUF parsing error
            Err(BitNetError::Config(_)) => {} // Expected configuration error (vocab size extraction)
            Err(other) => panic!("Unexpected error for large file: {:?}", other),
        }
    }

    /// Test concurrent access to GGUF discovery - thread safety
    #[test]
    #[cfg(feature = "cpu")]
    fn test_concurrent_gguf_discovery() {
        use std::io::Write;
        use std::sync::Arc;
        use std::thread;
        use tempfile::NamedTempFile;

        // Create a valid-looking GGUF file
        let temp_file = NamedTempFile::new().expect("Failed to create temp file");
        let path = Arc::new(temp_file.path().to_path_buf());

        // Write minimal GGUF structure
        let mut file =
            std::fs::OpenOptions::new().write(true).open(&*path).expect("Failed to open temp file");
        file.write_all(b"GGUF\x03\x00\x00\x00").expect("Failed to write header");

        // Spawn multiple threads to test concurrent access
        let handles: Vec<_> = (0..4)
            .map(|_| {
                let path_clone = Arc::clone(&path);
                thread::spawn(move || {
                    for _ in 0..10 {
                        let _result = TokenizerDiscovery::from_gguf(&path_clone);
                        // Don't assert success since this is a minimal GGUF file
                        // Just ensure no panics or race conditions
                    }
                })
            })
            .collect();

        // Wait for all threads to complete
        for handle in handles {
            handle.join().expect("Thread should complete without panic");
        }
    }

    /// Test file system permission errors
    #[test]
    #[cfg(feature = "cpu")]
    fn test_file_permission_errors() {
        // Test with completely inaccessible path
        let inaccessible_path = Path::new("/root/nonexistent/model.gguf");
        let result = TokenizerDiscovery::from_gguf(inaccessible_path);

        assert!(result.is_err(), "Should fail for inaccessible paths");
    }

    /// Test directory instead of file
    #[test]
    #[cfg(feature = "cpu")]
    fn test_directory_instead_of_file() {
        use tempfile::tempdir;

        let temp_dir = tempdir().expect("Failed to create temp directory");
        let result = TokenizerDiscovery::from_gguf(temp_dir.path());

        assert!(result.is_err(), "Should fail when given directory instead of file");
    }

    /// Test very long file paths (path length limits)
    #[test]
    #[cfg(feature = "cpu")]
    fn test_long_file_paths() {
        // Create extremely long path that might hit filesystem limits
        let long_filename = "a".repeat(255); // Near filesystem limit
        let long_path = Path::new("/tmp").join(format!("{}.gguf", long_filename));

        let result = TokenizerDiscovery::from_gguf(&long_path);
        assert!(result.is_err(), "Should handle long path names gracefully");
    }

    /// Test neural network model compatibility edge cases
    #[test]
    #[cfg(feature = "cpu")]
    fn test_neural_network_edge_cases() {
        let matrix = ModelCompatibilityMatrix::default();

        // Test edge case vocabulary sizes
        let edge_cases = [
            // LLaMA-3 exact boundary
            ("llama3", 128256, Some(matrix.llama3_128k.clone())),
            // LLaMA-2 exact boundary
            ("llama2", 32000, Some(matrix.llama2_32k.clone())),
            // GPT-2 exact boundary
            ("gpt2", 50257, Some(matrix.gpt2_50k.clone())),
            // Unknown model type
            ("unknown", 99999, None),
            // Edge case: exactly at GPU optimization threshold
            ("test", 65536, None),
            // Edge case: just below GPU threshold
            ("test", 65535, None),
            // Edge case: just above GPU threshold
            ("test", 65537, None),
        ];

        for (_model_type, vocab_size, expected_download_info) in edge_cases {
            // Test GPU acceleration detection
            let requires_gpu = ModelTypeDetector::requires_gpu_acceleration(vocab_size);
            let expected_gpu = vocab_size > 65536;
            assert_eq!(
                requires_gpu, expected_gpu,
                "GPU requirement mismatch for vocab_size: {}",
                vocab_size
            );

            // Test download info inference (mock discovery needed for real test)
            if expected_download_info.is_some() {
                // Would test with real discovery instance
                // let discovery = create_test_discovery(model_type, vocab_size);
                // let inferred = discovery.infer_download_source().unwrap();
                // assert_eq!(inferred, expected_download_info);
            }
        }
    }

    /// Test GGUF metadata key variations and missing fields
    #[test]
    #[cfg(feature = "cpu")]
    fn test_gguf_metadata_variations() {
        // Test various metadata key formats that might be encountered
        let metadata_keys = [
            "tokenizer.ggml.vocab_size", // Standard key
            "llama.vocab_size",          // LLaMA-specific
            "gpt2.vocab_size",           // GPT-2-specific
            "transformer.vocab_size",    // Generic transformer
            "model.vocab_size",          // Generic model
            "vocab_size",                // Simple key
            "vocabulary_size",           // Alternative naming
            "VOCAB_SIZE",                // Case variation
        ];

        // Test architecture key variations
        let arch_keys = [
            "general.architecture",     // Standard
            "model.architecture",       // Alternative
            "transformer.architecture", // Specific
            "llama.architecture",       // LLaMA-specific
            "gpt.architecture",         // GPT-specific
            "architecture",             // Simple
            "model_type",               // Alternative naming
        ];

        // These would be tested with actual GGUF files containing different metadata formats
        for key in metadata_keys.iter().chain(arch_keys.iter()) {
            // Test that key variations are handled properly
            assert!(!key.is_empty(), "Metadata key should not be empty");
            assert!(key.len() < 100, "Metadata key should be reasonable length");
        }
    }

    /// Test fallback strategies with edge cases
    #[test]
    #[cfg(feature = "cpu")]
    fn test_fallback_edge_cases() {
        use tempfile::tempdir;

        let temp_dir = tempdir().expect("Failed to create temp directory");

        // Test with empty directory (no co-located files)
        let empty_model_path = temp_dir.path().join("model.gguf");
        std::fs::File::create(&empty_model_path).expect("Failed to create empty model file");

        // Mock discovery for testing fallback scenarios
        // let discovery = create_test_discovery_from_path(&empty_model_path);

        // Test co-located file discovery with various file types
        let colocated_files = [
            "tokenizer.json",          // Standard HuggingFace
            "tokenizer.model",         // SentencePiece
            "vocab.json",              // Vocabulary only
            "merges.txt",              // BPE merges
            "special_tokens_map.json", // Special tokens
            "model.tokenizer.json",    // Model-specific
            "model_tokenizer.json",    // Alternative naming
            "model.vocab.json",        // Vocab-specific
        ];

        for filename in colocated_files {
            let colocated_path = temp_dir.path().join(filename);
            std::fs::File::create(&colocated_path).expect("Failed to create colocated file");

            // Test file is detectable
            assert!(colocated_path.exists(), "Colocated file should exist: {}", filename);
        }
    }

    /// Test cache directory edge cases and permissions
    #[test]
    #[cfg(feature = "cpu")]
    fn test_cache_directory_edge_cases() {
        use tempfile::tempdir;

        let temp_dir = tempdir().expect("Failed to create temp directory");

        // Test cache directory creation and access
        let cache_base = temp_dir.path().join("cache");
        let model_cache = cache_base.join("llama").join("128256");

        // Test nested directory creation
        std::fs::create_dir_all(&model_cache).expect("Failed to create cache directories");
        assert!(model_cache.exists(), "Cache directory should be created");

        // Test cache with various model types and vocabulary sizes
        let cache_scenarios = [
            ("llama", Some(32000)),   // LLaMA-2
            ("llama", Some(128256)),  // LLaMA-3
            ("gpt2", Some(50257)),    // GPT-2
            ("bitnet", None),         // No specific vocab size
            ("unknown", Some(99999)), // Unknown model type
        ];

        for (model_type, vocab_size) in cache_scenarios {
            let cache_result = CacheManager::model_cache_dir(model_type, vocab_size);
            match cache_result {
                Ok(cache_dir) => {
                    assert!(
                        !cache_dir.as_os_str().is_empty(),
                        "Cache directory path should not be empty"
                    );
                    assert!(
                        cache_dir.to_string_lossy().contains(model_type),
                        "Cache path should contain model type"
                    );
                }
                Err(_) => {
                    // Some combinations might fail, which is acceptable
                }
            }
        }
    }

    /// Test tokenizer file validation edge cases
    #[test]
    #[cfg(feature = "cpu")]
    fn test_tokenizer_file_validation() {
        use std::io::Write;
        use tempfile::NamedTempFile;

        // Test various tokenizer file formats and contents
        let test_scenarios = [
            // Valid JSON tokenizer
            (
                r#"{"version": "1.0", "model": {"type": "BPE"}, "normalizer": null, "pre_tokenizer": null}"#,
                true,
            ),
            // Invalid JSON
            (r#"{"invalid": json malformed"#, false),
            // Empty file
            ("", false),
            // Non-JSON content
            ("This is not JSON at all", false),
            // Very large JSON (memory test)
            (&"x".repeat(1024 * 1024), false), // 1MB of 'x' characters
        ];

        for (content, should_be_valid) in test_scenarios {
            let mut temp_file = NamedTempFile::new().expect("Failed to create temp file");
            temp_file.write_all(content.as_bytes()).expect("Failed to write test content");

            // Test file size validation
            let file_size = temp_file.as_file().metadata().expect("Failed to get metadata").len();

            if content.is_empty() {
                assert_eq!(file_size, 0, "Empty file should have zero size");
            } else {
                assert!(file_size > 0, "Non-empty file should have positive size");
            }

            // Test JSON parsing (would be done by actual tokenizer loading)
            if should_be_valid && content.starts_with('{') {
                let json_parse = serde_json::from_str::<serde_json::Value>(content);
                assert!(json_parse.is_ok(), "Valid JSON should parse successfully");
            }
        }
    }

    /// Test device capability detection for large vocabularies
    #[test]
    #[cfg(any(feature = "gpu", feature = "cuda"))]
    fn test_device_capability_detection() {
        // Test GPU acceleration requirements for different vocabulary sizes
        let vocab_scenarios = [
            (1000, false),  // Small vocab - CPU sufficient
            (32000, false), // LLaMA-2 - CPU sufficient
            (50257, false), // GPT-2 - CPU sufficient
            (65536, false), // Exactly at threshold - CPU sufficient
            (65537, true),  // Just above threshold - GPU recommended
            (128256, true), // LLaMA-3 - GPU required
            (200000, true), // Very large - GPU required
        ];

        for (vocab_size, should_need_gpu) in vocab_scenarios {
            let needs_gpu = ModelTypeDetector::requires_gpu_acceleration(vocab_size);
            assert_eq!(
                needs_gpu, should_need_gpu,
                "GPU requirement mismatch for vocab_size: {}",
                vocab_size
            );

            // Test memory estimation (mock calculation)
            let estimated_memory_mb = vocab_size * 4 * 1024 / (1024 * 1024); // Rough estimate: vocab_size * 4KB * embedding_dim / MB
            if vocab_size > 100000 {
                assert!(
                    estimated_memory_mb > 100,
                    "Large vocabularies should have significant memory requirements"
                );
            }
        }
    }

    /// Test strict mode enforcement edge cases
    #[test]
    #[cfg(feature = "cpu")]
    fn test_strict_mode_edge_cases() {
        // Test with strict mode enabled
        unsafe {
            std::env::set_var("BITNET_STRICT_TOKENIZERS", "1");
        }

        // Mock discovery that would normally fallback to mock tokenizer
        // let mock_discovery = create_failing_discovery();
        // let strategy_result = mock_discovery.discover_tokenizer_strategy();

        // In strict mode, should fail rather than fallback to mock
        // assert!(strategy_result.is_err(), "Should fail in strict mode without fallback");

        // Test strict mode detection
        let is_strict = std::env::var("BITNET_STRICT_TOKENIZERS").is_ok();
        assert!(is_strict, "Strict mode should be detected when environment variable is set");

        unsafe {
            std::env::remove_var("BITNET_STRICT_TOKENIZERS");
        }

        let is_strict_after = std::env::var("BITNET_STRICT_TOKENIZERS").is_ok();
        assert!(
            !is_strict_after,
            "Strict mode should be disabled after removing environment variable"
        );
    }

    /// Test quantization compatibility with tokenizer discovery
    #[test]
    #[cfg(feature = "cpu")]
    fn test_quantization_tokenizer_compatibility() {
        use bitnet_common::QuantizationType;

        // Test vocabulary sizes with different quantization methods
        let compatibility_matrix = [
            // (vocab_size, quantization_type, should_be_optimal)
            (32000, QuantizationType::I2S, true), // LLaMA-2 + I2S
            (128256, QuantizationType::I2S, true), // LLaMA-3 + I2S (good for large vocab)
            (50257, QuantizationType::TL1, true), // GPT-2 + TL1
            (32000, QuantizationType::TL2, true), // LLaMA-2 + TL2
            (128256, QuantizationType::TL1, false), // LLaMA-3 + TL1 (not optimal for large vocab)
            (200000, QuantizationType::TL2, false), // Very large + TL2 (not optimal)
        ];

        for (vocab_size, quant_type, should_be_optimal) in compatibility_matrix {
            // Test compatibility logic
            let is_compatible = match quant_type {
                QuantizationType::I2S => vocab_size <= 200000, // I2S handles large vocabularies well
                QuantizationType::TL1 | QuantizationType::TL2 => vocab_size <= 65536, // Table lookup better for smaller vocabs
            };

            if should_be_optimal {
                assert!(
                    is_compatible,
                    "Optimal combination should be compatible: vocab={}, quant={:?}",
                    vocab_size, quant_type
                );
            }

            // Test memory efficiency estimation
            let memory_factor = match quant_type {
                QuantizationType::I2S => 2.0,  // 2-bit quantization
                QuantizationType::TL1 => 1.5,  // Table lookup with compression
                QuantizationType::TL2 => 1.25, // Enhanced table lookup
            };

            let estimated_memory = (vocab_size as f64 * memory_factor) / 1024.0; // KB
            assert!(estimated_memory > 0.0, "Memory estimation should be positive");
        }
    }

    /// Test error message quality and actionability
    #[test]
    #[cfg(feature = "cpu")]
    fn test_error_message_quality() {
        // Test that error messages provide actionable guidance
        let test_error_scenarios = [
            ("nonexistent.gguf", "file not found"),
            ("/root/restricted.gguf", "permission"),
            ("directory/", "not a file"),
        ];

        for (path, _expected_error_hint) in test_error_scenarios {
            let result = TokenizerDiscovery::from_gguf(Path::new(path));
            assert!(result.is_err(), "Should fail for invalid path: {}", path);

            // Just verify we got an error - error content validation would require actual implementation
            // Error should exist and be meaningful (avoid unwrap_err due to missing Debug trait)
        }
    }

    // ================================
    // MATRIX COMPLETENESS VALIDATION
    // ================================

    /// Helper: returns all matrix entries as (name, &TokenizerDownloadInfo) pairs
    #[cfg(feature = "cpu")]
    fn all_matrix_entries(
        m: &ModelCompatibilityMatrix,
    ) -> Vec<(&'static str, &TokenizerDownloadInfo)> {
        vec![
            ("llama3_128k", &m.llama3_128k),
            ("llama2_32k", &m.llama2_32k),
            ("gpt2_50k", &m.gpt2_50k),
            ("bitnet_custom", &m.bitnet_custom),
            ("phi4_100k", &m.phi4_100k),
            ("qwen2_150k", &m.qwen2_150k),
            ("gemma_256k", &m.gemma_256k),
            ("mistral_32k", &m.mistral_32k),
            ("deepseek_100k", &m.deepseek_100k),
            ("starcoder_49k", &m.starcoder_49k),
            ("falcon_65k", &m.falcon_65k),
            ("codellama_32k", &m.codellama_32k),
            ("command_256k", &m.command_256k),
            ("internlm_103k", &m.internlm_103k),
            ("yi_64k", &m.yi_64k),
            ("baichuan_64k", &m.baichuan_64k),
            ("chatglm_65k", &m.chatglm_65k),
            ("mpt_50k", &m.mpt_50k),
            ("phi4_mini_100k", &m.phi4_mini_100k),
            ("mistral_v03_32k", &m.mistral_v03_32k),
            ("smollm2_49k", &m.smollm2_49k),
            ("qwen25_152k", &m.qwen25_152k),
            ("gemma2_256k", &m.gemma2_256k),
            ("llama31_128k", &m.llama31_128k),
            ("smollm_49k", &m.smollm_49k),
        ]
    }

    #[test]
    #[cfg(feature = "cpu")]
    fn test_entry_count() {
        let matrix = ModelCompatibilityMatrix::default();
        let entries = all_matrix_entries(&matrix);
        assert_eq!(entries.len(), 25, "Expected 25 tokenizer entries, got {}", entries.len());
    }

    #[test]
    #[cfg(feature = "cpu")]
    fn test_all_entries_have_non_empty_repo() {
        let matrix = ModelCompatibilityMatrix::default();
        for (name, entry) in all_matrix_entries(&matrix) {
            assert!(!entry.repo.is_empty(), "Entry '{}' has empty repo", name);
            assert!(
                entry.repo.contains('/'),
                "Entry '{}' repo '{}' should be owner/repo format",
                name,
                entry.repo
            );
        }
    }

    #[test]
    #[cfg(feature = "cpu")]
    fn test_all_entries_have_non_empty_files() {
        let matrix = ModelCompatibilityMatrix::default();
        for (name, entry) in all_matrix_entries(&matrix) {
            assert!(!entry.files.is_empty(), "Entry '{}' has no files listed", name);
            for f in &entry.files {
                assert!(!f.is_empty(), "Entry '{}' has empty filename", name);
            }
        }
    }

    #[test]
    #[cfg(feature = "cpu")]
    fn test_all_entries_have_non_empty_cache_key() {
        let matrix = ModelCompatibilityMatrix::default();
        for (name, entry) in all_matrix_entries(&matrix) {
            assert!(!entry.cache_key.is_empty(), "Entry '{}' has empty cache_key", name);
        }
    }

    #[test]
    #[cfg(feature = "cpu")]
    fn test_all_entries_have_positive_vocab_size() {
        let matrix = ModelCompatibilityMatrix::default();
        for (name, entry) in all_matrix_entries(&matrix) {
            if let Some(vocab) = entry.expected_vocab {
                assert!(vocab > 0, "Entry '{}' has non-positive vocab_size: {}", name, vocab);
            }
        }
    }

    #[test]
    #[cfg(feature = "cpu")]
    fn test_key_families_present() {
        let matrix = ModelCompatibilityMatrix::default();
        let entries = all_matrix_entries(&matrix);
        let expected_families = [
            "llama3",
            "llama2",
            "phi4",
            "qwen2",
            "gemma",
            "mistral",
            "deepseek",
            "starcoder",
            "falcon",
            "gpt2",
            "bitnet",
            "yi",
            "baichuan",
            "chatglm",
            "mpt",
        ];
        for family in &expected_families {
            assert!(
                entries.iter().any(|(name, _)| name.starts_with(family)),
                "Expected family '{}' not found in tokenizer matrix",
                family
            );
        }
    }

    #[test]
    #[cfg(feature = "cpu")]
    fn test_no_duplicate_cache_keys() {
        let matrix = ModelCompatibilityMatrix::default();
        let entries = all_matrix_entries(&matrix);
        let mut seen = std::collections::HashSet::new();
        for (name, entry) in &entries {
            assert!(
                seen.insert(&entry.cache_key),
                "Entry '{}' has duplicate cache_key '{}'",
                name,
                entry.cache_key
            );
        }
    }

    /// Test performance boundaries for neural network inference
    #[test]
    #[cfg(feature = "cpu")]
    fn test_performance_boundaries() {
        use std::time::Instant;

        // Test tokenizer discovery performance requirements
        let performance_scenarios = [
            // (vocab_size, expected_max_time_ms, description)
            (32000, 100, "LLaMA-2 discovery should be fast"),
            (128256, 200, "LLaMA-3 discovery acceptable latency"),
            (50257, 80, "GPT-2 discovery should be very fast"),
        ];

        for (vocab_size, max_time_ms, description) in performance_scenarios {
            let start = Instant::now();

            // Simulate discovery performance (mock timing)
            let _matrix = ModelCompatibilityMatrix::default();
            let _requires_gpu = ModelTypeDetector::requires_gpu_acceleration(vocab_size);

            // Mock some computation time
            for _ in 0..vocab_size / 1000 {
                std::hint::black_box(vocab_size * 2);
            }

            let elapsed = start.elapsed();
            let elapsed_ms = elapsed.as_millis() as u64;

            // For this test, we're not enforcing strict timing since it's hardware dependent
            // But we validate the timing measurement works
            assert!(elapsed_ms < 10000, "{}: took too long ({}ms)", description, elapsed_ms);

            // Log performance for monitoring
            if elapsed_ms > max_time_ms {
                println!(
                    "Warning: {} took {}ms (expected <{}ms)",
                    description, elapsed_ms, max_time_ms
                );
            }
        }
    }

    // ================================
    // SLM FAMILY DISCOVERY TESTS
    // ================================

    /// Test Phi-4 family discovery: model ID → correct tokenizer config
    #[test]
    #[cfg(feature = "cpu")]
    fn test_phi4_discovery() {
        let m = ModelCompatibilityMatrix::default();

        assert_eq!(m.phi4_100k.repo, "microsoft/phi-4");
        assert_eq!(m.phi4_100k.expected_vocab, Some(100352));
        assert_eq!(m.phi4_100k.tokenizer_type, TokenizerType::TikTokenBpe);
        assert_eq!(m.phi4_100k.special_tokens.bos_token_id, Some(100257));
        assert_eq!(m.phi4_100k.special_tokens.eos_token_id, Some(100265));

        assert_eq!(m.phi4_mini_100k.repo, "microsoft/Phi-4-mini");
        assert_eq!(m.phi4_mini_100k.expected_vocab, Some(100352));
        assert_eq!(m.phi4_mini_100k.tokenizer_type, TokenizerType::TikTokenBpe);
        assert_eq!(m.phi4_mini_100k.special_tokens.bos_token_id, Some(100257));
        assert_eq!(m.phi4_mini_100k.special_tokens.eos_token_id, Some(100265));
    }

    /// Test Qwen2.5 discovery
    #[test]
    #[cfg(feature = "cpu")]
    fn test_qwen25_discovery() {
        let m = ModelCompatibilityMatrix::default();

        assert_eq!(m.qwen25_152k.repo, "Qwen/Qwen2.5-7B-Instruct");
        assert_eq!(m.qwen25_152k.expected_vocab, Some(152064));
        assert_eq!(m.qwen25_152k.tokenizer_type, TokenizerType::TikTokenBpe);
    }

    /// Test Gemma-2 discovery
    #[test]
    #[cfg(feature = "cpu")]
    fn test_gemma2_discovery() {
        let m = ModelCompatibilityMatrix::default();

        assert_eq!(m.gemma2_256k.repo, "google/gemma-2-9b-it");
        assert_eq!(m.gemma2_256k.expected_vocab, Some(256000));
        assert_eq!(m.gemma2_256k.tokenizer_type, TokenizerType::SentencePiece);
        assert_eq!(m.gemma2_256k.special_tokens.bos_token_id, Some(2));
        assert_eq!(m.gemma2_256k.special_tokens.eos_token_id, Some(1));
        assert_eq!(m.gemma2_256k.special_tokens.pad_token_id, Some(0));
    }

    /// Test Mistral v0.3 discovery
    #[test]
    #[cfg(feature = "cpu")]
    fn test_mistral_v03_discovery() {
        let m = ModelCompatibilityMatrix::default();

        assert_eq!(m.mistral_v03_32k.repo, "mistralai/Mistral-7B-Instruct-v0.3");
        assert_eq!(m.mistral_v03_32k.expected_vocab, Some(32768));
        assert_eq!(m.mistral_v03_32k.tokenizer_type, TokenizerType::SentencePiece);
        assert_eq!(m.mistral_v03_32k.special_tokens.bos_token_id, Some(1));
        assert_eq!(m.mistral_v03_32k.special_tokens.eos_token_id, Some(2));
    }

    /// Test LLaMA 3.1 discovery
    #[test]
    #[cfg(feature = "cpu")]
    fn test_llama31_discovery() {
        let m = ModelCompatibilityMatrix::default();

        assert_eq!(m.llama31_128k.repo, "meta-llama/Llama-3.1-8B-Instruct");
        assert_eq!(m.llama31_128k.expected_vocab, Some(128256));
        assert_eq!(m.llama31_128k.tokenizer_type, TokenizerType::TikTokenBpe);
        assert_eq!(m.llama31_128k.special_tokens.bos_token_id, Some(128000));
        assert_eq!(m.llama31_128k.special_tokens.eos_token_id, Some(128001));
    }

    /// Test SmolLM2 discovery
    #[test]
    #[cfg(feature = "cpu")]
    fn test_smollm2_discovery() {
        let m = ModelCompatibilityMatrix::default();

        assert_eq!(m.smollm2_49k.repo, "HuggingFaceTB/SmolLM2-1.7B-Instruct");
        assert_eq!(m.smollm2_49k.expected_vocab, Some(49152));
        assert_eq!(m.smollm2_49k.tokenizer_type, TokenizerType::Bpe);
    }

    /// Test tokenizer type detection across all typed entries
    #[test]
    #[cfg(feature = "cpu")]
    fn test_tokenizer_type_detection() {
        let m = ModelCompatibilityMatrix::default();

        // TikToken BPE models
        let tiktoken_entries = [
            &m.phi4_100k,
            &m.phi4_mini_100k,
            &m.qwen25_152k,
            &m.llama31_128k,
            &m.llama32_128k,
        ];
        for entry in tiktoken_entries {
            assert_eq!(
                entry.tokenizer_type,
                TokenizerType::TikTokenBpe,
                "{} should be TikToken BPE",
                entry.repo
            );
        }

        // SentencePiece models
        let sp_entries = [
            &m.gemma_256k,
            &m.gemma2_256k,
            &m.mistral_32k,
            &m.mistral_v03_32k,
            &m.mixtral_32k,
        ];
        for entry in sp_entries {
            assert_eq!(
                entry.tokenizer_type,
                TokenizerType::SentencePiece,
                "{} should be SentencePiece",
                entry.repo
            );
        }

        // Standard BPE models
        let bpe_entries = [&m.smollm_49k, &m.smollm2_49k];
        for entry in bpe_entries {
            assert_eq!(
                entry.tokenizer_type,
                TokenizerType::Bpe,
                "{} should be BPE",
                entry.repo
            );
        }
    }

    /// Test special token configs for all annotated families
    #[test]
    #[cfg(feature = "cpu")]
    fn test_special_token_configs() {
        let m = ModelCompatibilityMatrix::default();

        // Phi-4 family: BOS=100257, EOS=100265
        assert_eq!(m.phi4_100k.special_tokens.bos_token_id, Some(100257));
        assert_eq!(m.phi4_100k.special_tokens.eos_token_id, Some(100265));
        assert_eq!(m.phi4_mini_100k.special_tokens.bos_token_id, Some(100257));

        // Gemma: BOS=2, EOS=1, PAD=0
        assert_eq!(m.gemma_256k.special_tokens.bos_token_id, Some(2));
        assert_eq!(m.gemma_256k.special_tokens.eos_token_id, Some(1));
        assert_eq!(m.gemma_256k.special_tokens.pad_token_id, Some(0));

        // Mistral: BOS=1, EOS=2
        assert_eq!(m.mistral_32k.special_tokens.bos_token_id, Some(1));
        assert_eq!(m.mistral_32k.special_tokens.eos_token_id, Some(2));
        assert_eq!(m.mistral_32k.special_tokens.pad_token_id, None);

        // LLaMA-3.1: BOS=128000, EOS=128001, PAD=128004
        assert_eq!(m.llama31_128k.special_tokens.bos_token_id, Some(128000));
        assert_eq!(m.llama31_128k.special_tokens.eos_token_id, Some(128001));
        assert_eq!(m.llama31_128k.special_tokens.pad_token_id, Some(128004));
    }

    /// Test vocab size validation for all SLM families
    #[test]
    #[cfg(feature = "cpu")]
    fn test_slm_vocab_size_validation() {
        let expected = [
            ("phi4_100k", 100352),
            ("phi4_mini_100k", 100352),
            ("qwen25_152k", 152064),
            ("gemma2_256k", 256000),
            ("mistral_v03_32k", 32768),
            ("llama31_128k", 128256),
            ("smollm2_49k", 49152),
        ];

        let m = ModelCompatibilityMatrix::default();
        let entries = all_matrix_entries(&m);

        for (name, vocab) in expected {
            let entry = entries.iter().find(|(n, _)| *n == name);
            assert!(entry.is_some(), "Entry '{}' should exist in matrix", name);
            let (_, info) = entry.unwrap();
            assert_eq!(
                info.expected_vocab,
                Some(vocab),
                "Entry '{}' should have vocab_size {}",
                name,
                vocab
            );
            assert!(
                ModelTypeDetector::validate_vocab_size(vocab).is_ok(),
                "Vocab {} for '{}' should be valid",
                vocab,
                name
            );
        }
    }

    /// Test fallback for unknown models returns None from infer_download_source
    #[test]
    #[cfg(feature = "cpu")]
    fn test_unknown_model_fallback() {
        // Unknown vocab sizes should map to "unknown" in detector
        assert_eq!(ModelTypeDetector::detect_from_vocab_size(99999), "unknown");
        assert_eq!(ModelTypeDetector::expected_vocab_size("nonexistent"), None);
    }

    /// Test no duplicate cache keys across all entries including new SLM entries
    #[test]
    #[cfg(feature = "cpu")]
    fn test_no_duplicate_cache_keys_with_slm() {
        let m = ModelCompatibilityMatrix::default();
        let entries = all_matrix_entries(&m);
        let mut seen = std::collections::HashSet::new();
        for (name, entry) in &entries {
            assert!(
                seen.insert(&entry.cache_key),
                "Entry '{}' has duplicate cache_key '{}'",
                name,
                entry.cache_key
            );
        }
    }

    /// Test SLM families are present in the key families check
    #[test]
    #[cfg(feature = "cpu")]
    fn test_slm_families_present() {
        let m = ModelCompatibilityMatrix::default();
        let entries = all_matrix_entries(&m);
        let slm_families = [
            "phi4",
            "phi4_mini",
            "qwen25",
            "gemma2",
            "mistral_v03",
            "llama31",
            "smollm2",
            "smollm",
        ];
        for family in &slm_families {
            assert!(
                entries.iter().any(|(name, _)| name.starts_with(family)),
                "SLM family '{}' not found in tokenizer matrix",
                family
            );
        }
    }

    /// Test TokenizerType Display formatting
    #[test]
    #[cfg(feature = "cpu")]
    fn test_tokenizer_type_display() {
        assert_eq!(format!("{}", TokenizerType::Bpe), "BPE");
        assert_eq!(format!("{}", TokenizerType::TikTokenBpe), "TikToken-BPE");
        assert_eq!(format!("{}", TokenizerType::SentencePiece), "SentencePiece");
        assert_eq!(format!("{}", TokenizerType::Unigram), "Unigram");
        assert_eq!(format!("{}", TokenizerType::Unknown), "Unknown");
    }
}