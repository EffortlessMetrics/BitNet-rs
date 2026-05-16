//! Tokenizer model-family compatibility metadata.

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
    pub(crate) fn basic(
        repo: &str,
        files: Vec<&str>,
        cache_key: &str,
        expected_vocab: Option<usize>,
    ) -> Self {
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
    pub(crate) fn with_type(
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
