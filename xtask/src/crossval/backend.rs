//! Backend selection for dual-backend cross-validation
//!
//! Supports two C++ reference implementations:
//! - bitnet.cpp: For BitNet models (microsoft/bitnet-b1.58-2B-4T-gguf)
//! - llama.cpp: For LLaMA models (llama-3, llama-2, etc.)

#![allow(dead_code)] // TODO: Backend helper methods to be used by preflight command

use clap::ValueEnum;
use std::path::Path;

/// C++ backend selection for cross-validation
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum CppBackend {
    /// Use bitnet.cpp for tokenization and evaluation
    ///
    /// Models: microsoft-bitnet-b1.58-2B-4T-gguf
    /// Tokenizer: bitnet.cpp's tokenizer API
    /// Required libs: libbitnet*.so
    #[value(name = "bitnet")]
    BitNet,

    /// Use llama.cpp for tokenization and evaluation
    ///
    /// Models: llama-3-instruct, llama-2, SmolLM3, etc.
    /// Tokenizer: llama.cpp's tokenizer API
    /// Required libs: libllama*.so, libggml*.so
    #[value(name = "llama")]
    Llama,
}

impl CppBackend {
    /// Auto-detect backend from model path
    ///
    /// Detection heuristics:
    /// - Path contains "bitnet" or "microsoft/bitnet" → BitNet
    /// - Path contains "llama" → Llama
    /// - Default: Llama (more common, safer fallback)
    ///
    /// # Examples
    ///
    /// ```
    /// use std::path::Path;
    /// use xtask::crossval::backend::CppBackend;
    ///
    /// let path = Path::new("models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf");
    /// assert_eq!(CppBackend::from_model_path(path), CppBackend::BitNet);
    ///
    /// let path = Path::new("models/llama-3-8b-instruct.gguf");
    /// assert_eq!(CppBackend::from_model_path(path), CppBackend::Llama);
    /// ```
    pub fn from_model_path(path: &Path) -> Self {
        let path_str = path.to_string_lossy().to_lowercase();

        if path_str.contains("bitnet") || path_str.contains("microsoft/bitnet") {
            Self::BitNet
        } else if path_str.contains("llama") {
            Self::Llama
        } else {
            // Conservative default: llama (more common, better tested)
            Self::Llama
        }
    }

    /// Get backend name for diagnostics
    ///
    /// # Examples
    ///
    /// ```
    /// # use xtask::crossval::backend::CppBackend;
    /// assert_eq!(CppBackend::BitNet.name(), "bitnet.cpp");
    /// assert_eq!(CppBackend::Llama.name(), "llama.cpp");
    /// ```
    pub fn name(&self) -> &'static str {
        match self {
            Self::BitNet => "bitnet.cpp",
            Self::Llama => "llama.cpp",
        }
    }

    /// Get required library patterns for preflight checks
    ///
    /// Returns library name prefixes that must be found during build.
    ///
    /// # Examples
    ///
    /// ```
    /// # use xtask::crossval::backend::CppBackend;
    /// assert_eq!(CppBackend::BitNet.required_libs(), &["libbitnet"]);
    /// assert_eq!(CppBackend::Llama.required_libs(), &["libllama", "libggml"]);
    /// ```
    pub fn required_libs(&self) -> &[&'static str] {
        match self {
            Self::BitNet => &["libbitnet"],
            Self::Llama => &["libllama", "libggml"],
        }
    }

    /// Get setup command for this backend
    ///
    /// Returns the command users should run to set up the C++ reference.
    /// Unified command - auto-detection handles both backends.
    pub fn setup_command(&self) -> &'static str {
        "eval \"$(cargo run -p xtask -- setup-cpp-auto --emit=sh)\""
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_autodetect_bitnet() {
        let path = Path::new("models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf");
        assert_eq!(CppBackend::from_model_path(path), CppBackend::BitNet);

        let path = Path::new("models/bitnet/model.gguf");
        assert_eq!(CppBackend::from_model_path(path), CppBackend::BitNet);
    }

    #[test]
    fn test_autodetect_llama() {
        let path = Path::new("models/llama-3-8b-instruct.gguf");
        assert_eq!(CppBackend::from_model_path(path), CppBackend::Llama);

        let path = Path::new("models/smollm3/SmolLM3-3B-F16.gguf");
        assert_eq!(CppBackend::from_model_path(path), CppBackend::Llama);
    }

    #[test]
    fn test_backend_metadata() {
        assert_eq!(CppBackend::BitNet.name(), "bitnet.cpp");
        assert_eq!(CppBackend::Llama.name(), "llama.cpp");

        assert_eq!(CppBackend::BitNet.required_libs(), &["libbitnet"]);
        assert_eq!(CppBackend::Llama.required_libs(), &["libllama", "libggml"]);
    }
}
