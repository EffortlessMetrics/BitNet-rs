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

/// Runtime backend detection fallback (post-install, pre-rebuild)
///
/// This function provides runtime detection when libraries are installed after xtask build.
/// It checks multiple sources in priority order:
///
/// 1. `BITNET_CROSSVAL_LIBDIR` (explicit override)
/// 2. Backend-specific granular overrides (`CROSSVAL_RPATH_BITNET`, `CROSSVAL_RPATH_LLAMA`)
/// 3. Backend home dir + subdirectories (`BITNET_CPP_DIR/build`, `LLAMA_CPP_DIR/build`, etc.)
///
/// # Arguments
///
/// * `backend` - The C++ backend to detect (BitNet or Llama)
///
/// # Returns
///
/// * `Ok((true, Some(path)))` - Backend libraries found at runtime, with matched path
/// * `Ok((false, None))` - Backend libraries not found
/// * `Err(String)` - Error during detection
///
/// # Platform-Specific Library Extensions
///
/// - Linux: `.so`
/// - macOS: `.dylib`
/// - Windows: `.dll`
pub fn detect_backend_runtime(
    backend: CppBackend,
) -> Result<(bool, Option<std::path::PathBuf>), String> {
    let mut candidates: Vec<std::path::PathBuf> = Vec::new();

    // Priority 1: BITNET_CROSSVAL_LIBDIR (explicit override)
    if let Ok(p) = std::env::var("BITNET_CROSSVAL_LIBDIR") {
        candidates.push(p.into());
    }

    // Priority 2: Granular overrides (backend-specific)
    match backend {
        CppBackend::BitNet => {
            if let Ok(p) = std::env::var("CROSSVAL_RPATH_BITNET") {
                candidates.push(p.into());
            }
        }
        CppBackend::Llama => {
            if let Ok(p) = std::env::var("CROSSVAL_RPATH_LLAMA") {
                candidates.push(p.into());
            }
        }
    }

    // Priority 3: Backend home directory + subdirectories
    let home_var = match backend {
        CppBackend::BitNet => "BITNET_CPP_DIR",
        CppBackend::Llama => "LLAMA_CPP_DIR",
    };

    if let Ok(root) = std::env::var(home_var) {
        let root_path = std::path::Path::new(&root);
        for sub in ["build", "build/bin", "build/lib"] {
            candidates.push(root_path.join(sub));
        }
    }

    // Check for required library filenames per platform
    let exts = if cfg!(target_os = "windows") {
        vec!["dll"]
    } else if cfg!(target_os = "macos") {
        vec!["dylib"]
    } else {
        vec!["so"]
    };

    let needs: &[&str] = match backend {
        CppBackend::BitNet => &["bitnet"],
        CppBackend::Llama => &["llama", "ggml"],
    };

    // Check each candidate directory and return first match with path
    for dir in candidates {
        if !dir.exists() {
            continue;
        }

        // Check if all required libraries are present
        let all_found = needs.iter().all(|stem| {
            exts.iter().any(|ext| {
                let lib_name = format_lib_name_ext(stem, ext);
                dir.join(&lib_name).exists()
            })
        });

        if all_found {
            return Ok((true, Some(dir)));
        }
    }

    Ok((false, None))
}

/// Format library name with specific extension (helper for runtime detection)
///
/// # Arguments
///
/// * `stem` - Library name stem (e.g., "bitnet", "llama")
/// * `ext` - File extension (e.g., "so", "dylib", "dll")
///
/// # Returns
///
/// Formatted library name:
/// - Windows: `{stem}.{ext}` (e.g., "bitnet.dll")
/// - Unix: `lib{stem}.{ext}` (e.g., "libbitnet.so")
fn format_lib_name_ext(stem: &str, ext: &str) -> String {
    if cfg!(target_os = "windows") {
        format!("{}.{}", stem, ext)
    } else {
        format!("lib{}.{}", stem, ext)
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
