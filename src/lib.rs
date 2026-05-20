//! # bitnet-rs — 1-bit LLM Inference Engine
//!
//! Pre-alpha (v0.2.1-dev) Rust inference engine for `BitNet` 1-bit large language models.
//!
//! ## Status
//!
//! This is **pre-alpha software**. Correctness, performance, and validation work is ongoing.
//! CPU inference with SIMD optimization works; GPU backends are scaffolded but not validated.
//! Do not use in production.
//!
//! ## Feature Flags
//!
//! Default features are **empty** — always specify features explicitly:
//!
//! - `cpu`: SIMD-optimised CPU inference (AVX2 / AVX-512 / NEON)
//! - `gpu`: GPU acceleration (CUDA umbrella; requires CUDA 12.x)
//! - `full-cli`: Enable all CLI subcommands
//! - `ffi`: C++ FFI bridge for cross-validation
//! - `fixtures`: GGUF fixture-based integration tests (test-only)
//!
//! ## Architecture
//!
//! The workspace contains ~200 crates. Key crates:
//!
//! - [`bitnet_common`]: Shared types, config, error types
//! - [`bitnet_models`]: GGUF / `SafeTensors` model loading
//! - [`bitnet_quantization`]: `I2_S`, `TL1`, `TL2` quantization
//! - `bitnet_kernels`: AVX2 / AVX-512 / NEON / CUDA compute kernels
//! - `bitnet_inference`: Autoregressive generation engine
//! - `bitnet_tokenizers`: Universal tokenizer with auto-discovery
//!
//! ## Quick Start
//!
//! ```rust,no_run
//! use bitnet::prelude::*;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Create a BitNet model config (load tensors from GGUF in practice)
//! let device = Device::Cpu;
//! let config = BitNetConfig::default();
//! let _model = BitNetModel::new(config, device);
//! # Ok(())
//! # }
//! ```

#![cfg_attr(docsrs, feature(doc_cfg))]
#![deny(missing_docs)]
#![warn(clippy::all, clippy::pedantic)]
#![allow(clippy::module_name_repetitions)]

/// Build metadata captured at compile time.
pub mod build_info;
mod constants;
/// Convenient imports for common `BitNet` types and traits.
pub mod prelude;

pub use build_info::*;
pub use constants::{MSRV, VERSION};

// Re-export core functionality
pub use bitnet_common as common;
pub use bitnet_models as models;
pub use bitnet_quantization as quantization;

#[cfg(feature = "inference")]
#[cfg_attr(docsrs, doc(cfg(feature = "inference")))]
pub use bitnet_inference as inference;

#[cfg(feature = "tokenizers")]
#[cfg_attr(docsrs, doc(cfg(feature = "tokenizers")))]
pub use bitnet_tokenizers as tokenizers;

#[cfg(feature = "kernels")]
#[cfg_attr(docsrs, doc(cfg(feature = "kernels")))]
pub use bitnet_kernels as kernels;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version() {
        #[allow(clippy::const_is_empty)]
        {
            assert!(!VERSION.is_empty());
        }
        // Version is read from Cargo.toml via env!(); don't hardcode it here
        assert!(VERSION.starts_with("0."), "expected 0.x version, got {VERSION}");
    }

    #[test]
    fn test_msrv() {
        assert_eq!(MSRV, "1.92.0");
    }

    #[test]
    fn test_build_info() {
        // These should not panic
        let _ = build_info::GIT_HASH;
        let _ = build_info::BUILD_TIMESTAMP;
        let _ = build_info::TARGET;
        let _ = build_info::RUSTC_VERSION;
        let _ = GIT_HASH;
        let _ = BUILD_TIMESTAMP;
        let _ = TARGET;
        let _ = RUSTC_VERSION;
    }

    #[test]
    fn test_prelude_imports() {
        use crate::prelude::*;
        // Test that prelude imports work
        let _config = BitNetConfig::default();
    }
}
