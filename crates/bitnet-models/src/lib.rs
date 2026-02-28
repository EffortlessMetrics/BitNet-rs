//! Model definitions and loading for BitNet inference

pub mod bitnet;
pub mod correction_policy;
pub mod fingerprint;
pub mod formats;
pub mod gguf_min;
pub mod gguf_parity;
pub mod gguf_simple;
pub mod loader;
pub mod minimal;
pub mod names;
pub mod production_loader;
pub mod qk256_utils;
pub mod quant;
pub mod security;
pub mod transformer;
pub mod weight_loader;
pub mod weight_mapper;

#[cfg(test)]
mod transformer_tests;

pub use bitnet::*;
pub use gguf_simple::GGUFLoaderConfig; // AC1: Export loader config for strict mode
#[allow(deprecated)]
pub use gguf_simple::load_gguf;
pub use gguf_simple::load_gguf_full;
pub use loader::*;
pub use production_loader::*;

// Export GGUF reader for tokenizer loading
pub use formats::gguf::GgufReader;

// Export weight mapper utilities for crossval tests
pub use weight_mapper::dry_run_remap_names;

// AC2: Re-export QK256 tolerance constants from bitnet-quantization (Issue #469)
pub use bitnet_quantization::{QK256_SIZE_TOLERANCE_PERCENT, qk256_tolerance_bytes};
