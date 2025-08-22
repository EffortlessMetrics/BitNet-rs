//! Model definitions and loading for BitNet inference

pub mod bitnet;
pub mod formats;
pub mod gguf_min;
pub mod gguf_parity;
pub mod gguf_simple;
pub mod loader;
pub mod minimal;
pub mod security;
pub mod transformer;
pub mod weight_mapper;

#[cfg(test)]
mod transformer_tests;

pub use bitnet::*;
pub use gguf_simple::load_gguf;
pub use loader::*;

// Export GGUF reader for tokenizer loading
pub use formats::gguf::GgufReader;
