//! Neural network layer implementations
//!
//! This module contains the core neural network layer implementations
//! for BitNet inference, including quantized linear layers and attention
//! mechanisms.

pub mod attention;
pub mod kv_cache_validation;
pub mod quantized_linear;

#[cfg(test)]
mod quantized_linear_tests;

pub use attention::BitNetAttention;
pub use kv_cache_validation::validate_kv_cache_dims;
pub use quantized_linear::{LookupTable, QuantizedLinear};
