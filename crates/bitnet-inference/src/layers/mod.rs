//! Neural network layer implementations
//!
//! This module contains the core neural network layer implementations
//! for BitNet inference, including quantized linear layers and attention
//! mechanisms.

pub mod attention;
pub mod quantized_linear;

pub use attention::BitNetAttention;
pub use quantized_linear::QuantizedLinear;
