//! CUDA element-wise operations for neural network inference.
//!
//! Provides element-wise arithmetic (add, subtract, multiply, divide, FMA),
//! activation functions (`ReLU`, `GELU`, `SiLU`/`SwiGLU`, sigmoid, tanh), and
//! broadcasting semantics (scalar-tensor, vector-tensor, tensor-tensor).
//!
//! All GPU-specific code is gated behind `#[cfg(any(feature = "gpu", feature = "cuda"))]`.
//! CPU reference implementations are always available for testing and fallback.

mod activation;
mod broadcast;
mod error;
mod ops;

pub use activation::{Activation, apply_activation, apply_activation_inplace};
pub use broadcast::BroadcastShape;
pub use error::{ElementWiseError, Result};
pub use ops::{
    add, add_inplace, div, div_inplace, fma, fma_inplace, mul, mul_inplace, sub, sub_inplace,
};
