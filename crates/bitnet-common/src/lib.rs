//! Common types, traits, and utilities for BitNet inference
//!
//! This crate provides the foundational types and abstractions used across
//! the BitNet ecosystem, including configuration, error handling, and tensor
//! abstractions.

pub mod config;
pub mod error;
pub mod math;
pub mod tensor;
pub mod types;

pub use config::*;
pub use error::*;
pub use math::ceil_div;
pub use tensor::*;
pub use types::*;
