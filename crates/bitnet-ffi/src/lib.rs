//! C API bindings for BitNet
//!
//! This module provides a comprehensive C API that serves as a drop-in replacement
//! for the existing BitNet C++ bindings. It maintains exact signature compatibility
//! while providing enhanced error handling, thread safety, and performance monitoring.

pub mod c_api;
pub mod config;
pub mod error;
pub mod inference;
pub mod llama_compat;
pub mod memory;
pub mod model;
pub mod streaming;
pub mod threading;

pub use c_api::*;
pub use config::*;
pub use error::*;
pub use inference::*;
pub use llama_compat::*;
pub use memory::*;
pub use model::*;
pub use streaming::*;
pub use threading::*;
