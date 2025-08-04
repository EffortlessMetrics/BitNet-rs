//! C API bindings for BitNet
//!
//! This module provides a comprehensive C API that serves as a drop-in replacement
//! for the existing BitNet C++ bindings. It maintains exact signature compatibility
//! while providing enhanced error handling, thread safety, and performance monitoring.

pub mod c_api;
pub mod error;
pub mod model;
pub mod config;
pub mod inference;
pub mod memory;
pub mod threading;
pub mod streaming;

pub use c_api::*;
pub use error::*;
pub use model::*;
pub use config::*;
pub use inference::*;
pub use memory::*;
pub use threading::*;
pub use streaming::*;