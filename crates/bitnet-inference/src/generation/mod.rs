//! Autoregressive Generation Implementation
//!
//! This module provides autoregressive text generation capabilities with
//! various sampling strategies and deterministic generation support.

pub mod autoregressive;
pub mod deterministic;
pub mod sampling;

pub use autoregressive::{AutoregressiveGenerator, GenerationConfig as GenConfig};
pub use deterministic::{DeterministicGenerator, set_deterministic_seed};
pub use sampling::{SamplingConfig as SampleConfig, SamplingStrategy};
