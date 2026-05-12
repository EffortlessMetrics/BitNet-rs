//! Focused helpers for the `run_simple_generation` command path.
//!
//! The top-level command still coordinates loading and decoding, while these
//! submodules isolate setup concerns that do not need decode-loop state.

pub(crate) mod environment;
pub(crate) mod model_format;
pub(crate) mod prompt;
